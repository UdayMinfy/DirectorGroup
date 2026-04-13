import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import SEARCH_TIMEZONE


RELATIVE_DATE_PATTERNS = (
    (re.compile(r"\bday before yesterday\b", re.IGNORECASE), -2, "day_before_yesterday"),
    (re.compile(r"\byesterday\b", re.IGNORECASE), -1, "yesterday"),
    (re.compile(r"\btoday\b", re.IGNORECASE), 0, "today"),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), 1, "tomorrow"),
)

RESULT_HINTS = ("won", "winner", "result", "score", "highlights", "summary")
SCHEDULE_HINTS = ("match", "matches", "fixture", "fixtures", "schedule", "playing", "live")
IPL_HINTS = ("ipl", "indian premier league")
MONTH_LOOKUP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def build_query_context(prompt):
    cleaned_prompt = " ".join((prompt or "").split()).strip()
    rewritten_prompt, resolved_date, relative_label = _replace_relative_dates(cleaned_prompt)
    lowered = cleaned_prompt.casefold()

    return {
        "original_prompt": cleaned_prompt,
        "rewritten_prompt": rewritten_prompt,
        "resolved_date": resolved_date,
        "relative_label": relative_label,
        "is_ipl": any(term in lowered for term in IPL_HINTS),
        "focus": _pick_focus(lowered),
    }


def rewrite_search_query(prompt):
    context = build_query_context(prompt)
    if not context["original_prompt"]:
        return ""

    if not context["is_ipl"]:
        return context["rewritten_prompt"]

    if context["resolved_date"]:
        return _build_ipl_date_query(context["focus"], context["resolved_date"], context["relative_label"])

    return f"{context['rewritten_prompt']} official IPL match fixture live score"


def _build_ipl_date_query(focus, resolved_date, relative_label):
    date_query = (
        f'IPL {focus} on {resolved_date} {relative_label} playing today live score fixture '
        'official match schedule scorecard'
    )
    return " ".join(date_query.split())


def _replace_relative_dates(prompt):
    current_date = datetime.now(ZoneInfo(SEARCH_TIMEZONE)).date()
    rewritten = prompt
    resolved_date = None
    relative_label = ""

    for pattern, offset_days, label in RELATIVE_DATE_PATTERNS:
        if pattern.search(rewritten):
            resolved = current_date + timedelta(days=offset_days)
            resolved_text = _format_date(resolved)
            rewritten = pattern.sub(f"on {resolved_text}", rewritten)
            resolved_date = resolved_text
            relative_label = label.replace("_", " ")
            break

    return rewritten, resolved_date, relative_label


def parse_resolved_date(date_text):
    if not date_text:
        return None

    normalized = re.sub(r"\s+", " ", date_text.strip())
    match = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", normalized)
    if not match:
        return None

    month_name, day_text, year_text = match.groups()
    month = MONTH_LOOKUP.get(month_name.casefold())
    if not month:
        return None

    return {
        "text": normalized,
        "month": month,
        "day": int(day_text),
        "year": int(year_text),
        "month_name": month_name,
    }


def _pick_focus(lowered_prompt):
    if any(term in lowered_prompt for term in RESULT_HINTS):
        return "match result winner score highlights"
    if any(term in lowered_prompt for term in SCHEDULE_HINTS):
        return "match today playing today fixture live score"
    return "match today fixture live score"


def _format_date(value):
    return f"{value.strftime('%B')} {value.day}, {value.year}"
