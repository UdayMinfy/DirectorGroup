import json
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_GUARDRAIL_ID, BEDROCK_GUARDRAIL_VERSION, BEDROCK_MODEL_ID, BEDROCK_REGION
from query_rewriter import parse_resolved_date


PREFERRED_IPL_DOMAINS = {
    "iplt20.com": 10,
    "www.iplt20.com": 10,
    "espncricinfo.com": 8,
    "www.espncricinfo.com": 8,
    "cricbuzz.com": 7,
    "www.cricbuzz.com": 7,
    "bbc.com": 6,
    "www.bbc.com": 6,
    "ndtv.com": 5,
    "www.ndtv.com": 5,
    "hindustantimes.com": 4,
    "www.hindustantimes.com": 4,
    "indianexpress.com": 4,
    "www.indianexpress.com": 4,
    "moneycontrol.com": 1,
    "www.moneycontrol.com": 1,
}

GOOD_PATH_HINTS = (
    "fixture",
    "fixtures",
    "schedule",
    "match",
    "matches",
    "result",
    "results",
    "live-score",
    "scorecard",
)

DATE_SPECIFIC_HINTS = (
    "today",
    "tomorrow",
    "yesterday",
    "march",
    "april",
    "may",
    "june",
)

BAD_PATH_HINTS = (
    "player",
    "players",
    "profile",
    "auction",
    "photo",
    "video",
    "standings",
    "points-table",
)

BAD_TITLE_HINTS = (
    "all 10 teams",
    "full list of fixtures",
    "full schedule",
    "standings",
)

RESTRICTED_HINTS = (
    "video",
    "highlights",
    "subscription",
    "premium",
    "sign-in",
    "login",
)


class ResultRanker:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    def rerank(self, prompt, search_query, websites, limit, requested_date_text=""):
        heuristic_ranked = _heuristic_rank(prompt, search_query, websites, requested_date_text)
        llm_ranked = self._llm_rank(prompt, search_query, heuristic_ranked, limit, requested_date_text)
        return llm_ranked[:limit] if llm_ranked else heuristic_ranked[:limit]

    def _llm_rank(self, prompt, search_query, websites, limit, requested_date_text=""):
        condensed = []
        for index, website in enumerate(websites, start=1):
            condensed.append(
                {
                    "index": index,
                    "title": website.get("title", ""),
                    "url": website.get("url", ""),
                    "snippet": (website.get("tavily_snippet") or website.get("fallback_content") or "")[:300],
                }
            )

        instruction = "\n\n".join(
            [
                "You rank search results for web extraction.",
                "Prefer famous, reputable domains and URLs likely to allow readable first-page extraction.",
                "Prefer score, result, match, schedule, and article pages.",
                "Avoid likely blocked or restricted pages such as video, highlights, login, subscription, premium, or anti-bot-prone URLs.",
                "Return JSON only with the key ranked_indices containing result indices in best-first order.",
                f"User prompt: {prompt}",
                f"Search query: {search_query}",
                f"Requested date text: {requested_date_text}",
                f"Limit: {limit}",
                f"Candidates: {json.dumps(condensed, ensure_ascii=True)}",
            ]
        )

        request_params = {
            "modelId": BEDROCK_MODEL_ID,
            "messages": [{"role": "user", "content": [{"text": instruction}]}],
        }
        if BEDROCK_GUARDRAIL_ID and BEDROCK_GUARDRAIL_VERSION:
            request_params["guardrailConfig"] = {
                "guardrailIdentifier": BEDROCK_GUARDRAIL_ID,
                "guardrailVersion": BEDROCK_GUARDRAIL_VERSION,
            }
        try:
            response = self.client.converse(**request_params)
            text = _extract_text(response)
            parsed = _parse_json_object(text)
            ordered_indices = parsed.get("ranked_indices", []) if isinstance(parsed, dict) else []
        except (ClientError, BotoCoreError, KeyError, TypeError, ValueError):
            return []

        chosen = []
        seen = set()
        by_index = {i: website for i, website in enumerate(websites, start=1)}
        for index in ordered_indices:
            try:
                numeric = int(index)
            except (TypeError, ValueError):
                continue
            if numeric in seen or numeric not in by_index:
                continue
            seen.add(numeric)
            chosen.append(by_index[numeric])

        for position, website in enumerate(websites, start=1):
            if position in seen:
                continue
            chosen.append(website)
        return chosen


def rerank_websites(prompt, search_query, websites, limit, requested_date_text=""):
    return ResultRanker().rerank(prompt, search_query, websites, limit, requested_date_text)


def _heuristic_rank(prompt, search_query, websites, requested_date_text=""):
    scored = []
    query_terms = _tokenize(f"{prompt} {search_query}")
    is_ipl = "ipl" in query_terms or ("indian" in query_terms and "premier" in query_terms)
    wants_specific_match = any(term in query_terms for term in {"today", "tomorrow", "yesterday", "playing", "live"})
    requested_date = parse_resolved_date(requested_date_text)

    for position, website in enumerate(websites):
        title = (website.get("title") or "").lower()
        url = website.get("url") or ""
        parsed = urlparse(url)
        domain = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        haystack = _tokenize(f"{title} {domain} {path}")

        score = 0
        score += max(0, 20 - position)
        score += len(query_terms.intersection(haystack)) * 4

        if is_ipl:
            score += PREFERRED_IPL_DOMAINS.get(domain, 0)

        if any(hint in title or hint in path for hint in GOOD_PATH_HINTS):
            score += 5

        if wants_specific_match and any(hint in title or hint in path for hint in DATE_SPECIFIC_HINTS):
            score += 7

        if wants_specific_match and ("live" in title or "live" in path or "score" in title or "score" in path):
            score += 6

        if wants_specific_match and any(bad in title for bad in BAD_TITLE_HINTS):
            score -= 8

        if any(hint in title or hint in path for hint in BAD_PATH_HINTS):
            score -= 6

        if any(hint in title or hint in path for hint in RESTRICTED_HINTS):
            score -= 10

        if is_ipl and ("ipl" in title or "ipl" in domain or "ipl" in path):
            score += 3

        if requested_date:
            score += _score_requested_date_match(title, path, url, requested_date)

        scored.append((score, website))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [website for _, website in scored]


def _score_requested_date_match(title, path, url, requested_date):
    score = 0
    month_name = requested_date["month_name"].casefold()
    day_text = str(requested_date["day"])
    year_text = str(requested_date["year"])
    compact_expected = f"{requested_date['month']:02d}{requested_date['day']:02d}{requested_date['year']}"
    title_path = f"{title} {path}"

    if month_name in title_path and day_text in title_path and year_text in title_path:
        score += 12

    compact_dates = _extract_compact_dates(url)
    if compact_dates:
        if compact_expected in compact_dates:
            score += 12
        else:
            score -= 10

    if f"match-{requested_date['day']}" in path or f"match {requested_date['day']}" in title_path:
        if month_name not in title_path and compact_expected not in compact_dates:
            score -= 12

    return score


def _extract_compact_dates(text):
    return set(part for part in __import__('re').findall(r"\b(\d{8})\b", text or ""))


def _tokenize(text):
    cleaned = (text or "").replace("/", " ").replace("-", " ").replace(",", " ")
    return {token for token in cleaned.lower().split() if token}


def _extract_text(response):
    output = response.get("output", {})
    message = output.get("message", {})
    content = message.get("content", [])
    parts = []
    for item in content:
        text = item.get("text")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _parse_json_object(text):
    if not text:
        return None
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    import re
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
