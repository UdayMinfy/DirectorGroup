import json
import logging
import re
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import AGENT_GUIDELINES, BEDROCK_GUARDRAIL_ID,  BEDROCK_GUARDRAIL_VERSION, BEDROCK_MODEL_ID, BEDROCK_REGION
from query_rewriter import build_query_context, parse_resolved_date


LOGGER = logging.getLogger(__name__)
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


class BedrockSummarizer:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    def summarize(self, user_prompt, sources):
        query_context = build_query_context(user_prompt)
        requested_date = parse_resolved_date(query_context.get("resolved_date") or "")
        fact_extractions = []
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for index, source in enumerate(sources, start=1):
            extracted_facts, usage = self._extract_source_facts(index, user_prompt, source, requested_date)
            fact_extractions.append(extracted_facts)
            self._add_usage(total_usage, usage)

        consensus = self._merge_facts(fact_extractions)
        system_prompt, user_content = self._build_consensus_prompt(user_prompt, consensus, fact_extractions)
        output_text, usage = self._invoke_text(user_content, "summarize", system_prompt=system_prompt)
        self._add_usage(total_usage, usage)

        LOGGER.info("Nova Pro summary generated")
        LOGGER.info(
            "Summary token usage - input: %s, output: %s, total: %s",
            total_usage["input_tokens"],
            total_usage["output_tokens"],
            total_usage["total_tokens"],
        )

        return {
            "model_id": BEDROCK_MODEL_ID,
            "summary": output_text,
            "usage": total_usage,
            "consensus": consensus,
            "fact_extractions": fact_extractions,
            "raw_prompt": f"{system_prompt}\n\n{user_content}",
        }

    def _invoke_text(self, prompt, operation_name, system_prompt=None):
        request_params = {
            "modelId": BEDROCK_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
        }
        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]
        if BEDROCK_GUARDRAIL_ID and BEDROCK_GUARDRAIL_VERSION:
            request_params["guardrailConfig"] = {
                "guardrailIdentifier": BEDROCK_GUARDRAIL_ID,
                "guardrailVersion": BEDROCK_GUARDRAIL_VERSION,
            }
        try:
            response = self.client.converse(**request_params)
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Bedrock %s call failed", operation_name)
            raise RuntimeError(f"Bedrock {operation_name} call failed: {error}") from error

        if response.get("stopReason") == "guardrail_intervened":
            LOGGER.warning("Guardrail blocked %s response", operation_name)
            raise RuntimeError(f"Guardrail blocked {operation_name} response")

        usage = response.get("usage", {})
        return self._extract_text(response), {
            "input_tokens": usage.get("inputTokens") or 0,
            "output_tokens": usage.get("outputTokens") or 0,
            "total_tokens": usage.get("totalTokens") or 0,
        }

    def _extract_source_facts(self, index, user_prompt, source, requested_date):
        system_prompt = "\n\n".join(
            [
                "You extract structured facts from a single source for later consensus merging.",
                "Use only the source provided below.",
                "Return JSON only with these keys:",
                (
                    '{"source_index": 0, "title": "", "url": "", "event": "", "date": "", '
                    '"winner": "", "participants": [], "score": "", "highlights": [], '
                    '"answer_relevance": "high|medium|low", "confidence": "high|medium|low", '
                    '"conflicts": [], "raw_supporting_facts": []}'
                ),
                "If a field is unknown, use an empty string or empty list.",
            ]
        )
        user_content = "\n\n".join(
            [
                f"User question: {user_prompt}",
                f"Source index: {index}",
                f"Title: {source.get('title')}",
                f"URL: {source.get('url')}",
                f"Content: {source.get('content')}",
            ]
        )

        output_text, usage = self._invoke_text(user_content, "fact extraction", system_prompt=system_prompt)
        parsed = self._parse_json_object(output_text)

        if not isinstance(parsed, dict):
            LOGGER.warning("Fact extraction returned non-JSON for source %s", source.get("url"))
            parsed = {}

        facts = {
            "source_index": index,
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "event": self._clean_text(parsed.get("event")),
            "date": self._clean_text(parsed.get("date")),
            "winner": self._clean_text(parsed.get("winner")),
            "participants": self._normalize_list(parsed.get("participants")),
            "score": self._clean_text(parsed.get("score")),
            "highlights": self._normalize_list(parsed.get("highlights")),
            "answer_relevance": self._clean_text(parsed.get("answer_relevance")) or "low",
            "confidence": self._clean_text(parsed.get("confidence")) or "low",
            "conflicts": self._normalize_list(parsed.get("conflicts")),
            "raw_supporting_facts": self._normalize_list(parsed.get("raw_supporting_facts")),
        }
        return self._apply_requested_date_validation(facts, source, requested_date), usage

    def _apply_requested_date_validation(self, facts, source, requested_date):
        if not requested_date:
            return facts

        detected_date = self._detect_source_date(facts, source)
        if detected_date:
            facts["detected_date"] = detected_date["text"]
            if not self._same_date(detected_date, requested_date):
                facts["answer_relevance"] = "low"
                facts["confidence"] = "low"
                facts["conflicts"] = facts.get("conflicts", []) + [
                    f"Source date {detected_date['text']} does not match requested date {requested_date['text']}."
                ]
                return facts

        if self._looks_like_match_number_confusion(source, requested_date):
            facts["answer_relevance"] = "low"
            facts["confidence"] = "low"
            facts["conflicts"] = facts.get("conflicts", []) + [
                f"Match number {requested_date['day']} was treated as a date-like signal without matching {requested_date['text']}."
            ]

        return facts

    def _detect_source_date(self, facts, source):
        candidates = [
            facts.get("date", ""),
            source.get("title", ""),
            source.get("url", ""),
            source.get("content", "")[:1500],
        ]

        for candidate in candidates:
            parsed = self._extract_date_from_text(candidate)
            if parsed:
                return parsed
        return None

    def _extract_date_from_text(self, text):
        if not text:
            return None

        lowered = text.lower()
        month_pattern = re.compile(r"\b(" + "|".join(MONTH_LOOKUP.keys()) + r")\s+(\d{1,2})(?:,)?\s+(\d{4})\b", re.IGNORECASE)
        match = month_pattern.search(lowered)
        if match:
            month_name, day_text, year_text = match.groups()
            return {
                "text": f"{month_name.title()} {int(day_text)}, {year_text}",
                "month": MONTH_LOOKUP[month_name.lower()],
                "day": int(day_text),
                "year": int(year_text),
                "month_name": month_name.title(),
            }

        compact = re.search(r"\b(\d{2})(\d{2})(\d{4})\b", text)
        if compact:
            month, day, year = compact.groups()
            try:
                dt = datetime(int(year), int(month), int(day))
            except ValueError:
                return None
            return {
                "text": f"{dt.strftime('%B')} {dt.day}, {dt.year}",
                "month": dt.month,
                "day": dt.day,
                "year": dt.year,
                "month_name": dt.strftime('%B'),
            }

        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
        if iso_match:
            year, month, day = iso_match.groups()
            try:
                dt = datetime(int(year), int(month), int(day))
            except ValueError:
                return None
            return {
                "text": f"{dt.strftime('%B')} {dt.day}, {dt.year}",
                "month": dt.month,
                "day": dt.day,
                "year": dt.year,
                "month_name": dt.strftime('%B'),
            }

        return None

    @staticmethod
    def _same_date(left, right):
        return (
            left.get("year") == right.get("year")
            and left.get("month") == right.get("month")
            and left.get("day") == right.get("day")
        )

    @staticmethod
    def _looks_like_match_number_confusion(source, requested_date):
        title = (source.get("title") or "").lower()
        url = (source.get("url") or "").lower()
        target = requested_date["day"]
        has_match_number = f"match {target}" in title or f"match-{target}" in url
        has_requested_month = requested_date["month_name"].lower() in f"{title} {url}"
        return has_match_number and not has_requested_month

    @classmethod
    def _merge_facts(cls, fact_extractions):
        relevant = [
            item
            for item in fact_extractions
            if item.get("answer_relevance") in {"high", "medium"}
        ] or fact_extractions

        return {
            "event": cls._pick_consensus_value(relevant, "event"),
            "date": cls._pick_consensus_value(relevant, "date"),
            "winner": cls._pick_consensus_value(relevant, "winner"),
            "score": cls._pick_consensus_value(relevant, "score"),
            "participants": cls._merge_list_values(relevant, "participants"),
            "highlights": cls._merge_list_values(relevant, "highlights"),
            "conflicts": cls._merge_list_values(relevant, "conflicts"),
            "supporting_sources": [item.get("url", "") for item in relevant if item.get("url")],
            "source_count": len(relevant),
        }

    @classmethod
    def _pick_consensus_value(cls, items, key):
        ranked = {}
        for item in items:
            value = cls._clean_text(item.get(key))
            if not value:
                continue

            normalized = value.casefold()
            bucket = ranked.setdefault(
                normalized,
                {"value": value, "count": 0, "weight": 0},
            )
            bucket["count"] += 1
            bucket["weight"] += cls._confidence_weight(item.get("confidence"))

        if not ranked:
            return ""

        best = max(
            ranked.values(),
            key=lambda entry: (entry["count"], entry["weight"], len(entry["value"])),
        )
        return best["value"]

    @classmethod
    def _merge_list_values(cls, items, key):
        merged = []
        seen = set()

        for item in items:
            for value in cls._normalize_list(item.get(key)):
                normalized = value.casefold()
                if normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(value)

        return merged

    @staticmethod
    def _confidence_weight(confidence):
        return {"high": 3, "medium": 2, "low": 1}.get(confidence, 0)

    @staticmethod
    def _build_consensus_prompt(user_prompt, consensus, fact_extractions):
        system_prompt = "\n\n".join(
            [
                "You are a research assistant.",
                "Answer only from the merged evidence below.",
                "Combine consistent facts from multiple sources into one answer.",
                "Prefer facts supported by multiple relevant sources.",
                "If there are conflicts, mention them briefly instead of silently picking one side.",
                "Keep the answer clear and helpful.",
                "End with a Sources section listing only the URLs actually used.",
                AGENT_GUIDELINES,
            ]
        )
        user_content = "\n\n".join(
            [
                f"User question: {user_prompt}",
                f"Consensus facts: {json.dumps(consensus, ensure_ascii=True)}",
                f"Per-source extracted facts: {json.dumps(fact_extractions, ensure_ascii=True)}",
            ]
        )
        return system_prompt, user_content

    @staticmethod
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

    @classmethod
    def _parse_json_object(cls, text):
        if not text:
            return None

        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _clean_text(value):
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _normalize_list(cls, value):
        if not value:
            return []
        if isinstance(value, list):
            return [cls._clean_text(item) for item in value if cls._clean_text(item)]
        cleaned = cls._clean_text(value)
        return [cleaned] if cleaned else []

    @staticmethod
    def _add_usage(total_usage, usage):
        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
