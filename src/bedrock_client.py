import json
import logging
import re
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_MODEL_ID, BEDROCK_REGION
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
        selected_sources, fact_extractions = self._prepare_sources_for_summary(sources, requested_date)
        consensus = self._build_fast_consensus(selected_sources, requested_date)
        prompt = self._build_summary_prompt(user_prompt, selected_sources, requested_date)
        output_text, usage = self._invoke_text(prompt, "summarize")

        LOGGER.info("Nova Pro summary generated")
        LOGGER.info(
            "Summary token usage - input: %s, output: %s, total: %s",
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
        )

        return {
            "model_id": BEDROCK_MODEL_ID,
            "summary": output_text,
            "usage": usage,
            "consensus": consensus,
            "fact_extractions": fact_extractions,
            "raw_prompt": prompt,
        }

    def _invoke_text(self, prompt, operation_name):
        try:
            response = self.client.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            )
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Bedrock %s call failed", operation_name)
            raise RuntimeError(f"Bedrock {operation_name} call failed: {error}") from error

        usage = response.get("usage", {})
        return self._extract_text(response), {
            "input_tokens": usage.get("inputTokens") or 0,
            "output_tokens": usage.get("outputTokens") or 0,
            "total_tokens": usage.get("totalTokens") or 0,
        }

    def _prepare_sources_for_summary(self, sources, requested_date):
        prepared = []
        matched = []

        for index, source in enumerate(sources, start=1):
            detected_date = self._detect_source_date({}, source)
            relevance = "high"
            if requested_date and detected_date and not self._same_date(detected_date, requested_date):
                relevance = "low"

            item = {
                "source_index": index,
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "content": source.get("content", ""),
                "detected_date": detected_date["text"] if detected_date else "",
                "answer_relevance": relevance,
            }
            prepared.append(item)
            if relevance != "low":
                matched.append(item)

        selected = matched or prepared
        fact_extractions = [
            {
                "source_index": item["source_index"],
                "title": item["title"],
                "url": item["url"],
                "date": item["detected_date"],
                "answer_relevance": item["answer_relevance"],
            }
            for item in selected
        ]
        return selected, fact_extractions

    @staticmethod
    def _build_fast_consensus(selected_sources, requested_date):
        supporting_sources = [item.get("url", "") for item in selected_sources if item.get("url")]
        return {
            "date": requested_date["text"] if requested_date else "",
            "supporting_sources": supporting_sources,
            "source_count": len(supporting_sources),
        }

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
    def _build_summary_prompt(user_prompt, selected_sources, requested_date):
        source_blocks = []
        for item in selected_sources:
            source_blocks.append(
                "\n".join(
                    [
                        f"Source index: {item.get('source_index')}",
                        f"Title: {item.get('title')}",
                        f"URL: {item.get('url')}",
                        f"Detected date: {item.get('detected_date')}",
                        f"Content: {item.get('content')}",
                    ]
                )
            )

        return "\n\n".join(
            [
                "You are a research assistant.",
                "Answer only from the sources below.",
                "Prefer facts repeated across multiple relevant sources.",
                "If a requested date is provided, prioritize sources that match that date.",
                "If there are conflicts, mention them briefly instead of silently picking one side.",
                "Keep the answer clear and helpful.",
                "End with a Sources section listing only the URLs actually used.",
                f"User question: {user_prompt}",
                f"Requested date: {requested_date['text'] if requested_date else ''}",
                f"Sources: {json.dumps(source_blocks, ensure_ascii=True)}",
            ]
        )

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
