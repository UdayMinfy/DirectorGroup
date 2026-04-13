import json
import logging
import re
from urllib.parse import urlparse

import requests

from config import (
    MIN_SEARCH_RESULTS,
    REQUEST_TIMEOUT_SECONDS,
    SEARCH_API_PROVIDER,
    SEARCH_RESULT_LIMIT,
    TAVILY_API_KEY,
    TAVILY_INCLUDE_ANSWER,
)


LOGGER = logging.getLogger(__name__)


class WebSearcher:
    def search(self, query, limit=SEARCH_RESULT_LIMIT):
        if SEARCH_API_PROVIDER != "tavily":
            raise RuntimeError(f"Unsupported search provider: {SEARCH_API_PROVIDER}")

        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not configured.")

        desired_count = max(limit, MIN_SEARCH_RESULTS)
        collected = []
        seen_urls = set()

        for search_query in self._build_queries(query):
            batch = self._search_once(search_query, desired_count)
            added = 0
            for item in batch:
                url = item["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(item)
                added += 1
                if len(collected) >= desired_count:
                    break

            LOGGER.info(
                "Tavily query '%s' added %s new results; collected %s total",
                search_query,
                added,
                len(collected),
            )
            if len(collected) >= desired_count:
                break

        ordered = self._prioritize_domain_diversity(collected)
        final_items = ordered[:desired_count]
        LOGGER.info("Search returned %s results from Tavily", len(final_items))
        return final_items

    def _search_once(self, query, limit):
        response = requests.post(
            "https://api.tavily.com/search",
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": limit,
                    "include_answer": TAVILY_INCLUDE_ANSWER,
                    "include_raw_content": False,
                }
            ),
        )
        response.raise_for_status()

        payload = response.json()
        answer = self._scrub_pii(self._clean_text(payload.get("answer", "")))

        items = []
        for index, item in enumerate(payload.get("results", [])):
            url = item.get("url")
            title = self._clean_text(item.get("title")) or url
            snippet = self._scrub_pii(self._clean_text(item.get("content")))
            if not url:
                continue

            fallback_parts = []
            if index == 0 and answer:
                fallback_parts.append(f"Tavily answer:\n{answer}")
            if snippet:
                fallback_parts.append(f"Tavily snippet:\n{snippet}")

            items.append(
                {
                    "title": title,
                    "url": url,
                    "tavily_answer": answer if index == 0 else "",
                    "tavily_snippet": snippet,
                    "fallback_content": "\n\n".join(fallback_parts).strip(),
                }
            )

            if len(items) >= limit:
                break

        return items

    @staticmethod
    def _build_queries(query):
        base = " ".join((query or "").split()).strip()
        if not base:
            return [""]

        queries = [base]
        lowered = base.lower()

        if "ipl" in lowered:
            queries.extend(
                [
                    f"{base} scorecard result",
                    f"{base} cricbuzz espncricinfo",
                    f"{base} site:espncricinfo.com OR site:cricbuzz.com OR site:hindustantimes.com",
                    f"{base} site:ndtv.com OR site:indiatoday.in OR site:timesofindia.indiatimes.com",
                ]
            )
        else:
            queries.extend(
                [
                    f"{base} official result",
                    f"{base} latest update",
                ]
            )

        deduped = []
        seen = set()
        for item in queries:
            normalized = item.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(item)
        return deduped

    @staticmethod
    def _prioritize_domain_diversity(items):
        by_domain = {}
        for item in items:
            domain = urlparse(item.get("url") or "").netloc.casefold()
            by_domain.setdefault(domain, []).append(item)

        ordered = []
        while by_domain:
            exhausted = []
            for domain, domain_items in by_domain.items():
                ordered.append(domain_items.pop(0))
                if not domain_items:
                    exhausted.append(domain)
            for domain in exhausted:
                by_domain.pop(domain, None)
        return ordered

    @staticmethod
    def _clean_text(text):
        if not text:
            return ""

        cleaned = re.sub(r"<.*?>", " ", str(text))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _scrub_pii(text):
        return re.sub(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "[EMAIL REDACTED]",
            text or "",
        )
