import logging
import json

import requests

from config import REQUEST_TIMEOUT_SECONDS, SEARCH_API_PROVIDER, SEARCH_RESULT_LIMIT, TAVILY_API_KEY


LOGGER = logging.getLogger(__name__)


class WebSearcher:
    def search(self, query, limit=SEARCH_RESULT_LIMIT):
        if SEARCH_API_PROVIDER != "tavily":
            raise RuntimeError(f"Unsupported search provider: {SEARCH_API_PROVIDER}")

        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not configured.")

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
                    "include_answer": False,
                    "include_raw_content": False,
                }
            ),
        )
        response.raise_for_status()

        payload = response.json()
        items = []
        for item in payload.get("results", []):
            url = item.get("url")
            title = item.get("title") or url
            if not url:
                continue
            items.append({"title": title, "url": url})
            if len(items) >= limit:
                break

        LOGGER.info("Search returned %s results from Tavily", len(items))
        return items
