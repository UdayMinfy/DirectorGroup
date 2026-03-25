import logging
import re

import requests
from bs4 import BeautifulSoup

from agentcore_browser_client import AgentCoreBrowserSession
from config import BROWSER_IDENTIFIER, MAX_PAGE_CHARS, MAX_SOURCE_CHARS, REQUEST_TIMEOUT_SECONDS


LOGGER = logging.getLogger(__name__)

NOISE_PATTERNS = (
    "accept cookies",
    "cookie policy",
    "privacy policy",
    "terms of service",
    "sign in",
    "sign up",
    "log in",
    "skip to content",
    "book a demo",
    "talk to sales",
    "watch on youtube",
    "pricing",
)


class BrowserExtractor:
    def extract(self, url):
        if BROWSER_IDENTIFIER:
            try:
                with AgentCoreBrowserSession() as session:
                    return self._extract_with_browser_session(url, session)
            except Exception:
                LOGGER.exception("Browser extraction failed for %s, falling back to HTTP parsing", url)

        return self._extract_with_requests(url)

    def extract_with_session(self, url, session):
        try:
            return self._extract_with_browser_session(url, session)
        except Exception:
            LOGGER.exception("Browser extraction failed for %s, falling back to HTTP parsing", url)
            return self._extract_with_requests(url)

    def _extract_with_browser_session(self, url, session):
        page = session.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_SECONDS * 1000)
            text = page.locator("body").inner_text(timeout=REQUEST_TIMEOUT_SECONDS * 1000)
            title = page.title()
        finally:
            page.close()

        return {
            "url": url,
            "title": title or url,
            "content": self._clean_content(text),
            "extraction_method": "agentcore_browser_playwright",
        }

    def _extract_with_requests(self, url):
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for node in soup(["script", "style", "noscript"]):
            node.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else url
        text = soup.get_text(" ", strip=True)

        return {
            "url": url,
            "title": title,
            "content": self._clean_content(text),
            "extraction_method": "http_fallback",
        }

    @staticmethod
    def _clean_content(text):
        raw = (text or "")[:MAX_PAGE_CHARS]
        segments = re.split(r"(?<=[.!?])\s+|\n+", raw)

        cleaned_segments = []
        seen = set()

        for segment in segments:
            normalized = " ".join(segment.split()).strip()
            if len(normalized) < 40:
                continue

            lowered = normalized.lower()
            if any(pattern in lowered for pattern in NOISE_PATTERNS):
                continue
            if normalized in seen:
                continue

            alpha_ratio = sum(char.isalpha() for char in normalized) / max(len(normalized), 1)
            if alpha_ratio < 0.6:
                continue

            seen.add(normalized)
            cleaned_segments.append(normalized)

            if len(" ".join(cleaned_segments)) >= MAX_SOURCE_CHARS:
                break

        condensed = " ".join(cleaned_segments)
        return condensed[:MAX_SOURCE_CHARS]
