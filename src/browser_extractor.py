import logging
import re

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

BLOCKED_PATTERNS = (
    "access denied",
    "forbidden",
    "request blocked",
    "temporarily unavailable",
    "enable javascript",
    "captcha",
    "bot verification",
    "press and hold",
    "cloudflare",
    "attention required",
)


class BrowserExtractor:
    def extract(self, url):
        if not BROWSER_IDENTIFIER:
            raise RuntimeError("BROWSER_IDENTIFIER/BROWSER_ID is not configured.")

        with AgentCoreBrowserSession() as session:
            return self._extract_with_browser_session(url, session)

    def _extract_with_browser_session(self, url, session):
        page = session.new_page()
        try:
            self._prepare_page(page)
            text = self._load_page_text(page, url)
            title = page.title()
        finally:
            page.close()

        if self._looks_blocked(text, title):
            raise RuntimeError(f"Blocked or anti-bot page detected for {url}")

        cleaned = self._clean_content(text)
        if len(cleaned) < 200:
            raise RuntimeError(f"Insufficient extractable content for {url}")

        return {
            "url": url,
            "title": title or url,
            "content": cleaned,
            "extraction_method": "agentcore_browser_playwright",
        }

    @staticmethod
    def _prepare_page(page):
        page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4] });
            """
        )

    def _load_page_text(self, page, url):
        # Increased timeout to 15 seconds for better reliability with slow websites
        page.goto(url, wait_until="domcontentloaded", timeout=15000)  # 15 seconds
        page.wait_for_timeout(250)
        self._expand_page(page)

        try:
            page.wait_for_function(
                "document.body && document.body.innerText && document.body.innerText.length > 500",
                timeout=1500,
            )
        except Exception:
            pass

        return page.evaluate(
            """
            () => {
                const selectors = ['main', 'article', '[role="main"]', '.content', '#content', 'body'];
                const parts = [];
                for (const selector of selectors) {
                    const node = document.querySelector(selector);
                    if (!node || !node.innerText) continue;
                    const value = node.innerText.trim();
                    if (value) parts.push(value);
                }
                return parts.join('\\n\\n');
            }
            """
        )

    @staticmethod
    def _expand_page(page):
        try:
            page.evaluate(
                """
                () => {
                    window.scrollTo(0, document.body.scrollHeight);
                    window.scrollTo(0, 0);
                }
                """
            )
        except Exception:
            pass

    @staticmethod
    def _looks_blocked(text, title=""):
        combined = f"{title}\n{text}".lower()
        return any(pattern in combined for pattern in BLOCKED_PATTERNS)

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
