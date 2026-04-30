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

        # Log character counts only (detailed content previews commented out for performance)
        LOGGER.info("Extracted content from %s - raw: %d chars, cleaned: %d chars", url, len(text), len(cleaned))
        # LOGGER.info("Extracted raw content from %s (%d chars): %s", url, len(text), text[:1500])
        # LOGGER.info("Extracted cleaned content from %s (%d chars): %s", url, len(cleaned), cleaned[:1500])

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
        # Load initial page
        page.goto(url, wait_until="domcontentloaded", timeout=8000)
        page.wait_for_timeout(500)

        # Wait for basic content
        try:
            page.wait_for_function(
                "document.body && document.body.innerText && document.body.innerText.length > 200",
                timeout=2000,
            )
        except Exception:
            pass

        # Extract initial content
        initial_text = self._extract_page_content(page)
        
        # Return initial content without navigating to linked pages (optimization)
        LOGGER.info("Extracted initial content (%d chars)", len(initial_text))
        return initial_text[:MAX_PAGE_CHARS]

    def _extract_page_content(self, page):
        return page.evaluate(
            """
            () => {
                // Priority selectors for main content
                const mainSelectors = [
                    'main',
                    'article',
                    '[role="main"]',
                    '.post-content',
                    '.entry-content',
                    '.content',
                    '#main-content',
                    '#content',
                    '.main-content'
                ];

                // Try main content selectors first
                for (const selector of mainSelectors) {
                    const element = document.querySelector(selector);
                    if (element && element.innerText && element.innerText.trim().length > 500) {
                        return element.innerText.trim();
                    }
                }

                // Fallback to body but exclude noise
                const body = document.body;
                if (!body) return '';

                const noiseSelectors = [
                    'nav', 'header', 'footer', '.nav', '.header', '.footer',
                    '.sidebar', '.advertisement', '.ads', '.social-share',
                    '.comments', '.related-posts', '.newsletter'
                ];

                noiseSelectors.forEach(selector => {
                    const elements = body.querySelectorAll(selector);
                    elements.forEach(el => el.remove());
                });

                const text = body.innerText || '';
                return text.trim();
            }
            """
        )

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
            # Step 1: Remove extra spaces
            normalized = " ".join(segment.split()).strip()
            
            # Skip empty segments
            if not normalized:
                continue

            # Step 2: Check for noise patterns
            lowered = normalized.lower()
            if any(pattern in lowered for pattern in NOISE_PATTERNS):
                continue
            
            # Step 3: Skip duplicates
            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned_segments.append(normalized)

            if len(" ".join(cleaned_segments)) >= MAX_SOURCE_CHARS:
                break

        condensed = " ".join(cleaned_segments)
        return condensed[:MAX_SOURCE_CHARS]
