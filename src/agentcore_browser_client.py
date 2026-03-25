import logging
import uuid

from bedrock_agentcore.tools.browser_client import BrowserClient
from playwright.sync_api import sync_playwright

from config import BROWSER_IDENTIFIER, BROWSER_REGION


LOGGER = logging.getLogger(__name__)


class AgentCoreBrowserSession:
    def __init__(self):
        self.client = BrowserClient(region=BROWSER_REGION)
        self.session_id = None
        self._playwright_manager = None
        self._playwright = None
        self._browser = None
        self._context = None

    def __enter__(self):
        if not BROWSER_IDENTIFIER:
            raise RuntimeError("BROWSER_IDENTIFIER is not configured.")

        self.session_id = self.client.start(
            identifier=BROWSER_IDENTIFIER,
            name=f"research-session-{uuid.uuid4().hex[:8]}",
            session_timeout_seconds=900,
        )
        LOGGER.info("Started AgentCore browser session %s", self.session_id)

        ws_url, headers = self.client.generate_ws_headers()
        self._playwright_manager = sync_playwright()
        self._playwright = self._playwright_manager.__enter__()
        self._browser = self._playwright.chromium.connect_over_cdp(ws_url, headers=headers)
        self._context = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._browser:
                self._browser.close()
            if self._playwright_manager:
                self._playwright_manager.__exit__(exc_type, exc_value, traceback)
            self.client.stop()
            LOGGER.info("Stopped AgentCore browser session %s", self.session_id)
        finally:
            self._context = None
            self._browser = None
            self._playwright = None
            self._playwright_manager = None
            self.session_id = None

    def new_page(self):
        if not self._context:
            raise RuntimeError("Browser context is not initialized.")
        return self._context.new_page()
