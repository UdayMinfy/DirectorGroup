import logging

from bedrock_client import BedrockSummarizer
from browser_extractor import BrowserExtractor
from web_search import WebSearcher


LOGGER = logging.getLogger(__name__)


class WebResearchOrchestrator:
    def __init__(self):
        self.searcher = WebSearcher()
        self.extractor = BrowserExtractor()
        self.summarizer = BedrockSummarizer()

    def run(self, prompt):
        LOGGER.info("Incoming prompt: %s", prompt)

        search_results = self.searcher.search(prompt)
        return self._extract_and_summarize(prompt, search_results)

    def run_for_urls(self, prompt, urls):
        LOGGER.info("Incoming prompt for explicit URLs: %s", prompt)
        search_results = [{"title": url, "url": url} for url in urls]
        return self._extract_and_summarize(prompt, search_results)

    def _extract_and_summarize(self, prompt, search_results):
        extracted_sources = []

        for result in search_results:
            try:
                extracted = self.extractor.extract(result["url"])
                extracted_sources.append(extracted)
                LOGGER.info(
                    "Extracted content from %s using %s",
                    extracted["url"],
                    extracted["extraction_method"],
                )
            except Exception:
                LOGGER.exception("Failed to extract content from %s", result["url"])

        if not extracted_sources:
            raise RuntimeError("No source content could be extracted from the top search results.")

        summary = self.summarizer.summarize(prompt, extracted_sources)

        LOGGER.info("User prompt: %s", prompt)
        LOGGER.info("Summary output: %s", summary["summary"])

        return {
            "prompt": prompt,
            "answer": summary["summary"],
            "model_id": summary["model_id"],
            "usage": summary["usage"],
            "sources": [
                {
                    "title": source["title"],
                    "url": source["url"],
                    "extraction_method": source["extraction_method"],
                    "content_preview": source["content"][:500],
                }
                for source in extracted_sources
            ],
        }
