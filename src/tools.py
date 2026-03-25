from bedrock_client import BedrockSummarizer
from browser_extractor import BrowserExtractor
from web_search import WebSearcher


class SearchWebTool:
    def __init__(self):
        self.searcher = WebSearcher()

    def run(self, query, limit):
        return self.searcher.search(query, limit=limit)


class BrowseExtractTool:
    def __init__(self):
        self.extractor = BrowserExtractor()

    def run(self, url):
        return self.extractor.extract(url)

    def run_with_session(self, url, session):
        return self.extractor.extract_with_session(url, session)


class SummarizeTool:
    def __init__(self):
        self.summarizer = BedrockSummarizer()

    def run(self, prompt, sources):
        return self.summarizer.summarize(prompt, sources)
