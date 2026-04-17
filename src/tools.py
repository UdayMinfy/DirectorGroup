from bedrock_client import BedrockSummarizer
from browser_extractor import BrowserExtractor
from intent_router import BedrockIntentRouter, BedrockKnowledgeResponder
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


class SummarizeTool:
    def __init__(self):
        self.summarizer = BedrockSummarizer()

    def run(self, prompt, sources):
        return self.summarizer.summarize(prompt, sources)


class IntentTool:
    def __init__(self):
        self.router = BedrockIntentRouter()

    def decide_context_strategy(self, prompt):
        return self.router.decide_context_strategy(prompt)


class KnowledgeAnswerTool:
    def __init__(self):
        self.responder = BedrockKnowledgeResponder()

    def run(self, prompt, history_text=""):
        return self.responder.answer(prompt, history_text=history_text)
