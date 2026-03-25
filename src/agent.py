import logging
from uuid import uuid4

from agentcore_browser_client import AgentCoreBrowserSession
from config import SEARCH_RESULT_LIMIT
from task_models import ExtractionTask
from tools import BrowseExtractTool, SearchWebTool, SummarizeTool


LOGGER = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self):
        self.search_tool = SearchWebTool()
        self.browse_extract_tool = BrowseExtractTool()
        self.summarize_tool = SummarizeTool()

    def run(self, prompt, urls=None):
        LOGGER.info("Agent received prompt: %s", prompt)

        if urls:
            websites = [{"title": url, "url": url} for url in urls]
            LOGGER.info("Agent will use %s explicit URLs", len(websites))
        else:
            websites = self.search_tool.run(prompt, SEARCH_RESULT_LIMIT)
            LOGGER.info("Agent search produced %s websites", len(websites))

        tasks = self._create_tasks(websites)
        extracted_sources = self._execute_tasks(tasks)

        if not extracted_sources:
            raise RuntimeError("The agent could not extract content from any websites.")

        summary = self.summarize_tool.run(prompt, extracted_sources)

        LOGGER.info("Agent produced final summary")
        return {
            "prompt": prompt,
            "answer": summary["summary"],
            "model_id": summary["model_id"],
            "usage": summary["usage"],
            "tasks": [self._serialize_task(task) for task in tasks],
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

    def _create_tasks(self, websites):
        tasks = []
        for website in websites:
            task = ExtractionTask(
                task_id=uuid4().hex[:8],
                title=website.get("title") or website["url"],
                url=website["url"],
            )
            tasks.append(task)

        LOGGER.info("Agent created %s extraction tasks", len(tasks))
        return tasks

    def _execute_tasks(self, tasks):
        extracted_sources = []

        with AgentCoreBrowserSession() as session:
            for task in tasks:
                LOGGER.info("Running extraction task %s for %s", task.task_id, task.url)
                task.status = "running"
                try:
                    extracted = self.browse_extract_tool.run_with_session(task.url, session)
                    task.status = "completed"
                    task.content = extracted["content"]
                    task.extraction_method = extracted["extraction_method"]
                    extracted_sources.append(extracted)
                    LOGGER.info(
                        "Completed task %s using %s",
                        task.task_id,
                        task.extraction_method,
                    )
                except Exception as error:
                    task.status = "failed"
                    task.error = str(error)
                    LOGGER.exception("Task %s failed for %s", task.task_id, task.url)

        return extracted_sources

    @staticmethod
    def _serialize_task(task):
        return {
            "task_id": task.task_id,
            "title": task.title,
            "url": task.url,
            "status": task.status,
            "extraction_method": task.extraction_method,
            "error": task.error,
        }
