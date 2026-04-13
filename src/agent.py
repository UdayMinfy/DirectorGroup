import logging
from uuid import uuid4

from agentcore_browser_client import AgentCoreBrowserSession
from chat_session_store import ChatSessionStore
from config import (
    EXTRACTION_SUCCESS_TARGET,
    MIN_SEARCH_RESULTS,
    SEARCH_CANDIDATE_LIMIT,
    SEARCH_RESULT_LIMIT,
    USE_BROWSER,
    USE_TAVILY_RESULT_ORDER,
)
from query_rewriter import build_query_context, rewrite_search_query
from result_ranker import rerank_websites
from task_models import ExtractionTask
from tools import BrowseExtractTool, IntentTool, KnowledgeAnswerTool, SearchWebTool, SummarizeTool, clean_response, is_math_expression


LOGGER = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self):
        self.intent_tool = IntentTool()
        self.knowledge_answer_tool = KnowledgeAnswerTool()
        self.search_tool = SearchWebTool()
        self.browse_extract_tool = BrowseExtractTool()
        self.summarize_tool = SummarizeTool()
        self.chat_session_store = ChatSessionStore()

    def run(self, prompt, urls=None, user_id="", session_id=""):
        LOGGER.info("Agent received prompt: %s", prompt)

        # Check if this is a math expression
        is_math, result = is_math_expression(prompt)
        if is_math:
            answer = str(result)
            self._store_session_turn(user_id, session_id, prompt, answer)
            return {
                "prompt": prompt,
                "answer": clean_response(answer),
                "model_id": "calculator",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "intent": {
                    "route": "calculator",
                    "reason": "Detected arithmetic expression",
                },
                "consensus": {},
                "fact_extractions": [],
                "tasks": [],
                "sources": [],
                "user_id": user_id,
                "session_id": session_id,
                "history_used": False,
            }

        effective_result_limit = max(SEARCH_RESULT_LIMIT, MIN_SEARCH_RESULTS)
        effective_candidate_limit = max(SEARCH_CANDIDATE_LIMIT, effective_result_limit)
        session_context = self._load_session_context(prompt, user_id, session_id)
        effective_prompt = session_context["effective_prompt"]
        history_usage = session_context["usage"]

        if urls:
            websites = [{"title": url, "url": url} for url in urls[:effective_result_limit]]
            LOGGER.info("Agent will use %s explicit URLs", len(websites))
            intent = {
                "route": "realtime",
                "reason": "Explicit URLs were provided, so browsing/extraction is required.",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        else:
            intent = self.intent_tool.run(effective_prompt)
            LOGGER.info("Intent route selected: %s (%s)", intent["route"], intent.get("reason", ""))

            if intent["route"] == "general":
                direct = self.knowledge_answer_tool.run(prompt, history_text=session_context["history_text"])
                self._store_session_turn(user_id, session_id, prompt, direct["answer"])
                LOGGER.info("Answered using model knowledge without browsing")
                return {
                    "prompt": prompt,
                    "answer": clean_response(direct["answer"]),
                    "model_id": direct["model_id"],
                    "usage": self._merge_usage(history_usage, intent.get("usage", {}), direct.get("usage", {})),
                    "intent": {
                        "route": intent["route"],
                        "reason": intent.get("reason", ""),
                    },
                    "consensus": {},
                    "fact_extractions": [],
                    "tasks": [],
                    "sources": [],
                    "user_id": user_id,
                    "session_id": session_id,
                    "history_used": session_context["history_used"],
                }

            query_context = build_query_context(prompt)
            search_query = rewrite_search_query(prompt)
            LOGGER.info("Agent rewritten search query: %s", search_query)
            candidates = self.search_tool.run(search_query, effective_candidate_limit)

            if USE_TAVILY_RESULT_ORDER:
                websites = candidates[:effective_result_limit]
                LOGGER.info(
                    "Using Tavily ranking order directly; selected %s websites",
                    len(websites),
                )
            else:
                websites = rerank_websites(
                    effective_prompt,
                    search_query,
                    candidates,
                    effective_result_limit,
                    requested_date_text=query_context.get("resolved_date") or "",
                )
                LOGGER.info(
                    "Agent search produced %s candidates and selected %s websites",
                    len(candidates),
                    len(websites),
                )

        tasks = self._create_tasks(websites, effective_result_limit)
        extracted_sources = self._execute_tasks(tasks, effective_result_limit)

        if not extracted_sources:
            extracted_sources = self._build_fallback_sources(tasks)

        if not extracted_sources:
            raise RuntimeError("The agent could not extract content from any websites.")

        summary = self.summarize_tool.run(effective_prompt, extracted_sources)
        self._store_session_turn(user_id, session_id, prompt, summary["summary"])

        LOGGER.info("Agent produced final summary")
        return {
            "prompt": prompt,
            "answer": clean_response(summary["summary"]),
            "model_id": summary["model_id"],
            "usage": self._merge_usage(history_usage, intent.get("usage", {}), summary.get("usage", {})),
            "intent": {
                "route": intent["route"],
                "reason": intent.get("reason", ""),
            },
            "consensus": summary.get("consensus", {}),
            "fact_extractions": summary.get("fact_extractions", []),
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
            "user_id": user_id,
            "session_id": session_id,
            "history_used": session_context["history_used"],
        }

    def _load_session_context(self, prompt, user_id, session_id):
        if not user_id or not session_id:
            return {
                "effective_prompt": prompt,
                "history_text": "",
                "history_used": False,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }

        history_text = self.chat_session_store.build_history_text(user_id, session_id)
        if not history_text:
            LOGGER.info("No prior messages found for session %s and user %s", session_id, user_id)
            return {
                "effective_prompt": prompt,
                "history_text": "",
                "history_used": False,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }

        effective_prompt = self._compose_prompt_with_history(prompt, history_text)
        LOGGER.info("Loaded prior messages for session %s and user %s", session_id, user_id)
        return {
            "effective_prompt": effective_prompt,
            "history_text": history_text,
            "history_used": True,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    @staticmethod
    def _compose_prompt_with_history(prompt, history_text):
        return "\n\n".join(
            [
                "Previous chat messages:",
                history_text,
                f"Latest user prompt: {prompt}",
            ]
        )

    def _create_tasks(self, websites, result_limit):
        tasks = []
        for website in websites[:result_limit]:
            task = ExtractionTask(
                task_id=uuid4().hex[:8],
                title=website.get("title") or website["url"],
                url=website["url"],
                fallback_content=(
                    website.get("fallback_content")
                    or website.get("content")
                    or website.get("snippet")
                    or ""
                ),
            )
            tasks.append(task)

        LOGGER.info("Agent created %s extraction tasks", len(tasks))
        return tasks

    def _execute_tasks(self, tasks, result_limit):
        if not USE_BROWSER:
            raise RuntimeError("USE_BROWSER is disabled; browser-only extraction requires USE_BROWSER=true.")

        return self._execute_tasks_with_browser(tasks, result_limit)

    def _execute_tasks_with_browser(self, tasks, result_limit):
        extracted_sources = []
        attempts = 0

        with AgentCoreBrowserSession() as session:
            for task in tasks:
                if attempts >= result_limit:
                    break
                if len(extracted_sources) >= EXTRACTION_SUCCESS_TARGET:
                    LOGGER.info(
                        "Reached extraction success target with %s websites; stopping early",
                        len(extracted_sources),
                    )
                    break

                attempts += 1
                LOGGER.info(
                    "Running extraction task %s for %s (attempt %s/%s)",
                    task.task_id,
                    task.url,
                    attempts,
                    result_limit,
                )
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
                    LOGGER.warning("Task %s failed for %s: %s", task.task_id, task.url, error)

        return extracted_sources

    @staticmethod
    def _build_fallback_sources(tasks):
        fallback_sources = []
        for task in tasks:
            if task.fallback_content:
                fallback_sources.append(
                    {
                        "url": task.url,
                        "title": task.title,
                        "content": task.fallback_content,
                        "extraction_method": "search_result_fallback",
                    }
                )

        return fallback_sources

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

    @staticmethod
    def _merge_usage(*usage_objects):
        total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for usage in usage_objects:
            if not isinstance(usage, dict):
                continue
            total["input_tokens"] += int(usage.get("input_tokens") or 0)
            total["output_tokens"] += int(usage.get("output_tokens") or 0)
            total["total_tokens"] += int(usage.get("total_tokens") or 0)
        return total

    def _store_session_turn(self, user_id, session_id, prompt, answer):
        if not user_id or not session_id:
            return

        self.chat_session_store.append_messages(
            user_id,
            session_id,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
        )
