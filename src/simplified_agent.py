import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

from browser_extractor import BrowserExtractor
from chat_session_store import ChatSessionStore
from config import (
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    REQUEST_TIMEOUT_SECONDS,
    TAVILY_API_KEY,
)

LOGGER = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are an intent router for the latest user message.
Return JSON only:
{"read_history": true, "use_web_search": true, "reason": "short reason"}
Rules:
- Set read_history = true only if the latest message depends on earlier chat context, such as follow-ups, omitted subjects, references like "it", "that", "this", "him", "them", corrections, or "tell me more".
- Set read_history = false if the latest message is self-contained.
- Set use_web_search = true if answering correctly needs current, recent, changing, date-sensitive, live, or externally verifiable information.
Examples: latest news, recent events, match results, schedules, prices, weather, current positions, or anything likely to have changed.
- Set use_web_search = false if the query can be answered from stable built-in knowledge.
Examples: historical facts, general explanations, concepts, writing help, and timeless information.
Decision modes:
- both false: self-contained + stable knowledge question, greetings, simple conversation
- read_history true, use_web_search false: follow-up about prior chat, but no fresh facts needed
- read_history false, use_web_search true: self-contained question needing fresh/current facts
- both true: follow-up that also needs fresh/current facts
Bias:
- If the message contains words like "latest", "current", "today", "recent", "now", prefer use_web_search = true.
- For simple greetings, questions, or conversational messages, prefer use_web_search = false.
- Only set use_web_search = true when external verification or current information is clearly needed.
Do not answer the user. Output JSON only."""

ANSWER_SYSTEM_PROMPT = """You are a helpful, polished conversational assistant.
Answer style:
- Start with the direct answer.
- Add brief explanation or examples only when useful.
- Keep simple answers short and complex answers clear.
- Write naturally, confidently, and conversationally.
Use of knowledge:
- Use built-in knowledge for stable topics.
- If web content is provided, prioritize and synthesize information from it for current/recent topics.
- If retrieved content is provided, use it to answer questions about current events, scores, prices, or time-sensitive information.
- Be transparent about data sources when appropriate.
Formatting:
- Make the answer easy to scan.
- When helpful, use light visual markers like: ✓, •, 👍
- Do not overuse symbols or make the response noisy.
Follow-up:
- Ask a short follow-up only if needed for clarity or if it genuinely helps the user continue."""


class SimplifiedResearchAgent:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        self.chat_session_store = ChatSessionStore()
        self.browser_extractor = BrowserExtractor()

    def run(self, prompt, user_id="", session_id=""):
        LOGGER.info("Agent received prompt: %s", prompt)

        # Step 1: Determine intent (use_history, use_web_search)
        intent_result = self._determine_intent(prompt)
        use_history = intent_result["read_history"]
        use_web_search = intent_result["use_web_search"]

        LOGGER.info("Intent detection result: use_history=%s, use_web_search=%s, reason='%s'",
                   use_history, use_web_search, intent_result.get("reason", ""))

        # Step 2: Gather data based on intent
        history_text = ""
        web_content = ""

        if use_history:
            history_text = self.chat_session_store.build_history_text(user_id, session_id)
            if history_text:
                LOGGER.info("Loaded history for session %s (%d chars)", session_id, len(history_text))
            else:
                LOGGER.info("No history found for session %s", session_id)

        if use_web_search:
            web_content = self._fetch_web_content(prompt)
            if web_content:
                LOGGER.info("Fetched web content (%d chars)", len(web_content))
            else:
                LOGGER.info("No web content fetched")
        else:
            LOGGER.info("Skipping web search as per intent")

        # Step 3: Single LLM call with all data
        answer_result = self._generate_answer(prompt, history_text, web_content)

        # Step 4: Store the conversation turn
        self._store_session_turn(user_id, session_id, prompt, answer_result["answer"])

        return {
            "prompt": prompt,
            "answer": answer_result["answer"],
            "model_id": answer_result["model_id"],
            "usage": self._merge_usage(intent_result.get("usage", {}), answer_result.get("usage", {})),
            "intent": {
                "read_history": use_history,
                "use_web_search": use_web_search,
                "reason": intent_result.get("reason", ""),
            },
            "user_id": user_id,
            "session_id": session_id,
            "history_used": bool(history_text),
        }

    def _determine_intent(self, prompt):
        """Determine if we need history and/or web search."""
        # Quick check for simple conversational prompts
        prompt_lower = prompt.strip().lower()
        simple_prompts = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "ok", "okay", "yes", "no", "sure", "cool", "nice"]

        if prompt_lower in simple_prompts or len(prompt.strip()) < 10:
            LOGGER.info("Intent detection: Simple conversational prompt detected, skipping LLM call")
            return {
                "read_history": False,
                "use_web_search": False,
                "reason": "Simple conversational prompt",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "model_id": BEDROCK_MODEL_ID,
            }

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        full_prompt = "\n\n".join([
            "Decide whether the assistant should read previous session history and/or use web search.",
            "Return JSON only with keys: read_history, use_web_search, reason.",
            f"Current UTC date: {today}",
            f"User prompt: {prompt}",
        ])

        output_text, usage = self._invoke_bedrock(full_prompt, "intent detection", INTENT_SYSTEM_PROMPT)

        # Extract JSON from response (handle markdown-wrapped JSON)
        json_text = self._extract_json_from_response(output_text)

        try:
            parsed = json.loads(json_text)
            result = {
                "read_history": bool(parsed.get("read_history", False)),
                "use_web_search": bool(parsed.get("use_web_search", False)),  # Changed default to False
                "reason": parsed.get("reason", "Intent detection completed"),
                "usage": usage,
                "model_id": BEDROCK_MODEL_ID,
            }
            LOGGER.info("LLM Intent detection: read_history=%s, use_web_search=%s, reason='%s'",
                       result["read_history"], result["use_web_search"], result["reason"])
            return result
        except json.JSONDecodeError as e:
            LOGGER.warning("Intent detection: Failed to parse LLM response: %s. Raw response: '%s'", str(e), output_text[:500])
            return {
                "read_history": False,
                "use_web_search": False,
                "reason": "Failed to parse intent response; defaulting to no history/web search",
                "usage": usage,
                "model_id": BEDROCK_MODEL_ID,
            }

    @staticmethod
    def _is_content_url(url):
        if not url:
            return False

        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
            path = (parsed.path or "").lower()
        except Exception:
            return False

        blocked_domains = {
            "youtube.com",
            "youtu.be",
            "vimeo.com",
            "dailymotion.com",
            "twitter.com",
            "facebook.com",
            "instagram.com",
            "tiktok.com",
        }

        if any(domain in hostname for domain in blocked_domains):
            return False

        if path.startswith("/watch") or "/watch" in path or path.startswith("/video") or "/video/" in path or "/embed" in path:
            return False

        return True

    def _fetch_web_content(self, prompt):
        """Fetch content from max 2 successful URLs using Tavily (try up to 10 URLs)."""
        if not TAVILY_API_KEY:
            LOGGER.warning("TAVILY_API_KEY not configured, skipping web search")
            return ""

        try:
            # Search with Tavily - get top 10 URLs
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": prompt,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                    "max_results": 10,  # Increased from 2 to 10
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            search_results = response.json()

            urls = [result["url"] for result in search_results.get("results", [])[:10]]
            LOGGER.info("Tavily returned %d URLs for query: %s", len(urls), prompt[:50])
            LOGGER.info("Tavily URLs: %s", urls)

            filtered_urls = [url for url in urls if self._is_content_url(url)]
            LOGGER.info("Filtered URLs: %d out of %d", len(filtered_urls), len(urls))
            LOGGER.info("Filtered URLs: %s", filtered_urls)

            if not filtered_urls:
                return ""

            # Try scraping all URLs in parallel, stop after 2 successful extractions
            successful_contents = []
            attempted_count = 0
            successful_count = 0

            with ThreadPoolExecutor(max_workers=10) as executor:  # Increased workers for 10 URLs
                future_to_url = {executor.submit(self._extract_url_content, url): url for url in filtered_urls}

                for future in as_completed(future_to_url):
                    attempted_count += 1
                    url = future_to_url[future]

                    try:
                        content = future.result()
                        if content:
                            successful_contents.append(content)
                            successful_count += 1
                            LOGGER.info("✅ Successfully extracted content from %s (%d/%d successful)",
                                       url, successful_count, attempted_count)

                            # Stop after 2 successful extractions
                            if successful_count >= 2:
                                LOGGER.info("Reached 2 successful extractions, cancelling remaining tasks")
                                # Cancel remaining futures
                                for remaining_future in future_to_url:
                                    if not remaining_future.done():
                                        remaining_future.cancel()
                                break
                        else:
                            LOGGER.info("❌ Failed to extract content from %s", url)

                    except Exception as e:
                        LOGGER.warning("Exception extracting from %s: %s", url, e)

            LOGGER.info("Web scraping summary: %d URLs attempted, %d successful extractions",
                       attempted_count, successful_count)

            return "\n\n".join(successful_contents)

        except Exception as e:
            LOGGER.warning("Web search failed: %s", e)
            return ""

    def _extract_url_content(self, url):
        """Extract clean content from a URL."""
        try:
            result = self.browser_extractor.extract(url)
            content = result.get("content", "")
            return f"Content from {url}:\n{content}" if content else ""
        except Exception as e:
            LOGGER.warning("Failed to extract content from %s: %s", url, e)
            return ""

    def _generate_answer(self, prompt, history_text="", web_content=""):
        """Generate final answer using single LLM call."""
        prompt_parts = []

        if web_content:
            prompt_parts.append("Use the provided web content to answer questions about current events, scores, or recent information.")
        else:
            prompt_parts.append("Use built-in model knowledge to answer the question.")

        if history_text:
            prompt_parts.append(f"Previous chat messages:\n{history_text}")

        if web_content:
            prompt_parts.append(f"Retrieved web content:\n{web_content}")

        prompt_parts.append(f"Latest user prompt: {prompt}")

        full_prompt = "\n\n".join(prompt_parts)

        answer_text, usage = self._invoke_bedrock(full_prompt, "answer generation", ANSWER_SYSTEM_PROMPT)

        return {
            "answer": answer_text,
            "usage": usage,
            "model_id": BEDROCK_MODEL_ID,
        }

    def _invoke_bedrock(self, prompt, operation_name, system_prompt):
        """Invoke Bedrock model."""
        try:
            response = self.client.converse(
                modelId=BEDROCK_MODEL_ID,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            )

            usage = response.get("usage", {})
            output_text = self._extract_text(response)

            return output_text, {
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            }

        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Bedrock %s call failed", operation_name)
            raise RuntimeError(f"Bedrock {operation_name} call failed: {error}") from error

    @staticmethod
    def _extract_text(response):
        """Extract text from Bedrock response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        parts = []
        for item in content:
            text = item.get("text")
            if text:
                parts.append(text)
        return "".join(parts).strip()

    @staticmethod
    def _extract_json_from_response(text):
        """Extract JSON from LLM response, handling markdown code blocks."""
        if not text:
            return text

        # Remove markdown code block wrappers
        text = text.strip()

        # Handle ```json ... ``` format
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()  # Remove ```json and ```
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()  # Remove generic ``` wrappers

        # Try to find JSON object if there are multiple parts
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return text

    @staticmethod
    def _merge_usage(*usages):
        """Merge usage dictionaries."""
        total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for usage in usages:
            if isinstance(usage, dict):
                total["input_tokens"] += usage.get("input_tokens", 0)
                total["output_tokens"] += usage.get("output_tokens", 0)
                total["total_tokens"] += usage.get("total_tokens", 0)
        return total

    def _store_session_turn(self, user_id, session_id, prompt, answer):
        """Store conversation turn in DynamoDB."""
        if not user_id or not session_id:
            return

        try:
            messages = [
                {"role": "user", "content": prompt, "timestamp": datetime.now(timezone.utc).isoformat()},
                {"role": "assistant", "content": answer, "timestamp": datetime.now(timezone.utc).isoformat()},
            ]
            self.chat_session_store.append_messages(user_id, session_id, messages)
        except Exception as e:
            LOGGER.warning("Failed to store session turn: %s", e)