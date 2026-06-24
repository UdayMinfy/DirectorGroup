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
    LITELLM_TOKENIZER_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    TAVILY_API_KEY,
)

try:
    from litellm import token_counter
except ImportError:
    token_counter = None

LOGGER = logging.getLogger(__name__)



INTENT_SYSTEM_PROMPT = """CRITICAL INSTRUCTION:
You will receive conversation history and a current user message. Your task is to determine if the current message references or depends on the conversation history.

Check:
- Pronouns (it, that, this, them, him, her, they) referring to earlier entities.
- Follow-ups (tell me more, continue, what about, how about, elaborate, and that).
- Whether the message would make NO SENSE without the conversation history.

Return JSON only:
{"read_history": bool, "use_web_search": bool, "reason": "short"}

Rules:
- read_history=true if the message depends on earlier topics, entities, questions, pronouns, or follow-up context.
- read_history=false if the message is fully self-contained.
- use_web_search=true only for current, changing, recent, or externally verifiable information (latest, today, now, recent, news, weather, prices, schedules, results, current events).
- use_web_search=false for stable knowledge, historical facts, concepts, explanations, or timeless information.

Modes:
false,false = self-contained + stable knowledge
true,false = history only
false,true = web only
true,true = history + web

Output JSON only."""



SEARCH_QUERY_SYSTEM_PROMPT = """Rewrite the user's follow-up into a standalone web search query using the previous two messages.

Rules:
- Resolve references and pronouns from context.
- Return only the query.
- No quotes or explanations.
- Keep it concise (5-12 words).
- Make it specific enough for web search."""


#ANSWER_SYSTEM_PROMPT = """You are a helpful assistant.
#- Answer directly and concisely.
#- Explain only when necessary.
#- Use bullets for complex answers.
#- Prefer retrieved content when provided.
#- Use **bold** for key points and suggest relevant follow-ups when useful."""

ANSWER_SYSTEM_PROMPT = """
Answer briefly. Use 1-3 sentences or bullets. Expand only if asked. Prefer retrieved content.
"""


def _sse(event_type, data):
    """Format a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class SimplifiedResearchAgent:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        self.chat_session_store = ChatSessionStore()
        self.browser_extractor = BrowserExtractor()

    def run_stream(self, prompt, user_id="", session_id=""):
        """Run the agent and yield SSE-formatted events for streaming."""

        # Step 1: Fetch last 2 messages for intent context
        last_two_messages = self._get_last_two_messages(user_id, session_id)
        if last_two_messages:
            LOGGER.info("Last 2 messages for intent (first 40 chars): %s", last_two_messages[:40])

        # Step 2: Determine intent using last 2 messages + current prompt
        intent_result = self._determine_intent(prompt, last_two_messages)
        use_history = intent_result["read_history"]
        use_web_search = intent_result["use_web_search"]
        LOGGER.info("Intent: use_history=%s, use_web_search=%s", use_history, use_web_search)

        # Step 2: Gather data
        history_text = ""
        web_content = ""
        search_query = prompt

        if use_history:
            history_text = self.chat_session_store.build_history_text(user_id, session_id)
            if history_text:
                LOGGER.info("Chat history retrieved (first 100 chars): %s", history_text[:100])

        if use_web_search:
            yield _sse("status", {"message": "Searching the web..."})
            if use_history and history_text:
                search_query = self._build_search_query(history_text, prompt)
            web_content = self._fetch_web_content(search_query)
            if web_content:
                yield _sse("status", {"message": "Analysing results..."})

        # Step 3: Stream answer from Bedrock
        full_answer = ""
        # Start with intent detection token usage
        usage = intent_result.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        try:
            for chunk, final_usage in self._stream_answer(prompt, history_text, web_content):
                if chunk:
                    full_answer += chunk
                    yield _sse("chunk", {"text": chunk})
                if final_usage:
                    # Add answer generation tokens to intent detection tokens
                    usage = {
                        "input_tokens": usage.get("input_tokens", 0) + final_usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0) + final_usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0) + final_usage.get("total_tokens", 0),
                    }
        except Exception:
            LOGGER.exception("Streaming answer generation failed")
            yield _sse("error", {"message": "An unexpected error occurred. Please try again."})
            return

        # Step 4: Store conversation turn
        self._store_session_turn(user_id, session_id, prompt, full_answer)

        # Step 5: Track usage
        budget_view = None
        try:
            from budget_service import TokenBudgetService
            budget_view = TokenBudgetService().track_usage(
                user_email=user_id,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        except Exception as e:
            LOGGER.warning("Failed to track usage: %s", e)

        # Step 6: Send final done event
        done_event = {
            "session_id": session_id,
            "usage": usage,
            "history_used": bool(history_text),
            "search_query_used": search_query if use_web_search else None,
        }
        
        # Add updated budget information if available
        if budget_view:
            done_event["consumed_tokens_daily"] = budget_view.get("consumed_tokens_daily", 0)
            done_event["consumed_tokens_monthly"] = budget_view.get("consumed_tokens_monthly", 0)
        
        yield _sse("done", done_event)

    def run(self, prompt, user_id="", session_id=""):
        LOGGER.info("Agent received prompt: %s", prompt)

        # Fetch last 2 messages for intent context
        last_two_messages = self._get_last_two_messages(user_id, session_id)
        if last_two_messages:
            LOGGER.info("Last 2 messages for intent (first 40 chars): %s", last_two_messages[:40])

        intent_result = self._determine_intent(prompt, last_two_messages)
        use_history = intent_result["read_history"]
        use_web_search = intent_result["use_web_search"]

        LOGGER.info("Intent detection result: use_history=%s, use_web_search=%s, reason='%s'",
                    use_history, use_web_search, intent_result.get("reason", ""))

        history_text = ""
        web_content = ""
        search_query = prompt

        if use_history:
            history_text = self.chat_session_store.build_history_text(user_id, session_id)
            if history_text:
                LOGGER.info("Loaded history for session %s (%d chars)", session_id, len(history_text))
                LOGGER.info("Chat history retrieved (first 100 chars): %s", history_text[:100])
            else:
                LOGGER.info("No history found for session %s", session_id)

        if use_web_search:
            if use_history and history_text:
                search_query = self._build_search_query(history_text, prompt)
                LOGGER.info("Rewritten search query: %s", search_query)
            web_content = self._fetch_web_content(search_query)
            if web_content:
                LOGGER.info("Fetched web content (%d chars)", len(web_content))
            else:
                LOGGER.info("No web content fetched")
        else:
            LOGGER.info("Skipping web search as per intent")

        answer_result = self._generate_answer(prompt, history_text, web_content)
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
            "search_query_used": search_query if use_web_search else None,
        }

    def _get_last_two_messages(self, user_id, session_id):
        """Fetch last 2 messages from chat history for intent context."""
        if not user_id or not session_id:
            return ""
        try:
            history_text = self.chat_session_store.build_history_text(user_id, session_id)
            if not history_text:
                return ""
            lines = [line for line in history_text.strip().splitlines() if line.strip()]
            return "\n".join(lines[-2:]) if len(lines) >= 2 else "\n".join(lines)
        except Exception as e:
            LOGGER.warning("Failed to fetch last 2 messages for intent detection: %s", e)
            return ""

    def _determine_intent(self, prompt, last_two_messages=""):
        """Determine if we need history and/or web search."""
        LOGGER.info("Intent detection using model: %s", BEDROCK_MODEL_ID)
        
        prompt_lower = prompt.strip().lower()
        simple_prompts = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "cool", "nice"]

        #if prompt_lower in simple_prompts or len(prompt.strip()) < 10:
        if prompt_lower in simple_prompts:
            LOGGER.info("Intent detection: Simple conversational prompt detected, skipping LLM call")
            return {
                "read_history": False,
                "use_web_search": False,
                "reason": "Simple conversational prompt",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "model_id": BEDROCK_MODEL_ID,
            }

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Build structured prompt with clear sections
        parts = [f"Current UTC date: {today}\n"]
        
        if last_two_messages:
            parts.append("=== CONVERSATION HISTORY ===")
            parts.append(last_two_messages)
            parts.append("\n=== CURRENT USER MESSAGE ===")
            parts.append(f'"{prompt}"')
            #parts.append("\n=== ANALYSIS TASK ===")
            #parts.append("Does the current user message reference, continue, or depend on the conversation history above?")
            #parts.append("Check for pronouns (it, that, this) or follow-up phrases (tell me more, continue, what about).")
            #parts.append("Determine: read_history (true/false) and use_web_search (true/false)")
        else:
            parts.append("=== USER MESSAGE ===")
            parts.append(f'"{prompt}"')
            #parts.append("\n=== ANALYSIS TASK ===")
            #parts.append("This is a standalone message with no conversation history.")
            #parts.append("Determine: read_history (should be false) and use_web_search (true if needs current info)")
        
        full_prompt = "\n".join(parts)
        
        LOGGER.info("[DEBUG] Intent detection - System prompt length: %d chars", len(INTENT_SYSTEM_PROMPT))
        LOGGER.info("[DEBUG] Intent detection - User prompt length: %d chars", len(full_prompt))
        LOGGER.info("[DEBUG] Intent detection - User prompt content: %s", full_prompt[:200])

        output_text, usage = self._invoke_bedrock(full_prompt, "intent detection", INTENT_SYSTEM_PROMPT)
        json_text = self._extract_json_from_response(output_text)

        try:
            parsed = json.loads(json_text)
            result = {
                "read_history": bool(parsed.get("read_history", False)),
                "use_web_search": bool(parsed.get("use_web_search", False)),
                "reason": parsed.get("reason", "Intent detection completed"),
                "usage": usage,
                "model_id": BEDROCK_MODEL_ID,
            }
            LOGGER.info("LLM Intent detection: read_history=%s, use_web_search=%s, reason='%s'",
                        result["read_history"], result["use_web_search"], result["reason"])
            return result
        except json.JSONDecodeError as e:
            LOGGER.warning("Intent detection: Failed to parse LLM response: %s. Raw: '%s'", str(e), output_text[:500])
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

        blocked_domains = {"youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
                           "twitter.com", "facebook.com", "instagram.com", "tiktok.com"}
        if any(domain in hostname for domain in blocked_domains):
            return False
        if path.startswith("/watch") or "/watch" in path or path.startswith("/video") or "/video/" in path or "/embed" in path:
            return False
        return True

    def _build_search_query(self, history_text, prompt):
        """Rewrite the user prompt as a standalone search query using last 2 history messages."""
        lines = [line for line in history_text.strip().splitlines() if line.strip()]
        last_two = "\n".join(lines[-2:]) if len(lines) >= 2 else "\n".join(lines)

        full_prompt = "\n\n".join([
            "Last 2 conversation messages:",
            last_two,
            f"User follow-up: {prompt}",
            "Rewrite the follow-up as a standalone web search query.",
        ])

        try:
            output_text, _ = self._invoke_bedrock(full_prompt, "search query rewrite", SEARCH_QUERY_SYSTEM_PROMPT)
            rewritten = output_text.strip().strip('"').strip("'")
            if rewritten:
                return rewritten
        except Exception as e:
            LOGGER.warning("Search query rewrite failed, using original prompt: %s", e)
        return prompt

    def _fetch_web_content(self, prompt):
        """Fetch content from max 2 successful URLs using Tavily."""
        if not TAVILY_API_KEY:
            LOGGER.warning("TAVILY_API_KEY not configured, skipping web search")
            return ""

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": prompt,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                    "max_results": 10,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            search_results = response.json()

            urls = [result["url"] for result in search_results.get("results", [])[:10]]
            filtered_urls = [url for url in urls if self._is_content_url(url)]
            LOGGER.info("Tavily: %d URLs, %d filtered for query: %s", len(urls), len(filtered_urls), prompt[:50])

            if not filtered_urls:
                return ""

            successful_contents = []
            successful_count = 0

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(self._extract_url_content, url): url for url in filtered_urls}
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content:
                            successful_contents.append(content)
                            successful_count += 1
                            if successful_count >= 2:
                                for remaining in future_to_url:
                                    if not remaining.done():
                                        remaining.cancel()
                                break
                    except Exception as e:
                        LOGGER.warning("Exception extracting from %s: %s", url, e)

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
        """Generate final answer using single LLM call (non-streaming)."""
        LOGGER.info("Answer generation using model: %s", BEDROCK_MODEL_ID)
        full_prompt = self._build_answer_prompt(prompt, history_text, web_content)
        answer_text, usage = self._invoke_bedrock(full_prompt, "answer generation", ANSWER_SYSTEM_PROMPT)
        return {"answer": answer_text, "usage": usage, "model_id": BEDROCK_MODEL_ID}

    def _stream_answer(self, prompt, history_text="", web_content=""):
        """Stream answer from Bedrock using converse_stream. Yields (chunk_text, None) then (None, usage)."""
        LOGGER.info("Answer generation using model: %s", BEDROCK_MODEL_ID)
        full_prompt = self._build_answer_prompt(prompt, history_text, web_content)
        
        LOGGER.info("[DEBUG] Answer generation - System prompt length: %d chars", len(ANSWER_SYSTEM_PROMPT))
        LOGGER.info("[DEBUG] Answer generation - User prompt length: %d chars", len(full_prompt))
        LOGGER.info("[DEBUG] Answer generation - User prompt content: %s", full_prompt[:200])
        LOGGER.info("[DEBUG] Answer generation - Has history: %s (%d chars)", bool(history_text), len(history_text))
        LOGGER.info("[DEBUG] Answer generation - Has web content: %s (%d chars)", bool(web_content), len(web_content))

        try:
            response = self.client.converse_stream(
                modelId=BEDROCK_MODEL_ID,
                system=[{"text": ANSWER_SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": full_prompt}]}],
            )

            input_tokens = 0
            output_tokens = 0

            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"].get("delta", {}).get("text", "")
                    if text:
                        yield text, None
                elif "metadata" in event:
                    token_usage = event["metadata"].get("usage", {})
                    input_tokens = token_usage.get("inputTokens", 0)
                    output_tokens = token_usage.get("outputTokens", 0)

            yield None, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Bedrock converse_stream failed")
            raise RuntimeError(f"Bedrock streaming call failed: {error}") from error

    @staticmethod
    def _build_answer_prompt(prompt, history_text="", web_content=""):
        """Build the full prompt for answer generation."""
        parts = []
        if web_content:
            parts.append("Use the provided web content to answer questions about current events, scores, or recent information.")
        else:
            parts.append("Use built-in model knowledge to answer the question.")
        if history_text:
            parts.append(f"Previous chat messages:\n{history_text}")
        if web_content:
            parts.append(f"Retrieved web content:\n{web_content}")
        parts.append(f"Latest user prompt: {prompt}")
        return "\n\n".join(parts)

    def _invoke_bedrock(self, prompt, operation_name, system_prompt):
        """Invoke Bedrock model and return text + usage counts."""
        try:
            response = self.client.converse(
                modelId=BEDROCK_MODEL_ID,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            )
            output_text = self._extract_text(response)
            usage = self._build_usage(system_prompt, prompt, output_text)
            
            LOGGER.info("[DEBUG] %s - Actual usage from response: input=%d, output=%d", 
                       operation_name, 
                       response.get('usage', {}).get('inputTokens', 0),
                       response.get('usage', {}).get('outputTokens', 0))
            LOGGER.info("[DEBUG] %s - LiteLLM calculated usage: input=%d, output=%d", 
                       operation_name, usage.get('input_tokens', 0), usage.get('output_tokens', 0))
            
            return output_text, usage
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Bedrock %s call failed", operation_name)
            raise RuntimeError(f"Bedrock {operation_name} call failed: {error}") from error

    def _build_usage(self, system_prompt, user_prompt, output_text):
        """Count tokens using LiteLLM token_counter function."""
        input_tokens = self._count_tokens_for_input(system_prompt, user_prompt)
        output_tokens = self._count_tokens_for_output(output_text)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _count_tokens_for_input(self, system_prompt, user_prompt):
        if token_counter is None:
            return 0
        try:
            messages = [{"role": "user", "content": user_prompt}]
            input_tokens = token_counter(model=LITELLM_TOKENIZER_MODEL, messages=messages)
            if system_prompt:
                input_tokens += token_counter(model=LITELLM_TOKENIZER_MODEL, text=system_prompt)
            return input_tokens
        except Exception as error:
            LOGGER.warning("LiteLLM input token counting failed: %s", error)
            return 0

    def _count_tokens_for_output(self, output_text):
        if token_counter is None:
            return 0
        try:
            return token_counter(model=LITELLM_TOKENIZER_MODEL, text=output_text)
        except Exception as error:
            LOGGER.warning("LiteLLM output token counting failed: %s", error)
            return 0

    @staticmethod
    def _extract_text(response):
        """Extract text from Bedrock converse response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        return "".join(item.get("text", "") for item in content).strip()

    @staticmethod
    def _extract_json_from_response(text):
        """Extract JSON from LLM response, handling markdown code blocks."""
        if not text:
            return text
        text = text.strip()
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
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
