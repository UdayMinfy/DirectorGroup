import json
import logging
import re
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_MODEL_ID, BEDROCK_REGION


LOGGER = logging.getLogger(__name__)

ROUTING_SYSTEM_PROMPT = """Classify the request as either general knowledge or realtime.
Return valid JSON only.
Choose realtime only when the answer needs current, live, web-updated, or date-sensitive facts.
Choose general for stable facts, math, coding, writing, explanations, and reasoning."""

HISTORY_SYSTEM_PROMPT = """Decide whether previous session messages are needed to answer the latest user message.
Return valid JSON only.
Use history when the user refers to earlier context, omitted subjects, prior constraints, preferences, or unresolved follow-ups.
If the latest message stands on its own, do not use history."""

KNOWLEDGE_SYSTEM_PROMPT = """You are a high-quality conversational assistant. Give answers that are accurate, clear, natural, and genuinely helpful.

Answering style:
- Start with the direct answer.
- Then add explanation, reasoning, examples, or practical detail when it improves the answer.
- Match the depth to the question: simple questions can be short, but important or complex questions should be well explained.
- Write like a strong frontier chat assistant: thoughtful, fluent, and context-aware, not robotic or generic.

Use of knowledge, memory, and retrieved content:
- Use built-in knowledge confidently for stable topics.
- If previous chat messages are provided, treat them as working memory for the user's subject, goals, preferences, constraints, and unresolved threads.
- If retrieved web or extracted content is provided, use it carefully and make it count: prioritize the most relevant facts, combine overlapping evidence, and avoid wasting useful extracted details.
- Do not ignore strong evidence from retrieved content, and do not repeat raw extracted text unnecessarily; synthesize it into a clean answer.
- When memory, retrieved content, and the latest user message differ, prioritize the latest user message, then the most reliable retrieved evidence.

Reasoning and quality:
- Be precise, but not dry.
- Highlight uncertainty briefly when needed.
- For time-sensitive topics, rely on retrieved content if available; otherwise say that live verification may be needed.
- Do not invent facts that are not supported by built-in knowledge or provided content.

Follow-up behavior:
- If the request is ambiguous or missing an important detail, ask a short, useful follow-up question.
- If the answer is complete enough already, do not ask unnecessary questions.
- When appropriate, end with one natural follow-up question that helps the user continue.

Overall goal:
- Make the response feel intelligent, polished, and useful.
- Use available memory and retrieved content wisely so the final answer is richer, more relevant, and more grounded."""


class BedrockIntentRouter:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    def classify(self, user_prompt):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prompt = "\n\n".join(
            [
                "Classify whether the user question requires real-time or web-updated information.",
                "Return JSON only with keys: route, reason.",
                'Allowed route values: "realtime" or "general".',
                "Choose realtime when latest/current/live/news/price/schedule/time-sensitive facts are needed.",
                "Choose general for timeless concepts, explanations, coding help, writing, math, and stable facts.",
                f"Current UTC date: {today}",
                f"User prompt: {user_prompt}",
            ]
        )

        output_text, usage = self._invoke_text(prompt, "intent classification", ROUTING_SYSTEM_PROMPT)
        parsed = self._parse_json_object(output_text)

        route = "realtime"
        reason = "Failed to parse intent response; defaulting to realtime for safer freshness."

        if isinstance(parsed, dict):
            candidate = (parsed.get("route") or "").strip().lower()
            if candidate in {"realtime", "general"}:
                route = candidate
            reason = (parsed.get("reason") or reason).strip()

        return {
            "route": route,
            "reason": reason,
            "usage": usage,
            "model_id": BEDROCK_MODEL_ID,
        }

    def should_read_history(self, user_prompt):
        prompt = "\n\n".join(
            [
                "Decide whether answering the latest user message requires previous chat messages from the same session.",
                "Return JSON only with keys: read_history, reason.",
                'read_history must be true when the user refers to earlier context like: continue, that, this, above, previous, same as before, explain more, summarize again, follow-up questions, or omitted subject references.',
                'read_history must be false when the latest user prompt is standalone and understandable by itself.',
                f"User prompt: {user_prompt}",
            ]
        )

        output_text, usage = self._invoke_text(prompt, "history intent classification", HISTORY_SYSTEM_PROMPT)
        parsed = self._parse_json_object(output_text)
        read_history = False
        reason = "Failed to parse history intent response; defaulting to not reading history."
        if isinstance(parsed, dict):
            read_history = bool(parsed.get("read_history"))
            reason = (parsed.get("reason") or reason).strip()
        return {
            "read_history": read_history,
            "reason": reason,
            "usage": usage,
            "model_id": BEDROCK_MODEL_ID,
        }

    def _invoke_text(self, prompt, operation_name, system_prompt):
        try:
            response = self.client.converse(
                modelId=BEDROCK_MODEL_ID,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            )
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Bedrock %s call failed", operation_name)
            raise RuntimeError(f"Bedrock {operation_name} call failed: {error}") from error

        usage = response.get("usage", {})
        return self._extract_text(response), {
            "input_tokens": usage.get("inputTokens") or 0,
            "output_tokens": usage.get("outputTokens") or 0,
            "total_tokens": usage.get("totalTokens") or 0,
        }

    @staticmethod
    def _extract_text(response):
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
    def _parse_json_object(text):
        if not text:
            return None

        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


class BedrockKnowledgeResponder:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    def answer(self, user_prompt, history_text=""):
        prompt_parts = [
            "Use built-in model knowledge only.",
            "Do not claim to have browsed the web.",
        ]
        if history_text:
            prompt_parts.append(f"Previous chat messages:\n{history_text}")
        prompt_parts.append(f"Latest user prompt: {user_prompt}")
        prompt = "\n\n".join(prompt_parts)

        answer_text, usage = self._invoke_text(prompt, "knowledge answer", KNOWLEDGE_SYSTEM_PROMPT)
        return {
            "answer": answer_text,
            "usage": usage,
            "model_id": BEDROCK_MODEL_ID,
        }

    def _invoke_text(self, prompt, operation_name, system_prompt):
        try:
            response = self.client.converse(
                modelId=BEDROCK_MODEL_ID,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            )
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Bedrock %s call failed", operation_name)
            raise RuntimeError(f"Bedrock {operation_name} call failed: {error}") from error

        usage = response.get("usage", {})
        return self._extract_text(response), {
            "input_tokens": usage.get("inputTokens") or 0,
            "output_tokens": usage.get("outputTokens") or 0,
            "total_tokens": usage.get("totalTokens") or 0,
        }

    @staticmethod
    def _extract_text(response):
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        parts = []
        for item in content:
            text = item.get("text")
            if text:
                parts.append(text)
        return "".join(parts).strip()




