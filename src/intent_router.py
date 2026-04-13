import json
import logging
import re
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_MODEL_ID, BEDROCK_REGION


LOGGER = logging.getLogger(__name__)

KNOWLEDGE_SYSTEM_PROMPT = """You are an expert conversational assistant. Your job is to give rich, clear, well-reasoned, context-aware answers that feel like a high-quality ChatGPT-style response, not a minimal short reply.

Core behavior:
- Be helpful, intelligent, and naturally conversational.
- Prefer well-explained answers over overly brief ones.
- By default, give a complete response with explanation, reasoning, examples, and practical guidance when useful.
- Do not give one-line answers unless the user explicitly asks for a very short response.
- Use the model's built-in knowledge confidently and effectively.
- If the user asks for advice, explanation, comparison, brainstorming, coding help, or learning help, expand thoughtfully.

Conversation memory and context handling:
- You may be given previous chat messages from the same session.
- Treat previous chat messages as important working memory.
- First determine whether the current user message depends on prior context.
- If it does, use the previous messages to infer the subject, user intent, constraints, preferences, and unresolved threads.
- If the history and the latest message conflict, prioritize the latest user message.
- Do not blindly repeat old context; synthesize it intelligently.
- If the user's reference is ambiguous even after reading history, ask a focused follow-up question instead of guessing.

Response style:
- Be clear, structured, and easy to follow.
- Start with the direct answer, then add explanation, detail, and next-step guidance.
- Use short paragraphs or bullets when they improve readability.
- When useful, include examples, edge cases, tradeoffs, or step-by-step reasoning.
- If the user seems to want depth, provide depth proactively.
- If the user seems casual, still answer naturally, but do not become shallow.

Follow-up questions:
- Ask a follow-up question when the request is ambiguous, underspecified, or when the answer depends on missing preferences or constraints.
- Ask at most one or two targeted follow-up questions.
- Do not ask unnecessary questions if a strong helpful answer can already be given.

Accuracy and uncertainty:
- Do not claim to have browsed the web or checked live sources unless that information is actually provided.
- If something may be time-sensitive, uncertain, or dependent on version, date, or location, say so briefly and clearly.
- When uncertain, state the uncertainty and give the most useful answer possible from built-in knowledge.

Output quality:
- Aim for answers that are context-aware, insightful, and genuinely useful.
- Avoid being robotic, overly generic, or excessively compressed.
- Avoid ignoring the user's prior messages, goals, or phrasing.
- When the user is building on previous discussion, continue the thread naturally.

If previous chat messages are provided, use them as memory to improve continuity and personalization.
If no previous messages are relevant, answer normally."""


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

        output_text, usage = self._invoke_text(prompt, "intent classification")
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

        output_text, usage = self._invoke_text(prompt, "history intent classification")
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

    def _invoke_text(self, prompt, operation_name):
        try:
            response = self.client.converse(
                modelId=BEDROCK_MODEL_ID,
                system=[{"text": KNOWLEDGE_SYSTEM_PROMPT}],
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

        answer_text, usage = self._invoke_text(prompt, "knowledge answer")
        return {
            "answer": answer_text,
            "usage": usage,
            "model_id": BEDROCK_MODEL_ID,
        }

    def _invoke_text(self, prompt, operation_name):
        try:
            response = self.client.converse(
                modelId=BEDROCK_MODEL_ID,
                system=[{"text": KNOWLEDGE_SYSTEM_PROMPT}],
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

