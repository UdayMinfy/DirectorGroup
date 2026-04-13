import json
import logging
import re
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_MODEL_ID, BEDROCK_REGION


LOGGER = logging.getLogger(__name__)


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
            "You are a helpful assistant.",
            "Answer from built-in model knowledge only.",
            "Do not claim to have browsed the web.",
            "If the answer could be time-sensitive or uncertain, state that briefly.",
        ]
        if history_text:
            prompt_parts.append(f"Previous chat messages:\n{history_text}")
        prompt_parts.append(f"User prompt: {user_prompt}")
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
