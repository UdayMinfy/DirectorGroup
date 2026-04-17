import json
import logging
import re
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_MODEL_ID, BEDROCK_REGION


LOGGER = logging.getLogger(__name__)

CONTEXT_DECISION_SYSTEM_PROMPT = """You are an intent router for the latest user message.
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
- both false: self-contained + stable knowledge question
- read_history true, use_web_search false: follow-up about prior chat, but no fresh facts needed
- read_history false, use_web_search true: self-contained question needing fresh/current facts
- both true: follow-up that also needs fresh/current facts
Bias:
- If the message contains words like "latest", "current", "today", "recent", "now", prefer use_web_search = true.
- If unsure whether facts may have changed, prefer use_web_search = true.
- If unsure whether the message refers to prior chat, prefer read_history = true.
Do not answer the user. Output JSON only."""

KNOWLEDGE_SYSTEM_PROMPT = """You are a helpful, polished conversational assistant.
Answer style:
- Start with the direct answer.
- Add brief explanation or examples only when useful.
- Keep simple answers short and complex answers clear.
- Write naturally, confidently, and conversationally.
Use of knowledge:
- Use built-in knowledge for stable topics.
- If prior chat is provided, use it as working memory.
- If retrieved content is provided, prioritize the most relevant facts and synthesize them clearly.
- If sources conflict, follow the latest user request first, then the strongest evidence.
- Do not invent facts.
Quality:
- Be accurate, clear, and concise.
- Mention uncertainty briefly when needed.
- For time-sensitive topics, rely on retrieved content if available; otherwise say live verification may be needed.
Formatting:
- Make the answer easy to scan.
- When helpful, use light visual markers like: ✓, •, 👍
- Do not overuse symbols or make the response noisy.
Follow-up:
- Ask a short follow-up only if needed for clarity or if it genuinely helps the user continue."""


class BedrockIntentRouter:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    def decide_context_strategy(self, user_prompt):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prompt = "\n\n".join(
            [
                "Decide whether the assistant should read previous session history and/or use web search.",
                "Return JSON only with keys: read_history, use_web_search, reason.",
                "Use read_history=true when the latest message depends on earlier chat context.",
                "Use use_web_search=true when the answer needs latest/current/live/news/price/schedule/date-sensitive facts.",
                "Use both=false for standalone stable questions.",
                "Use both=true when the user asks a follow-up that also needs fresh web information.",
                f"Current UTC date: {today}",
                f"User prompt: {user_prompt}",
            ]
        )

        output_text, usage = self._invoke_text(prompt, "context strategy decision", CONTEXT_DECISION_SYSTEM_PROMPT)
        parsed = self._parse_json_object(output_text)

        strategy = {
            "read_history": False,
            "use_web_search": True,
            "reason": "Failed to parse context strategy response; defaulting to web search without history.",
            "usage": usage,
            "model_id": BEDROCK_MODEL_ID,
        }
        if isinstance(parsed, dict):
            strategy["read_history"] = bool(parsed.get("read_history"))
            strategy["use_web_search"] = bool(parsed.get("use_web_search"))
            strategy["reason"] = (parsed.get("reason") or strategy["reason"]).strip()
        return strategy

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
