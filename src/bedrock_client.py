import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_MODEL_ID, BEDROCK_REGION


LOGGER = logging.getLogger(__name__)


class BedrockSummarizer:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    def summarize(self, user_prompt, sources):
        prompt = self._build_prompt(user_prompt, sources)

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
            LOGGER.exception("Bedrock summarize call failed")
            raise RuntimeError(f"Bedrock summarize call failed: {error}") from error

        usage = response.get("usage", {})
        output_text = self._extract_text(response)

        LOGGER.info("Nova Pro summary generated")
        LOGGER.info(
            "Summary token usage - input: %s, output: %s, total: %s",
            usage.get("inputTokens"),
            usage.get("outputTokens"),
            usage.get("totalTokens"),
        )

        return {
            "model_id": BEDROCK_MODEL_ID,
            "summary": output_text,
            "usage": {
                "input_tokens": usage.get("inputTokens"),
                "output_tokens": usage.get("outputTokens"),
                "total_tokens": usage.get("totalTokens"),
            },
            "raw_prompt": prompt,
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
    def _build_prompt(user_prompt, sources):
        source_sections = []
        for index, source in enumerate(sources, start=1):
            source_sections.append(
                "\n".join(
                    [
                        f"Source {index}",
                        f"Title: {source.get('title')}",
                        f"URL: {source.get('url')}",
                        f"Content: {source.get('content')}",
                    ]
                )
            )

        return "\n\n".join(
            [
                "You are a research assistant.",
                "Answer only from the provided sources.",
                "Give a short, clear summary.",
                "Prefer direct factual points over marketing language.",
                "End with a Sources section listing only the URLs you used.",
                f"User question: {user_prompt}",
                "Sources:",
                "\n\n".join(source_sections),
            ]
        )
