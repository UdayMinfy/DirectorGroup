import json
import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_REGION, NOVA_PRO_MODEL_ID, SUMMARY_CHUNK_SIZE, SUMMARY_WORD_LIMIT


LOGGER = logging.getLogger(__name__)


SUMMARIZATION_SYSTEM_PROMPT = f"""You are a conversation summarizer. 
Summarize the provided conversation messages into a concise {SUMMARY_WORD_LIMIT}-word summary.

Focus on:
- Key topics discussed
- Important questions asked
- Decisions or conclusions reached
- Main information exchanged

Be factual and preserve important context. Output only the summary text, no preamble."""


class ChatSummarizer:
    SUMMARY_CHUNK_SIZE = SUMMARY_CHUNK_SIZE  # Make it accessible as class attribute
    
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    def should_create_summary(self, total_messages, existing_summaries_count):
        """Check if we need to create a new summary."""
        if total_messages < SUMMARY_CHUNK_SIZE + 1:
            # Need at least SUMMARY_CHUNK_SIZE + 1 messages to trigger first summary
            return False
        
        # Calculate how many messages are not yet summarized
        summarized_messages = existing_summaries_count * SUMMARY_CHUNK_SIZE
        unsummarized_messages = total_messages - summarized_messages
        
        # Create summary when we have SUMMARY_CHUNK_SIZE + 1 unsummarized messages
        return unsummarized_messages >= SUMMARY_CHUNK_SIZE + 1

    def get_messages_to_summarize(self, all_messages, existing_summaries_count):
        """Extract the chunk of messages that should be summarized."""
        start_idx = existing_summaries_count * SUMMARY_CHUNK_SIZE
        end_idx = start_idx + SUMMARY_CHUNK_SIZE
        return all_messages[start_idx:end_idx]

    def generate_summary(self, messages_chunk):
        """Generate a summary for a chunk of messages using Nova Pro."""
        if not messages_chunk:
            return ""

        # Format messages for summarization
        formatted_messages = []
        for msg in messages_chunk:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")
        
        conversation_text = "\n".join(formatted_messages)
        user_prompt = f"Summarize this conversation in {SUMMARY_WORD_LIMIT} words or less:\n\n{conversation_text}"

        try:
            response = self.client.converse(
                modelId=NOVA_PRO_MODEL_ID,
                system=[{"text": SUMMARIZATION_SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            )
            
            summary = self._extract_text(response)
            LOGGER.info(
                "Generated summary for %d messages: %s (model: %s)",
                len(messages_chunk),
                summary[:100] + "..." if len(summary) > 100 else summary,
                NOVA_PRO_MODEL_ID,
            )
            return summary
            
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Failed to generate summary with Nova Pro")
            # Return a basic fallback summary
            return f"Conversation covering {len(messages_chunk)} messages (summary generation failed)"

    @staticmethod
    def _extract_text(response):
        """Extract text from Bedrock converse response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        return "".join(item.get("text", "") for item in content).strip()
