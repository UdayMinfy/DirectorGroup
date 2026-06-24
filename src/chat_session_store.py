import json
import logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import CHAT_SESSIONS_TABLE, DYNAMODB_REGION, SUMMARY_CHUNK_SIZE


LOGGER = logging.getLogger(__name__)


class ChatSessionStore:
    def __init__(self):
        resource = boto3.resource("dynamodb", region_name=DYNAMODB_REGION)
        self.table = resource.Table(CHAT_SESSIONS_TABLE)

    def read_messages(self, user_id, session_id):
        if not user_id or not session_id:
            return []

        try:
            response = self.table.get_item(Key={"user_id": user_id, "session_id": session_id})
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception(
                "Failed to read chat session %s for user %s from DynamoDB",
                session_id,
                user_id,
            )
            raise RuntimeError(
                f"Failed to read chat session {session_id} for user {user_id}: {error}"
            ) from error

        item = response.get("Item") or {}
        return self._normalize_messages(item.get("messages"))

    def read_summaries(self, user_id, session_id):
        """Read existing summaries from DynamoDB."""
        if not user_id or not session_id:
            return []

        try:
            response = self.table.get_item(Key={"user_id": user_id, "session_id": session_id})
        except (ClientError, BotoCoreError) as error:
            LOGGER.warning("Failed to read summaries: %s", error)
            return []

        item = response.get("Item") or {}
        summaries = item.get("summary", [])
        return summaries if isinstance(summaries, list) else []

    def build_history_text(self, user_id, session_id, max_messages=10):
        """Build history text with summaries + all unsummarized raw messages."""
        messages = self.read_messages(user_id, session_id)
        if not messages:
            return ""

        summaries = self.read_summaries(user_id, session_id)
        
        parts = []
        
        # Add all summaries first
        if summaries:
            for i, summary in enumerate(summaries, 1):
                parts.append(f"[Summary {i}]: {summary}")
            parts.append("")  # Empty line separator
            LOGGER.info("Loaded %d summaries for session %s", len(summaries), session_id)

        # Calculate how many messages are already summarized
        summarized_message_count = len(summaries) * SUMMARY_CHUNK_SIZE
        
        # Get all unsummarized messages (everything after the last summarized message)
        unsummarized_messages = messages[summarized_message_count:]
        
        if unsummarized_messages:
            parts.append("Recent messages:")
            for message in unsummarized_messages:
                role = (message.get("role") or message.get("sender") or "user").strip() or "user"
                content = (
                    message.get("content")
                    or message.get("text")
                    or message.get("message")
                    or message.get("body")
                    or ""
                )
                content = " ".join(str(content).split()).strip()
                if content:
                    parts.append(f"{role}: {content}")
            
            LOGGER.info(
                "Loaded %d unsummarized raw messages for session %s (messages %d-%d)",
                len(unsummarized_messages),
                session_id,
                summarized_message_count + 1,
                len(messages)
            )

        return "\n".join(parts)

    def append_messages(self, user_id, session_id, new_messages):
        """Append messages and trigger summarization if needed."""
        normalized_messages = self._normalize_messages(new_messages)
        if not user_id or not session_id or not normalized_messages:
            return

        existing_messages = self.read_messages(user_id, session_id)
        now = datetime.now(timezone.utc).isoformat()
        title = self._build_title(normalized_messages)
        update_expression = (
            "SET updated_at = :updated_at, "
            "created_at = if_not_exists(created_at, :created_at), "
            "messages = list_append(if_not_exists(messages, :empty_messages), :messages)"
        )
        expression_attribute_values = {
            ":updated_at": now,
            ":created_at": now,
            ":empty_messages": [],
            ":messages": normalized_messages,
        }

        if not existing_messages:
            update_expression = (
                "SET updated_at = :updated_at, "
                "created_at = if_not_exists(created_at, :created_at), "
                "title = :title, "
                "messages = list_append(if_not_exists(messages, :empty_messages), :messages)"
            )
            expression_attribute_values[":title"] = title

        try:
            self.table.update_item(
                Key={"user_id": user_id, "session_id": session_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values,
            )
            
            # After successful save, check if we need to create a summary
            self._check_and_create_summary(user_id, session_id, existing_messages, normalized_messages)
            
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception(
                "Failed to append chat session %s for user %s to DynamoDB",
                session_id,
                user_id,
            )
            raise RuntimeError(
                f"Failed to append chat session {session_id} for user {user_id}: {error}"
            ) from error

    def _check_and_create_summary(self, user_id, session_id, existing_messages, new_messages):
        """Check if summarization is needed and create summary if so."""
        try:
            from chat_summarizer import ChatSummarizer
            
            summarizer = ChatSummarizer()
            
            # Calculate total messages after adding new ones
            total_messages = len(existing_messages) + len(new_messages)
            existing_summaries = self.read_summaries(user_id, session_id)
            existing_summaries_count = len(existing_summaries)
            
            # Check if we should create a summary
            if summarizer.should_create_summary(total_messages, existing_summaries_count):
                # Get all messages including the newly added ones
                all_messages = existing_messages + new_messages
                
                # Get the chunk to summarize
                messages_to_summarize = summarizer.get_messages_to_summarize(all_messages, existing_summaries_count)
                
                # Generate summary
                summary = summarizer.generate_summary(messages_to_summarize)
                
                if summary:
                    # Append summary to DynamoDB
                    self._append_summary(user_id, session_id, summary)
                    LOGGER.info(
                        "Created summary %d for session %s (messages %d-%d)",
                        existing_summaries_count + 1,
                        session_id,
                        existing_summaries_count * summarizer.SUMMARY_CHUNK_SIZE + 1,
                        (existing_summaries_count + 1) * summarizer.SUMMARY_CHUNK_SIZE,
                    )
        except Exception as error:
            # Don't fail the main flow if summarization fails
            LOGGER.warning("Summarization failed for session %s: %s", session_id, error)

    def _append_summary(self, user_id, session_id, summary):
        """Append a summary to the summary array in DynamoDB."""
        try:
            self.table.update_item(
                Key={"user_id": user_id, "session_id": session_id},
                UpdateExpression="SET summary = list_append(if_not_exists(summary, :empty), :summary)",
                ExpressionAttributeValues={
                    ":empty": [],
                    ":summary": [summary],
                },
            )
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Failed to append summary to DynamoDB")
            raise

    @staticmethod
    def _normalize_messages(messages):
        if isinstance(messages, list):
            normalized = []
            for message in messages:
                if isinstance(message, dict):
                    normalized.append(message)
                elif isinstance(message, str) and message.strip():
                    normalized.append({"role": "user", "content": message.strip()})
            return normalized

        if isinstance(messages, str):
            stripped = messages.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [{"role": "user", "content": stripped}]
            if isinstance(parsed, list):
                return ChatSessionStore._normalize_messages(parsed)
            if isinstance(parsed, dict):
                return [parsed]
        return []

    @staticmethod
    def _build_title(messages):
        for message in messages:
            if (message.get("role") or "").strip().lower() != "user":
                continue
            content = (
                message.get("content")
                or message.get("text")
                or message.get("message")
                or message.get("body")
                or ""
            )
            title = " ".join(str(content).split()).strip()
            if title:
                return title[:200]
        return "Chat session"


