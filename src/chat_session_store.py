import json
import logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import CHAT_SESSIONS_TABLE, DYNAMODB_REGION, MAX_HISTORY_CHARS, MAX_HISTORY_MESSAGES


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

    def build_history_text(self, user_id, session_id):
        messages = self.read_messages(user_id, session_id)
        if not messages:
            return ""

        lines = []
        for message in messages[-MAX_HISTORY_MESSAGES:]:
            role = (message.get("role") or message.get("sender") or "user").strip() or "user"
            content = (
                message.get("content")
                or message.get("text")
                or message.get("message")
                or message.get("body")
                or ""
            )
            content = " ".join(str(content).split()).strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
            if len("\n".join(lines)) >= MAX_HISTORY_CHARS:
                break

        history_text = "\n".join(lines)
        return history_text[:MAX_HISTORY_CHARS]

    def append_messages(self, user_id, session_id, new_messages):
        normalized_messages = self._normalize_messages(new_messages)
        if not user_id or not session_id or not normalized_messages:
            return

        now = datetime.now(timezone.utc).isoformat()
        title = self._build_title(normalized_messages)

        try:
            self.table.update_item(
                Key={"user_id": user_id, "session_id": session_id},
                UpdateExpression=(
                    "SET updated_at = :updated_at, "
                    "created_at = if_not_exists(created_at, :created_at), "
                    "title = if_not_exists(title, :title), "
                    "messages = list_append(if_not_exists(messages, :empty_messages), :messages)"
                ),
                ExpressionAttributeValues={
                    ":updated_at": now,
                    ":created_at": now,
                    ":title": title,
                    ":empty_messages": [],
                    ":messages": normalized_messages,
                },
            )
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception(
                "Failed to append chat session %s for user %s to DynamoDB",
                session_id,
                user_id,
            )
            raise RuntimeError(
                f"Failed to append chat session {session_id} for user {user_id}: {error}"
            ) from error

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
