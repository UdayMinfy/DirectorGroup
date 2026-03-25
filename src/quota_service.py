import logging
from datetime import datetime, timezone
from uuid import uuid4

import psycopg

from config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    DEFAULT_DAILY_TOKEN_LIMIT,
    DEFAULT_MONTHLY_TOKEN_LIMIT,
)


LOGGER = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    def __init__(self, message, quota_snapshot):
        super().__init__(message)
        self.quota_snapshot = quota_snapshot


class TokenQuotaService:
    def __init__(self):
        self.conninfo = (
            f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
            f"user={DB_USER} password={DB_PASSWORD}"
        )

    def _connect(self):
        return psycopg.connect(self.conninfo)

    def ensure_user(self, external_user_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM token_plans WHERE plan_name = %s", ("default",))
            row = cur.fetchone()
            if row:
                plan_id = row[0]
            else:
                cur.execute(
                    """
                    INSERT INTO token_plans (plan_name, daily_token_limit, monthly_token_limit, is_active)
                    VALUES (%s, %s, %s, TRUE)
                    RETURNING id
                    """,
                    ("default", DEFAULT_DAILY_TOKEN_LIMIT, DEFAULT_MONTHLY_TOKEN_LIMIT),
                )
                plan_id = cur.fetchone()[0]

            cur.execute("SELECT id FROM users2 WHERE external_user_id = %s", (external_user_id,))
            row = cur.fetchone()
            if row:
                return row[0]

            cur.execute(
                """
                INSERT INTO users2 (external_user_id, status, plan_id)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (external_user_id, "active", plan_id),
            )
            return cur.fetchone()[0]

    def start_request(self, user_id, prompt):
        request_id = str(uuid4())
        usage_date, usage_month = self._periods()

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_token_usage_events (
                    user_id, request_id, prompt, status, usage_date, usage_month
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, request_id, prompt, "in_progress", usage_date, usage_month),
            )
            event_id = cur.fetchone()[0]

        return event_id, request_id

    def check_quota(self, user_id):
        usage_date, usage_month = self._periods()

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.daily_token_limit,
                    p.monthly_token_limit,
                    COALESCE(d.total_tokens_used, 0),
                    COALESCE(m.total_tokens_used, 0)
                FROM users2 u
                JOIN token_plans p ON p.id = u.plan_id
                LEFT JOIN user_token_usage_summary d
                    ON d.user_id = u.id
                    AND d.period_type = 'daily'
                    AND d.period_key = %s
                LEFT JOIN user_token_usage_summary m
                    ON m.user_id = u.id
                    AND m.period_type = 'monthly'
                    AND m.period_key = %s
                WHERE u.id = %s
                """,
                (usage_date, usage_month, user_id),
            )
            row = cur.fetchone()

        if not row:
            raise RuntimeError("User quota configuration could not be found.")

        daily_limit, monthly_limit, daily_used, monthly_used = row
        snapshot = {
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
            "daily_used": daily_used,
            "monthly_used": monthly_used,
            "daily_remaining": max(daily_limit - daily_used, 0),
            "monthly_remaining": max(monthly_limit - monthly_used, 0),
        }

        if daily_used >= daily_limit:
            raise QuotaExceededError("Daily token quota exceeded.", snapshot)
        if monthly_used >= monthly_limit:
            raise QuotaExceededError("Monthly token quota exceeded.", snapshot)

        return snapshot

    def mark_blocked(self, event_id, reason):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_token_usage_events
                SET status = %s, error_message = %s, completed_at = NOW()
                WHERE id = %s
                """,
                ("blocked", reason, event_id),
            )

    def mark_failed(self, event_id, reason):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_token_usage_events
                SET status = %s, error_message = %s, completed_at = NOW()
                WHERE id = %s
                """,
                ("failed", reason, event_id),
            )

    def record_success(self, event_id, user_id, prompt, answer, model_id, usage):
        usage_date, usage_month = self._periods()
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_token_usage_events
                SET
                    model_id = %s,
                    prompt = %s,
                    answer_preview = %s,
                    input_tokens = %s,
                    output_tokens = %s,
                    total_tokens = %s,
                    status = %s,
                    completed_at = NOW()
                WHERE id = %s
                """,
                (
                    model_id,
                    prompt,
                    answer[:1000],
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    "completed",
                    event_id,
                ),
            )

            self._upsert_summary(cur, user_id, "daily", usage_date, input_tokens, output_tokens, total_tokens)
            self._upsert_summary(cur, user_id, "monthly", usage_month, input_tokens, output_tokens, total_tokens)

        return self.check_quota(user_id)

    @staticmethod
    def _upsert_summary(cur, user_id, period_type, period_key, input_tokens, output_tokens, total_tokens):
        cur.execute(
            """
            INSERT INTO user_token_usage_summary (
                user_id, period_type, period_key,
                input_tokens_used, output_tokens_used, total_tokens_used,
                request_count, last_request_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, 1, NOW())
            ON CONFLICT (user_id, period_type, period_key)
            DO UPDATE SET
                input_tokens_used = user_token_usage_summary.input_tokens_used + EXCLUDED.input_tokens_used,
                output_tokens_used = user_token_usage_summary.output_tokens_used + EXCLUDED.output_tokens_used,
                total_tokens_used = user_token_usage_summary.total_tokens_used + EXCLUDED.total_tokens_used,
                request_count = user_token_usage_summary.request_count + 1,
                last_request_at = NOW(),
                updated_at = NOW()
            """,
            (user_id, period_type, period_key, input_tokens, output_tokens, total_tokens),
        )

    @staticmethod
    def _periods():
        now = datetime.now(timezone.utc)
        return now.date().isoformat(), now.strftime("%Y-%m")
