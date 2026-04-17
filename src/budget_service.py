import logging
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import (
    DEFAULT_DAILY_TOKEN_LIMIT,
    DEFAULT_MONTHLY_TOKEN_LIMIT,
    DYNAMODB_REGION,
    MODEL_INPUT_COST_PER_1K,
    MODEL_OUTPUT_COST_PER_1K,
    REQUEST_TOKEN_PRECHECK,
    TOKEN_BUDGETS_TABLE,
)


LOGGER = logging.getLogger(__name__)


class BudgetError(Exception):
    status_code = 400


class UserNotFoundError(BudgetError):
    status_code = 404


class AccountInactiveError(BudgetError):
    status_code = 403


class BudgetExceededError(BudgetError):
    status_code = 429


class TokenBudgetService:
    def __init__(self):
        resource = boto3.resource("dynamodb", region_name=DYNAMODB_REGION)
        self.table = resource.Table(TOKEN_BUDGETS_TABLE)

    def _get_user_or_raise(self, user_email):
        try:
            response = self.table.get_item(Key={"user_email": user_email})
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Failed to load token budget for %s", user_email)
            raise RuntimeError(f"Failed to load token budget for {user_email}: {error}") from error

        item = response.get("Item")
        if not item:
            raise UserNotFoundError(f"No token budget record found for {user_email}.")
        return item

    def validate_request(self, user_email, required_tokens=None):
        item = self._get_user_or_raise(user_email)
        status = str(item.get("status") or "").strip().lower()
        if status != "active":
            raise AccountInactiveError(f"Account {user_email} is not active.")

        daily_limit = self._read_limit(item, ("daily_token_limit", "daily_limit_tokens"), DEFAULT_DAILY_TOKEN_LIMIT)
        monthly_limit = self._read_limit(item, ("monthly_token_limit", "monthly_limit_tokens"), DEFAULT_MONTHLY_TOKEN_LIMIT)
        required = REQUEST_TOKEN_PRECHECK if required_tokens is None else int(required_tokens)

        daily_used = int(item.get("consumed_tokens_daily") or 0)
        monthly_used = int(item.get("consumed_tokens_monthly") or 0)

        if daily_limit and daily_used + required > daily_limit:
            remaining = max(daily_limit - daily_used, 0)
            raise BudgetExceededError(f"Daily token budget exceeded for {user_email}. Remaining tokens: {remaining}.")

        if monthly_limit and monthly_used + required > monthly_limit:
            remaining = max(monthly_limit - monthly_used, 0)
            raise BudgetExceededError(f"Monthly token budget exceeded for {user_email}. Remaining tokens: {remaining}.")

        return self._build_budget_view(item, daily_limit, monthly_limit)

    def track_usage(self, user_email, input_tokens, output_tokens):
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        total_tokens = input_tokens + output_tokens
        cost_delta = self._calculate_cost(input_tokens, output_tokens)
        now = datetime.now(timezone.utc).isoformat()

        update_expression = (
            "SET updated_at = :updated_at, "
            "consumed_tokens_daily = if_not_exists(consumed_tokens_daily, :zero) + :daily_add, "
            "consumed_tokens_monthly = if_not_exists(consumed_tokens_monthly, :zero) + :monthly_add, "
            "total_requests = if_not_exists(total_requests, :zero) + :request_add, "
            "total_cost_usd = if_not_exists(total_cost_usd, :zero_decimal) + :cost_add"
        )
        expression_values = {
            ":updated_at": now,
            ":zero": 0,
            ":daily_add": total_tokens,
            ":monthly_add": total_tokens,
            ":request_add": 1,
            ":zero_decimal": Decimal("0"),
            ":cost_add": cost_delta,
        }

        try:
            response = self.table.update_item(
                Key={"user_email": user_email},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values,
                ReturnValues="ALL_NEW",
            )
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Failed to update token usage for %s", user_email)
            raise RuntimeError(f"Failed to update token usage for {user_email}: {error}") from error

        item = response.get("Attributes") or self._get_user_or_raise(user_email)
        daily_limit = self._read_limit(item, ("daily_token_limit", "daily_limit_tokens"), DEFAULT_DAILY_TOKEN_LIMIT)
        monthly_limit = self._read_limit(item, ("monthly_token_limit", "monthly_limit_tokens"), DEFAULT_MONTHLY_TOKEN_LIMIT)
        return self._build_budget_view(item, daily_limit, monthly_limit)

    @staticmethod
    def _read_limit(item, keys, default_limit):
        for key in keys:
            if item.get(key) not in (None, ""):
                return int(item[key])
        return int(default_limit or 0)

    @staticmethod
    def _build_budget_view(item, daily_limit, monthly_limit):
        daily_used = int(item.get("consumed_tokens_daily") or 0)
        monthly_used = int(item.get("consumed_tokens_monthly") or 0)
        total_cost = float(item.get("total_cost_usd") or 0)
        return {
            "status": item.get("status", ""),
            "tier": item.get("tier", ""),
            "consumed_tokens_daily": daily_used,
            "consumed_tokens_monthly": monthly_used,
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
            "daily_remaining": max(daily_limit - daily_used, 0) if daily_limit else None,
            "monthly_remaining": max(monthly_limit - monthly_used, 0) if monthly_limit else None,
            "total_cost_usd": total_cost,
        }

    @staticmethod
    def _calculate_cost(input_tokens, output_tokens):
        input_cost = Decimal(str(MODEL_INPUT_COST_PER_1K)) * Decimal(input_tokens) / Decimal(1000)
        output_cost = Decimal(str(MODEL_OUTPUT_COST_PER_1K)) * Decimal(output_tokens) / Decimal(1000)
        return input_cost + output_cost
