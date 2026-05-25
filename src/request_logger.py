import json
import logging
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import (
    REQUEST_LOGS_TABLE,
    INPUT_TOKEN_COST_PER_1K,
    OUTPUT_TOKEN_COST_PER_1K,
    MODEL_NAME,
)

LOGGER = logging.getLogger(__name__)


class RequestLogger:
    """Service to log API requests to DynamoDB."""

    def __init__(self):
        self.dynamodb = boto3.resource("dynamodb")
        self.table = self.dynamodb.Table(REQUEST_LOGS_TABLE)

    def log_request(
        self,
        request_id,
        email,
        session_id,
        timestamp_ms,
        status,
        prompt_length,
        input_tokens,
        output_tokens,
        response_time_ms,
        model_used,
        error_message=None,
    ):
        """
        Log a request to DynamoDB.

        Args:
            request_id: Unique request identifier
            email: User email
            session_id: Chat session ID
            timestamp_ms: Request timestamp in milliseconds (Unix epoch)
            status: "Success", "Error", or "Timeout"
            prompt_length: Length of user prompt
            input_tokens: Tokens consumed from input
            output_tokens: Tokens generated in output
            response_time_ms: Total response time in milliseconds
            model_used: Model name (e.g., "claude-sonnet-4-5")
            error_message: Error message if status is "Error" (optional)
        """
        try:
            # Calculate total cost
            total_cost = self._calculate_cost(input_tokens, output_tokens)

            # Prepare item for DynamoDB
            item = {
                "request_id": request_id,
                "email": email,
                "session_id": session_id,
                "timestamp": timestamp_ms,  # Sort key
                "status": status,
                "prompt_length": prompt_length,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "response_time_ms": Decimal(str(response_time_ms)),  # Convert to Decimal
                "model_used": model_used,
                "total_cost": Decimal(str(round(total_cost, 6))),  # Convert to Decimal
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Add error message if present
            if error_message:
                item["error_message"] = error_message

            # Insert into DynamoDB
            self.table.put_item(Item=item)

            LOGGER.info(
                "Request logged: request_id=%s, email=%s, status=%s, "
                "input_tokens=%d, output_tokens=%d, cost=%.6f",
                request_id,
                email,
                status,
                input_tokens,
                output_tokens,
                total_cost,
            )

        except (ClientError, BotoCoreError) as error:
            LOGGER.error(
                "Failed to log request to DynamoDB: %s. request_id=%s",
                error,
                request_id,
            )
        except Exception as error:
            LOGGER.error(
                "Unexpected error logging request: %s. request_id=%s",
                error,
                request_id,
            )

    @staticmethod
    def _calculate_cost(input_tokens, output_tokens):
        """
        Calculate total cost based on token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in dollars
        """
        input_cost = (input_tokens / 1000) * INPUT_TOKEN_COST_PER_1K
        output_cost = (output_tokens / 1000) * OUTPUT_TOKEN_COST_PER_1K
        return input_cost + output_cost
