import base64
import json
import logging
import time
from uuid import uuid4

from simplified_agent import SimplifiedResearchAgent
from auth import AuthenticationError, extract_bearer_token, validate_access_token, get_user_email_from_token
from budget_service import TokenBudgetService
from config import IS_LOCAL_SAM, MAX_PROMPT_LENGTH, MODEL_NAME
from request_logger import RequestLogger


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

agent = SimplifiedResearchAgent()
budget_service = TokenBudgetService()
request_logger = RequestLogger()

SSE_HEADERS = {
    # SSE Essential
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-store, no-cache, must-revalidate, private",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    # Security
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-XSS-Protection": "1; mode=block",
    "Content-Security-Policy": "default-src 'self';"
}

JSON_HEADERS = {
    "Content-Type": "application/json",
}


def _parse_event(event):
    if not event or not isinstance(event, dict):
        return {}

    body = event.get("body")
    if event.get("isBase64Encoded") and isinstance(body, str):
        body = base64.b64decode(body).decode("utf-8")

    if isinstance(body, str):
        try:
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {"prompt": body}
    if isinstance(body, dict):
        return body
    return event


def _error_response(status_code, error_code, message):
    """Return a non-streaming error response."""
    return {
        "statusCode": status_code,
        "headers": JSON_HEADERS,
        "body": json.dumps({"error": error_code, "message": message}),
    }


def _sse_error(message):
    """Return an SSE-formatted error event string."""
    return f"event: error\ndata: {json.dumps({'message': message})}\n\n"


def _validate_jwt(event):
    """Validate JWT token on every request. Cannot be bypassed in production.
    Only skipped when running locally via SAM (IS_LOCAL_SAM=true).
    Returns the raw token string for downstream use.
    """
    if IS_LOCAL_SAM:
        LOGGER.warning("JWT validation skipped — local SAM environment only")
        return None

    try:
        token = extract_bearer_token(event)
        validate_access_token(token)
        LOGGER.info("JWT validated successfully")
        return token
    except AuthenticationError:
        raise
    except ValueError:
        raise AuthenticationError("Invalid or expired authorization token.")


def lambda_handler(event, context):
    request_id = str(uuid4())
    start_time = time.time()
    start_timestamp_ms = int(start_time * 1000)

    # ── Step 1: JWT Validation (always first) ──────────────────────────────
    try:
        token = _validate_jwt(event)
    except AuthenticationError as error:
        LOGGER.warning("Auth failed: %s", error)
        return _error_response(401, "Unauthorized", str(error))

    # ── Step 2: Get user email from Cognito ────────────────────────────────
    if IS_LOCAL_SAM:
        payload = _parse_event(event)
        user_email = str((payload or {}).get("email") or "test@example.com").strip()
    else:
        try:
            user_email = get_user_email_from_token(token)
        except AuthenticationError as error:
            LOGGER.warning("Failed to retrieve user email: %s", error)
            return _error_response(401, "Unauthorized", str(error))

    # ── Step 3: Parse request body ─────────────────────────────────────────
    payload = _parse_event(event)

    # ── Step 4: Validate prompt ────────────────────────────────────────────
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return _error_response(400, "Bad Request", "Request body must include a non-empty 'prompt' field.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        return _error_response(400, "Bad Request", f"Prompt exceeds maximum allowed length of {MAX_PROMPT_LENGTH} characters.")

    session_id = str(payload.get("session_id") or "").strip() or str(uuid4())
    prompt_length = len(prompt)

    # ── Step 5: Stream response ────────────────────────────────────────────
    LOGGER.info("Chat stream request: user=%s session=%s request_id=%s", user_email, session_id, request_id)

    # Variables to track response metrics
    input_tokens = 0
    output_tokens = 0
    response_status = "Success"
    error_message = None

    def generate():
        nonlocal input_tokens, output_tokens, response_status, error_message
        try:
            for sse_event in agent.run_stream(prompt, user_id=user_email, session_id=session_id):
                # Parse SSE event to extract token counts from done event
                if sse_event.startswith("event: done"):
                    # Extract data from done event
                    lines = sse_event.split("\n")
                    for line in lines:
                        if line.startswith("data: "):
                            try:
                                done_data = json.loads(line[6:])
                                usage = done_data.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)
                                output_tokens = usage.get("output_tokens", 0)
                                LOGGER.info(
                                    "Request completed: request_id=%s, input_tokens=%d, output_tokens=%d",
                                    request_id,
                                    input_tokens,
                                    output_tokens,
                                )
                            except json.JSONDecodeError:
                                LOGGER.warning("Failed to parse done event data")
                yield sse_event
        except Exception as error:
            LOGGER.exception("Streaming failed for user=%s request_id=%s", user_email, request_id)
            response_status = "Error"
            error_message = str(error)
            yield _sse_error("An unexpected error occurred. Please try again.")

    response_body = "".join(generate())

    # ── Step 6: Log request to DynamoDB ────────────────────────────────────
    end_time = time.time()
    response_time_ms = int((end_time - start_time) * 1000)

    try:
        request_logger.log_request(
            request_id=request_id,
            email=user_email,
            session_id=session_id,
            timestamp_ms=start_timestamp_ms,
            status=response_status,
            prompt_length=prompt_length,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_time_ms=response_time_ms,
            model_used=MODEL_NAME,
            error_message=error_message,
        )
    except Exception as error:
        LOGGER.error("Failed to log request: %s", error)

    return {
        "statusCode": 200,
        "headers": SSE_HEADERS,
        "body": response_body,
    }
