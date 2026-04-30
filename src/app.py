import base64
import json
import logging
from uuid import uuid4

from simplified_agent import SimplifiedResearchAgent
from auth import AuthenticationError, extract_bearer_token, validate_access_token
from budget_service import AccountInactiveError, BudgetExceededError, TokenBudgetService, UserNotFoundError
from config import JWT_VALIDATION_ENABLED


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

agent = SimplifiedResearchAgent()
budget_service = TokenBudgetService()

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Authorization,Content-Type,X-User-Email",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
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


def _build_response(status_code, payload, extra_headers=None):
    headers = dict(DEFAULT_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": json.dumps(payload),
    }


def _is_http_event(event):
    request_context = (event or {}).get("requestContext") or {}
    return isinstance(request_context.get("http"), dict)


def extract_user_email(event, payload):
    if JWT_VALIDATION_ENABLED and _is_http_event(event):
        try:
            token = extract_bearer_token(event)
            validate_access_token(token)
            LOGGER.info("JWT validated successfully")
        except (AuthenticationError, ValueError) as error:
            raise AuthenticationError(str(error)) from error

    user_email = str((payload or {}).get("email") or "").strip()
    if not user_email:
        user_email = str((payload or {}).get("user_id") or "").strip()
    if not user_email:
        raise AuthenticationError("Missing authenticated user email in payload.")

    budget_service._get_user_or_raise(user_email)
    return user_email


def lambda_handler(event, context):
    http_method = (((event or {}).get("requestContext") or {}).get("http") or {}).get("method", "")
    if http_method == "OPTIONS":
        LOGGER.info("OPTIONS preflight request allowed")
        return _build_response(200, {"ok": True})

    payload = _parse_event(event)
    prompt = str(payload.get("prompt") or "").strip()
    session_id = str(payload.get("session_id") or "").strip() or str(uuid4())

    if not prompt:
        return _build_response(400, {"error": "Request payload must include a non-empty 'prompt'."})

    request_id = str(uuid4())

    try:
        user_email = extract_user_email(event, payload)
        LOGGER.info("Chat request: user=%s", user_email)
        budget_snapshot = budget_service.validate_request(user_email)

        result = agent.run(prompt, user_id=user_email, session_id=session_id)

        usage = result.get("usage", {})
        budget_after = budget_service.track_usage(
            user_email=user_email,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
    except AuthenticationError as error:
        return _build_response(401, {"error": str(error), "request_id": request_id})
    except UserNotFoundError as error:
        return _build_response(error.status_code, {"error": str(error), "request_id": request_id})
    except AccountInactiveError as error:
        return _build_response(error.status_code, {"error": str(error), "request_id": request_id})
    except BudgetExceededError as error:
        return _build_response(error.status_code, {"error": str(error), "request_id": request_id})
    except Exception as error:
        LOGGER.exception("Research workflow failed")
        return _build_response(
            500,
            {
                "error": str(error),
                "request_id": request_id,
                "session_id": session_id,
            },
        )

    response_payload = {
        "answer": result.get("answer", ""),
        "usage": usage,
        "user_id": user_email,
        "session_id": result.get("session_id", session_id),
        "history_used": result.get("history_used", False),
        "budget": budget_after,
        "budget_before": budget_snapshot,
        "streaming_supported": False,
    }
    return _build_response(200, response_payload)
