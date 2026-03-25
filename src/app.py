import json
import logging

from agent import ResearchAgent
from quota_service import QuotaExceededError, TokenQuotaService


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

agent = ResearchAgent()
quota_service = TokenQuotaService()


def _parse_event(event):
    if not event:
        return {}

    if not isinstance(event, dict):
        return {}

    body = event.get("body")
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"prompt": body}
    if isinstance(body, dict):
        return body
    return event

def lambda_handler(event, context):
    payload = _parse_event(event)
    prompt = (payload.get("prompt") or "").strip()
    external_user_id = (payload.get("user_id") or "").strip()
    urls = payload.get("urls") or []

    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list):
        urls = []
    urls = [url.strip() for url in urls if isinstance(url, str) and url.strip()]

    if not prompt:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Request payload must include a non-empty 'prompt'."}),
        }
    if not external_user_id:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Request payload must include a non-empty 'user_id'."}),
        }

    user_id = quota_service.ensure_user(external_user_id)
    event_id, request_id = quota_service.start_request(user_id, prompt)

    try:
        quota_service.check_quota(user_id)
        if urls:
            result = agent.run(prompt, urls=urls)
        else:
            result = agent.run(prompt)
        updated_quota = quota_service.record_success(
            event_id=event_id,
            user_id=user_id,
            prompt=prompt,
            answer=result["answer"],
            model_id=result["model_id"],
            usage=result["usage"],
        )
        result["request_id"] = request_id
        result["quota"] = updated_quota
    except QuotaExceededError as error:
        quota_service.mark_blocked(event_id, str(error))
        return {
            "statusCode": 429,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(error), "quota": error.quota_snapshot, "request_id": request_id}),
        }
    except Exception as error:
        quota_service.mark_failed(event_id, str(error))
        LOGGER.exception("Research workflow failed")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(error), "request_id": request_id}),
        }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result),
    }
