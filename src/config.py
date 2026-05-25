import os


def _getenv(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _getboolenv(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BEDROCK_MODEL_ID = _getenv("BEDROCK_MODEL_ID", "au.anthropic.claude-sonnet-4-5-20250929-v1:0")
LITELLM_TOKENIZER_MODEL = _getenv("LITELLM_TOKENIZER_MODEL", "BEDROCK_MODEL_ID")
BEDROCK_REGION = _getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "ap-southeast-2"))
COGNITO_REGION = _getenv("COGNITO_REGION", BEDROCK_REGION)
COGNITO_USER_POOL_ID = _getenv("COGNITO_USER_POOL_ID", "")
COGNITO_APP_CLIENT_ID = _getenv("COGNITO_APP_CLIENT_ID", "")
JWT_ISSUER = _getenv("JWT_ISSUER", "")
JWT_JWKS_URL = _getenv("JWT_JWKS_URL", "")
MAX_PAGE_CHARS = int(_getenv("MAX_PAGE_CHARS", "18000"))
MAX_SOURCE_CHARS = int(_getenv("MAX_SOURCE_CHARS", "6000"))
BROWSER_IDENTIFIER = _getenv("BROWSER_IDENTIFIER", _getenv("BROWSER_ID", ""))
BROWSER_REGION = _getenv("BROWSER_REGION", BEDROCK_REGION)
REQUEST_TIMEOUT_SECONDS = int(_getenv("REQUEST_TIMEOUT_SECONDS", "8"))
TAVILY_API_KEY = _getenv("TAVILY_API_KEY", "")
IS_LOCAL_SAM = _getboolenv("AWS_SAM_LOCAL", False)
CHAT_SESSIONS_TABLE = _getenv("CHAT_SESSIONS_TABLE", "ChatSessions")
DYNAMODB_REGION = _getenv("DYNAMODB_REGION", "ap-southeast-2")
MAX_HISTORY_CHARS = int(_getenv("MAX_HISTORY_CHARS", "50000"))
TOKEN_BUDGETS_TABLE = _getenv("TOKEN_BUDGETS_TABLE", "DirectedGroup-UserTokenBudgets")
DEFAULT_DAILY_TOKEN_LIMIT = int(_getenv("DEFAULT_DAILY_TOKEN_LIMIT", "0"))
DEFAULT_MONTHLY_TOKEN_LIMIT = int(_getenv("DEFAULT_MONTHLY_TOKEN_LIMIT", "0"))
REQUEST_TOKEN_PRECHECK = int(_getenv("REQUEST_TOKEN_PRECHECK", "0"))
MODEL_INPUT_COST_PER_1K = float(_getenv("MODEL_INPUT_COST_PER_1K", "0"))
MODEL_OUTPUT_COST_PER_1K = float(_getenv("MODEL_OUTPUT_COST_PER_1K", "0"))
MAX_PROMPT_LENGTH = int(_getenv("MAX_PROMPT_LENGTH", "4000"))

# Request logging configuration
REQUEST_LOGS_TABLE = _getenv("REQUEST_LOGS_TABLE", "RequestLogs")
INPUT_TOKEN_COST_PER_1K = float(_getenv("INPUT_TOKEN_COST_PER_1K", "0.003"))  # Claude Sonnet input cost
OUTPUT_TOKEN_COST_PER_1K = float(_getenv("OUTPUT_TOKEN_COST_PER_1K", "0.015"))  # Claude Sonnet output cost
MODEL_NAME = _getenv("MODEL_NAME", "claude-sonnet-4-5")
