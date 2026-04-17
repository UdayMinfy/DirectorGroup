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
BEDROCK_REGION = _getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "ap-southeast-2"))
JWT_VALIDATION_ENABLED = _getboolenv("JWT_VALIDATION_ENABLED", True)
COGNITO_REGION = _getenv("COGNITO_REGION", BEDROCK_REGION)
COGNITO_USER_POOL_ID = _getenv("COGNITO_USER_POOL_ID", "")
COGNITO_APP_CLIENT_ID = _getenv("COGNITO_APP_CLIENT_ID", "")
JWT_ISSUER = _getenv("JWT_ISSUER", "")
JWT_JWKS_URL = _getenv("JWT_JWKS_URL", "")
HARDCODED_USER_EMAIL = _getenv("HARDCODED_USER_EMAIL", "")
SEARCH_RESULT_LIMIT = int(_getenv("SEARCH_RESULT_LIMIT", "10"))
SEARCH_CANDIDATE_LIMIT = int(_getenv("SEARCH_CANDIDATE_LIMIT", "10"))
MIN_SEARCH_RESULTS = int(_getenv("MIN_SEARCH_RESULTS", "7"))
EXTRACTION_SUCCESS_TARGET = int(_getenv("EXTRACTION_SUCCESS_TARGET", "2"))
MAX_PAGE_CHARS = int(_getenv("MAX_PAGE_CHARS", "18000"))
MAX_SOURCE_CHARS = int(_getenv("MAX_SOURCE_CHARS", "6000"))
BROWSER_IDENTIFIER = _getenv("BROWSER_IDENTIFIER", _getenv("BROWSER_ID", ""))
BROWSER_REGION = _getenv("BROWSER_REGION", BEDROCK_REGION)
REQUEST_TIMEOUT_SECONDS = int(_getenv("REQUEST_TIMEOUT_SECONDS", "8"))
SEARCH_API_PROVIDER = _getenv("SEARCH_API_PROVIDER", "tavily")
TAVILY_API_KEY = _getenv("TAVILY_API_KEY", "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")
TAVILY_INCLUDE_ANSWER = _getboolenv("TAVILY_INCLUDE_ANSWER", True)
USE_TAVILY_RESULT_ORDER = _getboolenv("USE_TAVILY_RESULT_ORDER", True)
SEARCH_TIMEZONE = _getenv("SEARCH_TIMEZONE", "Asia/Kolkata")
IS_LOCAL_SAM = _getboolenv("AWS_SAM_LOCAL", False)
USE_BROWSER = _getboolenv("USE_BROWSER", not IS_LOCAL_SAM)
CHAT_SESSIONS_TABLE = _getenv("CHAT_SESSIONS_TABLE", "ChatSessions")
DYNAMODB_REGION = _getenv("DYNAMODB_REGION", "ap-southeast-2")
MAX_HISTORY_MESSAGES = int(_getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_HISTORY_CHARS = int(_getenv("MAX_HISTORY_CHARS", "6000"))
TOKEN_BUDGETS_TABLE = _getenv("TOKEN_BUDGETS_TABLE", "DirectedGroup-UserTokenBudgets")
DB_HOST = _getenv("DB_HOST", "host.docker.internal")
DB_PORT = int(_getenv("DB_PORT", "5433"))
DB_NAME = _getenv("DB_NAME", "ecommerce_db")
DB_USER = _getenv("DB_USER", "ecommerce_user")
DB_PASSWORD = _getenv("DB_PASSWORD", "ecommerce_pass")
DEFAULT_DAILY_TOKEN_LIMIT = int(_getenv("DEFAULT_DAILY_TOKEN_LIMIT", "0"))
DEFAULT_MONTHLY_TOKEN_LIMIT = int(_getenv("DEFAULT_MONTHLY_TOKEN_LIMIT", "0"))
REQUEST_TOKEN_PRECHECK = int(_getenv("REQUEST_TOKEN_PRECHECK", "0"))
MODEL_INPUT_COST_PER_1K = float(_getenv("MODEL_INPUT_COST_PER_1K", "0"))
MODEL_OUTPUT_COST_PER_1K = float(_getenv("MODEL_OUTPUT_COST_PER_1K", "0"))

