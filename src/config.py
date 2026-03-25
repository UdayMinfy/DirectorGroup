import os


def _getenv(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


BEDROCK_MODEL_ID = _getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
BEDROCK_REGION = _getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
SEARCH_RESULT_LIMIT = int(_getenv("SEARCH_RESULT_LIMIT", "3"))
MAX_PAGE_CHARS = int(_getenv("MAX_PAGE_CHARS", "12000"))
MAX_SOURCE_CHARS = int(_getenv("MAX_SOURCE_CHARS", "4000"))
BROWSER_IDENTIFIER = _getenv("BROWSER_IDENTIFIER", "")
BROWSER_REGION = _getenv("BROWSER_REGION", BEDROCK_REGION)
REQUEST_TIMEOUT_SECONDS = int(_getenv("REQUEST_TIMEOUT_SECONDS", "12"))
SEARCH_API_PROVIDER = _getenv("SEARCH_API_PROVIDER", "tavily")
TAVILY_API_KEY = _getenv("TAVILY_API_KEY", "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")
DB_HOST = _getenv("DB_HOST", "host.docker.internal")
DB_PORT = int(_getenv("DB_PORT", "5433"))
DB_NAME = _getenv("DB_NAME", "ecommerce_db")
DB_USER = _getenv("DB_USER", "ecommerce_user")
DB_PASSWORD = _getenv("DB_PASSWORD", "ecommerce_pass")
DEFAULT_DAILY_TOKEN_LIMIT = int(_getenv("DEFAULT_DAILY_TOKEN_LIMIT", "20000"))
DEFAULT_MONTHLY_TOKEN_LIMIT = int(_getenv("DEFAULT_MONTHLY_TOKEN_LIMIT", "300000"))
