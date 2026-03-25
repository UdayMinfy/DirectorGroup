CREATE TABLE IF NOT EXISTS token_plans (
    id BIGSERIAL PRIMARY KEY,
    plan_name TEXT NOT NULL UNIQUE,
    daily_token_limit INTEGER NOT NULL,
    monthly_token_limit INTEGER NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users2 (
    id BIGSERIAL PRIMARY KEY,
    external_user_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active',
    plan_id BIGINT NOT NULL REFERENCES token_plans(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_token_usage_events (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users2(id),
    request_id UUID NOT NULL UNIQUE,
    prompt TEXT,
    answer_preview TEXT,
    model_id TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_message TEXT,
    usage_date DATE NOT NULL,
    usage_month TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS user_token_usage_summary (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users2(id),
    period_type TEXT NOT NULL,
    period_key TEXT NOT NULL,
    input_tokens_used INTEGER NOT NULL DEFAULT 0,
    output_tokens_used INTEGER NOT NULL DEFAULT 0,
    total_tokens_used INTEGER NOT NULL DEFAULT 0,
    request_count INTEGER NOT NULL DEFAULT 0,
    last_request_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, period_type, period_key)
);

INSERT INTO token_plans (plan_name, daily_token_limit, monthly_token_limit, is_active)
VALUES ('default', 20000, 300000, TRUE)
ON CONFLICT (plan_name) DO NOTHING;
