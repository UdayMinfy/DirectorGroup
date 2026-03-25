INSERT INTO token_plans (plan_name, daily_token_limit, monthly_token_limit, is_active)
VALUES
    ('default', 20000, 300000, TRUE),
    ('tiny-daily', 1500, 10000, TRUE)
ON CONFLICT (plan_name) DO UPDATE
SET
    daily_token_limit = EXCLUDED.daily_token_limit,
    monthly_token_limit = EXCLUDED.monthly_token_limit,
    is_active = EXCLUDED.is_active,
    updated_at = NOW();

INSERT INTO users2 (external_user_id, status, plan_id)
SELECT 'demo-user-1', 'active', id
FROM token_plans
WHERE plan_name = 'default'
ON CONFLICT (external_user_id) DO UPDATE
SET
    status = EXCLUDED.status,
    plan_id = EXCLUDED.plan_id,
    updated_at = NOW();

INSERT INTO users2 (external_user_id, status, plan_id)
SELECT 'demo-user-2', 'active', id
FROM token_plans
WHERE plan_name = 'default'
ON CONFLICT (external_user_id) DO UPDATE
SET
    status = EXCLUDED.status,
    plan_id = EXCLUDED.plan_id,
    updated_at = NOW();

INSERT INTO users2 (external_user_id, status, plan_id)
SELECT 'quota-blocked-user', 'active', id
FROM token_plans
WHERE plan_name = 'tiny-daily'
ON CONFLICT (external_user_id) DO UPDATE
SET
    status = EXCLUDED.status,
    plan_id = EXCLUDED.plan_id,
    updated_at = NOW();

INSERT INTO user_token_usage_summary (
    user_id,
    period_type,
    period_key,
    input_tokens_used,
    output_tokens_used,
    total_tokens_used,
    request_count,
    last_request_at
)
SELECT
    u.id,
    'daily',
    CURRENT_DATE::text,
    1000,
    500,
    1500,
    1,
    NOW()
FROM users2 u
WHERE u.external_user_id = 'quota-blocked-user'
ON CONFLICT (user_id, period_type, period_key) DO UPDATE
SET
    input_tokens_used = EXCLUDED.input_tokens_used,
    output_tokens_used = EXCLUDED.output_tokens_used,
    total_tokens_used = EXCLUDED.total_tokens_used,
    request_count = EXCLUDED.request_count,
    last_request_at = EXCLUDED.last_request_at,
    updated_at = NOW();

INSERT INTO user_token_usage_summary (
    user_id,
    period_type,
    period_key,
    input_tokens_used,
    output_tokens_used,
    total_tokens_used,
    request_count,
    last_request_at
)
SELECT
    u.id,
    'monthly',
    TO_CHAR(CURRENT_DATE, 'YYYY-MM'),
    1000,
    500,
    1500,
    1,
    NOW()
FROM users2 u
WHERE u.external_user_id = 'quota-blocked-user'
ON CONFLICT (user_id, period_type, period_key) DO UPDATE
SET
    input_tokens_used = EXCLUDED.input_tokens_used,
    output_tokens_used = EXCLUDED.output_tokens_used,
    total_tokens_used = EXCLUDED.total_tokens_used,
    request_count = EXCLUDED.request_count,
    last_request_at = EXCLUDED.last_request_at,
    updated_at = NOW();
