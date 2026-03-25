SELECT
    u.id,
    u.external_user_id,
    p.plan_name,
    p.daily_token_limit,
    p.monthly_token_limit
FROM users2 u
JOIN token_plans p ON p.id = u.plan_id
ORDER BY u.id;

SELECT
    u.external_user_id,
    s.period_type,
    s.period_key,
    s.input_tokens_used,
    s.output_tokens_used,
    s.total_tokens_used,
    s.request_count,
    s.last_request_at
FROM user_token_usage_summary s
JOIN users2 u ON u.id = s.user_id
ORDER BY u.external_user_id, s.period_type, s.period_key;

SELECT
    u.external_user_id,
    e.request_id,
    e.status,
    e.model_id,
    e.input_tokens,
    e.output_tokens,
    e.total_tokens,
    e.usage_date,
    e.created_at,
    e.completed_at
FROM user_token_usage_events e
JOIN users2 u ON u.id = e.user_id
ORDER BY e.created_at DESC
LIMIT 20;
