CREATE TABLE IF NOT EXISTS users2 (
    id BIGSERIAL PRIMARY KEY,
    external_user_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active',
    plan_id BIGINT NOT NULL REFERENCES token_plans(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT setval(
    pg_get_serial_sequence('users2', 'id'),
    COALESCE((SELECT MAX(id) FROM users2), 1),
    TRUE
);

ALTER TABLE user_token_usage_events DROP CONSTRAINT IF EXISTS user_token_usage_events_user_id_fkey;
ALTER TABLE user_token_usage_summary DROP CONSTRAINT IF EXISTS user_token_usage_summary_user_id_fkey;

ALTER TABLE user_token_usage_events
    ADD CONSTRAINT user_token_usage_events_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users2(id);

ALTER TABLE user_token_usage_summary
    ADD CONSTRAINT user_token_usage_summary_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users2(id);
