-- Synthetic training database for FinRAG Agentic extension.
-- This is NOT a real bank schema and contains no real customer data.
CREATE TABLE IF NOT EXISTS business_units (
    unit_id SMALLINT PRIMARY KEY,
    unit_name TEXT NOT NULL UNIQUE,
    rollout_group TEXT NOT NULL CHECK (rollout_group IN ('pilot', 'control'))
);

CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    unit_id SMALLINT NOT NULL REFERENCES business_units(unit_id),
    role_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS employee_cost_rates (
    rate_id BIGSERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(employee_id),
    valid_from DATE NOT NULL,
    valid_to DATE,
    hourly_cost_rub NUMERIC(12,2) NOT NULL CHECK (hourly_cost_rub >= 0),
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    UNIQUE (employee_id, valid_from)
);

CREATE TABLE IF NOT EXISTS evaluation_requests (
    request_id BIGINT PRIMARY KEY,
    unit_id SMALLINT NOT NULL REFERENCES business_units(unit_id),
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    complexity TEXT NOT NULL CHECK (complexity IN ('low', 'medium', 'high')),
    status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled', 'in_progress')),
    agent_assisted BOOLEAN NOT NULL DEFAULT FALSE,
    initiative_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS work_logs (
    work_log_id BIGSERIAL PRIMARY KEY,
    request_id BIGINT NOT NULL REFERENCES evaluation_requests(request_id),
    employee_id INTEGER NOT NULL REFERENCES employees(employee_id),
    work_date TIMESTAMP NOT NULL,
    task_type TEXT NOT NULL CHECK (task_type IN ('analysis', 'review', 'rework', 'coordination')),
    minutes_spent NUMERIC(10,2) NOT NULL CHECK (minutes_spent >= 0),
    UNIQUE (request_id, task_type)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id BIGSERIAL PRIMARY KEY,
    request_id BIGINT NOT NULL REFERENCES evaluation_requests(request_id),
    started_at TIMESTAMP NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'failed', 'fallback')),
    model_cost_rub NUMERIC(12,4) NOT NULL DEFAULT 0 CHECK (model_cost_rub >= 0),
    retry_count SMALLINT NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
    tool_calls_count SMALLINT NOT NULL DEFAULT 0 CHECK (tool_calls_count >= 0),
    UNIQUE (request_id, started_at)
);

CREATE TABLE IF NOT EXISTS quality_reviews (
    review_id BIGSERIAL PRIMARY KEY,
    request_id BIGINT NOT NULL REFERENCES evaluation_requests(request_id),
    reviewed_at TIMESTAMP NOT NULL,
    reviewer_type TEXT NOT NULL CHECK (reviewer_type IN ('human', 'automated')),
    score NUMERIC(5,4) CHECK (score >= 0 AND score <= 1),
    is_critical_error BOOLEAN NOT NULL DEFAULT FALSE,
    rework_minutes NUMERIC(10,2) NOT NULL DEFAULT 0 CHECK (rework_minutes >= 0),
    error_category TEXT,
    UNIQUE (request_id, reviewed_at)
);

CREATE TABLE IF NOT EXISTS agent_monthly_costs (
    cost_month DATE PRIMARY KEY,
    infrastructure_rub NUMERIC(14,2) NOT NULL DEFAULT 0,
    model_rub NUMERIC(14,2) NOT NULL DEFAULT 0,
    support_rub NUMERIC(14,2) NOT NULL DEFAULT 0,
    monitoring_rub NUMERIC(14,2) NOT NULL DEFAULT 0,
    CHECK (EXTRACT(DAY FROM cost_month) = 1),
    CHECK (infrastructure_rub >= 0 AND model_rub >= 0 AND support_rub >= 0 AND monitoring_rub >= 0)
);

CREATE TABLE IF NOT EXISTS initiative_cashflows (
    cashflow_id BIGSERIAL PRIMARY KEY,
    initiative_id TEXT NOT NULL,
    period_no INTEGER NOT NULL CHECK (period_no >= 0),
    period_date DATE NOT NULL,
    scenario TEXT NOT NULL CHECK (scenario IN ('conservative', 'base', 'optimistic')),
    cashflow_rub NUMERIC(16,2) NOT NULL,
    UNIQUE (initiative_id, period_no, scenario)
);

CREATE INDEX IF NOT EXISTS ix_requests_created ON evaluation_requests(created_at);
CREATE INDEX IF NOT EXISTS ix_requests_unit_created ON evaluation_requests(unit_id, created_at);
CREATE INDEX IF NOT EXISTS ix_work_logs_request ON work_logs(request_id);
CREATE INDEX IF NOT EXISTS ix_work_logs_employee_date ON work_logs(employee_id, work_date);
CREATE INDEX IF NOT EXISTS ix_agent_runs_request ON agent_runs(request_id);
CREATE INDEX IF NOT EXISTS ix_quality_reviews_request ON quality_reviews(request_id);


-- Least-privilege role used by the agent's SQL tools.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'finrag_reader') THEN
        CREATE ROLE finrag_reader LOGIN PASSWORD 'finrag_reader_password';
    END IF;
END
$$;
GRANT CONNECT ON DATABASE finrag_agentic TO finrag_reader;
GRANT USAGE ON SCHEMA public TO finrag_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO finrag_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE finrag IN SCHEMA public
    GRANT SELECT ON TABLES TO finrag_reader;
