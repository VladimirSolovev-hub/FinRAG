-- Deterministic synthetic sample: 4 business units x 5 months x 20 requests = 400 requests.
-- Pilot units use the agent starting July 2026; control units do not.
INSERT INTO business_units(unit_id, unit_name, rollout_group) VALUES
(1, 'Retail', 'pilot'),
(2, 'Corporate', 'pilot'),
(3, 'SME', 'control'),
(4, 'Cards', 'control')
ON CONFLICT (unit_id) DO NOTHING;

INSERT INTO employees(employee_id, unit_id, role_name) VALUES
(11,1,'analyst'),(12,1,'reviewer'),
(21,2,'analyst'),(22,2,'reviewer'),
(31,3,'analyst'),(32,3,'reviewer'),
(41,4,'analyst'),(42,4,'reviewer')
ON CONFLICT (employee_id) DO NOTHING;

INSERT INTO employee_cost_rates(employee_id, valid_from, valid_to, hourly_cost_rub)
SELECT employee_id, DATE '2026-01-01', NULL,
       CASE WHEN role_name='analyst' THEN 900 + unit_id*15 ELSE 1050 + unit_id*15 END
FROM employees
ON CONFLICT DO NOTHING;

WITH months AS (
    SELECT generate_series(0,4) AS month_idx
), seqs AS (
    SELECT generate_series(1,20) AS seq
), base AS (
    SELECT
        (u.unit_id::BIGINT * 100000 + m.month_idx::BIGINT * 100 + s.seq::BIGINT) AS request_id,
        u.unit_id,
        u.rollout_group,
        m.month_idx,
        s.seq,
        (DATE '2026-04-01' + (m.month_idx || ' months')::interval
          + INTERVAL '2 days'
          + (s.seq * 3 || ' hours')::interval)::timestamp AS created_at,
        CASE WHEN s.seq % 3 = 0 THEN 'high'
             WHEN s.seq % 3 = 1 THEN 'low' ELSE 'medium' END AS complexity,
        CASE WHEN s.seq % 10 = 0 THEN 'in_progress'
             WHEN s.seq % 17 = 0 THEN 'cancelled' ELSE 'completed' END AS status,
        (u.rollout_group='pilot' AND m.month_idx >= 3) AS agent_assisted,
        CASE WHEN s.seq % 4 = 0 THEN 'automation'
             WHEN s.seq % 4 = 1 THEN 'pricing'
             WHEN s.seq % 4 = 2 THEN 'marketing'
             ELSE 'product' END AS initiative_type
    FROM business_units u CROSS JOIN months m CROSS JOIN seqs s
)
INSERT INTO evaluation_requests(
    request_id, unit_id, created_at, completed_at, complexity, status, agent_assisted, initiative_type
)
SELECT
    request_id,
    unit_id,
    created_at,
    CASE WHEN status='completed'
         THEN created_at + INTERVAL '1 day' + (seq || ' minutes')::interval
         ELSE NULL END,
    complexity,
    status,
    agent_assisted,
    initiative_type
FROM base
ON CONFLICT (request_id) DO NOTHING;

-- Worklogs are split into analysis/review entries. Active human minutes are not wall-clock duration.
WITH req AS (
    SELECT r.*, bu.rollout_group,
        CASE
          WHEN bu.rollout_group='pilot' AND r.created_at < TIMESTAMP '2026-07-01' THEN 30
          WHEN bu.rollout_group='pilot' AND r.created_at >= TIMESTAMP '2026-07-01' THEN 10
          WHEN bu.rollout_group='control' AND r.created_at < TIMESTAMP '2026-07-01' THEN 28
          ELSE 25
        END
        + CASE r.complexity WHEN 'low' THEN 0 WHEN 'medium' THEN 5 ELSE 12 END
        + ((r.request_id % 7)::INTEGER - 3) AS total_minutes
    FROM evaluation_requests r JOIN business_units bu USING(unit_id)
    WHERE r.status='completed'
), expanded AS (
    SELECT request_id, unit_id, created_at, 1 AS part,
           (GREATEST(total_minutes,2)*0.75)::numeric(10,2) AS minutes_spent,
           'analysis'::text AS task_type,
           (unit_id*10+1) AS employee_id
    FROM req
    UNION ALL
    SELECT request_id, unit_id, created_at, 2,
           (GREATEST(total_minutes,2)*0.25)::numeric(10,2),
           'review', (unit_id*10+2)
    FROM req
)
INSERT INTO work_logs(request_id, employee_id, work_date, task_type, minutes_spent)
SELECT request_id, employee_id, created_at + (part || ' hours')::interval, task_type, minutes_spent
FROM expanded
WHERE (request_id % 29) <> 0
ON CONFLICT DO NOTHING;

-- One agent run per assisted request; some runs fail and fall back to a human.
INSERT INTO agent_runs(request_id, started_at, status, model_cost_rub, retry_count, fallback_used, tool_calls_count)
SELECT
    r.request_id,
    r.created_at + INTERVAL '5 minutes',
    CASE WHEN r.request_id % 13 = 0 THEN 'fallback'
         WHEN r.request_id % 19 = 0 THEN 'failed'
         ELSE 'success' END,
    ROUND((0.7 + (r.request_id % 8) * 0.13)::numeric, 4),
    CASE WHEN r.request_id % 11 = 0 THEN 2 WHEN r.request_id % 7 = 0 THEN 1 ELSE 0 END,
    (r.request_id % 13 = 0 OR r.request_id % 19 = 0),
    2 + (r.request_id % 4)
FROM evaluation_requests r
WHERE r.agent_assisted
ON CONFLICT DO NOTHING;

-- One quality review for completed requests. Synthetic critical errors are rare and visible.
INSERT INTO quality_reviews(request_id, reviewed_at, reviewer_type, score, is_critical_error, rework_minutes, error_category)
SELECT
    r.request_id,
    COALESCE(r.completed_at, r.created_at) + INTERVAL '2 hours',
    CASE WHEN r.request_id % 2 = 0 THEN 'human' ELSE 'automated' END,
    CASE WHEN r.request_id % 37 = 0 THEN 0.55
         WHEN r.agent_assisted AND r.request_id % 17 = 0 THEN 0.72
         ELSE 0.88 + ((r.request_id % 10)::numeric / 100.0) END,
    (r.request_id % 37 = 0),
    CASE WHEN r.request_id % 17 = 0 THEN 12 ELSE 0 END,
    CASE WHEN r.request_id % 37 = 0 THEN 'critical_calculation'
         WHEN r.request_id % 17 = 0 THEN 'missing_assumption'
         ELSE NULL END
FROM evaluation_requests r
WHERE r.status='completed'
ON CONFLICT DO NOTHING;

INSERT INTO agent_monthly_costs(cost_month, infrastructure_rub, model_rub, support_rub, monitoring_rub) VALUES
(DATE '2026-07-01', 80000, 170000, 200000, 50000),
(DATE '2026-08-01', 80000, 185000, 200000, 50000)
ON CONFLICT (cost_month) DO NOTHING;

-- Illustrative scenario cashflows for a separate initiative (not the agent's own savings).
INSERT INTO initiative_cashflows(initiative_id, period_no, period_date, scenario, cashflow_rub)
SELECT
    'AUTO_FIN_ASSESSMENT', month_no,
    (DATE '2026-07-01' + (month_no || ' months')::interval)::date,
    scenario,
    CASE scenario
      WHEN 'conservative' THEN CASE WHEN month_no=0 THEN -8000000 ELSE 2500000 END
      WHEN 'base' THEN CASE WHEN month_no=0 THEN -8000000 ELSE 5500000 END
      ELSE CASE WHEN month_no=0 THEN -8000000 ELSE 7000000 END
    END
FROM generate_series(0,12) AS month_no
CROSS JOIN (VALUES ('conservative'),('base'),('optimistic')) AS s(scenario)
ON CONFLICT (initiative_id, period_no, scenario) DO NOTHING;
