# Synthetic PostgreSQL data dictionary

All records are fabricated for interview practice. They do not represent a real bank, its customers or internal systems.

| Table | Grain | Primary purpose |
|---|---|---|
| `business_units` | One row per unit | Pilot/control assignment |
| `employees` | One row per demo employee | Unit and role lookup |
| `employee_cost_rates` | One effective-dated rate per employee and start date | Cost of active labour |
| `evaluation_requests` | One row per financial assessment request | Request status, complexity, created/completed time |
| `work_logs` | One row per request/task type | Active employee minutes |
| `agent_runs` | One row per execution attempt | Success, fallback, retry and per-run model cost |
| `quality_reviews` | One row per review event | Quality score, critical errors and rework |
| `agent_monthly_costs` | One row per month | Infrastructure, model invoice, support and monitoring |
| `initiative_cashflows` | One row per initiative/scenario/period | Illustrative scenario cash flows |

## Important grains and joins

- `evaluation_requests.request_id` is the grain for request-level process analysis.
- `work_logs` is one-to-many from a request. Aggregate by `request_id` before joining to other one-to-many tables.
- `agent_runs` is one-to-many because retries can produce several attempts. Use `COUNT(DISTINCT request_id)` for requests with agent usage; use `COUNT(run_id)` for attempts.
- `quality_reviews` is an event table. If multiple reviews are added, distinguish review-level averages from request-level latest/final quality.
- `employee_cost_rates` is effective-dated. Join each work log to the rate active on its work date. The demo has non-overlapping periods and a unique `(employee_id, valid_from)` key.
- `agent_monthly_costs.model_rub` is an invoice-level monthly amount. Do not add it to the sum of `agent_runs.model_cost_rub` unless their definitions are proven non-overlapping; the demo stores both separately for reconciliation practice.

## Synthetic assignment and rollout

- `Retail` and `Corporate` are `pilot` units.
- `SME` and `Cards` are `control` units.
- Agent usage begins in pilot units on 2026-07-01.
- Requests cover April–August 2026; 20 requests per unit and month.
- Some completed requests intentionally have no work log to create a missing-data case.
- DiD is descriptive training only. Four clusters do not support robust cluster-level causal inference.

## Cashflow query tool

`get_initiative_cashflows(initiative_id, scenario)` returns the ordered monthly/periodic cashflow rows, including period 0. Pass the returned `cashflow_rub` values in `period_no` order to `calculate_npv_from_cashflows`; use `periods_per_year=12` for monthly records. The seeded `AUTO_FIN_ASSESSMENT` scenarios are fabricated teaching data.
