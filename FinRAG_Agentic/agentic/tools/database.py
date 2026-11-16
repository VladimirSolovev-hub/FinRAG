from __future__ import annotations
import json
from datetime import date
from typing import Any

from langchain_core.tools import tool
from agentic.db import fetch_all


def _iso_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть датой ISO YYYY-MM-DD") from exc


@tool
def get_process_kpis(start_date: str, end_date: str) -> str:
    """Получить KPI процесса по пилотной и контрольной группам за период.

    Даты задаются в ISO YYYY-MM-DD; end_date не включается. Возвращает число
    заявок, completion rate, средние/медианные активные трудозатраты и стоимость
    труда на завершённую оценку. БД учебная, синтетическая.
    """
    start, end = _iso_date(start_date, "start_date"), _iso_date(end_date, "end_date")
    if end <= start:
        raise ValueError("end_date должен быть позже start_date")
    sql = """WITH effort AS (
        SELECT request_id, SUM(minutes_spent) AS human_minutes
        FROM work_logs GROUP BY request_id
    ), labor_cost AS (
        SELECT wl.request_id,
               SUM(wl.minutes_spent * er.hourly_cost_rub / 60.0) AS labor_cost_rub
        FROM work_logs wl
        JOIN employee_cost_rates er
          ON er.employee_id = wl.employee_id
         AND wl.work_date >= er.valid_from
         AND (er.valid_to IS NULL OR wl.work_date < er.valid_to)
        GROUP BY wl.request_id
    )
    SELECT bu.rollout_group,
           COUNT(*) AS total_requests,
           COUNT(*) FILTER (WHERE r.status='completed') AS completed_requests,
           ROUND(100.0 * COUNT(*) FILTER (WHERE r.status='completed') / NULLIF(COUNT(*),0), 2) AS completion_rate_pct,
           ROUND(AVG(e.human_minutes) FILTER (WHERE r.status='completed'), 2) AS avg_human_minutes,
           ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY e.human_minutes) FILTER (WHERE r.status='completed'))::numeric, 2) AS median_human_minutes,
           COUNT(*) FILTER (WHERE r.status='completed' AND e.human_minutes IS NULL) AS completed_requests_without_worklogs,
           ROUND(AVG(lc.labor_cost_rub) FILTER (WHERE r.status='completed'), 2) AS avg_labor_cost_per_completed_request_rub
    FROM evaluation_requests r
    JOIN business_units bu USING(unit_id)
    LEFT JOIN effort e USING(request_id)
    LEFT JOIN labor_cost lc USING(request_id)
    WHERE r.created_at >= %s AND r.created_at < %s
    GROUP BY bu.rollout_group
    ORDER BY bu.rollout_group"""
    rows = fetch_all(sql, (start, end))
    return json.dumps({"data_type": "synthetic_demo", "period": [start_date, end_date], "rows": rows}, ensure_ascii=False, default=str)


@tool
def get_did_effect(start_date: str, end_date: str, rollout_date: str) -> str:
    """Оценить описательную DiD-разницу в средних активных минутах на заявку.

    start_date и end_date задают общий период, end_date не включается;
    rollout_date — дата начала внедрения. Возвращает четыре средних и DiD.
    Сам по себе расчёт не доказывает причинность: требуется проверка параллельных трендов.
    """
    start, end = _iso_date(start_date, "start_date"), _iso_date(end_date, "end_date")
    rollout = _iso_date(rollout_date, "rollout_date")
    if end <= start or not start < rollout < end:
        raise ValueError("Требуется start_date < rollout_date < end_date")
    sql = """WITH effort AS (
        SELECT request_id, SUM(minutes_spent) AS human_minutes
        FROM work_logs GROUP BY request_id
    ), request_level AS (
        SELECT bu.rollout_group,
               CASE WHEN r.created_at < %s THEN 'before' ELSE 'after' END AS period,
               e.human_minutes
        FROM evaluation_requests r
        JOIN business_units bu USING(unit_id)
        LEFT JOIN effort e USING(request_id)
        WHERE r.created_at >= %s AND r.created_at < %s
          AND r.status='completed'
    ), means AS (
        SELECT rollout_group, period, AVG(human_minutes) AS avg_minutes,
               COUNT(*) AS completed_requests,
               COUNT(human_minutes) AS requests_with_worklogs,
               COUNT(*) FILTER (WHERE human_minutes IS NULL) AS requests_missing_worklogs
        FROM request_level GROUP BY rollout_group, period
    ), p AS (
        SELECT
          MAX(avg_minutes) FILTER (WHERE rollout_group='pilot' AND period='before') AS pilot_before,
          MAX(avg_minutes) FILTER (WHERE rollout_group='pilot' AND period='after') AS pilot_after,
          MAX(avg_minutes) FILTER (WHERE rollout_group='control' AND period='before') AS control_before,
          MAX(avg_minutes) FILTER (WHERE rollout_group='control' AND period='after') AS control_after,
          MAX(completed_requests) FILTER (WHERE rollout_group='pilot' AND period='before') AS pilot_before_n,
          MAX(completed_requests) FILTER (WHERE rollout_group='pilot' AND period='after') AS pilot_after_n,
          MAX(completed_requests) FILTER (WHERE rollout_group='control' AND period='before') AS control_before_n,
          MAX(completed_requests) FILTER (WHERE rollout_group='control' AND period='after') AS control_after_n,
          MAX(requests_missing_worklogs) FILTER (WHERE rollout_group='pilot' AND period='before') AS pilot_before_missing_logs,
          MAX(requests_missing_worklogs) FILTER (WHERE rollout_group='pilot' AND period='after') AS pilot_after_missing_logs,
          MAX(requests_missing_worklogs) FILTER (WHERE rollout_group='control' AND period='before') AS control_before_missing_logs,
          MAX(requests_missing_worklogs) FILTER (WHERE rollout_group='control' AND period='after') AS control_after_missing_logs
        FROM means
    )
    SELECT pilot_before, pilot_after, control_before, control_after,
           pilot_before_n, pilot_after_n, control_before_n, control_after_n,
           pilot_before_missing_logs, pilot_after_missing_logs,
           control_before_missing_logs, control_after_missing_logs,
           (pilot_after-pilot_before)-(control_after-control_before) AS did_minutes,
           -((pilot_after-pilot_before)-(control_after-control_before)) AS estimated_minutes_saved
    FROM p"""
    rows = fetch_all(sql, (rollout, start, end))
    return json.dumps({
        "data_type": "synthetic_demo",
        "period": [start_date, end_date],
        "rollout_date": rollout_date,
        "rows": rows,
        "interpretation_warning": "Descriptive DiD only. With four units (two pilot, two control), cluster-level causal inference is weak; inspect pre-trends and concurrent changes.",
    }, ensure_ascii=False, default=str)


@tool
def get_quality_metrics(start_date: str, end_date: str) -> str:
    """Получить качество финансовых оценок и долю критических ошибок за период."""
    start, end = _iso_date(start_date, "start_date"), _iso_date(end_date, "end_date")
    if end <= start:
        raise ValueError("end_date должен быть позже start_date")
    sql = """SELECT bu.rollout_group,
        COUNT(DISTINCT qr.request_id) AS reviewed_requests,
        ROUND(AVG(qr.score),4) AS avg_quality_score,
        COUNT(DISTINCT qr.request_id) FILTER (WHERE qr.is_critical_error) AS critical_error_requests,
        ROUND(100.0 * COUNT(DISTINCT qr.request_id) FILTER (WHERE qr.is_critical_error)
              / NULLIF(COUNT(DISTINCT qr.request_id),0),2) AS critical_error_rate_pct,
        SUM(qr.rework_minutes) AS total_rework_minutes,
        ROUND(AVG(qr.rework_minutes),2) AS avg_rework_minutes_per_review
    FROM quality_reviews qr
    JOIN evaluation_requests r USING(request_id)
    JOIN business_units bu USING(unit_id)
    WHERE qr.reviewed_at >= %s AND qr.reviewed_at < %s
    GROUP BY bu.rollout_group ORDER BY bu.rollout_group"""
    rows = fetch_all(sql, (start, end))
    return json.dumps({"data_type": "synthetic_demo", "period": [start_date, end_date], "rows": rows}, ensure_ascii=False, default=str)


@tool
def get_agent_run_metrics(start_date: str, end_date: str) -> str:
    """Получить метрики использования AI-агента: запуски, success, fallback, retries и model cost."""
    start, end = _iso_date(start_date, "start_date"), _iso_date(end_date, "end_date")
    if end <= start:
        raise ValueError("end_date должен быть позже start_date")
    sql = """SELECT bu.rollout_group,
        COUNT(ar.run_id) AS total_runs,
        COUNT(DISTINCT ar.request_id) AS requests_with_agent,
        COUNT(ar.run_id) FILTER (WHERE ar.status='success') AS successful_runs,
        ROUND(100.0 * COUNT(ar.run_id) FILTER (WHERE ar.status='success') / NULLIF(COUNT(ar.run_id),0),2) AS run_success_rate_pct,
        COUNT(ar.run_id) FILTER (WHERE ar.fallback_used) AS fallback_runs,
        ROUND(100.0 * COUNT(ar.run_id) FILTER (WHERE ar.fallback_used) / NULLIF(COUNT(ar.run_id),0),2) AS fallback_rate_pct,
        ROUND(AVG(ar.retry_count),2) AS avg_retry_count,
        ROUND(SUM(ar.model_cost_rub),2) AS model_cost_from_runs_rub
    FROM evaluation_requests r
    JOIN business_units bu USING(unit_id)
    LEFT JOIN agent_runs ar USING(request_id)
    WHERE r.created_at >= %s AND r.created_at < %s
      AND r.agent_assisted = TRUE
    GROUP BY bu.rollout_group ORDER BY bu.rollout_group"""
    rows = fetch_all(sql, (start, end))
    return json.dumps({"data_type": "synthetic_demo", "period": [start_date, end_date], "rows": rows}, ensure_ascii=False, default=str)


@tool
def get_complexity_mix(start_date: str, end_date: str) -> str:
    """Проверить состав заявок по сложности между pilot/control за заданный период."""
    start, end = _iso_date(start_date, "start_date"), _iso_date(end_date, "end_date")
    if end <= start:
        raise ValueError("end_date должен быть позже start_date")
    sql = """SELECT bu.rollout_group, r.complexity, COUNT(*) AS request_count,
        COUNT(*) FILTER (WHERE r.status='completed') AS completed_count
    FROM evaluation_requests r JOIN business_units bu USING(unit_id)
    WHERE r.created_at >= %s AND r.created_at < %s
    GROUP BY bu.rollout_group, r.complexity
    ORDER BY bu.rollout_group, r.complexity"""
    rows = fetch_all(sql, (start, end))
    return json.dumps({"data_type": "synthetic_demo", "period": [start_date, end_date], "rows": rows}, ensure_ascii=False, default=str)


@tool
def get_initiative_cashflows(initiative_id: str, scenario: str = "base") -> str:
    """Получить денежные потоки конкретной инициативы по сценариям.

    scenario принимает conservative, base или optimistic. period_no=0 —
    первоначальный денежный поток; последующие периоды идут по возрастанию.
    Данные демонстрационные.
    """
    allowed = {"conservative", "base", "optimistic"}
    if scenario not in allowed:
        raise ValueError(f"scenario должен быть одним из: {', '.join(sorted(allowed))}")
    if not initiative_id.strip():
        raise ValueError("initiative_id не должен быть пустым")
    sql = """SELECT initiative_id, scenario, period_no, period_date, cashflow_rub
    FROM initiative_cashflows
    WHERE initiative_id = %s AND scenario = %s
    ORDER BY period_no"""
    rows = fetch_all(sql, (initiative_id, scenario))
    if not rows:
        return json.dumps({"status": "not_found", "initiative_id": initiative_id, "scenario": scenario, "cashflows": []}, ensure_ascii=False)
    return json.dumps({"status": "ok", "data_type": "synthetic_demo", "initiative_id": initiative_id, "scenario": scenario, "cashflows": rows}, ensure_ascii=False, default=str)


@tool
def get_agent_monthly_costs(start_date: str, end_date: str) -> str:
    """Получить месячные расходы за календарные месяцы, пересекающие интервал [start_date, end_date).

    Значение месяца возвращается целиком; частичное пропорционирование неполного месяца не выполняется.
    """
    start, end = _iso_date(start_date, "start_date"), _iso_date(end_date, "end_date")
    if end <= start:
        raise ValueError("end_date должен быть позже start_date")
    sql = """SELECT cost_month, infrastructure_rub, model_rub, support_rub, monitoring_rub,
        (infrastructure_rub+model_rub+support_rub+monitoring_rub) AS total_opex_rub
    FROM agent_monthly_costs
    WHERE cost_month >= date_trunc('month', %s::date)::date
      AND cost_month < (date_trunc('month', %s::date - INTERVAL '1 day') + INTERVAL '1 month')::date
    ORDER BY cost_month"""
    rows = fetch_all(sql, (start, end))
    return json.dumps({"data_type": "synthetic_demo", "rows": rows}, ensure_ascii=False, default=str)
