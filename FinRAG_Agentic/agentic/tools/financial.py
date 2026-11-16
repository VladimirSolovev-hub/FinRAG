from __future__ import annotations
import json
from langchain_core.tools import tool
from agentic.tools.finance import calculate_business_case as _calculate, calculate_periodic_cashflows as _calculate_flows


@tool
def calculate_business_case(
    monthly_volume: int,
    baseline_minutes: float,
    agent_minutes: float,
    hourly_cost_rub: float,
    monetization_rate: float,
    monthly_fixed_opex_rub: float,
    implementation_cost_rub: float,
    annual_discount_rate: float,
    forecast_months: int = 36,
    variable_agent_cost_per_request_rub: float = 0.0,
    additional_human_minutes_per_request: float = 0.0,
    incremental_value_per_request_rub: float = 0.0,
    monthly_volume_growth_rate: float = 0.0,
    ramp_months: int = 1,
) -> str:
    """Детерминированно рассчитать финансовый бизнес-кейс.

    monetization_rate — доля стоимости высвобождённой мощности, которая реально
    превращается в выгоду (0..1). annual_discount_rate — эффективная годовая ставка
    (например, 0.20 для 20%). Первоначальные инвестиции — поток t=0.
    """
    result = _calculate(
        monthly_volume=monthly_volume,
        baseline_minutes=baseline_minutes,
        agent_minutes=agent_minutes,
        hourly_cost_rub=hourly_cost_rub,
        monetization_rate=monetization_rate,
        monthly_fixed_opex_rub=monthly_fixed_opex_rub,
        implementation_cost_rub=implementation_cost_rub,
        annual_discount_rate=annual_discount_rate,
        forecast_months=forecast_months,
        variable_agent_cost_per_request_rub=variable_agent_cost_per_request_rub,
        additional_human_minutes_per_request=additional_human_minutes_per_request,
        incremental_value_per_request_rub=incremental_value_per_request_rub,
        monthly_volume_growth_rate=monthly_volume_growth_rate,
        ramp_months=ramp_months,
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def calculate_npv_from_cashflows(
    cash_flows: list[float],
    annual_discount_rate: float,
    periods_per_year: int = 12,
) -> str:
    """Рассчитать NPV, IRR и окупаемость по готовому ряду денежных потоков.

    cash_flows[0] — поток в t=0 (обычно CAPEX со знаком минус); последующие
    элементы — равномерно расположенные потоки. periods_per_year=12 для месяцев,
    1 для годовых потоков. annual_discount_rate — эффективная годовая ставка.
    """
    result = _calculate_flows(
        cash_flows=cash_flows,
        annual_discount_rate=annual_discount_rate,
        periods_per_year=periods_per_year,
    )
    return json.dumps(result, ensure_ascii=False)
