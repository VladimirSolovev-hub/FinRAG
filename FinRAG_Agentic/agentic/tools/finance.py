from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Sequence
import math


@dataclass
class FinancialCaseResult:
    currency: str
    minutes_saved_per_request: float
    effective_minutes_saved_per_request: float
    monthly_capacity_hours: float
    monthly_capacity_value_rub: float
    monetization_rate: float
    realized_labor_benefit_rub: float
    incremental_value_rub: float
    monthly_opex_rub: float
    monthly_net_benefit_rub: float
    implementation_cost_rub: float
    simple_payback_months: float | None
    discounted_payback_months: float | None
    npv_rub: float
    roi_pct: float
    monthly_irr_pct: float | None
    annualized_irr_pct: float | None
    forecast_months: int
    assumptions: list[str]
    warnings: list[str]
    monthly_cashflows: list[float]


def _npv_monthly(rate: float, flows: Sequence[float]) -> float:
    if rate <= -1:
        raise ValueError("Месячная ставка дисконтирования должна быть выше -100%")
    return sum(cf / ((1.0 + rate) ** period) for period, cf in enumerate(flows))


def _irr_bisection(flows: Sequence[float]) -> float | None:
    """Monthly IRR for conventional cashflows with exactly one sign change."""
    signs = [1 if x > 1e-9 else -1 for x in flows if abs(x) > 1e-9]
    sign_changes = sum(a != b for a, b in zip(signs, signs[1:]))
    if sign_changes != 1:
        return None
    low, high = -0.9999, 1.0
    f_low, f_high = _npv_monthly(low, flows), _npv_monthly(high, flows)
    while f_low * f_high > 0 and high < 1_000_000:
        high = high * 2 + 1
        f_high = _npv_monthly(high, flows)
    if f_low * f_high > 0:
        return None
    for _ in range(250):
        mid = (low + high) / 2
        f_mid = _npv_monthly(mid, flows)
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2


def _payback_months(flows: Sequence[float], discounted: bool = False, monthly_rate: float = 0.0) -> float | None:
    if flows and abs(flows[0]) < 1e-9:
        return 0.0
    cumulative = 0.0
    for month, amount in enumerate(flows):
        value = amount / ((1 + monthly_rate) ** month) if discounted else amount
        previous = cumulative
        cumulative += value
        if month > 0 and previous < 0 <= cumulative:
            if value <= 0:
                return float(month)
            fraction = -previous / value
            return (month - 1) + fraction
    return None


def calculate_periodic_cashflows(
    cash_flows: Sequence[float],
    annual_discount_rate: float,
    periods_per_year: int = 12,
) -> dict:
    """Calculate NPV/IRR/payback for periodic cashflows, with t=0 at index 0."""
    if not cash_flows:
        raise ValueError("cash_flows должен содержать хотя бы один период t=0")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year должен быть > 0")
    if not math.isfinite(float(annual_discount_rate)) or annual_discount_rate <= -1:
        raise ValueError("annual_discount_rate должен быть конечным и выше -100%")
    values = [float(x) for x in cash_flows]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Все денежные потоки должны быть конечными числами")
    periodic_rate = (1 + annual_discount_rate) ** (1 / periods_per_year) - 1
    npv = _npv_monthly(periodic_rate, values)
    periodic_irr = _irr_bisection(values)
    annualized_irr = ((1 + periodic_irr) ** periods_per_year - 1) if periodic_irr is not None else None
    payback_periods = _payback_months(values)
    discounted_payback_periods = _payback_months(values, discounted=True, monthly_rate=periodic_rate)
    return {
        "npv_rub": npv,
        "annual_discount_rate": annual_discount_rate,
        "periodic_discount_rate": periodic_rate,
        "periods_per_year": periods_per_year,
        "periodic_irr_pct": periodic_irr * 100 if periodic_irr is not None else None,
        "annualized_irr_pct": annualized_irr * 100 if annualized_irr is not None else None,
        "simple_payback_periods": payback_periods,
        "discounted_payback_periods": discounted_payback_periods,
        "cashflow_count_including_t0": len(values),
        "assumptions": [
            "cash_flows[0] is the initial cashflow at t=0",
            "all subsequent cashflows are equally spaced",
            "annual_discount_rate is an effective annual rate",
        ],
        "warnings": [] if periodic_irr is not None else ["IRR is not reported: a unique conventional root was not found."],
    }


def calculate_business_case(
    *,
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
) -> dict:
    """Build monthly cashflows. Released capacity is monetized only by monetization_rate.

    annual_discount_rate is an effective annual rate; it is converted to an
    effective monthly rate. Initial capex is a t=0 cashflow. Monthly operations
    begin in month 1. All values are nominal RUB.
    """
    if monthly_volume < 0 or forecast_months <= 0 or ramp_months <= 0:
        raise ValueError("Объём должен быть >= 0, горизонт и ramp_months должны быть > 0")
    if baseline_minutes < 0 or agent_minutes < 0:
        raise ValueError("Время обработки не может быть отрицательным")
    numeric_values = {
        "baseline_minutes": baseline_minutes,
        "agent_minutes": agent_minutes,
        "hourly_cost_rub": hourly_cost_rub,
        "monetization_rate": monetization_rate,
        "monthly_fixed_opex_rub": monthly_fixed_opex_rub,
        "implementation_cost_rub": implementation_cost_rub,
        "annual_discount_rate": annual_discount_rate,
        "variable_agent_cost_per_request_rub": variable_agent_cost_per_request_rub,
        "additional_human_minutes_per_request": additional_human_minutes_per_request,
        "incremental_value_per_request_rub": incremental_value_per_request_rub,
        "monthly_volume_growth_rate": monthly_volume_growth_rate,
    }
    for name, value in numeric_values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} должен быть конечным числом")
    numeric_nonnegative = {
        "hourly_cost_rub": hourly_cost_rub,
        "monthly_fixed_opex_rub": monthly_fixed_opex_rub,
        "implementation_cost_rub": implementation_cost_rub,
        "variable_agent_cost_per_request_rub": variable_agent_cost_per_request_rub,
        "additional_human_minutes_per_request": additional_human_minutes_per_request,
        "incremental_value_per_request_rub": incremental_value_per_request_rub,
    }
    for name, value in numeric_nonnegative.items():
        if value < 0:
            raise ValueError(f"{name} не может быть отрицательным")
    if not 0 <= monetization_rate <= 1:
        raise ValueError("monetization_rate должен находиться в диапазоне [0, 1]")
    if annual_discount_rate <= -1:
        raise ValueError("annual_discount_rate должен быть выше -100%")
    if monthly_volume_growth_rate <= -1:
        raise ValueError("monthly_volume_growth_rate должен быть выше -100%")

    effective_minutes = baseline_minutes - agent_minutes - additional_human_minutes_per_request
    capacity_hours = monthly_volume * effective_minutes / 60.0
    capacity_value = capacity_hours * hourly_cost_rub
    monthly_rate = (1 + annual_discount_rate) ** (1 / 12) - 1

    cashflows = [-float(implementation_cost_rub)]
    monthly_benefits: list[float] = []
    monthly_opex_values: list[float] = []
    monthly_net_values: list[float] = []
    for month in range(1, forecast_months + 1):
        ramp = min(1.0, month / ramp_months)
        volume = monthly_volume * ((1 + monthly_volume_growth_rate) ** (month - 1))
        monthly_capacity_value = volume * effective_minutes / 60.0 * hourly_cost_rub * ramp
        realized_labor = monthly_capacity_value * monetization_rate
        incremental = volume * incremental_value_per_request_rub * ramp
        opex = monthly_fixed_opex_rub + volume * ramp * variable_agent_cost_per_request_rub
        net = realized_labor + incremental - opex
        cashflows.append(net)
        monthly_benefits.append(realized_labor + incremental)
        monthly_opex_values.append(opex)
        monthly_net_values.append(net)

    npv = _npv_monthly(monthly_rate, cashflows)
    irr_monthly = _irr_bisection(cashflows)
    irr_annual = ((1 + irr_monthly) ** 12 - 1) if irr_monthly is not None else None
    total_net_after_capex = sum(cashflows)
    roi = (total_net_after_capex / implementation_cost_rub * 100) if implementation_cost_rub > 0 else None

    warnings = []
    assumptions = [
        "Все денежные суммы указаны в RUB.",
        "Первоначальные инвестиции учитываются в момент t=0.",
        "Годовая ставка преобразована в эффективную месячную ставку.",
        "Высвобождённая мощность монетизируется только через monetization_rate.",
        "Модель не включает налоги, инфляцию и оборотный капитал, если они не заданы отдельно.",
    ]
    if monetization_rate < 1:
        warnings.append("Часть высвобождённой мощности не монетизируется; capacity value не равна cashable savings.")
    if effective_minutes < 0:
        warnings.append("После учёта дополнительной ручной работы агент увеличивает трудозатраты на оценку.")
    if monthly_volume == 0:
        warnings.append("Объём равен нулю: операционная выгода отсутствует.")
    if irr_monthly is None:
        warnings.append("IRR не рассчитан: нет единственного корректного корня для заданного потока.")
    if _payback_months(cashflows) is None:
        warnings.append("В пределах заданного горизонта проект не окупается по простому денежному потоку.")

    return asdict(FinancialCaseResult(
        currency="RUB",
        minutes_saved_per_request=baseline_minutes - agent_minutes,
        effective_minutes_saved_per_request=effective_minutes,
        monthly_capacity_hours=capacity_hours,
        monthly_capacity_value_rub=capacity_value,
        monetization_rate=monetization_rate,
        realized_labor_benefit_rub=(capacity_value * monetization_rate),
        incremental_value_rub=monthly_volume * incremental_value_per_request_rub,
        monthly_opex_rub=(monthly_fixed_opex_rub + monthly_volume * variable_agent_cost_per_request_rub),
        monthly_net_benefit_rub=(capacity_value * monetization_rate + monthly_volume * incremental_value_per_request_rub - monthly_fixed_opex_rub - monthly_volume * variable_agent_cost_per_request_rub),
        implementation_cost_rub=implementation_cost_rub,
        simple_payback_months=_payback_months(cashflows),
        discounted_payback_months=_payback_months(cashflows, discounted=True, monthly_rate=monthly_rate),
        npv_rub=npv,
        roi_pct=roi,
        monthly_irr_pct=(irr_monthly * 100 if irr_monthly is not None else None),
        annualized_irr_pct=(irr_annual * 100 if irr_annual is not None else None),
        forecast_months=forecast_months,
        assumptions=assumptions,
        warnings=warnings,
        monthly_cashflows=cashflows,
    ))
