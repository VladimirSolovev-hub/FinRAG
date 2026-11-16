import math
import pytest
from agentic.tools.finance import calculate_business_case, calculate_periodic_cashflows


def base(**overrides):
    args = dict(
        monthly_volume=20_000,
        baseline_minutes=30,
        agent_minutes=10,
        hourly_cost_rub=900,
        monetization_rate=1.0,
        monthly_fixed_opex_rub=500_000,
        implementation_cost_rub=8_000_000,
        annual_discount_rate=0.20,
        forecast_months=36,
    )
    args.update(overrides)
    return calculate_business_case(**args)


def test_full_monetization_baseline_values():
    result = base()
    assert math.isclose(result["monthly_capacity_value_rub"], 6_000_000, rel_tol=1e-9)
    assert math.isclose(result["monthly_net_benefit_rub"], 5_500_000, rel_tol=1e-9)
    assert math.isclose(result["simple_payback_months"], 8_000_000 / 5_500_000, rel_tol=1e-9)
    assert math.isclose(result["npv_rub"], 143_352_695.68404156, rel_tol=1e-8)


def test_zero_monetization_has_no_payback():
    result = base(monetization_rate=0)
    assert result["realized_labor_benefit_rub"] == 0
    assert result["monthly_net_benefit_rub"] == -500_000
    assert result["simple_payback_months"] is None
    assert result["npv_rub"] < 0


def test_validation_rejects_invalid_monetization():
    with pytest.raises(ValueError):
        base(monetization_rate=1.2)


def test_additional_manual_work_reduces_savings():
    result = base(additional_human_minutes_per_request=5)
    assert result["effective_minutes_saved_per_request"] == 15
    assert result["monthly_capacity_value_rub"] == 4_500_000


def test_zero_capex_has_zero_payback():
    result = base(implementation_cost_rub=0)
    assert result["simple_payback_months"] == 0.0


def test_negative_time_savings_are_not_hidden():
    result = base(agent_minutes=40)
    assert result["effective_minutes_saved_per_request"] == -10
    assert result["monthly_net_benefit_rub"] < 0
    assert result["warnings"]


def test_periodic_cashflow_npv_matches_monthly_model():
    flows = [-8_000_000] + [5_500_000] * 36
    result = calculate_periodic_cashflows(flows, annual_discount_rate=0.20, periods_per_year=12)
    assert math.isclose(result["npv_rub"], 143_352_695.68404156, rel_tol=1e-8)
    assert result["simple_payback_periods"] is not None


def test_periodic_cashflow_validation():
    with pytest.raises(ValueError):
        calculate_periodic_cashflows([], annual_discount_rate=0.2)
    with pytest.raises(ValueError):
        calculate_periodic_cashflows([-1, 2], annual_discount_rate=0.2, periods_per_year=0)
