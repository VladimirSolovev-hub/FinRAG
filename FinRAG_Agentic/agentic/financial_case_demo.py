from __future__ import annotations
import json
from agentic.tools.finance import calculate_business_case


def main() -> None:
    result = calculate_business_case(
        monthly_volume=20_000,
        baseline_minutes=30,
        agent_minutes=10,
        hourly_cost_rub=900,
        monetization_rate=0.25,
        monthly_fixed_opex_rub=500_000,
        implementation_cost_rub=8_000_000,
        annual_discount_rate=0.20,
        forecast_months=36,
        ramp_months=3,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
