from __future__ import annotations
from agentic.config import settings
from agentic.tools.financial import calculate_business_case, calculate_npv_from_cashflows
from agentic.tools.retrieval import search_financial_knowledge
from agentic.tools.database import (
    get_process_kpis,
    get_did_effect,
    get_quality_metrics,
    get_agent_run_metrics,
    get_complexity_mix,
    get_agent_monthly_costs,
    get_initiative_cashflows,
)


def build_tools():
    tools = [calculate_business_case, calculate_npv_from_cashflows]
    if settings.enable_rag:
        tools.append(search_financial_knowledge)
    if settings.enable_database:
        tools.extend([
            get_process_kpis,
            get_did_effect,
            get_quality_metrics,
            get_agent_run_metrics,
            get_complexity_mix,
            get_agent_monthly_costs,
            get_initiative_cashflows,
        ])
    return tools
