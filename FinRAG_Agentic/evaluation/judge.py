from __future__ import annotations
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agentic.config import settings


class CriterionResult(BaseModel):
    criterion: str
    rating_0_to_4: int = Field(ge=0, le=4)
    evidence_quote: str = Field(description="Краткая дословная цитата из ответа; пустая строка, если доказательства нет")
    rationale: str
    critical_failure: bool = False


class JudgeResult(BaseModel):
    criteria: list[CriterionResult]
    missing_information: list[str]
    unsupported_claims: list[str]
    needs_human_review: bool = True
    overall_comment: str


CRITERIA = [
    ("financial_interpretation", "Различает ли ответ capacity value и подтверждённую cashable savings?"),
    ("assumptions", "Обозначены ли недостающие входы и сценарные допущения?"),
    ("methodology", "Корректна ли финансовая методология и причинная интерпретация?"),
    ("completeness", "Учтены ли CAPEX, OPEX, качество, ручные проверки и ограничения?"),
    ("traceability", "Можно ли проследить источники чисел и ключевых утверждений?"),
    ("recommendation", "Соответствует ли сила рекомендации уровню доказательств?"),
]


JUDGE_SYSTEM = """Ты — независимый оценщик ответов финансового аналитического AI-агента.
Оценивай только фактическую и методологическую корректность. Не награждай красноречие.
Не считай собственные знания заменой данным кейса. Если входные данные не содержат параметр,
проверь, обозначил ли агент его как неизвестный/допущение. Не выдумывай цитаты: evidence_quote
должен быть дословным фрагментом ответа или пустой строкой. Рейтинг 0..4: 0 отсутствует/опасно;
1 серьёзно неполно; 2 частично; 3 в основном корректно; 4 полно и корректно.
Если агент заявляет подтверждённую экономию без доказательства монетизации, это critical_failure.
Если агент утверждает причинный эффект без надлежащего дизайна/оговорок, это critical_failure.
Если неверный расчёт меняет бизнес-решение, это critical_failure.
Всегда устанавливай needs_human_review=true: результат LLM-as-a-judge — вспомогательный, не окончательный.
"""


def judge_answer(case: dict, answer: str) -> dict:
    if not settings.llm_api_key:
        raise RuntimeError("Для LLM-judge не задан LLM_API_KEY/GROQ_API_KEY")
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        temperature=0,
        timeout=settings.llm_timeout,
        max_retries=2,
    ).with_structured_output(JudgeResult, method="function_calling")
    criteria_text = "\n".join(f"- {name}: {description}" for name, description in CRITERIA)
    payload = {
        "case_id": case.get("id"),
        "user_query": case.get("query"),
        "case_expectations": {
            "must_mention": case.get("must_mention", []),
            "must_not_claim": case.get("must_not_claim", []),
            "critical_rules": case.get("critical_rules", []),
        },
        "agent_answer": answer,
    }
    result = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=(
            f"Оцени ответ по критериям ниже.\n{criteria_text}\n\n"
            f"КЕЙС И ОЖИДАНИЯ:\n{payload}\n\n"
            "Верни по одному CriterionResult для каждого критерия. Не делай выводов о фактах,"
            " которые не представлены в кейсе или ответе."
        )),
    ])
    data = result.model_dump()
    weights = {
        "financial_interpretation": 20,
        "assumptions": 10,
        "methodology": 10,
        "completeness": 15,
        "traceability": 10,
        "recommendation": 5,
    }
    score_by_id = {item.criterion: item.rating_0_to_4 for item in data["criteria"]}
    weighted = sum(weights[key] * score_by_id.get(key, 0) / 4 for key in weights)
    # Numerical correctness (25) and SQL/tool correctness (5) must be assessed
    # deterministically by code-specific tests, not delegated to this language judge.
    critical = [item.criterion for item in data["criteria"] if item.critical_failure]
    data["llm_judge_weighted_score_70"] = round(weighted, 2)
    data["critical_failures"] = critical
    data["verdict"] = "FAIL" if critical else ("REVIEW" if data["needs_human_review"] else "PASS")
    data["score_scope_note"] = "LLM judge covers 70 rubric points only; numeric/tool correctness is assessed separately. Not a ground-truth evaluator."
    return data
