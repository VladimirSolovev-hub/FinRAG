from __future__ import annotations
import json
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agentic.config import settings
from agentic.evidence import EVIDENCE_REVIEW_SYSTEM, EvidenceAssessment
from agentic.prompts import SYSTEM_PROMPT
from agentic.state import AgentState
from agentic.tools.registry import build_tools


def _latest_retrieval_payload(messages: list[Any]) -> tuple[bool, str, list[dict]]:
    """Return whether the latest tool batch called RAG and its source snippets."""
    latest_ai_index = None
    latest_calls: list[dict] = []
    for index in range(len(messages) - 1, -1, -1):
        calls = getattr(messages[index], "tool_calls", None) or []
        if calls:
            latest_ai_index = index
            latest_calls = calls
            break
    if latest_ai_index is None:
        return False, "", []
    if not any(call.get("name") == "search_financial_knowledge" for call in latest_calls):
        return False, "", []

    question = next(
        (str(message.content) for message in messages if getattr(message, "type", "") == "human"),
        "",
    )
    sources: list[dict] = []
    for message in messages[latest_ai_index + 1:]:
        if getattr(message, "type", "") != "tool" or getattr(message, "name", "") != "search_financial_knowledge":
            continue
        try:
            payload = json.loads(message.content) if isinstance(message.content, str) else {}
        except json.JSONDecodeError:
            payload = {}
        for source in payload.get("sources", []) if isinstance(payload, dict) else []:
            sources.append({
                "doc_id": source.get("doc_id"),
                "title": source.get("title"),
                "source_name": source.get("source_name"),
                "source_type": source.get("source_type"),
                "text": str(source.get("text", ""))[:1800],
            })
    return True, question, sources[:8]


@lru_cache(maxsize=1)
def build_graph():
    if not settings.llm_api_key:
        raise RuntimeError("Не задан LLM_API_KEY (или GROQ_API_KEY) в .env.agentic")
    tools = build_tools()
    base_llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=2,
    )
    agent_llm = base_llm.bind_tools(tools)
    tools_without_rag = [tool for tool in tools if getattr(tool, "name", "") != "search_financial_knowledge"]
    agent_llm_without_rag = base_llm.bind_tools(tools_without_rag)
    evidence_llm = base_llm.with_structured_output(EvidenceAssessment, method="function_calling")

    def agent_node(state: AgentState) -> dict[str, Any]:
        review_limit = max(1, settings.max_evidence_search_rounds)
        # Enforce the retrieval retry limit at tool-binding time: after the limit,
        # the model no longer receives the RAG tool in its available tool schema.
        active_llm = (
            agent_llm_without_rag
            if settings.enable_rag and state.get("evidence_review_count", 0) >= review_limit
            else agent_llm
        )
        # Keep system instructions at the front of the provider message list.
        current_messages = state["messages"]
        system_messages = [m for m in current_messages if getattr(m, "type", "") == "system"]
        conversation_messages = [m for m in current_messages if getattr(m, "type", "") != "system"]
        response = active_llm.invoke(system_messages + conversation_messages)
        return {"messages": [response]}

    def evidence_review_node(state: AgentState) -> dict[str, Any]:
        was_rag_called, question, sources = _latest_retrieval_payload(state["messages"])
        if not was_rag_called:
            return {}

        attempt = int(state.get("evidence_review_count", 0)) + 1
        if not sources:
            assessment = {
                "document_evidence_sufficient": False,
                "reason": "RAG tool returned no source excerpts.",
                "covered_aspects": [],
                "missing_aspects": ["Документальные подтверждения по запросу не найдены."],
                "suggested_queries": [question],
                "source_ids_used": [],
                "review_status": "no_sources",
            }
        else:
            try:
                reviewed = evidence_llm.invoke([
                    SystemMessage(content=EVIDENCE_REVIEW_SYSTEM),
                    HumanMessage(content=json.dumps({
                        "user_question": question,
                        "retrieved_sources": sources,
                    }, ensure_ascii=False)),
                ])
                assessment = reviewed.model_dump()
                assessment["review_status"] = "completed"
            except Exception as exc:
                assessment = {
                    "document_evidence_sufficient": False,
                    "reason": f"Evidence review failed: {type(exc).__name__}",
                    "covered_aspects": [],
                    "missing_aspects": ["Автоматическая проверка достаточности источников не выполнена."],
                    "suggested_queries": [],
                    "source_ids_used": [],
                    "review_status": "review_error",
                }

        assessment["attempt"] = attempt
        assessment["source_count"] = len(sources)
        limit = max(1, settings.max_evidence_search_rounds)
        if assessment["document_evidence_sufficient"]:
            instruction = (
                "EVIDENCE REVIEW: найденные документы признаны достаточными для "
                "документально-обоснованной части ответа. Используй только подтверждённые "
                "фрагментами утверждения; для числовых показателей всё равно используй "
                "SQL/финансовые tools."
            )
        elif assessment.get("review_status") == "review_error":
            instruction = (
                "EVIDENCE REVIEW: автоматическая проверка источников завершилась ошибкой. "
                "Не считай достаточность доказательств подтверждённой. Можешь использовать "
                "найденные фрагменты с явной оговоркой, но не делай неподтверждённых выводов."
            )
        elif attempt < limit:
            suggestions = assessment.get("suggested_queries") or [question]
            instruction = (
                "EVIDENCE REVIEW: найденных документов недостаточно. Если документальные "
                "сведения нужны для ответа, выполни ещё один поиск с переформулированным "
                "запросом из списка: "
                + json.dumps(suggestions, ensure_ascii=False)
                + ". Не повторяй идентичный запрос без изменения. Не делай неподтверждённых утверждений."
            )
        else:
            instruction = (
                "EVIDENCE REVIEW: достигнут лимит попыток поиска. Не запускай дополнительные "
                "RAG-поиски. Ответь только на части вопроса, которые подтверждаются имеющимися "
                "источниками/данными; явно перечисли, какой документальной информации не хватает."
            )

        return {
            "messages": [SystemMessage(content=instruction)],
            "evidence_reviews": [assessment],
            "evidence_review_count": attempt,
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    workflow.add_node("evidence_review", evidence_review_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "evidence_review")
    workflow.add_edge("evidence_review", "agent")
    return workflow.compile()


def invoke_agent(question: str, thread_id: str = "local-demo") -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Вопрос не должен быть пустым")
    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
            "evidence_reviews": [],
            "evidence_review_count": 0,
        },
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(10, settings.max_tool_rounds * 2 + 4),
        },
    )
    messages = result["messages"]
    final_text = ""
    tool_events = []
    for message in messages:
        if getattr(message, "type", "") == "ai" and isinstance(getattr(message, "content", None), str) and message.content.strip():
            final_text = message.content
        for call in getattr(message, "tool_calls", []) or []:
            tool_events.append({"tool": call.get("name"), "args": call.get("args")})
        if getattr(message, "type", "") == "tool":
            tool_events.append({"tool_result": getattr(message, "name", None), "content_preview": str(message.content)[:500]})
    return {
        "answer": final_text,
        "tool_events": tool_events,
        "evidence_reviews": result.get("evidence_reviews", []),
        "messages": messages,
    }
