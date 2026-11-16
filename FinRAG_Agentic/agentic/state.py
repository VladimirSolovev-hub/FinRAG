from __future__ import annotations
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    evidence_reviews: Annotated[list[dict], operator.add]
    evidence_review_count: int
