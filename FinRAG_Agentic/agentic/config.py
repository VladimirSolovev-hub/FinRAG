from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repository root, not from the current shell directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / ".env.agentic", override=False)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://finrag:finrag_dev_password@localhost:54329/finrag_agentic",
    )
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "90"))
    max_tool_rounds: int = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "8"))
    max_evidence_search_rounds: int = int(os.getenv("AGENT_MAX_EVIDENCE_SEARCH_ROUNDS", "2"))
    trace_path: Path = Path(os.getenv("AGENT_TRACE_PATH", "agentic/logs/agent_runs.jsonl"))
    enable_rag: bool = _bool("AGENT_ENABLE_RAG", True)
    enable_database: bool = _bool("AGENT_ENABLE_DATABASE", True)
    corpus_path: Path = Path(os.getenv("FINRAG_CORPUS_PATH", "data/processed/unified_corpus_v3.parquet"))
    chroma_path: Path = Path(os.getenv("FINRAG_CHROMA_PATH", "data/chroma_db"))
    collection_name: str = os.getenv("FINRAG_COLLECTION", "finrag")
    embedding_model: str = os.getenv("FINRAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    reranker_model: str | None = os.getenv("FINRAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3") or None
    device: str | None = os.getenv("FINRAG_DEVICE") or None


settings = Settings()
