from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

from agentic.config import REPO_ROOT, settings


def main() -> int:
    print("FinRAG Agentic environment check")
    print(f"Repository root: {REPO_ROOT}")
    print(f"LLM key configured: {'yes' if settings.llm_api_key else 'no'}")
    print(f"Database URL configured: {'yes' if settings.database_url else 'no'}")
    print(f"RAG enabled: {settings.enable_rag}")
    print(f"Database tools enabled: {settings.enable_database}")
    required = ["langgraph", "langchain_openai", "psycopg", "dotenv"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages:", ", ".join(missing))
        return 1
    if not (REPO_ROOT / "retrieval.py").exists():
        print("WARNING: root retrieval.py not found. Add this folder inside the FinRAG repo root.")
    else:
        print("Root retrieval.py: found")
    if not settings.corpus_path.is_absolute():
        corpus_path = REPO_ROOT / settings.corpus_path
    else:
        corpus_path = settings.corpus_path
    print(f"Corpus exists: {corpus_path.exists()} ({corpus_path})")
    print("Healthcheck completed. Database connectivity and LLM calls are tested at runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
