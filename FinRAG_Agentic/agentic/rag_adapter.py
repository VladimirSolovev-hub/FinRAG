from __future__ import annotations
import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentic.config import REPO_ROOT, settings


@lru_cache(maxsize=1)
def get_retriever() -> Any:
    """Load the existing root-level FinRAG HybridRetriever lazily.

    The extension intentionally reuses retrieval.py instead of duplicating the
    user's existing dense + BM25 + RRF + Cross-Encoder implementation.
    """
    import sys
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("retrieval")
    retriever_cls = getattr(module, "HybridRetriever")
    corpus = settings.corpus_path
    chroma = settings.chroma_path
    if not corpus.is_absolute():
        corpus = REPO_ROOT / corpus
    if not chroma.is_absolute():
        chroma = REPO_ROOT / chroma
    kwargs = {
        "corpus_path": corpus,
        "chroma_path": chroma,
        "collection_name": settings.collection_name,
        "embedding_model_name": settings.embedding_model,
        "reranker_model_name": settings.reranker_model,
    }
    if settings.device:
        kwargs["device"] = settings.device
    return retriever_cls(**kwargs)


def search_finrag(query: str, top_k: int = 5, source_type: str | None = None) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Поисковый запрос не должен быть пустым")
    retriever = get_retriever()
    hits = retriever.search(
        query,
        k=max(1, min(int(top_k), 10)),
        source_type=source_type,
        dense_candidates=100,
        sparse_candidates=100,
        rerank_candidates=50,
    )
    results = []
    for hit in hits:
        metadata = dict(hit.metadata or {})
        results.append({
            "doc_id": hit.doc_id,
            "title": metadata.get("title"),
            "source_name": metadata.get("source_name"),
            "source_type": metadata.get("source_type"),
            "published_date": metadata.get("published_date"),
            "score": float(hit.score),
            "score_type": hit.score_type,
            "text": hit.text,
            "metadata": metadata,
        })
    return results
