"""Hybrid retrieval for the canonical FinRAG corpus v3.

Pipeline: multilingual-E5 dense retrieval + BM25 -> reciprocal-rank fusion -> optional
BGE cross-encoder reranking. Filters are explicit: publication dates are independent
from RFSD reporting years and period-end dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import chromadb
import polars as pl
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from data_parser import (
    CORPUS_COLUMNS,
    DEFAULT_CORPUS_PATH,
    corpus_fingerprint,
    normalize_iso_date,
    normalize_region,
)

DEFAULT_CHROMA_PATH = Path("./data/chroma_db")
DEFAULT_COLLECTION_NAME = "finrag"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
INDEX_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SearchHit:
    """One retrieved document; score_type explains the meaning of `score`."""

    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    score_type: str  # "rrf" or "reranker"


@dataclass(frozen=True)
class _Document:
    doc_id: str
    text: str
    source_type: str
    published_date_ordinal: int
    reporting_year: int
    region: str
    okved_section: str
    metadata: dict[str, Any]


def choose_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def parse_date(value: Any) -> date | None:
    normalized = normalize_iso_date(value)
    if not normalized:
        return None
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError, json.JSONDecodeError):
        return {}


def tokenize(text: str) -> list[str]:
    """Unicode-aware tokenizer preserving Cyrillic text, numbers and identifiers."""
    return re.findall(r"\w+", text.casefold())


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Fuse rank-ordered document IDs; raw scores from different systems are ignored."""
    if rrf_k <= 0:
        raise ValueError("rrf_k должен быть положительным числом")
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    """Loads the corpus, validates its index version, and exposes a single search API."""

    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
        chroma_path: str | Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str | None = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        skip_rerank_sources: Iterable[str] = ("telegram",),
    ) -> None:
        self.device = device or choose_device()
        self.skip_rerank_sources = set(skip_rerank_sources)
        self.corpus_path = Path(corpus_path)
        self.collection_name = collection_name

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Корпус не найден: {self.corpus_path.resolve()}")
        df = pl.read_parquet(self.corpus_path)
        missing = set(CORPUS_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Корпус не соответствует схеме v3; отсутствуют колонки: {sorted(missing)}. "
                "Сначала выполните исправленный data_parser.ipynb."
            )
        if df.height == 0:
            raise ValueError("Корпус пуст")

        actual_fingerprint = corpus_fingerprint(df.iter_rows(named=True))
        self._documents: list[_Document] = []
        self._documents_by_id: dict[str, _Document] = {}
        seen_ids: set[str] = set()

        for row in df.iter_rows(named=True):
            doc_id = _text(row.get("doc_id"))
            body = _text(row.get("text"))
            if not doc_id or not body:
                raise ValueError("В корпусе есть пустой doc_id или text; пересоберите корпус")
            if doc_id in seen_ids:
                raise ValueError(f"В корпусе повторяется doc_id: {doc_id!r}")
            seen_ids.add(doc_id)

            published_date = parse_date(row.get("published_date"))
            source_metadata = _parse_json_dict(row.get("metadata"))
            reporting_year = _as_int(row.get("reporting_year"))
            # Legacy RFSD metadata may also carry year; the canonical field wins.
            if not reporting_year:
                reporting_year = _as_int(source_metadata.get("reporting_year", source_metadata.get("year")))
            if not reporting_year:
                reporting_year = 0

            region_raw = _text(source_metadata.get("region"))
            region_name = _text(source_metadata.get("region_name")) or normalize_region(region_raw)
            section = _text(source_metadata.get("okved_section")).upper()
            title = _text(row.get("title"))
            source_type = _text(row.get("source_type"))
            published_string = published_date.isoformat() if published_date else ""
            period_end = normalize_iso_date(row.get("period_end_date")) or ""
            legacy_date = normalize_iso_date(row.get("date")) or published_string or period_end

            public_metadata: dict[str, Any] = {
                "source_type": source_type,
                "source_name": _text(row.get("source_name")),
                "date": legacy_date,
                "published_date": published_string,
                "period_end_date": period_end,
                "reporting_year": reporting_year or None,
                "title": title,
                "topic": _text(row.get("topic")),
                "url": _text(row.get("url")),
                "source_metadata": source_metadata,
            }
            # Flatten source-specific metadata for convenience while preserving the original.
            for key, value in source_metadata.items():
                if key not in public_metadata:
                    public_metadata[key] = value

            document = _Document(
                doc_id=doc_id,
                text=body,
                source_type=source_type,
                published_date_ordinal=published_date.toordinal() if published_date else 0,
                reporting_year=reporting_year,
                region=region_name,
                okved_section=section,
                metadata=public_metadata,
            )
            self._documents.append(document)
            self._documents_by_id[doc_id] = document

        self.client = chromadb.PersistentClient(path=str(chroma_path))
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Коллекция ChromaDB {collection_name!r} не найдена. "
                "Сначала выполните `python index_data_refactored.py`."
            ) from exc

        self.collection_count = self.collection.count()
        index_metadata = self.collection.metadata or {}
        if _as_int(index_metadata.get("schema_version"), -1) != INDEX_SCHEMA_VERSION:
            raise RuntimeError(
                f"Нужен индекс схемы {INDEX_SCHEMA_VERSION}, обнаружена схема "
                f"{index_metadata.get('schema_version')!r}. Пересоберите корпус и индекс."
            )
        indexed_fingerprint = index_metadata.get("corpus_fingerprint")
        if indexed_fingerprint != actual_fingerprint:
            raise RuntimeError(
                "Корпус Parquet и индекс ChromaDB созданы из разных версий данных. "
                "Перезапустите индексатор с `--rebuild`."
            )
        if self.collection_count != len(self._documents_by_id):
            raise RuntimeError(
                f"Количество записей не совпало: ChromaDB={self.collection_count:,}, "
                f"Parquet={len(self._documents_by_id):,}. Пересоберите индекс."
            )

        # Expensive models are loaded only after corpus/index consistency is verified.
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.reranker = (
            CrossEncoder(reranker_model_name, device=self.device)
            if reranker_model_name
            else None
        )
        self.bm25 = BM25Okapi([tokenize(document.text) or ["__empty__"] for document in self._documents])
        self._bm25_documents = self._documents

        print(
            f"HybridRetriever готов: device={self.device}, ChromaDB={self.collection_count:,}, "
            f"BM25={len(self._bm25_documents):,}, "
            f"reranker={'on' if self.reranker else 'off'}"
        )

    @staticmethod
    def _normalize_date_filters(
        date_from: str | date | None, date_to: str | date | None
    ) -> tuple[date | None, date | None]:
        start = parse_date(date_from)
        end = parse_date(date_to)
        if date_from is not None and start is None:
            raise ValueError(f"Некорректный date_from: {date_from!r}; ожидается YYYY-MM-DD")
        if date_to is not None and end is None:
            raise ValueError(f"Некорректный date_to: {date_to!r}; ожидается YYYY-MM-DD")
        if start and end and start > end:
            raise ValueError("date_from не может быть позже date_to")
        return start, end

    def _build_chroma_where(
        self,
        source_type: str | None,
        date_from: date | None,
        date_to: date | None,
        reporting_year: int | None,
        region: str | None,
        okved_section: str | None,
    ) -> dict[str, Any] | None:
        conditions: list[dict[str, Any]] = []
        if source_type:
            conditions.append({"source_type": {"$eq": source_type}})
        if date_from or date_to:
            start_ordinal = date_from.toordinal() if date_from else 1
            conditions.append({"date_ordinal": {"$gte": start_ordinal}})
            if date_to:
                conditions.append({"date_ordinal": {"$lte": date_to.toordinal()}})
        if reporting_year is not None:
            conditions.append({"reporting_year": {"$eq": int(reporting_year)}})
        if region:
            normalized_region = normalize_region(region)
            conditions.append({"region": {"$eq": normalized_region}})
        if okved_section:
            conditions.append({"okved_section": {"$eq": okved_section.upper()}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _matches_python_filters(
        self,
        document: _Document,
        source_type: str | None,
        start: date | None,
        end: date | None,
        reporting_year: int | None,
        region: str | None,
        okved_section: str | None,
    ) -> bool:
        if source_type and document.source_type != source_type:
            return False
        if start or end:
            ordinal = document.published_date_ordinal
            if ordinal == 0:
                return False
            if start and ordinal < start.toordinal():
                return False
            if end and ordinal > end.toordinal():
                return False
        if reporting_year is not None and document.reporting_year != int(reporting_year):
            return False
        if region and document.region.casefold() != normalize_region(region).casefold():
            return False
        if okved_section and document.okved_section != okved_section.upper():
            return False
        return True

    def _dense_search(
        self,
        query: str,
        limit: int,
        source_type: str | None,
        start: date | None,
        end: date | None,
        reporting_year: int | None,
        region: str | None,
        okved_section: str | None,
    ) -> list[str]:
        if limit <= 0 or self.collection_count <= 0:
            return []
        query_embedding = self.embedding_model.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(limit, self.collection_count),
            "include": ["metadatas", "documents", "distances"],
        }
        where = self._build_chroma_where(
            source_type, start, end, reporting_year, region, okved_section
        )
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        return list(result["ids"][0]) if result.get("ids") else []

    def _bm25_search(
        self,
        query: str,
        limit: int,
        source_type: str | None,
        start: date | None,
        end: date | None,
        reporting_year: int | None,
        region: str | None,
        okved_section: str | None,
    ) -> list[str]:
        if limit <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        filtered: list[tuple[str, float]] = []
        for index, document in enumerate(self._bm25_documents):
            if not self._matches_python_filters(
                document, source_type, start, end, reporting_year, region, okved_section
            ):
                continue
            score = float(scores[index])
            if score > 0:
                filtered.append((document.doc_id, score))
        filtered.sort(key=lambda item: (-item[1], item[0]))
        return [doc_id for doc_id, _ in filtered[:limit]]

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        source_type: str | None = None,
        date_from: str | date | None = None,
        date_to: str | date | None = None,
        reporting_year: int | None = None,
        region: str | None = None,
        okved_section: str | None = None,
        use_reranker: bool | None = None,
        dense_candidates: int = 100,
        sparse_candidates: int = 100,
        rerank_candidates: int = 50,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        """Run hybrid search.

        date_from/date_to filter publication dates, and never infer automatically from
        the query. reporting_year filters RFSD fiscal year. `region` accepts a known
        region code (e.g. moscow) or display name (e.g. Москва); OKVED uses section code.
        """
        query = query.strip()
        if not query:
            raise ValueError("query не должен быть пустым")
        if k <= 0:
            raise ValueError("k должен быть положительным числом")
        if min(dense_candidates, sparse_candidates, rerank_candidates) <= 0:
            raise ValueError("Параметры числа кандидатов должны быть положительными")
        if reporting_year is not None and not (1900 <= int(reporting_year) <= 2200):
            raise ValueError("reporting_year выглядит некорректным")

        start, end = self._normalize_date_filters(date_from, date_to)
        if use_reranker is None:
            use_reranker = source_type not in self.skip_rerank_sources
        if use_reranker and self.reranker is None:
            raise RuntimeError(
                "Реранкер отключён при инициализации. Укажите reranker_model_name "
                "или вызовите search(..., use_reranker=False)."
            )

        dense_ids = self._dense_search(
            query, dense_candidates, source_type, start, end,
            reporting_year, region, okved_section,
        )
        sparse_ids = self._bm25_search(
            query, sparse_candidates, source_type, start, end,
            reporting_year, region, okved_section,
        )
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], rrf_k=rrf_k)
        if not fused:
            return []

        if not use_reranker:
            return [
                SearchHit(
                    doc_id=doc_id,
                    text=self._documents_by_id[doc_id].text,
                    metadata=dict(self._documents_by_id[doc_id].metadata),
                    score=float(score),
                    score_type="rrf",
                )
                for doc_id, score in fused[:k]
                if doc_id in self._documents_by_id
            ]

        candidates: list[tuple[str, _Document, float]] = []
        for doc_id, rrf_score in fused:
            document = self._documents_by_id.get(doc_id)
            if document and document.text:
                candidates.append((doc_id, document, rrf_score))
            if len(candidates) >= rerank_candidates:
                break
        if not candidates:
            return []

        raw_scores = self.reranker.predict(
            [[query, document.text] for _, document, _ in candidates],
            batch_size=32,
            show_progress_bar=False,
        )
        order = sorted(
            range(len(candidates)),
            key=lambda idx: (-float(raw_scores[idx]), candidates[idx][0]),
        )[:k]
        return [
            SearchHit(
                doc_id=candidates[idx][0],
                text=candidates[idx][1].text,
                metadata=dict(candidates[idx][1].metadata),
                score=float(raw_scores[idx]),
                score_type="reranker",
            )
            for idx in order
        ]


def print_results(
    query: str,
    hits: Sequence[SearchHit],
    preview_chars: int = 300,
) -> None:
    """Print search results with explicit publication/reporting-date distinction."""
    print(f"\n{'=' * 92}\n🔎 {query}\n{'=' * 92}")
    if not hits:
        print("Ничего не найдено.")
        return
    for rank, hit in enumerate(hits, start=1):
        meta = hit.metadata
        source_type = meta.get("source_type", "")
        source_name = meta.get("source_name", "")
        published_date = meta.get("published_date", "")
        period_end = meta.get("period_end_date", "")
        if published_date:
            date_label = f"публикация: {published_date}"
        elif period_end:
            date_label = f"конец периода: {period_end}"
        else:
            date_label = "дата не указана"
        reporting_year = meta.get("reporting_year")
        year_label = f" | отчётный год: {reporting_year}" if reporting_year else ""
        print(
            f"\n{rank}. [{hit.doc_id}] {source_type} | {source_name} | {date_label}"
            f"{year_label} | {hit.score_type}={hit.score:.4f}"
        )
        if meta.get("title"):
            print(f"   Заголовок: {meta['title']}")
        if meta.get("url"):
            print(f"   URL: {meta['url']}")
        preview = hit.text.replace("\n", " ").strip()
        if len(preview) > preview_chars:
            preview = preview[:preview_chars].rstrip() + "…"
        print(f"   {preview}")
