"""Build/replace the FinRAG ChromaDB index from canonical corpus v3.

The replacement index is built as a staging collection first. The active collection
is switched only after all batches have been embedded and written successfully.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
from typing import Any

import chromadb
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data_parser import (
    CORPUS_COLUMNS,
    DEFAULT_CORPUS_PATH,
    corpus_fingerprint,
    normalize_iso_date,
    normalize_region,
)

CORPUS_PATH = DEFAULT_CORPUS_PATH
CHROMA_PATH = Path("./data/chroma_db")
COLLECTION_NAME = "finrag"
STAGING_COLLECTION_NAME = "finrag_staging"
BACKUP_COLLECTION_NAME = "finrag_backup"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
SCHEMA_VERSION = 3
BATCH_SIZE = 256
ENCODE_BATCH_SIZE = 32


def choose_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _clean_text(value: Any) -> str:
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


def _metadata_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _to_chroma_metadata(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert canonical corpus data to scalar Chroma metadata."""
    pub_date = normalize_iso_date(row.get("published_date"))
    period_end = normalize_iso_date(row.get("period_end_date"))
    legacy_date = normalize_iso_date(row.get("date"))
    topic = _clean_text(row.get("topic"))
    url = _clean_text(row.get("url"))
    source_extra = _metadata_json(row.get("metadata"))

    metadata: dict[str, str | int | float | bool] = {
        "source_type": _clean_text(row.get("source_type")),
        "source_name": _clean_text(row.get("source_name")),
        "date": legacy_date or pub_date or period_end or "",
        "published_date": pub_date or "",
        "period_end_date": period_end or "",
        # 0 means unknown; `date_ordinal` is based only on actual publication date.
        "date_ordinal": date.fromisoformat(pub_date).toordinal() if pub_date else 0,
        "reporting_year": _as_int(row.get("reporting_year")),
        "title": _clean_text(row.get("title"))[:200],
        "topic": topic,
        "url": url,
    }

    region_code = _clean_text(source_extra.get("region"))
    region_name = _clean_text(source_extra.get("region_name")) or normalize_region(region_code)
    okved_section = _clean_text(source_extra.get("okved_section")).upper()
    inn = _clean_text(source_extra.get("inn"))
    if region_name:
        metadata["region"] = region_name
    if region_code:
        metadata["region_code"] = region_code.casefold()
    if okved_section:
        metadata["okved_section"] = okved_section
    if inn:
        metadata["inn"] = inn

    for key in ("revenue", "rank_by_revenue"):
        value = source_extra.get(key)
        if value is None:
            continue
        try:
            number = float(value) if key == "revenue" else int(value)
            if math.isfinite(float(number)):
                metadata[key] = number
        except (TypeError, ValueError, OverflowError):
            pass
    return metadata


def _collection_names(client) -> set[str]:
    return {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }


def _recover_collection_state(client) -> set[str]:
    """Restore a backup if a prior process stopped between the two rename steps."""
    names = _collection_names(client)
    if BACKUP_COLLECTION_NAME in names and COLLECTION_NAME not in names:
        client.get_collection(BACKUP_COLLECTION_NAME).modify(name=COLLECTION_NAME)
        names = _collection_names(client)
    elif BACKUP_COLLECTION_NAME in names and COLLECTION_NAME in names:
        # Active collection exists, so the backup is left over from a completed swap.
        client.delete_collection(BACKUP_COLLECTION_NAME)
        names = _collection_names(client)
    return names


def _activate_staging_collection(client, staging_collection) -> None:
    """Switch staging into place, restoring the existing index on switch failure."""
    names = _collection_names(client)
    had_active = COLLECTION_NAME in names
    if had_active:
        if BACKUP_COLLECTION_NAME in names:
            client.delete_collection(BACKUP_COLLECTION_NAME)
        client.get_collection(COLLECTION_NAME).modify(name=BACKUP_COLLECTION_NAME)
    try:
        staging_collection.modify(name=COLLECTION_NAME)
    except Exception:
        names = _collection_names(client)
        if COLLECTION_NAME not in names and BACKUP_COLLECTION_NAME in names:
            client.get_collection(BACKUP_COLLECTION_NAME).modify(name=COLLECTION_NAME)
        raise
    if had_active:
        client.delete_collection(BACKUP_COLLECTION_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить индекс FinRAG в ChromaDB")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="заменить существующий индекс после успешного построения staging-коллекции",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Корпус не найден: {CORPUS_PATH.resolve()}. Сначала выполните data_parser.ipynb."
        )

    df = pl.read_parquet(CORPUS_PATH)
    missing = set(CORPUS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Корпус не соответствует схеме v3; отсутствуют колонки: {sorted(missing)}. "
            "Пересоберите корпус через data_parser.ipynb."
        )
    if df.height == 0:
        raise ValueError("Корпус пуст")

    fingerprint = corpus_fingerprint(df.iter_rows(named=True))
    ids_seen: set[str] = set()
    records: list[tuple[str, str, dict[str, str | int | float | bool]]] = []
    for row in df.iter_rows(named=True):
        doc_id = _clean_text(row.get("doc_id"))
        text = _clean_text(row.get("text"))
        if not doc_id or not text:
            raise ValueError("В корпусе есть пустой doc_id или text; сначала проверьте data_parser")
        if doc_id in ids_seen:
            raise ValueError(f"В корпусе повторяется doc_id: {doc_id!r}")
        ids_seen.add(doc_id)
        records.append((doc_id, text, _to_chroma_metadata(row)))

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    names = _recover_collection_state(client)
    if COLLECTION_NAME in names and not args.rebuild:
        raise RuntimeError(
            f"Коллекция {COLLECTION_NAME!r} уже существует. Для замены используйте "
            "`python index_data_refactored.py --rebuild`. Существующий индекс не затронут."
        )
    if STAGING_COLLECTION_NAME in names:
        # Staging is never used by retrieval. It can be removed before a fresh attempt.
        client.delete_collection(STAGING_COLLECTION_NAME)

    device = choose_device()
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Устройство: {device}")
    print(f"Корпус: {CORPUS_PATH} ({len(records):,} документов)")
    print(f"Fingerprint: {fingerprint[:16]}…")

    staging = client.create_collection(
        name=STAGING_COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "schema_version": SCHEMA_VERSION,
            "corpus_fingerprint": fingerprint,
            "embedding_model": EMBEDDING_MODEL,
        },
    )
    try:
        for start in tqdm(range(0, len(records), BATCH_SIZE), desc="Индексация"):
            batch = records[start:start + BATCH_SIZE]
            texts = [record[1] for record in batch]
            embeddings = model.encode(
                [f"passage: {text}" for text in texts],
                batch_size=ENCODE_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            staging.add(
                ids=[record[0] for record in batch],
                documents=texts,
                metadatas=[record[2] for record in batch],
                embeddings=embeddings.tolist(),
            )
        if staging.count() != len(records):
            raise RuntimeError(
                f"В staging-коллекции {staging.count()} документов при ожидании {len(records)}"
            )
        _activate_staging_collection(client, staging)
    except Exception:
        # Preserve active collection; remove incomplete staging collection when possible.
        try:
            names = _collection_names(client)
            if STAGING_COLLECTION_NAME in names:
                client.delete_collection(STAGING_COLLECTION_NAME)
        except Exception:
            pass
        raise

    active = client.get_collection(COLLECTION_NAME)
    print(
        f"Готово: {active.count():,} документов в {COLLECTION_NAME!r}; "
        f"schema_version={SCHEMA_VERSION}"
    )


if __name__ == "__main__":
    main()
