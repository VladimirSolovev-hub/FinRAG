from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        decoded = json.loads(str(value))
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, TypeError):
        return {}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _stable_hash(text: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()


def _split_for_doc(doc_id: str) -> str:
    """Stable grouped split; all questions for a document stay in one split."""
    bucket = int(_stable_hash(doc_id, "finrag-eval-split")[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _ru_number(value: Any) -> str:
    # Mirrors data_parser.fmt_number without depending on the whole parser at import time.
    number = float(value)
    absolute = abs(number)
    if absolute >= 1e9:
        return f"{number / 1e9:.2f} млрд"
    if absolute >= 1e6:
        return f"{number / 1e6:.1f} млн"
    if absolute >= 1e3:
        return f"{number / 1e3:.0f} тыс"
    return f"{number:.0f}"


def _make_auto_qa(row: dict[str, Any], sections: dict[str, str], normalize_region) -> list[dict[str, Any]]:
    meta = _read_meta(row.get("metadata"))
    doc_id = _clean(row.get("doc_id"))
    inn = _clean(meta.get("inn"))
    year_value = row.get("reporting_year") or meta.get("reporting_year") or meta.get("year")
    try:
        year = int(year_value)
    except (TypeError, ValueError):
        return []
    if not doc_id or not inn or not 1900 <= year <= 2200:
        return []

    base = {
        "gold_doc_ids": [doc_id],
        "relevance": {doc_id: 3},
        "filters": {"source_type": "structured", "reporting_year": year},
        "answerable": True,
        "split": _split_for_doc(doc_id),
        "source_type": "structured",
        "review_status": "auto_generated",
        "generation_method": "rfsd_metadata_lookup_template",
    }
    result: list[dict[str, Any]] = []

    revenue = meta.get("revenue")
    if revenue is not None:
        try:
            revenue_num = float(revenue)
            if revenue_num > 0:
                answer_amount = _ru_number(revenue_num)
                result.append({
                    **base,
                    "qid": f"{doc_id}::revenue",
                    "category": "rfsd_lookup_revenue",
                    "query": f"Какая выручка указана в финансовой отчётности организации с ИНН {inn} за {year} год?",
                    "reference_answer": (
                        f"В карточке RFSD организации с ИНН {inn} за {year} год "
                        f"указана выручка {answer_amount} руб."
                    ),
                    "required_facts": [inn, str(year), answer_amount],
                })
        except (TypeError, ValueError, OverflowError):
            pass

    region_raw = _clean(meta.get("region_name") or meta.get("region"))
    region = normalize_region(region_raw) if region_raw else ""
    if region:
        result.append({
            **base,
            "qid": f"{doc_id}::region",
            "category": "rfsd_lookup_region",
            "query": f"В каком регионе зарегистрирована организация с ИНН {inn} по отчётности за {year} год?",
            "reference_answer": f"В карточке RFSD организации с ИНН {inn} за {year} год указан регион: {region}.",
            "required_facts": [inn, str(year), region],
        })

    section_code = _clean(meta.get("okved_section")).upper()
    section_name = sections.get(section_code)
    if section_code and section_name:
        result.append({
            **base,
            "qid": f"{doc_id}::okved_section",
            "category": "rfsd_lookup_activity",
            "query": f"К какому виду деятельности относится организация с ИНН {inn} за {year} год?",
            "reference_answer": (
                f"В карточке RFSD указан раздел ОКВЭД {section_code}: {section_name}."
            ),
            "required_facts": [inn, str(year), section_code, section_name],
        })
    return result


def _make_annotation_queue(rows: list[dict[str, Any]], limit_per_source: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_type = _clean(row.get("source_type")) or "unknown"
        if source_type == "structured":
            continue
        if not _clean(row.get("doc_id")) or not _clean(row.get("text")):
            continue
        by_source.setdefault(source_type, []).append(row)

    for source_type, group in sorted(by_source.items()):
        group.sort(key=lambda r: _stable_hash(_clean(r.get("doc_id")), "queue-select"))
        for row in group[:limit_per_source]:
            doc_id = _clean(row.get("doc_id"))
            title = _clean(row.get("title")) or _clean(row.get("text"))[:120]
            body = " ".join(_clean(row.get("text")).split())
            selected.append({
                "qid": f"manual_{hashlib.sha1(doc_id.encode('utf-8')).hexdigest()[:12]}",
                "split": _split_for_doc(doc_id),
                "source_type": source_type,
                "source_name": _clean(row.get("source_name")),
                "published_date": _clean(row.get("published_date")),
                "title": title,
                "url": _clean(row.get("url")),
                "candidate_doc_id": doc_id,
                "relevant_doc_ids": f"{doc_id}:3",
                "filters_json": json.dumps({"source_type": source_type}, ensure_ascii=False),
                "query": "",
                "question_draft": f"Найди источник, в котором рассматривается тема: {title}",
                "reference_answer": "",
                "answerable": "true",
                "review_status": "needs_review",
                "text_excerpt": body[:1600],
                # The text used for question drafting is retained separately from the short preview.
                # This lets an LLM draft from a larger source while keeping the CSV reviewable.
                "text_for_annotation": body,
                "review_instructions": (
                    "Проверьте предложенные query, reference_answer и evidence_quote по тексту; "
                    "проверьте candidate_doc_id и relevant_doc_ids, поправьте при необходимости, "
                    "затем поставьте review_status=approved. Не подтверждайте неподкреплённые факты."
                ),
            })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/unified_corpus_v3.parquet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evaluation/data")
    parser.add_argument("--max-rfsd-docs-per-year", type=int, default=200,
                        help="Максимум документов RFSD на год для стартового lookup-бенчмарка")
    parser.add_argument("--annotation-docs-per-source", type=int, default=25,
                        help="Сколько текстовых документов каждого типа добавить в очередь ручной разметки")
    args = parser.parse_args()

    if args.max_rfsd_docs_per_year <= 0 or args.annotation_docs_per_source < 0:
        parser.error("Лимиты должны быть положительными (annotation может быть 0)")
    if not args.corpus.exists():
        raise FileNotFoundError(
            f"Корпус не найден: {args.corpus.resolve()}. Сначала постройте unified_corpus_v3.parquet."
        )

    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Нужен polars: установите зависимости из requirements.txt") from exc

    from data_parser import OKVED_SECTIONS, corpus_fingerprint, normalize_region

    frame = pl.read_parquet(args.corpus)
    expected = {"doc_id", "text", "source_type", "reporting_year", "metadata"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"В корпусе отсутствуют необходимые колонки: {sorted(missing)}")
    if frame.height == 0:
        raise ValueError("Корпус пуст")

    rows = frame.to_dicts()
    structured: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if _clean(row.get("source_type")) != "structured":
            continue
        meta = _read_meta(row.get("metadata"))
        year = row.get("reporting_year") or meta.get("reporting_year") or meta.get("year")
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            continue
        structured.setdefault(year_int, []).append(row)

    chosen_rows: list[dict[str, Any]] = []
    for year, group in sorted(structured.items()):
        group.sort(key=lambda r: _stable_hash(_clean(r.get("doc_id")), "rfsd-select"))
        chosen_rows.extend(group[:args.max_rfsd_docs_per_year])

    benchmark: list[dict[str, Any]] = []
    for row in chosen_rows:
        benchmark.extend(_make_auto_qa(row, OKVED_SECTIONS, normalize_region))
    benchmark.sort(key=lambda item: (item["split"], item["category"], item["qid"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = args.out_dir / "rfsd_lookup_benchmark.jsonl"
    with benchmark_path.open("w", encoding="utf-8") as stream:
        for item in benchmark:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    queue = _make_annotation_queue(rows, args.annotation_docs_per_source)
    queue_path = args.out_dir / "annotation_queue.csv"
    queue_fields = list(queue[0].keys()) if queue else [
        "qid", "split", "source_type", "source_name", "published_date", "title", "url",
        "candidate_doc_id", "relevant_doc_ids", "filters_json", "query", "question_draft",
        "reference_answer", "answerable", "review_status", "text_excerpt", "text_for_annotation", "review_instructions",
    ]
    with queue_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=queue_fields)
        writer.writeheader()
        writer.writerows(queue)

    try:
        fingerprint = corpus_fingerprint(frame.iter_rows(named=True))
    except Exception:
        fingerprint = None
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(args.corpus.resolve()),
        "corpus_rows": frame.height,
        "corpus_fingerprint": fingerprint,
        "rfsd_docs_sampled": len(chosen_rows),
        "auto_questions": len(benchmark),
        "auto_questions_by_category": dict(Counter(q["category"] for q in benchmark)),
        "auto_questions_by_split": dict(Counter(q["split"] for q in benchmark)),
        "annotation_queue_rows": len(queue),
        "annotation_queue_by_source": dict(Counter(q["source_type"] for q in queue)),
        "warning": (
            "RFSD questions are deterministic exact-lookup templates derived from the corpus. "
            "They are a reproducible smoke/lookup benchmark, not a substitute for human-reviewed "
            "semantic questions. Only reviewed rows from annotation_queue.csv should be added to the QA benchmark."
        ),
    }
    (args.out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Корпус: {frame.height:,} документов; fingerprint={fingerprint[:16] + '…' if fingerprint else 'n/a'}")
    print(f"RFSD-документов отобрано: {len(chosen_rows):,}")
    print(f"Автоматических RFSD-вопросов создано: {len(benchmark):,} -> {benchmark_path}")
    print(f"Вопросов по split: {dict(Counter(q['split'] for q in benchmark))}")
    print(f"Очередь ручной разметки: {len(queue):,} -> {queue_path}")
    if not benchmark:
        print("ПРЕДУПРЕЖДЕНИЕ: не создано ни одного RFSD-вопроса. Проверьте metadata с inn/revenue.")
    print("Сохраните эти файлы как версию baseline; не пересоздавайте их после просмотра результатов без необходимости.")


if __name__ == "__main__":
    main()
