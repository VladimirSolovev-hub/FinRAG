"""Compile rows from annotation_queue.csv into an evaluation JSONL."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "evaluation/data/annotation_queue.csv"
DEFAULT_OUTPUT = ROOT / "evaluation/data/reviewed_benchmark.jsonl"


def _bool(value: str) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "да"}


def _qrels(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for part in (value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            doc_id, grade_text = part.rsplit(":", 1)
            grade = int(grade_text)
        else:
            doc_id, grade = part, 3
        doc_id = doc_id.strip()
        if doc_id:
            if grade < 0 or grade > 3:
                raise ValueError(f"Оценка релевантности должна быть 0..3: {part!r}")
            if grade > 0:
                result[doc_id] = grade
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-splits", nargs="+", choices=["train", "validation", "test"],
                        default=["train", "validation", "test"])
    args = parser.parse_args()
    if not args.queue.exists():
        raise FileNotFoundError(args.queue)

    compiled = []
    errors = []
    with args.queue.open("r", newline="", encoding="utf-8-sig") as stream:
        for row_no, row in enumerate(csv.DictReader(stream), start=2):
            if row.get("review_status", "").strip().casefold() != "approved":
                continue
            try:
                query = (row.get("query") or "").strip()
                answer = (row.get("reference_answer") or "").strip()
                if not query or not answer:
                    raise ValueError("для approved обязательны query и reference_answer")
                qrels = _qrels(row.get("relevant_doc_ids", ""))
                if _bool(row.get("answerable", "true")) and not qrels:
                    raise ValueError("для answerable=true требуется хотя бы один relevant_doc_id")
                filters = json.loads(row.get("filters_json") or "{}")
                if not isinstance(filters, dict):
                    raise ValueError("filters_json должен быть JSON-объектом")
                split = (row.get("split") or "test").strip().casefold()
                if split not in args.include_splits:
                    continue
                qid = (row.get("qid") or f"manual_{row_no}").strip()
                evidence_quote = (row.get("evidence_quote") or "").strip()
                text_excerpt = (row.get("text_excerpt") or "").strip()
                text_for_annotation = (row.get("text_for_annotation") or text_excerpt).strip()
                # If an evidence quote is supplied, ensure it is supported by the saved source text.
                # Rows manually authored without a quote remain usable, but are marked explicitly.
                evidence_verified = False
                if evidence_quote and text_excerpt:
                    normalize_ws = lambda value: " ".join(value.split())
                    evidence_verified = normalize_ws(evidence_quote) in normalize_ws(text_for_annotation)
                    if not evidence_verified:
                        raise ValueError("evidence_quote не найден в text_excerpt")
                compiled.append({
                    "qid": qid,
                    "query": query,
                    "filters": filters,
                    "gold_doc_ids": list(qrels),
                    "relevance": qrels,
                    "reference_answer": answer,
                    "required_facts": [],
                    "answerable": _bool(row.get("answerable", "true")),
                    "split": split,
                    "category": row.get("draft_category") or f"manual_{row.get('source_type') or 'unknown'}",
                    "source_type": row.get("source_type") or "unknown",
                    "review_status": "approved",
                    "generation_method": "llm_draft_human_reviewed" if row.get("draft_status") else "human_reviewed",
                    "candidate_doc_id": row.get("candidate_doc_id", ""),
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "text_excerpt": text_excerpt,
                    "text_for_annotation": text_for_annotation,
                    "evidence_quote": evidence_quote,
                    "evidence_verified": evidence_verified,
                    "ambiguity_note": row.get("ambiguity_note", ""),
                })
            except Exception as exc:
                errors.append(f"строка {row_no}: {exc}")

    if errors:
        raise ValueError("Ошибки в размеченной очереди:\n- " + "\n- ".join(errors[:25]))
    if not compiled:
        raise ValueError("Нет строк со review_status=approved. Разметьте queue и повторите запуск.")
    ids = [row["qid"] for row in compiled]
    if len(set(ids)) != len(ids):
        raise ValueError("В compiled benchmark есть дублирующиеся qid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in sorted(compiled, key=lambda item: item["qid"]):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Скомпилировано вопросов: {len(compiled)} -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
