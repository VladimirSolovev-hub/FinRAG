"""Evaluate FinRAG retrieval against a JSONL qrels/reference-answer benchmark.

Modes: hybrid (production defaults), hybrid_rrf (dense+BM25+RRF without reranker),
dense (Chroma only), bm25 (lexical only), or all (all four on the same queries).
The reference answers are recorded for the next generation-evaluation phase; this
script evaluates retrieval and does not score generated answers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VALID_FILTERS = {"source_type", "date_from", "date_to", "reporting_year", "region", "okved_section"}
MODES = ("hybrid", "hybrid_rrf", "dense", "bm25")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSONL, строка {line_no}: {exc}") from exc
            if not item.get("qid") or not item.get("query"):
                raise ValueError(f"В строке {line_no} отсутствует qid или query")
            items.append(item)
    return items


def get_qrels(item: dict[str, Any]) -> dict[str, int]:
    qrels = item.get("relevance")
    if isinstance(qrels, dict) and qrels:
        return {str(key): int(value) for key, value in qrels.items() if int(value) > 0}
    ids = item.get("gold_doc_ids") or []
    return {str(doc_id): 1 for doc_id in ids}


def recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if not qrels:
        return math.nan
    return len(set(ranked_ids[:k]) & set(qrels)) / len(qrels)


def precision_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k должен быть положительным")
    return len(set(ranked_ids[:k]) & set(qrels)) / k


def reciprocal_rank_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in qrels:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if not qrels:
        return math.nan
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        grade = qrels.get(doc_id, 0)
        if grade > 0:
            dcg += (2**grade - 1) / math.log2(rank + 1)
    ideal_grades = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1)
               for rank, grade in enumerate(ideal_grades, start=1))
    return dcg / idcg if idcg else 0.0


def per_question_metrics(ranked_ids: list[str], qrels: dict[str, int], ks: Iterable[int]) -> dict[str, float]:
    values: dict[str, float] = {}
    for k in ks:
        values[f"recall@{k}"] = recall_at_k(ranked_ids, qrels, k)
        values[f"precision@{k}"] = precision_at_k(ranked_ids, qrels, k)
        values[f"mrr@{k}"] = reciprocal_rank_at_k(ranked_ids, qrels, k)
        values[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, qrels, k)
        values[f"hit_rate@{k}"] = float(bool(set(ranked_ids[:k]) & set(qrels)))
    return values


def _mean(items: list[float]) -> float | None:
    finite = [float(value) for value in items if math.isfinite(float(value))]
    return statistics.mean(finite) if finite else None


def _filters(item: dict[str, Any]) -> dict[str, Any]:
    filters = item.get("filters") or {}
    if not isinstance(filters, dict):
        raise ValueError(f"{item.get('qid')}: filters должен быть объектом")
    unknown = set(filters) - VALID_FILTERS
    if unknown:
        raise ValueError(f"{item.get('qid')}: неподдерживаемые фильтры {sorted(unknown)}")
    return filters


def _search_one(retriever, item: dict[str, Any], mode: str, k: int,
                dense_candidates: int, sparse_candidates: int, rerank_candidates: int):
    filters = _filters(item)
    if mode == "hybrid":
        return retriever.search(
            item["query"], k=k, dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates, rerank_candidates=rerank_candidates,
            **filters,
        )
    if mode == "hybrid_rrf":
        return retriever.search(
            item["query"], k=k, use_reranker=False,
            dense_candidates=dense_candidates, sparse_candidates=sparse_candidates,
            rerank_candidates=rerank_candidates, **filters,
        )

    start, end = retriever._normalize_date_filters(filters.get("date_from"), filters.get("date_to"))
    kwargs = {
        "source_type": filters.get("source_type"),
        "start": start,
        "end": end,
        "reporting_year": filters.get("reporting_year"),
        "region": filters.get("region"),
        "okved_section": filters.get("okved_section"),
    }
    if mode == "dense":
        doc_ids = retriever._dense_search(item["query"], k, **kwargs)
    elif mode == "bm25":
        doc_ids = retriever._bm25_search(item["query"], k, **kwargs)
    else:
        raise ValueError(f"Неизвестный mode: {mode}")

    # Component-only modes need rank order, not the raw numeric score.
    from retrieval import SearchHit
    hits = []
    for rank, doc_id in enumerate(doc_ids, start=1):
        document = retriever._documents_by_id.get(doc_id)
        if document:
            hits.append(SearchHit(
                doc_id=doc_id,
                text=document.text,
                metadata=dict(document.metadata),
                score=1.0 / rank,
                score_type=f"{mode}_rank",
            ))
    return hits


def _summarize(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_questions": len(rows)}
    metric_names = [f"{metric}@{k}" for k in ks
                    for metric in ("recall", "precision", "mrr", "ndcg", "hit_rate")]
    for name in metric_names:
        summary[name] = _mean([row[name] for row in rows])
    summary["mean_latency_seconds"] = _mean([row["latency_seconds"] for row in rows])
    summary["median_latency_seconds"] = statistics.median([row["latency_seconds"] for row in rows]) if rows else None
    summary["empty_result_rate"] = _mean([float(row["n_results"] == 0) for row in rows])
    return summary


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "evaluation/data/rfsd_lookup_benchmark.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/unified_corpus_v3.parquet")
    parser.add_argument("--chroma", type=Path, default=ROOT / "data/chroma_db")
    parser.add_argument("--collection", default="finrag")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base",
                        help="must match the embedding model used to build the selected Chroma index")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3",
                        help="cross-encoder reranker model ID/path; use none only for runs that do not invoke hybrid reranking")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--sparse-candidates", type=int, default=100)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if any(k <= 0 for k in args.k):
        parser.error("Все k должны быть положительными")
    if not args.benchmark.exists():
        raise FileNotFoundError(
            f"Бенчмарк не найден: {args.benchmark.resolve()}. Сначала запустите "
            "python evaluation/build_baseline_dataset.py"
        )
    if not args.corpus.exists():
        raise FileNotFoundError(f"Корпус не найден: {args.corpus.resolve()}")
    items = load_jsonl(args.benchmark)
    items = [item for item in items if args.split == "all" or item.get("split", "test") == args.split]
    # Retrieval has no meaningful gold-document metric for deliberately unanswerable items.
    items = [item for item in items if item.get("answerable", True) and get_qrels(item)]
    items = [item for item in items if item.get("review_status") in {"auto_generated", "approved"}]
    if not items:
        raise ValueError(
            f"После фильтрации нет вопросов для split={args.split!r}. Проверьте dataset_manifest.json; "
            "для ручных вопросов используйте compile_reviewed_benchmark.py."
        )

    # Load the whole retriever once; every mode is evaluated on exactly the same questions.
    from retrieval import HybridRetriever
    retriever = HybridRetriever(
        corpus_path=args.corpus,
        chroma_path=args.chroma,
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
        reranker_model_name=(None if args.reranker_model.strip().casefold() == "none" else args.reranker_model),
    )
    modes = MODES if args.mode == "all" else (args.mode,)
    max_k = max(args.k)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (ROOT / "evaluation/results" / f"baseline_{run_stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(args.benchmark.resolve()),
        "corpus_path": str(args.corpus.resolve()),
        "chroma_path": str(args.chroma.resolve()),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "split": args.split,
        "n_questions": len(items),
        "modes": list(modes),
        "k": args.k,
        "dense_candidates": args.dense_candidates,
        "sparse_candidates": args.sparse_candidates,
        "rerank_candidates": args.rerank_candidates,
        "index_metadata": retriever.collection.metadata or {},
        "caveat": (
            "Automatic RFSD lookup questions are template-generated and include exact INN/year identifiers. "
            "Results measure exact structured lookup reliability; use approved human-written questions to "
            "claim general semantic retrieval quality."
        ),
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    all_summaries: dict[str, Any] = {}
    for mode in modes:
        print(f"\n=== Оценка mode={mode}; вопросов={len(items)}; split={args.split} ===")
        details: list[dict[str, Any]] = []
        ranking_path = output_dir / f"rankings_{mode}.jsonl"
        with ranking_path.open("w", encoding="utf-8") as ranking_stream:
            for item_index, item in enumerate(items, start=1):
                started = time.perf_counter()
                hits = _search_one(
                    retriever, item, mode, max_k,
                    args.dense_candidates, args.sparse_candidates, args.rerank_candidates,
                )
                latency = time.perf_counter() - started
                ranked_ids = [hit.doc_id for hit in hits]
                qrels = get_qrels(item)
                metrics = per_question_metrics(ranked_ids, qrels, args.k)
                detail = {
                    "qid": item["qid"],
                    "query": item["query"],
                    "category": item.get("category", "unknown"),
                    "source_type": item.get("source_type", "unknown"),
                    "split": item.get("split", "test"),
                    "gold_doc_ids": list(qrels),
                    "ranked_doc_ids": ranked_ids,
                    "top1_correct": bool(ranked_ids and ranked_ids[0] in qrels),
                    "n_results": len(ranked_ids),
                    "latency_seconds": latency,
                    "reference_answer": item.get("reference_answer", ""),
                    **metrics,
                }
                details.append(detail)
                ranking_stream.write(json.dumps({
                    **detail,
                    "results": [{
                        "doc_id": hit.doc_id,
                        "score": hit.score,
                        "score_type": hit.score_type,
                        "metadata": hit.metadata,
                        "text_excerpt": " ".join(hit.text.split())[:1200],
                    } for hit in hits],
                }, ensure_ascii=False, sort_keys=True) + "\n")
                if item_index % 25 == 0 or item_index == len(items):
                    print(f"  {item_index}/{len(items)} вопросов обработано")

        summary = _summarize(details, args.k)
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in details:
            by_category[row["category"]].append(row)
        summary["by_category"] = {category: _summarize(group, args.k)
                                  for category, group in sorted(by_category.items())}
        all_summaries[mode] = summary
        _write_json(output_dir / f"summary_{mode}.json", summary)

        csv_path = output_dir / f"per_question_{mode}.csv"
        fieldnames = list(details[0].keys()) if details else ["qid"]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in details:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})
        print(json.dumps({key: summary.get(key) for key in [f"recall@10", f"mrr@10", f"ndcg@10", "mean_latency_seconds"]},
                         ensure_ascii=False, indent=2))

    _write_json(output_dir / "comparison_summary.json", all_summaries)
    print(f"\nОценка завершена. Отчёты сохранены: {output_dir.resolve()}")
    print("Reference answers записаны в per-question и rankings-файлы, но здесь не оценивается генерация LLM.")


if __name__ == "__main__":
    main()
