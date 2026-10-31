"""RAGAS evaluation and a small retrieval ablation for FinRAG.

Phase 1 (generate): run the existing answer generator on benchmark questions with
and without the reranker and persist all responses/contexts to JSONL.
Phase 2 (score): score persisted results using RAGAS metrics and a Groq-hosted judge.

`--limit 0` means all test questions. Generation and scoring are resumable.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation import (  # noqa: E402
    DEFAULT_MODEL,
    _load_project_env,
    answer_question,
)
from retrieval import HybridRetriever  # noqa: E402

DEFAULT_BENCHMARK = ROOT / "evaluation/data/reviewed_benchmark_audited_v2.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation/results/ragas_ablation"
VARIANTS = ("full", "no_reranker")


class RetrieverVariant:
    """Thin proxy: preserve the existing search path, optionally disable reranking."""

    def __init__(self, retriever: HybridRetriever, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported variant: {variant}")
        self._retriever = retriever
        self._variant = variant

    def search(self, query: str, **kwargs):
        if self._variant == "no_reranker":
            kwargs["use_reranker"] = False
        # For `full`, do not override defaults: this preserves HybridRetriever's
        # production behavior (including its source-specific reranker exclusions).
        return self._retriever.search(query, **kwargs)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()


def _load_benchmark(path: Path, split: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split", "test") != split:
            continue
        if row.get("answerable", True) is False:
            continue
        if row.get("review_status") not in {"approved", "auto_generated"}:
            continue
        reference = str(row.get("reference_answer") or "").strip()
        gold_ids = row.get("gold_doc_ids") or []
        if not reference or not gold_ids:
            continue
        if not isinstance(gold_ids, list):
            continue
        selected.append(row)
    return selected


def _get_retrieved_contexts(retriever: HybridRetriever, result: dict[str, Any], max_chars: int) -> list[str]:
    """Recreate the actual text passed into generation, in ranking order."""
    contexts: list[str] = []
    for source in result.get("retrieved_sources", []):
        doc_id = source.get("doc_id")
        doc = getattr(retriever, "_documents_by_id", {}).get(doc_id)
        if doc is None:
            continue
        text = str(getattr(doc, "text", "") or "").strip()
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[Фрагмент сокращён по длине]"
        if text:
            contexts.append(text)
    return contexts


def _select_questions(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def generate_phase(args: argparse.Namespace) -> None:
    _load_project_env()
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise RuntimeError("GROQ_API_KEY не найден в окружении или в .env проекта")
    if not args.benchmark.exists():
        raise FileNotFoundError(f"Benchmark не найден: {args.benchmark}")

    benchmark_rows = _load_benchmark(args.benchmark, args.split)
    if not benchmark_rows:
        raise RuntimeError(f"Не найдено подходящих benchmark-вопросов для split={args.split!r}")
    selected_rows = _select_questions(benchmark_rows, args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = args.output_dir / "generations.jsonl"
    existing = _read_jsonl(generations_path) if not args.no_resume else []
    completed = {
        (str(row.get("qid")), str(row.get("variant")))
        for row in existing
        if row.get("qid") and row.get("variant")
    }
    if args.no_resume:
        generations_path.write_text("", encoding="utf-8")

    print(f"Benchmark: {args.benchmark}")
    print(f"Split: {args.split}; questions selected: {len(selected_rows)}")
    print(f"Variants: {', '.join(args.variants)}")
    print(f"Output: {generations_path}")

    print("Загружаю retrieval (корпус, Chroma, embeddings и reranker)...", file=sys.stderr)
    retriever = HybridRetriever(
        corpus_path=args.corpus,
        chroma_path=args.chroma,
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
    )

    for item in selected_rows:
        qid = str(item.get("qid") or "")
        question = str(item.get("query") or "").strip()
        if not qid or not question:
            print("Skipping record without qid/query", file=sys.stderr)
            continue
        for variant in args.variants:
            key = (qid, variant)
            if not args.no_resume and key in completed:
                print(f"SKIP {qid} / {variant} (already saved)")
                continue
            proxy = RetrieverVariant(retriever, variant)
            started = time.perf_counter()
            result = answer_question(
                proxy,  # answer_question only requires a .search() method.
                question,
                model=args.model,
                top_k=args.top_k,
                dense_candidates=args.dense_candidates,
                sparse_candidates=args.sparse_candidates,
                rerank_candidates=args.rerank_candidates,
                max_doc_chars=args.max_doc_chars,
                max_completion_tokens=args.max_completion_tokens,
                timeout=args.timeout,
                source_type=(item.get("filters") or {}).get("source_type"),
                date_from=(item.get("filters") or {}).get("date_from"),
                date_to=(item.get("filters") or {}).get("date_to"),
                reporting_year=(item.get("filters") or {}).get("reporting_year"),
                region=(item.get("filters") or {}).get("region"),
                okved_section=(item.get("filters") or {}).get("okved_section"),
            )
            contexts = _get_retrieved_contexts(retriever, result, args.max_doc_chars)
            source_ids = [str(source.get("doc_id")) for source in result.get("retrieved_sources", [])]
            record = {
                "qid": qid,
                "variant": variant,
                "split": item.get("split", args.split),
                "category": item.get("category"),
                "source_type": item.get("source_type"),
                "question": question,
                "reference": str(item.get("reference_answer") or "").strip(),
                "gold_doc_ids": item.get("gold_doc_ids") or [],
                "response": result.get("answer", ""),
                "retrieved_doc_ids": source_ids,
                "retrieved_contexts": contexts,
                "citation_warnings": result.get("citation_warnings", []),
                "cited_sources": result.get("cited_sources", []),
                "model": result.get("model") or args.model,
                "usage": result.get("usage", {}),
                "retrieval_seconds": result.get("retrieval_seconds"),
                "generation_seconds": result.get("generation_seconds"),
                "total_seconds": time.perf_counter() - started,
                "generator_config": {
                    "top_k": args.top_k,
                    "dense_candidates": args.dense_candidates,
                    "sparse_candidates": args.sparse_candidates,
                    "rerank_candidates": args.rerank_candidates,
                    "max_doc_chars": args.max_doc_chars,
                    "max_completion_tokens": args.max_completion_tokens,
                },
                "benchmark_path": str(args.benchmark),
            }
            _append_jsonl(generations_path, record)
            completed.add(key)
            print(
                f"Saved {qid} / {variant}: retrieval={record['retrieval_seconds']:.2f}s, "
                f"generation={record['generation_seconds']:.2f}s, contexts={len(contexts)}"
            )


def _metric_kwargs(metric_name: str, row: dict[str, Any]) -> dict[str, Any]:
    if metric_name == "faithfulness":
        return {
            "user_input": row["question"],
            "response": row["response"],
            "retrieved_contexts": row["retrieved_contexts"],
        }
    if metric_name == "context_precision":
        return {
            "user_input": row["question"],
            "reference": row["reference"],
            "retrieved_contexts": row["retrieved_contexts"],
        }
    if metric_name == "context_recall":
        return {
            "user_input": row["question"],
            "reference": row["reference"],
            "retrieved_contexts": row["retrieved_contexts"],
        }
    if metric_name == "factual_correctness":
        return {"response": row["response"], "reference": row["reference"]}
    if metric_name == "answer_relevancy":
        return {"user_input": row["question"], "response": row["response"]}
    raise KeyError(metric_name)


async def score_phase(args: argparse.Namespace) -> None:
    _load_project_env()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY не найден в окружении или в .env проекта")

    try:
        import instructor
        from openai import AsyncOpenAI
        from ragas.llms import InstructorLLM
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS/OpenAI SDK не установлены. Выполните: "
            "/usr/bin/python3 -m pip install 'ragas>=0.4,<0.5' 'openai>=1.0'"
        ) from exc

    generations_path = args.output_dir / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(
            f"Не найден {generations_path}. Сначала запустите --phase generate."
        )
    rows = _read_jsonl(generations_path)
    rows = [r for r in rows if r.get("question") and r.get("response") is not None and r.get("reference")]
    # Restrict the score phase to the selected variants, if supplied.
    rows = [r for r in rows if r.get("variant") in args.variants]
    if args.limit > 0:
        selected_qids = []
        for row in rows:
            qid = str(row.get("qid"))
            if qid not in selected_qids:
                selected_qids.append(qid)
            if len(selected_qids) >= args.limit:
                break
        rows = [row for row in rows if str(row.get("qid")) in set(selected_qids)]

    scores_path = args.output_dir / "ragas_scores.jsonl"
    previous = _read_jsonl(scores_path) if not args.no_resume else []
    latest_scores = {
        (str(old.get("qid")), str(old.get("variant"))): old
        for old in previous
        if old.get("qid") and old.get("variant")
    }
    if args.no_resume:
        scores_path.write_text("", encoding="utf-8")

    openai_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=args.timeout,
    )

    # Instructor сам управляет MD_JSON.
    # mode больше не передаётся через аргументы RAGAS.
    instructor_client = instructor.from_openai(
        openai_client,
        mode=instructor.Mode.MD_JSON,
    )

    judge = InstructorLLM(
        client=instructor_client,
        model=args.judge_model,
        provider="openai",
        max_tokens=args.judge_max_tokens,
        temperature=0.6,
        top_p=0.95,
        reasoning_effort=args.judge_reasoning_effort,
    )

    # Use a local multilingual embedding model for AnswerRelevancy; no second
    # embedding API key is required. CPU is deliberate to avoid competing with MPS.
    embeddings = HuggingFaceEmbeddings(
        model=args.ragas_embedding_model,
        device=args.ragas_embedding_device,
        batch_size=8,
    )
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "factual_correctness": FactualCorrectness(llm=judge, mode="f1"),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
    }

    print(f"RAGAS rows in scope: {len(rows)}")
    print(f"Judge model: {args.judge_model}")
    print("Metrics: " + ", ".join(metrics.keys()))
    for row in rows:
        key = (str(row.get("qid")), str(row.get("variant")))
        previous_record = latest_scores.get(key) if not args.no_resume else None
        if previous_record and all(
            isinstance(previous_record.get("scores", {}).get(name), (int, float))
            for name in metrics
        ):
            print(f"SKIP scores {key[0]} / {key[1]} (all metrics already saved)")
            continue

        record = {
            "qid": row["qid"],
            "variant": row["variant"],
            "question": row["question"],
            "scores": dict((previous_record or {}).get("scores") or {}),
            "errors": dict((previous_record or {}).get("errors") or {}),
            "judge_model": args.judge_model,
        }
        for metric_name, metric in metrics.items():
            existing_value = record["scores"].get(metric_name)
            if isinstance(existing_value, (int, float)):
                continue
            try:
                metric_result = await metric.ascore(**_metric_kwargs(metric_name, row))
                record["scores"][metric_name] = float(metric_result.value)
                record["errors"].pop(metric_name, None)
                print(f"{key[0]} / {key[1]} / {metric_name}: {metric_result.value}")
            except Exception as exc:  # Continue and persist partial results for auditability.
                record["scores"][metric_name] = None
                record["errors"][metric_name] = f"{type(exc).__name__}: {exc}"
                print(f"ERROR {key[0]} / {key[1]} / {metric_name}: {exc}", file=sys.stderr)
        _append_jsonl(scores_path, record)
        latest_scores[key] = record

    _write_summary(scores_path, args.output_dir / "ragas_summary.csv")
    print(f"Scores: {scores_path}")
    print(f"Summary: {args.output_dir / 'ragas_summary.csv'}")


def _write_summary(scores_path: Path, summary_path: Path) -> None:
    # Keep only the latest record per question/variant so retries do not double-count.
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(scores_path):
        latest[(str(row.get("qid")), str(row.get("variant")))] = row
    rows = list(latest.values())
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        variant = str(row.get("variant", "unknown"))
        for metric, value in (row.get("scores") or {}).items():
            if isinstance(value, (int, float)):
                aggregates[variant][metric].append(float(value))
    metric_names = sorted({m for variant in aggregates.values() for m in variant})
    columns = ["variant", "n_scored"] + [f"mean_{m}" for m in metric_names]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for variant in sorted(aggregates):
            values = aggregates[variant]
            out: dict[str, Any] = {"variant": variant}
            out["n_scored"] = max((len(values[m]) for m in metric_names), default=0)
            for metric in metric_names:
                scores = values.get(metric, [])
                out[f"mean_{metric}"] = round(sum(scores) / len(scores), 6) if scores else ""
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "score"), default="generate")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--limit", type=int, default=2, help="First N questions for a pilot; 0 = all selected questions")
    parser.add_argument("--variants", default="full,no_reranker",
                        help="Comma-separated subset: full,no_reranker")
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", DEFAULT_MODEL), help="Generation model on Groq")
    parser.add_argument("--judge-model", default=os.getenv("GROQ_JUDGE_MODEL", DEFAULT_MODEL), help="RAGAS judge model on Groq")
    parser.add_argument("--judge-max-tokens", type=int, default=4096,
                        help="Completion token budget for RAGAS judge (includes reasoning tokens)")
    parser.add_argument("--judge-reasoning-effort", choices=["low", "medium", "high"], default="low",
                        help="Reasoning effort for GPT-OSS RAGAS judge")
    parser.add_argument("--ragas-embedding-model", default="intfloat/multilingual-e5-base", help="Local multilingual embedding model for answer relevancy")
    parser.add_argument("--ragas-embedding-device", default="cpu", choices=("cpu", "mps"), help="Device for RAGAS answer-relevancy embeddings")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/unified_corpus_v3.parquet")
    parser.add_argument("--chroma", type=Path, default=ROOT / "data/chroma_db")
    parser.add_argument("--collection", default="finrag")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--sparse-candidates", type=int, default=100)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--max-doc-chars", type=int, default=4500)
    parser.add_argument("--max-completion-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--no-resume", action="store_true", help="Overwrite the relevant output file instead of skipping saved qid/variant rows")
    args = parser.parse_args()
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        parser.error(f"Unknown variants: {sorted(unknown)}. Allowed: {', '.join(VARIANTS)}")
    if not args.variants:
        parser.error("At least one variant is required")
    if args.limit < 0:
        parser.error("--limit должен быть >= 0")
    if args.top_k <= 0:
        parser.error("--top-k должен быть положительным")
    return args


def main() -> int:
    args = parse_args()
    if args.phase == "generate":
        generate_phase(args)
    else:
        asyncio.run(score_phase(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
