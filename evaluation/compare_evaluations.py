"""Compare aggregate retrieval metrics from a frozen baseline and a new evaluation run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = ("recall@10", "mrr@10", "ndcg@10", "mean_latency_seconds")


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="path to frozen baseline comparison_summary.json")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="path to new evaluation comparison_summary.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional CSV destination; defaults beside candidate summary")
    args = parser.parse_args()

    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)
    rows = []
    for mode in sorted(set(baseline) | set(candidate)):
        before = baseline.get(mode, {})
        after = candidate.get(mode, {})
        for metric in METRICS:
            base_value = before.get(metric)
            cand_value = after.get(metric)
            if base_value is None or cand_value is None:
                delta = None
            else:
                delta = float(cand_value) - float(base_value)
            rows.append({
                "mode": mode,
                "metric": metric,
                "baseline": base_value,
                "candidate": cand_value,
                "delta_candidate_minus_baseline": delta,
                "interpretation": (
                    "lower_is_better" if metric == "mean_latency_seconds" else "higher_is_better"
                ),
            })

    output = args.output or args.candidate.parent / "comparison_to_baseline.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["mode"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Baseline: {args.baseline.resolve()}")
    print(f"Candidate: {args.candidate.resolve()}")
    print(f"Comparison saved: {output.resolve()}")
    print("Delta = candidate - baseline; for latency, negative is better.")
    for row in rows:
        if row["metric"] == "recall@10" or row["metric"] == "mrr@10" or row["metric"] == "ndcg@10":
            print(f"{row['mode']:12s} {row['metric']:24s} baseline={row['baseline']} candidate={row['candidate']} delta={row['delta_candidate_minus_baseline']}")
        elif row["metric"] == "mean_latency_seconds":
            print(f"{row['mode']:12s} {row['metric']:24s} baseline={row['baseline']} candidate={row['candidate']} delta={row['delta_candidate_minus_baseline']}")


if __name__ == "__main__":
    main()
