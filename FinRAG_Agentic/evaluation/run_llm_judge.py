from __future__ import annotations
import json
from pathlib import Path

from evaluation.judge import judge_answer

BASE = Path(__file__).parent
DATASET = BASE / "datasets" / "agent_benchmark.jsonl"
RESULTS = BASE / "results" / "benchmark_results.json"
OUTPUT = BASE / "results" / "llm_judge_results.json"


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit("Сначала запусти python -m evaluation.run_benchmark")
    cases = {x["id"]: x for x in (json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip())}
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    judged = []
    for result in results:
        case = cases.get(result["id"])
        answer = result.get("answer", "")
        if not case or not answer:
            judged.append({"id": result.get("id"), "error": "No matching case or empty answer", "needs_human_review": True})
            continue
        print(f"Judging {result['id']}...")
        try:
            judged.append({"id": result["id"], **judge_answer(case, answer)})
        except Exception as exc:
            judged.append({"id": result["id"], "error": str(exc), "needs_human_review": True})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}")
    print("LLM-as-a-judge is advisory only. Expert review is still required for high-risk financial conclusions.")


if __name__ == "__main__":
    main()
