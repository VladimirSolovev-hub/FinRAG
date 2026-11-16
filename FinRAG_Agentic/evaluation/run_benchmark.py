from __future__ import annotations
import json
import re
from pathlib import Path
from agentic.graph import invoke_agent

DATASET = Path(__file__).parent / "datasets" / "agent_benchmark.jsonl"


def main() -> None:
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in cases:
        print(f"\n[{case['id']}] {case['query']}")
        try:
            output = invoke_agent(case["query"], thread_id=f"eval-{case['id']}")
            answer = output["answer"] or ""
            called = {event.get("tool") for event in output["tool_events"] if event.get("tool")}
            missing_tools = sorted(set(case.get("required_tools", [])) - called)
            lower = answer.casefold()
            missing_phrases = [x for x in case.get("must_mention", []) if x.casefold() not in lower]
            forbidden = [x for x in case.get("must_not_claim", []) if x.casefold() in lower]
            # This harness checks tool-call names and simple phrase heuristics only.
            # Phrase checks are language-sensitive and are triage signals, not semantic proof.
            passed = not (missing_tools or missing_phrases or forbidden)
            row = {
                "id": case["id"], "passed_heuristics": passed,
                "missing_tools": missing_tools,
                "missing_phrases": missing_phrases,
                "forbidden_phrases": forbidden,
                "manual_review_required": case.get("manual_review_required", True),
                "answer": answer,
                "evidence_reviews": output.get("evidence_reviews", []),
            }
            results.append(row)
            print("PASS heuristics" if passed else "FAIL heuristics")
            if row["manual_review_required"]:
                print("MANUAL REVIEW REQUIRED")
        except Exception as exc:
            results.append({"id": case["id"], "passed_heuristics": False, "error": str(exc), "manual_review_required": True})
            print("ERROR:", exc)
    out = Path(__file__).parent / "results" / "benchmark_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    total = len(results)
    passed = sum(bool(x.get("passed_heuristics")) for x in results)
    print(f"Heuristic pass rate: {passed}/{total} ({passed/total:.1%})")
    print("Important: heuristic pass rate is not a semantic quality score.")


if __name__ == "__main__":
    main()
