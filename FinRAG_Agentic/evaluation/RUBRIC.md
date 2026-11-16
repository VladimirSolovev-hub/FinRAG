# Agent evaluation rubric

The benchmark has three distinct evaluation layers. Do not collapse them into one LLM-judge score.

## A. Deterministic checks (automated; release gate)

- Financial formula unit tests: NPV, IRR, payback, ROI, input validation.
- Tool contract: required tool called, required args present, tool result schema valid.
- Data checks: SQL result row counts, null handling, duplicate grain checks.
- Output contract: required JSON fields and data types where structured output is used.

A critical numeric mismatch, unsafe database action, fabricated source ID, or unsupported claim of confirmed cash savings is a release-blocking failure.

## B. Heuristic checks (automated; triage only)

The current benchmark runner checks required tool names and selected phrases. This catches regressions but does not prove semantic correctness. Phrase matching is language-sensitive and can produce false positives/negatives.

## C. Expert review (manual; initial gold creation and sampled regression audit)

For each case, domain experts agree on:
- expected factual claims;
- acceptable alternatives in method selection;
- required assumptions and caveats;
- prohibited conclusions;
- critical-error conditions.

Use two independent reviewers for ambiguous methodology cases, resolve disagreements, and version the rubric. After calibration, use expert review on all new/changed high-risk cases and a stratified sample of routine regression runs.

## Suggested scorecard (100 points)

| Dimension | Weight | Typical checks |
|---|---:|---|
| Numerical correctness | 25 | Formula, sign, timing, rounding, units |
| Financial interpretation | 20 | Capacity value vs cashable savings |
| Model completeness | 15 | CAPEX, OPEX, human review, variable cost |
| Assumptions and limitations | 10 | Missing inputs clearly identified |
| Causal evaluation | 10 | Counterfactual and DiD assumptions |
| Traceability | 10 | Source, query, date range, version |
| Quality and risk | 5 | Rework, critical errors, guardrails |
| Recommendation | 5 | Strength of conclusion matches evidence |

## Critical failures override the numeric score

- fabricated input values or sources;
- incorrect financial formula that changes the decision;
- capacity value represented as confirmed cash savings without evidence;
- causal impact claimed without a credible design;
- recommendation to scale despite known unresolved critical calculation errors;
- unsupported currency/unit conversion or period mismatch.
