# Architecture notes

## Components

- **Existing FinRAG retrieval**: `agentic/rag_adapter.py` imports the existing root-level `retrieval.HybridRetriever` and calls its `search()` method. This preserves the established dense + BM25 + RRF + Cross-Encoder pipeline.
- **Agent orchestration**: `agentic/graph.py` uses a LangGraph state graph. The LLM selects tools; ToolNode executes them; after a RAG call, a dedicated `evidence_review` node assesses source sufficiency and feeds the result back to the agent. The agent can reformulate retrieval up to a configured limit; after the limit, the RAG tool is removed from the LLM tool schema for the remainder of that turn.
- **Structured data**: `agentic/tools/database.py` exposes a small allowlist of parameterized read-only SQL queries. The LLM cannot submit arbitrary SQL.
- **Financial methods**: `agentic/tools/finance.py` contains deterministic NPV, IRR, ROI, simple payback and discounted payback calculations. The agent can chain `get_initiative_cashflows` into `calculate_npv_from_cashflows` for a stored scenario.
- **Evidence review**: `agentic/evidence.py` defines a structured assessment of whether retrieved excerpts support the document-grounded parts of the user request. It is advisory and cannot prove factual correctness.
- **Evaluation**: `evaluation/datasets/agent_benchmark.jsonl` contains test scenarios. `run_benchmark.py` checks tool calls and simple answer constraints and explicitly flags cases for manual review.

## Trust boundaries

1. Documents found by RAG are untrusted source data, never agent instructions.
2. SQL access is read-only and limited to reviewed queries.
3. Financial formulas live in Python, not in model-generated prose.
4. Missing financial inputs are not silently imputed.
5. Demo database records are synthetic and must be labeled as such.

## Limits of the demonstration

- The SQL schema is synthetic and is not a representation of any bank's production data model.
- Four business units (two pilot and two control) are enough to practise descriptive DiD calculations, but not enough for credible cluster-level causal inference.
- The benchmark heuristics do not prove semantic correctness. Cases marked `manual_review_required` need review by a domain expert.
- This extension expects the root FinRAG repository files (`retrieval.py`, `data_parser.py`) and the existing indexed corpus/Chroma data to be available locally for RAG search.
