# GitHub release checklist for FinRAG

## Before making the repository public

- [ ] Review the actual repository tree and update the README structure accordingly.
- [ ] Add a pinned `requirements.txt` or `pyproject.toml` generated from the working environment; do not guess dependency versions.
- [ ] Add a `.env.example` containing variable names only, never real credentials.
- [ ] Ensure `.env`, API keys, access tokens, private certificates, and local credentials are not tracked.
- [ ] Do not commit `data/chroma_db/`, large local Parquet/JSONL corpora, raw Telegram exports, checkpoints, model weights, caches, or temporary outputs unless their redistribution is explicitly permitted.
- [ ] Inspect `evaluation/results/ragas_ablation/generations.jsonl`: it includes full generated answers, retrieved contexts, document IDs, usage, and an absolute local benchmark path. Keep it private or publish only the aggregated `RESULTS.md` and CSV summaries.
- [ ] Review benchmark records and source attribution, licenses, copyright, and privacy requirements before publishing any questions or source excerpts.
- [ ] Add a clear license only after deciding what license applies to original code and confirming rights to included assets/data. A code license does not automatically license third-party datasets.
- [ ] Remove local usernames and absolute paths from manifests, notebooks, logs, and config files.
- [ ] Verify that public README commands match the actual CLI and that the repository can be installed from a clean environment.
- [ ] Add a minimal smoke test that does not require a paid API call, if possible.
- [ ] State clearly that this is a research prototype and not financial/legal advice.

## Suggested `.gitignore` additions

```gitignore
.env
.env.*
!.env.example
.DS_Store
__pycache__/
*.py[cod]
.venv/
venv/

# Local corpora, indexes and generated data
data/
evaluation/results/
*.parquet
*.sqlite
*.db
# Add explicit private JSONL paths if they contain source text or non-public benchmark data.
# Example: evaluation/data/reviewed_benchmark_audited_v2.jsonl

# Model checkpoints and caches
models/
checkpoints/
.cache/
```

Review this list before adopting it: if a small, sanitized benchmark or a carefully curated example JSONL is intentionally part of the public repository, explicitly unignore that exact path. Do not blanket-ignore files that the project needs to reproduce its tests.
