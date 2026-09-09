# Contributing to FinSight

## Setup

```bash
uv sync                       # runtime + dev deps (from pyproject.toml [dependency-groups].dev)
uv run pre-commit install     # ruff-check --fix + ruff-format on commit
docker compose up qdrant -d   # only needed for integration/eval work
```

Python 3.11 (`.python-version`). All dependencies are managed with `uv`; `uv.lock` is committed — update it (`uv lock`) in the same commit as any `pyproject.toml` dependency change.

## Before you push

The CI (`.github/workflows/ci.yml`) runs exactly three checks on every push/PR to `main`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```

All three must pass. `ruff` config is in `pyproject.toml` (`line-length = 100`, rules `E`/`F`/`I`, `E501` ignored, double quotes). Run `uv run ruff format .` to autofix formatting.

`tests/` is unit-only — no Qdrant or Groq required, and it must stay that way so CI needs no services. Anything that needs a live API belongs in `evaluation/`.

## Architecture rules to respect

These are the invariants the design depends on. A change that breaks one needs a matching update to [docs/DECISIONS.md](docs/DECISIONS.md) and a reviewer conversation.

1. **Citations are structural.** Every claim that reaches a user carries a `Citation` (document, page, verbatim snippet). Don't add a code path that emits a figure without one.
2. **No LLM arithmetic.** Currency/scale conversion and any derived figure is done in `core/unit_normalizer.py` with `Decimal`, or not at all. The Analyst / Comparator / Synthesizer prompts forbid computed figures — keep them that way.
3. **The AuditorAgent is a hard gate.** `UNVERIFIABLE` claims must have no code path to the synthesised report. They go in the audit log's `blocked_unverifiable` only.
4. **One LLM dispatch point.** All model calls go through `core/groq_client.py` (retry → reserve → hedge). Don't instantiate an OpenAI client elsewhere.
5. **Single process.** In-process state (compiled `StateGraph` singleton, BM25 cache, `MetricsStore`) assumes one uvicorn worker. Don't add state that breaks under that assumption without also making it shared.
6. **Every run emits an `AuditLog`.** Even the failure/"insufficient evidence" paths.
7. **Memory reaches the Planner only.** Recalled session/user context (`memory_context`) may shape `PLANNER_PROMPT`; it must never be threaded into `SYNTHESIZER_PROMPT` or any claim/citation path. That's what keeps a user's own unreviewed history from becoming an unverified figure in the report.
8. **Memory failures degrade, never block.** `recall_memory`/`remember` nodes must catch and continue — a Qdrant/`finsight_memory` outage should cost a query its recalled context or its write, not the query itself.

## Making a change

- Branch off `main`: `feat/…`, `fix/…`, `docs/…`, `chore/…`, `perf/…`, `tests/…`.
- Keep commits focused — one logical change each. The project history intentionally has many small commits.
- Commit message: imperative subject, a body explaining *why* when it isn't obvious.
- Update docs in the same PR as the behaviour change:
  - new/changed module → [docs/LLD.md](docs/LLD.md)
  - new component, requirement, or cross-cutting concern → [docs/HLD.md](docs/HLD.md)
  - a design trade-off you deliberately made → [docs/DECISIONS.md](docs/DECISIONS.md) (append a numbered entry; don't rewrite old ones)
  - operational impact → [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Add/adjust unit tests under `tests/` for any logic change in `core/`, `retrieval/`, `ingestion/`, or the route helpers.
- If the change touches retrieval, ranking, agent orchestration, scoring, or prompts, run the eval harness against a locally running API before opening the PR:
  ```bash
  uvicorn api.main:app &        # with a real GROQ_API_KEY and an ingested corpus
  python evaluation/harness.py evaluation/queries/happy_path.json
  python evaluation/harness.py evaluation/queries/adversarial.json
  ```
  Note the pass counts in the PR description. Results land in `evaluation/results/` (git-ignored).

## Adding an agent

1. New async function in `agents/` — a plain coroutine, no framework decorators.
2. Add a node for it in `orchestration/graph.py::build_graph()` and wire it into the edge list (or a `Send`-based fan-out, if it's per-subtask like `retrieve_analyze`). Both `/query` and `/query/stream` run the same compiled graph, so there is only one pipeline to keep in step now.
3. Add it to `orchestration/graph.py::AGENT_SEQUENCE` and, if it has its own latency budget, to `MetricsStore._agent_timings`.
4. Document it: HLD §5.2 table, LLD §3.

## What not to commit

`.env`, `qdrant_storage/`, `audit_logs/`, `data/filings/`, `evaluation/results/`, caches — all already in `.gitignore`. Don't force-add them.
