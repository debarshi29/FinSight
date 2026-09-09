# FinSight — Architecture

This document describes the system design decisions, data flows, component responsibilities, and technical models that make up FinSight. It is the single authoritative reference for anyone contributing to, evaluating, or auditing the system.

---

## Design Goals

The architecture is shaped by a single constraint that separates FinSight from a standard RAG system: **any claim that cannot be verified against a verbatim source passage must be blocked before reaching the user, not warned about after the fact.**

This forces several design decisions that would be unnecessary in a looser system:

- Citations are a structural field, not a cosmetic annotation. Every agent hop carries citations as typed fields on the LangGraph `FinSightState` and the pipeline cannot produce output without them.
- The AuditorAgent is a separate structural pass. It is not a prompt instruction asking the model to be careful. It calls the LLM again on each claim individually and blocks `UNVERIFIABLE` results before synthesis.
- The audit log is a first-class output, not a side effect. It records the plan, every retrieval, every claim, what was blocked, and which agents were invoked — for every query run.

---

## System Overview

```
                       ┌────────────────────────────────┐
                       │   FastAPI  (api/main.py)        │
                       │   ├── POST /query               │
                       │   ├── POST /ingest/upload       │
                       │   └── GET  /eval/*              │
                       └────────────────┬───────────────┘
                                        │
                       ┌────────────────▼───────────────┐
                       │  orchestration/runner.py         │
                       │  run_pipeline() — drives          │
                       │  graph.astream(), derives          │
                       │  per-agent latencies                │
                       └────────────┬───────────────────┘
                                    │  astream(state)
                       ┌────────────▼───────────────────┐
                       │  StateGraph (orchestration/graph)│
                       │  compiled once, process singleton │
                       └────────────┬───────────────────┘
                                    │
                       ┌────────────▼───────────────────┐
                       │  plan (node)                      │
                       │  PlannerAgent — chat_completion    │
                       │  via core/groq_client               │
                       └────────────┬───────────────────┘
                                    │  Send() × N subtasks
                       ┌────────────▼───────────────────┐
                       │  retrieve_analyze (node, × N, ∥)  │
                       │  RetrieverAgent: BM25+Dense+RRF   │
                       │  +Rerank+Confidence                 │
                       │  AnalystAgent: KPI extraction,      │
                       │  cited claims                        │
                       └────────────┬───────────────────┘
                                    │  subtask_results (reducer)
                    ┌───────────────┴───────────────┐
                    │      superstep — parallel      │
          ┌─────────▼─────────┐         ┌───────────▼──────────┐
          │  audit (node)       │         │  compare (node)         │
          │  AuditorAgent —      │         │  ComparatorAgent —       │
          │  batch LLM            │         │  cross-doc synthesis,     │
          │  entailment →          │         │  delta analysis,           │
          │  VERIFIED /              │         │  anomaly flags,              │
          │  UNCERTAIN /              │         │  multi-source citations      │
          │  UNVERIFIABLE            │         └───────────┬──────────┘
          │  (blocked)                │                     │
          └─────────┬─────────┘                     │
                    └───────────────┬─────────────────┘
                                    │  verified + uncertain + comparison
                       ┌────────────▼───────────────────┐
                       │  synthesize (node)                │
                       │  SynthesizerAgent —                 │
                       │  chat_completion_hedged             │
                       │  → Structured report                 │
                       └────────────┬───────────────────┘
                                    │
                        ┌───────────▼────────────────┐
                        │  AnalysisReport  +  AuditLog  (JSON)      │
                        └───────────────────────────────────────────┘
```

---

## Orchestration: LangGraph `StateGraph`

### Why It's the Orchestration Layer

The pipeline shape never varies with the query: plan → (retrieve + analyse per subtask, in parallel) → audit ∥ compare → synthesise. Only the *width* of the fan-out (how many subtasks) is dynamic. LangGraph's `Send` API expresses exactly that — a fixed topology with a data-dependent number of parallel branches — without hand-rolling `asyncio.gather` and a manual join. See [DECISIONS.md](DECISIONS.md) Decision 11 for the full reasoning and what this replaced (Semantic Kernel).

The four computational agents — Retriever, Analyst, Auditor, Comparator — are plain async functions called directly from graph nodes (`orchestration/graph.py`). The Planner and Synthesizer are prompt-only roles: `agents/router.py::plan_task` and `agents/synthesizer.py::synthesize_report` call `core/groq_client.chat_completion` / `chat_completion_hedged` directly (retry / reserve-endpoint / hedging path) rather than being graph nodes with side effects of their own.

### Graph Singleton

`orchestration/graph.py::build_graph()` constructs and compiles a single `StateGraph` at first use; `get_graph()` returns the cached instance for the lifetime of the process.

```
orchestration/graph.py
  build_graph()
    ├── add_node("plan", plan_node)                    # PlannerAgent
    ├── add_node("retrieve_analyze", retrieve_analyze_node)  # Retriever + Analyst
    ├── add_node("audit", audit_node)                  # AuditorAgent
    ├── add_node("compare", compare_node)               # ComparatorAgent
    ├── add_node("synthesize", synthesize_node)         # SynthesizerAgent
    ├── add_edge(START, "plan")
    ├── add_conditional_edges("plan", fan_out, ["retrieve_analyze"])  # Send × N
    ├── add_edge("retrieve_analyze", "audit")
    ├── add_edge("retrieve_analyze", "compare")          # audit ∥ compare superstep
    ├── add_edge("audit", "synthesize")
    ├── add_edge("compare", "synthesize")
    └── add_edge("synthesize", END)
```

### Nodes

| Node | File | What It Does |
|---|---|---|
| `plan` | `orchestration/graph.py::plan_node` | Calls `agents/router.py::plan_task` — Groq decomposes the query into 2–6 subtasks |
| `retrieve_analyze` | `orchestration/graph.py::retrieve_analyze_node` | One instance per subtask via `Send`; calls `agents/retriever.py` then `agents/analyst.py`; never raises — a failing subtask contributes an `errors` entry and nothing else |
| `audit` | `orchestration/graph.py::audit_node` | Calls `agents/auditor.py::audit_claims` over every extracted claim in one batch call; deliberately does not catch exceptions — an auditor failure aborts the whole run |
| `compare` | `orchestration/graph.py::compare_node` | Calls `agents/comparator.py::compare_results` over unit-normalised subtask results; catches exceptions and degrades to an empty comparison rather than sinking a run with verified claims |
| `synthesize` | `orchestration/graph.py::synthesize_node` | Calls `agents/synthesizer.py::synthesize_report` |

### `FinSightState`

State is a typed `TypedDict` (`orchestration/graph.py::FinSightState`), not a JSON string re-parsed at each hop. Fan-out fields (`subtask_results`, `errors`) use `Annotated[list[...], operator.add]` reducers so LangGraph merges the parallel `retrieve_analyze` branches automatically:

```
query, company_filter, fiscal_year_filter, confidence_threshold, task_id, streaming   (inputs)
    │
plan_node → subtasks: list[str]
    │
retrieve_analyze_node (× N, merged via operator.add) → subtask_results: list[SubtaskResult], errors: list[dict]
    │
audit_node → verified, uncertain, unverifiable
compare_node → comparison                                    (same superstep)
    │
synthesize_node → summary
```

Citations and computed data are structural fields on typed dataclasses/TypedDicts throughout — never serialised to JSON and re-parsed between hops.

### Streaming

Nodes emit progress via LangGraph's custom stream channel (`langgraph.config.get_stream_writer`), wrapped in `orchestration/graph.py::_emit` as a best-effort no-op when nobody is streaming. `orchestration/runner.py::run_pipeline` consumes `astream(state, stream_mode=["values", "updates", "custom"])`, forwards `custom` events to an optional callback (the SSE route), and derives per-agent latency from `updates` events — the graph layer never imports `api`, so it has no HTTP awareness.

---

## Identity and Memory

Two nodes bracket the pipeline: `recall_memory` (before `plan`) and `remember` (after `synthesize`), backed by `memory/store.py::MemoryService`. Both are best-effort — a Qdrant outage degrades to no recalled context / no write, never a failed query. See [DECISIONS.md](DECISIONS.md) Decision 12 for the full reasoning.

**Identity.** `api/middleware/auth.py::AuthMiddleware` (pure ASGI, header-only) resolves `Authorization: Bearer <key>` to a `user_id` via `settings.api_keys` (`"key:user_id,..."`). Unconfigured (the default), every request is `user_id="anonymous"` — local dev, the eval harness, and CI need no setup. This is identity for scoping memory, not an account system: no RBAC, no signup, one flat key→id mapping.

**Storage.** One new Qdrant collection, `finsight_memory`, holding `MemoryRecord` points (`core/models.py`): `memory_id, user_id, session_id, task_id, text, timestamp`. `retrieval/qdrant_store.py::QdrantStore` gained two additive constructor parameters (`collection`, `vector_size`) rather than a second client class, so every existing retrieval call site is unaffected. One record is written per completed turn — there is no separate LLM-based "consolidation" step, deliberately: a compliance system's memory should never be a new synthesis of what happened, only a record of it.

**Recall — two different queries against the same records:**

| | Short-term (this session) | Long-term (this user, across sessions) |
|---|---|---|
| Filter | `session_id` exact match | `user_id` exact match |
| Ranking | Recency (`timestamp`) | Vector similarity to the current query |
| Limit | `settings.session_max_turns` (6) | `settings.long_term_top_k` (3) |
| Question answered | "What did we just discuss?" | "Has this user asked about this before?" |

**Where memory can and cannot reach.** `memory/consolidate.py::format_memory_context` renders both into one text block, injected only into `PLANNER_PROMPT` (`agents/router.py::plan_task`'s new `memory_context` parameter). `SYNTHESIZER_PROMPT` is untouched. This is the load-bearing constraint: memory can bias what the Planner searches for, but it can never become a claim, because it never enters the claims/citations path the Auditor and Synthesizer operate on. A user's own unreviewed past queries shaping today's search terms is an acceptable UX tradeoff; the same content shaping today's *report* would not be.

**Transparency.** `AuditLog` now carries `user_id` and `session_id`. `GET /sessions/{id}` and `GET /memory` let a caller inspect exactly what's recalled about them (their own records only — a 404, not silent filtering, on someone else's session); `DELETE` on both lets them clear it.

---

## Retrieval Architecture

The retrieval pipeline uses two separate scoring mechanisms combined via rank fusion, followed by neural reranking. Each stage has a distinct role.

```
Query string
    │
    ├─────────────────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         ▼
BM25 (rank-bm25)                                Dense (sentence-transformers)
BM25Okapi over Qdrant payload text              384-dim cosine similarity
Exact match: "20.7%", "FY2024", "INR"          Semantic: "profitability trend"
Rank list: [chunk_id → BM25 score]             Rank list: [chunk_id → cosine score]
    │                                                         │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              Reciprocal Rank Fusion (k=60)
              retrieval/hybrid.py
              Score = Σ 1 / (k + rank_i)
              Scale-invariant — avoids BM25 vs cosine magnitude mismatch
                         │
                         ▼
              Cross-encoder reranker
              ms-marco-MiniLM-L6-v2
              Joint (query, passage) attention: O(n) per candidate
              Replaces stage-1 scores entirely — not a rescore weight
                         │
                         ▼
              Confidence scoring (5 signals)
              retrieval/confidence.py
                         │
                         ▼
              RankedChunk list with confidence, section_type, citations
```

### Why Two Retrieval Stages

**Stage 1 — bi-encoder** (BM25 + dense): Produces a candidate pool in O(1) per query. Fast, but bi-encoder scores are not calibrated — each passage is scored independently, so the relative scores between passages are unreliable.

**Stage 2 — cross-encoder**: The cross-encoder sees the full `(query, passage)` pair jointly and produces a calibrated relevance score. This is slower (O(n) per candidate) but substantially more accurate. The cross-encoder's output replaces stage-1 scores entirely; it is not blended.

Running the cross-encoder over the entire corpus would be prohibitively slow. Restricting it to the top-N candidate pool from stage 1 achieves cross-encoder accuracy at bi-encoder speed.

### Reciprocal Rank Fusion

BM25 scores are in different units than cosine similarity. A naive weighted average would be dominated by whichever scorer produces larger absolute values, which varies by query and corpus.

RRF converts both to rank lists and fuses them:

```
score(chunk) = Σ_ranker  1 / (k + rank_i)
```

`k=60` is a smoothing constant that dampens the advantage of rank-1 vs rank-2. The result is a unified rank list with no dependence on original score magnitudes.

---

## Confidence Scoring

`retrieval/confidence.py` computes a composite confidence score from five signals.

| Signal | Weight | Source |
|---|---|---|
| Retrieval score | 35% | Cross-encoder (stage 2) output |
| Section type | 25% | `metadata.py` regex detection |
| Freshness | 15% | Fiscal year proximity to current year |
| Cross-filing consistency | 15% | Same fact in multiple documents |
| Retrieval consistency | 10% | BM25 + dense both ranked it highly |

**Section type weights:**

| Section | Weight | Rationale |
|---|---|---|
| `audited_financials` | 1.00 | Highest — externally audited numbers |
| `notes` | 0.85 | Part of the audited statements; note-specific detail |
| `mda` | 0.65 | Management narrative; unaudited |
| `unknown` | 0.50 | No section detected |
| `letter` | 0.40 | Qualitative; promotional tone possible |

*(Values as implemented in `ingestion/metadata.py::section_type_confidence_weight`.)*

The composite confidence feeds directly into the AuditorAgent's three-tier classification.

---

## AuditorAgent Design

The AuditorAgent (`agents/auditor.py`) is a structural verification pass, not a prompt guardrail.

### Why This Matters

A prompt instruction like "only use information from the provided context" can be overridden by sufficiently confident model responses or adversarial prompts. The AuditorAgent cannot be bypassed this way because it is a separate LLM call that sees each claim individually with its supporting snippet and determines whether the snippet entails the claim.

### Three-Tier Classification

```
For each claim:
    if no snippet provided:
        → UNVERIFIABLE (blocked)
    elif snippet entails claim AND confidence ≥ threshold:
        → VERIFIED
    elif snippet weakly supports claim OR confidence < threshold:
        → UNCERTAIN
    elif snippet contradicts claim:
        → UNVERIFIABLE (blocked)
```

**VERIFIED** claims proceed to SynthesizerAgent.
**UNCERTAIN** claims proceed to SynthesizerAgent with a lower-confidence flag, clearly labelled in the report.
**UNVERIFIABLE** claims are blocked. They appear in the audit log under `blocked_unverifiable` but never in the user-visible report.

Default confidence threshold: `0.65` (configurable via `.env`).

---

## Ingestion Pipeline

```
PDF file (PyMuPDF)
    │
    ▼ ingestion/parser.py
    Page text + bounding boxes + page number
    │
    ▼ ingestion/metadata.py
    Company detection (regex on first 3 pages)
    Fiscal year extraction (FY2024 / 2023–24 / March 2022)
    Section type detection (heading regex per chunk)
    │
    ▼ ingestion/chunker.py
    Heading-aware sliding window: target 400 tokens, overlap 80 tokens
    SHA-256 deduplication: skip chunk if hash already in collection
    Chunk ID = sha256(company + fiscal_year + page + text)
    │
    ▼ retrieval/qdrant_store.py
    Upsert to Qdrant: vector (384-dim) + full payload
    Payload: text, page, chunk_id, company, fiscal_year, section_type, doc_id
```

### Section Type Detection

`ingestion/metadata.py` uses heading-proximity regex to classify each chunk. The heading must appear within the first two lines of the chunk or as a standalone page heading.

| Pattern examples | Classified as |
|---|---|
| `Independent Auditors' Report`, `Consolidated Balance Sheet` | `audited_financials` |
| `Management Discussion and Analysis`, `MD&A` | `mda` |
| `Notes to Financial Statements`, `Note [0-9]` | `notes` |
| `Dear Shareholders`, `Chairman's Message` | `letter` |

Correct section classification is worth 25% of the composite confidence score — the highest non-retrieval signal. A fact from an audited financial statement deserves structurally higher confidence than the same number mentioned in a management letter.

---

## Data Models

All shared types are defined in `core/models.py`.

### Citation

The Citation is the fundamental unit of traceability. Every claim carries one.

```python
@dataclass
class Citation:
    document: str        # filename of the source PDF
    page: int            # 1-indexed page number
    snippet: str         # verbatim passage (≤ 500 chars) supporting the claim
    claim: str           # the claim this citation supports
    confidence: float    # composite 5-signal score [0, 1]
    section_type: str    # audited_financials | mda | notes | letter | unknown
```

### AuditedClaim

```python
@dataclass
class AuditedClaim:
    claim: str
    citation: Citation
    audit_status: AuditStatus    # VERIFIED | UNCERTAIN | UNVERIFIABLE
    audit_reason: str            # one sentence from AuditorAgent
```

### Chunk and RankedChunk

```python
@dataclass
class Chunk:
    chunk_id: str        # SHA-256 of (company, fiscal_year, page, text)
    text: str
    page: int
    company: str
    fiscal_year: str
    section_type: str
    doc_id: str

@dataclass
class RankedChunk:
    chunk: Chunk
    score: float         # cross-encoder output after stage 2
    confidence: float    # 5-signal composite
```

### AuditLog

Written to `audit_logs/<task_id>.json` after every query run.

```python
@dataclass
class AuditLog:
    task_id: str
    timestamp: str                    # ISO 8601 UTC
    user_query: str
    plan: list[str]                   # subtasks produced by PlannerAgent
    retrievals: dict[str, list[str]]  # subtask → list of chunk_ids retrieved
    claims: list[dict]                # all non-blocked claims with citations
    flagged_uncertain: list[str]      # claims that reached UNCERTAIN
    blocked_unverifiable: list[str]   # claims that were blocked
    agents_invoked: list[str]         # ordered list of agents invoked
    latency_ms: int
```

---

## API Layer

`api/main.py` configures the FastAPI application with a lifespan context manager that starts Qdrant and compiles the LangGraph `StateGraph` singleton at startup.

### Guardrails Middleware

`api/middleware/guardrails.py` scans every incoming request body for prompt injection patterns before the route handler runs. Detected patterns are logged and the request is rejected with HTTP 400.

Common patterns detected:
- `ignore previous instructions`
- `disregard the above`
- `pretend you are`
- Direct injection via angle brackets or JSON escaping

### Routes

| Route | Handler | Notes |
|---|---|---|
| `POST /query` | `api/routes/query.py` | Full 6-agent pipeline, blocking |
| `POST /query/stream` | `api/routes/query_stream.py` | Same pipeline, SSE progress events (`start … done`/`error`) |
| `POST /ingest/upload` | `api/routes/ingest.py` | Multipart PDF, triggers ingestion pipeline |
| `GET /eval/collection` | `api/routes/eval.py` | Qdrant vector count and status |
| `GET /eval/audit-logs` | `api/routes/eval.py` | Lists saved audit log files |
| `GET /eval/audit-logs/{id}` | `api/routes/eval.py` | Returns full audit log JSON |
| `GET /metrics` | `api/routes/metrics.py` | `MetricsStore` snapshot (latency percentiles, per-agent timings, claim rates) |
| `GET /health` | `api/main.py` | Liveness probe |

---

## Observability

`observability/tracer.py` configures `structlog` with structured JSON output and sets up the OpenTelemetry tracer provider. All agent hops use the `@traced` decorator which records span start/end, name, and any exception.

Log output is JSON per line with fields: `timestamp`, `level`, `event`, `task_id`, `agent`, `latency_ms`, and agent-specific fields.

Optional: set `OTEL_EXPORTER_OTLP_ENDPOINT` in `.env` to export traces to a Jaeger or Grafana Tempo instance.

---

## Infrastructure

### Qdrant

Single collection: `finsight_chunks` (configurable via `QDRANT_COLLECTION`). Vector size: 384 (matching `all-MiniLM-L6-v2`). Distance: Cosine.

Payload fields indexed for filtering:
- `company` (keyword)
- `fiscal_year` (keyword)
- `section_type` (keyword)
- `doc_id` (keyword)

Qdrant runs in Docker via `docker-compose.yml`. The API container declares a dependency with healthcheck so Qdrant is ready before the API starts accepting requests.

### Container Networking

| Container | Internal hostname | Port |
|---|---|---|
| Qdrant | `qdrant` | 6333 |
| API | `api` | 8000 |

`QDRANT_HOST=qdrant` in the API container's environment ensures the async Qdrant client uses the Docker service name, not `localhost`.

---

## Directory Structure

```
finsight/
├── orchestration/
│   ├── graph.py            # StateGraph — nodes, Send fan-out, FinSightState, compiled singleton
│   └── runner.py           # run_pipeline() — astream() driver, per-agent latency accounting
├── memory/
│   ├── store.py            # MemoryService — session recall/write, long-term recall (Qdrant)
│   └── consolidate.py      # build_turn_text(), format_memory_context() — pure, no LLM call
├── agents/
│   ├── router.py          # plan_task() — PlannerAgent, chat_completion (not a graph node)
│   ├── retriever.py       # RetrieverAgent — retrieve(), called from retrieve_analyze_node
│   ├── analyst.py         # AnalystAgent — analyze_chunks(), called from retrieve_analyze_node
│   ├── comparator.py      # ComparatorAgent — compare_results(), called from compare_node
│   ├── auditor.py         # AuditorAgent — audit_claims(), called from audit_node
│   └── synthesizer.py     # synthesize_report() — chat_completion_hedged, called from synthesize_node
├── retrieval/
│   ├── qdrant_store.py    # QdrantStore: ensure_collection, upsert, dense_search, scroll
│   ├── embedder.py        # SentenceTransformer local wrapper
│   ├── bm25.py            # BM25Okapi built over Qdrant payload scroll
│   ├── hybrid.py          # reciprocal_rank_fusion(ranked_lists, k=60)
│   ├── reranker.py        # CrossEncoder rescoring
│   └── confidence.py      # compute_confidence() — 5-signal composite
├── ingestion/
│   ├── parser.py          # extract_pages() via PyMuPDF
│   ├── chunker.py         # chunk_document(), _deduplicate()
│   ├── metadata.py        # detect_section_type(), detect_fiscal_year(), detect_company()
│   └── pipeline.py        # ingest_file(), ingest_directory()
├── api/
│   ├── main.py            # FastAPI app, lifespan, /health
│   ├── routes/
│   │   ├── query.py       # POST /query — run_pipeline() over the compiled graph, blocking
│   │   ├── query_stream.py# POST /query/stream — same graph, SSE progress events
│   │   ├── ingest.py      # POST /ingest/upload
│   │   ├── eval.py        # GET /eval/*
│   │   ├── sessions.py    # GET/DELETE /sessions/{id} — caller's own turns only
│   │   └── memory.py      # GET/DELETE /memory — caller's own long-term records
│   └── middleware/
│       ├── auth.py        # API key → request.state.user_id (pure ASGI)
│       └── guardrails.py  # prompt injection detection middleware
├── core/
│   ├── prompts.py         # Planner/Synthesizer prompt templates
│   ├── models.py          # Citation, Chunk, RankedChunk, AuditedClaim, AuditLog
│   ├── config.py          # Settings (pydantic-settings, .env)
│   └── groq_client.py     # chat_completion() / chat_completion_hedged() async wrapper
├── evaluation/
│   ├── harness.py         # run_harness(query_file) — async test runner
│   └── queries/
│       ├── happy_path.json     # 4 verifiable queries against seed corpus
│       └── adversarial.json    # 4 hallucination-eliciting queries
├── observability/
│   └── tracer.py          # setup_tracing(), @traced decorator
├── tests/                 # pytest unit tests
├── docs/
│   ├── README.md          # documentation index
│   ├── HLD.md             # high-level design
│   ├── LLD.md             # low-level design
│   ├── ARCHITECTURE.md    # this file
│   ├── DECISIONS.md       # design decisions with reasoning
│   └── RUNBOOK.md         # operations runbook
├── data/
│   └── filings/           # seed PDFs placed here (git-ignored)
├── audit_logs/            # per-run JSON artifacts (git-ignored)
├── docker-compose.yml     # Qdrant + API services
├── Dockerfile             # multi-stage: uv builder → slim non-root runtime
├── pyproject.toml         # deps + ruff config (line-length=100)
└── .env.example           # all configurable settings
```
