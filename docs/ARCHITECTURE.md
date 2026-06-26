# FinSight — Architecture

This document describes the system design decisions, data flows, component responsibilities, and technical models that make up FinSight. It is the single authoritative reference for anyone contributing to, evaluating, or auditing the system.

---

## Design Goals

The architecture is shaped by a single constraint that separates FinSight from a standard RAG system: **any claim that cannot be verified against a verbatim source passage must be blocked before reaching the user, not warned about after the fact.**

This forces several design decisions that would be unnecessary in a looser system:

- Citations are a structural field, not a cosmetic annotation. Every agent hop carries citations in `KernelArguments` and the pipeline cannot produce output without them.
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
                       │   SK Kernel  (core/sk_kernel)  │
                       │   ├── OpenAIChatCompletion      │
                       │   │   (Groq endpoint)           │
                       │   ├── Native plugins:           │
                       │   │   Retriever, Analyst,       │
                       │   │   Auditor, Comparator       │
                       │   └── Semantic functions:       │
                       │       Planner, Synthesizer      │
                       └────────────┬───────────────────┘
                                    │  kernel.invoke()
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
     ┌───────▼──────┐   ┌───────────▼────────┐   ┌───────▼──────────┐
     │  PlannerAgent│   │  RetrieverAgent     │   │  AnalystAgent    │
     │  (semantic   │   │  (native plugin)    │   │  (native plugin) │
     │  function)   │   │  BM25+Dense+RRF     │   │  KPI extraction  │
     │  decompose() │   │  +Rerank+Confidence │   │  cited claims    │
     └──────────────┘   └──────────┬──────────┘   └──────┬───────────┘
                                   │                      │
                       ┌───────────▼──────────────────────▼──────────┐
                       │         AuditorAgent (native plugin)         │
                       │  LLM entailment per claim → VERIFIED /       │
                       │  UNCERTAIN / UNVERIFIABLE (blocked)          │
                       └───────────────────────┬─────────────────────┘
                                               │
                       ┌───────────────────────▼─────────────────────┐
                       │       ComparatorAgent (native plugin)        │
                       │  Cross-doc synthesis, delta analysis,        │
                       │  anomaly flags, multi-source citations       │
                       └───────────────────────┬─────────────────────┘
                                               │
                       ┌───────────────────────▼─────────────────────┐
                       │    SynthesizerAgent (semantic function)      │
                       │  kernel.invoke("Synthesizer", "synthesize")  │
                       │  → Structured report from verified claims    │
                       └───────────────────────┬─────────────────────┘
                                               │
                        ┌──────────────────────▼────────────────────┐
                        │  AnalysisReport  +  AuditLog  (JSON)      │
                        └───────────────────────────────────────────┘
```

---

## Semantic Kernel Design

### Why SK Is Load-Bearing

Semantic Kernel is not a thin API wrapper in this system. Removing it would require reimplementing plugin registration, context passing (KernelArguments), and prompt template rendering. Every agent hop goes through `kernel.invoke()`.

### Kernel Singleton

`core/sk_kernel.py` builds a single `sk.Kernel` instance at startup. It is reused for the lifetime of the process. Lazy initialization via `_kernel: sk.Kernel | None = None` with `get_kernel()`.

```
core/sk_kernel.py
  _build_kernel()
    ├── kernel.add_service(OpenAIChatCompletion)
    │   └── async_client = AsyncOpenAI(base_url=groq_base_url)
    │                                            ↓
    │                             Groq — Llama 3.3 70B
    ├── kernel.add_plugin(RetrieverPlugin, "Retriever")
    ├── kernel.add_plugin(AnalystPlugin,   "Analyst")
    ├── kernel.add_plugin(AuditorPlugin,   "Auditor")
    ├── kernel.add_plugin(ComparatorPlugin,"Comparator")
    ├── kernel.add_function("Planner",    KernelFunctionFromPrompt(...))
    └── kernel.add_function("Synthesizer",KernelFunctionFromPrompt(...))
```

### Two Types of SK Functions

| Type | Plugin Name | File | Invocation |
|---|---|---|---|
| Semantic function | `Planner` | `core/sk_kernel.py` | `kernel.invoke("Planner", "decompose", KernelArguments(user_task=...))` |
| Semantic function | `Synthesizer` | `core/sk_kernel.py` | `kernel.invoke("Synthesizer", "synthesize", KernelArguments(query=..., verified_claims=..., ...))` |
| Native plugin | `Retriever` | `agents/retriever.py` | `kernel.invoke("Retriever", "retrieve", KernelArguments(subtask=...))` |
| Native plugin | `Analyst` | `agents/analyst.py` | `kernel.invoke("Analyst", "analyze", KernelArguments(subtask=..., chunks_json=...))` |
| Native plugin | `Auditor` | `agents/auditor.py` | `kernel.invoke("Auditor", "audit", KernelArguments(claims_json=..., confidence_threshold=...))` |
| Native plugin | `Comparator` | `agents/comparator.py` | `kernel.invoke("Comparator", "compare", KernelArguments(subtask_results_json=..., original_query=...))` |

**Semantic functions** are `KernelFunctionFromPrompt` instances. SK renders the `{{$variable}}` template, applies token limits, and dispatches to Groq. The prompt itself is the logic — the LLM produces a plan or a report.

**Native plugins** are Python classes with `@kernel_function`-decorated methods. SK calls the Python function directly via `kernel.invoke()`. The function does real work (BM25 search, cross-encoder reranking, entailment checking) and returns a JSON string that threads into the next hop via `KernelArguments`.

### KernelArguments Context Chain

Citations and computed data pass through the pipeline as JSON strings inside `KernelArguments`. They are never lost between hops:

```
user_task (str)
    │ KernelArguments(user_task=...)
    ▼
Planner.decompose → subtasks (list[str])
    │
    │ KernelArguments(subtask=subtask)
    ▼
Retriever.retrieve → chunks_json (JSON str with citations)
    │
    │ KernelArguments(subtask=subtask, chunks_json=chunks_json)
    ▼
Analyst.analyze → analysis_json (JSON str with cited claims)
    │
    │ KernelArguments(claims_json=all_claims_json, confidence_threshold=...)
    ▼
Auditor.audit → audit_json (verified / uncertain / unverifiable)
    │
    │ KernelArguments(subtask_results_json=..., original_query=...)
    ▼
Comparator.compare → comparison_json (deltas, anomalies)
    │
    │ KernelArguments(query=..., verified_claims=..., uncertain_claims=..., comparison=...)
    ▼
Synthesizer.synthesize → final report (str)
```

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
| `audited_financials` | 1.0 | Highest — externally audited numbers |
| `mda` | 0.8 | Management narrative; unaudited |
| `notes` | 0.7 | Supplementary; context-dependent |
| `letter` | 0.5 | Qualitative; promotional tone possible |
| `unknown` | 0.4 | No section detected |

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

`api/main.py` configures the FastAPI application with a lifespan context manager that starts Qdrant and warms the SK kernel at startup.

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
| `POST /query` | `api/routes/query.py` | Full 6-agent pipeline |
| `POST /ingest/upload` | `api/routes/ingest.py` | Multipart PDF, triggers ingestion pipeline |
| `GET /eval/collection` | `api/routes/eval.py` | Qdrant vector count and status |
| `GET /eval/audit-logs` | `api/routes/eval.py` | Lists saved audit log files |
| `GET /eval/audit-logs/{id}` | `api/routes/eval.py` | Returns full audit log JSON |
| `GET /health` | `api/main.py` | Liveness probe |

---

## Observability

`observability/tracer.py` configures `structlog` with structured JSON output and sets up the OpenTelemetry tracer provider. All agent hops use the `@traced` decorator which records span start/end, name, and any exception.

Log output is JSON per line with fields: `timestamp`, `level`, `event`, `task_id`, `agent`, `latency_ms`, and agent-specific fields.

Optional: set `OTEL_EXPORTER_OTLP_ENDPOINT` in `.env` to export traces to a Jaeger or Grafana Tempo instance.

---

## Infrastructure

### Qdrant

Single collection: `financial_filings`. Vector size: 384 (matching `all-MiniLM-L6-v2`). Distance: Cosine.

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
├── agents/
│   ├── router.py          # plan_task() — PlannerAgent via kernel.invoke
│   ├── retriever.py       # RetrieverPlugin — @kernel_function retrieve()
│   ├── analyst.py         # AnalystPlugin  — @kernel_function analyze()
│   ├── comparator.py      # ComparatorPlugin — @kernel_function compare()
│   ├── auditor.py         # AuditorPlugin  — @kernel_function audit()
│   └── synthesizer.py     # synthesize_report() — via kernel.invoke Synthesizer
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
│   │   ├── query.py       # POST /query — full kernel.invoke pipeline
│   │   ├── ingest.py      # POST /ingest/upload
│   │   └── eval.py        # GET /eval/*
│   └── middleware/
│       └── guardrails.py  # prompt injection detection middleware
├── core/
│   ├── sk_kernel.py       # _build_kernel(), get_kernel() singleton
│   ├── models.py          # Citation, Chunk, RankedChunk, AuditedClaim, AuditLog
│   ├── config.py          # Settings (pydantic-settings, .env)
│   └── groq_client.py     # chat_completion() async wrapper
├── evaluation/
│   ├── harness.py         # run_harness(query_file) — async test runner
│   └── queries/
│       ├── happy_path.json     # 4 verifiable queries against seed corpus
│       └── adversarial.json    # 4 hallucination-eliciting queries
├── observability/
│   └── tracer.py          # setup_tracing(), @traced decorator
├── tests/                 # pytest unit tests
├── docs/
│   ├── ARCHITECTURE.md    # this file
│   └── DECISIONS.md       # design decisions with reasoning
├── data/
│   └── filings/           # seed PDFs placed here (git-ignored)
├── audit_logs/            # per-run JSON artifacts (git-ignored)
├── docker-compose.yml     # Qdrant + API services
├── Dockerfile             # python:3.11-slim, uv install
├── pyproject.toml         # deps + ruff config (line-length=100)
└── .env.example           # all configurable settings
```
