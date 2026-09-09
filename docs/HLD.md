# FinSight — High-Level Design (HLD)

**Status:** current as of the `main` branch
**Audience:** engineers, reviewers, and evaluators who need the system-level picture before reading code or the [Low-Level Design](LLD.md).
**Companion documents:** [ARCHITECTURE.md](ARCHITECTURE.md) (narrative deep-dive), [DECISIONS.md](DECISIONS.md) (rationale log), [LLD.md](LLD.md) (module-level detail).

---

## 1. Purpose and Scope

### 1.1 Problem

An analyst asks a natural-language question about a set of financial filings ("compare Infosys and TCS operating margins FY2024–FY2026 and flag anomalies"). A conventional RAG chatbot answers fluently but gives no usable guarantee that each figure in the answer actually appears in a source document. In a compliance setting that is the wrong failure mode: an unverifiable AI output is worse than no output.

### 1.2 What FinSight does

FinSight is a multi-agent Retrieval-Augmented Generation system that produces a structured analytical report in which:

- every claim is bound to a verbatim source passage (document + page + snippet);
- claims whose snippet does not entail them are **blocked before synthesis**, not flagged afterwards;
- every run emits a machine-readable audit log (plan, retrievals, claims, blocks, agents, latency).

### 1.3 In scope

- Ingestion of digital (non-scanned) financial-filing PDFs.
- Hybrid retrieval (keyword + dense + rerank) over a single vector collection.
- A fixed six-role agent pipeline orchestrated per query.
- Structural hallucination control (the AuditorAgent).
- Deterministic financial-unit normalisation prior to cross-document comparison.
- A FastAPI service exposing blocking and streaming query endpoints, ingestion, metrics, and audit-log retrieval.
- A single-page web UI and a metrics dashboard.

### 1.4 Out of scope

- OCR / scanned-document ingestion.
- Full multi-tenant accounts, RBAC, or login flows. Simple per-user identity (API key → `user_id`) *is* in scope as of F14 — see below — but that's a single flat namespace, not real account/role management.
- Durable/clustered vector storage or horizontal scale-out of the API.
- Real-time market data; the corpus is whatever has been ingested.
- Model fine-tuning — all models are used off-the-shelf.

---

## 2. Stakeholders and Goals

| Stakeholder | Primary concern | How the design serves it |
|---|---|---|
| Compliance analyst (end user) | Can I trust and defend every figure? | Structural citations, three-tier claim status, per-run audit log |
| Reviewer / evaluator | Does it resist hallucination? | Adversarial eval suite; AuditorAgent block path; audit log of blocked claims |
| Operator | Can I run and observe it? | Docker-compose stack, structured logs, `/metrics`, `/health`, optional OTel |
| Contributor | Where do I make a change safely? | Single LLM dispatch point, typed data models, module boundaries (see LLD) |

---

## 3. Requirements

### 3.1 Functional

| ID | Requirement |
|---|---|
| F1 | Accept a natural-language analytical query and return a structured Markdown report. |
| F2 | Decompose each query into 2–6 ordered retrieval subtasks; the plan varies with the query. |
| F3 | Retrieve evidence with hybrid keyword + dense retrieval, rank fusion, and cross-encoder reranking. |
| F4 | Attach a composite confidence score and a section-type label to every retrieved chunk. |
| F5 | Extract only claims stated verbatim in a chunk; never emit a figure that required arithmetic. |
| F6 | Verify each claim against its snippet with a separate LLM entailment pass; classify VERIFIED / UNCERTAIN / UNVERIFIABLE. |
| F7 | Block UNVERIFIABLE claims — they must never reach the synthesised report; record them in the audit log. |
| F8 | Normalise currency/scale (USD, million, lakh, billion → ₹ crore) deterministically in Python before cross-document comparison. |
| F9 | Produce a cross-document comparison (deltas, anomaly flags, multi-source claims). |
| F10 | Persist a JSON audit log per run and expose it over the API. |
| F11 | Ingest a PDF on demand: parse, chunk, embed, upsert; deduplicate re-ingested content. |
| F12 | Stream pipeline progress events over SSE for the interactive UI. |
| F13 | Reject requests whose body matches known prompt-injection patterns. |
| F14 | Resolve an `Authorization: Bearer <key>` header to a `user_id` when `API_KEYS` is configured; otherwise treat every caller as `user_id="anonymous"`. |
| F15 | Recall the current session's recent turns (short-term memory) and inject them into the Planner prompt so follow-up queries resolve against prior context. |
| F16 | Recall a user's semantically relevant past turns across sessions (long-term memory) and inject them into the Planner prompt only — never the Synthesizer, so memory can shape what gets searched for but can never become an unverified claim. |

### 3.2 Non-functional

| ID | Requirement | Design response |
|---|---|---|
| N1 Traceability | Every user-visible figure resolves to document + page + snippet. | `Citation` is a mandatory field threaded through every hop. |
| N2 Auditability | A third party can reconstruct what happened in a run. | `AuditLog` dataclass persisted to `audit_logs/<task_id>.json`. |
| N3 Resilience | A single slow/failed LLM endpoint must not fail the request. | Retry-with-backoff, reserve endpoint, hedged calls on the streaming path. |
| N4 Latency | Interactive use; correctness outranks speed. | Subtasks run in parallel; auditor/comparator run concurrently on the stream path; BM25 corpus cached per process. |
| N5 Observability | Per-agent latency and error rate visible without a debugger. | `structlog` structured events, in-process `MetricsStore`, `/metrics`, `/dashboard`, optional OTel spans. |
| N6 Determinism where it matters | Arithmetic must not be delegated to an LLM. | `core/unit_normalizer.py` does all currency/scale math with `Decimal`. |
| N7 Portability | Runs on a laptop with no paid accounts beyond one free LLM key. | Local embedding + reranker models; Qdrant in Docker; Groq free tier. |

---

## 4. System Context

```
        ┌──────────────┐        HTTPS / SSE        ┌────────────────────────────┐
        │  Analyst      │ ───────────────────────▶ │        FinSight API         │
        │  (browser /   │ ◀─────────────────────── │      (FastAPI, one proc)    │
        │   curl)       │   report + audit log     └─────────────┬──────────────┘
        └──────────────┘                                         │
                                                                 │ async
                    ┌────────────────────────────────────────────┼───────────────────────┐
                    │                                            │                       │
                    ▼                                            ▼                       ▼
        ┌────────────────────┐                      ┌──────────────────────┐   ┌────────────────────┐
        │  Groq LLM endpoint │                      │  Qdrant (Docker)     │   │ Local ML models    │
        │  (OpenAI-compat)   │                      │  vector + payload    │   │ MiniLM emb/rerank  │
        │  + optional reserve│                      │  store               │   │ (in-process)       │
        └────────────────────┘                      └──────────────────────┘   └────────────────────┘
```

External dependencies:

| Dependency | Role | Failure behaviour |
|---|---|---|
| Groq (Llama 3.3 70B) via OpenAI-compatible endpoint | All LLM reasoning (plan, analyse, audit, compare, synthesise) | Retry ×3 with backoff → reserve endpoint (if configured) → `RuntimeError` surfaced as HTTP 500 / SSE `error` |
| Optional reserve LLM (any OpenAI-compatible URL) | Take over on primary rate-limit / 503 / timeout; hedge partner on the stream path | If also fails: `"Both LLM endpoints failed"` |
| Qdrant | Vector search + payload storage + filtered scroll | Empty collection → empty retrieval → "insufficient evidence" report; connection error → subtask yields no chunks |
| HuggingFace Hub (first run only) | Download `all-MiniLM-L6-v2` (~90 MB) and `ms-marco-MiniLM-L6-v2` (~67 MB) | Cached in the Docker layer / local cache afterwards |

---

## 5. Architecture Overview

### 5.1 Layered view

```
┌───────────────────────────────────────────────────────────────────────┐
│ Interface layer            api/main.py · routes/* · static/*           │
│   POST /query  POST /query/stream  POST /ingest/upload                 │
│   GET /metrics /eval/* /health  GET/DELETE /sessions /memory           │
│   ASGI AuthMiddleware (API-key → user_id) · GuardrailsMiddleware       │
├───────────────────────────────────────────────────────────────────────┤
│ Orchestration layer        orchestration/graph.py · runner.py          │
│   LangGraph StateGraph: recall_memory → plan → Send×N retrieve_analyze │
│   → audit ∥ compare (superstep) → synthesize → remember; runner drives │
│   astream()                                                             │
├───────────────────────────────────────────────────────────────────────┤
│ Agent layer                agents/*  ·  memory/*                       │
│   Planner · Retriever · Analyst · Auditor · Comparator · Synthesizer   │
│   MemoryService (session recall/write, long-term recall)               │
├───────────────────────────────────────────────────────────────────────┤
│ Capability layer                                                       │
│   retrieval/*  (bm25, embedder, hybrid RRF, reranker, confidence,      │
│                 qdrant_store — also backs memory/store.py)             │
│   ingestion/*  (parser, chunker, metadata, pipeline)                   │
│   core/unit_normalizer.py  (deterministic currency/scale math)         │
├───────────────────────────────────────────────────────────────────────┤
│ Foundation layer           core/*  ·  observability/*                  │
│   prompts (Planner/Synthesizer templates) · groq_client (LLM + reserve │
│   + hedging) · config (pydantic-settings) · models (dataclasses)       │
│   tracer (structlog / OTel) · api/metrics_store (in-proc metrics)      │
└───────────────────────────────────────────────────────────────────────┘
```

### 5.2 The six agent roles

| # | Agent | Kind | Input → Output | Responsibility |
|---|---|---|---|---|
| 1 | **PlannerAgent** | Prompt (`PLANNER_PROMPT`) | query → `list[str]` subtasks | Decompose the query into 2–6 searchable subtasks. |
| 2 | **RetrieverAgent** | Graph node (`retrieve_analyze`) | subtask → ranked chunks | BM25 + dense search → RRF fusion → cross-encoder rerank → 5-signal confidence. |
| 3 | **AnalystAgent** | Graph node (`retrieve_analyze`, LLM inside) | subtask + chunks → claims + KPIs | Extract verbatim-stated claims with supporting text; refuse derived figures. |
| 4 | **AuditorAgent** | Graph node (`audit`, LLM inside) | all claims + threshold + query → verified / uncertain / unverifiable | Per-claim entailment + fabricated-event check; classify and block. |
| 5 | **ComparatorAgent** | Graph node (`compare`, LLM inside) | normalised subtask results + query → deltas / anomalies / cross-doc claims | Place verbatim figures side by side; flag material differences; no arithmetic. |
| 6 | **SynthesizerAgent** | Graph node (`synthesize`), prompt (`SYNTHESIZER_PROMPT`) | query + verified + uncertain + comparison → Markdown report | Assemble the report; reproduce only verbatim / pipeline-labelled figures. |

> **Note on orchestration.** `orchestration/graph.py` is a single compiled LangGraph `StateGraph`, a process-wide singleton. Retriever, Analyst, Auditor, and Comparator are plain async functions called directly from graph nodes — there is no dispatch layer between the node and the agent function. Planner and Synthesizer remain prompt-only roles: `agents/router.py::plan_task` and `agents/synthesizer.py::synthesize_report` call `core/groq_client.chat_completion` / `chat_completion_hedged` directly, keeping those two calls on the retry/reserve/hedge path. This replaced a Semantic Kernel dispatch layer; see [DECISIONS.md](DECISIONS.md) Decision 11.

> **Note on memory.** Two additional best-effort nodes bracket the pipeline: `recall_memory` (before `plan`) reads short-term (`session_id`-scoped, recency) and long-term (`user_id`-scoped, vector-similarity) turns via `memory/store.py::MemoryService` and folds them into the Planner prompt only; `remember` (after `synthesize`) writes the completed turn. Neither ever reaches the Synthesizer — memory can shape what gets searched for, never what gets asserted as a claim. See [DECISIONS.md](DECISIONS.md) Decision 12.

### 5.3 Request flow — `POST /query` (blocking)

```
client ──▶ GuardrailsMiddleware ──▶ query.run_query ──▶ runner.run_pipeline(graph)
                                       │
   1. plan node          PlannerAgent — chat_completion(PLANNER_PROMPT) → subtasks[]
                                       │
   2. retrieve_analyze node, one per subtask via Send(), run in parallel:
        RetrieverAgent   retrieve_chunks(...)                          → ranked chunks
        AnalystAgent     analyze_chunks(...)                           → {kpis, claims}
                                       │  merged into subtask_results (reducer)
   3. audit ∥ compare — one superstep, neither depends on the other:
        audit node       AuditorAgent — audit_claims(claims, threshold, → {verified,
                            original_query)                                uncertain,
                                                                             unverifiable}
        compare node      normalize_subtask_results() (deterministic ₹-crore)
                            ComparatorAgent — compare_results(...)      → {deltas, …}
                                       │
   4. synthesize node    SynthesizerAgent — synthesize_report()        → Markdown
                          (chat_completion_hedged(SYNTH))
                                       │
   5. build AuditLog → write audit_logs/<task_id>.json
      build AnalysisReport → metrics.record_complete → return JSON
```

### 5.4 Request flow — `POST /query/stream` (SSE)

Same graph, driven through the same `run_pipeline`, with these differences:

- The `on_event` callback forwards each custom-stream event straight into the SSE response instead of being collected for a single JSON return.
- Events emitted in order: `start → planned → retrieved* → analyzed* → audited → compared → done` (or `error`).
- **Auditor and Comparator already run concurrently on both routes** — they are one LangGraph superstep, not something the stream route adds; wall-clock is `max`, not `sum`, on `/query` as well as `/query/stream` (see [DECISIONS.md](DECISIONS.md) Decision 11 tradeoffs).

---

## 6. Data Design (high level)

All shared types are frozen-ish dataclasses in `core/models.py`. Full field semantics: [LLD.md §4](LLD.md#4-data-models).

| Type | Meaning | Lifetime |
|---|---|---|
| `Chunk` | One indexed passage + provenance (`chunk_id`, `doc_id`, `source`, `page`, `section`, `section_type`, `token_count`, `fiscal_year`, `company`). | Persisted as Qdrant payload. |
| `RankedChunk` | `Chunk` + `retrieval_score` (cross-encoder) + `confidence_score` (5-signal composite). | Per query, in memory. |
| `Citation` | `document`, `page`, `snippet` (≤500 chars), `claim`, `confidence`, `section_type`. | Embedded in every claim and in the audit log. |
| `AuditedClaim` | `claim`, `citation`, `audit_status` (enum), `audit_reason`. | Per query; verified+uncertain reach the report. |
| `AuditLog` | `task_id`, `timestamp`, `user_query`, `plan`, `retrievals` (subtask→chunk_ids), `claims`, `flagged_uncertain`, `blocked_unverifiable`, `agents_invoked`, `latency_ms`. | Persisted per run. |
| `AnalysisReport` | `task_id`, `query`, `summary` (Markdown), `verified_claims`, `uncertain_claims`, `audit_log`. | API response body. |

**Vector store:** one Qdrant collection (`finsight_chunks` by default), 384-dim vectors, cosine distance. Payload carries the full `Chunk`. Filterable fields: `company`, `fiscal_year`, `section_type`, `doc_id`.

**Audit logs:** newline-free pretty JSON, one file per `task_id`, under `AUDIT_LOG_DIR` (default `audit_logs/`, git-ignored, volume-mounted in Docker).

---

## 7. Key Design Decisions (index)

Full context in [DECISIONS.md](DECISIONS.md). The load-bearing ones:

| # | Decision | One-line rationale |
|---|---|---|
| D1–D3 | *(superseded by D11)* Semantic Kernel over LangGraph, `kernel.invoke()` at every hop, native plugins vs. prompt functions | Historical — see D11. |
| D3 | Retriever/Analyst/Auditor/Comparator as computational graph nodes; Planner/Synthesizer as prompt-only `chat_completion` calls | Match the tool to the work — real computation vs. pure text-in/text-out; the split survives D11, only the framework changed. |
| D4 | Reciprocal Rank Fusion (k=60), not weighted score blend | Scale-invariant across BM25 vs. cosine. |
| D5 | Two-stage retrieval (bi-encoder → cross-encoder) | Cross-encoder accuracy at bi-encoder speed via a small candidate pool. |
| D6 | AuditorAgent as a separate structural pass | A prompt instruction competes with model priors; a separate call does not. |
| D7 | Three-tier confidence at 0.65 / 0.50 | Binary pass/fail discards signal a compliance analyst needs. |
| D8 | Heading-aware sliding-window chunker (400 / 80 tokens) | Respect the strong heading structure of filings; overlap keeps boundary figures retrievable. |
| D9 | Qdrant over Chroma / managed | Async client, indexed payload filters, self-contained in Docker. |
| D10 | Groq / Llama 3.3 70B | Free tier, OpenAI-compatible, fast enough that retrieval is the bottleneck. |
| D11 | LangGraph `StateGraph` over Semantic Kernel, reversing D1–D3 | Fixed topology with dynamic `Send` fan-out expresses the real pipeline shape; typed state replaces JSON-string round-tripping; audit ∥ compare becomes an explicit superstep. |
| — | Deterministic unit normaliser before Comparator | The LLM must never do currency/scale arithmetic. |

---

## 8. Cross-Cutting Concerns

### 8.1 Hallucination control (defence in depth)

1. **Retrieval confidence** — 5-signal composite score gates weak evidence.
2. **Analyst extraction rules** — prompt forbids derived figures; `supporting_text` must be verbatim.
3. **Deterministic normalisation** — `unit_normalizer` removes any need for the LLM to convert units.
4. **AuditorAgent** — separate per-claim entailment call; `UNVERIFIABLE` has no code path to the report.
5. **Synthesizer output rules** — reproduce only verbatim or `[converted from …]`-labelled figures; no arithmetic.
6. **Audit log** — blocked claims are recorded so a reviewer can see what was excluded and why.

### 8.2 Resilience

| Failure | Handling |
|---|---|
| LLM rate-limit / 503 | `chat_completion` retries with delays `[1, 2, 4]s`, then reserve endpoint. |
| LLM timeout | No retry — straight to reserve (timeouts rarely recover). |
| LLM slow (stream path) | `chat_completion_hedged` fires reserve after `hedge_after` seconds, takes the winner. |
| Retriever exception on a subtask | Subtask returns `None`; other subtasks proceed. |
| Analyst JSON parse failure | Regex-extract a JSON object; else `{"kpis": [], "claims": []}`. |
| Auditor batch parse failure | Pad/truncate to input length; missing verdicts default to `uncertain`. |
| Empty collection / no chunks | Report states "insufficient evidence"; run still produces an audit log. |

### 8.3 Security

- `AuthMiddleware` (pure ASGI, header-only) resolves `Authorization: Bearer <key>` to a `user_id` via `API_KEYS`; registered after `GuardrailsMiddleware` so it rejects an unauthenticated request (401) before the body is buffered/scanned. Empty `API_KEYS` (default) disables it — every caller is `user_id="anonymous"`. This is identity for scoping memory, not RBAC — see [DECISIONS.md](DECISIONS.md) Decision 12.
- `GuardrailsMiddleware` (pure ASGI) buffers and scans every `POST`/`PUT` body for injection patterns (`ignore … instructions`, `you are now`, `jailbreak`, `<script`, `[system]`, …) → HTTP 400 before any handler runs. Implemented as pure ASGI specifically so it does not break SSE streaming.
- No secrets in the repo — `.env` is git-ignored; `.env.example` documents every key.
- CORS is currently `allow_origins=["*"]` — acceptable for a single-trust-domain demo; tighten before any shared deployment.
- `/sessions/{id}` and `/memory` enforce access control (404, not just filtering) against the caller's `user_id` — but only when `API_KEYS` is configured. Unconfigured, every caller shares the `"anonymous"` identity and can see each other's memory by design (dev/demo mode).

### 8.4 Observability

- **Logs:** `structlog`; `LOG_FORMAT=text` (dev, coloured) or `json` (containers). `@traced` decorator logs start/complete/error + latency for async functions.
- **Metrics:** in-process `MetricsStore` — query count, error rate, latency percentiles + buckets, per-agent latency (avg/p95/min/max), claim verified/uncertain/blocked rates, recent queries and errors. Exposed at `GET /metrics`, visualised at `/dashboard`.
- **Tracing:** optional OpenTelemetry (`OTEL_ENABLED=true`) → OTLP gRPC exporter → any compatible backend.

### 8.5 Configuration

Single `Settings` object (`core/config.py`, pydantic-settings, `.env`-backed). Groups: Groq + reserve LLM, Qdrant, embedding/reranker model ids, chunking (`chunk_size` 400 / `chunk_overlap` 80), retrieval (`retrieval_top_k` 20 / `rerank_top_k` 5), confidence thresholds (0.65 / 0.50), `audit_log_dir`, OTel, logging. Full table: [LLD.md §9](LLD.md#9-configuration-reference).

---

## 9. Deployment View

```
docker-compose.yml
├── qdrant   (qdrant/qdrant)   :6333  volume ./qdrant_storage   TCP healthcheck
└── api      (multi-stage img) :8000  volumes ./data ./audit_logs
             depends_on: qdrant healthy
             LOG_FORMAT=json   json-file log driver  (50 MB × 5 rotation)
```

- **Image:** multi-stage — builder installs deps with `uv`; slim runtime carries no build toolchain; runs as a non-root user; container healthcheck on `/health`.
- **Local dev alternative:** run only `qdrant` in Docker, `uv sync`, `uvicorn api.main:app --reload`.
- **Scale:** single API process (in-process metrics and BM25 cache assume this). Qdrant is single-node. Scaling out is explicitly out of scope.

### 9.1 Runtime topology

| Process | Count | State |
|---|---|---|
| FastAPI / uvicorn | 1 | In-process: compiled `StateGraph` singleton, BM25 corpus cache, `MetricsStore`, loaded ML models. |
| Qdrant | 1 | On-disk vectors + payload under `qdrant_storage/`. |
| LLM endpoint(s) | external | Stateless from FinSight's side. |

---

## 10. Constraints and Assumptions

- **Corpus:** well-formed digital PDFs with a genuine heading hierarchy and font-size contrast (Infosys / TCS / Wipro annual reports are the reference set). Scanned PDFs will chunk poorly.
- **Company detection** is keyword-based (`infosys`, `tcs`, `tata consultancy`, `wipro`) — extending to other issuers means extending `_COMPANY_HINTS`.
- **Exchange rate** for USD→INR normalisation is a fixed `₹84` constant (approx FY2026 average); every converted figure is labelled `[converted from … at ₹84/USD, approx]`.
- **Single fiscal calendar** assumed comparable across issuers (April–March for the reference set).
- **LLM determinism:** all reasoning calls use `temperature=0.0` (synthesis `0.1`); output is still not byte-stable.
- **Free-tier rate limits** on Groq will be hit under load-testing; the reserve endpoint is the mitigation.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation | Residual |
|---|---|---|---|
| Heading detection fails on an atypical PDF | Bad chunk boundaries → weaker retrieval | Font-size ratio + regex; overlap; SHA-256 dedup | Manual spot-check on new issuers |
| LLM ignores the "no derived figures" rule | A computed figure slips into a claim | Analyst + Comparator + Synthesizer prompts all forbid it; Auditor entailment; deterministic normaliser removes the main temptation | Non-zero; audit log allows review |
| Both LLM endpoints down | Request fails | Retry + reserve + hedge | Hard failure surfaced honestly (no fabricated answer) |
| In-process metrics / BM25 cache lost on restart | Metrics reset; first query rebuilds BM25 | Acceptable for single-process design | N/A |
| Fixed FX rate drifts from reality | Converted comparatives are approximate | Every conversion is labelled `approx`; anomaly flag set on any USD-converted delta row | Documented assumption |
| CORS `*` + auth optional | Anyone on the network can query; unconfigured deployments share one "anonymous" identity | Configure `API_KEYS` for real per-user isolation; deploy behind a trusted boundary otherwise | Tighten before shared use |
| Memory outage (Qdrant `finsight_memory` unavailable) | `recall_memory`/`remember` nodes fail | Both are best-effort — catch and degrade to no context / no write | Query still succeeds; that turn just isn't recalled later |
| Long-term memory content is unfiltered LLM-adjacent user history | A user's own past queries shape future plans, unreviewed | Memory only ever reaches the Planner prompt — it never becomes a claim or reaches the Synthesizer | Documented in Decision 12; `/memory` lets a user inspect/delete their own history |

---

## 12. Traceability — requirements to components

| Req | Realised by |
|---|---|
| F1, F12 | `api/routes/query.py`, `api/routes/query_stream.py` |
| F2 | `PLANNER_PROMPT`, `_parse_subtasks`, `agents/router.py` |
| F3, F4 | `agents/retriever.py`, `retrieval/{bm25,embedder,hybrid,reranker,confidence}.py`, `retrieval/qdrant_store.py` |
| F5 | `agents/analyst.py` (`_SYSTEM` extraction rules) |
| F6, F7 | `agents/auditor.py` (`_BATCH_SYSTEM`, `audit_claims`) |
| F8 | `core/unit_normalizer.py` |
| F9 | `agents/comparator.py` |
| F10 | `core/models.py::AuditLog`, `query.py::_save_audit_log`, `api/routes/eval.py` |
| F11 | `ingestion/pipeline.py`, `ingestion/{parser,chunker,metadata}.py`, `api/routes/ingest.py` |
| F13 | `api/middleware/guardrails.py` |
| N1–N2 | `core/models.py`, audit-log persistence |
| N3 | `core/groq_client.py` (retry / reserve / hedge) |
| N5 | `observability/tracer.py`, `api/metrics_store.py`, `/metrics`, `/dashboard` |
| N6 | `core/unit_normalizer.py` |
