# FinSight

**Regulatory-Grade Financial Intelligence System**

> A multi-agent RAG system built for compliance: every claim is traceable to a verbatim source passage, hallucinations trigger a hard block rather than a warning, and every analysis run produces a machine-readable audit log. Built for environments where an unverifiable AI output is worse than no output.

---

## Why This Is Not a RAG Chatbot

The standard GenAI portfolio project in 2026 is a RAG chatbot over PDFs. FinSight is designed around a different question: *what would a GenAI system need to look like for a compliance team to actually trust it?*

| Property | Typical RAG chatbot | FinSight |
|---|---|---|
| Citations | Added at the end, cosmetic | Structural — pipeline cannot produce output without them |
| Hallucination handling | Prompt instruction ("only use context") | Separate AuditorAgent performs LLM entailment checking; UNVERIFIABLE claims are blocked before synthesis |
| Task decomposition | Fixed prompt or hardcoded chain | SK Stepwise Planner — the plan changes with the query |
| Audit trail | None | Per-run JSON artifact: every claim, every retrieval, every agent invoked |
| Cross-document reasoning | Single-document or naive merge | ComparatorAgent synthesises multi-source claims with multi-source citations |
| Confidence | Binary (answer / no answer) | Five-signal composite score; three-tier status (VERIFIED / UNCERTAIN / UNVERIFIABLE) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Query (NL)                       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  PlannerAgent  ·  SK semantic function                  │
│  → Groq decomposes query into 2–6 ordered subtasks      │
└─────────────────────────┬───────────────────────────────┘
                          │  per subtask (parallel)
                   ┌──────▼──────┐
                   │             │
          ┌────────▼─────────────▼────────┐
          │  RetrieverAgent · SK native   │
          │  BM25 + Dense → RRF → Rerank  │
          │  → ranked chunks + citations  │
          └────────────────┬──────────────┘
                           │  chunks_json
          ┌────────────────▼──────────────┐
          │  AnalystAgent · SK native     │
          │  KPI extraction, trend anal.  │
          │  → cited claims per subtask   │
          └────────────────┬──────────────┘
                           │ all subtask results
          ┌────────────────▼──────────────┐
          │  ComparatorAgent · SK native  │
          │  Cross-doc synthesis, deltas  │
          │  → anomaly flags, multi-cite  │
          └────────────────┬──────────────┘
                           │
          ┌────────────────▼──────────────┐
          │  AuditorAgent · SK native     │
          │  Batch LLM entailment check   │
          │  VERIFIED / UNCERTAIN /       │
          │  UNVERIFIABLE (blocked)       │
          └────────────────┬──────────────┘
                           │ verified + uncertain only
          ┌────────────────▼──────────────┐
          │  SynthesizerAgent · SK func.  │
          │  → structured GFM report      │
          └────────────────┬──────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│       Structured Report  +  Audit Log (JSON)             │
└─────────────────────────────────────────────────────────┘
```

### Semantic Kernel Role

SK is not a convenience wrapper — it is the orchestration layer.

| SK Concept | Where Used | What It Does |
|---|---|---|
| `Kernel` singleton | `core/sk_kernel.py` | Single LLM service + plugin registry for the whole app |
| Native plugin | Retriever, Analyst, Auditor, Comparator | Python functions with `@kernel_function` |
| Semantic function | Planner, Synthesizer | `KernelFunctionFromPrompt` — kernel renders templates before dispatching to Groq |
| `KernelArguments` | Every agent hop | Threads `subtask → chunks → claims → report` forward without dropping citations |

---

## Retrieval Pipeline

```
Query
  │
  ├──► BM25 (rank-bm25)          ─── keyword precision: exact figures, ticker symbols
  │                                                        │
  └──► Dense (all-MiniLM-L6-v2)  ─── semantic coverage    │
                                                           │
                        ◄──── RRF fusion (k=60) ──────────┘
                               scale-invariant rank merge
                                           │
                        ◄──── Cross-encoder reranker ──────
                               ms-marco-MiniLM: joint (query, passage) attention
                                           │
                        ◄──── Confidence scoring ──────────
                               5-signal composite (retrieval, section type,
                               freshness, cross-filing, retrieval consistency)
```

---

## Quickstart — Docker (recommended)

The full stack (Qdrant + API) runs with a single command. The Dockerfile uses a multi-stage build: dependencies are installed in a builder stage and copied into a slim runtime image — no build toolchain ships in the final container.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin)
- A free [Groq API key](https://console.groq.com)
- Annual report PDFs for Infosys, TCS, Wipro (see [Seed Data](#seed-data))

### 1 — Clone and configure

```bash
git clone https://github.com/debarshi29/FinSight.git
cd FinSight
cp .env.example .env
# Set GROQ_API_KEY=<your key> in .env
# Optionally set FALLBACK_API_KEY / FALLBACK_MODEL / FALLBACK_BASE_URL for a reserve LLM
```

### 2 — Build and start

```bash
docker compose up --build
```

This builds the image, starts Qdrant (health-checked), then starts the API. The server is ready when you see:

```
finsight_api  | {"event": "finsight.startup", "level": "info", ...}
```

- **Web UI:** http://localhost:8000/ui
- **Metrics dashboard:** http://localhost:8000/dashboard
- **API docs (Swagger):** http://localhost:8000/docs
- **Qdrant UI:** http://localhost:6333/dashboard

### 3 — Ingest financial filings

Place PDFs in `data/filings/`, then POST them to the running container:

```bash
# Ingest a single file
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@data/filings/Infosys_AR_2024.pdf"

# Or ingest a whole directory from outside the container
for f in data/filings/*.pdf; do
  curl -s -X POST http://localhost:8000/ingest/upload -F "file=@$f"
  echo " ← $f"
done
```

> **First run:** The embedding model (~90 MB) and cross-encoder (~67 MB) are downloaded from HuggingFace on first use. Subsequent starts use the local Docker layer cache.

### 4 — Run a query

```bash
# Streaming (SSE) — live pipeline events + final result
curl -N -X POST http://localhost:8000/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "Compare Infosys and TCS operating margins FY2022–2024 and flag anomalies"}'

# Blocking — waits for full result
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Compare Infosys and TCS operating margins FY2022–2024 and flag anomalies"}'
```

---

## Quickstart — Local Development

For IDE-level debugging or faster iteration without rebuilding the image:

```bash
# 1. Start only Qdrant in Docker
docker compose up qdrant -d

# 2. Install Python deps
uv sync

# 3. Start the API with hot reload
uvicorn api.main:app --reload
# API at http://localhost:8000
```

---

## Logging

FinSight uses [structlog](https://www.structlog.org) for structured logging throughout the agent pipeline.

| `LOG_FORMAT` value | Output style | When to use |
|---|---|---|
| `text` (default) | Coloured console output | Local development |
| `json` | Newline-delimited JSON | Containers, log aggregators (CloudWatch, Datadog, Loki) |

`docker-compose.yml` sets `LOG_FORMAT=json` automatically. Docker's `json-file` log driver (configured with `max-size: 50m / max-file: 5`) captures all output and prevents disk exhaustion.

**View live container logs:**
```bash
docker compose logs -f api      # structured JSON events
docker compose logs -f qdrant   # Qdrant HTTP access log
```

**Log levels** are controlled by `LOG_LEVEL` (default `INFO`). Set `LOG_LEVEL=DEBUG` in `.env` for per-agent trace output including retrieval chunk scores and entailment reasoning.

---

## Sample Output

**Input query:** `"Compare Infosys and TCS operating margins FY2022-2024 and flag any anomalies"`

```json
{
  "task_id": "3f8a2d91-...",
  "query": "Compare Infosys and TCS operating margins FY2022-2024 and flag any anomalies",
  "summary": "## Executive Summary\n\nInfosys operating margin declined from 23.0% in FY2022 to 20.7% in FY2024...\n\n## Key Findings\n...",
  "verified_claims": [
    {
      "claim": "Infosys operating margin for FY2024 was 20.7%",
      "citation": {
        "document": "Infosys_AR_2024.pdf",
        "page": 31,
        "snippet": "Operating profit margin for FY2024 stood at 20.7%, compared to 21.5% in FY2023.",
        "confidence": 0.91,
        "section_type": "audited_financials"
      },
      "audit_status": "verified",
      "audit_reason": "Snippet directly states the figure; section type is audited_financials."
    }
  ],
  "uncertain_claims": [
    {
      "claim": "TCS margin pressure primarily driven by wage hikes in Q2 FY2023",
      "citation": {
        "document": "TCS_AR_2023.pdf",
        "page": 44,
        "snippet": "Wage revision impact was partially offset by operational efficiencies...",
        "confidence": 0.58,
        "section_type": "mda"
      },
      "audit_status": "uncertain",
      "audit_reason": "MD&A source (lower evidential weight); does not exclusively attribute margin pressure to wage hikes."
    }
  ],
  "audit_log": {
    "plan": ["Infosys operating margin FY2022", "...", "TCS operating margin FY2024"],
    "blocked_unverifiable": [],
    "agents_invoked": ["PlannerAgent", "RetrieverAgent", "AnalystAgent", "ComparatorAgent", "AuditorAgent", "SynthesizerAgent"],
    "latency_ms": 36200
  }
}
```

---

## Seed Data

The system is designed to be demonstrated on real public filings, not synthetic data.

| Company | Filings | Source |
|---|---|---|
| Infosys | Annual Report FY2022, FY2023, FY2024 | [infosys.com/investors](https://www.infosys.com/investors.html) |
| TCS | Annual Report FY2022, FY2023, FY2024 | [tcs.com/investors](https://www.tcs.com/investors) |
| Wipro | Annual Report FY2022, FY2023, FY2024 | [wipro.com/investors](https://www.wipro.com/investors/) |

Indian IT sector filings are ideal: comparable business models, consistent April–March fiscal years, and enough variation in margin trajectories to make cross-document comparison non-trivial.

**Sample analysis tasks:**
1. Compare Infosys and TCS operating margins FY2022–2024 and flag any anomalies or risk disclosures related to margin pressure.
2. What were the key revenue drivers for Wipro in FY2024 and how did they differ from FY2022?
3. Identify all mentions of attrition risk across all three companies in FY2023 and summarise the mitigations disclosed.
4. What guidance did each company provide for FY2025 capex and how does it compare to actual FY2024 capex?

---

## Evaluation

```bash
# Happy-path queries — verifiable facts in the corpus
python evaluation/harness.py evaluation/queries/happy_path.json

# Adversarial queries — future dates, fabricated events, cross-domain
python evaluation/harness.py evaluation/queries/adversarial.json
```

Results are written as JSON to `evaluation/results/`. Current scores: **4/4 happy-path**, **3/4 adversarial** (the fourth times out on a 5-company cross-domain query by design).

**What is measured:**
- **Retrieval recall** — are the right chunks returned for each subtask?
- **Hallucination block rate** — what % of adversarial queries are blocked or flagged at AuditorAgent?
- **Citation accuracy** — does the snippet actually entail the claim?

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Blocking analysis. Body: `{"query": "...", "company_filter": "...", "fiscal_year_filter": "..."}` |
| `/query/stream` | POST | SSE streaming — emits `start`, `planned`, `retrieved`, `analyzed`, `audited`, `compared`, `done`, `error` events |
| `/ingest/upload` | POST | Upload a PDF for ingestion. Multipart form: `file=@report.pdf` |
| `/eval/collection` | GET | Qdrant collection stats (vector count, status) |
| `/eval/audit-logs` | GET | List stored audit log files |
| `/eval/audit-logs/{id}` | GET | Retrieve a specific audit log |
| `/metrics` | GET | Per-agent latency percentiles, error rate, query count |
| `/health` | GET | Liveness check — returns `{"status": "ok"}` |
| `/ui` | GET | Web UI (query interface + live pipeline visualiser) |
| `/dashboard` | GET | Metrics dashboard |

Full interactive docs: `http://localhost:8000/docs`

---

## Environment Variables

All variables are optional except `GROQ_API_KEY`. Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Primary LLM |
| `FALLBACK_API_KEY` | — | Reserve LLM API key (OpenAI-compatible) |
| `FALLBACK_MODEL` | — | Reserve LLM model ID |
| `FALLBACK_BASE_URL` | — | Reserve LLM base URL |
| `QDRANT_HOST` | `localhost` | Qdrant host (`qdrant` inside Docker Compose) |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `LOG_FORMAT` | `text` | Log renderer: `text` (coloured) or `json` (structured) |
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry trace export |
| `OTEL_ENDPOINT` | `http://localhost:4317` | OTLP gRPC endpoint |

---

## Stack

| Component | Technology | Rationale |
|---|---|---|
| LLM | Groq — Llama 3.3 70B | Free tier, fastest open inference; OpenAI-compatible endpoint slots into SK |
| Fallback LLM | Any OpenAI-compatible endpoint | Activated automatically on primary timeout; same credentials interface |
| Orchestration | Semantic Kernel (Python) | Planner-based — dynamic task decomposition, native plugin system, KernelArguments context passing |
| Vector store | Qdrant (Docker) | Production-grade, local, free; async client; payload-level filtering |
| Embeddings | `all-MiniLM-L6-v2` | Local — no API dependency; 384-dim, strong retrieval quality |
| Reranker | `ms-marco-MiniLM-L6-v2` | Cross-encoder: joint (query, passage) attention; better than bi-encoder similarity alone |
| API | FastAPI + async SSE | Native async throughout; SSE streaming for live pipeline events |
| PDF parsing | PyMuPDF | Best positional extraction; bounding boxes for snippet location |
| Keyword retrieval | rank-bm25 | Exact figure matching (`20.7%`, `FY2024`) where dense retrieval undershoots |
| Containerisation | Docker multi-stage | Builder stage installs deps; slim runtime image with no build toolchain |
| Logging | structlog + Docker json-file | JSON-rendered logs in containers; log rotation (50 MB / 5 files) |
| Observability | OpenTelemetry (optional) | Full trace per agent hop; OTLP export to any compatible backend |

---

## Repository Structure

```
finsight/
├── agents/
│   ├── router.py          # PlannerAgent — SK semantic function
│   ├── retriever.py       # RetrieverAgent — SK native plugin, hybrid retrieval
│   ├── analyst.py         # AnalystAgent — SK native plugin, KPI extraction
│   ├── comparator.py      # ComparatorAgent — SK native plugin, cross-doc synthesis
│   ├── auditor.py         # AuditorAgent — SK native plugin, batch entailment check
│   └── synthesizer.py     # SynthesizerAgent — delegates to SK semantic function
├── retrieval/
│   ├── qdrant_store.py    # Async Qdrant client wrapper
│   ├── embedder.py        # sentence-transformers local embeddings
│   ├── bm25.py            # BM25Okapi over Qdrant payload text
│   ├── hybrid.py          # Reciprocal Rank Fusion (k=60)
│   ├── reranker.py        # Cross-encoder rescoring
│   └── confidence.py      # 5-signal composite confidence model
├── ingestion/
│   ├── parser.py          # PyMuPDF page extraction
│   ├── chunker.py         # Heading-aware sliding window chunker
│   ├── metadata.py        # Section type, fiscal year, company detection
│   └── pipeline.py        # Full ingest orchestration
├── api/
│   ├── main.py            # FastAPI app, lifespan, static serving
│   ├── metrics_store.py   # In-process per-agent latency + query metrics
│   ├── routes/
│   │   ├── query.py       # POST /query — blocking pipeline
│   │   ├── query_stream.py# POST /query/stream — SSE streaming pipeline
│   │   ├── ingest.py      # POST /ingest/upload
│   │   ├── eval.py        # GET /eval/*
│   │   └── metrics.py     # GET /metrics
│   ├── middleware/
│   │   └── guardrails.py  # Prompt injection detection (pure ASGI)
│   └── static/
│       ├── index.html     # Web UI — query interface + live pipeline visualiser
│       └── dashboard.html # Metrics dashboard
├── core/
│   ├── sk_kernel.py       # SK kernel singleton + plugin registration
│   ├── models.py          # Citation, Chunk, AuditedClaim, AuditLog dataclasses
│   ├── config.py          # pydantic-settings (all env vars)
│   └── groq_client.py     # Async Groq/OpenAI client wrapper with fallback
├── observability/
│   └── tracer.py          # structlog config (text/json), @traced decorator, OTel
├── evaluation/
│   ├── harness.py         # Async test runner
│   └── queries/           # happy_path.json, adversarial.json
├── tests/                 # Unit tests — chunker, confidence, ingestion, models
├── data/filings/          # Place seed PDFs here (volume-mounted in Docker)
├── audit_logs/            # Per-run JSON artifacts (volume-mounted in Docker)
├── qdrant_storage/        # Qdrant on-disk vectors (volume-mounted in Docker)
├── docker-compose.yml     # Qdrant + API services with health checks, log rotation
├── Dockerfile             # Multi-stage: builder (uv + deps) → runtime (slim)
├── .dockerignore          # Excludes .env, qdrant_storage, tests, __pycache__, etc.
├── pyproject.toml         # Dependencies + ruff + pytest config
└── .env.example           # All supported environment variables with defaults
```
