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
│  kernel.invoke("Planner", "decompose", ...)             │
│  → Groq decomposes query into 2–6 ordered subtasks      │
└─────────────────────────┬───────────────────────────────┘
                          │  per subtask
                   ┌──────▼──────┐
                   │             │
          ┌────────▼─────────────▼────────┐
          │  RetrieverAgent · SK native   │
          │  BM25 + Dense → RRF → Rerank  │
          │  → ranked chunks + citations  │
          └────────────────┬──────────────┘
                           │  chunks_json (KernelArguments)
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
          │  LLM entailment: snippet ⊨    │
          │  claim?  VERIFIED / UNCERTAIN  │
          │  / UNVERIFIABLE (blocked)     │
          └────────────────┬──────────────┘
                           │ verified + uncertain only
          ┌────────────────▼──────────────┐
          │  SynthesizerAgent · SK func.  │
          │  kernel.invoke("Synthesizer") │
          │  → structured report (text)   │
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
| Native plugin | Retriever, Analyst, Auditor, Comparator | Python functions with `@kernel_function`; invoked via `kernel.invoke()` |
| Semantic function | Planner, Synthesizer | `KernelFunctionFromPrompt` — kernel renders `{{$variable}}` templates before dispatching to Groq |
| `KernelArguments` | Every agent hop | Threads `subtask → chunks_json → analysis_json → claims_json → report` forward without dropping citations |

If you removed SK and replaced it with direct Groq API calls, you would need to re-implement the plugin registry, context passing, and prompt template rendering. SK is structural, not cosmetic.

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

## Quickstart

### Prerequisites

- Docker (for Qdrant)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- A free [Groq API key](https://console.groq.com)
- Annual report PDFs for Infosys, TCS, Wipro FY2022–2024 (see [Seed Data](#seed-data))

### 1 — Clone and configure

```bash
git clone https://github.com/debarshi29/FinSight.git
cd FinSight
cp .env.example .env
# Open .env and set GROQ_API_KEY=<your key>
```

### 2 — Start Qdrant

```bash
docker compose up qdrant -d
# Qdrant UI available at http://localhost:6333/dashboard
```

### 3 — Install dependencies

```bash
uv sync
```

> First run downloads the embedding model (~90 MB) and cross-encoder (~67 MB) from HuggingFace. Subsequent runs use the local cache.

### 4 — Ingest financial filings

Place PDFs in `data/filings/`, then:

```bash
python -c "
import asyncio
from ingestion.pipeline import ingest_directory
asyncio.run(ingest_directory('data/filings'))
"
```

### 5 — Start the API

```bash
uvicorn api.main:app --reload
# API docs at http://localhost:8000/docs
```

### 6 — Run a query

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Compare Infosys and TCS operating margins FY2022-2024 and flag any anomalies"
  }'
```

### One-command Docker

```bash
docker compose up
```

Starts Qdrant and the API together. Qdrant health-checked before API starts.

---

## Sample Output

**Input query:** `"Compare Infosys and TCS operating margins FY2022-2024 and flag any anomalies"`

```json
{
  "task_id": "3f8a2d91-...",
  "query": "Compare Infosys and TCS operating margins FY2022-2024 and flag any anomalies",
  "summary": "## Executive Summary\n\nInfosys operating margin declined from 23.0% in FY2022 to 20.7% in FY2024, a 2.3pp compression over the period...\n\n## Key Findings\n...",
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
      "audit_reason": "Snippet from MD&A (lower evidential weight); does not exclusively attribute margin pressure to wage hikes."
    }
  ],
  "audit_log": {
    "task_id": "3f8a2d91-...",
    "timestamp": "2026-06-26T12:43:00Z",
    "plan": [
      "Infosys operating margin FY2022",
      "Infosys operating margin FY2023",
      "Infosys operating margin FY2024",
      "TCS operating margin FY2022",
      "TCS operating margin FY2023",
      "TCS operating margin FY2024"
    ],
    "flagged_uncertain": ["TCS margin pressure primarily driven by wage hikes in Q2 FY2023"],
    "blocked_unverifiable": [],
    "agents_invoked": ["PlannerAgent", "RetrieverAgent", "AnalystAgent", "ComparatorAgent", "AuditorAgent", "SynthesizerAgent"],
    "latency_ms": 4120
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

**Sample analysis tasks to run against the seed data:**
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

The harness scores each result against its expected behaviour and writes a JSON report to `evaluation/results/`.

**What is measured:**
- **Retrieval recall** — are the right chunks returned for each subtask?
- **Hallucination block rate** — what % of adversarial queries are blocked or flagged at AuditorAgent?
- **Citation accuracy** — does the snippet actually entail the claim (checked per run by AuditorAgent)?

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Run a full analysis. Body: `{"query": "..."}` |
| `/ingest/upload` | POST | Upload a PDF for ingestion. Multipart form. |
| `/eval/collection` | GET | Qdrant collection stats (vector count, status) |
| `/eval/audit-logs` | GET | List stored audit log files |
| `/eval/audit-logs/{id}` | GET | Retrieve a specific audit log |
| `/health` | GET | Liveness check |

Full interactive docs: `http://localhost:8000/docs`

---

## Stack

| Component | Technology | Rationale |
|---|---|---|
| LLM | Groq — Llama 3.3 70B | Free tier, fastest open inference; OpenAI-compatible endpoint slots into SK |
| Orchestration | Semantic Kernel (Python) | Planner-based — dynamic task decomposition, native plugin system, KernelArguments context passing |
| Vector Store | Qdrant (Docker) | Production-grade, local, free; async client; payload-level filtering |
| Embeddings | `all-MiniLM-L6-v2` | Local — no API dependency; 384-dim, strong retrieval quality |
| Reranker | `ms-marco-MiniLM-L6-v2` | Cross-encoder: joint (query, passage) attention; better than bi-encoder similarity alone |
| API | FastAPI + async | Native async throughout the retrieval and agent pipeline |
| PDF Parsing | PyMuPDF | Best positional extraction; bounding boxes for snippet location |
| Keyword retrieval | rank-bm25 | Exact figure matching (`20.7%`, `FY2024`) where dense retrieval undershoots |
| Observability | structlog + OpenTelemetry | Full trace per agent hop; optional OTLP export |

---

## Repository Structure

```
finsight/
├── agents/
│   ├── router.py          # PlannerAgent — SK semantic function via kernel.invoke
│   ├── retriever.py       # RetrieverAgent — SK native plugin, hybrid retrieval
│   ├── analyst.py         # AnalystAgent — SK native plugin, KPI extraction
│   ├── comparator.py      # ComparatorAgent — SK native plugin, cross-doc synthesis
│   ├── auditor.py         # AuditorAgent — SK native plugin, entailment checking
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
│   ├── main.py            # FastAPI app, lifespan, middleware
│   ├── routes/
│   │   ├── query.py       # POST /query — full pipeline via kernel.invoke
│   │   ├── ingest.py      # POST /ingest/upload
│   │   └── eval.py        # GET /eval/*
│   └── middleware/
│       └── guardrails.py  # Prompt injection detection
├── core/
│   ├── sk_kernel.py       # SK kernel singleton + plugin registration
│   ├── models.py          # Citation, Chunk, AuditedClaim, AuditLog dataclasses
│   ├── config.py          # pydantic-settings
│   └── groq_client.py     # Groq async client wrapper
├── evaluation/
│   ├── harness.py         # Async test runner
│   └── queries/           # Happy-path and adversarial query sets
├── observability/
│   └── tracer.py          # structlog config + @traced decorator
├── tests/                 # Unit tests for retrieval, ingestion, models
├── docs/
│   ├── ARCHITECTURE.md    # Full system design and component map
│   └── DECISIONS.md       # Every non-obvious design decision with reasoning
├── data/filings/          # Place seed PDFs here
├── audit_logs/            # Per-run JSON artifacts (git-ignored)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Full system design: pipeline, SK design, retrieval architecture, data models, confidence scoring
- **[DECISIONS.md](docs/DECISIONS.md)** — Every non-obvious design decision with context, options considered, reasoning, and tradeoffs accepted
