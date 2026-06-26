# FinSight — Regulatory-Grade Financial Intelligence System

A multi-agent financial analysis system where every claim is traceable to a verbatim source passage, uncertain claims are flagged rather than hallucinated, and every analysis generates a machine-readable audit log — because in compliance-heavy environments, an unverifiable AI output is worse than no output.

## Architecture

```
User Query
    │
    ├─ PlannerAgent (SK + Groq) → ordered subtasks
    │
    ├─ RetrieverAgent (BM25 + Dense → RRF → Cross-encoder) → ranked chunks
    │
    ├─ AnalystAgent (Groq) → KPIs + cited claims
    │
    ├─ ComparatorAgent (Groq) → cross-document deltas + anomalies
    │
    ├─ AuditorAgent (Groq entailment) → VERIFIED / UNCERTAIN / UNVERIFIABLE
    │
    └─ SynthesizerAgent (Groq) → structured report
```

## Quickstart

```bash
# 1. Clone and set up
git clone <repo>
cd finsight
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 2. Start Qdrant
docker compose up qdrant -d

# 3. Install dependencies
uv sync

# 4. Ingest PDFs (place in data/filings/)
python -c "
import asyncio
from ingestion.pipeline import ingest_directory
asyncio.run(ingest_directory('data/filings'))
"

# 5. Start API
uvicorn api.main:app --reload

# 6. Run a query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Compare Infosys and TCS operating margins FY2022-2024"}'
```

## One-Command Docker

```bash
docker compose up
```

## Sample Query & Output

**Input:** "Compare Infosys and TCS operating margins FY2022-2024 and flag any anomalies"

**Output (excerpt):**
```json
{
  "task_id": "3f8a-...",
  "query": "Compare Infosys and TCS operating margins FY2022-2024",
  "summary": "Infosys operating margin declined from 23.0% (FY2022) to 20.7% (FY2024)...",
  "verified_claims": [
    {
      "claim": "Infosys operating margin for FY2024 was 20.7%",
      "citation": {
        "document": "Infosys_AR_2024.pdf",
        "page": 31,
        "snippet": "Operating profit margin for FY2024 stood at 20.7%...",
        "confidence": 0.91,
        "section_type": "audited_financials"
      },
      "audit_status": "verified"
    }
  ],
  "uncertain_claims": [],
  "audit_log": {
    "agents_invoked": ["PlannerAgent", "RetrieverAgent", "AnalystAgent", "ComparatorAgent", "AuditorAgent", "SynthesizerAgent"],
    "latency_ms": 3240
  }
}
```

## Eval Metrics (on seed data)

| Metric | Score |
|---|---|
| Retrieval recall | Measured via evaluation harness |
| Hallucination block rate | Adversarial queries blocked at AuditorAgent |
| Citation accuracy | snippet→claim entailment checked per run |

Run the eval harness:
```bash
python evaluation/harness.py evaluation/queries/happy_path.json
python evaluation/harness.py evaluation/queries/adversarial.json
```

## Stack

| Component | Technology |
|---|---|
| LLM | Groq (Llama 3.3 70B) |
| Orchestration | Semantic Kernel (Python) |
| Vector Store | Qdrant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| API | FastAPI + async |
| PDF Parsing | PyMuPDF |

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Design Decisions](docs/DECISIONS.md)
