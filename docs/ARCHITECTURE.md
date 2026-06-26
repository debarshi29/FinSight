# FinSight — System Architecture

## Overview

FinSight is a regulatory-grade financial analysis system where every claim is traceable to a verbatim source passage, uncertain claims are explicitly flagged, and every analysis run produces a machine-readable audit log.

## High-Level Pipeline

```
User Query (natural language)
    │
    ▼
PlannerAgent — Groq LLM decomposes query into 2-6 ordered subtasks
    │
    ▼ (per subtask)
RetrieverAgent — Hybrid BM25 + Dense retrieval → RRF fusion → Cross-encoder reranking
    │
    ▼
AnalystAgent — KPI extraction, trend analysis, claim generation with citations
    │
    ▼
ComparatorAgent — Cross-document synthesis, delta calculation, anomaly flagging
    │
    ▼
AuditorAgent — Structural entailment check: VERIFIED / UNCERTAIN / UNVERIFIABLE
    │
    ▼
SynthesizerAgent — Assembles final report (only verified + flagged uncertain claims)
    │
    ▼
Structured Report + Audit Log (JSON artifact in audit_logs/)
```

## Component Map

| Component | Technology | File |
|---|---|---|
| LLM Inference | Groq (Llama 3.3 70B) | core/groq_client.py |
| Orchestration | Semantic Kernel | core/sk_kernel.py |
| Vector Store | Qdrant (Docker) | retrieval/qdrant_store.py |
| Embeddings | all-MiniLM-L6-v2 | retrieval/embedder.py |
| BM25 | rank-bm25 | retrieval/bm25.py |
| Hybrid Fusion | RRF (k=60) | retrieval/hybrid.py |
| Reranker | ms-marco-MiniLM-L6-v2 | retrieval/reranker.py |
| Confidence | 5-signal composite | retrieval/confidence.py |
| PDF Parsing | PyMuPDF | ingestion/parser.py |
| Chunking | Sliding window + headings | ingestion/chunker.py |
| Section Detection | Regex heuristics | ingestion/metadata.py |
| API | FastAPI + async | api/main.py |
| Observability | structlog + OTel | observability/tracer.py |

## The Citation Object

Every claim carries a Citation object through every agent hop:

```json
{
  "document": "Infosys_AR_2024.pdf",
  "page": 31,
  "snippet": "Operating profit margin for FY2024 stood at 20.7%, compared to 21.5% in FY2023.",
  "claim": "Infosys operating margin declined 0.8pp in FY2024",
  "confidence": 0.91,
  "section_type": "audited_financials"
}
```

## Source Confidence Model (5 signals)

| Signal | Weight | High Score | Low Score |
|---|---|---|---|
| Retrieval score | 35% | Direct answer after reranking | Tangential passage |
| Section type | 25% | Audited financials | MD&A, letter |
| Freshness | 15% | Most recent fiscal year | 3+ year old filing |
| Cross-filing consistency | 15% | Same figure in 3+ filings | Appears once |
| Retrieval consistency | 10% | Surfaces for 3+ query variants | Single phrasing only |

## AuditorAgent — Three-Tier Status

| Status | Condition | Action |
|---|---|---|
| VERIFIED | Snippet entails claim; confidence ≥ 0.65 | Included unmarked |
| UNCERTAIN | Low confidence (0.50–0.65) or weak support | Included with [UNCERTAIN] flag |
| UNVERIFIABLE | No snippet; or snippet contradicts claim | Blocked entirely; logged |

## Retrieval Architecture

1. **BM25**: keyword matching over Qdrant payload text — exact figure matching
2. **Dense**: cosine similarity via all-MiniLM-L6-v2 — semantic matching
3. **RRF fusion**: combines both ranked lists (scale-invariant)
4. **Cross-encoder reranking**: ms-marco rescores top-20 candidates jointly

See DECISIONS.md for why each choice was made.

## Audit Log Format

```json
{
  "task_id": "uuid-v4",
  "timestamp": "2024-03-15T10:23:00Z",
  "user_query": "Compare Infosys and TCS margins FY2022-2024",
  "plan": ["retrieve_infosys_margins", "retrieve_tcs_margins", "compare_deltas"],
  "retrievals": {"retrieve_infosys_margins": ["chunk_ids + scores"]},
  "claims": [{"claim": "...", "citation": {...}, "audit_status": "verified"}],
  "flagged_uncertain": ["..."],
  "blocked_unverifiable": [],
  "agents_invoked": ["PlannerAgent", "RetrieverAgent", "AnalystAgent", "..."],
  "latency_ms": 3240
}
```
