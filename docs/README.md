# FinSight Documentation

Start here. Each document has a distinct job — they are meant to be read in roughly this order.

| Document | What it is | Read it when you want to… |
|---|---|---|
| [../README.md](../README.md) | Project overview, quickstart, API reference, stack table | Run the system or pitch it |
| [HLD.md](HLD.md) | **High-Level Design** — requirements, system context, layered architecture, the six agent roles, request flows, cross-cutting concerns, deployment view, risks, requirement→component traceability | Understand *what* the system is and *why* it's shaped this way, before touching code |
| [LLD.md](LLD.md) | **Low-Level Design** — module map, every module's signatures and behaviour, data models field-by-field, the retrieval/chunker/normalizer algorithms as implemented, API request/response shapes, error-handling matrix, extension points, testing map | Make a specific code change |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Narrative deep-dive on the design (LangGraph `StateGraph` orchestration, two-stage retrieval, RRF, confidence model, AuditorAgent, ingestion) | Read a longer-form explanation of one subsystem |
| [DECISIONS.md](DECISIONS.md) | Numbered decision log — context, options considered, choice, trade-offs accepted | Know why an alternative was rejected, or record a new trade-off |
| [RUNBOOK.md](RUNBOOK.md) | Operations runbook — start/stop, ingestion, observability, audit logs, incident table, backup/recovery, hardening checklist | Run, observe, or recover a deployment |
| [MEDIUM_ARTICLE.md](MEDIUM_ARTICLE.md) | Long-form article draft on the anti-hallucination design | Write or reference external-facing material |
| [FinSight_Documentation.pdf](FinSight_Documentation.pdf) / [`.tex`](FinSight_Documentation.tex) | Full LaTeX technical reference (module-by-module, data models, API/config tables, deployment topology) | Want a single printable reference |
| [FinSight_Project_Specification.pdf](FinSight_Project_Specification.pdf) | The original project specification | Check the initial brief |

## How the docs stay honest

`HLD.md` and `LLD.md` are written against the `main` branch and transcribed from the implementation. Where the code and the older prose in `ARCHITECTURE.md` disagree, the code wins and the LLD says so. If you change behaviour, update the relevant doc in the same PR — see [../CONTRIBUTING.md](../CONTRIBUTING.md).

## One-paragraph summary

FinSight is a multi-agent RAG system for financial filings built around one constraint: every user-visible figure must resolve to a verbatim source passage, and any claim that can't be verified is blocked before synthesis rather than flagged after. A per-query LLM plan fans out — via a LangGraph `StateGraph`'s `Send` API — into parallel hybrid retrieval (BM25 + dense → RRF → cross-encoder rerank → 5-signal confidence); an Analyst extracts only verbatim-stated claims; a deterministic Python normalizer converts all currency/scale figures to ₹ crore; a separate Auditor pass classifies each claim VERIFIED / UNCERTAIN / UNVERIFIABLE concurrently with a Comparator that lays verbatim figures side by side (one graph superstep); a Synthesizer writes the report from verified evidence only. Every run persists a machine-readable audit log.
