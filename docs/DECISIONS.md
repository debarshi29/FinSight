# FinSight — Design Decisions

This document records every non-obvious design decision made in building FinSight. For each decision, it records the context that forced a choice, the options that were genuinely considered, the decision taken, the reasoning behind it, and the tradeoffs that were explicitly accepted.

Decisions are listed in the order they were made. The date marks when the decision was finalised, not when the implementation was completed.

---

## Decision 1 — Semantic Kernel over LangGraph for multi-agent orchestration

**Date:** 2026-06-26

**Context:** FinSight requires multi-agent orchestration with dynamic task decomposition. A user asking "compare margins FY2022–2024" should get a different execution plan than "summarise attrition risk across three companies." The orchestration layer must support this without requiring a graph topology to be hardcoded for every query type.

**Options considered:**

- **LangGraph** — graph-based, explicit node/edge topology, battle-tested at scale, large community. Well-suited to pipelines with known structure.
- **Semantic Kernel (SK)** — planner-based, Microsoft-backed, native Python 1.x SDK with plugin system and `KernelFunctionFromPrompt` for LLM-driven task decomposition. First-class `KernelArguments` context passing.
- **Raw async Python** — no framework. Full control, zero overhead, but requires reimplementing plugin discovery, context threading, and prompt rendering.

**Decision:** Semantic Kernel with `OpenAIChatCompletion` configured against Groq's OpenAI-compatible endpoint.

**Reasoning:** SK's planner model is the right fit for dynamic decomposition: the `Planner.decompose` semantic function asks the LLM to generate a subtask list for each specific query. Two different queries produce two different plans. This is the core value proposition of FinSight — a hardcoded graph topology cannot achieve it. LangGraph's strengths (explicit control flow, conditional edges) are assets in pipelines where the topology is known in advance and needs to be reliable; they add complexity when the topology needs to vary.

The `KernelArguments` context chain also eliminates a class of bugs: citations cannot be dropped between hops because they are serialised into the argument passed to `kernel.invoke()`. A raw async implementation would require manual context threading.

**Tradeoffs accepted:** SK Python 1.x is less battle-tested than LangGraph. More registration ceremony (explicit `kernel.add_plugin()` per agent). The Stepwise Planner can generate invalid subtask lists on ambiguous queries — mitigated by the fallback list parser in `agents/router.py`.

---

## Decision 2 — kernel.invoke() at every agent hop, not direct Groq API calls

**Date:** 2026-06-26

**Context:** After the initial implementation, the code had `@kernel_function` decorators on agent methods but was calling `chat_completion()` directly inside them. SK was present but not load-bearing — removing it would not have changed behaviour.

**Options considered:**

- **Direct Groq calls (`chat_completion()`)** — simpler, one fewer abstraction layer, slightly less latency.
- **`kernel.invoke()` at every hop** — all LLM calls and plugin calls route through the SK kernel.

**Decision:** Rewrite to route every agent invocation through `kernel.invoke()` with `KernelArguments`.

**Reasoning:** The direct-call approach defeats the purpose of having SK. The kernel's plugin registry, context passing, and prompt template rendering are only exercised if everything goes through `kernel.invoke()`. More importantly, `KernelFunctionFromPrompt` only renders `{{$variable}}` template syntax when the kernel dispatches the call — calling Groq directly bypasses the template rendering entirely, meaning semantic functions stop working as intended.

The SK refactor also enforces a single point of LLM dispatch. Adding observability, retry logic, rate limiting, or model switching in the future requires changing `core/sk_kernel.py` in one place, not hunting through every agent file.

**Tradeoffs accepted:** Minor additional serialisation overhead (Python → JSON string → kernel.invoke → JSON string → Python) at each hop. Accepted because the structural benefits outweigh it.

---

## Decision 3 — Native plugins for 4 agents, semantic functions for 2

**Date:** 2026-06-26

**Context:** SK offers two types of registered functions: native plugins (Python methods with `@kernel_function`) and semantic functions (`KernelFunctionFromPrompt`, prompt templates).

**Options considered:**

- **All native plugins** — straightforward, all logic in Python, easy to test.
- **All semantic functions** — all logic in prompts, no Python implementation required.
- **Mixed** — native for agents with real computation, semantic for pure LLM reasoning tasks.

**Decision:** Native plugins for Retriever, Analyst, Auditor, Comparator. Semantic functions for Planner, Synthesizer.

**Reasoning:** The distinction reflects what each agent actually does:

- **Retriever** performs BM25 search, ANN vector search, RRF fusion, cross-encoder reranking, and confidence scoring. These are deterministic computations. A semantic function cannot do them.
- **Analyst** and **Auditor** call Groq internally but must also parse JSON, handle errors, and construct typed dataclasses. They need Python.
- **Comparator** performs cross-document delta analysis and also calls Groq — same reasoning as Analyst.
- **Planner** and **Synthesizer** are pure LLM tasks: given text in, get text out. They have no side effects, no external calls, and no structured computation. `KernelFunctionFromPrompt` with `{{$variable}}` template syntax is exactly the right tool for these.

**Tradeoffs accepted:** The mixed design means two different mental models for plugin and semantic function agents. Documented here and in ARCHITECTURE.md to make the distinction explicit.

---

## Decision 4 — Reciprocal Rank Fusion over weighted score combination for hybrid retrieval

**Date:** 2026-06-26

**Context:** Combining BM25 keyword scores and dense cosine similarity scores into a single ranked list for the retrieval candidate pool.

**Options considered:**

- **Weighted linear combination:** `score = α × bm25_score + β × dense_score` — simple, tunable.
- **Score normalisation then combine:** normalise both to [0,1] before weighting — addresses scale mismatch but fragile.
- **Reciprocal Rank Fusion (RRF):** `score = Σ 1 / (k + rank_i)` — operates on ranks, not scores.

**Decision:** RRF with k=60 (Cormack et al. 2009 default).

**Reasoning:** BM25 and cosine similarity are in completely different units. BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded [0,1]. Any fixed weighting `α, β` will produce results that vary across corpora and query types. Normalisation is fragile — it requires knowing the score distribution in advance or computing it per-query, which adds latency and instability.

RRF is scale-invariant. It only cares about which document was ranked first, second, third by each retriever. The `k=60` constant prevents the top-ranked document from overwhelming everything else. Empirically, RRF consistently outperforms weighted combination in IR benchmarks (TREC, MS MARCO) without requiring tuning.

**Tradeoffs accepted:** RRF discards score magnitude information. Two documents ranked #1 and #2 contribute nearly equal scores regardless of how much better #1 was. This is acceptable because the cross-encoder reranker (stage 2) rescores the entire candidate pool using full joint attention, producing calibrated scores that replace the RRF scores entirely.

---

## Decision 5 — Two-stage retrieval: bi-encoder for candidates, cross-encoder for reranking

**Date:** 2026-06-26

**Context:** Needed accurate relevance scoring at query time without prohibitive latency.

**Options considered:**

- **Bi-encoder only (dense similarity):** Fast O(1) with ANN index. Encodes query and document independently — scores are not jointly calibrated.
- **Cross-encoder only:** Joint (query, document) attention. Highly accurate. O(n) per query — too slow for real-time use over a large corpus.
- **Two-stage:** Bi-encoder retrieves top-N candidates, cross-encoder reranks.

**Decision:** Two-stage with `all-MiniLM-L6-v2` (bi-encoder) + `ms-marco-MiniLM-L6-v2` (cross-encoder).

**Reasoning:** The key insight is that cross-encoder accuracy comes from seeing the query and document jointly — it cannot be done in a single forward pass at ANN-index speed. The two-stage architecture delegates the "fast but approximate" problem to the bi-encoder and the "slow but accurate" problem to the cross-encoder, which only runs on a small candidate pool.

`all-MiniLM-L6-v2` is a strong general-purpose bi-encoder (384-dim, 80M+ downloads). `ms-marco-MiniLM-L6-v2` is fine-tuned specifically on the MS MARCO passage ranking benchmark — the standard reranking dataset for this task. Both models run locally with no API dependency.

**Tradeoffs accepted:** Two model loads at startup (~150 MB total). Cross-encoder adds 200–500ms per query depending on candidate pool size. Acceptable for compliance-grade analysis where retrieval correctness matters more than sub-second latency.

---

## Decision 6 — AuditorAgent as a separate structural pass, not a prompt guardrail

**Date:** 2026-06-26

**Context:** Needed a mechanism to prevent hallucinated claims from appearing in the final report.

**Options considered:**

- **Prompt instruction:** "Only use information explicitly stated in the provided context. If unsure, say so." — Zero additional latency, but the model can override it.
- **Post-hoc filtering:** Filter out claims with confidence below a threshold — uses the retrieval confidence signal only, no LLM entailment check.
- **Separate AuditorAgent pass:** Individual LLM entailment check per claim with three-tier classification.

**Decision:** Separate AuditorAgent (`agents/auditor.py`) performing per-claim entailment verification.

**Reasoning:** Prompt instructions exist inside the same context window as the generation. A model with strong priors about a financial figure (e.g. "Apple's market cap is $3T") will sometimes generate that figure confidently regardless of what the retrieved context says. The instruction competes with the model's training distribution.

The AuditorAgent cannot be bypassed this way. It is a separate LLM call that asks: "Does this specific snippet literally support this specific claim?" Each claim is evaluated in isolation with the supporting snippet visible. If the snippet does not entail the claim — or there is no snippet — the claim is classified UNVERIFIABLE and blocked before SynthesizerAgent runs. There is no code path from UNVERIFIABLE to the final report.

**Tradeoffs accepted:** Additional LLM call per claim. This is the single largest latency contributor. Justified because the alternative — a hallucinated figure in a compliance report — is a much worse failure mode. The audit log records every blocked claim so analysts can review what was excluded.

---

## Decision 7 — Composite confidence threshold at 0.65 with three-tier classification

**Date:** 2026-06-26

**Context:** Needed a threshold for classifying retrieval confidence into actionable tiers for the AuditorAgent.

**Options considered:**

- **Binary (pass/fail at 0.5):** Simple. VERIFIED above, UNVERIFIABLE below.
- **Three-tier with configurable thresholds:** VERIFIED / UNCERTAIN / UNVERIFIABLE with separate thresholds.
- **Per-section-type thresholds:** Audited financials get a different cutoff than MD&A.

**Decision:** Three-tier classification at 0.65 (VERIFIED) / 0.50 (UNCERTAIN) / below 0.50 (UNVERIFIABLE). Single configurable threshold via `CONFIDENCE_THRESHOLD` in `.env`.

**Reasoning:** Binary pass/fail loses signal. A claim at 0.52 confidence with a supporting snippet from an MD&A section is genuinely different from one at 0.91 from an audited financials section — treating them identically discards information a compliance analyst needs. The UNCERTAIN tier exists precisely for this: the claim reaches the report, but it is explicitly labelled as uncertain, and its source and confidence are visible.

The 0.65 default was calibrated against the 5-signal composite model: a chunk from `audited_financials` (section weight 1.0) that appears in multiple filings and is retrieved consistently by both BM25 and dense will naturally score above 0.65. Chunks from weaker sections or single-filing sources typically score 0.50–0.65.

**Tradeoffs accepted:** The threshold is somewhat arbitrary and domain-dependent. It is configurable for this reason. The eval harness (`evaluation/harness.py`) provides an empirical way to assess the threshold against the seed corpus.

---

## Decision 8 — Heading-aware sliding window chunker (400-token target, 80-token overlap)

**Date:** 2026-06-26

**Context:** Needed to chunk financial PDFs for vector indexing. Financial filings have strong heading structure (Balance Sheet, Notes to Accounts, MD&A) and precise numeric content where chunk boundaries matter.

**Options considered:**

- **Fixed-size chunks, no overlap:** Simple, predictable. Splits mid-sentence or mid-table frequently.
- **Sentence-level chunks:** Natural boundaries. Highly variable chunk sizes — some financial sentences span multiple lines; some paragraphs are a single number.
- **Paragraph-level chunks:** Good for prose sections but financial tables produce very short paragraphs.
- **Heading-aware sliding window with overlap:** New chunk at each detected heading; remainder chunked at token limit with overlap.

**Decision:** Heading-aware sliding window with 400-token target and 80-token overlap. SHA-256 deduplication on chunk content.

**Reasoning:** Financial filings have reliable heading structure. Respect it: when a new section heading is detected, start a new chunk regardless of the current token count. This keeps semantically cohesive content together (all of "Notes to Account 5" in one chunk, not split across two). Within a section, 400 tokens is large enough to contain a full paragraph with supporting figures but small enough for the cross-encoder to process efficiently.

The 80-token overlap ensures that a figure appearing at the boundary between two chunks (a common occurrence in financial tables that span paragraphs) is present in both chunks and therefore retrievable regardless of which chunk scores higher.

SHA-256 deduplication prevents the same passage from being indexed twice when a filing is re-ingested. The hash is computed over `(company, fiscal_year, page, text)` — re-ingesting the same PDF with the same metadata produces no new chunks.

**Tradeoffs accepted:** Heading detection uses regex + font-size proximity heuristics which can fail on scanned PDFs or PDFs with non-standard heading formatting. For the target corpus (Infosys, TCS, Wipro annual reports — all well-formatted digital PDFs), this is reliable.

---

## Decision 9 — Qdrant over Chroma or a managed vector service

**Date:** 2026-06-26

**Context:** Needed a vector store with async client, payload filtering, and the ability to run locally without an API key or account.

**Options considered:**

- **Chroma** — local, simple, popular for prototypes. Limited filtering, synchronous client, lower throughput.
- **Pinecone / Weaviate Cloud** — managed, scalable. Requires API key and incurs cost; adds external dependency for a portfolio project.
- **Qdrant** — open-source, Docker-deployable, async Python client, full payload filtering with indexed fields, gRPC + HTTP, production-grade.

**Decision:** Qdrant, Docker-deployed, accessed via the async `qdrant-client` Python SDK.

**Reasoning:** Qdrant's async client is a first-class requirement — the entire FastAPI stack is async, and a synchronous vector store call would block the event loop. Chroma's Python client is synchronous. Qdrant's payload filtering (filter by `company`, `fiscal_year`, `section_type`) is used in the retrieval pipeline to scope searches per-query. The Docker deployment means no API key, no cost, and no external service dependency — the entire system is self-contained.

**Tradeoffs accepted:** Requires Docker to run. Qdrant is heavier than Chroma (~200 MB Docker image vs in-process). Acceptable for the production-grade positioning of the system.

---

## Decision 10 — Groq (Llama 3.3 70B) as the LLM backend

**Date:** 2026-06-26

**Context:** Needed a capable LLM for planning, analysis, entailment checking, and synthesis. Cost and API availability matter for a portfolio project that may be demonstrated many times.

**Options considered:**

- **OpenAI GPT-4o** — highest capability, expensive, requires paid account.
- **Anthropic Claude** — strong reasoning, competitive pricing, different API format.
- **Groq — Llama 3.3 70B** — free tier, OpenAI-compatible endpoint, fastest open inference (hardware-accelerated via LPU chips).

**Decision:** Groq with Llama 3.3 70B via OpenAI-compatible endpoint.

**Reasoning:** Groq's free tier provides sufficient quota for development and demonstration. The OpenAI-compatible endpoint means it slots directly into Semantic Kernel's `OpenAIChatCompletion` connector with a single `base_url` override — no custom connector required. Llama 3.3 70B performs comparably to GPT-4o-mini on structured JSON extraction tasks (the primary use case for Analyst, Auditor, and Comparator agents). Groq's LPU inference is fast enough that the retrieval pipeline (cross-encoder, BM25) becomes the latency bottleneck rather than the LLM calls.

**Tradeoffs accepted:** Groq's free tier has rate limits. High-volume testing will hit them. The LLM backend is abstracted behind `core/groq_client.py` and the SK `OpenAIChatCompletion` service — swapping to OpenAI or Anthropic requires changing `core/sk_kernel.py` and `core/groq_client.py`, not the agent implementations.
