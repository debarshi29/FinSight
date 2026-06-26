# FinSight — Design Decisions

## 2026-06-26 — Why Semantic Kernel over LangChain

**Context:** Needed an orchestration framework for multi-agent financial analysis with dynamic task decomposition.

**Options considered:**
- LangGraph: graph-based, explicit topology, battle-tested
- Semantic Kernel (SK): planner-based, dynamic ordering via LLM, Microsoft-backed

**Decision:** Semantic Kernel with OpenAIChatCompletion configured against Groq's OpenAI-compatible endpoint.

**Reasoning:** FinSight's core value is dynamic task decomposition — a user asking "compare margins FY2022-2024" should get a different plan than "summarise attrition risk." SK's planner-based model achieves this without hardcoding a graph topology. The kernel manages context passing between plugin function calls natively.

**Tradeoffs accepted:** SK Python is less battle-tested than LangGraph. More ceremony for plugin registration. Planner can generate invalid plans on ambiguous queries (mitigated by fallback parsing in PlannerAgent).

---

## 2026-06-26 — RRF over weighted score combination for hybrid retrieval

**Context:** Combining BM25 keyword scores and dense cosine similarity scores into a single ranked list.

**Options considered:**
- Weighted linear combination: score = α×bm25 + β×dense
- Reciprocal Rank Fusion (RRF): score = Σ 1/(k + rank_i)

**Decision:** RRF with k=60 (Cormack et al. 2009 default).

**Reasoning:** BM25 and cosine similarity operate on completely different scales (BM25 can be arbitrarily large; cosine is bounded [0,1]). Normalising both is fragile and corpus-dependent. RRF is scale-invariant — it operates on ranks, not scores. Empirically outperforms weighted combination in most IR benchmarks.

**Tradeoffs accepted:** RRF loses score magnitude information. Two documents ranked #1 and #2 contribute almost equally regardless of how much better #1 is. Acceptable because the cross-encoder reranker rescores the top-k pool using full joint attention anyway.

---

## 2026-06-26 — Two-stage retrieval: bi-encoder + cross-encoder

**Context:** Needed accurate relevance scoring at query time.

**Options considered:**
- Vector similarity only (bi-encoder): fast, O(1) with ANN index
- Cross-encoder only: accurate but O(n) — too slow for large corpora
- Two-stage: bi-encoder retrieves candidates, cross-encoder reranks

**Decision:** Two-stage with all-MiniLM-L6-v2 (bi-encoder) + ms-marco-MiniLM-L6-v2 (cross-encoder).

**Reasoning:** Bi-encoders encode query and document independently — fast for ANN search. Cross-encoders encode them jointly with full attention — far more accurate but O(n) at query time. The two-stage pattern gives us both: fast candidate retrieval, accurate final ranking.

**Tradeoffs accepted:** Two model loads at startup. Cross-encoder adds ~200-500ms latency per query depending on candidate pool size. Acceptable for compliance-grade analysis where correctness > speed.

---

## 2026-06-26 — AuditorAgent as separate structural pass

**Context:** Needed to prevent hallucinated claims from appearing in final reports.

**Options considered:**
- Prompt instruction: "only answer from provided context"
- Separate AuditorAgent performing entailment checking

**Decision:** Separate AuditorAgent with three-tier status (VERIFIED / UNCERTAIN / UNVERIFIABLE).

**Reasoning:** Prompt instructions can be overridden by strong model priors or adversarial inputs. A separate agent performing "does this snippet entail this claim?" is a structural check that runs after all claims are assembled. UNVERIFIABLE claims are blocked before SynthesizerAgent — they cannot appear in the report regardless of what other agents produced.

**Tradeoffs accepted:** Additional LLM call per claim. Increases latency and cost. Justified because the alternative (hallucinated claims in a compliance report) is worse than no output.

---

## 2026-06-26 — Composite confidence threshold at 0.65

**Context:** Needed a threshold below which claims are flagged UNCERTAIN rather than VERIFIED.

**Options considered:**
- Binary pass/fail at 0.5
- Three-tier at 0.65 / 0.50
- Configurable per section type

**Decision:** Configurable composite threshold defaulting to 0.65. Claims below 0.50 are UNVERIFIABLE (blocked). Claims 0.50–0.65 are UNCERTAIN (flagged). Claims above 0.65 are VERIFIED (included unmarked).

**Reasoning:** Binary pass/fail loses useful signal. An uncertain claim with explicit flagging is more valuable to a compliance analyst than a silent exclusion. The 0.65 threshold was calibrated against the five-signal composite model where audited_financials + recent + consistent retrieval naturally scores above 0.65.

**Tradeoffs accepted:** Threshold is somewhat arbitrary. DECISIONS.md documents it so it can be adjusted empirically using the eval harness.

---

## 2026-06-26 — Sliding window chunker with 400-token target, 80-token overlap

**Context:** Needed to chunk financial PDFs for vector indexing.

**Options considered:**
- Fixed-size chunks (no overlap): simple but splits mid-sentence
- Sentence-level chunks: natural boundaries but variable size
- Sliding window with heading-aware splitting + overlap

**Decision:** Heading-aware sliding window: new chunk at each detected heading, overflow chunked at 400 tokens with 80-token overlap.

**Reasoning:** Financial filings have strong heading structure (Income Statement, Notes to Accounts). Respecting headings keeps semantically related content together. 80-token overlap ensures figures that straddle chunk boundaries are retrievable from either chunk.

**Tradeoffs accepted:** Heading detection uses font-size heuristics which can fail on poorly formatted PDFs. Deduplication by SHA-256 hash prevents duplicate chunks from appearing in the index.
