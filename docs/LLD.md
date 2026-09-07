# FinSight — Low-Level Design (LLD)

**Status:** current as of the `main` branch.
**Audience:** anyone modifying the code. Read [HLD.md](HLD.md) first for the system-level picture.
**Convention:** signatures and behaviour below are transcribed from the implementation. Where a comment says "current implementation note", the code and the higher-level prose in [ARCHITECTURE.md](ARCHITECTURE.md) diverge slightly — the code is authoritative.

---

## 1. Module Map

```
core/                         foundation — no intra-project deps except each other
  config.py                   Settings (pydantic-settings)                → everything
  models.py                   dataclasses + enums                          → everything
  sk_kernel.py                Kernel singleton, PLANNER/SYNTHESIZER prompt → agents, routes
  groq_client.py              chat_completion / chat_completion_hedged     → agents, routes
  unit_normalizer.py          deterministic ₹-crore conversion             → routes (pre-Comparator)

retrieval/                    capability — retrieval primitives
  qdrant_store.py             QdrantStore (async client wrapper)
  embedder.py                 SentenceTransformer singleton (384-dim)
  bm25.py                     BM25Retriever (rank-bm25 over payload text)
  hybrid.py                   reciprocal_rank_fusion(lists, k=60)
  reranker.py                 CrossEncoder singleton + rerank()
  confidence.py               compute_confidence() 5-signal, build_ranked_chunks()

ingestion/                    capability — PDF → chunks
  parser.py                   parse_pdf() via PyMuPDF ("dict" mode)
  metadata.py                 detect_section_type / _fiscal_year / _company + weights
  chunker.py                  chunk_document() heading-aware sliding window + dedup
  pipeline.py                 ingest_pdf() / ingest_directory()

agents/                       one module per role
  router.py                   plan_task()          — SK-native Planner call (reference)
  retriever.py                RetrieverPlugin      — @kernel_function retrieve()
  analyst.py                  AnalystPlugin        — @kernel_function analyze()
  auditor.py                  AuditorPlugin        — @kernel_function audit()
  comparator.py               ComparatorPlugin     — @kernel_function compare()
  synthesizer.py              synthesize_report()  — direct chat_completion(SYNTHESIZER_PROMPT)

api/
  main.py                     create_app(): middleware, routers, static, /health
  metrics_store.py            MetricsStore + module singleton `metrics`
  middleware/guardrails.py    GuardrailsMiddleware (pure ASGI) + detect_injection()
  routes/query.py             POST /query (blocking) + shared parse/persist helpers
  routes/query_stream.py      POST /query/stream (SSE); imports helpers from query.py
  routes/ingest.py            POST /ingest/upload
  routes/eval.py              GET /eval/collection, /eval/audit-logs[/{id}]
  routes/metrics.py           GET /metrics

observability/tracer.py       setup_tracing(), @traced decorator, optional OTel

evaluation/harness.py         run_harness(query_file) — HTTP client against a running API
```

Dependency direction is strictly downward: `api → agents → retrieval/ingestion/core`. `core` depends only on third-party packages. No cycles (`sk_kernel` imports the agent plugins lazily inside `_build_kernel()` to avoid an import cycle).

---

## 2. Foundation Layer

### 2.1 `core/config.py`

`Settings(BaseSettings)` — `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`. Single module-level instance `settings`. Fields in [§9](#9-configuration-reference).

### 2.2 `core/sk_kernel.py`

| Symbol | Type | Notes |
|---|---|---|
| `PLANNER_PROMPT` | `str` | `{user_task}` placeholder (Python `str.format`, **not** SK `{{$var}}`). Asks for "ONLY valid JSON — a list of strings", 2–6 items. |
| `SYNTHESIZER_PROMPT` | `str` | `{query} {task_id} {verified_claims} {uncertain_claims} {comparison}` placeholders. Contains the hard output rules (no arithmetic, omit unlabelled derived figures, keep `[converted from …]` labels, exact section headers). |
| `_PLANNER_PROMPT`, `_SYNTHESIZER_PROMPT` | `str` | Back-compat aliases. |
| `get_kernel()` | `-> sk.Kernel` | Lazy singleton (`_kernel`). |
| `get_fallback_kernel()` | `-> sk.Kernel \| None` | Reserve kernel; `None` if `groq_api_key` unset. |
| `reset_kernel()` | `-> None` | Clears both singletons (test hook). |
| `_build_kernel(use_fallback=False)` | `-> sk.Kernel` | Adds one `OpenAIChatCompletion` service + 4 native plugins + 2 prompt functions. |

**Endpoint selection inside `_build_kernel`:**

```
use_fallback=True                          → Groq creds/model (reserve acts as primary)
fallback_api_key & _model & _base_url set  → fallback endpoint is primary
otherwise                                  → Groq endpoint
```

Plugins registered: `Retriever`, `Analyst`, `Auditor`, `Comparator` (native). Prompt functions registered: `Planner.decompose`, `Synthesizer.synthesize` (`KernelFunctionFromPrompt`).

### 2.3 `core/groq_client.py`

Module singletons: `_primary_client / _primary_model / _reserve_client / _reserve_model`, built once by `_init_clients()`.

- If `fallback_*` all set → fallback endpoint is **primary** (timeout 45 s), Groq is **reserve** (timeout 60 s).
- Else → Groq is primary (timeout 60 s), no reserve.
- `_RETRY_DELAYS = [1.0, 2.0, 4.0]`.

| Function | Behaviour |
|---|---|
| `_call(client, model, messages, temperature, max_tokens, response_format)` | One `chat.completions.create`; returns `choices[0].message.content or ""`. |
| `chat_completion(messages, model=None, temperature=0.1, max_tokens=2048, response_format=None)` | Try primary; on `RateLimitError` / `APIStatusError(503)` retry after each delay; on `APITimeoutError` / `APIConnectionError` **break immediately** (no retry). Then try reserve once. If no reserve → `RuntimeError`. If reserve fails → `RuntimeError("Both LLM endpoints failed")`. |
| `chat_completion_hedged(messages, hedge_after=8.0, temperature=0.0, max_tokens=2048, response_format=None)` | No reserve → delegates to `chat_completion`. Else: start primary task, `asyncio.wait({primary}, timeout=hedge_after)`; if primary finished cleanly return it; otherwise start reserve, `wait(FIRST_COMPLETED)`, cancel the loser, return the winner (falling back to the other task if the winner raised). |
| `get_groq_async_client()` | Back-compat; returns reserve if present else primary. |

`response_format` is threaded through but no caller currently sets it.

---

## 3. Agent Layer

### 3.1 PlannerAgent

**Reference form** — `agents/router.py::plan_task(user_task: str) -> list[str]`:
`kernel.invoke("Planner", "decompose", KernelArguments(user_task=…))` → parse JSON list; on failure, parse a bullet/numbered list (`lstrip("-•1234567890.) ")`, keep lines >10 chars, cap 6); ultimate fallback `[user_task]`.

**As actually called in the routes** — `query.py` / `query_stream.py` build the prompt with `PLANNER_PROMPT.format(user_task=req.query)` and call `chat_completion(...)` (blocking) or `chat_completion_hedged(..., hedge_after=8.0)` (stream), then `_parse_subtasks(text, fallback=req.query)` (same parsing logic as `plan_task`). This keeps the planner call on the retry/reserve/hedge path. Both forms share the prompt template.

Params: `max_tokens=500`, `temperature=0.0`.

### 3.2 RetrieverAgent — `agents/retriever.py`

`class RetrieverPlugin` holds `QdrantStore`, a lazily-built `BM25Retriever`, the cached `_all_payloads`, an `asyncio.Lock`, and `_collection_ready`.

`_ensure_bm25()` — double-checked locking: first caller does `ensure_collection()` + `scroll_all()` + builds `BM25Retriever(self._all_payloads)`; subsequent callers (including parallel subtasks) are no-ops. **One `scroll_all` per process lifetime.**

`@kernel_function retrieve(subtask, company_filter="", fiscal_year_filter="") -> str` (JSON array). Builds `filters` dict from non-empty args, calls `retrieve_chunks(...)`, serialises each `RankedChunk` to `{chunk_id, text, source, page, section_type, company, fiscal_year, confidence}`.

`retrieve_chunks(query, store=None, top_k=None, filters=None, bm25=None, all_payloads=None) -> list[RankedChunk]`:

1. `top_k = top_k or settings.retrieval_top_k` (20).
2. If `all_payloads is None` (standalone/tests) → build corpus from scratch.
3. Empty corpus → `[]`.
4. `_payload_matches` — case-insensitive substring match; `"FY2024"` → strips leading `fy` → `"2024"`, checks both forms against the stored value.
5. If `filters` → filter `bm25_payloads` in memory; no matches → `[]`.
6. If `bm25 is None or filters` → build a fresh `BM25Retriever(bm25_payloads)` (filtered subset); else reuse the cached index.
7. `dense_filters` — only `company` is passed to Qdrant (exact `MatchValue`); `fiscal_year` is handled by the in-memory BM25 pre-filter.
8. `asyncio.gather(to_thread(bm25.search, query, top_k), store.dense_search(query_vec, top_k, dense_filters))`.
9. `reciprocal_rank_fusion([bm25_results, dense_results])`.
10. `to_thread(rerank, query, fused, top_k)`.
11. `build_ranked_chunks(reranked, all_payloads)`.

### 3.3 AnalystAgent — `agents/analyst.py`

`@kernel_function analyze(subtask: str, chunks_json: str) -> str`. Parses `chunks_json`; on `JSONDecodeError` → `{"kpis": [], "claims": []}`. Calls `analyze_chunks`.

`analyze_chunks(subtask, chunks_data) -> dict`:
- Context = first `settings.rerank_top_k` (5) chunks, each rendered as `[Source: …, Page …, Section: …, Confidence: 0.xx]\n<text>`.
- `chat_completion([system=_SYSTEM, user="Subtask: …\n\nChunks:\n…"], temperature=0.0, max_tokens=1500)`.
- Parse: strict `json.loads`; else `re.search(r"\{[\s\S]+\}", content)`; else log `analyst.parse_failed` and return empty.

`_SYSTEM` contract (extraction rules): only verbatim-stated figures; `supporting_text` must be a verbatim excerpt; **no derived figures** (per-unit, computed ratios, unit/currency conversions, cross-company combinations); a printed ratio may be extracted only if `supporting_text` contains the exact printed phrase; `confidence` hint 0.9 audited / 0.75 notes / 0.6 mda|letter; ≤5 claims per call. Output JSON: `{kpis:[{metric,value,period,company}], claims:[{claim,supporting_text,source_doc,page,confidence,section_type}]}`.

### 3.4 AuditorAgent — `agents/auditor.py`

`@kernel_function audit(claims_json, confidence_threshold="0.65", original_query="") -> str`. Parse `claims_json`; on failure → `{"verified":[],"uncertain":[],"unverifiable":[]}`. Calls `audit_claims(claims, {}, threshold, original_query=…)`. Serialises `{verified:[to_dict], uncertain:[to_dict], unverifiable:[claim strings]}`.

`audit_claims(claims, chunks_by_subtask, confidence_threshold=None, original_query="") -> (verified, uncertain, unverifiable)`:
- Split: claims with no `supporting_text` → **instant unverifiable** (`Citation.snippet=""`, `confidence=0.0`, reason `"No supporting snippet provided"`).
- The rest → `_batch_entailment(...)` — **one** LLM call for all claims.
- Map each verdict to `AuditStatus`; unknown status string → `UNCERTAIN`. Build `Citation` from `source_doc / page / supporting_text[:500] / confidence / section_type`.

`_batch_entailment(claims_data, threshold, original_query="") -> list[dict]`:
- Payload = JSON array of `{index, user_query, claim, snippet(≤400), confidence(2dp), threshold}`.
- `chat_completion([system=_BATCH_SYSTEM, user=payload], temperature=0.0, max_tokens=min(200*N, 4096))`.
- Strip ```` ``` ```` fences; try `json.loads`; else regex `\[[\s\S]+\]`. If the list is short, **pad** with `{"audit_status":"uncertain","reason":"Missing verdict"}`; truncate to `len(claims_data)`. Total parse failure → all `uncertain`, reason `"Batch parse failed"`.

`_BATCH_SYSTEM` — two ordered checks:
- **Check 1 — fabricated-event detection** (fires rarely; all three must hold): query has an explicit transaction/statement action verb (`acquiring`, `merged with`, `announced partnership with`, …) directed at a named external company; that company is **not** Infosys/TCS/Wipro/HCL/Tech Mahindra; the claim is about an unrelated topic. → `unverifiable`.
- **Check 2 — snippet entailment:** `verified` = snippet states/implies the claim **and** `confidence >= threshold`; `uncertain` = weak/partial support **or** `confidence < threshold`; `unverifiable` = snippet absent/irrelevant/contradictory.
- Output: JSON array, one `{audit_status, reason}` per claim, **same order** as input.

### 3.5 ComparatorAgent — `agents/comparator.py`

`@kernel_function compare(subtask_results_json, original_query) -> str`. Parse (`[]` on failure) → `compare_results`.

`compare_results(subtask_results, original_query) -> dict`:
- Empty → `{deltas:[], cross_document_claims:[], summary:"No data retrieved."}`.
- `combined = json.dumps(subtask_results, indent=2)[:6000]`.
- `chat_completion([system=_SYSTEM, user="Query: …\n\nSubtask Results:\n…"], temperature=0.0, max_tokens=2000)`.
- Parse: `json.loads`; else regex `\{[\s\S]+\}`; else `{deltas:[], cross_document_claims:[], summary: content[:500]}`.

`_SYSTEM` contract: **no arithmetic of any kind** (per-unit, currency conversion, scale conversion, % change, any quotient/product/sum/difference); **no cross-company arithmetic**; unit normalisation is **already done** — copy `[converted from …]` labels verbatim, never recompute; missing value → `"N/A — not in retrieved data"`; a delta row requires both values in the **same unit and currency**; summary and `cross_document_claims` are verbatim passthrough only. `anomaly=true` when a comparable figure deviates >15%, or units/currencies mismatch, or a figure contradicts an auditor statement, or (explicitly) when a USD→₹ conversion was applied (`at ₹84/USD` in the label). Output JSON: `{deltas:[{metric, company_a, value_a, period_a, source_a, company_b, value_b, period_b, source_b, delta, anomaly, anomaly_reason}], cross_document_claims:[{claim, sources[], pages[]}], summary}`.

### 3.6 SynthesizerAgent — `agents/synthesizer.py`

`synthesize_report(query, verified_claims, uncertain_claims, comparison, task_id) -> str`:
- No verified and no uncertain claims → fixed "Insufficient evidence found. …" string (no LLM call).
- Else `SYNTHESIZER_PROMPT.format(query, task_id, verified_claims=json[:3000], uncertain_claims=json[:1000], comparison=json[:1500])` → `chat_completion([user=prompt], max_tokens=2048, temperature=0.1)`.
- Output sections (exact headers): `## Executive Summary`, `## Key Findings`, `## Comparative Analysis`, `## Risk Flags`. Uncertain claims inline-prefixed `[UNCERTAIN]`.

---

## 4. Data Models (`core/models.py`)

### 4.1 Enums

| Enum | Members (value) |
|---|---|
| `SectionType(str, Enum)` | `AUDITED_FINANCIALS="audited_financials"`, `MDA="mda"`, `NOTES="notes"`, `LETTER="letter"`, `UNKNOWN="unknown"` |
| `AuditStatus(str, Enum)` | `VERIFIED="verified"`, `UNCERTAIN="uncertain"`, `UNVERIFIABLE="unverifiable"` |

### 4.2 Dataclasses

**`Citation`** — `document: str`, `page: int`, `snippet: str` (verbatim, callers cap at 500), `claim: str`, `confidence: float [0,1]`, `section_type: str`. `to_dict()`.

**`Chunk`** — `chunk_id, doc_id, source, text, page, section, section_type: SectionType, token_count, fiscal_year="", company=""`. `to_payload()` → dict with `section_type.value`; `from_payload(dict)` inverse (defaults `section_type` to `"unknown"`).

**`RankedChunk`** — `chunk: Chunk`, `retrieval_score: float` (cross-encoder), `confidence_score: float` (5-signal). `to_citation(claim="")` → `Citation(document=chunk.source, page=chunk.page, snippet=chunk.text[:500], confidence=confidence_score, section_type=chunk.section_type.value)`.

**`AuditedClaim`** — `claim: str`, `citation: Citation`, `audit_status: AuditStatus`, `audit_reason: str`. `to_dict()`.

**`SubtaskResult`** — `subtask, ranked_chunks: list[RankedChunk], kpis: list[dict], claims: list[AuditedClaim], agents_used: list[str] = []`. (Defined for completeness; the routes pass plain dicts of the same shape.)

**`AuditLog`** — `task_id, timestamp` (`"%Y-%m-%dT%H:%M:%SZ"` UTC), `user_query, plan: list[str], retrievals: dict[str, list[str]]` (subtask → chunk_ids), `claims: list[dict]` (verified+uncertain, `to_dict`), `flagged_uncertain: list[str]` (claim texts), `blocked_unverifiable: list[str]`, `agents_invoked: list[str]` (de-duplicated, order preserved), `latency_ms: int`. `to_dict()`.

**`AnalysisReport`** — `task_id, query, summary: str` (Markdown), `verified_claims: list[AuditedClaim]`, `uncertain_claims: list[AuditedClaim]`, `audit_log: AuditLog`. `to_dict()` → API response body.

---

## 5. Retrieval Layer

### 5.1 `retrieval/qdrant_store.py`

`VECTOR_SIZE = 384`. `QdrantStore` wraps `AsyncQdrantClient(host, port)`, collection = `settings.qdrant_collection`.

| Method | Notes |
|---|---|
| `ensure_collection()` | Create with `VectorParams(size=384, distance=COSINE)` if absent. |
| `upsert_chunks(chunks, embeddings)` | `PointStruct(id=abs(hash(chunk_id)) % 10**15, vector=emb, payload=chunk.to_payload())`. |
| `dense_search(query_vector, top_k=20, filters=None)` | `filters` dict → `Filter(must=[FieldCondition(key, MatchValue(value))])`. Returns `[{score, payload}]`. |
| `scroll_all(batch_size=100)` | Paginates `scroll` until `next_offset is None`; returns all payloads (no vectors). |
| `delete_by_doc_id(doc_id)` | Filter delete on `doc_id`. |
| `collection_info()` | `{name, vectors_count (or points_count), points_count, status}`. |

### 5.2 `retrieval/embedder.py` / `retrieval/reranker.py`

- `get_embedder()` → lazy `SentenceTransformer(settings.embedding_model)`. `embed_query(str) -> list[float]` and `embed_texts(list[str])` both `normalize_embeddings=True`.
- `get_reranker()` → lazy `CrossEncoder(settings.reranker_model)`. `rerank(query, candidates, top_k=None)` — builds `(query, text)` pairs, `predict`, sorts desc, returns `top_k or settings.rerank_top_k` items each with an added `rerank_score`.

### 5.3 `retrieval/bm25.py`

`_tokenize(text)` → `text.lower()` then `re.findall(r"\b[a-z0-9][a-z0-9.%]*", text)` — **`%` and `.` are kept inside tokens** so `"20.7%"` and `"fy2024"` survive as single tokens.

`BM25Retriever(payloads)` builds `BM25Okapi([_tokenize(p["text"]) for p in payloads])`. `search(query, top_k=20)` → `get_scores(tokens)`, sort desc, keep `score > 0`, return `[{score, payload}]`.

### 5.4 `retrieval/hybrid.py`

```
reciprocal_rank_fusion(ranked_lists, k=60):
    for each list, for rank, item in enumerate(list, start=1):
        chunk_id = payload["chunk_id"]  (fallback: hash of text[:50])
        scores[chunk_id] += 1.0 / (k + rank)
    return items sorted by score desc  → [{score, payload}]
```

Scale-invariant: only ranks matter, so BM25's unbounded scores and cosine's `[0,1]` never need reconciling. `k=60` from Cormack et al. 2009.

### 5.5 `retrieval/confidence.py`

`compute_confidence(payload, retrieval_score, rerank_score=None, all_payloads_for_consistency=None, query_variants_hit=1) -> float`:

| Signal | Weight | Computation |
|---|---|---|
| `retrieval` | **0.35** | `base = clamp(rerank_score if not None else retrieval_score, 0, 1)` |
| `section` | **0.25** | `section_type_confidence_weight(SectionType(payload.section_type))` |
| `freshness` | **0.15** | `_freshness_score(fiscal_year)` |
| `cross_filing` | **0.15** | `_cross_filing_score(payload, all_payloads)` |
| `consistency` | **0.10** | `min(1.0, query_variants_hit / 3.0)` — currently always `1/3 ≈ 0.333` (no caller passes >1) |

`composite = Σ weight·signal`, `round(…, 4)`.

- `_freshness_score(fy)` — empty → `0.5`; for year in `2025..2019`, if `str(year) in fy` → `max(0.2, 1.0 - 0.2·(2025-year))`; no match → `0.4`.
- `_cross_filing_score(payload, all)` — no `all` or no `company` → `0.5`; count distinct `doc_id` for the same company: `≥3 → 1.0`, `==2 → 0.75`, else `0.4`.
- `section_type_confidence_weight` (in `ingestion/metadata.py`): **AUDITED_FINANCIALS 1.0 · NOTES 0.85 · MDA 0.65 · LETTER 0.40 · UNKNOWN 0.50**. *(ARCHITECTURE.md's older table lists different values; these are the live weights.)*

`build_ranked_chunks(fused_results, all_payloads=None)` → for each item: `retrieval_score = item.get("rerank_score", item.get("score", 0.0))`, `compute_confidence(...)`, `Chunk.from_payload(payload)` → `RankedChunk`.

---

## 6. Ingestion Layer

### 6.1 `ingestion/parser.py`

- `parse_pdf(path) -> list[{page_num (1-idx), width, height, blocks}]` — PyMuPDF `page.get_text("dict")`.
- `extract_text_with_positions(path)` — spans with `bbox / size / flags / page` (available for snippet localisation; not on the ingest hot path).
- `get_page_text(path, page_num)` — plain text of one 1-indexed page.

### 6.2 `ingestion/metadata.py`

Regex lists (case-insensitive): `_AUDITED_PATTERNS`, `_MDA_PATTERNS`, `_NOTES_PATTERNS`, `_LETTER_PATTERNS`.

- `detect_section_type(text, section_heading="") -> SectionType` — check audited patterns first; **if `"note"` appears in the combined string, downgrade audited → NOTES**; then notes, then MDA, then letter; default `UNKNOWN`.
- `detect_fiscal_year(text) -> str` — `\b(fy|fiscal year|year ended)\s*(20\d{2}[-/]?\d{0,2})\b`; normalises `/`/spaces to `-`. Fallback: first of `2024..2020` found as a bare substring; else `""`.
- `detect_company(source_filename, text="") -> str` — lower-cases and replaces `_`/`-` with spaces over `filename + text[:200]`; `_COMPANY_HINTS`: `infosys→Infosys`, `tcs→TCS`, `tata consultancy→TCS`, `wipro→Wipro`; else `""`.
- `section_type_confidence_weight(section_type) -> float` — the weight table in [§5.5](#55-retrievalconfidencepy).

### 6.3 `ingestion/chunker.py` — heading-aware sliding window

Helpers: `_is_heading(span, body_size)` → `span["size"] > body_size * 1.2`. `_body_size(blocks)` → modal span size (`Counter.most_common(1)`), default `12.0`. `_block_text(block)` joins line/span text. `_make_chunk_id(doc_id, page, index, text)` → `"{doc_id}_p{page}_c{index}_{sha256(f'{doc_id}:{page}:{index}:{text[:100]}')[:12]}"`.

`chunk_document(pages, doc_id, source, target_tokens=None, overlap_tokens=None) -> list[Chunk]`:
- `target_tokens = settings.chunk_size` (400), `overlap_tokens = settings.chunk_overlap` (80).
- `company` from filename; `fiscal_year` from the concatenation of all block text.
- **Per page**, reset `current_tokens=[]`, `current_section="unknown"`, `chunk_index=0`. For each text block (`type==0`, has lines):
  - `is_head = _is_heading(first_span, body_size)`.
  - If `is_head` **and** `len(current_tokens) >= overlap_tokens` → **flush** the current buffer as a chunk, then keep only the last `overlap_tokens` tokens as the seed of the next chunk, set `current_section = heading text`.
  - Else: if `is_head`, update `current_section`; extend `current_tokens` with the block's words; while `len(current_tokens) >= target_tokens + overlap_tokens`, emit a `target_tokens`-token window and advance by `target_tokens - overlap_tokens` (→ 80-token overlap).
  - At page end, flush any remaining `current_tokens`.
- Return `_deduplicate(chunks)`.

`_deduplicate(chunks)` — drop a chunk if `sha256(chunk.text[:200])` was already seen. *(Content-prefix hash — this is the effective dedup key in code; the `chunk_id` string also embeds `text[:100]`.)*

`token` == whitespace-split word (no sub-word tokeniser).

### 6.4 `ingestion/pipeline.py`

`ingest_pdf(pdf_path, doc_id=None, overwrite=True) -> dict`:
1. `FileNotFoundError` if missing. `doc_id = doc_id or path.stem.lower().replace(" ", "_")`.
2. `parse_pdf` → `chunk_document(pages, doc_id, source)`.
3. No chunks → `{doc_id, chunks_indexed: 0, status: "empty"}`.
4. `get_embedder().encode([c.text …], normalize_embeddings=True, show_progress_bar=True)`.
5. `QdrantStore().ensure_collection()`; if `overwrite` → `delete_by_doc_id(doc_id)`; `upsert_chunks`.
6. → `{doc_id, source, chunks_indexed, status: "success", company, fiscal_year}` (company/fy from `chunks[0]`).

`ingest_directory(directory, pattern="*.pdf")` — sequential `ingest_pdf` per file; per-file errors captured as `{file, status: "error", error}`.

---

## 7. `core/unit_normalizer.py` — deterministic ₹-crore conversion

**Purpose:** remove every reason for an LLM to do arithmetic. All math is `Decimal`.

Constants: `USD_TO_INR = Decimal("84")`. `_USD_BN_TO_CRORE = 84 * 100` (`$1 bn = ₹84 bn = ₹8,400 cr`).

Scale tables (multiplier to reach the common unit):

| INR word (`_INR_SCALE`, → crore) | | USD word (`_USD_SCALE`, → USD-billion) |
|---|---|---|
| `crore(s)/cr` = 1 | | `trillion(s)/tn` = 1000 |
| `lakh(s)/lac(s)` = 0.01 | | `billion(s)/bn` = 1 |
| `million(s)/mn` = 0.1 | | `million(s)/mn` = 0.001 |
| `billion(s)/bn` = 100 | | `thousand(s)` = 0.000001 |
| `thousand(s)` = 0.001 | | |

Regexes: `_NUM = r"[0-9][0-9,]*(?:\.[0-9]*)?"`. `_INR_RE` matches `(₹|Rs.?|INR)\s*<num>\s*<inr-scale-word>?`. `_USD_RE` matches `(USD|\$)\s*<num>\s*<usd-scale-word>?`. Case-insensitive.

`normalize_text(text) -> str` = `_replace_usd(_replace_inr(text))`:
- **INR:** unknown scale word → unchanged. Multiplier `1` (already crore) → rebuilt as `"₹<num> crore"` (whitespace normalised only). Otherwise → `"₹<fmt(value)> crore [converted from ₹<num> <scale>]"`.
- **USD:** no scale word → **left unchanged** (assumed not a revenue-scale figure). Unknown scale word → unchanged. Otherwise → `"₹<fmt(value)> crore [converted from $<num> <scale> at ₹84/USD, approx]"`.
- `_fmt(Decimal)` — `quantize(0.01, ROUND_HALF_UP)`, thousands separators, strip trailing zeros / dot.

`normalize_subtask_results(subtask_results) -> list[dict]` — shallow copy; for each result, `normalize_text` the `claim` and `supporting_text` fields of every claim and the `value / label / description` fields of every KPI. Originals not mutated.

Worked examples (from the module docstring):

| Input | Output |
|---|---|
| `$30 billion` | `₹2,52,000 crore [converted from $30 billion at ₹84/USD, approx]` |
| `₹10,478 million` | `₹1,047.8 crore [converted from ₹10,478 million]` |
| `₹1,78,650 crore` | `₹1,78,650 crore` (unchanged) |
| `growth of 15.2%` | unchanged |
| `270,000 employees` | unchanged |

---

## 8. API Layer

### 8.1 App assembly — `api/main.py`

`create_app()` → `FastAPI(title="FinSight", version="1.0.0", lifespan=…)`. Middleware order: `CORSMiddleware(allow_origins=["*"], methods=["*"], headers=["*"])`, then `GuardrailsMiddleware`. Routers: `ingest, query, query_stream, eval, metrics`. Static: `/ui` (index.html, `no-store`), `/ui/*` mounted `StaticFiles(html=True)`, `/dashboard` (dashboard.html), `/` → redirect `/ui`. `/health` → `{"status":"ok","model": settings.groq_model}`. `lifespan` calls `setup_tracing()` on startup.

### 8.2 `POST /query` — `api/routes/query.py`

Request `QueryRequest`: `query: str`, `company_filter: str|None`, `fiscal_year_filter: str|None`, `confidence_threshold: float|None`.

Empty `query` → `HTTPException(400)`. Otherwise runs the six stages ([§5.3 of HLD](HLD.md#53-request-flow--post-query-blocking)); `metrics.record_start()` at entry, `metrics.record_agent_latency(...)` per stage, `_save_audit_log`, `metrics.record_complete`, returns `AnalysisReport.to_dict()`.

Shared helpers (also imported by the stream route):

| Helper | Behaviour |
|---|---|
| `_parse_subtasks(content, fallback) -> list[str]` | JSON list of strings; else bullet/number parse (>10 chars, ≤6); else `[fallback]`. |
| `_parse_audit_result(json) -> (verified, uncertain, unverifiable)` | `_deserialize_claims` on `verified` / `uncertain`; `unverifiable` passed through as strings. |
| `_safe_confidence(raw) -> float` | `float()` guard; NaN / out-of-range → clamp to `[0,1]` or `0.5`. |
| `_deserialize_claims(list[dict]) -> list[AuditedClaim]` | Build `Citation` + `AuditedClaim`; unknown `audit_status` → `uncertain`; per-item exceptions skipped. |
| `_save_audit_log(AuditLog)` | `mkdir(exist_ok=True)`, write `audit_logs/<task_id>.json` (`indent=2`). |

### 8.3 `POST /query/stream` — `api/routes/query_stream.py`

`StreamingResponse(generate(), media_type="text/event-stream", headers={Cache-Control: no-cache, X-Accel-Buffering: no})`. `_event(name, data)` → `data: {json}\n\n`.

Differences vs. blocking: hedged planner (`hedge_after=8.0`); `_run_subtask(kernel, subtask, company_filter, fiscal_year_filter)` returns `(events, result|None)` and is `gather`-ed; **Auditor + Comparator via `asyncio.gather(_audit(), _compare(), return_exceptions=True)`**; each stage yields an event; auditor exception → `error` event + return; comparator exception → `error` event + empty comparison; final `done` carries `AnalysisReport.to_dict()`. Event sequence: `start, planned, retrieved*, analyzed*, audited, compared, done | error`.

### 8.4 Other routes

| Route | Handler | Response |
|---|---|---|
| `POST /ingest/upload` | `ingest.py` | Rejects non-`.pdf` (400). Writes upload to a `NamedTemporaryFile`, `ingest_pdf(tmp, doc_id=doc_id or filename.stem, overwrite=…)`, unlinks temp, returns the pipeline dict. |
| `GET /eval/collection` | `eval.py` | `QdrantStore().collection_info()` or `{"error": …}`. |
| `GET /eval/audit-logs` | `eval.py` | `{logs: [newest 20 filenames], total}`. |
| `GET /eval/audit-logs/{id}` | `eval.py` | Parsed JSON of `<id>` or `<id>.json`; `{"error": "Log not found"}` otherwise. |
| `GET /metrics` | `metrics.py` | `metrics.snapshot()`. |

### 8.5 `GuardrailsMiddleware` — `api/middleware/guardrails.py`

Pure ASGI (not `BaseHTTPMiddleware`) — deliberately, because `BaseHTTPMiddleware` captures `receive` in a closure and breaks SSE disconnect handling ("Unexpected message received: http.request").

`__call__`: pass through unless `scope["type"]=="http"` and method in `("POST","PUT")`. Buffer the whole body (`while more_body`). `detect_injection(body.decode(errors="ignore"))` → `JSONResponse(400, {"error":"Request contains disallowed content"})`. Otherwise wrap downstream `receive` so the **first** call replays the buffered body and **all subsequent** calls delegate to the real `receive` (so `http.disconnect` still reaches the disconnect listener).

`_INJECTION_PATTERNS` (compiled, `IGNORECASE`): `ignore (all|previous|above)? instructions`, `forget (everything|all|your instructions)`, `you are now`, `jailbreak`, `disregard.*prompt`, `</?script`, `system:\s*you`, `\[system\]`.

### 8.6 `MetricsStore` — `api/metrics_store.py`

Single process, GIL-protected (no lock). Bounded `deque`s: `_latency_values` (200), per-agent timings (100 each for the six agents), `_recent` (50), `_errors` (20). `LATENCY_BUCKETS`: `<500ms, 500ms–1s, 1s–2s, 2s–5s, 5s–10s, >10s`.

| Method | Effect |
|---|---|
| `record_start()` | `_inflight += 1` |
| `record_complete(task_id, query, latency_ms, verified, uncertain, blocked)` | decrement inflight; bump totals; append latency; bucket it; push `QueryRecord`. |
| `record_error(task_id, query, stage, detail)` | decrement inflight; `_total_errors += 1`; push error + errored `QueryRecord`. |
| `record_agent_latency(agent, ms)` | append if `agent` is one of the six known keys. |
| `snapshot()` | `{uptime_s, total_queries, total_errors, inflight, error_rate, latency:{avg,p50,p95,min,max,buckets}, claims:{verified,uncertain,blocked,total,*_rate}, agents:{<name>:{avg_ms,p95_ms,min_ms,max_ms,count}}, recent_queries[≤20], recent_errors[≤10]}`. Percentiles from the sorted recent window (`vals[int(len·0.95)]`). |

Module singleton `metrics` imported by the routes.

---

## 9. Configuration Reference

`core/config.py::Settings` — env keys are the upper-cased field names; `.env` is loaded, unknown keys ignored.

| Field | Default | Purpose |
|---|---|---|
| `groq_api_key` | `""` | Groq / primary key (required in practice). |
| `groq_model` | `llama-3.3-70b-versatile` | Primary model id. |
| `groq_base_url` | `https://api.groq.com/openai/v1` | Primary endpoint. |
| `fallback_api_key` / `fallback_model` / `fallback_base_url` | `""` | Reserve LLM (any OpenAI-compatible). If all three are set, this becomes **primary** and Groq becomes the reserve. |
| `qdrant_host` | `localhost` (`qdrant` in compose) | Qdrant host. |
| `qdrant_port` | `6333` | Qdrant HTTP port. |
| `qdrant_collection` | `finsight_chunks` | Collection name. |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder (384-dim). |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Cross-encoder. |
| `chunk_size` | `400` | Target tokens per chunk. |
| `chunk_overlap` | `80` | Overlap tokens (and heading-flush threshold). |
| `retrieval_top_k` | `20` | Candidates from BM25 and dense each. |
| `rerank_top_k` | `5` | Chunks kept after rerank; also the Analyst context window. |
| `confidence_threshold` | `0.65` | Auditor VERIFIED cutoff. |
| `hallucination_fallback_threshold` | `0.50` | UNCERTAIN floor (below → UNVERIFIABLE). |
| `audit_log_dir` | `audit_logs` | Audit-log output directory. |
| `otel_enabled` | `false` | Enable OTLP span export. |
| `otel_endpoint` | `http://localhost:4317` | OTLP gRPC endpoint. |
| `log_level` | `INFO` | structlog filtering level. |
| `log_format` | `text` | `text` (coloured) or `json` (NDJSON; also forced when `otel_enabled`). |

---

## 10. Error Handling Matrix

| Location | Failure | Handling | User-visible result |
|---|---|---|---|
| `groq_client.chat_completion` | 429 / 503 | retry `[1,2,4]s` then reserve | transparent if a retry/reserve succeeds |
| `groq_client.chat_completion` | timeout / connection | no retry → reserve | transparent if reserve succeeds |
| `groq_client` (both fail) | — | `RuntimeError` | HTTP 500 (blocking) / SSE `error` |
| `query.py::_run_subtask_sync` | Retriever raises / no chunks | return `None` | that subtask contributes nothing |
| `analyst.analyze_chunks` | bad JSON | regex salvage → `{kpis:[],claims:[]}` | fewer/no claims for the subtask |
| `auditor._batch_entailment` | bad JSON / short list | pad + truncate → `uncertain` | claims land in UNCERTAIN, not dropped |
| `comparator.compare_results` | bad JSON | regex salvage → empty comparison | report has no "Comparative Analysis" content |
| `synthesizer.synthesize_report` | no claims at all | fixed "insufficient evidence" string | honest no-answer |
| `query_stream.generate` | Planner / Auditor / Synth raises | `metrics.record_error` + `error` event + stop | SSE `error`, stream closes |
| `GuardrailsMiddleware` | injection pattern | `JSONResponse(400)` before handler | HTTP 400 |
| `ingest.ingest_upload` | non-PDF | `HTTPException(400)` | HTTP 400 |
| `pipeline.ingest_pdf` | file missing | `FileNotFoundError` | HTTP 500 |
| `retrieve_chunks` | empty collection | `log.warning` → `[]` | "insufficient evidence" report |

---

## 11. Extension Points

| Want to… | Change |
|---|---|
| Swap the LLM provider | `core/config.py` (`groq_*` or `fallback_*`) — no agent code changes; SK service + `groq_client` both read `settings`. |
| Add an agent | New `@kernel_function` plugin class → register in `core/sk_kernel.py::_build_kernel` → wire a `kernel.invoke` call into the route pipeline. |
| Add a retrieval signal | Extend `compute_confidence` (add a weight, keep the sum at 1.0) in `retrieval/confidence.py`. |
| Support another issuer | Extend `_COMPANY_HINTS` in `ingestion/metadata.py`; check heading regexes cover its filing style. |
| Change chunk sizing | `CHUNK_SIZE` / `CHUNK_OVERLAP` env vars. |
| New currency / FX rate | `core/unit_normalizer.py` — add a scale table + regex; `USD_TO_INR` is a single `Decimal` constant. |
| Tune hallucination strictness | `CONFIDENCE_THRESHOLD` / `HALLUCINATION_FALLBACK_THRESHOLD`; or the `_BATCH_SYSTEM` prompt in `agents/auditor.py`. |
| Add an injection pattern | `_INJECTION_PATTERNS` in `api/middleware/guardrails.py`. |
| Export traces | `OTEL_ENABLED=true`, `OTEL_ENDPOINT=<otlp-grpc>`. |

---

## 12. Testing Map

`tests/` (pytest, `asyncio_mode=auto`, `testpaths=["tests"]`) — 102 unit tests, no Qdrant/Groq required:

| File | Covers |
|---|---|
| `test_models.py` | dataclass round-trips, enum values, `to_dict` / `from_payload`. |
| `test_chunker.py` | `ingestion.chunker._deduplicate`, `_make_chunk_id`. |
| `test_ingestion.py` | `ingestion/metadata` detectors and section weights. |
| `test_retrieval.py` | `bm25._tokenize`, `BM25Retriever`, `reciprocal_rank_fusion`. |
| `test_confidence.py` | `compute_confidence` signal math and clamping. |
| `test_unit_normalizer.py` | INR/USD conversion, label format, no-op cases (50 cases). |

End-to-end: `evaluation/harness.py` runs query files (`happy_path.json`, `adversarial.json`) against a **running** API over HTTP and scores each case by `expected_behavior`; results → `evaluation/results/harness_<ts>.json` (git-ignored). CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, and `pytest tests/` on push/PR to `main`.
