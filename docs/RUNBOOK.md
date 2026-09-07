# FinSight — Operations Runbook

Practical procedures for running, observing, and recovering the system. Design rationale lives in [HLD.md](HLD.md) / [LLD.md](LLD.md); this file is task-oriented.

---

## 1. Topology recap

| Process | Port | State it owns |
|---|---|---|
| `finsight_api` (uvicorn, **1 worker**) | 8000 | SK kernel singleton, BM25 corpus cache, `MetricsStore`, loaded MiniLM models — all in-process, all lost on restart. |
| `finsight_qdrant` | 6333 (HTTP), 6334 (gRPC) | On-disk vectors + payload under `./qdrant_storage`. |
| LLM endpoint(s) | external | Stateless. Primary = Groq unless `FALLBACK_*` set (then fallback is primary, Groq is reserve). |

Single worker is deliberate — in-process metrics and the BM25 cache are not shared across workers. Do not raise `--workers`.

---

## 2. Start / stop

### Full stack (Docker)

```bash
cp .env.example .env          # then set GROQ_API_KEY
docker compose up --build -d   # qdrant starts first, API waits for its healthcheck
docker compose ps              # both should read "healthy"
docker compose logs -f api
docker compose down            # stop; add -v to also wipe qdrant_storage
```

Ready signal in the logs: `{"event": "finsight.startup", ...}`.

### Local dev (API on host)

```bash
docker compose up qdrant -d
uv sync
uvicorn api.main:app --reload      # http://localhost:8000
```

### Endpoints

| URL | Purpose |
|---|---|
| `/ui` | Query interface + live pipeline visualiser |
| `/dashboard` | Metrics dashboard |
| `/docs` | Swagger |
| `/health` | Liveness — `{"status":"ok","model":"..."}` |
| `http://localhost:6333/dashboard` | Qdrant UI |

---

## 3. First-run expectations

- On first query/ingest the API downloads `all-MiniLM-L6-v2` (~90 MB) and `ms-marco-MiniLM-L6-v2` (~67 MB) from HuggingFace. The container healthcheck has a 60 s `start-period` to cover this; a cold first request can take noticeably longer.
- The first query in a process also triggers a one-time `scroll_all` of the Qdrant collection to build the BM25 index (`retriever.bm25_built` log event, with `corpus_size`). Subsequent queries reuse it.

---

## 4. Ingestion

```bash
# one file
curl -X POST http://localhost:8000/ingest/upload -F "file=@data/filings/Infosys_Annual_Report_FY2024.pdf"

# a directory
for f in data/filings/*.pdf; do
  curl -s -X POST http://localhost:8000/ingest/upload -F "file=@$f"; echo " <- $f"
done
```

Response shape: `{doc_id, source, chunks_indexed, status, company, fiscal_year}`.

- `status: "empty"` → the PDF parsed to zero chunks. Almost always a scanned/image PDF (no extractable text blocks) or a heading structure the chunker could not segment.
- `company: ""` or `fiscal_year: ""` → detection missed. Extend `_COMPANY_HINTS` / check `_FISCAL_YEAR_RE` in `ingestion/metadata.py`. Retrieval still works but the confidence `freshness` and `cross_filing` signals degrade.
- Re-ingesting the same file is safe: `ingest_pdf` defaults to `overwrite=True` (deletes the `doc_id` first) and the chunker deduplicates on a content hash.

### Verify the collection

```bash
curl -s http://localhost:8000/eval/collection
# {"name":"finsight_chunks","vectors_count":N,"points_count":N,"status":"green"}
```

`points_count: 0` after ingestion that reported success → check that `QDRANT_HOST` resolves (inside compose it must be `qdrant`, not `localhost`).

---

## 5. Running a query

```bash
# blocking
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query":"Compare Infosys and TCS operating margins FY2024-FY2026 and flag anomalies"}'

# streaming (progress events + final result)
curl -N -X POST http://localhost:8000/query/stream -H 'Content-Type: application/json' \
  -d '{"query":"..."}'
```

Optional body fields: `company_filter`, `fiscal_year_filter`, `confidence_threshold`.

SSE event order: `start → planned → retrieved* → analyzed* → audited → compared → done` (or `error` with a `stage`).

---

## 6. Observability

### Metrics

```bash
curl -s http://localhost:8000/metrics | jq
```

Key fields: `error_rate`, `latency.{p50_ms,p95_ms}` and `latency.buckets`, `agents.<Name>.{avg_ms,p95_ms,count}`, `claims.{verified_rate,uncertain_rate,blocked_rate}`, `recent_errors`.

All windows are bounded in memory (latency 200, per-agent 100, recent queries 50, errors 20) and reset on restart.

### Logs

`structlog`. `LOG_FORMAT=json` in compose; `text` locally. Useful events:

| Event | Meaning |
|---|---|
| `finsight.startup` / `finsight.shutdown` | lifespan boundaries |
| `retriever.bm25_built` | BM25 index built (once per process); `corpus_size` |
| `retriever.empty_collection` | Qdrant returned nothing to index |
| `retriever.filter_no_match` | `company_filter` / `fiscal_year_filter` excluded every chunk |
| `analyst.parse_failed` / `auditor.batch_parse_failed` / `comparator.parse_failed` | LLM returned unparseable JSON; pipeline degraded gracefully |
| `llm.primary_retry` / `llm.primary_timeout` / `llm.using_reserve` / `llm.hedging` | LLM resilience path engaged |
| `query.complete` / `stream.*` | per-run summary (verified / uncertain / blocked / latency) |

```bash
docker compose logs -f api | grep -E 'llm\.|parse_failed'
```

### Tracing

Set `OTEL_ENABLED=true` and `OTEL_ENDPOINT=<otlp-grpc>` in `.env`. If the OTel packages are missing you get `tracing.otel_not_installed` and the app continues without spans.

---

## 7. Audit logs

One JSON file per run at `audit_logs/<task_id>.json` (dir = `AUDIT_LOG_DIR`, volume-mounted in compose, git-ignored).

```bash
curl -s http://localhost:8000/eval/audit-logs            # newest 20 filenames + total
curl -s http://localhost:8000/eval/audit-logs/<task_id>  # full log
```

Each log carries `plan`, `retrievals` (subtask → chunk_ids), `claims` (verified + uncertain), `flagged_uncertain`, `blocked_unverifiable`, `agents_invoked`, `latency_ms`. To review what the AuditorAgent excluded, read `blocked_unverifiable`.

The directory grows unbounded — there is no rotation. Prune with a cron/job if disk matters:

```bash
find audit_logs -name '*.json' -mtime +30 -delete
```

---

## 8. Common incidents

| Symptom | Likely cause | Action |
|---|---|---|
| `/health` never goes healthy | Model download still running, or import error | `docker compose logs api`; wait past `start-period`; check `GROQ_API_KEY` is set |
| Every query → HTTP 500 / SSE `error` at `PlannerAgent` | LLM key invalid, or both endpoints down | Verify `GROQ_API_KEY`; check `llm.*` logs; `curl` the Groq endpoint; configure `FALLBACK_*` |
| Reports say "insufficient evidence" for known facts | Empty/short collection, or filters too tight | `GET /eval/collection`; re-ingest; drop `company_filter`/`fiscal_year_filter`; check `retriever.filter_no_match` |
| High p95, frequent `llm.hedging` / `llm.using_reserve` | Primary endpoint slow or rate-limited | Expected under Groq free-tier load; configure a reserve endpoint; reduce concurrency |
| `blocked_rate` unexpectedly high | Threshold too strict, or weak sources | Lower `CONFIDENCE_THRESHOLD`; inspect `blocked_unverifiable` + `audit_reason` in the log |
| Answers contain a computed ratio / converted figure without a label | LLM broke an extraction rule | Inspect the audit log; tighten the Analyst/Comparator/Synthesizer prompts; confirm `unit_normalizer` ran (stream/sync both call `normalize_subtask_results`) |
| `400 Request contains disallowed content` | Query body matched an injection pattern | Legit? Adjust `_INJECTION_PATTERNS` in `api/middleware/guardrails.py` |
| Metrics all zero after a while | API restarted (in-process store) | Expected; no action |
| Qdrant unhealthy on boot | Port 6333 taken, or corrupt `qdrant_storage` | Free the port; `docker compose down -v` to reset storage (destroys the index — re-ingest) |

---

## 9. Backup & recovery

| Asset | Backup | Restore |
|---|---|---|
| Vector index | Copy `./qdrant_storage/` while stopped, **or** just keep the source PDFs and re-ingest | Restore the dir, or re-run ingestion |
| Audit logs | Copy `./audit_logs/` | Copy back |
| Config | `.env` (store in a secret manager, never in git) | Recreate from `.env.example` |
| Source PDFs | `./data/filings/` (git-ignored) | Re-download from the issuers |

The vector index is fully reproducible from the PDFs + code, so treating `qdrant_storage/` as disposable is fine.

---

## 10. Config changes that need a restart

Everything — `Settings` is read once at process start and the kernel/clients are singletons. After editing `.env`:

```bash
docker compose up -d --force-recreate api      # Docker
# or just restart uvicorn locally
```

Thresholds (`CONFIDENCE_THRESHOLD`) can also be overridden per request via the `confidence_threshold` body field without a restart.

---

## 11. Deployment hardening checklist (before any shared/network deployment)

- [ ] Restrict CORS — `allow_origins` is `["*"]` in `api/main.py`.
- [ ] Put the API behind an authenticating reverse proxy — there is no authn/z.
- [ ] Move `GROQ_API_KEY` / `FALLBACK_*` into a secret manager, not a file on disk.
- [ ] Add audit-log rotation/retention.
- [ ] Pin the Qdrant image tag (currently `:latest`).
- [ ] Set resource limits on both containers.
- [ ] Decide whether `qdrant_storage/` needs real backups for your RPO.
