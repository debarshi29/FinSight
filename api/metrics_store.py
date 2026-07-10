"""In-memory metrics store for production observability."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class QueryRecord:
    task_id: str
    query: str
    latency_ms: int
    verified: int
    uncertain: int
    blocked: int
    error: bool
    timestamp: float


class MetricsStore:
    """Lightweight thread-safe (GIL-protected) metrics collector.

    All mutations are simple int increments or deque appends — both atomic
    under CPython's GIL, so no asyncio.Lock is needed for a single process.
    """

    LATENCY_BUCKETS = [
        ("<500ms", 0, 500),
        ("500ms–1s", 500, 1000),
        ("1s–2s", 1000, 2000),
        ("2s–5s", 2000, 5000),
        ("5s–10s", 5000, 10000),
        (">10s", 10000, None),
    ]

    def __init__(self) -> None:
        self._start = time.time()
        self._total_queries: int = 0
        self._total_errors: int = 0
        self._inflight: int = 0

        self._latency_values: deque[int] = deque(maxlen=200)
        self._latency_bucket_counts: list[int] = [0] * len(self.LATENCY_BUCKETS)
        self._latency_sum: int = 0

        self._verified_total: int = 0
        self._uncertain_total: int = 0
        self._blocked_total: int = 0

        # Per-agent: list of individual ms timings
        self._agent_timings: dict[str, deque[int]] = {
            "PlannerAgent": deque(maxlen=100),
            "RetrieverAgent": deque(maxlen=100),
            "AnalystAgent": deque(maxlen=100),
            "AuditorAgent": deque(maxlen=100),
            "ComparatorAgent": deque(maxlen=100),
            "SynthesizerAgent": deque(maxlen=100),
        }

        self._recent: deque[QueryRecord] = deque(maxlen=50)
        self._errors: deque[dict] = deque(maxlen=20)

    # ── Public recording API ──────────────────────────────────────────────────

    def record_start(self) -> None:
        self._inflight += 1

    def record_complete(
        self,
        task_id: str,
        query: str,
        latency_ms: int,
        verified: int,
        uncertain: int,
        blocked: int,
    ) -> None:
        self._inflight = max(0, self._inflight - 1)
        self._total_queries += 1
        self._latency_values.append(latency_ms)
        self._latency_sum += latency_ms
        self._verified_total += verified
        self._uncertain_total += uncertain
        self._blocked_total += blocked

        for i, (_, lo, hi) in enumerate(self.LATENCY_BUCKETS):
            if hi is None or latency_ms < hi:
                if latency_ms >= lo:
                    self._latency_bucket_counts[i] += 1
                    break

        self._recent.appendleft(
            QueryRecord(
                task_id=task_id,
                query=query,
                latency_ms=latency_ms,
                verified=verified,
                uncertain=uncertain,
                blocked=blocked,
                error=False,
                timestamp=time.time(),
            )
        )

    def record_error(self, task_id: str, query: str, stage: str, detail: str) -> None:
        self._inflight = max(0, self._inflight - 1)
        self._total_errors += 1
        self._errors.appendleft(
            {
                "task_id": task_id[:8],
                "query": query[:80],
                "stage": stage,
                "detail": detail[:200],
                "ts": time.time(),
            }
        )
        self._recent.appendleft(
            QueryRecord(
                task_id=task_id,
                query=query,
                latency_ms=0,
                verified=0,
                uncertain=0,
                blocked=0,
                error=True,
                timestamp=time.time(),
            )
        )

    def record_agent_latency(self, agent: str, ms: int) -> None:
        if agent in self._agent_timings:
            self._agent_timings[agent].append(ms)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        total_q = max(self._total_queries, 1)  # avoid div-by-zero
        claims_total = self._verified_total + self._uncertain_total + self._blocked_total

        # Latency percentiles from recent window
        vals = sorted(self._latency_values)
        p50 = vals[len(vals) // 2] if vals else 0
        p95 = vals[int(len(vals) * 0.95)] if vals else 0
        avg = self._latency_sum // max(self._total_queries, 1)
        lat_min = vals[0] if vals else 0
        lat_max = vals[-1] if vals else 0

        agents: dict[str, dict] = {}
        for name, times in self._agent_timings.items():
            tlist = sorted(times)
            if tlist:
                agents[name] = {
                    "avg_ms": int(sum(tlist) / len(tlist)),
                    "p95_ms": tlist[int(len(tlist) * 0.95)],
                    "min_ms": tlist[0],
                    "max_ms": tlist[-1],
                    "count": len(tlist),
                }
            else:
                agents[name] = {"avg_ms": 0, "p95_ms": 0, "min_ms": 0, "max_ms": 0, "count": 0}

        return {
            "uptime_s": int(time.time() - self._start),
            "total_queries": self._total_queries,
            "total_errors": self._total_errors,
            "inflight": self._inflight,
            "error_rate": round(self._total_errors / total_q, 4),
            "latency": {
                "avg_ms": avg,
                "p50_ms": p50,
                "p95_ms": p95,
                "min_ms": lat_min,
                "max_ms": lat_max,
                "buckets": [
                    {"label": label, "count": count}
                    for (label, _, __), count in zip(
                        self.LATENCY_BUCKETS, self._latency_bucket_counts
                    )
                ],
            },
            "claims": {
                "verified": self._verified_total,
                "uncertain": self._uncertain_total,
                "blocked": self._blocked_total,
                "total": claims_total,
                "verified_rate": round(self._verified_total / max(claims_total, 1), 4),
                "uncertain_rate": round(self._uncertain_total / max(claims_total, 1), 4),
                "blocked_rate": round(self._blocked_total / max(claims_total, 1), 4),
            },
            "agents": agents,
            "recent_queries": [
                {
                    "task_id": r.task_id[:8],
                    "query": r.query[:60],
                    "latency_ms": r.latency_ms,
                    "verified": r.verified,
                    "uncertain": r.uncertain,
                    "blocked": r.blocked,
                    "error": r.error,
                    "ts": r.timestamp,
                }
                for r in list(self._recent)[:20]
            ],
            "recent_errors": list(self._errors)[:10],
        }


# Module-level singleton — imported by routes.
metrics = MetricsStore()
