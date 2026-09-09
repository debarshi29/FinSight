from __future__ import annotations

import asyncio
import json
import time
import uuid

import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.metrics_store import metrics
from api.routes.query import QueryRequest, _save_audit_log
from core.models import AnalysisReport, AuditLog
from orchestration.graph import AGENT_SEQUENCE, build_retrievals, initial_state
from orchestration.runner import run_pipeline

log = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])


def _event(name: str, data: dict) -> str:
    payload = json.dumps({"event": name, **data})
    return f"data: {payload}\n\n"


@router.post("/stream")
async def run_query_stream(req: QueryRequest):
    async def generate():
        task_id = str(uuid.uuid4())
        start_ms = time.time()

        metrics.record_start()
        log.info("stream.start", task_id=task_id[:8], query=req.query[:80])
        yield _event("start", {"task_id": task_id, "query": req.query})

        state = initial_state(
            req.query,
            task_id,
            company_filter=req.company_filter,
            fiscal_year_filter=req.fiscal_year_filter,
            confidence_threshold=req.confidence_threshold,
            streaming=True,
        )

        # The graph runs in a driver task and pushes progress events onto a queue;
        # this generator drains the queue into the SSE stream.
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(ev: dict) -> None:
            await queue.put(("event", ev))

        async def driver() -> None:
            try:
                final, timings = await run_pipeline(state, on_event=on_event)
                await queue.put(("final", (final, timings)))
            except Exception as exc:  # noqa: BLE001 - reported over SSE
                await queue.put(("error", exc))

        drv = asyncio.create_task(driver())

        final: dict = {}
        timings: dict[str, int] = {}
        failure: Exception | None = None

        while True:
            kind, payload = await queue.get()
            if kind == "event":
                name = payload.get("event", "message")
                yield _event(name, {k: v for k, v in payload.items() if k != "event"})
            elif kind == "final":
                final, timings = payload
                break
            else:  # "error"
                failure = payload
                break

        await drv

        if failure is not None:
            metrics.record_error(task_id, req.query, "pipeline", str(failure))
            log.exception("stream.failed", task_id=task_id[:8], detail=str(failure))
            yield _event("error", {"stage": "pipeline", "detail": str(failure)[:300]})
            return

        for agent, ms in timings.items():
            metrics.record_agent_latency(agent, ms)

        verified = final.get("verified", [])
        uncertain = final.get("uncertain", [])
        blocked = final.get("unverifiable", [])
        summary = final.get("summary", "")
        subtasks = final.get("subtasks", [])
        latency_ms = int((time.time() - start_ms) * 1000)

        try:
            audit_log = AuditLog(
                task_id=task_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                user_query=req.query,
                plan=subtasks,
                retrievals=build_retrievals(final),
                claims=[c.to_dict() for c in verified + uncertain],
                flagged_uncertain=[c.claim for c in uncertain],
                blocked_unverifiable=blocked,
                agents_invoked=list(AGENT_SEQUENCE),
                latency_ms=latency_ms,
            )
            _save_audit_log(audit_log)

            report = AnalysisReport(
                task_id=task_id,
                query=req.query,
                summary=summary,
                verified_claims=verified,
                uncertain_claims=uncertain,
                audit_log=audit_log,
            )

            metrics.record_complete(
                task_id=task_id,
                query=req.query,
                latency_ms=latency_ms,
                verified=len(verified),
                uncertain=len(uncertain),
                blocked=len(blocked),
            )
            yield _event("done", {"result": report.to_dict()})
        except Exception as exc:  # noqa: BLE001
            log.exception("stream.done_failed", detail=str(exc))
            yield _event("error", {"stage": "finalize", "detail": str(exc)[:300]})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
