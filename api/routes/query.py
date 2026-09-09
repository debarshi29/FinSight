from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.metrics_store import metrics
from core.config import settings
from core.models import AnalysisReport, AuditLog
from orchestration.graph import AGENT_SEQUENCE, build_retrievals, initial_state
from orchestration.runner import run_pipeline

log = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str
    company_filter: str | None = None
    fiscal_year_filter: str | None = None
    confidence_threshold: float | None = None


@router.post("")
async def run_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    task_id = str(uuid.uuid4())
    start_ms = time.time()

    metrics.record_start()
    log.info("query.start", task_id=task_id, query=req.query[:100])

    state = initial_state(
        req.query,
        task_id,
        company_filter=req.company_filter,
        fiscal_year_filter=req.fiscal_year_filter,
        confidence_threshold=req.confidence_threshold,
        streaming=False,
    )

    try:
        final, timings = await run_pipeline(state)
    except Exception as exc:  # noqa: BLE001 - surface a failed run honestly
        metrics.record_error(task_id, req.query, "pipeline", str(exc))
        log.exception("query.failed", task_id=task_id, detail=str(exc))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    for agent, ms in timings.items():
        metrics.record_agent_latency(agent, ms)

    subtasks = final.get("subtasks", [])
    verified = final.get("verified", [])
    uncertain = final.get("uncertain", [])
    blocked = final.get("unverifiable", [])
    summary = final.get("summary", "")
    log.info("query.audited", task_id=task_id, verified=len(verified), uncertain=len(uncertain))

    latency_ms = int((time.time() - start_ms) * 1000)

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
    log.info(
        "query.complete",
        task_id=task_id,
        verified=len(verified),
        uncertain=len(uncertain),
        blocked=len(blocked),
        latency_ms=latency_ms,
    )
    return report.to_dict()


def _save_audit_log(audit_log: AuditLog) -> None:
    log_dir = Path(settings.audit_log_dir)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{audit_log.task_id}.json"
    log_path.write_text(json.dumps(audit_log.to_dict(), indent=2))
