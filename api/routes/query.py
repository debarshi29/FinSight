from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.analyst import analyze_chunks
from agents.auditor import audit_claims
from agents.comparator import compare_results
from agents.retriever import retrieve_chunks
from agents.router import plan_task
from agents.synthesizer import synthesize_report
from core.config import settings
from core.models import AnalysisReport, AuditLog

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
    agents_invoked = []

    log.info("query.start", task_id=task_id, query=req.query[:100])

    # PlannerAgent
    agents_invoked.append("PlannerAgent")
    subtasks = await plan_task(req.query)

    all_claims = []
    all_kpis = []
    subtask_results = []
    retrievals: dict[str, list[str]] = {}

    for subtask in subtasks:
        # RetrieverAgent
        agents_invoked.append("RetrieverAgent")
        ranked_chunks = await retrieve_chunks(subtask)

        if not ranked_chunks:
            continue

        retrievals[subtask] = [r.chunk.chunk_id for r in ranked_chunks]
        chunks_data = [
            {
                "text": r.chunk.text,
                "source": r.chunk.source,
                "page": r.chunk.page,
                "section_type": r.chunk.section_type.value,
                "confidence": r.confidence_score,
                "company": r.chunk.company,
                "fiscal_year": r.chunk.fiscal_year,
            }
            for r in ranked_chunks
        ]

        # AnalystAgent
        agents_invoked.append("AnalystAgent")
        analysis = await analyze_chunks(subtask, chunks_data)
        kpis = analysis.get("kpis", [])
        claims = analysis.get("claims", [])

        all_kpis.extend(kpis)
        all_claims.extend(claims)
        subtask_results.append({"subtask": subtask, "kpis": kpis, "claims": claims})

    # AuditorAgent
    agents_invoked.append("AuditorAgent")
    threshold = req.confidence_threshold or settings.confidence_threshold
    verified, uncertain, unverifiable = await audit_claims(all_claims, {}, threshold)

    # ComparatorAgent
    agents_invoked.append("ComparatorAgent")
    comparison = await compare_results(subtask_results, req.query)

    # SynthesizerAgent
    agents_invoked.append("SynthesizerAgent")
    summary = await synthesize_report(req.query, verified, uncertain, comparison, task_id)

    latency_ms = int((time.time() - start_ms) * 1000)

    audit_log = AuditLog(
        task_id=task_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        user_query=req.query,
        plan=subtasks,
        retrievals=retrievals,
        claims=[c.to_dict() for c in verified + uncertain],
        flagged_uncertain=[c.claim for c in uncertain],
        blocked_unverifiable=[c.claim for c in unverifiable],
        agents_invoked=list(dict.fromkeys(agents_invoked)),
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

    log.info(
        "query.complete",
        task_id=task_id,
        verified=len(verified),
        uncertain=len(uncertain),
        blocked=len(unverifiable),
        latency_ms=latency_ms,
    )
    return report.to_dict()


def _save_audit_log(audit_log: AuditLog) -> None:
    log_dir = Path(settings.audit_log_dir)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{audit_log.task_id}.json"
    log_path.write_text(json.dumps(audit_log.to_dict(), indent=2))
