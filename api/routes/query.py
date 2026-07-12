from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from semantic_kernel.functions import KernelArguments

from agents.synthesizer import synthesize_report
from api.metrics_store import metrics
from core.config import settings
from core.groq_client import chat_completion
from core.models import AnalysisReport, AuditedClaim, AuditLog, AuditStatus, Citation
from core.sk_kernel import PLANNER_PROMPT, get_kernel
from core.unit_normalizer import normalize_subtask_results

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
    agents_invoked: list[str] = []

    metrics.record_start()
    kernel = get_kernel()
    log.info("query.start", task_id=task_id, query=req.query[:100])

    # ── PlannerAgent — direct LLM call, no SK tool injection ─────────────────
    agents_invoked.append("PlannerAgent")
    _t = time.time()
    plan_text = await chat_completion(
        messages=[{"role": "user", "content": PLANNER_PROMPT.format(user_task=req.query)}],
        max_tokens=500,
        temperature=0.0,
    )
    metrics.record_agent_latency("PlannerAgent", int((time.time() - _t) * 1000))
    subtasks = _parse_subtasks(plan_text, req.query)
    log.info("planner.subtasks", count=len(subtasks), subtasks=subtasks)

    all_claims: list[dict] = []
    subtask_results: list[dict] = []
    retrievals: dict[str, list[str]] = {}

    # ── Retriever + Analyst — all subtasks in parallel ────────────────────────
    agents_invoked.append("RetrieverAgent")
    agents_invoked.append("AnalystAgent")

    async def _run_subtask_sync(subtask: str) -> dict | None:
        try:
            retrieve_result = await kernel.invoke(
                plugin_name="Retriever",
                function_name="retrieve",
                arguments=KernelArguments(
                    subtask=subtask,
                    company_filter=req.company_filter or "",
                    fiscal_year_filter=req.fiscal_year_filter or "",
                ),
            )
            chunks_json = str(retrieve_result)
            chunks_data = json.loads(chunks_json)
        except Exception:
            return None

        if not chunks_data:
            return None

        try:
            analysis_result = await kernel.invoke(
                plugin_name="Analyst",
                function_name="analyze",
                arguments=KernelArguments(subtask=subtask, chunks_json=chunks_json),
            )
            analysis = json.loads(str(analysis_result))
        except Exception:
            analysis = {"kpis": [], "claims": []}

        return {
            "subtask": subtask,
            "chunks": [c.get("chunk_id", "") for c in chunks_data],
            "claims": analysis.get("claims", []),
            "kpis": analysis.get("kpis", []),
        }

    _t = time.time()
    parallel_results = await asyncio.gather(*[_run_subtask_sync(s) for s in subtasks])
    ra_ms = int((time.time() - _t) * 1000)
    metrics.record_agent_latency("RetrieverAgent", ra_ms)
    metrics.record_agent_latency("AnalystAgent", ra_ms)

    for result in parallel_results:
        if result:
            all_claims.extend(result["claims"])
            subtask_results.append(
                {"subtask": result["subtask"], "kpis": result["kpis"], "claims": result["claims"]}
            )
            retrievals[result["subtask"]] = result["chunks"]

    # ── AuditorAgent ─────────────────────────────────────────────────────────
    agents_invoked.append("AuditorAgent")
    threshold = req.confidence_threshold or settings.confidence_threshold
    _t = time.time()
    audit_result = await kernel.invoke(
        plugin_name="Auditor",
        function_name="audit",
        arguments=KernelArguments(
            claims_json=json.dumps(all_claims),
            confidence_threshold=str(threshold),
            original_query=req.query,
        ),
    )
    metrics.record_agent_latency("AuditorAgent", int((time.time() - _t) * 1000))

    verified, uncertain, unverifiable_claims = _parse_audit_result(str(audit_result))

    # ── ComparatorAgent ───────────────────────────────────────────────────────
    agents_invoked.append("ComparatorAgent")
    _t = time.time()
    normalized_subtask_results = normalize_subtask_results(subtask_results)
    compare_result = await kernel.invoke(
        plugin_name="Comparator",
        function_name="compare",
        arguments=KernelArguments(
            subtask_results_json=json.dumps(normalized_subtask_results),
            original_query=req.query,
        ),
    )
    metrics.record_agent_latency("ComparatorAgent", int((time.time() - _t) * 1000))

    try:
        comparison = json.loads(str(compare_result))
    except json.JSONDecodeError:
        comparison = {"deltas": [], "cross_document_claims": [], "summary": ""}

    # ── SynthesizerAgent — direct LLM call, no SK tool injection ─────────────
    agents_invoked.append("SynthesizerAgent")
    _t = time.time()
    summary = await synthesize_report(req.query, verified, uncertain, comparison, task_id)
    metrics.record_agent_latency("SynthesizerAgent", int((time.time() - _t) * 1000))

    latency_ms = int((time.time() - start_ms) * 1000)

    audit_log = AuditLog(
        task_id=task_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        user_query=req.query,
        plan=subtasks,
        retrievals=retrievals,
        claims=[c.to_dict() for c in verified + uncertain],
        flagged_uncertain=[c.claim for c in uncertain],
        blocked_unverifiable=unverifiable_claims,
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

    metrics.record_complete(
        task_id=task_id,
        query=req.query,
        latency_ms=latency_ms,
        verified=len(verified),
        uncertain=len(uncertain),
        blocked=len(unverifiable_claims),
    )
    log.info(
        "query.complete",
        task_id=task_id,
        verified=len(verified),
        uncertain=len(uncertain),
        blocked=len(unverifiable_claims),
        latency_ms=latency_ms,
    )
    return report.to_dict()


def _parse_subtasks(content: str, fallback: str) -> list[str]:

    try:
        subtasks = json.loads(content)
        if isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks):
            return subtasks
    except (json.JSONDecodeError, ValueError):
        pass
    lines = [ln.strip().lstrip("-•1234567890.) ") for ln in content.splitlines() if ln.strip()]
    subtasks = [ln for ln in lines if len(ln) > 10][:6]
    return subtasks if subtasks else [fallback]


def _parse_audit_result(
    audit_json: str,
) -> tuple[list[AuditedClaim], list[AuditedClaim], list[str]]:
    try:
        data = json.loads(audit_json)
    except json.JSONDecodeError:
        return [], [], []

    verified = _deserialize_claims(data.get("verified", []))
    uncertain = _deserialize_claims(data.get("uncertain", []))
    unverifiable = data.get("unverifiable", [])
    return verified, uncertain, unverifiable


def _safe_confidence(raw: object) -> float:
    try:
        v = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if v != v or not (0.0 <= v <= 1.0):  # nan / out-of-range guard
        return max(0.0, min(1.0, v)) if v == v else 0.5
    return v


def _deserialize_claims(raw: list[dict]) -> list[AuditedClaim]:
    result = []
    for item in raw:
        try:
            cit = item.get("citation", {})
            citation = Citation(
                document=cit.get("document", "unknown"),
                page=int(cit.get("page") or 0),
                snippet=cit.get("snippet", ""),
                claim=cit.get("claim", ""),
                confidence=_safe_confidence(cit.get("confidence", 0.5)),
                section_type=cit.get("section_type", "unknown"),
            )
            result.append(
                AuditedClaim(
                    claim=item.get("claim", ""),
                    citation=citation,
                    audit_status=AuditStatus(item.get("audit_status", "uncertain")),
                    audit_reason=item.get("audit_reason", ""),
                )
            )
        except Exception:
            continue
    return result


def _save_audit_log(audit_log: AuditLog) -> None:
    log_dir = Path(settings.audit_log_dir)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{audit_log.task_id}.json"
    log_path.write_text(json.dumps(audit_log.to_dict(), indent=2))
