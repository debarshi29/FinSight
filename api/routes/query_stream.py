from __future__ import annotations

import json
import time
import uuid

import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from semantic_kernel.functions import KernelArguments

from agents.synthesizer import synthesize_report
from api.metrics_store import metrics
from core.config import settings
from core.models import AnalysisReport, AuditLog
from core.sk_kernel import get_fallback_kernel, get_kernel

from .query import (
    QueryRequest,
    _parse_audit_result,
    _parse_subtasks,
    _save_audit_log,
)

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
        kernel = get_kernel()
        fallback_kernel = get_fallback_kernel()
        agents_invoked: list[str] = []

        metrics.record_start()
        log.info("stream.start", task_id=task_id[:8], query=req.query[:80])
        yield _event("start", {"task_id": task_id, "query": req.query})

        # ── PlannerAgent ────────────────────────────────────────────────────
        agents_invoked.append("PlannerAgent")
        _t = time.time()
        try:
            plan_result = await kernel.invoke(
                plugin_name="Planner",
                function_name="decompose",
                arguments=KernelArguments(user_task=req.query),
            )
        except Exception as exc:
            if fallback_kernel:
                try:
                    plan_result = await fallback_kernel.invoke(
                        plugin_name="Planner",
                        function_name="decompose",
                        arguments=KernelArguments(user_task=req.query),
                    )
                except Exception as exc2:
                    metrics.record_error(task_id, req.query, "PlannerAgent", str(exc2))
                    yield _event("error", {"stage": "PlannerAgent", "detail": str(exc2)})
                    return
            else:
                metrics.record_error(task_id, req.query, "PlannerAgent", str(exc))
                yield _event("error", {"stage": "PlannerAgent", "detail": str(exc)})
                return
        metrics.record_agent_latency("PlannerAgent", int((time.time() - _t) * 1000))
        subtasks = _parse_subtasks(str(plan_result), req.query)
        log.info("stream.planned", task_id=task_id[:8], subtasks=len(subtasks))
        yield _event("planned", {"subtasks": subtasks})

        all_claims: list[dict] = []
        subtask_results: list[dict] = []
        retrievals: dict[str, list[str]] = {}

        for subtask in subtasks:
            # ── RetrieverAgent ───────────────────────────────────────────────
            agents_invoked.append("RetrieverAgent")
            _t = time.time()
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
                try:
                    chunks_data = json.loads(chunks_json)
                except json.JSONDecodeError:
                    chunks_data = []
            except Exception as exc:
                yield _event(
                    "error", {"stage": "RetrieverAgent", "subtask": subtask, "detail": str(exc)}
                )
                continue
            metrics.record_agent_latency("RetrieverAgent", int((time.time() - _t) * 1000))

            retrievals[subtask] = [c.get("chunk_id", "") for c in chunks_data]
            yield _event("retrieved", {"subtask": subtask, "chunks": len(chunks_data)})

            if not chunks_data:
                continue

            # ── AnalystAgent ─────────────────────────────────────────────────
            agents_invoked.append("AnalystAgent")
            _t = time.time()
            try:
                analysis_result = await kernel.invoke(
                    plugin_name="Analyst",
                    function_name="analyze",
                    arguments=KernelArguments(subtask=subtask, chunks_json=chunks_json),
                )
                try:
                    analysis = json.loads(str(analysis_result))
                except json.JSONDecodeError:
                    analysis = {"kpis": [], "claims": []}
            except Exception as exc:
                yield _event(
                    "error", {"stage": "AnalystAgent", "subtask": subtask, "detail": str(exc)}
                )
                continue
            metrics.record_agent_latency("AnalystAgent", int((time.time() - _t) * 1000))

            claims = analysis.get("claims", [])
            all_claims.extend(claims)
            subtask_results.append(
                {"subtask": subtask, "kpis": analysis.get("kpis", []), "claims": claims}
            )
            yield _event("analyzed", {"subtask": subtask, "claims": len(claims)})

        # ── AuditorAgent ─────────────────────────────────────────────────────
        log.info("stream.auditor_start", task_id=task_id[:8], total_claims=len(all_claims))
        agents_invoked.append("AuditorAgent")
        _t = time.time()
        try:
            threshold = req.confidence_threshold or settings.confidence_threshold
            audit_result = await kernel.invoke(
                plugin_name="Auditor",
                function_name="audit",
                arguments=KernelArguments(
                    claims_json=json.dumps(all_claims),
                    confidence_threshold=str(threshold),
                ),
            )
            verified, uncertain, unverifiable_claims = _parse_audit_result(str(audit_result))
        except Exception as exc:
            metrics.record_error(task_id, req.query, "AuditorAgent", str(exc))
            yield _event("error", {"stage": "AuditorAgent", "detail": str(exc)})
            return
        metrics.record_agent_latency("AuditorAgent", int((time.time() - _t) * 1000))
        log.info(
            "stream.audited", task_id=task_id[:8], verified=len(verified), uncertain=len(uncertain)
        )

        yield _event(
            "audited",
            {
                "verified": len(verified),
                "uncertain": len(uncertain),
                "blocked": len(unverifiable_claims),
            },
        )

        # ── ComparatorAgent ──────────────────────────────────────────────────
        agents_invoked.append("ComparatorAgent")
        _t = time.time()
        try:
            compare_result = await kernel.invoke(
                plugin_name="Comparator",
                function_name="compare",
                arguments=KernelArguments(
                    subtask_results_json=json.dumps(subtask_results),
                    original_query=req.query,
                ),
            )
            try:
                comparison = json.loads(str(compare_result))
            except json.JSONDecodeError:
                comparison = {"deltas": [], "cross_document_claims": [], "summary": ""}
        except Exception as exc:
            yield _event("error", {"stage": "ComparatorAgent", "detail": str(exc)})
            comparison = {"deltas": [], "cross_document_claims": [], "summary": ""}
        metrics.record_agent_latency("ComparatorAgent", int((time.time() - _t) * 1000))

        yield _event("compared", {"deltas": len(comparison.get("deltas", []))})

        # ── SynthesizerAgent ─────────────────────────────────────────────────
        log.info("stream.synthesizer_start", task_id=task_id[:8])
        agents_invoked.append("SynthesizerAgent")
        _t = time.time()
        try:
            summary = await synthesize_report(req.query, verified, uncertain, comparison, task_id)
        except Exception as exc:
            if fallback_kernel:
                try:
                    summary = await synthesize_report(
                        req.query, verified, uncertain, comparison, task_id, kernel=fallback_kernel
                    )
                except Exception as exc2:
                    metrics.record_error(task_id, req.query, "SynthesizerAgent", str(exc2))
                    yield _event("error", {"stage": "SynthesizerAgent", "detail": str(exc2)})
                    return
            else:
                metrics.record_error(task_id, req.query, "SynthesizerAgent", str(exc))
                yield _event("error", {"stage": "SynthesizerAgent", "detail": str(exc)})
                return
        metrics.record_agent_latency("SynthesizerAgent", int((time.time() - _t) * 1000))
        log.info("stream.synthesizer_done", task_id=task_id[:8])

        latency_ms = int((time.time() - start_ms) * 1000)

        try:
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
            yield _event("done", {"result": report.to_dict()})
        except Exception as exc:
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
