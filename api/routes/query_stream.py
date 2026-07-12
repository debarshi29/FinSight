from __future__ import annotations

import asyncio
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
from core.groq_client import chat_completion_hedged
from core.models import AnalysisReport, AuditLog
from core.sk_kernel import PLANNER_PROMPT, get_kernel
from core.unit_normalizer import normalize_subtask_results

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


async def _run_subtask(
    kernel,
    subtask: str,
    company_filter: str,
    fiscal_year_filter: str,
) -> tuple[list[tuple[str, dict]], dict | None]:
    """
    Run Retriever + Analyst for one subtask.
    Returns (sse_events, result_dict | None).
    Runs fully async so multiple subtasks can be gather()ed in parallel.
    """
    events: list[tuple[str, dict]] = []

    # ── RetrieverAgent ────────────────────────────────────────────────
    try:
        retrieve_result = await kernel.invoke(
            plugin_name="Retriever",
            function_name="retrieve",
            arguments=KernelArguments(
                subtask=subtask,
                company_filter=company_filter,
                fiscal_year_filter=fiscal_year_filter,
            ),
        )
        chunks_json = str(retrieve_result)
        try:
            chunks_data = json.loads(chunks_json)
        except json.JSONDecodeError:
            chunks_data = []
    except Exception as exc:
        events.append(
            ("error", {"stage": "RetrieverAgent", "subtask": subtask, "detail": str(exc)})
        )
        return events, None

    events.append(("retrieved", {"subtask": subtask, "chunks": len(chunks_data)}))

    if not chunks_data:
        return events, None

    # ── AnalystAgent ──────────────────────────────────────────────────
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
        events.append(("error", {"stage": "AnalystAgent", "subtask": subtask, "detail": str(exc)}))
        return events, None

    claims = analysis.get("claims", [])
    events.append(("analyzed", {"subtask": subtask, "claims": len(claims)}))

    return events, {
        "subtask": subtask,
        "chunks": [c.get("chunk_id", "") for c in chunks_data],
        "claims": claims,
        "kpis": analysis.get("kpis", []),
    }


@router.post("/stream")
async def run_query_stream(req: QueryRequest):
    async def generate():
        task_id = str(uuid.uuid4())
        start_ms = time.time()
        kernel = get_kernel()
        agents_invoked: list[str] = []

        metrics.record_start()
        log.info("stream.start", task_id=task_id[:8], query=req.query[:80])
        yield _event("start", {"task_id": task_id, "query": req.query})

        # ── PlannerAgent — direct LLM call, no SK tool injection ─────
        agents_invoked.append("PlannerAgent")
        _t = time.time()
        try:
            plan_text = await chat_completion_hedged(
                messages=[{"role": "user", "content": PLANNER_PROMPT.format(user_task=req.query)}],
                max_tokens=500,
                temperature=0.0,
                hedge_after=8.0,
            )
        except Exception as exc:
            metrics.record_error(task_id, req.query, "PlannerAgent", str(exc))
            yield _event("error", {"stage": "PlannerAgent", "detail": str(exc)})
            return
        metrics.record_agent_latency("PlannerAgent", int((time.time() - _t) * 1000))
        subtasks = _parse_subtasks(plan_text, req.query)
        log.info("stream.planned", task_id=task_id[:8], subtasks=len(subtasks))
        yield _event("planned", {"subtasks": subtasks})

        # ── Retriever + Analyst — all subtasks in parallel ────────────
        agents_invoked.append("RetrieverAgent")
        agents_invoked.append("AnalystAgent")
        _t = time.time()

        subtask_coros = [
            _run_subtask(
                kernel,
                subtask,
                req.company_filter or "",
                req.fiscal_year_filter or "",
            )
            for subtask in subtasks
        ]
        subtask_outputs = await asyncio.gather(*subtask_coros)

        all_claims: list[dict] = []
        subtask_results: list[dict] = []
        retrievals: dict[str, list[str]] = {}

        for events, result in subtask_outputs:
            for ev_name, ev_data in events:
                yield _event(ev_name, ev_data)
            if result:
                all_claims.extend(result["claims"])
                subtask_results.append(
                    {
                        "subtask": result["subtask"],
                        "kpis": result["kpis"],
                        "claims": result["claims"],
                    }
                )
                retrievals[result["subtask"]] = result["chunks"]

        ra_ms = int((time.time() - _t) * 1000)
        metrics.record_agent_latency("RetrieverAgent", ra_ms)
        metrics.record_agent_latency("AnalystAgent", ra_ms)

        # ── AuditorAgent + ComparatorAgent — run concurrently ────────
        # Neither depends on the other; both need only subtask_results.
        # Wall-clock = max(auditor, comparator) instead of their sum.
        log.info("stream.audit_compare_start", task_id=task_id[:8], total_claims=len(all_claims))
        agents_invoked.extend(["AuditorAgent", "ComparatorAgent"])
        _t = time.time()
        threshold = req.confidence_threshold or settings.confidence_threshold

        # Normalize financial units to ₹ crore before comparator sees the data.
        # This is deterministic Python math — the LLM never does arithmetic.
        normalized_subtask_results = normalize_subtask_results(subtask_results)

        async def _audit() -> str:
            res = await kernel.invoke(
                plugin_name="Auditor",
                function_name="audit",
                arguments=KernelArguments(
                    claims_json=json.dumps(all_claims),
                    confidence_threshold=str(threshold),
                ),
            )
            return str(res)

        async def _compare() -> str:
            res = await kernel.invoke(
                plugin_name="Comparator",
                function_name="compare",
                arguments=KernelArguments(
                    subtask_results_json=json.dumps(normalized_subtask_results),
                    original_query=req.query,
                ),
            )
            return str(res)

        audit_raw, compare_raw = await asyncio.gather(_audit(), _compare(), return_exceptions=True)
        ac_ms = int((time.time() - _t) * 1000)
        metrics.record_agent_latency("AuditorAgent", ac_ms)
        metrics.record_agent_latency("ComparatorAgent", ac_ms)

        # Process auditor result
        if isinstance(audit_raw, Exception):
            metrics.record_error(task_id, req.query, "AuditorAgent", str(audit_raw))
            yield _event("error", {"stage": "AuditorAgent", "detail": str(audit_raw)})
            return
        verified, uncertain, unverifiable_claims = _parse_audit_result(audit_raw)
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

        # Process comparator result
        if isinstance(compare_raw, Exception):
            yield _event("error", {"stage": "ComparatorAgent", "detail": str(compare_raw)})
            comparison: dict = {"deltas": [], "cross_document_claims": [], "summary": ""}
        else:
            try:
                comparison = json.loads(compare_raw)
            except json.JSONDecodeError:
                comparison = {"deltas": [], "cross_document_claims": [], "summary": ""}
        yield _event("compared", {"deltas": len(comparison.get("deltas", []))})

        # ── SynthesizerAgent — direct LLM call ────────────────────────
        log.info("stream.synthesizer_start", task_id=task_id[:8])
        agents_invoked.append("SynthesizerAgent")
        _t = time.time()
        try:
            summary = await synthesize_report(req.query, verified, uncertain, comparison, task_id)
        except Exception as exc:
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
