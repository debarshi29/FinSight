"""LangGraph orchestration for the FinSight agent pipeline.

A single ``StateGraph`` replaces the Semantic Kernel dispatch layer. The plan is
still generated per query by the Planner; the graph fans that plan out with the
``Send`` API, so there is no fixed topology to maintain.

Shape::

    START ─▶ recall_memory ─▶ plan ─▶ (Send × N) retrieve_analyze ─┬─▶ audit ────┐
                                                                   └─▶ compare ──┴─▶ synthesize ─▶ remember ─▶ END

``retrieve_analyze`` instances run in parallel; ``audit`` and ``compare`` run in
one superstep (neither depends on the other); ``synthesize`` waits for both.
``recall_memory`` and ``remember`` are best-effort: a memory outage degrades to
no context / no write rather than failing the run, and recalled memory only ever
reaches the Planner prompt — never the Synthesizer — so it can shape what gets
searched for but can never become an unverified claim in the report.

Nodes emit progress via the custom stream channel (``get_stream_writer``) so the
SSE endpoint can surface ``memory_recalled`` / ``planned`` / ``retrieved`` /
``analyzed`` / ``audited`` / ``compared`` / ``remembered`` events without the
graph knowing about HTTP. This layer imports agents, core, and memory only —
never ``api`` — so latency accounting is done by the caller around ``astream`` /
``ainvoke``.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.analyst import analyze_chunks
from agents.auditor import audit_claims
from agents.comparator import compare_results
from agents.retriever import get_retrieval_service, ranked_chunk_to_dict
from agents.router import plan_task
from agents.synthesizer import synthesize_report
from core.config import settings
from core.models import AuditedClaim
from core.unit_normalizer import normalize_subtask_results
from memory.consolidate import build_turn_text, format_memory_context
from memory.store import get_memory_service

log = structlog.get_logger()

# Canonical agent order for the audit log. Every stage always executes, so this
# is fixed rather than derived from node-completion order (racy for the parallel
# audit/compare superstep). MemoryAgent covers both the recall and the write node.
AGENT_SEQUENCE = [
    "MemoryAgent",
    "PlannerAgent",
    "RetrieverAgent",
    "AnalystAgent",
    "AuditorAgent",
    "ComparatorAgent",
    "SynthesizerAgent",
]

_EMPTY_COMPARISON: dict[str, Any] = {
    "deltas": [],
    "cross_document_claims": [],
    "summary": "",
}


def _emit(event: str, **data: Any) -> None:
    """Best-effort custom-stream event; a no-op when nobody is streaming."""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"event": event, **data})
    except Exception:  # pragma: no cover - streaming not active
        pass


# ── State ────────────────────────────────────────────────────────────────────


class SubtaskResult(TypedDict):
    subtask: str
    chunks: list[str]  # chunk_ids retrieved
    claims: list[dict]
    kpis: list[dict]


class FinSightState(TypedDict, total=False):
    # inputs
    query: str
    company_filter: str
    fiscal_year_filter: str
    confidence_threshold: float
    task_id: str
    streaming: bool
    user_id: str
    session_id: str
    # memory
    memory_context: str
    # plan
    subtasks: list[str]
    # fan-out accumulation
    subtask_results: Annotated[list[SubtaskResult], operator.add]
    errors: Annotated[list[dict], operator.add]
    # audit
    verified: list[AuditedClaim]
    uncertain: list[AuditedClaim]
    unverifiable: list[str]
    # compare
    comparison: dict
    # synthesize
    summary: str


# ── Nodes ────────────────────────────────────────────────────────────────────


async def recall_memory_node(state: FinSightState) -> dict:
    """Best-effort — a memory outage must never block a query. Populates
    ``memory_context``, which only ever reaches the Planner prompt: it shapes
    what gets searched for, never what gets asserted as a claim, so it cannot
    become an unverified figure in the synthesized report."""
    if not settings.memory_enabled:
        return {"memory_context": ""}

    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    if not user_id or not session_id:
        return {"memory_context": ""}

    try:
        service = get_memory_service()
        short_term = await service.recall_session(session_id)
        long_term = await service.recall_long_term(user_id, state["query"])
    except Exception as exc:  # noqa: BLE001 - degrade, don't block the query
        _emit("error", stage="MemoryAgent", detail=str(exc))
        return {"memory_context": ""}

    memory_context = format_memory_context(short_term, long_term)
    _emit("memory_recalled", short_term=len(short_term), long_term=len(long_term))
    return {"memory_context": memory_context}


async def plan_node(state: FinSightState) -> dict:
    subtasks = await plan_task(
        state["query"],
        streaming=state.get("streaming", False),
        memory_context=state.get("memory_context", ""),
    )
    _emit("planned", subtasks=subtasks)
    return {"subtasks": subtasks}


def fan_out(state: FinSightState) -> list[Send]:
    """Conditional edge — one retrieve_analyze branch per subtask."""
    return [
        Send(
            "retrieve_analyze",
            {
                "subtask": subtask,
                "company_filter": state.get("company_filter", ""),
                "fiscal_year_filter": state.get("fiscal_year_filter", ""),
            },
        )
        for subtask in state["subtasks"]
    ]


async def retrieve_analyze_node(payload: dict) -> dict:
    """Retriever + Analyst for a single subtask. Never raises: a failing subtask
    contributes an error event and nothing else, exactly like the old pipeline."""
    subtask = payload["subtask"]

    try:
        ranked = await get_retrieval_service().retrieve(
            subtask,
            payload.get("company_filter", ""),
            payload.get("fiscal_year_filter", ""),
        )
    except Exception as exc:  # noqa: BLE001 - isolate one subtask's failure
        _emit("error", stage="RetrieverAgent", subtask=subtask, detail=str(exc))
        return {"errors": [{"stage": "RetrieverAgent", "subtask": subtask, "detail": str(exc)}]}

    chunks_data = [ranked_chunk_to_dict(r) for r in ranked]
    _emit("retrieved", subtask=subtask, chunks=len(chunks_data))

    if not chunks_data:
        return {}

    try:
        analysis = await analyze_chunks(subtask, chunks_data)
    except Exception as exc:  # noqa: BLE001
        _emit("error", stage="AnalystAgent", subtask=subtask, detail=str(exc))
        return {"errors": [{"stage": "AnalystAgent", "subtask": subtask, "detail": str(exc)}]}

    claims = analysis.get("claims", [])
    _emit("analyzed", subtask=subtask, claims=len(claims))

    return {
        "subtask_results": [
            {
                "subtask": subtask,
                "chunks": [c.get("chunk_id", "") for c in chunks_data],
                "claims": claims,
                "kpis": analysis.get("kpis", []),
            }
        ]
    }


async def audit_node(state: FinSightState) -> dict:
    """AuditorAgent — batch entailment over every extracted claim.

    Deliberately does not catch: an auditor failure aborts the whole run (the
    report must never ship unaudited claims)."""
    all_claims = [c for r in state.get("subtask_results", []) for c in r["claims"]]
    threshold = state.get("confidence_threshold") or settings.confidence_threshold

    verified, uncertain, unverifiable = await audit_claims(
        all_claims, {}, threshold, original_query=state["query"]
    )

    blocked = [c.claim for c in unverifiable]
    _emit("audited", verified=len(verified), uncertain=len(uncertain), blocked=len(blocked))
    return {"verified": verified, "uncertain": uncertain, "unverifiable": blocked}


async def compare_node(state: FinSightState) -> dict:
    """ComparatorAgent — cross-document deltas over unit-normalised results.

    Catches: a comparator failure degrades to an empty comparison rather than
    sinking a run that has verified claims."""
    results = [
        {"subtask": r["subtask"], "kpis": r["kpis"], "claims": r["claims"]}
        for r in state.get("subtask_results", [])
    ]
    normalized = normalize_subtask_results(results)

    try:
        comparison = await compare_results(normalized, state["query"])
    except Exception as exc:  # noqa: BLE001
        _emit("error", stage="ComparatorAgent", detail=str(exc))
        return {"comparison": dict(_EMPTY_COMPARISON)}

    _emit("compared", deltas=len(comparison.get("deltas", [])))
    return {"comparison": comparison}


async def synthesize_node(state: FinSightState) -> dict:
    summary = await synthesize_report(
        state["query"],
        state.get("verified", []),
        state.get("uncertain", []),
        state.get("comparison") or dict(_EMPTY_COMPARISON),
        state["task_id"],
    )
    return {"summary": summary}


async def remember_node(state: FinSightState) -> dict:
    """Best-effort write of this completed turn, for the next turn's recall.
    Never fails the run — a memory-write outage should not cost the user their
    already-synthesized report."""
    if not settings.memory_enabled:
        return {}

    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    if not user_id or not session_id:
        return {}

    try:
        text = build_turn_text(state["query"], state.get("subtasks", []), state.get("summary", ""))
        await get_memory_service().write(
            user_id=user_id, session_id=session_id, task_id=state["task_id"], text=text
        )
    except Exception as exc:  # noqa: BLE001 - degrade, don't fail an already-synthesized report
        _emit("error", stage="MemoryAgent", detail=str(exc))
        return {}

    _emit("remembered", session_id=session_id[:8])
    return {}


# ── Assembly ─────────────────────────────────────────────────────────────────


def build_graph():
    """Construct and compile the FinSight pipeline graph."""
    g = StateGraph(FinSightState)

    g.add_node("recall_memory", recall_memory_node)
    g.add_node("plan", plan_node)
    g.add_node("retrieve_analyze", retrieve_analyze_node)
    g.add_node("audit", audit_node)
    g.add_node("compare", compare_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("remember", remember_node)

    g.add_edge(START, "recall_memory")
    g.add_edge("recall_memory", "plan")
    g.add_conditional_edges("plan", fan_out, ["retrieve_analyze"])
    g.add_edge("retrieve_analyze", "audit")
    g.add_edge("retrieve_analyze", "compare")
    g.add_edge("audit", "synthesize")
    g.add_edge("compare", "synthesize")
    g.add_edge("synthesize", "remember")
    g.add_edge("remember", END)

    return g.compile()


_graph = None


def get_graph():
    """Return the process-wide compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def initial_state(
    query: str,
    task_id: str,
    *,
    company_filter: str | None = None,
    fiscal_year_filter: str | None = None,
    confidence_threshold: float | None = None,
    streaming: bool = False,
    user_id: str = "anonymous",
    session_id: str | None = None,
) -> FinSightState:
    return {
        "query": query,
        "task_id": task_id,
        "company_filter": company_filter or "",
        "fiscal_year_filter": fiscal_year_filter or "",
        "confidence_threshold": confidence_threshold or settings.confidence_threshold,
        "streaming": streaming,
        "user_id": user_id,
        "session_id": session_id or str(uuid.uuid4()),
        "subtask_results": [],
        "errors": [],
    }


def build_retrievals(state: FinSightState) -> dict[str, list[str]]:
    """subtask → retrieved chunk_ids, for the audit log."""
    return {r["subtask"]: r["chunks"] for r in state.get("subtask_results", [])}
