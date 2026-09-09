"""Drive the compiled graph and account for per-stage latency.

The API routes call :func:`run_pipeline`; it streams the graph, forwards custom
progress events to an optional callback (the SSE endpoint), keeps the final
accumulated state, and derives the six per-agent timings the ``MetricsStore``
expects.

Timing model: Retriever and Analyst share the fan-out superstep wall-clock;
Auditor and Comparator share their parallel superstep wall-clock. This matches
the streaming pipeline's accounting and is applied uniformly to both routes.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from orchestration.graph import FinSightState, get_graph

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


async def run_pipeline(
    state: FinSightState,
    *,
    on_event: EventCallback | None = None,
) -> tuple[FinSightState, dict[str, int]]:
    """Run the graph to completion.

    Returns ``(final_state, agent_latencies_ms)`` where the latency dict is keyed
    by the six ``*Agent`` names.
    """
    graph = get_graph()

    final: FinSightState = {}
    timings: dict[str, int] = {}

    t_start = time.time()
    t_plan = t_ra = t_ac = None

    async for mode, chunk in graph.astream(state, stream_mode=["values", "updates", "custom"]):
        if mode == "custom":
            if on_event is not None:
                await on_event(chunk)
        elif mode == "values":
            final = chunk  # full accumulated state after each superstep
        elif mode == "updates":
            now = time.time()
            for node in chunk or {}:
                if node == "plan":
                    t_plan = now
                    timings["PlannerAgent"] = int((now - t_start) * 1000)
                elif node == "retrieve_analyze":
                    t_ra = now
                elif node in ("audit", "compare"):
                    t_ac = now
                elif node == "synthesize":
                    base = t_ac or t_ra or t_plan or t_start
                    timings["SynthesizerAgent"] = int((now - base) * 1000)

    ra_ms = int(((t_ra or t_plan or t_start) - (t_plan or t_start)) * 1000)
    ac_ms = int(((t_ac or t_ra or t_start) - (t_ra or t_plan or t_start)) * 1000)
    timings.setdefault("PlannerAgent", 0)
    timings["RetrieverAgent"] = ra_ms
    timings["AnalystAgent"] = ra_ms
    timings["AuditorAgent"] = ac_ms
    timings["ComparatorAgent"] = ac_ms
    timings.setdefault("SynthesizerAgent", 0)

    return final, timings
