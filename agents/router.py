from __future__ import annotations

import json

import structlog
from semantic_kernel.functions import KernelArguments

from core.sk_kernel import get_kernel

log = structlog.get_logger()


async def plan_task(user_task: str) -> list[str]:
    """
    PlannerAgent: invokes the Planner.decompose semantic function via the SK
    kernel. The kernel renders the {{$user_task}} prompt template, dispatches
    to Groq, and returns the result — no raw chat_completion call here.

    This is load-bearing SK: the plan changes dynamically with every query
    because the LLM generates it, not a hardcoded graph topology.
    """
    kernel = get_kernel()
    log.info("planner.start", task=user_task[:100])

    result = await kernel.invoke(
        plugin_name="Planner",
        function_name="decompose",
        arguments=KernelArguments(user_task=user_task),
    )
    content = str(result).strip()

    try:
        subtasks = json.loads(content)
        if isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks):
            log.info("planner.complete", subtasks=len(subtasks))
            return subtasks
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: parse bullet/numbered list
    lines = [ln.strip().lstrip("-•1234567890.) ") for ln in content.splitlines() if ln.strip()]
    subtasks = [ln for ln in lines if len(ln) > 10][:6]
    log.warning("planner.fallback_parse", subtasks=len(subtasks))
    return subtasks if subtasks else [user_task]
