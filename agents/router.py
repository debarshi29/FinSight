from __future__ import annotations

import json

import structlog

from core.groq_client import chat_completion, chat_completion_hedged
from core.prompts import PLANNER_PROMPT

log = structlog.get_logger()


def parse_subtasks(content: str, fallback: str) -> list[str]:
    """Parse the Planner LLM output into a subtask list.

    Accepts a JSON array of strings; falls back to a bullet/numbered-list parse
    (lines > 10 chars, capped at 6); ultimately returns ``[fallback]``.
    """
    try:
        subtasks = json.loads(content)
        if isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks):
            return subtasks
    except (json.JSONDecodeError, ValueError):
        pass

    lines = [ln.strip().lstrip("-•1234567890.) ") for ln in content.splitlines() if ln.strip()]
    subtasks = [ln for ln in lines if len(ln) > 10][:6]
    return subtasks if subtasks else [fallback]


async def plan_task(user_task: str, *, streaming: bool = False) -> list[str]:
    """PlannerAgent — decompose a query into 2–6 ordered retrieval subtasks.

    A plain LLM call on the retry/reserve path (``chat_completion``), or the
    hedged path (``chat_completion_hedged``, ``hedge_after=8s``) when serving the
    streaming endpoint. The plan is regenerated per query — there is no fixed
    topology.
    """
    log.info("planner.start", task=user_task[:100], streaming=streaming)
    messages = [{"role": "user", "content": PLANNER_PROMPT.format(user_task=user_task)}]

    if streaming:
        text = await chat_completion_hedged(
            messages, hedge_after=8.0, max_tokens=500, temperature=0.0
        )
    else:
        text = await chat_completion(messages, max_tokens=500, temperature=0.0)

    subtasks = parse_subtasks(text, fallback=user_task)
    log.info("planner.complete", subtasks=len(subtasks))
    return subtasks
