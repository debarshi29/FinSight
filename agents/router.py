from __future__ import annotations

import json

import structlog

from core.groq_client import chat_completion

log = structlog.get_logger()

_SYSTEM = (
    "You are a financial analysis planning agent. Given a user analysis task, "
    "decompose it into an ordered list of 2-6 specific retrieval subtasks. "
    "Each subtask must be a precise, searchable query targeting a specific "
    "financial metric, time period, or company.\n\n"
    "Output ONLY valid JSON — a list of strings. No markdown, no explanation.\n\n"
    "Example:\n"
    'Input: "Compare Infosys and TCS operating margins FY2022-2024"\n'
    'Output: ["Infosys operating margin FY2022", "Infosys operating margin FY2023", '
    '"Infosys operating margin FY2024", "TCS operating margin FY2022", '
    '"TCS operating margin FY2023", "TCS operating margin FY2024"]'
)


async def plan_task(user_task: str) -> list[str]:
    """PlannerAgent: decompose a user task into ordered subtasks using Groq."""
    log.info("planner.start", task=user_task[:100])

    content = await chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_task},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    try:
        subtasks = json.loads(content.strip())
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
