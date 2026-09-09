"""Pure text-formatting helpers for turning a completed turn into a memory record,
and turning recalled records back into the block injected into the Planner prompt.

Deliberately no LLM call here: the turn text is built directly from the query,
plan, and summary already produced by the pipeline. A separate LLM-based
"consolidation" step would add another hallucination surface to a system whose
whole guarantee is traceability — so memory content is always something the
pipeline already said, never a new synthesis of it.
"""

from __future__ import annotations

from core.models import MemoryRecord


def build_turn_text(query: str, subtasks: list[str], summary: str) -> str:
    """Compact, embeddable summary of one completed turn."""
    plan = "; ".join(subtasks) if subtasks else "(no subtasks)"
    excerpt = summary.strip().replace("\n", " ")[:400] if summary else "(no summary)"
    return f"Query: {query}\nPlan: {plan}\nSummary: {excerpt}"


def format_memory_context(short_term: list[MemoryRecord], long_term: list[MemoryRecord]) -> str:
    """Render recalled records into the block appended to PLANNER_PROMPT.

    Empty when there is nothing to recall, so the placeholder degrades to a no-op
    rather than an awkward empty section header.
    """
    if not short_term and not long_term:
        return ""

    parts: list[str] = []
    if short_term:
        # Most-recent last, so the immediately preceding turn reads closest to
        # the new query in the prompt.
        recent = "\n".join(f"- {r.text}" for r in reversed(short_term))
        parts.append(f"Recent turns in this session:\n{recent}")
    if long_term:
        past = "\n".join(f"- {r.text}" for r in long_term)
        parts.append(f"Relevant past queries from this user:\n{past}")

    return "\n\n".join(parts)
