from __future__ import annotations

from core.models import MemoryRecord
from memory.consolidate import build_turn_text, format_memory_context


def _record(text: str, memory_id: str = "m1") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id="alice",
        session_id="s1",
        task_id="t1",
        text=text,
        timestamp="2026-09-09T00:00:00Z",
    )


def test_build_turn_text_includes_query_plan_summary():
    text = build_turn_text(
        "Compare Infosys and TCS margins",
        ["Infosys margin FY2024", "TCS margin FY2024"],
        "Infosys margin declined to 20.7%",
    )
    assert "Compare Infosys and TCS margins" in text
    assert "Infosys margin FY2024" in text
    assert "20.7%" in text


def test_build_turn_text_handles_missing_subtasks_and_summary():
    text = build_turn_text("q", [], "")
    assert "(no subtasks)" in text
    assert "(no summary)" in text


def test_build_turn_text_truncates_long_summary():
    text = build_turn_text("q", ["s"], "x" * 1000)
    # "Summary: " + at most 400 chars
    summary_line = [ln for ln in text.splitlines() if ln.startswith("Summary:")][0]
    assert len(summary_line) <= len("Summary: ") + 400


def test_format_memory_context_empty_when_nothing_recalled():
    assert format_memory_context([], []) == ""


def test_format_memory_context_short_term_only():
    ctx = format_memory_context([_record("turn one"), _record("turn two", "m2")], [])
    assert "Recent turns in this session" in ctx
    assert "turn one" in ctx and "turn two" in ctx
    assert "Relevant past queries" not in ctx


def test_format_memory_context_long_term_only():
    ctx = format_memory_context([], [_record("past turn")])
    assert "Relevant past queries" in ctx
    assert "past turn" in ctx
    assert "Recent turns" not in ctx


def test_format_memory_context_both():
    ctx = format_memory_context([_record("recent")], [_record("distant")])
    assert "recent" in ctx and "distant" in ctx
    assert ctx.index("Recent turns") < ctx.index("Relevant past queries")
