"""Construction and wiring tests for the LangGraph orchestration.

No network: the agent entry points are monkeypatched, so this exercises the
graph topology, fan-out, parallel audit/compare superstep, and state
accumulation — not the LLM or Qdrant.
"""

from __future__ import annotations

import pytest

from core.models import AuditedClaim, AuditStatus, Citation
from orchestration import graph as G


def _claim(text: str, status: AuditStatus = AuditStatus.VERIFIED) -> AuditedClaim:
    return AuditedClaim(
        claim=text,
        citation=Citation(
            document="doc.pdf",
            page=1,
            snippet=text,
            claim=text,
            confidence=0.9,
            section_type="audited_financials",
        ),
        audit_status=status,
        audit_reason="test",
    )


def test_build_graph_has_all_nodes():
    compiled = G.build_graph()
    nodes = set(compiled.get_graph().nodes)
    assert {"plan", "retrieve_analyze", "audit", "compare", "synthesize"} <= nodes


def test_get_graph_is_singleton():
    assert G.get_graph() is G.get_graph()


def test_fan_out_emits_one_send_per_subtask():
    state = {
        "subtasks": ["a b c d e", "f g h i j"],
        "company_filter": "Infosys",
        "fiscal_year_filter": "",
    }
    sends = G.fan_out(state)
    assert len(sends) == 2
    assert all(s.node == "retrieve_analyze" for s in sends)
    assert sends[0].arg["subtask"] == "a b c d e"
    assert sends[0].arg["company_filter"] == "Infosys"


def test_build_retrievals_maps_subtask_to_chunk_ids():
    state = {
        "subtask_results": [
            {"subtask": "s1", "chunks": ["c1", "c2"], "claims": [], "kpis": []},
            {"subtask": "s2", "chunks": ["c3"], "claims": [], "kpis": []},
        ]
    }
    assert G.build_retrievals(state) == {"s1": ["c1", "c2"], "s2": ["c3"]}


def test_initial_state_defaults(monkeypatch):
    st = G.initial_state("q", "tid")
    assert st["query"] == "q" and st["task_id"] == "tid"
    assert st["subtask_results"] == [] and st["errors"] == []
    assert st["streaming"] is False


class _FakeRetrieval:
    async def retrieve(self, subtask, company_filter="", fiscal_year_filter=""):
        return []  # ranked_chunk_to_dict is never reached


@pytest.fixture
def stub_agents(monkeypatch):
    async def fake_plan(query, *, streaming=False):
        return ["subtask one xxxx", "subtask two xxxx"]

    calls: dict[str, int] = {"analyze": 0, "audit": 0, "compare": 0, "synth": 0}

    async def fake_retrieve_service_retrieve(
        self, subtask, company_filter="", fiscal_year_filter=""
    ):
        # Return one chunk-like object; ranked_chunk_to_dict is patched below.
        return [object()]

    def fake_to_dict(r):
        return {
            "chunk_id": "c1",
            "text": "t",
            "source": "d.pdf",
            "page": 1,
            "section_type": "notes",
            "company": "Infosys",
            "fiscal_year": "2024",
            "confidence": 0.8,
        }

    async def fake_analyze(subtask, chunks):
        calls["analyze"] += 1
        return {
            "kpis": [],
            "claims": [
                {
                    "claim": f"claim for {subtask}",
                    "supporting_text": "x",
                    "source_doc": "d.pdf",
                    "page": 1,
                    "confidence": 0.9,
                    "section_type": "notes",
                }
            ],
        }

    async def fake_audit(claims, chunks_by_subtask, threshold=None, original_query=""):
        calls["audit"] += 1
        return [_claim(c["claim"]) for c in claims], [], []

    async def fake_compare(results, query):
        calls["compare"] += 1
        return {"deltas": [{"metric": "m"}], "cross_document_claims": [], "summary": "s"}

    async def fake_synth(query, verified, uncertain, comparison, task_id):
        calls["synth"] += 1
        return "## Executive Summary\nok"

    monkeypatch.setattr(G, "plan_task", fake_plan)
    monkeypatch.setattr(G, "get_retrieval_service", lambda: _FakeRetrieval())
    monkeypatch.setattr(_FakeRetrieval, "retrieve", fake_retrieve_service_retrieve, raising=False)
    monkeypatch.setattr(G, "ranked_chunk_to_dict", fake_to_dict)
    monkeypatch.setattr(G, "analyze_chunks", fake_analyze)
    monkeypatch.setattr(G, "audit_claims", fake_audit)
    monkeypatch.setattr(G, "compare_results", fake_compare)
    monkeypatch.setattr(G, "synthesize_report", fake_synth)
    G._graph = None  # force rebuild against patched names
    yield calls
    G._graph = None


async def test_end_to_end_graph_run(stub_agents):
    calls = stub_agents
    final = await G.build_graph().ainvoke(G.initial_state("compare margins", "tid-1"))

    assert final["subtasks"] == ["subtask one xxxx", "subtask two xxxx"]
    assert len(final["subtask_results"]) == 2  # one per subtask, accumulated
    assert calls["analyze"] == 2
    assert calls["audit"] == 1 and calls["compare"] == 1 and calls["synth"] == 1
    assert len(final["verified"]) == 2
    assert final["unverifiable"] == []
    assert final["comparison"]["deltas"] == [{"metric": "m"}]
    assert final["summary"].startswith("## Executive Summary")
    assert G.build_retrievals(final) == {"subtask one xxxx": ["c1"], "subtask two xxxx": ["c1"]}


async def test_comparator_failure_degrades_to_empty(stub_agents, monkeypatch):
    async def boom(results, query):
        raise RuntimeError("comparator down")

    monkeypatch.setattr(G, "compare_results", boom)
    final = await G.build_graph().ainvoke(G.initial_state("q", "tid-2"))
    assert final["comparison"] == {"deltas": [], "cross_document_claims": [], "summary": ""}
    assert final["summary"].startswith("## Executive Summary")  # run still completes


async def test_auditor_failure_aborts_run(stub_agents, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("auditor down")

    monkeypatch.setattr(G, "audit_claims", boom)
    with pytest.raises(Exception, match="auditor down"):
        await G.build_graph().ainvoke(G.initial_state("q", "tid-3"))
