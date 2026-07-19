"""TDD for the six read-only research specialists.

Pins the bounded-tool boundary (an agent calls ONLY its allowlisted read-only
tools) and the provenance invariant (every number a specialist writes is a tool
receipt) — the two properties that keep the layer read-only / non-transmission.
"""
from __future__ import annotations

import asyncio

from simulation.llm_compare.blackboard import EvidenceBlackboard
from simulation.llm_compare.specialists import (
    SPECIALISTS,
    Specialist,
    call_allowlisted,
    extract_leaf_facts,
    route_specialists,
    run_specialist,
)


class _FakeResult:
    def __init__(self, content, is_error=False):
        self.content = content
        self.is_error = is_error


class _FakeServer:
    """Records tool calls and returns a numeric payload."""

    def __init__(self):
        self.calls = []

    def call_tool(self, name, args):
        self.calls.append(name)
        return _FakeResult({"status": "ok",
                            "details": {"score": 0.557, "n": 48},
                            "provenance": {"server_version": "x"}})


def _spec(name):
    return next(s for s in SPECIALISTS if s.name == name)


# ── allowlist / tool boundary ─────────────────────────────────────────────────
def test_specialist_calls_only_allowlisted_tools():
    srv = _FakeServer()
    bb = EvidenceBlackboard()
    spec = _spec("forecast_intelligence")
    asyncio.run(run_specialist(spec, srv, bb))
    assert srv.calls == list(spec.tools)          # exactly its allowlist, nothing else


def test_call_allowlisted_rejects_foreign_tool():
    srv = _FakeServer()
    spec = _spec("forecast_intelligence")          # not allowed: epi.query_db
    res = call_allowlisted(srv, spec, "epi.query_db", {"sql": "SELECT 1"})
    assert res.is_error is True
    assert res.content["error"] == "tool_not_allowed"
    assert srv.calls == []                          # foreign tool never dispatched


def test_every_number_written_is_receipted():
    srv = _FakeServer()
    bb = EvidenceBlackboard()
    spec = _spec("statistical_verification")        # 2 tools
    out = asyncio.run(run_specialist(spec, srv, bb))
    assert out["facts_written"] > 0
    for e in bb.snapshot():
        if e.value is not None and isinstance(e.value, (int, float)):
            assert e.provenance and e.provenance.get("tool")     # tool receipt present
    # the receipted number really appears in its tool payload
    facts = bb.facts_for_verifier()
    assert any("0.557" in f for f in facts) and any("48" in f for f in facts)


# ── leaf extraction ───────────────────────────────────────────────────────────
def test_extract_leaf_facts_skips_provenance():
    leaves = dict(extract_leaf_facts(
        {"status": "ok", "details": {"score": 0.9},
         "provenance": {"server_version": "0.1.0", "db_vintage_ts": "t"}}))
    assert "details.score" in leaves and leaves["details.score"] == 0.9
    assert not any("server_version" in k for k in leaves)   # bookkeeping skipped


# ── routing ───────────────────────────────────────────────────────────────────
def test_route_specialists_by_keyword():
    picked = {s.name for s in route_specialists("백신 배분 전략은?")}
    assert "intervention_optimization" in picked
    picked_fc = {s.name for s in route_specialists("champion 예측 WIS는?")}
    assert "forecast_intelligence" in picked_fc
    assert route_specialists("xyz unknown") == SPECIALISTS   # none match → all


# ── real server smoke (offline; numbers may be sparse when data unavailable) ───
def test_real_server_smoke_read_only_and_provenanced():
    from simulation.server.mcp_epi import EpiMCPServer
    srv = EpiMCPServer()
    bb = EvidenceBlackboard()
    spec = _spec("statistical_verification")
    out = asyncio.run(run_specialist(spec, srv, bb))
    assert set(r["tool"] for r in out["receipts"]) == set(spec.tools)
    for e in bb.snapshot():                                   # all writes receipted
        assert e.provenance and e.provenance.get("tool", "").startswith("epi.")
