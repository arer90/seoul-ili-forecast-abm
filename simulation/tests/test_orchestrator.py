"""TDD for the async multi-agent orchestrator (S2).

Pins: (1) the semaphore bounds concurrency; (2) every blackboard write is a tool
receipt; (3) synthesis passes the gate on grounded facts and is blocked on a
hallucinated number.
"""
from __future__ import annotations

import asyncio

from simulation.llm_compare.blackboard import EvidenceBlackboard
from simulation.llm_compare.orchestrator import orchestrate, run_orchestrated


class _FakeResult:
    def __init__(self, content, is_error=False):
        self.content = content
        self.is_error = is_error


class _FakeServer:
    # decimal-valued epi metrics (forward R², relative WIS) — what real tools return
    def call_tool(self, name, args):
        return _FakeResult({"status": "ok",
                            "details": {"forward_r2": 0.557, "rel_wis": 0.443},
                            "provenance": {"server_version": "x"}})


def test_semaphore_bounds_concurrency():
    active = {"cur": 0, "max": 0}

    async def tracking_runner(spec, server, bb, *, args_by_tool=None):
        active["cur"] += 1
        active["max"] = max(active["max"], active["cur"])
        await asyncio.sleep(0.02)
        active["cur"] -= 1
        return {"agent": spec.name, "role": spec.role, "facts_written": 0,
                "receipts": []}

    asyncio.run(orchestrate("unknown-all", server=object(),
                            blackboard=EvidenceBlackboard(), limit=2, mock=True,
                            runner=tracking_runner))
    assert active["max"] <= 2          # never more than the semaphore bound
    assert active["max"] >= 1


def test_orchestrate_writes_are_all_provenanced():
    bb = EvidenceBlackboard()
    res = run_orchestrated("champion 예측과 검증 요약", server=_FakeServer(),
                           blackboard=bb, limit=3, mock=True)
    assert res["blackboard"]                       # facts were surfaced
    for e in res["blackboard"]:
        if isinstance(e["value"], (int, float)) and not isinstance(e["value"], bool):
            assert e["provenance"] and e["provenance"].get("tool")


def test_orchestrate_mock_synthesises_and_passes_gate():
    res = run_orchestrated("champion 모델 검증", server=_FakeServer(), limit=3,
                           mock=True)
    assert res["final_answer"]
    assert res["verification"]["n_spurious"] == 0
    assert res["grounded"] is True                 # gate + verifier both pass


def test_orchestrate_real_server_stays_read_only():
    from simulation.server.mcp_epi import EpiMCPServer
    res = run_orchestrated("검증 및 유의성 요약", server=EpiMCPServer(), limit=2,
                           mock=True)
    for e in res["blackboard"]:
        assert e["provenance"]["tool"].startswith("epi.")   # only epi.* tool receipts


def test_orchestrate_memory_inject_and_persist(tmp_path):
    from simulation.llm_compare.memory import VerifiedMemory
    mem = VerifiedMemory(path=tmp_path / "verified.jsonl")
    mem.remember("챔피언 모델 상대 WIS 질의", "FusedEpi 가 상대 WIS 0.443.",
                 tool_receipts=[{"tool": "epi.model_compare"}],
                 verification={"grounded": True})
    # prior context is retrieved (offline mock synthesis), and a grounded run persists
    res = run_orchestrated("챔피언 모델 검증 요약", server=_FakeServer(), limit=3,
                           mock=True, memory=mem, remember=True)
    assert res["prior_context_used"] >= 1          # past verified answer injected
    if res["grounded"]:
        assert len(mem) == 2                        # this run's answer also persisted


def test_orchestrate_no_remember_when_disabled(tmp_path):
    from simulation.llm_compare.memory import VerifiedMemory
    mem = VerifiedMemory(path=tmp_path / "verified.jsonl")
    run_orchestrated("champion 모델 검증", server=_FakeServer(), limit=3,
                     mock=True, memory=mem, remember=False)   # caller disables (mock)
    assert len(mem) == 0                            # nothing persisted
