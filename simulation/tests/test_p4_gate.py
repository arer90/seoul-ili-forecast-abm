"""TDD for the mandatory P4 delivery gate (verify_grounding + guardrail).

Every numeric claim in an ARIA answer must trace to a retrieved/receipted fact.
A fabricated number is routed for review and NEVER silently delivered; a grounded
answer passes. This is the same leak-free arbiter the 3-LLM panel requires so the
LLM layer stays read-only (it cannot emit a self-generated epidemic number).
"""
from __future__ import annotations


def test_gate_passes_grounded_answer():
    from simulation.llm_compare.agentic_rag import guardrail
    facts = ["forward_r2_behavior_on=0.557", "forward_r2_behavior_off=0.0408"]
    ans = "행동 ON 구성의 R² 0.557 가 OFF 0.0408 보다 우세합니다."
    gate = guardrail(ans, facts, question="behavior on vs off?")
    assert gate["safe"] is True and gate["action"] == "pass"


def test_gate_rejects_fabricated_number():
    from simulation.llm_compare.agentic_rag import guardrail
    facts = ["forward_r2_behavior_on=0.557"]
    ans = "이 모형은 R² 0.999 로 완벽하며 정확도 0.87 입니다."  # 0.999/0.87 not in facts
    gate = guardrail(ans, facts, question="how good?")
    assert gate["safe"] is False and gate["action"] == "route_for_review"


def test_crew_gate_end_to_end_blocks_hallucination(monkeypatch):
    """The blackboard-backed crew + gate blocks a hallucinated number end to end."""
    from simulation.llm_compare.aria_multiagent import MultiAgentARIA
    from simulation.llm_compare.agentic_rag import guardrail
    crew = MultiAgentARIA(mock=True)
    # force the Analyst to keep emitting a fabricated number (even on revision)
    monkeypatch.setattr(crew, "_ask",
                        lambda k, p: "R² 0.999 로 완벽." if k == "analyst"
                        else crew._mock_reply(k, p))
    res = crew.consult("행동 ABM 전향 예측 R²는?")
    assert res["verification"]["grounded"] is False        # verifier caught it
    gate = guardrail(res["final_answer"], res["retrieved_facts"], question="q")
    assert gate["safe"] is False                           # gate refuses delivery


def test_crew_gate_end_to_end_passes_grounded():
    from simulation.llm_compare.aria_multiagent import MultiAgentARIA
    from simulation.llm_compare.agentic_rag import guardrail
    crew = MultiAgentARIA(mock=True)
    res = crew.consult("행동 ABM 전향 예측 R²는?")
    # mock analyst cites gold values verbatim → grounded → gate passes
    assert res["verification"]["n_spurious"] == 0
    gate = guardrail(res["final_answer"], res["retrieved_facts"], question="q")
    assert gate["safe"] is True
