"""ARIA 3-agent crew (AutoGen/Ollama) — structural + grounding-gate smoke tests.

Offline by design: every test runs in --mock mode (deterministic stub) so CI
never needs Ollama up. The grounding-gate (verify_grounding) tests assert the
leak-free arbiter discriminates grounded vs hallucinated answers, independent of
any LLM. matched / mismatch / empty / spurious / revision cases (6-8).
"""
import json


# ── Stage-1: real-artifact fact retrieval ─────────────────────────────────────
def test_retrieve_facts_three_artifacts():
    from simulation.llm_compare.aria_multiagent import retrieve_facts
    pool = retrieve_facts()
    assert set(pool) == {"forward", "counterfactual", "champion"}
    for t in pool:
        assert pool[t]["facts"] and pool[t]["source"].endswith(".json")
    # forward pool carries the real behavior-on/off R² facts
    fwd = " ".join(pool["forward"]["facts"])
    assert "forward_r2_behavior_on=" in fwd and "forward_r2_behavior_off=" in fwd
    # champion pool carries the real champion + SPA test
    champ = " ".join(pool["champion"]["facts"])
    assert "champion=FusedEpi" in champ and "spa_p_value=" in champ


# ── Stage-3: CoVe-style grounding gate discriminates ──────────────────────────
def test_verify_grounding_matched_grounded():
    from simulation.llm_compare.aria_multiagent import verify_grounding
    facts = ["forward_r2_behavior_on=0.557", "forward_r2_behavior_off=0.0408"]
    grounded = "행동 ON R²=0.557 가 OFF R²=0.0408 보다 우세합니다."
    v = verify_grounding(grounded, facts)
    assert v["grounded"] is True and v["n_spurious"] == 0 and v["fact_recall"] == 1.0


def test_verify_grounding_hallucinated_flagged():
    from simulation.llm_compare.aria_multiagent import verify_grounding
    facts = ["forward_r2_behavior_on=0.557", "forward_r2_behavior_off=0.0408"]
    hallucinated = "이 모형은 R²=0.99 로 완벽하며 정확도 0.87 입니다."  # neither in facts
    v = verify_grounding(hallucinated, facts)
    assert v["grounded"] is False and v["n_spurious"] >= 1
    assert "0.99" in v["spurious_numbers"]


def test_verify_grounding_empty_and_mismatch_safe():
    from simulation.llm_compare.aria_multiagent import verify_grounding
    assert verify_grounding("", [])["grounded"] is False              # empty
    assert verify_grounding("숫자 없음", ["r2=0.5"])["grounded"] is False  # no cite
    assert verify_grounding("r2=0.5", ["r2=0.5"])["grounded"] is True     # matched


# ── Query routing ─────────────────────────────────────────────────────────────
def test_topic_router_picks_relevant_artifacts():
    from simulation.llm_compare.aria_multiagent import _topics_for_query
    assert "forward" in _topics_for_query("행동 ABM 전향 예측이 나은가?")
    assert "counterfactual" in _topics_for_query("백신 배분 전략은?")
    # unknown query → all topics (never empty)
    assert _topics_for_query("xyz") == ["forward", "counterfactual", "champion"]


# ── Full 3-agent pipeline (mock) ──────────────────────────────────────────────
def test_consult_mock_three_stage_trace():
    from simulation.llm_compare.aria_multiagent import MultiAgentARIA
    crew = MultiAgentARIA(mock=True)
    res = crew.consult("행동을 켠 ABM 전향 예측이 나은가? 수치 근거로.")
    roles = [s["role"] for s in res["trace"]]
    assert roles[:3] == ["Retriever/Grounder", "Analyst", "Verifier/Critic"]
    assert res["final_answer"] and "verification" in res
    # mock analyst cites gold values verbatim → stays grounded
    assert res["verification"]["n_spurious"] == 0


def test_run_demo_mock_payload_shape():
    from simulation.llm_compare.aria_multiagent import run_demo
    p = run_demo(mock=True)
    assert p["title_realized"] == "3-Agent LLM Architecture"
    assert p["ollama_api_key_required"] is False
    assert len(p["agents"]) == 3
    assert p["comparison"]["multiagent_crew"]["agents"] == 3
    assert p["comparison"]["single_pass_aria"]["agents"] == 1
    assert p["comparison"]["multiagent_crew"]["verification_stage"] is True
    assert len(p["consultations"]) == 2
    # honest: limitations recorded
    assert any("small" in lim.lower() for lim in p["honest_limitations"])
    # serializable
    json.dumps(p, ensure_ascii=False)


def test_revision_path_on_spurious(monkeypatch):
    """When the Analyst emits a spurious number, the gate triggers one revision."""
    from simulation.llm_compare import aria_multiagent as M
    crew = M.MultiAgentARIA(mock=True)
    calls = {"n": 0}
    orig = crew._mock_reply

    def _bad_then_good(key, prompt):
        if key == "analyst" and calls["n"] == 0:
            calls["n"] += 1
            return "이 모형은 R²=0.999 로 완벽합니다."  # spurious → triggers revision
        return orig(key, prompt)

    monkeypatch.setattr(crew, "_ask", lambda k, p: _bad_then_good(k, p))
    res = crew.consult("행동 ABM 전향 예측 R²는?")
    assert res["revised"] is True
    assert any(s["role"] == "Analyst(revision)" for s in res["trace"])
