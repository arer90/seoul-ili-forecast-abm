"""Smoke tests for agentic-RAG loop + critic + runtime guardrail (offline, stubbed).

Run per-file:
    .venv/bin/python -m pytest simulation/tests/test_agentic_rag.py -q
"""
from simulation.llm_compare import agentic_rag as ar
from simulation.llm_compare.backends import LLMResponse


def test_ungrounded_numbers():
    assert ar.ungrounded_numbers("VE is 60% and R0 is 1.3", ["VE is 60% in trials"]) == ["1.3"]
    assert ar.ungrounded_numbers("VE 60%", ["coverage 60%"]) == []


def test_critic_flags_ungrounded():
    c = ar.critic_review("q", "Efficacy is 99%.", ["Efficacy around 60%."])
    assert c["grounded"] is False
    assert "99" in str(c["ungrounded_numbers"])


def test_critic_grounded_passes():
    c = ar.critic_review("q", "Efficacy is 60%.", ["Efficacy is 60% per the trial."])
    assert c["grounded"] is True


def test_guardrail_routes_ungrounded():
    g = ar.guardrail("R0 is 1.3", ["no numbers here"], question="q")
    assert g["action"] == "route_for_review" and g["safe"] is False


def test_guardrail_passes_grounded():
    g = ar.guardrail("value 60%", ["the value is 60%"], question="q")
    assert g["action"] == "pass" and g["safe"] is True


def test_agentic_retrieve_loop():
    calls = {"n": 0}
    def retriever(q, k):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"id": "a", "title": "doc a"}]
        return [{"id": "b", "title": "doc b"}, {"id": "c", "title": "doc c"}]

    class J:
        tier = "mock"
        backend_id = "j"
        def __init__(self):
            self.calls = 0
        def is_available(self):
            return True
        def generate(self, prompt, **kw):
            if "Can the QUESTION be answered" in prompt:
                self.calls += 1
                return LLMResponse("j", "j", "NO" if self.calls == 1 else "YES", 1.0)
            if "alternative" in prompt:
                return LLMResponse("j", "j", "broader influenza query", 1.0)
            return LLMResponse("j", "j", "x", 1.0)

    out = ar.agentic_retrieve("q", retriever, k=3, max_rounds=3, judge=J(), min_hits=1)
    assert out["rounds"] >= 2
    assert len(out["hits"]) == 3            # a + b + c, de-duplicated
    assert len(out["queries"]) >= 2         # reformulated at least once


def test_agentic_answer_pipeline():
    def retriever(q, k):
        return [{"id": "1", "title": "flu", "abstract": "VE is 60%"}]
    def generator(q, hits):
        return "Vaccine efficacy is 60%."
    out = ar.agentic_answer("how effective?", retriever=retriever, generator=generator)
    assert out["answer"] == "Vaccine efficacy is 60%."
    assert out["guardrail"]["action"] == "pass"   # 60% is grounded in context
