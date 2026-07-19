"""Smoke tests for the LLM-judge RAGAS-equivalent metrics (offline, deterministic).

Run per-file (macOS test-suite policy):
    .venv/bin/python -m pytest simulation/tests/test_ragas_metrics.py -q

Verifies: (1) OFF by default -> deterministic proxy; (2) score parsing;
(3) enabled LLM-judge path with a deterministic stub backend; (4) judge-failure
falls back to the proxy. No live model is ever required.
"""
from simulation.llm_compare import ragas_metrics as rm
from simulation.llm_compare.backends import LLMResponse


def _stub(scores):
    """Deterministic judge stub: replies 'KEY=<score>' for whichever metric the
    prompt asks for (each prompt embeds exactly one of the four metric keys)."""
    class S:
        backend_id = "stub"
        tier = "mock"
        def is_available(self):
            return True
        def generate(self, prompt, **kw):
            for k, v in scores.items():
                if k in prompt:
                    return LLMResponse(backend_id="stub", model="stub",
                                       text=f"{k}={v}", latency_ms=1.0)
            return LLMResponse(backend_id="stub", model="stub", text="0.5", latency_ms=1.0)
    return S()


def test_parse_score():
    assert rm._parse_score("FAITHFULNESS=0.8", "FAITHFULNESS") == 0.8
    assert rm._parse_score("the score is 0.42 overall", "FAITHFULNESS") == 0.42
    assert rm._parse_score("no number here", "X") is None


def test_disabled_defaults_to_proxy(monkeypatch):
    monkeypatch.delenv("MPH_RAGAS_LLM", raising=False)
    out = rm.ragas_eval("why vaccinate?",
                        "Vaccination reduces influenza risk [data:ve].",
                        ["Vaccination reduces influenza risk in the elderly."])
    assert out["llm_judge_enabled"] is False
    assert out["per_metric_method"]["faithfulness"] == "proxy"
    assert isinstance(out["faithfulness"], float)


def test_enabled_llm_judge(monkeypatch):
    monkeypatch.setenv("MPH_RAGAS_LLM", "1")
    judge = _stub({"FAITHFULNESS": "0.75", "PRECISION": "0.6",
                   "RECALL": "0.9", "RELEVANCY": "0.85"})
    out = rm.ragas_eval("q", "a", ["c1", "c2"], backend=judge)
    assert out["method"] == "llm", out
    assert out["faithfulness"] == 0.75
    assert out["context_precision"] == 0.6
    assert out["context_recall"] == 0.9
    assert out["answer_relevancy"] == 0.85


def test_judge_failure_falls_back_to_proxy(monkeypatch):
    monkeypatch.setenv("MPH_RAGAS_LLM", "1")
    class Bad:
        backend_id = "bad"
        tier = "mock"
        def is_available(self):
            return True
        def generate(self, *a, **k):
            return LLMResponse(backend_id="bad", model="b", text="",
                               latency_ms=1.0, error="boom")
    out = rm.faithfulness_llm("Vaccination reduces influenza risk.",
                              ["Vaccination reduces influenza risk."], backend=Bad())
    assert out["method"] == "proxy", out
    assert isinstance(out["faithfulness"], float)


def test_empty_inputs_safe(monkeypatch):
    monkeypatch.delenv("MPH_RAGAS_LLM", raising=False)
    out = rm.ragas_eval("", "", [])
    assert "faithfulness" in out and "answer_relevancy" in out
