"""Smoke tests for hybrid GraphRAG + LLM-judge reranking (offline, deterministic).

Run per-file:
    .venv/bin/python -m pytest simulation/tests/test_hybrid_rerank.py -q

Verifies: (1) OFF by default -> None (vector RAG fallback); (2) score parsing;
(3) LLM-judge rerank reorders by judged relevance; (4) no judge backend -> order
preserved (graceful). Does NOT build the real GraphRAG index (kept fast).
"""
from simulation.server.rag import hybrid_rerank as hr
from simulation.llm_compare.backends import LLMResponse


class _StubJudge:
    tier = "mock"
    backend_id = "stub"
    def is_available(self):
        return True
    def generate(self, prompt, **kw):
        score = "0.9" if "vaccine" in prompt.lower() else "0.1"
        return LLMResponse(backend_id="stub", model="s", text=score, latency_ms=1.0)


def test_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("MPH_GRAPH_RAG", raising=False)
    assert hr.hybrid_rag_search("influenza vaccination", k=3) is None


def test_rerank_score():
    assert hr._rerank_score("0.9") == 0.9
    assert hr._rerank_score("relevance is 0.42 here") == 0.42
    assert hr._rerank_score("no number") is None


def test_llm_rerank_reorders():
    hits = [{"title": "general flu trends", "abstract": "", "score": 0.5},
            {"title": "vaccine efficacy study", "abstract": "", "score": 0.4}]
    out = hr._llm_rerank("best prevention method", hits, backend=_StubJudge())
    assert out[0]["title"] == "vaccine efficacy study"
    assert out[0]["rerank_score"] == 0.9


def test_llm_rerank_no_judge_preserves_order(monkeypatch):
    monkeypatch.setattr(hr, "_pick_judge", lambda: None)
    hits = [{"title": "a", "score": 0.5}, {"title": "b", "score": 0.4}]
    out = hr._llm_rerank("q", hits, top_n=2)
    assert [h["title"] for h in out] == ["a", "b"]
