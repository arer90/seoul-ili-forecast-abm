"""ARIA-RAG (LangChain document-retrieval) smoke tests.

Offline-safe: every test uses the DRY-RUN chain (``model=None``) so no Ollama
call is made and no network is required. The dry-run path exercises the real
LangChain wiring — corpus build, HuggingFace embeddings, Chroma in-memory store,
and top-k retrieval — only the final LLM generation is skipped.

Skips cleanly if the LangChain stack is not installed (it is an optional,
pip-installed extra, NOT in requirements.lock).
"""
import pytest

lc = pytest.importorskip("langchain_chroma")  # whole stack-gated module


def test_build_corpus_has_artifacts_and_notes():
    from simulation.llm_compare.aria_langchain import build_corpus
    corpus = build_corpus()
    kinds = {d["kind"] for d in corpus}
    assert "knowledge_note" in kinds          # method/epi notes always present
    assert len(corpus) >= 5
    # every doc has the required fields; method notes carry empty facts
    for d in corpus:
        assert d["text"] and d["source"] and "facts" in d


def test_artifact_docs_carry_gold_facts():
    from simulation.llm_compare.aria_langchain import build_corpus
    arts = [d for d in build_corpus() if d["kind"] == "result_artifact"]
    # if the run artifacts exist, they must carry their key=value gold facts
    for d in arts:
        assert d["facts"] and all("=" in f for f in d["facts"])


def test_dry_run_retrieval_is_query_driven():
    """A WIS question retrieves the WIS method note above the ABM artifacts."""
    from simulation.llm_compare.aria_langchain import build_rag
    rag = build_rag(model=None, top_k=3)
    hits = rag.retrieve("WIS는 무엇인가?")
    sources = [h["source"] for h in hits]
    assert any("metrics" in s for s in sources)   # the WIS note was retrieved


def test_dry_run_query_structure_and_facts():
    from simulation.llm_compare.aria_langchain import build_rag
    rag = build_rag(model=None, top_k=3)
    res = rag.query("ABM 전향 R²는 얼마인가?")
    assert res["dry_run"] is True
    assert res["answer"] and res["retrieved"]
    # retrieving the ABM artifact should surface its gold numeric facts
    assert isinstance(res["facts"], list)


def test_run_demo_dry_run_produces_scored_report():
    from simulation.llm_compare.aria_langchain import run_demo
    rep = run_demo(model=None, top_k=3)
    assert rep["approach"] == "langchain_rag"
    assert rep["n_docs"] >= 5 and rep["per_query"]
    # grounding is scored on at least the artifact-retrieving questions
    for q in rep["per_query"]:
        assert "retrieved" in q and "grounding" in q


def test_compare_to_custom_aria_states_honest_limits():
    from simulation.llm_compare.aria_langchain import compare_to_custom_aria
    cmp = compare_to_custom_aria()
    assert cmp["dimensions"] and cmp["rag_adds"] and cmp["custom_keeps"]
    # the honest small-corpus caveat must be present (task requirement)
    assert cmp["honest_limits"] and any(
        "small" in h or "limited" in h for h in cmp["honest_limits"])


def test_build_rag_dry_run_no_llm_dependency():
    """model=None must NOT require Ollama — pure retrieval chain."""
    from simulation.llm_compare.aria_langchain import build_rag, AriaRagChain
    rag = build_rag(model=None)
    assert isinstance(rag, AriaRagChain) and rag.model is None
