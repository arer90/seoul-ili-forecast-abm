"""Tests for the hybrid graph RAG and the RAG groundedness eval.

Covers the fusion/parse units, real-corpus retrieval quality (relevant, cited),
and the groundedness/citation metrics that make the RAG layer auditable.
"""
from __future__ import annotations

import pytest

from simulation.server.rag.graph_rag import GraphRAG, _rrf_fuse, _split_mesh
from simulation.llm_compare.judge import groundedness_score, citation_support


def test_rrf_fuse_combines_rankings() -> None:
    a = [(1, 0.9), (2, 0.8), (3, 0.7)]
    b = [(3, 0.95), (1, 0.6)]
    ids = [i for i, _ in _rrf_fuse([a, b])]
    assert set(ids[:2]) == {1, 3}


def test_split_mesh() -> None:
    assert _split_mesh("Influenza; Vaccines, Humans") == ["influenza", "vaccines", "humans"]
    assert _split_mesh("") == []


def test_groundedness_grounded_vs_hallucinated() -> None:
    ev = ["oseltamivir resistance emerged in pandemic H1N1 influenza in Korea"]
    assert groundedness_score("oseltamivir resistance influenza Korea", ev) > 0.9
    assert groundedness_score("quantum cryptography blockchain tokenomics dispute", ev) < 0.2
    assert groundedness_score("", ev) == 1.0
    assert groundedness_score("anything", []) == 0.0


def test_citation_support() -> None:
    assert citation_support("see [20394719] and [99999999]", ["20394719"]) == 0.5
    assert citation_support("no citation here", ["20394719"]) == 0.0
    assert citation_support("uses §4.13", ["§4.13"]) == 1.0


@pytest.fixture(scope="module")
def rag() -> GraphRAG:
    r = GraphRAG()
    r.build_index()
    return r


def test_build_index_real_corpus(rag: GraphRAG) -> None:
    assert rag._docs is not None and len(rag._docs) > 100
    assert rag._tfidf is not None


def test_query_relevant_cited_and_grounded(rag: GraphRAG) -> None:
    r = rag.query("oseltamivir antiviral resistance influenza", hop=2, top_k=5)
    assert r.evidence
    assert all(e.get("pmid") for e in r.evidence)
    blob = " ".join(e["title"].lower() for e in r.evidence)
    assert any(t in blob for t in ("influenza", "oseltamivir", "antiviral", "resistance"))
    assert 0.0 <= r.confidence <= 1.0
    ev_texts = [f"{e['snippet']} {e['title']}" for e in r.evidence]
    # the extractive answer is assembled from the evidence -> must be grounded
    assert groundedness_score(r.answer, ev_texts) > 0.5
