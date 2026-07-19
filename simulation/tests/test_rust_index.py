"""TDD for the BM25 sparse index (Rust tantivy + numpy fallback).

Verifies the numpy Okapi-BM25 fallback ranks correctly (so the index ALWAYS works
with zero compiled deps) and that the active backend matches wheel availability.
"""
from __future__ import annotations

from simulation.server.rag.rust_index import (
    RUST_INDEX_AVAILABLE,
    BM25Index,
    tokenize,
)

DOCS = [
    "influenza vaccine effectiveness in elderly adults",
    "commuter mobility and district transmission of a respiratory virus",
    "bayesian probabilistic forecasting of seasonal influenza incidence",
    "mask wearing behavior reduces contact rate during outbreaks",
]


def test_tokenize_alphanumeric_lowercase():
    assert tokenize("Influenza-A, H1N1!") == ["influenza", "a", "h1n1"]


def test_numpy_bm25_ranks_relevant_doc_first():
    idx = BM25Index(DOCS, prefer_rust=False)
    assert idx.backend == "numpy"
    hits = idx.search("influenza vaccine elderly", k=2)
    assert hits and hits[0][0] == 0                 # doc 0 is the vaccine/elderly doc


def test_numpy_bm25_second_query():
    idx = BM25Index(DOCS, prefer_rust=False)
    hits = idx.search("commuter district transmission", k=1)
    assert hits and hits[0][0] == 1                 # doc 1 is the commuter/district doc


def test_empty_query_and_no_overlap():
    idx = BM25Index(DOCS, prefer_rust=False)
    assert idx.search("", 3) == []
    assert idx.search("quantum blockchain cryptography", 3) == []   # no term overlap


def test_empty_corpus():
    idx = BM25Index([], prefer_rust=False)
    assert len(idx) == 0 and idx.search("influenza", 3) == []


def test_default_backend_matches_wheel_availability():
    idx = BM25Index(DOCS)                            # prefer_rust=True (default)
    assert idx.backend == ("tantivy" if RUST_INDEX_AVAILABLE else "numpy")
    hits = idx.search("forecasting seasonal influenza", k=1)
    assert hits and hits[0][0] == 2                  # doc 2 regardless of backend
