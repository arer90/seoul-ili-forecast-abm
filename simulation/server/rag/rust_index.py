"""simulation.server.rag.rust_index
=====================================
A Rust-accelerated BM25 sparse index for the ARIA literature RAG, with a correct
pure-Python (numpy) Okapi-BM25 fallback.

Design (honest):
  * The LLM call dominates end-to-end latency by orders of magnitude, so the point
    of the Rust arm is BM25 **correctness** (real length normalization + saturating
    term frequency, which a TF-IDF cosine only approximates) and fast indexing —
    NOT a latency win the user will feel over a few-thousand-doc corpus.
  * Portability first (project principle #1, mirroring ``seir_core`` → numba): the
    Rust extension (``tantivy``) is a guarded import. When it is absent, an exact
    numpy Okapi-BM25 implementation is used, so this module ALWAYS works with zero
    compiled dependencies and returns BM25-ranked results either way.

Install the Rust arm (optional) from the pinned, isolated requirement file::

    pip install -r simulation/llm_compare/requirements-aria-rt.txt   # tantivy wheel

``tantivy`` ships prebuilt wheels (macOS/Linux/Windows), needs no Rust toolchain,
and has no python dependencies, so it is zero-churn — but it is never required.
"""
from __future__ import annotations

import math
import re
from typing import Optional

__all__ = ["RUST_INDEX_AVAILABLE", "BM25Index", "tokenize"]

try:  # guarded — mirrors simulation.abm.agent_kernel seir_core→numba fallback
    import tantivy  # type: ignore
    RUST_INDEX_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional wheel
    tantivy = None
    RUST_INDEX_AVAILABLE = False

# Unicode word tokens — keeps English alphanumerics AND CJK/Korean (space-separated
# Korean advisory memory would otherwise tokenize to nothing under [A-Za-z0-9]+).
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode-word tokenizer (shared by the numpy fallback)."""
    return _TOKEN.findall((text or "").lower())


class _NumpyBM25:
    """Exact Okapi BM25 over a small corpus, numpy-vectorised. Always available."""

    def __init__(self, docs: list[str], *, k1: float = 1.5, b: float = 0.75):
        import numpy as np
        self._np = np
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in toks], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        vocab: dict[str, int] = {}
        for t in toks:
            for w in set(t):
                vocab.setdefault(w, len(vocab))
        self.vocab = vocab
        n_docs, n_terms = len(docs), len(vocab)
        self.tf = np.zeros((n_docs, n_terms), dtype=np.float32)
        for i, t in enumerate(toks):
            for w in t:
                self.tf[i, vocab[w]] += 1.0
        df = (self.tf > 0).sum(axis=0)
        # BM25 idf with +1 smoothing (non-negative)
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        np = self._np
        q_ids = [self.vocab[w] for w in tokenize(query) if w in self.vocab]
        if not q_ids or self.tf.shape[0] == 0:
            return []
        tf = self.tf[:, q_ids]                                  # (n_docs, |q|)
        idf = self.idf[q_ids]                                   # (|q|,)
        denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[:, None] / (self.avgdl or 1.0)))
        scores = (idf * (tf * (self.k1 + 1.0)) / np.where(denom == 0, 1.0, denom)).sum(axis=1)
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0.0]


class _TantivyBM25:
    """tantivy (Rust) BM25 index over the corpus. Built only when the wheel exists."""

    def __init__(self, docs: list[str]):
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("body", stored=False, tokenizer_name="default")
        sb.add_integer_field("doc_id", stored=True, indexed=True)
        self._schema = sb.build()
        self._index = tantivy.Index(self._schema)
        writer = self._index.writer()
        for i, d in enumerate(docs):
            doc = tantivy.Document()
            doc.add_text("body", d or "")
            doc.add_integer("doc_id", i)
            writer.add_document(doc)
        writer.commit()
        self._index.reload()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        searcher = self._index.searcher()
        q = self._index.parse_query(" ".join(toks), ["body"])
        hits = searcher.search(q, k).hits
        out = []
        for score, addr in hits:
            doc = searcher.doc(addr)
            out.append((int(doc["doc_id"][0]), float(score)))
        return out


class BM25Index:
    """BM25 sparse index — Rust (tantivy) when available, exact numpy BM25 otherwise.

    Small interface (:meth:`search`), rich implementation: the backend is chosen at
    construction and hidden. Either backend returns BM25-ranked ``(doc_index,
    score)`` pairs, so callers get correct BM25 semantics regardless of whether the
    compiled wheel is installed.

    Args:
        docs: the corpus texts (one string per document).
        prefer_rust: use tantivy when available (default True); set False to force
            the numpy backend (used in tests to exercise the fallback).

    Attributes:
        backend: ``"tantivy"`` or ``"numpy"`` — which arm is active.
    """

    def __init__(self, docs: list[str], *, prefer_rust: bool = True):
        self._n = len(docs)
        if prefer_rust and RUST_INDEX_AVAILABLE:
            try:
                self._impl = _TantivyBM25(docs)
                self.backend = "tantivy"
                return
            except Exception:  # pragma: no cover - defensive: fall back on any tantivy error
                pass
        self._impl = _NumpyBM25(docs)
        self.backend = "numpy"

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Top-``k`` ``(doc_index, bm25_score)`` for ``query`` (descending)."""
        if self._n == 0:
            return []
        return self._impl.search(query, max(1, int(k)))

    def __len__(self) -> int:
        return self._n
