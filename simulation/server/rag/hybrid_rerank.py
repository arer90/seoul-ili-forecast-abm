"""Hybrid GraphRAG retrieval + optional LLM-judge reranking for literature_rag.

Additive, flag-gated, offline. Default OFF -> the served retriever stays the
vector RAG in :mod:`simulation.server.rag` (``_h_literature_rag``).

- ``MPH_GRAPH_RAG=1`` -> route literature_rag through the multi-hop hybrid
  GraphRAG (:mod:`simulation.server.rag.graph_rag`: TF-IDF sparse + dense MiniLM +
  Reciprocal-Rank-Fusion + mesh-graph expansion over the PubMed corpus), giving
  the hybrid (sparse+dense) retrieval the vector-only path lacks.
- ``MPH_RAG_RERANK=1`` -> rerank the top hits with an offline LLM judge. A true
  cross-encoder reranker model is NOT cached offline, so an LLM-as-reranker over
  the project's existing Ollama backend is used (graceful: if no judge backend is
  available the original fused order is kept).

Both degrade gracefully to ``None`` (caller falls back to the vector RAG) on any
failure, so enabling a flag can never crash the served tool.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)

_GRAPHRAG = None          # module singleton (index built once per process)
_GRAPHRAG_FAILED = False


def graph_rag_enabled() -> bool:
    """True iff MPH_GRAPH_RAG is truthy (default OFF — vector RAG stays served)."""
    return os.environ.get("MPH_GRAPH_RAG", "0") not in ("", "0", "false", "False")


def rerank_enabled() -> bool:
    """True iff MPH_RAG_RERANK is truthy (default OFF)."""
    return os.environ.get("MPH_RAG_RERANK", "0") not in ("", "0", "false", "False")


def _get_graphrag():
    """Lazily build the GraphRAG index once per process; None if unavailable.

    build_index is ~10 s over the PubMed corpus, so it is cached module-level and
    only attempted once (a failure latches so we do not re-pay the cost).
    """
    global _GRAPHRAG, _GRAPHRAG_FAILED
    if _GRAPHRAG is not None:
        return _GRAPHRAG
    if _GRAPHRAG_FAILED:
        return None
    try:
        from simulation.server.rag.graph_rag import GraphRAG
        g = GraphRAG()
        g.build_index()
        _GRAPHRAG = g
        return g
    except Exception as e:  # no corpus / no deps -> fall back to vector RAG
        log.warning("hybrid_rerank: GraphRAG unavailable (%s) -> vector RAG fallback", e)
        _GRAPHRAG_FAILED = True
        return None


def _pick_judge():
    """First available offline judge backend (Ollama/local preferred); None if none."""
    try:
        from simulation.llm_compare.backends import discover_backends
        backends = discover_backends()
        for tier in ("ollama", "local", "api"):
            for b in backends:
                if getattr(b, "tier", "") == tier and b.is_available():
                    return b
    except Exception:
        pass
    return None


def _rerank_score(text: str) -> Optional[float]:
    """Parse a 0-1 relevance score from a judge reply; None if unparseable."""
    if not text:
        return None
    m = re.search(r"(0?\.\d+|1\.0+|[01])", text)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


def _llm_rerank(query: str, hits: list[dict], *, backend=None,
                top_n: Optional[int] = None) -> list[dict]:
    """Rerank ``hits`` by LLM-judged query-document relevance; stable fallback.

    Args:
        query: the user query.
        hits: candidate hits (each a dict with ``title``/``abstract``).
        backend: optional judge backend; else auto-resolved (offline Ollama first).
        top_n: keep only the top-N after reranking (None = keep all, reordered).

    Returns:
        Hits reordered by descending judge score (with ``rerank_score`` added). If
        no judge backend is available the original order is preserved (graceful).

    Side effects: one short LLM judge call per hit when a backend is available.
    """
    b = backend or _pick_judge()
    if b is None or not hits:
        return hits[:top_n] if top_n else hits
    scored = []
    for h in hits:
        doc = (str(h.get("title", "")) + " " + str(h.get("abstract", "")))[:600]
        prompt = ("Rate how relevant this DOCUMENT is to the QUESTION on a 0-1 "
                  "scale. Output only the number.\nQUESTION: " + query
                  + "\nDOCUMENT: " + doc)
        s = None
        try:
            r = b.generate(prompt, temperature=0.0, max_tokens=16)
            if not getattr(r, "error", ""):
                s = _rerank_score(getattr(r, "text", "") or "")
        except Exception:
            s = None
        scored.append((s if s is not None else float(h.get("score") or 0.0), h))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    out = [dict(h, rerank_score=round(float(s), 3)) for s, h in scored]
    return out[:top_n] if top_n else out


def hybrid_rag_search(query: str, k: int = 5, *,
                      rerank: Optional[bool] = None) -> Optional[list[dict]]:
    """Hybrid GraphRAG retrieval (+ optional LLM rerank) in literature_rag format.

    Args:
        query: natural-language query.
        k: number of hits to return.
        rerank: force rerank on/off; None = honor ``MPH_RAG_RERANK``.

    Returns:
        A list of hit dicts ``{id, title, abstract, year, tags, doi, pmid, journal,
        score[, rerank_score]}`` (literature_rag shape), or None when GraphRAG is
        disabled/unavailable (caller falls back to the vector RAG).

    Performance: one-time ~10 s index build, then ~0.1 s/query; + one LLM call per
    candidate when reranking. Side effects: builds/caches the GraphRAG index.
    """
    if not graph_rag_enabled():
        return None
    g = _get_graphrag()
    if g is None:
        return None
    do_rerank = rerank if rerank is not None else rerank_enabled()
    try:
        # over-retrieve when reranking so the reranker has candidates to reorder
        res = g.query(query, top_k=max(k, k * 3 if do_rerank else k))
        evidence = getattr(res, "evidence", None) or []
    except Exception as e:
        log.warning("hybrid_rerank: GraphRAG query failed (%s)", e)
        return None
    hits = [{
        "id": str(e.get("pmid", "")),
        "title": e.get("title", ""),
        "abstract": (e.get("snippet") or "")[:500],
        "year": e.get("year"),
        "tags": [],
        "doi": None,
        "pmid": e.get("pmid"),
        "journal": e.get("journal"),
        "score": e.get("score"),
    } for e in evidence]
    if do_rerank:
        hits = _llm_rerank(query, hits, top_n=k)
    else:
        hits = hits[:k]
    return hits
