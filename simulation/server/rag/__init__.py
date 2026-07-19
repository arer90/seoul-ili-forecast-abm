"""Lightweight vector RAG over project bibliography.

Backend: LanceDB (embedded, Arrow-native, single directory).
Embedding: sentence-transformers all-MiniLM-L6-v2 (384d, 90MB, CPU-friendly).

Usage:
    # One-time index build from static citations (+ optional PDFs later):
    from simulation.server.rag import build_index, semantic_search
    build_index()                    # populates simulation/results/rag_index/

    # Query:
    hits = semantic_search("antiviral efficacy oseltamivir", k=5)

Falls back gracefully: if lancedb / sentence-transformers is missing,
`semantic_search` returns None and callers should use the static catalogue.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
_INDEX_DIR = get_results_dir() / "rag_index"
_TABLE_NAME = "citations"
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

_model_singleton = None
_db_singleton = None


def _load_model():
    """Lazy-load the sentence-transformer. Cached per process."""
    global _model_singleton
    if _model_singleton is not None:
        return _model_singleton
    try:
        from sentence_transformers import SentenceTransformer
        _model_singleton = SentenceTransformer(_EMBEDDING_MODEL, device="cpu")
        return _model_singleton
    except ImportError:
        log.debug("sentence-transformers not installed")
        return None


def _open_db():
    """Return LanceDB connection or None if unavailable."""
    global _db_singleton
    if _db_singleton is not None:
        return _db_singleton
    try:
        import lancedb
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _db_singleton = lancedb.connect(str(_INDEX_DIR))
        return _db_singleton
    except ImportError:
        log.debug("lancedb not installed")
        return None


def build_index(entries: Optional[list[dict]] = None, *, rebuild: bool = False) -> dict:
    """Build (or rebuild) the vector index from static citations + optional PDFs.

    `entries` = list of dicts with at least {id, title, abstract (or text), tags}.
    If None, pulls from simulation.server.static_citations.STATIC_CITATIONS.

    Returns {"indexed": N, "dim": D, "path": str}.
    """
    db = _open_db()
    model = _load_model()
    if db is None or model is None:
        return {"error": "lancedb or sentence-transformers unavailable"}

    if entries is None:
        try:
            from simulation.server.static_citations import STATIC_CITATIONS
            # STATIC_CITATIONS is a tuple of Citation dataclass — convert to dict
            entries = [c.to_dict() if hasattr(c, "to_dict") else c
                       for c in STATIC_CITATIONS]
        except ImportError:
            return {"error": "no entries + static_citations import failed"}

    # Prep rows
    # Body-text resolution order (for the embedding input AND the stored
    # `abstract` display field): `relevance` > `abstract` > `text`.
    # Rationale: the project's Citation dataclass uses `relevance` (a
    # 1-sentence plain-English note on WHY this paper matters to MPH) as the
    # best semantic summary; generic dicts may supply `abstract` instead.
    def _body(e: dict) -> str:
        for k in ("relevance", "abstract", "text"):
            v = e.get(k, "")
            if v:
                return v
        return ""

    rows = []
    texts = []
    for e in entries:
        # Accept dict or Citation (with to_dict)
        if hasattr(e, "to_dict"):
            e = e.to_dict()
        body = _body(e)
        text = " ".join(filter(None, [
            e.get("title", ""),
            body,
            " ".join(e.get("tags", []) or []),
        ])).strip()
        if not text:
            continue
        texts.append(text)
        rows.append({
            "id": str(e.get("id", "")),
            "title": e.get("title", ""),
            "abstract": body,  # same string the encoder saw — no fallback drift
            "year": int(e.get("year", 0) or 0),
            "tags": ",".join(e.get("tags", []) or []),
            "doi": e.get("doi", e.get("doi_or_url", "")),
        })

    if not rows:
        return {"error": "no valid entries to index"}

    # Encode
    import numpy as np
    log.info(f"encoding {len(texts)} citations with {_EMBEDDING_MODEL}...")
    vecs = model.encode(texts, show_progress_bar=False, batch_size=32,
                        convert_to_numpy=True, normalize_embeddings=True)
    # Silent zip truncation guard: model.encode should always return one vec
    # per text, but encode() has had bugs in some sentence-transformer versions
    # where batch-size mismatches drop rows. Fail loudly instead of silently
    # mis-labelling the index.
    if len(vecs) != len(rows):
        return {"error": f"encoder returned {len(vecs)} vectors for {len(rows)} rows"}
    for row, vec in zip(rows, vecs):
        row["vector"] = vec.astype(np.float32).tolist()

    # (Re)create table
    if _TABLE_NAME in db.table_names() and rebuild:
        db.drop_table(_TABLE_NAME)
    if _TABLE_NAME in db.table_names():
        tbl = db.open_table(_TABLE_NAME)
        tbl.add(rows)
    else:
        tbl = db.create_table(_TABLE_NAME, data=rows)

    return {"indexed": len(rows), "dim": _EMBEDDING_DIM, "path": str(_INDEX_DIR)}


def semantic_search(query: str, k: int = 5) -> Optional[list[dict]]:
    """Return top-k citation hits for `query`. None if backend unavailable."""
    db = _open_db()
    model = _load_model()
    if db is None or model is None:
        return None
    if _TABLE_NAME not in db.table_names():
        return None

    import numpy as np
    tbl = db.open_table(_TABLE_NAME)
    qvec = model.encode([query], show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)[0]
    hits = tbl.search(qvec.astype(np.float32)).limit(int(k)).to_list()
    # LanceDB default metric is L2. For normalized embeddings:
    #   L2_distance = sqrt(2 - 2*cos_sim)  ⇒  cos_sim = 1 - L2^2 / 2
    # Clamp to [0, 1] for presentation.
    return [
        {
            "id": h.get("id"),
            "title": h.get("title"),
            "abstract": (h.get("abstract") or "")[:500],
            "year": h.get("year"),
            "tags": (h.get("tags") or "").split(",") if h.get("tags") else [],
            "doi": h.get("doi") or None,
            "score": round(max(0.0, min(1.0, 1.0 - float(h.get("_distance", 0))**2 / 2.0)), 4),
        }
        for h in hits
    ]


def rag_info() -> dict:
    """Diagnostic."""
    db = _open_db()
    model = _load_model()
    return {
        "lancedb_available": db is not None,
        "embedding_model_available": model is not None,
        "index_dir": str(_INDEX_DIR),
        "table_exists": _TABLE_NAME in (db.table_names() if db else []),
    }
