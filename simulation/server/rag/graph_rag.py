"""Graph RAG skeleton — `literature_rag` MCP tool 업그레이드 prep.

목적
----
현재 `static_citations.py` 의 20-entry 카탈로그 → 동적 graph-based RAG.
- PubMed abstracts (14,143 rows) + 우리 SQLite DB (disease_master)
- 엔터티 (disease/drug/region/week) ↔ 관계 (treats/causes/prevents) 그래프
- Multi-hop query: "2024 ILI 시즌 약한 이유 + COVID NPI 잔여 효과"

Status
------
**EXPERIMENTAL — NOT in the served / evaluated pipeline.** A working hybrid
retriever (TF-IDF + dense embedding + RRF fusion + mesh-graph expansion over the
PubMed corpus) is implemented below, but it is **not wired into the served
``epi.literature_rag`` MCP tool** — that tool uses the vector RAG in
``rag/__init__.py``. ``_extractive_answer`` is non-generative (extractive only).
Do NOT describe this as a production capability: the paper must report the
**vector RAG** as the served retriever and GraphRAG as prototyped future work.
(2026-06-06 D5 honesty relabel — was mislabeled "SKELETON / interface only".)

다음 sprint 계획
---------------
1. (1주) Microsoft GraphRAG 또는 LightRAG 통합 — 선택
2. (1주) PubMed → entity extraction (질병/약/지역) — Claude / Gemma 활용
3. (3일) Neo4j 또는 NetworkX in-memory graph 구축
4. (3일) `_h_literature_rag` 가 graph_rag 자동 호출 (옵션)
5. (3일) Self-RAG (답변 자기검증) + Conformal RAG (신뢰도 PI)

API 스펙 (구현 예정)
--------------------
    from simulation.server.rag.graph_rag import GraphRAG

    rag = GraphRAG(
        pubmed_dir="simulation/data/collected/pubmed_abstracts",
        sqlite_db="simulation/data/db/epi_real_seoul.db",
        backend="lightrag",   # "lightrag" | "microsoft_graphrag" | "neo4j"
        cache_dir="simulation/results/rag_index/graph",
    )

    # Multi-hop query
    answer = rag.query(
        "2024 ILI 시즌이 2019 대비 약한 이유",
        hop=3,
        with_citations=True,
        with_pi=True,    # Conformal RAG: 신뢰도
    )
    # → {
    #     "answer": "...",
    #     "evidence": [{"source": ..., "year": ..., "doi": ...}],
    #     "graph_path": [["2024 시즌", "vaccine_coverage_↑"], ["NPI", "...]]],
    #     "confidence": 0.78,
    # }
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 1. 환경 검증
# ══════════════════════════════════════════════════════════

def check_graph_rag_env() -> dict:
    """Graph RAG 백엔드 후보 + 의존성 검증."""
    status = {
        "lightrag": False,
        "microsoft_graphrag": False,
        "neo4j": False,
        "networkx": False,
        "ollama": False,
        "anthropic": False,
        "duckdb_vss": False,
    }

    try:
        import lightrag    # noqa: F401
        status["lightrag"] = True
    except ImportError:
        pass

    try:
        import graphrag    # noqa: F401  (Microsoft)
        status["microsoft_graphrag"] = True
    except ImportError:
        pass

    try:
        from neo4j import GraphDatabase    # noqa: F401
        status["neo4j"] = True
    except ImportError:
        pass

    try:
        import networkx    # noqa: F401
        status["networkx"] = True
    except ImportError:
        pass

    try:
        import shutil
        if shutil.which("ollama"):
            status["ollama"] = True
    except Exception:
        pass

    try:
        import anthropic    # noqa: F401
        status["anthropic"] = True
    except ImportError:
        pass

    try:
        import duckdb
        # VSS extension 가능 여부 확인
        con = duckdb.connect(":memory:")
        try:
            con.execute("INSTALL vss")
            con.execute("LOAD vss")
            status["duckdb_vss"] = True
        except Exception:
            pass
        con.close()
    except ImportError:
        pass

    return status


# ══════════════════════════════════════════════════════════
# 2. 데이터 source 검증
# ══════════════════════════════════════════════════════════

def list_pubmed_abstracts(pubmed_dir: str = "simulation/data/collected/pubmed_abstracts") -> dict:
    """PubMed abstract CSV 파일 목록 + 총 row 수."""
    p = Path(pubmed_dir)
    if not p.exists():
        return {"exists": False, "files": [], "n_rows": 0}

    files = sorted(p.glob("*.csv"))
    n_rows = 0
    for f in files:
        try:
            with f.open(encoding="utf-8") as fp:
                n_rows += sum(1 for _ in fp) - 1    # header 제외
        except Exception:
            pass

    return {
        "exists": True,
        "n_files": len(files),
        "n_rows": n_rows,
        "files_sample": [f.name for f in files[:5]],
    }


def list_disease_master(sqlite_db: str = "simulation/data/db/epi_real_seoul.db") -> dict:
    """Disease master 테이블 (그래프 entity 후보)."""
    if not Path(sqlite_db).exists():
        return {"exists": False, "n_diseases": 0}

    try:
        from simulation.database import safe_connect
        with safe_connect() as con:
            cur = con.execute("SELECT COUNT(DISTINCT disease_nm) FROM disease_master")
            n = cur.fetchone()[0]
            cur = con.execute("SELECT disease_nm FROM disease_master LIMIT 5")
            sample = [r[0] for r in cur.fetchall()]
        return {"exists": True, "n_diseases": n, "sample": sample}
    except Exception as e:
        log.debug(f"disease_master 조회 실패: {e}")
        return {"exists": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
# 3. 인터페이스 (구현 예정)
# ══════════════════════════════════════════════════════════

@dataclass
class GraphRAGConfig:
    pubmed_dir: str = "simulation/data/collected/pubmed_abstracts"
    sqlite_db: str = "simulation/data/db/epi_real_seoul.db"
    backend: str = "lightrag"   # "lightrag" | "microsoft_graphrag" | "neo4j"
    cache_dir: str = field(default_factory=lambda: str(get_results_dir() / "rag_index" / "graph"))
    # Local-only default: all-MiniLM-L6-v2 ships in the HF cache here, so dense
    # retrieval needs no download. "BAAI/bge-m3" (Korean-strong) is an optional
    # upgrade that triggers a ~2GB download — set it explicitly to opt in.
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm: str = "claude"   # "claude" | "gemma" | "openai"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 10


@dataclass
class GraphRAGResult:
    answer: str
    evidence: list[dict] = field(default_factory=list)
    graph_path: list[list[str]] = field(default_factory=list)
    confidence: Optional[float] = None
    model: Optional[str] = None


_DENSE_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
_GENERIC_MESH_DOC_CAP = 150  # mesh terms in more docs than this are too generic


def _split_mesh(raw: str) -> list[str]:
    """Split a mesh_terms / keywords cell into normalised terms."""
    if not raw:
        return []
    out: list[str] = []
    for part in str(raw).replace(";", "|").replace(",", "|").split("|"):
        t = part.strip().lower()
        if len(t) >= 3:
            out.append(t)
    return out


def _load_pubmed_corpus(pubmed_dir: str) -> list[dict]:
    """Load and pmid-dedup PubMed abstracts into doc dicts.

    Returns a list of ``{pmid,title,abstract,journal,year,mesh_terms}`` dicts.
    Robust to per-file read errors (skips the file, logs a warning).
    """
    import polars as pl

    files = sorted(Path(pubmed_dir).glob("*.csv"))
    frames = []
    for f in files:
        try:
            frames.append(pl.read_csv(f, infer_schema_length=0))
        except Exception as e:  # pragma: no cover - corrupt file guard
            log.warning("graph_rag: skip %s (%s)", f.name, e)
    if not frames:
        return []
    df = pl.concat(frames, how="vertical_relaxed")
    if "pmid" in df.columns:
        df = df.unique(subset=["pmid"], keep="last")
    docs: list[dict] = []
    for row in df.iter_rows(named=True):
        pmid = str(row.get("pmid") or "").strip()
        if not pmid:
            continue
        docs.append({
            "pmid": pmid,
            "title": (row.get("title") or "").strip(),
            "abstract": (row.get("abstract") or "").strip(),
            "journal": (row.get("journal") or "").strip(),
            "year": (row.get("year") or "").strip(),
            "mesh_terms": (row.get("mesh_terms") or "").strip(),
        })
    return docs


def _rrf_fuse(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of several (doc_idx, score) ranking lists.

    RRF score = sum_r 1/(k + rank_r(doc)). Robustly combines sparse and dense
    rankings without needing comparable score scales.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class GraphRAG:
    """Hybrid (sparse + dense) graph RAG over PubMed abstracts + disease master.

    Small interface — :meth:`build_index` then :meth:`query` — over a rich
    implementation: TF-IDF sparse retrieval, dense retrieval with a locally
    cached sentence-transformer (no download), Reciprocal Rank Fusion of the
    two, and mesh-term graph expansion for multi-hop recall. Degrades to
    sparse-only if sentence-transformers or the embedding model is unavailable.

    Side effects: caches dense embeddings under ``config.cache_dir``.
    """

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or GraphRAGConfig()
        self._docs: Optional[list[dict]] = None
        self._tfidf = None          # (vectorizer, sparse matrix)
        self._dense = None          # (model, np.ndarray embeddings) or None
        self._doc_mesh: list[set[str]] = []
        self._mesh_to_docs: dict[str, list[int]] = {}
        self._indexed = False

    def build_index(self, force: bool = False) -> dict:
        """Build sparse + dense + mesh-graph indices over the corpus.

        Args:
            force: re-encode dense embeddings even if a cache exists.

        Returns:
            ``{"n_docs", "sparse", "dense", "mesh_terms"}`` summary dict.

        Raises:
            RuntimeError: if no PubMed abstracts are found to index.
        """
        docs = _load_pubmed_corpus(self.config.pubmed_dir)
        if not docs:
            raise RuntimeError(
                f"no PubMed abstracts under {self.config.pubmed_dir}"
            )
        self._docs = docs
        corpus = [
            f"{d['title']} . {d['abstract']} . {d['mesh_terms']}" for d in docs
        ]
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), stop_words="english"
        )
        matrix = vec.fit_transform(corpus)
        self._tfidf = (vec, matrix)
        # Optional Rust BM25 sparse arm (opt-in via MPH_ARIA_RUST_INDEX). Default
        # OFF keeps the sklearn TF-IDF path unchanged → zero retrieval regression.
        # When on: tantivy BM25 if the wheel is installed, else an exact numpy
        # Okapi-BM25 fallback (portability; both give correct BM25 ranking).
        self._bm25 = None
        if os.environ.get("MPH_ARIA_RUST_INDEX", "").strip().lower() in {"1", "true", "yes", "on"}:
            from simulation.server.rag.rust_index import BM25Index
            self._bm25 = BM25Index(corpus)
            log.info("GraphRAG sparse arm = BM25 (%s backend)", self._bm25.backend)
        self._dense = self._build_dense(corpus, force)
        self._build_mesh_index(docs)
        self._indexed = True
        return {
            "n_docs": len(docs),
            "sparse": True,
            "dense": self._dense is not None,
            "mesh_terms": len(self._mesh_to_docs),
        }

    def _build_dense(self, corpus: list[str], force: bool):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            log.warning("graph_rag: sentence-transformers unavailable -> sparse-only")
            return None
        import numpy as np

        cache = Path(self.config.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        emb_path = cache / "dense_emb.npy"
        model = None
        for name in (self.config.embed_model, _DENSE_FALLBACK):
            try:
                model = SentenceTransformer(name)
                break
            except Exception as e:
                log.warning("graph_rag: dense model %s unavailable (%s)", name, e)
        if model is None:
            log.warning("graph_rag: no dense model loadable -> sparse-only")
            return None
        if emb_path.exists() and not force:
            emb = np.load(emb_path)
            if emb.shape[0] == len(corpus):
                return (model, np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0))
        emb = model.encode(
            corpus, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True,
        )
        # empty title+abstract docs normalize to NaN; zero them so they simply
        # carry no dense signal (sparse still indexes them) instead of poisoning
        # the matmul with NaN/overflow.
        emb = np.nan_to_num(np.asarray(emb, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        np.save(emb_path, emb)
        return (model, emb)

    def _build_mesh_index(self, docs: list[dict]) -> None:
        self._doc_mesh = [set(_split_mesh(d["mesh_terms"])) for d in docs]
        mesh_to_docs: dict[str, list[int]] = {}
        for i, terms in enumerate(self._doc_mesh):
            for t in terms:
                mesh_to_docs.setdefault(t, []).append(i)
        # drop ultra-generic terms that would over-connect the graph
        self._mesh_to_docs = {
            t: idxs for t, idxs in mesh_to_docs.items()
            if 2 <= len(idxs) <= _GENERIC_MESH_DOC_CAP
        }

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        bm25 = getattr(self, "_bm25", None)
        if bm25 is not None:                       # opt-in Rust/numpy BM25 arm
            return bm25.search(query, k)
        from sklearn.metrics.pairwise import linear_kernel

        vec, matrix = self._tfidf
        qv = vec.transform([query])
        sims = linear_kernel(qv, matrix).ravel()
        idx = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0.0]

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._dense is None:
            return []
        import numpy as np

        model, emb = self._dense
        qe = np.nan_to_num(
            np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        # float32 BLAS matmul can emit spurious divide/overflow RuntimeWarnings
        # on some CPUs even when the result is correct; suppress + sanitize.
        with np.errstate(all="ignore"):
            sims = np.nan_to_num(emb @ qe.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        idx = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in idx]

    def _graph_expand(self, seed_idx: list[int], hop: int) -> list[tuple[int, str, int]]:
        """Return (seed, shared_mesh, neighbour) links from mesh co-occurrence."""
        links: list[tuple[int, str, int]] = []
        seen: set[int] = set(seed_idx)
        frontier = list(seed_idx)
        for _ in range(max(0, hop - 1)):
            nxt: list[int] = []
            for s in frontier:
                for term in self._doc_mesh[s]:
                    for nb in self._mesh_to_docs.get(term, []):
                        if nb not in seen:
                            links.append((s, term, nb))
                            seen.add(nb)
                            nxt.append(nb)
                            if len(links) >= 40:
                                return links
            frontier = nxt
        return links

    def query(self, query: str, hop: int = 2,
              with_citations: bool = True,
              with_pi: bool = False,
              top_k: Optional[int] = None) -> GraphRAGResult:
        """Hybrid retrieve + mesh-graph expand for ``query``.

        Args:
            query: natural-language question.
            hop: mesh-graph expansion depth (1 = retrieval only).
            with_citations: include pmid/journal/year in each evidence row.
            with_pi: reserved for a future conformal-RAG confidence interval.
            top_k: number of primary hits (defaults to ``config.top_k``).

        Returns:
            :class:`GraphRAGResult` with an extractive answer, ranked evidence
            (each with a pmid citation), the mesh graph_path, and a confidence
            in [0, 1] from the top fused score.
        """
        if not self._indexed:
            self.build_index()
        top_k = int(top_k or self.config.top_k)
        pool = top_k * 3
        sparse = self._sparse_search(query, pool)
        dense = self._dense_search(query, pool)
        fused = _rrf_fuse([sparse, dense])[:top_k] if dense else sparse[:top_k]
        primary = [i for i, _ in fused]
        links = self._graph_expand(primary, hop) if hop > 1 else []

        evidence = []
        for idx, score in fused:
            d = self._docs[idx]
            row = {
                "snippet": (d["abstract"] or d["title"])[:300],
                "title": d["title"],
                "score": round(float(score), 5),
            }
            if with_citations:
                row.update(pmid=d["pmid"], journal=d["journal"], year=d["year"])
            evidence.append(row)

        graph_path = [
            [self._docs[a]["pmid"], term, self._docs[b]["pmid"]]
            for a, term, b in links
        ]
        top_cos = 0.0
        if dense:
            top_cos = max(top_cos, dense[0][1])
        if sparse:
            top_cos = max(top_cos, sparse[0][1])
        answer = self._extractive_answer(query, fused)
        return GraphRAGResult(
            answer=answer,
            evidence=evidence,
            graph_path=graph_path,
            confidence=round(float(max(0.0, min(top_cos, 1.0))), 4),
            model=(self.config.embed_model if self._dense else "tfidf-sparse-only"),
        )

    def _extractive_answer(self, query: str, fused: list[tuple[int, float]]) -> str:
        """Concatenate the most relevant abstract sentences as a grounded,
        non-generative answer (the LLM consultation layer does generation)."""
        if not fused:
            return "No relevant literature found in the indexed corpus."
        parts = []
        for idx, _ in fused[:3]:
            d = self._docs[idx]
            head = d["title"] or d["abstract"][:120]
            parts.append(f"[{d['pmid']}] {head}")
        return " ".join(parts)


# ══════════════════════════════════════════════════════════
# 4. main — 환경 검증 (수동 실행 용)
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Graph RAG 환경 검증")
    print("=" * 60)

    print("\n[1] Backend / 의존성")
    env = check_graph_rag_env()
    for k, v in env.items():
        flag = "✓" if v else "✗"
        print(f"  {flag} {k}")

    print("\n[2] PubMed abstracts")
    pm = list_pubmed_abstracts()
    print(f"  exists: {pm['exists']}, files: {pm.get('n_files', 0)}, "
          f"rows: {pm.get('n_rows', 0):,}")

    print("\n[3] Disease master")
    dm = list_disease_master()
    print(f"  exists: {dm['exists']}, n_diseases: {dm.get('n_diseases', 0)}")
    if dm.get("sample"):
        print(f"  sample: {dm['sample']}")

    print("\n[4] 권장 install (없는 것)")
    if not env["lightrag"]:
        print("  uv pip install lightrag-hku           # Hong Kong U LightRAG")
    if not env["networkx"]:
        print("  uv pip install networkx               # in-memory graph")
    if not env["duckdb_vss"]:
        print("  uv pip install duckdb                 # 이미 사용 중, VSS 추가")

    print()
    print("=" * 60)
    print("  Status: EXPERIMENTAL prototype — NOT wired into served epi.literature_rag")
    print("=" * 60)


if __name__ == "__main__":
    main()
