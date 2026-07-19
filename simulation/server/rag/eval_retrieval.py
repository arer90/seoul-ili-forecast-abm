"""D5 (M7 SCI-grade): retrieval-quality evaluation for the literature RAG.

A reviewer distrusts a RAG claim with no retrieval metrics. This harness reports
recall@k / hit@k / MRR over a hand-labelled gold set of query→citation pairs, for
whichever retriever is active:

  - ``static`` : the always-available token-overlap scorer (``score_citations``)
                 that backs ``epi.literature_rag`` when the vector index is absent.
  - ``vector`` : the LanceDB + all-MiniLM-L6-v2 semantic retriever (evaluated only
                 if ``lancedb`` / ``sentence-transformers`` + a built index exist).

The metrics turn "we have RAG" into "our RAG retrieves the relevant reference in
the top-k X% of the time" — a small but publishable evidence table. The gold set
is the 20-entry curated catalogue (``static_citations``); the PubMed corpus
(14k abstracts) is available for future indexing (see ``graph_rag.py``).

Run::

    python -m simulation.server.rag.eval_retrieval
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

#: (query, {relevant citation ids}) — hand-labelled against static_citations.
GOLD: tuple[tuple[str, frozenset[str]], ...] = (
    ("time-varying reproduction number estimation EpiEstim", frozenset({"cori_2013_epiestim"})),
    ("weighted interval score proper scoring rule probabilistic forecast", frozenset({"gneiting_2007_wis"})),
    ("Diebold-Mariano test comparing predictive accuracy", frozenset({"diebold_mariano_1995"})),
    ("FluSight interval forecast evaluation coverage", frozenset({"flusight_bracher_2021"})),
    ("model confidence set forecasting comparison", frozenset({"hansen_mcs_2011"})),
    ("SEIR compartmental mathematical theory of epidemics", frozenset({"kermack_mckendrick_1927"})),
    ("endemic SIR threshold reproductive number review", frozenset({"hethcote_2000"})),
    ("Seoul commuter mobility population density spatial transmission", frozenset({"seoul_covid_spatial_2021"})),
    ("immunity debt rebound after COVID hygiene measures", frozenset({"immunity_debt_2022"})),
    ("NPI suppression of influenza surveillance under-ascertainment",
     frozenset({"npi_collateral_2022", "flu_suppression_2020"})),
    ("KDCA notifiable infectious disease sentinel surveillance Korea", frozenset({"kdca_guidelines_2024"})),
    ("Wilson confidence interval binomial small sample proportion", frozenset({"wilson_1927_ci"})),
    ("Kupiec unconditional coverage test prediction interval", frozenset({"kupiec_1995"})),
    ("Brier score decomposition reliability resolution", frozenset({"murphy_brier_1973"})),
    ("sMAPE MAPE forecast accuracy zero series denominator", frozenset({"hyndman_2018_sMAPE"})),
    ("cost-loss decision-theoretic forecast value skill score", frozenset({"bosse_2026_costloss"})),
    ("STROBE observational study reporting guideline checklist", frozenset({"strobe_2007"})),
    ("ecological fallacy aggregation bias spatial interpretation", frozenset({"ecological_fallacy_1994"})),
    ("LISA spatial clustering tuberculosis Seoul gu", frozenset({"tb_spatial_korea_2020"})),
)

#: A retriever maps (query, k) → ranked list of citation ids (or None if unavailable).
Retriever = Callable[[str, int], Optional[list[str]]]


def static_retriever(query: str, k: int) -> list[str]:
    """Token-overlap scorer (always available; backs the served tool fallback)."""
    from ..static_citations import score_citations

    return [c.id for _score, c in score_citations(query, k=k)]


def vector_retriever(query: str, k: int) -> Optional[list[str]]:
    """LanceDB + MiniLM semantic retriever, or None if index/libs are missing."""
    try:
        from . import semantic_search

        hits = semantic_search(query, k=k)
    except Exception:
        return None
    if not hits:
        return None
    out: list[str] = []
    for h in hits:
        cid = h.get("id") if isinstance(h, dict) else getattr(h, "id", None)
        if cid:
            out.append(cid)
    return out or None


def evaluate(retriever: Retriever,
             gold: Iterable[tuple[str, frozenset[str]]] = GOLD,
             k: int = 5) -> dict:
    """Compute recall@k / hit@k / MRR for ``retriever`` over ``gold``.

    Args:
        retriever: (query, k) → ranked citation ids, or None if unavailable.
        gold: (query, relevant-ids) pairs.
        k: cutoff.

    Returns:
        ``{k, n_queries, recall_at_k, hit_at_k, mrr, available}``. When the
        retriever returns None for every query, ``available`` is False and the
        rate metrics are 0.0 (caller should skip reporting it).
    """
    gold = list(gold)
    recalls: list[float] = []
    hits: list[float] = []
    rrs: list[float] = []
    n_available = 0
    for query, relevant in gold:
        ranked = retriever(query, k)
        if ranked is None:
            recalls.append(0.0); hits.append(0.0); rrs.append(0.0)
            continue
        n_available += 1
        ranked = ranked[:k]
        n_found = sum(1 for r in ranked if r in relevant)
        recalls.append(n_found / len(relevant))
        hits.append(1.0 if n_found else 0.0)
        rr = 0.0
        for rank, rid in enumerate(ranked, 1):
            if rid in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    n = len(gold)
    return {
        "k": k,
        "n_queries": n,
        "recall_at_k": sum(recalls) / n if n else 0.0,
        "hit_at_k": sum(hits) / n if n else 0.0,
        "mrr": sum(rrs) / n if n else 0.0,
        "available": n_available > 0,
    }


def main() -> int:
    print("=" * 64)
    print("  Literature-RAG retrieval evaluation (gold set: 19 query→citation)")
    print("=" * 64)
    for name, retr in (("static", static_retriever), ("vector", vector_retriever)):
        res = evaluate(retr)
        if not res["available"]:
            print(f"  {name:8s}: UNAVAILABLE (libs/index missing) — skipped")
            continue
        print(f"  {name:8s}: recall@{res['k']}={res['recall_at_k']:.3f}  "
              f"hit@{res['k']}={res['hit_at_k']:.3f}  MRR={res['mrr']:.3f}  "
              f"(n={res['n_queries']})")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
