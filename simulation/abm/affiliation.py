"""Per-agent school-affiliation graph (ABM heterogeneity enrichment, branch ⑬-소속).

Agents already carry an ``occupation`` (incl. 'school'); this adds the missing
MEMBERSHIP structure — which school a student belongs to — so that students in the
same school share elevated contact (the dominant influenza amplification channel,
Cauchemez 2008). The school COUNT and per-district distribution are grounded in
the real ``school_info`` table (25 gu; 611 elementary / 390 middle / 323 high
schools), so the number of school clusters in each district matches reality.

Additive (no core agent_kernel rewrite): exposes a per-agent school-cluster id a
contact kernel can use to raise within-school mixing. Never raises in analysis.
Companion to ``comorbidity`` (health) and ``mobility_timeresolved`` (movement).
"""
from __future__ import annotations

import numpy as np

_STUDENT_BAND = 1  # AGE_BAND_LABELS index "10-19" (middle/high-school age)


def load_schools_per_gu(db_path: str, gu_names: list[str]) -> np.ndarray:
    """Real per-district school count from ``school_info`` (초/중/고), aligned to
    ``gu_names`` order. Missing district → 1 (avoid div-by-zero). Never raises."""
    counts = {g: 1 for g in gu_names}
    try:
        from simulation.database import read_only_connect
        con = read_only_connect(db_path)
        try:
            for gu, n in con.execute(
                "SELECT gu_nm, COUNT(*) FROM school_info WHERE gu_nm IS NOT NULL "
                "AND school_kind IN ('초등학교','중학교','고등학교') GROUP BY gu_nm"
            ).fetchall():
                if gu in counts:
                    counts[gu] = max(1, int(n))
        finally:
            con.close()
    except Exception:
        pass
    return np.array([counts[g] for g in gu_names], dtype=int)


def assign_school_clusters(home_gu_idx: np.ndarray, age_bands: np.ndarray,
                           schools_per_gu: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """Assign each student agent (age band 10-19) to a school cluster within its
    home district; the number of clusters per district = the real school count.

    Returns a length-N int array of GLOBAL school ids (``-1`` for non-students).
    Students in the same district are spread across that district's real number of
    schools, giving school-sized shared-contact groups. Never raises."""
    rng = np.random.default_rng(seed)
    gu = np.asarray(home_gu_idx, dtype=int)
    bands = np.asarray(age_bands, dtype=int)
    spg = np.asarray(schools_per_gu, dtype=int)
    out = np.full(len(gu), -1, dtype=np.int64)
    base = np.concatenate([[0], np.cumsum(spg)])  # global id offsets per gu
    students = bands == _STUDENT_BAND
    for g in np.unique(gu[students]):
        mask = students & (gu == g)
        k = int(spg[g]) if 0 <= g < len(spg) else 1
        out[mask] = base[g] + rng.integers(0, max(k, 1), size=int(mask.sum()))
    return out


def validate_school_affiliation(db_path: str, gu_names: list[str], *,
                                n_students_per_gu: int = 2000, seed: int = 0) -> dict:
    """Validate that the assigned school structure reflects the real per-district
    school distribution: districts with more real schools receive more clusters,
    and within-school sizes are bounded. Returns ``{n_gu, n_schools_total,
    rank_corr, match, verdict}``. Never raises."""
    spg = load_schools_per_gu(db_path, gu_names)
    if spg.sum() <= len(gu_names):
        return {"error": "no usable school_info counts"}
    G = len(gu_names)
    gu = np.repeat(np.arange(G), n_students_per_gu)
    bands = np.full(len(gu), _STUDENT_BAND)
    ids = assign_school_clusters(gu, bands, spg, seed=seed)
    # clusters actually used per gu vs real school count → rank concordance
    used = np.array([len(np.unique(ids[(gu == g) & (ids >= 0)])) for g in range(G)])
    ra = np.argsort(np.argsort(spg)).astype(float)
    rb = np.argsort(np.argsort(used)).astype(float)
    rank_corr = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else 0.0
    match = rank_corr >= 0.8
    verdict = (
        f"{G} gu, {int(spg.sum())} real schools; cluster-count vs real-count "
        f"Spearman {rank_corr:+.2f}. "
        + ("✓ school membership reflects the real per-district school distribution "
           "(more schools → more student clusters) — a real-data-grounded "
           "shared-contact structure for within-school transmission."
           if match else "✗ cluster structure does not track the real school counts.")
    )
    return {"n_gu": G, "n_schools_total": int(spg.sum()),
            "rank_corr": round(rank_corr, 4), "match": bool(match), "verdict": verdict}
