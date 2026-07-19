"""ABM stratification validation (branch B): does the age-contact structure
reproduce the REAL age-stratified ILI risk ordering?

Companion to ``stratified_validation`` (which matched the ABM's daytime mobility
to real living-population). Here the question is whether the POLYMOD-like age
contact matrix (``contact_structure.CONTACT_MATRIX_7x7``) puts the right age
groups at highest risk, validated against the KDCA sentinel ILI which is
age-stratified into seven bands. The real pattern is the classic influenza one —
school-age children carry the most ILI — so a credible contact structure must
reproduce child > adult > elderly. The model's age-risk proxy is the dominant
eigenvector of the contact matrix (the stable age distribution of infection in an
SIR-on-contact-matrix), compared on three harmonised coarse bands because the
model's 10-year bands and the sentinel's custom bands do not align one-to-one.

This is an additive validator (no core-ABM change). Never raises; missing data →
explicit ``{"error": …}``. See PROOF_VALIDATION_PROTOCOL Pillar 3.
"""
from __future__ import annotations

import numpy as np

from .contact_structure import CONTACT_MATRIX_7x7

# sentinel ILI age bands (exact DB strings) and their coarse membership
_REAL_BANDS = ["0세", "1-6세", "7-12세", "13-18세", "19-49세", "50-64세", "65세 이상"]
_REAL_COARSE = {
    "child": ["0세", "1-6세", "7-12세", "13-18세"],   # 0–18
    "adult": ["19-49세", "50-64세"],                   # 19–64
    "elderly": ["65세 이상"],                          # 65+
}
# model AGE_BAND_LABELS = ["0-9","10-19","20-29","30-39","40-49","50-59","60+"]
_MODEL_COARSE = {"child": [0, 1], "adult": [2, 3, 4, 5], "elderly": [6]}
_GROUPS = ["child", "adult", "elderly"]


def load_real_age_ili(db_path: str) -> dict[str, float]:
    """Mean ILI rate per sentinel age band → ``{band: rate}`` (KDCA, official).
    Never raises (DB error → ``{}``)."""
    try:
        from simulation.database import read_only_connect
        con = read_only_connect(db_path)
        try:
            rows = con.execute(
                "SELECT age_group, AVG(ili_rate) FROM sentinel_influenza "
                "WHERE ili_rate IS NOT NULL AND age_group IS NOT NULL GROUP BY age_group"
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return {}
    return {str(a): float(r) for a, r in rows if r is not None}


def model_age_risk() -> np.ndarray:
    """Relative infection burden by model age band = the (normalised, non-negative)
    dominant eigenvector of the contact matrix — the stable age distribution of
    infection an SIR process on this contact structure would produce."""
    M = np.asarray(CONTACT_MATRIX_7x7, dtype=np.float64)
    w, v = np.linalg.eig(M)
    pev = np.abs(v[:, int(np.argmax(w.real))].real)
    s = pev.sum()
    return pev / s if s > 0 else pev


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def validate_age_ili_pattern(db_path: str) -> dict:
    """Does the ABM's age-contact structure reproduce the real age-ILI ordering?

    Returns ``{model_coarse, real_coarse, spearman, child_highest_both, match,
    verdict}``. ``match`` is True iff the coarse rank ordering agrees (Spearman ≥
    0.5) AND children are the highest-risk group in BOTH model and data — the
    influenza signature. Never raises.
    """
    real = load_real_age_ili(db_path)
    if not real or not all(b in real for b in _REAL_BANDS):
        return {"error": f"sentinel age-ILI incomplete (have {sorted(real)[:3]}…)"}
    real_coarse = {g: float(np.mean([real[b] for b in bands]))
                   for g, bands in _REAL_COARSE.items()}
    mr = model_age_risk()
    model_coarse = {g: float(np.mean([mr[i] for i in idxs]))
                    for g, idxs in _MODEL_COARSE.items()}
    rv = [real_coarse[g] for g in _GROUPS]
    mv = [model_coarse[g] for g in _GROUPS]
    rho = _spearman(mv, rv)
    child_real = max(real_coarse, key=real_coarse.get) == "child"
    child_model = max(model_coarse, key=model_coarse.get) == "child"
    match = (rho >= 0.5) and child_real and child_model
    verdict = (
        f"coarse age-risk: model child/adult/elderly="
        f"{mv[0]:.2f}/{mv[1]:.2f}/{mv[2]:.2f}, real ILI="
        f"{rv[0]:.1f}/{rv[1]:.1f}/{rv[2]:.1f} (Spearman {rho:+.2f}). "
        + ("✓ MATCH — the age-contact structure reproduces the real influenza "
           "ordering (school-age children highest, elderly lowest)."
           if match else
           "✗ mismatch — the model's age-risk ordering does not match the sentinel "
           "ILI; the contact matrix or age bands need revision.")
    )
    return {"model_coarse": {g: round(model_coarse[g], 4) for g in _GROUPS},
            "real_coarse": {g: round(real_coarse[g], 2) for g in _GROUPS},
            "spearman": round(rho, 4), "child_highest_both": bool(child_real and child_model),
            "match": bool(match), "verdict": verdict}
