"""Multiple testing correction (audit Stage 1.3, Task #15).

BH-FDR / Holm-Bonferroni / Bonferroni family-wise correction.

Audit context (TRIPOD+AI 2024 + Benjamini-Hochberg 1995):
    53 model × 22 region × 77 horizon × 54 metric = 4,848,228 comparisons.
    FWER 폭증 위험. R6 (pairwise DM) + R8 (composite scoring) +
    R10 (per_model_eval — 이미 일부 BH-FDR 적용) 의 p-value
    family 일괄 보정 필요. (R10 champion-designation g175 binding 제거 2026-06-05 — champion=best-WIS.)

Reference:
    - Benjamini Y & Hochberg Y (1995) "Controlling the false discovery rate:
      a practical and powerful approach to multiple testing"
      JRSSB 57(1):289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
    - Holm S (1979) "A simple sequentially rejective multiple test procedure"
      Scand J Stat 6(2):65-70. doi: (no DOI, classic).

Family definitions (audit Stage 1.3 권장):
    Family A: 4 criteria (R²/MAPE/WIS/PICP95) — correlated → effective dim
    Family B: per-region (22 region per metric)
    Family C: per-horizon (77 horizon)
    Family D: per-model (53 model)

Each family applies BH-FDR(q=0.05) independently.

D-5 gray-box contract:
    - statsmodels.stats.multitest.multipletests wrap (NaN-safe extension)
    - Returns dict[str, bool] (test_name → pass_corrected)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "apply_bh_fdr",
    "apply_holm",
    "apply_bonferroni",
    "effective_number_of_tests_pca",
    "adjust_pvalues",
]


def apply_bh_fdr(
    p_values: dict[str, float],
    q: float = 0.05,
) -> dict:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: dict[test_name, p_value]. NaN allowed (skip).
        q: FDR target (default 0.05).

    Returns:
        dict {
            "reject": dict[str, bool],       # test_name → pass (reject H0 after BH)
            "pvals_corrected": dict[str, float],
            "method": "fdr_bh",
            "q": float,
            "n_tests": int,
            "n_rejected": int,
            "reference": "Benjamini & Hochberg (1995) doi:10.1111/j.2517-6161.1995.tb02031.x",
        }

    Performance: O(N log N) sort.
    Side effects: 없음 (pure function).
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return _fallback_adjust(p_values, q, "fdr_bh", import_error=True)

    return _wrap_multipletests(p_values, q, method="fdr_bh",
                                reference="Benjamini & Hochberg (1995) doi:10.1111/j.2517-6161.1995.tb02031.x")


def apply_holm(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict:
    """Holm-Bonferroni step-down correction (controls FWER)."""
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return _fallback_adjust(p_values, alpha, "holm", import_error=True)

    return _wrap_multipletests(p_values, alpha, method="holm",
                                reference="Holm (1979) Scand J Stat 6(2):65-70")


def apply_bonferroni(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict:
    """Bonferroni correction (most conservative, controls FWER)."""
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return _fallback_adjust(p_values, alpha, "bonferroni", import_error=True)

    return _wrap_multipletests(p_values, alpha, method="bonferroni",
                                reference="Bonferroni (1936) — classic")


def adjust_pvalues(
    p_values: list[float] | np.ndarray,
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> np.ndarray:
    """Backward-compat wrapper (simulation/analytics/epidemiological.py 등에서 사용).

    Args:
        p_values: list of p-values (NaN allowed).
        method: 'fdr_bh' / 'holm' / 'bonferroni'.
        alpha: 0.05 default.

    Returns:
        np.ndarray of adjusted p-values (NaN preserved).
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return np.asarray(p_values, dtype=np.float64)  # passthrough

    arr = np.asarray(p_values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return arr

    finite_pvals = arr[finite_mask]
    reject, pvals_corr, _, _ = multipletests(finite_pvals, alpha=alpha, method=method)

    out = np.full_like(arr, np.nan)
    out[finite_mask] = pvals_corr
    return out


def effective_number_of_tests_pca(
    correlation_matrix: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """Effective number of independent tests via PCA (Galwey 2009).

    Used for audit Stage 1.2 G3 — 4 criteria (R²/MAPE/WIS/PICP95) 의
    correlation 으로 effective dimension < 4 일 수 있음.

    Args:
        correlation_matrix: (n, n) symmetric correlation matrix.
        threshold: variance explained threshold (default 0.95).

    Returns:
        int — number of PC needed to explain ≥ threshold variance.

    Reference:
        Galwey NW (2009) "A new measure of the effective number of tests"
        Genet Epidemiol 33(7):559-568. doi:10.1002/gepi.20408
    """
    if correlation_matrix is None or len(correlation_matrix) == 0:
        return 0
    try:
        eigs = np.linalg.eigvalsh(correlation_matrix)
        eigs = eigs[eigs > 1e-10]  # drop ~0
        eigs = np.sort(eigs)[::-1]  # descending
        total = eigs.sum()
        if total <= 0:
            return 0
        cumvar = np.cumsum(eigs) / total
        # 첫 K PC 가 threshold 충족
        for k in range(len(cumvar)):
            if cumvar[k] >= threshold:
                return k + 1
        return len(eigs)
    except Exception:
        return correlation_matrix.shape[0]  # fallback: n_tests


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────


def _wrap_multipletests(
    p_values: dict[str, float],
    alpha: float,
    *,
    method: str,
    reference: str,
) -> dict:
    """statsmodels.multipletests wrap with NaN handling."""
    from statsmodels.stats.multitest import multipletests

    if not p_values:
        return _empty_result(alpha, method, reference)

    names = list(p_values.keys())
    pvals = np.array([p_values[n] for n in names], dtype=np.float64)

    finite_mask = np.isfinite(pvals)
    if not finite_mask.any():
        return {
            "reject": {n: False for n in names},
            "pvals_corrected": {n: float("nan") for n in names},
            "method": method, "q": alpha,
            "n_tests": 0, "n_rejected": 0,
            "reference": reference,
        }

    # only finite p-values participate
    finite_names = [names[i] for i in range(len(names)) if finite_mask[i]]
    finite_pvals = pvals[finite_mask]
    reject, pvals_corr, _, _ = multipletests(finite_pvals, alpha=alpha, method=method)

    out_reject = {n: False for n in names}
    out_corr = {n: float("nan") for n in names}
    for i, n in enumerate(finite_names):
        out_reject[n] = bool(reject[i])
        out_corr[n] = float(pvals_corr[i])

    return {
        "reject": out_reject,
        "pvals_corrected": out_corr,
        "method": method,
        "q": alpha,
        "n_tests": int(len(finite_pvals)),
        "n_rejected": int(reject.sum()),
        "reference": reference,
    }


def _fallback_adjust(p_values: dict, alpha: float, method: str, *, import_error: bool) -> dict:
    """statsmodels 없을 때 passthrough (warning)."""
    import warnings
    if import_error:
        warnings.warn(
            "statsmodels not available — multiple testing correction skipped. "
            "Install: pip install statsmodels.",
            RuntimeWarning, stacklevel=3,
        )
    return {
        "reject": {n: (np.isfinite(p) and p < alpha) for n, p in p_values.items()},
        "pvals_corrected": dict(p_values),
        "method": f"{method}_unavailable_passthrough",
        "q": alpha,
        "n_tests": len(p_values),
        "n_rejected": sum(1 for p in p_values.values() if np.isfinite(p) and p < alpha),
        "reference": "statsmodels import failed",
    }


def _empty_result(alpha: float, method: str, reference: str) -> dict:
    return {
        "reject": {}, "pvals_corrected": {},
        "method": method, "q": alpha,
        "n_tests": 0, "n_rejected": 0,
        "reference": reference,
    }
