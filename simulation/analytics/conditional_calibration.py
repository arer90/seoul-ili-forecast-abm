"""Conditional calibration metrics (audit Stage 3.2, Task #20).

Czado, Gneiting & Held (2009) nonrandomized PIT for count data +
marginal calibration diagram + conditional coverage by ILI tier +
Romano-Patterson-Candès (2019) CQR adjustment hooks.

Reference:
    - Czado C, Gneiting T, Held L (2009)
      "Predictive Model Assessment for Count Data"
      Biometrics 65(4):1254-1261. doi:10.1111/j.1541-0420.2009.01191.x
      "Our proposals include a nonrandomized version of the probability
       integral transform, marginal calibration diagrams, and proper
       scoring rules"
    - Gneiting T, Balabdaoui F, Raftery AE (2007)
      JRSSB 69(2):243-268. doi:10.1111/j.1467-9868.2007.00587.x
    - Romano Y, Patterson E, Candès EJ (2019)
      "Conformalized Quantile Regression"
      NeurIPS 32. arXiv:1905.03222

Audit context (TRIPOD+AI 2024):
    marginal PIT (compute_full_metrics 의 기존 pit_mean/pit_std/pit_ks_p) 는
    *평균적* 보정만 보장. ILI count-like seasonal data 의 high-incidence /
    low-incidence 구간별 conditional calibration 부재 → audit 비판 A6.

D-5 gray-box contract:
    - NaN-safe (절대 raise X)
    - O(n + n_bins) per metric
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from simulation.config_global import Z95  # SSOT (2026-05-28)

__all__ = [
    "nonrandomized_pit",
    "marginal_calibration_diff",
    "conditional_coverage_by_tier",
    "compute_conditional_calibration_block",
]


def nonrandomized_pit(
    y: np.ndarray,
    pred_mean: np.ndarray,
    pred_var: np.ndarray,
    *,
    family: str = "continuous",
) -> dict:
    """PIT for predictive calibration assessment.

    ⚠ Reference scope correction (external audit 2026-05-27):
        Czado, Gneiting & Held (2009) Biometrics 65(4):1254-1261, doi:10.1111/
        j.1541-0420.2009.01191.x — nonrandomized PIT 의 핵심 목적은 *count data
        의 discreteness 에서 발생하는 PIT jump* 를 randomization 없이 처리.
        ILI rate (continuous ratio scale) 에는 standard continuous PIT
        (Gneiting/Balabdaoui/Raftery 2007) 적용이 적절.

        본 함수의 family default = "continuous" (Gaussian standard PIT).
        Czado-style nonrandomized 는 family ∈ {"poisson", "nbinom"} 시만 적용.

    For continuous predictive distributions (default): PIT(y) = F_N(y; μ, σ).
        Reference: Gneiting/Balabdaoui/Raftery (2007) JRSS-B 69(2):243-268,
        doi:10.1111/j.1467-9868.2007.00587.x.
    For discrete (count) — Czado et al. (2009) nonrandomized:
        PIT_nonrand = (F(y-1) + F(y)) / 2 (midpoint of jump).
    Both methods produce uniform[0,1] under calibration.

    Args:
        y: observed (n,) — continuous (rate) or count.
        pred_mean: predicted mean (n,).
        pred_var: predicted variance (n,). Must be > 0.
        family: "continuous" (default, Gaussian PIT, Gneiting 2007) /
                "normal" (alias for "continuous") /
                "poisson" (count, Czado 2009 nonrandomized) /
                "nbinom" (count Negative Binomial, Czado 2009 nonrandomized).

    Returns:
        dict {
            "pit_nonrand_mean": float,    # expected 0.5 under calibration
            "pit_nonrand_std":  float,    # expected 1/sqrt(12) ≈ 0.289
            "pit_nonrand_ks_p": float,    # KS test vs uniform (p > 0.05 = OK)
            "n_valid": int,
            "family": str,
            "reference": str,             # Gneiting 2007 (continuous) or Czado 2009 (count)
        }
    """
    # Default reference: Gneiting (continuous); Czado only for count families
    if family in ("poisson", "nbinom"):
        _ref = "Czado, Gneiting & Held (2009) Biometrics 65:1254, doi:10.1111/j.1541-0420.2009.01191.x"
    else:
        _ref = "Gneiting, Balabdaoui, Raftery (2007) JRSS-B 69:243, doi:10.1111/j.1467-9868.2007.00587.x"

    out = {
        "pit_nonrand_mean": float("nan"),
        "pit_nonrand_std":  float("nan"),
        "pit_nonrand_ks_p": float("nan"),
        "n_valid": 0,
        "family": family,
        "reference": _ref,
    }
    if y is None or pred_mean is None or pred_var is None:
        return out
    yt = np.asarray(y, dtype=np.float64)
    pm = np.asarray(pred_mean, dtype=np.float64)
    pv = np.asarray(pred_var, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(pm) & np.isfinite(pv) & (pv > 0)
    if mask.sum() < 4:
        return out
    yt, pm, pv = yt[mask], pm[mask], pv[mask]
    n = len(yt)
    out["n_valid"] = n
    sd = np.sqrt(pv)

    try:
        from scipy.stats import norm, nbinom, poisson, kstest
    except ImportError:
        return out

    if family in ("continuous", "normal"):
        # PIT = F(y) for continuous Gaussian (Gneiting et al. 2007)
        pit_vals = norm.cdf(yt, loc=pm, scale=sd)
    elif family == "poisson":
        # Nonrandomized: (F(y-1) + F(y)) / 2
        lam = np.maximum(pm, 1e-6)
        y_int = np.round(yt).astype(int)
        f_y = poisson.cdf(y_int, lam)
        f_y_minus = poisson.cdf(np.maximum(y_int - 1, 0), lam)
        pit_vals = 0.5 * (f_y_minus + f_y)
    elif family == "nbinom":
        # NB parameterization: variance = mu + mu^2 / k → k = mu^2 / (var - mu) (if var > mu)
        mu = np.maximum(pm, 1e-6)
        k = np.where(pv > mu, mu * mu / np.maximum(pv - mu, 1e-6), 100.0)
        p = k / (k + mu)
        y_int = np.round(yt).astype(int)
        try:
            f_y = nbinom.cdf(y_int, k, p)
            f_y_minus = nbinom.cdf(np.maximum(y_int - 1, 0), k, p)
            pit_vals = 0.5 * (f_y_minus + f_y)
        except Exception:
            pit_vals = norm.cdf(yt, loc=pm, scale=sd)  # fallback
    else:
        pit_vals = norm.cdf(yt, loc=pm, scale=sd)

    out["pit_nonrand_mean"] = float(np.mean(pit_vals))
    out["pit_nonrand_std"] = float(np.std(pit_vals, ddof=1))
    try:
        out["pit_nonrand_ks_p"] = float(kstest(pit_vals, "uniform").pvalue)
    except Exception:
        pass
    return out


def marginal_calibration_diff(
    y: np.ndarray,
    pred_mean: np.ndarray,
    pred_var: np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,
) -> dict:
    """Marginal calibration: max |F_pred(z) - F_obs(z)| over support.

    Czado, Gneiting & Held (2009) — comparison of mean predictive CDF to
    empirical observed CDF.

    Returns:
        dict {
            "marginal_calib_max_diff": float (max KS-like distance),
            "marginal_calib_mean_diff": float (mean absolute diff),
            "n_grid": int,
        }
    """
    out = {
        "marginal_calib_max_diff": float("nan"),
        "marginal_calib_mean_diff": float("nan"),
        "n_grid": 0,
    }
    if y is None or pred_mean is None or pred_var is None:
        return out
    yt = np.asarray(y, dtype=np.float64)
    pm = np.asarray(pred_mean, dtype=np.float64)
    pv = np.asarray(pred_var, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(pm) & np.isfinite(pv) & (pv > 0)
    if mask.sum() < 4:
        return out
    yt, pm, pv = yt[mask], pm[mask], pv[mask]

    try:
        from scipy.stats import norm
    except ImportError:
        return out

    if grid is None:
        z_min = min(np.min(yt), np.min(pm - 3 * np.sqrt(pv)))
        z_max = max(np.max(yt), np.max(pm + 3 * np.sqrt(pv)))
        grid = np.linspace(z_min, z_max, 100)

    sd = np.sqrt(pv)
    # F_pred(z) = mean over n of F_i(z)  (mean predictive CDF)
    # F_obs(z) = empirical CDF of y
    f_pred = np.array([np.mean(norm.cdf(z, loc=pm, scale=sd)) for z in grid])
    f_obs = np.array([np.mean(yt <= z) for z in grid])
    diff = np.abs(f_pred - f_obs)
    out["marginal_calib_max_diff"] = float(np.max(diff))
    out["marginal_calib_mean_diff"] = float(np.mean(diff))
    out["n_grid"] = len(grid)
    return out


def conditional_coverage_by_tier(
    y: np.ndarray,
    pred_mean: np.ndarray,
    pred_sigma: float,
    *,
    n_tiers: int = 2,
    z_score: float = Z95,
) -> dict:
    """Conditional 95% PI coverage by ILI tier (high/low).

    Audit Stage 3.2 — ILI data 의 high-incidence vs low-incidence tier 별
    coverage 차이 보고. tier 1 = below median, tier 2 = above median (n_tiers=2).

    Args:
        y: observed (n,)
        pred_mean: (n,)
        pred_sigma: scalar (residual std).
        n_tiers: 2 (binary) or 3 (low/mid/high).
        z_score: 1.96 for 95% nominal.

    Returns:
        dict {
            "picp95_tier_<i>": float for i in 0..n_tiers-1,
            "picp95_tier_<i>_n": int,
            "tier_thresholds": list[float],
        }
    """
    out = {"tier_thresholds": []}
    if y is None or pred_mean is None:
        return out
    yt = np.asarray(y, dtype=np.float64)
    pm = np.asarray(pred_mean, dtype=np.float64)
    sigma = max(float(pred_sigma), 1e-6)
    mask = np.isfinite(yt) & np.isfinite(pm)
    if mask.sum() < 4:
        return out
    yt, pm = yt[mask], pm[mask]
    n = len(yt)

    lo = pm - z_score * sigma
    hi = pm + z_score * sigma
    in_pi = (yt >= lo) & (yt <= hi)

    # Tier by quantile of yt
    quantiles = np.linspace(0, 1, n_tiers + 1)[1:-1]
    thresholds = [float(np.quantile(yt, q)) for q in quantiles]
    out["tier_thresholds"] = thresholds

    for i in range(n_tiers):
        if i == 0:
            tier_mask = yt < (thresholds[0] if thresholds else np.inf)
        elif i == n_tiers - 1:
            tier_mask = yt >= thresholds[-1]
        else:
            tier_mask = (yt >= thresholds[i - 1]) & (yt < thresholds[i])
        n_tier = int(tier_mask.sum())
        out[f"picp95_tier_{i}"] = (float(in_pi[tier_mask].mean())
                                     if n_tier > 0 else float("nan"))
        out[f"picp95_tier_{i}_n"] = n_tier

    return out


def compute_conditional_calibration_block(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    sigma: float = 1.0,
    family: str = "continuous",  # audit 2026-05-27: ILI rate = continuous, default fix
) -> dict:
    """Single-call wrapper for audit Stage 3.2 — 3 calibration metric block.

    Used by compute_full_metrics (metric_eval.py) to add audit-grade
    conditional calibration without disrupting existing 54-metric output.

    Returns merged dict (keys prefixed for clarity):
        - pit_nonrand_mean / pit_nonrand_std / pit_nonrand_ks_p
        - marginal_calib_max_diff / marginal_calib_mean_diff
        - picp95_tier_0 / picp95_tier_1 (low/high ILI tier)
    """
    sigma_v = max(float(sigma), 1e-6)
    pred_var = np.full_like(np.asarray(y_pred, dtype=np.float64), sigma_v ** 2)

    pit = nonrandomized_pit(y_test, y_pred, pred_var, family=family)
    marg = marginal_calibration_diff(y_test, y_pred, pred_var)
    cond = conditional_coverage_by_tier(y_test, y_pred, sigma_v, n_tiers=2)

    out = {
        "pit_nonrand_mean": pit["pit_nonrand_mean"],
        "pit_nonrand_std":  pit["pit_nonrand_std"],
        "pit_nonrand_ks_p": pit["pit_nonrand_ks_p"],
        "marginal_calib_max_diff":  marg["marginal_calib_max_diff"],
        "marginal_calib_mean_diff": marg["marginal_calib_mean_diff"],
        "picp95_low_tier":  cond.get("picp95_tier_0",  float("nan")),
        "picp95_high_tier": cond.get("picp95_tier_1",  float("nan")),
        "_conditional_calib_meta": {
            "pit_family": family,
            "tier_thresholds": cond.get("tier_thresholds", []),
            "reference": "Czado, Gneiting & Held (2009) doi:10.1111/j.1541-0420.2009.01191.x",
        },
    }
    return out
