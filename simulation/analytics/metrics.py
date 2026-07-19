"""
Reusable metrics and comparison functions for epidemic forecasting.

Point-accuracy:
  - _calc_metrics: R², RMSE, MSE, MAE, MAPE, sMAPE

Probabilistic:
  - crps_gaussian: CRPS under Gaussian predictive (diagnostics.weighted_interval_score
                   covers WIS; combine via analytics.__init__).
  - pinball_loss: single-quantile loss (building block for WIS / quantile
                  regression evaluation).
  - pi_coverage: empirical coverage vs nominal for a (lower, upper) PI.
  - pi_calibration_table: coverage at a vector of nominal levels.

Forecast comparison:
  - diebold_mariano: DM test (Harvey et al. 1997 correction).
  - mcnemar_test: paired-binary direction-accuracy comparison.

Epi / clinical:
  - peak_week_error / peak_intensity_error: seasonal-peak fidelity.
  - direction_accuracy: % of correct up/down moves.
  - brier_score / brier_skill_score: probabilistic classification vs
                   climatological baseline.
  - binary_clinical_rates: sensitivity/specificity/PPV/NPV at a threshold.
  - decision_curve: net benefit across decision thresholds
                   (Vickers & Elkin 2006).

Uncertainty / multiple testing:
  - bootstrap_ci: percentile + BCa (DiCiccio & Efron 1996).
  - adjust_pvalues: Bonferroni / Holm / Benjamini-Hochberg FDR
                   (backs simulation.models runner per-model DM tests).
"""

import numpy as np
from scipy import stats
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any


def diebold_mariano(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for forecast comparison (Harvey et al. 1997 correction).

    Tests H0: E[d_t] = 0, where d_t = e1_t² - e2_t²

    Args:
        actual: Ground truth values
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        h: Forecast horizon for autocorrelation adjustment

    Returns:
        dm_stat: DM test statistic (corrected)
        p_value: Two-tailed p-value
    """
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1 ** 2 - e2 ** 2
    nn = len(d)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_sum += np.cov(d[:-k], d[k:], ddof=1)[0, 1]
    var_d = (gamma0 + 2 * gamma_sum) / nn
    if var_d <= 0:
        return 0.0, 1.0
    dm = d_bar / np.sqrt(var_d)
    correction = np.sqrt((nn + 1 - 2 * h + h * (h - 1) / nn) / nn)
    dm_corrected = dm * correction
    p_value = 2 * (1 - stats.t.cdf(abs(dm_corrected), df=nn - 1))
    return float(dm_corrected), float(p_value)


def crps_gaussian(
    actual: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray
) -> np.ndarray:
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.

    CRPS = E_F[|X - y|] for F = N(μ, σ²)
    Analytical formula via Gneiting & Raftery (2007) JASA 102(477):359-378
    Eq. (5); see also Matheson & Winkler (1976) for the original form.

    Args:
        actual: Observed values
        pred_mean: Mean of predictive distribution
        pred_std: Standard deviation of predictive distribution

    Returns:
        crps: CRPS value at each time point
    """
    z = (actual - pred_mean) / np.maximum(pred_std, 1e-8)
    return pred_std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def _calc_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Calculate standard forecast accuracy metrics on a subset.

    Returns:
        Dict with r2, mse, rmse, mae, mape (None if no positive actuals),
        smape, n, or None if the subset has <3 observations.
    """
    a, pr = actual[mask], predicted[mask]
    if len(a) < 3:
        return None

    ss_res = np.sum((a - pr) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mse = float(np.mean((a - pr) ** 2))
    rmse = float(np.sqrt(mse))

    m = a > 0
    mape = float(np.mean(np.abs((a[m] - pr[m]) / a[m])) * 100) if m.any() else None
    mae = float(np.mean(np.abs(a - pr)))

    denom = np.abs(a) + np.abs(pr)
    keep = denom > 0
    smape = (
        float(np.mean(2.0 * np.abs(a[keep] - pr[keep]) / denom[keep]) * 100)
        if keep.any() else None
    )

    return {
        "r2": round(r2, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2) if mape is not None else None,
        "smape": round(smape, 2) if smape is not None else None,
        "mae": round(mae, 2),
        "n": int(mask.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════
# Probabilistic metrics
# ══════════════════════════════════════════════════════════════════════════
def pinball_loss(
    y_true: np.ndarray,
    y_quantile: np.ndarray,
    quantile: float,
) -> float:
    """Quantile / pinball loss for a single predictive quantile.

    L_q(y, q̂) = (q - 1{y<q̂}) · (y - q̂), averaged. 0 is perfect; lower better.
    Building block for quantile regression evaluation and for constructing
    the Interval Score from (lower, upper) = (q_{α/2}, q_{1-α/2}).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_quantile = np.asarray(y_quantile, dtype=float)
    diff = y_true - y_quantile
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def pi_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal: float,
) -> Dict[str, float]:
    """Empirical coverage and mean width of a prediction interval.

    Returns the raw coverage, the deviation from nominal, and the mean
    width — the triple used by conformal PI audits (phase10_intervals)
    and by reviewer-facing calibration tables.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    mean_cov = float(np.mean(covered)) if covered.size else float("nan")
    mean_width = float(np.mean(upper - lower)) if covered.size else float("nan")
    return {
        "nominal": float(nominal),
        "empirical": mean_cov,
        "deviation": mean_cov - float(nominal),
        "mean_width": mean_width,
        "n": int(covered.size),
    }


def pi_calibration_table(
    y_true: np.ndarray,
    lower_by_level: Dict[float, np.ndarray],
    upper_by_level: Dict[float, np.ndarray],
) -> List[Dict[str, float]]:
    """Coverage table across a set of nominal levels.

    `lower_by_level` / `upper_by_level` map nominal coverage (e.g. 0.9) to
    the arrays of lower / upper quantile forecasts. Returns a list ordered
    by ascending level, suitable for a calibration plot.
    """
    levels = sorted(set(lower_by_level) & set(upper_by_level))
    rows: List[Dict[str, float]] = []
    for lvl in levels:
        rows.append(pi_coverage(
            y_true,
            lower_by_level[lvl], upper_by_level[lvl],
            nominal=lvl,
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Forecast comparison — paired binary direction accuracy
# ══════════════════════════════════════════════════════════════════════════
def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    use_exact_cutoff: int = 25,
) -> Tuple[float, float]:
    """McNemar test on paired correct/incorrect classifications.

    `correct_a`, `correct_b` are 0/1 arrays (same length) telling whether
    model A / B got each time step right (e.g. direction of change).
    Uses the exact binomial test when the discordant cell count is small
    (≤ `use_exact_cutoff`), else the chi² continuity-corrected statistic.

    Returns (statistic, two-sided p-value). `statistic` is the number of
    b-wins (exact branch) or the chi² value (asymptotic branch).
    """
    a = np.asarray(correct_a).astype(int)
    b = np.asarray(correct_b).astype(int)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("correct_a and correct_b must be 1-D and same length")

    # Discordant counts
    n01 = int(np.sum((a == 0) & (b == 1)))  # A wrong, B right
    n10 = int(np.sum((a == 1) & (b == 0)))  # A right, B wrong
    n_disc = n01 + n10
    if n_disc == 0:
        return 0.0, 1.0

    if n_disc <= use_exact_cutoff:
        # Exact binomial two-sided
        k = min(n01, n10)
        p = 2.0 * stats.binom.cdf(k, n_disc, 0.5)
        return float(min(n01, n10)), float(min(p, 1.0))

    # Asymptotic chi² with Edwards' continuity correction
    stat = (abs(n01 - n10) - 1.0) ** 2 / n_disc
    p = float(1.0 - stats.chi2.cdf(stat, df=1))
    return float(stat), p


# ══════════════════════════════════════════════════════════════════════════
# Epidemic-curve metrics (seasonal-peak fidelity + direction)
# ══════════════════════════════════════════════════════════════════════════
def peak_week_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    tolerance_weeks: int = 1,
) -> Dict[str, float]:
    """Absolute error in the index-of-peak, plus a hit flag.

    Epi-surveillance standard (CDC FluSight, ECDC): the primary target is
    not the weekly RMSE but whether the forecast peaks at the right week.
    `tolerance_weeks` decides what counts as a "hit" (default ±1).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0 or predicted.size == 0:
        return {"abs_weeks": float("nan"), "hit": 0.0, "peak_actual": -1.0,
                "peak_pred": -1.0}

    peak_a = int(np.argmax(actual))
    peak_p = int(np.argmax(predicted[: len(actual)]))
    abs_err = abs(peak_a - peak_p)
    return {
        "abs_weeks": float(abs_err),
        "hit": float(abs_err <= tolerance_weeks),
        "peak_actual": float(peak_a),
        "peak_pred": float(peak_p),
    }


def peak_intensity_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    log_scale: bool = False,
) -> Dict[str, float]:
    """Error in peak magnitude (ILI rate at peak week).

    `log_scale=True` returns the log-ratio magnitude, which is what
    CDC/ECDC ensembles tend to report to avoid scale dominance by wide
    epidemics.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0 or predicted.size == 0:
        return {"abs_err": float("nan"), "rel_err": float("nan")}

    peak_a_val = float(np.max(actual))
    peak_p_val = float(np.max(predicted[: len(actual)]))
    abs_err = abs(peak_a_val - peak_p_val)
    rel_err = abs_err / max(peak_a_val, 1e-8)
    out = {
        "abs_err": abs_err,
        "rel_err": rel_err,
        "peak_actual_value": peak_a_val,
        "peak_pred_value": peak_p_val,
    }
    if log_scale:
        out["log_ratio"] = float(
            np.log(max(peak_p_val, 1e-8)) - np.log(max(peak_a_val, 1e-8))
        )
    return out


def direction_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """Fraction of weeks where the sign of Δ is predicted correctly.

    Returns both the raw accuracy and a per-step 0/1 indicator array
    (for McNemar on two models; use `mcnemar_test(indicator_a,
    indicator_b)`).
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    n = min(len(a), len(p))
    if n < 2:
        return {"accuracy": float("nan"), "n_moves": 0.0}

    da = np.sign(np.diff(a[:n]))
    dp = np.sign(np.diff(p[:n]))
    correct = (da == dp).astype(float)
    return {
        "accuracy": float(correct.mean()),
        "n_moves": float(correct.size),
        "correct": correct,   # consumable by mcnemar_test
    }


# ══════════════════════════════════════════════════════════════════════════
# Brier score + skill score vs seasonal climatology
# ══════════════════════════════════════════════════════════════════════════
def brier_score(
    event_true: np.ndarray,
    event_prob: np.ndarray,
) -> float:
    """Mean squared error of a probabilistic binary forecast."""
    e = np.asarray(event_true, dtype=float)
    p = np.asarray(event_prob, dtype=float)
    return float(np.mean((p - e) ** 2))


def brier_skill_score(
    event_true: np.ndarray,
    event_prob: np.ndarray,
    event_prob_reference: np.ndarray,
) -> float:
    """BSS = 1 - BS(forecast) / BS(reference).

    Positive → forecast beats reference (usually seasonal climatology);
    0 → equal; negative → worse than baseline. For ILI rate, the
    reference is typically the climatological week-of-year mean prob of
    the "above CDC threshold" event (cf. Ray et al. 2020, FluSight).
    """
    bs_f = brier_score(event_true, event_prob)
    bs_ref = brier_score(event_true, event_prob_reference)
    if bs_ref <= 0:
        return float("nan")
    return float(1.0 - bs_f / bs_ref)


def brier_decomposition(
    event_true: np.ndarray,
    event_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Murphy (1973) decomposition of the Brier score.

    Decomposes ``BS = REL − RES + UNC`` for a probabilistic binary
    forecast, where:

    - ``REL`` (Reliability) = Σ_k (n_k/N) · (p̄_k − f̄_k)²
      The mean squared bias between binned forecast probability and the
      observed event frequency in that bin. **Lower is better** — measures
      calibration. Zero ⇒ perfect calibration.

    - ``RES`` (Resolution) = Σ_k (n_k/N) · (f̄_k − f̄)²
      The variance of bin-conditional event frequencies around the
      marginal climatology. **Higher is better** — measures discrimination.

    - ``UNC`` (Uncertainty) = f̄ · (1 − f̄)
      Climatological irreducible uncertainty. Depends only on base rate.

    Identity (modulo binning noise): ``BS ≈ REL − RES + UNC``.

    Why include this alongside ``brier_score`` / ``brier_skill_score``:
    a single Brier value cannot tell you WHY a model performs well or
    poorly. The decomposition separates calibration errors (REL) from
    discrimination skill (RES) so reviewers can locate the failure mode
    (CDC FluSight + EU Hub report this routinely).

    Args:
        event_true: 0/1 binary outcomes, shape (N,).
        event_prob: forecast probabilities in [0, 1], shape (N,).
        n_bins:     number of equal-width forecast probability bins
                    (default 10; matches the standard reliability diagram).

    Returns:
        Dict with keys ``brier``, ``reliability``, ``resolution``,
        ``uncertainty``, ``identity_residual`` (BS − (REL − RES + UNC),
        ≈0 if binning fine enough), ``n``, ``n_bins_used``.
    """
    e = np.asarray(event_true, dtype=float).ravel()
    p = np.asarray(event_prob, dtype=float).ravel()
    n = len(e)
    if n != len(p) or n == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan"),
                "identity_residual": float("nan"), "n": n, "n_bins_used": 0}

    # Sanitize
    p = np.clip(p, 0.0, 1.0)
    e = np.clip(e, 0.0, 1.0)

    bs = float(np.mean((p - e) ** 2))
    f_bar = float(np.mean(e))  # base rate
    unc = float(f_bar * (1.0 - f_bar))

    # Equal-width bins on [0, 1]. Empty bins contribute 0 (skipped).
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-inclusive last bin; use np.digitize with right=False then clamp.
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    rel = 0.0
    res = 0.0
    bins_used = 0
    for k in range(n_bins):
        mask = bin_idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        p_bar_k = float(p[mask].mean())
        f_bar_k = float(e[mask].mean())
        w_k = n_k / n
        rel += w_k * (p_bar_k - f_bar_k) ** 2
        res += w_k * (f_bar_k - f_bar) ** 2
        bins_used += 1

    residual = bs - (rel - res + unc)
    return {
        "brier":             round(bs, 6),
        "reliability":       round(rel, 6),
        "resolution":        round(res, 6),
        "uncertainty":       round(unc, 6),
        "identity_residual": round(residual, 6),
        "n":                 n,
        "n_bins_used":       bins_used,
    }


# ══════════════════════════════════════════════════════════════════════════
# Sprint S5 (2026-05-26) — 11 metric implementations referenced by
# `simulation/scripts/generate_4method_report.py` but never coded.
# Adds: pearson_r, spearman_r, calibration_slope, calibration_intercept,
# hosmer_lemeshow (chi² + p_value), c_index (Harrell concordance),
# s_index (skill index aggregate), epi_peak_mae, epi_season_total_mae.
# ══════════════════════════════════════════════════════════════════════════

def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson product-moment correlation coefficient.

    Returns NaN if either array has zero variance.

    Args:
        y_true: observed values, shape (n,).
        y_pred: predicted values, shape (n,).

    Returns:
        float in [-1, 1], or NaN on degenerate input.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if mask.sum() < 2:
        return float("nan")
    y_t = y_t[mask]; y_p = y_p[mask]
    if y_t.std(ddof=0) == 0 or y_p.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(y_t, y_p)[0, 1])


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient (Pearson r on ranks).

    Robust to monotone transformations of either variable; ideal for
    forecast-skill ranking comparisons.

    Args:
        y_true: observed values, shape (n,).
        y_pred: predicted values, shape (n,).

    Returns:
        float in [-1, 1], or NaN on degenerate input.
    """
    from scipy.stats import spearmanr
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if mask.sum() < 2:
        return float("nan")
    try:
        rho, _p = spearmanr(y_t[mask], y_p[mask])
        return float(rho) if np.isfinite(rho) else float("nan")
    except Exception:
        return float("nan")


def calibration_slope_intercept(
    event_true: np.ndarray,
    event_prob: np.ndarray,
) -> Dict[str, float]:
    """Cox (1958) calibration slope + intercept via logistic regression.

    Fits ``logit(p_obs) = α + β · logit(p_pred)``. Perfect calibration
    yields α=0, β=1. Slope <1 → over-confidence; >1 → under-confidence.

    Args:
        event_true: 0/1 binary outcomes, shape (n,).
        event_prob: forecast probabilities in (0, 1).

    Returns:
        Dict with ``calibration_slope`` and ``calibration_intercept``.
    """
    from sklearn.linear_model import LogisticRegression
    e = np.asarray(event_true, dtype=float).ravel()
    p = np.clip(np.asarray(event_prob, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    if e.size != p.size or e.size < 4:
        return {"calibration_slope": float("nan"),
                "calibration_intercept": float("nan")}
    # Avoid degenerate cases (all 0 / all 1)
    if e.sum() == 0 or e.sum() == e.size:
        return {"calibration_slope": float("nan"),
                "calibration_intercept": float("nan")}
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        lr.fit(logit_p, e.astype(int))
        return {
            "calibration_slope":     float(lr.coef_[0, 0]),
            "calibration_intercept": float(lr.intercept_[0]),
        }
    except Exception:
        return {"calibration_slope": float("nan"),
                "calibration_intercept": float("nan")}


def hosmer_lemeshow(
    event_true: np.ndarray,
    event_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Hosmer-Lemeshow (1980) goodness-of-fit chi² test.

    Bins predicted probabilities into ``n_bins`` deciles; tests whether
    observed event frequencies match expected. df = ``n_bins − 2``.
    Low p (<0.05) ⇒ poor calibration. Standard for clinical prediction
    models (logistic regression diagnostics).

    Args:
        event_true: 0/1 binary outcomes, shape (n,).
        event_prob: forecast probabilities in [0, 1].
        n_bins:     number of probability bins (default 10).

    Returns:
        Dict with ``hl_chi2``, ``hl_p_value``, ``hl_bins_used``.
    """
    from scipy.stats import chi2
    e = np.asarray(event_true, dtype=float).ravel()
    p = np.clip(np.asarray(event_prob, dtype=float).ravel(), 0.0, 1.0)
    n = e.size
    if n != p.size or n < n_bins:
        return {"hl_chi2": float("nan"), "hl_p_value": float("nan"),
                "hl_bins_used": 0}

    # Quantile bins (deciles).
    try:
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] = -np.inf; edges[-1] = np.inf
    except Exception:
        return {"hl_chi2": float("nan"), "hl_p_value": float("nan"),
                "hl_bins_used": 0}

    chi2_stat = 0.0
    bins_used = 0
    for k in range(n_bins):
        mask = (p >= edges[k]) & (p < edges[k + 1])
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        o_k = float(e[mask].sum())
        e_k = float(p[mask].sum())  # expected count under model
        denom = e_k * (1.0 - e_k / n_k) if 0 < e_k < n_k else 0.0
        if denom <= 0:
            continue
        chi2_stat += (o_k - e_k) ** 2 / denom
        bins_used += 1

    df = max(1, bins_used - 2)
    p_value = float(1.0 - chi2.cdf(chi2_stat, df=df))
    return {
        "hl_chi2":      round(chi2_stat, 6),
        "hl_p_value":   round(p_value, 6),
        "hl_bins_used": bins_used,
    }


def c_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harrell's concordance index (1982) — rank-based discrimination.

    For continuous outcomes, computes the fraction of comparable pairs
    where higher predicted value goes with higher observed value::

        c = (concordant + 0.5·ties) / total_comparable_pairs

    Range [0, 1]; 0.5 = random; 1.0 = perfect rank discrimination.
    Equivalent to AUC for binary outcomes. For the time-series ILI
    forecast, this measures whether the model preserves the *ordering*
    of weeks by severity (separable from level/scale calibration).

    Args:
        y_true: observed values, shape (n,).
        y_pred: predicted values, shape (n,).

    Returns:
        float in [0, 1], or NaN if no comparable pairs exist.

    Performance: O(n²) — fine for ILI test slabs (n ≤ 70).
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if mask.sum() < 2:
        return float("nan")
    y_t = y_t[mask]; y_p = y_p[mask]
    n = y_t.size
    concordant = 0
    discordant = 0
    ties = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_t[i] == y_t[j]:
                continue  # not comparable on outcome
            pairs += 1
            if (y_t[i] > y_t[j] and y_p[i] > y_p[j]) or \
               (y_t[i] < y_t[j] and y_p[i] < y_p[j]):
                concordant += 1
            elif y_p[i] == y_p[j]:
                ties += 1
            else:
                discordant += 1
    if pairs == 0:
        return float("nan")
    return float((concordant + 0.5 * ties) / pairs)


def s_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    wis_model: Optional[float] = None,
    wis_climatology: Optional[float] = None,
    sensitivity: Optional[float] = None,
    lead_time_weeks: Optional[float] = None,
) -> float:
    """Skill / Surveillance index — aggregate forecast-utility score.

    Combines two clinically-relevant signals:

      1. **Probabilistic skill** = ``1 − WIS_model / WIS_climatology`` if both
         provided; ranges (−∞, 1], 0 = match climatology, >0 = beat baseline.

      2. **Surveillance utility** = harmonic mean of alert-detection
         ``sensitivity`` and ``lead_time_weeks`` rescaled to [0, 1] (assuming
         max useful lead = 4 weeks).

    Final ``s_index = 0.5 × skill + 0.5 × utility`` if both available; falls
    back to either alone. If neither is provided, returns NaN.

    Args:
        y_true:           observed values (used for fallback if WIS args absent).
        y_pred:           predicted values (used for fallback).
        wis_model:        model WIS (lower is better).
        wis_climatology:  baseline WIS for skill score.
        sensitivity:      alert detection sensitivity in [0, 1].
        lead_time_weeks:  forecast lead time (negative = lagging).

    Returns:
        float aggregate score, or NaN if insufficient inputs.

    Notes:
        Definition is project-specific (not a published index). Documented as
        an aggregate utility metric for the 4-method comparison report.
    """
    skill = float("nan")
    if wis_model is not None and wis_climatology is not None and wis_climatology > 0:
        skill = 1.0 - float(wis_model) / float(wis_climatology)

    utility = float("nan")
    if sensitivity is not None and lead_time_weeks is not None:
        # Rescale lead time to [0, 1]: 0 = no lead or lag, 1 = ≥4 weeks ahead.
        lt_norm = max(0.0, min(1.0, float(lead_time_weeks) / 4.0))
        sens = max(0.0, min(1.0, float(sensitivity)))
        if sens + lt_norm > 0:
            utility = 2.0 * sens * lt_norm / (sens + lt_norm)

    if np.isfinite(skill) and np.isfinite(utility):
        return float(0.5 * skill + 0.5 * utility)
    if np.isfinite(skill):
        return float(skill)
    if np.isfinite(utility):
        return float(utility)
    return float("nan")


def epi_peak_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    peak_window: int = 2,
) -> float:
    """MAE in a ±peak_window weeks neighborhood around the observed peak.

    Focuses error measurement on the surge season — the period clinicians
    care about most. Default window: peak ± 2 weeks (5-week window).

    Args:
        y_true:      observed values.
        y_pred:      predicted values.
        peak_window: half-width of window around peak (default 2 → 5 weeks).

    Returns:
        MAE over the peak window, or NaN if y_true is empty.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    if y_t.size == 0 or y_t.size != y_p.size:
        return float("nan")
    peak_idx = int(np.argmax(y_t))
    lo = max(0, peak_idx - peak_window)
    hi = min(y_t.size, peak_idx + peak_window + 1)
    return float(np.mean(np.abs(y_p[lo:hi] - y_t[lo:hi])))


def epi_season_total_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Absolute error of the cumulative season-total ILI burden.

    Equivalent to ``|sum(pred) − sum(obs)|``. Used by FluSight and EU Hub
    as a season-level burden metric (Reich et al. 2019 PNAS).

    Args:
        y_true: observed weekly ILI rate, shape (n,).
        y_pred: predicted weekly ILI rate, shape (n,).

    Returns:
        Absolute difference of season totals, or NaN on empty input.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    if y_t.size == 0 or y_t.size != y_p.size:
        return float("nan")
    return float(abs(np.sum(y_p) - np.sum(y_t)))


# ══════════════════════════════════════════════════════════════════════════
# Sprint S6 (2026-05-26) — 12 metrics flagged missing per 9fix glossary
# + ROC AUC for influenza alert binary classification.
# These were either in compute_full_metrics (metric_eval.py) but not
# surfaced in phase11 row dict, or new (PI relative width, ROC AUC).
# ══════════════════════════════════════════════════════════════════════════

def roc_auc(event_true: np.ndarray, event_prob: np.ndarray) -> float:
    """ROC AUC for binary outbreak/alert classification.

    Integrates over all thresholds — sensitive to ranking regardless of
    calibration. Complements `c_index` (which compares rank on continuous
    outcome) by working directly on the binary alert event.

    Standard in influenza alert literature (CDC FluSight overshoot
    detection, EARS aberration). KDCA NIP also reports AUC.

    Args:
        event_true: 0/1 binary outcomes, shape (n,).
        event_prob: forecast probabilities (or any score), shape (n,).

    Returns:
        AUC in [0, 1] (0.5 = random); NaN if all-positive / all-negative
        (AUC undefined when only one class present).
    """
    from sklearn.metrics import roc_auc_score
    e = np.asarray(event_true).astype(int).ravel()
    p = np.asarray(event_prob, dtype=float).ravel()
    if e.size != p.size or e.size < 2:
        return float("nan")
    # Need both classes present
    if e.sum() == 0 or e.sum() == e.size:
        return float("nan")
    try:
        return float(roc_auc_score(e, p))
    except Exception:
        return float("nan")


def pi_relative_widths(
    y_true: np.ndarray,
    pi_widths: dict[str, float],
    *,
    cap: float = 150.0,
) -> dict[str, float]:
    """Relative PI widths = PI_width / |mean(y_true)|, with G-215 cap.

    Used by the 9fix metric system (pi50_rel_width, pi80_rel_width,
    pi95_rel_width). A value of 1.0 means the interval is as wide as the
    observed mean ILI rate; >1 = oversized; <1 = sharp.

    G-215 cap=150 prevents runaway relative widths on near-zero ILI weeks.

    Args:
        y_true: observed values, shape (n,).
        pi_widths: dict with keys like 'pi50_width', 'pi80_width', 'pi95_width'.
        cap: maximum relative width (default 150 per G-215).

    Returns:
        Dict with 'pi50_rel_width', 'pi80_rel_width', 'pi95_rel_width' keys.
    """
    y_mean = float(np.abs(np.mean(np.asarray(y_true, dtype=float))))
    if y_mean <= 0:
        return {f"pi{lvl}_rel_width": float("nan") for lvl in (50, 80, 95)}
    out: dict[str, float] = {}
    for lvl in (50, 80, 95):
        w = pi_widths.get(f"pi{lvl}_width", float("nan"))
        if not np.isfinite(w):
            out[f"pi{lvl}_rel_width"] = float("nan")
        else:
            rel = float(w) / y_mean
            out[f"pi{lvl}_rel_width"] = min(rel, cap)
    return out


def cost_skill_ratios(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Asymmetric cost-skill scores at common false-negative:false-positive
    ratios (3:1, 5:1, 10:1).

    Cost-weighted skill score for binary alert. FN_cost ratio captures the
    relative penalty of missing an outbreak vs. a false alarm.

    Formula::

        cost = FN_cost * fn + FP_cost * fp
        baseline_cost = FN_cost * (n_positive)        # always-negative
        skill = 1 - cost / baseline_cost   (higher = better)

    Args:
        y_true: observed ILI rate, shape (n,).
        y_pred: predicted ILI rate, shape (n,).
        threshold: alert threshold.

    Returns:
        Dict with 'cost_skill_3to1', '5to1', '10to1' keys.
    """
    e = (np.asarray(y_true, dtype=float) > threshold).astype(int)
    p = (np.asarray(y_pred, dtype=float) > threshold).astype(int)
    if e.size != p.size or e.size == 0:
        return {f"cost_skill_{r}to1": float("nan") for r in (3, 5, 10)}

    fn = int(((e == 1) & (p == 0)).sum())
    fp = int(((e == 0) & (p == 1)).sum())
    n_pos = int(e.sum())
    if n_pos == 0:
        return {f"cost_skill_{r}to1": float("nan") for r in (3, 5, 10)}

    out: dict[str, float] = {}
    for r in (3, 5, 10):
        fn_cost = float(r)
        fp_cost = 1.0
        cost = fn_cost * fn + fp_cost * fp
        baseline = fn_cost * n_pos  # always-negative classifier
        out[f"cost_skill_{r}to1"] = float(1.0 - cost / baseline) if baseline > 0 else float("nan")
    return out


def diebold_mariano_vs_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Diebold-Mariano test (Harvey-corrected) vs a baseline predictor.

    Tests whether two forecasts have equal accuracy. p<0.05 → significantly
    different. Default baseline = lag-1 persistence (y_baseline[i] = y_true[i-1]).

    Args:
        y_true:     observed values, shape (n,).
        y_pred:     model predictions, shape (n,).
        y_baseline: baseline predictions; defaults to lag-1 persistence.

    Returns:
        Dict with 'dm_z_stat' and 'dm_p_value'.
    """
    from simulation.analytics.metrics import diebold_mariano
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    if y_t.size < 3 or y_t.size != y_p.size:
        return {"dm_z_stat": float("nan"), "dm_p_value": float("nan")}
    if y_baseline is None:
        y_b = np.concatenate([[y_t[0]], y_t[:-1]])  # lag-1 persistence
    else:
        y_b = np.asarray(y_baseline, dtype=float).ravel()
        if y_b.size != y_t.size:
            return {"dm_z_stat": float("nan"), "dm_p_value": float("nan")}
    try:
        z, p = diebold_mariano(y_t, y_p, y_b, h=1)
        return {"dm_z_stat": float(z), "dm_p_value": float(p)}
    except Exception:
        return {"dm_z_stat": float("nan"), "dm_p_value": float("nan")}


def lead_time_weeks_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float,
) -> float:
    """Lead time (in weeks) of model alert vs observed threshold crossing.

    Returns positive if model alerts BEFORE observation crosses threshold
    (early warning); negative if AFTER (lagging). NaN if neither crosses.

    Args:
        y_true:    observed values, shape (n,).
        y_pred:    predicted values, shape (n,).
        threshold: alert threshold.

    Returns:
        Lead time in weeks (positive = early warning, negative = lag), or NaN.
    """
    e = np.asarray(y_true, dtype=float) > threshold
    p = np.asarray(y_pred, dtype=float) > threshold
    if not e.any() or not p.any():
        return float("nan")
    first_obs = int(np.argmax(e))   # first True index
    first_pred = int(np.argmax(p))
    return float(first_obs - first_pred)  # positive = pred earlier


# ══════════════════════════════════════════════════════════════════════════
# Sprint S7 (2026-05-26) — full classification/confusion-matrix metric suite
# User: "AUR-ROC나 다른 ROC도 있지 않아? F1 score도 있고 Prediction table로
#       accuracy, 민감도, 특이도 같은것들도 말이야?"
# Adds: AUPRC + partial AUC (high-spec); F0.5 + F2; full confusion matrix
# (tp/tn/fp/fn) + accuracy + balanced_accuracy + g_mean + prevalence;
# diagnostic odds ratio + markedness + informedness (alias Youden's J).
# ══════════════════════════════════════════════════════════════════════════

def roc_family_metrics(
    event_true: np.ndarray,
    event_prob: np.ndarray,
    *,
    high_spec_fpr_max: float = 0.1,
) -> Dict[str, float]:
    """Full ROC family: AUC + AUPRC + partial AUC at high specificity.

    Three complementary discrimination scores:

    - ``roc_auc``: standard AUC; integrates TPR over all FPRs.
    - ``auprc``: Area Under Precision-Recall curve (= average_precision).
       More informative than ROC AUC under class imbalance (typical for
       outbreak detection where positive weeks are rare). FluSight/RESP-LENS
       reports both.
    - ``partial_auc_high_spec``: ROC AUC restricted to FPR ≤ ``high_spec_fpr_max``
       (default 0.1). Clinical screening standard — measures discrimination
       only in the operationally-useful low-false-alarm region.

    Args:
        event_true: 0/1 binary outcomes, shape (n,).
        event_prob: forecast probabilities or scores, shape (n,).
        high_spec_fpr_max: upper FPR bound for partial AUC (default 0.1
            → "specificity ≥ 0.9" region).

    Returns:
        Dict with ``roc_auc``, ``auprc``, ``partial_auc_high_spec``.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
    e = np.asarray(event_true).astype(int).ravel()
    p = np.asarray(event_prob, dtype=float).ravel()
    if e.size != p.size or e.size < 2 or e.sum() in (0, e.size):
        return {"roc_auc": float("nan"), "auprc": float("nan"),
                "partial_auc_high_spec": float("nan")}
    out: Dict[str, float] = {}
    try:
        out["roc_auc"] = float(roc_auc_score(e, p))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(e, p))
    except Exception:
        out["auprc"] = float("nan")
    try:
        fpr, tpr, _ = roc_curve(e, p)
        mask = fpr <= high_spec_fpr_max
        if mask.sum() >= 2:
            # Use full integration: prepend (0,0); trapezoid up to cap.
            out["partial_auc_high_spec"] = float(auc(fpr[mask], tpr[mask]))
        else:
            out["partial_auc_high_spec"] = float("nan")
    except Exception:
        out["partial_auc_high_spec"] = float("nan")
    return out


def f_beta_scores(
    event_true: np.ndarray,
    event_pred: np.ndarray,
) -> Dict[str, float]:
    """F-β family at β ∈ {0.5, 1, 2}.

    - ``f1_score``    : harmonic mean of precision + recall (balanced).
    - ``f2_score``    : β=2, weights recall higher — outbreak detection
                        favors recall (don't miss surges).
    - ``f05_score``   : β=0.5, weights precision higher — favors fewer
                        false alarms (resource-constrained labs).

    Args:
        event_true: 0/1 binary outcomes.
        event_pred: 0/1 binary predictions.

    Returns:
        Dict with ``f1_score``, ``f2_score``, ``f05_score``.
    """
    from sklearn.metrics import fbeta_score
    e = np.asarray(event_true).astype(int).ravel()
    p = np.asarray(event_pred).astype(int).ravel()
    if e.size != p.size or e.size == 0:
        return {"f1_score": float("nan"), "f2_score": float("nan"),
                "f05_score": float("nan")}
    out: Dict[str, float] = {}
    for beta, name in [(1.0, "f1_score"), (2.0, "f2_score"), (0.5, "f05_score")]:
        try:
            out[name] = float(fbeta_score(e, p, beta=beta, zero_division=0))
        except Exception:
            out[name] = float("nan")
    return out


def confusion_matrix_table(
    event_true: np.ndarray,
    event_pred: np.ndarray,
) -> Dict[str, float]:
    """Full confusion-matrix prediction table for binary outbreak alert.

    Returns the standard 2×2 cells (TP, TN, FP, FN) plus derived metrics:

    - ``accuracy``         = (TP + TN) / n
    - ``balanced_accuracy``= (sens + spec) / 2  (imbalance-robust)
    - ``prevalence``       = (TP + FN) / n      (base rate of positives)
    - ``g_mean``           = sqrt(sens × spec)  (geometric mean —
                             imbalanced classifier standard)

    Args:
        event_true: 0/1 binary outcomes, shape (n,).
        event_pred: 0/1 binary predictions, shape (n,).

    Returns:
        Dict with ``tp``, ``tn``, ``fp``, ``fn`` (int), ``accuracy``,
        ``balanced_accuracy``, ``prevalence``, ``g_mean`` (float).
    """
    e = np.asarray(event_true).astype(int).ravel()
    p = np.asarray(event_pred).astype(int).ravel()
    if e.size != p.size or e.size == 0:
        return {"tp": 0, "tn": 0, "fp": 0, "fn": 0,
                "accuracy": float("nan"), "balanced_accuracy": float("nan"),
                "prevalence": float("nan"), "g_mean": float("nan")}
    tp = int(((e == 1) & (p == 1)).sum())
    tn = int(((e == 0) & (p == 0)).sum())
    fp = int(((e == 0) & (p == 1)).sum())
    fn = int(((e == 1) & (p == 0)).sum())
    n = e.size
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prev = (tp + fn) / n
    out: Dict[str, float] = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": float((sens + spec) / 2.0),
        "prevalence": float(prev),
        "g_mean": float(np.sqrt(sens * spec)) if sens * spec >= 0 else float("nan"),
    }
    return out


def clinical_diagnostic_metrics(
    event_true: np.ndarray,
    event_pred: np.ndarray,
) -> Dict[str, float]:
    """Diagnostic Odds Ratio + Markedness + Informedness (Youden's J).

    Classical clinical-epidemiology diagnostic-test summary statistics:

    - ``dor`` = (TP/FN) / (FP/TN) = (sens × spec) / ((1-sens) × (1-spec))
       Single number summarizing test discrimination. >1 = useful;
       gold-standard tests have DOR ≥ 20 (Glas et al. 2003).
    - ``markedness``    = PPV + NPV − 1  (Powers 2011)
    - ``informedness``  = sens + spec − 1 = Youden's J (Youden 1950)
                          Equivalent to Youden index; in [-1, 1].

    Args:
        event_true: 0/1 binary outcomes.
        event_pred: 0/1 binary predictions.

    Returns:
        Dict with ``dor``, ``markedness``, ``informedness``,
        ``youden_j`` (= informedness, alias for legacy 9fix compat).
    """
    e = np.asarray(event_true).astype(int).ravel()
    p = np.asarray(event_pred).astype(int).ravel()
    if e.size != p.size or e.size == 0:
        return {"dor": float("nan"), "markedness": float("nan"),
                "informedness": float("nan"), "youden_j": float("nan")}
    tp = int(((e == 1) & (p == 1)).sum())
    tn = int(((e == 0) & (p == 0)).sum())
    fp = int(((e == 0) & (p == 1)).sum())
    fn = int(((e == 1) & (p == 0)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv  = tp / max(1, tp + fp)
    npv  = tn / max(1, tn + fn)
    # DOR: avoid division by zero (use +0.5 continuity correction if any cell = 0)
    if fp == 0 or fn == 0:
        a, b, c, d = tp + 0.5, fn + 0.5, fp + 0.5, tn + 0.5
        dor = (a * d) / (b * c)
    else:
        dor = (tp * tn) / (fp * fn)
    return {
        "dor":          float(dor),
        "markedness":   float(ppv + npv - 1.0),
        "informedness": float(sens + spec - 1.0),
        "youden_j":     float(sens + spec - 1.0),  # alias of informedness
    }


# ══════════════════════════════════════════════════════════════════════════
# Clinical / binary decision metrics
# ══════════════════════════════════════════════════════════════════════════
def binary_clinical_rates(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    """Sensitivity, specificity, PPV, NPV, prevalence, accuracy, F1.

    `y_true` is 0/1 ground truth; `y_score` is a continuous score; the
    binary decision is `y_score >= threshold`. All rates have `_ci_lo`
    / `_ci_hi` Wilson 95% bounds, and `nnt` / `nns` are returned as the
    numbers-needed-to-treat and screen implied by Sens/Spec against the
    sample prevalence.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    pred = (s >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    n = tp + fp + tn + fn

    def _rate(k: int, m: int) -> Tuple[float, float, float]:
        if m == 0:
            return float("nan"), float("nan"), float("nan")
        p_hat = k / m
        lo, hi = _wilson_ci(k, m, alpha=0.05)
        return p_hat, lo, hi

    sens, sens_lo, sens_hi = _rate(tp, tp + fn)
    spec, spec_lo, spec_hi = _rate(tn, tn + fp)
    ppv,  ppv_lo,  ppv_hi  = _rate(tp, tp + fp)
    npv,  npv_lo,  npv_hi  = _rate(tn, tn + fn)

    prevalence = (tp + fn) / n if n else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) > 0 else float("nan")

    # NNT = 1 / absolute risk reduction at this threshold
    #   ARR = sensitivity - (1 - specificity) * prevalence / (1 - prevalence)
    # Guard against degenerate 0 / 0 in rare-event regimes.
    arr = sens - (1.0 - spec)
    nnt = float("inf") if arr <= 0 else 1.0 / arr
    nns = float("inf") if prevalence <= 0 or sens <= 0 else 1.0 / (prevalence * sens)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
        "prevalence": prevalence, "accuracy": acc, "f1": f1,
        "sensitivity": sens, "sensitivity_ci_lo": sens_lo, "sensitivity_ci_hi": sens_hi,
        "specificity": spec, "specificity_ci_lo": spec_lo, "specificity_ci_hi": spec_hi,
        "ppv": ppv, "ppv_ci_lo": ppv_lo, "ppv_ci_hi": ppv_hi,
        "npv": npv, "npv_ci_lo": npv_lo, "npv_ci_hi": npv_hi,
        "nnt": nnt, "nns": nns,
        "threshold": float(threshold),
    }


def decision_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Decision-curve analysis (Vickers & Elkin 2006).

    Net benefit at threshold p_t =
        (TP / n) - (FP / n) · p_t / (1 - p_t).
    Returns the curve plus the "treat-all" and "treat-none" references
    at each threshold, so the caller can plot all three.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    n = len(y)
    if n == 0:
        return []
    prevalence = float(np.mean(y))

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 30)

    out: List[Dict[str, float]] = []
    for pt in thresholds:
        if pt <= 0.0 or pt >= 1.0:
            continue
        pred = (s >= pt).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        nb_model = tp / n - (fp / n) * (pt / (1.0 - pt))
        nb_all   = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        out.append({
            "threshold": float(pt),
            "nb_model": float(nb_model),
            "nb_treat_all": float(nb_all),
            "nb_treat_none": 0.0,
        })
    return out


# ══════════════════════════════════════════════════════════════════════════
# Uncertainty — bootstrap CI (percentile + BCa)
# ══════════════════════════════════════════════════════════════════════════
def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "bca",
    random_state: Optional[int] = None,
    block_len: int = 0,
) -> Dict[str, float]:
    """Nonparametric bootstrap CI for a scalar statistic.

    Args:
        data: 1-D sample array.
        statistic: scalar-returning callable.
        n_boot: number of bootstrap replicates.
        alpha: significance level (default 0.05 → 95% CI).
        method: "percentile" for the naive quantile interval, or "bca" for
            the bias-corrected and accelerated interval (DiCiccio & Efron 1996).
            Defaults to BCa because it is the recommended choice when the
            statistic is potentially biased (e.g. R² on small folds).
        random_state: RNG seed for reproducibility.
        block_len: moving-block bootstrap (Künsch 1989). 0 = iid resampling
            (default, preserves prior behavior). >0 = resample contiguous
            blocks of `block_len` indices to preserve serial dependence
            in time-series statistics (PICP, WIS, ΔPICP). A common rule of
            thumb is ``int(np.sqrt(n))`` for weakly-dependent series.

    Returns:
        Dict with ``estimate``, ``ci_lo``, ``ci_hi``, ``se``, ``method``,
        ``n``, and (for BCa) ``z0`` + ``a_hat``. When ``block_len > 0`` the
        returned ``method`` string is suffixed with ``+block{L}``.

    Notes:
        For time-series uncertainty (e.g. ΔPICP, ΔWIS across resampled test
        windows), prefer ``block_len = int(np.sqrt(n))`` over iid. iid
        bootstrap underestimates CI width when residuals are autocorrelated;
        the moving block bootstrap recovers proper coverage for stationary
        weakly-dependent series.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 2:
        return {"estimate": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "se": float("nan"), "method": method, "n": n}
    rng = np.random.default_rng(random_state)

    theta_hat = float(statistic(x))
    boots = np.empty(n_boot)

    if block_len and block_len > 0:
        # Moving-block bootstrap (Künsch 1989): sample contiguous blocks.
        # Number of blocks ceil(n / block_len). Trim to length n at the end.
        L = int(block_len)
        n_blocks = (n + L - 1) // L
        max_start = max(1, n - L + 1)
        for b in range(n_boot):
            starts = rng.integers(0, max_start, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
            boots[b] = statistic(x[idx])
        method_label = f"{method}+block{L}"
    else:
        # iid bootstrap (legacy default).
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[b] = statistic(x[idx])
        method_label = method
    se = float(np.std(boots, ddof=1))

    if method == "percentile":
        lo = float(np.quantile(boots, alpha / 2.0))
        hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
        return {"estimate": theta_hat, "ci_lo": lo, "ci_hi": hi,
                "se": se, "method": method_label, "n": n}

    # BCa
    prop_below = float(np.mean(boots < theta_hat))
    prop_below = min(max(prop_below, 1.0 / (10 * n_boot)), 1.0 - 1.0 / (10 * n_boot))
    z0 = stats.norm.ppf(prop_below)

    # Acceleration via jackknife
    jack = np.empty(n)
    for i in range(n):
        jack[i] = statistic(np.delete(x, i))
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a_hat = 0.0 if den == 0 else float(num / den)

    z_lo = stats.norm.ppf(alpha / 2.0)
    z_hi = stats.norm.ppf(1.0 - alpha / 2.0)

    def _adjust(z: float) -> float:
        denom = 1.0 - a_hat * (z0 + z)
        if denom == 0:
            return float("nan")
        return stats.norm.cdf(z0 + (z0 + z) / denom)

    lo_pct = _adjust(z_lo)
    hi_pct = _adjust(z_hi)
    lo = float(np.quantile(boots, np.clip(lo_pct, 0.0, 1.0)))
    hi = float(np.quantile(boots, np.clip(hi_pct, 0.0, 1.0)))
    return {"estimate": theta_hat, "ci_lo": lo, "ci_hi": hi,
            "se": se, "method": method_label, "n": n, "z0": float(z0), "a_hat": a_hat}


# ══════════════════════════════════════════════════════════════════════════
# Multiple testing corrections
# ══════════════════════════════════════════════════════════════════════════
def adjust_pvalues(
    pvalues: Sequence[float],
    method: str = "fdr_bh",
) -> Dict[str, Any]:
    """Family-wise / FDR adjustment for a vector of p-values.

    `method`:
      - "bonferroni" — classical FWER (conservative).
      - "holm"       — step-down Holm-Bonferroni (uniformly more powerful
                       than bonferroni while still controlling FWER).
      - "fdr_bh"     — Benjamini-Hochberg FDR (default; standard for
                       model-shootout DM tests where hundreds of pairs
                       are compared).

    Returns a dict with the input, adjusted p-values, and a 5%-rejection
    mask. Uses scipy.stats.false_discovery_control / combine_pvalues
    primitives to avoid pulling statsmodels just for this.
    """
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    if n == 0:
        return {"method": method, "p_raw": p, "p_adj": p.copy(),
                "reject": np.zeros(0, dtype=bool)}

    m = method.lower()
    if m == "bonferroni":
        p_adj = np.minimum(p * n, 1.0)
    elif m == "holm":
        order = np.argsort(p)
        adj_sorted = np.empty(n)
        running = 0.0
        for rank, idx in enumerate(order):
            candidate = (n - rank) * p[idx]
            running = max(running, candidate)
            adj_sorted[rank] = min(running, 1.0)
        p_adj = np.empty(n)
        p_adj[order] = adj_sorted
    elif m in ("fdr_bh", "bh"):
        order = np.argsort(p)
        sorted_p = p[order]
        ranks = np.arange(1, n + 1)
        bh_vals = sorted_p * n / ranks
        # enforce monotone non-decreasing from the largest down
        for i in range(n - 2, -1, -1):
            bh_vals[i] = min(bh_vals[i], bh_vals[i + 1])
        bh_vals = np.clip(bh_vals, 0.0, 1.0)
        p_adj = np.empty(n)
        p_adj[order] = bh_vals
    else:
        raise ValueError(f"unknown method: {method!r}")

    return {
        "method": m,
        "p_raw": p,
        "p_adj": p_adj,
        "reject": p_adj < 0.05,
    }


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def _wilson_ci(k: int, n: int, *, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score CI for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))



# ======================================================================
# Epidemic phase metrics (ILI forecasting -- Biggerstaff 2016, Cowling 2020)
# ======================================================================

def epidemic_phase_metrics(
    y_test,
    y_pred,
    threshold: float,
) -> dict:
    """Epidemic-phase accuracy metrics for ILI forecasting.

    Evaluates how accurately the model captures epidemic dynamics beyond
    point-accuracy: total burden, growth trajectory, epidemic duration,
    and onset/peak timing.

    Args:
        y_test:     observed ILI rate (n,)
        y_pred:     point forecast   (n,)
        threshold:  epidemic alert threshold (e.g. KDCA 8.6 per 1000)

    Returns:
        dict with keys:
          attack_rate_relerr    -- relative error of cumulative ILI burden
          growth_rate_corr      -- Pearson r of week-over-week growth rates
          epidemic_duration_err -- signed error in weeks above threshold
          season_onset_err      -- signed week-index error of first exceedance
                                   = onset_pred_week − onset_true_week
                                   (+) model onset LATER than observed
          early_warning_lead    -- weeks model predicts onset before observed
                                   = onset_true_week − onset_pred_week
                                   (+) model onset EARLIER (early warning)
                                   NOTE: early_warning_lead = −season_onset_err
                                   always. Both keys are kept for directional
                                   clarity (sign convention differs between
                                   onset-accuracy and clinical-lead frames).

    Performance: O(n). NaN-safe.
    Caller responsibility: threshold > 0; y_test/y_pred finite-masked externally.
    """
    nan = float("nan")
    try:
        yt = np.asarray(y_test, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if not mask.any():
            return dict(attack_rate_relerr=nan, growth_rate_corr=nan,
                        epidemic_duration_err=nan, season_onset_err=nan,
                        early_warning_lead=nan)
        a, p = yt[mask], yp[mask]
        thr = float(threshold)

        # Attack rate (cumulative burden -- proportional to total case burden)
        ar_true = float(np.sum(a))
        ar_pred = float(np.sum(p))
        attack_relerr = ((ar_pred - ar_true) / ar_true) if ar_true > 1e-6 else nan

        # Growth rate correlation (Cowling 2020 -- model captures trend dynamics)
        # ILI units are per-1,000 visits. Epsilon 0.1 avoids 1000× amplification
        # of off-season near-zero weeks (1e-6 would cause extreme outlier growth rates).
        if len(a) >= 3:
            _ILI_EPS = 0.1
            gr_true = np.diff(a) / np.maximum(a[:-1], _ILI_EPS)
            gr_pred = np.diff(p) / np.maximum(p[:-1], _ILI_EPS)
            gm = np.isfinite(gr_true) & np.isfinite(gr_pred)
            if gm.sum() >= 3:
                from scipy.stats import pearsonr
                gr_corr, _ = pearsonr(gr_true[gm], gr_pred[gm])
                gr_corr = float(gr_corr)
            else:
                gr_corr = nan
        else:
            gr_corr = nan

        # Epidemic duration (weeks above threshold)
        dur_true = int(np.sum(a > thr))
        dur_pred = int(np.sum(p > thr))
        dur_err  = float(dur_pred - dur_true)

        # Season onset (first week above threshold)
        onset_true_arr = np.where(a > thr)[0]
        onset_pred_arr = np.where(p > thr)[0]
        if len(onset_true_arr) > 0 and len(onset_pred_arr) > 0:
            onset_err = float(onset_pred_arr[0] - onset_true_arr[0])
        else:
            onset_err = nan

        # Early warning lead (how many weeks BEFORE obs onset the model predicted)
        if len(onset_true_arr) > 0 and len(onset_pred_arr) > 0:
            lead = float(onset_true_arr[0] - onset_pred_arr[0])  # positive = lead time
        else:
            lead = nan

        return dict(
            attack_rate_relerr=float(attack_relerr) if np.isfinite(attack_relerr) else nan,
            growth_rate_corr=gr_corr,
            epidemic_duration_err=dur_err,
            season_onset_err=onset_err,
            early_warning_lead=lead,
        )
    except Exception:
        return dict(attack_rate_relerr=nan, growth_rate_corr=nan,
                    epidemic_duration_err=nan, season_onset_err=nan,
                    early_warning_lead=nan)


# ======================================================================
# Advanced clinical metrics (Chicco 2020, Vickers 2006, Altman 1994)
# ======================================================================

def advanced_clinical_metrics_ext(
    y_test,
    y_pred,
    threshold: float,
    prior_prob: float = 0.30,
) -> dict:
    """Extended clinical classification metrics at a binary alert threshold.

    Supplements binary_clinical_rates() with MCC, Cohen's kappa, likelihood
    ratios, and net benefit -- standard metrics for clinical decision tool
    validation (Chicco & Jurman 2020, Vickers & Elkin 2006).

    Args:
        y_test:     observed ILI rate (n,)
        y_pred:     point forecast   (n,)
        threshold:  epidemic alert threshold (binary split)
        prior_prob: prior probability for net benefit denominator (default 0.30)

    Returns:
        dict with keys:
          mcc              -- Matthews Correlation Coefficient in [-1, 1]
          cohens_kappa     -- Cohen's kappa (agreement beyond chance)
          lr_positive      -- sensitivity / (1 - specificity)
          lr_negative      -- (1 - sensitivity) / specificity
          net_benefit_default -- net benefit at threshold `prior_prob`

    Performance: O(n). NaN-safe.
    Caller responsibility: threshold > 0.
    """
    nan = float("nan")
    try:
        yt = np.asarray(y_test, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if not mask.any():
            return dict(mcc=nan, cohens_kappa=nan, lr_positive=nan,
                        lr_negative=nan, net_benefit_default=nan)
        a, p = yt[mask], yp[mask]
        thr = float(threshold)
        n = len(a)

        obs_bin  = (a > thr).astype(int)
        pred_bin = (p > thr).astype(int)
        tp = int(np.sum((obs_bin == 1) & (pred_bin == 1)))
        fp = int(np.sum((obs_bin == 0) & (pred_bin == 1)))
        tn = int(np.sum((obs_bin == 0) & (pred_bin == 0)))
        fn = int(np.sum((obs_bin == 1) & (pred_bin == 0)))

        # MCC (Chicco & Jurman 2020)
        denom_mcc = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (float(tp * tn - fp * fn) / np.sqrt(denom_mcc)
               if denom_mcc > 0 else nan)

        # Cohen's kappa
        p_obs = (tp + tn) / n if n > 0 else nan
        p_e   = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n) if n > 0 else nan
        kappa = ((p_obs - p_e) / (1.0 - p_e)
                 if (p_e is not nan and abs(1.0 - p_e) > 1e-9) else nan)

        # Likelihood ratios (Altman & Bland 1994)
        sens = tp / (tp + fn) if (tp + fn) > 0 else nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else nan
        lr_pos = (sens / (1.0 - spec)
                  if (spec is not nan and abs(1.0 - spec) > 1e-9 and
                      sens is not nan) else nan)
        lr_neg = ((1.0 - sens) / spec
                  if (spec is not nan and spec > 1e-9 and
                      sens is not nan) else nan)

        # Net benefit (Vickers & Elkin 2006) at prior_prob threshold
        pt = float(prior_prob)
        if 0.0 < pt < 1.0 and n > 0:
            nb = tp / n - (fp / n) * (pt / (1.0 - pt))
        else:
            nb = nan

        return dict(
            mcc=float(mcc) if (mcc is not nan and np.isfinite(mcc)) else nan,
            cohens_kappa=float(kappa) if (kappa is not nan and np.isfinite(kappa)) else nan,
            lr_positive=float(lr_pos) if (lr_pos is not nan and np.isfinite(lr_pos)) else nan,
            lr_negative=float(lr_neg) if (lr_neg is not nan and np.isfinite(lr_neg)) else nan,
            net_benefit_default=float(nb) if (nb is not nan and np.isfinite(nb)) else nan,
        )
    except Exception:
        return dict(mcc=nan, cohens_kappa=nan, lr_positive=nan,
                    lr_negative=nan, net_benefit_default=nan)
