"""
Hub-standard metrics for SCI-tier comparability with FluSight / RespiCast.
==========================================================================

This module bundles the metric variants that an external SCI-tier reviewer
audit (2026-04-25) flagged as required for hub-comparability:

  C. K=11 PI levels (Bracher 2021 full / FluSight 23-quantile spec)
  D. Log-transformed WIS (Bosse et al. 2023, FluSight 2024-25 standard)
  E. Pairwise tournament relative WIS (Sherratt 2023 eLife)
  F. Wilson exact CI for empirical PI coverage at small n

The base WIS / pinball / coverage primitives live in
`simulation.analytics.metrics` + `simulation.analytics.diagnostics`. This
module composes them into the hub-canonical reporting layer.

References:
  Bosse NI et al. (2023). Scoring epidemiological forecasts on transformed
    scales. PLoS Comp Bio 19(8):e1011393.
  Sherratt K et al. (2023). Predictive performance of multi-model ensemble
    forecasts of COVID-19 across European nations. eLife 12:e81916.
  Bracher J et al. (2021). Evaluating epidemic forecasts in an interval
    format. PLOS Comp Bio 17(2):e1008618.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ─── C. K=11 PI levels (FluSight 23-quantile spec) ──────────────────────
#
# FluSight (CDC) requires forecasts at 23 quantiles:
#   {0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
#    0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900,
#    0.950, 0.975, 0.990}
# These correspond to K=11 central prediction intervals (PI levels) plus
# the median.  The 11 α values (1 − 2 × q_low) are:
FLUSIGHT_ALPHAS: tuple = (
    0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
)
FLUSIGHT_QUANTILES: tuple = (
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
)


def k11_pi_widths_from_residuals(
    abs_residuals: np.ndarray,
    alphas: tuple = FLUSIGHT_ALPHAS,
) -> dict[float, float]:
    """Lei 2018 / Vovk 2005 split-conformal half-widths for K=11 levels.

    Returns: {alpha: q_alpha}  — half-width of the (1−α) PI.
    """
    res = np.sort(np.abs(np.asarray(abs_residuals, dtype=np.float64)))
    res = res[np.isfinite(res)]
    n = len(res)
    if n == 0:
        return {a: float("inf") for a in alphas}
    out = {}
    for a in alphas:
        k = int(np.ceil((n + 1) * (1.0 - a))) - 1
        k = max(0, min(k, n - 1))
        out[float(a)] = float(res[k])
    return out


# ─── D. Log-transformed WIS (Bosse 2023, FluSight 2024-25 standard) ─────


def weighted_interval_score_logscale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: float,
    alphas: tuple = FLUSIGHT_ALPHAS,
    epsilon: float = 1.0,
) -> np.ndarray:
    """WIS on log-transformed scale (Gaussian-σ closed-form).

    DEPRECATED for primary ILI evaluation as of 2026-05-26 S8 — use
    `weighted_interval_score_logscale_empirical()` instead.

    Bosse et al. 2023 (PLoS Comp Bio 19:e1011393).
    """
    from .diagnostics import weighted_interval_score
    yt = np.log(np.asarray(y_true, dtype=np.float64) + epsilon)
    yp = np.log(np.asarray(y_pred, dtype=np.float64) + epsilon)
    # σ in log-space ≈ σ_raw / mean(y + ε) (delta method approximation)
    mean_y = float(np.mean(np.asarray(y_pred) + epsilon))
    sigma_log = float(sigma) / max(mean_y, epsilon)
    return weighted_interval_score(yt, yp, sigma_log, alphas=alphas)


def weighted_interval_score_logscale_empirical(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    alphas: tuple = FLUSIGHT_ALPHAS,
    epsilon: float = 1.0,
) -> np.ndarray:
    """Log-scale WIS with empirical residual quantiles (Bosse 2023 + Lei 2018).

    Computes WIS on log(y+ε) with empirical |log-residual| quantiles for PI
    half-widths — no Gaussian assumption. FluSight 2024-25 primary headline
    metric, made distribution-free.

    Added 2026-05-26 (S8 Tier C migration).

    Args:
        y_true: observed (n,)
        y_pred: point forecast (n,)
        residuals: raw-scale residuals (will be log-transformed internally for
                   half-width calibration on log scale).
        alphas: K PI levels
        epsilon: log-shift (default 1.0 per FluSight)
    """
    from .diagnostics import weighted_interval_score_empirical
    yt = np.log(np.asarray(y_true, dtype=np.float64) + epsilon)
    yp = np.log(np.asarray(y_pred, dtype=np.float64) + epsilon)
    # Log-space residual: derive from raw `residuals` via delta-method-equivalent
    # remapping. Simpler/cleaner: use observed log-scale residual when y_train
    # available; here approximate with raw_residual / (mean(y) + ε).
    raw_res = np.asarray(residuals, dtype=np.float64)
    raw_res = raw_res[np.isfinite(raw_res)]
    if len(raw_res) < 2:
        return np.full(len(yt), float("inf"))
    mean_y = float(np.mean(np.abs(np.asarray(y_pred)) + epsilon))
    log_res = raw_res / max(mean_y, epsilon)  # delta-method log-scale residuals
    return weighted_interval_score_empirical(yt, yp, log_res, alphas=list(alphas))


# ─── E. Pairwise tournament relative WIS (Sherratt 2023) ────────────────


def pairwise_relative_wis(
    wis_per_model: dict[str, np.ndarray],
    geometric_mean: bool = True,
) -> dict[str, float]:
    """Pairwise tournament relative WIS per Sherratt 2023 eLife.

    For each model M:
      θ_M = geomean( WIS_M / WIS_M' )  for all M' ≠ M  (across forecast targets)
      = product over M' of (WIS_M / WIS_M')^(1/K-1)

    Lower θ_M = M is better than the average of all opponents.

    Args:
      wis_per_model: {model_name: array of WIS scores per forecast target}.
        Arrays must all have the same length (paired comparison).

    Returns: {model_name: relative_WIS}
    """
    names = list(wis_per_model.keys())
    if len(names) < 2:
        raise ValueError("need ≥2 models for pairwise tournament")
    # Stack per-target WIS scores
    arrs = {n: np.asarray(w, dtype=np.float64) for n, w in wis_per_model.items()}
    L = next(iter(arrs.values())).shape[0]
    for n, a in arrs.items():
        if a.shape[0] != L:
            raise ValueError(f"WIS length mismatch: {n} has {a.shape[0]} vs {L}")

    rel: dict[str, float] = {}
    for m in names:
        ratios = []
        for opp in names:
            if opp == m:
                continue
            wm = arrs[m]
            wo = arrs[opp]
            mask = (wm > 0) & (wo > 0) & np.isfinite(wm) & np.isfinite(wo)
            if not mask.any():
                continue
            r = wm[mask] / wo[mask]
            ratios.append(np.exp(np.mean(np.log(r))))  # geomean per-pair
        if not ratios:
            rel[m] = float("nan")
            continue
        if geometric_mean:
            rel[m] = float(np.exp(np.mean(np.log(ratios))))
        else:
            rel[m] = float(np.mean(ratios))
    return rel


# ─── F. Wilson exact CI for PI coverage ─────────────────────────────────


def wilson_score_ci(
    n_hits: int, n_total: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Wilson score interval — recommended for empirical proportions at
    small n. Returns (point, lower, upper).

    Reference: Wilson (1927). For survey of intervals see Brown, Cai &
    DasGupta (2001). When n_total < 30 and Wilson lower hits 0 or upper
    hits 1, prefer Clopper-Pearson exact (also implemented below).
    """
    if n_total <= 0:
        return float("nan"), float("nan"), float("nan")
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p_hat = n_hits / n_total
    denom = 1.0 + z * z / n_total
    centre = (p_hat + z * z / (2.0 * n_total)) / denom
    half = (z * math.sqrt(p_hat * (1.0 - p_hat) / n_total
                          + z * z / (4.0 * n_total ** 2))) / denom
    return float(p_hat), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def clopper_pearson_ci(
    n_hits: int, n_total: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Clopper-Pearson exact binomial CI. Conservative; preferable to Wilson
    when n is very small (n < 20) or extreme (k=0 or k=n)."""
    if n_total <= 0:
        return float("nan"), float("nan"), float("nan")
    from scipy.stats import beta
    p_hat = n_hits / n_total
    if n_hits == 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(alpha / 2.0, n_hits, n_total - n_hits + 1))
    if n_hits == n_total:
        hi = 1.0
    else:
        hi = float(beta.ppf(1.0 - alpha / 2.0, n_hits + 1, n_total - n_hits))
    return float(p_hat), lo, hi


def coverage_with_exact_ci(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *, nominal: float, method: str = "wilson",
) -> dict:
    """Empirical coverage + exact CI (Wilson by default, Clopper-Pearson
    optional). Replaces normal-approximation `pi_coverage` in the small-n
    regime (FluSight reviewers expect this at n < 30)."""
    yt = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return {"empirical": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": 0, "method": method}
    n = int(mask.sum())
    n_hit = int(((yt >= lo) & (yt <= hi))[mask].sum())
    fn = wilson_score_ci if method == "wilson" else clopper_pearson_ci
    p, ci_lo, ci_hi = fn(n_hit, n)
    return {
        "empirical": p,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "deviation": p - float(nominal),
        "mean_width": float(np.mean(hi[mask] - lo[mask])),
        "n_hits": n_hit,
        "n": n,
        "method": method,
    }


# ─── H. MASE — Mean Absolute Scaled Error (Hyndman & Koehler 2006) ──────


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006, IJF 22:679-688).

        MASE = MAE(forecast) / MAE(naive_seasonal_one_step_in_train)

    `seasonality` = 1 → naive 1-step (random walk) baseline
    `seasonality` = 52 → seasonal naive (same week last year)

    Interpretation:
      - MASE < 1: forecast beats naive baseline
      - MASE = 1: tied
      - MASE > 1: worse than naive

    Reference: scoringutils mase(), pyfts mase(), Hyndman 2006 IJF.
    Hub-canonical for time-series forecasting.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    mae_forecast = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
    if len(y_train) <= seasonality:
        return float("nan")
    naive_resid = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    naive_resid = naive_resid[np.isfinite(naive_resid)]
    if len(naive_resid) == 0:
        return float("nan")
    mae_naive = float(np.mean(naive_resid))
    if mae_naive < 1e-10:
        return float("nan")
    return mae_forecast / mae_naive


def median_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """MdAPE — median of |pct error|. Robust to outliers vs MAPE.

    Common alongside MAPE in the time-series literature
    (Hyndman & Koehler 2006).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    nz = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not nz.any():
        return float("nan")
    pe = np.abs((y_pred[nz] - y_true[nz]) / y_true[nz]) * 100.0
    return float(np.median(pe))


# ─── I. Additional forecasting metrics (Bias, Theil's U, log-score, etc.) ───


def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Error (signed Bias). Positive = systematic over-prediction.

    Reference: Hyndman & Koehler 2006. Often paired with MAE to detect
    direction of systematic error.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return float("nan")
    return float(np.mean(yp[mask] - yt[mask]))


def msle(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Squared Logarithmic Error.

    MSLE = mean( (log(y_pred + ε) - log(y_true + ε))^2 )

    Reference: Tofallis 2015 J. Operational Research Society 66:1352
    Useful when targets span orders of magnitude (ILI rate at low values).
    """
    yt = np.asarray(y_true, dtype=np.float64) + epsilon
    yp = np.asarray(y_pred, dtype=np.float64) + epsilon
    mask = (yt > 0) & (yp > 0) & np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return float("nan")
    return float(np.mean((np.log(yp[mask]) - np.log(yt[mask])) ** 2))


def theils_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Theil's U (U2) statistic — forecast accuracy vs naive 1-step.

    U = sqrt( mean( ((y_pred[t+1] - y_true[t+1]) / y_true[t])^2 ) ) /
        sqrt( mean( ((y_true[t+1] - y_true[t]) / y_true[t])^2 ) )

    U < 1: forecast beats naive
    U = 1: tied with naive (random walk)
    U > 1: worse than naive

    Reference: Theil 1966 'Applied Economic Forecasting'; Bliemel 1973 MS 19:444
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) < 2 or len(yp) != len(yt):
        return float("nan")
    yt_lag = yt[:-1]
    yt_now = yt[1:]
    yp_now = yp[1:]
    mask = (yt_lag != 0) & np.isfinite(yt_lag) & np.isfinite(yt_now) & np.isfinite(yp_now)
    if not mask.any():
        return float("nan")
    num = np.mean(((yp_now[mask] - yt_now[mask]) / yt_lag[mask]) ** 2)
    denom = np.mean(((yt_now[mask] - yt_lag[mask]) / yt_lag[mask]) ** 2)
    if denom < 1e-10:
        return float("nan")
    return float(np.sqrt(num) / np.sqrt(denom))


def log_score_gaussian(
    y_true: np.ndarray, y_pred: np.ndarray, sigma: float
) -> float:
    """Negative log-likelihood under Gaussian predictive distribution.

    log_score = -log( (2πσ²)^{-1/2} · exp(-(y-μ)²/(2σ²)) ) (mean over t)
              = 0.5·log(2πσ²) + (y-μ)²/(2σ²) (averaged)

    Lower is better. Reference: Gneiting & Raftery 2007 JASA 102:359
    Standard alongside CRPS in CDC FluSight.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    sig = max(float(sigma), 1e-6)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return float("nan")
    sq = (yt[mask] - yp[mask]) ** 2 / (2.0 * sig ** 2)
    return float(0.5 * np.log(2.0 * np.pi * sig ** 2) + np.mean(sq))


def crps_empirical(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> float:
    """Sample-based CRPS estimator (non-parametric).

    For predictive samples X_1, ..., X_M and observation y:
        CRPS(F, y) = E|X - y| - 0.5 · E|X - X'|

    Computed per observation, averaged. More accurate than Gaussian
    closed-form when the predictive distribution is non-Gaussian.

    Args:
      y_true: shape (n,)
      samples: shape (n, M) — M samples per observation

    Reference: Bracher 2021 §2.1; Krüger et al. 2021 J. Bus. Econ. Stat.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    sm = np.asarray(samples, dtype=np.float64)
    if sm.ndim == 1:
        sm = sm.reshape(-1, 1)
    n, M = sm.shape
    if n != len(yt) or M < 2:
        return float("nan")
    crps_per = np.zeros(n)
    for t in range(n):
        if not np.isfinite(yt[t]):
            crps_per[t] = np.nan
            continue
        Xt = sm[t][np.isfinite(sm[t])]
        if len(Xt) < 2:
            crps_per[t] = np.nan
            continue
        crps_per[t] = (np.mean(np.abs(Xt - yt[t]))
                       - 0.5 * np.mean(np.abs(Xt[:, None] - Xt[None, :])))
    return float(np.nanmean(crps_per))


def relative_skill_score(
    score_model: float, score_baseline: float, lower_is_better: bool = True,
) -> float:
    """Generic skill score: 1 - score_model/score_baseline.

    >0 = beats baseline. =0 tied. <0 worse than baseline.
    Used for relative-MAE, relative-WIS, relative-CRPS.

    Reference: Murphy 1973; Bracher 2021.
    """
    if abs(score_baseline) < 1e-10:
        return float("nan")
    if lower_is_better:
        return float(1.0 - score_model / score_baseline)
    return float(score_model / score_baseline - 1.0)


# ─── G. Hansen Superior Predictive Ability test ─────────────────────────


def hansen_spa_test(
    losses_per_model: dict[str, np.ndarray],
    benchmark_name: str,
    n_bootstrap: int = 5000,
    block_length: Optional[int] = None,
    seed: int = 0,
) -> dict[str, float]:
    """Hansen (2005) Superior Predictive Ability test.

    Tests whether a benchmark model is significantly inferior to ANY of K-1
    competitor models, while correctly controlling for the multiple-
    comparisons problem. Replaces naive best-of-K cherry-pick.

    H_0:  E[L_b - L_k] ≤ 0  for ALL k       (benchmark is best)
    H_1:  E[L_b - L_k] > 0  for SOME k       (some competitor is better)

    Reference: Hansen PR (2005). A test for superior predictive ability.
      J. Business & Econ. Statistics 23(4):365-380.

    Args:
      losses_per_model: {model_name: per-period loss array (length n)}
      benchmark_name: model treated as H_0 (e.g., "persistence" or current best)
      n_bootstrap: number of stationary-bootstrap resamples
      block_length: bootstrap block length (default = n^(1/3))
      seed: RNG seed

    Returns: {"spa_p_value": float, "consistent_p": float,
              "benchmark": str, "n_models": int, "n_periods": int}
    """
    rng = np.random.default_rng(seed)
    names = list(losses_per_model.keys())
    if benchmark_name not in names:
        raise ValueError(f"benchmark {benchmark_name} not in models")
    K = len(names)
    L = np.stack([losses_per_model[n] for n in names])  # (K, n)
    n = L.shape[1]
    if block_length is None:
        block_length = max(1, int(n ** (1.0 / 3.0)))
    bench_idx = names.index(benchmark_name)

    # d_k = L_bench - L_k    (positive ⇒ k is better than benchmark)
    d = L[bench_idx][None, :] - L
    d_bar = d.mean(axis=1)

    # Studentized statistic
    var_d = d.var(axis=1, ddof=1)
    var_d = np.maximum(var_d, 1e-10)
    se_d = np.sqrt(var_d / n)
    t_studentized = d_bar / se_d
    t_max = float(np.max(t_studentized))  # SPA test stat

    # Stationary bootstrap (Politis-Romano 1994) for null distribution
    boot_max = np.zeros(n_bootstrap, dtype=np.float64)
    p_block = 1.0 / block_length
    for b in range(n_bootstrap):
        # Generate stationary-bootstrap indices
        idx = np.zeros(n, dtype=np.int64)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            if rng.random() < p_block:
                idx[t] = rng.integers(n)
            else:
                idx[t] = (idx[t - 1] + 1) % n
        d_boot = d[:, idx]
        d_bar_boot = d_boot.mean(axis=1)
        # Re-center per Hansen's Z statistic (consistent variant uses thresholding)
        thresh = -np.sqrt(2.0 * var_d * np.log(np.log(n)) / n)
        recentred = d_bar_boot - np.where(d_bar > thresh, d_bar, 0.0)
        boot_max[b] = float(np.max(recentred / se_d))

    spa_p = float(np.mean(boot_max >= t_max))
    return {
        "spa_p_value": spa_p,
        "test_statistic": t_max,
        "benchmark": benchmark_name,
        "n_models": K,
        "n_periods": n,
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
        "interpretation": (
            f"H_0: {benchmark_name} is best. "
            f"p={spa_p:.4f} → "
            + ("REJECT H_0 (some competitor significantly better)"
               if spa_p < 0.05
               else "fail to reject (benchmark may be best)")
        ),
    }



# ─── J. WIS decomposition into sharpness / underpred / overpred (Bracher 2021) ─


def weighted_interval_score_components(
    y_test,
    y_pred,
    sigma: float,
    alphas: tuple = FLUSIGHT_ALPHAS,
) -> dict:
    """WIS 3-component decomposition (Gaussian-σ closed-form) per Bracher 2021.

    DEPRECATED for primary ILI evaluation as of 2026-05-26 S8 — use
    `weighted_interval_score_components_empirical()` instead.

    Decomposes WIS = sharpness + underprediction_penalty + overprediction_penalty
    (averaged over all K alpha levels and the full test set).

    Reference: Bracher J et al. (2021). Evaluating epidemic forecasts in an
      interval format. PLOS Comp Bio 17(2):e1008618, eq. (3)-(4).

    Args:
        y_test:  observed ILI rate (n,)
        y_pred:  point forecast (n,)
        sigma:   residual std used to build symmetric Gaussian PI
        alphas:  K prediction-interval levels (FLUSIGHT_ALPHAS default = K=11)

    Returns:
        dict with keys:
          wis_sharpness    -- mean PI half-width contribution (lower = sharper)
          wis_underpred    -- mean penalty for observed < lower bound
          wis_overpred     -- mean penalty for observed > upper bound
          wis_total_decomp -- sum of three components (cross-check vs wis)

    Performance: O(n*K) time, negligible memory.
    Side effects: none (pure function). NaN-safe -- returns NaN on failure.
    Caller responsibility: sanitize_predictions(y_pred) before calling.
    """
    try:
        from scipy.stats import norm as _norm
        yt = np.asarray(y_test, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        sig = max(float(sigma), 1e-6)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if not mask.any():
            return {
                "wis_sharpness": float("nan"), "wis_underpred": float("nan"),
                "wis_overpred": float("nan"),  "wis_total_decomp": float("nan"),
            }
        a, p = yt[mask], yp[mask]
        K = len(alphas)
        K_half = K + 0.5          # Bracher 2021 denominator (K + median weight 0.5)
        sharp_sum = under_sum = over_sum = 0.0
        for alpha in alphas:
            z = float(_norm.ppf(1.0 - alpha / 2.0))
            lo = p - z * sig
            hi = p + z * sig
            # Bracher 2021 eq.(4): IS_alpha = (u-l) + (2/alpha)*[max(l-y,0)+max(y-u,0)]
            # WIS = 1/(K+0.5) * [|y-m|/2 + sum_k alpha_k/2 * IS_{alpha_k}]
            # Expanding: sharpness contribution = alpha/2 * (u-l)
            #            underpred contribution = max(l-y, 0)   [alpha/2 * 2/alpha cancels]
            #            overpred  contribution = max(y-u, 0)   [same cancellation]
            sharp_sum += float(np.mean((alpha / 2.0) * (hi - lo)))
            under_sum += float(np.mean(np.maximum(0.0, lo - a)))   # no 2/alpha factor
            over_sum  += float(np.mean(np.maximum(0.0, a  - hi)))  # no 2/alpha factor
        # Normalise by K+0.5; add median component |y-m|/(K+0.5)
        # NOTE: Bracher 2021 eq.(4) writes 0.5*|y-m| in the numerator, but
        # diagnostics.weighted_interval_score() (the SSOT scalar WIS used in the
        # 4-criteria filter) omits that 0.5 and computes |y-m|/(K+0.5) directly.
        # We match diagnostics.py here so wis_total_decomp ≈ wis (not ≈ wis/2 + bias).
        # For Gaussian forecast, median = point prediction = y_pred.
        sharp_v  = sharp_sum / K_half
        under_v  = under_sum / K_half
        over_v   = over_sum  / K_half
        median_v = float(np.mean(np.abs(a - p))) / K_half   # matches diagnostics.py
        total_v  = median_v + sharp_v + under_v + over_v    # ≈ wis from diagnostics
        return {
            "wis_sharpness":    round(sharp_v,  6),
            "wis_underpred":    round(under_v,  6),
            "wis_overpred":     round(over_v,   6),
            "wis_total_decomp": round(total_v,  6),
        }
    except Exception:
        return {
            "wis_sharpness": float("nan"), "wis_underpred": float("nan"),
            "wis_overpred": float("nan"),  "wis_total_decomp": float("nan"),
        }


def weighted_interval_score_components_empirical(
    y_test,
    y_pred,
    residuals,
    alphas: tuple = FLUSIGHT_ALPHAS,
) -> dict:
    """WIS 3-component decomposition with empirical residual quantiles.

    Same Bracher 2021 decomposition as `weighted_interval_score_components()`,
    but PI half-widths come from empirical |residuals| quantiles (Lei 2018
    split-conformal) instead of Gaussian z·σ.

    Added 2026-05-26 (S8 Tier C) — Codex/Gemini consensus migration.

    Args:
        y_test:     observed (n,)
        y_pred:     point forecast (n,)
        residuals:  OOF or train residuals for half-width calibration
        alphas:     K PI levels (FLUSIGHT_ALPHAS default = K=11)

    Returns:
        dict {wis_sharpness, wis_underpred, wis_overpred, wis_total_decomp}

    Side effects: none. NaN-safe.
    Caller responsibility: residuals must be in-sample (not from y_test).
    """
    try:
        yt = np.asarray(y_test, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        res = np.asarray(residuals, dtype=np.float64)
        res = res[np.isfinite(res)]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if not mask.any() or len(res) < 2:
            return {
                "wis_sharpness": float("nan"), "wis_underpred": float("nan"),
                "wis_overpred": float("nan"),  "wis_total_decomp": float("nan"),
            }
        a, p = yt[mask], yp[mask]
        K = len(alphas)
        K_half = K + 0.5
        q_alpha = k11_pi_widths_from_residuals(np.abs(res), tuple(alphas))
        sharp_sum = under_sum = over_sum = 0.0
        for alpha in alphas:
            q = q_alpha.get(float(alpha), float("inf"))
            if not np.isfinite(q):
                continue
            lo = p - q
            hi = p + q
            # Same decomposition formula as Gaussian path, but with q empirical
            sharp_sum += float(np.mean((alpha / 2.0) * (hi - lo)))  # = alpha * q
            under_sum += float(np.mean(np.maximum(0.0, lo - a)))
            over_sum  += float(np.mean(np.maximum(0.0, a  - hi)))
        sharp_v  = sharp_sum / K_half
        under_v  = under_sum / K_half
        over_v   = over_sum  / K_half
        median_v = float(np.mean(np.abs(a - p))) / K_half
        total_v  = median_v + sharp_v + under_v + over_v
        return {
            "wis_sharpness":    round(sharp_v,  6),
            "wis_underpred":    round(under_v,  6),
            "wis_overpred":     round(over_v,   6),
            "wis_total_decomp": round(total_v,  6),
        }
    except Exception:
        return {
            "wis_sharpness": float("nan"), "wis_underpred": float("nan"),
            "wis_overpred": float("nan"),  "wis_total_decomp": float("nan"),
        }
