"""
Probabilistic forecast diagnostics and validation.

Contains:
 - pit_values: Probability Integral Transform for calibration assessment
 - calibration_check: Coverage at nominal levels (Diebold-Mariano style)
 - weighted_interval_score: WIS for probabilistic forecasts (Bracher et al. 2021)
 - model_confidence_set: MCS Hansen, Lunde & Nason (2011)
 - coverage_gap_by_regime: PI coverage broken out by pre/during/post-COVID
 (Phase C1 — pair with phase6 conformal PI to expose regime shifts)
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple, Sequence, Union


def pit_values(y_true: np.ndarray, y_pred: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute Probability Integral Transform (PIT) values.

    Assumes Normal(y_pred, sigma²) predictive distribution.
    Well-calibrated forecast -> PIT ~ Uniform(0,1)

    Args:
        y_true: Observed values
        y_pred: Predicted means
        sigma: Predicted standard deviation (scalar or array)

    Returns:
        pit: PIT values on [0, 1]
    """
    z = (y_true - y_pred) / np.maximum(sigma, 1e-10)
    return stats.norm.cdf(z)


def calibration_check(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: float,
    levels: Optional[List[float]] = None
) -> List[Dict[str, float]]:
    """
    Check empirical coverage at multiple nominal confidence levels.

    Args:
        y_true: Observed values
        y_pred: Predicted means
        sigma: Predicted standard deviation (scalar or array)
        levels: Nominal confidence levels. Default: [0.10, 0.20, ..., 0.95]

    Returns:
        List of dicts with keys: nominal, empirical, deviation
    """
    if levels is None:
        levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    results = []
    for alpha in levels:
        z = stats.norm.ppf(1 - (1 - alpha) / 2)
        lower = y_pred - z * sigma
        upper = y_pred + z * sigma
        covered = np.mean((y_true >= lower) & (y_true <= upper))
        results.append({
            "nominal": alpha,
            "empirical": float(covered),
            "deviation": float(covered - alpha),
        })
    return results


def weighted_interval_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: float,
    alphas: Optional[List[float]] = None
) -> np.ndarray:
    """
    Weighted Interval Score (Bracher et al. 2021) — Gaussian-σ closed-form.

    DEPRECATED for primary ILI evaluation as of 2026-05-26 S8 — use
    `weighted_interval_score_empirical()` instead (no Gaussian assumption).
    Kept for back-compat with phase12 4-criteria filter pre-migration.

    WIS = |y - median| / (K+0.5) + (1/(K+0.5)) * sum_k [ alpha_k/2 * width_k + penalty ]
    where penalty = max(L-y, 0) + max(y-U, 0), L = pred - z·σ, U = pred + z·σ.

    Args:
        y_true: Observed values
        y_pred: Predicted means (median)
        sigma: Predicted standard deviation (Gaussian PI)
        alphas: Prediction interval levels

    Returns:
        wis: WIS value at each time point
    """
    if alphas is None:
        alphas = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    K = len(alphas)
    n = len(y_true)
    wis_total = np.zeros(n)

    # Dispersion (absolute error of median)
    wis_total += np.abs(y_true - y_pred) / (K + 0.5)

    for alpha in alphas:
        z = stats.norm.ppf(1 - alpha / 2)
        lower = y_pred - z * sigma
        upper = y_pred + z * sigma
        width = upper - lower

        # Overprediction
        overpred = np.maximum(lower - y_true, 0)
        # Underprediction
        underpred = np.maximum(y_true - upper, 0)

        interval_score = (alpha / 2) * width + overpred + underpred
        wis_total += interval_score / (K + 0.5)

    return wis_total


def weighted_interval_score_empirical(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    alphas: Optional[List[float]] = None,
) -> np.ndarray:
    """WIS using empirical residual quantiles (Lei 2018 / Vovk 2005 split-conformal).

    Same Bracher 2021 formula as `weighted_interval_score`, but PI half-widths
    come from empirical |residuals| order statistics instead of Gaussian
    Φ⁻¹(1-α/2)·σ. Methodologically defensible for non-Gaussian residual
    distributions (right-skewed ILI, low-count weeks).

    Added 2026-05-26 (S8 Tier C) — Codex/Gemini consensus migration.

    Args:
        y_true: observed (n,)
        y_pred: point forecast (n,)
        residuals: OOF or train residuals (≥2 finite values required)
                   for half-width calibration. Use `y_train - pred_train` (not
                   `y_test - pred_test` — that would leak).
        alphas: K PI levels (default K=11 FluSight); same defaults as Gaussian.

    Returns:
        wis: WIS value at each time point (n,)

    Reference:
      - Bracher et al. (2021). PLOS Comp Bio 17(2):e1008618 eq.(3)-(4)
      - Lei et al. (2018). JASA 113(523):1094-1111 (split-conformal)
      - Vovk et al. (2005). Algorithmic Learning in a Random World

    Caller responsibility: residuals must be in-sample (no leakage from
    y_test). For 4-criteria filter or phase11/12, use OOF residuals.
    """
    from .hub_metrics import k11_pi_widths_from_residuals

    if alphas is None:
        alphas = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    K = len(alphas)
    n = len(y_true)
    wis_total = np.zeros(n)

    # Median dispersion |y - m| / (K+0.5) — same as Gaussian path
    wis_total += np.abs(y_true - y_pred) / (K + 0.5)

    # Empirical residual quantile half-widths (Bracher 2021 + Lei 2018)
    q_alpha = k11_pi_widths_from_residuals(np.abs(residuals), tuple(alphas))

    for alpha in alphas:
        q = q_alpha.get(float(alpha), float("inf"))
        if not np.isfinite(q):
            wis_total += float("inf")
            continue
        lower = y_pred - q
        upper = y_pred + q
        width = 2.0 * q  # symmetric empirical PI

        overpred = np.maximum(lower - y_true, 0)
        underpred = np.maximum(y_true - upper, 0)
        interval_score = (alpha / 2.0) * width + overpred + underpred
        wis_total += interval_score / (K + 0.5)

    return wis_total


def model_confidence_set(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    alpha: float = 0.10,
    n_boot: int = 1000
) -> Dict[str, Any]:
    """
    Model Confidence Set (Hansen, Lunde & Nason 2011, Econometrica).

    Identifies the set of statistically equivalent best-performing models.
    Uses T_max statistic with bootstrap p-values.

    Args:
        y_true: Observed values
        predictions: Dict mapping model names to predictions
        alpha: Significance level for elimination (default 0.10)
        n_boot: Number of bootstrap samples

    Returns:
        Dictionary with keys:
          - mcs_models: List of surviving models
          - mcs_size: Number of models in MCS
          - alpha: Significance level used
          - eliminated: List of eliminated models with iteration info
    """
    model_names = list(predictions.keys())
    losses = {}
    for name in model_names:
        p = predictions[name][:len(y_true)]
        losses[name] = (y_true[:len(p)] - p) ** 2  # squared error loss

    surviving = list(model_names)
    eliminated = []

    for iteration in range(len(model_names)):
        if len(surviving) <= 1:
            break

        # Compute pairwise loss differentials
        n_obs = min(len(losses[s]) for s in surviving)
        loss_matrix = np.column_stack([losses[s][:n_obs] for s in surviving])
        mean_losses = loss_matrix.mean(axis=0)

        # T_max: max over j of t_j where t_j tests H0: E[d_bar_j] = 0
        avg_loss = mean_losses.mean()
        d_bar = mean_losses - avg_loss

        # Bootstrap variance
        t_stats = np.zeros(len(surviving))
        for j in range(len(surviving)):
            diffs = loss_matrix[:, j] - loss_matrix.mean(axis=1)
            boot_means = np.zeros(n_boot)
            for b in range(n_boot):
                idx = np.random.choice(n_obs, n_obs, replace=True)
                boot_means[b] = diffs[idx].mean()
            se = np.std(boot_means)
            t_stats[j] = d_bar[j] / max(se, 1e-10)

        t_max = np.max(t_stats)
        worst_model_idx = np.argmax(t_stats)

        # Bootstrap p-value for T_max
        boot_t_max = np.zeros(n_boot)
        for b in range(n_boot):
            idx = np.random.choice(n_obs, n_obs, replace=True)
            boot_loss = loss_matrix[idx]
            boot_mean = boot_loss.mean(axis=0)
            boot_avg = boot_mean.mean()
            boot_d = boot_mean - boot_avg
            boot_se = np.zeros(len(surviving))
            for j in range(len(surviving)):
                boot_diffs = boot_loss[:, j] - boot_loss.mean(axis=1)
                boot_se[j] = max(np.std(boot_diffs), 1e-10)
            boot_t = boot_d / boot_se
            boot_t_max[b] = np.max(boot_t)

        p_value = float(np.mean(boot_t_max >= t_max))

        if p_value < alpha:
            worst_name = surviving[worst_model_idx]
            eliminated.append({
                "model": worst_name,
                "t_stat": float(t_stats[worst_model_idx]),
                "p_value": p_value,
                "iteration": iteration + 1,
            })
            surviving.pop(worst_model_idx)
        else:
            break

    return {
        "mcs_models": surviving,
        "mcs_size": len(surviving),
        "alpha": alpha,
        "eliminated": eliminated,
    }


# ══════════════════════════════════════════════════════════════════════════
# Phase C1 — Coverage gap by COVID regime
#
# PI audits that pool across the 2020 NPI structural break hide regime-
# specific under/overcoverage. Mirrors `phase9_dm_test._build_regime_masks`
# so DM tests and coverage gaps reference the same pre/during/post slices.
# ══════════════════════════════════════════════════════════════════════════

_COVID_START = "2020-03-01"
_COVID_END = "2023-01-01"


def _regime_masks(
    n: int,
    dates: Optional[Sequence] = None,
    boundaries: Tuple[str, str] = (_COVID_START, _COVID_END),
) -> Dict[str, np.ndarray]:
    """Pre/during/post-COVID masks over an n-length series.

    Calendar-accurate when ``dates`` is supplied (datetime64-coercible,
    aligned to y_true). Otherwise falls back to the 47/36/17 proportional
    index split used by ``phase9_dm_test._build_regime_masks``.
    """
    import warnings as _warnings
    out: Dict[str, np.ndarray] = {}
    if dates is not None and len(dates) == n:
        d = np.asarray(dates, dtype="datetime64[D]")
        b0 = np.datetime64(boundaries[0])
        b1 = np.datetime64(boundaries[1])
        pre = d < b0
        post = d >= b1
        during = (~pre) & (~post)
        out["pre_covid"] = pre
        out["during_covid"] = during
        out["post_covid"] = post
    else:
        # Round 3 audit G4 (2026-05-26): silent fallback warning.
        # When calendar dates are unavailable, COVID regime boundaries are
        # ESTIMATED via fixed 47/36/17 proportional index split. This is
        # a heuristic — actual COVID period in the input series may differ.
        # Reviewer-visible disclosure: regimes are NOT calendar-validated.
        _warnings.warn(
            "_regime_masks: `dates` missing or length mismatch — falling back "
            "to fixed 47/36/17 proportional index split (heuristic, NOT "
            "calendar-validated). For SCI publication, supply `dates` argument "
            "from phase1['dates'][test_start:test_end] to use _COVID_START "
            f"({_COVID_START}) / _COVID_END ({_COVID_END}) boundaries.",
            UserWarning, stacklevel=2,
        )
        i1 = int(round(n * 0.47))
        i2 = int(round(n * 0.83))
        pre = np.zeros(n, dtype=bool);    pre[:i1] = True
        during = np.zeros(n, dtype=bool); during[i1:i2] = True
        post = np.zeros(n, dtype=bool);   post[i2:] = True
        out["pre_covid"] = pre
        out["during_covid"] = during
        out["post_covid"] = post
    out["global"] = np.ones(n, dtype=bool)
    return out


def coverage_gap_by_regime(
    y_true: Union[np.ndarray, Sequence[float]],
    lower: Union[np.ndarray, Sequence[float]],
    upper: Union[np.ndarray, Sequence[float]],
    *,
    nominal: float,
    dates: Optional[Sequence] = None,
    boundaries: Tuple[str, str] = (_COVID_START, _COVID_END),
) -> List[Dict[str, Any]]:
    """Empirical PI coverage and mean width, split by COVID regime.

    Args:
        y_true: observed values
        lower, upper: PI endpoints (same length as y_true)
        nominal: target coverage (e.g. 0.90); reported as-is for the gap
        dates: optional datetime64-coercible vector aligned to y_true
        boundaries: (covid_start, covid_end). Defaults to 2020-03-01 / 2023-01-01.

    Returns:
        One dict per regime (pre_covid, during_covid, post_covid, global)
        with keys: regime, n, coverage, gap (=coverage-nominal), mean_width.
        Regimes with n=0 are skipped.
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (len(y) == len(lo) == len(hi)):
        raise ValueError(
            f"length mismatch: y={len(y)} lower={len(lo)} upper={len(hi)}"
        )
    n = len(y)
    if n == 0:
        return []
    masks = _regime_masks(n, dates=dates, boundaries=boundaries)
    covered = ((y >= lo) & (y <= hi)).astype(float)
    widths = hi - lo
    rows: List[Dict[str, Any]] = []
    for regime in ("pre_covid", "during_covid", "post_covid", "global"):
        m = masks[regime]
        k = int(m.sum())
        if k == 0:
            continue
        cov = float(covered[m].mean())
        width = float(widths[m].mean())
        rows.append({
            "regime": regime,
            "n": k,
            "coverage": cov,
            "gap": cov - float(nominal),
            "mean_width": width,
            "nominal": float(nominal),
        })
    return rows


def coverage_gap_table(
    y_true: Union[np.ndarray, Sequence[float]],
    lower_by_level: Dict[float, np.ndarray],
    upper_by_level: Dict[float, np.ndarray],
    *,
    dates: Optional[Sequence] = None,
    boundaries: Tuple[str, str] = (_COVID_START, _COVID_END),
) -> List[Dict[str, Any]]:
    """Stack `coverage_gap_by_regime` across a set of nominal levels.

    Returns a flat list of {level, regime, n, coverage, gap, mean_width}
    suitable for a long-form plot or markdown table.
    """
    levels = sorted(set(lower_by_level) & set(upper_by_level))
    out: List[Dict[str, Any]] = []
    for lvl in levels:
        for row in coverage_gap_by_regime(
            y_true,
            lower_by_level[lvl],
            upper_by_level[lvl],
            nominal=lvl,
            dates=dates,
            boundaries=boundaries,
        ):
            out.append({"level": float(lvl), **row})
    return out
