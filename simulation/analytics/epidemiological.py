"""
Epidemiological estimation and detection methods.

Contains:
  - estimate_rt_cori: Bayesian Rt estimation via Cori et al. (2013)
  - serfling_regression: Sinusoidal baseline + excess detection (Serfling 1963)
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional


def estimate_rt_cori(
    incidence: np.ndarray,
    serial_interval_mean: float = 3.6,
    serial_interval_sd: float = 1.6,
    window: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cori et al. (2013, Am J Epidemiol): Bayesian Rt estimation.

    Simplified version using ratio estimator with Gamma prior.
    Serial interval typical for ILI: 3.6 days (range 1-7).

    Args:
        incidence: Weekly incidence counts
        serial_interval_mean: Mean of serial interval (days)
        serial_interval_sd: SD of serial interval (days)
        window: Not used in current implementation, kept for compatibility

    Returns:
        rt: Point estimates of Rt
        rt_lower: Lower credible interval (2.5%)
        rt_upper: Upper credible interval (97.5%)
    """
    n = len(incidence)

    # Discretized serial interval (weekly)
    # For weekly data, SI weights: w[0]=prob in week 0, w[1]=prob in week 1, etc.
    max_si = 4  # max 4 weeks
    si_weights = np.zeros(max_si + 1)
    for k in range(1, max_si + 1):
        # Gamma distribution for serial interval
        shape = (serial_interval_mean / serial_interval_sd) ** 2
        scale = serial_interval_sd ** 2 / serial_interval_mean
        si_weights[k] = (
            stats.gamma.cdf(k * 7, a=shape, scale=scale * 7)
            - stats.gamma.cdf((k - 1) * 7, a=shape, scale=scale * 7)
        )
    si_weights = si_weights / si_weights.sum()

    rt = np.full(n, np.nan)
    rt_lower = np.full(n, np.nan)
    rt_upper = np.full(n, np.nan)

    for t in range(max_si, n):
        # Lambda_t = sum over s of I_{t-s} * w_s
        lambda_t = 0
        for s in range(1, min(max_si + 1, t + 1)):
            lambda_t += incidence[t - s] * si_weights[s]

        if lambda_t > 0.1:  # minimum denominator
            # Posterior: Gamma(a + I_t, 1/(1/b + lambda_t))
            a_prior, b_prior = 1, 5  # weakly informative
            a_post = a_prior + incidence[t]
            b_post = 1 / (1 / b_prior + lambda_t)
            rt[t] = a_post * b_post
            rt_lower[t] = stats.gamma.ppf(0.025, a_post, scale=b_post)
            rt_upper[t] = stats.gamma.ppf(0.975, a_post, scale=b_post)

    return rt, rt_lower, rt_upper


def serfling_regression(
    y: np.ndarray,
    weeks_index: np.ndarray,
    fit_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Serfling RE (1963, Am J Public Health): sinusoidal baseline regression.

    Fits: y = b0 + b1*t + b2*sin(2πt/52) + b3*cos(2πt/52) + b4*sin(4πt/52) + b5*cos(4πt/52)

    Args:
        y: Time series of ILI rates or counts
        weeks_index: Week indices (typically 0 to n-1)
        fit_mask: Boolean mask, True for non-epidemic weeks used for fitting.
                  If None, uses baseline below 75th percentile.

    Returns:
        baseline: Fitted baseline level at each time point
        upper_threshold: 95th percentile threshold (baseline + 1.645*sigma)
        excess: Excess ILI = max(y - threshold, 0)
    """
    t = np.arange(len(y), dtype=float)

    # Design matrix
    X = np.column_stack([
        np.ones(len(y)),
        t,
        np.sin(2 * np.pi * t / 52),
        np.cos(2 * np.pi * t / 52),
        np.sin(4 * np.pi * t / 52),
        np.cos(4 * np.pi * t / 52),
    ])

    if fit_mask is None:
        # Use non-epidemic weeks: below 75th percentile
        fit_mask = y <= np.percentile(y, 75)

    # Fit on non-epidemic weeks
    X_fit = X[fit_mask]
    y_fit = y[fit_mask]

    # OLS
    try:
        beta, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), np.nan)

    baseline = X @ beta
    residuals = y_fit - X_fit @ beta
    sigma = np.std(residuals)

    # Upper threshold (95th percentile)
    upper_threshold = baseline + 1.645 * sigma

    # Excess ILI
    excess = np.maximum(y - upper_threshold, 0)

    return baseline, upper_threshold, excess
