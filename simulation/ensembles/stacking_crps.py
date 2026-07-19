"""
Stacking on CRPS (Yao, Vehtari, Simpson, Gelman 2018) — proper-score
ensemble combination.
====================================================================

Bayesian Model Averaging (BMA) is **flawed in M-open settings**: under model
misspecification (which always holds in practice), BMA weights converge to a
single best model rather than a balanced average. Yao et al. 2018 (Bayesian
Analysis 13(3):917-1007) propose **stacking on a proper scoring rule** which
chooses ensemble weights `w` to minimize the held-out CRPS / log score:

    minimize_w  CRPS( Σ_k w_k · F_k, y_obs )    s.t.  w_k ≥ 0, Σ w_k = 1

For Gaussian predictive distributions F_k = N(μ_k, σ_k²), the linear-pool
predictive mean is Σ w_k μ_k and the linear-pool variance is
Σ w_k (σ_k² + μ_k²) − (Σ w_k μ_k)². CRPS for the linear pool can be computed
in closed form.

Reference:
  Yao Y, Vehtari A, Simpson D, Gelman A (2018). Using stacking to average
  Bayesian predictive distributions (with discussion). Bayesian Analysis
  13(3):917-1007. doi:10.1214/17-BA1091

Companion: Wadsworth & Niemi (2025) arXiv:2509.04203 — Gibbs-posterior
stacking on CRPS, validated on FluSight 2023-24.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)


def crps_gaussian_scalar(y: float, mu: float, sigma: float) -> float:
    """Scalar Gaussian CRPS (Gneiting & Raftery 2007 JASA Eq. 5)."""
    from scipy.stats import norm
    sigma = max(float(sigma), 1e-8)
    z = (y - mu) / sigma
    return float(sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)))


def stacking_weights_crps(
    predictions_per_model: dict[str, np.ndarray],
    observations: np.ndarray,
    sigma_per_model: Optional[dict[str, float]] = None,
    method: str = "SLSQP",
    initial_weights: Optional[np.ndarray] = None,
    n_starts: int = 5,
) -> dict:
    """Find ensemble weights minimizing held-out CRPS.

    Args:
      predictions_per_model: {model_name: pred_array (length n)}
      observations: ground truth array (length n)
      sigma_per_model: {model_name: float} predictive σ. If None, use std
        of in-sample residuals across all models.
      method: scipy.optimize method (default SLSQP for sum-to-1 + nonneg).
      n_starts: random restarts to escape local minima (CRPS surface is
        convex in w under linear pooling, but starts help with numerics).

    Returns: {weights, crps_train, model_names, sigma_used}
    """
    names = list(predictions_per_model.keys())
    K = len(names)
    if K == 0:
        raise ValueError("at least one model required")

    P = np.stack([np.asarray(predictions_per_model[n], dtype=np.float64)
                  for n in names])  # (K, n)
    y = np.asarray(observations, dtype=np.float64)
    n = len(y)
    if P.shape[1] != n:
        raise ValueError(f"prediction length {P.shape[1]} != obs length {n}")

    # σ per model — fall back to residual std if not provided
    if sigma_per_model is None:
        sigmas = np.array([float(np.std(P[k] - y)) for k in range(K)])
        sigmas = np.maximum(sigmas, 1e-3)
    else:
        sigmas = np.array([sigma_per_model.get(n, 1.0) for n in names])

    def total_crps(weights: np.ndarray) -> float:
        # Linear pool: μ_pool[t] = Σ w_k μ_k[t]
        # σ_pool[t]² = Σ w_k σ_k² (using approximation; exact is mixture variance)
        mu_pool = (weights[:, None] * P).sum(axis=0)
        sigma_pool = float(np.sqrt(np.sum(weights * sigmas ** 2)))
        crps = sum(crps_gaussian_scalar(y[t], mu_pool[t], sigma_pool)
                   for t in range(n))
        return crps / n

    cons = (
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: w},  # w ≥ 0
    )
    bounds = [(0.0, 1.0)] * K

    best = None
    rng = np.random.default_rng(seed=0)
    starts = []
    if initial_weights is not None:
        starts.append(initial_weights / max(float(initial_weights.sum()), 1e-8))
    starts.append(np.full(K, 1.0 / K))  # equal-weight start
    for _ in range(max(0, n_starts - len(starts))):
        w0 = rng.dirichlet(np.ones(K))
        starts.append(w0)

    for w0 in starts:
        try:
            res = minimize(total_crps, w0, method=method,
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 200, "ftol": 1e-8})
            if best is None or (res.success and res.fun < best.fun):
                best = res
        except Exception as e:
            log.debug(f"stacking start failed: {e}")
            continue

    if best is None:
        raise RuntimeError("all stacking optimization starts failed")
    w = np.clip(best.x, 0.0, None)
    w = w / max(float(w.sum()), 1e-8)
    log.info(f"  [stacking-CRPS] weights: " +
             ", ".join(f"{n}={w[i]:.3f}" for i, n in enumerate(names)))
    return {
        "weights": {names[i]: float(w[i]) for i in range(K)},
        "weight_array": w,
        "crps_train": float(best.fun),
        "model_names": names,
        "sigma_used": dict(zip(names, sigmas.tolist())),
        "method": "stacking_on_CRPS_Yao_2018",
    }


def equally_weighted_median_ensemble(
    predictions_per_model: dict[str, np.ndarray]
) -> np.ndarray:
    """Sherratt 2023 eLife showed median ensemble outperforms mean and
    trained variants for COVID-19 forecasts. Trivial baseline.
    """
    P = np.stack([np.asarray(predictions_per_model[n], dtype=np.float64)
                  for n in predictions_per_model])
    return np.median(P, axis=0)


def predict_with_stacking(
    weights: dict[str, float],
    predictions_per_model: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply learned stacking weights to a new prediction set."""
    out = np.zeros(next(iter(predictions_per_model.values())).shape, dtype=np.float64)
    for name, w in weights.items():
        if name in predictions_per_model:
            out += float(w) * np.asarray(predictions_per_model[name], dtype=np.float64)
    return out
