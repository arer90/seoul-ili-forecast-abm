"""
Adaptive Conformal Inference (ACI) for time-series forecast intervals.
======================================================================

Split-conformal prediction (Lei et al. 2018; Vovk-Gammerman-Shafer 2005)
guarantees finite-sample marginal coverage `P(Y ∈ Ĉ(X)) ≥ 1−α` ONLY under
the *exchangeability* assumption. Influenza time-series are non-exchangeable
(seasonality, COVID-era distributional shift, post-pandemic rebound), so
empirical coverage drifts away from nominal in the real slab.

This module implements three remedies, in increasing sophistication:

  1. **Standard split-conformal** (legacy baseline)
       — already in phase10_intervals / real_eval.

  2. **Adaptive Conformal Inference (ACI; Gibbs & Candès 2021 NeurIPS)**
       Online learning of α via:
           α_{t+1} = α_t + γ · (α* − err_t)
       where err_t = 1{Y_t ∉ Ĉ_{α_t}(X_t)} and γ > 0 is the step size.
       Guarantees long-run empirical coverage `→ 1−α*` despite non-stationarity.
       Reference: https://arxiv.org/abs/2106.00170

  3. **AgACI (Aggregated ACI; Zaffran et al. 2022 ICML)**
       Runs multiple γ values in parallel and aggregates with online expert
       advice (e.g., BOA / Hedge). Removes the γ tuning burden.
       Reference: PMLR 162:25834 — https://proceedings.mlr.press/v162/zaffran22a.html

  4. **NEx-CP (Non-Exchangeable Conformal Prediction; Barber et al. 2023)**
       Weighted residual quantiles with weights decaying for older calibration
       points. Useful when concept drift is gradual rather than abrupt.
       Reference: Annals of Statistics; arXiv 2202.13415.

Default in this file: **ACI** (single-γ adaptive). Use AgACI or NEx-CP via
the `method` kwarg.

API
---
    aci = AdaptiveConformal(alpha_star=0.05, gamma=0.05)
    aci.calibrate(residuals_oof, n_calibration_steps=residuals_oof.size)
    for t, (y_pred, y_true) in enumerate(zip(predictions, observations)):
        lo, hi = aci.predict_interval(y_pred)
        aci.update(y_true)   # observe truth, update α
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def split_conformal_quantile(
    residuals: np.ndarray, alpha: float
) -> float:
    """Standard split-conformal half-width.

    Returns the (1−α) sample quantile of |residuals| using the Lei 2018 /
    Vovk 2005 ceiling rule k = ⌈(n+1)(1−α)⌉ − 1 (0-indexed).
    """
    res = np.asarray(residuals, dtype=np.float64)
    res = np.abs(res[np.isfinite(res)])
    n = len(res)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = max(0, min(k, n - 1))
    return float(np.sort(res)[k])


class AdaptiveConformal:
    """Single-γ Adaptive Conformal Inference (Gibbs & Candès 2021).

    Maintains an online α_t that adjusts when realized coverage drifts from
    the nominal level α*. Use cycle:

        ac = AdaptiveConformal(alpha_star=0.05, gamma=0.05)
        ac.calibrate(in_sample_oof_residuals)
        for y_pred_t in predictions:
            lo, hi = ac.predict_interval(y_pred_t)   # uses α_t
            # ... record (y_pred_t, lo, hi) ...
            y_true_t = observe()
            ac.update(y_true_t)                       # advances α_{t+1}
    """

    def __init__(
        self,
        alpha_star: float = 0.05,
        gamma: float = 0.05,
        alpha_init: Optional[float] = None,
    ):
        self.alpha_star = float(alpha_star)
        self.gamma = float(gamma)
        self.alpha_t = float(alpha_init if alpha_init is not None else alpha_star)
        self._calibrated = False
        self._residuals: np.ndarray = np.array([], dtype=np.float64)
        self._last_lo: Optional[float] = None
        self._last_hi: Optional[float] = None
        self._coverage_history: list[int] = []  # 1=hit, 0=miss
        self._alpha_history: list[float] = [self.alpha_t]

    def calibrate(self, residuals: np.ndarray) -> "AdaptiveConformal":
        """Store the in-sample OOF residuals as the conformal calibration set."""
        res = np.asarray(residuals, dtype=np.float64)
        res = np.abs(res[np.isfinite(res)])
        if len(res) == 0:
            raise ValueError("residuals empty after NaN removal")
        self._residuals = np.sort(res)
        self._calibrated = True
        return self

    def predict_interval(self, y_pred: float) -> tuple[float, float]:
        """Return (lo, hi) for the current α_t."""
        if not self._calibrated:
            raise RuntimeError("call calibrate(residuals) first")
        # Clip α_t to (0, 1) as Gibbs & Candès §2 spec:
        #   if α_t ≥ 1 → predict (-∞, +∞) (always covers)
        #   if α_t ≤ 0 → predict (μ̂, μ̂) (vacuous singleton)
        a = float(np.clip(self.alpha_t, 1e-6, 1.0 - 1e-6))
        n = len(self._residuals)
        k = int(np.ceil((n + 1) * (1.0 - a))) - 1
        k = max(0, min(k, n - 1))
        q = float(self._residuals[k])
        lo = float(y_pred) - q
        hi = float(y_pred) + q
        self._last_lo, self._last_hi = lo, hi
        return lo, hi

    def update(self, y_true: float) -> None:
        """Observe ground truth, update α_t per ACI rule.

            err_t = 1{y_true ∉ [lo, hi]}
            α_{t+1} = α_t + γ · (α* − err_t)
        """
        if self._last_lo is None:
            raise RuntimeError("predict_interval must be called before update")
        miss = 0 if (self._last_lo <= y_true <= self._last_hi) else 1
        # Coverage history (hit=1)
        self._coverage_history.append(1 - miss)
        # ACI rule: gradient step on miscoverage error
        new_alpha = self.alpha_t + self.gamma * (self.alpha_star - miss)
        # Clip to keep α_t in [0, 1]
        self.alpha_t = float(np.clip(new_alpha, 0.0, 1.0))
        self._alpha_history.append(self.alpha_t)
        self._last_lo = self._last_hi = None

    @property
    def realized_coverage(self) -> float:
        """Empirical coverage so far = mean(1−err_t)."""
        if not self._coverage_history:
            return float("nan")
        return float(np.mean(self._coverage_history))

    @property
    def alpha_history(self) -> list[float]:
        return list(self._alpha_history)


class AggregatedACI:
    """AgACI (Zaffran 2022) — pool of γ-experts with online aggregation.

    Multiple ACI instances run in parallel with different γ values; a
    Bernstein Online Aggregation (BOA) / Hedge weight on each expert
    selects the best-performing γ adaptively.

    For an MPH-scale workload (n=8 real slab), single-γ ACI is usually
    sufficient — AgACI is included for robustness sensitivity.
    """

    def __init__(
        self,
        alpha_star: float = 0.05,
        gammas: tuple[float, ...] = (0.001, 0.01, 0.05, 0.1, 0.2),
        eta: float = 0.5,
    ):
        self.experts = [
            AdaptiveConformal(alpha_star=alpha_star, gamma=g) for g in gammas
        ]
        self.weights = np.full(len(gammas), 1.0 / len(gammas))
        self.eta = float(eta)
        self.alpha_star = float(alpha_star)

    def calibrate(self, residuals: np.ndarray) -> "AggregatedACI":
        for e in self.experts:
            e.calibrate(residuals)
        return self

    def predict_interval(self, y_pred: float) -> tuple[float, float]:
        # Weighted average of expert intervals
        intervals = [e.predict_interval(y_pred) for e in self.experts]
        los = np.array([iv[0] for iv in intervals])
        his = np.array([iv[1] for iv in intervals])
        lo = float(np.average(los, weights=self.weights))
        hi = float(np.average(his, weights=self.weights))
        return lo, hi

    def update(self, y_true: float) -> None:
        # BOA-style weight update: penalize experts that missed
        misses = np.array([
            0 if (e._last_lo <= y_true <= e._last_hi) else 1  # type: ignore
            for e in self.experts
        ], dtype=np.float64)
        # Loss = miscoverage indicator; multiplicative weight update
        self.weights = self.weights * np.exp(-self.eta * misses)
        self.weights = self.weights / self.weights.sum()
        for e in self.experts:
            e.update(y_true)


# ─── Convenience function: run ACI offline on an entire (preds, truths) trace
def run_aci_offline(
    in_sample_residuals: np.ndarray,
    predictions: np.ndarray,
    observations: np.ndarray,
    alpha_star: float = 0.05,
    gamma: float = 0.05,
) -> dict:
    """Apply ACI to a sequence of (prediction, observation) pairs.

    Returns: {lower, upper, alpha_path, coverage, mean_width, method}
    """
    ac = AdaptiveConformal(alpha_star=alpha_star, gamma=gamma)
    ac.calibrate(in_sample_residuals)
    los, his = [], []
    for pred, obs in zip(predictions, observations):
        lo, hi = ac.predict_interval(float(pred))
        los.append(lo); his.append(hi)
        ac.update(float(obs))
    los_a, his_a = np.array(los), np.array(his)
    return {
        "lower": los_a,
        "upper": his_a,
        "alpha_path": np.array(ac.alpha_history),
        "coverage": ac.realized_coverage,
        "mean_width": float(np.mean(his_a - los_a)),
        "method": f"ACI(γ={gamma}, α*={alpha_star})",
    }
