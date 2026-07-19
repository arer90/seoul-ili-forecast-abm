"""
simulation/models/wallinga_teunis.py
=====================================
Wallinga-Teunis Rt + ILI forecasting (Wallinga & Teunis 2004).

[학술 배경]
Wallinga & Teunis (2004) AJE 160(6):509-516:
  R(t) = Σ_j p_{ij} where p_{ij} ∝ w(t_i - t_j)
  (case-level pairing likelihood — retrospective Rt estimator)

Aggregate version:
  R(t) = I(t) / Σ_{τ>0} I(t-τ) × w(τ)
  where w is generation interval pmf.

[Cori 2013 vs Wallinga 2004]
  - Wallinga: case-pair likelihood, retrospective (needs future data)
  - Cori (EpiEstim): forward-window posterior, real-time
  - 본 구현: aggregate Wallinga (Fraser 2007 reformulation)

[참조]
- Wallinga J, Teunis P (2004). "Different epidemic curves for SARS reveal
  similar impacts of control measures". AJE 160(6):509-516.
- Fraser C (2007). "Estimating individual and household reproduction numbers
  in an emerging epidemic". PLoS ONE 2(8):e758.
"""
from __future__ import annotations

import logging

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

_INFLUENZA_SI_WEEKLY = np.array([
    0.50, 0.30, 0.12, 0.05, 0.02, 0.005, 0.005,
], dtype=np.float64)
_INFLUENZA_SI_WEEKLY /= _INFLUENZA_SI_WEEKLY.sum()


class WallingaTeunisForecaster(BaseForecaster):
    """Wallinga-Teunis Rt + renewal ILI forecaster (2004)."""

    meta = ModelMeta(
        name="Wallinga-Teunis",
        category="epi", level=5, min_data=40,
        description=(
            "Wallinga-Teunis Rt (2004) — aggregate retrospective estimator + "
            "renewal forecast."
        ),
        dependencies=[],
    )

    def __init__(self, smoothing_window: int = 4):
        super().__init__()
        self._y_train = None
        self._rt_recent = 1.0
        self._si = _INFLUENZA_SI_WEEKLY
        self._smooth = int(smoothing_window)
        self._y_max = 100.0
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "WallingaTeunisForecaster":
        from simulation.models.base import _validate_shapes
        _validate_shapes(X_train, y_train, name="Wallinga-Teunis.fit", min_n=20)

        self._y_train = np.maximum(y_train.astype(np.float64), 0.01)
        self._y_max = float(np.max(self._y_train))

        # Aggregate Wallinga-Teunis Rt:
        # R(t) = I(t) / Σ_{τ} I(t-τ) × w(τ)
        n = len(self._y_train)
        rt_series = np.ones(n)
        for t in range(len(self._si), n):
            denom = sum(self._y_train[t - τ] * self._si[τ - 1]
                          for τ in range(1, len(self._si) + 1))
            if denom > 0:
                rt_series[t] = self._y_train[t] / denom

        # Smoothed recent Rt
        if n >= self._smooth:
            self._rt_recent = float(np.mean(rt_series[-self._smooth:]))
        else:
            self._rt_recent = 1.0
        # Bound Rt to reasonable range (avoid blowup)
        self._rt_recent = float(np.clip(self._rt_recent, 0.5, 2.5))
        self._fitted = True
        log.info(f"  [Wallinga-Teunis] Rt_recent (last {self._smooth} wks) = "
                   f"{self._rt_recent:.3f}")
        return self

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        from simulation.models.base import sanitize_predictions

        n_test = len(X_test)
        if not self._fitted or self._y_train is None:
            return sanitize_predictions(np.full(n_test, 1.0))

        history = list(self._y_train[-len(self._si):])
        # G-327 (2026-06-20, 사용자: rolling): y_observed 주면 history 에 **관측값** append(self-feeding
        #   renewal 드리프트→음수 회피, 매주 1-step). 없으면 자기예측(legacy 단일원점).
        _obs = (np.asarray(y_observed, dtype=np.float64)
                if y_observed is not None and len(y_observed) == n_test else None)
        preds = []
        for _t in range(n_test):
            recent_rev = history[-len(self._si):][::-1]
            renewal = sum(r * w for r, w in zip(recent_rev, self._si))
            next_val = self._rt_recent * renewal
            preds.append(next_val)
            history.append(float(_obs[_t]) if _obs is not None else next_val)

        pred = np.clip(np.array(preds), 0.0, self._y_max * 5.0)
        return sanitize_predictions(pred)


try:
    REGISTRY.register(WallingaTeunisForecaster)
    log.info("[wallinga_teunis] WallingaTeunisForecaster 등록됨 (2004)")
except Exception as _e:
    log.warning(f"[wallinga_teunis] 등록 skip: {_e}")
