"""
simulation/models/epiestim_models.py
=====================================
EpiEstim (Cori 2013) Rt-based ILI rate forecaster.

[학술 배경]
EpiEstim (Cori et al. 2013, AJE 178(9):1505-1512) 은 효과적 재생산수 (Rt)
실시간 추정 표준. Bayesian posterior:
  Rt | I_{1..t} ~ Gamma(a + ΣI_τ, 1/(b + ΣI_τ/Λ_τ))
where Λ_τ = serial interval weighted history.

[ILI rate 적용]
- ILI rate → cases proxy (× population × 0.01)
- Rt estimation via epyestim.r_covid (Cori 2013 method)
- Renewal equation forecast:
    I_{t+1} = R_t × Σ_{s=1}^{t} I_{t-s+1} × w_s
  where w_s = serial interval pmf.

[참조]
- Cori A, Ferguson NM, Fraser C, Cauchemez S (2013). "A new framework and
  software to estimate time-varying reproduction numbers during epidemics".
  AJE 178(9):1505-1512.
- epyestim package: https://github.com/lo-hfk/epyestim
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


# Influenza serial interval (Vink 2014 meta-analysis: mean 2.85 days, sd 0.93)
# Weekly aggregation → ~0.4 weeks mean. Approximate Gamma pmf:
_INFLUENZA_SI_WEEKLY = np.array([
    0.50, 0.30, 0.12, 0.05, 0.02, 0.005, 0.005,
], dtype=np.float64)
_INFLUENZA_SI_WEEKLY /= _INFLUENZA_SI_WEEKLY.sum()


class EpiEstimForecaster(BaseForecaster):
    """EpiEstim Rt + renewal equation ILI rate forecaster (Cori 2013).

    학습 (fit):
      1. y_train (ILI rate) → cases proxy
      2. epyestim.bagging_r 로 Rt time series 추정
      3. 최근 mean Rt + serial interval 저장

    예측 (predict):
      1. test horizon = len(X_test)
      2. renewal equation:
         I_{t+1} = Rt × Σ_s I_{t-s+1} × w_s
      3. autoregressive forecast (no covariates — pure Rt model)
    """

    meta = ModelMeta(
        name="EpiEstim",
        category="epi",
        level=5,
        min_data=60,
        description=(
            "EpiEstim Rt (Cori 2013) + renewal equation forecast. "
            "Bayesian Rt estimator — 학술 표준."
        ),
        dependencies=["epyestim"],
    )

    def __init__(self, window_size: int = 7, alpha: float = 3.0,
                  beta: float = 1.0):
        super().__init__()
        self._y_train = None
        self._rt_recent = 1.0
        self._si = _INFLUENZA_SI_WEEKLY
        self._window = int(window_size)
        self._alpha = float(alpha)  # Gamma prior shape
        self._beta = float(beta)    # Gamma prior rate
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "EpiEstimForecaster":
        from simulation.models.base import _validate_shapes
        _validate_shapes(X_train, y_train, name="EpiEstim.fit", min_n=30)

        # Save y_train (predict 에서 renewal init 으로 사용)
        self._y_train = np.maximum(y_train, 0.01).astype(np.float64)

        try:
            import epyestim
            # Cases proxy = ILI rate × 100 (scale up to avoid epyestim 0 floor)
            cases = pd.Series(self._y_train * 100.0,
                                index=pd.date_range("2020-01-01",
                                                       periods=len(self._y_train),
                                                       freq="W"))
            # EpiEstim Rt — gt_distribution 기본 (COVID); flu SI 도 적용 가능
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rt_df = epyestim.bagging_r(cases,
                                              gt_distribution=self._extend_si_to_daily(),
                                              smoothing_window=self._window,
                                              n_samples=20,
                                              quantiles=(0.5,))
            if len(rt_df) > 0 and "Q0.5" in rt_df.columns:
                self._rt_recent = float(rt_df["Q0.5"].tail(self._window).mean())
            else:
                self._rt_recent = 1.0
            self._fitted = True
            log.info(f"  [EpiEstim] Rt_recent (last {self._window} wks) = {self._rt_recent:.3f}")
        except Exception as e:
            log.warning(f"  [EpiEstim] Rt 추정 실패: {e} → Rt=1.0 fallback")
            self._rt_recent = 1.0
            self._fitted = True
        return self

    def _extend_si_to_daily(self) -> np.ndarray:
        """주간 SI → 일간 SI (epyestim 가 daily 기대)."""
        # 주간 7-element SI → ~49 일 daily pmf (linearly interpolated)
        weekly = self._si
        daily = np.repeat(weekly, 7) / 7.0
        daily = daily / daily.sum()
        return daily

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        from simulation.models.base import sanitize_predictions

        n_test = len(X_test)
        if not self._fitted or self._y_train is None:
            return sanitize_predictions(np.full(n_test, 1.0))

        # Renewal forecast:
        # I_{t+1} = Rt × Σ_s I_{t-s+1} × w_s
        # init: last len(si) ILI from train
        history = list(self._y_train[-len(self._si):])
        # G-327 (2026-06-20, 사용자: rolling): y_observed 주면 history 에 **관측값** append(self-feeding
        #   renewal 드리프트→음수 회피, 매주 1-step). 없으면 자기예측(legacy 단일원점).
        _obs = (np.asarray(y_observed, dtype=np.float64)
                if y_observed is not None and len(y_observed) == n_test else None)
        preds = []
        for _t in range(n_test):
            # Renewal sum: Σ I_{t-s+1} × w_s
            recent = history[-len(self._si):]
            recent_rev = recent[::-1]  # reverse so w_0 multiplies most recent
            renewal = sum(r * w for r, w in zip(recent_rev, self._si))
            next_val = self._rt_recent * renewal
            preds.append(next_val)
            history.append(float(_obs[_t]) if _obs is not None else next_val)

        pred = np.array(preds, dtype=np.float64)
        pred = np.clip(pred, 0.0, float(np.max(self._y_train) * 5.0))
        return sanitize_predictions(pred)


# ═══════════════════════════════════════════════════════════════════════════
try:
    REGISTRY.register(EpiEstimForecaster)
    log.info("[epiestim_models] EpiEstimForecaster 등록됨 (Cori 2013)")
except Exception as _e:
    log.warning(f"[epiestim_models] 등록 skip: {_e}")
