"""
simulation/models/ears_models.py
=================================
EARS C1/C2/C3 outbreak detection (CDC standard, Hutwagner 2003).

[학술 배경]
Early Aberration Reporting System (EARS) — CDC standard for syndromic
surveillance. Three algorithms:

  C1: standardized z-score with 7-day baseline (no shift)
      Y_t > μ_{t-7..t-1} + 3 × σ_{t-7..t-1}
  C2: 2-week shift baseline (avoid contamination by event itself)
      Baseline = t-9 .. t-3 (7 obs)
  C3: 3-week composite (sum of last 3 C2 stats > 3-week threshold)

[ILI rate 적용]
- 학습: historical mean / std (rolling window)
- 예측: forecast = baseline + threshold-corrected adjustment
- 정확한 EARS 는 outbreak detection only — ILI prediction 으로 확장

[참조]
- Hutwagner L, Thompson W, Seeman GM, Treadwell T (2003). "The bioterrorism
  preparedness and response Early Aberration Reporting System (EARS)".
  Journal of Urban Health 80(2 Suppl 1):i89-i96.
- Fricker RD, Hegler BL, Dunfee DA (2008). "Comparing syndromic surveillance
  detection methods". Statistics in Medicine 27:3407-3429.
"""
from __future__ import annotations

import logging

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


class EarsC1Forecaster(BaseForecaster):
    """EARS-C1 baseline forecast (no shift, rolling 7-period mean + sd)."""

    meta = ModelMeta(
        name="EARS-C1", category="epi", level=3, min_data=20,
        description="EARS-C1: μ_{t-7..t-1} + 3σ outbreak detection baseline.",
        dependencies=[],
    )

    def __init__(self, window: int = 7, shift: int = 0):
        super().__init__()
        self._y_train = None
        self._window = int(window)
        self._shift = int(shift)
        self._mean = 0.0
        self._std = 1.0
        self._y_max = 100.0
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "EarsC1Forecaster":
        from simulation.models.base import _validate_shapes
        _validate_shapes(X_train, y_train, name="EARS-C1.fit", min_n=10)

        self._y_train = y_train.astype(np.float64)
        # Tail window for forecast
        end = len(self._y_train) - self._shift
        start = max(0, end - self._window)
        tail = self._y_train[start:end]
        self._mean = float(np.mean(tail))
        self._std = float(np.std(tail) + 1e-3)
        self._y_max = float(np.max(self._y_train))
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.base import sanitize_predictions
        n_test = len(X_test)
        # Forecast = baseline mean (no upper threshold added; that's for
        # outbreak detection rather than point forecast)
        pred = np.full(n_test, self._mean, dtype=np.float64)
        pred = np.clip(pred, 0.0, self._y_max * 5.0)
        return sanitize_predictions(pred)


class EarsC2Forecaster(EarsC1Forecaster):
    """EARS-C2: 2-period shift to avoid baseline contamination."""

    meta = ModelMeta(
        name="EARS-C2", category="epi", level=3, min_data=20,
        description="EARS-C2: 2-week shift baseline (Hutwagner 2003).",
        dependencies=[],
    )

    def __init__(self):
        super().__init__(window=7, shift=2)


class EarsC3Forecaster(BaseForecaster):
    """EARS-C3: 3-week composite (3 × C2 cumulative)."""

    meta = ModelMeta(
        name="EARS-C3", category="epi", level=3, min_data=20,
        description="EARS-C3: 3-period composite of C2 statistics.",
        dependencies=[],
    )

    def __init__(self):
        super().__init__()
        self._c2_pred = 0.0
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "EarsC3Forecaster":
        # 2026-06-16 G-274: C3 가 C2._mean 만 복사 → byte-identical(중복 모델). EARS-C3
        # (Hutwagner 2003)=3-period composite. 점예측엔 detection z-sum 이 안 맞으므로 3개
        # shift(0/1/2) baseline 평균으로 충실한 3-기간 합성(단일 shift=2 인 C2 와 numerically 구분).
        _means = []
        for _sh in (0, 1, 2):
            _c = EarsC1Forecaster(window=7, shift=_sh)
            _c.fit(X_train, y_train)
            _means.append(float(_c._mean))
        self._c2_pred = float(np.mean(_means)) if _means else 0.0
        self._y_max = float(np.max(y_train)) if len(y_train) else 0.0
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.base import sanitize_predictions
        n_test = len(X_test)
        pred = np.full(n_test, self._c2_pred, dtype=np.float64)
        pred = np.clip(pred, 0.0, self._y_max * 5.0)
        return sanitize_predictions(pred)


try:
    REGISTRY.register(EarsC1Forecaster)
    REGISTRY.register(EarsC2Forecaster)
    REGISTRY.register(EarsC3Forecaster)
    log.info("[ears_models] EARS-C1/C2/C3 등록됨 (Hutwagner 2003)")
except Exception as _e:
    log.warning(f"[ears_models] 등록 skip: {_e}")
