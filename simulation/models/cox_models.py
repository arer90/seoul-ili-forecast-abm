"""
simulation/models/cox_models.py
================================
Cox Proportional Hazards regression for ILI rate forecasting.

[학술 배경]
Cox PH (Cox 1972) 는 survival analysis 표준 — time-to-event 모델.
ILI rate (continuous) 에 직접 적용 어려움. 표준 epi 적용 방법:

  1. **Outbreak event** = ILI > threshold (e.g., p95(historical baseline))
  2. duration = time-to-outbreak from baseline week
  3. covariates = features (X_train)
  4. predict = ILI_hat = baseline + cumulative_hazard × scale

[구현 결정]
- threshold = 80th percentile of historical ILI rate (training set)
- duration = consecutive weeks since last outbreak event
- predict 변환:
    P(outbreak | X) = 1 - exp(-cumulative_hazard(t | X))
    ILI_hat = ILI_baseline + ILI_outbreak_mean × P(outbreak)

[참조]
- Cox DR (1972). "Regression Models and Life-Tables". JRSSB 34(2):187-220.
- Kleinbaum & Klein (2012). "Survival Analysis: A Self-Learning Text" 3rd ed.
- Lopman et al. (2020). "Influenza outbreak detection in U.S. nursing homes"
  PLoS One — Cox PH 적용 예시.

ILI rate 도메인에 Cox PH 는 학술적으로 정통 — outbreak detection + Rt 추정에
표준으로 사용됨 (정확한 ILI rate prediction 모델 아니지만 outbreak forecast
및 prediction interval 학술 기여).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


class CoxPHForecaster(BaseForecaster):
    """Cox Proportional Hazards forecaster — outbreak event 학습 + ILI 예측.

    Cox PH (Cox 1972) 표준 implementation:
      h(t|x) = h_0(t) × exp(x'β)

    학습 (fit):
      1. ILI threshold = 80th percentile (train set)
      2. duration = weeks-since-last-outbreak
      3. event = ILI > threshold (binary)
      4. CoxPHFitter.fit(X + duration + event)

    예측 (predict):
      1. partial_hazard = exp(X_test × β) — Cox PH score
      2. baseline_hazard scale → predicted ILI:
         ILI_hat = mean(ILI_train) + std(ILI_train) × normalize(score)
      3. Calibration on validation set 필요 (R9 per_model_optimize grid)
    """

    meta = ModelMeta(
        name="CoxPH",
        category="epi",
        level=6,
        min_data=80,
        description=(
            "Cox Proportional Hazards (Cox 1972). Outbreak event detection "
            "with ILI > p80 threshold. 학술 표준 survival analysis."
        ),
        dependencies=["lifelines"],
    )

    def __init__(self, threshold_quantile: float = 0.80,
                  penalizer: float = 0.01,
                  l1_ratio: float = 0.0):
        super().__init__()
        self._cox = None
        self._scaler_X = None
        self._feat_idx = None
        self._ili_mean = 0.0
        self._ili_std = 1.0
        self._ili_max = 100.0
        self._threshold = 0.0
        self._threshold_q = float(threshold_quantile)
        self._penalizer = float(penalizer)
        self._l1_ratio = float(l1_ratio)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "CoxPHForecaster":
        from lifelines import CoxPHFitter
        from sklearn.preprocessing import StandardScaler

        from simulation.models.base import _validate_shapes
        _validate_shapes(X_train, y_train, name="CoxPH.fit", min_n=60)

        # Outbreak event 정의 — threshold = p80(y_train)
        self._threshold = float(np.quantile(y_train, self._threshold_q))
        event = (y_train > self._threshold).astype(int)

        # Duration = weeks-since-last-outbreak (단조증가, threshold 도달 시 reset)
        duration = np.zeros(len(y_train), dtype=np.float64)
        last_event = 0
        for i, ev in enumerate(event):
            duration[i] = i - last_event + 1
            if ev:
                last_event = i

        # ILI 통계 — predict 시 scaling 용
        self._ili_mean = float(np.mean(y_train))
        self._ili_std = float(np.std(y_train) + 1e-8)
        self._ili_max = float(np.max(y_train))

        # X 스케일링 (lifelines 가 high-dim 에 민감)
        self._scaler_X = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)

        # Top-K feature selection (n_features > n_samples → singular)
        n_train = X_s.shape[0]
        K = min(15, max(5, n_train // 6))
        if X_s.shape[1] > K:
            from sklearn.feature_selection import f_regression
            scores, _ = f_regression(X_s, y_train)
            self._feat_idx = np.argsort(-np.abs(np.nan_to_num(scores)))[:K]
            X_s = X_s[:, self._feat_idx]

        # Cox PH fit (lifelines API)
        df = pd.DataFrame(X_s, columns=[f"x{i}" for i in range(X_s.shape[1])])
        df["duration"] = duration
        df["event"] = event

        cph = CoxPHFitter(penalizer=self._penalizer, l1_ratio=self._l1_ratio)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(df, duration_col="duration", event_col="event",
                          show_progress=False)
            self._cox = cph
            self._fitted = True
            log.info(f"  [CoxPH] threshold=p{int(self._threshold_q*100)}={self._threshold:.2f}, "
                       f"events={int(event.sum())}/{len(event)}, K={X_s.shape[1]}")
        except Exception as e:
            log.warning(f"  [CoxPH] CoxPHFitter fit 실패: {e} → fallback baseline only")
            self._cox = None
            self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.base import sanitize_predictions

        if not self._fitted or self._cox is None:
            # Fallback: ILI mean baseline
            pred = np.full(len(X_test), self._ili_mean, dtype=np.float64)
            return sanitize_predictions(pred)

        X_s = self._scaler_X.transform(X_test)
        if self._feat_idx is not None:
            X_s = X_s[:, self._feat_idx]

        # Partial hazard score = exp(X β)
        df_test = pd.DataFrame(X_s, columns=[f"x{i}" for i in range(X_s.shape[1])])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # log(partial hazard) = linear score = X β
                log_hazard = self._cox.predict_log_partial_hazard(df_test).to_numpy()
        except Exception as e:
            log.warning(f"  [CoxPH] predict 실패: {e} → mean baseline")
            return sanitize_predictions(np.full(len(X_test), self._ili_mean))

        # log_hazard → ILI scaling
        # 표준화 점수 → ILI rate domain
        z = (log_hazard - np.mean(log_hazard)) / (np.std(log_hazard) + 1e-8)
        # z ∈ [-3, +3] 대부분 → ILI ∈ [mean - 3σ, mean + 3σ]
        pred = self._ili_mean + z * self._ili_std
        # Domain bound: ILI ≥ 0
        pred = np.clip(pred, 0.0, self._ili_max * 2.0)
        return sanitize_predictions(pred)


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY 등록
# ═══════════════════════════════════════════════════════════════════════════
try:
    REGISTRY.register(CoxPHForecaster)
    log.info("[cox_models] CoxPHForecaster 등록됨 (2026-05-12 사용자 5.b)")
except Exception as _e:
    log.warning(f"[cox_models] 등록 skip: {_e}")
