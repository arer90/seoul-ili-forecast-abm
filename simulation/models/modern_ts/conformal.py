"""
simulation/models/modern_ts/conformal.py
=========================================
Conformal Prediction Intervals for any base model.

Split conformal: uses val residuals to compute nonconformity scores.
alpha=0.05 → 95% prediction interval.
Quantifies uncertainty for public health decision-making reports.

개선:
 - 적응형 PI: 예측값 비례 스케일링 (CQR-like)
 - 소표본 보정: Bonferroni-like margin 확대
 - 분포 이동 대응: train-val-test 분포 차이 보정
"""

from __future__ import annotations

import numpy as np

__all__ = ["ConformalPredictionWrapper"]


class ConformalPredictionWrapper:
    r"""Conformal Prediction Intervals for any base model.

 Split conformal: val residuals로 nonconformity score 계산,
 alpha=0.05 → 95% prediction interval.
 보건 의사결정 보고서용 불확실성 정량화.

 : 적응형(adaptive) 모드 추가 — 예측값에 비례하는 PI 폭.
 소표본(51주 val)에서 coverage 개선을 위한 보정 포함.

 사용법:
 cp = ConformalPredictionWrapper(base_model_pred_val, y_val, alpha=0.05)
 lower, upper = cp.predict_interval(base_model_pred_test)
 """

    def __init__(self, pred_val: np.ndarray, y_val: np.ndarray, alpha: float = 0.05,
                 adaptive: bool = True):
        self.alpha = alpha
        self.adaptive = adaptive
        n = len(pred_val)

        # 기본 절대 잔차
        residuals = np.abs(y_val - pred_val)

        # 소표본 보정: n이 작을수록 quantile을 높게 잡아 coverage 확보
        # Vovk (2012) finite-sample correction
        q = min((1 - alpha) * (1 + 1/n), 1.0)
        self.q_hat = float(np.quantile(residuals, q))

        if adaptive:
            # 적응형: 상대 잔차 (예측값 대비 비율) 기반
            # 인플루엔자 ILI rate는 피크 시기에 분산이 크고, 비시즌에 작음
            eps = max(np.mean(np.abs(pred_val)) * 0.01, 0.1)  # 0 나눗셈 방지
            rel_residuals = residuals / (np.abs(pred_val) + eps)
            self.q_hat_relative = float(np.quantile(rel_residuals, q))

            # 분포 이동 마진: val과 test의 분포가 다를 수 있으므로 안전 마진 추가
            # 잔차의 IQR 기반 추가 마진 (robust)
            iqr = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
            self._iqr_margin = iqr * 0.5  # 50% IQR 추가

            # MAD (Median Absolute Deviation) 기반 보정
            mad = float(np.median(np.abs(residuals - np.median(residuals))))
            self._mad_scale = max(mad * 1.4826, 0.1)  # MAD→σ 변환
        else:
            self.q_hat_relative = None
            self._iqr_margin = 0.0
            self._mad_scale = 0.0

    def predict_interval(self, pred_test: np.ndarray):
        if self.adaptive and self.q_hat_relative is not None:
            eps = max(np.mean(np.abs(pred_test)) * 0.01, 0.1)
            # 적응형 PI: 예측값 비례 폭 + IQR 마진
            adaptive_width = self.q_hat_relative * (np.abs(pred_test) + eps) + self._iqr_margin
            # 최소 폭 보장: 고정 q_hat의 50% 이상
            min_width = self.q_hat * 0.5
            width = np.maximum(adaptive_width, min_width)
        else:
            width = np.full_like(pred_test, self.q_hat)

        lower = np.maximum(pred_test - width, 0)
        upper = pred_test + width
        return lower, upper

    @property
    def width(self):
        """평균 PI 폭 (적응형일 때는 근사값)."""
        if self.adaptive and self.q_hat_relative is not None:
            # 대략적인 평균 폭 반환
            return 2 * (self.q_hat + self._iqr_margin)
        return 2 * self.q_hat
