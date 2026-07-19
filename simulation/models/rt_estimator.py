"""
simulation/models/rt_estimator.py
==================================
Rt (유효재생산수) 역추정 모듈 -- EpiEstim(Cori et al. 2013) 베이지안 방법론

구성:
  1. RtEstimator:   독립 유틸리티 클래스
     - estimate(): ILI 시계열 → 슬라이딩 윈도우 베이지안 Rt 추정
     - 감마분포 serial interval + 공액사전(Gamma) 후진 기반

  2. compute_rt_features(): 헬퍼 함수
     - feature_engine.py에서 호출: ILI → Rt 특성(Rt_mean, Rt_trend, phase)

  3. RtForecaster:  BaseForecaster 서브클래스
     - fit(): Rt 역추정 + Ridge 회귀 (원본 특성 + Rt 특성)
     - predict(): 마지막 Rt 추세 + 기본 특성으로 예측
     - meta: physics 범주, level=15

역학 배경:
  - Serial interval: Gamma(mean=2.6일, sd=1.5일) (Cowling et al. 2009)
  - 갱신 방정식: I_t = sum_{s=1}^{t-1} I_s * w_{t-s} * R_t
  - 베이지안: R_t | 데이터 ~ Gamma(α_posterior, β_posterior)

참고: 2026-04-10 기준
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import gamma

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. RtEstimator -- 독립 유틸리티 클래스
# ═══════════════════════════════════════════════════════════════

class RtEstimator:
    """
    EpiEstim(Cori et al. 2013) 기반 Rt 베이지안 역추정.

    Serial interval을 감마분포로 이산화 후,
    슬라이딩 윈도우 방식으로 각 시점 Rt의 사후분포(Gamma) 추정.
    """

    def __init__(self, window_size: int = 7, prior_shape: float = 1.0, prior_rate: float = 0.5):
        """
        Parameters:
            window_size: 슬라이딩 윈도우 너비 (일/주 단위)
            prior_shape: 감마 사전 α (default: 약정보 = 1.0)
            prior_rate: 감마 사전 β (default: 약정보 = 0.5)
        """
        self.window_size = window_size
        self.prior_shape = prior_shape
        self.prior_rate = prior_rate
        self._si_weights = None  # Cached serial interval 가중치

    def _compute_serial_interval_weights(
        self,
        serial_interval_mean: float = 2.6,
        serial_interval_sd: float = 1.5,
        max_delay: int = 30,
    ) -> np.ndarray:
        """
        Serial interval (감마분포)을 이산 가중치로 변환.

        Gamma 분포:
          shape α = (mean/sd)^2
          rate β = mean/sd^2

        반환: w[0..max_delay] -- 정규화된 가중치 (합=1)
        """
        # Gamma 파라미터 계산
        alpha = (serial_interval_mean / serial_interval_sd) ** 2
        beta = serial_interval_mean / (serial_interval_sd ** 2)

        # 일 1부터 max_delay까지 확률질량 계산
        days = np.arange(1, max_delay + 1, dtype=float)
        # PDF 값
        pdf_vals = gamma.pdf(days, a=alpha, scale=1/beta)
        # 정규화
        weights = pdf_vals / pdf_vals.sum()

        return weights

    def estimate(
        self,
        ili_series: np.ndarray,
        serial_interval_mean: float = 2.6,
        serial_interval_sd: float = 1.5,
    ) -> pd.DataFrame:
        """
        ILI 시계열로부터 Rt 역추정 (베이지안 공액).

        Parameters:
            ili_series: (n_timepoints,) 정수 발생 건수 또는 비율
            serial_interval_mean: 평균 잠복기 (일)
            serial_interval_sd: 잠복기 표준편차 (일)

        Returns:
            DataFrame:
              - t: 시점 인덱스
              - Rt_mean: E[R_t | 데이터]
              - Rt_lower: 95% CI 하한
              - Rt_upper: 95% CI 상한
              - n_incident: 해당 윈도우의 발생 건수
        """
        ili_series = np.asarray(ili_series, dtype=float)
        n = len(ili_series)

        # Serial interval 가중치 캐싱
        self._si_weights = self._compute_serial_interval_weights(
            serial_interval_mean, serial_interval_sd
        )

        results = []

        # 슬라이딩 윈도우: window_size 이후부터 시작
        for t in range(self.window_size, n):
            # 윈도우 [t - window_size + 1, t] 구간의 발생 건수
            window_cases = ili_series[t - self.window_size + 1:t + 1]
            total_cases = window_cases.sum()

            # 초기 조건: 발생 건수 0이면 Rt 추정 불가 → skip
            if total_cases < 1:
                results.append({
                    "t": t,
                    "Rt_mean": np.nan,
                    "Rt_lower": np.nan,
                    "Rt_upper": np.nan,
                    "n_incident": 0,
                })
                continue

            # 갱신 방정식: λ_t = sum_{s=1}^{t-1} I_s * w_{t-s} * R
            # 과거 발생에 serial interval 가중치를 곱함
            lambda_t = 0.0
            for lag in range(1, min(t, len(self._si_weights))):
                if t - lag >= 0:
                    lambda_t += ili_series[t - lag] * self._si_weights[lag]

            # 베이지안 공액: Gamma(α_prior, β_prior) → Gamma(α_posterior, β_posterior)
            # likelihood: I_t ~ Poisson(R_t * λ_t)
            # posterior: Gamma(α_prior + I_t, β_prior + λ_t)
            alpha_post = self.prior_shape + window_cases[-1]  # 마지막 타임스텝 발생
            beta_post = self.prior_rate + lambda_t

            # 사후 Rt 분포
            rt_mean = alpha_post / beta_post if beta_post > 0 else 0.0
            rt_var = alpha_post / (beta_post ** 2) if beta_post > 0 else np.inf

            # 95% CI (감마분포 분위수)
            if beta_post > 0 and alpha_post > 0:
                rt_lower = gamma.ppf(0.025, a=alpha_post, scale=1/beta_post)
                rt_upper = gamma.ppf(0.975, a=alpha_post, scale=1/beta_post)
            else:
                rt_lower = rt_upper = np.nan

            results.append({
                "t": t,
                "Rt_mean": rt_mean,
                "Rt_lower": rt_lower,
                "Rt_upper": rt_upper,
                "n_incident": total_cases,
            })

        return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 2. 헬퍼 함수: compute_rt_features()
# ═══════════════════════════════════════════════════════════════

def compute_rt_features(
    ili_series: np.ndarray,
    serial_interval_mean: float = 2.6,
    serial_interval_sd: float = 1.5,
    window_size: int = 7,
) -> pd.DataFrame:
    """
    ILI 시계열로부터 Rt 기반 특성 계산.

    feature_engine.py에서 호출하여 원본 특성에 추가.

    Parameters:
        ili_series: (n_samples,) ILI 비율 또는 발생 건수
        serial_interval_mean: 평균 잠복기
        serial_interval_sd: 잠복기 표준편차
        window_size: 슬라이딩 윈도우 너비

    Returns:
        DataFrame with columns:
          - index: 원본 시계열 인덱스 (0~n-1)
          - Rt_mean: 추정 Rt 평균
          - Rt_lower: 95% CI 하한
          - Rt_upper: 95% CI 상한
          - Rt_trend: 3주 기울기 (Rt 변화 추세)
          - Rt_accel: 3주 가속도 (Rt 변화의 가속)
          - epidemic_phase: 0 (R<1, 감소), 1 (R≥1, 성장)

    주의:
      - 초기 window_size 샘플은 NaN (부족한 이력)
      - Rt_trend/accel도 부족한 샘플 구간에서 NaN
    """
    estimator = RtEstimator(window_size=window_size)
    rt_df = estimator.estimate(ili_series, serial_interval_mean, serial_interval_sd)

    # 원본 시계열 길이만큼 결과 DataFrame 초기화
    n = len(ili_series)
    result = pd.DataFrame({
        "Rt_mean": np.nan,
        "Rt_lower": np.nan,
        "Rt_upper": np.nan,
        "Rt_trend": np.nan,
        "Rt_accel": np.nan,
        "epidemic_phase": np.nan,
    }, index=range(n))

    # Rt 값 채우기: rt_df.t는 window_size부터 시작
    for _, row in rt_df.iterrows():
        t = int(row["t"])
        if 0 <= t < n:
            result.loc[t, "Rt_mean"] = row["Rt_mean"]
            result.loc[t, "Rt_lower"] = row["Rt_lower"]
            result.loc[t, "Rt_upper"] = row["Rt_upper"]

    # Rt_trend: 3주(또는 3샘플) 선형 기울기
    if n >= 4:
        for i in range(3, n):
            rt_window = result.loc[max(0, i-3):i, "Rt_mean"].dropna()
            if len(rt_window) >= 3:
                x = np.arange(len(rt_window), dtype=float)
                y = rt_window.values
                # 최소제곱 기울기
                slope = np.polyfit(x, y, 1)[0]
                result.loc[i, "Rt_trend"] = slope

    # Rt_accel: 3주 기울기의 변화 (2계 차분)
    if n >= 6:
        trends = result["Rt_trend"].values
        for i in range(6, n):
            if not np.isnan(trends[i]) and not np.isnan(trends[i-3]):
                accel = trends[i] - trends[i-3]
                result.loc[i, "Rt_accel"] = accel

    # epidemic_phase: Rt 상태 표시
    # Rt >= 1: 성장 (1), Rt < 1: 감소 (0), NaN: -1
    result["epidemic_phase"] = (
        result["Rt_mean"].apply(lambda x: 1 if (not np.isnan(x) and x >= 1.0) else
                                           (0 if not np.isnan(x) else np.nan))
    )

    return result


# ═══════════════════════════════════════════════════════════════
# 3. RtForecaster -- BaseForecaster 서브클래스 (Level 15)
# ═══════════════════════════════════════════════════════════════

class RtForecaster(BaseForecaster):
    """
    Rt 역추정 기반 증강 예측기.

    설계:
      1. 학습 데이터에서 Rt 역추정 (EpiEstim)
      2. Rt 특성(Rt_mean, Rt_trend, Rt_accel, phase) 생성
      3. 원본 특성 + Rt 특성으로 Ridge 회귀 학습
      4. 예측 시: 마지막 Rt 상태 + 기본 특성으로 예측

    메타정보:
      - name: "Rt-Augmented"
      - category: "physics" (역학 모델 기반)
      - level: 15 (중간 복잡도)
      - min_data: 50 (Rt 추정용 최소 샘플)
    """

    meta = ModelMeta(
        name="Rt-Augmented",
        category="physics",
        level=15,
        min_data=50,
        description="Rt역추정 기반 증강 예측기. EpiEstim 베이지안 + Ridge 회귀.",
        dependencies=["scipy", "sklearn"],
    )

    def __init__(self):
        super().__init__()
        self._ridge_model = None
        self._scaler_X = None
        self._scaler_y = None
        self._rt_estimator = None
        self._last_rt_mean = 1.0  # 기본값
        self._last_rt_trend = 0.0
        self._last_ili = None
        self._n_rt_features = 4  # Rt_mean, Rt_trend, Rt_accel, phase

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> RtForecaster:
        """
        학습: ILI 시계열 → Rt 역추정 → 특성 증강 → Ridge 회귀.

        Parameters:
            X_train: (n_samples, n_features) 기본 특성
            y_train: (n_samples,) ILI 비율
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        y_train = np.asarray(y_train, dtype=float)
        X_train = np.asarray(X_train, dtype=float)

        # 1. Rt 역추정
        log.info(f"  [Rt-Augmented] Rt 역추정 시작 (샘플 {len(y_train)})")
        self._rt_estimator = RtEstimator(window_size=7)
        rt_df = self._rt_estimator.estimate(y_train)

        # 2. Rt 특성 생성
        rt_features = compute_rt_features(y_train)
        rt_cols = ["Rt_mean", "Rt_trend", "Rt_accel", "epidemic_phase"]
        X_rt = rt_features[rt_cols].values

        # 3. 결합: 원본 + Rt 특성
        X_augmented = np.hstack([X_train, X_rt])

        # NaN 처리: 초기 window_size 행은 Rt 값 없음 → 평균값으로 채우기
        for col_idx in range(X_train.shape[1], X_augmented.shape[1]):
            col = X_augmented[:, col_idx]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                fill_val = np.nanmean(valid)
                col[np.isnan(col)] = fill_val
            else:
                col[np.isnan(col)] = 0.0

        # 4. 정규화
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_augmented)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # 5. Ridge 회귀 (정규화 강도 α 고정)
        self._ridge_model = Ridge(alpha=1.0, random_state=42)
        self._ridge_model.fit(X_s, y_s)

        # 마지막 Rt 상태 저장 (predict에서 사용)
        if len(rt_df) > 0:
            last_row = rt_df.iloc[-1]
            self._last_rt_mean = last_row["Rt_mean"] if not np.isnan(last_row["Rt_mean"]) else 1.0
            self._last_ili = y_train[-1]

        self._fitted = True
        log.info(
            f"  [Rt-Augmented] Ridge 학습 완료. "
            f"마지막 Rt={self._last_rt_mean:.3f}, ILI={self._last_ili:.4f}"
        )
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        예측: 마지막 Rt 상태 + 기본 특성으로 예측.

        Parameters:
            X_test: (n_test_samples, n_features) 기본 특성

        Returns:
            (n_test_samples,) 음수 클리핑 포함 예측값
        """
        if not self._fitted or self._ridge_model is None:
            log.warning("[Rt-Augmented] 모델이 학습되지 않음")
            return np.maximum(np.mean(X_test, axis=1), 0)

        X_test = np.asarray(X_test, dtype=float)
        n_test = len(X_test)

        # 마지막 Rt 상태 기반 특성 생성
        # 간단한 방식: 마지막 Rt 값 유지 + 작은 감쇠
        rt_augment = np.zeros((n_test, self._n_rt_features))
        for i in range(n_test):
            # Rt_mean: 기하평균 감쇠 (매 스텝 5% 감쇠)
            rt_val = self._last_rt_mean * (0.95 ** (i + 1))
            rt_augment[i, 0] = rt_val
            # Rt_trend: 선형 감소 (마지막 추세가 계속됨)
            rt_augment[i, 1] = self._last_rt_trend * 0.9 ** (i + 1)
            # Rt_accel: 0 (가속 없다고 가정)
            rt_augment[i, 2] = 0.0
            # epidemic_phase: Rt 상태 유지
            rt_augment[i, 3] = 1.0 if rt_val >= 1.0 else 0.0

        # 결합
        X_augmented = np.hstack([X_test, rt_augment])

        # 정규화 + 예측
        X_s = self._scaler_X.transform(X_augmented)
        pred_s = self._ridge_model.predict(X_s)
        pred = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

        return np.maximum(pred, 0)  # ILI rate ≥ 0


# ── 등록 ──
REGISTRY.register(RtForecaster)
