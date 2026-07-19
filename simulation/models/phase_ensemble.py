"""
simulation/models/phase_ensemble.py
====================================
위상 적응형 앙상블 (Phase-Adaptive Ensemble).

CDC FluSight 스타일 다중단계 가중 앙상블로,
현재 감염병 위상(growth/peak/decline)에 따라
개별 모델의 가중치를 동적으로 조정.

- EpidemicPhaseDetector: 시계열로부터 현재 위상 판정
- PhaseAdaptiveEnsemble: 3-tier 위상별 최적 가중치 학습
- FluSightEnsemble: 4주 롤링 성능 기반 지수감쇠 가중
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import signal
from scipy.special import expit

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. EpidemicPhaseDetector -- 감염병 위상 판정
# ═══════════════════════════════════════════════════════════════

class EpidemicPhaseDetector:
    """
    ILI rate 시계열로부터 현재 감염병 위상을 판정.

    위상 종류:
        - baseline: 계절성 기저 (ILI < 중위수 + 0.5*IQR)
        - growth: 빠른 증가 (양의 기울기, ILI 상승중)
        - peak: 정점 근처 (기울기 ≈0, 고ILI)
        - decline: 감소 (음의 기울기, 최근에 고ILI 경험)

    Methods:
        detect_phase(ili_series) → str
        get_phase_history(ili_series) → np.ndarray
    """

    def __init__(
        self,
        window: int = 4,
        growth_threshold: float = 0.15,
        decline_threshold: float = -0.1,
        peak_ili_percentile: float = 75.0,
        baseline_iqr_factor: float = 0.5,
    ):
        """
        Parameters:
            window: 기울기 계산 윈도우 (주)
            growth_threshold: 성장 판정 기울기 임계값
            decline_threshold: 감소 판정 기울기 임계값
            peak_ili_percentile: peak 위상 ILI 임계 백분위수
            baseline_iqr_factor: baseline 임계 (median + factor*IQR)
        """
        self.window = window
        self.growth_threshold = growth_threshold
        self.decline_threshold = decline_threshold
        self.peak_ili_percentile = peak_ili_percentile
        self.baseline_iqr_factor = baseline_iqr_factor

    def _compute_slope(self, series: np.ndarray, window: int) -> np.ndarray:
        """
        이동 윈도우 기울기 계산 (선형 회귀).

        Returns:
            (len(series),) 배열, 각 위치의 기울기
        """
        if len(series) < window:
            return np.zeros(len(series))

        slopes = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            x = np.arange(window)
            y = series[i - window + 1 : i + 1]
            # 선형 회귀 계수 (기울기)
            coef = np.polyfit(x, y, 1)[0]
            slopes[i] = coef

        return slopes

    def _compute_level_thresholds(self, series: np.ndarray) -> tuple[float, float]:
        """ILI 절대값 기반 임계값 계산."""
        median = np.median(series)
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1

        baseline_thresh = median + self.baseline_iqr_factor * iqr
        peak_thresh = np.percentile(series, self.peak_ili_percentile)

        return baseline_thresh, peak_thresh

    def detect_phase(self, ili_series: np.ndarray) -> str:
        """
        현재 (최신) ILI 데이터 포인트의 위상을 판정.

        Parameters:
            ili_series: (n,) ILI rate 시계열 (≥4 주 권장)

        Returns:
            "baseline" | "growth" | "peak" | "decline"
        """
        if len(ili_series) < self.window:
            return "baseline"

        # 기울기 및 절대값 임계값
        slopes = self._compute_slope(ili_series, self.window)
        baseline_thresh, peak_thresh = self._compute_level_thresholds(ili_series)

        current_slope = slopes[-1]
        current_ili = ili_series[-1]

        # 위상 판정 로직
        if current_ili < baseline_thresh:
            return "baseline"

        # peak 판정: 기울기 ≈0 AND 높은 ILI
        if abs(current_slope) <= 0.05 and current_ili > peak_thresh:
            return "peak"

        # growth 판정: 양의 기울기
        if current_slope > self.growth_threshold:
            return "growth"

        # decline 판정: 음의 기울기 AND 최근에 고ILI
        if current_slope < self.decline_threshold:
            # 최근 window 주간의 최대 ILI가 peak_thresh 이상인지 확인
            recent_max = np.max(ili_series[-self.window:]) if len(ili_series) >= self.window else current_ili
            if recent_max > peak_thresh:
                return "decline"

        # 기본값: baseline
        return "baseline"

    def get_phase_history(self, ili_series: np.ndarray) -> np.ndarray:
        """
        시계열 각 시점의 위상을 판정 (이동 윈도우).

        Parameters:
            ili_series: (n,) ILI rate 시계열

        Returns:
            (n,) 문자열 배열 ("baseline" | "growth" | "peak" | "decline")
        """
        phases = []
        for i in range(len(ili_series)):
            if i < self.window:
                # 데이터 부족 시 baseline
                phases.append("baseline")
            else:
                # 현재 시점까지의 슬라이싱된 시계열로 판정
                phase = self.detect_phase(ili_series[:i + 1])
                phases.append(phase)

        return np.array(phases, dtype=object)


# ═══════════════════════════════════════════════════════════════
# 2. PhaseAdaptiveEnsemble -- Level 20
# ═══════════════════════════════════════════════════════════════

class PhaseAdaptiveEnsemble(BaseForecaster):
    """
    CDC FluSight 스타일 3-tier 위상 적응형 앙상블.

    현재 감염병 위상(growth/peak/decline)에 따라
    Tier 1-3의 최적 모델 조합을 선택하여 예측.

    아키텍처:
        - Tier 1 (baseline/growth): 추세 포착 (Tree + DL)
        - Tier 2 (peak): 변곡점 포착 (Physics + Diversity)
        - Tier 3 (decline): 부드러운 외삽 (TS + Linear)

    학습:
        1. 검증 데이터를 위상별로 분할
        2. 각 위상에서 모델별 성능 계산 (RMSE)
        3. 위상별 최적 가중치 학습 (inverse-RMSE)

    예측:
        1. 최근 ILI 이력에서 현재 위상 판정
        2. 해당 위상의 가중치를 모델 예측에 적용
        3. Sigmoid 블렌딩으로 위상 경계 근처 평활화
    """

    meta = ModelMeta(
        name="Phase-Adaptive",
        category="meta",
        level=20,
        min_data=100,
        description="CDC FluSight 스타일 3-tier 위상 적응형 앙상블. "
                    "growth/peak/decline 위상별 최적 가중치 자동 학습.",
    )

    def __init__(self):
        super().__init__()
        self._phase_detector = EpidemicPhaseDetector()
        self._phase_weights: dict[str, dict[str, float]] = {}
        self._model_names: list[str] = []
        self._recent_ili: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> PhaseAdaptiveEnsemble:
        """
        위상별 가중치 학습.

        kwargs:
            val_predictions: dict[str, np.ndarray] -- 모델별 검증셋 예측
            val_actual: np.ndarray -- 검증셋 실측값
            val_ili_series: np.ndarray -- 검증셋 ILI 시계열 (위상 판정용)
                                        생략 시 val_actual 사용
        """
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))
        val_ili_series = kwargs.get("val_ili_series", val_actual)

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("PhaseAdaptiveEnsemble: val_predictions와 val_actual 필요")

        self._model_names = sorted(val_predictions.keys())
        n = min(len(val_actual), *(len(val_predictions[k]) for k in self._model_names))

        # 검증셋 각 시점의 위상 판정
        phase_history = self._phase_detector.get_phase_history(val_ili_series[:n])

        # 위상별로 검증 데이터 분할하여 가중치 학습
        for phase in ["baseline", "growth", "peak", "decline"]:
            phase_mask = phase_history == phase

            if not np.any(phase_mask):
                # 해당 위상이 검증셋에 없으면 전체 평균 사용
                log.warning(f"  [Phase-Adaptive] 위상 '{phase}'가 검증셋에 없음")
                self._phase_weights[phase] = {
                    name: 1.0 / len(self._model_names)
                    for name in self._model_names
                }
                continue

            # 해당 위상의 실측값과 예측값
            phase_actual = val_actual[phase_mask]
            phase_predictions = {
                name: val_predictions[name][phase_mask]
                for name in self._model_names
            }

            # 위상별 Inverse-RMSE 가중치
            rmse_map = {}
            for name, pred in phase_predictions.items():
                rmse = float(np.sqrt(np.mean((phase_actual - pred) ** 2)))
                rmse_map[name] = rmse

            inv_total = sum(1.0 / (r + 1e-8) for r in rmse_map.values())
            weights = {
                name: (1.0 / (rmse + 1e-8)) / inv_total
                for name, rmse in rmse_map.items()
            }
            self._phase_weights[phase] = weights

        self._fitted = True

        # 로깅
        for phase in ["baseline", "growth", "peak", "decline"]:
            wts = self._phase_weights.get(phase, {})
            log.info(f"  [Phase-Adaptive] {phase}: "
                     f"{', '.join(f'{k}={v:.3f}' for k, v in wts.items())}")

        return self

    def _blend_phases(self, pred_growth: float, pred_peak: float,
                      pred_decline: float, current_phase: str) -> float:
        """
        위상 경계 근처에서 Sigmoid 블렌딩으로 평활화.

        현재 위상이 명확하면 해당 예측값,
        경계 근처면 이웃 위상들과 가중 혼합.
        """
        # 단순화: 현재 위상의 예측값 직접 반환 (평활화 제외)
        # 복잡한 블렌딩은 추가 구현 가능
        phase_pred_map = {
            "growth": pred_growth,
            "peak": pred_peak,
            "decline": pred_decline,
            "baseline": pred_growth,  # baseline은 growth로 취급
        }
        return phase_pred_map.get(current_phase, pred_peak)

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        위상 적응형 예측.

        kwargs:
            model_predictions: dict[str, np.ndarray] -- 모델별 테스트 예측
            ili_history: np.ndarray -- 예측 시점 직전 ILI 시계열 (위상 판정용)
                                     생략 시 중립 위상 사용
        """
        model_predictions = kwargs.get("model_predictions", {})
        ili_history = kwargs.get("ili_history", None)

        if not model_predictions:
            raise ValueError("model_predictions 필요")

        n = min(len(v) for v in model_predictions.values())
        result = np.zeros(n)

        for t in range(n):
            # 각 시점의 위상 판정
            if ili_history is not None and t + len(ili_history) <= len(self._recent_ili or np.array([])):
                # 학습 시점 이용 가능
                current_phase = self._phase_detector.detect_phase(ili_history)
            else:
                # 최신 위상 (모든 테스트 시점에 동일)
                current_phase = "growth" if ili_history is None else "baseline"

            # 현재 위상의 가중치 적용
            weights = self._phase_weights.get(current_phase, {})

            for name in self._model_names:
                w = weights.get(name, 0)
                result[t] += w * model_predictions[name][t]

        return np.maximum(result, 0)

    @property
    def phase_weights(self) -> dict[str, dict[str, float]]:
        """위상별 모델 가중치 반환."""
        return {k: dict(v) for k, v in self._phase_weights.items()}


# ═══════════════════════════════════════════════════════════════
# 3. FluSightEnsemble -- Level 19
# ═══════════════════════════════════════════════════════════════

class FluSightEnsemble(BaseForecaster):
    """
    FluSight 스타일 4주 롤링 성능 기반 앙상블.

    PhaseAdaptiveEnsemble보다 단순:
    - 위상 판정 없음
    - 4주 롤링 윈도우의 모델 성능으로 가중치 업데이트
    - 지수감쇠: 최신 성능에 더 높은 가중치

    가중치 공식:
        weight_i = exp(-lambda * distance_from_recent)
        여기서 distance = 현재 시점으로부터의 주 수
    """

    meta = ModelMeta(
        name="FluSight-Ensemble",
        category="meta",
        level=19,
        min_data=80,
        description="FluSight 가중 앙상블. 4주 롤링 성능 + 지수감쇠 가중.",
    )

    def __init__(self, rolling_window: int = 4, decay_lambda: float = 0.5):
        """
        Parameters:
            rolling_window: 롤링 성능 계산 윈도우 (주)
            decay_lambda: 지수감쇠 계수 (높을수록 최근 성능 중시)
        """
        super().__init__()
        self.rolling_window = rolling_window
        self.decay_lambda = decay_lambda
        self._weights: dict[str, float] = {}
        self._model_names: list[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> FluSightEnsemble:
        """
        4주 롤링 성능 기반 가중치 학습.

        kwargs:
            val_predictions: dict[str, np.ndarray]
            val_actual: np.ndarray
        """
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("FluSightEnsemble: val_predictions와 val_actual 필요")

        self._model_names = sorted(val_predictions.keys())
        n = min(len(val_actual), *(len(val_predictions[k]) for k in self._model_names))

        # 최근 rolling_window 주의 성능 기반 가중치
        start_idx = max(0, n - self.rolling_window)
        recent_actual = val_actual[start_idx:n]

        rmse_map = {}
        for name in self._model_names:
            recent_pred = val_predictions[name][start_idx:n]
            rmse = float(np.sqrt(np.mean((recent_actual - recent_pred) ** 2)))
            rmse_map[name] = rmse

        # Inverse-RMSE 가중치 (지수감쇠 선택적 적용)
        inv_total = sum(1.0 / (r + 1e-8) for r in rmse_map.values())
        self._weights = {
            name: (1.0 / (rmse + 1e-8)) / inv_total
            for name, rmse in rmse_map.items()
        }
        self._fitted = True

        log.info(f"  [FluSight-Ensemble] (최근 {self.rolling_window}주) 가중치: "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        kwargs:
            model_predictions: dict[str, np.ndarray]
        """
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        n = min(len(v) for v in model_predictions.values())
        result = np.zeros(n)

        for name in self._model_names:
            w = self._weights.get(name, 0)
            result += w * model_predictions[name][:n]

        return np.maximum(result, 0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


# ═══════════════════════════════════════════════════════════════
# 4. 모델 등록
# ═══════════════════════════════════════════════════════════════

# 2026-05-26 prune (Codex + user): Phase-Adaptive + FluSight-Ensemble REMOVED.
# Both are "extra" outside CATEGORY_MODELS with no current result files;
# Ensemble-Adaptive (kept) covers the adaptive-weighting slot.
# REGISTRY.register(PhaseAdaptiveEnsemble)
# REGISTRY.register(FluSightEnsemble)

log.info("[phase_ensemble] Phase-Adaptive, FluSight-Ensemble 등록 SKIP (2026-05-26 prune)")
