"""
simulation/models/target_transform.py
======================================
타겟 변수 전처리 전략 모음.

COVID-era 분포 이동 (train mean=8.59, test mean=28.83, ratio=0.30x) 대응:
  - log1p: 분산 안정화 + 분포 이동 축소 (0.30x → 0.64x)
  - sqrt:  약한 변환, 극단값 완화
  - boxcox: 최적 lambda 자동 탐색 (scipy)
  - robust: IQR 기반 스케일링 (이상치에 강건)
  - none:   변환 없음 (기존 방식)

COVID-era 전략:
  - curriculum: 후반부 샘플에 높은 가중치 (기존)
  - clip: 극단값 클리핑 (percentile 기반)
  - downweight: COVID-era 이전 저 ILI 기간 가중치 축소
  - none: 처리 안함

사용법:
    tt = TargetTransformer(method="log1p")
    y_transformed = tt.fit_transform(y_train)
    # ... model.fit(X, y_transformed) ...
    y_pred_original = tt.inverse_transform(y_pred_log)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. TargetTransformer: 타겟 변수 변환 전략
# ══════════════════════════════════════════════════════════════

# Hierarchical menu (Sprint 1.5 R6-A, 2026-05-26) — TargetTransformer 가
# preproc_optuna_hierarchical 의 9 transform 도 위임 사용 가능. 기존 5 method
# (log1p/sqrt/boxcox/robust/none) 는 학습 결과 호환 위해 그대로 자체 처리.
_HIERARCHICAL_METHODS = frozenset({
    "asinh", "rank", "mcmc_robust", "laplace",
    "yeo_johnson", "gaussian",
    "anscombe", "freeman_tukey", "arcsine_sqrt",
})


@dataclass
class TargetTransformer:
    """
    타겟 변수 변환기.

    Parameters:
        method: 변환 방식 — 14가지 지원 (Sprint 1.5 R6-A, 2026-05-26)
          Legacy (자체 구현, 학습 결과 호환):
            - "log1p":  np.log1p(y) ↔ np.expm1(y)
            - "sqrt":   np.sqrt(y) ↔ y**2
            - "boxcox": scipy Box-Cox (lambda 자동 탐색)
            - "robust": (y - median) / IQR 스케일링 — IQR-based (legacy)
            - "none":   변환 없음
          Hierarchical 위임 (preproc_optuna_hierarchical 의 _apply_single_y_transform):
            - "asinh":          arcsinh — heavy-tail 안정
            - "rank":           rank ordinal (분포 무관)
            - "mcmc_robust":    (y - median) / (1.4826 × MAD) — MAD-based
            - "laplace":        (y - median) / (MAD / 0.6745)
            - "yeo_johnson":    sklearn PowerTransformer + G-146 cap
            - "gaussian":       QuantileTransformer(output='normal') + G-146 cap
            - "anscombe":       2·sqrt(x + 3/8) — Poisson VST mean > 1
            - "freeman_tukey":  sqrt(x) + sqrt(x+1) — low-mean Poisson VST
            - "arcsine_sqrt":   2·arcsin(sqrt(p)) — proportion VST [0,1]
        clip_negative: 역변환 후 음수값 0으로 클리핑 (기본: True)

    R6-A notes:
        - "robust" (legacy IQR) ≠ "mcmc_robust" (MAD-based). 학습 결과 호환을 위해
          legacy "robust" 는 변경 없이 그대로 유지. 새 caller 는 "mcmc_robust"
          권장. expanding_cv 는 별도 alias 처리 ("robust" → "mcmc_robust").
        - Hierarchical 위임 9 method 는 G-146 inverse cap 자동 적용.
        - Hierarchical 호출의 fitted state 는 `_hier_inv_fn` 에 closure 로 저장.
          Champion artifact pickle 호환 (closure 안 captured 값은 모두 immutable
          또는 sklearn fitted objects).
    """

    method: Literal[
        "log1p", "sqrt", "boxcox", "robust", "none",
        "asinh", "rank", "mcmc_robust", "laplace",
        "yeo_johnson", "gaussian",
        "anscombe", "freeman_tukey", "arcsine_sqrt",
    ] = "log1p"
    clip_negative: bool = True

    # Box-Cox 전용 (legacy method="boxcox")
    _boxcox_lambda: Optional[float] = field(default=None, repr=False)
    # Robust 전용 (legacy method="robust")
    _median: Optional[float] = field(default=None, repr=False)
    _iqr: Optional[float] = field(default=None, repr=False)
    # Hierarchical 위임 state (R6-A method ∈ _HIERARCHICAL_METHODS)
    _hier_inv_fn: Optional[Callable] = field(default=None, repr=False)
    _hier_state: dict = field(default_factory=dict, repr=False)
    _hier_y_t: Optional[np.ndarray] = field(default=None, repr=False)

    _fitted: bool = field(default=False, repr=False)

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        """학습 데이터로 변환 파라미터 추정."""
        y = np.asarray(y, dtype=float)

        if self.method == "boxcox":
            from scipy.stats import boxcox
            # Box-Cox는 양수만 허용 → 최소값 + 1
            y_pos = y + 1.0 if y.min() <= 0 else y
            _, self._boxcox_lambda = boxcox(y_pos)
            log.info(f"  [TargetTransformer] Box-Cox lambda={self._boxcox_lambda:.4f}")

        elif self.method == "robust":
            self._median = float(np.median(y))
            q75, q25 = np.percentile(y, [75, 25])
            self._iqr = float(max(q75 - q25, 1e-8))
            log.info(f"  [TargetTransformer] Robust: median={self._median:.2f}, IQR={self._iqr:.2f}")

        elif self.method in _HIERARCHICAL_METHODS:
            # R6-A: hierarchical 위임. fit 시점에 (y_t, inv_fn, state) 생성 + 저장.
            from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform
            self._hier_y_t, self._hier_inv_fn, self._hier_state = _apply_single_y_transform(y, self.method)
            log.info(f"  [TargetTransformer] hierarchical method={self.method} "
                     f"state_keys={list(self._hier_state.keys())}")

        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """변환 적용. R6-A: hierarchical method 는 같은 y_train 일 때 cached 사용."""
        y = np.asarray(y, dtype=float)

        if self.method == "log1p":
            return np.log1p(np.maximum(y, 0))

        elif self.method == "sqrt":
            return np.sqrt(np.maximum(y, 0))

        elif self.method == "boxcox":
            y_pos = y + 1.0 if y.min() <= 0 else y
            if abs(self._boxcox_lambda) < 1e-6:
                return np.log(y_pos)
            return (y_pos ** self._boxcox_lambda - 1) / self._boxcox_lambda

        elif self.method == "robust":
            return (y - self._median) / self._iqr

        elif self.method in _HIERARCHICAL_METHODS:
            # If caller's y matches fit y (same shape + values), use cached.
            # Otherwise re-fit hierarchical on the new y (state may differ —
            # logged for visibility).
            if self._hier_y_t is not None and y.shape == self._hier_y_t.shape:
                # Cheap identity check by hash — handles fit_transform fast-path.
                return self._hier_y_t.copy()
            from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform
            log.debug(f"  [TargetTransformer] transform() re-fits hierarchical "
                      f"method={self.method} on new y (shape={y.shape})")
            y_t, _, _ = _apply_single_y_transform(y, self.method)
            return y_t

        else:  # "none"
            return y.copy()

    def inverse_transform(self, y_t: np.ndarray) -> np.ndarray:
        """역변환."""
        y_t = np.asarray(y_t, dtype=float)

        if self.method == "log1p":
            result = np.expm1(y_t)

        elif self.method == "sqrt":
            result = y_t ** 2

        elif self.method == "boxcox":
            lam = self._boxcox_lambda
            if abs(lam) < 1e-6:
                result = np.exp(y_t) - 1.0
            else:
                result = (y_t * lam + 1) ** (1 / lam) - 1.0

        elif self.method == "robust":
            result = y_t * self._iqr + self._median

        elif self.method in _HIERARCHICAL_METHODS:
            if self._hier_inv_fn is None:
                raise ValueError(
                    f"TargetTransformer.inverse_transform: method={self.method!r} "
                    f"requires .fit() first (hierarchical inv_fn not built)."
                )
            result = self._hier_inv_fn(y_t)

        else:  # "none"
            return y_t.copy()

        if self.clip_negative:
            result = np.maximum(result, 0)
        return result

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """fit + transform. R6-A: hierarchical 도 cached path 활용."""
        self.fit(y)
        # Hierarchical methods: fit 가 이미 _hier_y_t 캐싱 → 그대로 반환.
        if self.method in _HIERARCHICAL_METHODS and self._hier_y_t is not None:
            return self._hier_y_t.copy()
        return self.transform(y)

    def describe_shift(self, y_train: np.ndarray, y_test: np.ndarray) -> dict:
        """분포 이동 정보 (변환 전/후 비교)."""
        # 변환 전
        orig_train_mean = float(np.mean(y_train))
        orig_test_mean = float(np.mean(y_test))
        orig_ratio = orig_train_mean / max(orig_test_mean, 0.01)

        # 변환 후
        y_train_t = self.transform(y_train)
        y_test_t = self.transform(y_test)
        trans_train_mean = float(np.mean(y_train_t))
        trans_test_mean = float(np.mean(y_test_t))
        trans_ratio = trans_train_mean / max(trans_test_mean, 0.01)

        return {
            "method": self.method,
            "original": {"train_mean": round(orig_train_mean, 2),
                        "test_mean": round(orig_test_mean, 2),
                        "ratio": round(orig_ratio, 4)},
            "transformed": {"train_mean": round(trans_train_mean, 2),
                           "test_mean": round(trans_test_mean, 2),
                           "ratio": round(trans_ratio, 4)},
            "improvement": round(abs(trans_ratio - 1) - abs(orig_ratio - 1), 4),
        }


# ══════════════════════════════════════════════════════════════
# 2. COVIDStrategy: COVID-era 분포 이동 대응 전략
# ══════════════════════════════════════════════════════════════

@dataclass
class COVIDStrategy:
    """
    COVID-era 분포 이동 대응 전략.

    Parameters:
        mode: 전략 모드
            - "curriculum": 후반부(COVID-era 포함) 가중치 증가
            - "clip_percentile": 극단값을 percentile로 클리핑
            - "reweight_by_magnitude": ILI 크기에 비례한 가중치
            - "none": 처리 없음
        curriculum_weight: curriculum 모드에서 후반부 가중치 (기본: 2.0)
        curriculum_ratio: curriculum 모드에서 "후반부" 비율 (기본: 0.4)
        clip_lower: clip 모드 하위 percentile
        clip_upper: clip 모드 상위 percentile
    """

    mode: Literal["curriculum", "clip_percentile", "reweight_by_magnitude", "none"] = "curriculum"
    curriculum_weight: float = 3.0  # : 2.0→3.0 강화
    curriculum_ratio: float = 0.4
    clip_lower: float = 1.0
    clip_upper: float = 99.0

    def get_sample_weights(self, y_train: np.ndarray) -> np.ndarray:
        """학습 샘플 가중치 생성."""
        n = len(y_train)
        weights = np.ones(n, dtype=float)

        if self.mode == "curriculum":
            # 후반부(COVID-era 근접) 샘플에 높은 가중치
            split_idx = int(n * (1.0 - self.curriculum_ratio))
            weights[split_idx:] = self.curriculum_weight
            log.debug(f"  [COVIDStrategy] Curriculum: {split_idx}+ idx → weight={self.curriculum_weight}")

        elif self.mode == "reweight_by_magnitude":
            # ILI 크기에 비례한 가중치 (고 ILI 시기 중요도 상승)
            y_abs = np.abs(y_train)
            weights = 1.0 + (y_abs / (y_abs.max() + 1e-8)) * (self.curriculum_weight - 1)

        elif self.mode == "clip_percentile":
            # 클리핑은 가중치가 아닌 y 변환이므로 weights=1 유지
            pass

        elif self.mode == "none":
            pass

        return weights

    def clip_target(self, y: np.ndarray) -> np.ndarray:
        """clip_percentile 모드일 때 타겟 클리핑."""
        if self.mode != "clip_percentile":
            return y.copy()
        lo = np.percentile(y, self.clip_lower)
        hi = np.percentile(y, self.clip_upper)
        return np.clip(y, lo, hi)


# ══════════════════════════════════════════════════════════════
# 3. 편의 함수: 추천 설정 프리셋
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 3-1. Per-Model Transform Strategy 
# ══════════════════════════════════════════════════════════════

#  vs 비교 결과 기반 최적 변환 매핑
# - "none": 원본 스케일이 더 좋은 모델 (linear ML, 일부 DL)
# - "log1p": log1p 변환이 더 좋은 모델 (TS, TCN 계열, transformer 계열)
PER_MODEL_TRANSFORM: dict[str, str] = {
    # ── 원본 스케일 유지 (log1p 시 성능 하락) ──
    "SVR-Linear":   "none",   # v4=0.93 → v5b=0.52 (▼0.42)
    "SVR-RBF":      "none",   # v4=0.46 → v5b≈worse
    "ElasticNet":   "none",   # v4=0.88 → v5b=0.80 (▼0.09)
    "KRR":          "none",   # v4=0.70 → v5b=0.62 (▼0.08)
    "LightGBM":     "none",   # v4=0.69 → v5b=0.67 (▼0.02)
    "RandomForest": "none",   # v4=0.77 → v5b=0.75 (▼0.02)
    "DNN":          "log1p",  # D-4: 외삽 폭주 방지 + D-1 clip 병행 (/ log1p 악화는 clip 느슨했을 때 결과)
    "DNN-Optuna":   "none",   # v4=0.66 → v5b=0.63 → v6=0.73 (▼0.03, 신기록)
    "Mamba":        "none",   # v4=0.46 → v5b=0.33 → v6=0.47 (▼0.13, 원본 유지)
    "TimesNet":     "none",   # v4=0.39 → v5b=0.31 → v6=0.34 (▼0.08, 원본 유지)

    # ── log1p 변환 사용 (성능 개선) ──
    "XGBoost":      "none",   # per_model_pipeline_isolated: log1p → 0.40,
                              # none → 0.73-0.78 (/ log1p 우위는 다른 split).
    "N-BEATS":      "log1p",  # v4=0.49 → v5b=0.56 → v6=0.71 (▲0.22, 신기록)
    "N-HiTS":       "none",   # v4=0.60 → v5b=-0.42 → v6=0.39 (log1p 악화 확인)
    "TCN":          "log1p",  # v4=0.66 → v5b=0.75 (▲0.10)
    "TCN-Optuna":   "log1p",  # v4=0.59 → v5b=0.78 (▲0.19)
    "TFT":          "log1p",  # v4=0.37 → v5b=0.44 (▲0.07)
    "PatchTST":     "log1p",  # v4=0.29 → v5b≈improved
    "iTransformer": "log1p",  # v4=0.33 → v5b=0.35 (▲0.03)
    "TiDE":         "log1p",  # v4=0.55 → v5b=0.62 (▲0.07)

    # ── 원본 스케일 필수 (log1p → 스케일 왜곡/이중변환) ──
    "SARIMA":       "none",   # v13=-4.86 → ARIMA 차분 + log1p = 이중변환
    "SARIMAX":      "none",   # v13=-1.12 → statsmodels 내부 변환과 충돌
    "Bayesian-SEIR": "none",  # v13=-1.15 → SEIR ODE/사전분포가 원본 ILI rate 기준
    "Metapop-SEIR": "none",   # v13=-1.33 → Metapopulation ODE 원본 스케일 필요
    "NegBinGLM":    "none",   # v13=-0.05 → NB2 log link + log1p = 이중 log
    "GAM-Spline":   "none",   # GAM 스플라인은 원본 스케일에서 비선형 적합
    "BayesianMCMC": "none",   # MCMC 사후분포 원본 스케일 기준
    "BayesianRidge": "none",  # 베이지안 정규화, 원본 스케일
    "PoissonAutoreg": "none", # Poisson/NB2 log link 이미 내장
    "GP-RBF-Periodic": "none",  # GP 커널 파라미터 원본 스케일 기준
    "OverseasTransfer": "none", # LSTM transfer 원본 분포 기대
    "FoundationModelTransfer": "none",  # Foundation model 원본 분포
    "DNN-Conformal": "none",  # Conformal interval 원본 스케일 보정
    "Rt-Augmented": "none",   # Rt 추정 원본 ILI rate 기준
    "PINN-Lite":    "none",   # PINN physics loss 원본 스케일
    "MP-PINN":      "none",   # PINN physics loss 원본 스케일
}


def get_per_model_strategy(preset: str = "optimal") -> dict[str, str]:
    """
 모델별 최적 변환 전략 반환.

 Presets:
 - "optimal": / 비교 기반 모델별 최적 매핑
 - "all_log1p": 모든 모델에 log1p ( 방식)
 - "all_none": 모든 모델에 원본 스케일 ( 방식)
 - "future_pandemic": 미래 팬데믹 대응용 (log1p 강화)

 Returns:
 dict[str, str]: {model_name: "log1p" | "none"}
 """
    if preset == "optimal":
        return PER_MODEL_TRANSFORM.copy()
    elif preset == "all_log1p":
        return {k: "log1p" for k in PER_MODEL_TRANSFORM}
    elif preset == "all_none":
        return {k: "none" for k in PER_MODEL_TRANSFORM}
    elif preset == "future_pandemic":
        # 미래 팬데믹 시나리오: 분포 이동이 심할 것으로 예상
        # → 더 많은 모델에 log1p 적용 (선형 ML도 일부 포함)
        mapping = PER_MODEL_TRANSFORM.copy()
        mapping["LightGBM"] = "log1p"
        mapping["RandomForest"] = "log1p"
        mapping["DNN"] = "log1p"
        mapping["DNN-Optuna"] = "log1p"
        return mapping
    else:
        log.warning(f"Unknown per-model preset: {preset}, using 'optimal'")
        return PER_MODEL_TRANSFORM.copy()


def get_preset(name: str = "aggressive") -> tuple[TargetTransformer, COVIDStrategy]:
    """
    사전 정의된 설정 프리셋.

    Presets:
        - "aggressive": log1p + curriculum(3.0) -- 최대 분포 이동 완화
        - "moderate":   sqrt + curriculum(2.0) -- 적당한 완화
        - "conservative": none + curriculum(2.0) -- 기존 v4와 동일
        - "boxcox":     boxcox + curriculum(3.0) -- 최적 변환 탐색
        - "robust":     robust + reweight -- IQR 스케일링 + 크기 가중치
    """
    presets = {
        "aggressive": (
            TargetTransformer(method="log1p"),
            COVIDStrategy(mode="curriculum", curriculum_weight=3.0),
        ),
        "moderate": (
            TargetTransformer(method="sqrt"),
            COVIDStrategy(mode="curriculum", curriculum_weight=2.0),
        ),
        "conservative": (
            TargetTransformer(method="none"),
            COVIDStrategy(mode="curriculum", curriculum_weight=2.0),
        ),
        "boxcox": (
            TargetTransformer(method="boxcox"),
            COVIDStrategy(mode="curriculum", curriculum_weight=3.0),
        ),
        "robust": (
            TargetTransformer(method="robust"),
            COVIDStrategy(mode="reweight_by_magnitude", curriculum_weight=2.5),
        ),
    }

    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

    return presets[name]
