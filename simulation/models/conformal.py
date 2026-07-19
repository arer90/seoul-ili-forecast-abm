"""
simulation/models/conformal.py
==============================
분포무관(Distribution-free) 불확실성 정량화를 위한 Conformal Prediction 모듈.

설계:
  1. ConformalPredictor: 모든 BaseForecaster를 감싼 래퍼
     - Split Conformal Prediction (Vovk et al.)
     - Conformalized Quantile Regression (CQR) 변형
  
  2. AdaptiveConformalPredictor: 온라인 적응형
     - Adaptive Conformal Inference (ACI, Gibbs & Candès 2021)
     - 시간 경과에 따른 분포 이동 대응
  
  3. ConformalForecaster: BaseForecaster 구현
     - 내부 DNN + Conformal PI
     - fit() → 80% 학습, 20% 보정
     - predict() → 점 예측 + 신뢰 구간 저장

ILI rate(‰) 전용 -- 음수 클리핑 필수
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import stats

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


# Numba JIT path for Jackknife+/CV+ hot loops. Falls back to pure-numpy.
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


# ══════════════════════════════════════════════════════════════════════════
# Phase C2 — Jackknife+ / CV+ (Barber, Candès, Ramdas, Tibshirani 2021
# "Predictive inference with the jackknife+", Annals of Statistics)
#
# Finite-sample coverage ≥ 1 - 2α (no exchangeability assumption needed
# beyond i.i.d.), strictly better than jackknife (no coverage guarantee)
# and often tighter than split conformal for small n because every
# training point contributes to both fitting and calibration.
#
# These are pure functions that take per-fold (or per-LOO) test predictions
# and a cal residual vector. They do not refit models — the caller supplies
# the fold infrastructure. See phase6_wfcv for the production wiring.
# ══════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _order_stat_ceil(values_sorted: np.ndarray, level: float) -> float:
    """⌈level·(n+1)⌉-th order statistic. `values_sorted` MUST be pre-sorted asc."""
    n = values_sorted.shape[0]
    if n == 0:
        return np.nan
    k = int(np.ceil(level * (n + 1)))
    if k < 1:
        k = 1
    elif k > n:
        k = n
    return values_sorted[k - 1]


@njit(cache=True, fastmath=True)
def _order_stat_floor(values_sorted: np.ndarray, level: float) -> float:
    """⌊level·(n+1)⌋-th order statistic. `values_sorted` MUST be pre-sorted asc."""
    n = values_sorted.shape[0]
    if n == 0:
        return np.nan
    k = int(np.floor(level * (n + 1)))
    if k < 1:
        k = 1
    elif k > n:
        k = n
    return values_sorted[k - 1]


def _qplus(values: np.ndarray, level: float) -> float:
    """q⁺_level from Barber+2021: ⌈level·(n+1)⌉-th smallest.

    Clipped to [1, n] so level → 1 picks the max, level → 0 picks the min+.
    Numba-JIT path used when available (5-10× on hot sort+index path).
    """
    v = np.sort(np.ascontiguousarray(values, dtype=np.float64))
    return float(_order_stat_ceil(v, level))


def _qminus(values: np.ndarray, level: float) -> float:
    """q⁻_level: ⌊level·(n+1)⌋-th smallest (identity q⁻_α(S) = -q⁺_α(-S))."""
    v = np.sort(np.ascontiguousarray(values, dtype=np.float64))
    return float(_order_stat_floor(v, level))


@njit(cache=True, fastmath=True)
def _jackknife_plus_loop_jit(
    fp: np.ndarray, r: np.ndarray, alpha: float
) -> tuple:
    """Hot loop: per-test-point lower/upper via ⌊α(n+1)⌋ / ⌈(1-α)(n+1)⌉ order stats
    of (fp[:,t] - r) and (fp[:,t] + r) respectively. Returns two (n_test,) arrays.
    """
    n_cal, n_test = fp.shape
    lower = np.empty(n_test, dtype=np.float64)
    upper = np.empty(n_test, dtype=np.float64)
    for t in range(n_test):
        low = np.empty(n_cal, dtype=np.float64)
        high = np.empty(n_cal, dtype=np.float64)
        for i in range(n_cal):
            low[i] = fp[i, t] - r[i]
            high[i] = fp[i, t] + r[i]
        low.sort()
        high.sort()
        lower[t] = _order_stat_floor(low, alpha)
        upper[t] = _order_stat_ceil(high, 1.0 - alpha)
    return lower, upper


def jackknife_plus_interval(
    fold_preds_test: np.ndarray,
    residuals_cal: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Jackknife+ / CV+ prediction intervals (Barber+2021 Eq 3).

    Args:
        fold_preds_test: shape (n_cal, n_test). Row i is the prediction
            of the fold-i leave-out model on every test point. For
            Jackknife+ each fold holds out exactly one training point;
            for CV+ fold_preds_test[i] is the k-fold prediction of the
            model that held out the fold containing training point i
            (so rows within the same fold share a vector).
        residuals_cal: shape (n_cal,). Absolute residuals |y_i - μ_{-i}(x_i)|.
        alpha: mis-coverage rate (1 - α coverage target).

    Returns:
        (lower, upper): shape (n_test,) each.

    Coverage: ≥ 1 - 2α in finite samples under exchangeability
        (Barber+2021 Thm 1). Width ≤ 2 × split-conformal width in the worst
        case but typically tighter.
    """
    fp = np.asarray(fold_preds_test, dtype=float)
    r = np.asarray(residuals_cal, dtype=float)
    if fp.ndim != 2:
        raise ValueError(f"fold_preds_test must be 2-D, got shape {fp.shape}")
    if fp.shape[0] != r.shape[0]:
        raise ValueError(
            f"n_cal mismatch: fold_preds_test rows={fp.shape[0]} "
            f"vs residuals_cal={r.shape[0]}"
        )
    n_cal, n_test = fp.shape
    if n_cal == 0:
        return np.array([]), np.array([])
    if _HAS_NUMBA:
        return _jackknife_plus_loop_jit(
            np.ascontiguousarray(fp, dtype=np.float64),
            np.ascontiguousarray(r, dtype=np.float64),
            float(alpha),
        )
    lower = np.empty(n_test, dtype=float)
    upper = np.empty(n_test, dtype=float)
    for t in range(n_test):
        lower[t] = _qminus(fp[:, t] - r, alpha)
        upper[t] = _qplus(fp[:, t] + r, 1.0 - alpha)
    return lower, upper


def cv_plus_interval(
    fold_preds_test_by_fold: dict,
    fold_indices: dict,
    residuals_cal: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """CV+ intervals when per-fold test predictions are stored separately.

    Args:
        fold_preds_test_by_fold: {fold_id: preds_on_test (n_test,)}.
        fold_indices: {fold_id: training indices held out in that fold}.
        residuals_cal: shape (n_cal,). Indexed by the same training indices
            referenced in fold_indices.
        alpha: mis-coverage rate.

    Convenience wrapper that expands the compact fold representation into
    the (n_cal, n_test) matrix expected by `jackknife_plus_interval`.
    """
    all_idx = sorted(i for ids in fold_indices.values() for i in ids)
    n_cal = len(all_idx)
    if n_cal == 0:
        return np.array([]), np.array([])
    n_test = next(iter(fold_preds_test_by_fold.values())).shape[0]
    fp = np.empty((n_cal, n_test), dtype=float)
    pos = {idx: row for row, idx in enumerate(all_idx)}
    for fold_id, test_preds in fold_preds_test_by_fold.items():
        for i in fold_indices[fold_id]:
            fp[pos[i]] = test_preds
    r = np.asarray(residuals_cal, dtype=float)[all_idx]
    return jackknife_plus_interval(fp, r, alpha=alpha)


def split_conformal_interval(
    pred_test: np.ndarray,
    residuals_cal: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split-conformal interval (Lei+2018 Thm 2.2) — kept here for
    parity so callers can A/B against Jackknife+. Identical to the
    `ceil((n+1)(1-α))`-th-order-statistic formula used by
    phase10_intervals._conformal_pi.
    """
    pt = np.asarray(pred_test, dtype=float)
    r = np.asarray(residuals_cal, dtype=float)
    n = r.shape[0]
    if n == 0:
        return pt.copy(), pt.copy()
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    q = float(np.sort(np.abs(r))[k - 1])
    return pt - q, pt + q


# ══════════════════════════════════════════════════════════════════════════
# Prompt A — log1p-space split conformal (Tier A ①)
#
# Motivation: in raw ILI‰ space a 3‰ residual at ILI=10‰ and a 25‰ residual
# at ILI=90‰ share the same weight, so the calibration quantile is pulled
# up by peak errors and widths balloon in the low-incidence regime.
# Moving to log1p(y) space makes residuals proportional and lets CP adapt
# to heteroscedasticity without touching the point predictor.
#
# This is a *pure function* companion to split_conformal_interval(); the
# caller supplies raw (y_cal, y_pred_cal, y_pred_test) and the function
# routes residual computation through the chosen space and inverse-maps
# the interval back to raw space.
# ══════════════════════════════════════════════════════════════════════════

def split_conformal_interval_space(
    y_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 0.1,
    residual_space: Literal["raw", "log1p"] = "raw",
) -> Tuple[np.ndarray, np.ndarray]:
    """Split-conformal PI with selectable residual space.

    Parameters
    ----------
    y_cal          : observed calibration targets  (n_cal,)
    y_pred_cal     : point predictions on cal set  (n_cal,)
    y_pred_test    : point predictions on test set (n_test,)
    alpha          : miscoverage rate (0.05 → 95 % PI)
    residual_space : "raw"   — |y_true - y_pred|   on raw ‰  (legacy)
                     "log1p" — |log1p(y_true) - log1p(y_pred)| in log space
                               and the quantile is mapped back via expm1

    Returns
    -------
    (lower, upper) : arrays (n_test,) in raw ‰.  `lower` is clipped to 0.

    Notes
    -----
    * legacy residual_space="raw" is bit-identical to split_conformal_interval,
      so every existing caller is untouched.
    * residual_space="log1p" requires y ≥ 0 — we `clip(y, 0, None)` defensively.
    * k = ⌈(n+1)(1-α)⌉ (clamped to [1, n]) — matches phase10_intervals._conformal_pi.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    y_pred_cal = np.asarray(y_pred_cal, dtype=float)
    y_pred_test = np.asarray(y_pred_test, dtype=float)
    n = y_cal.shape[0]
    if n == 0:
        return y_pred_test.copy(), y_pred_test.copy()

    if residual_space == "raw":
        scores = np.abs(y_cal - y_pred_cal)
    elif residual_space == "log1p":
        y_c = np.log1p(np.clip(y_cal, 0.0, None))
        yp_c = np.log1p(np.clip(y_pred_cal, 0.0, None))
        scores = np.abs(y_c - yp_c)
    else:
        raise ValueError(
            f"unknown residual_space={residual_space!r}; expected 'raw' | 'log1p'"
        )

    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    q = float(np.sort(scores)[k - 1])

    if residual_space == "raw":
        lower = y_pred_test - q
        upper = y_pred_test + q
    else:  # log1p
        yp_t = np.log1p(np.clip(y_pred_test, 0.0, None))
        lower = np.expm1(yp_t - q)
        upper = np.expm1(yp_t + q)

    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)
    return lower, upper


def conformal_interval(
    *,
    method: Literal["split", "jackknife_plus", "cv_plus"],
    alpha: float = 0.1,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to a conformal PI method by name.

    kwargs forwarded to the target function:
      - split:           pred_test, residuals_cal
      - jackknife_plus:  fold_preds_test (n_cal × n_test), residuals_cal
      - cv_plus:         fold_preds_test_by_fold, fold_indices, residuals_cal
    """
    if method == "split":
        return split_conformal_interval(
            kwargs["pred_test"], kwargs["residuals_cal"], alpha=alpha
        )
    if method == "jackknife_plus":
        return jackknife_plus_interval(
            kwargs["fold_preds_test"], kwargs["residuals_cal"], alpha=alpha
        )
    if method == "cv_plus":
        return cv_plus_interval(
            kwargs["fold_preds_test_by_fold"],
            kwargs["fold_indices"],
            kwargs["residuals_cal"],
            alpha=alpha,
        )
    raise ValueError(
        f"unknown conformal method {method!r}; "
        "expected 'split' | 'jackknife_plus' | 'cv_plus'"
    )


# ── Conformal Prediction 래퍼 ──

class ConformalPredictor:
    """
    Split Conformal Prediction (Vovk et al.) + CQR 변형.
    
    모든 BaseForecaster를 감싸서 분포무관 불확실성 정량화 제공.
    
    Nonconformity score = |y_i - ŷ_i|
    Quantile q = ceil((1-α)(n+1))/n percentile
    Prediction Interval: [ŷ - q, ŷ + q]
    """
    
    def __init__(self, base_model: BaseForecaster, alpha: float = 0.1):
        """
        Parameters:
            base_model: 학습된 BaseForecaster 인스턴스
            alpha: 오류율 (기본값 0.1 = 90% 신뢰도)
        """
        if not base_model.is_fitted:
            raise ValueError("base_model이 학습되지 않았습니다. fit() 후 전달하세요.")
        
        self.base_model = base_model
        self.alpha = alpha
        
        # 보정 데이터로부터 계산한 nonconformity scores
        self.cal_scores: np.ndarray = None  # shape (n_cal,)
        self.quantile: float = None  # PI 너비 결정
        
        self._fitted_conformal = False
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """
        보정 집합에서 nonconformity scores 계산.
        
        Parameters:
            X_cal: 보정 특성 (n_cal, n_features)
            y_cal: 보정 관측값 (n_cal,)
        """
        if X_cal.shape[0] == 0:
            raise ValueError("보정 집합이 비어있습니다.")
        
        # 기본 모델 예측
        pred_cal = self.base_model.predict(X_cal)
        
        # Nonconformity scores: 절댓값 잔차
        self.cal_scores = np.abs(y_cal - pred_cal)
        
        # Quantile 계산
        # q_level = ceil((n+1)(1-α)) / n
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = np.clip(q_level, 0, 1)  # [0, 1] 범위 확인
        
        self.quantile = np.quantile(self.cal_scores, q_level)
        
        self._fitted_conformal = True
        log.info(
            f"[Conformal] 보정 완료: n_cal={n}, "
            f"α={self.alpha}, q_level={q_level:.4f}, quantile={self.quantile:.4f}"
        )
    
    def predict_interval(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        불확실성 정량화를 포함한 예측.
        
        Returns:
            (point_pred, lower, upper): 점 예측, 신뢰 구간
            - point_pred: shape (n_test,)
            - lower, upper: shape (n_test,)
        """
        if not self._fitted_conformal:
            raise RuntimeError("calibrate()를 먼저 호출하세요.")
        
        if self.quantile is None:
            raise RuntimeError("quantile이 계산되지 않았습니다.")
        
        # 기본 모델 예측
        point_pred = self.base_model.predict(X_test)
        
        # PI: [ŷ - q, ŷ + q]
        lower = point_pred - self.quantile
        upper = point_pred + self.quantile
        
        # ILI rate ≥ 0 -- 하한 클리핑
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, 0)  # 상한도 확인
        
        return point_pred, lower, upper
    
    def get_coverage(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """
        테스트 집합에서 실제 적중률(coverage) 계산.
        
        Returns:
            coverage: [0, 1] -- 1.0이 목표
        """
        _, lower, upper = self.predict_interval(X_test)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        return coverage


# ── 적응형 Conformal Prediction ──

class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive Conformal Inference (ACI, Gibbs & Candès 2021).
    
    온라인으로 α를 조정하여 시간 경과에 따른 분포 이동에 대응.
    
    α_t+1 = α_t + γ(α_target - error_t)
    여기서 error_t = I(y_t ∉ PI_t) ∈ {0, 1}
    """
    
    def __init__(
        self,
        base_model: BaseForecaster,
        alpha: float = 0.1,
        gamma: float = 0.01,
    ):
        """
        Parameters:
            base_model: 학습된 BaseForecaster
            alpha: 목표 오류율
            gamma: 적응형 학습률 (0.001~0.05 범위)
        """
        super().__init__(base_model, alpha=alpha)
        self.gamma = gamma
        self.alpha_target = alpha
        
        # 온라인 추적
        self.alpha_t = alpha
        self.history_alpha: list[float] = []
        self.history_error: list[int] = []
        self.n_updates = 0
    
    def online_update(self, y_true: float, lower: float, upper: float) -> None:
        """
        새로운 데이터점에서 α 적응적 업데이트.
        
        Parameters:
            y_true: 실제 값
            lower, upper: 이전 단계의 PI
        """
        if not self._fitted_conformal:
            raise RuntimeError("calibrate()를 먼저 호출하세요.")
        
        # Error: 실제값이 구간 밖인지 여부
        error_t = int((y_true < lower) or (y_true > upper))
        
        # α 업데이트
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - error_t)
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.5)  # 합리적 범위
        
        # 이력 저장
        self.history_alpha.append(self.alpha_t)
        self.history_error.append(error_t)
        self.n_updates += 1
        
        if self.n_updates % 10 == 0:
            recent_error = np.mean(self.history_error[-10:])
            log.info(
                f"[ACI] Update {self.n_updates}: "
                f"α={self.alpha_t:.4f}, recent_error={recent_error:.3f}"
            )
    
    def predict_interval(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        적응형 α를 사용한 PI 계산.
        
        주의: 온라인 설정에서는 한 번에 하나의 표본씩 호출.
        """
        if not self._fitted_conformal:
            raise RuntimeError("calibrate()를 먼저 호출하세요.")
        
        point_pred = self.base_model.predict(X_test)
        
        # 현재 α에서 quantile 재계산
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha_t)) / n
        q_level = np.clip(q_level, 0, 1)
        quantile_adaptive = np.quantile(self.cal_scores, q_level)
        
        lower = point_pred - quantile_adaptive
        upper = point_pred + quantile_adaptive
        
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, 0)
        
        return point_pred, lower, upper


# ── BaseForecaster 구현 ──

class ConformalForecaster(BaseForecaster):
    """
    DNN + Adaptive Conformal PI를 결합한 BaseForecaster.
    
    Architecture:
      1. 학습 데이터: 80% proper training, 20% calibration
      2. Proper: 내부 DNN 학습
      3. Calibration: Conformal PI 보정
      4. Prediction: 점 예측 + 불확실성 정량화
    """
    
    meta = ModelMeta(
        name="DNN-Conformal",
        category="dl",
        level=16,
        min_data=80,
        # G-285 (2026-06-16, 3자 감사): 정직 표기 — 내부 base 는 Ridge(alpha=1.0)(아래 __init__/fit
        #   참조), DNN 아님. 'DNN-Conformal' 이름은 역사적(registry/web/53-list 호환 위해 유지).
        #   모델의 가치 = Adaptive Conformal PI(분포무관 구간)는 실재. 점추정 base 만 Ridge.
        description="Ridge(α=1.0) base + Adaptive Conformal PI (분포무관 구간; 이름의 'DNN'은 역사적)",
        requires_gpu=False,
        dependencies=["scikit-learn", "numpy", "scipy"],
    )
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        """
        Parameters:
            alpha: 목표 오류율
            gamma: ACI 학습률
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # 내부 모델 (기본: sklearn Ridge -- lightweight & 빠름)
        self._inner_model: Optional[BaseForecaster] = None
        
        # Conformal predictor
        self._conformal: Optional[AdaptiveConformalPredictor] = None
        
        # 마지막 예측의 PI 저장
        self._last_intervals: tuple[np.ndarray, np.ndarray] = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "ConformalForecaster":
        """
        학습 + 보정.
        
        1. X_train/y_train → 80% proper, 20% calibration
        2. Proper set으로 내부 DNN 학습
        3. Calibration set으로 Conformal PI 보정
        """
        n = len(X_train)
        n_proper = int(0.8 * n)
        
        # 데이터 분할 (순서대로) — : per-call rng, global state 오염 방지
        idx = np.arange(n)
        rng = np.random.default_rng(42)  # 재현성 고정
        rng.shuffle(idx)
        
        idx_proper = idx[:n_proper]
        idx_cal = idx[n_proper:]
        
        X_proper, y_proper = X_train[idx_proper], y_train[idx_proper]
        X_cal, y_cal = X_train[idx_cal], y_train[idx_cal]
        
        log.info(
            f"[ConformalForecaster] 데이터 분할: "
            f"proper={len(X_proper)}, cal={len(X_cal)}"
        )
        
        # 내부 모델 초기화 (Ridge 기본값)
        if self._inner_model is None:
            from sklearn.linear_model import Ridge
            
            # Ridge를 BaseForecaster 패턴으로 감싸기
            class RidgeForecaster(BaseForecaster):
                meta = ModelMeta(
                    name="_Ridge",
                    category="linear",
                    level=2,
                    min_data=10,
                )
                
                def __init__(self):
                    super().__init__()
                    self.model = None   # G-329d: RidgeCV at fit (α 선택)

                def fit(self, X, y, **kwargs):
                    # G-329d (2026-06-20, 3AI feature/HP 워크플로 H-1): frozen Ridge(alpha=1.0) 가
                    #   under-regularized × full-pool feature → 체계적 과소예측(bias −13.9, test R²0.466).
                    #   RidgeCV(efficient LOO, cv=None)로 α 선택 → leakage 0(train 내부)·closed-form 빠름.
                    #   실측(워크플로): α=100 → bias −0.37 (vs α=1.0 −10.32).
                    from sklearn.linear_model import RidgeCV
                    try:
                        self.model = RidgeCV(alphas=np.logspace(-2, 3, 20))
                        self.model.fit(X, y)
                    except Exception:
                        self.model = Ridge(alpha=10.0)   # fallback (α=1.0 보다 보수적)
                        self.model.fit(X, y)
                    self._fitted = True
                    return self
                
                def predict(self, X, **kwargs):
                    pred = self.model.predict(X)
                    return np.maximum(pred, 0)
            
            self._inner_model = RidgeForecaster()
        
        # 내부 모델 학습
        self._inner_model.fit(X_proper, y_proper)
        
        # Conformal predictor 초기화 & 보정
        self._conformal = AdaptiveConformalPredictor(
            self._inner_model, alpha=self.alpha, gamma=self.gamma
        )
        self._conformal.calibrate(X_cal, y_cal)
        
        self._fitted = True
        log.info(f"[ConformalForecaster] 학습 완료")
        
        return self
    
    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        점 예측 반환 (PI는 내부에 저장).
        
        self._last_intervals에 (lower, upper) 저장됨.
        """
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        
        point_pred, lower, upper = self._conformal.predict_interval(X_test)
        
        # PI 저장
        self._last_intervals = (lower, upper)
        
        return np.maximum(point_pred, 0)
    
    def get_prediction_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        마지막 predict() 호출에서의 신뢰 구간.
        
        Returns:
            (lower, upper): shape (n_test,)
        """
        if self._last_intervals is None:
            raise RuntimeError("predict()을 먼저 호출하세요.")
        
        return self._last_intervals
    
    def get_adaptive_alpha(self) -> float:
        """현재 적응형 α 값."""
        if self._conformal is None:
            return self.alpha
        return self._conformal.alpha_t


# ══════════════════════════════════════════════════════════════════════════
# Prompt B — Tier A full: stateful SplitConformal / CQRSplit /
#                 AdaptiveConformalTracker
#
# Three new classes implementing handoff_v22_6_pi_stack §1 Tier A.
# They all sit on top of the existing pure-function primitives so legacy
# callers (ConformalPredictor, _conformal_pi, split_conformal_interval*) are
# untouched.
#
#   SplitConformal(alpha, residual_space, window_weeks, method, _native_space)
#       - absolute-residual split conformal
#       - residual_space ∈ {"raw", "log1p"}, window_weeks=W trims cal to
#         last-W residuals (rolling-52w), _native_space=True keeps raw
#         space for models already fitted in log space (TabularDNN-Lite,
#         PINN-Lite) to avoid a double log1p transform.
#
#   CQRSplit(alpha, residual_space, window_weeks)
#       - Conformalized Quantile Regression (Romano-Patterson-Candès 2019).
#       - Takes pre-computed (q_lo_cal, q_hi_cal) and (q_lo_test, q_hi_test)
#         from any quantile regressor; calibrates the conformity score
#         s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)) and returns the
#         interval [q_lo - q_hat, q_hi + q_hat] (in the chosen space).
#
#   AdaptiveConformalTracker(base, alpha, gamma, alpha_clip)
#       - Gibbs & Candès (2021) ACI online wrapper.
#       - α_{t+1} = clip(α_t + γ·(α_target − 1{miss_t}), alpha_clip_lo, hi).
#       - step(y_true, **kw) produces one-step PI and advances α.
#       - For correctness, only use in a *sliding simulation*; never call
#         one-shot against a batch of test points.
#
# Every class is strict on shapes and returns raw-space arrays clipped at 0.
# ══════════════════════════════════════════════════════════════════════════

def cqr_split_interval(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    alpha: float = 0.1,
    residual_space: Literal["raw", "log1p"] = "raw",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-function CQR split interval (Romano-Patterson-Candès 2019).

    s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))       (conformity score)
    q_hat = ceil((n+1)(1-α))-th smallest of s         (finite-sample quantile)
    PI(x) = [q_lo(x) - q_hat, q_hi(x) + q_hat]        (in chosen space)

    For residual_space="log1p", scores are computed in log1p-space and the
    interval is inverse-mapped via expm1.  y_cal and quantile predictions
    must be non-negative; we clip defensively.
    """
    y = np.asarray(y_cal, dtype=float)
    ql = np.asarray(q_lo_cal, dtype=float)
    qh = np.asarray(q_hi_cal, dtype=float)
    ql_t = np.asarray(q_lo_test, dtype=float)
    qh_t = np.asarray(q_hi_test, dtype=float)
    n = y.shape[0]
    if n == 0:
        return ql_t.copy(), qh_t.copy()

    if residual_space == "raw":
        scores = np.maximum(ql - y, y - qh)
    elif residual_space == "log1p":
        yc = np.log1p(np.clip(y, 0.0, None))
        qlc = np.log1p(np.clip(ql, 0.0, None))
        qhc = np.log1p(np.clip(qh, 0.0, None))
        scores = np.maximum(qlc - yc, yc - qhc)
    else:
        raise ValueError(
            f"unknown residual_space={residual_space!r}; expected 'raw' | 'log1p'"
        )

    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    q_hat = float(np.sort(scores)[k - 1])

    if residual_space == "raw":
        lower = ql_t - q_hat
        upper = qh_t + q_hat
    else:
        ql_tc = np.log1p(np.clip(ql_t, 0.0, None))
        qh_tc = np.log1p(np.clip(qh_t, 0.0, None))
        lower = np.expm1(ql_tc - q_hat)
        upper = np.expm1(qh_tc + q_hat)
    return np.maximum(lower, 0.0), np.maximum(upper, 0.0)


class SplitConformal:
    """Stateful split-conformal PI with residual_space / window_weeks / method.

    Usage:
        sc = SplitConformal(alpha=0.05, residual_space="log1p",
                            window_weeks=52, method="absolute",
                            _native_space=False)
        sc.calibrate(y_cal, y_pred_cal)
        lo, hi = sc.predict_interval(y_pred_test)

    Notes
    -----
    * `method` is kept for forward-compat; only "absolute" is implemented
      here.  CQR lives in its own class (`CQRSplit`) because it requires
      two predicted quantiles instead of one point prediction.
    * When `window_weeks` is set and `len(y_cal) > window_weeks`, only the
      last `window_weeks` residuals are used for calibration.  The input
      vectors must already be in chronological order.
    * `_native_space=True` forces residual_space="raw" *on top of* any
      caller-set value — used by models that already fitted in log-space
      internally and want absolute residuals in their native output.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        residual_space: Literal["raw", "log1p"] = "raw",
        window_weeks: Optional[int] = None,
        method: Literal["absolute"] = "absolute",
        _native_space: bool = False,
    ) -> None:
        if method != "absolute":
            raise ValueError(
                f"SplitConformal.method={method!r}: only 'absolute' supported; "
                "use CQRSplit for method='cqr'."
            )
        if residual_space not in ("raw", "log1p"):
            raise ValueError(
                f"unknown residual_space={residual_space!r}; expected 'raw' | 'log1p'"
            )
        self.alpha = float(alpha)
        self.residual_space = residual_space
        self.window_weeks = int(window_weeks) if window_weeks else None
        self.method = method
        self._native_space = bool(_native_space)

        self._effective_space: str = "raw" if self._native_space else residual_space
        self._scores_sorted: Optional[np.ndarray] = None
        self._n_cal: int = 0
        self._q_cached: Optional[float] = None  # last-computed q, for diagnostics

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray) -> "SplitConformal":
        y = np.asarray(y_cal, dtype=float)
        yp = np.asarray(y_pred_cal, dtype=float)
        if y.shape != yp.shape:
            raise ValueError(
                f"y_cal shape {y.shape} != y_pred_cal shape {yp.shape}"
            )
        if self.window_weeks is not None and self.window_weeks < y.shape[0]:
            y = y[-self.window_weeks:]
            yp = yp[-self.window_weeks:]

        if self._effective_space == "raw":
            scores = np.abs(y - yp)
        else:  # log1p
            scores = np.abs(
                np.log1p(np.clip(y, 0.0, None)) - np.log1p(np.clip(yp, 0.0, None))
            )
        self._scores_sorted = np.sort(scores)
        self._n_cal = int(scores.shape[0])
        self._q_cached = None
        return self

    def _quantile(self, alpha: float) -> float:
        if self._scores_sorted is None or self._n_cal == 0:
            raise RuntimeError("call calibrate() first")
        n = self._n_cal
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        return float(self._scores_sorted[k - 1])

    def predict_interval(
        self,
        y_pred_test: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = self.alpha if alpha is None else float(alpha)
        q = self._quantile(a)
        self._q_cached = q
        yp = np.asarray(y_pred_test, dtype=float)
        if self._effective_space == "raw":
            lo = yp - q
            hi = yp + q
        else:
            ypc = np.log1p(np.clip(yp, 0.0, None))
            lo = np.expm1(ypc - q)
            hi = np.expm1(ypc + q)
        return np.maximum(lo, 0.0), np.maximum(hi, 0.0)

    # diagnostic helpers
    @property
    def q_hat(self) -> Optional[float]:
        return self._q_cached

    @property
    def n_cal_effective(self) -> int:
        return self._n_cal


class CQRSplit:
    """Conformalized Quantile Regression (Romano-Patterson-Candès 2019).

    Stateful wrapper around `cqr_split_interval`.  Requires *predicted*
    quantiles q_lo, q_hi at levels α/2 and 1-α/2 from an upstream quantile
    regressor (LightGBM objective='quantile', sklearn GBR loss='quantile',
    statsmodels QuantReg, etc.).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        residual_space: Literal["raw", "log1p"] = "raw",
        window_weeks: Optional[int] = None,
    ) -> None:
        if residual_space not in ("raw", "log1p"):
            raise ValueError(
                f"unknown residual_space={residual_space!r}; expected 'raw' | 'log1p'"
            )
        self.alpha = float(alpha)
        self.residual_space = residual_space
        self.window_weeks = int(window_weeks) if window_weeks else None

        self._scores_sorted: Optional[np.ndarray] = None
        self._n_cal: int = 0
        self._q_cached: Optional[float] = None

    def calibrate(
        self,
        y_cal: np.ndarray,
        q_lo_cal: np.ndarray,
        q_hi_cal: np.ndarray,
    ) -> "CQRSplit":
        y = np.asarray(y_cal, dtype=float)
        ql = np.asarray(q_lo_cal, dtype=float)
        qh = np.asarray(q_hi_cal, dtype=float)
        if not (y.shape == ql.shape == qh.shape):
            raise ValueError(
                f"calibrate shape mismatch: y={y.shape}, q_lo={ql.shape}, "
                f"q_hi={qh.shape}"
            )
        if self.window_weeks is not None and self.window_weeks < y.shape[0]:
            y = y[-self.window_weeks:]
            ql = ql[-self.window_weeks:]
            qh = qh[-self.window_weeks:]

        if self.residual_space == "raw":
            scores = np.maximum(ql - y, y - qh)
        else:
            yc = np.log1p(np.clip(y, 0.0, None))
            qlc = np.log1p(np.clip(ql, 0.0, None))
            qhc = np.log1p(np.clip(qh, 0.0, None))
            scores = np.maximum(qlc - yc, yc - qhc)
        self._scores_sorted = np.sort(scores)
        self._n_cal = int(scores.shape[0])
        self._q_cached = None
        return self

    def _quantile(self, alpha: float) -> float:
        if self._scores_sorted is None or self._n_cal == 0:
            raise RuntimeError("call calibrate() first")
        n = self._n_cal
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        return float(self._scores_sorted[k - 1])

    def predict_interval(
        self,
        q_lo_test: np.ndarray,
        q_hi_test: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = self.alpha if alpha is None else float(alpha)
        q_hat = self._quantile(a)
        self._q_cached = q_hat
        ql_t = np.asarray(q_lo_test, dtype=float)
        qh_t = np.asarray(q_hi_test, dtype=float)
        if self.residual_space == "raw":
            lo = ql_t - q_hat
            hi = qh_t + q_hat
        else:
            ql_tc = np.log1p(np.clip(ql_t, 0.0, None))
            qh_tc = np.log1p(np.clip(qh_t, 0.0, None))
            lo = np.expm1(ql_tc - q_hat)
            hi = np.expm1(qh_tc + q_hat)
        return np.maximum(lo, 0.0), np.maximum(hi, 0.0)

    @property
    def q_hat(self) -> Optional[float]:
        return self._q_cached

    @property
    def n_cal_effective(self) -> int:
        return self._n_cal


class AdaptiveConformalTracker:
    """Gibbs & Candès (2021) Adaptive Conformal Inference — online wrapper.

    Only correct under *sliding simulation* — never call `.step()` on a
    pre-batched set of test points, because the update rule depends on
    observing y_t between PI_t and PI_{t+1}.

    Wraps a `SplitConformal` or `CQRSplit`; the base object must be already
    calibrated.  Each call to `.step()` returns the one-step PI at α_t and
    then moves α_t ← α_t + γ·(α_target − miss_t).
    """

    def __init__(
        self,
        base,
        alpha: float = 0.05,
        gamma: float = 0.05,
        alpha_clip: Tuple[float, float] = (0.001, 0.499),
    ) -> None:
        if not isinstance(base, (SplitConformal, CQRSplit)):
            raise TypeError(
                f"base must be SplitConformal | CQRSplit, got {type(base).__name__}"
            )
        if getattr(base, "_scores_sorted", None) is None:
            raise RuntimeError("base must already be calibrated (call base.calibrate() first)")
        self.base = base
        self.alpha_target = float(alpha)
        self.alpha_t = float(alpha)
        self.gamma = float(gamma)
        self.alpha_lo, self.alpha_hi = float(alpha_clip[0]), float(alpha_clip[1])
        self.history_alpha: list[float] = [self.alpha_t]
        self.history_miss: list[int] = []
        self.n_updates: int = 0

    def step(
        self,
        y_true: float,
        *,
        y_pred_test: Optional[float] = None,
        q_lo_test: Optional[float] = None,
        q_hi_test: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Advance one time step.

        Pass `y_pred_test` (single scalar) when wrapping SplitConformal,
        or (`q_lo_test`, `q_hi_test`) when wrapping CQRSplit.
        Returns the (lo, hi) at α_t before α is updated.
        """
        if isinstance(self.base, SplitConformal):
            if y_pred_test is None:
                raise ValueError("SplitConformal step requires y_pred_test")
            lo_arr, hi_arr = self.base.predict_interval(
                np.array([y_pred_test], dtype=float), alpha=self.alpha_t
            )
        else:  # CQRSplit
            if q_lo_test is None or q_hi_test is None:
                raise ValueError(
                    "CQRSplit step requires q_lo_test and q_hi_test"
                )
            lo_arr, hi_arr = self.base.predict_interval(
                np.array([q_lo_test], dtype=float),
                np.array([q_hi_test], dtype=float),
                alpha=self.alpha_t,
            )
        lo = float(lo_arr[0])
        hi = float(hi_arr[0])

        miss = int((y_true < lo) or (y_true > hi))
        self.history_miss.append(miss)

        # ACI update: α_{t+1} = α_t + γ·(α_target − miss_t).
        # Miss=1 → α shrinks → NEXT interval is wider.
        # Miss=0 → α grows  → NEXT interval is tighter.
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - miss)
        self.alpha_t = float(np.clip(self.alpha_t, self.alpha_lo, self.alpha_hi))
        self.history_alpha.append(self.alpha_t)
        self.n_updates += 1
        return lo, hi

    def empirical_coverage(self) -> float:
        if not self.history_miss:
            return float("nan")
        return 1.0 - (sum(self.history_miss) / len(self.history_miss))


# ── 등록 ──

REGISTRY.register(ConformalForecaster)

log.info("[conformal.py] ConformalForecaster 등록 완료")

# Package C B-A: Mondrian Conformal — per-group quantile
def package_c_mondrian(
    residuals,
    groups,
    alpha: float = 0.05,
    min_per_group: int = 5,
    fallback_global: bool = True,
):
    """Per-group quantile of residuals.

    Args:
        residuals: 1-D array of (y_true - y_pred) on calibration set
        groups: 1-D array of same length, integer group label per sample
        alpha: 1-alpha coverage (0.05 → 95% PI)
        min_per_group: 그룹 sample <이 값이면 글로벌 fallback
        fallback_global: True면 small-group 시 글로벌 quantile

    Returns:
        dict[group_id → quantile]
    """
    import numpy as _np_pc

    residuals = _np_pc.asarray(residuals)
    groups = _np_pc.asarray(groups)
    q_target = 1.0 - alpha
    global_q = _np_pc.quantile(_np_pc.abs(residuals), q_target)

    out = {}
    unique_groups = _np_pc.unique(groups)
    for g in unique_groups:
        mask = groups == g
        n_g = int(mask.sum())
        if n_g >= min_per_group:
            out[int(g)] = float(_np_pc.quantile(_np_pc.abs(residuals[mask]), q_target))
        elif fallback_global:
            out[int(g)] = float(global_q)
        else:
            out[int(g)] = float("nan")
    out["__global__"] = float(global_q)
    return out


def package_c_mondrian_apply(predictions, groups, group_quantiles: dict):
    """Apply group-specific quantiles to predictions for PI.

    Returns (lower, upper) np.ndarray of same length as predictions.
    """
    import numpy as _np_pc
    predictions = _np_pc.asarray(predictions)
    groups = _np_pc.asarray(groups)
    q_arr = _np_pc.array([
        group_quantiles.get(int(g), group_quantiles["__global__"])
        for g in groups
    ])
    return predictions - q_arr, predictions + q_arr
