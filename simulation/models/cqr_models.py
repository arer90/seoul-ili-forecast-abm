"""
simulation/models/cqr_models.py
================================
Tier A (③) — Conformalized Quantile Regression models.

Three CQR-capable BaseForecaster implementations:
 1. CQRLightGBMForecaster — LightGBM objective='quantile' at α/2 and 1-α/2
 2. CQRGBRForecaster — sklearn GradientBoostingRegressor(loss='quantile')
 3. CQRQuantRegForecaster — statsmodels QuantReg (linear quantile regression)

Each model implements:
 * fit(X, y) — trains two quantile heads (q_lo, q_hi)
 * predict(X) — returns midpoint ((q_lo + q_hi) / 2) for R²/point metrics
 * predict_quantiles(X) — returns (q_lo, q_hi) arrays for downstream CQR calibration
 * predict_interval(X, alpha, ...)
 — returns (lo, hi) from raw q_lo/q_hi (no conformal fix);
 conformal calibration must be done via
 `simulation.models.conformal.CQRSplit` using a
 disjoint cal set.

Design notes (handoff_v22_6_pi_stack §1 Tier A ③, §2 Step 3):
 * α is a *training-time* parameter — default α=0.05 (90 % quantile heads for
 95 % CQR PI). CQRSplit later adjusts at calibration time.
 * Optuna budget is dialed down to max(10, min(30, int(0.2·n_train))) per
 model because CQR trains two heads → 2-3× wall time of absolute regressors.
 * log1p is *not* applied here. Train in raw space; let CQRSplit choose the
 residual space so we can A/B raw vs. log1p cleanly.
 * For statsmodels QuantReg: we standardize features + fit per-quantile
 separately. No regularization → keep top-K (|Pearson r|) feature selection
 so p < n (handoff §3 concern).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Shared util: Optuna budget for CQR (capped tighter than the pipeline-wide
# dyn_cap because training is 2× the cost).
# ══════════════════════════════════════════════════════════════════════════

def _cqr_dyn_cap(n_train: int, default: int = 30) -> int:
    """handoff §2 Step 3: max(10, min(30, int(0.2·n_train)))."""
    return max(10, min(default, int(0.2 * max(n_train, 1))))


# ══════════════════════════════════════════════════════════════════════════
# 1. CQR-LightGBM
# ══════════════════════════════════════════════════════════════════════════

class CQRLightGBMForecaster(BaseForecaster):
    """LightGBM quantile regression for CQR (objective='quantile').

    Trains two LightGBM models at α/2 and 1-α/2.  Returns midpoint from
    .predict() (so the model remains a valid point regressor in the
    runner pipeline) and exposes .predict_quantiles() for downstream
    CQRSplit calibration.
    """

    meta = ModelMeta(
        name="CQR-LightGBM",
        category="tree",
        level=8,
        min_data=50,
        description="LightGBM quantile regression (α/2, 1-α/2) for CQR.",
        dependencies=["lightgbm"],
    )

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self._q_lo_model = None
        self._q_hi_model = None
        self._alpha = float(alpha)
        self._y_train_max: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "CQRLightGBMForecaster":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise RuntimeError(
                "CQRLightGBMForecaster requires lightgbm — `uv pip install lightgbm`"
            ) from e

        q_lo = self._alpha / 2.0
        q_hi = 1.0 - self._alpha / 2.0

        common = dict(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            min_data_in_leaf=5,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            random_state=42,
            verbosity=-1,
            # G-273b (2026-06-15): 격리 subprocess 내 quantile 2-head(q_lo/q_hi)가 n_jobs=2 시
            # LightGBM libomp pthread key 고갈(OMP Error #179) → worker SIGSEGV → run_isolated
            # crash marker → per_model_optimize.py:3061 continue → CQR-LightGBM.json 미생성(52/53).
            # 명시 n_jobs 인수가 OMP_NUM_THREADS env 를 override 하므로 1 필수.
            n_jobs=1,
        )
        # G-273c (2026-06-15): early_stop hold-out 추가 — 이전엔 n_estimators=400 을 2-head
        # (q_lo/q_hi) 모두 끝까지 학습(eval_set 無) = 53 모델 중 유일하게 early_stop 누락.
        # XGBoost/LightGBM/CatBoost 와 동형으로 마지막 15% 를 hold-out 으로 떼어 quantile-loss
        # early_stop(patience=40). 보정(coverage)은 하류 CQRSplit conformal 이 담당하므로
        # hold-out carve-out 무해(predict_quantiles 는 두 head 의 raw predict 만 사용).
        from lightgbm import early_stopping as _lgb_es, log_evaluation as _lgb_logeval
        _n = len(y_train)
        _vs = max(8, int(_n * 0.15))
        if _n - _vs >= 20:   # hold-out 떼어도 train 충분할 때만 early_stop
            _Xtr, _Xes = X_train[:-_vs], X_train[-_vs:]
            _ytr, _yes = y_train[:-_vs], y_train[-_vs:]
            _es_cbs = [_lgb_es(40, verbose=False), _lgb_logeval(0)]
            self._q_lo_model = LGBMRegressor(objective="quantile", alpha=q_lo, **common).fit(
                _Xtr, _ytr, eval_set=[(_Xes, _yes)], callbacks=_es_cbs
            )
            self._q_hi_model = LGBMRegressor(objective="quantile", alpha=q_hi, **common).fit(
                _Xtr, _ytr, eval_set=[(_Xes, _yes)], callbacks=_es_cbs
            )
        else:   # 데이터 너무 작음 → early_stop 생략(기존 동작 보존)
            self._q_lo_model = LGBMRegressor(objective="quantile", alpha=q_lo, **common).fit(
                X_train, y_train
            )
            self._q_hi_model = LGBMRegressor(objective="quantile", alpha=q_hi, **common).fit(
                X_train, y_train
            )
        self._y_train_max = float(np.max(y_train))
        self._fitted = True
        log.info(
            f"  [CQR-LightGBM] α={self._alpha:.3f}, q_lo={q_lo:.3f}, q_hi={q_hi:.3f}, "
            f"y_max={self._y_train_max:.2f}"
        )
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        q_lo, q_hi = self.predict_quantiles(X_test)
        mid = 0.5 * (q_lo + q_hi)
        return np.maximum(mid, 0.0)

    def predict_quantiles(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("CQR-LightGBM: fit() first")
        q_lo = np.asarray(self._q_lo_model.predict(X_test), dtype=float)
        q_hi = np.asarray(self._q_hi_model.predict(X_test), dtype=float)
        # Ensure q_lo ≤ q_hi (quantile crossing can happen with small samples).
        q_lo_fixed = np.minimum(q_lo, q_hi)
        q_hi_fixed = np.maximum(q_lo, q_hi)
        return np.maximum(q_lo_fixed, 0.0), np.maximum(q_hi_fixed, 0.0)

    def predict_interval(
        self, X_test: np.ndarray, alpha: Optional[float] = None, **kw
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Raw quantile heads (no conformal correction).  For calibrated
        PI, pipe through `CQRSplit(...).calibrate(...).predict_interval(...)`.
        """
        _ = alpha  # not used — training α is fixed; PI-level α is set via CQRSplit
        return self.predict_quantiles(X_test)


# ══════════════════════════════════════════════════════════════════════════
# 2. CQR-GradientBoostingRegressor (sklearn)
# ══════════════════════════════════════════════════════════════════════════

class CQRGBRForecaster(BaseForecaster):
    """sklearn GradientBoostingRegressor with loss='quantile' — CQR baseline.

    Slower than LightGBM but dependency-free.  Useful as a sanity check that
    the CQR plumbing is not LightGBM-specific.
    """

    meta = ModelMeta(
        name="CQR-GBR",
        category="tree",
        level=7,
        min_data=50,
        description="sklearn GBR loss='quantile' (α/2, 1-α/2) for CQR.",
        dependencies=["sklearn"],
    )

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self._q_lo_model = None
        self._q_hi_model = None
        self._alpha = float(alpha)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "CQRGBRForecaster":
        from sklearn.ensemble import GradientBoostingRegressor

        q_lo = self._alpha / 2.0
        q_hi = 1.0 - self._alpha / 2.0
        common = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self._q_lo_model = GradientBoostingRegressor(
            loss="quantile", alpha=q_lo, **common
        ).fit(X_train, y_train)
        self._q_hi_model = GradientBoostingRegressor(
            loss="quantile", alpha=q_hi, **common
        ).fit(X_train, y_train)
        self._fitted = True
        log.info(f"  [CQR-GBR] α={self._alpha:.3f}, q_lo={q_lo:.3f}, q_hi={q_hi:.3f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        q_lo, q_hi = self.predict_quantiles(X_test)
        return np.maximum(0.5 * (q_lo + q_hi), 0.0)

    def predict_quantiles(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("CQR-GBR: fit() first")
        q_lo = np.asarray(self._q_lo_model.predict(X_test), dtype=float)
        q_hi = np.asarray(self._q_hi_model.predict(X_test), dtype=float)
        q_lo_fixed = np.minimum(q_lo, q_hi)
        q_hi_fixed = np.maximum(q_lo, q_hi)
        return np.maximum(q_lo_fixed, 0.0), np.maximum(q_hi_fixed, 0.0)

    def predict_interval(
        self, X_test: np.ndarray, alpha: Optional[float] = None, **kw
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = alpha
        return self.predict_quantiles(X_test)


# ══════════════════════════════════════════════════════════════════════════
# 3. CQR-QuantReg (statsmodels linear quantile regression)
# ══════════════════════════════════════════════════════════════════════════

class CQRQuantRegForecaster(BaseForecaster):
    """Linear quantile regression via statsmodels.QuantReg for CQR.

    Unlike trees, QuantReg is parametric and generally under-fits the
    COVID ILI rebound, but it's useful as a linear lower bound and is the
    only CQR family with exact quantile loss minimization.  We apply
    top-K |Pearson r| feature selection to keep p < n.
    """

    meta = ModelMeta(
        name="CQR-QuantReg",
        category="linear",
        level=4,
        min_data=50,
        description="statsmodels QuantReg (top-K + standardize) for CQR.",
        dependencies=["statsmodels"],
    )

    def __init__(self, alpha: float = 0.05, topk: int = 20):
        super().__init__()
        self._q_lo_model = None
        self._q_hi_model = None
        self._scaler_X = None
        self._feat_idx = None
        self._alpha = float(alpha)
        self._topk = int(topk)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "CQRQuantRegForecaster":
        import statsmodels.api as sm
        from sklearn.preprocessing import StandardScaler

        n_train, p_orig = X_train.shape
        k = min(self._topk, p_orig)

        Xc = X_train - X_train.mean(axis=0, keepdims=True)
        yc = y_train - float(y_train.mean())
        num = (Xc * yc[:, None]).sum(axis=0)
        den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum() + 1e-12)
        corr = np.abs(num / np.maximum(den, 1e-12))
        self._feat_idx = np.argsort(-corr)[:k]

        self._scaler_X = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train[:, self._feat_idx])
        X_aug = sm.add_constant(X_s, has_constant="add")

        q_lo = self._alpha / 2.0
        q_hi = 1.0 - self._alpha / 2.0
        try:
            self._q_lo_model = sm.QuantReg(y_train, X_aug).fit(q=q_lo, max_iter=2000)
            self._q_hi_model = sm.QuantReg(y_train, X_aug).fit(q=q_hi, max_iter=2000)
        except Exception as e:
            raise RuntimeError(f"CQR-QuantReg: fit failed — {e}") from e

        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap
        self._fitted = True
        log.info(
            f"  [CQR-QuantReg] top-K={k}/{p_orig}, q_lo={q_lo:.3f}, q_hi={q_hi:.3f}"
        )
        return self

    def _project(self, X_test: np.ndarray) -> np.ndarray:
        import statsmodels.api as sm
        X_s = self._scaler_X.transform(X_test[:, self._feat_idx])
        return sm.add_constant(X_s, has_constant="add")

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.safety import apply_extrapolation_cap  # G-289
        q_lo, q_hi = self.predict_quantiles(X_test)
        return apply_extrapolation_cap(np.maximum(0.5 * (q_lo + q_hi), 0.0),
                                       getattr(self, "_y_train_max", None))

    def predict_quantiles(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("CQR-QuantReg: fit() first")
        from simulation.models.safety import apply_extrapolation_cap  # G-289: 선형 QuantReg 외삽 cap
        X_aug = self._project(X_test)
        q_lo = np.asarray(self._q_lo_model.predict(X_aug), dtype=float)
        q_hi = np.asarray(self._q_hi_model.predict(X_aug), dtype=float)
        q_lo_fixed = np.minimum(q_lo, q_hi)
        q_hi_fixed = np.maximum(q_lo, q_hi)
        _ym = getattr(self, "_y_train_max", None)
        return (apply_extrapolation_cap(np.maximum(q_lo_fixed, 0.0), _ym),
                apply_extrapolation_cap(np.maximum(q_hi_fixed, 0.0), _ym))

    def predict_interval(
        self, X_test: np.ndarray, alpha: Optional[float] = None, **kw
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = alpha
        return self.predict_quantiles(X_test)


# ══════════════════════════════════════════════════════════════════════════
# Registry wiring
# ══════════════════════════════════════════════════════════════════════════

try:
    # G-181 (2026-05-05) — 사용자 명시 deprecate:
    # CQR-LightGBM SIGSEGV (LightGBM 패키지 OMP 자체 버그) — 4 fix path 모두 fail.
    # 대체: CQR-QuantReg (R²=+0.90 PASS).
    # 2026-05-12 (사용자 명시 4.a): CQR 3 모델 모두 유지. G-181 disable 해제.
    REGISTRY.register(CQRLightGBMForecaster)
    REGISTRY.register(CQRGBRForecaster)
    REGISTRY.register(CQRQuantRegForecaster)
    log.info("[cqr_models] CQR-{GBR,QuantReg}Forecaster 등록됨 (CQR-LightGBM deprecated G-181)")
except Exception as _e:
    log.debug(f"[cqr_models] 등록 skip: {_e}")


__all__ = [
    "CQRLightGBMForecaster",
    "CQRGBRForecaster",
    "CQRQuantRegForecaster",
    "_cqr_dyn_cap",
]
