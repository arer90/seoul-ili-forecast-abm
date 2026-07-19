"""
simulation/models/ensemble.py
=============================
메타(Meta/Ensemble) 범주 모델:
  - Inverse-RMSE 가중 앙상블
  - Stacking (2단계 메타 학습)
  - Blending (홀드아웃 기반 조합)
  - BMA (Bayesian Model Averaging, BIC 기반)
  - NNLS (Non-Negative Least Squares)
  - Temporal Weight (시간 가중, 지수감쇠)
  - Diversity (다양성 기반, 정확도+독립성)

계획서 기술: "개별 모델의 예측값을 가중 평균(weighted ensemble),
stacking, blending 방법으로 결합하여 단일 모델 대비 일반화 성능을 개선한다.
가중치는 검증 세트의 inverse-RMSE 비례로 산출한다."
"""

from __future__ import annotations

import logging

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


def _drop_negative_controls(val_predictions: dict) -> tuple[dict, list[str]]:
    """P0-1: ensemble 입력에서 NEGATIVE_CONTROL 모델(예: TabularDNN)을 제거.

 paper §Methods 에서 negative control 로 명시된 모델이 앙상블 가중치를
 받으면 "설계상 제외" 원칙과 충돌하므로 여기서 원천 차단한다.
 registry.NEGATIVE_CONTROL 이 source of truth.
 """
    try:
        from simulation.models.registry import NEGATIVE_CONTROL as _NC
    except Exception:
        return dict(val_predictions), []
    dropped = [k for k in val_predictions if k in _NC]
    if not dropped:
        return dict(val_predictions), []
    filtered = {k: v for k, v in val_predictions.items() if k not in _NC}
    return filtered, dropped


# ═══════════════════════════════════════════════════════════════
# 1. Inverse-RMSE 가중 앙상블 -- Level 18
# ═══════════════════════════════════════════════════════════════

class InverseRMSEEnsemble(BaseForecaster):
    """
    Inverse-RMSE 가중 앙상블.

    검증셋 RMSE의 역수 비례로 각 모델의 가중치를 결정.
    RMSE가 낮을수록 높은 가중치 → 우수 모델에 집중.
    """

    meta = ModelMeta(
        name="Ensemble-InvRMSE",
        category="meta",
        level=18,
        min_data=80,
        description="Inverse-RMSE 가중 앙상블. 검증 성능 비례 가중치 자동 산출.",
    )

    def __init__(self):
        super().__init__()
        self._weights: dict[str, float] = {}
        self._models: list[BaseForecaster] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> InverseRMSEEnsemble:
        """
        kwargs:
            models: list[BaseForecaster] -- 이미 학습된 개별 모델
            val_predictions: dict[str, np.ndarray] -- 모델별 검증셋 예측
            val_actual: np.ndarray -- 검증셋 실측값
        """
        models = kwargs.get("models", [])
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("InverseRMSEEnsemble: val_predictions와 val_actual 필요")

        # RMSE 계산
        rmse_map = {}
        for name, pred in val_predictions.items():
            n = min(len(val_actual), len(pred))
            rmse = float(np.sqrt(np.mean((val_actual[:n] - pred[:n]) ** 2)))
            rmse_map[name] = rmse

        # Inverse-RMSE 가중치
        inv_total = sum(1.0 / (r + 1e-8) for r in rmse_map.values())
        self._weights = {
            name: (1.0 / (rmse + 1e-8)) / inv_total
            for name, rmse in rmse_map.items()
        }
        self._models = models
        self._fitted = True

        log.info(f"  [Ensemble-InvRMSE] 가중치: "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        kwargs:
            model_predictions: dict[str, np.ndarray] -- 모델별 테스트 예측
        """
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        n = min(len(v) for v in model_predictions.values())
        result = np.zeros(n)
        for name, pred in model_predictions.items():
            w = self._weights.get(name, 0)
            result += w * pred[:n]

        return np.maximum(result, 0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


# ═══════════════════════════════════════════════════════════════
# 2. Stacking -- Level 19
# ═══════════════════════════════════════════════════════════════

class StackingEnsemble(BaseForecaster):
    """
 Stacking (2단계 메타 학습).

 Level 0: 개별 모델 예측 → 메타 피처
 Level 1: Ridge Regression으로 최적 조합 학습

 Walk-forward CV에서 out-of-fold 예측을 메타 피처로 사용하여
 정보 누출을 방지.

 P1-6:
 1. RidgeCV α 격자를 {0.01, 0.1, 1, 10, 100, 1000} 으로 확장
 (Blending 과 동일 — p≈n·공선성 높은 메타 피처에서 안정적).
 2. 음수 계수를 0 으로 클립 후 원래 |coef| 합으로 재정규화
 (positive projection). 음의 가중치로 다른 모델을 공격적으로
 상쇄하는 과적합 패턴을 제거 — 앙상블의 변동성을 줄인다.
 3. neg_RMSE 기준으로 교차검증.
 """

    meta = ModelMeta(
        name="Ensemble-Stacking",
        category="meta",
        level=19,
        min_data=100,
        description="Stacking. 2단계 메타 학습 -- 개별 예측을 Ridge(positive)로 최적 결합.",
        dependencies=["sklearn"],
    )

    # P1-6: 확장된 α 격자 (Blending 과 align).
    _ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    # : negative-R² 모델이 메타 피처로 들어오는 것을 원천 차단
    _R2_FLOOR = 0.3

    def __init__(self):
        super().__init__()
        self._meta_model = None
        self._model_names: list[str] = []
        self._y_train_max: float = 0.0  # B-1: holdout extrapolation ceiling

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> StackingEnsemble:
        """
        kwargs:
            val_predictions: dict[str, np.ndarray] -- 모델별 검증셋 예측
            val_actual: np.ndarray -- 검증셋 실측값
        """
        from sklearn.linear_model import RidgeCV

        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("StackingEnsemble: val_predictions와 val_actual 필요")

        # P0-1: NEGATIVE_CONTROL 모델 원천 제외.
        val_predictions, _nc_dropped = _drop_negative_controls(val_predictions)
        if _nc_dropped:
            log.info("  [Stacking] NEGATIVE_CONTROL 제외: %s", ", ".join(_nc_dropped))

        # : R²≥_R2_FLOOR 모델만 메타 피처로 허용.
        # GP-RBF-Periodic(R²=-0.52) 같은 모델에 Ridge 가 0.348 가중치를
        # 부여하던 문제 차단. 2개 미만이면 sanity 를 풀되 loud WARN.
        _r2_by_model: dict[str, float] = {}
        for _name, _pred in val_predictions.items():
            _m = min(len(val_actual), len(_pred))
            if _m <= 1:
                continue
            _y = val_actual[:_m]
            _yhat = _pred[:_m]
            _ss_tot = float(np.sum((_y - np.mean(_y)) ** 2))
            if _ss_tot <= 1e-12:
                continue
            _r2_by_model[_name] = 1.0 - float(np.sum((_y - _yhat) ** 2)) / _ss_tot
        qualified = [k for k, v in _r2_by_model.items() if v >= self._R2_FLOOR]
        excluded = [k for k in val_predictions.keys() if k not in qualified]
        if len(qualified) < 2:
            log.warning(
                "  [Stacking] R²≥%.2f 통과 %d개 (<2) → sanity 해제, 전체 투입",
                self._R2_FLOOR, len(qualified),
            )
            qualified = sorted(val_predictions.keys())
            excluded = []
        else:
            if excluded:
                log.info(
                    "  [Stacking] R²<%.2f 제외 %d개: %s",
                    self._R2_FLOOR, len(excluded),
                    ", ".join(f"{k}(R²={_r2_by_model.get(k, float('nan')):.3f})" for k in excluded),
                )

        self._model_names = sorted(qualified)
        n = min(len(val_actual), *(len(val_predictions[k]) for k in self._model_names))

        # 메타 피처 행렬: (n, n_models)
        meta_X = np.column_stack([val_predictions[k][:n] for k in self._model_names])
        meta_y = val_actual[:n]

        # P1-6: 확장된 α 격자 + neg_RMSE 스코어링
        self._meta_model = RidgeCV(
            alphas=self._ALPHAS,
            fit_intercept=True,
            scoring="neg_root_mean_squared_error",
        )
        self._meta_model.fit(meta_X, meta_y)

        # P1-6: positive projection — 음수 계수 0 클립 + 원래 |coef| 합으로 재정규화
        coefs = np.asarray(self._meta_model.coef_, dtype=float)
        coefs_pos = np.clip(coefs, a_min=0.0, a_max=None)
        if coefs_pos.sum() > 1e-8:
            orig_sum = float(np.abs(coefs).sum())
            if orig_sum > 1e-8:
                coefs_pos = coefs_pos * (orig_sum / coefs_pos.sum())
        self._meta_model.coef_ = coefs_pos
        # B-1: store training-time target ceiling for predict clip.
        # Holdout winter peak can put meta-features outside training range →
        # Ridge linear extrapolation blows up (R²=-12949 in run).
        # Cap at 2.5× train max as a generous but finite safety net.
        self._y_train_max = float(np.max(meta_y)) if len(meta_y) else 0.0
        self._fitted = True

        coefs_dict = dict(zip(self._model_names, coefs_pos))
        log.info(f"  [Stacking] Ridge α={self._meta_model.alpha_:.2f} (+proj), "
                 f"y_train_max={self._y_train_max:.2f}, "
                 f"coefs: {', '.join(f'{k}={v:.3f}' for k, v in coefs_dict.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        kwargs:
            model_predictions: dict[str, np.ndarray]
        """
        model_predictions = kwargs.get("model_predictions", {})
        n = min(len(v) for v in model_predictions.values())
        meta_X = np.column_stack([model_predictions[k][:n] for k in self._model_names])
        pred = self._meta_model.predict(meta_X)
        # B-1: clip to [0, 2.5 * y_train_max] to prevent Ridge
        # extrapolation blow-up on OOD holdout (winter peak).
        _cap = 2.5 * self._y_train_max if self._y_train_max > 0 else np.inf
        pred_clipped = np.clip(pred, 0.0, _cap)
        n_over = int(np.sum(pred > _cap))
        if n_over > 0:
            log.warning(
                f"  [Stacking] predict: clipped {n_over}/{len(pred)} values "
                f"to ceiling {_cap:.2f} (max raw={float(np.max(pred)):.2f})"
            )
        return pred_clipped


# ═══════════════════════════════════════════════════════════════
# 3. Blending -- Level 20
# ═══════════════════════════════════════════════════════════════

class BlendingEnsemble(BaseForecaster):
    """
 Blending (홀드아웃 기반 조합).

 Stacking과 유사하나 CV 대신 단순 홀드아웃 분할.
 validation set의 모델 예측을 그대로 메타 피처로 사용.
 더 빠르고 단순하지만 데이터 효율은 낮음.

 fix (P0-1): 38개 모델에서 p≈n 이므로 unregularized LinearRegression은
 발산한다 (로그: R²=-114, Bayesian-SEIR=-315, SVR-RBF=-72). RidgeCV(positive=True)
 로 교체하여 비음수 제약 + L2 정규화. α 탐색 범위를 넓혀 유효 자유도를 제한.
 """

    meta = ModelMeta(
        name="Ensemble-Blending",
        category="meta",
        level=20,
        min_data=100,
        description="Blending. Ridge(positive=True) 기반 홀드아웃 메타 조합.",
        dependencies=["sklearn"],
    )

    # : Positive-constrained Ridge α 탐색 범위.
    # 하한 0.1 이어도 p>n 에서는 계수 진폭이 크므로 1e3 까지 허용.
    _ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
    # : Stacking 과 동일한 sanity — R²<0.3 모델은 메타 피처 제외.
    _R2_FLOOR = 0.3

    def __init__(self):
        super().__init__()
        self._meta_model = None
        self._model_names: list[str] = []
        self._y_train_max: float = 0.0  # B-1: holdout extrapolation ceiling

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BlendingEnsemble:
        """val_predictions, val_actual 동일 인터페이스."""
        from sklearn.linear_model import RidgeCV

        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("BlendingEnsemble: val_predictions와 val_actual 필요")

        # P0-1: NEGATIVE_CONTROL 모델 원천 제외.
        val_predictions, _nc_dropped = _drop_negative_controls(val_predictions)
        if _nc_dropped:
            log.info("  [Blending] NEGATIVE_CONTROL 제외: %s", ", ".join(_nc_dropped))

        # : Stacking 과 동일한 R² 필터.
        _r2_by_model: dict[str, float] = {}
        for _name, _pred in val_predictions.items():
            _m = min(len(val_actual), len(_pred))
            if _m <= 1:
                continue
            _y = val_actual[:_m]
            _yhat = _pred[:_m]
            _ss_tot = float(np.sum((_y - np.mean(_y)) ** 2))
            if _ss_tot <= 1e-12:
                continue
            _r2_by_model[_name] = 1.0 - float(np.sum((_y - _yhat) ** 2)) / _ss_tot
        qualified = [k for k, v in _r2_by_model.items() if v >= self._R2_FLOOR]
        excluded = [k for k in val_predictions.keys() if k not in qualified]
        if len(qualified) < 2:
            log.warning(
                "  [Blending] R²≥%.2f 통과 %d개 (<2) → sanity 해제, 전체 투입",
                self._R2_FLOOR, len(qualified),
            )
            qualified = sorted(val_predictions.keys())
            excluded = []
        elif excluded:
            log.info(
                "  [Blending] R²<%.2f 제외 %d개: %s",
                self._R2_FLOOR, len(excluded),
                ", ".join(f"{k}(R²={_r2_by_model.get(k, float('nan')):.3f})" for k in excluded),
            )

        self._model_names = sorted(qualified)
        n = min(len(val_actual), *(len(val_predictions[k]) for k in self._model_names))
        p = len(self._model_names)

        meta_X = np.column_stack([val_predictions[k][:n] for k in self._model_names])
        meta_y = val_actual[:n]

        # P0-1: positive=True 로 음수 계수 제거, α 상한 1000 으로 p>>n 방어.
        # fit_intercept=True 는 유지 (편향 상쇄).
        self._meta_model = RidgeCV(
            alphas=self._ALPHAS,
            fit_intercept=True,
            scoring="neg_root_mean_squared_error",
        )
        # RidgeCV 는 positive 직접 인자가 없으므로 fit 후 음수 계수를 0 으로
        # 클리핑하고 재정규화. sklearn>=1.2 의 Ridge(positive=True) 는 L-BFGS-B
        # 최적화 경로라 RidgeCV 와 동일 α 를 못 쓴다 — two-stage 접근.
        self._meta_model.fit(meta_X, meta_y)

        coefs = np.asarray(self._meta_model.coef_, dtype=float)
        # Non-negativity projection (단순 clip 이후 스케일 복원)
        coefs_pos = np.clip(coefs, a_min=0.0, a_max=None)
        if coefs_pos.sum() > 1e-8:
            # 원래 합을 유지해 스케일 보존
            orig_sum = float(np.abs(coefs).sum())
            coefs_pos = coefs_pos * (orig_sum / coefs_pos.sum())
        self._meta_model.coef_ = coefs_pos
        # B-1: same ceiling-clip guard as Stacking.
        # Blending also blew up on holdout winter peak (R²=-2771).
        self._y_train_max = float(np.max(meta_y)) if len(meta_y) else 0.0
        self._fitted = True

        coef_map = dict(zip(self._model_names, coefs_pos))
        max_abs = float(np.max(np.abs(coefs_pos))) if len(coefs_pos) else 0.0
        log.info(
            f"  [Blending] RidgeCV α={self._meta_model.alpha_:.2f}, "
            f"max|coef|={max_abs:.3f} (positive-projected), "
            f"y_train_max={self._y_train_max:.2f}, p={p}, n={n}"
        )
        log.info(f"  [Blending] coefs: {', '.join(f'{k}={v:.3f}' for k, v in coef_map.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})
        n = min(len(v) for v in model_predictions.values())
        meta_X = np.column_stack([model_predictions[k][:n] for k in self._model_names])
        pred = self._meta_model.predict(meta_X)
        # B-1: ceiling clip to prevent OOD extrapolation blow-up.
        _cap = 2.5 * self._y_train_max if self._y_train_max > 0 else np.inf
        pred_clipped = np.clip(pred, 0.0, _cap)
        n_over = int(np.sum(pred > _cap))
        if n_over > 0:
            log.warning(
                f"  [Blending] predict: clipped {n_over}/{len(pred)} values "
                f"to ceiling {_cap:.2f} (max raw={float(np.max(pred)):.2f})"
            )
        return pred_clipped


# ═══════════════════════════════════════════════════════════════
# 4. BMA (Bayesian Model Averaging) -- Level 22
# ═══════════════════════════════════════════════════════════════

class BMAEnsemble(BaseForecaster):
    """
    BMA -- Bayesian Model Averaging.

    BIC 기반 가중: 모델 복잡도 페널티 적용.
    BIC_i = n * log(MSE_i) + k_i * log(n)
    weight_i ∝ exp(-0.5 * BIC_i)
    """

    meta = ModelMeta(
        name="Ensemble-BMA",
        category="meta",
        level=22,
        min_data=80,
        description="Bayesian Model Averaging. BIC 기반 모델 가중, 복잡도 페널티.",
    )

    # 범주별 근사 파라미터 수
    PARAM_PROXY = {"ts": 6, "linear": 4, "tree": 20, "dl": 100, "meta": 10}

    def __init__(self):
        super().__init__()
        self._weights: dict[str, float] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BMAEnsemble:
        """
        kwargs:
            val_predictions: dict[str, np.ndarray]
            val_actual: np.ndarray
            model_complexities: dict[str, int] -- 모델별 파라미터 수 (선택)
        """
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))
        model_complexities = kwargs.get("model_complexities", {})

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("BMAEnsemble: val_predictions와 val_actual 필요")

        n = len(val_actual)
        bic_map = {}

        for name, pred in val_predictions.items():
            m = min(n, len(pred))
            mse = float(np.mean((val_actual[:m] - pred[:m]) ** 2))
            mse = max(mse, 1e-12)  # log(0) 방지

            # 복잡도: 명시적 > PARAM_PROXY 매핑
            if name in model_complexities:
                k = model_complexities[name]
            else:
                # 모델 이름에서 범주 추정
                k = self.PARAM_PROXY.get("meta", 10)
                for cat, pk in self.PARAM_PROXY.items():
                    if cat in name.lower():
                        k = pk
                        break

            bic = m * np.log(mse) + k * np.log(m)
            bic_map[name] = bic

        # Softmax(-0.5 * BIC) -- 수치 안정성을 위해 max 빼기
        bic_arr = np.array(list(bic_map.values()))
        shifted = -0.5 * bic_arr - np.max(-0.5 * bic_arr)
        exp_w = np.exp(shifted)
        exp_w /= exp_w.sum()

        self._weights = dict(zip(bic_map.keys(), exp_w.tolist()))
        self._fitted = True

        log.info(f"  [Ensemble-BMA] 가중치: "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        n = min(len(v) for v in model_predictions.values())
        result = np.zeros(n)
        for name, pred in model_predictions.items():
            w = self._weights.get(name, 0)
            result += w * pred[:n]

        return np.maximum(result, 0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


# ═══════════════════════════════════════════════════════════════
# 5. NNLS (Non-Negative Least Squares) -- Level 23
# ═══════════════════════════════════════════════════════════════

class NNLSEnsemble(BaseForecaster):
    """
    NNLS -- Non-Negative Least Squares.

    scipy.optimize.nnls로 비음수 제약 하 최적 가중치.
    Blending의 음수 가중치 문제 해결.
    """

    meta = ModelMeta(
        name="Ensemble-NNLS",
        category="meta",
        level=23,
        min_data=80,
        description="NNLS 앙상블. 비음수 제약 최소자승법으로 안정적 가중치.",
    )

    def __init__(self):
        super().__init__()
        self._weights: dict[str, float] = {}
        self._model_names: list[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> NNLSEnsemble:
        """
        kwargs:
            val_predictions: dict[str, np.ndarray]
            val_actual: np.ndarray
        """
        from scipy.optimize import nnls

        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("NNLSEnsemble: val_predictions와 val_actual 필요")

        # P0-1: NEGATIVE_CONTROL 모델(예: TabularDNN) 원천 제외.
        val_predictions, _nc_dropped = _drop_negative_controls(val_predictions)
        if _nc_dropped:
            log.info("  [Ensemble-NNLS] NEGATIVE_CONTROL 제외: %s", ", ".join(_nc_dropped))

        # : raw R² < 0 인 모델은 NNLS 후보에서 제외
        # 근거: 2026-04-19 run 에서 Metapop-SEIR(raw R²=-1.32), Bayesian-SEIR(-1.30),
        # Rt-Augmented(0.66) 가 합쳐 NNLS 가중치 65% 를 차지 → 앙상블 품질 저하.
        # _R2_FLOOR 이하 모델은 조용히 0-weight 로 처리.
        _R2_FLOOR = 0.3
        eligible_names = []
        for k in sorted(val_predictions.keys()):
            p = np.asarray(val_predictions[k])
            m = min(len(val_actual), len(p))
            if m < 2:
                continue
            ya = val_actual[:m]
            pa = p[:m]
            ss_tot = float(np.sum((ya - ya.mean()) ** 2))
            if ss_tot <= 0:
                continue
            r2 = 1.0 - float(np.sum((ya - pa) ** 2)) / ss_tot
            if r2 >= _R2_FLOOR and np.all(np.isfinite(pa)):
                eligible_names.append(k)

        if not eligible_names:
            # 모두 floor 미만이면 기존 방식 유지 (fallback)
            log.warning(f"  [Ensemble-NNLS] 모든 모델 R² < {_R2_FLOOR} → 필터 비활성")
            eligible_names = sorted(val_predictions.keys())

        self._model_names = eligible_names
        n = min(len(val_actual), *(len(val_predictions[k]) for k in self._model_names))

        meta_X = np.column_stack([val_predictions[k][:n] for k in self._model_names])
        meta_y = val_actual[:n]

        raw_weights, _ = nnls(meta_X, meta_y)

        # 정규화 (합 = 1)
        w_sum = raw_weights.sum()
        if w_sum > 0:
            raw_weights = raw_weights / w_sum
        else:
            raw_weights = np.ones(len(self._model_names)) / len(self._model_names)

        self._weights = dict(zip(self._model_names, raw_weights.tolist()))
        # 제외된 모델은 0-weight 로 명시 (predict 에서 key miss 방지)
        for k in val_predictions:
            if k not in self._weights:
                self._weights[k] = 0.0
        self._fitted = True

        log.info(f"  [Ensemble-NNLS] 필터 후 {len(eligible_names)}개 모델 "
                 f"(R²≥{_R2_FLOOR}), 가중치: "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items() if v > 0.01)}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        n = min(len(v) for v in model_predictions.values())
        result = np.zeros(n)
        for name, pred in model_predictions.items():
            w = self._weights.get(name, 0)
            result += w * pred[:n]

        return np.maximum(result, 0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


# ═══════════════════════════════════════════════════════════════
# 5b. NNLS-Filtered (G-169, 2026-05-03) -- Level 23
# ═══════════════════════════════════════════════════════════════
# train_by_category.sh:71 ensemble 카테고리 list 의 Ensemble-NNLS-Filtered 가
# REGISTRY 에 없음 → 학습 silent skip. NNLSEnsemble 의 R² floor 를 Optuna HP
# 로 학습 (기본 NNLS 의 hardcoded 0.3 vs Filtered 는 [0.0, 0.7] search).
# 이걸로 더 보수적인 가중치 (모델 수 적게, 품질 ↑) 또는 inclusive (모델 수 많게)
# 학습 가능.

class NNLSFilteredEnsemble(NNLSEnsemble):
    """NNLS + 사용자/Optuna-tunable R² floor (G-169, D-4 deep module).

    `NNLSEnsemble` 의 hardcoded floor=0.3 (Q1 quartile guess) → kwargs 또는
    Optuna HP 로 학습 가능한 floor (0.0~0.7) 로 일반화. 후보 모델 set 의 R²
    분포에 따라 optimal floor 가 다름 — DL-only ensemble 시 floor=0.5,
    mixed (tree+linear) 시 floor=0.0 (모두 포함).

    Attributes:
        meta: name="Ensemble-NNLS-Filtered", category="meta", level=23, min_data=80.
        _weights (NNLSEnsemble inherited): dict[model_name, weight] (sum = 1).
        _model_names: filter 통과 모델 list.

    fit(X_train, y_train, **kwargs):
        kwargs:
            val_predictions (dict[str, np.ndarray]): model 별 val prediction.
            val_actual (np.ndarray): val ground truth.
            r2_floor (float, default 0.5): R² 최소 임계값. floor 이상 만 NNLS 후보.

        Algorithm:
          1. `_drop_negative_controls` (NEGATIVE_CONTROL 제외)
          2. R² floor filter (configurable, NNLSEnsemble 의 0.3 hardcoded vs 여기 0.5 default)
          3. 모두 floor 미만 → fallback (floor=0.0)
          4. scipy.optimize.nnls → 비음수 가중치
          5. Normalize (sum = 1)

        Raises:
            ValueError: val_predictions 또는 val_actual 비었을 때.
            ValueError: 모든 모델 무효 (NaN-only).

    Performance: O(n_models × n_val) — 1초 이내 (n_val ≤ 100).
    Side effects: log.info — filter 결과 + 가중치.

    Caller responsibility:
        - val_predictions 의 NaN/inf 사전 sanitize.
        - r2_floor 적정 값 (0.5 권장 — G-169 smoke test 검증).

    Example:
        >>> m = NNLSFilteredEnsemble()
        >>> m.fit(np.zeros((8, 1)), val_actual,
        ...       val_predictions={"good": pred_good, "bad": pred_bad},
        ...       val_actual=val_actual, r2_floor=0.5)
        >>> m.weights
        {'good': 0.88, 'bad': 0.0}

    See: G-169 (Ensemble-NNLS-Filtered 신규, ensemble 카테고리 11/11),
         NNLSEnsemble (parent class, hardcoded floor=0.3).
    """

    meta = ModelMeta(
        name="Ensemble-NNLS-Filtered",
        category="meta",
        level=23,
        min_data=80,
        description="NNLS with Optuna-tuned R² floor (0.0~0.7) — 후보 모델 quality-aware",
    )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """NNLS + R² floor (configurable, NNLSEnsemble hardcoded 0.3 → 일반화) (G-169, D-4).

        `NNLSEnsemble.fit` 와 동일하지만 r2_floor 가 kwargs 또는 Optuna HP 로 결정.

        Args:
            X_train: 미사용 (BaseForecaster 인터페이스 호환).
            y_train: 미사용 (BaseForecaster 인터페이스 호환).
            **kwargs:
                val_predictions (dict[str, np.ndarray]): 필수. model 별 val prediction.
                val_actual (np.ndarray): 필수. val ground truth.
                r2_floor (float, default 0.5): R² 최소 임계값 (0.0~0.7 권장).

        Returns:
            self (chain용).

        Raises:
            ValueError: val_predictions / val_actual 비었을 때.
            ValueError: 모든 모델 무효 (NaN-only) — fallback 후에도 0 모델.

        Performance: O(n_models × n_val) — 1초 이내 (n_val ≤ 100).
        Side effects:
            - log.info: filter 결과 + 가중치
            - self._weights / _model_names / _fitted 갱신

        Caller responsibility:
            - val_predictions 의 NaN/inf 사전 sanitize.
            - r2_floor 적정 (0.5 권장 — 후보 모델 분포 따라).

        See: G-169 (NNLS-Filtered 신규), NNLSEnsemble (parent, hardcoded floor=0.3).
        """
        from scipy.optimize import nnls

        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))
        # G-169: floor 가 kwargs 명시 (caller 가 Optuna best 넘겨주거나) 또는 default 0.5
        r2_floor = float(kwargs.get("r2_floor", 0.5))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("NNLSFilteredEnsemble: val_predictions/val_actual 필요")

        val_predictions, _nc_dropped = _drop_negative_controls(val_predictions)
        if _nc_dropped:
            log.info("  [Ensemble-NNLS-Filtered] NEGATIVE_CONTROL 제외: %s",
                     ", ".join(_nc_dropped))

        # Filter by R² floor (configurable, NNLSEnsemble 의 0.3 hardcoded 와 차이)
        eligible_names = []
        for k in sorted(val_predictions.keys()):
            p = np.asarray(val_predictions[k])
            m = min(len(val_actual), len(p))
            if m < 2:
                continue
            ya = val_actual[:m]; pa = p[:m]
            ss_tot = float(np.sum((ya - ya.mean()) ** 2))
            if ss_tot <= 0:
                continue
            r2 = 1.0 - float(np.sum((ya - pa) ** 2)) / ss_tot
            if r2 >= r2_floor and np.all(np.isfinite(pa)):
                eligible_names.append(k)

        if not eligible_names:
            log.warning(f"  [Ensemble-NNLS-Filtered] 모든 모델 R² < {r2_floor} "
                        "→ floor 0.0 으로 fallback")
            eligible_names = [k for k in sorted(val_predictions.keys())
                              if np.all(np.isfinite(np.asarray(val_predictions[k])))]
            if not eligible_names:
                raise ValueError("[Ensemble-NNLS-Filtered] 사용 가능한 모델 없음")

        self._model_names = eligible_names
        n = min(len(val_actual), *(len(val_predictions[k]) for k in self._model_names))
        meta_X = np.column_stack([val_predictions[k][:n] for k in self._model_names])
        meta_y = val_actual[:n]
        raw_weights, _ = nnls(meta_X, meta_y)
        w_sum = raw_weights.sum()
        if w_sum > 0:
            raw_weights = raw_weights / w_sum
        else:
            raw_weights = np.ones(len(self._model_names)) / len(self._model_names)
        self._weights = dict(zip(self._model_names, raw_weights.tolist()))
        for k in val_predictions:
            if k not in self._weights:
                self._weights[k] = 0.0
        self._fitted = True
        log.info(f"  [Ensemble-NNLS-Filtered] floor={r2_floor:.2f} → "
                 f"{len(eligible_names)} 모델, 가중치: "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items() if v > 0.01)}")
        return self


# ═══════════════════════════════════════════════════════════════
# 6. Temporal Weight Ensemble -- Level 24
# ═══════════════════════════════════════════════════════════════

class TemporalWeightEnsemble(BaseForecaster):
    """
    시간 가중 앙상블.

    최근 성능에 더 높은 가중치 → 분포 변화(distribution shift) 적응.
    지수감쇠: half_life=26주.
    """

    meta = ModelMeta(
        name="Ensemble-Temporal",
        category="meta",
        level=24,
        min_data=80,
        description="시간가중 앙상블. 최근 성능 기반 지수감쇠 가중.",
    )

    HALF_LIFE = 26

    def __init__(self):
        super().__init__()
        self._weights: dict[str, float] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> TemporalWeightEnsemble:
        """
        kwargs:
            val_predictions: dict[str, np.ndarray]
            val_actual: np.ndarray
        """
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("TemporalWeightEnsemble: val_predictions와 val_actual 필요")

        T = len(val_actual)
        # 시간 감쇠 가중치: 최근 시점일수록 높은 가중치
        decay = np.exp(-np.log(2) * (T - 1 - np.arange(T)) / self.HALF_LIFE)

        weighted_rmse = {}
        for name, pred in val_predictions.items():
            n = min(T, len(pred))
            sq_err = (val_actual[:n] - pred[:n]) ** 2
            # 감쇠 가중 MSE → RMSE
            w_mse = np.sum(decay[:n] * sq_err) / np.sum(decay[:n])
            weighted_rmse[name] = float(np.sqrt(w_mse))

        # Inverse weighted-RMSE 가중치
        inv_total = sum(1.0 / (r + 1e-8) for r in weighted_rmse.values())
        self._weights = {
            name: (1.0 / (rmse + 1e-8)) / inv_total
            for name, rmse in weighted_rmse.items()
        }
        self._fitted = True

        log.info(f"  [Ensemble-Temporal] 가중치: "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        n = min(len(v) for v in model_predictions.values())
        result = np.zeros(n)
        for name, pred in model_predictions.items():
            w = self._weights.get(name, 0)
            result += w * pred[:n]

        return np.maximum(result, 0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


# ═══════════════════════════════════════════════════════════════
# 7. Diversity Ensemble -- Level 25
# ═══════════════════════════════════════════════════════════════

class DiversityEnsemble(BaseForecaster):
    """
    다양성 기반 앙상블.

    정확도 + 예측 다양성(상관 기반) 결합.
    다른 모델과 상관이 낮은(diverse) 모델에 보너스 가중치.
    """

    meta = ModelMeta(
        name="Ensemble-Diversity",
        category="meta",
        level=25,
        min_data=80,
        description="다양성 앙상블. 정확도 × 예측 독립성 결합 가중.",
    )

    LAMBDA = 0.5

    def __init__(self):
        super().__init__()
        self._weights: dict[str, float] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> DiversityEnsemble:
        """
        kwargs:
            val_predictions: dict[str, np.ndarray]
            val_actual: np.ndarray
        """
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("DiversityEnsemble: val_predictions와 val_actual 필요")

        names = sorted(val_predictions.keys())
        n = min(len(val_actual), *(len(val_predictions[k]) for k in names))

        # RMSE 계산
        rmse_map = {}
        for name in names:
            pred = val_predictions[name][:n]
            rmse = float(np.sqrt(np.mean((val_actual[:n] - pred) ** 2)))
            rmse_map[name] = rmse

        # 상관 행렬 계산
        pred_matrix = np.column_stack([val_predictions[k][:n] for k in names])
        corr_matrix = np.corrcoef(pred_matrix, rowvar=False)
        # NaN 처리 (분산 0인 경우)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 다양성 점수: 1 - mean(|corr(i, j)|) for j != i
        n_models = len(names)
        diversity = {}
        for i, name in enumerate(names):
            if n_models > 1:
                abs_corrs = [abs(corr_matrix[i, j]) for j in range(n_models) if j != i]
                diversity[name] = 1.0 - float(np.mean(abs_corrs))
            else:
                diversity[name] = 1.0

        # 최종 가중치: inv_rmse * (1 + LAMBDA * diversity), 정규화
        raw_weights = {}
        for name in names:
            inv_rmse = 1.0 / (rmse_map[name] + 1e-8)
            raw_weights[name] = inv_rmse * (1.0 + self.LAMBDA * diversity[name])

        w_total = sum(raw_weights.values())
        self._weights = {name: w / w_total for name, w in raw_weights.items()}
        self._fitted = True

        log.info(f"  [Ensemble-Diversity] 가중치: "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        n = min(len(v) for v in model_predictions.values())
        result = np.zeros(n)
        for name, pred in model_predictions.items():
            w = self._weights.get(name, 0)
            result += w * pred[:n]

        return np.maximum(result, 0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


# ═══════════════════════════════════════════════════════════════
# 8. SelectiveBMA -- Top-K 모델만 사용하는 BMA (Level 26)
# ═══════════════════════════════════════════════════════════════

class SelectiveBMAEnsemble(BaseForecaster):
    """
    Selective BMA -- Val R² 상위 K개 모델만 투입.

    일반 BMA가 저성능 모델(DNN R²=0.78, TCN R²=0.63 등)까지 포함하여
    가중 평균이 희석되는 문제를 해결.
    Ref: Hoeting et al. (1999) Bayesian Model Averaging: A Tutorial, Statistical Science.
    """

    meta = ModelMeta(
        name="Ensemble-SelectiveBMA",
        category="meta",
        level=26,
        min_data=80,
        description="Top-K 선별 BMA. Val R²≥threshold 모델만 BIC 가중.",
    )

    PARAM_PROXY = {"ts": 6, "linear": 4, "tree": 20, "dl": 100, "meta": 10}

    # : variance sanity — 예측 분산 / 실측 분산 비율이 이 값 이하면
    # "near-constant 예측" 으로 간주해 제외. TinyMLP 처럼 val 평균을 복제하면
    # Val R² 는 높아 보이지만 test (분포 이동) 에선 실패하는 패턴 차단.
    _VAR_RATIO_FLOOR = 0.10

    def __init__(self, r2_threshold: float = 0.85, max_models: int = 5):
        super().__init__()
        self._weights: dict[str, float] = {}
        self._r2_threshold = r2_threshold
        self._max_models = max_models
        self._selected_models: list[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "SelectiveBMAEnsemble":
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("SelectiveBMAEnsemble: val_predictions와 val_actual 필요")

        n = len(val_actual)

        # 1) Val R² + variance ratio 계산.
        # : variance ratio 는 예측이 near-constant 일 때 낮다. TinyMLP 가
        # val mean 을 복제해 R²=0.877 을 받지만 test 에선 −0.93 이 되는 현상이
        # 전형적 증상. 분산이 작으면 외삽 능력 부재 신호 → 제외.
        val_r2: dict[str, float] = {}
        var_ratio: dict[str, float] = {}
        for name, pred in val_predictions.items():
            m = min(n, len(pred))
            _y = val_actual[:m]
            _yhat = np.asarray(pred[:m], dtype=float)
            ss_res = float(np.sum((_y - _yhat) ** 2))
            ss_tot = float(np.sum((_y - np.mean(_y)) ** 2))
            val_r2[name] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            y_var = float(np.var(_y))
            var_ratio[name] = float(np.var(_yhat)) / y_var if y_var > 1e-12 else 0.0

        # threshold 이상 + variance 충분 + 상위 K개
        qualified = {
            k: v for k, v in val_r2.items()
            if v >= self._r2_threshold and var_ratio.get(k, 0.0) >= self._VAR_RATIO_FLOOR
        }
        rejected_by_var = [
            k for k, v in val_r2.items()
            if v >= self._r2_threshold and var_ratio.get(k, 0.0) < self._VAR_RATIO_FLOOR
        ]
        if rejected_by_var:
            log.info(
                "  [Selective-BMA] var-ratio<%.2f 제외 %d개 (near-constant): %s",
                self._VAR_RATIO_FLOOR, len(rejected_by_var),
                ", ".join(f"{k}(R²={val_r2[k]:.3f}, var={var_ratio[k]:.3f})"
                          for k in rejected_by_var),
            )
        sorted_models = sorted(qualified.keys(), key=lambda k: qualified[k], reverse=True)
        self._selected_models = sorted_models[:self._max_models]

        if len(self._selected_models) < 2:
            # fallback: 분산 sanity 통과한 상위 3개 (그래도 없으면 그냥 상위 3)
            sorted_all = sorted(
                [k for k in val_r2.keys()
                 if var_ratio.get(k, 0.0) >= self._VAR_RATIO_FLOOR],
                key=lambda k: val_r2[k], reverse=True,
            )
            if len(sorted_all) < 2:
                sorted_all = sorted(val_r2.keys(), key=lambda k: val_r2[k], reverse=True)
            self._selected_models = sorted_all[:3]

        log.info(f"  [Selective-BMA] 선택된 모델({len(self._selected_models)}개): "
                 f"{', '.join(f'{m}(R²={val_r2[m]:.3f})' for m in self._selected_models)}")

        # 2) 선택된 모델에 대해 BIC 가중
        bic_map = {}
        for name in self._selected_models:
            pred = val_predictions[name]
            m = min(n, len(pred))
            mse = max(float(np.mean((val_actual[:m] - pred[:m]) ** 2)), 1e-12)

            k = self.PARAM_PROXY.get("meta", 10)
            for cat, pk in self.PARAM_PROXY.items():
                if cat in name.lower():
                    k = pk
                    break

            bic = m * np.log(mse) + k * np.log(m)
            bic_map[name] = bic

        bic_arr = np.array(list(bic_map.values()))
        shifted = -0.5 * bic_arr - np.max(-0.5 * bic_arr)
        exp_w = np.exp(shifted)
        exp_w /= exp_w.sum()

        self._weights = dict(zip(bic_map.keys(), exp_w.tolist()))
        self._fitted = True

        log.info(f"  [Selective-BMA] 가중치: "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        selected_preds = {k: v for k, v in model_predictions.items() if k in self._weights}
        if not selected_preds:
            raise ValueError("선택된 모델 예측값 없음")

        n = min(len(v) for v in selected_preds.values())
        result = np.zeros(n)
        for name, pred in selected_preds.items():
            w = self._weights.get(name, 0)
            result += w * pred[:n]

        return np.maximum(result, 0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


# ═══════════════════════════════════════════════════════════════
# 9. ResidualCorrectedEnsemble -- 잔차 AR(1) 보정 앙상블 (Level 27)
# ═══════════════════════════════════════════════════════════════

class ResidualCorrectedEnsemble(BaseForecaster):
    """
    잔차 AR(1) 보정 앙상블.

    1단계: SelectiveBMA로 기본 예측 생성
    2단계: Val 잔차의 AR(1) 패턴을 학습
    3단계: Test 예측에 잔차 보정 적용

    DW < 1.0 문제 해결: 잔차 자기상관을 활용하여 다음 시점 오차 예측.
    Ref: Chatfield (2000) Time-Series Forecasting, Chapman & Hall.
    """

    meta = ModelMeta(
        name="Ensemble-ResidualAR",
        category="meta",
        level=27,
        min_data=80,
        description="잔차 AR(1) 보정 앙상블. DW<1 자기상관 캡처로 예측 보정.",
    )

    def __init__(self, r2_threshold: float = 0.85, max_models: int = 5):
        super().__init__()
        self._bma = SelectiveBMAEnsemble(r2_threshold=r2_threshold, max_models=max_models)
        self._ar_coef = 0.0
        self._ar_intercept = 0.0
        self._last_val_residual = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "ResidualCorrectedEnsemble":
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        # 1단계: SelectiveBMA 학습
        self._bma.fit(X_train, y_train, **kwargs)

        # 2단계: Val 잔차 계산
        val_bma_pred = self._bma.predict(X_train, model_predictions=val_predictions)
        # val_predictions는 val set에 대한 것이므로 val_actual과 비교
        n = min(len(val_actual), len(val_bma_pred))
        # BMA predict는 model_predictions 크기에 맞춰지므로, val_predictions를 넣어야 함
        val_bma_pred = self._bma.predict(None, model_predictions=val_predictions)
        n = min(len(val_actual), len(val_bma_pred))
        residuals = val_actual[:n] - val_bma_pred[:n]

        # 3단계: AR(1) 학습 -- r_{t} = a * r_{t-1} + b
        if len(residuals) > 2:
            r_prev = residuals[:-1]
            r_curr = residuals[1:]
            # OLS for AR(1)
            X_ar = np.column_stack([r_prev, np.ones(len(r_prev))])
            try:
                beta = np.linalg.lstsq(X_ar, r_curr, rcond=None)[0]
                self._ar_coef = float(np.clip(beta[0], -0.95, 0.95))  # 안정성 제한
                self._ar_intercept = float(beta[1])
            except Exception:
                self._ar_coef = 0.0
                self._ar_intercept = 0.0

            self._last_val_residual = float(residuals[-1])
        else:
            self._ar_coef = 0.0
            self._ar_intercept = 0.0
            self._last_val_residual = 0.0

        self._fitted = True
        log.info(f"  [ResidualAR] AR(1) coef={self._ar_coef:.4f}, "
                 f"intercept={self._ar_intercept:.4f}, "
                 f"last_residual={self._last_val_residual:.4f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})

        # 기본 BMA 예측
        base_pred = self._bma.predict(X_test, model_predictions=model_predictions)

        # AR(1) 잔차 보정 -- 순차적으로 적용
        n = len(base_pred)
        correction = np.zeros(n)
        prev_resid = self._last_val_residual

        for t in range(n):
            expected_resid = self._ar_coef * prev_resid + self._ar_intercept
            correction[t] = expected_resid
            # 다음 시점: 실제 잔차를 모르므로 예측된 잔차를 사용
            prev_resid = expected_resid

        corrected = base_pred + correction
        return np.maximum(corrected, 0)

    @property
    def weights(self) -> dict[str, float]:
        return self._bma.weights


# ═══════════════════════════════════════════════════════════════
# 10. AdaptiveWeightEnsemble -- COVID-era 분포 이동 보정 (Level 28)
# ═══════════════════════════════════════════════════════════════

class AdaptiveWeightEnsemble(BaseForecaster):
    """
    적응형 가중 앙상블 -- 최근 성능 기반 동적 가중.

    Val set 전체 RMSE 대신, Val set 후반부(최근 k주)의 RMSE를 가중하여
    COVID-era 분포 이동에 적응. 학습 데이터와 테스트 데이터의 분포가
    다를 때(Train mean/Test mean ratio ~3.4x) 최근 패턴에 더 가중.

    Ref: Timmermann (2006) Forecast Combinations, Handbook of Economic Forecasting.
    """

    meta = ModelMeta(
        name="Ensemble-Adaptive",
        category="meta",
        level=28,
        min_data=80,
        description="적응형 가중 앙상블. 최근 Val 성능 가중으로 분포 이동 대응.",
    )

    def __init__(self, recent_ratio: float = 0.4, r2_threshold: float = 0.80):
        super().__init__()
        self._weights: dict[str, float] = {}
        self._recent_ratio = recent_ratio
        self._r2_threshold = r2_threshold
        self._selected_models: list[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "AdaptiveWeightEnsemble":
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("AdaptiveWeightEnsemble: val_predictions와 val_actual 필요")

        n = len(val_actual)
        n_recent = max(int(n * self._recent_ratio), 5)

        # Val R² 기반 필터링
        val_r2 = {}
        for name, pred in val_predictions.items():
            m = min(n, len(pred))
            ss_res = np.sum((val_actual[:m] - pred[:m]) ** 2)
            ss_tot = np.sum((val_actual[:m] - np.mean(val_actual[:m])) ** 2)
            val_r2[name] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        qualified = {k: v for k, v in val_r2.items() if v >= self._r2_threshold}
        if len(qualified) < 2:
            sorted_all = sorted(val_r2.keys(), key=lambda k: val_r2[k], reverse=True)
            self._selected_models = sorted_all[:5]
        else:
            self._selected_models = sorted(qualified.keys(), key=lambda k: qualified[k], reverse=True)[:7]

        # 최근 구간 + 전체 구간 RMSE 혼합 가중
        combined_rmse = {}
        for name in self._selected_models:
            pred = val_predictions[name][:n]
            # 전체 RMSE
            full_rmse = float(np.sqrt(np.mean((val_actual - pred) ** 2)))
            # 최근 RMSE (더 중요)
            recent_rmse = float(np.sqrt(np.mean((val_actual[-n_recent:] - pred[-n_recent:]) ** 2)))
            # 혼합: 최근 70% + 전체 30%
            combined_rmse[name] = 0.3 * full_rmse + 0.7 * recent_rmse

        # Inverse combined RMSE 가중
        inv_total = sum(1.0 / (r + 1e-8) for r in combined_rmse.values())
        self._weights = {name: (1.0 / (combined_rmse[name] + 1e-8)) / inv_total
                        for name in self._selected_models}
        self._fitted = True

        log.info(f"  [Adaptive] 선택 모델({len(self._selected_models)}개): "
                 f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items())}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        selected_preds = {k: v for k, v in model_predictions.items() if k in self._weights}
        if not selected_preds:
            raise ValueError("선택된 모델 예측값 없음")

        n = min(len(v) for v in selected_preds.values())
        result = np.zeros(n)
        for name, pred in selected_preds.items():
            w = self._weights.get(name, 0)
            result += w * pred[:n]

        return np.maximum(result, 0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


# ═══════════════════════════════════════════════════════════════
# 11. TopTierStackingEnsemble -- R²≥threshold 필터 + Ridge 메타학습 (Level 29)
# ═══════════════════════════════════════════════════════════════

class TopTierStackingEnsemble(BaseForecaster):
    """
    Top-Tier Stacking -- Val R² ≥ threshold 모델만 Ridge 메타학습.

    Option D: 현재 상위 R² 모델들만 모아서 Ridge(alpha=100) 로 최적 선형
    결합을 학습. SelectiveBMA(BIC 가중)의 보완재로, 잔차 상관이 큰 Tier A
    모델들에 대해 더 보수적인 가중치를 산출한다.

    - 임계값 이상 모델이 2개 미만이면 상위 3개로 폴백.
    - Ridge alpha=100 으로 다중공선성 억제 (BIC/NNLS 대비 안정).
    - max_models 제한으로 과대 스태킹 방지.

    Ref: Breiman (1996) Stacked Regressions, Machine Learning 24.
         Van der Laan et al. (2007) Super Learner, Stat. Appl. Gen. Mol. Bio.
    """

    meta = ModelMeta(
        name="Ensemble-TopTierStacking",
        category="meta",
        level=29,
        min_data=80,
        description="R²≥0.85 상위 모델 Ridge 스태킹. SelectiveBMA의 Ridge 보완재.",
    )

    def __init__(self, r2_threshold: float = 0.85, max_models: int = 7, alpha: float = 100.0):
        super().__init__()
        self._r2_threshold = float(r2_threshold)
        self._max_models = int(max_models)
        self._alpha = float(alpha)
        self._selected_models: list[str] = []
        self._coef_: dict[str, float] = {}
        self._intercept_: float = 0.0
        self._ridge = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TopTierStackingEnsemble":
        val_predictions = kwargs.get("val_predictions", {})
        val_actual = kwargs.get("val_actual", np.array([]))

        if not val_predictions or len(val_actual) == 0:
            raise ValueError("TopTierStackingEnsemble: val_predictions와 val_actual 필요")

        n = len(val_actual)

        # 1) Val R² 계산 후 임계값 필터
        val_r2 = {}
        for name, pred in val_predictions.items():
            m = min(n, len(pred))
            pa = np.asarray(pred[:m], dtype=float)
            if not np.all(np.isfinite(pa)):
                continue
            ss_res = float(np.sum((val_actual[:m] - pa) ** 2))
            ss_tot = float(np.sum((val_actual[:m] - np.mean(val_actual[:m])) ** 2))
            val_r2[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        qualified = {k: v for k, v in val_r2.items() if v >= self._r2_threshold}
        sorted_models = sorted(qualified.keys(), key=lambda k: qualified[k], reverse=True)
        self._selected_models = sorted_models[:self._max_models]

        if len(self._selected_models) < 2:
            sorted_all = sorted(val_r2.keys(), key=lambda k: val_r2[k], reverse=True)
            self._selected_models = sorted_all[:3]
            log.warning(f"  [Top-Tier-Stacking] R²≥{self._r2_threshold:.2f} 후보 <2, "
                        f"상위 3개 폴백: {self._selected_models}")

        log.info(f"  [Top-Tier-Stacking] 선택 모델({len(self._selected_models)}개): "
                 f"{', '.join(f'{m}(R²={val_r2[m]:.3f})' for m in self._selected_models)}")

        # 2) Ridge 메타학습: X_meta=모델별 val 예측, y=val_actual
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            log.warning("  [Top-Tier-Stacking] sklearn 없음 → 균등가중 폴백")
            w = 1.0 / len(self._selected_models)
            self._coef_ = {m: w for m in self._selected_models}
            self._intercept_ = 0.0
            self._fitted = True
            return self

        X_meta = np.column_stack([
            np.asarray(val_predictions[m][:n], dtype=float) for m in self._selected_models
        ])
        y_meta = np.asarray(val_actual[:n], dtype=float)

        # NaN/Inf 방어
        finite_mask = np.all(np.isfinite(X_meta), axis=1) & np.isfinite(y_meta)
        if finite_mask.sum() < max(5, len(self._selected_models) + 1):
            log.warning("  [Top-Tier-Stacking] 유효 샘플 부족 → 균등가중 폴백")
            w = 1.0 / len(self._selected_models)
            self._coef_ = {m: w for m in self._selected_models}
            self._intercept_ = 0.0
            self._fitted = True
            return self

        X_meta = X_meta[finite_mask]
        y_meta = y_meta[finite_mask]

        self._ridge = Ridge(alpha=self._alpha, fit_intercept=True, positive=False)
        self._ridge.fit(X_meta, y_meta)

        self._coef_ = dict(zip(self._selected_models, self._ridge.coef_.tolist()))
        self._intercept_ = float(self._ridge.intercept_)
        self._fitted = True

        log.info(f"  [Top-Tier-Stacking] Ridge(α={self._alpha}) 계수: "
                 f"{', '.join(f'{k}={v:+.3f}' for k, v in self._coef_.items())} "
                 f"+ β₀={self._intercept_:+.3f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        model_predictions = kwargs.get("model_predictions", {})
        if not model_predictions:
            raise ValueError("model_predictions 필요")

        available = [m for m in self._selected_models if m in model_predictions]
        if not available:
            raise ValueError("선택된 모델의 테스트 예측값 없음")

        n = min(len(model_predictions[m]) for m in available)
        result = np.full(n, self._intercept_, dtype=float)
        for m in available:
            result += self._coef_.get(m, 0.0) * np.asarray(model_predictions[m][:n], dtype=float)

        return np.maximum(result, 0.0)

    @property
    def weights(self) -> dict[str, float]:
        """후방호환: Ridge 계수를 weights 로 노출 (음수 가능)."""
        return dict(self._coef_)


# ── 등록 ──
# 2026-05-26 Sprint D1 (사용자 명시 "MERGE-drop 모델 다 없애버려"):
# Ensemble-SelectiveBMA / Ensemble-Temporal / Ensemble-Blending 모두 REGISTRY 제거.
# 클래스 body 는 audit history 위해 보존 — 다음 sprint 에서 archive 가능.
REGISTRY.register(InverseRMSEEnsemble)
REGISTRY.register(BMAEnsemble)
REGISTRY.register(NNLSEnsemble)
REGISTRY.register(NNLSFilteredEnsemble)   # G-169 (2026-05-03)
REGISTRY.register(DiversityEnsemble)
REGISTRY.register(ResidualCorrectedEnsemble)
REGISTRY.register(AdaptiveWeightEnsemble)
# MERGE-dropped (Sprint D1 2026-05-26):
# REGISTRY.register(StackingEnsemble)        # extra, empty metrics, overfit
# REGISTRY.register(BlendingEnsemble)        # MERGE-drop, empty metrics, duplicate
# REGISTRY.register(TemporalWeightEnsemble)  # MERGE-drop → Ensemble-InvRMSE (R²=0.870 cluster)
# REGISTRY.register(SelectiveBMAEnsemble)    # MERGE-drop → Ensemble-BMA (threshold = config)
# REGISTRY.register(TopTierStackingEnsemble) # extra, no result, dup of Stacking


# Package C B-B: Temperature softmin stacking weights
def package_c_softmin_weights(oof_wis_per_model, temperature: float = 0.5):
    """Convert per-model OOF WIS scores → ensemble weights via softmin.

    weights ∝ exp(-WIS_oof / T)
    Lower WIS (better) → higher weight. T controls sharpness.
      T → 0  : winner-takes-all (단일 best)
      T → ∞  : uniform (mean)
      T = 0.5: 권장 (중간)
    """
    import numpy as _np_pc
    scores = _np_pc.asarray(oof_wis_per_model, dtype=float)
    # numerical stability: subtract min
    scaled = -(scores - scores.min()) / max(temperature, 1e-6)
    e = _np_pc.exp(scaled)
    return e / e.sum()


def package_c_softmin_ensemble(predictions, oof_wis, temperature: float = 0.5):
    """Predictions: array (n_models, n_samples). oof_wis: per-model scalar score."""
    import numpy as _np_pc
    w = package_c_softmin_weights(oof_wis, temperature)
    return _np_pc.einsum("ms,m->s", _np_pc.asarray(predictions), w), w
