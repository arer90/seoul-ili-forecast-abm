"""
simulation/models/hhh4_models.py
=================================
hhh4-equivalent (Held & Paul 2012) NegBin GLM + seasonal harmonics + AR(1).

[학술 배경]
hhh4 (Held & Höhle 2014, Held & Paul 2012) — R surveillance package 의
endemic-epidemic 모델. 단일 region simplification:

  log E[Y_t] = λ_t + Σ_k (α_k cos(2πk t/52) + β_k sin(2πk t/52)) + φ × Y_{t-1}

where:
  - λ_t: time-varying baseline (intercept + covariates)
  - K seasonal harmonics (default K=2)
  - φ: AR(1) coefficient
  - Y_t ~ NegBin(μ_t, ψ)

[Paper 명시]
paper/methodology_real_forecast.md:210, 463:
  "hhh4_equivalent (S2-B) — NegBin GLM with K=2 seasonal harmonics + AR(1)"

[참조]
- Held L, Höhle M, Hofmann M (2005). "A statistical framework for the
  analysis of multivariate infectious disease surveillance counts".
  Statistical Modelling 5(3):187-199.
- Held L, Paul M (2012). "Modeling seasonality in space-time infectious
  disease surveillance data". Biometrical Journal 54(6):824-843.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


class Hhh4EquivalentForecaster(BaseForecaster):
    """hhh4-equivalent (Held & Paul 2012) — single-region NegBin GLM.

    Design:
      log μ_t = β_0 + β · X_t + Σ (α_k sin + γ_k cos) + φ Y_{t-1}
      Y_t ~ NegBin(μ_t, ψ)

    K=2 seasonal harmonics (period=52 weeks).
    AR(1) regressor: lagged Y_{t-1} (NegBin GLM 가 자동 처리).
    """

    meta = ModelMeta(
        name="hhh4-equivalent",
        category="epi",
        level=5,
        min_data=80,
        description=(
            "hhh4 (Held & Paul 2012) NegBin GLM + K=2 seasonal harmonics + "
            "AR(1). Single-region simplification of R surveillance::hhh4."
        ),
        dependencies=["statsmodels"],
    )

    def __init__(self, K_harmonics: int = 2, period: int = 52,
                  topk: int = 15, alpha_init: float = 1.0):
        super().__init__()
        self._glm_res = None
        self._scaler_X = None
        self._feat_idx = None
        self._K = int(K_harmonics)
        self._period = int(period)
        self._topk = int(topk)
        self._alpha_init = float(alpha_init)
        self._t_train_end = 0
        self._y_last = 0.0
        self._y_max = 100.0
        self._fitted = False

    def _seasonal_features(self, t: np.ndarray) -> np.ndarray:
        """K seasonal harmonics: sin/cos for k=1..K with period P."""
        feats = []
        for k in range(1, self._K + 1):
            feats.append(np.sin(2 * np.pi * k * t / self._period))
            feats.append(np.cos(2 * np.pi * k * t / self._period))
        return np.column_stack(feats)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "Hhh4EquivalentForecaster":
        import statsmodels.api as sm
        from sklearn.feature_selection import f_regression
        from sklearn.preprocessing import StandardScaler

        from simulation.models.base import _validate_shapes
        _validate_shapes(X_train, y_train, name="hhh4-equivalent.fit", min_n=60)

        n_train = len(y_train)
        self._t_train_end = n_train
        self._y_last = float(y_train[-1])
        self._y_max = float(np.max(y_train))

        # Standardize covariates
        self._scaler_X = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)

        # Top-K features (high-dim handling)
        K_feat = min(self._topk, max(5, n_train // 8))
        if X_s.shape[1] > K_feat:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores, _ = f_regression(X_s, y_train)
                self._feat_idx = np.argsort(-np.abs(np.nan_to_num(scores)))[:K_feat]
                X_s = X_s[:, self._feat_idx]
            except Exception:
                self._feat_idx = np.arange(min(K_feat, X_s.shape[1]))
                X_s = X_s[:, self._feat_idx]

        # Seasonal harmonics
        t = np.arange(n_train, dtype=np.float64)
        seasonal = self._seasonal_features(t)

        # AR(1) regressor: Y_{t-1}, first row use mean
        y_lag1 = np.concatenate([[float(np.mean(y_train))], y_train[:-1]])

        # G-278b (2026-06-16, 3자 감사): AR(1) 항을 log1p 로 — NB log-link 에서 raw y_prev 되먹임이
        #   μ=exp(β·y_prev) 지수폭발(val_wis 33367, 정상 sibling 3.6); 1.5× point cap 으로도 WIS 미해결.
        #   log1p → μ=(1+y_prev)^β power-law 로 안정(sibling hhh4_benchmark.py:69 와 동형).
        _ar_col = np.log1p(np.clip(y_lag1, 0, None))

        # Design matrix: [X_topk, seasonal, AR(1)=log1p, intercept]
        design = np.column_stack([X_s, seasonal, _ar_col.reshape(-1, 1)])
        design = sm.add_constant(design, has_constant="add")

        # NegBin GLM fit
        y_train_int = np.maximum(np.round(y_train).astype(int), 0)  # NegBin needs non-neg int
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                glm = sm.GLM(y_train_int, design,
                                family=sm.families.NegativeBinomial(alpha=self._alpha_init))
                self._glm_res = glm.fit(maxiter=100, disp=False)
            self._fitted = True
            log.info(f"  [hhh4-eq] NegBin GLM converged, K={self._K} harmonics, "
                       f"n_feat={X_s.shape[1] + 2*self._K + 1}")
        except Exception as e:
            log.warning(f"  [hhh4-eq] GLM fit 실패: {e} → fallback mean")
            self._glm_res = None
            self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        import statsmodels.api as sm
        from simulation.models.base import sanitize_predictions

        n_test = len(X_test)
        if not self._fitted or self._glm_res is None:
            return sanitize_predictions(np.full(n_test, self._y_last))

        X_s = self._scaler_X.transform(X_test)
        if self._feat_idx is not None:
            X_s = X_s[:, self._feat_idx]

        # Seasonal harmonics — continue time index from train_end
        t = np.arange(self._t_train_end, self._t_train_end + n_test,
                       dtype=np.float64)
        seasonal = self._seasonal_features(t)

        # AR(1): iteratively predict (first step uses _y_last)
        # G-327 (2026-06-20, 사용자: rolling): y_observed 주면 AR(1) 입력을 **관측값**으로(self-feeding
        #   누적 과소예측→음수 회피, 매주 1-step). 없으면 자기예측 y_prev(legacy 단일원점).
        _obs = (np.asarray(y_observed, dtype=np.float64)
                if y_observed is not None and len(y_observed) == n_test else None)
        preds = []
        y_prev = self._y_last
        for i in range(n_test):
            _ar = float(_obs[i - 1]) if (_obs is not None and i > 0) else y_prev
            # G-278b: AR(1) 입력도 log1p (fit 과 동형) — 되먹임 지수폭발 차단
            design_i = np.concatenate([X_s[i], seasonal[i], [np.log1p(max(_ar, 0.0))]])
            design_i = sm.add_constant(design_i.reshape(1, -1),
                                          has_constant="add", prepend=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mu = float(self._glm_res.predict(design_i)[0])
            except Exception:
                mu = y_prev
            # G-258b (2026-06-15): AR(1) y_prev 되먹임이 NB log-link 에서 exp 폭발
            # (val_wis 33367, 정상 sibling 3.6) → per-step cap 으로 중간 mu 를 묶어야
            # y_prev 누적 발산 차단. 최종 clip(아래 line)은 루프 종료 後라 미차단.
            # _y_max = fit y_train 파생(누수 0). 2026-06-16 G-274: 3× cap(G-258b)이 transform
            # 공간(예 HIER_categorical)서 AR(1) 누적 발산을 못 막음(val_wis 33367 잔존) → 1.5× 로
            # 조임. 1.5×y_max = ILI peak 여유는 유지하되 누적 폭발 강하게 차단.
            mu = min(mu, self._y_max * 1.5)
            preds.append(mu)
            y_prev = mu

        pred = np.array(preds, dtype=np.float64)
        pred = np.clip(pred, 0.0, self._y_max * 5.0)
        return sanitize_predictions(pred)


# ═══════════════════════════════════════════════════════════════════════════
try:
    REGISTRY.register(Hhh4EquivalentForecaster)
    log.info("[hhh4_models] Hhh4EquivalentForecaster 등록됨 (Held & Paul 2012)")
except Exception as _e:
    log.warning(f"[hhh4_models] 등록 skip: {_e}")
