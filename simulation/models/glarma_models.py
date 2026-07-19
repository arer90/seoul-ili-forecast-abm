"""
simulation/models/glarma_models.py
===================================
GLARMA (Generalized Linear ARMA) for ILI count data — Davis et al. (2003).

[학술 배경]
GLARMA = GLM with autoregressive and moving average error in mean:
  η_t = X_t β + Z_t
  Z_t = Σ_p φ_p (Y_{t-p} - μ_{t-p})/√Var(Y_{t-p}) + Σ_q θ_q ε_{t-q}
  Y_t | Z_t ~ NegBin(μ_t, ψ),  log μ_t = η_t

[Davis 2003, Dunsmuir & Scott 2015]
- Pearson residual feedback (AR component)
- standardized error MA component
- Negative binomial or Poisson family

[구현 결정]
- 본 구현: statsmodels GLM (NegBin) + manually injected AR(1) Pearson lag.
  (MA component omitted — pure AR(1) GLARMA-lite)
- 정통 GLARMA python package는 PyPI 미제공 (R glarma 만).

[참조]
- Davis RA, Dunsmuir WTM, Streett SB (2003). "Observation-driven models for
  Poisson counts". Biometrika 90(4):777-790.
- Dunsmuir WTM, Scott DJ (2015). "The glarma Package for Observation-Driven
  Time Series Regression of Counts". JSS 67(7):1-36.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


class GLARMAForecaster(BaseForecaster):
    """GLARMA-lite (NegBin GLM + AR(1) Pearson residual feedback).

    Davis et al. (2003) — observation-driven Poisson/NegBin time series.
    본 구현: statsmodels GLM 으로 NegBin fit + AR(1) Pearson residual lag.
    """

    meta = ModelMeta(
        name="GLARMA", category="epi", level=5, min_data=60,
        description=(
            "GLARMA-lite: NegBin GLM + AR(1) Pearson residual feedback "
            "(Davis 2003)."
        ),
        dependencies=["statsmodels"],
    )

    def __init__(self, topk: int = 12, alpha_init: float = 1.0):
        super().__init__()
        self._glm_res = None
        self._scaler_X = None
        self._feat_idx = None
        self._phi = 0.0  # AR(1) coefficient
        self._mu_last = 0.0
        self._resid_last = 0.0
        self._topk = int(topk)
        self._alpha = float(alpha_init)
        self._y_max = 100.0
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "GLARMAForecaster":
        import statsmodels.api as sm
        from sklearn.feature_selection import f_regression
        from sklearn.preprocessing import StandardScaler
        from simulation.models.base import _validate_shapes
        _validate_shapes(X_train, y_train, name="GLARMA.fit", min_n=30)

        self._y_max = float(np.max(y_train))

        # Standardize + top-K
        self._scaler_X = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        K = min(self._topk, max(5, len(y_train) // 8))
        if X_s.shape[1] > K:
            try:
                scores, _ = f_regression(X_s, y_train)
                self._feat_idx = np.argsort(-np.abs(np.nan_to_num(scores)))[:K]
                X_s = X_s[:, self._feat_idx]
            except Exception:
                self._feat_idx = np.arange(min(K, X_s.shape[1]))
                X_s = X_s[:, self._feat_idx]

        # First-pass GLM (no AR feedback)
        X_design = sm.add_constant(X_s, has_constant="add")
        y_int = np.maximum(np.round(y_train).astype(int), 0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                glm0 = sm.GLM(y_int, X_design,
                                  family=sm.families.NegativeBinomial(alpha=self._alpha))
                res0 = glm0.fit(maxiter=100, disp=False)
            mu0 = np.maximum(res0.predict(X_design), 1e-3)
            # Pearson residuals
            var0 = mu0 + self._alpha * mu0 ** 2  # NegBin variance
            pearson = (y_train - mu0) / np.sqrt(var0 + 1e-6)

            # AR(1) feedback: extend design with lag-1 pearson residual
            pearson_lag1 = np.concatenate([[0.0], pearson[:-1]])
            X_design2 = np.column_stack([X_design, pearson_lag1.reshape(-1, 1)])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                glm1 = sm.GLM(y_int, X_design2,
                                  family=sm.families.NegativeBinomial(alpha=self._alpha))
                self._glm_res = glm1.fit(maxiter=100, disp=False)
            # Coefficients
            self._phi = float(self._glm_res.params[-1])
            self._mu_last = float(np.maximum(self._glm_res.predict(X_design2)[-1], 1e-3))
            self._resid_last = float(pearson[-1])
            self._fitted = True
            log.info(f"  [GLARMA] φ_AR1 = {self._phi:.3f}, K={X_s.shape[1]}")
        except Exception as e:
            log.warning(f"  [GLARMA] fit 실패: {e} → fallback mean")
            self._glm_res = None
            self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        import statsmodels.api as sm
        from simulation.models.base import sanitize_predictions

        n_test = len(X_test)
        if not self._fitted or self._glm_res is None:
            return sanitize_predictions(np.full(n_test, self._mu_last))

        X_s = self._scaler_X.transform(X_test)
        if self._feat_idx is not None:
            X_s = X_s[:, self._feat_idx]
        X_design = sm.add_constant(X_s, has_constant="add")

        # G-327 (2026-06-20, 사용자: rolling): GLARMA = observation-driven (Davis 2003) — 관측 y 주면
        #   매주 pearson 잔차를 **관측값**으로 재계산해 AR(1) feedback 갱신(=올바른 1-step). 없으면
        #   resid_lag 동결(legacy 단일원점 = error feedback 없어 static 외삽 발산, G-319b cap 으로 미봉).
        _obs = (np.asarray(y_observed, dtype=np.float64)
                if y_observed is not None and len(y_observed) == n_test else None)
        # AR(1) iterative
        preds = []
        resid_lag = self._resid_last
        for i in range(n_test):
            design_i = np.concatenate([X_design[i], [resid_lag]])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mu = float(self._glm_res.predict(design_i.reshape(1, -1))[0])
            except Exception:
                mu = self._mu_last
            preds.append(mu)
            if _obs is not None:
                # observation-driven: 관측 y[i] 와 예측 μ[i] 로 pearson 잔차 갱신 → i+1 의 AR(1) lag.
                #   i+1 예측에 _obs[:i+1] 만 쓰임 = leak-free 1-step.
                _var_i = mu + self._alpha * mu ** 2
                resid_lag = float((_obs[i] - mu) / np.sqrt(_var_i + 1e-6))
            # else: resid_lag 동결 (legacy)
        # G-319b: cap 5×→2× (NB 가족 정합). static AR 반복외삽(error feedback 없음)이 위로
        #   발산(pmax=167.3 실측) — 2× 로 bound. 근본 발산은 rolling 1-step 평가로 회피.
        pred = np.clip(np.array(preds), 0.0, self._y_max * 2.0)
        return sanitize_predictions(pred)


try:
    REGISTRY.register(GLARMAForecaster)
    log.info("[glarma_models] GLARMAForecaster 등록됨 (Davis 2003)")
except Exception as _e:
    log.warning(f"[glarma_models] 등록 skip: {_e}")
