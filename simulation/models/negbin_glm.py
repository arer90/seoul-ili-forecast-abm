"""
simulation/models/negbin_glm.py
===================================
NegBinGLM V7 — true statsmodels Negative Binomial GLM.

V6 (epi_models.py) 의 salvage 는 RidgeCV(log1p(y)) 였다 (NB2 가 p=309, n=234 에서 divergence).
V7 는 진짜 NB-GLM 을 top-K feature selection + 자동 알파 추정으로 안정화시킨다.

구조:
  1. top-K |Pearson r| feature selection (기본 k=15)
  2. StandardScaler + 절편 (exog 에 col of 1s 추가)
  3. alpha 자동 추정: GLM Poisson 잔차로 Cameron-Trivedi aux regression
  4. sm.GLM(y, X, family=NegativeBinomial(alpha=est_alpha), link=log) 로 fit
  5. predict 는 exp(Xβ), train max 의 2배로 clip (V6 와 동일 안전장치)

Falls back to V6 salvage 결과 (RidgeCV log1p) 시에만 log.warning + metric 추적.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


def _w_glum_available() -> bool:
    """glum(진짜 NB-GLM) 설치 여부 — NegBinGLM-Glum 테스트/가드용."""
    try:
        import glum  # noqa: F401
        return True
    except ImportError:
        return False


class NegBinGLMForecaster(BaseForecaster):
    """True NB-GLM (statsmodels) with Cameron-Trivedi alpha estimation + top-K selection."""

    meta = ModelMeta(
        name="NegBinGLM-V7",
        category="epi",
        level=5,
        min_data=60,
        description="Negative Binomial GLM (statsmodels, top-K + auto-alpha, Cameron-Trivedi).",
        dependencies=["statsmodels"],
    )

    def __init__(self, topk: int = 15, alpha_max: float = 10.0):
        super().__init__()
        self._result = None
        self._scaler_X = None
        self._feat_idx = None
        self._topk = topk
        self._alpha_max = alpha_max
        self._est_alpha = 1.0
        self._y_train_max = 0.0
        self._fallback = False
        self._v6 = None  # V6 salvage fallback

    def _estimate_alpha(self, X: np.ndarray, y: np.ndarray) -> float:
        """Cameron & Trivedi (1990) auxiliary regression:
        ((y - μ)² - μ) / μ ~ α·μ   (linear through origin).
        Poisson GLM fit 으로 μ 얻고 OLS 로 α 추정.
        """
        try:
            import statsmodels.api as sm
            pois = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=200, disp=0)
            mu = np.clip(pois.fittedvalues, 1e-3, None)
            aux_y = ((y - mu) ** 2 - mu) / mu
            # OLS through origin
            aux = sm.OLS(aux_y, mu).fit()
            alpha = float(aux.params[0])
            # clip to sensible range
            return float(np.clip(alpha, 1e-4, self._alpha_max))
        except Exception as e:
            log.warning(f"  [NegBinGLM-V7] alpha 추정 실패: {e} → α=1.0 fallback")
            return 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "NegBinGLMForecaster":
        from sklearn.preprocessing import StandardScaler

        n_train, p_orig = X_train.shape
        k = min(self._topk, p_orig)

        # 1. top-K |corr| feature selection
        Xc = X_train - X_train.mean(axis=0, keepdims=True)
        yc = y_train - float(y_train.mean())
        num = (Xc * yc[:, None]).sum(axis=0)
        den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum() + 1e-12)
        corr = np.abs(num / np.maximum(den, 1e-12))
        self._feat_idx = np.argsort(-corr)[:k]
        X_sel = X_train[:, self._feat_idx]

        # 2. Standardize + intercept
        self._scaler_X = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_sel)
        try:
            import statsmodels.api as sm
        except ImportError:
            log.error("  [NegBinGLM-V7] statsmodels 없음 → V6 salvage fallback")
            return self._fit_v6_fallback(X_train, y_train)

        X_with_const = sm.add_constant(X_s, has_constant="add")

        # 3. y = ILI rate (continuous, 비음수). NB-GLM 은 본래 count 분포지만 여기선 rate 를
        #    **그대로 사용**(반올림/offset 없음 — 변수명 y_int 는 역사적, 실제 float rate).
        #    statsmodels IRLS 는 float y 허용하나 count 우도를 rate 에 적용 = mild misspecification.
        #    검토 2026-06-05 (3-LLM): 엄밀히는 raw count + log(노출) offset 또는 rate-모델이 정석.
        #    현 champion 은 경험적 우수(test WIS 최저)하나 논문엔 이 근사를 명시·정당화 필요.
        y_int = np.maximum(y_train, 0).astype(float)

        # 4. alpha 자동 추정
        self._est_alpha = self._estimate_alpha(X_with_const, y_int)

        # 5. NB-GLM fit with IRLS
        try:
            fam = sm.families.NegativeBinomial(alpha=self._est_alpha)
            self._result = sm.GLM(y_int, X_with_const, family=fam).fit(
                maxiter=500, scale=1.0, disp=0, tol=1e-6
            )
            self._y_train_max = float(np.max(y_train))
            self._fitted = True

            # sanity check: train fit R²
            pred_train = self._result.predict(X_with_const)
            ss_res = float(np.sum((y_int - pred_train) ** 2))
            ss_tot = float(np.sum((y_int - y_int.mean()) ** 2))
            r2_train = 1 - ss_res / max(ss_tot, 1e-9)
            log.info(
                f"  [NegBinGLM-V7] top-K={k}/{p_orig}, α={self._est_alpha:.4f}, "
                f"train R²={r2_train:.4f}, converged={self._result.converged}, "
                f"deviance={float(self._result.deviance):.2f}"
            )
            if not self._result.converged or r2_train < 0.3:
                log.warning(f"  [NegBinGLM-V7] 수렴 불안정 → V6 salvage fallback")
                return self._fit_v6_fallback(X_train, y_train)
            return self
        except Exception as e:
            log.warning(f"  [NegBinGLM-V7] GLM fit 실패: {e} → V6 salvage fallback")
            return self._fit_v6_fallback(X_train, y_train)

    def _fit_v6_fallback(self, X_train: np.ndarray, y_train: np.ndarray) -> "NegBinGLMForecaster":
        """V6 salvage (RidgeCV + log1p) — NB divergence 시 안전망."""
        from simulation.models.epi_models import NegBinGLMForecaster
        self._v6 = NegBinGLMForecaster(topk=20)
        self._v6.fit(X_train, y_train)
        self._fallback = True
        self._used_fallback = True   # G-286 (3자 감사): 공개 플래그 — V7 이 V6 로 fallback(=V6 중복)임을
        #   artifact/리포트서 식별 가능하게(옛 _fallback 만으론 silent → V7≡V6 인지 알 수 없었음).
        self._fitted = True
        log.warning("  [NegBinGLM-V7] true-NB 미수렴 → V6 salvage fallback (이 run 의 V7 = V6 와 동일)")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("NegBinGLM-V7: fit() 먼저 호출")
        if self._fallback:
            return self._v6.predict(X_test)

        import statsmodels.api as sm
        X_sel = X_test[:, self._feat_idx]
        X_s = self._scaler_X.transform(X_sel)
        X_with_const = sm.add_constant(X_s, has_constant="add")
        pred = self._result.predict(X_with_const)
        # cap at 2 × train max (NB predict 발산 방지)
        cap = 2.0 * self._y_train_max
        return np.clip(pred, 0.0, cap)

    def predict_interval(
        self, X_test: np.ndarray, alpha: float = 0.05, n_samples: int = 2000, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tier C ⑦ — native NB predictive interval via parametric bootstrap.

 For each test row:
 μ_i = self._result.predict(x_i) (conditional mean)
 α_nb = self._est_alpha (NB2 overdispersion)
 Draw N_i ~ NB(μ_i, α_nb) via scipy.stats.nbinom
 where size r = 1/α, prob p = r/(r+μ)
 (lo_i, hi_i) = empirical α/2, 1-α/2 quantiles.

 Falls back to V6 log-normal approximation when V7 itself fell back
 to the V6 salvage path. The interval is capped at 2·y_train_max
 to prevent runaway NB tails in extrapolation regimes.
 """
        if not self._fitted:
            raise RuntimeError("NegBinGLM-V7: fit() 먼저 호출")
        if self._fallback:
            return self._v6.predict_interval(X_test, alpha=alpha)

        import statsmodels.api as sm
        from scipy.stats import nbinom
        X_sel = X_test[:, self._feat_idx]
        X_s = self._scaler_X.transform(X_sel)
        X_with_const = sm.add_constant(X_s, has_constant="add")
        mu = np.clip(self._result.predict(X_with_const), 1e-6, 2.0 * self._y_train_max)

        r = 1.0 / max(self._est_alpha, 1e-6)          # NB2 size parameter
        p = r / (r + mu)                               # success probability
        rng = np.random.RandomState(2026)
        # scipy nbinom.rvs accepts broadcast p (shape = n_test); sample n_samples cols
        samples = nbinom.rvs(r, p[:, None], size=(len(mu), int(n_samples)), random_state=rng)
        lo = np.quantile(samples, alpha / 2.0, axis=1).astype(float)
        hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=1).astype(float)

        cap = 2.0 * self._y_train_max if self._y_train_max > 0 else np.inf
        lo = np.clip(lo, 0.0, cap)
        hi = np.clip(hi, 0.0, cap)
        return lo, hi


class GlumNBForecaster(BaseForecaster):
    """True Negative-Binomial GLM via glum (elastic-net, FULL pool) — G-263 (2026-06-13).

    V7(NegBinGLMForecaster)와 같은 NB2 우도지만 **접근이 다름**: V7 은 hard top-K |Pearson r| cut
    으로 p>n 을 통제(feature 버림), 본 모델은 **glum 의 L1+L2 elastic-net 연속 shrinkage 로 전체
    feature pool 을 통제**(정보 손실 적음). log link 라 자연 비음수 + peak 외삽 가능(실측 ILI test
    r2=0.878, 예측 max 105.5 > train max — 트리 cap 능가). 사용자 add 확정(2026-06-13).

    Caller responsibility: y ≥ 0 (rate; NB count 우도를 rate 에 적용 = V7 과 동일 mild
    misspecification — 논문에 명시·정당화 필요). Performance: O(n·p·iter) IRLS, glum 은 numba-fast.
    """

    meta = ModelMeta(
        name="NegBinGLM-Glum",
        category="epi",
        level=5,
        min_data=60,
        description="Negative Binomial GLM (glum, elastic-net full-pool, log link). "
                    "V7 의 hard top-K 대신 연속 L1/L2 shrinkage — peak 외삽 가능 (G-263).",
        dependencies=["glum"],
    )

    def __init__(self, l1_ratio: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._l1_ratio = float(l1_ratio)
        self._alpha = float(alpha)
        self._y_train_max = 0.0
        self._fallback = False
        self._v6 = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "GlumNBForecaster":
        from sklearn.preprocessing import StandardScaler
        # HP override (pipeline stage-3): l1_ratio / alpha
        self._l1_ratio = float(kwargs.get("l1_ratio", self._l1_ratio))
        self._alpha = float(kwargs.get("alpha", self._alpha))

        self._scaler_X = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y = np.clip(np.asarray(y_train, dtype=float), 1e-6, None)
        self._y_train_max = float(np.max(y_train))
        try:
            import warnings as _w
            from glum import GeneralizedLinearRegressor
            # G-319c (2026-06-19, 전체 라인업 감사): glum 이 registry-load multi-lib OpenMP 환경(macOS)서
            #   SEGFAULT→V6 fallback(3 NB CSV byte-identical 원인) — 진짜 test R²=0.882 모델을 잃고 있었음.
            #   G-303 은 process-level OMP=1(전체 단일스레드화)만 고려해 비채택했으나, **fit-시점
            #   threadpool_limits(1) 가 targeted 해법**: glum fit 동안만 BLAS/OpenMP 1-thread → SEGFAULT
            #   회피(heavy 프로세스 재현테스트서 train R²=0.952 검증), run 나머지는 multi-thread 유지.
            from threadpoolctl import threadpool_limits
            m = GeneralizedLinearRegressor(
                family="negative.binomial", alpha=self._alpha, l1_ratio=self._l1_ratio,
                fit_intercept=True, scale_predictors=False, max_iter=300,
            )
            # glum 의 정규화 path 탐색이 benign matmul over/underflow 경고를 다수 발생 — 결과는 유한·정상.
            # multi-day run 로그 오염 방지로 fit + sanity predict 둘 다 억제 (caller 영향 0).
            with threadpool_limits(limits=1), np.errstate(all="ignore"), _w.catch_warnings():
                _w.simplefilter("ignore")
                m.fit(X_s, y)
                pred_tr = np.asarray(m.predict(X_s), float)   # sanity predict 도 블록 안 (경고 누수 차단)
            ss_res = float(np.sum((y - pred_tr) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2_tr = 1 - ss_res / max(ss_tot, 1e-9)
            if not np.all(np.isfinite(pred_tr)) or r2_tr < 0.2:
                log.warning(f"  [NegBinGLM-Glum] 불안정(train R²={r2_tr:.3f}) → V6 salvage fallback")
                return self._fit_v6_fallback(X_train, y_train)
            self._model = m
            self._fitted = True
            log.info(f"  [NegBinGLM-Glum] elastic-net(l1={self._l1_ratio:.2f}, α={self._alpha:.3f}) "
                     f"train R²={r2_tr:.4f}")
            return self
        except Exception as e:
            log.warning(f"  [NegBinGLM-Glum] glum fit 실패: {e} → V6 salvage fallback")
            return self._fit_v6_fallback(X_train, y_train)

    def _fit_v6_fallback(self, X_train: np.ndarray, y_train: np.ndarray) -> "GlumNBForecaster":
        from simulation.models.epi_models import NegBinGLMForecaster as _V6
        self._v6 = _V6(topk=20); self._v6.fit(X_train, y_train)
        self._fallback = True; self._fitted = True
        self._used_fallback = True   # G-319c: V7 의 G-286 과 정합 — Glum 이 V6 로 fallback(=V6 중복)임을
        #   artifact/리포트서 식별 가능하게(threadpool_limits 로 이제 거의 fallback 안 하나, 안전망 유지).
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("NegBinGLM-Glum: fit() 먼저 호출")
        if self._fallback:
            return self._v6.predict(X_test)
        import warnings as _w
        X_s = self._scaler_X.transform(X_test)
        with np.errstate(all="ignore"), _w.catch_warnings():
            _w.simplefilter("ignore")
            pred = np.asarray(self._model.predict(X_s), float)
        cap = 2.0 * self._y_train_max if self._y_train_max > 0 else np.inf
        return np.clip(pred, 0.0, cap)   # NB predict 발산 방지 (V7 과 동일 2×cap)


# 레지스트리 등록
for _cls in (NegBinGLMForecaster, GlumNBForecaster):
    try:
        REGISTRY.register(_cls)
        log.info(f"[negbin_glm] {_cls.meta.name} 등록됨")
    except Exception as _e:
        log.debug(f"[negbin_glm] {_cls.meta.name} 등록 skip: {_e}")
