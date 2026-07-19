"""
simulation/models/ts_models.py
==============================
시계열(Time Series) 범주 모델: SARIMA, SARIMAX

- SARIMA: 단변량 계절 ARIMA (기준선)
- SARIMAX: 외생변수 포함 SARIMA (기상, COVID 더미 등)

ILI rate(‰) 주간 시계열 전용.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from simulation.models.base import (
    BaseForecaster, TimeSeriesForecaster, ModelMeta, REGISTRY,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. SARIMA -- Level 0 (기준선)
# ═══════════════════════════════════════════════════════════════

class SARIMAForecaster(TimeSeriesForecaster):
    """
    Seasonal ARIMA -- 전통 시계열 기준선.

    log1p 변환 후 학습, 복수 (p,d,q)(P,D,Q,52) 후보 중 AIC 최소 선택.
    """

    meta = ModelMeta(
        name="SARIMA",
        category="ts",
        level=0,
        min_data=104,
        description="계절 ARIMA 기준선. 자기상관+계절성을 ARIMA 프레임워크로 모델링.",
        dependencies=["statsmodels"],
    )

    def __init__(self):
        super().__init__()
        self._fit_result = None
        self._seasonal_period = 52

    def fit_series(self, series: np.ndarray, **kwargs) -> SARIMAForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # transform-fix (2026-06-21): internal log1p REMOVED — fit on the RAW series (identity).
        #   G-271 intent (positivity so D=1 seasonal differencing does not extrapolate negative →
        #   forecast clip(0) 0-floor) is now supplied by the preproc layer: the single y-transform
        #   is DATA-DRIVEN by the preproc Optuna search, not hardcoded here, so there is no internal
        #   log1p/expm1 round-trip that exploded on out-of-range peaks. forecast/rolling just clip≥0.
        self._y_max = float(np.max(series)) if len(series) else 100.0  # G-284: 외삽 cap 기준
        train_series = np.maximum(series.astype(float), 0.0)

        orders = []
        # G-025: sp=52는 MLE 수렴 극도로 느림 (70분+) → sp=13,26만 사용
        for sp in [26, 13]:
            if len(series) < sp * 2 + 10:
                continue
            orders.extend([
                ((1, 1, 1), (1, 1, 1, sp)),
                ((1, 0, 1), (1, 1, 0, sp)),
                ((2, 1, 1), (1, 1, 1, sp)),
                ((0, 1, 1), (0, 1, 1, sp)),
                ((1, 1, 0), (1, 1, 0, sp)),
                ((3, 1, 1), (1, 0, 1, sp)),
            ])
        # 비계절 ARIMA도 포함 (fallback)
        orders.extend([
            ((2, 1, 2), (0, 0, 0, 1)),
            ((3, 1, 2), (0, 0, 0, 1)),
        ])

        best_fit, best_aic = None, np.inf
        # G-303 (2026-06-17, 검증 적발): SARIMAX(G-290)와 동일 수렴-guard. 옛 SARIMA 는
        #   mle_retvals['converged'] 미확인 → 비수렴(낮은 AIC) fit 을 champion 으로 silently 수용
        #   가능(SARIMA 는 reported baseline). 수렴 fit 별도 추적 후 우선.
        best_conv_fit, best_conv_aic, best_conv_seasonal = None, np.inf, None
        for order, seasonal in orders:
            try:
                model = SARIMAX(
                    train_series, order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=200)
                _conv = bool(getattr(fit, "mle_retvals", {}) and
                             fit.mle_retvals.get("converged", True))
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
                    self._seasonal_period = seasonal[-1]
                if _conv and fit.aic < best_conv_aic:
                    best_conv_aic = fit.aic
                    best_conv_fit = fit
                    best_conv_seasonal = seasonal[-1]
            except Exception:
                continue

        # 수렴 fit 우선(비수렴 silently 수용 방지); 없으면 best-AIC + 경고
        if best_conv_fit is not None:
            best_fit, best_aic = best_conv_fit, best_conv_aic
            self._seasonal_period = best_conv_seasonal
        elif best_fit is not None:
            log.warning("  [SARIMA] 수렴한 order 없음 → best-AIC(비수렴) fallback")

        if best_fit is None:
            raise RuntimeError("SARIMA: 모든 후보 order 실패")

        self._fit_result = best_fit
        self._fitted = True
        log.info(f"  [SARIMA] AIC={best_aic:.1f}, order={best_fit.specification['order']}, "
                 f"seasonal={best_fit.specification['seasonal_order']}")
        return self

    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("SARIMA: 학습되지 않음")
        # transform-fix (2026-06-21): forecast on the RAW series (no expm1). The single y-transform
        #   is supplied by the preproc layer; here we only enforce the domain floor (≥0) and a 2×
        #   raw-space upper cap as an extrapolation backstop (G-284 intent, now in linear space).
        _fc = np.asarray(self._fit_result.forecast(steps=steps), dtype=float)
        _cap = 2.0 * getattr(self, "_y_max", 100.0)  # 보수적 2× 상한
        return np.clip(_fc, 0.0, _cap)

    def rolling_1step(self, y_observed: np.ndarray, **kwargs) -> np.ndarray:
        """G-321: statsmodels append(refit=False) rolling 1-step — 싸다(filter only, refit X). 각 test
        주를 관측 과거로 1주 예측 = feature 모델과 동일 task(공정). A/B: SARIMA static −1.01 → rolling +0.86.

        transform-fix (2026-06-21): RAW-series rolling (no log1p/expm1) — the single y-transform is
        now supplied by the preproc layer; rolling only clips ≥0 and applies the 2× raw upper cap."""
        if self._fit_result is None:
            return super().rolling_1step(y_observed, **kwargs)
        _cap = 2.0 * getattr(self, "_y_max", 100.0)
        preds = np.full(len(y_observed), np.nan, dtype=float)
        ext = self._fit_result
        for i in range(len(y_observed)):
            try:
                fc = float(np.asarray(ext.forecast(1)).ravel()[0])
                preds[i] = float(np.clip(fc, 0.0, _cap))
                ext = ext.append([max(float(y_observed[i]), 0.0)], refit=False)
            except Exception:
                preds[i] = float(y_observed[i - 1]) if i > 0 else 0.0
        return np.clip(preds, 0.0, _cap)


# ═══════════════════════════════════════════════════════════════
# 2. SARIMAX -- Level 1 (외생변수 포함)
# ═══════════════════════════════════════════════════════════════

class SARIMAXForecaster(BaseForecaster):
    """
    SARIMAX -- 외생변수(기상, COVID 더미 등) 포함 계절 ARIMA.

    fit(X_train, y_train): X_train이 외생변수 행렬
    predict(X_test): X_test 외생변수로 예측
    """

    meta = ModelMeta(
        name="SARIMAX",
        category="ts",
        level=1,
        min_data=104,
        description="외생변수 포함 계절 ARIMA. 기상·COVID 더미 등 공변량 활용.",
        dependencies=["statsmodels"],
    )

    def __init__(self):
        super().__init__()
        self._fit_result = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SARIMAXForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # G-025: sp=52는 341주에서 MLE 수렴 극도로 느림 (70분+)
        # SARIMA auto-detect가 sp=13 선택 → SARIMAX도 13 사용
        sp = kwargs.get("seasonal_period", 13)
        # : runner에서 log1p 변환 담당 → raw y 직접 사용
        y_log = y_train.astype(float)
        self._y_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-284: 외삽 cap 기준

        # 외생변수: 피처 수가 과다하면 처음 10개만 사용 (과적합 방지)
        max_exog = min(X_train.shape[1], 10)
        exog = X_train[:, :max_exog] if max_exog > 0 else None
        self._n_exog = max_exog

        orders = [
            ((1, 1, 1), (1, 1, 1, sp)),
            ((1, 0, 1), (1, 1, 0, sp)),
            ((2, 1, 1), (0, 1, 1, sp)),
        ]

        best_fit, best_aic = None, np.inf
        # G-290 (2026-06-17, 3자 감사): 수렴한 fit 우선 — 옛 코드는 mle_retvals['converged'] 미확인 →
        #   비수렴(낮은 AIC) fit 을 champion 으로 silently 수용 → test R² 회귀. 수렴 fit 별도 추적.
        best_conv_fit, best_conv_aic = None, np.inf
        for order, seasonal in orders:
            try:
                model = SARIMAX(
                    y_log, exog=exog,
                    order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=200)
                _conv = bool(getattr(fit, "mle_retvals", {}) and
                             fit.mle_retvals.get("converged", True))
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
                if _conv and fit.aic < best_conv_aic:
                    best_conv_aic = fit.aic
                    best_conv_fit = fit
            except Exception:
                continue

        # 수렴한 fit 이 있으면 그걸 채택(비수렴 우선 방지); 없으면 best-AIC + 경고
        if best_conv_fit is not None:
            best_fit, best_aic = best_conv_fit, best_conv_aic
        elif best_fit is not None:
            log.warning("  [SARIMAX] 수렴한 order 없음 → best-AIC(비수렴) fallback")

        if best_fit is None:
            raise RuntimeError("SARIMAX: 모든 후보 order 실패")

        self._fit_result = best_fit
        self._fitted = True
        log.info(f"  [SARIMAX] AIC={best_aic:.1f}")
        return self

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("SARIMAX: 학습되지 않음")
        n_exog = getattr(self, "_n_exog", X_test.shape[1])
        exog = X_test[:, :n_exog] if n_exog > 0 else None
        _cap = 2.0 * getattr(self, "_y_max", 0.0)   # G-284: 상한 cap (외삽 발산 가드)
        # G-321 (2026-06-19): y_observed 주면 rolling 1-step(exog 포함, append refit=False) = feature
        #   모델과 동일 task(공정). 각 step 관측 과거+exog로 1주 예측. 없으면 단일원점(legacy).
        if y_observed is not None and len(y_observed) == len(X_test):
            preds = np.full(len(X_test), np.nan, dtype=float)
            ext = self._fit_result
            for i in range(len(X_test)):
                ex_i = exog[i:i + 1] if exog is not None else None
                try:
                    preds[i] = float(np.asarray(ext.forecast(1, exog=ex_i)).ravel()[0])
                    ext = ext.append([float(y_observed[i])], exog=ex_i, refit=False)
                except Exception:
                    preds[i] = float(y_observed[i - 1]) if i > 0 else 0.0
            return np.clip(preds, 0.0, _cap if _cap > 0 else None)
        # : runner에서 역변환 담당 → expm1 제거. SARIMAX exog × 차분 후 발산(R²=-5.4e10) 차단(G-180/284).
        forecast_vals = self._fit_result.forecast(steps=len(X_test), exog=exog)
        return np.clip(np.asarray(forecast_vals), 0.0, _cap if _cap > 0 else None)


# ═══════════════════════════════════════════════════════════════
# 3. ARIMA -- KUIRB §4-(2) 공식 baseline
# ═══════════════════════════════════════════════════════════════

class ARIMAForecaster(TimeSeriesForecaster):
    """ARIMA(p,d,q) — 계절성 없는 순수 ARIMA. KUIRB 계획서의 공식 baseline.

    R `forecast::auto.arima()` 의 Python statsmodels 대응. 계획서
    "기준선 모델로 ARIMA를 설정한다" (§4-(2) 비교군 설정) 정합.
    """
    meta = ModelMeta(
        name="ARIMA",
        category="ts",
        level=0,
        min_data=52,
        description="ARIMA(p,d,q) KUIRB 공식 baseline — R auto.arima 대응.",
        dependencies=["statsmodels"],
    )

    def __init__(self):
        super().__init__()
        self._fit_result = None

    def fit_series(self, series: np.ndarray, **kwargs) -> "ARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = series.astype(float)
        # R auto.arima-style grid — 계절 차수 제거
        orders = [
            (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (3, 1, 1),
            (0, 1, 1), (1, 1, 0), (2, 0, 2), (3, 1, 2), (1, 0, 1),
        ]
        best_fit, best_aic = None, np.inf
        for order in orders:
            try:
                mdl = SARIMAX(
                    y, order=order, seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                fit = mdl.fit(disp=False, maxiter=200)
                if fit.aic < best_aic:
                    best_aic, best_fit = fit.aic, fit
            except Exception:
                continue
        if best_fit is None:
            raise RuntimeError("ARIMA: 모든 후보 order 실패")
        self._fit_result = best_fit
        self._fitted = True
        log.info(f"  [ARIMA] AIC={best_aic:.1f}, order={best_fit.specification['order']}")
        return self

    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("ARIMA: 학습되지 않음")
        # G-180 P2 (2026-05-05): clip(0, ...) ILI 도메인 제약
        # ARIMA 차분 후 base level 복원 실패 → negative pred 차단
        pred = self._fit_result.forecast(steps=steps)
        return np.clip(np.asarray(pred), 0.0, None)

    def rolling_1step(self, y_observed: np.ndarray, **kwargs) -> np.ndarray:
        """G-321: statsmodels append(refit=False) rolling 1-step — raw 공간(ARIMA는 log1p 미사용).
        각 test 주를 관측 과거로 1주 예측 = feature 모델과 동일 task(공정). A/B: static −0.89 → rolling +0.86."""
        if self._fit_result is None:
            return super().rolling_1step(y_observed, **kwargs)
        preds = np.full(len(y_observed), np.nan, dtype=float)
        ext = self._fit_result
        for i in range(len(y_observed)):
            try:
                preds[i] = float(np.asarray(ext.forecast(1)).ravel()[0])
                ext = ext.append([float(y_observed[i])], refit=False)
            except Exception:
                preds[i] = float(y_observed[i - 1]) if i > 0 else 0.0
        return np.clip(preds, 0.0, None)


# ═══════════════════════════════════════════════════════════════
# 4. Theta — Assimakopoulos & Nikolopoulos (2000) M3-winning baseline
# ═══════════════════════════════════════════════════════════════

class ThetaForecaster(TimeSeriesForecaster):
    """Theta method (Assimakopoulos & Nikolopoulos 2000) — M3-winning baseline.

    Decomposes the series into two ``θ-lines``:
      - ``θ=0``: long-term linear regression trend
      - ``θ=2``: short-term curvature, forecast via SES on the detrended part
    The final forecast is the average of both, with the trend reconstructed.

    statsmodels' ``ThetaModel`` implements the standard variant with optional
    deseasonalisation (multiplicative or additive). Setting ``deseasonalize=True``
    detects seasonality via the classical decomposition; we keep the period at
    ``52`` for weekly ILI data.

    Paper relevance:
        Sprint S3 (2026-05-26) — added because every M3/M4/M5 forecasting
        competition uses Theta as a top-tier baseline. Excluding it from the
        comparison panel invites the "why no Theta?" reviewer question.
    """

    meta = ModelMeta(
        name="Theta",
        category="ts",
        level=0,
        min_data=52,
        description=(
            "Assimakopoulos & Nikolopoulos 2000 Theta method — M3 winner. "
            "Decomposes into θ=0 trend + θ=2 SES; classical baseline."
        ),
        dependencies=["statsmodels"],
    )

    def __init__(self):
        super().__init__()
        self._fit_result = None
        self._period = 52
        self._b_alpha = None   # G-338: B-style fit-once rolling state gate (α/seasonal/trend)

    def fit_series(self, series: np.ndarray, **kwargs) -> "ThetaForecaster":
        from statsmodels.tsa.forecasting.theta import ThetaModel

        # Runner handles log1p/inverse — pass raw series.
        y = np.asarray(series, dtype=float)
        if y.size < self.meta.min_data:
            raise RuntimeError(
                f"Theta: insufficient data n={y.size} < min_data={self.meta.min_data}"
            )

        # Pick period: weekly (52) when long enough, else fall back to short
        # variants (26 or 13). ThetaModel needs n ≥ 2·period to deseasonalise.
        candidate_periods = [52, 26, 13, 4]
        for p in candidate_periods:
            if y.size >= 2 * p:
                self._period = p
                break
        else:
            self._period = 1  # no seasonality — degenerate case

        # ThetaModel: deseasonalize=True when period > 1 and data length supports it.
        try:
            model = ThetaModel(
                y,
                period=self._period if self._period > 1 else None,
                deseasonalize=self._period > 1,
                method="auto",  # auto-pick additive vs multiplicative
            )
            self._fit_result = model.fit()
        except Exception as e:
            # Fallback: no deseasonalisation (period=None) for very short series
            try:
                model = ThetaModel(y, period=None, deseasonalize=False)
                self._fit_result = model.fit()
                self._period = 1
            except Exception as e2:
                raise RuntimeError(f"Theta: fit failed even without seasonality: {e2}") from e

        # G-338 (2026-06-24, 사용자): B-style fit-once rolling state — estimate α/seasonal/trend
        #   ONCE here so rolling_1step feeds OBSERVED values through FIXED params (no per-origin
        #   refit). Makes the eval panel a UNIFORM fit-once + rolling-observed protocol (§8.6),
        #   symmetric with statsmodels append(refit=False) (ARIMA/SARIMA) + epi observed-feed.
        #   Additive deseasonalize = zero-safe (ILI has 0-rate weeks; multiplicative crashes).
        #   Verified ≈ legacy per-origin refit (max|Δ|≈1.1, R² 0.938 vs 0.934; α window-stable Δα<0.011).
        try:
            p = int(self._period)
            try:
                a = float(self._fit_result.params["alpha"])
            except Exception:
                a = float(np.ravel(np.asarray(self._fit_result.params))[-1])
            a = float(min(max(a, 1e-3), 0.999))
            if p > 1 and y.size >= 2 * p:
                from statsmodels.tsa.seasonal import seasonal_decompose
                seas = np.asarray(
                    seasonal_decompose(y, period=p, model="additive",
                                       extrapolate_trend="freq").seasonal, dtype=float)
                cyc = np.array([seas[j::p].mean() for j in range(p)], dtype=float)
                cyc = cyc - cyc.mean()
            else:
                cyc = np.zeros(max(p, 1), dtype=float)
            idx = np.arange(y.size)
            des = y - cyc[idx % len(cyc)]
            self._b_b0 = float(np.polyfit(idx, des, 1)[0]) if y.size >= 3 else 0.0
            lvl = float(des[0])
            for v in des[1:]:
                lvl = a * float(v) + (1.0 - a) * lvl
            self._b_alpha, self._b_cycle, self._b_level, self._b_n = a, cyc, lvl, int(y.size)
        except Exception as _be:
            self._b_alpha = None   # rolling_1step → base refit fallback (safe)
            log.debug(f"  [Theta] B-state unavailable ({_be}); rolling falls back to refit")

        self._fitted = True
        log.info(f"  [Theta] period={self._period} (Assimakopoulos & Nikolopoulos 2000)")
        return self

    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("Theta: not fitted")
        fc = np.asarray(self._fit_result.forecast(steps=int(steps)), dtype=float)
        # G-180 P2: clip(0, ...) ILI domain constraint (rate ≥ 0).
        return np.clip(fc, 0.0, None)

    def rolling_1step(self, y_observed: np.ndarray, **kwargs) -> np.ndarray:
        """G-338: B-style fit-once rolling 1-step — fixed α/seasonal/trend, observed-feed.

        Symmetric-refit (§8.6, 2026-06-24, 사용자): params estimated ONCE in fit_series; each
        origin feeds the OBSERVED series through the FIXED-α SES level recursion + fixed additive
        seasonal + fixed Theta drift — **NO per-origin parameter re-estimation**. Mirrors
        statsmodels append(refit=False) (ARIMA/SARIMA) + epi observed-feed → the whole eval panel
        is now a uniform fit-once + rolling-observed protocol (배포 충실; Tashman 2000 = eval가
        실제 배포 조건 복제). Theta 가 패널의 마지막 A-style(매 origin 재추정) 잔존이었음.

        Verified ≈ legacy per-origin refit (max|Δ|≈1.1, R² 0.938 vs 0.934) because α is
        window-stable (Δα<0.011). Falls back to base refit if B-state unavailable.

        Args:
            y_observed: (n_test,) hold-out 관측 y (raw; Theta=identity transform). i 예측엔
                y_observed[:i] 만 사용 → leak-free 1-step.

        Returns:
            (n_test,) 1-step rolling 예측 (clip ≥ 0).
        """
        a = getattr(self, "_b_alpha", None)
        if a is None:
            return super().rolling_1step(y_observed, **kwargs)   # safe fallback = base refit
        yo = np.asarray(y_observed, dtype=float)
        cyc = np.asarray(self._b_cycle, dtype=float)
        p = len(cyc)
        b0 = float(self._b_b0)
        lvl = float(self._b_level)
        n = int(self._b_n)
        preds = np.full(len(yo), np.nan, dtype=float)
        for i in range(len(yo)):
            drift = 0.5 * b0 * ((1.0 - (1.0 - a) ** n) / a)
            preds[i] = max(lvl + drift + float(cyc[n % p]), 0.0)
            lvl = a * (float(yo[i]) - float(cyc[n % p])) + (1.0 - a) * lvl
            n += 1
        return preds


# ── 등록 ──
REGISTRY.register(ARIMAForecaster)
REGISTRY.register(SARIMAForecaster)
REGISTRY.register(SARIMAXForecaster)
REGISTRY.register(ThetaForecaster)
