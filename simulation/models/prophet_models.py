"""
simulation/models/prophet_models.py
====================================
Facebook PROPHET (Taylor & Letham 2018) — ILI rate forecasting.

[학술 배경]
Prophet (Taylor & Letham 2018, The American Statistician 72(1)):
  y(t) = g(t) + s(t) + h(t) + ε
where:
  - g(t): piecewise linear/logistic trend
  - s(t): Fourier seasonality (yearly + weekly)
  - h(t): holiday effects
  - ε: i.i.d. noise

[ILI rate 적용]
- 시계열 (date, ili_rate) → Prophet fit
- forecast horizon = len(X_test)
- covariates 무시 (Prophet 은 univariate; extra_regressors 추가 가능)

[참조]
- Taylor SJ, Letham B (2018). "Forecasting at scale". TAS 72(1):37-45.
- prophet package: https://github.com/facebook/prophet
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """PROPHET (Taylor & Letham 2018) univariate ILI forecaster.

    fit: pd.DataFrame(ds, y) → Prophet.fit
    predict: make_future_dataframe(periods=n_test) → forecast
    """

    meta = ModelMeta(
        name="PROPHET",
        category="epi",
        level=6,
        min_data=80,
        description=(
            "Facebook PROPHET (Taylor & Letham 2018). Bayesian additive "
            "regression: trend + Fourier seasonality + holidays."
        ),
        dependencies=["prophet"],
    )

    def __init__(self, yearly_seasonality: bool = True,
                  weekly_seasonality: bool = False,
                  changepoint_prior_scale: float = 0.05):
        super().__init__()
        self._prophet = None
        self._yearly = bool(yearly_seasonality)
        self._weekly = bool(weekly_seasonality)
        self._cp_scale = float(changepoint_prior_scale)
        self._last_date = None
        self._y_max = 100.0
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "ProphetForecaster":
        from prophet import Prophet
        from simulation.models.base import _validate_shapes
        _validate_shapes(X_train, y_train, name="PROPHET.fit", min_n=30)

        # Date index — weekly cadence
        n = len(y_train)
        dates = pd.date_range("2018-01-01", periods=n, freq="W")
        self._last_date = dates[-1]
        self._y_max = float(np.max(y_train))

        df = pd.DataFrame({"ds": dates, "y": y_train.astype(np.float64)})

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Prophet 자체 log 끄기
                import logging as _lg
                _lg.getLogger("prophet").setLevel(_lg.WARNING)
                _lg.getLogger("cmdstanpy").setLevel(_lg.WARNING)
                m = Prophet(
                    yearly_seasonality=self._yearly,
                    weekly_seasonality=self._weekly,
                    daily_seasonality=False,
                    changepoint_prior_scale=self._cp_scale,
                )
                m.fit(df)
            self._prophet = m
            self._fitted = True
            log.info(f"  [PROPHET] n_changepoints={m.n_changepoints}, "
                       f"yearly={self._yearly}")
        except Exception as e:
            log.warning(f"  [PROPHET] fit 실패: {e} → mean fallback")
            self._prophet = None
            self._y_mean = float(np.mean(y_train))
            self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.base import sanitize_predictions

        n_test = len(X_test)
        if not self._fitted or self._prophet is None:
            return sanitize_predictions(np.full(n_test, getattr(self, "_y_mean", 1.0)))

        future_dates = pd.date_range(
            self._last_date + pd.Timedelta(weeks=1),
            periods=n_test, freq="W"
        )
        future_df = pd.DataFrame({"ds": future_dates})

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self._prophet.predict(future_df)
            pred = forecast["yhat"].to_numpy()
        except Exception as e:
            log.warning(f"  [PROPHET] predict 실패: {e}")
            return sanitize_predictions(np.full(n_test, getattr(self, "_y_mean", 1.0)))

        pred = np.clip(pred, 0.0, self._y_max * 5.0)
        return sanitize_predictions(pred)


# ═══════════════════════════════════════════════════════════════════════════
try:
    REGISTRY.register(ProphetForecaster)
    log.info("[prophet_models] ProphetForecaster 등록됨 (Taylor & Letham 2018)")
except Exception as _e:
    log.warning(f"[prophet_models] 등록 skip: {_e}")
