"""
hhh4-style endemic-epidemic benchmark for ILI rate forecasting.
================================================================

The R `surveillance::hhh4` model (Held & Paul 2012, Biostatistics) is the
canonical statistical baseline for sentinel ILI surveillance forecasting.
It decomposes the conditional mean as

    λ_t = ν_t (endemic) + ε_t · Y_{t-1} (epidemic, autoregressive)

with negative-binomial observation noise. For a single-region scalar ILI
rate (this thesis's setup), the canonical hhh4 reduces to a NEGATIVE
BINOMIAL GLM with seasonal harmonics + AR(1) lag — which is exactly what
the existing `simulation/models/epi_models.py:NegBinGLM` implements.

This module exposes a minimal, single-line wrapper that:
  1. Wraps NegBinGLM with explicit seasonal+AR endemic-epidemic structure
  2. Reports it as the "hhh4-equivalent" baseline in R9 / R10
  3. Cites Held & Paul 2012 explicitly

References:
  Held L, Paul M (2012). Modeling seasonality in space-time infectious
    disease surveillance data. Biometrical Journal 54(6):824-843.
  Meyer S, Held L, Höhle M (2017). Spatio-temporal analysis of epidemic
    phenomena using the R package surveillance. JSS 77(11).

For the full hhh4 with spatial neighborhoods (gu × gu commuter graph),
use the R `surveillance` package via rpy2; this thesis uses the
single-region reduction as the practical benchmark.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class HHH4Equivalent:
    """Endemic-epidemic NegBin-GLM benchmark (single-region hhh4 reduction).

    Model:
        log(λ_t) = β_0 + β_AR · log(Y_{t-1} + 1)
                 + Σ_k [α_k cos(2πkt/T) + β_k sin(2πkt/T)]   # endemic seasonality
        Y_t | λ_t ~ NegBin(λ_t, φ)

    Default: K=2 harmonics (annual + semi-annual) at T=52 weeks. AR(1) only.

    Use:
        m = HHH4Equivalent(harmonics=2, period=52)
        m.fit(y_train)              # series in time order
        y_hat = m.predict_h(h=1)    # 1-step-ahead point forecast (mean)
        sigma = m.sigma_           # in-sample NB-residual std for PI
    """

    def __init__(self, harmonics: int = 2, period: int = 52, ar_order: int = 1):
        self.harmonics = int(harmonics)
        self.period = int(period)
        self.ar_order = int(ar_order)
        self._beta: Optional[np.ndarray] = None
        self._fitted_y: Optional[np.ndarray] = None
        self.sigma_: Optional[float] = None

    def _design(self, t_index: np.ndarray, y_lag: np.ndarray) -> np.ndarray:
        """Design matrix: [1, log(y_lag+1), cos(2πk t/T), sin(2πk t/T) for k=1..K]"""
        cols = [np.ones_like(t_index, dtype=np.float64)]
        cols.append(np.log(np.clip(y_lag, 0, None) + 1.0))
        for k in range(1, self.harmonics + 1):
            ang = 2.0 * np.pi * k * t_index / self.period
            cols.append(np.cos(ang))
            cols.append(np.sin(ang))
        return np.column_stack(cols)

    def fit(self, y: np.ndarray) -> "HHH4Equivalent":
        y = np.asarray(y, dtype=np.float64)
        if len(y) <= self.ar_order + 2 * self.harmonics + 2:
            raise ValueError("series too short for hhh4-equivalent fit")
        # Build training arrays: y_t depends on y_{t-1}
        y_target = y[self.ar_order:]
        y_lag = y[: -self.ar_order]
        t_idx = np.arange(self.ar_order, len(y), dtype=np.float64)
        X = self._design(t_idx, y_lag)

        # Fit log-linear via WLS-IRLS (poor man's GLM with NB link).
        # For NegBin, MLE requires statsmodels; if unavailable, fall back to
        # OLS on log(y_target+1) which is the canonical surveillance start.
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")  # NegBin α not-set warning is benign
                import statsmodels.api as sm
                model = sm.GLM(y_target, X, family=sm.families.NegativeBinomial())
                res = model.fit(disp=0)
            self._beta = np.asarray(res.params, dtype=np.float64)
            # In-sample residual std on the response scale
            mu_hat = np.exp(X @ self._beta)
            resid = y_target - mu_hat
            self.sigma_ = float(np.std(resid))
            self._link = "log_negbin"
        except Exception as _ie:
            log.warning(f"  [hhh4-equiv] statsmodels NegBin failed ({_ie}); "
                        f"falling back to OLS on log(y+1)")
            y_log = np.log(y_target + 1.0)
            # G-275 base layer: log-space OLS fallback 은 exp() 로 증폭되므로 safe_lstsq 로 β 폭발 차단
            from simulation.models.safety import safe_lstsq
            beta = safe_lstsq(X, y_log)
            self._beta = beta
            mu_hat = np.exp(X @ beta) - 1.0
            resid = y_target - mu_hat
            self.sigma_ = float(np.std(resid))
            self._link = "log_ols_fallback"
        self._fitted_y = y
        return self

    def predict_h(self, h: int = 1) -> np.ndarray:
        """Recursive h-step forecast from the end of the fitted series."""
        if self._beta is None or self._fitted_y is None:
            raise RuntimeError("call fit() first")
        y_hist = list(self._fitted_y)
        n_total = len(y_hist)
        out = np.zeros(h, dtype=np.float64)
        for step in range(h):
            t = float(n_total + step)
            y_lag = y_hist[-1]
            X = self._design(np.array([t]), np.array([y_lag]))
            log_mu = float((X @ self._beta)[0])
            mu = np.exp(log_mu) if self._link == "log_negbin" else max(np.exp(log_mu) - 1.0, 0.0)
            out[step] = mu
            y_hist.append(mu)  # recursive: feed the prediction back as next lag
        return out

    def predict(self, n_steps: int = 1) -> np.ndarray:
        return self.predict_h(h=n_steps)


def hhh4_benchmark_oneshot(y_in: np.ndarray, n_real: int) -> dict:
    """Single-line benchmark: fit on y_in, recursive predict n_real steps."""
    m = HHH4Equivalent(harmonics=2, period=52, ar_order=1)
    m.fit(y_in)
    return {
        "predictions": m.predict_h(h=n_real),
        "sigma": float(m.sigma_) if m.sigma_ is not None else 1.0,
        "name": "hhh4_equivalent",
        "citation": "Held & Paul (2012) Biometrical Journal 54:824 "
                    "(NegBin GLM with K=2 seasonal harmonics + AR(1))",
    }
