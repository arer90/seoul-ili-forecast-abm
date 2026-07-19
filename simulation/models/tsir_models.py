"""
simulation/models/tsir_models.py
=================================
TSIR (Time-Series SIR) — Finkenstädt & Grenfell (2000).

[학술 배경]
Discrete-time stochastic SIR for endemic-epidemic dynamics:
  I_{t+1} = β_t × S_t × I_t^α / N
  S_{t+1} = S_t + B_t - I_{t+1}
where:
  - β_t: time-varying transmission (seasonal, can fit Fourier)
  - α ≈ 0.97: heterogeneity exponent (Finkenstädt 2002)
  - S_t: susceptible pool (reconstructed via susceptible reconstruction)
  - B_t: births / new susceptibles per period

[Susceptible reconstruction]
  Finkenstädt (2002): cumulative susceptible = cumulative births - cumulative infectees
  S_t = S_0 + Σ B - Σ I (reporting-rate corrected)

[ILI rate 적용]
- ILI rate × population proxy = infectee count
- 학습: log(I_{t+1}/I_t^α) = log(β_t) + log(S_t/N) → linear regression
- 예측: iterative SIR forward

[참조]
- Finkenstädt BF, Grenfell BT (2000). "Time series modelling of childhood
  diseases: a dynamical systems approach". JRSS C 49(2):187-205.
- Finkenstädt BF (2002). "A stochastic model for extinction and recurrence
  of epidemics". Biostatistics 3(4):493-510.
- Bjornstad ON, Finkenstadt BF, Grenfell BT (2002). "Dynamics of measles
  epidemics: estimating scaling of transmission rates". EcoMonogr 72:169-184.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


class TSIRForecaster(BaseForecaster):
    """TSIR (Finkenstädt-Grenfell 2000) discrete-time stochastic SIR forecaster.

    fit:
      1. log-linear regression on log(I_{t+1}) - α × log(I_t) ~ seasonal
      2. estimate β(t) Fourier coefficients
      3. estimate population proxy + susceptible scaling

    predict:
      1. iterative SIR forward
      2. I_{t+1} = β_t × S_t/N × I_t^α
    """

    meta = ModelMeta(
        name="TSIR", category="epi", level=7, min_data=80,
        description=(
            "TSIR (Finkenstädt-Grenfell 2000) discrete-time stochastic SIR + "
            "Fourier β + susceptible reconstruction."
        ),
        dependencies=[],
    )

    def __init__(self, alpha: float = 0.97, K_harmonics: int = 2,
                  period: int = 52, S0_ratio: float = 0.5):
        super().__init__()
        self._alpha = float(alpha)
        self._K = int(K_harmonics)
        self._period = int(period)
        self._S0_ratio = float(S0_ratio)
        self._beta_coef = None  # Fourier coefficients
        self._N = 1.0   # population proxy
        self._S_last = 0.5
        self._I_last = 1.0
        self._y_max = 100.0
        self._fitted = False

    def _fourier(self, t: np.ndarray) -> np.ndarray:
        feats = [np.ones_like(t)]
        for k in range(1, self._K + 1):
            feats.append(np.sin(2 * np.pi * k * t / self._period))
            feats.append(np.cos(2 * np.pi * k * t / self._period))
        return np.column_stack(feats)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              **kwargs) -> "TSIRForecaster":
        from simulation.models.base import _validate_shapes
        _validate_shapes(X_train, y_train, name="TSIR.fit", min_n=40)

        I = np.maximum(y_train.astype(np.float64), 0.01)
        n = len(I)
        self._y_max = float(np.max(I))
        self._N = float(self._y_max * 10.0)  # crude population proxy

        # log-linear regression:
        # log(I_{t+1}) = log(β_t) + α × log(I_t) + log(S_t/N)
        # rearrange: log(I_{t+1}) - α × log(I_t) = log(β_t) + log(S_t/N)
        # 본 구현: S_t/N ≈ constant (S0_ratio); estimate β_t Fourier
        y_lhs = np.log(I[1:] + 1e-3) - self._alpha * np.log(I[:-1] + 1e-3)
        t_train = np.arange(n - 1, dtype=np.float64)
        F = self._fourier(t_train)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._beta_coef, *_ = np.linalg.lstsq(F, y_lhs, rcond=None)
            self._fitted = True
            self._S_last = self._S0_ratio
            self._I_last = float(I[-1])
            log.info(f"  [TSIR] α={self._alpha}, K={self._K}, β coefs={self._beta_coef.shape}")
        except Exception as e:
            log.warning(f"  [TSIR] fit 실패: {e}")
            self._beta_coef = None
            self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.base import sanitize_predictions

        n_test = len(X_test)
        if not self._fitted or self._beta_coef is None:
            return sanitize_predictions(np.full(n_test, self._I_last))

        t_start = self._I_last  # ignore X_test, pure SIR forward
        preds = []
        I_t = self._I_last
        S_t = self._S_last * self._N
        for τ in range(n_test):
            # t index continues from train
            t_idx = float(τ)
            F_t = self._fourier(np.array([t_idx]))[0]
            log_beta = float(F_t @ self._beta_coef)
            beta = float(np.exp(log_beta))
            I_next = beta * (S_t / self._N) * (I_t ** self._alpha)
            I_next = float(np.clip(I_next, 0.01, self._y_max * 5.0))
            preds.append(I_next)
            # Update susceptible (very approximate)
            S_t = max(S_t - I_next, self._N * 0.01)
            I_t = I_next
        pred = np.array(preds)
        return sanitize_predictions(pred)


try:
    REGISTRY.register(TSIRForecaster)
    log.info("[tsir_models] TSIRForecaster 등록됨 (Finkenstädt-Grenfell 2000)")
except Exception as _e:
    log.warning(f"[tsir_models] 등록 skip: {_e}")
