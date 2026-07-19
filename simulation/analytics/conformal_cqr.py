"""Conformalized Quantile Regression (CQR) — Romano et al. 2019.

Provides a single deep module: ``CQRForecaster`` that fits two LightGBM
quantile models (q=α/2, q=1-α/2) on training data, then applies
Vovk-style conformity correction on a calibration set to guarantee
finite-sample 1-α coverage.

Target use: post-hoc PI replacement for production models that have
under-coverage (PICP95 < 0.85) on the test slab. NegBinGLM (1위 R²+WIS)
phase 12 result showed PICP95=0.62 (Wilson [0.51, 0.73]); CQR raises
this to 1.00 with WIS 1.65 (vs native 16.07) — verified on
``simulation/results/picp95_experiment_NegBinGLM.json``.

References:
- Romano, Patterson, Candès (2019). Conformalized Quantile Regression.
  NeurIPS. arXiv:1905.03222
- Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random
  World. Springer.

ENGINEERING_PRINCIPLES.md:
- D-4 deep module: 1 fit + 1 predict_interval, internal: 2 LightGBM
  quantile models + Vovk finite-sample correction (K = ceil((n+1)(1-α))).
- D-5 gray-box: caller responsibility — ``cal`` set MUST be exchangeable
  with test (i.i.d. or distribution-shift recalibration via ``recalibrate``).
- #5 reproducibility: deterministic when ``random_state`` set.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["CQRForecaster", "apply_cqr_to_predictions"]


class CQRForecaster:
    """Conformalized Quantile Regression with LightGBM backbone.

    Args:
        alpha: target miscoverage (0.05 → 95% PI). Standard literature uses 0.05.
        n_estimators: per quantile model (default 200, fast on n~300).
        learning_rate: LightGBM step (default 0.05).
        num_leaves: LightGBM tree complexity (default 31).
        min_child_samples: minimum samples per leaf (default 5; small for n<300).
        random_state: deterministic seed.

    Performance: ~0.5-1.0 s fit on n=266, p=309 (M1 / 8 cores).
    Memory: 2 LightGBM models × ~5 MB = ~10 MB.
    Side effects: 0 (no global state).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 5,
        random_state: int = 42,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self._params = dict(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            min_child_samples=int(min_child_samples),
            random_state=int(random_state),
            verbose=-1,
            force_col_wise=True,
        )
        self._m_lo: Any = None
        self._m_hi: Any = None
        self._Q: float | None = None
        self._n_cal: int = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CQRForecaster":
        """Fit two LightGBM quantile regressors on training data.

        Args:
            X_train: (n, p) feature matrix.
            y_train: (n,) target.

        Returns: self (for chaining).
        Raises: ImportError if ``lightgbm`` not installed.
        """
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "CQRForecaster requires lightgbm. Install via 'uv pip install lightgbm'."
            ) from e
        self._m_lo = lgb.LGBMRegressor(
            objective="quantile", alpha=self.alpha / 2.0, **self._params
        )
        self._m_hi = lgb.LGBMRegressor(
            objective="quantile", alpha=1.0 - self.alpha / 2.0, **self._params
        )
        self._m_lo.fit(X_train, y_train)
        self._m_hi.fit(X_train, y_train)
        self._Q = None  # reset; must call calibrate after fit
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """Compute Vovk finite-sample conformity quantile on calibration set.

        Conformity score per cal point: max(qlo - y, y - qhi) — signed
        violation distance. The (1-α)-quantile (with k = ceil((n+1)(1-α)))
        gives finite-sample-valid coverage (Vovk 2005 Thm 2.1).

        Args:
            X_cal: (n_cal, p) calibration features.
            y_cal: (n_cal,) calibration target.

        Returns: Q (float) — the conformity correction.

        Raises: RuntimeError if ``fit`` not called first.
        """
        if self._m_lo is None or self._m_hi is None:
            raise RuntimeError("Must call fit() before calibrate()")
        qlo_cal = self._m_lo.predict(X_cal)
        qhi_cal = self._m_hi.predict(X_cal)
        # Signed conformity: max(qlo - y, y - qhi). Positive = miss.
        score = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
        n_cal = len(score)
        if n_cal < 5:
            log.warning(
                "[CQR] cal set too small (n=%d); coverage guarantee weak", n_cal
            )
        # Finite-sample correction: k = ceil((n+1)(1-α)), 1-indexed
        k = int(np.ceil((n_cal + 1) * (1.0 - self.alpha)))
        sorted_score = np.sort(score)
        Q = float(sorted_score[min(k - 1, n_cal - 1)])
        self._Q = Q
        self._n_cal = n_cal
        return Q

    def predict_interval(
        self, X_test: np.ndarray, nonneg: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (1-α) prediction interval bounds.

        Args:
            X_test: (n_test, p) features.
            nonneg: clamp lower bound to 0 (ILI rate ≥ 0 domain constraint).

        Returns:
            (lo, hi): each shape (n_test,).

        Raises: RuntimeError if ``fit`` or ``calibrate`` not called.
        """
        if self._m_lo is None or self._Q is None:
            raise RuntimeError(
                "Must call fit() then calibrate() before predict_interval()"
            )
        qlo = self._m_lo.predict(X_test)
        qhi = self._m_hi.predict(X_test)
        lo = qlo - self._Q
        hi = qhi + self._Q
        if nonneg:
            lo = np.maximum(0.0, lo)
        return lo, hi

    @property
    def conformity_quantile(self) -> float | None:
        """Q value used for prediction interval correction. None until calibrate."""
        return self._Q


def apply_cqr_to_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.05,
    nonneg: bool = True,
) -> dict:
    """Convenience: fit + calibrate + predict in one call.

    Returns dict with: ``lo``, ``hi``, ``Q``, ``n_cal``, ``alpha``, ``method``.
    """
    cqr = CQRForecaster(alpha=alpha)
    cqr.fit(X_train, y_train)
    Q = cqr.calibrate(X_cal, y_cal)
    lo, hi = cqr.predict_interval(X_test, nonneg=nonneg)
    return {
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "Q": Q,
        "n_cal": len(y_cal),
        "alpha": alpha,
        "method": "CQR-LightGBM (Romano 2019, Vovk finite-sample)",
    }
