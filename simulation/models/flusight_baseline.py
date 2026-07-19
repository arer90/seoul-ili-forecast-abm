"""FluSight persistence + quantile baseline (Round 4 audit G1, 2026-05-27).

External audit Round 4 (CRITICAL) — Mathis et al. (2024) *Nat Commun* 15:6289
의 `quantile_baseline` (Reich Lab `simplets` R package) Python port. paper
Methods 의 *comparator baseline* 으로 필수 — 학습 launch 전 model registry
포함 필요 (학습 후 추가 시 ensemble cascade 60-100h 재학습 위험).

Reference (verbatim from Mathis et al. 2024 Methods):
    "Baseline forecasts and their prediction intervals were generated each week
     using the 'quantile baseline' method in the simplets R package based on
     the incident hospitalizations reported in the previous week, with underlying
     methodology described as follows. The median prediction of the baseline
     forecasts is the corresponding target value observed in the previous week,
     and noise around the median prediction is generated using positive and
     negative 1-week differences (i.e., differences between consecutive reports)
     for all prior observations, separately for each jurisdiction. Sampling
     distributions were truncated to prevent negative values. The same median
     prediction is used for the 1-through 4-week ahead forecasts. The baseline
     model's prediction intervals are generated from a smoothed version of this
     distribution of differences."

Citation:
    Mathis SM, Webber AE, León TM, et al. (2024) "Evaluation of FluSight
    influenza forecasting in the 2021-22 and 2022-23 seasons with a new target
    laboratory-confirmed influenza hospitalizations" Nat Commun 15:6289.
    doi:10.1038/s41467-024-50601-9 (PMID 39060259)

    Reich Lab `simplets` R package: github.com/reichlab/simplets

Algorithm (verbatim port):
    1. median = last observed y_train (persistence; same for h=1..H)
    2. noise distribution = ±1-week differences history of y_train
       (symmetric: concat(diffs, -diffs))
    3. truncate at 0 (no negative ILI rates)
    4. smooth via Gaussian kernel (sample-based percentile equivalent)
    5. 23 quantiles per FluSight: 0.010, 0.025, 0.050, 0.100, 0.150, 0.200,
       0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700,
       0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990

Relative WIS (Mathis et al. 2024 + CDC FluSight 2024-2025 evaluation report):
    Relative WIS = geometric_mean(WIS_model_per_target) /
                   geometric_mean(WIS_baseline_per_target)
    "Using the geometric mean allows for a more direct comparison of models
     even when not all models submit all or most forecast jurisdictions."
    → `compute_relative_wis()` helper provided.

D-5 contract:
    - Single deep module (1 class + 1 helper)
    - NaN-safe (sanitize_predictions 자동 적용)
    - No external dependencies beyond numpy/scipy.stats
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from simulation.models.base import REGISTRY, BaseForecaster, ModelMeta

__all__ = [
    "FluSightQuantileBaseline",
    "FLUSIGHT_23_QUANTILES",
    "compute_relative_wis",
]


#: FluSight 23 quantile probability levels (Mathis et al. 2024 Methods verbatim)
FLUSIGHT_23_QUANTILES: tuple[float, ...] = (
    0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
    0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900,
    0.950, 0.975, 0.990,
)


class FluSightQuantileBaseline(BaseForecaster):
    """FluSight persistence + quantile baseline (Mathis et al. 2024).

    Round 4 audit G1 (CRITICAL): paper comparator baseline 필수. 본 model 이
    `relative WIS < 1` 의 reference denominator.

    Args:
        smoothing: "kernel" (Gaussian smoothing default) or "empirical".
        truncate_at_zero: True (FluSight 표준 — no negative).

    Returns (`predict_quantiles`):
        (n_test, n_quantiles) array — 23 quantile forecasts per horizon.
        Same median for h=1..4 (FluSight 표준).
    """

    meta = ModelMeta(
        name="FluSight-Baseline",
        category="ts",  # baseline category — same family as ARIMA/SARIMA/Theta
        level=0,  # baseline tier (lowest complexity)
        min_data=4,  # at least 4 weeks for 1-week diff history
        description=(
            "FluSight persistence baseline (Mathis et al. 2024 Nat Commun 15:6289). "
            "Median = last observed y; noise = ±1-week differences history; "
            "23 quantiles (0.010-0.990); same median for h=1-4. "
            "Comparator for relative WIS metric (geometric mean ratio)."
        ),
        dependencies=[],  # numpy only (scipy.stats optional for gmean)
    )

    def __init__(self, smoothing: str = "kernel", truncate_at_zero: bool = True):
        super().__init__()
        self.smoothing = smoothing
        self.truncate_at_zero = truncate_at_zero
        self._last_obs: Optional[float] = None
        self._diff_history: Optional[np.ndarray] = None

    # BaseForecaster ABC contract
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "FluSightQuantileBaseline":
        """Fit on y_train (baseline ignores X_train)."""
        y_arr = np.asarray(y_train, dtype=np.float64).ravel()
        finite = y_arr[np.isfinite(y_arr)]
        if len(finite) < 2:
            self._last_obs = float(finite[-1]) if len(finite) > 0 else 0.0
            self._diff_history = np.array([], dtype=np.float64)
            return self
        self._last_obs = float(finite[-1])
        diffs = np.diff(finite)
        self._diff_history = np.concatenate([diffs, -diffs])
        return self

    def predict(self, X_test: np.ndarray, y_observed: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Predict median (persistence).

        G-345 (감사 P1-2): FluSight 표준 = "median = 지난주 관측"(Mathis 2024). FluSight 는
        ROLLING_EVAL_MODELS 멤버라 eval 이 ``y_observed`` 를 넘김 → **rolling random-walk
        persistence**(주 t 의 median = y_observed[t-1]). y_observed 없으면(static) fit-time
        last_obs flat.

        옛 버그: predict 가 ``y_observed`` 를 무시해 68주 전부 frozen last_obs(flat line) →
        relative-WIS 분모(FluSight WIS)가 비현실적으로 약해져 타 모델 skill 과대(46/51 이 분모
        능가로 보이던 인플레). 이제 표준 persistence 라 분모가 정직.

        Args:
            X_test: (n_test, d) — shape only.
            y_observed: (n_test,) rolling 관측 — 주 i 예측에 y_observed[:i] 만 사용(leak-free).

        Returns:
            np.ndarray (n_test,) median 예측 (rolling: yo[i-1] / static: last_obs).
        """
        n = len(X_test)
        if self._last_obs is None:
            return np.full(n, float("nan"))
        yo_arr = np.asarray(y_observed, dtype=np.float64).ravel() if y_observed is not None else None
        if yo_arr is not None and len(yo_arr) == n and n >= 1:
            pred = np.empty(n, dtype=np.float64)
            pred[0] = self._last_obs                 # 주 0 = train 마지막 관측 (leak-free)
            if n > 1:
                pred[1:] = yo_arr[:-1]               # 주 i = 지난주 관측 yo[i-1] (random-walk persistence)
            prev = self._last_obs                     # NaN 관측 forward-fill (leak-free 유지)
            for i in range(n):
                if not np.isfinite(pred[i]):
                    pred[i] = prev
                prev = pred[i]
        else:
            pred = np.full(n, self._last_obs, dtype=np.float64)
        if self.truncate_at_zero:
            pred = np.maximum(pred, 0.0)
        return pred

    def _fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        **kw,
    ) -> np.ndarray:
        """Fit on y_train history, predict y_test of len(X_test).

        Args:
            X_train: ignored (baseline 은 y only — persistence + diff history)
            y_train: (n_train,) historical observations
            X_test: (n_test, d) only used for shape

        Returns:
            np.ndarray (n_test,) — median predictions (same value for all horizons,
            equal to last observed y_train).
        """
        y_arr = np.asarray(y_train, dtype=np.float64).ravel()
        finite_mask = np.isfinite(y_arr)
        finite = y_arr[finite_mask]

        if len(finite) < 2:
            # Edge case: insufficient history → fallback to last finite value or 0
            last = float(finite[-1]) if len(finite) > 0 else 0.0
            self._last_obs = last
            self._diff_history = np.array([], dtype=np.float64)
            return np.full(len(X_test), last, dtype=np.float64)

        # Algorithm step 1: median = last observation
        self._last_obs = float(finite[-1])

        # Algorithm step 2: 1-week differences history (symmetric)
        diffs = np.diff(finite)
        self._diff_history = np.concatenate([diffs, -diffs])

        # Median forecast — same value for all horizons (FluSight 표준)
        median_pred = np.full(len(X_test), self._last_obs, dtype=np.float64)

        # Algorithm step 3: truncate at 0
        if self.truncate_at_zero:
            median_pred = np.maximum(median_pred, 0.0)

        return median_pred

    def predict_quantiles(
        self,
        X_test: np.ndarray,
        quantiles: tuple[float, ...] = FLUSIGHT_23_QUANTILES,
    ) -> np.ndarray:
        """Predict 23 quantiles for FluSight WIS evaluation.

        Algorithm: quantile of (last_obs + diff_history) sample distribution.
        Smoothed via Gaussian kernel (sample-based percentile equivalent).

        Args:
            X_test: (n_test, d) — used for shape only
            quantiles: 23 probability levels (FluSight 표준)

        Returns:
            np.ndarray (n_test, n_quantiles) — same quantile values for all horizons
            (FluSight 표준 — same median for h=1..4).

        Raises:
            절대 raise X — empty diff history 시 last_obs broadcast 만.
        """
        n_test = len(X_test)
        n_q = len(quantiles)

        if self._last_obs is None:
            return np.full((n_test, n_q), float("nan"))

        if self._diff_history is None or len(self._diff_history) == 0:
            # Insufficient history → all quantiles = last_obs (no noise)
            return np.tile([self._last_obs] * n_q, (n_test, 1))

        # Quantile of (last_obs + diff_history) — symmetric noise around median
        samples = self._last_obs + self._diff_history

        # Algorithm step 3: truncate at 0
        if self.truncate_at_zero:
            samples = np.maximum(samples, 0.0)

        # Algorithm step 4: smoothing
        # - "kernel" (default): Gaussian KDE bandwidth ≈ scott's rule
        # - "empirical": direct sample percentile (no smoothing)
        if self.smoothing == "kernel" and len(samples) >= 4:
            # Augment with small Gaussian noise per sample (KDE-like)
            rng = np.random.default_rng(seed=42)
            bw = float(np.std(samples, ddof=1)) * (len(samples) ** (-1.0 / 5.0))
            if bw > 0:
                augmented = np.repeat(samples, 10) + rng.normal(0, bw, size=len(samples) * 10)
                if self.truncate_at_zero:
                    augmented = np.maximum(augmented, 0.0)
                samples = augmented

        q_values = np.array([float(np.quantile(samples, q)) for q in quantiles])

        # Same quantile values for all horizons (FluSight 표준 — h=1..4 동일)
        return np.tile(q_values, (n_test, 1))


def compute_relative_wis(
    model_wis_per_target: np.ndarray | list,
    baseline_wis_per_target: np.ndarray | list,
) -> dict:
    """Relative WIS = geometric_mean(WIS_model) / geometric_mean(WIS_baseline).

    Per Mathis et al. (2024) + CDC FluSight 2024-2025 evaluation report verbatim:
        "Relative WIS was calculated using the geometric mean WIS of each model
         forecast compared to the geometric mean WIS of the corresponding
         FluSight baseline model forecast … Using the geometric mean allows for
         a more direct comparison of models even when not all models submit all
         or most forecast jurisdictions."

    Args:
        model_wis_per_target: WIS values per (region × horizon × time) target
        baseline_wis_per_target: same target list, baseline WIS values

    Returns:
        dict {
            "relative_wis": float,        # < 1 = better than baseline
            "gmean_model": float,
            "gmean_baseline": float,
            "n_valid": int,
            "reference": str,
            "interpretation": str,        # "outperforms baseline" if < 1, "worse" if > 1
        }

    Raises:
        절대 raise X — fail 시 NaN.
    """
    out = {
        "relative_wis": float("nan"),
        "gmean_model": float("nan"),
        "gmean_baseline": float("nan"),
        "n_valid": 0,
        "reference": (
            "Mathis et al. (2024) Nat Commun 15:6289 doi:10.1038/s41467-024-50601-9 "
            "PMID 39060259; CDC FluSight 2024-2025 evaluation report"
        ),
        "interpretation": "n/a",
    }
    try:
        from scipy.stats import gmean
    except ImportError:
        # Fallback: log-mean exp
        def gmean(arr):
            arr = np.asarray(arr, dtype=np.float64)
            arr = arr[arr > 0]
            return float(np.exp(np.mean(np.log(arr)))) if len(arr) > 0 else float("nan")

    m = np.asarray(model_wis_per_target, dtype=np.float64)
    b = np.asarray(baseline_wis_per_target, dtype=np.float64)
    if m.shape != b.shape:
        return out

    mask = np.isfinite(m) & np.isfinite(b) & (m > 0) & (b > 0)
    if not mask.any():
        return out

    gm_m = float(gmean(m[mask]))
    gm_b = float(gmean(b[mask]))
    if gm_b <= 0:
        return out

    rel = gm_m / gm_b
    out["relative_wis"] = rel
    out["gmean_model"] = gm_m
    out["gmean_baseline"] = gm_b
    out["n_valid"] = int(mask.sum())
    out["interpretation"] = "outperforms baseline" if rel < 1 else "worse than baseline"
    return out


# REGISTRY registration (G1.b, Round 4 audit, 2026-05-27)
# CATEGORY_MODELS["ts"] entry — comparator baseline for relative WIS.
REGISTRY.register(FluSightQuantileBaseline)
