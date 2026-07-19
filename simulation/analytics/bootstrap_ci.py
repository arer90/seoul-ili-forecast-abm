"""Bootstrap CI for metrics (audit Stage 1.2, Task #14).

Generic bootstrap confidence interval for any metric function.
Per-metric bootstrap CI for diagnostic reporting (champion = best-WIS).
    (4-criteria/g175 strict-pass + composite z-score helper 제거 2026-06-05 — dead.)

References:
    - Efron B (1979) "Bootstrap methods: another look at the jackknife"
      Annals of Statistics 7(1):1-26. doi:10.1214/aos/1176344552
    - Davison & Hinkley (1997) "Bootstrap Methods and their Application"
      Cambridge University Press, ISBN 978-0-521-57471-6
    - Politis & Romano (1994) "The stationary bootstrap"
      JASA 89(428):1303-1313. doi:10.1080/01621459.1994.10476870
      (block bootstrap for time-series — R6 DM test compatible)

D-5 gray-box contract:
    - NaN-safe (vals filtered before quantile)
    - seed-aware (default=42; multi-seed wrapper 가능)
    - O(n_boot * metric_fn_cost) — per-metric eval(R10) 1회 산출 후 cache
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = [
    "bootstrap_metric_ci",
    "bootstrap_metric_ci_paired",
    "block_bootstrap_metric_ci",
]


def bootstrap_metric_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int = 42,
    return_dist: bool = False,
) -> dict:
    """Bootstrap CI for metric_fn(y_true, y_pred).

    Args:
        metric_fn: (y_true, y_pred) → float. NaN 가능.
        y_true: (n,) — finite.
        y_pred: (n,) — finite.
        n_boot: 1000 default (Efron 권장 ≥ 1000 for percentile CI).
        conf: 0.95 default (95% CI).
        seed: 42 default.
        return_dist: True → 전체 bootstrap distribution 도 반환.

    Returns:
        dict {
            "point": float,      # metric_fn(y_true, y_pred) point estimate
            "ci_lo": float,      # lower CI bound (percentile)
            "ci_hi": float,      # upper CI bound (percentile)
            "ci_method": "percentile",
            "n_valid": int,      # n_boot 중 finite metric 산출 횟수
            "dist": np.ndarray,  # only if return_dist=True
        }

    Performance: O(n_boot * metric_fn_cost). 1000 boot × R² → ~10 ms.
    Side effects: 없음 (pure function, seeded).
    Caller responsibility:
        - metric_fn 이 raise X (NaN return 권장)
        - y_true / y_pred 길이 일치 + finite (caller 가 mask)
    """
    out = {
        "point": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "ci_method": "percentile",
        "n_valid": 0,
    }

    if y_true is None or y_pred is None:
        return out
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = len(yt)
    if n < 4 or n != len(yp):
        return out

    # point estimate
    try:
        out["point"] = float(metric_fn(yt, yp))
    except Exception:
        pass

    # bootstrap
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # iid resample with replacement
        try:
            vals[b] = float(metric_fn(yt[idx], yp[idx]))
        except Exception:
            vals[b] = np.nan

    finite = vals[np.isfinite(vals)]
    out["n_valid"] = int(len(finite))
    if len(finite) >= 10:
        alpha = (1.0 - conf) / 2.0
        out["ci_lo"] = float(np.quantile(finite, alpha))
        out["ci_hi"] = float(np.quantile(finite, 1.0 - alpha))

    if return_dist:
        out["dist"] = vals

    return out


def bootstrap_metric_ci_paired(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI for metric difference (metric_A - metric_B).

    Used by PROMOTE_V2 (Task #21) — v1 vs v2 metric difference paired DM 보완.

    Returns:
        dict {
            "point_a": float, "point_b": float, "point_diff": float,
            "ci_lo_diff": float, "ci_hi_diff": float,
            "n_valid": int,
        }
    """
    out = {
        "point_a": float("nan"), "point_b": float("nan"), "point_diff": float("nan"),
        "ci_lo_diff": float("nan"), "ci_hi_diff": float("nan"),
        "n_valid": 0,
    }
    if y_true is None or y_pred_a is None or y_pred_b is None:
        return out
    yt = np.asarray(y_true, dtype=np.float64)
    yp_a = np.asarray(y_pred_a, dtype=np.float64)
    yp_b = np.asarray(y_pred_b, dtype=np.float64)
    n = len(yt)
    if n < 4 or n != len(yp_a) or n != len(yp_b):
        return out

    try:
        out["point_a"] = float(metric_fn(yt, yp_a))
        out["point_b"] = float(metric_fn(yt, yp_b))
        out["point_diff"] = out["point_a"] - out["point_b"]
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            m_a = float(metric_fn(yt[idx], yp_a[idx]))
            m_b = float(metric_fn(yt[idx], yp_b[idx]))
            diffs[b] = m_a - m_b
        except Exception:
            diffs[b] = np.nan

    finite = diffs[np.isfinite(diffs)]
    out["n_valid"] = int(len(finite))
    if len(finite) >= 10:
        alpha = (1.0 - conf) / 2.0
        out["ci_lo_diff"] = float(np.quantile(finite, alpha))
        out["ci_hi_diff"] = float(np.quantile(finite, 1.0 - alpha))

    return out


def block_bootstrap_metric_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    block_size: int = 8,
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int = 42,
) -> dict:
    """Politis-Romano stationary block bootstrap (time-series).

    Used for WIS / coverage / lead_time 의 CI — iid bootstrap 이 시계열
    autocorrelation 무시. Block bootstrap 이 시계열 적합.

    Reference: Politis & Romano (1994) doi:10.1080/01621459.1994.10476870

    Args:
        block_size: 기대 block 길이 (geometric distribution). 8 default.
    """
    out = {
        "point": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "ci_method": f"stationary_block_bootstrap(block_size={block_size})",
        "n_valid": 0,
    }

    if y_true is None or y_pred is None:
        return out
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = len(yt)
    if n < block_size * 2 or n != len(yp):
        return out

    try:
        out["point"] = float(metric_fn(yt, yp))
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    p = 1.0 / block_size
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        # stationary bootstrap (Politis-Romano)
        idx = np.zeros(n, dtype=np.int64)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            block_len = max(1, int(rng.geometric(p)))
            for j in range(block_len):
                if i + j >= n:
                    break
                idx[i + j] = (start + j) % n
            i += block_len

        try:
            vals[b] = float(metric_fn(yt[idx], yp[idx]))
        except Exception:
            vals[b] = np.nan

    finite = vals[np.isfinite(vals)]
    out["n_valid"] = int(len(finite))
    if len(finite) >= 10:
        alpha = (1.0 - conf) / 2.0
        out["ci_lo"] = float(np.quantile(finite, alpha))
        out["ci_hi"] = float(np.quantile(finite, 1.0 - alpha))

    return out


# (g175_strict_pass_from_ci + composite_score_z 제거 2026-06-05 — 4-criteria/g175
#  완전 폐지, dead helper. champion = best-WIS. live = bootstrap_metric_ci* 만.)
