"""Peak-aware 가중 평가 (FusedEpi peak-aware loss/eval) — 모델-비종속.

공중보건 자원배치는 **outbreak/peak 정확도**가 핵심이다 (평시 baseline 오차 1단위 ≠
유행 정점 오차 1단위). 이 모듈은 임의 forecaster의 점·구간 예측을 받아 peak 구간을
upweight한 평가 지표를 제공한다. 모든 함수가 모델/엔진에 무관 — y_true/y_pred/bounds만
받으며 특정 모델을 하드코딩하지 않는다.

핵심 개념 (ubiquitous language):
  - "peak 구간 (peak region)"  : 관측 시계열 y가 분위(quantile) 임계 이상인 시점 집합.
  - "가중치 (weights)"        : 시점별 비음(non-negative) 가중치. 평시=1.0, peak=상향.
  - "peak skill"              : peak 구간에서 baseline 대비 상대 정확도 (높을수록 우수).

좌표계: 모든 입력은 **원공간 (y_orig)** — transform 공간 함정 회피 (G-321/349).
        weights는 음수 없음 (불변식). 평탄(분산 0) 시계열은 균일 가중 (1.0).

설계 출처 (formula):
  - Bracher et al. (2021) PLOS Comp Bio 17(2):e1008618 — WIS 분해.
  - Gneiting & Raftery (2007) JASA 102(477):359-378 — interval score.
  - peak-가중 = WIS 시점별 항에 weights[t] 곱 후 가중평균 (FluSight peak-skill 관행).

Side effects: 없음 (순수 함수). Disk/DB/global state 미접촉.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# FluSight 표준 11-level (Bracher 2021); diagnostics.weighted_interval_score 기본값과 동일.
DEFAULT_ALPHAS: Tuple[float, ...] = (
    0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
)


def peak_weights(
    y: Sequence[float],
    *,
    quantile: float = 0.9,
    peak_weight: float = 3.0,
    base_weight: float = 1.0,
    smooth: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """관측 시계열의 outbreak/peak 시점에 상향 가중치를 부여한다 (모델-비종속).

    y의 `quantile` 분위를 임계값으로, 그 이상인 시점을 "peak 구간"으로 보고 `peak_weight`,
    그 외 평시는 `base_weight`를 부여한다. 공중보건 자원배치 평가에서 유행 정점 오차를
    평시 오차보다 무겁게 채점하기 위함.

    동작 규칙:
      - peak 임계 = np.quantile(y_finite, quantile). y[t] >= 임계 → peak_weight.
      - 평탄 시계열 (모든 유한값 동일 = 분산 0) → 전 구간 base_weight 균일 (불변식).
      - NaN/±inf 시점 → base_weight (peak 판정서 제외, leak/폭발 회피).
      - smooth=True: 이웃(±1) peak 인접 시점도 peak_weight로 1칸 팽창 (정점 어깨 포함).

    Args:
        y: 관측 시계열 (n,). 원공간 (y_orig) 권장. 길이 ≥ 1.
        quantile: peak 임계 분위. 0 < quantile < 1 (기본 0.9 = 상위 10%).
        peak_weight: peak 구간 가중치. ≥ 0 (기본 3.0).
        base_weight: 평시 가중치. ≥ 0 (기본 1.0).
        smooth: True면 peak 인접 ±1 시점도 peak로 확장.
        seed: 결정성 RNG seed (현재 무작위성 없음 — 미래 jitter 대비 예약, default_rng).

    Returns:
        weights: (n,) 비음 float64 가중치. 평시=base_weight, peak=peak_weight.

    Raises:
        ValueError: y가 비었거나(n=0), quantile ∉ (0,1), peak_weight/base_weight < 0.

    Performance: O(n) time (분위 1회), O(n) memory. n≈337 (서울 ILI) <1ms.
    Side effects: 없음. RNG는 로컬 (np.random.default_rng(seed)) — global seed 미오염.
    Caller responsibility: y는 원공간 관측값 (transform 공간 금지, G-321). 음수 ILI 없음.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.size
    if n == 0:
        raise ValueError("peak_weights: y는 비어 있을 수 없습니다 (n=0).")
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"peak_weights: quantile은 (0,1) 범위여야 합니다 (got {quantile}).")
    if peak_weight < 0.0 or base_weight < 0.0:
        raise ValueError(
            f"peak_weights: 가중치는 음수일 수 없습니다 "
            f"(peak_weight={peak_weight}, base_weight={base_weight})."
        )

    # 미래 jitter/bootstrap 확장을 위한 결정성 RNG 예약 (현재는 미사용, 결정성 계약 유지).
    _rng = np.random.default_rng(seed)  # noqa: F841

    finite = np.isfinite(y_arr)
    weights = np.full(n, float(base_weight), dtype=np.float64)

    if not finite.any():
        # 전 시점 비유한 → peak 판정 불가, 균일 base.
        return weights

    y_finite = y_arr[finite]
    # 평탄 (분산 0) 시계열 → 균일 가중 (불변식: 평탄은 peak 없음).
    if np.ptp(y_finite) == 0.0:
        return weights

    threshold = float(np.quantile(y_finite, quantile))
    # NaN 시점은 (y >= threshold) 가 False 라 자동 평시 처리.
    is_peak = finite & (y_arr >= threshold)

    if smooth and is_peak.any():
        shifted_l = np.zeros(n, dtype=bool)
        shifted_r = np.zeros(n, dtype=bool)
        shifted_l[:-1] = is_peak[1:]
        shifted_r[1:] = is_peak[:-1]
        is_peak = is_peak | shifted_l | shifted_r

    weights[is_peak] = float(peak_weight)
    return weights


def weighted_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    weights: Sequence[float],
) -> Dict[str, float]:
    """peak-가중 점예측 정확도 지표 묶음 (peak vs overall RMSE/MAE + peak skill).

    weights 가 비균일이면 peak 구간(weights > base) 오차가 더 무겁게 반영된다.
    `peak_*` 키는 weights 가 base 초과인 시점(=peak 구간)만, `overall_*` 키는 전 시점
    균일 채점이라 둘은 일반적으로 다르다 (불변식: peak_rmse ≠ overall_rmse, 비균일 시).

    지표 정의:
      - overall_rmse / overall_mae : 전 시점 균일 RMSE/MAE.
      - peak_rmse / peak_mae       : peak 구간(weights > min(weights))만의 RMSE/MAE.
                                     peak 구간 없으면 (전부 균일) overall 과 동일.
      - peak_skill                 : 1 - peak_wrmse / base_wrmse_naive. peak 구간서
                                     가중 RMSE 가 "persistence baseline (직전 관측)" 대비
                                     얼마나 우수한지 (1=완벽, 0=baseline 동급, <0=열등).
                                     peak 구간 <2 시점 또는 baseline 0 → NaN.

    Args:
        y_true: 관측 (n,). 원공간.
        y_pred: 예측 (n,). 원공간. y_true 와 길이 일치.
        weights: 시점별 비음 가중치 (n,). `peak_weights()` 산출 권장.

    Returns:
        dict: {peak_rmse, peak_mae, overall_rmse, overall_mae, peak_skill, n_peak}.
              모든 값 float (peak_skill/일부는 NaN 가능). n_peak=peak 시점 수.

    Raises:
        ValueError: 길이 불일치, n=0, weights 음수 포함.

    Performance: O(n) time/memory. <1ms for n≈337.
    Side effects: 없음.
    Caller responsibility: y_true/y_pred 원공간, NaN-free 권장 (NaN 시점은 마스크 제외).
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()

    if yt.size == 0:
        raise ValueError("weighted_metrics: y_true는 비어 있을 수 없습니다 (n=0).")
    if not (yt.size == yp.size == w.size):
        raise ValueError(
            f"weighted_metrics: shape 불일치 "
            f"(y_true={yt.size}, y_pred={yp.size}, weights={w.size})."
        )
    if np.any(w < 0.0):
        raise ValueError("weighted_metrics: weights에 음수가 있습니다.")

    # NaN/inf 시점 제외 (leak/폭발 회피) — 모든 입력에서 유한한 시점만 채점.
    finite = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(w)
    if not finite.any():
        return {
            "peak_rmse": float("nan"),
            "peak_mae": float("nan"),
            "overall_rmse": float("nan"),
            "overall_mae": float("nan"),
            "peak_skill": float("nan"),
            "n_peak": 0.0,
        }

    yt_f = yt[finite]
    yp_f = yp[finite]
    w_f = w[finite]
    err = yt_f - yp_f
    abs_err = np.abs(err)

    overall_rmse = float(np.sqrt(np.mean(err ** 2)))
    overall_mae = float(np.mean(abs_err))

    # peak 구간 = 가중치가 최소 가중치 초과인 시점 (peak_weights 의 peak_weight 시점).
    w_min = float(np.min(w_f))
    peak_mask = w_f > w_min
    n_peak = int(np.count_nonzero(peak_mask))

    if n_peak > 0:
        peak_err = err[peak_mask]
        peak_rmse = float(np.sqrt(np.mean(peak_err ** 2)))
        peak_mae = float(np.mean(np.abs(peak_err)))
    else:
        # 균일 가중 (peak 구간 없음) → overall 과 동일 (평탄 시계열 경로).
        peak_rmse = overall_rmse
        peak_mae = overall_mae

    peak_skill = _peak_skill(yt_f, yp_f, peak_mask)

    return {
        "peak_rmse": peak_rmse,
        "peak_mae": peak_mae,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "peak_skill": peak_skill,
        "n_peak": float(n_peak),
    }


def peak_aware_wis(
    y: Sequence[float],
    bounds: Dict[float, Tuple[Sequence[float], Sequence[float]]],
    alphas: Sequence[float],
    weights: Sequence[float],
    median: Optional[Sequence[float]] = None,
) -> float:
    """peak-가중 Weighted Interval Score (Bracher 2021 + 시점별 가중) — 모델-비종속.

    표준 WIS 를 시점별로 계산한 뒤 (per-point), peak 구간을 upweight한 가중평균으로
    스칼라화한다. 즉 WIS_peak = Σ_t w_t·wis_t / Σ_t w_t. weights 가 균일(전부 같은 값)이면
    표준 WIS 평균과 정확히 동일 (불변식). bounds 는 `{alpha: (lower, upper)}` dict —
    `adaptive_conformal.wis_from_bounds`/FluSight 와 동일 규약.

    per-point WIS (Bracher 2021 eq.3):
        wis_t = 1/(K+0.5) · [ 0.5·|y_t - m_t| + Σ_k (α_k/2)·IS_{α_k}(y_t) ]
    where IS_α(y) = (U-L) + (2/α)(L-y)·1[y<L] + (2/α)(y-U)·1[y>U]  (Gneiting-Raftery).

    Args:
        y: 관측 (n,). 원공간.
        bounds: {alpha: (lower, upper)} — 각 (n,) 구간 하·상한. lower ≤ upper 권장.
        alphas: 채점할 PI level 들 (bounds 키의 부분집합). 비면 NaN.
        weights: 시점별 비음 가중치 (n,). peak 시점 상향 권장 (`peak_weights()`).
        median: 중앙값 점예측 (n,). None 이면 가장 넓은 분위 구간 중점을 proxy 사용.

    Returns:
        float: peak-가중 스칼라 WIS (낮을수록 우수). 사용 가능 level 없으면 NaN.

    Raises:
        ValueError: shape 불일치, n=0, weights 음수, Σweights ≤ 0.

    Performance: O(n·K) time (K=len(alphas)), O(n) memory. <1ms for n≈337, K=11.
    Side effects: 없음.
    Caller responsibility: bounds 의 lower/upper 는 원공간 PI (transform 공간 금지).
        weights 는 y 와 동일 origin (leak-free: 평가 시점 weights 만, 미래 미참조).
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.size
    if n == 0:
        raise ValueError("peak_aware_wis: y는 비어 있을 수 없습니다 (n=0).")

    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n:
        raise ValueError(
            f"peak_aware_wis: weights shape 불일치 (y={n}, weights={w.size})."
        )
    if np.any(w < 0.0):
        raise ValueError("peak_aware_wis: weights에 음수가 있습니다.")

    per_point = _per_point_wis(y_arr, bounds, alphas, median)
    if per_point is None:
        return float("nan")

    # NaN-free 시점만 가중평균 (구간 결측/비유한 회피).
    finite = np.isfinite(per_point) & np.isfinite(w)
    if not finite.any():
        return float("nan")
    w_used = w[finite]
    w_sum = float(np.sum(w_used))
    if w_sum <= 0.0:
        raise ValueError("peak_aware_wis: 사용 가능 시점의 weights 합이 0 이하입니다.")

    return float(np.sum(w_used * per_point[finite]) / w_sum)


# --------------------------------------------------------------------------- #
# 내부 헬퍼 (private) — 캡슐화된 구현부 (deep module 의 rich implementation).
# --------------------------------------------------------------------------- #
def _interval_score(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float
) -> np.ndarray:
    """단일 level Gneiting-Raftery interval score (시점별, n,).

    IS_α(y) = (hi-lo) + (2/α)(lo-y)·1[y<lo] + (2/α)(y-hi)·1[y>hi].
    """
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    return (
        (hi - lo)
        + (2.0 / alpha) * (lo - y) * (y < lo)
        + (2.0 / alpha) * (y - hi) * (y > hi)
    )


def _per_point_wis(
    y: np.ndarray,
    bounds: Dict[float, Tuple[Sequence[float], Sequence[float]]],
    alphas: Sequence[float],
    median: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    """bounds dict 로부터 시점별 WIS (n,) 계산. 사용 level 없으면 None.

    median None → 가장 넓은 분위(=최대 α) 구간 중점을 median proxy (wis_from_bounds 규약).
    """
    ks = [a for a in alphas if a in bounds]
    if not ks:
        return None

    n = y.size
    if median is None:
        a0 = max(ks)  # 최대 α = 가장 넓은(=가장 낮은 신뢰) 구간 → 중점 ≈ median.
        lo0, hi0 = bounds[a0]
        lo0 = np.asarray(lo0, dtype=np.float64).ravel()
        hi0 = np.asarray(hi0, dtype=np.float64).ravel()
        m = 0.5 * (lo0 + hi0)
    else:
        m = np.asarray(median, dtype=np.float64).ravel()
    if m.size != n:
        raise ValueError(
            f"peak_aware_wis: median shape 불일치 (y={n}, median={m.size})."
        )

    acc = 0.5 * np.abs(y - m)
    for a in ks:
        lo, hi = bounds[a]
        lo = np.asarray(lo, dtype=np.float64).ravel()
        hi = np.asarray(hi, dtype=np.float64).ravel()
        if lo.size != n or hi.size != n:
            raise ValueError(
                f"peak_aware_wis: bounds[{a}] shape 불일치 "
                f"(y={n}, lo={lo.size}, hi={hi.size})."
            )
        acc = acc + (a / 2.0) * _interval_score(y, lo, hi, a)
    return acc / (len(ks) + 0.5)


def _peak_skill(
    y_true: np.ndarray, y_pred: np.ndarray, peak_mask: np.ndarray
) -> float:
    """peak 구간 가중 RMSE 의 persistence-baseline 대비 skill (1 - mse/mse_naive).

    baseline = persistence (직전 관측 y_{t-1} 로 y_t 예측). peak 구간 <2 시점 또는
    baseline MSE 0 → NaN (정의 불가). 높을수록 우수 (1=완벽, ≤0=baseline 이하).
    """
    if peak_mask is None or np.count_nonzero(peak_mask) < 2:
        return float("nan")

    # persistence baseline: y_hat_t = y_{t-1}; 첫 시점은 자기 자신 (오차 0 제외 위해 shift).
    naive = np.empty_like(y_true)
    naive[0] = y_true[0]
    naive[1:] = y_true[:-1]

    model_err = (y_true - y_pred)[peak_mask]
    naive_err = (y_true - naive)[peak_mask]

    mse_model = float(np.mean(model_err ** 2))
    mse_naive = float(np.mean(naive_err ** 2))
    if mse_naive == 0.0:
        return float("nan")
    return 1.0 - mse_model / mse_naive
