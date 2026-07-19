"""Post-hoc Pareto analysis for calibration-sharpness trade-off
(audit Stage 2.2, Task #17).

Approach:
    Single-objective WIS minimize 유지 (학습 entry 변경 최소화) +
    학습 후 각 trial 의 (wis, |picp95 - 0.95|) 산출 →
    Pareto front 추출 + HV-based champion 결정.

    Audit 의 spirit (calibration-sharpness trade-off 명시) 충족하면서
    학습 시간 영향 없음. 본격 multi-objective Optuna (MOTPESampler) 도입은
    별도 sprint 후보 (학습 시간 2-3배 증가).

Reference:
    Gneiting T, Balabdaoui F, Raftery AE (2007)
    "Probabilistic forecasts, calibration and sharpness"
    JRSSB 69(2):243-268. doi:10.1111/j.1467-9868.2007.00587.x
    "maximizing the sharpness of the predictive distributions subject to
     calibration."

    Bracher J, Ray EL, Gneiting T, Reich NG (2021)
    "Evaluating epidemic forecasts in an interval format"
    PLoS Comput Biol 17(2):e1008618. doi:10.1371/journal.pcbi.1008618

D-5 gray-box contract:
    - O(N²) pareto front extraction; for N=53 model OK.
    - HV computation: 2D hypervolume (simple closed-form).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "extract_pareto_front",
    "hypervolume_2d",
    "pareto_champion",
    "calibration_sharpness_decomposition",
]


def calibration_sharpness_decomposition(
    picp95: float, picp95_target: float = 0.95
) -> dict:
    """Audit Stage 2.2 — explicit calibration metric.

    |picp95 - 0.95| = miscalibration in 95% PI (lower better).
    """
    return {
        "miscalibration_picp95": (abs(picp95 - picp95_target)
                                    if np.isfinite(picp95) else float("nan")),
        "target": picp95_target,
    }


def extract_pareto_front(
    objectives: np.ndarray,
    names: Optional[list[str]] = None,
    *,
    minimize: bool = True,
) -> dict:
    """Pareto front extraction (2 or more objectives).

    Args:
        objectives: (n_points, n_obj) — each row a point in objective space.
                     Lower=better if minimize=True.
        names: (n_points,) point identifiers.
        minimize: True = lower better, False = higher better.

    Returns:
        dict {
            "is_pareto": list[bool] (n_points),  # True = on Pareto front
            "pareto_names": list[str],            # names on front
            "pareto_objectives": list[list[float]],  # objectives of front members
        }

    Performance: O(N^2) — for N=53 model OK.
    """
    out = {"is_pareto": [], "pareto_names": [], "pareto_objectives": []}
    if objectives is None or len(objectives) == 0:
        return out
    obj = np.asarray(objectives, dtype=np.float64)
    if obj.ndim != 2:
        return out
    n, k = obj.shape
    if names is None:
        names = [f"point_{i}" for i in range(n)]

    sign = 1.0 if minimize else -1.0
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(obj[i]).all():
            is_pareto[i] = False
            continue
        for j in range(n):
            if i == j or not np.isfinite(obj[j]).all():
                continue
            # j dominates i if: all obj_j <= obj_i AND any obj_j < obj_i
            dominates_all = np.all(sign * obj[j] <= sign * obj[i])
            dominates_strict = np.any(sign * obj[j] < sign * obj[i])
            if dominates_all and dominates_strict:
                is_pareto[i] = False
                break

    out["is_pareto"] = is_pareto.tolist()
    out["pareto_names"] = [names[i] for i in range(n) if is_pareto[i]]
    out["pareto_objectives"] = obj[is_pareto].tolist()
    return out


def hypervolume_2d(
    pareto_points: np.ndarray,
    reference_point: tuple[float, float],
    *,
    minimize: bool = True,
) -> float:
    """2D hypervolume (HV) for Pareto front.

    HV = area of dominated region between Pareto front and reference point.
    For minimize: HV = sum of rectangles (each (x_{i+1} - x_i) × (ref_y - y_i)).

    Args:
        pareto_points: (n_pareto, 2) — Pareto front (lower better).
        reference_point: (ref_x, ref_y) — worst-case reference.

    Returns:
        float — hypervolume.
    """
    if len(pareto_points) == 0:
        return 0.0
    pts = np.asarray(pareto_points, dtype=np.float64)
    if pts.shape[1] != 2:
        return float("nan")
    ref_x, ref_y = reference_point

    if minimize:
        # sort by x ascending, dominance check: y must be decreasing
        order = np.argsort(pts[:, 0])
        pts_sorted = pts[order]
        # HV: sum (x_{i+1} - x_i) × (ref_y - y_i)
        hv = 0.0
        prev_x = pts_sorted[0, 0]
        for i in range(len(pts_sorted)):
            x_i, y_i = pts_sorted[i]
            if y_i >= ref_y or x_i >= ref_x:
                continue
            x_next = pts_sorted[i + 1, 0] if i + 1 < len(pts_sorted) else ref_x
            x_next = min(x_next, ref_x)
            width = x_next - x_i
            height = ref_y - y_i
            if width > 0 and height > 0:
                hv += width * height
        return float(hv)
    else:
        # maximize: similar but flipped
        return float("nan")  # not implemented for max


def pareto_champion(
    wis_values: dict[str, float],
    picp95_values: dict[str, float],
    *,
    picp95_target: float = 0.95,
    wis_ceiling: float = 6.0,
    miscalib_ceiling: float = 0.05,
) -> dict:
    """Audit Stage 2.2 — calibration-sharpness Pareto champion.

    Objective: minimize (wis, |picp95 - 0.95|).
    Filter: wis <= wis_ceiling AND |picp95 - 0.95| <= miscalib_ceiling.

    Args:
        wis_values: dict[model, wis]
        picp95_values: dict[model, picp95]
        picp95_target: 0.95 (FluSight nominal)
        wis_ceiling: 6.0 (per G-175 forward)
        miscalib_ceiling: 0.05 (5%p tolerance)

    Returns:
        dict {
            "pareto_members": list[str],
            "pareto_filtered": list[str],   # 추가 filter 적용 후 (champion candidate)
            "hypervolume": float,
            "best_by_wis": str,
            "best_by_calib": str,
            "reference": "Gneiting/Balabdaoui/Raftery (2007) doi:10.1111/j.1467-9868.2007.00587.x"
        }
    """
    common_names = sorted(set(wis_values.keys()) & set(picp95_values.keys()))
    if not common_names:
        return {
            "pareto_members": [], "pareto_filtered": [],
            "hypervolume": float("nan"),
            "best_by_wis": "", "best_by_calib": "",
            "reference": "Gneiting/Balabdaoui/Raftery (2007) doi:10.1111/j.1467-9868.2007.00587.x"
        }

    wis = np.array([wis_values[n] for n in common_names], dtype=np.float64)
    picp = np.array([picp95_values[n] for n in common_names], dtype=np.float64)
    miscalib = np.abs(picp - picp95_target)

    objectives = np.column_stack([wis, miscalib])
    front = extract_pareto_front(objectives, common_names, minimize=True)
    members = front["pareto_names"]

    # Filter: wis <= ceiling AND miscalib <= ceiling
    filtered = []
    for n in members:
        i = common_names.index(n)
        if (np.isfinite(wis[i]) and wis[i] <= wis_ceiling
                and np.isfinite(miscalib[i]) and miscalib[i] <= miscalib_ceiling):
            filtered.append(n)

    # HV (using wis_ceiling + 1 as reference upper bound)
    ref = (wis_ceiling * 1.5, miscalib_ceiling * 2.0)
    hv = hypervolume_2d(np.array(front["pareto_objectives"]) if members else np.empty((0, 2)),
                          ref, minimize=True)

    # Best by single criterion
    finite_wis = [(n, wis_values[n]) for n in common_names if np.isfinite(wis_values[n])]
    best_wis = min(finite_wis, key=lambda x: x[1])[0] if finite_wis else ""
    finite_calib = [(n, abs(picp95_values[n] - picp95_target))
                     for n in common_names if np.isfinite(picp95_values[n])]
    best_calib = min(finite_calib, key=lambda x: x[1])[0] if finite_calib else ""

    return {
        "pareto_members": members,
        "pareto_filtered": filtered,
        "hypervolume": hv,
        "best_by_wis": best_wis,
        "best_by_calib": best_calib,
        "n_models": len(common_names),
        "n_pareto": len(members),
        "n_filtered": len(filtered),
        "reference": "Gneiting/Balabdaoui/Raftery (2007) doi:10.1111/j.1467-9868.2007.00587.x"
    }
