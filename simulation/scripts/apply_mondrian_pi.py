"""Mondrian Conformal PI — post-hoc 적용 (Tier 1 #1, Package C B-A wiring).

학습 종료 후 calibration residuals 에 Mondrian conformal 을 적용해
**per-group PI quantile** 산출. 글로벌 conformal 보다 PICP@95 정확도 향상.

ENGINEERING_PRINCIPLES.md §원칙 #5 (재현성): seasonal group / district group 별 quantile.

사용:
    .venv/bin/python -m simulation.scripts.apply_mondrian_pi \\
        --residuals simulation/results/per_model_optimal/ElasticNet.pkl \\
        --groups seasonal \\
        --alpha 0.05

출력:
    simulation/results/mondrian_pi_<model>.json — per-group PI bounds + PICP

근거:
    Foygel-Barber et al. 2021 — Mondrian Conformal
    Romano et al. 2019 — Conformalized Quantile Regression
    Vovk et al. 2005 — Algorithmic Learning in a Random World

ENGINEERING_PRINCIPLES.md 원칙 매핑:
    #5 재현성 — calibration set + group 정의가 결정적
    #4 KISS — 단일 utility, 모든 모델 동일 호출
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np


def seasonal_group(weeks: np.ndarray) -> np.ndarray:
    """Season ID by week-of-year (1-52).

    Returns 0/1/2:
      0 = winter peak (week 49-12)
      1 = shoulder    (week 13-24, 37-48)
      2 = summer low  (week 25-36)
    """
    weeks = np.asarray(weeks) % 52 + 1   # 1-52
    out = np.full_like(weeks, 1, dtype=int)  # default = shoulder
    out[(weeks >= 49) | (weeks <= 12)] = 0   # winter
    out[(weeks >= 25) & (weeks <= 36)] = 2   # summer
    return out


def magnitude_group(y: np.ndarray, n_groups: int = 3) -> np.ndarray:
    """ILI magnitude tertile (low / mid / high)."""
    y = np.asarray(y)
    quantiles = np.quantile(y, np.linspace(0, 1, n_groups + 1)[1:-1])
    return np.digitize(y, quantiles)


def mondrian_quantile(
    residuals: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.05,
    min_per_group: int = 5,
    fallback_global: bool = True,
) -> dict[int, float]:
    """Per-group |residual| quantile (1-α).

    Args:
        residuals: 1-D array of (y_true - y_pred) on calibration set
        groups: 1-D array of same length, integer group label
        alpha: 1-alpha coverage (0.05 → 95% PI)
        min_per_group: 그룹 sample <이 값이면 글로벌 fallback
        fallback_global: True 면 small-group → 글로벌 quantile 대체

    Returns:
        {group_id: quantile}, 추가 키 "__global__" = 글로벌 quantile
    """
    residuals = np.asarray(residuals)
    groups = np.asarray(groups)
    q_target = 1.0 - alpha
    global_q = float(np.quantile(np.abs(residuals), q_target))

    out: dict[int, float] = {}
    for g in np.unique(groups):
        mask = groups == g
        n_g = int(mask.sum())
        if n_g >= min_per_group:
            out[int(g)] = float(np.quantile(np.abs(residuals[mask]), q_target))
        elif fallback_global:
            out[int(g)] = global_q
        else:
            out[int(g)] = float("nan")
    out["__global__"] = global_q
    return out


def apply_pi(
    predictions: np.ndarray,
    groups: np.ndarray,
    group_quantiles: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply group-specific quantiles → (lower, upper) PI bounds."""
    predictions = np.asarray(predictions)
    groups = np.asarray(groups)
    q_arr = np.array([
        group_quantiles.get(int(g), group_quantiles["__global__"])
        for g in groups
    ])
    return predictions - q_arr, predictions + q_arr


def evaluate_picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict:
    """PICP@95 + per-group breakdown."""
    y_true = np.asarray(y_true)
    inside = (y_true >= lower) & (y_true <= upper)
    overall = float(inside.mean())
    out = {"PICP_overall": overall, "n": int(len(y_true))}
    if groups is not None:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() > 0:
                out[f"PICP_group_{int(g)}"] = float(inside[mask].mean())
                out[f"n_group_{int(g)}"] = int(mask.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residuals", required=True, help="path to residuals.npy or .json")
    ap.add_argument("--predictions", required=True, help="path to test predictions")
    ap.add_argument("--y-true", required=True)
    ap.add_argument("--weeks", required=True, help="week index for groups")
    ap.add_argument("--groups", choices=("seasonal", "magnitude"), default="seasonal")
    ap.add_argument("--alpha", type=float, default=0.05)
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--out", default=str(get_results_dir() / "mondrian_pi.json"))
    args = ap.parse_args()

    # Load arrays (numpy or json)
    def _load(p):
        p = Path(p)
        if p.suffix == ".npy":
            return np.load(p)
        return np.array(json.loads(p.read_text(encoding="utf-8")))

    residuals = _load(args.residuals)
    predictions = _load(args.predictions)
    y_true = _load(args.y_true)
    weeks = _load(args.weeks)

    if args.groups == "seasonal":
        groups = seasonal_group(weeks)
        group_names = ["winter_peak", "shoulder", "summer_low"]
    else:
        groups = magnitude_group(y_true)
        group_names = ["low", "mid", "high"]

    # Mondrian quantile + PI
    q = mondrian_quantile(residuals, groups, alpha=args.alpha)
    lower, upper = apply_pi(predictions, groups, q)

    # Evaluation
    metrics = evaluate_picp(y_true, lower, upper, groups)

    # Output
    out = {
        "method": "Mondrian Conformal (Foygel-Barber 2021)",
        "alpha": args.alpha,
        "group_strategy": args.groups,
        "group_names": group_names,
        "quantiles_per_group": q,
        "metrics": metrics,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"✓ Mondrian PI 저장: {args.out}")
    print(f"  PICP overall: {metrics['PICP_overall']:.3f} (target: {1-args.alpha:.2f})")
    for k, v in metrics.items():
        if k.startswith("PICP_group"):
            print(f"    {k}: {v:.3f}")


if __name__ == "__main__":
    sys.exit(main())
