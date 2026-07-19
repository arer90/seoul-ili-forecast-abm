"""A7 (M7 SCI-grade): alert operating curve + matched-sensitivity lead-time.

``alert_f1`` and ``lead_time_weeks`` are two of the three Table-1 PRIMARY metrics,
but were reported at a SINGLE season-specific KDCA threshold — fragile when the
slab has few crossing events and the KDCA threshold itself is a noisy mean+2σ.
This sweeps the alert threshold (F1 / sensitivity / specificity per threshold,
with Wilson CIs on sensitivity for the small-n binomial) and selects the
threshold at a fixed sensitivity operating point so cross-model lead-times are
compared like-with-like, not at the raw KDCA crossing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from simulation.analytics.hub_metrics import wilson_score_ci


def alert_operating_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray,
    alpha: float = 0.05,
) -> list[dict]:
    """F1 / sensitivity / specificity swept over alert thresholds.

    An "alert" at threshold ``thr`` = value ≥ ``thr``. For each threshold returns
    F1, sensitivity (+ Wilson CI), specificity, and the event count, so alert
    performance is a curve, not one fragile operating point.

    Args:
        y_true: (n,) observed values.
        y_pred: (n,) forecast values.
        thresholds: iterable of alert thresholds to sweep.
        alpha: CI miscoverage (0.05 → 95% Wilson CI on sensitivity).

    Returns:
        list of per-threshold dicts ``{threshold, f1, sensitivity, sens_ci,
        specificity, n_events, tp, fp, fn, tn}``. Never raises.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out: list[dict] = []
    for thr in np.asarray(thresholds, dtype=np.float64):
        event = y_true >= thr
        alert = y_pred >= thr
        tp = int(np.sum(event & alert))
        fp = int(np.sum(~event & alert))
        fn = int(np.sum(event & ~alert))
        tn = int(np.sum(~event & ~alert))
        n_pos, n_neg = tp + fn, tn + fp
        sens = tp / n_pos if n_pos else float("nan")
        spec = tn / n_neg if n_neg else float("nan")
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom else float("nan")
        _pt, lo, hi = wilson_score_ci(tp, n_pos, alpha) if n_pos else (
            float("nan"), float("nan"), float("nan"))
        out.append({
            "threshold": float(thr), "f1": round(f1, 4) if np.isfinite(f1) else f1,
            "sensitivity": sens, "sens_ci": [lo, hi], "specificity": spec,
            "n_events": n_pos, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return out


def threshold_at_sensitivity(curve: list[dict], target_sens: float = 0.9) -> Optional[float]:
    """Highest (most specific) threshold whose sensitivity ≥ ``target_sens``.

    Used to put every model at a MATCHED sensitivity operating point before
    comparing lead-times, instead of the raw KDCA threshold (A7/M7). Returns None
    when no swept threshold reaches the target.
    """
    ok = [r for r in curve
          if isinstance(r.get("sensitivity"), float)
          and np.isfinite(r["sensitivity"]) and r["sensitivity"] >= target_sens]
    if not ok:
        return None
    return max(ok, key=lambda r: r["threshold"])["threshold"]
