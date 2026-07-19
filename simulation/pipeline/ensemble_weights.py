"""
simulation.pipeline.ensemble_weights
=====================================
Phase D1 — NNLS stacking weights on the WF-CV OOF slab.

The `simulation.models.ensemble.NNLSEnsemble` fits weights on a single
15% validation split (see runner.py:737 — `val_predictions`), which is
too small for stable meta-learning on 343 weeks of Seoul ILI. The
walk-forward out-of-fold signal (phase 7) covers ~280 weeks per model,
giving stacking far more leverage.

This module exposes a pure function that takes the OOF signal (as
returned by phase6_wfcv.run_wfcv) and returns NNLS-normalized,
non-negative weights that sum to 1 — the canonical output shape
expected by `.models.ensemble.NNLSEnsemble.weights`.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy.optimize import nnls

log = logging.getLogger(__name__)

_R2_FLOOR_DEFAULT = 0.3


def _oof_r2(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 2:
        return -np.inf
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return -np.inf
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot


def oof_nnls_weights(
    oof_predictions: Dict[str, np.ndarray],
    y_all: np.ndarray,
    *,
    holdout_start: Optional[int] = None,
    r2_floor: float = _R2_FLOOR_DEFAULT,
) -> Dict[str, float]:
    """NNLS-fit stacking weights from the OOF calibration slab.

 Args:
 oof_predictions: ``{model_name: np.ndarray(n)}``. NaNs mark
 fold-gap positions that were never predicted; those are
 dropped from the design matrix.
 y_all: targets aligned to the OOF slab.
 holdout_start: when set, restrict the design to ``[0, holdout_start)``
 so the weights never see holdout data (mirror phase10_intervals
 S0-1 semantics).
 r2_floor: drop candidates whose OOF R² (on the cal slab) is below
 this threshold. Matches ensemble.NNLSEnsemble's 0.3 floor
 (— prevents structurally bad learners from dominating
 the weight sum).

 Returns:
 ``{name: weight}`` with weights ≥ 0, summing to 1 across accepted
 models. Models filtered out by r2_floor appear with weight 0.
 Empty dict if no candidate passes.
 """
    y = np.asarray(y_all, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y_all must be 1-D, got shape {y.shape}")
    n = y.shape[0]
    end = int(holdout_start) if holdout_start is not None else n
    end = max(0, min(end, n))
    if end == 0:
        return {}

    y_cal = y[:end]

    # Collect models with finite predictions and passing the R² floor.
    accepted: Dict[str, np.ndarray] = {}
    rejected: Dict[str, float] = {}
    for name in sorted(oof_predictions.keys()):
        p_full = np.asarray(oof_predictions[name], dtype=float)
        if p_full.shape[0] < end:
            continue
        p_cal = p_full[:end]
        # joint-valid indices across this model
        mask = np.isfinite(p_cal)
        if int(mask.sum()) < 20:
            continue
        r2 = _oof_r2(y_cal[mask], p_cal[mask])
        if r2 < r2_floor:
            rejected[name] = r2
            continue
        accepted[name] = p_cal

    if not accepted:
        if rejected:
            log.warning(
                "  [oof_nnls_weights] no candidate passes R² floor %.2f; "
                "rejected %s",
                r2_floor,
                {k: round(v, 3) for k, v in rejected.items()},
            )
        return {}

    # Intersect valid indices across the accepted set so the design is
    # rectangular (nnls needs complete rows).
    joint = np.ones(end, dtype=bool)
    for p in accepted.values():
        joint &= np.isfinite(p)
    if int(joint.sum()) < 20:
        return {}

    names = list(accepted.keys())
    X = np.column_stack([accepted[n][joint] for n in names])
    y_fit = y_cal[joint]

    raw, _ = nnls(X, y_fit)
    s = float(raw.sum())
    if s > 0:
        w = raw / s
    else:
        # Degenerate — all zeros. Fall back to equal weights across
        # accepted models rather than dropping the ensemble entirely.
        w = np.ones(len(names)) / len(names)

    weights: Dict[str, float] = {n: float(v) for n, v in zip(names, w)}
    for n in rejected:
        weights[n] = 0.0
    return weights


def blend_oof_predictions(
    oof_predictions: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """Produce the blended OOF series implied by ``weights``.

    NaN in any weighted contributor propagates to the output at that
    position — the caller can decide to fill or mask.
    """
    if not oof_predictions or not weights:
        return np.array([])
    # Length = min over contributing models
    contribs = [(n, w) for n, w in weights.items() if w > 0 and n in oof_predictions]
    if not contribs:
        return np.array([])
    m = min(oof_predictions[n].shape[0] for n, _ in contribs)
    out = np.zeros(m, dtype=float)
    total = 0.0
    for n, w in contribs:
        out += w * np.asarray(oof_predictions[n][:m], dtype=float)
        total += w
    if total > 0 and abs(total - 1.0) > 1e-9:
        out /= total
    return out
