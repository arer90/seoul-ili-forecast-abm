"""
simulation.verifier.epi_validity
=================================
Epidemiological plausibility guards (§11, RECOMMENDED_PIPELINE.md ).

Checks SEIR-V-D parameters + forecasts against literature-backed ranges.
Out-of-range values fail the verifier hook so we never publish predictions
that violate basic epi constraints (e.g. negative incidence, R0 > 20).

Ranges (widely-cited seasonal influenza values):
 * R0 [0.5, 4.0] Biggerstaff 2014, Chowell 2008
 * σ (E→I rate) [1/4, 1.0]/d Carrat 2008 incubation 1-4d
 * γ (I→R rate) [1/7, 1/2]/d Tuite 2010 infectious 2-7d
 * VE [0.0, 0.85] CDC VE estimates 2010-2023
 * ILI rate [0, 100] ‰ biological upper bound
 * weekly cases [0, 1e7] sanity cap (Seoul pop = 9.4M)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .decorators import CheckerResult

log = logging.getLogger(__name__)


class EpiValidityError(RuntimeError):
    """Raised when epi parameters or outputs violate plausibility ranges."""


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float
    unit: str = ""
    citation: str = ""

    def check(self, x: float) -> bool:
        return self.lo <= x <= self.hi


# Stage 4 tightenings (see docs/internal/stage_plan.md §Stage 4):
#   Re  : lower bound 0.0 → 0.3  (below 0.3 implies elimination, not endemic seasonal flu)
#   VE  : 0.0–0.85 → 0.50–0.95   (TND effective VE for seasonal match, CDC 2010-2023)
#   ifr : 0.0–0.10 → 0.0001–0.05 (seasonal ILI 0.01–5 %; upper bound stays 5 % pandemic cap)
# R0 kept wide (0.5–4.0) because R0 describes a fully-susceptible pop, which is
# not the paper target; Re is the operationally meaningful one.
EPI_RANGE: dict[str, Range] = {
    "R0":        Range(0.5, 4.0, "", "Biggerstaff 2014"),
    "Re":        Range(0.3, 5.0, "", "Cori 2013 EpiEstim + Stage 4 floor"),
    "sigma":     Range(1/4, 1.0, "/day", "Carrat 2008 incubation 1-4d"),
    "gamma":     Range(1/7, 1/2, "/day", "Tuite 2010 infectious 2-7d"),
    "VE":        Range(0.10, 0.95, "", "CDC TND 2010-2023; floor 0.50→0.10 (C4/M7) to admit drift-mismatch seasons (published TND VE 10-60%)"),
    "ili_rate":  Range(0.0, 100.0, "permille", "biological cap"),
    "weekly_cases": Range(0.0, 1e7, "", "Seoul pop 9.4M"),
    "ifr":       Range(0.0001, 0.05, "", "seasonal ILI 0.01-5 %"),
    "cfr":       Range(0.0, 0.5, "", "upper-bound"),
}

# Stage 4: operational thresholds that are not "bounds on a single value"
# but apply to sequences / relationships. Exposed as module-level constants so
# tests and the gate can reference them explicitly.
RT_DELTA_CAP: float = 1.5        # |Rt(w+1) - Rt(w)| ≤ this
COMPARTMENT_TOL: float = 1e-6    # |S+E+I+R+V+D - N| / N ≤ this
SEASONAL_PEAK_WEEKS: frozenset[int] = (
    frozenset(range(48, 53)) | frozenset(range(1, 9))
)  # Korean influenza seasonality window (W48-W8)

# Known influenza outbreak peak anchors in the Seoul ILI record. Tolerance is
# ±2 ISO weeks; predictions that miss the anchor by more than that are flagged.
# Source: KDCA 주간 감염병소식 archives; see docs/internal/analysis_protocol.md.
KNOWN_OUTBREAK_PEAKS: dict[str, tuple[int, int]] = {
    # label: (iso_year, iso_week)
    "2009_H1N1":   (2009, 45),
    "2017-18_H3":  (2018, 2),
    "2020_COVID":  (2020, 10),   # NPI-induced drop, not a peak — checked as trough
}


def check_epi_validity(
    params: Optional[dict[str, Any]] = None,
    predictions: Optional[np.ndarray] = None,
    *,
    ranges: Optional[dict[str, Range]] = None,
    raise_on_fail: bool = False,
) -> CheckerResult:
    """Validate epidemiological parameters and/or prediction arrays.

    Parameters
    ----------
    params : dict
        e.g. {"R0": 1.3, "sigma": 0.5, "gamma": 0.25, "VE": 0.4}
    predictions : np.ndarray
        Forecast array. Will check: no NaN, no negative, no huge outliers.
    ranges : optional override
    """
    ranges = ranges or EPI_RANGE
    violations: list[str] = []
    warnings_: list[str] = []

    # 1) Param-range checks
    if params:
        for k, v in params.items():
            if k not in ranges:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                violations.append(f"{k}={v!r} (non-numeric)")
                continue
            if not ranges[k].check(fv):
                violations.append(
                    f"{k}={fv:.4g} outside [{ranges[k].lo}, {ranges[k].hi}] "
                    f"({ranges[k].unit}) [{ranges[k].citation}]"
                )

    # 2) Prediction-array checks
    if predictions is not None:
        arr = np.asarray(predictions, dtype=float).ravel()
        if arr.size:
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            n_neg = int((arr < 0).sum())
            max_v = float(np.nanmax(arr)) if arr.size else float("nan")
            if n_nan:
                violations.append(f"predictions: {n_nan} NaN values")
            if n_inf:
                violations.append(f"predictions: {n_inf} Inf values")
            if n_neg:
                violations.append(f"predictions: {n_neg} negative values "
                                  "(ILI rate must be ≥ 0)")
            # Range sanity: ILI in permille
            ili_max = ranges.get("ili_rate", Range(0, 100)).hi
            if np.isfinite(max_v) and max_v > ili_max:
                warnings_.append(
                    f"predictions: max={max_v:.2f} exceeds ILI cap {ili_max} ‰"
                )

    status = "ok"
    if violations:
        status = "fail"
    elif warnings_:
        status = "warn"

    details = {"violations": violations, "warnings": warnings_}
    if raise_on_fail and status == "fail":
        raise EpiValidityError("; ".join(violations))
    return CheckerResult(status=status, checker="epi_validity", details=details)


# ══════════════════════════════════════════════════════════════════════════
# Stage 4 — operational gate checks
# ══════════════════════════════════════════════════════════════════════════
#
# The single-value EPI_RANGE test above catches parameter typos but misses
# three classes of failure that show up in real forecasts:
#
#   (1) Rt sequence with implausible week-over-week jumps — typically a
#       sign of a model hallucinating a regime shift, not an epidemic.
#   (2) Metapopulation simulators that quietly leak mass (S+E+I+R+V+D ≠ N)
#       after an integrator step, making the "total population" drift.
#   (3) Forecasts whose peak falls in July-August for Korean influenza,
#       which is non-physical given the seasonal window W48–W8.
#
# Each function below is small and independent so tests can drive them in
# isolation; ``run_epi_validity_gate`` composes them.


def _iso_week_in_set(week: int, wk_set: frozenset[int]) -> bool:
    """Helper: tolerate int or float ISO-week values."""
    try:
        return int(week) in wk_set
    except (TypeError, ValueError):
        return False


def check_rt_sequence(
    rt_values,
    *,
    lo: float = 0.3,
    hi: float = 8.0,   # G-184 (2026-05-06): 5.0 → 8.0 (ENGINEERING_PRINCIPLES.md Rt ∈ [0.3, 8] spec 일치, pandemic scenario 허용)
    delta_cap: float = RT_DELTA_CAP,
) -> CheckerResult:
    """Validate a sequence of reproduction-number estimates.

    Two constraints:
      (a) each Rt_t ∈ [lo, hi] — below 0.3 implies elimination, above 8 is
          beyond any observed influenza estimate (Cori 2013, seasonal ~1.5;
          1918 pandemic peak ~3-4; safety margin 8.0 per ENGINEERING_PRINCIPLES.md spec).
      (b) |Rt_{t+1} − Rt_t| ≤ delta_cap — week-over-week swings bigger
          than this are not biologically realistic for influenza at the
          weekly aggregation used by KDCA.
    """
    arr = np.asarray(rt_values, dtype=float).ravel()
    violations: list[str] = []
    if arr.size == 0:
        return CheckerResult(status="warn", checker="rt_sequence",
                             details={"warnings": ["empty Rt array"]})

    n_nan = int(np.isnan(arr).sum())
    if n_nan:
        violations.append(f"Rt: {n_nan} NaN values")

    if arr.size:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lo_viol = int((finite < lo).sum())
            hi_viol = int((finite > hi).sum())
            if lo_viol:
                violations.append(
                    f"Rt: {lo_viol} values < {lo} (below elimination floor)"
                )
            if hi_viol:
                violations.append(
                    f"Rt: {hi_viol} values > {hi} (above Cori 2013 ceiling)"
                )

    if arr.size >= 2:
        dR = np.diff(arr)
        n_big = int((np.abs(dR) > delta_cap).sum())
        if n_big:
            max_jump = float(np.nanmax(np.abs(dR)))
            violations.append(
                f"Rt: {n_big} week-over-week |ΔRt| > {delta_cap} "
                f"(max |ΔRt| = {max_jump:.2f})"
            )

    status = "fail" if violations else "ok"
    return CheckerResult(
        status=status, checker="rt_sequence",
        details={"violations": violations, "n": int(arr.size)},
    )


def check_compartment_conservation(
    compartments: dict[str, Any] | np.ndarray,
    N_total,
    *,
    tol: float = COMPARTMENT_TOL,
) -> CheckerResult:
    """Verify S+E+I+R+V+D = N per timestep.

    ``compartments``:
        * dict of arrays keyed by compartment name (e.g. {"S": ..., "E": ...}),
          all of shape (T,) or (T, G). Unknown keys are summed regardless.
        * or a 2D/3D array whose last-axis sum equals the compartment total.
    ``N_total``:
        scalar or array broadcastable to the compartment shape.
    """
    if isinstance(compartments, dict):
        arrs = [np.asarray(v, dtype=float) for v in compartments.values()]
        if not arrs:
            return CheckerResult(status="warn", checker="compartment_conservation",
                                 details={"warnings": ["empty compartment dict"]})
        try:
            stack = np.stack(arrs, axis=-1)   # (T[,G], K)
        except ValueError as e:
            return CheckerResult(
                status="fail", checker="compartment_conservation",
                details={"violations": [f"shape mismatch across compartments: {e}"]},
            )
    else:
        stack = np.asarray(compartments, dtype=float)

    total = stack.sum(axis=-1)
    N_arr = np.asarray(N_total, dtype=float)
    try:
        resid = np.abs(total - N_arr)
    except ValueError as e:
        return CheckerResult(
            status="fail", checker="compartment_conservation",
            details={"violations": [f"N_total shape incompatible: {e}"]},
        )
    denom = np.where(np.abs(N_arr) > 0, np.abs(N_arr), 1.0)
    rel_err = resid / denom
    max_rel = float(np.nanmax(rel_err)) if rel_err.size else 0.0

    violations: list[str] = []
    if not np.all(np.isfinite(total)):
        violations.append("compartment totals contain NaN/Inf")
    if max_rel > tol:
        n_bad = int((rel_err > tol).sum())
        violations.append(
            f"S+E+I+R+V+D ≠ N: {n_bad} timesteps with rel-error > {tol:.1e} "
            f"(max = {max_rel:.2e})"
        )

    status = "fail" if violations else "ok"
    return CheckerResult(
        status=status, checker="compartment_conservation",
        details={"violations": violations, "max_rel_err": max_rel},
    )


def _peak_index(arr: np.ndarray) -> int:
    """Argmax that tolerates NaN (skip if any finite values exist)."""
    if not arr.size:
        return -1
    if np.all(np.isnan(arr)):
        return -1
    return int(np.nanargmax(arr))


def check_seasonal_peak(
    predictions,
    iso_weeks,
    *,
    allowed: frozenset[int] = SEASONAL_PEAK_WEEKS,
) -> CheckerResult:
    """Predicted season peak week must fall in the allowed ISO-week set.

    For Korean seasonal influenza the empirical peak window is W48–W8
    (wrapping through New Year). Predictions that put the peak in
    July–August are non-physical and should be flagged.

    ``predictions`` : (T,) array of weekly rates/counts
    ``iso_weeks``   : (T,) array of ISO week numbers (1-53) aligned with predictions
    """
    arr = np.asarray(predictions, dtype=float).ravel()
    wks = np.asarray(iso_weeks).ravel()
    if arr.size == 0 or wks.size == 0:
        return CheckerResult(status="warn", checker="seasonal_peak",
                             details={"warnings": ["empty predictions or weeks"]})
    if arr.size != wks.size:
        return CheckerResult(
            status="fail", checker="seasonal_peak",
            details={"violations": [
                f"length mismatch: predictions={arr.size} vs weeks={wks.size}"
            ]},
        )
    idx = _peak_index(arr)
    if idx < 0:
        return CheckerResult(status="warn", checker="seasonal_peak",
                             details={"warnings": ["predictions all NaN"]})
    peak_week = wks[idx]
    ok = _iso_week_in_set(peak_week, allowed)
    details = {"peak_week": int(peak_week) if _iso_week_in_set(peak_week, allowed | frozenset(range(1, 54))) else peak_week,
               "allowed_window": sorted(allowed)}
    if ok:
        return CheckerResult(status="ok", checker="seasonal_peak", details=details)
    details["violations"] = [
        f"peak at ISO-W{peak_week} is outside the seasonal window W48–W8"
    ]
    return CheckerResult(status="fail", checker="seasonal_peak", details=details)


def check_outbreak_alignment(
    predictions,
    iso_year_weeks,
    *,
    anchors: dict[str, tuple[int, int]] = KNOWN_OUTBREAK_PEAKS,
    tolerance_weeks: int = 2,
) -> CheckerResult:
    """Known outbreak peaks (2009 H1N1 wk45, 2017-18 wk02) must align.

    For each anchor in ``anchors`` that falls inside the covered period,
    the predicted value at the anchor week must be among the top-decile
    of the whole series (local maximum check). Miss distance > tolerance
    flags the anchor as misaligned.

    ``iso_year_weeks``: (T, 2) array of (iso_year, iso_week) pairs.
    """
    arr = np.asarray(predictions, dtype=float).ravel()
    iyw = np.asarray(iso_year_weeks)
    if arr.size == 0 or iyw.size == 0:
        return CheckerResult(status="warn", checker="outbreak_alignment",
                             details={"warnings": ["empty predictions or dates"]})
    if iyw.ndim != 2 or iyw.shape[1] != 2 or iyw.shape[0] != arr.size:
        return CheckerResult(
            status="fail", checker="outbreak_alignment",
            details={"violations": [
                f"iso_year_weeks shape invalid: expected ({arr.size}, 2), "
                f"got {iyw.shape}"
            ]},
        )
    if np.all(np.isnan(arr)):
        return CheckerResult(status="warn", checker="outbreak_alignment",
                             details={"warnings": ["predictions all NaN"]})

    threshold = float(np.nanpercentile(arr, 90))
    violations: list[str] = []
    details_anchors: dict[str, dict] = {}

    for label, (y, w) in anchors.items():
        # Find rows within tolerance of the anchor
        mask = np.zeros(arr.size, dtype=bool)
        for i in range(arr.size):
            iy, iw = int(iyw[i, 0]), int(iyw[i, 1])
            if iy != y:
                continue
            # ISO-week circular distance within the same year (good enough
            # for the Jan-peak anchors; H1N1 anchor has no wrap risk).
            if abs(iw - w) <= tolerance_weeks:
                mask[i] = True
        if not mask.any():
            details_anchors[label] = {"covered": False}
            continue
        local_max = float(np.nanmax(arr[mask]))
        details_anchors[label] = {
            "covered": True,
            "local_max": local_max,
            "global_p90": threshold,
            "aligned": bool(local_max >= threshold),
        }
        if local_max < threshold:
            violations.append(
                f"anchor {label} ({y}-W{w}) local max {local_max:.2f} "
                f"< global 90th percentile {threshold:.2f}"
            )

    status = "fail" if violations else ("warn" if not details_anchors else "ok")
    return CheckerResult(
        status=status, checker="outbreak_alignment",
        details={"violations": violations, "reference_points": details_anchors,
                 "tolerance_weeks": tolerance_weeks},
    )


def run_epi_validity_gate(
    model_outputs: dict[str, dict],
    *,
    strict_exclude: bool = False,
) -> dict[str, dict]:
    """Apply all applicable Stage-4 checks to each model's output.

    ``model_outputs`` shape::
        {
            "XGBoost":  {"predictions": np.ndarray,
                         "iso_weeks":   np.ndarray (optional, 1d),
                         "iso_year_weeks": np.ndarray (optional, 2d),
                         "rt":          np.ndarray (optional, 1d),
                         "compartments": dict (optional, SEIR models only),
                         "N_total":     float or array (optional, SEIR models),
                         "params":      dict (optional, SEIR models)},
            ...
        }

    Returns::
        {model_name: {"status": "ok" | "warn" | "fail",
                      "checks": {checker_name: CheckerResult.details},
                      "violations": [flattened list of violation strings],
                      "exclude_from_ensemble": bool}}

    ``strict_exclude=True`` flips the ``exclude_from_ensemble`` flag on
    any ``fail``. The default (False) only FLAGS — callers decide
    whether to drop. This matches the Stage-4 design note
    ("flag 만 저장, 강제 제외는 opt-in").
    """
    report: dict[str, dict] = {}
    for name, out in model_outputs.items():
        checks: dict[str, dict] = {}
        violations: list[str] = []

        preds = out.get("predictions")
        params = out.get("params")
        rt = out.get("rt")
        iso_weeks = out.get("iso_weeks")
        iso_yw = out.get("iso_year_weeks")
        comps = out.get("compartments")
        N_total = out.get("N_total")

        # 1) Base param + prediction sanity (existing check)
        base = check_epi_validity(params=params, predictions=preds)
        checks[base.checker] = base.details
        violations.extend(base.details.get("violations", []))

        # 2) Rt sequence (only if provided)
        if rt is not None:
            r = check_rt_sequence(rt)
            checks[r.checker] = r.details
            violations.extend(r.details.get("violations", []))

        # 3) Compartment conservation (SEIR-V-D / PINN only)
        if comps is not None and N_total is not None:
            c = check_compartment_conservation(comps, N_total)
            checks[c.checker] = c.details
            violations.extend(c.details.get("violations", []))

        # 4) Seasonal peak
        if preds is not None and iso_weeks is not None:
            s = check_seasonal_peak(preds, iso_weeks)
            checks[s.checker] = s.details
            violations.extend(s.details.get("violations", []))

        # 5) Known outbreak alignment
        if preds is not None and iso_yw is not None:
            a = check_outbreak_alignment(preds, iso_yw)
            checks[a.checker] = a.details
            violations.extend(a.details.get("violations", []))

        status = "fail" if violations else "ok"
        report[name] = {
            "status": status,
            "checks": checks,
            "violations": violations,
            "exclude_from_ensemble": bool(strict_exclude and status == "fail"),
        }
    return report
