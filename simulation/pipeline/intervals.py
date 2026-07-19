"""
R7: Bootstrap PI + Conformal PI + Ablation Study
======================================================
예측구간 (PI) 추정 및 피처 그룹별 기여도 분석.

Phase C1 add-on: after computing the split-conformal quantile on
OOF residuals, evaluate empirical coverage by COVID regime
(pre/during/post/global) using `analytics.diagnostics.coverage_gap_by_regime`.
The regime table is appended to each model entry as `regime_coverage`.

Phase E1 guard (S0-2 re-classification): conformal PI residuals
must be computed in the same output space as `y_all`. If a downstream
model returns predictions in a transformed space (e.g. `log1p(y)`), the
caller MUST apply the inverse before feeding into run_intervals. A
heuristic sanity check (`_assert_same_residual_space`) compares the
location/scale of `y_all` against each predictor and warns/raises when
the distributions look wildly inconsistent.
"""
import logging
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
import time
import numpy as np
from typing import Dict, Literal, Optional, Sequence

log = logging.getLogger(__name__)


# Numba JIT path for bootstrap hot loop. Falls back to pure-numpy if unavailable.
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


@njit(cache=True, fastmath=True)
def _bootstrap_quantiles_jit(
    residuals: np.ndarray, n_boot: int, q_lo: float, q_hi: float, seed: int
) -> tuple:
    """Return (mean_lower_bound, mean_upper_bound) over `n_boot` resamples.

    `residuals` is a 1-D float64 array. `q_lo` / `q_hi` are percentiles in [0, 100].
    Uses numba-native RNG for determinism; `seed` controls the stream.
    """
    n = residuals.shape[0]
    np.random.seed(seed)  # numba-native seed (per-thread by default)
    lo_sum = 0.0
    hi_sum = 0.0
    sample = np.empty(n, dtype=np.float64)
    for _ in range(n_boot):
        for k in range(n):
            sample[k] = residuals[np.random.randint(0, n)]
        lo_sum += np.percentile(sample, q_lo)
        hi_sum += np.percentile(sample, q_hi)
    return lo_sum / n_boot, hi_sum / n_boot


def _bootstrap_pi(y_true, y_pred, n_boot=2000, alpha=0.05):
    """Bootstrap 예측구간 (PI).

    Numba-JIT 경로 (`_bootstrap_quantiles_jit`) 사용 가능 시 10-30× 가속.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residuals = y_true - y_pred
    n = residuals.size
    if n == 0:
        return {"width": 0.0, "coverage": 0.0}

    q_lo = alpha / 2.0 * 100.0
    q_hi = (1.0 - alpha / 2.0) * 100.0

    if _HAS_NUMBA:
        mean_lower, mean_upper = _bootstrap_quantiles_jit(
            np.ascontiguousarray(residuals), int(n_boot), q_lo, q_hi, seed=42
        )
    else:
        rng = np.random.RandomState(42)
        lower_bounds = np.empty(n_boot)
        upper_bounds = np.empty(n_boot)
        for i in range(n_boot):
            boot_idx = rng.randint(0, n, size=n)
            boot_res = residuals[boot_idx]
            lower_bounds[i] = np.percentile(boot_res, q_lo)
            upper_bounds[i] = np.percentile(boot_res, q_hi)
        mean_lower = float(np.mean(lower_bounds))
        mean_upper = float(np.mean(upper_bounds))

    lower = y_pred + mean_lower
    upper = y_pred + mean_upper
    width = float(np.mean(upper - lower))
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return {"width": round(width, 2), "coverage": round(coverage, 4)}


def _conformal_pi(y_cal, pred_cal, y_test, pred_test, alpha=0.05):
    """Split Conformal Prediction Interval.

    Fix (S1-3, 2026-04-14): use the proper ceil((n+1)(1-alpha))-th order
    statistic instead of the biased `np.percentile(..., (1-alpha)*100*(1+1/n))`.
    The percentile version under-covers for small n because linear
    interpolation skips the correct order statistic. Ref: Lei et al. (2018)
    "Distribution-Free Predictive Inference", Thm 2.2 -- finite-sample
    coverage >= 1-alpha is guaranteed only with the ceiling-based quantile.
    """
    scores = np.abs(y_cal - pred_cal)
    n = len(scores)
    if n == 0:
        return {"width": 0.0, "coverage": 0.0, "quantile": 0.0}
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)             # clip to [1, n]
    q = float(np.sort(scores)[k - 1]) # k-th smallest (1-indexed)
    lower = pred_test - q
    upper = pred_test + q
    width = float(2 * q)
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper))) if len(y_test) > 0 else 0.0
    return {"width": round(width, 2), "coverage": round(coverage, 4), "quantile": round(q, 4)}


def _cv_plus_pi_from_folds(
    y_all: np.ndarray,
    oof_pred: np.ndarray,
    fold_holdout_preds: np.ndarray,
    fold_val_indices: Sequence,
    y_test: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """F3: CV+ interval (Barber+2021) wired from R4 WF-CV folds.

 Aggregates per-fold holdout predictions and per-fold OOF residuals
 via `conformal.cv_plus_interval` to produce order-statistic based
 intervals that ARE valid for dependent fold predictors.

 Args:
 y_all: ground truth across the full series.
 oof_pred: OOF predictions (NaN outside fold val windows).
 fold_holdout_preds: shape (K, H) — row k is fold-k model's
 prediction on the H-length holdout slab.
 fold_val_indices: list of (val_start, val_end) per completed fold,
 same ordering as rows in fold_holdout_preds.
 y_test: holdout-slab truth (length H).
 alpha: miscoverage level (default 0.05 → 95% PI).
 """
    from simulation.models.conformal import cv_plus_interval

    if fold_holdout_preds is None or len(fold_val_indices) == 0:
        return {}
    if fold_holdout_preds.shape[0] != len(fold_val_indices):
        return {"error": "fold_holdout_preds rows ≠ len(fold_val_indices)"}

    fold_preds_by_fold: Dict[int, np.ndarray] = {}
    fold_indices_map: Dict[int, list] = {}
    residuals_buf: Dict[int, float] = {}

    for k, (vs, ve) in enumerate(fold_val_indices):
        hp_k = fold_holdout_preds[k]
        cal_idx = [i for i in range(int(vs), int(ve))
                   if i < len(oof_pred) and np.isfinite(oof_pred[i])]
        if not cal_idx:
            continue
        fold_preds_by_fold[k] = np.asarray(hp_k, dtype=float)
        fold_indices_map[k] = cal_idx
        for i in cal_idx:
            residuals_buf[i] = abs(float(y_all[i]) - float(oof_pred[i]))

    if not fold_preds_by_fold:
        return {}

    n_series = len(oof_pred)
    residuals_cal = np.zeros(n_series, dtype=float)
    for i, r in residuals_buf.items():
        residuals_cal[i] = r

    try:
        lower, upper = cv_plus_interval(
            fold_preds_by_fold, fold_indices_map, residuals_cal, alpha=alpha
        )
    except Exception as e:
        return {"error": f"cv_plus_interval failed: {e}"}

    yt = np.asarray(y_test, dtype=float)
    if lower.shape[0] != yt.shape[0]:
        return {"error": f"cv_plus length {lower.shape[0]} ≠ y_test {yt.shape[0]}"}

    width = float(np.mean(upper - lower))
    coverage = float(np.mean((yt >= lower) & (yt <= upper))) if yt.size else 0.0
    return {
        "width": round(width, 2),
        "coverage": round(coverage, 4),
        "n_folds": len(fold_preds_by_fold),
        "n_cal": int(sum(len(v) for v in fold_indices_map.values())),
    }


def _assert_same_residual_space(
    y_all: np.ndarray,
    predictions: Dict[str, np.ndarray],
    *,
    mode: Literal["raise", "warn", "off"] = "warn",
    scale_tol: float = 8.0,
    loc_tol_iqr: float = 20.0,
) -> Dict[str, Dict[str, float]]:
    """S0-2 guard: verify predictions share y_all's output space.

    If some downstream module stored a log1p-transformed forecast in
    `oof_predictions` but y_all is in the raw ILI-rate space, the
    absolute-residual conformal quantile will be meaningless — huge
    in one direction, trivially tight in the other. We catch this
    heuristically by comparing each predictor's IQR-scale and median
    against y_all.

    Args:
        mode:
            - ``"raise"``  : ValueError on mismatch (production wiring
                             should use this once call sites are clean)
            - ``"warn"``   : emit log.warning and continue (default — the
                             pipeline currently uses only raw-space models,
                             so mismatch indicates an upstream bug)
            - ``"off"``    : no-op, return diagnostic dict only
        scale_tol: ratio threshold on IQR (predictor IQR / y_all IQR
                   outside [1/scale_tol, scale_tol] → mismatch).
        loc_tol_iqr: median offset threshold in units of y_all IQR.

    Returns:
        Per-model diagnostic dict with keys y_iqr, p_iqr, ratio, offset_iqr,
        flag ∈ {"ok","scale","location","empty"}.
    """
    y = np.asarray(y_all, dtype=float)
    y_valid = y[np.isfinite(y)]
    if y_valid.size < 20:
        return {}
    y_med = float(np.median(y_valid))
    y_q1, y_q3 = np.percentile(y_valid, [25, 75])
    y_iqr = max(float(y_q3 - y_q1), 1e-12)

    diagnostics: Dict[str, Dict[str, float]] = {}
    mismatches: list[str] = []
    for name, pred in predictions.items():
        p = np.asarray(pred, dtype=float)
        p_valid = p[np.isfinite(p)]
        if p_valid.size < 20:
            diagnostics[name] = {"flag": "empty"}
            continue
        p_med = float(np.median(p_valid))
        p_q1, p_q3 = np.percentile(p_valid, [25, 75])
        p_iqr = max(float(p_q3 - p_q1), 1e-12)
        ratio = p_iqr / y_iqr
        offset = abs(p_med - y_med) / y_iqr
        flag = "ok"
        if ratio > scale_tol or ratio < 1.0 / scale_tol:
            flag = "scale"
        elif offset > loc_tol_iqr:
            flag = "location"
        diagnostics[name] = {
            "y_iqr": y_iqr, "p_iqr": p_iqr,
            "ratio": ratio, "offset_iqr": offset, "flag": flag,
        }
        if flag != "ok":
            mismatches.append(f"{name}(flag={flag}, ratio={ratio:.2f}, offset={offset:.2f}·IQR)")

    if mismatches:
        msg = (
            f"[R4 E1] residual-space mismatch: {len(mismatches)} predictor(s) "
            f"appear to be in a different scale than y_all. "
            f"Details: {', '.join(mismatches[:5])}"
            + (" …" if len(mismatches) > 5 else "")
            + ". If a model uses log1p/box-cox internally, invert before passing "
            "to run_intervals."
        )
        if mode == "raise":
            raise ValueError(msg)
        if mode == "warn":
            log.warning(msg)
    return diagnostics


def _regime_coverage_from_q(y_all: np.ndarray,
                            oof_pred: np.ndarray,
                            q: float,
                            nominal: float,
                            dates: Optional[Sequence] = None) -> list:
    """Apply symmetric split-conformal PI [pred-q, pred+q] to the full
    OOF slab and compute pre/during/post-COVID coverage.

    Skipped rows where oof_pred is NaN. Returns an empty list if fewer
    than 20 valid points remain.
    """
    valid = ~np.isnan(oof_pred)
    if int(valid.sum()) < 20:
        return []
    y = y_all[valid]
    p = oof_pred[valid]
    lo = p - q
    hi = p + q
    sub_dates = None
    if dates is not None and len(dates) == len(y_all):
        sub_dates = np.asarray(dates)[valid]
    try:
        from simulation.analytics.diagnostics import coverage_gap_by_regime
        return coverage_gap_by_regime(
            y, lo, hi, nominal=nominal, dates=sub_dates
        )
    except Exception as e:
        log.debug("coverage_gap_by_regime failed: %s", e)
        return []


from simulation.utils.resource_tracker import track_resources


@track_resources("phase10_intervals")
def run_intervals(y_all, oof_predictions, config,
               holdout_predictions=None, holdout_start=None,
               fold_holdout_predictions=None,
               fold_val_indices=None) -> dict:
    """R7: Bootstrap PI + Split-Conformal PI (+ CV+ when provided).

 S0-1 fix: if `holdout_start` and `holdout_predictions` are provided,
 split conformal uses OOF residuals (strictly < holdout_start, never
 touched the holdout) as CALIBRATION and the holdout slab as TEST.
 This is the only configuration with finite-sample coverage guarantees.

 F3: when R4 also emits `fold_holdout_predictions` (per-model
 (K, H) matrices of per-fold predictions on the holdout slab), a CV+
 interval (Barber+2021) is computed alongside split conformal. Added
 under each model's dict as `"cv_plus"`.

 Falls back to the legacy OOF-internal cal/test split only when
 holdout is not supplied, and loudly warns that coverage is optimistic.
 """
    from .utils.logging_util import phase_banner, fmt_time
    phase_banner("R7", "Bootstrap PI + Split-Conformal PI (S0-1 holdout)")

    t0 = time.time()
    n = len(y_all)
    pi_results = {}
    # Pull dates from config if wired (matches phase9_dm_test semantics);
    # otherwise coverage_gap_by_regime falls back to 47/36/17 index split.
    dates = None
    obj = config
    for a in ("data", "dates"):
        obj = getattr(obj, a, None)
        if obj is None:
            break
    if obj is not None:
        dates = obj

    # Phase E1: residual-space sanity check. Pull mode from
    # config.scoring.residual_space_mode if set, else default to "warn".
    mode = "warn"
    sc = getattr(config, "scoring", None)
    if sc is not None:
        mode = getattr(sc, "residual_space_mode", mode)
    residual_space_diag = _assert_same_residual_space(
        y_all, oof_predictions, mode=mode
    )

    has_holdout = (
        holdout_start is not None
        and holdout_start < n
        and holdout_predictions is not None
        and len(holdout_predictions) > 0
    )
    if has_holdout:
        h_y = y_all[holdout_start:]
        log.info(f"  [S0-1] holdout conformal: cal=OOF[<{holdout_start}], "
                 f"test=holdout[{holdout_start}:{n}] ({n - holdout_start} pts)")
    else:
        log.warning("  [S0-1] no holdout supplied -- falling back to legacy "
                    "OOF-internal cal/test split (PI is OPTIMISTIC).")

    for model_name, oof_pred in oof_predictions.items():
        valid = ~np.isnan(oof_pred)
        # Only use OOF indices strictly before holdout for calibration
        if has_holdout:
            cal_mask = valid.copy()
            cal_mask[holdout_start:] = False
        else:
            cal_mask = valid

        if cal_mask.sum() < 20:
            continue

        y_cal = y_all[cal_mask]
        p_cal = oof_pred[cal_mask]

        # Bootstrap PI from calibration residuals (unchanged semantics)
        boot = _bootstrap_pi(y_cal, p_cal, n_boot=2000)

        # Split conformal: cal = OOF (pre-holdout), test = holdout slab
        if has_holdout and model_name in holdout_predictions:
            h_pred = np.asarray(holdout_predictions[model_name], dtype=np.float64)
            if h_pred.shape[0] == h_y.shape[0]:
                conf = _conformal_pi(y_cal, p_cal, h_y, h_pred)
                conf["source"] = "holdout"
            else:
                log.warning(f"  [{model_name}] holdout pred shape mismatch; "
                            f"skipping conformal")
                conf = {"width": 0.0, "coverage": 0.0, "quantile": 0.0,
                        "source": "skipped_shape_mismatch"}
        else:
            # Legacy fallback: cal/test split inside OOF
            mid = len(y_cal) // 2
            conf = _conformal_pi(y_cal[:mid], p_cal[:mid],
                                 y_cal[mid:], p_cal[mid:])
            conf["source"] = "oof_internal_split_OPTIMISTIC"

        # Phase C1: regime-wise coverage using the conformal quantile
        # applied to the full OOF slab. 0.95 nominal (alpha=0.05) matches
        # _conformal_pi's default.
        q = float(conf.get("quantile", 0.0))
        regime_rows = _regime_coverage_from_q(
            y_all, oof_pred, q, nominal=0.95, dates=dates
        ) if q > 0 else []

        # F3: CV+ — run only when R4 supplied fold-level holdout
        # predictions AND we actually have a holdout slab to evaluate on.
        cv_plus_entry: dict = {}
        if (has_holdout
                and fold_holdout_predictions is not None
                and model_name in fold_holdout_predictions
                and fold_val_indices is not None
                and model_name in fold_val_indices):
            cv_plus_entry = _cv_plus_pi_from_folds(
                y_all=y_all,
                oof_pred=oof_pred,
                fold_holdout_preds=fold_holdout_predictions[model_name],
                fold_val_indices=fold_val_indices[model_name],
                y_test=h_y,
                alpha=0.05,
            )

        pi_results[model_name] = {
            "bootstrap": boot,
            "conformal": conf,
            "regime_coverage": regime_rows,
            "cv_plus": cv_plus_entry,   # F3 — {} when unavailable
        }

        # R8.2 (2026-05-26): full 134-key SSOT eval at R7 PI-calibrated state.
        # Trajectory: R4 OOF → R7 PI-calibrated → R8 final.
        # Captures pi95_coverage / pi95_width with the just-calibrated conformal q.
        try:
            from simulation.pipeline.phase_evaluator import evaluate_predictions_full
            if has_holdout and model_name in (holdout_predictions or {}):
                y_eval = h_y
                p_eval = np.asarray(holdout_predictions[model_name], dtype=np.float64)
            else:
                y_eval = y_cal
                p_eval = p_cal
            mask = np.isfinite(y_eval) & np.isfinite(p_eval)
            if mask.sum() >= 5:
                full_r8 = evaluate_predictions_full(
                    y_test=y_eval[mask],
                    y_pred=p_eval[mask],
                    residuals=(y_eval[mask] - p_eval[mask]),
                    y_train_pool=y_cal,
                    threshold=GLOBAL.filter.alert_threshold,
                    phase_id=f"R7_pi_{model_name}",
                    enable_bootstrap_ci=False,
                )
                pi_results[model_name]["phase_eval_r8"] = full_r8
        except Exception as _e:
            pi_results[model_name]["phase_eval_r8_err"] = str(_e)

        log.info(f"  [{model_name:20s}] Bootstrap: width={boot['width']:.1f}, "
                 f"cov={boot['coverage']:.1%} | "
                 f"Conformal({conf.get('source','?')[:8]}): "
                 f"width={conf['width']:.1f}, cov={conf['coverage']:.1%}")
        if cv_plus_entry and "width" in cv_plus_entry:
            log.info(
                f"    CV+: width={cv_plus_entry['width']:.1f}, "
                f"cov={cv_plus_entry['coverage']:.1%} "
                f"(K={cv_plus_entry.get('n_folds','?')}, "
                f"n_cal={cv_plus_entry.get('n_cal','?')})"
            )
        if regime_rows:
            parts = [
                f"{r['regime'][:3]}={r['coverage']:.1%}"
                for r in regime_rows if r["regime"] != "global"
            ]
            log.info(f"    regime cov(0.95): {' '.join(parts)}")

    elapsed = time.time() - t0
    log.info(f"  ✓ R7 완료 [{fmt_time(elapsed)}]")
    return {
        "pi_results": pi_results,
        "holdout_used": bool(has_holdout),
        "holdout_start": holdout_start,
        "residual_space_diag": residual_space_diag,
        "elapsed": elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════
# Tier A — run_intervals_extended
#
# Evaluates a fixed set of 8 method_keys on the same OOF/holdout inputs that
# `run_intervals` consumes, plus optional CQR and native-posterior predictions.
#
# method_keys:
#   1. split_absolute_raw_full          — baseline (= legacy conformal branch)
#   2. split_absolute_log1p_full        — raw quantile of |log1p(y) - log1p(ŷ)|
#   3. split_absolute_log1p_window52    — same but cal trimmed to last 52 weeks
#   4. split_cqr_raw_full                — CQR in raw space (requires cqr_predictions)
#   5. split_cqr_log1p_window52         — PRIMARY target (CQR + log1p + 52w)
#   6. aci_split_cqr_log1p_window52     — ACI-wrapped sliding simulation on CQR
#   7. cvplus_log1p_window52             — jackknife+/CV+ with log1p residuals
#   8. native_posterior                  — NegBin / Bayesian native PI (no conformal)
#
# Inputs (in addition to the run_intervals signature):
#   cqr_predictions: {model: {"q_lo_cal": ..., "q_hi_cal": ...,
#                             "q_lo_test": ..., "q_hi_test": ...}}
#   posterior_predictions: {model: {"lower": ..., "upper": ...}}
#   aci_gamma: ACI learning rate (default 0.05)
# ══════════════════════════════════════════════════════════════════════════

def _method_metrics(
    y_test: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict:
    """PICP, MPIW, Winkler (single-α special case of WIS)."""
    y = np.asarray(y_test, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.size == 0 or y.size != lo.size or y.size != hi.size:
        return {"picp": 0.0, "mpiw": 0.0, "winkler": 0.0, "n_test": int(y.size)}
    below = y < lo
    above = y > hi
    width = hi - lo
    penalty = np.where(below, (2.0 / alpha) * (lo - y), 0.0) + np.where(
        above, (2.0 / alpha) * (y - hi), 0.0
    )
    winkler = float(np.mean(width + penalty))
    return {
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "mpiw": float(np.mean(width)),
        "winkler": winkler,
        "n_test": int(y.size),
    }


def _apply_split_absolute(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    p_test: np.ndarray,
    alpha: float,
    residual_space: str,
    window_weeks: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (lo, hi, q_hat)."""
    from simulation.models.conformal import SplitConformal
    sc = SplitConformal(
        alpha=alpha, residual_space=residual_space,
        window_weeks=window_weeks, method="absolute",
    )
    sc.calibrate(y_cal, p_cal)
    lo, hi = sc.predict_interval(p_test)
    return lo, hi, float(sc.q_hat if sc.q_hat is not None else 0.0)


def _apply_split_cqr(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    alpha: float,
    residual_space: str,
    window_weeks: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    from simulation.models.conformal import CQRSplit
    cq = CQRSplit(
        alpha=alpha, residual_space=residual_space, window_weeks=window_weeks,
    )
    cq.calibrate(y_cal, q_lo_cal, q_hi_cal)
    lo, hi = cq.predict_interval(q_lo_test, q_hi_test)
    return lo, hi, float(cq.q_hat if cq.q_hat is not None else 0.0)


def _apply_aci_cqr_sliding(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    y_test: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    alpha: float,
    residual_space: str,
    window_weeks: Optional[int] = 52,
    gamma: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """ACI-wrapped CQR: sliding simulation across test horizon."""
    from simulation.models.conformal import CQRSplit, AdaptiveConformalTracker
    cq = CQRSplit(
        alpha=alpha, residual_space=residual_space, window_weeks=window_weeks,
    )
    cq.calibrate(y_cal, q_lo_cal, q_hi_cal)
    tracker = AdaptiveConformalTracker(cq, alpha=alpha, gamma=gamma)
    lows, highs = [], []
    for y_true, ql, qh in zip(y_test, q_lo_test, q_hi_test):
        lo, hi = tracker.step(float(y_true), q_lo_test=float(ql), q_hi_test=float(qh))
        lows.append(lo)
        highs.append(hi)
    return np.asarray(lows), np.asarray(highs), list(tracker.history_alpha)


def run_intervals_extended(
    y_all: np.ndarray,
    oof_predictions: dict,
    config,
    *,
    holdout_predictions: Optional[dict] = None,
    holdout_start: Optional[int] = None,
    fold_holdout_predictions: Optional[dict] = None,
    fold_val_indices: Optional[dict] = None,
    cqr_predictions: Optional[dict] = None,
    posterior_predictions: Optional[dict] = None,
    alpha: float = 0.05,
    window_weeks: int = 52,
    aci_gamma: float = 0.05,
    native_space_models: Optional[set] = None,
) -> dict:
    """Tier A PI sweep across 8 method_keys.

 Returns:
 {
 "version": "",
 "alpha": 0.05, "nominal": 0.95, "window_weeks": 52,
 "method_keys": [...],
 "per_method": {method_key: {model: {picp,mpiw,winkler,n_test,q_hat}}},
 "per_model_best": {model: (method_key, metric_dict)},
 "aggregate": {method_key: {picp_median, mpiw_median, winkler_median,
 n_models, n_at_or_above_nominal}},
 "config_echo": {...},
 "holdout_used": bool,
 }
 """
    t0 = time.time()
    native_space_models = set(native_space_models or [])

    has_holdout = (
        holdout_start is not None and holdout_start < len(y_all)
        and holdout_predictions is not None and len(holdout_predictions) > 0
    )
    if not has_holdout:
        log.warning("  [run_intervals_extended] no holdout supplied — all method_keys "
                    "fall back to OOF-internal split (OPTIMISTIC).")
    y_test = y_all[holdout_start:] if has_holdout else None

    method_keys = [
        "split_absolute_raw_full",
        "split_absolute_log1p_full",
        "split_absolute_log1p_window52",
        "split_cqr_raw_full",
        "split_cqr_log1p_window52",
        "aci_split_cqr_log1p_window52",
        "cvplus_log1p_window52",
        "native_posterior",
    ]

    per_method: dict = {k: {} for k in method_keys}

    # Collect cal/test pairs per base model (absolute-residual methods)
    for model_name, oof_pred in oof_predictions.items():
        valid = ~np.isnan(oof_pred)
        if has_holdout:
            cal_mask = valid.copy()
            cal_mask[holdout_start:] = False
        else:
            cal_mask = valid
        if cal_mask.sum() < 20:
            continue
        y_cal = y_all[cal_mask]
        p_cal = oof_pred[cal_mask]

        if has_holdout and model_name in holdout_predictions:
            p_test = np.asarray(holdout_predictions[model_name], dtype=float)
            y_tst = y_test
        else:
            mid = len(y_cal) // 2
            y_tst = y_cal[mid:]
            p_test = p_cal[mid:]
            y_cal = y_cal[:mid]
            p_cal = p_cal[:mid]

        # Skip log1p branches for native-space models so we don't double-transform.
        native = model_name in native_space_models

        # 1. split_absolute_raw_full
        try:
            lo, hi, q_hat = _apply_split_absolute(
                y_cal, p_cal, p_test, alpha=alpha, residual_space="raw",
                window_weeks=None,
            )
            m = _method_metrics(y_tst, lo, hi, alpha)
            m["q_hat"] = q_hat
            per_method["split_absolute_raw_full"][model_name] = m
        except Exception as e:
            log.debug(f"[v2] split_absolute_raw_full {model_name}: {e}")

        # 2. split_absolute_log1p_full (skip native)
        if not native:
            try:
                lo, hi, q_hat = _apply_split_absolute(
                    y_cal, p_cal, p_test, alpha=alpha, residual_space="log1p",
                    window_weeks=None,
                )
                m = _method_metrics(y_tst, lo, hi, alpha)
                m["q_hat"] = q_hat
                per_method["split_absolute_log1p_full"][model_name] = m
            except Exception as e:
                log.debug(f"[v2] split_absolute_log1p_full {model_name}: {e}")

        # 3. split_absolute_log1p_window52 (skip native)
        if not native:
            try:
                lo, hi, q_hat = _apply_split_absolute(
                    y_cal, p_cal, p_test, alpha=alpha, residual_space="log1p",
                    window_weeks=window_weeks,
                )
                m = _method_metrics(y_tst, lo, hi, alpha)
                m["q_hat"] = q_hat
                m["window_weeks"] = window_weeks
                per_method["split_absolute_log1p_window52"][model_name] = m
            except Exception as e:
                log.debug(f"[v2] split_absolute_log1p_window52 {model_name}: {e}")

    # 4–6. CQR methods (only for models with cqr_predictions)
    if cqr_predictions:
        for model_name, qp in cqr_predictions.items():
            try:
                y_cal_q = qp.get("y_cal")
                q_lo_cal = qp.get("q_lo_cal")
                q_hi_cal = qp.get("q_hi_cal")
                q_lo_test = qp.get("q_lo_test")
                q_hi_test = qp.get("q_hi_test")
                y_test_q = qp.get("y_test", y_test)
                if any(v is None for v in [y_cal_q, q_lo_cal, q_hi_cal, q_lo_test, q_hi_test]):
                    log.debug(f"[v2] CQR {model_name}: missing inputs, skipping")
                    continue
                y_cal_q = np.asarray(y_cal_q, dtype=float)
                q_lo_cal = np.asarray(q_lo_cal, dtype=float)
                q_hi_cal = np.asarray(q_hi_cal, dtype=float)
                q_lo_test = np.asarray(q_lo_test, dtype=float)
                q_hi_test = np.asarray(q_hi_test, dtype=float)
                y_test_q = np.asarray(y_test_q, dtype=float)

                # 4. split_cqr_raw_full
                try:
                    lo, hi, q_hat = _apply_split_cqr(
                        y_cal_q, q_lo_cal, q_hi_cal, q_lo_test, q_hi_test,
                        alpha=alpha, residual_space="raw", window_weeks=None,
                    )
                    m = _method_metrics(y_test_q, lo, hi, alpha)
                    m["q_hat"] = q_hat
                    per_method["split_cqr_raw_full"][model_name] = m
                except Exception as e:
                    log.debug(f"[v2] split_cqr_raw_full {model_name}: {e}")

                # 5. split_cqr_log1p_window52 (PRIMARY)
                try:
                    lo, hi, q_hat = _apply_split_cqr(
                        y_cal_q, q_lo_cal, q_hi_cal, q_lo_test, q_hi_test,
                        alpha=alpha, residual_space="log1p", window_weeks=window_weeks,
                    )
                    m = _method_metrics(y_test_q, lo, hi, alpha)
                    m["q_hat"] = q_hat
                    m["window_weeks"] = window_weeks
                    per_method["split_cqr_log1p_window52"][model_name] = m
                except Exception as e:
                    log.debug(f"[v2] split_cqr_log1p_window52 {model_name}: {e}")

                # 6. aci_split_cqr_log1p_window52 (sliding simulation)
                try:
                    lo, hi, alpha_hist = _apply_aci_cqr_sliding(
                        y_cal_q, q_lo_cal, q_hi_cal,
                        y_test_q, q_lo_test, q_hi_test,
                        alpha=alpha, residual_space="log1p",
                        window_weeks=window_weeks, gamma=aci_gamma,
                    )
                    m = _method_metrics(y_test_q, lo, hi, alpha)
                    m["window_weeks"] = window_weeks
                    m["aci_gamma"] = aci_gamma
                    m["alpha_final"] = float(alpha_hist[-1]) if alpha_hist else alpha
                    m["alpha_mean"] = float(np.mean(alpha_hist)) if alpha_hist else alpha
                    per_method["aci_split_cqr_log1p_window52"][model_name] = m
                except Exception as e:
                    log.debug(f"[v2] aci_split_cqr_log1p_window52 {model_name}: {e}")
            except Exception as e:
                log.debug(f"[v2] CQR block failure for {model_name}: {e}")

    # 7. cvplus_log1p_window52
    if fold_holdout_predictions and fold_val_indices and has_holdout:
        from simulation.models.conformal import cv_plus_interval
        for model_name, oof_pred in oof_predictions.items():
            if model_name not in fold_holdout_predictions:
                continue
            if model_name not in fold_val_indices:
                continue
            native = model_name in native_space_models
            if native:
                continue  # caller is already in native space; cv_plus_log1p skipped
            try:
                fold_preds_by_fold: dict = {}
                fold_indices_map: dict = {}
                residuals_buf: dict = {}

                fold_mat = fold_holdout_predictions[model_name]
                for k, (vs, ve) in enumerate(fold_val_indices[model_name]):
                    hp_k = fold_mat[k]
                    cal_idx = [
                        i for i in range(int(vs), int(ve))
                        if i < len(oof_pred) and np.isfinite(oof_pred[i])
                        and (i < holdout_start)
                    ]
                    if not cal_idx:
                        continue
                    fold_preds_by_fold[k] = np.log1p(np.maximum(np.asarray(hp_k, dtype=float), 0.0))
                    fold_indices_map[k] = cal_idx
                    for i in cal_idx:
                        # log1p-space residual
                        y_log = np.log1p(max(float(y_all[i]), 0.0))
                        p_log = np.log1p(max(float(oof_pred[i]), 0.0))
                        residuals_buf[i] = abs(y_log - p_log)
                        # window trim: drop oldest indices if beyond window_weeks
                if window_weeks is not None:
                    recent_cutoff = holdout_start - window_weeks
                    for k in list(fold_indices_map.keys()):
                        kept = [i for i in fold_indices_map[k] if i >= recent_cutoff]
                        if kept:
                            fold_indices_map[k] = kept
                        else:
                            fold_indices_map.pop(k, None)
                            fold_preds_by_fold.pop(k, None)
                if not fold_preds_by_fold:
                    continue

                n_series = len(oof_pred)
                residuals_cal = np.zeros(n_series, dtype=float)
                for i, r in residuals_buf.items():
                    residuals_cal[i] = r

                lo_log, hi_log = cv_plus_interval(
                    fold_preds_by_fold, fold_indices_map, residuals_cal, alpha=alpha
                )
                lo = np.maximum(np.expm1(lo_log), 0.0)
                hi = np.maximum(np.expm1(hi_log), 0.0)
                m = _method_metrics(y_all[holdout_start:], lo, hi, alpha)
                m["window_weeks"] = window_weeks
                m["n_folds"] = len(fold_preds_by_fold)
                per_method["cvplus_log1p_window52"][model_name] = m
            except Exception as e:
                log.debug(f"[v2] cvplus_log1p_window52 {model_name}: {e}")

    # 8. native_posterior (NegBin / Bayesian native PI)
    if posterior_predictions:
        for model_name, pp in posterior_predictions.items():
            try:
                lo = np.asarray(pp.get("lower"), dtype=float)
                hi = np.asarray(pp.get("upper"), dtype=float)
                y_ref = pp.get("y_test", y_test)
                if y_ref is None or lo.size == 0 or hi.size == 0:
                    continue
                m = _method_metrics(np.asarray(y_ref, dtype=float), lo, hi, alpha)
                per_method["native_posterior"][model_name] = m
            except Exception as e:
                log.debug(f"[v2] native_posterior {model_name}: {e}")

    # Aggregate per method
    aggregate: dict = {}
    nominal = 1.0 - alpha
    for k in method_keys:
        entries = per_method[k]
        if not entries:
            aggregate[k] = {"picp_median": float("nan"), "mpiw_median": float("nan"),
                             "winkler_median": float("nan"),
                             "n_models": 0, "n_at_or_above_nominal": 0}
            continue
        picps = np.asarray([v["picp"] for v in entries.values()], dtype=float)
        mpiws = np.asarray([v["mpiw"] for v in entries.values()], dtype=float)
        winks = np.asarray([v["winkler"] for v in entries.values()], dtype=float)
        aggregate[k] = {
            "picp_median": float(np.median(picps)),
            "mpiw_median": float(np.median(mpiws)),
            "winkler_median": float(np.median(winks)),
            "n_models": int(entries.__len__()),
            "n_at_or_above_nominal": int(np.sum(picps >= nominal)),
        }

    # per_model_best: for each model, select method with lowest Winkler at PICP ≥ nominal;
    # if none meet nominal, fall back to lowest Winkler overall.
    all_models = set()
    for k in method_keys:
        all_models.update(per_method[k].keys())
    per_model_best: dict = {}
    for mod in all_models:
        candidates = [
            (k, per_method[k][mod]) for k in method_keys if mod in per_method[k]
        ]
        # First look at PICP ≥ nominal, pick minimum Winkler
        above = [c for c in candidates if c[1]["picp"] >= nominal]
        chosen = min(above, key=lambda c: c[1]["winkler"]) if above else (
            min(candidates, key=lambda c: c[1]["winkler"]) if candidates else None
        )
        if chosen:
            per_model_best[mod] = {"method_key": chosen[0], "metrics": chosen[1]}

    elapsed = time.time() - t0
    log.info(f"  ✓ run_intervals_extended 완료 [{elapsed:.2f}s] — "
             f"{sum(len(v) for v in per_method.values())} method×model entries")
    return {
        "version": "1.0",
        "alpha": alpha,
        "nominal": nominal,
        "window_weeks": window_weeks,
        "aci_gamma": aci_gamma,
        "method_keys": method_keys,
        "per_method": per_method,
        "per_model_best": per_model_best,
        "aggregate": aggregate,
        "config_echo": {
            "has_holdout": bool(has_holdout),
            "holdout_start": holdout_start,
            "n_total": int(len(y_all)),
            "native_space_models": sorted(native_space_models),
        },
        "holdout_used": bool(has_holdout),
        "elapsed": elapsed,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase10_extended = run_intervals_extended
run_phase10 = run_intervals
