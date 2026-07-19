"""Calibration/coverage ablation for the FusedEpi champion forecaster.

Companion to ``scripts/ablation_fusedepi.py`` (which measures only mean
test-WIS).  This script is intentionally isolated from the live pipeline/model
code: it *imports* the frozen data split and the variant subclasses from
``ablation_fusedepi`` rather than re-defining or modifying them, then — for the
SAME 68-week rolling hold-out — measures whether the FusedEpi components
(adaptive conformal, do-no-harm, mechanistic anchor, residual correction,
dynamic alpha) improve prediction-interval CALIBRATION even where they do not
move the mean WIS.

For each variant it computes on model-native ``predict_quantiles`` output:
    (a) 95% and 50% prediction-interval COVERAGE (PICP) + Wilson 95% CI,
    (b) mean interval WIDTH at 95% and 50%,
    (c) a PIT / multi-level calibration summary:
        - PIT via monotone interpolation over the 23 FluSight quantiles
          (mean, std, KS distance from Uniform(0,1), 10-bin histogram),
        - 11-level coverage-calibration curve (FLUSIGHT_ALPHAS) + MACE
          (mean absolute calibration error) and MSCE (mean signed error).

Side effects:
    Writes one JSON file under /private/tmp/.../scratchpad/ablation.

NO fabrication: every number is produced by the code below at run time.  If a
variant raises, its entry records the error string instead of any invented
value.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("OPTUNA_ISOLATE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np

from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS, FLUSIGHT_QUANTILES
from simulation.analytics.metrics import pi_coverage
from simulation.models.fused_epi import FusedEpiForecaster

# Reuse — do NOT re-define — the frozen split loader and the variant subclasses
# from the mean-WIS ablation so both scripts measure the identical construct.
from scripts.ablation_fusedepi import (
    NoAnchorFusedEpi,
    NoDoNoHarmFusedEpi,
    NoResidualFusedEpi,
    StaticAlphaFusedEpi,
    _quantile_array,
    cleanup_model,
    load_split,
    seed_all,
)

LOG = logging.getLogger("ablation_fusedepi_calib")
OUT_JSON = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/ablation/"
    "fusedepi_calib.json"
)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (small-n coverage audit).

    Args:
        k: number of covered observations.
        n: total observations.
        z: standard-normal quantile (default = 1.96 for 95%).

    Returns:
        (lo, hi) bounds of the Wilson interval, clipped to [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def _pit_from_quantiles(y: np.ndarray, q: dict, levels: tuple) -> dict:
    """Non-randomized PIT via monotone interpolation over the quantile grid.

    For each observation, the reported quantile forecasts F^{-1}(level) are made
    monotone (running max) and PIT = F(y) is read off by inverse interpolation.
    Observations below/above the quantile envelope clamp to the extreme levels;
    the fraction that clamp is reported as a tail diagnostic.

    Args:
        y: observed values (n,).
        q: {level: array(n,)} from predict_quantiles.
        levels: ascending tuple of quantile levels present in ``q``.

    Returns:
        dict with pit_mean, pit_std, pit_ks_vs_uniform, pit_hist_10bin,
        n_below_envelope, n_above_envelope, monotone_repairs.
    """
    y = np.asarray(y, dtype=float).ravel()
    lv = np.asarray([float(l) for l in levels], dtype=float)
    order = np.argsort(lv)
    lv = lv[order]
    # (n, L) matrix of quantile values at ascending levels.
    Q = np.column_stack([np.asarray(q[levels[i]], dtype=float) for i in order])
    n, L = Q.shape
    pit = np.empty(n, dtype=float)
    below = above = repairs = 0
    for i in range(n):
        qv = Q[i].copy()
        mono = np.maximum.accumulate(qv)
        if np.any(mono != qv):
            repairs += 1
        if y[i] <= mono[0]:
            below += 1
        if y[i] >= mono[-1]:
            above += 1
        pit[i] = float(np.interp(y[i], mono, lv))
    ps = np.sort(pit)
    cdf_hi = np.arange(1, n + 1) / n
    cdf_lo = np.arange(0, n) / n
    ks = float(max(np.max(cdf_hi - ps), np.max(ps - cdf_lo))) if n else float("nan")
    hist, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
    return {
        "pit_mean": float(np.mean(pit)),
        "pit_std": float(np.std(pit)),
        "pit_ks_vs_uniform": ks,
        "pit_hist_10bin_frac": [float(h / n) for h in hist],
        "n_below_envelope": int(below),
        "n_above_envelope": int(above),
        "monotone_repairs": int(repairs),
    }


def compute_calibration(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Coverage / width / PIT summary for one fitted variant on the hold-out."""
    levels = tuple(float(x) for x in FLUSIGHT_QUANTILES)
    q = model.predict_quantiles(X_test, y_observed=y_test, levels=levels)
    y = np.asarray(y_test, dtype=float).ravel()
    n = int(len(y))

    def _cov(lo_level: float, hi_level: float, nominal: float) -> dict:
        lo = _quantile_array(q, lo_level)
        hi = _quantile_array(q, hi_level)
        c = pi_coverage(y, lo, hi, nominal=nominal)
        k = int(np.sum((y >= lo) & (y <= hi)))
        wlo, whi = wilson_ci(k, n)
        return {
            "nominal": float(nominal),
            "picp": float(c["empirical"]),
            "picp_deviation": float(c["deviation"]),
            "wilson95_lo": wlo,
            "wilson95_hi": whi,
            "mean_width": float(c["mean_width"]),
            "n_covered": k,
            "n": n,
        }

    pi95 = _cov(0.025, 0.975, 0.95)
    pi50 = _cov(0.25, 0.75, 0.50)

    # 11-level coverage-calibration curve (FluSight central PIs).
    curve = []
    abs_errs = []
    signed_errs = []
    for a in FLUSIGHT_ALPHAS:
        nominal = 1.0 - float(a)
        lo = _quantile_array(q, float(a) / 2.0)
        hi = _quantile_array(q, 1.0 - float(a) / 2.0)
        c = pi_coverage(y, lo, hi, nominal=nominal)
        curve.append({
            "nominal": round(nominal, 4),
            "picp": float(c["empirical"]),
            "mean_width": float(c["mean_width"]),
        })
        abs_errs.append(abs(float(c["empirical"]) - nominal))
        signed_errs.append(float(c["empirical"]) - nominal)

    pit = _pit_from_quantiles(y, q, levels)

    return {
        "pi95": pi95,
        "pi50": pi50,
        "calibration_curve_11level": curve,
        "mace_11level": float(np.mean(abs_errs)),   # mean |empirical - nominal|
        "msce_11level": float(np.mean(signed_errs)),  # mean signed (>0 = over-cover)
        "pit": pit,
        "alpha_blend": float(getattr(model, "_alpha", float("nan"))),
        "use_asym_conformal": bool(getattr(model, "_use_asym", False)),
        "adaptive_conf": bool(getattr(model, "adaptive_conf", False)),
    }


def run_variant(
    name: str,
    factory: Callable[[], FusedEpiForecaster],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit one variant and score its interval calibration on the hold-out."""
    seed_all(42)
    t0 = time.time()
    LOG.info("[%s] fit start", name)
    model = factory()
    model.fit(X_train, y_train)
    out = compute_calibration(model, X_test, y_test)
    out["elapsed_sec"] = round(time.time() - t0, 3)
    LOG.info(
        "[%s] PICP95=%.3f (w=%.3f) PICP50=%.3f (w=%.3f) MACE=%.3f",
        name, out["pi95"]["picp"], out["pi95"]["mean_width"],
        out["pi50"]["picp"], out["pi50"]["mean_width"], out["mace_11level"],
    )
    cleanup_model(model)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(42)
    X_train, y_train, X_test, y_test, split_meta = load_split()
    LOG.info(
        "loaded split: train_pool=%s test=%s basic_features=%s y_test[min=%.3f max=%.3f mean=%.3f]",
        len(y_train), len(y_test), X_train.shape[1],
        float(np.min(y_test)), float(np.max(y_test)), float(np.mean(y_test)),
    )

    variants: list[tuple[str, Callable[[], FusedEpiForecaster]]] = [
        ("full", lambda: FusedEpiForecaster()),
        ("no_anchor", lambda: NoAnchorFusedEpi()),
        ("no_residual", lambda: NoResidualFusedEpi()),
        ("static_alpha", lambda: StaticAlphaFusedEpi()),
        ("no_donoharm", lambda: NoDoNoHarmFusedEpi()),
        ("no_adaptive_conformal", lambda: FusedEpiForecaster(adaptive_conf=False)),
    ]

    results: dict[str, dict] = {}
    for name, factory in variants:
        try:
            results[name] = run_variant(name, factory, X_train, y_train, X_test, y_test)
        except Exception as e:  # noqa: BLE001 — record, never fabricate
            LOG.exception("[%s] failed", name)
            results[name] = {"error": str(e), "n_test": int(len(y_test))}

    results["_meta"] = {
        "protocol": (
            "run_data(PipelineConfig) frozen split; BASIC eval features via "
            "_resolve_eval_features; fit train+val pool; rolling hold-out "
            "predict_quantiles(X_test, y_observed=y_test, levels=FLUSIGHT_QUANTILES); "
            "coverage via simulation.analytics.metrics.pi_coverage; PIT via monotone "
            "interpolation over the 23 FluSight quantiles."
        ),
        "y_test_summary": {
            "min": float(np.min(y_test)),
            "max": float(np.max(y_test)),
            "mean": float(np.mean(y_test)),
            "integer_valued": bool(np.allclose(y_test, np.round(y_test))),
        },
        "caveat_conformal_levels": (
            "predict_quantiles conformal-adjusts only the (0.025,0.975) and "
            "(0.25,0.75) interval pairs (the keys seeded in _fit_conformal). The "
            "other FluSight quantile levels are raw NegBin ppf, so the 23-level PIT "
            "and 11-level curve mix conformal-adjusted and unadjusted quantiles; the "
            "95%/50% PICP+width are fully model-native and are the primary readout."
        ),
        "split": split_meta,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", OUT_JSON)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
