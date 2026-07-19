"""Regime-stratified component ablation for the FusedEpi champion forecaster.

Companion to ``scripts/ablation_fusedepi.py`` (mean test-WIS) and
``scripts/ablation_fusedepi_calib.py`` (mean interval calibration). Those two
scripts showed the FusedEpi components do NOT help on the 68-week MEAN WIS or
mean coverage. The mean, however, is dominated by the many mid/quiet weeks; a
component that only matters during the epidemic peak would be invisible in the
mean. This script re-scores the SAME six variants on the SAME frozen 68-week
rolling hold-out, but STRATIFIED by epidemic regime, to test whether
adaptive-conformal / do-no-harm / mechanistic-anchor / residual / dynamic-alpha
lower WIS or improve 95% coverage SPECIFICALLY in the elevated/peak regime.

Two independent, complementary regime definitions are reported (both are
offered by the task):

    (A) KDCA epidemic threshold  — season-aware, published, leak-free.
        ``elevated`` = week ILI > KDCA_THRESHOLD(season); else ``quiet``.
        (8.6 for the 2024-25 season, 9.1 for the 2025-26 season — the exact
        values the live per_model_eval alert metrics use.)
    (B) Test-week tertiles       — ``peak`` = top tertile of the 68 test-week
        ILI values, ``quiet`` = bottom tertile, ``mid`` = middle tertile.
        (This is a stratification of the outcome, not model tuning, so it does
        not leak into any model; it is reported alongside (A) as a robustness
        cross-check that does not depend on the published threshold.)

For every variant × regime it reports:
    - test WIS  (mean of per-week Bracher-2021 WIS within the regime),
    - PICP95    (empirical 95% interval coverage within the regime),
    - PICP50, mean 95%/50% interval width  (context),
    - n weeks in the regime,
    - delta_vs_full  (variant − full) for WIS and PICP95 within the regime.

The variant subclasses, the frozen split loader, the seed helper, the model
cleanup helper, and the quantile-array accessor are IMPORTED from
``scripts.ablation_fusedepi`` — they are NOT re-defined — so this script scores
the identical construct on the identical split/protocol.

Side effects:
    Writes one JSON file under /private/tmp/.../scratchpad/elevate/regime.json.

NO fabrication: every number below is produced by the code at run time. If a
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

from simulation.analytics.adaptive_conformal import wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS, FLUSIGHT_QUANTILES
from simulation.models.fused_epi import FusedEpiForecaster
from simulation.pipeline.config import PipelineConfig
from simulation.pipeline.data import run_data
from simulation.pipeline.real_eval import _kdca_threshold_for, _season_for

# Reuse — do NOT re-define — the frozen split, variant subclasses, and helpers
# from the mean-WIS ablation so every script measures the identical construct.
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

LOG = logging.getLogger("ablation_fusedepi_regime")
OUT_JSON = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/elevate/regime.json"
)


def load_test_dates(meta: dict) -> np.ndarray:
    """Return the datetime64 dates for the frozen test slab, aligned to y_test.

    Uses the SAME ``run_data(PipelineConfig())`` structure that ``load_split``
    consumes; ``dates`` is length ``n`` and aligned to ``y_all`` before feature
    resolution, so ``dates[test_start:test_end]`` matches ``y_test`` week-for-week.

    Args:
        meta: the split-meta dict returned by ``load_split`` (needs n,
            test_start, test_end, n_test).

    Returns:
        (n_test,) ``datetime64`` array of test-week start dates.

    Raises:
        RuntimeError: if ``run_data`` returns no dates or the length/alignment
            does not match the frozen split (fail-loud, never silently proceed).
    """
    data = run_data(PipelineConfig())
    dates = data.get("dates")
    if dates is None:
        raise RuntimeError("run_data returned dates=None — cannot stratify by regime")
    dates = np.asarray(dates)
    if len(dates) != int(meta["n"]):
        raise RuntimeError(
            f"dates length {len(dates)} != split n {meta['n']} — split drift, refusing"
        )
    td = dates[int(meta["test_start"]): int(meta["test_end"])]
    if len(td) != int(meta["n_test"]):
        raise RuntimeError(
            f"test-date slab {len(td)} != n_test {meta['n_test']} — refusing"
        )
    return td


def regime_masks(y_test: np.ndarray, test_dates: np.ndarray) -> dict:
    """Build the boolean regime masks used to stratify the hold-out weeks.

    Args:
        y_test: (n_test,) observed ILI rate on the hold-out.
        test_dates: (n_test,) datetime64 test-week start dates.

    Returns:
        dict with two regime families:
          - "kdca": {"elevated": mask, "quiet": mask, "threshold_per_week": [...],
                     "seasons": [...]}  (elevated = y > KDCA season threshold)
          - "tertile": {"peak": mask, "mid": mask, "quiet": mask,
                        "q33": float, "q67": float}
    """
    y = np.asarray(y_test, dtype=float).ravel()
    thr = np.array([float(_kdca_threshold_for(d)) for d in test_dates], dtype=float)
    seasons = [int(_season_for(d)) if _season_for(d) is not None else -1 for d in test_dates]
    kdca_elev = y > thr
    q33, q67 = (float(v) for v in np.quantile(y, [1.0 / 3.0, 2.0 / 3.0]))
    peak = y >= q67
    quiet_t = y <= q33
    mid = (~peak) & (~quiet_t)
    return {
        "kdca": {
            "elevated": kdca_elev,
            "quiet": ~kdca_elev,
            "threshold_per_week": thr.tolist(),
            "seasons": seasons,
        },
        "tertile": {
            "peak": peak,
            "mid": mid,
            "quiet": quiet_t,
            "q33": q33,
            "q67": q67,
        },
    }


def per_week_scores(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Per-week WIS + 95%/50% coverage/width from model-native quantiles.

    Uses the SAME rolling ``predict_quantiles(..., y_observed=y_test)`` protocol
    and the SAME ``wis_from_bounds`` scorer as ``ablation_fusedepi.score_native_wis``,
    but returns the PER-WEEK arrays (not the mean) so they can be stratified.

    Args:
        model: fitted FusedEpi variant.
        X_test: (n_test, d) BASIC-feature hold-out design.
        y_test: (n_test,) observed hold-out ILI.

    Returns:
        dict of per-week arrays: wis (n,), cov95 (n, bool), cov50 (n, bool),
        width95 (n,), width50 (n,).
    """
    levels = tuple(float(q) for q in FLUSIGHT_QUANTILES)
    q = model.predict_quantiles(X_test, y_observed=y_test, levels=levels)
    y = np.asarray(y_test, dtype=float).ravel()

    bounds = {}
    for alpha in FLUSIGHT_ALPHAS:
        lo = _quantile_array(q, float(alpha) / 2.0)
        hi = _quantile_array(q, 1.0 - float(alpha) / 2.0)
        bounds[float(alpha)] = (lo, hi)
    median = _quantile_array(q, 0.5)
    wis = np.asarray(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=median), dtype=float)
    if not np.isfinite(wis).any():
        raise RuntimeError("native per-week WIS produced no finite values")

    lo95 = _quantile_array(q, 0.025)
    hi95 = _quantile_array(q, 0.975)
    lo50 = _quantile_array(q, 0.25)
    hi50 = _quantile_array(q, 0.75)
    return {
        "wis": wis,
        "cov95": (y >= lo95) & (y <= hi95),
        "cov50": (y >= lo50) & (y <= hi50),
        "width95": np.asarray(hi95 - lo95, dtype=float),
        "width50": np.asarray(hi50 - lo50, dtype=float),
    }


def _regime_summary(scores: dict, mask: np.ndarray) -> dict:
    """WIS/PICP/width summary for one regime slice (masked weeks)."""
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "wis": None, "picp95": None, "picp50": None,
                "mean_width95": None, "mean_width50": None}
    wis = scores["wis"][m]
    return {
        "n": n,
        "wis": float(np.nanmean(wis)),
        "picp95": float(np.mean(scores["cov95"][m])),
        "picp50": float(np.mean(scores["cov50"][m])),
        "mean_width95": float(np.mean(scores["width95"][m])),
        "mean_width50": float(np.mean(scores["width50"][m])),
        "n_covered95": int(np.sum(scores["cov95"][m])),
    }


def run_variant(
    name: str,
    factory: Callable[[], FusedEpiForecaster],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    masks: dict,
) -> dict:
    """Fit one variant and produce its per-regime WIS/coverage on the hold-out."""
    seed_all(42)
    t0 = time.time()
    LOG.info("[%s] fit start", name)
    model = factory()
    model.fit(X_train, y_train)
    scores = per_week_scores(model, X_test, y_test)

    out = {
        "alpha": float(getattr(model, "_alpha", float("nan"))),
        "adaptive_conf": bool(getattr(model, "adaptive_conf", False)),
        "overall": _regime_summary(scores, np.ones(len(y_test), dtype=bool)),
        "kdca": {
            "elevated": _regime_summary(scores, masks["kdca"]["elevated"]),
            "quiet": _regime_summary(scores, masks["kdca"]["quiet"]),
        },
        "tertile": {
            "peak": _regime_summary(scores, masks["tertile"]["peak"]),
            "mid": _regime_summary(scores, masks["tertile"]["mid"]),
            "quiet": _regime_summary(scores, masks["tertile"]["quiet"]),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    LOG.info(
        "[%s] KDCA elev WIS=%.4f PICP95=%.3f (n=%d) | quiet WIS=%.4f PICP95=%.3f (n=%d)",
        name,
        out["kdca"]["elevated"]["wis"], out["kdca"]["elevated"]["picp95"],
        out["kdca"]["elevated"]["n"],
        out["kdca"]["quiet"]["wis"], out["kdca"]["quiet"]["picp95"],
        out["kdca"]["quiet"]["n"],
    )
    cleanup_model(model)
    return out


def _attach_deltas(results: dict) -> None:
    """Add delta_vs_full (WIS and PICP95) per regime for every variant in place."""
    full = results.get("full")
    if not isinstance(full, dict) or "kdca" not in full:
        return
    regime_paths = [
        ("kdca", "elevated"), ("kdca", "quiet"),
        ("tertile", "peak"), ("tertile", "mid"), ("tertile", "quiet"),
    ]
    for name, row in results.items():
        if name == "full" or not isinstance(row, dict) or "kdca" not in row:
            continue
        for fam, reg in regime_paths:
            f = full[fam][reg]
            r = row[fam][reg]
            d = {}
            if isinstance(r.get("wis"), (int, float)) and isinstance(f.get("wis"), (int, float)):
                d["wis"] = float(r["wis"] - f["wis"])
            else:
                d["wis"] = None
            if isinstance(r.get("picp95"), (int, float)) and isinstance(f.get("picp95"), (int, float)):
                d["picp95"] = float(r["picp95"] - f["picp95"])
            else:
                d["picp95"] = None
            r["delta_vs_full"] = d
    # full's own deltas are zero by construction (documented, not fabricated).
    for fam, reg in regime_paths:
        full[fam][reg]["delta_vs_full"] = {"wis": 0.0, "picp95": 0.0}


def _build_verdict(results: dict) -> dict:
    """Answer the key question with the raw per-regime deltas (peak regime).

    A component 'helps' in a regime iff removing it RAISES WIS there, i.e. the
    ``no_X`` variant's ``delta_vs_full.wis`` is POSITIVE in that regime. For
    coverage we report whether removing the component moves PICP95 away from
    (positive contribution) or toward the 0.95 nominal.
    """
    def dw(variant: str, fam: str, reg: str):
        row = results.get(variant, {})
        try:
            return row[fam][reg]["delta_vs_full"]["wis"]
        except Exception:
            return None

    def dp(variant: str, fam: str, reg: str):
        row = results.get(variant, {})
        try:
            return row[fam][reg]["delta_vs_full"]["picp95"]
        except Exception:
            return None

    verdict = {}
    for comp, variant in [
        ("adaptive_conformal", "no_adaptive_conformal"),
        ("do_no_harm", "no_donoharm"),
        ("mechanistic_anchor", "no_anchor"),
        ("residual_correction", "no_residual"),
        ("dynamic_alpha", "static_alpha"),
    ]:
        entry = {}
        for fam, reg in [("kdca", "elevated"), ("tertile", "peak"),
                         ("kdca", "quiet"), ("tertile", "quiet")]:
            key = f"{fam}_{reg}"
            wdelta = dw(variant, fam, reg)
            pdelta = dp(variant, fam, reg)
            entry[key] = {
                "removing_component_delta_wis": wdelta,
                "removing_component_delta_picp95": pdelta,
                "component_lowers_wis_here": (
                    bool(wdelta > 0) if isinstance(wdelta, (int, float)) else None
                ),
            }
        verdict[comp] = entry
    return verdict


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(42)
    X_train, y_train, X_test, y_test, split_meta = load_split()
    test_dates = load_test_dates(split_meta)
    masks = regime_masks(y_test, test_dates)

    LOG.info(
        "split: pool=%d test=%d basicfeat=%d | KDCA elevated=%d quiet=%d | "
        "tertile peak=%d mid=%d quiet=%d (q33=%.3f q67=%.3f)",
        len(y_train), len(y_test), X_train.shape[1],
        int(masks["kdca"]["elevated"].sum()), int(masks["kdca"]["quiet"].sum()),
        int(masks["tertile"]["peak"].sum()), int(masks["tertile"]["mid"].sum()),
        int(masks["tertile"]["quiet"].sum()),
        masks["tertile"]["q33"], masks["tertile"]["q67"],
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
            results[name] = run_variant(
                name, factory, X_train, y_train, X_test, y_test, masks
            )
        except Exception as e:  # noqa: BLE001 — record, never fabricate
            LOG.exception("[%s] failed", name)
            results[name] = {"error": str(e), "n_test": int(len(y_test))}

    _attach_deltas(results)
    verdict = _build_verdict(results)

    y = np.asarray(y_test, dtype=float).ravel()
    results["_meta"] = {
        "protocol": (
            "run_data(PipelineConfig) frozen split; BASIC eval features via "
            "_resolve_eval_features; fit train+val pool; rolling hold-out "
            "predict_quantiles(X_test, y_observed=y_test, levels=FLUSIGHT_QUANTILES); "
            "per-week WIS via simulation.analytics.adaptive_conformal.wis_from_bounds; "
            "per-week 95%/50% coverage from native (0.025,0.975)/(0.25,0.75) bounds. "
            "Variant subclasses + split loader imported from scripts.ablation_fusedepi."
        ),
        "regime_definitions": {
            "kdca_epidemic_threshold": (
                "elevated = week ILI > KDCA season threshold (8.6 for 2024-25, "
                "9.1 for 2025-26) via simulation.pipeline.real_eval._kdca_threshold_for; "
                "published + season-aware + leak-free (identical to live alert metrics)."
            ),
            "tertile": (
                "peak = top tertile of the 68 test-week ILI values, quiet = bottom "
                "tertile, mid = middle tertile. Outcome stratification only; no model "
                "sees the labels, so no leakage into any variant."
            ),
        },
        "regime_counts": {
            "kdca_elevated": int(masks["kdca"]["elevated"].sum()),
            "kdca_quiet": int(masks["kdca"]["quiet"].sum()),
            "tertile_peak": int(masks["tertile"]["peak"].sum()),
            "tertile_mid": int(masks["tertile"]["mid"].sum()),
            "tertile_quiet": int(masks["tertile"]["quiet"].sum()),
            "tertile_q33": masks["tertile"]["q33"],
            "tertile_q67": masks["tertile"]["q67"],
        },
        "y_test_summary": {
            "min": float(np.min(y)), "median": float(np.median(y)),
            "mean": float(np.mean(y)), "max": float(np.max(y)),
        },
        "test_dates_first_last": [str(test_dates[0]), str(test_dates[-1])],
        "split": split_meta,
        "verdict_key": (
            "In verdict, each component maps to its ablation variant. "
            "'removing_component_delta_wis' > 0 means the FULL model (with the "
            "component) has LOWER WIS in that regime, i.e. the component HELPS. "
            "'component_lowers_wis_here' is that boolean. PICP95 delta is the "
            "coverage change from removing the component (nominal 95%)."
        ),
    }
    results["verdict"] = verdict

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", OUT_JSON)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
