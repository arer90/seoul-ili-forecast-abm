"""Experiment 2 — season-blocked multi-season back-test for FusedEpi.

Addresses the single-season limitation of the frozen-split evaluation
(scripts/ablation_fusedepi.py) by running a SEASON-BLOCKED rolling-origin
back-test over the full 2019-2025 Seoul weekly ILI series.

Protocol (leak-free, matched to ablation_fusedepi.py):
    * Data:      run_data(PipelineConfig()) — the SAME frozen pipeline.  The
                 returned in-sample matrix (337 weeks) and the reserved real
                 slab (17 weeks) share identical columns, so they are stitched
                 back into the full chronological 2019-09 → 2026-06 series.
    * Features:  BASIC eval subset (lag + seasonal, 13 cols) via
                 _resolve_eval_features — identical to ablation.
    * Split:     season-blocked EXPANDING-window rolling-origin.  A season
                 labelled "YYYY/YYYY+1" spans weeks with date in
                 [Aug-1 YYYY, Aug-1 YYYY+1) (the summer trough cleanly
                 separates consecutive influenza epidemics).  For each testable
                 season we TRAIN on every week strictly before the season start
                 and TEST on the season's weeks — nothing from the test window
                 (or any later week) ever touches the fit.
    * Rolling:   one-step-ahead.  FusedEpi/TiRex feed y_observed back as they
                 roll; NegBinGLM/ARIMA use observed lag features / append(refit
                 =False) — every forecast uses only past-observed values.
    * Scoring:   per-season WIS via simulation.analytics.adaptive_conformal
                 .wis_from_bounds (the R10 WIS helper), plus MAE / RMSE.

Model panel (all reconstructed cheaply, no live pipeline code touched):
    * FusedEpi      — the champion (imported live class).
    * TiRex         — FusedEpi base with the TabPFN residual disabled
                      (NoResidualFusedEpi from ablation_fusedepi.py).  Isolates
                      exactly the value the residual layer adds.
    * NegBinGLM     — NegBinGLMForecaster (V7, statsmodels NB-GLM), native NB
                      parametric-bootstrap intervals.
    * ARIMA         — statsmodels SARIMAX (order picked by AIC on the training
                      pool, matching the pipeline ARIMA baseline), rolling
                      one-step with Gaussian predictive intervals.

KEY QUESTION answered: is FusedEpi's skill CONSISTENT across seasons or a
single-season artifact?

Side effects:
    Writes one JSON file to the scratchpad elevate/ directory.
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
from simulation.models.negbin_glm import NegBinGLMForecaster
from simulation.models.ts_models import ARIMAForecaster
from simulation.pipeline.config import PipelineConfig
from simulation.pipeline.data import run_data
from simulation.pipeline.runner import _resolve_eval_features

# Reuse the frozen-split ablation helpers so the WIS machinery is byte-identical.
from scripts.ablation_fusedepi import (
    NoResidualFusedEpi,
    _quantile_array,
    score_native_wis,
    seed_all,
)

LOG = logging.getLogger("fusedepi_multiseason")
OUT_JSON = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/elevate/multiseason.json"
)

# Minimum training-pool length before a season may be used as a test block.
# FusedEpi.meta.min_data = 70 and min_ctx = 52; we require a comfortable margin
# so TiRex rolling + a real calibration tail exist.
MIN_TRAIN_WEEKS = 90


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — full chronological series in BASIC feature space
# ─────────────────────────────────────────────────────────────────────────────
def load_full_series():
    """Stitch in-sample + real slab into the full 2019-2025 weekly ILI series.

    Returns:
        X_full: (N, 13) BASIC eval features, chronological.
        y_full: (N,) ILI rate.
        dates:  (N,) datetime64[D], chronological.
        meta:   dict with column names and provenance.
    """
    data = run_data(PipelineConfig())
    X_in = np.asarray(data["X_all"], dtype=float)
    y_in = np.asarray(data["y_all"], dtype=float).ravel()
    feature_cols = list(data["feature_cols"])
    dates_in = data.get("dates")
    real_X = data.get("real_X")
    real_y = data.get("real_y")
    real_dates = data.get("real_dates")

    def _as_days(d):
        d = np.asarray(d)
        return d if d.dtype.kind == "M" else np.asarray(d, dtype="datetime64[D]")

    if dates_in is None:
        raise RuntimeError("run_data returned no dates — cannot season-block")
    dates_in = _as_days(dates_in)

    if real_X is not None and real_y is not None and real_dates is not None:
        real_X = np.asarray(real_X, dtype=float)
        real_y = np.asarray(real_y, dtype=float).ravel()
        real_dates = _as_days(real_dates)
        if real_X.shape[1] != X_in.shape[1]:
            raise RuntimeError(
                f"real_X cols {real_X.shape[1]} != in-sample {X_in.shape[1]}"
            )
        X_all = np.vstack([X_in, real_X])
        y_all = np.concatenate([y_in, real_y])
        dates = np.concatenate([dates_in, real_dates])
    else:
        X_all, y_all, dates = X_in, y_in, dates_in

    order = np.argsort(dates, kind="stable")
    X_all = X_all[order]
    y_all = y_all[order]
    dates = dates[order]

    X_eval, eval_cols, _basic_idx = _resolve_eval_features(
        X_all, feature_cols, eval_basic=True
    )
    meta = {
        "n_full": int(len(y_all)),
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "n_features_basic": int(X_eval.shape[1]),
        "feature_cols": eval_cols,
        "stitched_real_slab": real_X is not None,
    }
    return np.asarray(X_eval, dtype=float), y_all, dates, meta


def season_label_for(dates: np.ndarray) -> np.ndarray:
    """Assign each week to an influenza-season year (Aug-1 boundary)."""
    dt = dates.astype("datetime64[D]")
    years = dt.astype("datetime64[Y]").astype(int) + 1970
    months = (dt.astype("datetime64[M]").astype(int) % 12) + 1
    return np.where(months >= 8, years, years - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_pred - y_true
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def score_fused_family(model, X_test, y_test) -> dict:
    """WIS (native NegBin/conformal quantiles) + MAE/RMSE for FusedEpi/TiRex."""
    wis = score_native_wis(model, X_test, y_test)
    point = np.asarray(model.predict(X_test, y_observed=y_test), dtype=float)
    mae, rmse = mae_rmse(y_test, point)
    return {"wis": float(wis), "mae": mae, "rmse": rmse}


def score_negbinglm(model, X_test, y_test) -> dict:
    """WIS from native NB parametric-bootstrap intervals + MAE/RMSE."""
    point = np.asarray(model.predict(X_test), dtype=float)
    bounds = {}
    for alpha in FLUSIGHT_ALPHAS:
        lo, hi = model.predict_interval(X_test, alpha=float(alpha))
        bounds[float(alpha)] = (
            np.asarray(lo, dtype=float),
            np.asarray(hi, dtype=float),
        )
    wis_arr = wis_from_bounds(y_test, bounds, FLUSIGHT_ALPHAS, median=point)
    wis_arr = np.asarray(wis_arr, dtype=float)
    if not np.isfinite(wis_arr).any():
        raise RuntimeError("NegBinGLM native WIS produced no finite values")
    mae, rmse = mae_rmse(y_test, point)
    return {"wis": float(np.nanmean(wis_arr)), "mae": mae, "rmse": rmse}


def score_arima_rolling(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Rolling one-step ARIMA with Gaussian predictive intervals → WIS/MAE/RMSE.

    Fits SARIMAX order by AIC on the training pool (via ARIMAForecaster), then
    rolls one step at a time: forecast(1) → append the observed value with
    refit=False.  Quantiles from the state-space predictive mean/variance.
    """
    from scipy.stats import norm

    arima = ARIMAForecaster()
    arima.fit_series(np.asarray(y_train, dtype=float))
    ext = arima._fit_result

    n = len(y_test)
    means = np.empty(n, dtype=float)
    ses = np.empty(n, dtype=float)
    for i in range(n):
        fc = ext.get_forecast(1)
        m = float(np.asarray(fc.predicted_mean).ravel()[0])
        try:
            s = float(np.asarray(fc.se_mean).ravel()[0])
        except Exception:
            s = float(np.asarray(fc.var_pred_mean).ravel()[0]) ** 0.5
        means[i] = m
        ses[i] = s if np.isfinite(s) and s > 0 else 1e-6
        ext = ext.append([float(y_test[i])], refit=False)

    point = np.clip(means, 0.0, None)
    bounds = {}
    for alpha in FLUSIGHT_ALPHAS:
        z = float(norm.ppf(1.0 - float(alpha) / 2.0))
        lo = np.clip(means - z * ses, 0.0, None)
        hi = np.clip(means + z * ses, 0.0, None)
        bounds[float(alpha)] = (lo, hi)
    wis_arr = wis_from_bounds(y_test, bounds, FLUSIGHT_ALPHAS, median=point)
    wis_arr = np.asarray(wis_arr, dtype=float)
    if not np.isfinite(wis_arr).any():
        raise RuntimeError("ARIMA WIS produced no finite values")
    mae, rmse = mae_rmse(y_test, point)
    order = tuple(int(v) for v in arima._fit_result.specification["order"])
    return {
        "wis": float(np.nanmean(wis_arr)),
        "mae": mae,
        "rmse": rmse,
        "order": order,
    }


def cleanup(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Main season-blocked back-test
# ─────────────────────────────────────────────────────────────────────────────
def run_season(name: str, X_train, y_train, X_test, y_test) -> dict:
    """Fit + score every model on one season block; failures recorded per model."""
    models: dict[str, dict] = {}

    def _try(model_name, fn):
        seed_all(42)
        t0 = time.time()
        try:
            res = fn()
            res["elapsed_sec"] = round(time.time() - t0, 2)
            LOG.info(
                "[%s/%s] WIS=%.4f MAE=%.4f RMSE=%.4f (%.1fs)",
                name, model_name, res["wis"], res["mae"], res["rmse"],
                res["elapsed_sec"],
            )
            models[model_name] = res
        except Exception as e:  # noqa: BLE001 — record and continue
            LOG.exception("[%s/%s] failed", name, model_name)
            models[model_name] = {"wis": None, "mae": None, "rmse": None, "error": str(e)}

    def _fused():
        m = FusedEpiForecaster()
        m.fit(X_train, y_train)
        out = score_fused_family(m, X_test, y_test)
        out["alpha"] = float(getattr(m, "_alpha", float("nan")))
        cleanup(m)
        return out

    def _tirex():
        m = NoResidualFusedEpi()
        m.fit(X_train, y_train)
        out = score_fused_family(m, X_test, y_test)
        cleanup(m)
        return out

    def _negbin():
        m = NegBinGLMForecaster()
        m.fit(X_train, y_train)
        out = score_negbinglm(m, X_test, y_test)
        out["used_fallback"] = bool(getattr(m, "_fallback", False))
        cleanup(m)
        return out

    def _arima():
        return score_arima_rolling(y_train, y_test)

    _try("FusedEpi", _fused)
    _try("TiRex", _tirex)
    _try("NegBinGLM", _negbin)
    _try("ARIMA", _arima)
    return models


def build_summary(seasons: list[dict]) -> dict:
    """Per-season relative WIS + consistency verdict for FusedEpi."""
    baselines = ["TiRex", "NegBinGLM", "ARIMA"]
    per_season = []
    fused_wis_list = []
    win_flags = []
    for s in seasons:
        m = s["models"]
        f = m.get("FusedEpi", {}).get("wis")
        row = {"season": s["season"], "fusedepi_wis": f}
        if not isinstance(f, (int, float)) or not np.isfinite(f):
            per_season.append(row)
            continue
        fused_wis_list.append(f)
        base_vals = {}
        for b in baselines:
            bw = m.get(b, {}).get("wis")
            if isinstance(bw, (int, float)) and np.isfinite(bw):
                base_vals[b] = bw
                row[f"relwis_vs_{b}"] = round(f / bw, 4) if bw > 0 else None
                row[f"delta_vs_{b}"] = round(f - bw, 4)
        if base_vals:
            best_base_name = min(base_vals, key=base_vals.get)
            best_base = base_vals[best_base_name]
            row["best_baseline"] = best_base_name
            row["best_baseline_wis"] = round(best_base, 4)
            row["relwis_vs_best_baseline"] = round(f / best_base, 4) if best_base > 0 else None
            won = f <= best_base
            row["fusedepi_beats_best_baseline"] = bool(won)
            win_flags.append(won)
        per_season.append(row)

    n_win = int(sum(1 for w in win_flags if w))
    n_eval = len(win_flags)
    rel_best = [
        r["relwis_vs_best_baseline"]
        for r in per_season
        if r.get("relwis_vs_best_baseline") is not None
    ]
    summary = {
        "seasons_evaluated": n_eval,
        "fusedepi_beats_best_baseline_count": n_win,
        "fusedepi_beats_best_baseline_fraction": (
            round(n_win / n_eval, 3) if n_eval else None
        ),
        "relwis_vs_best_baseline_min": round(min(rel_best), 4) if rel_best else None,
        "relwis_vs_best_baseline_max": round(max(rel_best), 4) if rel_best else None,
        "relwis_vs_best_baseline_mean": (
            round(float(np.mean(rel_best)), 4) if rel_best else None
        ),
        "fusedepi_wis_min": round(min(fused_wis_list), 4) if fused_wis_list else None,
        "fusedepi_wis_max": round(max(fused_wis_list), 4) if fused_wis_list else None,
        "per_season": per_season,
    }
    return summary


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    seed_all(42)
    X_full, y_full, dates, data_meta = load_full_series()
    labels = season_label_for(dates)
    uniq = sorted(set(int(v) for v in labels))
    LOG.info("full series: n=%d %s → %s seasons=%s",
             len(y_full), data_meta["date_min"], data_meta["date_max"], uniq)

    seasons_out: list[dict] = []
    for sy in uniq:
        idx = np.where(labels == sy)[0]
        s_start, s_end = int(idx.min()), int(idx.max()) + 1
        n_train = s_start
        if n_train < MIN_TRAIN_WEEKS:
            LOG.info("skip season %d/%d: train_pool=%d < %d",
                     sy, sy + 1, n_train, MIN_TRAIN_WEEKS)
            continue
        X_train, y_train = X_full[:s_start], y_full[:s_start]
        X_test, y_test = X_full[s_start:s_end], y_full[s_start:s_end]
        label = f"{sy}/{sy + 1}"
        LOG.info(
            "=== season %s: train[0:%d] test[%d:%d] (n_test=%d, peak=%.2f) ===",
            label, s_start, s_start, s_end, len(y_test), float(np.max(y_test)),
        )
        models = run_season(label, X_train, y_train, X_test, y_test)
        seasons_out.append({
            "season": label,
            "test_start_date": str(dates[s_start]),
            "test_end_date": str(dates[s_end - 1]),
            "n_test": int(len(y_test)),
            "train_pool_weeks": int(n_train),
            "peak_obs": round(float(np.max(y_test)), 3),
            "mean_obs": round(float(np.mean(y_test)), 3),
            "models": models,
        })

    summary = build_summary(seasons_out)

    # Verdict
    n_eval = summary["seasons_evaluated"]
    n_win = summary["fusedepi_beats_best_baseline_count"]
    rel_mean = summary["relwis_vs_best_baseline_mean"]
    rel_max = summary["relwis_vs_best_baseline_max"]
    if n_eval >= 3:
        consistency = (
            f"FusedEpi beat the best per-season baseline in {n_win}/{n_eval} seasons "
            f"(mean rel-WIS vs best baseline = {rel_mean}, worst season = {rel_max}). "
        )
        if rel_max is not None and rel_max <= 1.05 and n_win >= max(1, n_eval - 1):
            consistency += (
                "Skill is CONSISTENT across seasons — not a single-season artifact "
                "(never materially worse than the strongest baseline in any season)."
            )
        elif n_win >= 1 and rel_mean is not None and rel_mean <= 1.0:
            consistency += (
                "Skill is BROADLY consistent — on average at or below the best "
                "baseline, though the margin varies by season."
            )
        else:
            consistency += (
                "Skill is NOT uniformly dominant — advantage is season-dependent; "
                "treat headline single-season numbers with caution."
            )
    else:
        consistency = (
            f"Only {n_eval} season block(s) had enough training history — "
            "insufficient for a strong multi-season consistency claim."
        )

    out = {
        "protocol": {
            "data": "run_data(PipelineConfig()); in-sample + real slab stitched",
            "features": "BASIC eval subset via _resolve_eval_features (13 cols)",
            "split": "season-blocked expanding-window rolling-origin; season = "
                     "[Aug-1 Y, Aug-1 Y+1); train = weeks before season start",
            "rolling": "one-step-ahead; y_observed fed back (FusedEpi/TiRex), "
                       "observed lag features (NegBinGLM), append(refit=False) (ARIMA)",
            "wis": "simulation.analytics.adaptive_conformal.wis_from_bounds "
                   "over FLUSIGHT_ALPHAS (K=11)",
            "min_train_weeks": MIN_TRAIN_WEEKS,
            "models": {
                "FusedEpi": "live champion class",
                "TiRex": "NoResidualFusedEpi (FusedEpi base, residual disabled)",
                "NegBinGLM": "NegBinGLMForecaster V7 (statsmodels NB-GLM)",
                "ARIMA": "SARIMAX AIC-selected order, rolling 1-step Gaussian PI",
            },
        },
        "data_meta": data_meta,
        "seasons": seasons_out,
        "summary": summary,
        "verdict": consistency,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", OUT_JSON)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
