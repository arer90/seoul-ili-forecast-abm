#!/usr/bin/env python
"""EXPERIMENT 3 — WIS/coverage comparison: FusedEpi vs conformalized generic fusion.

Background
----------
`scripts/fusedepi_vs_stacking.py` compared FusedEpi against generic fusion of the
same base models on POINT error only (MAE/RMSE) and found FusedEpi does NOT beat a
trivial average / NNLS stack on the point. But FusedEpi's claimed edge is in
CALIBRATED PROBABILISTIC intervals (learned dynamic-alpha + do-no-harm + NegBin
count head + adaptive-conformal), not point accuracy. This script tests that claim
on WIS + PI coverage.

Key question
------------
Does FusedEpi's learned dynamic-alpha + do-no-harm + count head beat a GENERIC
conformalized stacking ensemble of {TiRex, NegBinGLM, TabPFN, ARIMA} on WIS and
coverage — OVERALL and at the epidemic PEAK?

Design (leak-free, same frozen split/protocol as ablation_fusedepi.py)
----------------------------------------------------------------------
1. Base + FusedEpi POINT predictions come from the OFFICIAL frozen artifacts
   (`per_model_optimal/<Model>.json:refit_test_predictions`, identical to
   `results/csv/predictions_<Model>.csv`). These are the R9-optimised, deployed
   base models on the frozen 68-week hold-out test (run_data pool_end=269,
   n_test=68 — verified to match run_data to 1e-15).

2. GENERIC FUSION point predictors on the 68-week test:
     - equal_weight        : mean of the 4 base preds (no fit; leak-free).
     - inverse_oofwis       : weights ∝ 1/oof_wis (leak-free OOF weights, pre-test).
     - nnls_heldout / invrmse_heldout : weights learned on the FIRST 34 test weeks
       (pseudo-validation), scored on the LAST 34 (Protocol B). This ADVANTAGES the
       generic stack — it sees recent test data FusedEpi never retrained on.
     - nnls_oracle          : NNLS fit on all 68 test weeks, scored in-sample —
       a LEAKY optimistic ceiling for the generic method.

3. CONFORMAL WRAPPER — the SAME model-agnostic adaptive-conformal method from
   FusedEpi's own module (`simulation.analytics.adaptive_conformal`). We use the
   pure-online Conformal-PID variant `online_conformal_bounds`, which is the
   designated leak-free method for point predictors that have NO in-sample residual
   (TiRex and ARIMA genuinely lack one — hence their native test WIS is NaN in the
   official metrics — so a fusion containing them is exactly this case). Each step
   i's interval uses only past obs[0..i-1]; y[i] is appended AFTER the interval is
   set → leak-free (same guarantee FusedEpi's rolling conformal has).

   The SAME wrapper (same window, same ki) is applied to EVERY point predictor,
   including a re-derived "FusedEpi point + same wrapper" line, so any WIS gap is
   attributable to the POINT prediction (fusion mechanism), not the conformal
   machinery. Additionally we cite FusedEpi's NATIVE official WIS/coverage (its
   bespoke NegBin count head + asym Conformal-PID) to see whether that bespoke
   machinery adds anything beyond the generic wrapper.

4. Metrics: WIS via `wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=pred)`
   (identical to per_model_eval R10), PICP95 = empirical coverage of the 95% PI.
   Reported OVERALL (full 68) and in the PEAK regime (y >= 75th pct of test y).

No live pipeline/model code is modified. Writes one JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[1]
OPT_DIR = REPO / "simulation/results/per_model_optimal"
METRICS = REPO / "simulation/results/per_model_eval/per_model_metrics.csv"
OUT = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/elevate/fusion_wis.json"
)

import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from simulation.analytics.adaptive_conformal import (  # noqa: E402
    online_conformal_bounds, wis_from_bounds,
)
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS  # noqa: E402

BASE_MODELS = ["TiRex", "NegBinGLM", "TabPFN", "ARIMA"]
CHAMPION = "FusedEpi"
PRIMARY_WINDOW = 30       # matches FusedEpi conf_window=30
PRIMARY_KI = 0.2          # matches FusedEpi pid_ki=0.2
WINDOW_SWEEP = (20, 30, 40)


def load_frozen(model: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (refit_test_predictions, y_true, oof_wis) from the official R9 JSON."""
    j = json.loads((OPT_DIR / f"{model}.json").read_text())
    pred = np.asarray(j["refit_test_predictions"], dtype=float).ravel()
    oof_wis = float((j.get("val_metrics", {}) or {}).get("oof_wis", np.nan))
    # y_true from the frozen CSV (all models share identical y_true — verified)
    csv = pd.read_csv(REPO / f"simulation/results/csv/predictions_{model}.csv")
    csv = csv.sort_values("idx")
    y_true = csv["y_true"].to_numpy(float)
    return pred, y_true, oof_wis


def score_point_predictor(pred, y, idx_mask, window, ki):
    """Conformalize a point predictor with online adaptive conformal → WIS + PICP95.

    Args:
        pred, y: (n,) test point predictions and observations (rolling order).
        idx_mask: boolean (n,) selecting the scoring subset (overall or peak).
        window, ki: online Conformal-PID rolling window and I-gain.
    Returns: dict(wis, picp95, picp80, picp50, mean_width95, n).
    """
    pred = np.asarray(pred, float).ravel()
    y = np.asarray(y, float).ravel()
    bounds = online_conformal_bounds(pred, y, FLUSIGHT_ALPHAS, window=window, ki=ki)
    wis_arr = wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=pred)
    m = np.asarray(idx_mask, bool)

    def _picp(alpha):
        lo, hi = bounds[alpha]
        cov = (y >= lo) & (y <= hi)
        return float(np.mean(cov[m]))

    lo95, hi95 = bounds[0.05]
    return {
        "wis": float(np.mean(np.asarray(wis_arr)[m])),
        "picp95": _picp(0.05),
        "picp80": _picp(0.20),
        "picp50": _picp(0.50),
        "mean_width95": float(np.mean((hi95 - lo95)[m])),
        "n": int(np.sum(m)),
    }


def main() -> dict:
    # ── load frozen predictions ──
    preds = {}
    oof_wis = {}
    y_true = None
    for m in BASE_MODELS + [CHAMPION]:
        p, yt, ow = load_frozen(m)
        preds[m] = p
        oof_wis[m] = ow
        if y_true is None:
            y_true = yt
        else:
            assert np.max(np.abs(yt - y_true)) < 1e-9, f"y_true mismatch {m}"
    n = len(y_true)
    X = np.column_stack([preds[m] for m in BASE_MODELS])  # (n, 4)

    # ── peak regime = weeks in the top quartile of test incidence ──
    peak_thr = float(np.quantile(y_true, 0.75))
    peak_mask = y_true >= peak_thr
    overall_mask = np.ones(n, dtype=bool)
    alt_thr = 40.0
    alt_peak_mask = y_true >= alt_thr

    # ── generic fusion point predictors (full 68) ──
    fusions_full = {}
    fusions_full["equal_weight"] = X.mean(axis=1)
    inv = np.array([1.0 / oof_wis[m] for m in BASE_MODELS])
    inv = inv / inv.sum()
    fusions_full["inverse_oofwis"] = X @ inv
    # oracle NNLS (leaky ceiling: fit on all 68)
    w_oracle, _ = nnls(X, y_true)
    fusions_full["nnls_oracle"] = X @ w_oracle

    # single base + FusedEpi point, all under the SAME wrapper
    point_predictors_full = {
        **fusions_full,
        "best_single_TiRex": preds["TiRex"],
        "FusedEpi_point": preds[CHAMPION],
    }
    for m in BASE_MODELS:
        point_predictors_full[f"base_{m}"] = preds[m]

    # ── Protocol B: learn weights on first 34, score on last 34 ──
    cut = n // 2
    fit_sl, eval_sl = slice(0, cut), slice(cut, n)
    Xf, yf = X[fit_sl], y_true[fit_sl]
    fit_rmse = np.array([np.sqrt(np.mean((yf - Xf[:, i]) ** 2)) for i in range(4)])
    inv_w = (1.0 / fit_rmse) / np.sum(1.0 / fit_rmse)
    nnls_w, _ = nnls(Xf, yf)
    protoB_full_preds = {
        "invrmse_heldout": X @ inv_w,
        "nnls_heldout": X @ nnls_w,
        "equal_weight": X.mean(axis=1),
        "FusedEpi_point": preds[CHAMPION],
    }
    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[eval_sl] = True
    # peak within the eval (last 34) window
    protoB_peak_mask = eval_mask & (y_true >= float(np.quantile(y_true[eval_sl], 0.75)))

    # ── score everything at the primary window ──
    def score_block(pp_dict, masks):
        out = {}
        for name, p in pp_dict.items():
            out[name] = {
                mk: score_point_predictor(p, y_true, mask, PRIMARY_WINDOW, PRIMARY_KI)
                for mk, mask in masks.items()
            }
        return out

    results_full = score_block(
        point_predictors_full,
        {"overall_68": overall_mask, "peak_top25pct": peak_mask, "peak_ge40": alt_peak_mask},
    )
    results_protoB = score_block(
        protoB_full_preds,
        {"heldout_last34": eval_mask, "heldout_peak": protoB_peak_mask},
    )

    # ── window-sweep robustness on the headline fusions + FusedEpi_point ──
    sweep = {}
    for w in WINDOW_SWEEP:
        sweep[str(w)] = {
            name: {
                "overall_wis": score_point_predictor(p, y_true, overall_mask, w, PRIMARY_KI)["wis"],
                "peak_wis": score_point_predictor(p, y_true, peak_mask, w, PRIMARY_KI)["wis"],
                "peak_picp95": score_point_predictor(p, y_true, peak_mask, w, PRIMARY_KI)["picp95"],
            }
            for name, p in {
                "equal_weight": fusions_full["equal_weight"],
                "inverse_oofwis": fusions_full["inverse_oofwis"],
                "nnls_oracle": fusions_full["nnls_oracle"],
                "FusedEpi_point": preds[CHAMPION],
            }.items()
        }

    # ── FusedEpi NATIVE official reference (its bespoke NegBin + asym Conformal-PID) ──
    met = pd.read_csv(METRICS)

    def off(m, col):
        r = met.loc[met.model == m, col]
        return None if r.empty or pd.isna(r.iloc[0]) else float(r.iloc[0])

    native_ref = {
        m: {
            "wis": off(m, "wis"),
            "pi95_coverage": off(m, "pi95_coverage"),
            "oof_wis": off(m, "oof_wis"),
            "relative_wis_pairwise": off(m, "relative_wis_pairwise"),
        }
        for m in [CHAMPION] + BASE_MODELS
    }

    # ── verdict ──
    fe_overall = results_full["FusedEpi_point"]["overall_68"]["wis"]
    fe_peak = results_full["FusedEpi_point"]["peak_top25pct"]["wis"]
    fe_peak_picp = results_full["FusedEpi_point"]["peak_top25pct"]["picp95"]
    gen_overall = {k: results_full[k]["overall_68"]["wis"]
                   for k in ("equal_weight", "inverse_oofwis", "nnls_oracle")}
    gen_peak = {k: results_full[k]["peak_top25pct"]["wis"]
                for k in ("equal_weight", "inverse_oofwis", "nnls_oracle")}
    best_gen_overall = min(gen_overall, key=gen_overall.get)
    best_gen_peak = min(gen_peak, key=gen_peak.get)

    verdict = {
        "same_wrapper_overall": {
            "fusedepi_point_wis": fe_overall,
            "generic_fusion_wis": gen_overall,
            "best_generic": best_gen_overall,
            "fusedepi_beats_all_generic": bool(all(fe_overall < v for v in gen_overall.values())),
            "fusedepi_beats_best_generic_incl_oracle": bool(fe_overall < gen_overall[best_gen_overall]),
        },
        "same_wrapper_peak_top25pct": {
            "fusedepi_point_wis": fe_peak,
            "fusedepi_point_picp95": fe_peak_picp,
            "generic_fusion_wis": gen_peak,
            "best_generic": best_gen_peak,
            "fusedepi_beats_all_generic": bool(all(fe_peak < v for v in gen_peak.values())),
            "fusedepi_beats_best_generic_incl_oracle": bool(fe_peak < gen_peak[best_gen_peak]),
        },
        "native_vs_wrapper": {
            "fusedepi_native_official_wis": native_ref[CHAMPION]["wis"],
            "fusedepi_native_official_picp95": native_ref[CHAMPION]["pi95_coverage"],
            "fusedepi_point_under_generic_wrapper_wis": fe_overall,
            "note": (
                "native = deployed FusedEpi (NegBin count head + asym Conformal-PID). "
                "wrapper = same FusedEpi point through the generic online conformal used "
                "for the fusion baselines. Compares the bespoke conformal machinery."
            ),
        },
    }

    result = {
        "question": (
            "Does FusedEpi's learned dynamic-alpha + do-no-harm + count head beat a "
            "generic CONFORMALIZED stacking ensemble on WIS and coverage, overall and "
            "at the peak?"
        ),
        "protocol": {
            "split": "run_data frozen split, pool_end=269, n_test=68 (verified 1e-15 vs run_data)",
            "point_source": "OFFICIAL frozen refit_test_predictions (== results/csv/predictions_<M>.csv)",
            "conformal": (
                "simulation.analytics.adaptive_conformal.online_conformal_bounds — pure-online "
                "Conformal-PID (the leak-free method for predictors without in-sample residuals, "
                "which TiRex/ARIMA genuinely lack). Same window=%d, ki=%.2f for ALL predictors."
                % (PRIMARY_WINDOW, PRIMARY_KI)
            ),
            "wis": "wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=pred) — identical to R10 per_model_eval",
            "peak_definition": {
                "peak_top25pct_threshold": peak_thr,
                "n_peak": int(peak_mask.sum()),
                "alt_peak_ge40_n": int(alt_peak_mask.sum()),
            },
        },
        "n_test": n,
        "base_oof_wis": oof_wis,
        "inverse_oofwis_weights": {m: round(float(w), 4) for m, w in zip(BASE_MODELS, inv)},
        "nnls_oracle_weights": {m: round(float(w), 4) for m, w in zip(BASE_MODELS, w_oracle)},
        "protoB_invrmse_weights": {m: round(float(w), 4) for m, w in zip(BASE_MODELS, inv_w)},
        "protoB_nnls_weights": {m: round(float(w), 4) for m, w in zip(BASE_MODELS, nnls_w)},
        "results_full68_same_wrapper": results_full,
        "results_protocolB_heldout": results_protoB,
        "window_sweep": sweep,
        "fusedepi_native_official_reference": native_ref,
        "verdict": verdict,
        "caveats": [
            "POINT predictions are the official R9-optimised frozen base models (strongest "
            "available), so the generic stack is built from strong bases — not weakened.",
            "The generic online conformal wrapper is applied IDENTICALLY to FusedEpi's own "
            "point ('FusedEpi_point'), so any WIS gap under the same wrapper isolates the "
            "POINT/fusion mechanism from the conformal machinery.",
            "FusedEpi's NATIVE official WIS/coverage (NegBin count head + asym Conformal-PID) "
            "is cited separately; it differs from FusedEpi_point-under-wrapper because the "
            "native path uses a count likelihood + asymmetric PID.",
            "inverse_oofwis weights use leak-free OOF WIS (pre-test); nnls_oracle is a LEAKY "
            "in-sample ceiling; Protocol B learns weights on the first 34 test weeks and scores "
            "the last 34 (advantages the generic stack).",
            "Pure-online conformal has a cold-start: the first ~window weeks build the buffer "
            "from scratch. It is identical for every predictor, so relative comparison is fair.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2, ensure_ascii=False))
