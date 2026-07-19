#!/usr/bin/env python
"""FusedEpi vs. generic fusion baselines on the frozen 68-week hold-out test.

Question
--------
Does FusedEpi's *learned dynamic-alpha* fusion actually beat GENERIC fusion of the
same strong base models (equal-weight averaging, inverse-RMSE weighting, NNLS
stacking)? If a trivial average of {TiRex, NegBinGLM, TabPFN, ARIMA} matches
FusedEpi, the learned fusion adds nothing.

Data (all frozen — NO retraining here)
--------------------------------------
- Base + FusedEpi frozen TEST predictions:
    simulation/results/csv/predictions_<Model>.csv  (cols: split,idx,y_true,y_pred)
  These contain the 68-week hold-out test ONLY, and POINT predictions ONLY
  (no quantile/interval columns).
- Official per-model metrics (cross-reference):
    simulation/results/per_model_eval/per_model_metrics.csv

Honest constraints baked into the protocol
------------------------------------------
1. No frozen VALIDATION predictions exist for the strong base models, so NNLS /
   inverse-RMSE weights cannot be learned on a true out-of-sample validation
   split. We therefore learn those weights on a WITHIN-TEST pseudo-validation
   (the first half of the 68 test weeks) and EVALUATE on the held-out second
   half — and we evaluate FusedEpi on the SAME held-out second half so the
   comparison is apples-to-apples. This *advantages* the generic stack (it gets
   to see recent test data that FusedEpi never retrained on).
2. We also report an ORACLE / in-sample fit (weights fit on all 68 test weeks,
   scored on the same 68 weeks). This is leaky and is an OPTIMISTIC UPPER BOUND
   for the generic method. If FusedEpi beats even the oracle stack, the result
   is decisive.
3. Equal-weight averaging needs NO fitting, so it is scored leak-free on the
   full 68 weeks.
4. Intervals are UNAVAILABLE for the point-fusion baselines, so WIS is NOT
   computed for them. We compare on point error (MAE / RMSE). FusedEpi's native
   WIS is reported for context only (its edge is known to be in calibrated
   intervals, not point error).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[1]
CSV_DIR = REPO / "simulation/results/csv"
METRICS = REPO / "simulation/results/per_model_eval/per_model_metrics.csv"
OUT = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/ablation/fusedepi_vs_stacking.json"
)

BASE_MODELS = ["TiRex", "NegBinGLM", "TabPFN", "ARIMA"]
CHAMPION = "FusedEpi"


def load_frozen(model: str) -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / f"predictions_{model}.csv")
    return df.sort_values("idx").reset_index(drop=True)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> dict:
    # ---- load frozen test predictions, aligned on idx ----
    frames = {m: load_frozen(m) for m in BASE_MODELS + [CHAMPION]}
    y_true = frames[CHAMPION]["y_true"].to_numpy(float)
    n = len(y_true)

    # sanity: identical y_true across all files
    for m in BASE_MODELS + [CHAMPION]:
        assert np.max(np.abs(frames[m]["y_true"].to_numpy(float) - y_true)) < 1e-9, m

    # base prediction matrix X: (n, k)
    X = np.column_stack([frames[m]["y_pred"].to_numpy(float) for m in BASE_MODELS])
    fused = frames[CHAMPION]["y_pred"].to_numpy(float)

    # ---- individual base models (full 68) ----
    base_scores = {
        m: {"mae": mae(y_true, X[:, i]), "rmse": rmse(y_true, X[:, i])}
        for i, m in enumerate(BASE_MODELS)
    }
    best_single = min(base_scores, key=lambda m: base_scores[m]["mae"])

    # ---- FusedEpi frozen (full 68) ----
    fused_full = {"mae": mae(y_true, fused), "rmse": rmse(y_true, fused)}

    # ================================================================
    # PROTOCOL A — leak-free, full 68 weeks (no fitted parameters)
    #   equal-weight average vs FusedEpi
    # ================================================================
    eq_full = X.mean(axis=1)
    protocol_A = {
        "note": "Full 68-week hold-out; no fitted weights; strictly leak-free.",
        "equal_weight": {"mae": mae(y_true, eq_full), "rmse": rmse(y_true, eq_full)},
        "fusedepi_frozen": fused_full,
        "best_single_base": {"model": best_single, **base_scores[best_single]},
    }

    # ================================================================
    # PROTOCOL B — within-test pseudo-validation split.
    #   Learn inverse-RMSE + NNLS weights on FIRST half (fit),
    #   evaluate all methods on SECOND half (eval). Apples-to-apples,
    #   and advantages the generic stack (sees recent test data).
    # ================================================================
    cut = n // 2  # 34
    fit_sl, eval_sl = slice(0, cut), slice(cut, n)
    yf, ye = y_true[fit_sl], y_true[eval_sl]
    Xf, Xe = X[fit_sl], X[eval_sl]

    # inverse-RMSE weights from fit half
    fit_rmse = np.array([rmse(yf, Xf[:, i]) for i in range(len(BASE_MODELS))])
    inv_w = (1.0 / fit_rmse) / np.sum(1.0 / fit_rmse)
    invrmse_eval = Xe @ inv_w

    # NNLS stacking weights from fit half (non-negative, no intercept)
    nnls_w, _ = nnls(Xf, yf)
    nnls_eval = Xe @ nnls_w

    # equal-weight on eval half (leak-free reference on same window)
    eq_eval = Xe.mean(axis=1)

    protocol_B = {
        "note": (
            f"Weights learned on first {cut} test weeks (pseudo-validation), "
            f"scored on held-out last {n - cut} weeks. FusedEpi scored on the "
            "same last weeks. Generic stack is advantaged (sees recent data "
            "FusedEpi never retrained on)."
        ),
        "n_fit": cut,
        "n_eval": n - cut,
        "inverse_rmse_weights": {
            m: round(float(w), 4) for m, w in zip(BASE_MODELS, inv_w)
        },
        "nnls_weights": {m: round(float(w), 4) for m, w in zip(BASE_MODELS, nnls_w)},
        "equal_weight": {"mae": mae(ye, eq_eval), "rmse": rmse(ye, eq_eval)},
        "inverse_rmse": {"mae": mae(ye, invrmse_eval), "rmse": rmse(ye, invrmse_eval)},
        "nnls_stacking": {"mae": mae(ye, nnls_eval), "rmse": rmse(ye, nnls_eval)},
        "fusedepi_frozen": {"mae": mae(ye, fused[eval_sl]), "rmse": rmse(ye, fused[eval_sl])},
    }

    # ================================================================
    # ORACLE / leaky upper bound — fit on all 68, score on all 68.
    #   Optimistic upper bound for the generic method.
    # ================================================================
    full_rmse = np.array([rmse(y_true, X[:, i]) for i in range(len(BASE_MODELS))])
    inv_w_full = (1.0 / full_rmse) / np.sum(1.0 / full_rmse)
    invrmse_oracle = X @ inv_w_full
    nnls_w_full, _ = nnls(X, y_true)
    nnls_oracle = X @ nnls_w_full
    oracle = {
        "note": (
            "LEAKY upper bound: weights fit on all 68 test weeks then scored on "
            "the same 68 weeks. Optimistic ceiling for generic stacking; not a "
            "deployable number. If FusedEpi beats this, the win is decisive."
        ),
        "inverse_rmse_weights": {
            m: round(float(w), 4) for m, w in zip(BASE_MODELS, inv_w_full)
        },
        "nnls_weights": {m: round(float(w), 4) for m, w in zip(BASE_MODELS, nnls_w_full)},
        "inverse_rmse": {"mae": mae(y_true, invrmse_oracle), "rmse": rmse(y_true, invrmse_oracle)},
        "nnls_stacking": {"mae": mae(y_true, nnls_oracle), "rmse": rmse(y_true, nnls_oracle)},
        "fusedepi_frozen": fused_full,
    }

    # ---- official metrics cross-reference (incl. WIS which baselines lack) ----
    met = pd.read_csv(METRICS)

    def off(m, col):
        r = met.loc[met.model == m, col]
        return None if r.empty or pd.isna(r.iloc[0]) else float(r.iloc[0])

    official = {
        m: {"mae": off(m, "mae"), "rmse": off(m, "rmse"), "wis": off(m, "wis"),
            "oof_wis": off(m, "oof_wis"), "relative_wis_pairwise": off(m, "relative_wis_pairwise")}
        for m in [CHAMPION] + BASE_MODELS
    }

    # ---- verdict logic (point error) ----
    def beats(a, b, key="mae"):
        return a[key] < b[key]

    verdict = {
        "A_full68_equalweight_vs_fusedepi": {
            "fusedepi_mae": protocol_A["fusedepi_frozen"]["mae"],
            "equalweight_mae": protocol_A["equal_weight"]["mae"],
            "fusedepi_wins_mae": beats(protocol_A["fusedepi_frozen"], protocol_A["equal_weight"], "mae"),
            "fusedepi_wins_rmse": beats(protocol_A["fusedepi_frozen"], protocol_A["equal_weight"], "rmse"),
        },
        "B_heldout_nnls_vs_fusedepi": {
            "fusedepi_mae": protocol_B["fusedepi_frozen"]["mae"],
            "nnls_mae": protocol_B["nnls_stacking"]["mae"],
            "fusedepi_wins_mae_vs_nnls": beats(protocol_B["fusedepi_frozen"], protocol_B["nnls_stacking"], "mae"),
            "fusedepi_wins_mae_vs_invrmse": beats(protocol_B["fusedepi_frozen"], protocol_B["inverse_rmse"], "mae"),
            "fusedepi_wins_mae_vs_equal": beats(protocol_B["fusedepi_frozen"], protocol_B["equal_weight"], "mae"),
        },
        "oracle_leaky_nnls_vs_fusedepi": {
            "fusedepi_mae": oracle["fusedepi_frozen"]["mae"],
            "nnls_oracle_mae": oracle["nnls_stacking"]["mae"],
            "fusedepi_wins_mae_vs_oracle_nnls": beats(oracle["fusedepi_frozen"], oracle["nnls_stacking"], "mae"),
        },
    }

    result = {
        "question": "Does FusedEpi's learned dynamic-alpha fusion beat generic "
                    "averaging / inverse-RMSE / NNLS stacking of the same base models?",
        "base_models": BASE_MODELS,
        "n_test": n,
        "individual_base_test_full68": base_scores,
        "best_single_base": best_single,
        "protocol_A_leakfree_full68": protocol_A,
        "protocol_B_heldout_half": protocol_B,
        "oracle_leaky_insample": oracle,
        "official_metrics_crossref": official,
        "verdict": verdict,
        "caveats": [
            "Frozen prediction CSVs contain POINT predictions only (no quantile "
            "columns), so WIS cannot be computed for point-fusion baselines; "
            "comparison is on MAE/RMSE. FusedEpi's WIS is reported from the "
            "official metrics CSV for context only.",
            "No frozen validation predictions exist for the strong base models; "
            "NNLS/inverse-RMSE weights are learned on a within-test pseudo-"
            "validation half (Protocol B) or in-sample (oracle). Both advantage "
            "the generic stack relative to FusedEpi.",
            f"FusedEpi recomputed from its frozen CSV (MAE={fused_full['mae']:.4f}, "
            f"RMSE={fused_full['rmse']:.4f}) differs slightly from the official "
            f"metrics CSV (MAE={official[CHAMPION]['mae']:.4f}, "
            f"RMSE={official[CHAMPION]['rmse']:.4f}); the CSV reflects a later "
            "conformal recalibration. Head-to-head uses the frozen CSV so all "
            "methods share identical y_true.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2))
