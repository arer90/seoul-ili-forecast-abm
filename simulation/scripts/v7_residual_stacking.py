"""V7: residual stacking + LightGBM blend on top of V4 graph model.

Purpose
-------
V4 (graph + temporal aggregate) hit test R²=0.8055 in 2026-04-21 smoke run.
V7 asks: can a LightGBM baseline on the same X_agg_temp push this past 0.85?

Five ensembles compared on the same 240-train / 51-val / 48-test split:

  V4_only    — V4 graph model alone (reference, reuses gedn_variant_comparison)
  LGBM_only  — LightGBM on X_agg_temp alone (no graph)
  Blend_50   — 0.5·V4 + 0.5·LGBM (equal-weight average)
  Blend_W    — weighted by inverse validation MSE of V4 and LGBM
  Ridge_Meta — Ridge meta-learner (V4_pred, LGBM_pred) → y, fit on train predictions
  Stack_Res  — V4_pred + LGBM_residual_pred, where LGBM fits on (y - V4_pred)

All ensembles are evaluated on the held-out test split. Train-range predictions
of V4 come from the newly-exposed `train_pred` return of `run_variant`
(gedn_variant_comparison.py).

Causality / leakage
-------------------
- X_agg_temp is built causally (lag_y via np.shift, NaN filled with train-slice mean).
- Z-score on X_agg_temp fits on train slice only.
- LGBM trains on X_agg_temp[tr], predicts X_agg_temp[va]/[te] — no test leakage.
- Ridge meta fits on TRAIN predictions (V4_train_pred, LGBM_train_pred) → y_train,
  then applies to test. Training-set predictions are slightly optimistic for V4
  since they come from a fitted model, but they're still apples-to-apples.

Output
------
  simulation/results/v7_residual_stacking/
    results.json     — {ensemble: {test_r2, mae, rmse, val_r2, ...}}
    run.log
    predictions.csv  — per-week (val+test) truth vs each ensemble prediction
    summary.md       — ranked summary table
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("v7_stack")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2) + 1e-12)
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "n": int(len(y_true)),
    }


def _prepare_X_agg_temp(X_agg_temp: np.ndarray, tr: slice) -> np.ndarray:
    """NaN-fill with train-slice mean, then leave raw (LGBM handles scale).

    Mirrors the NaN-fill in run_variant so LGBM sees the same inputs as V4.
    """
    out = X_agg_temp.copy()
    for j in range(out.shape[1]):
        col_tr = out[tr, j]
        if np.isnan(col_tr).all():
            fill = 0.0
        else:
            fill = float(np.nanmean(col_tr))
        out[np.isnan(out[:, j]), j] = fill
    return out


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _fit_lgbm(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
              *, seed: int, rounds: int = 400) -> tuple:
    import lightgbm as lgb
    train_ds = lgb.Dataset(X_tr, label=y_tr)
    val_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": seed,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=rounds,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def main() -> int:
    ap = argparse.ArgumentParser()
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--out", default=str(get_results_dir() / "v7_residual_stacking"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ssl-epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default="simulation/data/db/epi_real_seoul.db")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _configure_logging(out / "run.log")

    # Reuse loaders + V4 training from the sister script.
    from simulation.scripts.gedn_variant_comparison import (
        load_aligned_data,
        run_variant,
        _temporal_split,
    )

    log.info(f"[v7] args={vars(args)}")
    log.info("[v7] loading aligned data ...")
    t0 = time.time()
    data = load_aligned_data(args.db)
    log.info(
        f"[v7] data ready: X_gu={data['X_gu'].shape}, "
        f"X_agg_temp={data['X_agg_temp'].shape}, y={data['y'].shape}, "
        f"load_wall={time.time()-t0:.1f}s"
    )

    y = data["y"]
    X_agg_temp_raw = data["X_agg_temp"]
    T = len(y)
    tr, va, te = _temporal_split(T, 0.70, 0.15)

    # --- V4 training (reuses run_variant; now returns train_pred too) ---
    log.info("[v7] stage-1: training V4 graph model ...")
    v4_t0 = time.time()
    v4 = run_variant("V4", data, epochs=args.epochs, seed=args.seed,
                    ssl_epochs=args.ssl_epochs)
    v4_wall = time.time() - v4_t0
    log.info(
        f"[v7] V4 done test_r2={v4['test']['r2']:.4f} val_r2={v4['val']['r2']:.4f} "
        f"wall={v4_wall:.1f}s"
    )

    v4_tr_pred = np.asarray(v4["train_pred"], dtype=np.float64)
    v4_va_pred = np.asarray(v4["val_pred"], dtype=np.float64)
    v4_te_pred = np.asarray(v4["test_pred"], dtype=np.float64)

    # --- LGBM baseline on X_agg_temp ---
    log.info("[v7] stage-2: training LightGBM baseline on X_agg_temp ...")
    X_at = _prepare_X_agg_temp(X_agg_temp_raw, tr)
    X_tr = X_at[tr]; X_va = X_at[va]; X_te = X_at[te]
    y_tr = y[tr].astype(np.float64)
    y_va = y[va].astype(np.float64)
    y_te = y[te].astype(np.float64)

    lgbm = _fit_lgbm(X_tr, y_tr, X_va, y_va, seed=args.seed)
    lgbm_tr_pred = lgbm.predict(X_tr, num_iteration=lgbm.best_iteration)
    lgbm_va_pred = lgbm.predict(X_va, num_iteration=lgbm.best_iteration)
    lgbm_te_pred = lgbm.predict(X_te, num_iteration=lgbm.best_iteration)

    # --- Ensembles ---
    log.info("[v7] stage-3: building ensembles ...")

    # 1) V4 only
    e_v4 = {"val": _metrics(y_va, v4_va_pred), "test": _metrics(y_te, v4_te_pred)}

    # 2) LGBM only
    e_lgbm = {"val": _metrics(y_va, lgbm_va_pred), "test": _metrics(y_te, lgbm_te_pred)}

    # 3) 50/50 blend
    blend50_va = 0.5 * v4_va_pred + 0.5 * lgbm_va_pred
    blend50_te = 0.5 * v4_te_pred + 0.5 * lgbm_te_pred
    e_blend50 = {"val": _metrics(y_va, blend50_va), "test": _metrics(y_te, blend50_te)}

    # 4) Weighted by inverse validation MSE
    mse_v4 = float(np.mean((y_va - v4_va_pred) ** 2) + 1e-12)
    mse_lgbm = float(np.mean((y_va - lgbm_va_pred) ** 2) + 1e-12)
    inv_v4 = 1.0 / mse_v4
    inv_lgbm = 1.0 / mse_lgbm
    w_v4 = inv_v4 / (inv_v4 + inv_lgbm)
    w_lgbm = inv_lgbm / (inv_v4 + inv_lgbm)
    blendw_va = w_v4 * v4_va_pred + w_lgbm * lgbm_va_pred
    blendw_te = w_v4 * v4_te_pred + w_lgbm * lgbm_te_pred
    e_blendw = {
        "val": _metrics(y_va, blendw_va),
        "test": _metrics(y_te, blendw_te),
        "w_v4": w_v4, "w_lgbm": w_lgbm,
    }

    # 5) Ridge meta-learner over (V4_pred, LGBM_pred) → y, fit on train predictions
    from sklearn.linear_model import Ridge
    Z_tr = np.column_stack([v4_tr_pred, lgbm_tr_pred])
    Z_va = np.column_stack([v4_va_pred, lgbm_va_pred])
    Z_te = np.column_stack([v4_te_pred, lgbm_te_pred])
    ridge = Ridge(alpha=1.0, random_state=args.seed)
    ridge.fit(Z_tr, y_tr)
    ridge_va = ridge.predict(Z_va)
    ridge_te = ridge.predict(Z_te)
    e_ridge = {
        "val": _metrics(y_va, ridge_va),
        "test": _metrics(y_te, ridge_te),
        "coef": [float(c) for c in ridge.coef_],
        "intercept": float(ridge.intercept_),
    }

    # 6) Residual stacking: V4_pred + LGBM fit on (y_train - V4_train_pred)
    resid_tr = y_tr - v4_tr_pred
    resid_va_target = y_va - v4_va_pred
    lgbm_resid = _fit_lgbm(X_tr, resid_tr, X_va, resid_va_target, seed=args.seed)
    resid_va_pred = lgbm_resid.predict(X_va, num_iteration=lgbm_resid.best_iteration)
    resid_te_pred = lgbm_resid.predict(X_te, num_iteration=lgbm_resid.best_iteration)
    stack_va = v4_va_pred + resid_va_pred
    stack_te = v4_te_pred + resid_te_pred
    e_stack = {"val": _metrics(y_va, stack_va), "test": _metrics(y_te, stack_te)}

    ensembles = {
        "V4_only": e_v4,
        "LGBM_only": e_lgbm,
        "Blend_50": e_blend50,
        "Blend_W": e_blendw,
        "Ridge_Meta": e_ridge,
        "Stack_Res": e_stack,
    }

    for name, m in ensembles.items():
        t = m["test"]
        v = m["val"]
        log.info(
            f"[v7] {name:10s} val_r2={v['r2']:7.4f} "
            f"test_r2={t['r2']:7.4f} mae={t['mae']:6.3f} rmse={t['rmse']:6.3f}"
        )

    # Save JSON
    out_json = out / "results.json"
    payload = {
        "meta": {
            "seed": args.seed,
            "epochs": args.epochs,
            "ssl_epochs": args.ssl_epochs,
            "n_train": int(tr.stop - tr.start),
            "n_val": int(va.stop - va.start),
            "n_test": int(te.stop - te.start),
            "weeks": {
                "first": data["weeks"][0],
                "last": data["weeks"][-1],
            },
            "feat_names_agg_temp": data["feat_names_agg_temp"],
        },
        "ensembles": ensembles,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info(f"[v7] wrote {out_json}")

    # Predictions CSV
    weeks_va = data["weeks"][va.start:va.stop]
    weeks_te = data["weeks"][te.start:te.stop]
    out_csv = out / "predictions.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = [
            "week", "split", "y_true",
            "V4", "LGBM", "Blend_50", "Blend_W", "Ridge_Meta", "Stack_Res",
        ]
        w.writerow(header)
        for i, wk in enumerate(weeks_va):
            w.writerow([
                wk, "val", float(y_va[i]),
                float(v4_va_pred[i]), float(lgbm_va_pred[i]),
                float(blend50_va[i]), float(blendw_va[i]),
                float(ridge_va[i]), float(stack_va[i]),
            ])
        for i, wk in enumerate(weeks_te):
            w.writerow([
                wk, "test", float(y_te[i]),
                float(v4_te_pred[i]), float(lgbm_te_pred[i]),
                float(blend50_te[i]), float(blendw_te[i]),
                float(ridge_te[i]), float(stack_te[i]),
            ])
    log.info(f"[v7] wrote {out_csv}")

    # Markdown summary
    ranked = sorted(ensembles.items(), key=lambda kv: -kv[1]["test"]["r2"])
    md = out / "summary.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# V7 residual stacking — Seoul ILI graph+LGBM ensembles\n\n")
        f.write(f"- Generated: 2026-04-21\n")
        f.write(f"- n_train={tr.stop-tr.start}, n_val={va.stop-va.start}, "
                f"n_test={te.stop-te.start}\n")
        f.write(f"- Feat agg_temp: {data['feat_names_agg_temp']}\n\n")
        f.write("## Test-set ranking (by R²)\n\n")
        f.write("| Rank | Ensemble | Test R² | Test MAE | Test RMSE | Val R² |\n")
        f.write("|------|----------|--------:|---------:|----------:|-------:|\n")
        for rk, (name, m) in enumerate(ranked, 1):
            t = m["test"]; v = m["val"]
            f.write(
                f"| {rk} | `{name}` | {t['r2']:.4f} | {t['mae']:.3f} | "
                f"{t['rmse']:.3f} | {v['r2']:.4f} |\n"
            )
        f.write("\n## Notes\n\n")
        f.write("- V4 = graph (V3 arch + SSL) + temporal aggregate branch "
                "(lag_y 1/2/4/8, temp_avg, temp_avg_lag1, sin/cos(52)).\n")
        f.write("- LGBM uses the same 8 X_agg_temp features (no per-gu signal).\n")
        f.write(f"- Blend_W weights: V4={e_blendw['w_v4']:.3f}, "
                f"LGBM={e_blendw['w_lgbm']:.3f} (inverse-val-MSE)\n")
        f.write(f"- Ridge_Meta coefs: {e_ridge['coef']}, intercept={e_ridge['intercept']:.3f}\n")
    log.info(f"[v7] wrote {md}")

    # Stdout summary (ASCII-only to survive Windows cp949 consoles)
    print("\n" + "=" * 70)
    print("V7 RESIDUAL STACKING - SEOUL ILI GRAPH+LGBM ENSEMBLES")
    print("=" * 70)
    print(f"{'Ensemble':<12} {'Test R2':>10} {'Test MAE':>10} {'Test RMSE':>10} {'Val R2':>10}")
    for name, m in ranked:
        t = m["test"]; v = m["val"]
        print(f"{name:<12} {t['r2']:10.4f} {t['mae']:10.3f} "
              f"{t['rmse']:10.3f} {v['r2']:10.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
