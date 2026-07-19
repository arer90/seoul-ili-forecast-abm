"""GE-DNN V1 vs V2 vs V3 apples-to-apples comparison.

Purpose
-------
Answer the user's question from 2026-04-21:
 "노드와 엣지를 통한 지역구 행렬을 만들어서 하지 못하는거야?"
i.e. why is GE-DNN performing so poorly (R²=0.0249 in dry / ~0 in ),
and would a proper per-gu node/edge matrix help?

This script runs three variants with IDENTICAL:
 * train / val / test split (temporal, same cut-points)
 * seed (numpy + torch, incl. deterministic_algorithms)
 * Seoul-aggregate ILI target (from sentinel_influenza)
 * commuter adjacency (from CommutingMatrix, symmetric-normalized)
 * HPs (node_hidden=64, mlp_hidden=128, dropout=0.2, gelu, LayerNorm,
 kaiming init, lr=5e-4, bs=32, 200 epochs, patience 25)
 * no Optuna (HPs frozen — this is an ablation, not a race)

What differs per variant:
 * V1 Broadcast : input = X_agg (per-gu sum aggregated to Seoul-level),
 broadcast to 25 identical nodes (current GE-DNN design).
 * V2 PerGu : input = X_gu (25 gu-specific rows), real spatial signal
 flows through the GCN.
 * V2A : V2 + aggregate branch with static X_agg (sum of per-gu).
 * V3 Pretrained: V2 arch + 50-epoch masked-gu SSL pre-training on
 training-range X_gu before fine-tuning.
 * V3A : V3 + aggregate branch with static X_agg.
 * V4 TemporalAgg: V3 arch + SSL + aggregate branch with **temporal** features
 X_agg_temp (T, 8) = [lag_y 1/2/4/8, temp_avg, temp_avg_lag1,
 sin(2π·w/52), cos(2π·w/52)]. Unlike V1..V3A whose agg branch
 carries only static per-gu pop/age/etc., V4's agg branch carries
 the dynamic signal that the target actually follows (seasonality
 + autoregression + weather). Expected to escape the negative-R²
 regime of V1..V3A.

Outputs
-------
 simulation/results/gedn_variant_comparison/
 └── results.json # {variant: {test_r2, test_mae, test_rmse, wall_s, ...}}
 └── run.log # stdout/stderr of the run
 └── predictions.csv # per-week predictions for all 3 variants + truth

All outputs ASCII-only so Windows cp949 terminals won't crash (see the
`✓`-encoding bug that tripped verify_v22_7_fixes.py run 2).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("gedn_compare")


# ----------------------------------------------------------------------
# Deterministic seed setup (mirrors runner.run_pipeline block)
# ----------------------------------------------------------------------
def _seed_all(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------
# Data prep — align sentinel_ili target to per-gu weeks
# ----------------------------------------------------------------------
def _cal_to_iso(d) -> str:
    """Calendar date → 'YYYY-Www' ISO week label, capped at W52."""
    import datetime as dt
    if isinstance(d, dt.datetime):
        d = d.date()
    y, w, _ = d.isocalendar()
    # per_gu_loader clamps weeks 53→52; mirror that here.
    if w > 52:
        w = 52
    return f"{y}-W{w:02d}"


def load_aligned_data(db_path: str):
    """Load y (Seoul ILI rate) aligned to per-gu feature weeks.

    Returns
    -------
    X_gu        : (T, 25, K_gu) float32
    X_agg       : (T, K_gu)     float32 — per-gu sum, Seoul-level aggregate (V1/V2A/V3A)
    X_agg_temp  : (T, 8)        float32 — V4 temporal-aggregate features
                   cols = [lag_y_1, lag_y_2, lag_y_4, lag_y_8,
                           temp_avg, temp_avg_lag1, sin(2π·w/52), cos(2π·w/52)]
                  Causality: lag_y_k uses REAL past y (shift by k, NaN at start).
                  NaN filling is deferred to run_variant (uses train-slice mean to
                  avoid leakage). At inference the lag values are 1-step-ahead
                  real past y (standard TS eval, same as R4 WF-CV).
    y           : (T,)          float32 — sentinel ILI rate, Seoul-wide avg
    weeks       : list[str]     aligned ISO-week labels
    gu_order    : list[str]
    feat_names_gu : list[str]
    feat_names_agg_temp : list[str]  V4 aggregate feature names
    """
    from simulation.models.feature_engine.per_gu_loader import build_per_gu_bundle
    from simulation.models.feature_engine.loaders import _load_sentinel_ili, _load_weather

    # 1. target
    ili_df = _load_sentinel_ili(db_path)            # polars DataFrame
    # Extract (cal_date, ili_rate) pairs, drop nulls.
    y_df = ili_df.select(["cal_date", "ili_rate"]).drop_nulls()
    # Build ISO labels.
    cal_dates = y_df["cal_date"].to_list()
    ili_rate = y_df["ili_rate"].to_list()
    iso_labels = [_cal_to_iso(d) for d in cal_dates]
    y_by_label = {}
    for lbl, v in zip(iso_labels, ili_rate):
        # If duplicate labels (week 53 → 52 folded), average.
        y_by_label.setdefault(lbl, []).append(v)
    y_by_label = {k: float(np.mean(v)) for k, v in y_by_label.items()}

    # 2. per-gu features, aligned to target weeks
    target_labels = sorted(y_by_label.keys())
    bundle = build_per_gu_bundle(target_week_labels=target_labels)

    # 3. align y to bundle.week_labels (target_labels ∩ pop_weeks order).
    y = np.asarray([y_by_label[w] for w in bundle.week_labels], dtype=np.float32)
    X_gu = bundle.X_gu.astype(np.float32)            # (T, 25, K)
    X_agg = X_gu.sum(axis=1).astype(np.float32)      # (T, K) — Seoul-total

    # 4. V4 temporal aggregate branch — weather + lag-y + fourier
    weather_weekly = _load_weather(db_path)          # polars: week_start, temp_avg, ...
    # week_start (Date) → ISO "YYYY-Www" label to match bundle.week_labels
    ws_dates = weather_weekly["week_start"].to_list()
    ws_temp = weather_weekly["temp_avg"].to_list()
    temp_by_label: dict[str, list[float]] = {}
    for d, t in zip(ws_dates, ws_temp):
        if t is None:
            continue
        lbl = _cal_to_iso(d)
        temp_by_label.setdefault(lbl, []).append(float(t))
    temp_by_label = {k: float(np.mean(v)) for k, v in temp_by_label.items()}

    T = len(bundle.week_labels)
    temp_weekly = np.full(T, np.nan, dtype=np.float32)
    sin_w = np.zeros(T, dtype=np.float32)
    cos_w = np.zeros(T, dtype=np.float32)
    for i, lbl in enumerate(bundle.week_labels):
        if lbl in temp_by_label:
            temp_weekly[i] = temp_by_label[lbl]
        try:
            wk = int(lbl.split("-W")[-1])
        except ValueError:
            wk = 1
        sin_w[i] = np.sin(2.0 * np.pi * wk / 52.0)
        cos_w[i] = np.cos(2.0 * np.pi * wk / 52.0)

    # Causal lag features (NaN at start, filled later in run_variant using train-slice mean).
    def _shift(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.full_like(arr, np.nan, dtype=np.float32)
        if k < len(arr):
            out[k:] = arr[:-k]
        return out

    lag_y_1 = _shift(y, 1)
    lag_y_2 = _shift(y, 2)
    lag_y_4 = _shift(y, 4)
    lag_y_8 = _shift(y, 8)
    temp_lag1 = _shift(temp_weekly, 1)

    X_agg_temp = np.column_stack([
        lag_y_1, lag_y_2, lag_y_4, lag_y_8,
        temp_weekly, temp_lag1,
        sin_w, cos_w,
    ]).astype(np.float32)

    return {
        "X_gu": X_gu,
        "X_agg": X_agg,
        "X_agg_temp": X_agg_temp,
        "y": y,
        "weeks": list(bundle.week_labels),
        "gu_order": list(bundle.gu_order),
        "feat_names_gu": list(bundle.feature_names),
        "feat_names_agg_temp": [
            "lag_y_1", "lag_y_2", "lag_y_4", "lag_y_8",
            "temp_avg", "temp_avg_lag1", "sin_w52", "cos_w52",
        ],
    }


def _temporal_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15):
    """Return (idx_tr, idx_va, idx_te) contiguous index slices."""
    n_tr = int(round(n * train_frac))
    n_va = int(round(n * val_frac))
    n_te = n - n_tr - n_va
    if n_te < 5:
        n_te = 5
        n_va = max(5, n - n_tr - n_te)
    return (
        slice(0, n_tr),
        slice(n_tr, n_tr + n_va),
        slice(n_tr + n_va, n),
    )


def _fit_zscore(train_arr: np.ndarray, axis_reduce) -> tuple[np.ndarray, np.ndarray]:
    mu = train_arr.mean(axis=axis_reduce, keepdims=False)
    sd = train_arr.std(axis=axis_reduce, keepdims=False) + 1e-6
    return mu, sd


# ----------------------------------------------------------------------
# Metric helpers — sklearn fallback-free
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Run single variant
# ----------------------------------------------------------------------
def run_variant(
    variant: str,
    data: dict,
    *,
    epochs: int,
    seed: int,
    ssl_epochs: int,
) -> dict:
    """Train one variant, return metrics + timing + predictions."""
    import simulation.models.graph_models_variants as gmv2

    _seed_all(seed)

    X_gu = data["X_gu"]                              # (T, 25, K_gu)
    X_agg_static = data["X_agg"]                     # (T, K_gu) — per-gu sum (static agg)
    X_agg_temp = data.get("X_agg_temp")              # (T, 8) or None — V4 temporal agg
    y = data["y"]                                     # (T,)
    T, N, K = X_gu.shape

    tr, va, te = _temporal_split(T, 0.70, 0.15)
    n_tr = tr.stop - tr.start

    # Z-score — fit on train only (no leakage).
    # X_gu: reduce over (T_train, 25) per-feature → shape (K,)
    mu_gu = X_gu[tr].mean(axis=(0, 1))
    sd_gu = X_gu[tr].std(axis=(0, 1)) + 1e-6
    Xgu_n = (X_gu - mu_gu) / sd_gu

    # V1/V2A/V3A use static X_agg (sum of per-gu); V4 uses X_agg_temp (lag-y + temp + fourier).
    # Decide which aggregate tensor to use, then z-score on train.
    variant_upper = variant.upper()
    if variant_upper == "V4":
        if X_agg_temp is None:
            raise ValueError(
                "V4 requires X_agg_temp in data dict. "
                "Update load_aligned_data to include temporal aggregate features."
            )
        Xagg_src = X_agg_temp.copy()
        # NaN fill — lag features have NaN at start. Fill with train-slice mean per column
        # (causal: only train range used for imputation prior).
        for j in range(Xagg_src.shape[1]):
            col_tr = Xagg_src[tr, j]
            nan_tr = np.isnan(col_tr)
            if nan_tr.all():
                fill = 0.0
            else:
                fill = float(np.nanmean(col_tr))
            col_full = Xagg_src[:, j]
            col_full[np.isnan(col_full)] = fill
            Xagg_src[:, j] = col_full
    else:
        Xagg_src = X_agg_static

    mu_agg = Xagg_src[tr].mean(axis=0)
    sd_agg = Xagg_src[tr].std(axis=0) + 1e-6
    Xagg_n = (Xagg_src - mu_agg) / sd_agg

    # y: same for target
    mu_y = float(y[tr].mean())
    sd_y = float(y[tr].std() + 1e-6)
    y_n = (y - mu_y) / sd_y

    adj_norm = gmv2._normalize_commuter_adjacency()

    # Variant → (base arch, use aggregate branch?) mapping:
    #   V1       : V1-broadcast (current GE-DNN)       use_agg=True   (Xagg_src = static sum)
    #   V2       : V2-pergu, no SSL, no agg            use_agg=False
    #   V2A      : V2-pergu, no SSL, + agg branch      use_agg=True   (static sum)
    #   V3       : V2-pergu + SSL, no agg              use_agg=False
    #   V3A      : V2-pergu + SSL, + agg branch        use_agg=True   (static sum)
    #   V4       : V2-pergu + SSL, + agg branch        use_agg=True   (TEMPORAL agg = lag-y + temp + fourier)
    if variant_upper == "V4":
        base = "V3"
        use_agg = True
    else:
        use_agg = variant.endswith("A")
        base = variant.rstrip("A")                       # V1 / V2 / V3

    # --- build model ---
    t_build = time.time()
    if base == "V1":
        # V1 uses the aggregate vector (Seoul-total). Same path as current GE-DNN.
        K_v1 = Xagg_n.shape[1]
        model = gmv2.build_v1_broadcast_model(K_v1, adj_norm, n_nodes=25)
    elif base in ("V2", "V3"):
        k_agg = Xagg_n.shape[1] if use_agg else 0
        model = gmv2.build_v2_pergu_model(
            k_gu=K, adj_norm=adj_norm, k_aggregate=k_agg, n_nodes=25,
        )
    else:
        raise ValueError(f"unknown variant {variant}")

    # --- V3 SSL pre-training ---
    ssl_info = None
    if base == "V3":
        ssl_t0 = time.time()
        ssl_info = gmv2.pretrain_v3_ssl(
            model, Xgu_n[tr], epochs=ssl_epochs, lr=1e-3,
            mask_ratio=0.15, batch_size=16, verbose=True,
        )
        ssl_info["wall_s"] = time.time() - ssl_t0
        log.info(
            f"[{variant}] SSL pre-train: best={ssl_info['best_loss']:.4f} "
            f"wall={ssl_info['wall_s']:.1f}s"
        )

    # --- supervised fit ---
    fit_t0 = time.time()
    if base == "V1":
        fit = gmv2.train_variant(
            model, variant="V1",
            y_train=y_n[tr], x_agg_train=Xagg_n[tr],
            epochs=epochs, lr=5e-4, batch_size=32, patience=25,
            weight_decay=1e-3, graph_reg_weight=0.01, verbose=False,
        )
    else:
        fit = gmv2.train_variant(
            model, variant=base,
            y_train=y_n[tr], x_gu_train=Xgu_n[tr],
            x_agg_train=(Xagg_n[tr] if use_agg else None),
            epochs=epochs, lr=5e-4, batch_size=32, patience=25,
            weight_decay=1e-3, graph_reg_weight=0.01, verbose=False,
        )
    fit_wall = time.time() - fit_t0

    # --- predict on train + val + test (train included for V7 residual stacking) ---
    if base == "V1":
        train_pred_n = gmv2.predict_variant(
            model, variant="V1", x_agg=Xagg_n[tr],
        )
        val_pred_n = gmv2.predict_variant(
            model, variant="V1", x_agg=Xagg_n[va],
        )
        test_pred_n = gmv2.predict_variant(
            model, variant="V1", x_agg=Xagg_n[te],
        )
    else:
        train_pred_n = gmv2.predict_variant(
            model, variant=base, x_gu=Xgu_n[tr],
            x_agg=(Xagg_n[tr] if use_agg else None),
        )
        val_pred_n = gmv2.predict_variant(
            model, variant=base, x_gu=Xgu_n[va],
            x_agg=(Xagg_n[va] if use_agg else None),
        )
        test_pred_n = gmv2.predict_variant(
            model, variant=base, x_gu=Xgu_n[te],
            x_agg=(Xagg_n[te] if use_agg else None),
        )

    train_pred = train_pred_n * sd_y + mu_y
    val_pred = val_pred_n * sd_y + mu_y
    test_pred = test_pred_n * sd_y + mu_y
    val_metrics = _metrics(y[va], val_pred)
    test_metrics = _metrics(y[te], test_pred)

    total_wall = time.time() - t_build
    log.info(
        f"[{variant}] done test_r2={test_metrics['r2']:.4f} "
        f"mae={test_metrics['mae']:.3f} rmse={test_metrics['rmse']:.3f} "
        f"fit_wall={fit_wall:.1f}s total={total_wall:.1f}s"
    )

    train_metrics = _metrics(y[tr], train_pred)

    return {
        "variant": variant,
        "n_train": n_tr,
        "n_val": va.stop - va.start,
        "n_test": te.stop - te.start,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_val_loss": fit["best_val_loss"],
        "epochs_run": fit["epochs_run"],
        "fit_wall_s": fit_wall,
        "total_wall_s": total_wall,
        "ssl": ssl_info,
        "train_pred": train_pred.tolist(),
        "val_pred": val_pred.tolist(),
        "test_pred": test_pred.tolist(),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
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


def main() -> int:
    ap = argparse.ArgumentParser()
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--out", default=str(get_results_dir() / "gedn_variant_comparison"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ssl-epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default="simulation/data/db/epi_real_seoul.db")
    ap.add_argument("--variants", nargs="+", default=["V1", "V2", "V3"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _configure_logging(out / "run.log")

    log.info(f"[compare] args={vars(args)}")
    log.info("[compare] loading aligned real data ...")
    t0 = time.time()
    data = load_aligned_data(args.db)
    log.info(
        f"[compare] data ready: X_gu={data['X_gu'].shape}, "
        f"X_agg={data['X_agg'].shape}, "
        f"X_agg_temp={data['X_agg_temp'].shape}, "
        f"y={data['y'].shape}, "
        f"weeks=[{data['weeks'][0]} .. {data['weeks'][-1]}], "
        f"load_wall={time.time()-t0:.1f}s"
    )
    log.info(f"[compare] V4 agg-temp features: {data['feat_names_agg_temp']}")

    results = {
        "meta": {
            "epochs": args.epochs,
            "ssl_epochs": args.ssl_epochs,
            "seed": args.seed,
            "n_weeks": int(data["X_gu"].shape[0]),
            "k_gu": int(data["X_gu"].shape[2]),
            "k_agg": int(data["X_agg"].shape[1]),
            "k_agg_temp": int(data["X_agg_temp"].shape[1]),
            "feat_names_agg_temp": data["feat_names_agg_temp"],
            "gu_order": data["gu_order"],
            "first_week": data["weeks"][0],
            "last_week": data["weeks"][-1],
        },
        "variants": {},
    }

    for v in args.variants:
        log.info(f"\n{'='*60}\n[compare] running variant {v}\n{'='*60}")
        res = run_variant(
            v, data,
            epochs=args.epochs, seed=args.seed,
            ssl_epochs=args.ssl_epochs,
        )
        results["variants"][v] = res

    # Save JSON
    out_json = out / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"[compare] wrote {out_json}")

    # Save CSV with predictions (val + test rows for each variant).
    tr, va, te = _temporal_split(data["X_gu"].shape[0], 0.70, 0.15)
    weeks_va = data["weeks"][va.start:va.stop]
    weeks_te = data["weeks"][te.start:te.stop]
    import csv
    out_csv = out / "predictions.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["week", "split", "y_true"] + [f"pred_{v}" for v in args.variants]
        w.writerow(header)
        for i, wk in enumerate(weeks_va):
            row = [wk, "val", float(data["y"][va.start + i])]
            for v in args.variants:
                row.append(float(results["variants"][v]["val_pred"][i]))
            w.writerow(row)
        for i, wk in enumerate(weeks_te):
            row = [wk, "test", float(data["y"][te.start + i])]
            for v in args.variants:
                row.append(float(results["variants"][v]["test_pred"][i]))
            w.writerow(row)
    log.info(f"[compare] wrote {out_csv}")

    # Summary line
    print("\n" + "=" * 60)
    print("VARIANT COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<6} {'Test R2':>10} {'Test MAE':>10} {'Test RMSE':>10} {'Wall(s)':>10}")
    for v in args.variants:
        r = results["variants"][v]
        print(f"{v:<6} {r['test']['r2']:10.4f} {r['test']['mae']:10.3f} "
              f"{r['test']['rmse']:10.3f} {r['total_wall_s']:10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
