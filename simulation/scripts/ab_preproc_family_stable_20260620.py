#!/usr/bin/env python3
"""A/B: per-family preproc Y/X stable set (Q1 워크플로, 2026-06-20).

전 가족 대표 모델에 Y∈{identity,laplace,mcmc_robust} × X∈{none,individual(standard),group}
hold-out(test 68주) R² + in-range(y≤67) R² 측정. BASIC feature(lag+seasonal 13개) 사용,
authoritative split(run_data n=337 → train 242 | val 27 | test 68).

read-only: 라이브 코드 import 만, 무수정. .venv/bin/python 로 직접 실행.
"""
from __future__ import annotations
import os, sys, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from simulation.pipeline.preproc_optuna_hierarchical import (
    _apply_single_y_transform, _build_single_x_scaler,
    _categorize_feature_groups, data_driven_group_scalers,
)
from sklearn.compose import ColumnTransformer

BASIC = [
    "ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag4", "ili_rate_lag52",
    "sin_month", "cos_month",
    "fourier_sin_h1", "fourier_cos_h1", "fourier_sin_h2", "fourier_cos_h2",
    "fourier_sin_h3", "fourier_cos_h3", "season_idx",
]


def load_split():
    import polars as pl
    df = pl.read_parquet("simulation/cache/feature_cache.parquet")
    n = min(len(df), 337)          # authoritative n=337 (HWP §3)
    df = df.head(n)
    cols = [c for c in BASIC if c in df.columns]
    X = df.select(cols).to_numpy().astype(np.float64)
    y = df["ili_rate"].to_numpy().astype(np.float64)
    # nan→0 (lag52 head)
    X = np.nan_to_num(X, nan=0.0)
    # split: test=68 (ceil), val=27, train=rest
    n_test = 68
    n_val = 27
    n_train = n - n_test - n_val
    Xtr, ytr = X[:n_train], y[:n_train]
    Xv, yv = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    Xte, yte = X[n_train + n_val:], y[n_train + n_val:]
    # train pool = train+val (phase13 OOF uses pool); refit on pool, eval on test
    Xpool = np.vstack([Xtr, Xv])
    ypool = np.concatenate([ytr, yv])
    return Xpool, ypool, Xte, yte, cols


def r2(yt, yp):
    yt = np.asarray(yt); yp = np.asarray(yp)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def apply_x(Xtr, Xte, mode, cols):
    if mode == "none":
        return Xtr.copy(), Xte.copy()
    if mode == "individual":
        sc = _build_single_x_scaler("standard")
        return sc.fit_transform(Xtr), sc.transform(Xte)
    if mode == "group":
        groups = _categorize_feature_groups(cols)
        ddmap = data_driven_group_scalers(Xtr, groups)
        transformers = []
        for g in sorted(groups):
            transformers.append((f"grp_{g}", _build_single_x_scaler(ddmap.get(g, "standard")), groups[g]))
        ct = ColumnTransformer(transformers, remainder="passthrough")
        return ct.fit_transform(Xtr), ct.transform(Xte)
    raise ValueError(mode)


def apply_y(ytr, name):
    if name == "identity":
        return ytr.copy(), (lambda x: np.asarray(x)), {}
    yt, inv, st = _apply_single_y_transform(ytr, name)
    return yt, inv, st


def make_models():
    """family → callable factory (fresh estimator each call)."""
    from sklearn.linear_model import ElasticNet, BayesianRidge
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    try:
        from lightgbm import LGBMRegressor
        tree = ("tree/LightGBM", lambda: LGBMRegressor(n_estimators=200, num_leaves=15,
                                                       learning_rate=0.05, min_child_samples=10,
                                                       verbose=-1, n_jobs=1))
    except Exception:
        tree = ("tree/RandomForest", lambda: RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=42))
    from sklearn.neural_network import MLPRegressor
    return [
        ("linear/ElasticNet", lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ("linear/BayesianRidge", lambda: BayesianRidge()),
        ("kernel/KRR", lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01)),
        ("kernel/SVR-RBF", lambda: SVR(C=10.0, gamma=0.01, epsilon=0.1)),
        tree,
        ("dl-tabular/MLP", lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                                                early_stopping=True, random_state=42)),
    ]


def main():
    Xpool, ypool, Xte, yte, cols = load_split()
    ymax_tr = float(np.max(ypool))
    in_range = yte <= 67.0
    print(f"# split: pool={len(ypool)} test={len(yte)} | train+val max y={ymax_tr:.1f} "
          f"test peak={float(np.max(yte)):.1f} | in-range(y<=67)={int(in_range.sum())}/{len(yte)}")
    print(f"# features({len(cols)}): {cols}")
    print()

    Y_CHOICES = ["identity", "laplace", "mcmc_robust"]
    X_CHOICES = ["none", "individual", "group"]

    results = {}
    for fam_name, factory in make_models():
        results[fam_name] = {}
        for ym in Y_CHOICES:
            for xm in X_CHOICES:
                try:
                    ytr_t, inv_y, _ = apply_y(ypool, ym)
                    Xtr_s, Xte_s = apply_x(Xpool, Xte, xm, cols)
                    m = factory()
                    m.fit(Xtr_s, ytr_t)
                    yp_t = m.predict(Xte_s)
                    yp = np.asarray(inv_y(yp_t)).ravel()
                    yp = np.where(np.isfinite(yp), yp, float(np.median(ypool)))
                    full = r2(yte, yp)
                    inr = r2(yte[in_range], yp[in_range])
                    pmax = float(np.max(yp))
                    results[fam_name][f"{ym}|{xm}"] = (round(full, 3), round(inr, 3), round(pmax, 1))
                except Exception as e:
                    results[fam_name][f"{ym}|{xm}"] = ("ERR", str(e)[:40], 0)

    # print as family tables
    for fam_name, cells in results.items():
        print(f"=== {fam_name} ===")
        print(f"{'Y\\X':<12}" + "".join(f"{x:>26}" for x in X_CHOICES))
        for ym in Y_CHOICES:
            row = f"{ym:<12}"
            for xm in X_CHOICES:
                v = cells[f"{ym}|{xm}"]
                if v[0] == "ERR":
                    row += f"{'ERR':>26}"
                else:
                    row += f"{f'full{v[0]:+.2f} in{v[1]:+.2f} pk{v[2]:.0f}':>26}"
            print(row)
        print()

    out = "simulation/cache/ab_preproc_family_stable_20260620.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"# saved {out}")


if __name__ == "__main__":
    main()
