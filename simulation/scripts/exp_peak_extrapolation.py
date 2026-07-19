"""Heavy controlled experiment — peak extrapolation across model families (G-254/G-255).

Verifies the two root causes that codex + gemini independently flagged for the Seoul-ILI
phase-13 collapse (LightGBM r2=0.313, CatBoost -0.306):

  (A) Tree models cannot predict above the training target max regardless of n_estimators
      (leaf = average of train targets ⇒ ŷ ≤ max(y_train)). Non-tree models (neural, linear,
      kernel) can extrapolate. The `rank`/`gaussian`/`arcsine_sqrt` y-transforms compound the
      tree cap (empirical-CDF inverse bounded to train support).
  (B) OOF-CV selection by `np.median` over folds discards the (few) outbreak folds, so a
      peak-blind config wins selection — `np.mean` (or peak-aware) generalizes to the test peak.

Design: exact pipeline split recovered from predictions_LightGBM.csv (train max ≈ 66.9,
TEST max = 100.7). 16 models across 3 families, with a DEPTH SWEEP (trees: n_estimators
200 vs 1200; neural: epochs 80 vs 600) and a TRANSFORM sweep (rank vs identity vs log1p).

Run: .venv/bin/python -m simulation.scripts.exp_peak_extrapolation
"""
from __future__ import annotations

import warnings
import numpy as np
import polars as pl
import csv

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform

SEED = 42
np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# Data — exact pipeline split (recovered from the saved test predictions)
# ════════════════════════════════════════════════════════════════════════════
def load_split(n_test: int = 80):
    """Chronological train/test split (gemini C3 fix — replaces value-matching that could
    shuffle time order and misalign lag features). Sort by week_start, hold out the last
    ``n_test`` weeks as test. The 2024-12 epidemic peak (ILI 100.7) falls in this tail while
    the train pool maxes near ~67, giving the genuine out-of-distribution (test > train) setting
    without any value-matching ambiguity.
    """
    df = pl.read_parquet("simulation/cache/feature_cache.parquet").to_pandas()
    df = df.sort_values("week_start").reset_index(drop=True)   # TIME ORDER
    y = df["ili_rate"].values.astype(float)
    n = len(y)
    te = np.arange(n - n_test, n)
    tr = np.arange(0, n - n_test)
    drop = [c for c in df.columns if c in ("ili_rate", "week_start", "hfmd_rate")
            or c.startswith("rt_popdet") or c.startswith("rt_temp")]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])
    Xv = X.fillna(X.median()).values.astype(np.float32)
    lag_cols = [i for i, c in enumerate(X.columns) if c.startswith("ili_rate_lag")][:8]
    return Xv, y, tr, te, lag_cols


def transform_y(name, ytr):
    if name == "identity":
        return ytr.copy(), (lambda x: np.asarray(x))
    yt, inv, _ = _apply_single_y_transform(ytr.copy(), name)
    return np.asarray(yt), inv


def metrics(yte, pred, nonpeak):
    return pred.max(), r2_score(yte, pred), r2_score(yte[nonpeak], pred[nonpeak])


# ════════════════════════════════════════════════════════════════════════════
# Model families
# ════════════════════════════════════════════════════════════════════════════
def tree_factories(n_est):
    import xgboost as xgb, lightgbm as lgb
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                  HistGradientBoostingRegressor, GradientBoostingRegressor)
    from catboost import CatBoostRegressor
    return {
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=n_est, max_depth=5,
                                            learning_rate=0.05, random_state=SEED, verbosity=0),
        "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=n_est, num_leaves=31,
                                              learning_rate=0.05, random_state=SEED, verbose=-1),
        "CatBoost": lambda: CatBoostRegressor(iterations=n_est, depth=6, learning_rate=0.05,
                                              random_state=SEED, verbose=0),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=n_est, random_state=SEED, n_jobs=2),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=n_est, random_state=SEED, n_jobs=2),
        "HistGBM": lambda: HistGradientBoostingRegressor(max_iter=n_est, learning_rate=0.05,
                                                         random_state=SEED),
        "GradientBoost": lambda: GradientBoostingRegressor(n_estimators=n_est, max_depth=3,
                                                          learning_rate=0.05, random_state=SEED),
    }


def linear_factories():
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    return {
        "Ridge": lambda: Ridge(alpha=10.0, random_state=SEED),
        "ElasticNet": lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=SEED),
        "SVR-RBF": lambda: SVR(C=10.0, gamma="scale"),
        "KNN": lambda: KNeighborsRegressor(n_neighbors=7),
    }


def fit_predict_sklearn(factory, Xtr, yt, Xte, scale_x):
    if scale_x:
        xs = StandardScaler().fit(Xtr)
        Xtr, Xte = xs.transform(Xtr), xs.transform(Xte)
    m = factory(); m.fit(Xtr, np.asarray(yt).ravel())
    return np.asarray(m.predict(Xte)).ravel()


# ── Neural (PyTorch) — epochs sweep, multi-seed (gemini C3: ranking-grade) ───
def fit_predict_neural(kind, Xtr, yt, Xte, epochs, lag_cols, seeds=(42, 1, 7)):
    """Average test predictions over ``seeds`` independent inits — a single seed is fine for
    an existence proof (can the model exceed train max?) but too noisy to RANK families."""
    import torch
    preds = [_fit_predict_neural_single(kind, Xtr, yt, Xte, epochs, lag_cols, s) for s in seeds]
    return np.mean(np.vstack(preds), axis=0)


def _fit_predict_neural_single(kind, Xtr, yt, Xte, epochs, lag_cols, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    yt = np.asarray(yt, dtype=np.float32).ravel()
    ymu, ysd = float(yt.mean()), float(yt.std() + 1e-8)
    yts = (yt - ymu) / ysd

    if kind in ("LSTM-lags", "GRU-lags"):
        # recurrent over the autoregressive ILI lag history (n, seq, 1)
        Xtr_seq = Xtr[:, lag_cols][:, :, None]
        Xte_seq = Xte[:, lag_cols][:, :, None]
        sc = StandardScaler().fit(Xtr_seq.reshape(-1, 1))
        Xtr_t = torch.tensor(sc.transform(Xtr_seq.reshape(-1, 1)).reshape(Xtr_seq.shape), dtype=torch.float32, device=dev)
        Xte_t = torch.tensor(sc.transform(Xte_seq.reshape(-1, 1)).reshape(Xte_seq.shape), dtype=torch.float32, device=dev)
        rnn = (nn.LSTM if kind == "LSTM-lags" else nn.GRU)(1, 64, batch_first=True)

        class RNNReg(nn.Module):
            def __init__(self):
                super().__init__(); self.rnn = rnn; self.head = nn.Linear(64, 1)
            def forward(self, x):
                o, _ = self.rnn(x); return self.head(o[:, -1, :]).squeeze(-1)
        model = RNNReg().to(dev)
    else:
        xs = StandardScaler().fit(Xtr)
        Xtr_t = torch.tensor(xs.transform(Xtr), dtype=torch.float32, device=dev)
        Xte_t = torch.tensor(xs.transform(Xte), dtype=torch.float32, device=dev)
        d = Xtr.shape[1]
        if kind == "MLP-shallow":
            model = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        elif kind == "MLP-deep":
            model = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        elif kind == "MLP-wide":
            model = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
        elif kind == "ResMLP":
            class Res(nn.Module):
                def __init__(self):
                    super().__init__(); self.i = nn.Linear(d, 64)
                    self.b = nn.Linear(64, 64); self.o = nn.Linear(64, 1)
                def forward(self, x):
                    h = torch.relu(self.i(x)); h = h + torch.relu(self.b(h)); return self.o(h).squeeze(-1)
            model = Res().to(dev)
        model = model.to(dev)

        def fwd(x):
            return model(x).squeeze(-1)
    if kind in ("LSTM-lags", "GRU-lags"):
        def fwd(x):
            return model(x)

    Yt = torch.tensor(yts, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(); out = fwd(Xtr_t); loss = lossf(out, Yt); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = fwd(Xte_t).detach().cpu().numpy().ravel()
    return p * ysd + ymu  # back to transformed space


# ════════════════════════════════════════════════════════════════════════════
# Part A — cap / extrapolation across families × depth × transform
# ════════════════════════════════════════════════════════════════════════════
def part_a(Xv, y, tr, te, lag_cols):
    Xtr, Xte, ytr, yte = Xv[tr], Xv[te], y[tr], y[te]
    mu = yte.mean(); nonpeak = np.argsort(np.abs(yte - mu))[:-5]
    train_max = ytr.max()
    print(f"\n{'='*92}\nPART A — peak 외삽 능력 (train max={train_max:.1f}, TEST max={yte.max():.1f}, 평균={mu:.1f})")
    print(f"{'='*92}")
    print(f"{'model':14s}{'family':8s}{'depth':>12s} │ {'rank 천장/r2':>16s} │ {'identity 천장/r2':>18s} │ {'log1p 천장/r2':>16s}")
    print("─" * 92)

    rows = []

    def row(name, fam, depth_label, run):
        cells = {}
        for tf in ("rank", "identity", "log1p"):
            yt, inv = transform_y(tf, ytr)
            try:
                pred = np.asarray(inv(run(yt))).ravel()
                ceil, r2, npr2 = metrics(yte, pred, nonpeak)
                cells[tf] = (ceil, r2, npr2)
            except Exception as e:
                cells[tf] = (float("nan"), float("nan"), float("nan"))
        rows.append((name, fam, depth_label, cells))
        def f(tf): return f"{cells[tf][0]:5.0f}/{cells[tf][1]:+.2f}"
        print(f"{name:14s}{fam:8s}{depth_label:>12s} │ {f('rank'):>16s} │ {f('identity'):>18s} │ {f('log1p'):>16s}")

    # Trees — n_estimators sweep
    for n_est, lbl in [(200, "n_est=200"), (1200, "n_est=1200")]:
        for name, fac in tree_factories(n_est).items():
            row(name, "tree", lbl, lambda yt, fac=fac: fit_predict_sklearn(fac, Xtr, yt, Xte, False))
    print("─" * 92)
    # Neural — epochs sweep
    for ep, lbl in [(80, "ep=80"), (600, "ep=600")]:
        for kind in ("MLP-shallow", "MLP-deep", "MLP-wide", "ResMLP", "LSTM-lags"):
            row(kind, "neural", lbl, lambda yt, k=kind, e=ep: fit_predict_neural(k, Xtr, yt, Xte, e, lag_cols))
    print("─" * 92)
    # Linear / kernel — single (no depth knob)
    for name, fac in linear_factories().items():
        row(name, "linear", "—", lambda yt, fac=fac: fit_predict_sklearn(fac, Xtr, yt, Xte, True))

    return rows, train_max


# ════════════════════════════════════════════════════════════════════════════
# Part B — OOF aggregation: median vs mean selection
# ════════════════════════════════════════════════════════════════════════════
def walk_forward_oof(factory_kind, factory, transform, Xtr, ytr, lag_cols, n_folds=5, scale_x=False):
    """Return per-fold val WIS-proxy (here: MAE in original space) for one config."""
    n = len(ytr); fold = n // (n_folds + 1)
    scores, fold_max = [], []
    for k in range(1, n_folds + 1):
        end = fold * k
        if end < 20 or end + fold > n:
            continue
        Xt, yt2 = Xtr[:end], ytr[:end]
        Xv, yv = Xtr[end:end + fold], ytr[end:end + fold]
        if len(yv) < 4:
            continue
        ytt, inv = transform_y(transform, yt2)
        try:
            if factory_kind == "neural":
                p = fit_predict_neural(factory, Xt, ytt, Xv, 300, lag_cols)
            else:
                p = fit_predict_sklearn(factory, Xt, ytt, Xv, scale_x)
            pred = np.asarray(inv(p)).ravel()
            scores.append(float(np.mean(np.abs(yv - pred))))
            fold_max.append(float(yv.max()))
        except Exception:
            continue
    return scores, fold_max


def part_b(Xv, y, tr, te, lag_cols):
    import lightgbm as lgb, xgboost as xgb
    from catboost import CatBoostRegressor
    Xtr, Xte, ytr, yte = Xv[tr], Xv[te], y[tr], y[te]
    mu = yte.mean(); nonpeak = np.argsort(np.abs(yte - mu))[:-5]
    print(f"\n{'='*92}\nPART B — OOF 집계: median(현재) vs mean(제안) 중 어느 게 test peak에 일반화?")
    print(f"{'='*92}")
    models = {
        "LightGBM": ("tree", lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=SEED, verbose=-1), False),
        "CatBoost": ("tree", lambda: CatBoostRegressor(iterations=400, learning_rate=0.05, random_state=SEED, verbose=0), False),
        "XGBoost": ("tree", lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, random_state=SEED, verbosity=0), False),
        "MLP-deep": ("neural", "MLP-deep", False),
    }
    transforms = ("rank", "log1p", "identity")
    print(f"{'model':12s} │ {'OOF-median 선택→test r2':>26s} │ {'OOF-mean 선택→test r2':>24s} │ 일치?")
    print("─" * 92)
    summary = []
    for name, spec in models.items():
        fam = spec[0]
        oof = {}
        for tf in transforms:
            if fam == "neural":
                sc, fmax = walk_forward_oof("neural", spec[1], tf, Xtr, ytr, lag_cols)
            else:
                sc, fmax = walk_forward_oof("tree", spec[1], tf, Xtr, ytr, lag_cols, scale_x=spec[2])
            if sc:
                oof[tf] = (float(np.median(sc)), float(np.mean(sc)))
        # test r2 per transform (full refit)
        test_r2 = {}
        for tf in transforms:
            yt, inv = transform_y(tf, ytr)
            if fam == "neural":
                p = fit_predict_neural(spec[1], Xtr, yt, Xte, 600, lag_cols)
            else:
                p = fit_predict_sklearn(spec[1], Xtr, yt, Xte, spec[2])
            pred = np.asarray(inv(p)).ravel()
            test_r2[tf] = r2_score(yte, pred)
        med_pick = min(oof, key=lambda t: oof[t][0])
        mean_pick = min(oof, key=lambda t: oof[t][1])
        agree = "✓" if med_pick == mean_pick else "✗"
        print(f"{name:12s} │ {med_pick+' → '+f'{test_r2[med_pick]:+.3f}':>26s} │ "
              f"{mean_pick+' → '+f'{test_r2[mean_pick]:+.3f}':>24s} │ {agree}")
        summary.append((name, med_pick, test_r2[med_pick], mean_pick, test_r2[mean_pick]))
    return summary


def main():
    Xv, y, tr, te, lag_cols = load_split()
    rows, train_max = part_a(Xv, y, tr, te, lag_cols)

    # ── Part A 집계 결론 ──
    print(f"\n{'─'*92}\n[Part A 결론]")
    def fam_ceiling(fam, depth_filter=None):
        vals = [c["identity"][0] for nm, f, dl, c in rows if f == fam
                and (depth_filter is None or depth_filter in dl) and np.isfinite(c["identity"][0])]
        return np.nanmean(vals) if vals else float("nan")
    tree_lo = fam_ceiling("tree", "200"); tree_hi = fam_ceiling("tree", "1200")
    neural_lo = fam_ceiling("neural", "ep=80"); neural_hi = fam_ceiling("neural", "ep=600")
    lin = fam_ceiling("linear")
    print(f"  트리 identity 평균천장: n_est=200 → {tree_lo:.1f} | n_est=1200 → {tree_hi:.1f}  (train max={train_max:.1f})")
    print(f"    → n_est 무관 cap? Δ={abs(tree_hi-tree_lo):.1f} (작으면 구조적 cap 확인)")
    print(f"  신경망 identity 평균천장: ep=80 → {neural_lo:.1f} | ep=600 → {neural_hi:.1f}")
    print(f"    → epoch로 외삽 상승? Δ={neural_hi-neural_lo:+.1f}")
    print(f"  선형/커널 identity 평균천장: {lin:.1f}  (>train max면 외삽 가능)")
    # rank vs identity 전반
    better = sum(1 for nm, f, dl, c in rows
                 if np.isfinite(c["rank"][1]) and np.isfinite(c["identity"][1]) and c["identity"][1] > c["rank"][1])
    tot = sum(1 for nm, f, dl, c in rows if np.isfinite(c["rank"][1]) and np.isfinite(c["identity"][1]))
    print(f"  identity > rank (test r2): {better}/{tot} 케이스")

    summary = part_b(Xv, y, tr, te, lag_cols)
    print(f"\n{'─'*92}\n[Part B 결론]")
    mean_wins = sum(1 for n, mp, mr, ep, er in summary if er > mr + 1e-9)
    print(f"  OOF-mean 선택이 OOF-median 선택보다 test r2 높음: {mean_wins}/{len(summary)} 모델")
    print(f"{'='*92}\n[실험 끝]")


if __name__ == "__main__":
    main()
