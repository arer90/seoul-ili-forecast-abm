"""Why does the OOF objective pick log1p over identity (which is in the search space)?
And would an objective fix (sanity penalty on exploded predictions) make Optuna pick identity
itself — so the G-256 hard restriction wouldn't be needed?

Per-fold walk-forward OOF on the TRAIN POOL ONLY (no test), MLP-deep, log1p vs identity.
For each fold report MAE and the max prediction vs the fold's train-so-far max — that tells us
(a) whether the OOF rewards log1p, and (b) whether log1p actually blows up *inside* the OOF
(if it does, a sanity penalty can catch it; if it doesn't, only a hard restriction can).
"""
from __future__ import annotations
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error

from simulation.scripts.exp_peak_extrapolation import load_split, _fit_predict_neural_single, transform_y

Xv, y, tr, te, lag_cols = load_split()
Xtr, ytr = Xv[tr], y[tr]   # train pool only
n = len(ytr); n_folds = 5; fs = n // (n_folds + 1)
SANITY_MULT = 3.0   # a prediction above SANITY_MULT × train-so-far max = "exploded"

print(f"train pool n={n}, walk-forward {n_folds} folds, MLP-deep ep=400")
print(f"{'fold':5s}{'train_max':>10s}{'val_max':>9s} │ {'identity MAE/maxpred':>22s} │ {'log1p MAE/maxpred':>22s} │ log1p 폭주?")
print("─" * 100)

agg = {"identity": [], "log1p": []}
agg_sane = {"identity": [], "log1p": []}
for k in range(1, n_folds + 1):
    end = fs * k
    if end < 30 or end + fs > n:
        continue
    Xt, yt = Xtr[:end], ytr[:end]
    Xv2, yv = Xtr[end:end + fs], ytr[end:end + fs]
    tmax = yt.max()
    row = {}
    for tf in ("identity", "log1p"):
        ytt, inv = transform_y(tf, yt)
        p = np.asarray(inv(_fit_predict_neural_single("MLP-deep", Xt, ytt, Xv2, 400, lag_cols, 42))).ravel()
        mae = mean_absolute_error(yv, p)
        row[tf] = (mae, p.max())
        agg[tf].append(mae)
        # sanity-penalized score: if any prediction exploded past SANITY_MULT×train max → big penalty
        exploded = p.max() > SANITY_MULT * tmax
        agg_sane[tf].append(mae + (1e4 if exploded else 0.0))
    blow = "★ 폭주" if row["log1p"][1] > SANITY_MULT * tmax else "-"
    print(f"{k:<5d}{tmax:>10.1f}{yv.max():>9.1f} │ {row['identity'][0]:>10.2f} /{row['identity'][1]:>9.1f} │ "
          f"{row['log1p'][0]:>10.2f} /{row['log1p'][1]:>9.1f} │ {blow}")

print("─" * 100)
mi, ml = float(np.mean(agg["identity"])), float(np.mean(agg["log1p"]))
print(f"[plain OOF mean MAE]   identity={mi:.2f}  log1p={ml:.2f}  → Optuna picks: {'identity' if mi<ml else 'log1p'}")
si, sl = float(np.mean(agg_sane["identity"])), float(np.mean(agg_sane["log1p"]))
print(f"[sanity-penalized]     identity={si:.2f}  log1p={sl:.2f}  → Optuna picks: {'identity' if si<sl else 'log1p'}")
print()
print("해석: plain OOF가 log1p를 고르면 = 목적함수가 폭주를 못 봄(제한 필요).")
print("      sanity 페널티가 identity로 뒤집으면 = 목적함수 수정으로 제한 대체 가능.")
print("      단 log1p가 OOF fold서 '폭주' 안 하면(★ 없음) sanity도 못 잡음 → 제한만이 유일 보장.")
