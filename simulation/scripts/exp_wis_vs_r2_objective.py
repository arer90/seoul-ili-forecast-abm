"""Would selecting by r2 instead of WIS/MAE avoid the bad transforms on its own? (user, 2026-06-12)

For ONE model (argv), run the FULL transform pool through walk-forward OOF and report, with NO
sanity penalty, which transform each objective would pick:
  - argmin mean |error|  (MAE — a WIS-like point objective)
  - argmax mean r2        (variance-explained; squares the residual → peak-failure-sensitive)
If r2 naturally lands on a SAFE transform where MAE picks an exploding one, that supports the
user's intuition that r2 is more self-protecting. Per-model process (macOS OMP, G-251).

Run: for m in MLP-deep Ridge ElasticNet CatBoost RandomForest; do python -m ...exp_wis_vs_r2_objective $m; done
"""
from __future__ import annotations
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, r2_score

from simulation.scripts.exp_peak_extrapolation import load_split, transform_y

TRANSFORMS = ["identity", "log1p", "sqrt", "asinh", "mcmc_robust"]
SAFE = {"identity", "mcmc_robust", "laplace"}
Xv, y, tr, te, lag_cols = load_split()
Xtr, ytr = Xv[tr], y[tr]
n = len(ytr); n_folds = 5; fs = n // (n_folds + 1)


def build_fit_fn(name):
    from simulation.scripts.exp_per_model_trial_min import build_fit_fn as _b
    return _b(name)


def oof_mae_r2(fit_fn, transform):
    maes, r2s = [], []
    for k in range(1, n_folds + 1):
        end = fs * k
        if end < 30 or end + fs > n:
            continue
        Xt, yt = Xtr[:end], ytr[:end]
        Xv2, yv = Xtr[end:end + fs], ytr[end:end + fs]
        ytt, inv = transform_y(transform, yt)
        try:
            p = np.asarray(inv(fit_fn(Xt, ytt, Xv2, None))).ravel()
        except Exception:
            continue
        maes.append(mean_absolute_error(yv, p))
        r2s.append(r2_score(yv, p) if len(yv) > 1 else float("nan"))
    if not maes:
        return float("inf"), float("-inf")
    return float(np.mean(maes)), float(np.mean(r2s))


if __name__ == "__main__":
    name = sys.argv[1]
    fit_fn = build_fit_fn(name)
    res = {t: oof_mae_r2(fit_fn, t) for t in TRANSFORMS}
    mae_pick = min(res, key=lambda t: res[t][0])
    r2_pick = max(res, key=lambda t: res[t][1])
    cells = " ".join(f"{t}:MAE{res[t][0]:.1f}/r2{max(res[t][1],-99):.2f}" for t in TRANSFORMS)
    print(f"{name:13s} │ {cells}")
    print(f"   → MAE(WIS류) 선택: {mae_pick} ({'안전' if mae_pick in SAFE else '⚠외삽'})"
          f"   |   r2 선택: {r2_pick} ({'안전' if r2_pick in SAFE else '⚠외삽'})"
          f"   |   {'동일' if mae_pick==r2_pick else 'r2가 다르게 고름'}")
