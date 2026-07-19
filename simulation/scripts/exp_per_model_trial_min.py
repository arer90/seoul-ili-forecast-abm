"""Fast per-model OOF "trial minimum" scan (user request, 2026-06-12).

For each model, run the FULL y-transform pool (identity + VST + linear-inverse) through the
SAME selection objective the pipeline uses — regime-conditional OOF mean WITH the G-256c sanity
penalty — and report the minimum-OOF transform (= what Optuna would pick) per model. The point:
verify that with the penalty (and NO hard pool restriction), every model's argmin-OOF lands on a
SAFE (non-exploding) transform on its own.

Fast budget (user): trees n_estimators=20, neural epochs=10. Tiny epochs make the neural nets
under-trained and prone to blow up — a deliberate stress test of the penalty.

Run: .venv/bin/python -m simulation.scripts.exp_per_model_trial_min
"""
from __future__ import annotations
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error

from simulation.scripts.exp_peak_extrapolation import (
    load_split, transform_y, fit_predict_sklearn, _fit_predict_neural_single,
)
from simulation.pipeline.per_model_optimize import _sanity_penalize_wis
from simulation.pipeline._inline_optuna_3stage import _aggregate_oof_folds

N_EST = 20
EPOCHS = 10
TRANSFORMS = ["identity", "log1p", "sqrt", "asinh", "mcmc_robust"]
SAFE = {"identity", "mcmc_robust", "laplace"}   # linear-inverse / passthrough (never explode)

Xv, y, tr, te, lag_cols = load_split()
Xtr, ytr = Xv[tr], y[tr]
n = len(ytr); n_folds = 5; fs = n // (n_folds + 1)
outbreak_level = float(np.percentile(ytr, 75))


def oof_for(fit_fn, scale_x, transform):
    """Walk-forward OOF: per-fold MAE + G-256c sanity penalty, aggregated regime-conditional."""
    scores, fmaxes = [], []
    for k in range(1, n_folds + 1):
        end = fs * k
        if end < 30 or end + fs > n:
            continue
        Xt, yt = Xtr[:end], ytr[:end]
        Xv2, yv = Xtr[end:end + fs], ytr[end:end + fs]
        ytt, inv = transform_y(transform, yt)
        try:
            p = np.asarray(inv(fit_fn(Xt, ytt, Xv2, scale_x))).ravel()
        except Exception:
            continue
        mae = mean_absolute_error(yv, p)
        scores.append(_sanity_penalize_wis(mae, p, float(yt.max())))   # penalty here
        fmaxes.append(float(yv.max()))
    if not scores:
        return float("inf")
    return _aggregate_oof_folds(scores, fmaxes, outbreak_level)


# (name, kind, fit_fn) — fit_fn(Xt, yt, Xv, scale_x) returns transformed-space prediction
def sk(factory, scale):
    return lambda Xt, yt, Xv2, _s: fit_predict_sklearn(factory, Xt, yt, Xv2, scale)


def nn(kind):
    return lambda Xt, yt, Xv2, _s: _fit_predict_neural_single(kind, Xt, yt, Xv2, EPOCHS, lag_cols, 42)


MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "HistGBM",
               "MLP-deep", "LSTM-lags", "Ridge", "ElasticNet", "KNN"]


def build_fit_fn(name):
    """Lazily build ONE model's fit_fn — importing ONLY its library, so XGBoost and LightGBM
    never co-load libomp in the same process (macOS OMP #179 / segfault, G-251)."""
    if name == "XGBoost":
        import xgboost as xgb
        return sk(lambda: xgb.XGBRegressor(n_estimators=N_EST, max_depth=5, learning_rate=0.1,
                                           random_state=42, verbosity=0), False)
    if name == "LightGBM":
        import lightgbm as lgb
        return sk(lambda: lgb.LGBMRegressor(n_estimators=N_EST, num_leaves=31, learning_rate=0.1,
                                            random_state=42, verbose=-1), False)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return sk(lambda: CatBoostRegressor(iterations=N_EST, depth=6, learning_rate=0.1,
                                            random_state=42, verbose=0), False)
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return sk(lambda: RandomForestRegressor(n_estimators=N_EST, random_state=42, n_jobs=2), False)
    if name == "HistGBM":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return sk(lambda: HistGradientBoostingRegressor(max_iter=N_EST, learning_rate=0.1,
                                                        random_state=42), False)
    if name in ("MLP-deep", "LSTM-lags"):
        return nn(name)
    from simulation.scripts.exp_peak_extrapolation import linear_factories
    return sk(linear_factories()[name], True)


def model_row(name) -> str:
    """One model's OOF across transforms + the argmin (selected) transform."""
    fit_fn = build_fit_fn(name)
    oof = {t: oof_for(fit_fn, None, t) for t in TRANSFORMS}
    pick = min(oof, key=lambda t: oof[t])
    safe = pick in SAFE or oof[pick] < 1e3
    cells = " ".join((f"{oof[t]:>11.1f}" if oof[t] < 1e4 else f"{'EXPLODE':>11s}") for t in TRANSFORMS)
    return f"{name:13s} │ {cells} │  {pick:11s} {'✓' if safe else '✗ 폭주선택'}\tSAFE={int(safe)}"


def _header() -> str:
    return (f"per-model trial minimum (n_est={N_EST}, epochs={EPOCHS}) — OOF = regime-mean + sanity penalty\n"
            f"train pool n={n}, outbreak_level(75pct)={outbreak_level:.1f}, sanity cap=3×fold-train-max\n"
            f"{'model':13s} │ " + " ".join(f"{t:>11s}" for t in TRANSFORMS) + " │  선택(min)   안전?")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--header":
        print(_header()); print("─" * 100)
    elif arg in MODEL_NAMES:
        print(model_row(arg), flush=True)
    else:
        print(_header()); print("─" * 100)
        for nm in MODEL_NAMES:
            print(model_row(nm), flush=True)
