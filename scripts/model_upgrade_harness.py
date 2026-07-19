"""Standalone fast model-upgrade TDD harness (user 2026-06-13).

For each model, compare the INCUMBENT vs a SOTA CANDIDATE on the REAL Seoul ILI series under
IDENTICAL conditions (same split, fast settings) → r2 (direct + rolling) / WIS / MAE. Decision bar
(user): replace if candidate >= incumbent (prefer the newer SOTA).

STANDALONE by design — imports NO `simulation` package (so it runs unchanged in an isolated venv,
e.g. .venv_modeltest with statsforecast/tabpfn whose deps would downgrade the main env's scipy).
Reads the feature cache directly via polars. Each target runs in ONE process (OMP isolation,
G-251); a driver runs incumbent + candidate in separate procs and diffs the JSON.

Run ONE target, print JSON metrics:
    .venv/bin/python scripts/model_upgrade_harness.py <target_id> [--n-test 80]
    .venv_modeltest/bin/python scripts/model_upgrade_harness.py sf_autoarima
List targets:
    .venv/bin/python scripts/model_upgrade_harness.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

CACHE = "simulation/cache/feature_cache.parquet"


# ── data ──────────────────────────────────────────────────────────────────────
def load_series(n_test: int):
    """ILI series in time order + tabular feature matrix. Returns y, X, lag_cols, split idx."""
    df = pl.read_parquet(CACHE).sort("week_start")
    y = df["ili_rate"].to_numpy().astype(np.float64)
    n = len(y)
    # tabular features: numeric cols except target/date
    drop = {"ili_rate", "week_start"}
    feat_cols = [c for c, t in zip(df.columns, df.dtypes)
                 if c not in drop and t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    X = df.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float64)
    tr = np.arange(0, n - n_test)
    te = np.arange(n - n_test, n)
    return y, X, feat_cols, tr, te


# ── metrics ───────────────────────────────────────────────────────────────────
def _r2(yt, p):
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - float(np.sum((yt - p) ** 2)) / ss if ss > 0 else float("nan")


def _wis_point(yt, pred, sigma):
    """Sigma-scaled WIS (mirrors tree_models._fold_wis): treat pred as median, build symmetric
    Gaussian quantile intervals at the 11 FluSight levels, average WIS. Lower = better."""
    from scipy.stats import norm
    levels = np.array([0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    K = len(levels)
    sigma = max(float(sigma), 1e-3)
    wis = np.abs(yt - pred) / (K + 0.5)
    for a in levels:
        z = norm.ppf(1 - a)            # upper z for central (1-2a) interval
        lo, hi = pred - z * sigma, pred + z * sigma
        w = a
        wis = wis + (w * (hi - lo)
                     + 2 * np.maximum(lo - yt, 0)
                     + 2 * np.maximum(yt - hi, 0)) / (K + 0.5)
    return float(np.mean(wis))


def metrics(yt, pred_direct, y_train, roll=None):
    """roll = None or (yt_roll, pred_roll) — rolling 1-step subset (may be < full test)."""
    yt = np.asarray(yt, float)
    pd_ = np.asarray(pred_direct, float)
    nonpeak = yt < np.percentile(yt, 75)
    sigma = float(np.std(y_train))
    out = {
        "r2_direct": round(_r2(yt, pd_), 4),
        "r2_nonpeak": round(_r2(yt[nonpeak], pd_[nonpeak]), 4) if nonpeak.sum() > 2 else None,
        "mae_direct": round(float(np.mean(np.abs(yt - pd_))), 3),
        "wis_direct": round(_wis_point(yt, pd_, sigma), 3),
        "pred_max_direct": round(float(np.max(pd_)), 1),
        "y_test_max": round(float(np.max(yt)), 1),
    }
    if roll is not None:
        ytr, pr_ = np.asarray(roll[0], float), np.asarray(roll[1], float)
        out["r2_rolling"] = round(_r2(ytr, pr_), 4)
        out["mae_rolling"] = round(float(np.mean(np.abs(ytr - pr_))), 3)
        out["wis_rolling"] = round(_wis_point(ytr, pr_, sigma), 3)
        out["n_roll"] = int(len(ytr))
    return out


# ── univariate runners (y-series only) ────────────────────────────────────────
ROLL = True       # set False by --no-rolling
ROLL_N = 0        # >0 → only roll the LAST ROLL_N test points (faster for slow refit candidates)


def _roll_idx(te):
    return te if ROLL_N <= 0 else te[-ROLL_N:]


def _sm_arima(order, seasonal_order, y_train, y_full, te):
    """statsmodels ARIMA/SARIMAX — direct multi-step + rolling 1-step (refit-free filter)."""
    import statsmodels.api as sm
    n_test = len(te)
    m = sm.tsa.SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    direct = np.asarray(m.forecast(steps=n_test), float)
    if not ROLL:
        return direct, None
    # rolling 1-step: extend the fitted filter with true obs (no refit) → operational
    ridx = _roll_idx(te)
    res = m if ridx[0] == te[0] else m.append(y_full[te[0]:ridx[0]], refit=False)
    roll = []
    for t in ridx:
        roll.append(float(np.asarray(res.forecast(steps=1), float)[0]))
        res = res.append(y_full[t:t + 1], refit=False)
    return direct, (np.array(roll), ridx)


def _statsforecast(model_name, y_train, y_full, te, season_length=52):
    """Nixtla StatsForecast AutoARIMA/AutoETS/AutoTheta — direct + rolling 1-step (fixed order)."""
    from statsforecast import StatsForecast
    from statsforecast import models as M
    n_test = len(te)
    mk = {"AutoARIMA": lambda: M.AutoARIMA(season_length=season_length),
          "AutoETS":   lambda: M.AutoETS(season_length=season_length),
          "AutoTheta": lambda: M.AutoTheta(season_length=season_length),
          "MSTL":      lambda: M.MSTL(season_length=season_length)}[model_name]

    def _fit_forecast(train, h):
        sf = StatsForecast(models=[mk()], freq=1, n_jobs=1)
        df = pl.DataFrame({"unique_id": ["s"] * len(train), "ds": list(range(len(train))),
                           "y": list(map(float, train))}).to_pandas()
        sf.fit(df)
        fc = sf.predict(h=h)
        return np.asarray(fc[model_name].to_numpy(), float)[:h]

    direct = _fit_forecast(y_train, n_test)
    if not ROLL:
        return direct, None
    ridx = _roll_idx(te)
    roll = [float(_fit_forecast(y_full[:t], 1)[0]) for t in ridx]   # refit each step (slow but honest)
    return direct, (np.array(roll), ridx)


# ── tabular runners (X features → y) ──────────────────────────────────────────
def _sk_tabular(make, y, X, tr, te, scale=True, scale_y=False):
    """Fit a sklearn-API regressor. scale_y standardizes the target (neural nets need it — else
    raw-scale MSE blows up; trees are scale-invariant so scale_y=False)."""
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, Xte = X[tr], y[tr].copy(), X[te]
    if scale:
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    ymu, ysd = (float(ytr.mean()), float(ytr.std()) or 1.0) if scale_y else (0.0, 1.0)
    m = make().fit(Xtr, (ytr - ymu) / ysd)
    return np.asarray(m.predict(Xte), float) * ysd + ymu, None


def _tabpfn(y, X, tr, te):
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import get_cache_dir
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    # 공식 model_path 인자로 공개 repo서 받은 local .ckpt 직접 로드 (offline — 정식 지원 기능)
    ckpt = get_cache_dir() / "tabpfn-v2-regressor.ckpt"
    kw = {"device": "cpu", "ignore_pretraining_limits": True}
    if ckpt.exists():
        kw["model_path"] = str(ckpt)
    m = TabPFNRegressor(**kw).fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float), None


def _topk_idx(Xtr, ytr, k=20):
    """top-K |Pearson r| feature 선택 (NegBinGLM V6 salvage 와 동일)."""
    Xs = Xtr - Xtr.mean(0); ys = ytr - ytr.mean()
    denom = (np.sqrt((Xs ** 2).sum(0)) * np.sqrt((ys ** 2).sum()) + 1e-12)
    r = np.abs((Xs * ys[:, None]).sum(0) / denom)
    return np.argsort(r)[::-1][:k]


def _negbin_incumbent(y, X, tr, te):
    """현 NegBinGLM = V6 salvage = top-K |r| + RidgeCV on log1p(y) (epi_models.py:421 재현)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    idx = _topk_idx(Xtr, ytr, 20)
    sc = StandardScaler().fit(Xtr[:, idx])
    Xs, Xs_te = sc.transform(Xtr[:, idx]), sc.transform(Xte[:, idx])
    m = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3).fit(Xs, np.log1p(ytr))
    return np.expm1(np.asarray(m.predict(Xs_te), float)).clip(0), None


def _glum_nb(y, X, tr, te, l1_ratio=0.5, alpha=0.1):
    """진짜 Negative-Binomial GLM (glum, elastic-net) — full pool, p>n 을 L1/L2 로 통제 (log link)."""
    from glum import GeneralizedLinearRegressor
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    sc = StandardScaler().fit(Xtr); Xs, Xs_te = sc.transform(Xtr), sc.transform(Xte)
    m = GeneralizedLinearRegressor(family="negative.binomial", alpha=alpha, l1_ratio=l1_ratio,
                                   fit_intercept=True, scale_predictors=False, max_iter=300)
    m.fit(Xs, np.clip(ytr, 1e-6, None))
    return np.asarray(m.predict(Xs_te), float).clip(0), None


def _extratrees(y, X, tr, te):
    from sklearn.ensemble import ExtraTreesRegressor
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    m = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=2).fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float), None


def _cqr_catboost(y, X, tr, te):
    """CatBoost Quantile(α=0.5) median point — CQR base learner 후보 (vs CQR-LightGBM)."""
    from catboost import CatBoostRegressor
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    m = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, loss_function="Quantile:alpha=0.5",
                          random_state=42, verbose=0).fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float), None


def _cqr_lightgbm(y, X, tr, te):
    """LightGBM Quantile(α=0.5) median — 현 CQR-LightGBM base (비교 기준)."""
    import lightgbm as lgb
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    m = lgb.LGBMRegressor(n_estimators=200, objective="quantile", alpha=0.5,
                          random_state=42, verbose=-1).fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float), None


# ── target registry ───────────────────────────────────────────────────────────
def run_target(tid, y, X, lag_cols, tr, te):
    yt, ytr, yfull = y[te], y[tr], y
    # univariate incumbents (statsmodels)
    if tid == "arima":        d, r = _sm_arima((2, 1, 2), (0, 0, 0, 0), ytr, yfull, te)
    elif tid == "sarima":     d, r = _sm_arima((1, 1, 1), (1, 0, 1, 52), ytr, yfull, te)
    elif tid == "sarimax":    d, r = _sm_arima((1, 1, 1), (1, 0, 1, 52), ytr, yfull, te)  # exog 생략(공정 univariate)
    # univariate candidates (statsforecast)
    elif tid == "sf_autoarima": d, r = _statsforecast("AutoARIMA", ytr, yfull, te)
    elif tid == "sf_autoets":   d, r = _statsforecast("AutoETS", ytr, yfull, te)
    elif tid == "sf_autotheta": d, r = _statsforecast("AutoTheta", ytr, yfull, te)
    elif tid == "sf_mstl":      d, r = _statsforecast("MSTL", ytr, yfull, te)
    # tabular incumbents
    elif tid == "mlp":
        from sklearn.neural_network import MLPRegressor
        d, r = _sk_tabular(lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                                                early_stopping=True, random_state=42), y, X, tr, te, scale_y=True)
    elif tid == "lightgbm":
        import lightgbm as lgb
        d, r = _sk_tabular(lambda: lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1), y, X, tr, te, scale=False)
    # tabular candidate
    elif tid == "tabpfn":     d, r = _tabpfn(y, X, tr, te)
    # NegBinGLM: incumbent(RidgeCV-log1p salvage) vs glum 진짜 NB-GLM
    elif tid == "negbin_incumbent": d, r = _negbin_incumbent(y, X, tr, te)
    elif tid == "glum_nb":          d, r = _glum_nb(y, X, tr, te)
    elif tid == "glum_nb_l2":       d, r = _glum_nb(y, X, tr, te, l1_ratio=0.0, alpha=0.3)
    elif tid == "extratrees":       d, r = _extratrees(y, X, tr, te)
    elif tid == "cqr_catboost":     d, r = _cqr_catboost(y, X, tr, te)
    elif tid == "cqr_lightgbm_q":   d, r = _cqr_lightgbm(y, X, tr, te)
    else:
        raise SystemExit(f"unknown target: {tid}")
    # r = None (tabular/no-roll) OR (roll_pred, ridx) (univariate rolling subset)
    roll = None
    if r is not None:
        roll_pred, ridx = r
        roll = (y[ridx], roll_pred)
    return metrics(yt, d, ytr, roll=roll)


TARGETS = ["arima", "sarima", "sarimax", "sf_autoarima", "sf_autoets", "sf_autotheta", "sf_mstl",
           "mlp", "lightgbm", "tabpfn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?")
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--no-rolling", action="store_true", help="skip rolling 1-step (direct only)")
    ap.add_argument("--roll-n", type=int, default=0, help=">0 → roll only last N test pts (faster)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    global ROLL, ROLL_N
    ROLL = not args.no_rolling
    ROLL_N = args.roll_n
    if args.list or not args.target:
        print(json.dumps({"targets": TARGETS})); return
    y, X, lag_cols, tr, te = load_series(args.n_test)
    res = run_target(args.target, y, X, lag_cols, tr, te)
    print(json.dumps({"target": args.target, **res}))


if __name__ == "__main__":
    main()
