#!/usr/bin/env python
"""Overnight campaign — SOTA time-series conformal methods vs the fair baseline, leak-free
over the SAME 132 rolling 1-step origins (Seoul ILI weeks 205..336). Goal: beat TiRex+CQR
(WIS 2.4012, PICP95 0.9545) by a DM-significant WIS margin AND land PICP95 in [0.93,0.96].

Methods (all leak-free, past-only at every step; full FLUSIGHT quantile bounds for WIS):
  CQR_static        : reference fair baseline (TiRex point + expanding-window CQR offsets)  -> must ~2.40/0.95
  SPCI              : Xu&Xie 2023 — conditional residual quantiles from LAGGED residuals via
                      HistGBM-quantile (refit periodically), predicts q̂_τ(r_t | recent r's). Sharper on peaks.
  DtACI             : Gibbs&Candès 2022 — per-alpha online adaptive miscoverage with expert-γ
                      aggregation (pinball-weighted). Targets nominal robustly (calibration fixer).
  NexCP             : exponentially-weighted (recency) conformal quantile of |residual|.
Each is run on TWO base points: TiRex-alone, and TiRex+bagged-GBM-median (the boosted point).

No live/pipeline code touched. Deterministic. Prints a table + DM p vs the CQR_static base.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import load_split, build_features, predict_qy, MIN_CTX, K_CAL, PEAK_Y, FQ, MED_COL
from scripts.dec_boosted_mech_multiorigin import CONFIGS, fit_gbm, bagged_qy
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

ALPHAS = list(FLUSIGHT_ALPHAS)
FQL = [round(float(q), 4) for q in FQ]
T0 = 205
REFIT_K = 6
SPCI_LAGS = 8      # window of lagged residuals as SPCI features


def cp(k, nn, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, nn - k + 1)
    hi = 1.0 if k == nn else stats.beta.ppf(1 - a / 2, k + 1, nn - k)
    return round(float(lo), 3), round(float(hi), 3)


def wis_arr(bounds, y, med):
    return np.asarray(wis_from_bounds(y, bounds, ALPHAS, median=med), dtype=float)


def dm_pvalue(wis_a, wis_b):
    """Diebold-Mariano (HLN small-sample, h=1) on paired per-origin WIS. <0 => a better."""
    d = wis_a - wis_b
    n = len(d); dbar = d.mean()
    if np.allclose(d, 0):
        return 1.0, 0.0
    # h=1 -> just the variance; HLN correction factor sqrt((n+1)/n)
    var = np.var(d, ddof=1) / n
    if var <= 0:
        return 1.0, dbar
    dm = dbar / np.sqrt(var)
    dm_hln = dm * np.sqrt((n + 1) / n)
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(p), float(dbar)


# ───────────────────────── base setup ─────────────────────────
def setup():
    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test); ntot = ntr + nte
    frozen = np.asarray(json.loads((ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
                        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE); tirex_pool = d["tirex_pool"]
    y_full = np.concatenate([y_train, y_test]); cap = 2.0 * float(y_full.max())
    tirex_full = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    feat_full, foi_lag = build_features(y_train, y_test, X_train, X_test, tirex_full)
    return dict(y=y_full, tirex=tirex_full, feat=feat_full, cap=cap, ntot=ntot,
                ntr=ntr, tirex_pool=tirex_pool, frozen=frozen)


def boosted_median(S):
    """TiRex + bagged-GBM-median point over ALL weeks (per-origin refit, past-only)."""
    y, tirex, feat, cap, ntot = S["y"], S["tirex"], S["feat"], S["cap"], S["ntot"]
    r = y - tirex
    med = np.full(ntot, np.nan)
    for bstart in range(T0, ntot, REFIT_K):
        bend = min(bstart + REFIT_K, ntot)
        train_end = bstart - K_CAL
        tr = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat[tr], r[tr], cfg) for cfg in CONFIGS.values()]
        oi = np.arange(bstart, bend)
        qy = bagged_qy(gbm, feat[oi], tirex[oi], cap)
        med[oi] = qy[:, MED_COL]
    return med


# ───────────────────────── conformal methods ─────────────────────────
def m_cqr_static(point_full, resid_full, S, origins):
    """Expanding-window CQR: per-alpha half-width = past (1-a) quantile of |resid|."""
    cap = S["cap"]; n = len(origins)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    for j, t in enumerate(origins):
        past = np.abs(resid_full[MIN_CTX:t]); past = past[np.isfinite(past)]
        pt = point_full[t]
        for a in ALPHAS:
            q = np.quantile(past, 1 - a) if len(past) >= 5 else (past.max() if len(past) else 0.0)
            B[a][0][j] = max(0.0, pt - q); B[a][1][j] = min(cap, pt + q)
    return B


def m_nexcp(point_full, resid_full, S, origins, rho=0.99):
    """NexCP: exponentially-weighted (recency) conformal quantile of |resid|."""
    cap = S["cap"]; n = len(origins)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    for j, t in enumerate(origins):
        past = np.abs(resid_full[MIN_CTX:t]); past = past[np.isfinite(past)]
        pt = point_full[t]
        if len(past) < 5:
            for a in ALPHAS:
                B[a][0][j] = max(0.0, pt); B[a][1][j] = min(cap, pt)
            continue
        w = rho ** np.arange(len(past))[::-1]; w = w / w.sum()
        order = np.argsort(past); ps = past[order]; ws = w[order]; cw = np.cumsum(ws)
        for a in ALPHAS:
            idx = np.searchsorted(cw, 1 - a); idx = min(idx, len(ps) - 1)
            q = ps[idx]
            B[a][0][j] = max(0.0, pt - q); B[a][1][j] = min(cap, pt + q)
    return B


def m_dtaci(point_full, resid_full, S, origins, gammas=(0.001, 0.004, 0.016, 0.064, 0.256), eta=2.0):
    """DtACI (Gibbs&Candès 2022): per-alpha online adaptive miscoverage w/ expert aggregation."""
    cap = S["cap"]; n = len(origins)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    for a in ALPHAS:
        K = len(gammas)
        alpha_exp = np.full(K, a); w = np.ones(K) / K
        for j, t in enumerate(origins):
            past = np.abs(resid_full[MIN_CTX:t]); past = past[np.isfinite(past)]
            pt = point_full[t]
            if len(past) < 5:
                B[a][0][j] = max(0.0, pt); B[a][1][j] = min(cap, pt); continue
            # each expert radius = (1-alpha_exp) empirical quantile of past |resid|
            lv = np.clip(1 - alpha_exp, 0.0, 1.0)
            radii = np.quantile(past, lv)
            agg_alpha = float(np.clip(np.dot(w, alpha_exp), 1e-4, 0.999))
            radius = float(np.quantile(past, np.clip(1 - agg_alpha, 0, 1)))
            B[a][0][j] = max(0.0, pt - radius); B[a][1][j] = min(cap, pt + radius)
            # observe realized score, update experts
            s_t = abs(resid_full[t]) if np.isfinite(resid_full[t]) else 0.0
            err = (radii < s_t).astype(float)            # 1 = miss (score outside radius)
            pin = np.where(s_t > radii, a * (s_t - radii), (1 - a) * (radii - s_t))
            w = w * np.exp(-eta * pin); w = w / w.sum() if w.sum() > 0 else np.ones(K) / K
            alpha_exp = np.clip(alpha_exp + np.array(gammas) * (a - err), 1e-4, 0.999)
    return B


def m_spci(point_full, resid_full, S, origins):
    """SPCI (Xu&Xie 2023): conditional residual quantiles from LAGGED residuals via
    HistGBM-quantile (refit periodically). q̂_τ(r_t | r_{t-1..t-L}, seasonal)."""
    cap = S["cap"]; feat = S["feat"]; n = len(origins); ntot = S["ntot"]
    # build lagged-residual feature matrix for every week (past-only): [r_{t-1..t-L}, sin/cos week]
    Lr = np.full((ntot, SPCI_LAGS), np.nan)
    for lag in range(1, SPCI_LAGS + 1):
        Lr[lag:, lag - 1] = resid_full[:-lag]
    woy = (np.arange(ntot) % 52) / 52.0
    seas = np.column_stack([np.sin(2 * np.pi * woy), np.cos(2 * np.pi * woy)])
    Xr = np.column_stack([Lr, seas])
    cfg = dict(lr=0.05, it=300, leaves=8, msl=15, depth=3)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    models = None
    for j, t in enumerate(origins):
        if (t - T0) % REFIT_K == 0:
            tr = np.arange(MIN_CTX + SPCI_LAGS, t)                 # past pairs only
            tr = tr[np.isfinite(resid_full[tr]) & np.isfinite(Xr[tr]).all(1)]
            models = {}
            for q in FQL:
                m = HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=cfg["lr"],
                    max_iter=cfg["it"], max_leaf_nodes=cfg["leaves"], min_samples_leaf=cfg["msl"],
                    max_depth=cfg["depth"], l2_regularization=1.0, random_state=42)
                m.fit(Xr[tr], resid_full[tr]); models[q] = m
        xr = Xr[t:t + 1]
        pt = point_full[t]
        qhat = {q: float(models[q].predict(xr)[0]) for q in FQL}
        for a in ALPHAS:
            ql = round(a / 2.0, 4); qh = round(1 - a / 2.0, 4)
            lo = pt + min(qhat[ql], qhat[qh]); hi = pt + max(qhat[ql], qhat[qh])
            B[a][0][j] = float(np.clip(lo, 0.0, cap)); B[a][1][j] = float(np.clip(hi, 0.0, cap))
    return B


def median_from(point_full, origins):
    return point_full[origins]


def main():
    S = setup(); ntot = S["ntot"]
    origins = np.arange(T0, ntot); n = len(origins)
    y_seq = S["y"][origins]
    peak = y_seq >= PEAK_Y
    last34 = np.zeros(n, bool); last34[n - 34:] = True

    med_boost = boosted_median(S)
    r_tirex = S["y"] - S["tirex"]
    r_boost = S["y"] - med_boost

    bases = {"TiRex": (S["tirex"], r_tirex), "TiRex+GBM": (med_boost, r_boost)}
    methods = {"CQR_static": m_cqr_static, "NexCP": m_nexcp, "DtACI": m_dtaci, "SPCI": m_spci}

    # reference = fair baseline: TiRex + CQR_static
    ref_B = m_cqr_static(S["tirex"], r_tirex, S, origins)
    ref_med = median_from(S["tirex"], origins)
    ref_wis = wis_arr(ref_B, y_seq, ref_med)
    print(f"=== SOTA conformal vs fair baseline (TiRex+CQR), {n} leak-free origins (weeks {T0}..{ntot-1}) ===")
    print(f"    reference TiRex+CQR: WIS={ref_wis.mean():.4f}  (target: beat by DM p<0.05, PICP95 in [0.93,0.96])\n")
    hdr = f"{'base':>10s} {'method':>11s} | {'WIS':>7s} {'DMp':>7s} {'PICP95':>7s} {'k/N':>7s} {'CP95ci':>14s} {'W95':>6s} | {'peakP95':>7s} {'l34WIS':>7s}"
    print(hdr); print("-" * len(hdr))
    results = {"n": int(n), "ref_wis": round(float(ref_wis.mean()), 4), "rows": []}
    for bname, (pt, res) in bases.items():
        med = median_from(pt, origins)
        for mname, fn in methods.items():
            B = fn(pt, res, S, origins)
            w = wis_arr(B, y_seq, med)
            lo95, hi95 = B[0.05]; covv = (y_seq >= lo95) & (y_seq <= hi95); k = int(covv.sum())
            p, dbar = dm_pvalue(w, ref_wis)   # dbar<0 & p<0.05 => significantly better than ref
            row = {"base": bname, "method": mname, "wis": round(float(w.mean()), 4),
                   "dm_p_vs_ref": round(p, 4), "wis_delta_vs_ref": round(float(w.mean() - ref_wis.mean()), 4),
                   "picp95": round(k / n, 4), "k_of_n": f"{k}/{n}", "cp95ci": list(cp(k, n)),
                   "w95": round(float((hi95 - lo95).mean()), 2),
                   "peak_picp95": round(float(covv[peak].mean()), 3),
                   "last34_wis": round(float(w[last34].mean()), 4)}
            results["rows"].append(row)
            sig = "*" if (row["wis"] < ref_wis.mean() and p < 0.05) else " "
            cal = "✓" if 0.93 <= row["picp95"] <= 0.96 else " "
            print(f"{bname:>10s} {mname:>11s} | {row['wis']:>7.4f}{sig} {p:>7.4f} "
                  f"{row['picp95']:>6.4f}{cal} {row['k_of_n']:>7s} {str(row['cp95ci']):>14s} {row['w95']:>6.2f} | "
                  f"{row['peak_picp95']:>7.3f} {row['last34_wis']:>7.4f}")

    (ROOT / "scripts" / "_nov_conformal_sota.json").write_text(json.dumps(results, indent=2))
    winners = [r for r in results["rows"] if r["wis"] < results["ref_wis"] and r["dm_p_vs_ref"] < 0.05
               and 0.93 <= r["picp95"] <= 0.96 and r["last34_wis"] < 2.72]
    wlabel = [w["base"] + "/" + w["method"] for w in winners] or "NONE"
    print("\n=== DECISIVE WINNERS (WIS<ref & DM p<0.05 & PICP95 in [0.93,0.96] & last34<2.72): "
          + str(wlabel) + " ===")


if __name__ == "__main__":
    raise SystemExit(main())
