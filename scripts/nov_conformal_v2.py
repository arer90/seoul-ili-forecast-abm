#!/usr/bin/env python
"""Overnight campaign v2 — ONLINE COVERAGE GUARD on conditional residual quantiles (Codex #1/#2:
"you need a controller, not another static CQR pass"). Leak-free, SAME 132 origins (weeks 205..336).

The GBM residual learner gives WIS 2.2688 but OVER-covers at 0.985. Replace the fixed CQR offset with
an online ACI controller per interval level that targets nominal coverage exactly — should pull 0.985->~0.95
AND lower WIS (narrower intervals). Also apply the guard to SPCI conditional residual quantiles.

Methods over the origin sequence (all past-only):
  GBM_CQR      : bagged-GBM conditional quantiles + static CQR offset (reproduce overcover ~0.985)
  GBM_ACI      : bagged-GBM cond. quantiles + per-alpha ACI adaptive CQR offset (target nominal)   <- Codex #2
  GBM_ACIscale : bagged-GBM cond. quantiles + per-alpha multiplicative ACI width scaler
  SPCI_ACI     : SPCI cond. residual quantiles (HistGBM on lagged residuals) + per-alpha ACI guard  <- Codex #1
Compared to fair baseline TiRex+CQR (2.4012) with DM p + calibration + last-34.
No live/pipeline code touched. Deterministic.
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
from scripts.dec_boosted_mech import (load_split, build_features, MIN_CTX, K_CAL, PEAK_Y, FQ, MED_COL, FQ_COL)
from scripts.dec_boosted_mech_multiorigin import CONFIGS, fit_gbm, bagged_qy
from scripts.nov_conformal_sota import m_cqr_static, wis_arr, dm_pvalue, cp, setup, T0, REFIT_K, SPCI_LAGS
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS

ALPHAS = list(FLUSIGHT_ALPHAS)
FQL = [round(float(q), 4) for q in FQ]


def gbm_qy_seq(S, origins):
    """bagged-GBM conditional FLUSIGHT quantiles of residual r=y-tirex per origin (past-only refit)."""
    y, tirex, feat, cap, ntot = S["y"], S["tirex"], S["feat"], S["cap"], S["ntot"]
    r = y - tirex
    qy = np.zeros((len(origins), len(FQ)))
    for bstart in range(T0, ntot, REFIT_K):
        bend = min(bstart + REFIT_K, ntot); train_end = bstart - K_CAL
        tr = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat[tr], r[tr], cfg) for cfg in CONFIGS.values()]
        oi = np.arange(bstart, bend)
        qy[oi - T0] = bagged_qy(gbm, feat[oi], tirex[oi], cap)  # these are y-scale quantiles (tirex+r)
    return qy


def spci_qy_seq(S, origins):
    """SPCI conditional residual quantiles from lagged residuals (y-scale = tirex + q̂(r))."""
    y, tirex, cap, ntot = S["y"], S["tirex"], S["cap"], S["ntot"]
    resid = y - tirex
    Lr = np.full((ntot, SPCI_LAGS), np.nan)
    for lag in range(1, SPCI_LAGS + 1):
        Lr[lag:, lag - 1] = resid[:-lag]
    woy = (np.arange(ntot) % 52) / 52.0
    Xr = np.column_stack([Lr, np.sin(2 * np.pi * woy), np.cos(2 * np.pi * woy)])
    cfg = dict(lr=0.05, it=300, leaves=8, msl=15, depth=3)
    qy = np.zeros((len(origins), len(FQ))); models = None
    for j, t in enumerate(origins):
        if (t - T0) % REFIT_K == 0:
            tr = np.arange(MIN_CTX + SPCI_LAGS, t)
            tr = tr[np.isfinite(resid[tr]) & np.isfinite(Xr[tr]).all(1)]
            models = {}
            for q in FQL:
                m = HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=cfg["lr"],
                    max_iter=cfg["it"], max_leaf_nodes=cfg["leaves"], min_samples_leaf=cfg["msl"],
                    max_depth=cfg["depth"], l2_regularization=1.0, random_state=42)
                m.fit(Xr[tr], resid[tr]); models[q] = m
        pred = np.array([models[q].predict(Xr[t:t + 1])[0] for q in FQL])
        qy[j] = np.clip(tirex[t] + np.sort(pred), 0.0, cap)
    return qy


def bounds_static_cqr(qy_seq, y_seq, S, origins):
    """Static CQR: per-alpha offset from trailing calibration E-scores (expanding, past-only)."""
    cap = S["cap"]; n = len(origins)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    Ehist = {a: [] for a in ALPHAS}
    for j in range(n):
        for a in ALPHAS:
            cl = FQ_COL[round(a / 2.0, 4)]; ch = FQ_COL[round(1 - a / 2.0, 4)]
            past = np.asarray(Ehist[a])
            Q = np.quantile(past, min(1.0, (1 - a) * (1 + 1 / max(len(past), 1)))) if len(past) >= 5 else 0.0
            lo = np.clip(qy_seq[j, cl] - Q, 0, cap); hi = np.clip(qy_seq[j, ch] + Q, 0, cap)
            B[a][0][j] = lo; B[a][1][j] = hi
            E = max(qy_seq[j, cl] - y_seq[j], y_seq[j] - qy_seq[j, ch]); Ehist[a].append(E)
    return B


def bounds_aci_cqr(qy_seq, y_seq, S, origins, eta_map=None):
    """ACI on the CQR offset: per-alpha adaptive miscoverage a_t, offset=quantile(pastE, 1-a_t)."""
    cap = S["cap"]; n = len(origins)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    if eta_map is None:
        eta_map = {a: max(0.005, a * 0.1) for a in ALPHAS}
    for a in ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]; ch = FQ_COL[round(1 - a / 2.0, 4)]
        a_t = a; E = []; eta = eta_map[a]
        for j in range(n):
            past = np.asarray(E)
            lvl = float(np.clip(1 - a_t, 0.0, 1.0))
            Q = np.quantile(past, lvl) if len(past) >= 5 else 0.0
            lo = np.clip(qy_seq[j, cl] - Q, 0, cap); hi = np.clip(qy_seq[j, ch] + Q, 0, cap)
            B[a][0][j] = lo; B[a][1][j] = hi
            miss = 1.0 if (y_seq[j] < lo or y_seq[j] > hi) else 0.0
            a_t = float(np.clip(a_t + eta * (a - miss), 1e-3, 0.999))
            E.append(max(qy_seq[j, cl] - y_seq[j], y_seq[j] - qy_seq[j, ch]))
    return B


def bounds_aci_scale(qy_seq, y_seq, S, origins, eta=0.03):
    """Multiplicative ACI width scaler around the median: hi=med+s_t*(q_hi-med), lo=med-s_t*(med-q_lo)."""
    cap = S["cap"]; n = len(origins)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    for a in ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]; ch = FQ_COL[round(1 - a / 2.0, 4)]
        s_t = 1.0
        for j in range(n):
            med = qy_seq[j, MED_COL]
            lo = np.clip(med - s_t * (med - qy_seq[j, cl]), 0, cap)
            hi = np.clip(med + s_t * (qy_seq[j, ch] - med), 0, cap)
            B[a][0][j] = lo; B[a][1][j] = hi
            miss = 1.0 if (y_seq[j] < lo or y_seq[j] > hi) else 0.0
            # widen on miss, shrink on cover, toward target a
            s_t = float(np.clip(s_t + eta * (miss - a) * 5.0, 0.2, 5.0))
    return B


def evalrow(name, B, y_seq, med, ref_wis, peak, last34, n):
    w = wis_arr(B, y_seq, med)
    lo95, hi95 = B[0.05]; covv = (y_seq >= lo95) & (y_seq <= hi95); k = int(covv.sum())
    p, dbar = dm_pvalue(w, ref_wis)
    return {"method": name, "wis": round(float(w.mean()), 4), "dm_p_vs_ref": round(p, 4),
            "picp95": round(k / n, 4), "k_of_n": f"{k}/{n}", "cp95ci": list(cp(k, n)),
            "w95": round(float((hi95 - lo95).mean()), 2), "peak_picp95": round(float(covv[peak].mean()), 3),
            "last34_wis": round(float(w[last34].mean()), 4)}, w


def main():
    S = setup(); ntot = S["ntot"]; origins = np.arange(T0, ntot); n = len(origins)
    y_seq = S["y"][origins]; peak = y_seq >= PEAK_Y
    last34 = np.zeros(n, bool); last34[n - 34:] = True

    ref_B = m_cqr_static(S["tirex"], S["y"] - S["tirex"], S, origins)
    ref_med = S["tirex"][origins]; ref_wis = wis_arr(ref_B, y_seq, ref_med)
    print(f"=== v2 online-guard vs fair baseline TiRex+CQR (WIS {ref_wis.mean():.4f}), {n} leak-free origins ===\n")

    gbm_qy = gbm_qy_seq(S, origins); gbm_med = gbm_qy[:, MED_COL]
    spci_qy = spci_qy_seq(S, origins); spci_med = spci_qy[:, MED_COL]

    rows = []
    specs = [
        ("GBM_CQR",       bounds_static_cqr(gbm_qy, y_seq, S, origins), gbm_med),
        ("GBM_ACI",       bounds_aci_cqr(gbm_qy, y_seq, S, origins), gbm_med),
        ("GBM_ACIscale",  bounds_aci_scale(gbm_qy, y_seq, S, origins), gbm_med),
        ("SPCI_static",   bounds_static_cqr(spci_qy, y_seq, S, origins), spci_med),
        ("SPCI_ACI",      bounds_aci_cqr(spci_qy, y_seq, S, origins), spci_med),
    ]
    hdr = f"{'method':>13s} | {'WIS':>7s} {'DMp':>7s} {'d%':>6s} {'PICP95':>7s} {'k/N':>7s} {'W95':>6s} | {'peakP95':>7s} {'l34WIS':>7s}"
    print(hdr); print("-" * len(hdr))
    for name, B, med in specs:
        row, _ = evalrow(name, B, y_seq, med, ref_wis, peak, last34, n)
        rows.append(row)
        sig = "*" if (row["wis"] < ref_wis.mean() and row["dm_p_vs_ref"] < 0.05) else " "
        cal = "✓" if 0.93 <= row["picp95"] <= 0.96 else " "
        dpct = 100 * (row["wis"] - ref_wis.mean()) / ref_wis.mean()
        print(f"{name:>13s} | {row['wis']:>7.4f}{sig} {row['dm_p_vs_ref']:>7.4f} {dpct:>6.1f} "
              f"{row['picp95']:>6.4f}{cal} {row['k_of_n']:>7s} {row['w95']:>6.2f} | "
              f"{row['peak_picp95']:>7.3f} {row['last34_wis']:>7.4f}")

    (ROOT / "scripts" / "_nov_conformal_v2.json").write_text(
        json.dumps({"ref_wis": round(float(ref_wis.mean()), 4), "n": n, "rows": rows}, indent=2))
    win = [r for r in rows if r["wis"] < ref_wis.mean() and r["dm_p_vs_ref"] < 0.05
           and 0.93 <= r["picp95"] <= 0.96 and r["last34_wis"] < 2.72]
    print("\n=== DECISIVE (WIS<ref & DMp<0.05 & PICP95∈[0.93,0.96] & last34<2.72): "
          + str([w["method"] for w in win] or "NONE") + " ===")


if __name__ == "__main__":
    raise SystemExit(main())
