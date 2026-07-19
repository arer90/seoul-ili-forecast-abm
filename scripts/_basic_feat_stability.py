#!/usr/bin/env python
"""Can a BASIC-feature standalone quantile model + the project's STABILITY feature selection beat
default TiRex-native? Leak-free, EXPANDING per-block training (fair vs TiRex's expanding context).

BASIC 13 lag/seasonal features (S['feat'][:, :13]) -> project stability selection (select_features_stability,
Meinshausen-Bühlmann resampling frequency) on the expanding TRAIN pool -> HistGBM quantile models (one per
FluSight level) on the selected features predict y's quantiles directly -> expanding split-CQR -> WIS.
Compared to plain default TiRex-native on TEST[205,337), DM(HLN)+bootstrap. No live/pipeline edits.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts._exp_crosscountry as X
from scripts.nov_guard_v3 import setup
from scripts._tirex_headprobe_final import flusight  # native FluSight from cached deciles
from simulation.pipeline.feature_select_corr1se import select_features_stability
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

FQ = np.asarray(X.FQ, float); FQr = list(np.round(FQ, 4)); MEDI = FQr.index(0.5); NQ = len(FQ)
FQL = [float(round(q, 4)) for q in FQ]
MIN_CTX = 52; T0 = 205; REFIT_K = 15


def wis_arr(qmat, y, cap):
    qmat = np.clip(np.sort(qmat, 1), 0, cap); B = {}
    for a in FLUSIGHT_ALPHAS:
        B[a] = (qmat[:, FQr.index(round(a/2,4))], qmat[:, FQr.index(round(1-a/2,4))])
    return np.asarray(wis_from_bounds(y, B, list(FLUSIGHT_ALPHAS), median=qmat[:, MEDI]), float)


def dm_boot(wa, wb, L=6, nb=10000):
    d = wa-wb; n = len(d); v = np.var(d, ddof=1)/n
    hln = 2*(1-stats.t.cdf(abs(d.mean()/np.sqrt(v))*np.sqrt((n+1)/n), df=n-1)) if v > 0 else 1.0
    rng = np.random.RandomState(0); k = n//L; c = 0
    for _ in range(nb):
        idx = (rng.randint(0, n-L+1, size=k)[:, None]+np.arange(L)).ravel()[:n]
        if d[idx].mean() >= 0: c += 1
    return float(hln), float(c/nb)


def main():
    t0 = time.time()
    S = setup(); y = S["yf"]; feat = S["feat"]; N = S["ntot"]; cap = 2.0*float(np.nanmax(y[:T0]))
    basic = feat[:, :13]                                          # BASIC 13 lag/seasonal block
    dec = np.load(ROOT/"scripts"/"_tirex_native_deciles.npz")["dec"]
    origins = np.arange(T0, N); n = len(origins); y_te = y[origins]
    nat = np.array([flusight(dec[t], cap) for t in origins])
    nat_w = wis_arr(nat, y_te, cap)

    # BASIC-feature stability-selected quantile model, expanding per-block
    qmat = np.zeros((n, NQ)); sel_sizes = []
    for bstart in range(T0, N, REFIT_K):
        bend = min(bstart+REFIT_K, N)
        pool = np.arange(MIN_CTX, bstart)
        Xp = basic[pool]; yp = np.log1p(y[pool])                  # log1p target (project convention)
        try:
            sel = select_features_stability(Xp, yp, B=50, pi=0.6, seed=42)
            idx = sel["selected_indices"] if len(sel["selected_indices"]) else list(range(basic.shape[1]))
        except Exception:
            idx = list(range(basic.shape[1]))
        sel_sizes.append(len(idx))
        oi = np.arange(bstart, bend)
        preds = np.zeros((len(oi), NQ))
        for j, q in enumerate(FQL):
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05,
                max_iter=200, max_leaf_nodes=8, min_samples_leaf=15, l2_regularization=1.0,
                max_depth=3, random_state=42)
            m.fit(Xp[:, idx], yp)
            preds[:, j] = np.expm1(m.predict(basic[oi][:, idx]))
        qmat[oi-T0] = np.clip(np.sort(preds, 1), 0, cap)

    # raw model WIS + expanding-CQR-recalibrated WIS
    raw_w = wis_arr(qmat, y_te, cap)
    # expanding CQR on the model quantiles
    B = {a: (np.zeros(n), np.zeros(n)) for a in FLUSIGHT_ALPHAS}; Eh = {a: [] for a in FLUSIGHT_ALPHAS}
    for j in range(n):
        for a in FLUSIGHT_ALPHAS:
            cl = FQr.index(round(a/2,4)); ch = FQr.index(round(1-a/2,4)); pe = np.asarray(Eh[a])
            Q = np.quantile(pe, min(1.0,(1-a)*(1+1/max(len(pe),1)))) if len(pe) >= 5 else 0.0
            B[a][0][j] = np.clip(qmat[j,cl]-Q,0,cap); B[a][1][j] = np.clip(qmat[j,ch]+Q,0,cap)
            Eh[a].append(max(qmat[j,cl]-y_te[j], y_te[j]-qmat[j,ch]))
    cqr_w = np.asarray(wis_from_bounds(y_te, B, list(FLUSIGHT_ALPHAS), median=qmat[:,MEDI]), float)

    best_w = raw_w if raw_w.mean() < cqr_w.mean() else cqr_w
    hln, bp = dm_boot(best_w, nat_w)
    out = {
        "model": "BASIC-13-feature HistGBM-quantile + STABILITY selection, expanding per-block",
        "avg_features_selected": round(float(np.mean(sel_sizes)), 1), "of_13": 13,
        "tirex_native_wis": round(float(nat_w.mean()), 4),
        "basic_stability_raw_wis": round(float(raw_w.mean()), 4),
        "basic_stability_cqr_wis": round(float(cqr_w.mean()), 4),
        "best_basic_wis": round(float(best_w.mean()), 4),
        "delta_pct_vs_tirex": round(100*(best_w.mean()-nat_w.mean())/nat_w.mean(), 2),
        "dm_hln_p": round(hln, 4), "dm_boot_p": round(bp, 4),
        "beats_tirex": bool(best_w.mean() < nat_w.mean() and hln < 0.05 and bp < 0.05),
        "elapsed_s": round(time.time()-t0, 0),
    }
    (ROOT/"scripts"/"_basic_feat_stability.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nVERDICT:", "BASIC-feature stability model BEATS default TiRex!" if out["beats_tirex"]
          else "BASIC-feature stability model (%.4f) does NOT beat default TiRex-native (%.4f) — foundation model wins"
               % (out["best_basic_wis"], out["tirex_native_wis"]))


if __name__ == "__main__":
    raise SystemExit(main())
