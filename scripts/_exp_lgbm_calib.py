#!/usr/bin/env python
"""EXPERIMENT extension: LightGBM-corrected POINT ⊕ EXPANDING empirical residual quantiles.

Diagnosis from scripts/_exp_lgbm_qresid.py: the pure LightGBM conditional-quantile + fixed-seed
CQR path over-covers at 0.9848 (130/132) exactly like HistGBM — a distribution-shift artifact of
the [165,205) CQR seed being more volatile than the test period. That blocks PICP95 in [0.93,0.96].

The EXACT fair baseline (WIS 2.4012, PICP95 0.9545) avoids this because it builds quantiles from
EXPANDING past-residual empirical offsets (tirex_empirical_qy), which naturally track the residual
distribution and land at nominal. This experiment keeps that calibrated interval machinery but swaps
the TiRex point for a LightGBM-corrected point p = TiRex + LGBM_median(residual): a better point with
the same nominally-calibrated expanding-empirical interval + fixed CQR seed on [165,205).

Leak-free: LGBM point at each origin trains on weeks < bstart-K_CAL (rolling block refit); the
empirical residual pool at origin t uses r' = y - p strictly for weeks < t; CQR seed pre-T0.
Reuses helpers only; touches NO live/pipeline code.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.nov_guard_v3 as V
from scripts.dec_boosted_mech import (FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y,
                                      cqr_offsets, build_bounds_cqr)
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy

SEED_POOL = [42, 7, 123, 2024, 99]
P_START = 125  # first week with a rolling LGBM point (train_end=85 -> ~33 train weeks)


def fit_point(Xtr, r_tr, seed, objective):
    kw = dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1, random_state=seed)
    if objective == "q50":
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **kw)
    else:
        m = lgb.LGBMRegressor(objective="regression", **kw)
    m.fit(Xtr, r_tr)
    return m


def rolling_point(feat, tirex, yf, seeds, refit_k, objective, ntot):
    """Bagged LGBM residual-median point p[t]=TiRex[t]+LGBM_resid(feat[t]), past-only block refit."""
    r = yf - tirex
    p = np.full(ntot, np.nan)
    for bstart in range(P_START, ntot, refit_k):
        bend = min(bstart + refit_k, ntot)
        train_end = bstart - K_CAL
        tr = np.arange(MIN_CTX, train_end)
        models = [fit_point(feat[tr], r[tr], sd, objective) for sd in seeds]
        oi = np.arange(bstart, bend)
        pred = np.mean([m.predict(feat[oi]) for m in models], axis=0)
        p[oi] = tirex[oi] + pred
    return p


def empirical_qy(p_full, rp_full, idxs, cap):
    """qy = point + expanding empirical past-residual quantile offsets (strictly past-only)."""
    qy = np.zeros((len(idxs), len(FQ)), dtype=float)
    for k, t in enumerate(idxs):
        past = rp_full[:t]
        past = past[np.isfinite(past)]
        off = np.quantile(past, FQ)
        row = np.clip(p_full[t] + off, 0.0, cap)
        row.sort()
        qy[k] = row
    return qy


def main():
    t0 = time.time()
    S = V.setup()
    feat_full = S["feat"]; tirex = S["tirex"]; yf = S["yf"]; cap = S["cap"]; ntot = S["ntot"]
    feat_nomech = np.delete(feat_full, [13, 14, 15], axis=1)

    origins = np.arange(T0, ntot); n = len(origins)
    y = yf[origins]; peak = y >= PEAK_Y
    last34 = np.zeros(n, bool); last34[n - 34:] = True
    r_full = yf - tirex
    cal_idx = np.arange(T0 - K_CAL, T0)

    # ---- EXACT fair baseline (2.4012) ----
    qy_ref = tirex_empirical_qy(tirex, r_full, origins, cap)
    qy_ref_cal = tirex_empirical_qy(tirex, r_full, cal_idx, cap)
    cqr_ref = cqr_offsets(qy_ref_cal, yf[cal_idx])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_med = qy_ref[:, MED_COL]; ref_wis = V.wis_of(ref_B, y, ref_med)
    ref_cov = ((y >= ref_B[0.05][0]) & (y <= ref_B[0.05][1]))
    # baseline point AE (median absolute error of TiRex point) for context
    ref_pt_mae = float(np.mean(np.abs(y - tirex[origins])))

    print(f"=== LGBM-corrected point ⊕ expanding-empirical CQR, {n} leak-free origins ===")
    print(f"    fair baseline TiRex+emp-CQR: WIS={ref_wis.mean():.4f} PICP95={ref_cov.mean():.4f} "
          f"({int(ref_cov.sum())}/{n})  TiRex point MAE={ref_pt_mae:.3f}")
    print(f"    TARGET: WIS<2.4012 & DMp<0.05 & PICP95 in [0.93,0.96] & last34<2.72\n")

    configs = []
    for obj in ("q50", "l2"):
        for nseeds in (1, 5):
            for rk in (4, 6):
                for mech in (True, False):
                    configs.append({"obj": obj, "nseeds": nseeds, "rk": rk, "mech": mech,
                                    "tag": f"{obj}_{'m' if mech else 'nm'}_s{nseeds}_rk{rk}"})

    hdr = (f"{'config':>18s} | {'WIS':>7s} {'DMp':>7s} {'d%':>6s} {'PICP95':>7s} {'k/N':>7s} "
           f"{'CP95ci':>14s} {'W95':>6s} {'ptMAE':>6s} | {'pkP95':>6s} {'l34WIS':>7s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for c in configs:
        feat = feat_full if c["mech"] else feat_nomech
        seeds = SEED_POOL[:c["nseeds"]]
        p = rolling_point(feat, tirex, yf, seeds, c["rk"], c["obj"], ntot)
        rp = yf - p  # residual of the corrected point (nan where p undefined)
        qy = empirical_qy(p, rp, origins, cap)
        qy_cal = empirical_qy(p, rp, cal_idx, cap)
        cqr = cqr_offsets(qy_cal, yf[cal_idx])
        B = build_bounds_cqr(qy, cqr, cap)
        med = qy[:, MED_COL]
        w = V.wis_of(B, y, med)
        lo95, hi95 = B[0.05]; cov = (y >= lo95) & (y <= hi95); k = int(cov.sum())
        pval, dbar = V.dm(w, ref_wis)
        cplo, cphi = V.cp(k, n)
        pt_mae = float(np.mean(np.abs(y - p[origins])))
        row = {"config": c["tag"], "wis": round(float(w.mean()), 4), "dm_p": round(pval, 4),
               "picp95": round(k / n, 4), "k_of_n": f"{k}/{n}", "cp95ci": [cplo, cphi],
               "w95": round(float((hi95 - lo95).mean()), 2), "pt_mae": round(pt_mae, 3),
               "peak_picp95": round(float(cov[peak].mean()), 3),
               "last34_wis": round(float(w[last34].mean()), 4),
               "d_pct": round(100 * (float(w.mean()) - ref_wis.mean()) / ref_wis.mean(), 2)}
        rows.append(row)
        sig = "*" if (row["wis"] < ref_wis.mean() and pval < 0.05) else " "
        cal = "✓" if 0.93 <= row["picp95"] <= 0.96 else " "
        print(f"{c['tag']:>18s} | {row['wis']:>7.4f}{sig} {pval:>7.4f} {row['d_pct']:>6.2f} "
              f"{row['picp95']:>6.4f}{cal} {row['k_of_n']:>7s} {str(row['cp95ci']):>14s} "
              f"{row['w95']:>6.2f} {row['pt_mae']:>6.3f} | {row['peak_picp95']:>6.3f} {row['last34_wis']:>7.4f}")

    win = [r for r in rows if r["wis"] < ref_wis.mean() and r["dm_p"] < 0.05
           and 0.93 <= r["picp95"] <= 0.96 and r["last34_wis"] < 2.72]
    out = {"ref_wis": round(float(ref_wis.mean()), 4), "ref_picp95": round(float(ref_cov.mean()), 4),
           "ref_pt_mae": round(ref_pt_mae, 3), "n": n, "rows": rows,
           "winners": [w["config"] for w in win]}
    (ROOT / "scripts" / "_exp_lgbm_calib.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== DECISIVE (WIS<ref & DMp<0.05 & PICP95 in [0.93,0.96] & last34<2.72): "
          f"{[w['config'] for w in win] or 'NONE'} ===")
    if win:
        b = min(win, key=lambda r: r["wis"])
        print(f"    BEST: {b['config']}  WIS={b['wis']} (DM p={b['dm_p']})  "
              f"PICP95={b['picp95']} {b['cp95ci']}  last34={b['last34_wis']}")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())
