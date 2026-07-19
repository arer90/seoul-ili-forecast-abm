#!/usr/bin/env python
"""EXPERIMENT (Codex #3): LightGBM quantile-residual learner replacing HistGBM.

Drop-in swap of the residual learner inside the verified 132-origin rolling scaffold
(scripts/nov_guard_v3.py). Everything else identical to the current champion path:
  base   = TiRex 1-step point (frozen test + rolled pool, leak-free)
  learner= per-FluSight-quantile LGBMRegressor(objective='quantile', alpha=q, ...) on
           residual r = y - TiRex, monotone-rearranged, bagged over a few seeds
  conformal = static CQR, seed on [165,205) (== [T0-K_CAL, T0)), no foi-width mechanism
Compared per-origin to the EXACT fair baseline TiRex+empirical-CQR (WIS 2.4012) via DM.

Leak-free: per block the learner trains on weeks < bstart-K_CAL only; CQR seed pre-T0;
cap = 2*max(y) (train+test max is the same 201.4 constant used by the whole campaign).
Reuses helpers only; touches NO live/pipeline/dec_boosted_mech code.
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

FQL = [round(float(q), 4) for q in FQ]
SEED_POOL = [42, 7, 123, 2024, 99, 314, 271]


def fit_lgbm(Xtr, r_tr, seed):
    """One LGBM quantile regressor per FluSight level on residual r (train-only)."""
    models = {}
    for q in FQL:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=150, learning_rate=0.03,
            num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, deterministic=True,
            force_row_wise=True, verbose=-1, random_state=seed)
        m.fit(Xtr, r_tr)
        models[q] = m
    return models


def predict_qy(models, X, tirex, cap):
    """Raw y-quantile matrix (n,23) = TiRex + cond residual quantiles, monotone-rearranged."""
    n = len(X)
    qy = np.empty((n, len(FQL)), dtype=float)
    for j, q in enumerate(FQL):
        qy[:, j] = tirex + np.asarray(models[q].predict(X), dtype=float)
    qy = np.clip(qy, 0.0, cap)
    qy.sort(axis=1)
    return qy


def bagged_qy(model_lists, X, tirex, cap):
    stack = np.stack([predict_qy(g, X, tirex, cap) for g in model_lists], axis=0)
    return np.sort(stack.mean(axis=0), axis=1)


def build_lgb_qy(feat, tirex, yf, cap, idxs, seeds, refit_k):
    """Bagged-LGBM conditional FluSight quantiles at weeks idxs (past-only per-block refit)."""
    r = yf - tirex
    idxs = np.asarray(idxs)
    qy = np.zeros((len(idxs), len(FQL)), dtype=float)
    lo, hi = idxs.min(), idxs.max() + 1
    for bstart in range(lo, hi, refit_k):
        bend = min(bstart + refit_k, hi)
        train_end = bstart - K_CAL                       # strictly-past training cutoff
        tr = np.arange(MIN_CTX, train_end)
        gbm = [fit_lgbm(feat[tr], r[tr], sd) for sd in seeds]
        mask = (idxs >= bstart) & (idxs < bend)
        if mask.any():
            oi = idxs[mask]
            qy[mask] = bagged_qy(gbm, feat[oi], tirex[oi], cap)
    return qy


def main():
    t0 = time.time()
    S = V.setup()
    feat_full = S["feat"]; tirex = S["tirex"]; yf = S["yf"]; cap = S["cap"]; ntot = S["ntot"]
    feat_nomech = np.delete(feat_full, [13, 14, 15], axis=1)  # drop 3 mech_lag cols

    origins = np.arange(T0, ntot); n = len(origins)
    y = yf[origins]; peak = y >= PEAK_Y
    last34 = np.zeros(n, bool); last34[n - 34:] = True
    r_full = yf - tirex
    cal_idx = np.arange(T0 - K_CAL, T0)

    # ---- EXACT fair baseline (2.4012): TiRex point + empirical past-residual CQR ----
    qy_ref = tirex_empirical_qy(tirex, r_full, origins, cap)
    qy_ref_cal = tirex_empirical_qy(tirex, r_full, cal_idx, cap)
    cqr_ref = cqr_offsets(qy_ref_cal, yf[cal_idx])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_med = qy_ref[:, MED_COL]; ref_wis = V.wis_of(ref_B, y, ref_med)
    ref_cov = ((y >= ref_B[0.05][0]) & (y <= ref_B[0.05][1]))

    print(f"=== LightGBM quantile-residual (Codex #3), {n} leak-free origins (weeks {T0}..{ntot-1}) ===")
    print(f"    reference fair baseline TiRex+CQR: WIS={ref_wis.mean():.4f}  "
          f"PICP95={ref_cov.mean():.4f} ({int(ref_cov.sum())}/{n})")
    print(f"    current HistGBM static_cqr: WIS=2.2765 DMp=0.0572 PICP95=0.9848 last34=2.7164")
    print(f"    TARGET: WIS<2.4012 & DMp<0.05 & PICP95 in [0.93,0.96] & last34<2.72\n")

    configs = []
    for nseeds in (1, 3, 5):
        for rk in (4, 6):
            configs.append({"tag": f"mech_s{nseeds}_rk{rk}", "nseeds": nseeds, "rk": rk, "mech": True})
    for nseeds in (1, 5):
        for rk in (4, 6):
            configs.append({"tag": f"nomech_s{nseeds}_rk{rk}", "nseeds": nseeds, "rk": rk, "mech": False})

    hdr = (f"{'config':>16s} | {'WIS':>7s} {'DMp':>7s} {'d%':>6s} {'PICP95':>7s} {'k/N':>7s} "
           f"{'CP95ci':>14s} {'W95':>6s} | {'pkP95':>6s} {'l34WIS':>7s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for c in configs:
        feat = feat_full if c["mech"] else feat_nomech
        seeds = SEED_POOL[:c["nseeds"]]
        qy = build_lgb_qy(feat, tirex, yf, cap, origins, seeds, c["rk"])
        qy_cal = build_lgb_qy(feat, tirex, yf, cap, cal_idx, seeds, c["rk"])
        cqr = cqr_offsets(qy_cal, yf[cal_idx])
        B = build_bounds_cqr(qy, cqr, cap)
        med = qy[:, MED_COL]
        w = V.wis_of(B, y, med)
        lo95, hi95 = B[0.05]; cov = (y >= lo95) & (y <= hi95); k = int(cov.sum())
        p, dbar = V.dm(w, ref_wis)
        cplo, cphi = V.cp(k, n)
        row = {"config": c["tag"], "wis": round(float(w.mean()), 4), "dm_p": round(p, 4),
               "picp95": round(k / n, 4), "k_of_n": f"{k}/{n}", "cp95ci": [cplo, cphi],
               "w95": round(float((hi95 - lo95).mean()), 2),
               "peak_picp95": round(float(cov[peak].mean()), 3),
               "last34_wis": round(float(w[last34].mean()), 4),
               "d_pct": round(100 * (float(w.mean()) - ref_wis.mean()) / ref_wis.mean(), 2)}
        rows.append(row)
        sig = "*" if (row["wis"] < ref_wis.mean() and p < 0.05) else " "
        cal = "✓" if 0.93 <= row["picp95"] <= 0.96 else " "
        print(f"{c['tag']:>16s} | {row['wis']:>7.4f}{sig} {p:>7.4f} {row['d_pct']:>6.2f} "
              f"{row['picp95']:>6.4f}{cal} {row['k_of_n']:>7s} {str(row['cp95ci']):>14s} "
              f"{row['w95']:>6.2f} | {row['peak_picp95']:>6.3f} {row['last34_wis']:>7.4f}")

    win = [r for r in rows if r["wis"] < ref_wis.mean() and r["dm_p"] < 0.05
           and 0.93 <= r["picp95"] <= 0.96 and r["last34_wis"] < 2.72]
    out = {"ref_wis": round(float(ref_wis.mean()), 4), "ref_picp95": round(float(ref_cov.mean()), 4),
           "n": n, "rows": rows, "winners": [w["config"] for w in win]}
    (ROOT / "scripts" / "_exp_lgbm_qresid.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== DECISIVE (WIS<ref & DMp<0.05 & PICP95 in [0.93,0.96] & last34<2.72): "
          f"{[w['config'] for w in win] or 'NONE'} ===")
    if win:
        b = min(win, key=lambda r: r["wis"])
        print(f"    BEST: {b['config']}  WIS={b['wis']} (DM p={b['dm_p']})  "
              f"PICP95={b['picp95']} {b['cp95ci']}  last34={b['last34_wis']}")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())
