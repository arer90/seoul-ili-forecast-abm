#!/usr/bin/env python
"""EXPERIMENT sweep: can a tuned / capacity-bagged LightGBM conditional-quantile residual beat
the HistGBM incumbent (WIS 2.2765, DM p 0.0572) and clear DM p<0.05 vs the fair baseline 2.4012?

From _exp_lgbm_qresid.py the spec LightGBM (lr0.03, 150 trees, 7 leaves), seed-bagged, is 2.30
(wider than HistGBM 2.2765) and DM p 0.064. HistGBM's edge is a 6-DIVERSE-CAPACITY bag, not seeds.
This sweeps single LightGBM capacities and a diverse capacity-bag (mirroring HistGBM's weak..deep).
Same 132-origin leak-free scaffold, static CQR seed [165,205), mechanism features, no foi-width.
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

FQL = [round(float(q), 4) for q in FQ]

# diverse LightGBM capacities (mirror the HistGBM weak..deep bag)
BAG_CFGS = [
    dict(n_estimators=150, learning_rate=0.03, num_leaves=7,  min_child_samples=20),
    dict(n_estimators=150, learning_rate=0.03, num_leaves=4,  min_child_samples=25),
    dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10),
    dict(n_estimators=300, learning_rate=0.05, num_leaves=8,  min_child_samples=20),
    dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=8),
    dict(n_estimators=600, learning_rate=0.02, num_leaves=8,  min_child_samples=15),
]
SINGLE_CFGS = {
    "spec_7l":  dict(n_estimators=150, learning_rate=0.03, num_leaves=7,  min_child_samples=20),
    "mid_15l":  dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10),
    "deep_31l": dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=8),
    "lowlr_8l": dict(n_estimators=600, learning_rate=0.02, num_leaves=8,  min_child_samples=15),
}


def fit_one(Xtr, r_tr, cfg, seed=42):
    models = {}
    for q in FQL:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, deterministic=True,
            force_row_wise=True, verbose=-1, random_state=seed, **cfg)
        m.fit(Xtr, r_tr)
        models[q] = m
    return models


def predict_qy(models, X, tirex, cap):
    qy = np.empty((len(X), len(FQL)), dtype=float)
    for j, q in enumerate(FQL):
        qy[:, j] = tirex + np.asarray(models[q].predict(X), dtype=float)
    qy = np.clip(qy, 0.0, cap); qy.sort(axis=1)
    return qy


def bagged_qy(model_lists, X, tirex, cap):
    stack = np.stack([predict_qy(g, X, tirex, cap) for g in model_lists], axis=0)
    return np.sort(stack.mean(axis=0), axis=1)


def build_qy(feat, tirex, yf, cap, idxs, cfgs, refit_k):
    """cfgs = list of param dicts (bag). Past-only block refit."""
    r = yf - tirex
    idxs = np.asarray(idxs)
    qy = np.zeros((len(idxs), len(FQL)), dtype=float)
    lo, hi = idxs.min(), idxs.max() + 1
    for bstart in range(lo, hi, refit_k):
        bend = min(bstart + refit_k, hi)
        tr = np.arange(MIN_CTX, bstart - K_CAL)
        gbm = [fit_one(feat[tr], r[tr], cfg) for cfg in cfgs]
        mask = (idxs >= bstart) & (idxs < bend)
        if mask.any():
            oi = idxs[mask]
            qy[mask] = bagged_qy(gbm, feat[oi], tirex[oi], cap)
    return qy


def score(feat, tirex, yf, cap, origins, cal_idx, y, peak, last34, ref_wis, n, cfgs, refit_k):
    qy = build_qy(feat, tirex, yf, cap, origins, cfgs, refit_k)
    qy_cal = build_qy(feat, tirex, yf, cap, cal_idx, cfgs, refit_k)
    cqr = cqr_offsets(qy_cal, yf[cal_idx])
    B = build_bounds_cqr(qy, cqr, cap)
    med = qy[:, MED_COL]
    w = V.wis_of(B, y, med)
    lo95, hi95 = B[0.05]; cov = (y >= lo95) & (y <= hi95); k = int(cov.sum())
    pval, _ = V.dm(w, ref_wis)
    cplo, cphi = V.cp(k, n)
    return {"wis": round(float(w.mean()), 4), "dm_p": round(pval, 4), "picp95": round(k / n, 4),
            "k_of_n": f"{k}/{n}", "cp95ci": [cplo, cphi], "w95": round(float((hi95 - lo95).mean()), 2),
            "peak_picp95": round(float(cov[peak].mean()), 3), "last34_wis": round(float(w[last34].mean()), 4),
            "d_pct": round(100 * (float(w.mean()) - ref_wis.mean()) / ref_wis.mean(), 2)}


def main():
    t0 = time.time()
    S = V.setup()
    feat = S["feat"]; tirex = S["tirex"]; yf = S["yf"]; cap = S["cap"]; ntot = S["ntot"]
    origins = np.arange(T0, ntot); n = len(origins)
    y = yf[origins]; peak = y >= PEAK_Y
    last34 = np.zeros(n, bool); last34[n - 34:] = True
    r_full = yf - tirex
    cal_idx = np.arange(T0 - K_CAL, T0)

    qy_ref = tirex_empirical_qy(tirex, r_full, origins, cap)
    qy_ref_cal = tirex_empirical_qy(tirex, r_full, cal_idx, cap)
    cqr_ref = cqr_offsets(qy_ref_cal, yf[cal_idx])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_wis = V.wis_of(ref_B, y, qy_ref[:, MED_COL])

    print(f"=== LightGBM capacity sweep + capacity-bag, {n} leak-free origins ===")
    print(f"    fair baseline WIS={ref_wis.mean():.4f} ; HistGBM incumbent 2.2765 (DMp .0572, PICP95 .9848)")
    print(f"    TARGET: WIS<2.4012 & DMp<0.05 & PICP95 in [0.93,0.96] & last34<2.72\n")

    hdr = (f"{'config':>16s} | {'WIS':>7s} {'DMp':>7s} {'d%':>6s} {'PICP95':>7s} {'k/N':>7s} "
           f"{'CP95ci':>14s} {'W95':>6s} | {'pkP95':>6s} {'l34WIS':>7s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    trials = [(f"single_{tag}_rk6", [cfg], 6) for tag, cfg in SINGLE_CFGS.items()]
    trials += [("bag6_rk6", BAG_CFGS, 6), ("bag6_rk4", BAG_CFGS, 4)]
    for tag, cfgs, rk in trials:
        r = score(feat, tirex, yf, cap, origins, cal_idx, y, peak, last34, ref_wis, n, cfgs, rk)
        r["config"] = tag; rows.append(r)
        sig = "*" if (r["wis"] < ref_wis.mean() and r["dm_p"] < 0.05) else " "
        cal = "✓" if 0.93 <= r["picp95"] <= 0.96 else " "
        print(f"{tag:>16s} | {r['wis']:>7.4f}{sig} {r['dm_p']:>7.4f} {r['d_pct']:>6.2f} "
              f"{r['picp95']:>6.4f}{cal} {r['k_of_n']:>7s} {str(r['cp95ci']):>14s} "
              f"{r['w95']:>6.2f} | {r['peak_picp95']:>6.3f} {r['last34_wis']:>7.4f}")

    win = [r for r in rows if r["wis"] < ref_wis.mean() and r["dm_p"] < 0.05
           and 0.93 <= r["picp95"] <= 0.96 and r["last34_wis"] < 2.72]
    best_dm = min(rows, key=lambda r: r["dm_p"])
    out = {"ref_wis": round(float(ref_wis.mean()), 4), "n": n, "rows": rows,
           "winners": [w["config"] for w in win], "best_dm": best_dm}
    (ROOT / "scripts" / "_exp_lgbm_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== DECISIVE: {[w['config'] for w in win] or 'NONE'} ===")
    print(f"    best DM p = {best_dm['dm_p']} ({best_dm['config']}, WIS={best_dm['wis']}, "
          f"PICP95={best_dm['picp95']})")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())
