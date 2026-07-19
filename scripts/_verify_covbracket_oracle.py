#!/usr/bin/env python
"""ORACLE diagnostic for the coverage bracket (NOT leak-free — clearly labelled).

Purpose: distinguish WHY leak-free selection missed [0.94,0.96]:
  * STRUCTURAL: no w (blend) / c (scale) puts the TEST in [0.94,0.96] at all
                (a genuine resolution gap between the pid and cqr regimes), OR
  * SELECTION : the bracket IS reachable on the test path, but the past val tail
                picked the wrong scalar (regime shift val->test).

Scans the TEST-side blend coverage vs w and scale coverage vs c directly (oracle,
uses test outcomes) purely to characterise reachability. Reuses the cached bagged
GBM quantiles from _verify_covbracket.py — no refit, no live-code changes.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y,
    load_split, build_features, cqr_offsets, foi_multipliers, build_bounds_cqr, build_bounds_pid)
from scripts.dec_boosted_mech_multiorigin import CONFIGS, fit_gbm, bagged_qy
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

ALPHAS = list(FLUSIGHT_ALPHAS)
T0_VAL, T0_TEST, REFIT_K = 165, 205, 4
PID_GAMMA, CQR_GAMMA = 1.0, 0.75
QY_CACHE = D.SCRATCH / "covbracket_qy_full.npz"
LO, HI = 0.94, 0.96


def picp(b, y):
    lo, hi = b[0.05]; return float(np.mean((y >= lo) & (y <= hi)))
def wm(b, y, m):
    return float(np.mean(wis_from_bounds(y, b, ALPHAS, median=m)))
def blend(bp, bc, w):
    return {a: (w*bp[a][0]+(1-w)*bc[a][0], w*bp[a][1]+(1-w)*bc[a][1]) for a in ALPHAS}
def scale(b, med, c, cap):
    return {a: (np.clip(med-c*(med-b[a][0]),0,cap), np.clip(med+c*(b[a][1]-med),0,cap)) for a in ALPHAS}


def main():
    X_tr, y_tr, X_te, y_te_split, meta = load_split()
    ntr = len(y_tr); ntot = ntr + len(y_te_split)
    frozen = np.asarray(json.loads((ROOT/"simulation/results/per_model_optimal/TiRex.json").read_text())["refit_test_predictions"], float)
    d = np.load(D.TIREX_CACHE); tirex_pool = d["tirex_pool"]
    y_full = np.concatenate([y_tr, y_te_split]); cap = 2.0*float(np.max(y_full))
    tirex_full = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    feat_full, foi_lag = build_features(y_tr, y_te_split, X_tr, X_te, tirex_full)
    r_full = y_full - tirex_full
    z = np.load(QY_CACHE); origins_all, qy_all = z["origins"], z["qy"]
    sl = slice(T0_TEST - T0_VAL, ntot - T0_VAL)
    qy_te = qy_all[sl]; y_te = y_full[origins_all[sl]]; foi_te = foi_lag[origins_all[sl]]
    med_te = qy_te[:, MED_COL]; n = len(y_te)

    # cqr seed exactly like the verified/test run
    seed_train = np.arange(MIN_CTX, T0_TEST - K_CAL)
    seed_gbm = [fit_gbm(feat_full[seed_train], r_full[seed_train], cfg) for cfg in CONFIGS.values()]
    cal_idx = np.arange(T0_TEST - K_CAL, T0_TEST)
    cqr_test = cqr_offsets(bagged_qy(seed_gbm, feat_full[cal_idx], tirex_full[cal_idx], cap), y_full[cal_idx])
    foi_seed = foi_lag[MIN_CTX:T0_TEST]
    b_pid = build_bounds_pid(qy_te, cqr_test, y_te, cap, foi_mult=foi_multipliers(foi_te, foi_seed, PID_GAMMA))
    b_cqr = build_bounds_cqr(qy_te, cqr_test, cap, foi_mult=foi_multipliers(foi_te, foi_seed, CQR_GAMMA))

    def bracket_ks():
        return [k for k in range(n+1) if LO <= k/n <= HI]
    ks = bracket_ks()

    # ---- ORACLE blend scan over w ----
    ws = np.linspace(0, 1, 501)
    blend_hits, blend_curve = [], []
    for w in ws:
        b = blend(b_pid, b_cqr, w); p = picp(b, y_te)
        blend_curve.append((round(float(w),3), round(p,4)))
        if LO <= p <= HI:
            blend_hits.append({"w": round(float(w),3), "picp95": round(p,4), "wis": round(wm(b,y_te,med_te),4)})
    # ---- ORACLE scale scan over c on both bases ----
    cs = np.linspace(0.3, 3.0, 541)
    def scan_scale(base):
        hits = []
        for c in cs:
            b = scale(base, med_te, c, cap); p = picp(b, y_te)
            if LO <= p <= HI:
                hits.append({"c": round(float(c),3), "picp95": round(p,4), "wis": round(wm(b,y_te,med_te),4)})
        return hits
    scale_pid_hits = scan_scale(b_pid)
    scale_cqr_hits = scan_scale(b_cqr)

    # best in-bracket (min WIS) across all oracle constructions
    allhits = ([{**h, "kind": "blend"} for h in blend_hits]
               + [{**h, "kind": "scale_pid"} for h in scale_pid_hits]
               + [{**h, "kind": "scale_cqr"} for h in scale_cqr_hits])
    best = min(allhits, key=lambda h: h["wis"]) if allhits else None

    # coverage jump across the blend path (structural gap size)
    pic_vals = sorted({p for _, p in blend_curve})
    below = max([p for p in pic_vals if p < LO], default=None)
    above = min([p for p in pic_vals if p > HI], default=None)

    out = {
        "n_test": n, "bracket_ks": ks, "bracket_picps": [round(k/n,4) for k in ks],
        "note": "ORACLE (uses test outcomes) — reachability diagnostic only, NOT a leak-free selection",
        "endpoints_test": {"pid_mech_g1.0": {"picp95": round(picp(b_pid,y_te),4), "wis": round(wm(b_pid,y_te,med_te),4)},
                            "cqr_mech_g0.75": {"picp95": round(picp(b_cqr,y_te),4), "wis": round(wm(b_cqr,y_te,med_te),4)}},
        "blend_bracket_reachable": bool(blend_hits), "blend_in_bracket": blend_hits,
        "blend_nearest_below_LO": below, "blend_nearest_above_HI": above,
        "scale_pid_bracket_reachable": bool(scale_pid_hits), "scale_pid_in_bracket": scale_pid_hits[:8],
        "scale_cqr_bracket_reachable": bool(scale_cqr_hits), "scale_cqr_in_bracket": scale_cqr_hits[:8],
        "oracle_best_in_bracket_minWIS": best,
        "diagnosis": ("SELECTION (bracket reachable on path; val tail picked wrong scalar)"
                      if (blend_hits or scale_pid_hits or scale_cqr_hits)
                      else "STRUCTURAL (no w/c puts test in [0.94,0.96] — genuine resolution gap)"),
    }
    outp = ROOT/"scripts"/"_verify_covbracket_oracle.json"
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {outp}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
