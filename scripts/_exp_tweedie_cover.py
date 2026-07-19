#!/usr/bin/env python
"""OVERNIGHT EXPERIMENT (part 2) — restore nominal coverage on the SHARP Tweedie head.

Part-1 (_exp_tweedie.py) found: the pure-TiRex-mean Tweedie-scale head is SHARP and
DM-significantly beats the 2.4012 fair baseline (WIS ~2.25, DM p<0.001), but the
STATIC 40-week CQR seed [165,205) UNDER-covers (PICP95 0.86-0.92) because those
pre-peak weeks don't represent the peak-heavy test tail (exchangeability broken).
Diagnosis: 10/13 misses are above-hi UNDER-predictions on the season ramps.

Fix (all leak-free, past-only): recalibrate the CQR offset on RECENT past conformity
scores instead of a frozen seed, so the offset grows as the epidemic ramps and catches
the rising-limb under-predictions — which both raises coverage AND removes the 2/alpha
peak penalty (lowering WIS). Schemes on the head's conditional quantiles q_y:

  static      : frozen CQR seed [165,205) (part-1 baseline of the head)
  expand      : expanding CQR — at origin t, Q_a = quantile(E over [165,t)); at the
                first origin this EQUALS the static seed, then grows (strict generalization)
  trailW      : trailing-window CQR — Q_a from the last W past weeks [max(165,t-W), t)
  aci(eta)    : ACI on the CQR offset (Gibbs-Candes 2021), target 0.95, tuned pre-T0

Every score E_s at week s uses q_y(s) (per-block past-only fit) and the KNOWN y_s with
s < t only -> strictly leak-free. Reference stays the exact 2.4012 fair baseline; each
candidate's per-origin WIS is DM-tested against it. Honest selection is pre-T0: the
headline config is argmin-WIS on the validation origins [165,205) among configs whose
val PICP95 >= 0.93 (CQR calibrated from [125,·)). No live/pipeline edits.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dec_boosted_mech import (cqr_offsets, build_bounds_cqr,
                                       FQ_COL, MED_COL, K_CAL)
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup, dm, cp, wis_of, ALPHAS
from scripts._exp_tweedie import build_span, WSTART

P_GRID = (1.1, 1.2, 1.3, 1.4, 1.5)
QMETHODS = ("gamma", "pearson")
CAL_START = T0 - K_CAL          # 165: first-origin window == static seed
VAL_CAL_START = 125             # for validation origins [165,205)
TRAIL_WINDOWS = (26, 40, 52, 78)
ACI_ETAS = (0.02, 0.05, 0.1)


def rolling_cqr_bounds(Qspan, S, origins, cap, cal_start, window):
    """Leak-free rolling/expanding CQR. window=None -> expanding from cal_start,
    else trailing max(cal_start, t-window). Q_a recomputed per origin from past scores."""
    yf = S["yf"]
    n = len(origins)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    cols = {a: (FQ_COL[round(a / 2.0, 4)], FQ_COL[round(1 - a / 2.0, 4)]) for a in ALPHAS}
    for j, t in enumerate(origins):
        s0 = cal_start if window is None else max(cal_start, t - window)
        sw = np.arange(s0, t)
        qy_s = Qspan[sw - WSTART]
        y_s = yf[sw]
        qt = Qspan[t - WSTART]
        m = len(sw)
        beta_fac = 1.0 + 1.0 / max(m, 1)
        for a in ALPHAS:
            cl, ch = cols[a]
            E = np.maximum(qy_s[:, cl] - y_s, y_s - qy_s[:, ch])
            beta = min(1.0, (1.0 - a) * beta_fac)
            Q = max(0.0, float(np.quantile(E, beta)))
            B[a][0][j] = np.clip(qt[cl] - Q, 0.0, cap)
            B[a][1][j] = np.clip(qt[ch] + Q, 0.0, cap)
    return B


def aci_cqr_bounds(Qspan, S, origins, cap, cal_start, eta):
    """ACI on the CQR offset: per-alpha adaptive level a_t (target a), leak-free.
    Offset quantile taken over the expanding past-score buffer; a_t updated from past miss."""
    yf = S["yf"]
    n = len(origins)
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    cols = {a: (FQ_COL[round(a / 2.0, 4)], FQ_COL[round(1 - a / 2.0, 4)]) for a in ALPHAS}
    # seed score buffers from [cal_start, T0-origins... use expanding from cal_start
    for a in ALPHAS:
        cl, ch = cols[a]
        a_t = a
        E = []  # past conformity scores (expanding)
        # preload scores strictly before first origin
        pre = np.arange(cal_start, origins[0])
        qy_pre = Qspan[pre - WSTART]
        E = list(np.maximum(qy_pre[:, cl] - yf[pre], yf[pre] - qy_pre[:, ch]))
        for j, t in enumerate(origins):
            lvl = np.clip(1.0 - a_t, 0.0, 1.0)
            Q = float(np.quantile(E, lvl)) if len(E) >= 5 else 0.0
            Q = max(0.0, Q)
            qt = Qspan[t - WSTART]
            lo = np.clip(qt[cl] - Q, 0.0, cap)
            hi = np.clip(qt[ch] + Q, 0.0, cap)
            B[a][0][j] = lo
            B[a][1][j] = hi
            miss = 1.0 if (yf[t] < lo or yf[t] > hi) else 0.0
            a_t = float(np.clip(a_t + eta * (a - miss), 1e-3, 0.5))
            E.append(max(qt[cl] - yf[t], yf[t] - qt[ch]))
    return B


def static_bounds(Qspan, S, origins, cap, cal_start):
    cal = np.arange(cal_start, cal_start + K_CAL)
    qy = Qspan[origins - WSTART]
    qy_cal = Qspan[cal - WSTART]
    cqr = cqr_offsets(qy_cal, S["yf"][cal])
    return build_bounds_cqr(qy, cqr, cap)


def metrics(w, B, y, n, ref_wis):
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    p_dm, _ = dm(w, ref_wis)
    return dict(wis=round(float(w.mean()), 4), dm_p=round(float(p_dm), 4),
                picp95=round(k / n, 4), k=k, cp=cp(k, n),
                w95=round(float((hi95 - lo95).mean()), 2),
                last34=round(float(w[n - 34:].mean()), 4))


def build_schemes(Qspan, S, origins, cap, cal_start):
    schemes = {"static": static_bounds(Qspan, S, origins, cap, cal_start),
               "expand": rolling_cqr_bounds(Qspan, S, origins, cap, cal_start, None)}
    for W in TRAIL_WINDOWS:
        schemes[f"trail{W}"] = rolling_cqr_bounds(Qspan, S, origins, cap, cal_start, W)
    for eta in ACI_ETAS:
        schemes[f"aci{eta}"] = aci_cqr_bounds(Qspan, S, origins, cap, cal_start, eta)
    return schemes


def main():
    t0 = time.time()
    S = setup()
    ntot = S["ntot"]
    cap = S["cap"]
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S["yf"][origins]
    cal = np.arange(T0 - K_CAL, T0)
    r_full = S["yf"] - S["tirex"]

    qy_ref = tirex_empirical_qy(S["tirex"], r_full, origins, cap)
    cqr_ref = cqr_offsets(tirex_empirical_qy(S["tirex"], r_full, cal, cap), S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())

    val_origins = np.arange(T0 - K_CAL, T0)     # [165,205)
    y_val = S["yf"][val_origins]

    print(f"=== Coverage restoration on the sharp Tweedie head — {n} leak-free origins ===")
    print(f"    REFERENCE fair baseline: WIS={ref_mean:.4f}  PICP95=0.9545  last34={float(ref_wis[n-34:].mean()):.4f}")
    print(f"    TARGET: WIS<{ref_mean:.4f} & DM p<0.05 & PICP95 in [0.93,0.96] & last34<2.72\n")

    hdr = (f"{'config':>26s} | {'WIS':>7s} {'DMp':>7s} {'PICP95':>7s} {'k/N':>7s} "
           f"{'CP95ci':>15s} {'W95':>6s} {'l34':>7s} | {'valWIS':>7s} {'valP95':>6s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for qm_i, qm in enumerate(QMETHODS):
        for p in P_GRID:
            QG, QP, _ = build_span(S, p, "tirex")
            Qspan = QG if qm == "gamma" else QP
            test_schemes = build_schemes(Qspan, S, origins, cap, CAL_START)
            val_schemes = build_schemes(Qspan, S, val_origins, cap, VAL_CAL_START)
            med_test = Qspan[origins - WSTART][:, MED_COL]
            med_val = Qspan[val_origins - WSTART][:, MED_COL]
            for sch, B in test_schemes.items():
                w = wis_of(B, y, med_test)
                m = metrics(w, B, y, n, ref_wis)
                Bv = val_schemes[sch]
                wv = wis_of(Bv, y_val, med_val)
                lo_v, hi_v = Bv[0.05]
                m["val_wis"] = round(float(wv.mean()), 4)
                m["val_picp95"] = round(float(((y_val >= lo_v) & (y_val <= hi_v)).mean()), 4)
                name = f"tirex_{qm}_p{p}_{sch}"
                m["config"] = name
                rows.append(m)
                sig = "*" if (m["wis"] < ref_mean and m["dm_p"] < 0.05) else " "
                calm = "✓" if 0.93 <= m["picp95"] <= 0.96 else " "
                last_ok = "L" if m["last34"] < 2.72 else " "
                print(f"{name:>26s} | {m['wis']:>7.4f}{sig} {m['dm_p']:>7.4f} "
                      f"{m['picp95']:>6.4f}{calm} {str(m['k'])+'/'+str(n):>7s} {str(m['cp']):>15s} "
                      f"{m['w95']:>6.2f} {m['last34']:>6.4f}{last_ok} | {m['val_wis']:>7.4f} {m['val_picp95']:>6.3f}")

    # honest headline: argmin val WIS among val-calibrated (val PICP95>=0.93), pre-T0 only
    val_ok = [r for r in rows if r["val_picp95"] >= 0.93]
    headline = min(val_ok, key=lambda r: r["val_wis"]) if val_ok else min(rows, key=lambda r: r["val_wis"])
    ok = [r for r in rows if r["wis"] < ref_mean and r["dm_p"] < 0.05
          and 0.93 <= r["picp95"] <= 0.96 and r["last34"] < 2.72]
    best_test = min(ok, key=lambda r: r["wis"]) if ok else None

    out = {"ref_wis": round(ref_mean, 4), "n": n, "headline_val_selected": headline,
           "constraint_winners": ok, "rows": rows}
    (ROOT / "scripts" / "_exp_tweedie_cover.json").write_text(json.dumps(out, indent=2))

    print("\n--- HONEST headline (argmin pre-T0 val WIS s.t. val PICP95>=0.93) ---")
    h = headline
    allbars = bool(h["wis"] < ref_mean and h["dm_p"] < 0.05 and 0.93 <= h["picp95"] <= 0.96 and h["last34"] < 2.72)
    print(f"    {h['config']}: TEST WIS={h['wis']} (DM p={h['dm_p']}) PICP95={h['picp95']} {h['cp']} "
          f"last34={h['last34']} | valWIS={h['val_wis']} valP95={h['val_picp95']}")
    print(f"    ALL BARS (WIS<ref & DMp<0.05 & PICP95∈[0.93,0.96] & last34<2.72): {allbars}")
    print(f"\n--- configs clearing ALL bars: {[r['config'] for r in ok] or 'NONE'}")
    if best_test:
        print(f"    post-hoc best-WIS clearing all bars (transparency): {best_test['config']} "
              f"WIS={best_test['wis']} DMp={best_test['dm_p']} PICP95={best_test['picp95']} last34={best_test['last34']}")
    print(f"\nelapsed {time.time()-t0:.1f}s  wrote scripts/_exp_tweedie_cover.json")


if __name__ == "__main__":
    raise SystemExit(main())
