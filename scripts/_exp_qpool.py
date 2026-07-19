#!/usr/bin/env python
"""OVERNIGHT EXPERIMENT — CRPS/WIS-optimal quantile POOLING of the best interval methods.

The verified single-method leader on this frozen 132-origin leak-free rolling 1-step
benchmark (Seoul ILI weeks 205..336, T0=205, K_CAL=40) is the Tweedie distributional
head (WIS 2.2427, DM p 3.35e-6 vs the 2.4012 fair baseline, PICP95 0.9318). This script
asks whether POOLING its per-origin FluSight quantile matrix with two other strong but
DIFFERENTLY-STRUCTURED interval methods diversifies the quantile estimate enough for a
small consistent WIS cut below 2.2427.

Three ingredient methods (each its own established leak-free protocol, reproduced here):
  * TWEEDIE  : q = mu + Qz*mu^(p/2), mu=TiRex 1-step, p*=argmin pre-T0 val WIS, EXPANDING
               split-CQR seeded wk125.                       (scripts/_exp_tweedie*.py)   ~2.24
  * SPCI     : QRF conditional residual quantiles + width-optimal beta-search + expanding
               split-conformal; QRF leaf size = argmin pre-T0 val WIS.   (scripts/_exp_spci*.py) ~2.34
  * GBM-CQR  : bagged HistGBM residual conditional quantiles + static CQR seed [165,205).
               (scripts/nov_guard_v3.build_gbm_qy + dec_boosted_mech.build_bounds_cqr)    ~2.27

Each method yields a per-origin 23-column FluSight quantile matrix (its final CALIBRATED
interval endpoints + median). We POOL the matched columns across methods four ways, all
selected ONLY on the pre-T0 validation window [165,205):
  (1) vincent      : Vincentization = plain quantile averaging (equal 1/3 weights, fixed).
  (2) global_convex: one convex weight vector over the 3 methods (simplex grid step 0.05),
                     argmin val WIS.
  (3) per_alpha    : a SEPARATE convex weight per FluSight interval level (WIS decomposes
                     additively over levels, so each level's endpoints are pooled to argmin
                     that level's val interval score; the median is pooled to argmin val
                     |y-med|). The fully-flexible "per-quantile-level weight".
  (4) peak_tilt    : global_convex for the lower endpoints + median, but the UPPER endpoints
                     tilted (fraction f, val-tuned) toward the method with the best val
                     high-tail (y>=25) coverage — the "upper quantiles toward best peak
                     coverage" heuristic.

Monotonicity is enforced by a per-row sort of the pooled 23-column matrix (non-crossing
quantiles + nested intervals). Reference = the exact 2.4012 fair baseline AND the Tweedie
2.2427 head; every pooled scheme's per-origin WIS is DM-tested (HLN h=1) against both.
Cap = 2*max(y_train) train-only (spotless) is applied to the pooled matrix and cross-checked
against 2*max(y_full); leak-free by construction (all weights on pre-T0 val only). No
live/pipeline or existing-script edits — imports only.

HONEST by design: the single-method rows are printed alongside, so if pooling merely
reproduces the best single method (global weight collapses to Tweedie) that is reported
plainly, not dressed up as a win.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dec_boosted_mech import (cqr_offsets, build_bounds_cqr, FQ_COL, MED_COL, K_CAL)
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup as guard_setup, build_gbm_qy, wis_of, dm, cp, ALPHAS
from scripts._exp_tweedie import build_span, WSTART, P_GRID
from scripts._exp_tweedie_cover import rolling_cqr_bounds
from scripts._exp_spci import setup as spci_setup
from scripts._exp_spci_final import build as spci_build

VAL_LO, VAL_HI = 165, 205          # pre-T0 validation origins (weight/leaf/p selection)
VAL_CAL_START = 125                # CQR seed start for the validation window
CAL_START = VAL_LO                 # test CQR seed start (= 165), matches the headline scripts
SEED0_SPCI = 125
SPCI_LEAVES = (10, 14, 20)
NQ = 23
PEAK_Y = 50.0
HITAIL_Y = 25.0                    # val "high tail" threshold for the peak-coverage heuristic
GRID_STEP = 0.05
METHODS = ("tweedie", "spci", "gbm")

# ─────────────────────────── matrix <-> bounds ───────────────────────────
def bounds_to_matrix(B, med):
    """Pack per-alpha (lo,hi) endpoints + median into a (n,23) FluSight-quantile matrix."""
    n = len(med)
    M = np.zeros((n, NQ), dtype=float)
    M[:, MED_COL] = med
    for a in ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]
        ch = FQ_COL[round(1.0 - a / 2.0, 4)]
        M[:, cl] = B[a][0]
        M[:, ch] = B[a][1]
    return M


def matrix_to_bounds(M, cap=None):
    """Monotone-sort each row (non-crossing) and unpack to bounds dict + median."""
    Ms = np.sort(M, axis=1)
    if cap is not None:
        Ms = np.clip(Ms, 0.0, cap)
    B = {a: (Ms[:, FQ_COL[round(a / 2.0, 4)]], Ms[:, FQ_COL[round(1.0 - a / 2.0, 4)]])
         for a in ALPHAS}
    return B, Ms[:, MED_COL], Ms


# ─────────────────────────── simplex grid ───────────────────────────
def simplex_grid(step=GRID_STEP):
    ns = int(round(1.0 / step))
    pts = []
    for i in range(ns + 1):
        for j in range(ns + 1 - i):
            k = ns - i - j
            pts.append((i / ns, j / ns, k / ns))
    return np.asarray(pts, dtype=float)


# ─────────────────────────── ingredient method matrices ───────────────────────────
def build_tweedie(S, test_origins, val_origins, cap):
    yf = S["yf"]
    val_scores = {}
    QPs = {}
    for p in P_GRID:
        _, QP, _ = build_span(S, p, "tirex")
        QPs[p] = QP
        Bv = rolling_cqr_bounds(QP, S, val_origins, cap, VAL_CAL_START, None)
        medv = QP[val_origins - WSTART][:, MED_COL]
        val_scores[p] = float(wis_of(Bv, yf[val_origins], medv).mean())
    p_star = min(val_scores, key=val_scores.get)
    QP = QPs[p_star]
    Bt = rolling_cqr_bounds(QP, S, test_origins, cap, CAL_START, None)
    medt = QP[test_origins - WSTART][:, MED_COL]
    Bv = rolling_cqr_bounds(QP, S, val_origins, cap, VAL_CAL_START, None)
    medv = QP[val_origins - WSTART][:, MED_COL]
    # exact headline per-origin WIS (no round-trip sort) = the 2.2427 DM reference
    tw_wis_exact = wis_of(Bt, yf[test_origins], medt)
    info = {"p_star": p_star, "val_wis_by_p": {str(k): round(v, 4) for k, v in val_scores.items()},
            "headline_wis": round(float(tw_wis_exact.mean()), 4)}
    return bounds_to_matrix(Bt, medt), bounds_to_matrix(Bv, medv), info, tw_wis_exact


def build_gbm(S, test_origins, val_origins, cap):
    yf = S["yf"]
    cal_idx = np.arange(T0 - K_CAL, T0)              # [165,205) == val_origins range
    vcal_idx = np.arange(VAL_LO - K_CAL, VAL_LO)     # [125,165)
    G_test = build_gbm_qy(S, test_origins)
    G_165 = build_gbm_qy(S, np.arange(VAL_LO, VAL_HI))   # serves test-CQR cal AND val origins
    G_125 = build_gbm_qy(S, vcal_idx)
    # test bounds: static CQR seed [165,205)
    cqr_t = cqr_offsets(G_165, yf[cal_idx])
    Bt = build_bounds_cqr(G_test, cqr_t, cap)
    medt = G_test[:, MED_COL]
    # val bounds: static CQR seed [125,165)
    cqr_v = cqr_offsets(G_125, yf[vcal_idx])
    Bv = build_bounds_cqr(G_165, cqr_v, cap)
    medv = G_165[:, MED_COL]
    return bounds_to_matrix(Bt, medt), bounds_to_matrix(Bv, medv), {}


def build_spci(Ssp, test_origins, val_origins):
    yf = Ssp["yf"]
    best = None
    for msl in SPCI_LEAVES:
        r = spci_build(Ssp, msl, SEED0_SPCI, engine="beta")
        idx = r["idx"]; B = r["B"]; med = r["med"]; yv_ = r["y"]
        vm = (idx >= VAL_LO) & (idx < VAL_HI)
        Bv = {a: (B[a][0][vm], B[a][1][vm]) for a in ALPHAS}
        vw = float(wis_of(Bv, yv_[vm], med[vm]).mean())
        if best is None or vw < best[0]:
            best = (vw, msl, r)
    vw, msl, r = best
    idx = r["idx"]; B = r["B"]; med = r["med"]
    tm = idx >= T0
    vm = (idx >= VAL_LO) & (idx < VAL_HI)
    Bt = {a: (B[a][0][tm], B[a][1][tm]) for a in ALPHAS}
    Bv = {a: (B[a][0][vm], B[a][1][vm]) for a in ALPHAS}
    info = {"msl_star": msl, "val_wis": round(vw, 4)}
    return bounds_to_matrix(Bt, med[tm]), bounds_to_matrix(Bv, med[vm]), info


# ─────────────────────────── evaluation ───────────────────────────
def evaluate(M, y, ref_wis, tw_wis, cap):
    B, med, _ = matrix_to_bounds(M, cap)
    w = wis_of(B, y, med)
    lo, hi = B[0.05]
    cov = (y >= lo) & (y <= hi)
    k = int(cov.sum()); nn = len(y)
    peak = y >= PEAK_Y
    p_ref, d_ref = dm(w, ref_wis)
    p_tw, d_tw = dm(w, tw_wis)
    last34 = np.arange(nn) >= nn - 34
    return {
        "wis": round(float(w.mean()), 4),
        "dm_p_vs_ref": float(p_ref), "dm_mean_diff_vs_ref": round(float(d_ref), 4),
        "dm_p_vs_tweedie": float(p_tw), "dm_mean_diff_vs_tweedie": round(float(d_tw), 4),
        "picp95": round(k / nn, 4), "k_of_n": f"{k}/{nn}", "cp95ci": [round(v, 4) for v in cp(k, nn)],
        "peak_picp95": round(float(cov[peak].mean()), 3), "n_peak": int(peak.sum()),
        "last34_wis": round(float(w[last34].mean()), 4),
        "mean_w95": round(float((hi - lo).mean()), 3),
        "_w": w,
    }


def val_wis(M, yv, cap):
    B, med, _ = matrix_to_bounds(M, cap)
    return float(wis_of(B, yv, med).mean())


# ─────────────────────────── pooling schemes (weights on val only) ───────────────────────────
def pool_convex(Ms, w):
    out = np.zeros_like(Ms[0])
    for wi, Mi in zip(w, Ms):
        out += wi * Mi
    return out


def select_global(Mvals, yv, cap, grid):
    best = None
    for w in grid:
        s = val_wis(pool_convex(Mvals, w), yv, cap)
        if best is None or s < best[0]:
            best = (s, w)
    return best[1], best[0]


def select_per_alpha(Mvals, yv, grid):
    """Per-level convex weights: for each alpha argmin val interval score; median argmin val |err|."""
    w_alpha = {}
    for a in ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]; ch = FQ_COL[round(1.0 - a / 2.0, 4)]
        best = None
        for w in grid:
            lo = sum(w[m] * Mvals[m][:, cl] for m in range(3))
            hi = sum(w[m] * Mvals[m][:, ch] for m in range(3))
            IS = (hi - lo) + (2.0 / a) * np.clip(lo - yv, 0, None) + (2.0 / a) * np.clip(yv - hi, 0, None)
            s = float(IS.mean())
            if best is None or s < best[0]:
                best = (s, w)
        w_alpha[a] = best[1]
    best = None
    for w in grid:
        md = sum(w[m] * Mvals[m][:, MED_COL] for m in range(3))
        s = float(np.abs(yv - md).mean())
        if best is None or s < best[0]:
            best = (s, w)
    w_med = best[1]
    return w_alpha, w_med


def assemble_per_alpha(Ms, w_alpha, w_med):
    n = Ms[0].shape[0]
    M = np.zeros((n, NQ), dtype=float)
    M[:, MED_COL] = sum(w_med[m] * Ms[m][:, MED_COL] for m in range(3))
    for a in ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]; ch = FQ_COL[round(1.0 - a / 2.0, 4)]
        w = w_alpha[a]
        M[:, cl] = sum(w[m] * Ms[m][:, cl] for m in range(3))
        M[:, ch] = sum(w[m] * Ms[m][:, ch] for m in range(3))
    return M


def peak_method_on_val(Mvals, yv, cap):
    """Method with best val high-tail (y>=HITAIL_Y) 95% coverage."""
    hi_mask = yv >= HITAIL_Y
    picp = []
    for m in range(3):
        B, _, _ = matrix_to_bounds(Mvals[m], cap)
        lo, hi = B[0.05]
        cov = (yv >= lo) & (yv <= hi)
        picp.append(float(cov[hi_mask].mean()) if hi_mask.any() else float(cov.mean()))
    return int(np.argmax(picp)), picp


UPPER_COLS = [FQ_COL[round(1.0 - a / 2.0, 4)] for a in ALPHAS]  # upper endpoint columns


def apply_peak_tilt(Ms, w_global, peak_m, f):
    M = pool_convex(Ms, w_global).copy()
    for c in UPPER_COLS:
        base = sum(w_global[m] * Ms[m][:, c] for m in range(3))
        M[:, c] = (1.0 - f) * base + f * Ms[peak_m][:, c]
    return M


def select_peak_tilt(Mvals, yv, cap, w_global, peak_m):
    best = None
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = val_wis(apply_peak_tilt(Mvals, w_global, peak_m, f), yv, cap)
        if best is None or s < best[0]:
            best = (s, f)
    return best[1], best[0]


# ─────────────────────────── main ───────────────────────────
def main():
    t0 = time.time()
    S = guard_setup()
    Ssp = spci_setup()
    ntot = S["ntot"]
    cap_full = S["cap"]                                   # 2*max(y_full) = 201.4 (references' cap)
    cap_train = 2.0 * float(S["yf"][:269].max())          # 2*max(y_train) — spotless final clip
    yf = S["yf"]
    test_origins = np.arange(T0, ntot); n = len(test_origins)
    y = yf[test_origins]
    val_origins = np.arange(VAL_LO, VAL_HI)
    yv = yf[val_origins]

    # ---- references: exact 2.4012 fair baseline per-origin WIS ----
    r_full = yf - S["tirex"]
    cal = np.arange(T0 - K_CAL, T0)
    qy_ref = tirex_empirical_qy(S["tirex"], r_full, test_origins, cap_full)
    cqr_ref = cqr_offsets(tirex_empirical_qy(S["tirex"], r_full, cal, cap_full), yf[cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap_full)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())

    print(f"=== WIS-optimal quantile POOLING | {n} leak-free origins (weeks {T0}..{ntot-1}) ===")
    print(f"    fair baseline WIS={ref_mean:.4f} ; Tweedie head to beat = 2.2427")
    print(f"    caps: full(ref)={cap_full:.1f}  train-only(spotless)={cap_train:.1f}\n")

    # ---- ingredient method matrices (test + val) ----
    print("building ingredient methods ...")
    Mt_tw, Mv_tw, info_tw, tw_wis = build_tweedie(S, test_origins, val_origins, cap_full)
    print(f"  tweedie  p*={info_tw['p_star']}  headline_wis={info_tw['headline_wis']}  [{time.time()-t0:.0f}s]")
    Mt_gbm, Mv_gbm, _ = build_gbm(S, test_origins, val_origins, cap_full)
    print(f"  gbm-cqr  built  [{time.time()-t0:.0f}s]")
    Mt_sp, Mv_sp, info_sp = build_spci(Ssp, test_origins, val_origins)
    print(f"  spci     msl*={info_sp['msl_star']} (val_wis={info_sp['val_wis']})  [{time.time()-t0:.0f}s]")

    Mtest = [Mt_tw, Mt_sp, Mt_gbm]     # order: tweedie, spci, gbm
    Mval = [Mv_tw, Mv_sp, Mv_gbm]
    # tw_wis (from build_tweedie) = exact 2.2427 headline per-origin WIS = DM reference #2

    # ---- single-method readouts (pure weights, honesty anchor) ----
    grid = simplex_grid(GRID_STEP)
    rows = {}
    for name, M in zip(METHODS, Mtest):
        rows[f"single_{name}"] = evaluate(M, y, ref_wis, tw_wis, cap_train)
    val_single = {name: val_wis(M, yv, cap_train) for name, M in zip(METHODS, Mval)}

    # ---- (1) vincentization (equal weights, fixed) ----
    w_eq = (1 / 3, 1 / 3, 1 / 3)
    rows["pool_vincent"] = evaluate(pool_convex(Mtest, w_eq), y, ref_wis, tw_wis, cap_train)
    rows["pool_vincent"]["weights"] = [round(v, 3) for v in w_eq]
    rows["pool_vincent"]["val_wis"] = round(val_wis(pool_convex(Mval, w_eq), yv, cap_train), 4)

    # ---- (2) global convex (val-selected) ----
    w_g, vwg = select_global(Mval, yv, cap_train, grid)
    rows["pool_global"] = evaluate(pool_convex(Mtest, w_g), y, ref_wis, tw_wis, cap_train)
    rows["pool_global"]["weights"] = {m: round(float(w_g[i]), 3) for i, m in enumerate(METHODS)}
    rows["pool_global"]["val_wis"] = round(vwg, 4)

    # ---- (3) per-alpha convex (val-selected per level) ----
    w_alpha, w_med = select_per_alpha(Mval, yv, grid)
    M_pa_test = assemble_per_alpha(Mtest, w_alpha, w_med)
    M_pa_val = assemble_per_alpha(Mval, w_alpha, w_med)
    rows["pool_per_alpha"] = evaluate(M_pa_test, y, ref_wis, tw_wis, cap_train)
    rows["pool_per_alpha"]["weights_per_alpha"] = {
        str(a): {m: round(float(w_alpha[a][i]), 3) for i, m in enumerate(METHODS)} for a in ALPHAS}
    rows["pool_per_alpha"]["weights_median"] = {m: round(float(w_med[i]), 3) for i, m in enumerate(METHODS)}
    rows["pool_per_alpha"]["val_wis"] = round(val_wis(M_pa_val, yv, cap_train), 4)

    # ---- (4) peak-tilt (upper endpoints toward best val high-tail method) ----
    peak_m, peak_picp_val = peak_method_on_val(Mval, yv, cap_train)
    f_star, vwpt = select_peak_tilt(Mval, yv, cap_train, w_g, peak_m)
    rows["pool_peak_tilt"] = evaluate(apply_peak_tilt(Mtest, w_g, peak_m, f_star), y, ref_wis, tw_wis, cap_train)
    rows["pool_peak_tilt"]["peak_method"] = METHODS[peak_m]
    rows["pool_peak_tilt"]["peak_picp_val"] = {m: round(float(peak_picp_val[i]), 3) for i, m in enumerate(METHODS)}
    rows["pool_peak_tilt"]["tilt_f"] = f_star
    rows["pool_peak_tilt"]["lower_weights"] = {m: round(float(w_g[i]), 3) for i, m in enumerate(METHODS)}
    rows["pool_peak_tilt"]["val_wis"] = round(vwpt, 4)

    # ---- spotless cap cross-check on the best pooled scheme ----
    def wis_at_cap(Mbuild, cap):
        B, med, _ = matrix_to_bounds(Mbuild, cap)
        return round(float(wis_of(B, y, med).mean()), 6)
    cap_check = {
        "pool_global_wis_capfull": wis_at_cap(pool_convex(Mtest, w_g), cap_full),
        "pool_global_wis_captrain": wis_at_cap(pool_convex(Mtest, w_g), cap_train),
        "pool_per_alpha_wis_capfull": wis_at_cap(M_pa_test, cap_full),
        "pool_per_alpha_wis_captrain": wis_at_cap(M_pa_test, cap_train),
    }
    cap_binds = not (abs(cap_check["pool_global_wis_capfull"] - cap_check["pool_global_wis_captrain"]) < 1e-9
                     and abs(cap_check["pool_per_alpha_wis_capfull"] - cap_check["pool_per_alpha_wis_captrain"]) < 1e-9)

    # ─────────────────────────── report ───────────────────────────
    print(f"\n{'scheme':>16s} | {'WIS':>7s} {'DMvsBL':>8s} {'DMvsTw':>8s} {'PICP95':>7s} {'k/N':>7s} "
          f"{'CP95ci':>15s} {'pkP95':>6s} {'l34':>7s} {'W95':>6s} | {'valWIS':>7s}")
    print("-" * 118)
    order = ["single_tweedie", "single_spci", "single_gbm",
             "pool_vincent", "pool_global", "pool_per_alpha", "pool_peak_tilt"]
    for name in order:
        r = rows[name]
        vw = r.get("val_wis", val_single.get(name.replace("single_", ""), float("nan")))
        beats_tw = "*" if r["wis"] < 2.2427 else " "
        print(f"{name:>16s} | {r['wis']:>7.4f}{beats_tw} {r['dm_p_vs_ref']:>8.1e} {r['dm_p_vs_tweedie']:>8.4f} "
              f"{r['picp95']:>6.4f} {r['k_of_n']:>7s} {str(r['cp95ci']):>15s} {r['peak_picp95']:>6.3f} "
              f"{r['last34_wis']:>7.4f} {r['mean_w95']:>6.2f} | {vw:>7.4f}")

    print(f"\nweights  global      = {rows['pool_global']['weights']}")
    print(f"         vincent     = {rows['pool_vincent']['weights']}  (fixed equal)")
    print(f"         peak_tilt   = lower {rows['pool_peak_tilt']['lower_weights']} ; upper tilt f={f_star} "
          f"toward {METHODS[peak_m]} (val high-tail picp {rows['pool_peak_tilt']['peak_picp_val']})")
    print(f"         per_alpha   = (per level; see JSON)  median={rows['pool_per_alpha']['weights_median']}")
    print(f"\ncap check: {cap_check}  -> cap_binds={cap_binds} (False=spotless, train-cap==full-cap)")

    # verdict
    best_pool = min(["pool_vincent", "pool_global", "pool_per_alpha", "pool_peak_tilt"],
                    key=lambda k: rows[k]["wis"])
    bp = rows[best_pool]
    beats = bool(bp["wis"] < 2.2427)
    strict = bool(beats and bp["dm_p_vs_tweedie"] < 0.05)
    print(f"\n=== best pooled scheme = {best_pool}  WIS={bp['wis']}  (Tweedie 2.2427) ===")
    print(f"    beats Tweedie WIS: {beats}   |   DM-significant vs Tweedie (p<0.05): {strict} "
          f"(DM p vs Tweedie = {bp['dm_p_vs_tweedie']:.4f})")
    if not beats:
        gw = rows["pool_global"]["weights"]
        collapse = gw["tweedie"] >= 0.85
        print(f"    HONEST verdict: pooling does NOT beat the Tweedie head; global weight "
              f"{'COLLAPSES to Tweedie' if collapse else 'is mixed but no WIS cut'} "
              f"(tweedie weight={gw['tweedie']}).")

    out = {
        "n": n, "reference_wis": round(ref_mean, 4), "tweedie_wis": 2.2427,
        "methods_order": list(METHODS),
        "tweedie_info": info_tw, "spci_info": info_sp,
        "val_single_wis": {k: round(v, 4) for k, v in val_single.items()},
        "rows": {k: {kk: vv for kk, vv in v.items() if kk != "_w"} for k, v in rows.items()},
        "cap_check": cap_check, "cap_binds": cap_binds,
        "best_pool": best_pool, "beats_tweedie_wis": beats,
        "dm_significant_vs_tweedie": strict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (ROOT / "scripts" / "_exp_qpool.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/_exp_qpool.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    raise SystemExit(main())
