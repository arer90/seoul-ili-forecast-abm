#!/usr/bin/env python
"""OVERNIGHT COMBINATION — Tweedie distributional head  ⊕  SPCI conditional-QRF.

Two verified leak-free DECISIVE wins over the fair baseline TiRex+CQR (WIS 2.4012,
PICP95 0.9545, 126/132) on the SAME 132 rolling 1-step origins (Seoul ILI wk205..336,
T0=205, K_CAL=40):
    Tweedie : WIS 2.2427 (-6.6%), DM p 3.35e-6, PICP95 0.9318 (123/132), peak(y>=50) 0.870,
              last34 2.6491.  SHARP but overall coverage at the margin + weak on peaks.
    SPCI    : WIS 2.3449 (-2.3%), DM p 0.0104, PICP95 0.9545 (126/132, cleanest), last34 2.6772.
              Well-calibrated overall, wider.

This script REUSES the EXACT quantile generators of both winners (imports only; no edits to
any live/pipeline or existing scripts/_exp_*, dec_boosted_mech*):
  * Tweedie:  scripts._exp_tweedie.build_span (pearson residual-scale span, p* by pre-T0 val)
              + scripts._exp_tweedie_cover.rolling_cqr_bounds (EXPANDING split-CQR).
  * SPCI:     scripts._exp_spci.block_qrf_resid_grid / _yq_from_grid / bounds_beta / median_from_grid
              + scripts._exp_spci_final.conformity / calibrate_expanding (expanding conformal).

Both standalones are reproduced EXACTLY (asserted): w=1 -> 2.2427, w=0(bag) -> 2.3449.

Combinations tried (ALL selected leak-free by argmin PRE-T0 val WIS on origins [165,205)
ONLY; test origins never consulted in any config/weight choice):
  (a) quantile blend  q_comb(tau) = w*q_tweedie(tau) + (1-w)*q_spci(tau), median blended too,
      w in {0,0.25,0.5,0.75,1}.
  (b) peak-targeted envelope: Tweedie everywhere, but on RISING-LIMB origins widen the UPPER
      quantiles toward SPCI's (hi <- hi_tw + lam*max(0, hi_sp - hi_tw)); off-season stays pure
      Tweedie. Rising limb detected LEAK-FREE from the trailing signed residual mean
      g_t = mean(r[t-L:t]), r = y - TiRex (all weeks < t) -> TiRex under-predicting recently
      == epidemic accelerating (the exact mechanism behind the 10/13 above-hi peak misses).

Reference for DM = the exact 2.4012 per-origin WIS (tirex_empirical_qy + build_bounds_cqr,
the reference block of _exp_tweedie_final.py:92-98). DM = HLN h=1 paired per-origin WIS.
Leak-free #1: every weight / (L,thr,lam) chosen ONLY on [165,205); cap reported both full and
train-only (spotless 2*max(y_train)). REAL numbers only; honest verdict if it fails.
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

# ── shared constants / stats ──
from scripts.dec_boosted_mech import cqr_offsets, build_bounds_cqr, MED_COL, K_CAL, PEAK_Y
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup as tw_setup, dm, cp, wis_of, ALPHAS
# ── Tweedie generators (imported, EXACT) ──
from scripts._exp_tweedie import build_span, WSTART, P_GRID
from scripts._exp_tweedie_cover import rolling_cqr_bounds
# ── SPCI generators (imported, EXACT) ──
from scripts._exp_spci import (setup as spci_setup, block_qrf_resid_grid, _yq_from_grid,
                               bounds_beta, median_from_grid)
from scripts._exp_spci_final import conformity, calibrate_expanding, slice_bounds, VAL_LO

CAL_START = T0 - K_CAL          # 165  (test CQR / first expanding window == static seed)
VAL_CAL_START = 125             # sub-seed for validation origins [165,205)
VAL_LO_TW, VAL_HI = T0 - K_CAL, T0     # [165,205)  pre-T0 validation origins
SPCI_SEED0 = 125                # SPCI expanding-conformal seed (== its winning config)
LEAF_SET = (12, 16, 20, 24)     # SPCI bagged-QRF leaf set (== _exp_spci_report headline)
SPCI_IDX0 = 110                 # SPCI grid start (== _exp_spci_report)
STD_TWEEDIE, STD_SPCI, REF = 2.2427, 2.3449, 2.4012


# ─────────────────────────────── metrics ────────────────────────────────
def full_metrics(B, y, med, ref_wis, n):
    w = wis_of(B, y, med)
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    p_dm, dbar = dm(w, ref_wis)
    peak = y >= PEAK_Y
    return dict(
        wis=round(float(w.mean()), 4), dm_p=float(p_dm), dm_meandiff=round(float(dbar), 4),
        picp95=round(k / n, 4), k_of_n=f"{k}/{n}", cp95ci=[round(v, 4) for v in cp(k, n)],
        last34_wis=round(float(w[n - 34:].mean()), 4),
        mean_w95=round(float((hi95 - lo95).mean()), 3),
        peak_picp95=round(float(covv[peak].mean()), 4), n_peak=int(peak.sum()),
        peak_wis=round(float(w[peak].mean()), 4), _w=w,
    )


def val_wis_of(B, y, med):
    return float(wis_of(B, y, med).mean())


# ─────────────────────────── Tweedie side (EXACT) ───────────────────────────
def tweedie_span(S_tw, p):
    _QG, QP, _ = build_span(S_tw, p, "tirex")   # QP = pearson residual-scale span (the winner)
    return QP


def select_p_star(S_tw, cap):
    """argmin pre-T0 val WIS over P_GRID (expanding CQR, pearson head) — the EXACT _exp_tweedie_final rule."""
    val_o = np.arange(VAL_LO_TW, VAL_HI)
    y_val = S_tw["yf"][val_o]
    scores = {}
    for p in P_GRID:
        Q = tweedie_span(S_tw, p)
        Bv = rolling_cqr_bounds(Q, S_tw, val_o, cap, VAL_CAL_START, None)
        medv = Q[val_o - WSTART][:, MED_COL]
        scores[p] = round(val_wis_of(Bv, y_val, medv), 4)
    return min(scores, key=scores.get), scores


def tweedie_bounds(QP, S_tw, origins, cap, cal_start):
    B = rolling_cqr_bounds(QP, S_tw, origins, cap, cal_start, None)
    med = QP[origins - WSTART][:, MED_COL]
    return B, med


# ─────────────────────────── SPCI side (EXACT) ───────────────────────────
def spci_bag_bounds(S_sp):
    """Bagged-QRF SPCI (leaves 12/16/20/24) expanding-conformal — reproduces the 2.3449 headline.

    Returns calibrated bounds B over ev_idx (sorted origins >= VAL_LO=155), the conditional
    median med_ev, y_ev and ev_idx, so both the test slice (>=T0) and the val slice ([165,205))
    come from ONE leak-free calibrate_expanding call. cap = train-only (spotless)."""
    feat, r, tirex, yf, cap = S_sp["feat"], S_sp["r"], S_sp["tirex"], S_sp["yf"], S_sp["cap"]
    idx_all = np.arange(SPCI_IDX0, S_sp["ntot"])
    grids = [block_qrf_resid_grid(feat, r, idx_all,
                                  dict(n_estimators=400, min_samples_leaf=msl, max_features=0.6))
             for msl in LEAF_SET]
    bag = sum(grids) / len(grids)
    tir = tirex[idx_all]
    _yqfq, yqg = _yq_from_grid(bag, tir, cap)
    Braw = bounds_beta(yqg)
    med_all = median_from_grid(yqg, tir, True)
    y_all = yf[idx_all]
    E = conformity(Braw, y_all)
    B, ev_idx, order, ev_pos = calibrate_expanding(Braw, E, idx_all, VAL_LO, cap, SPCI_SEED0)
    med_ev = med_all[order][ev_pos]
    y_ev = y_all[order][ev_pos]
    return B, ev_idx, med_ev, y_ev


# ─────────────────────────── combination constructors ───────────────────────────
def blend_bounds(B_tw, B_sp, w):
    return {a: (w * B_tw[a][0] + (1 - w) * B_sp[a][0],
                w * B_tw[a][1] + (1 - w) * B_sp[a][1]) for a in ALPHAS}


def envelope_bounds(B_tw, B_sp, flag, lam):
    """Tweedie lower+median; on `flag` origins widen upper toward SPCI: hi = hi_tw + lam*max(0,hi_sp-hi_tw)."""
    out = {}
    for a in ALPHAS:
        lo_tw, hi_tw = B_tw[a]
        _lo_sp, hi_sp = B_sp[a]
        widen = lam * np.maximum(0.0, hi_sp - hi_tw)
        hi = np.where(flag, hi_tw + widen, hi_tw)
        out[a] = (lo_tw.copy(), hi)
    return out


def trailing_signed(r_full, origins, L):
    """g_t = mean(r[t-L:t]) using residuals for weeks < t only (leak-free rising-limb signal)."""
    g = np.zeros(len(origins))
    for j, t in enumerate(origins):
        seg = r_full[t - L:t]
        seg = seg[np.isfinite(seg)]
        g[j] = float(seg.mean()) if len(seg) else 0.0
    return g


def clean(m):
    return {k: v for k, v in m.items() if k != "_w"}


def main():
    t0 = time.time()
    # ---- Tweedie setup (cap = 2*max(y_full); reproduces reported 2.2427) ----
    S_tw = tw_setup()
    ntot = S_tw["ntot"]
    cap_tw = S_tw["cap"]
    cap_train = 2.0 * float(S_tw["yf"][:269].max())           # spotless train-only cap
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S_tw["yf"][origins]
    val_o = np.arange(VAL_LO_TW, VAL_HI)
    y_val = S_tw["yf"][val_o]
    r_full = S_tw["yf"] - S_tw["tirex"]

    # ---- exact 2.4012 reference (per-origin WIS for DM) ----
    cal = np.arange(CAL_START, T0)
    qy_ref = tirex_empirical_qy(S_tw["tirex"], r_full, origins, cap_tw)
    cqr_ref = cqr_offsets(tirex_empirical_qy(S_tw["tirex"], r_full, cal, cap_tw), S_tw["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap_tw)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())
    ref_k = int(((y >= ref_B[0.05][0]) & (y <= ref_B[0.05][1])).sum())
    assert abs(ref_mean - REF) < 5e-4, f"reference mismatch {ref_mean}"

    # ---- Tweedie standalone (p* by pre-T0 val) ----
    p_star, val_p_scores = select_p_star(S_tw, cap_tw)
    QP = tweedie_span(S_tw, p_star)
    B_tw, med_tw = tweedie_bounds(QP, S_tw, origins, cap_tw, CAL_START)
    B_tw_val, med_tw_val = tweedie_bounds(QP, S_tw, val_o, cap_tw, VAL_CAL_START)
    B_tw_train, _ = tweedie_bounds(QP, S_tw, origins, cap_train, CAL_START)
    m_tw = full_metrics(B_tw, y, med_tw, ref_wis, n)
    assert abs(m_tw["wis"] - STD_TWEEDIE) < 1e-3, f"Tweedie reproduce fail {m_tw['wis']}"

    # ---- SPCI standalone (bagged QRF; ONE leak-free calibrate call for test+val slices) ----
    S_sp = spci_setup()
    B_sp_all, ev_idx, med_ev, y_ev = spci_bag_bounds(S_sp)
    tst_m = ev_idx >= T0
    val_m = (ev_idx >= VAL_LO_TW) & (ev_idx < VAL_HI)
    assert np.array_equal(ev_idx[tst_m], origins), "SPCI test origin misalignment"
    assert np.array_equal(ev_idx[val_m], val_o), "SPCI val origin misalignment"
    assert np.allclose(y_ev[tst_m], y), "SPCI y misalignment"
    B_sp = slice_bounds(B_sp_all, tst_m); med_sp = med_ev[tst_m]
    B_sp_val = slice_bounds(B_sp_all, val_m); med_sp_val = med_ev[val_m]
    m_sp = full_metrics(B_sp, y, med_sp, ref_wis, n)
    assert abs(m_sp["wis"] - STD_SPCI) < 1e-3, f"SPCI reproduce fail {m_sp['wis']}"

    # cap for SPCI is train-only already; note it for the report
    cap_sp = S_sp["cap"]

    print("=" * 92)
    print(f"REFERENCE fair baseline TiRex+CQR : WIS={ref_mean:.4f}  PICP95={ref_k/n:.4f} ({ref_k}/{n})")
    print(f"STANDALONE Tweedie (p*={p_star})       : WIS={m_tw['wis']:.4f}  PICP95={m_tw['picp95']:.4f} "
          f"({m_tw['k_of_n']})  peakP95={m_tw['peak_picp95']:.3f}  last34={m_tw['last34_wis']:.4f}")
    print(f"STANDALONE SPCI  (bag{list(LEAF_SET)}) : WIS={m_sp['wis']:.4f}  PICP95={m_sp['picp95']:.4f} "
          f"({m_sp['k_of_n']})  peakP95={m_sp['peak_picp95']:.3f}  last34={m_sp['last34_wis']:.4f}")
    print(f"    (Tweedie p-val scores: {val_p_scores}; SPCI cap=train-only {cap_sp:.1f}, Tweedie cap=full {cap_tw:.1f})")
    print("=" * 92)

    # ---- candidate set (val-selected) ----
    cands = {}      # name -> dict(build test bounds, med, val bounds, val med, kind, params)
    # (a) quantile blend
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        cands[f"blend_w{w:.2f}"] = dict(
            kind="blend", w=w,
            B=blend_bounds(B_tw, B_sp, w), med=w * med_tw + (1 - w) * med_sp,
            Bv=blend_bounds(B_tw_val, B_sp_val, w), medv=w * med_tw_val + (1 - w) * med_sp_val)
    # (b) peak-targeted envelope (rising-limb widen upper toward SPCI)
    for L in (4, 6):
        g_t = trailing_signed(r_full, origins, L)
        g_v = trailing_signed(r_full, val_o, L)
        for thr in (0.0, 1.0):
            fl_t = g_t > thr
            fl_v = g_v > thr
            for lam in (0.5, 1.0):
                cands[f"env_L{L}_thr{thr:.1f}_lam{lam:.1f}"] = dict(
                    kind="env", L=L, thr=thr, lam=lam, n_flag=int(fl_t.sum()),
                    B=envelope_bounds(B_tw, B_sp, fl_t, lam), med=med_tw,
                    Bv=envelope_bounds(B_tw_val, B_sp_val, fl_v, lam), medv=med_tw_val)

    rows = []
    wis_arr = {}
    for name, c in cands.items():
        vw = val_wis_of(c["Bv"], y_val, c["medv"])
        mt = full_metrics(c["B"], y, c["med"], ref_wis, n)
        wis_arr[name] = mt["_w"]
        # DM of this candidate's per-origin WIS vs each STANDALONE (honest "beats both" test)
        dp_tw, dd_tw = dm(mt["_w"], m_tw["_w"])
        dp_sp, dd_sp = dm(mt["_w"], m_sp["_w"])
        row = dict(config=name, kind=c["kind"], val_wis=round(vw, 4), **clean(mt),
                   dm_p_vs_tweedie=round(float(dp_tw), 4), dm_diff_vs_tweedie=round(float(dd_tw), 4),
                   dm_p_vs_spci=round(float(dp_sp), 4), dm_diff_vs_spci=round(float(dd_sp), 4))
        for extra in ("w", "L", "thr", "lam", "n_flag"):
            if extra in c:
                row[extra] = c[extra]
        rows.append(row)

    # honest headline: argmin PRE-T0 val WIS over ALL candidates
    headline = min(rows, key=lambda r: r["val_wis"])
    # transparency: which candidate best raises peak coverage while WIS<=2.30 & DM p<0.05
    peak_ok = [r for r in rows if r["wis"] <= 2.30 and r["dm_p"] < 0.05]
    best_peak = max(peak_ok, key=lambda r: r["peak_picp95"]) if peak_ok else None

    # ---- print table ----
    hdr = (f"{'config':>22s} | {'valWIS':>7s} || {'WIS':>7s} {'DMp':>9s} {'PICP95':>7s} {'k/N':>7s} "
           f"{'pkP95':>6s} {'W95':>6s} {'last34':>7s}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["val_wis"]):
        star = "*" if r["config"] == headline["config"] else " "
        print(f"{r['config']:>22s}{star}| {r['val_wis']:>7.4f} || {r['wis']:>7.4f} {r['dm_p']:>9.2e} "
              f"{r['picp95']:>7.4f} {r['k_of_n']:>7s} {r['peak_picp95']:>6.3f} {r['mean_w95']:>6.2f} "
              f"{r['last34_wis']:>7.4f}")

    # ---- verdict ----
    h = headline
    beats_tw = h["wis"] < STD_TWEEDIE
    beats_sp = h["wis"] < STD_SPCI
    beats_both = bool(beats_tw and beats_sp)
    fixes_peak = bool(h["peak_picp95"] >= 0.90)
    is_pure_tw = h["config"] == "blend_w1.00"
    is_pure_sp = h["config"] == "blend_w0.00"
    dm_p_h = h["dm_p"]

    print("\n" + "=" * 92)
    print(f"HONEST HEADLINE (argmin pre-T0 val WIS [165,205)) : {h['config']}  (valWIS={h['val_wis']:.4f})")
    print(f"  TEST WIS       = {h['wis']:.4f}   (Tweedie {STD_TWEEDIE} / SPCI {STD_SPCI} / ref {REF})")
    print(f"  DM p vs 2.4012 = {dm_p_h:.2e}   (mean per-origin WIS diff = {h['dm_meandiff']:+.4f})")
    print(f"  PICP95         = {h['picp95']:.4f}  ({h['k_of_n']})   CP95 CI = {h['cp95ci']}")
    print(f"  peak PICP95    = {h['peak_picp95']:.4f}  (n_peak y>=50 = {h['n_peak']}; peakWIS={h['peak_wis']})")
    print(f"  last34 WIS     = {h['last34_wis']:.4f}   mean-W95 = {h['mean_w95']:.3f}")
    print(f"\n  beats Tweedie(2.2427)? {beats_tw} (DM p={h['dm_p_vs_tweedie']}, diff={h['dm_diff_vs_tweedie']:+.4f})   "
          f"beats SPCI(2.3449)? {beats_sp} (DM p={h['dm_p_vs_spci']}, diff={h['dm_diff_vs_spci']:+.4f})   beats BOTH? {beats_both}")
    print(f"  headline reduces to pure Tweedie? {is_pure_tw}   pure SPCI? {is_pure_sp}")
    print(f"  fixes peak coverage (>=0.90)? {fixes_peak}")
    if best_peak is not None:
        print(f"\n  TRANSPARENCY (NOT the honest pick) best peak-coverage cand w/ WIS<=2.30 & DMp<0.05:")
        print(f"    {best_peak['config']}: WIS={best_peak['wis']} peakP95={best_peak['peak_picp95']} "
              f"PICP95={best_peak['picp95']} DMp={best_peak['dm_p']:.2e}")

    # which standalone remains best (by WIS)
    best_standalone = "Tweedie" if STD_TWEEDIE <= STD_SPCI else "SPCI"
    print(f"\n  BEST STANDALONE by WIS = {best_standalone} ({min(STD_TWEEDIE, STD_SPCI)})")
    print("=" * 92)

    out = {
        "reference_wis": round(ref_mean, 4), "n": n,
        "standalones": {"tweedie": clean(m_tw), "spci_bag": clean(m_sp),
                        "tweedie_train_cap_wis": round(full_metrics(B_tw_train, y, med_tw, ref_wis, n)["wis"], 4)},
        "tweedie_p_star": p_star, "tweedie_val_p_scores": val_p_scores,
        "spci_config": f"bagged-QRF{list(LEAF_SET)} beta_expanding seed0={SPCI_SEED0}",
        "candidates": rows,
        "headline": {"config": h["config"], **{k: v for k, v in h.items() if k != "config"}},
        "verdict": {
            "beats_tweedie_2p2427": beats_tw, "beats_spci_2p3449": beats_sp,
            "beats_both_standalone": beats_both, "fixes_peak_coverage_ge0p90": fixes_peak,
            "headline_is_pure_tweedie": is_pure_tw, "headline_is_pure_spci": is_pure_sp,
            "best_standalone_by_wis": best_standalone,
            "best_standalone_wis": min(STD_TWEEDIE, STD_SPCI),
        },
        "transparency_best_peak_cand": clean(best_peak) if best_peak else None,
        "leak_free_notes": [
            "every weight w and envelope (L,thr,lam) selected ONLY by argmin val WIS over [165,205); test never consulted",
            "Tweedie: pearson residual-scale span (build_span) + expanding split-CQR (rolling_cqr_bounds), p* by pre-T0 val",
            "SPCI: bagged-QRF conditional residual quantiles + expanding conformal (calibrate_expanding), seed0=125, train-only cap",
            "envelope rising-limb flag g_t=mean(r[t-L:t]) uses residuals for weeks < t only (leak-free, past-only)",
            "reference = exact 2.4012 per-origin WIS; DM = HLN h=1 paired per-origin WIS",
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (ROOT / "scripts" / "_exp_combo.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/_exp_combo.json  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
