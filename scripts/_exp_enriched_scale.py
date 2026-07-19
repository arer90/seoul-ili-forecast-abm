#!/usr/bin/env python
"""OVERNIGHT EXPERIMENT — feature-enriched Tweedie residual-SCALE head.

The verified decisive win is the Tweedie residual-scale interval (point = TiRex 1-step;
FLUSIGHT quantiles q = mu + Qz*mu^(p/2), Qz = empirical past standardized-residual
quantiles, single GLOBAL power p*=1.5; EXPANDING split-CQR): WIS 2.2427, DM p 3.35e-6 vs
the 2.4012 fair baseline, PICP95 0.9318 (123/132), last34 2.6491, peak(y>=50) PICP95 0.87
over 132 leak-free rolling 1-step origins (weeks 205..336, T0=205, K_CAL=40).

Its one known weakness is PEAK under-coverage (0.87): the single global Qz cannot be both
sharp off-season AND wide on the rising limb. This script ENRICHES only the SCALE model
(point stays TiRex, distribution stays Tweedie residual-scale, conformal stays expanding
split-CQR) three ways, each a strict generalization that collapses to the p=1.5 baseline:

  (a) REGIME-conditional Qz — separate past standardized-residual quantile pools for
      off-season / rising-limb / peak, split by a leak-free trailing y-level signal;
      each pool James-Stein shrunk toward the global Qz by a pseudo-count k0 (small pools
      -> global). Data-driven per-block regime thresholds (block-quantiles of the signal).
  (b) HARMONIC quantile-scale — a small per-level linear quantile model (statsmodels
      QuantReg) of the standardized residual z on seasonal harmonics [sin,cos @ 52-wk
      (+104-wk)], giving a calendar-conditional Qz(tau | week-of-year) beyond mu^(p/2).
  (c) ASYMMETRIC power — steeper UPPER power p_hi (wider peaks) + gentler LOWER power
      p_lo (sharper off-season troughs): q = mu + Qz_lo*mu^(p_lo/2) for tau<0.5,
      mu + Qz_hi*mu^(p_hi/2) for tau>0.5.
  (d) COMBINED regime x asymmetric — the val-selected knobs of (a) and (c) fused.

HONEST selection: every knob (signal window, #regimes, k0, #harmonics, p, p_lo, p_hi) is
chosen by argmin WIS on the pre-T0 VALIDATION origins [165,205) ONLY (expanding CQR seeded
[125,·)); the 132 test origins are never used to pick anything. Leak-free: per-block head
trained/calibrated on weeks strictly before each origin (train_end = block_start - K_CAL);
regime signal & thresholds & harmonics & all conformity scores use past y only; refit every
REFIT_K=4 origins; cap = 2*max(y_train) (train-only). Reference = the exact 2.4012 fair
baseline; each candidate's per-origin WIS is DM-tested (HLN h=1) vs 2.4012 AND vs the
Tweedie p=1.5 baseline (2.2427). No live/pipeline or existing-script edits; NEW script only.
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
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dec_boosted_mech import (cqr_offsets, build_bounds_cqr, FQ, MED_COL,
                                      MIN_CTX, K_CAL, PEAK_Y)
from scripts.dec_boosted_mech_multiorigin import T0, REFIT_K
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup, dm, cp, wis_of
from scripts._exp_tweedie import build_span, WSTART
from scripts._exp_tweedie_cover import rolling_cqr_bounds

EPS = 1e-6
FQ = np.asarray(FQ, dtype=float)
CAL_START = T0 - K_CAL          # 165  (test expanding-CQR seed start)
VAL_CAL_START = 125             # validation expanding-CQR seed start
VAL_LO, VAL_HI = T0 - K_CAL, T0  # [165,205) honest selection window
P_BASE = 1.5                    # the winning single global power

# ── grids (all selected on pre-T0 val only) ──
REGIME_GRID = [dict(p=p, w=w, n_reg=nr, k0=k0)
               for p in (1.4, 1.5) for w in (3, 5) for nr in (2, 3) for k0 in (20, 50)]
HARM_GRID = [dict(p=p, n_harm=nh) for p in (1.4, 1.5) for nh in (1, 2)]
ASYM_GRID = [dict(p_lo=plo, p_hi=phi) for plo in (1.2, 1.3, 1.4)
             for phi in (1.5, 1.6, 1.7, 1.8)]


# ─────────────────────────── leak-free regime signal ───────────────────────────
def trailing_level(yf, w):
    """g[t] = mean(y[t-w:t]) using past y only (leak-free). NaN for t<MIN_CTX+1."""
    n = len(yf)
    g = np.full(n, np.nan)
    for t in range(MIN_CTX, n):
        lo = max(MIN_CTX, t - w)
        g[t] = yf[lo:t].mean() if t > lo else yf[t - 1]
    return g


# ─────────────────────────── span builders (Wspan,23) ───────────────────────────
def _blocks(ntot):
    for bstart in range(WSTART, ntot, REFIT_K):
        yield bstart, min(bstart + REFIT_K, ntot)


def build_span_regime(S, p, w, n_reg, k0):
    """Regime-conditional Qz, shrunk toward global by pseudo-count k0."""
    yf, tirex, cap, ntot = S["yf"], S["tirex"], S["cap"], S["ntot"]
    g = trailing_level(yf, w)
    W = np.arange(WSTART, ntot)
    Q = np.zeros((len(W), len(FQ)))
    qs = np.linspace(0.0, 1.0, n_reg + 1)[1:-1]     # interior split fractions
    for bstart, bend in _blocks(ntot):
        tr = np.arange(MIN_CTX, bstart - K_CAL)
        mu = np.clip(tirex[tr], EPS, None)
        z = (yf[tr] - tirex[tr]) / np.power(mu, p / 2.0)
        Qz_g = np.quantile(z, FQ)
        gtr = g[tr]
        thr = np.quantile(gtr[np.isfinite(gtr)], qs) if len(qs) else np.array([])
        lab_tr = np.searchsorted(thr, gtr)
        Qz_r = {}
        for r in range(n_reg):
            m = lab_tr == r
            nr = int(m.sum())
            base = np.quantile(z[m], FQ) if nr >= 5 else Qz_g
            Qz_r[r] = (nr * base + k0 * Qz_g) / (nr + k0)
        oi = np.arange(bstart, bend)
        mu_o = np.clip(tirex[oi], EPS, None)
        s_o = np.power(mu_o, p / 2.0)
        lab_o = np.searchsorted(thr, g[oi])
        rows = np.array([Qz_r.get(int(r), Qz_g) for r in lab_o])       # (nb,23)
        pe = mu_o[:, None] + rows * s_o[:, None]
        pe = np.clip(pe, 0.0, cap)
        pe.sort(axis=1)
        Q[oi - WSTART] = pe
    return Q


def build_span_harmonic(S, p, n_harm):
    """Calendar-conditional Qz via per-level QuantReg of z on seasonal harmonics."""
    from statsmodels.regression.quantile_regression import QuantReg
    yf, tirex, cap, ntot = S["yf"], S["tirex"], S["cap"], S["ntot"]
    t_all = np.arange(ntot)
    cols = [np.ones(ntot)]
    for h in range(1, n_harm + 1):
        cols.append(np.sin(2 * np.pi * h * t_all / 52.0))
        cols.append(np.cos(2 * np.pi * h * t_all / 52.0))
    H = np.column_stack(cols)                       # (ntot, 1+2*n_harm)
    W = np.arange(WSTART, ntot)
    Q = np.zeros((len(W), len(FQ)))
    for bstart, bend in _blocks(ntot):
        tr = np.arange(MIN_CTX, bstart - K_CAL)
        mu = np.clip(tirex[tr], EPS, None)
        z = (yf[tr] - tirex[tr]) / np.power(mu, p / 2.0)
        Qz_g = np.quantile(z, FQ)
        zlo, zhi = np.quantile(z, 0.005), np.quantile(z, 0.995)
        Htr = H[tr]
        oi = np.arange(bstart, bend)
        Hoi = H[oi]
        Qz_pred = np.zeros((len(oi), len(FQ)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j, tau in enumerate(FQ):
                try:
                    res = QuantReg(z, Htr).fit(q=float(tau), max_iter=200, p_tol=1e-4)
                    pv = np.asarray(res.predict(Hoi), dtype=float)
                    if not np.all(np.isfinite(pv)):
                        raise ValueError
                    Qz_pred[:, j] = np.clip(pv, zlo, zhi)
                except Exception:
                    Qz_pred[:, j] = Qz_g[j]
        mu_o = np.clip(tirex[oi], EPS, None)
        s_o = np.power(mu_o, p / 2.0)
        pe = mu_o[:, None] + Qz_pred * s_o[:, None]
        pe = np.clip(pe, 0.0, cap)
        pe.sort(axis=1)
        Q[oi - WSTART] = pe
    return Q


def build_span_asym(S, p_lo, p_hi):
    """Asymmetric power: p_lo scaling for tau<=0.5, p_hi scaling for tau>0.5."""
    yf, tirex, cap, ntot = S["yf"], S["tirex"], S["cap"], S["ntot"]
    W = np.arange(WSTART, ntot)
    Q = np.zeros((len(W), len(FQ)))
    lo_mask = FQ <= 0.5
    for bstart, bend in _blocks(ntot):
        tr = np.arange(MIN_CTX, bstart - K_CAL)
        mu = np.clip(tirex[tr], EPS, None)
        z_lo = (yf[tr] - tirex[tr]) / np.power(mu, p_lo / 2.0)
        z_hi = (yf[tr] - tirex[tr]) / np.power(mu, p_hi / 2.0)
        Qz_lo = np.quantile(z_lo, FQ)
        Qz_hi = np.quantile(z_hi, FQ)
        oi = np.arange(bstart, bend)
        mu_o = np.clip(tirex[oi], EPS, None)
        s_lo = np.power(mu_o, p_lo / 2.0)
        s_hi = np.power(mu_o, p_hi / 2.0)
        pe = np.where(lo_mask[None, :],
                      mu_o[:, None] + Qz_lo[None, :] * s_lo[:, None],
                      mu_o[:, None] + Qz_hi[None, :] * s_hi[:, None])
        pe = np.clip(pe, 0.0, cap)
        pe.sort(axis=1)
        Q[oi - WSTART] = pe
    return Q


def build_span_regime_asym(S, w, n_reg, k0, p_lo, p_hi):
    """Regime-conditional AND asymmetric: per regime, lo/hi power pools shrunk to global."""
    yf, tirex, cap, ntot = S["yf"], S["tirex"], S["cap"], S["ntot"]
    g = trailing_level(yf, w)
    W = np.arange(WSTART, ntot)
    Q = np.zeros((len(W), len(FQ)))
    qs = np.linspace(0.0, 1.0, n_reg + 1)[1:-1]
    lo_mask = FQ <= 0.5
    for bstart, bend in _blocks(ntot):
        tr = np.arange(MIN_CTX, bstart - K_CAL)
        mu = np.clip(tirex[tr], EPS, None)
        z_lo = (yf[tr] - tirex[tr]) / np.power(mu, p_lo / 2.0)
        z_hi = (yf[tr] - tirex[tr]) / np.power(mu, p_hi / 2.0)
        Qzlo_g, Qzhi_g = np.quantile(z_lo, FQ), np.quantile(z_hi, FQ)
        gtr = g[tr]
        thr = np.quantile(gtr[np.isfinite(gtr)], qs) if len(qs) else np.array([])
        lab_tr = np.searchsorted(thr, gtr)
        Rlo, Rhi = {}, {}
        for r in range(n_reg):
            m = lab_tr == r
            nr = int(m.sum())
            blo = np.quantile(z_lo[m], FQ) if nr >= 5 else Qzlo_g
            bhi = np.quantile(z_hi[m], FQ) if nr >= 5 else Qzhi_g
            Rlo[r] = (nr * blo + k0 * Qzlo_g) / (nr + k0)
            Rhi[r] = (nr * bhi + k0 * Qzhi_g) / (nr + k0)
        oi = np.arange(bstart, bend)
        mu_o = np.clip(tirex[oi], EPS, None)
        s_lo = np.power(mu_o, p_lo / 2.0)
        s_hi = np.power(mu_o, p_hi / 2.0)
        lab_o = np.searchsorted(thr, g[oi])
        rows_lo = np.array([Rlo.get(int(r), Qzlo_g) for r in lab_o])
        rows_hi = np.array([Rhi.get(int(r), Qzhi_g) for r in lab_o])
        pe = np.where(lo_mask[None, :],
                      mu_o[:, None] + rows_lo * s_lo[:, None],
                      mu_o[:, None] + rows_hi * s_hi[:, None])
        pe = np.clip(pe, 0.0, cap)
        pe.sort(axis=1)
        Q[oi - WSTART] = pe
    return Q


# ─────────────────────────── evaluation ───────────────────────────
def eval_span(Q, S, origins, cap, cal_start):
    B = rolling_cqr_bounds(Q, S, origins, cap, cal_start, None)
    med = Q[origins - WSTART][:, MED_COL]
    w = wis_of(B, S["yf"][origins], med)
    return w, B


def full_metrics(w, B, y, n, ref_wis, tw_wis):
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    peak = y >= PEAK_Y
    p_ref, d_ref = dm(w, ref_wis)
    p_tw, d_tw = dm(w, tw_wis)
    return dict(
        wis=round(float(w.mean()), 4),
        dm_p_vs_ref=float(p_ref), dm_meandiff_vs_ref=round(float(d_ref), 4),
        dm_p_vs_tweedie=float(p_tw), dm_meandiff_vs_tweedie=round(float(d_tw), 4),
        picp95=round(k / n, 4), k_of_n=f"{k}/{n}", cp95ci=[round(v, 4) for v in cp(k, n)],
        peak_picp95=round(float(covv[peak].mean()), 3), n_peak=int(peak.sum()),
        last34_wis=round(float(w[n - 34:].mean()), 4),
        mean_w95=round(float((hi95 - lo95).mean()), 3),
    )


def main():
    t0 = time.time()
    np.random.seed(42)
    S = setup()
    ntot = S["ntot"]
    cap = S["cap"]
    cap_train = 2.0 * float(S["yf"][:269].max())   # spotless train-only cap
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S["yf"][origins]
    peak = y >= PEAK_Y
    cal = np.arange(T0 - K_CAL, T0)
    r_full = S["yf"] - S["tirex"]
    val_origins = np.arange(VAL_LO, VAL_HI)
    y_val = S["yf"][val_origins]

    # ---- exact 2.4012 fair baseline (per-origin WIS for DM) ----
    qy_ref = tirex_empirical_qy(S["tirex"], r_full, origins, cap)
    cqr_ref = cqr_offsets(tirex_empirical_qy(S["tirex"], r_full, cal, cap), S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())

    # ---- Tweedie p=1.5 baseline reproduction (per-origin WIS for DM) ----
    _, QP_base, _ = build_span(S, P_BASE, "tirex")
    tw_w, tw_B = eval_span(QP_base, S, origins, cap, CAL_START)
    tw_mean = float(tw_w.mean())
    tw_lo, tw_hi = tw_B[0.05]
    tw_peakP = float(((y >= tw_lo) & (y <= tw_hi))[peak].mean())

    print(f"=== enriched Tweedie residual-SCALE head — {n} leak-free origins (weeks {T0}..{ntot-1}) ===")
    print(f"    2.4012 fair baseline reproduced: {ref_mean:.4f}")
    print(f"    Tweedie p=1.5 baseline reproduced: WIS={tw_mean:.4f}  PICP95={((y>=tw_lo)&(y<=tw_hi)).mean():.4f}"
          f"  peakP95={tw_peakP:.3f}  last34={float(tw_w[n-34:].mean()):.4f}")
    print(f"    TARGET: beat Tweedie {tw_mean:.4f} on TEST (config chosen on pre-T0 val [165,205) only)\n")

    rows = []

    def evaluate(name, family, Q):
        wv, _ = eval_span(Q, S, val_origins, cap, VAL_CAL_START)
        val_wis = float(wv.mean())
        wt, Bt = eval_span(Q, S, origins, cap, CAL_START)
        m = full_metrics(wt, Bt, y, n, ref_wis, tw_w)
        m.update(config=name, family=family, val_wis=round(val_wis, 4))
        rows.append(m)
        return m

    # baseline as a row (for reference)
    m0 = full_metrics(tw_w, tw_B, y, n, ref_wis, tw_w)
    wv0, _ = eval_span(QP_base, S, val_origins, cap, VAL_CAL_START)
    m0.update(config="tweedie_p1.5_global", family="baseline", val_wis=round(float(wv0.mean()), 4))
    rows.append(m0)

    print("  building (a) regime-conditional ...")
    for cfg in REGIME_GRID:
        Q = build_span_regime(S, cfg["p"], cfg["w"], cfg["n_reg"], cfg["k0"])
        evaluate(f"regime_p{cfg['p']}_w{cfg['w']}_r{cfg['n_reg']}_k{cfg['k0']}", "regime", Q)
    print("  building (b) harmonic quantile-scale ...")
    for cfg in HARM_GRID:
        Q = build_span_harmonic(S, cfg["p"], cfg["n_harm"])
        evaluate(f"harm_p{cfg['p']}_h{cfg['n_harm']}", "harmonic", Q)
    print("  building (c) asymmetric power ...")
    for cfg in ASYM_GRID:
        Q = build_span_asym(S, cfg["p_lo"], cfg["p_hi"])
        evaluate(f"asym_lo{cfg['p_lo']}_hi{cfg['p_hi']}", "asym", Q)

    # family val-winners
    fam_win = {}
    for fam in ("regime", "harmonic", "asym"):
        fr = [r for r in rows if r["family"] == fam]
        fam_win[fam] = min(fr, key=lambda r: r["val_wis"])

    # (d) combined regime x asym from val-selected knobs of (a) and (c)
    print("  building (d) combined regime x asymmetric (val-selected knobs) ...")
    rw = fam_win["regime"]["config"]           # regime_p{p}_w{w}_r{nr}_k{k0}
    parts = rw.split("_")
    w_sel = int(parts[2][1:]); nr_sel = int(parts[3][1:]); k0_sel = int(parts[4][1:])
    aw = fam_win["asym"]["config"]             # asym_lo{plo}_hi{phi}
    plo_sel = float(aw.split("_")[1][2:]); phi_sel = float(aw.split("_")[2][2:])
    Qd = build_span_regime_asym(S, w_sel, nr_sel, k0_sel, plo_sel, phi_sel)
    evaluate(f"combo_w{w_sel}_r{nr_sel}_k{k0_sel}_lo{plo_sel}_hi{phi_sel}", "combo", Qd)
    fam_win["combo"] = [r for r in rows if r["family"] == "combo"][0]

    # ---- print full table ----
    hdr = (f"{'config':>30s} | {'valWIS':>7s} | {'WIS':>7s} {'DMvsTw':>8s} {'DMvsRef':>8s} "
           f"{'PICP95':>7s} {'k/N':>7s} {'pkP95':>6s} {'l34':>7s} {'W95':>6s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["val_wis"]):
        beat = "*" if (r["wis"] < tw_mean and r["dm_p_vs_tweedie"] < 0.05) else \
               ("~" if r["wis"] < tw_mean else " ")
        print(f"{r['config']:>30s} | {r['val_wis']:>7.4f} | {r['wis']:>7.4f}{beat} "
              f"{r['dm_p_vs_tweedie']:>8.1e} {r['dm_p_vs_ref']:>8.1e} "
              f"{r['picp95']:>6.4f} {r['k_of_n']:>7s} {r['peak_picp95']:>6.3f} "
              f"{r['last34_wis']:>7.4f} {r['mean_w95']:>6.2f}")

    # ---- honest headline: argmin pre-T0 val WIS across ALL enriched configs ----
    enriched = [r for r in rows if r["family"] != "baseline"]
    headline = min(enriched, key=lambda r: r["val_wis"])
    beats_tw = bool(headline["wis"] < tw_mean and headline["dm_p_vs_tweedie"] < 0.05)
    beats_tw_point = bool(headline["wis"] < tw_mean)

    # spotless cap check on the headline span
    fam2fn = {}
    # rebuild the headline span deterministically for the cap check
    hc = headline["config"]
    if headline["family"] == "regime":
        pp = float(hc.split("_")[1][1:]); ww = int(hc.split("_")[2][1:])
        nn = int(hc.split("_")[3][1:]); kk = int(hc.split("_")[4][1:])
        Qh = build_span_regime(S, pp, ww, nn, kk)
    elif headline["family"] == "harmonic":
        pp = float(hc.split("_")[1][1:]); hh = int(hc.split("_")[2][1:])
        Qh = build_span_harmonic(S, pp, hh)
    elif headline["family"] == "asym":
        plo = float(hc.split("_")[1][2:]); phi = float(hc.split("_")[2][2:])
        Qh = build_span_asym(S, plo, phi)
    else:  # combo
        Qh = Qd
    wt_tr, Bt_tr = eval_span(Qh, S, origins, cap_train, CAL_START)
    cap_train_wis = round(float(wt_tr.mean()), 4)
    cap_binds = not (abs(cap_train_wis - headline["wis"]) < 1e-9)

    out = {
        "reference_wis_2p4012": round(ref_mean, 4),
        "tweedie_baseline_wis": round(tw_mean, 4),
        "tweedie_baseline_peak_picp95": round(tw_peakP, 3),
        "n": n, "n_peak": int(peak.sum()),
        "family_val_winners": {k: {kk: v[kk] for kk in
                                   ("config", "val_wis", "wis", "dm_p_vs_tweedie", "dm_p_vs_ref",
                                    "picp95", "k_of_n", "cp95ci", "peak_picp95", "last34_wis", "mean_w95")}
                               for k, v in fam_win.items()},
        "honest_headline": {k: headline[k] for k in
                            ("config", "family", "val_wis", "wis", "dm_p_vs_ref", "dm_p_vs_tweedie",
                             "dm_meandiff_vs_tweedie", "picp95", "k_of_n", "cp95ci",
                             "peak_picp95", "n_peak", "last34_wis", "mean_w95")},
        "headline_beats_tweedie_wis_and_dm": beats_tw,
        "headline_beats_tweedie_wis_point": beats_tw_point,
        "headline_cap_train_only_wis": cap_train_wis,
        "headline_cap_binds_train_only": bool(cap_binds),
        "rows": rows,
    }
    (ROOT / "scripts" / "_exp_enriched_scale.json").write_text(json.dumps(out, indent=2))

    print("\n--- FAMILY val-winners (honest, argmin pre-T0 val WIS within family) ---")
    for fam in ("regime", "harmonic", "asym", "combo"):
        v = fam_win[fam]
        b = "BEAT" if (v["wis"] < tw_mean and v["dm_p_vs_tweedie"] < 0.05) else \
            ("point<" if v["wis"] < tw_mean else "no")
        print(f"  {fam:>8s}: {v['config']:>30s}  TEST WIS={v['wis']:.4f} (Tw {tw_mean:.4f}, {b})  "
              f"DMvsTw={v['dm_p_vs_tweedie']:.3f}  PICP95={v['picp95']:.4f}  pk={v['peak_picp95']:.3f}  l34={v['last34_wis']:.4f}")

    print("\n=== HONEST HEADLINE (argmin pre-T0 val WIS across ALL enriched configs) ===")
    h = headline
    print(f"  {h['config']} [{h['family']}]  valWIS={h['val_wis']:.4f}")
    print(f"  TEST WIS      = {h['wis']:.4f}   (Tweedie {tw_mean:.4f}; delta {100*(h['wis']-tw_mean)/tw_mean:+.2f}%)")
    print(f"  DM p vs 2.4012 = {h['dm_p_vs_ref']:.2e}")
    print(f"  DM p vs Tweedie(2.2427) = {h['dm_p_vs_tweedie']:.4f}  (mean diff {h['dm_meandiff_vs_tweedie']:+.4f})")
    print(f"  PICP95        = {h['picp95']:.4f}  ({h['k_of_n']})  CP95 CI {h['cp95ci']}")
    print(f"  PEAK PICP95   = {h['peak_picp95']:.3f}  (Tweedie {tw_peakP:.3f}, n_peak={h['n_peak']})")
    print(f"  last34 WIS    = {h['last34_wis']:.4f}   mean-W95 = {h['mean_w95']:.3f}")
    print(f"  cap binds (train-only): {cap_binds}  (False = spotless)")
    print(f"\n  BEATS Tweedie (WIS< & DM p<0.05): {beats_tw}   (WIS-point-only: {beats_tw_point})")
    print(f"\nelapsed {time.time()-t0:.1f}s  wrote scripts/_exp_enriched_scale.json")


if __name__ == "__main__":
    raise SystemExit(main())
