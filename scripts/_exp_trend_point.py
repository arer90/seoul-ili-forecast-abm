#!/usr/bin/env python
"""OVERNIGHT EXPERIMENT — leak-free trend/momentum POINT correction for ramps.

DIAGNOSIS (from the Tweedie win 2.2427): peak coverage y>=50 ~0.87 is a SHARED failure —
the missed peak weeks are POINT UNDER-PREDICTIONS on the rising limb where the TiRex point
is too low, so BOTH the Tweedie and SPCI upper bounds fall short. Fixing it needs a BETTER
POINT on ramps, which would improve BOTH WIS and peak coverage.

This experiment corrects the POINT with a PAST-ONLY momentum/trend signal, then puts the SAME
Tweedie residual-scale interval (p*=1.5, EXPANDING split-CQR) around the CORRECTED point,
with residuals of the CORRECTED point recomputed past-only.

  corrected point:  p'_t = TiRex_t + kappa * trend_eff_t
    trend signals (all functions of PAST y only -> leak-free at a 1-step origin t):
      mom1    : y[t-1]-y[t-2]                          (1-step momentum)
      mom2    : 0.5*(y[t-1]-y[t-3])                    (2-step avg momentum)
      ols4    : slope of OLS on (y[t-4..t-1]) vs time  (short-window slope)
      posres4 : mean POSITIVE TiRex residual over last 4 wks (TiRex under-predicting recently)
      posres8 : same, last 8 wks
    gate:
      ungated : trend_eff = trend         (also lowers the point when trend<0)
      gated   : trend_eff = max(trend,0)  (raise ONLY on rising limbs; leave declines alone)
    kappa (and gate/trend) are chosen ONLY by argmin pre-T0 val WIS on [165,205)
      (val CQR sub-seed [125,165)) — NEVER the 132 test origins.  kappa=0 == pure-TiRex Tweedie
      (reproduces the 2.2427 anchor) and is the honest "no-correction" option in the same sweep.
    do-no-harm ONLINE blend: at each test origin pick corrected vs uncorrected(kappa=0) by which
      had the lower CUMULATIVE PAST per-origin WIS (buffer seeded on the val block [165,205)) —
      strictly leak-free; a positive online guard against ramp-overshoot variance.

  interval: Tweedie pearson residual-scale q = mu + Qz*mu^(p/2), mu=p'_t, Qz = empirical PAST
    standardized residuals of the CORRECTED point; p=1.5 fixed (the winning power); EXPANDING CQR.

Reference = the exact 2.4012 fair baseline (tirex_empirical_qy + build_bounds_cqr); every
candidate's per-origin WIS is DM-tested (HLN h=1) against it. cap = 2*max(y_full), with a
train-only cap (2*max y_train) cross-check on the headline. No live/pipeline or existing-script
edits — this is a NEW standalone script that only IMPORTS the reusable helpers.
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
                                      FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y)
from scripts.dec_boosted_mech_multiorigin import T0, REFIT_K
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup, dm, cp, wis_of, ALPHAS
from scripts._exp_tweedie import WSTART
from scripts._exp_tweedie_cover import rolling_cqr_bounds

EPS = 1e-6
P_STAR = 1.5                                # the winning Tweedie variance power (fixed a-priori)
CAL_START = T0 - K_CAL                      # 165: test CQR expanding start (== static seed @ first origin)
VAL_LO, VAL_HI = T0 - K_CAL, T0            # val origins [165,205)
VAL_CAL_START = WSTART                      # 125: val CQR sub-seed start

KAPPAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
TREND_TYPES = ("mom1", "mom2", "ols4", "posres4", "posres8")
GATES = ("ungated", "gated")


# ───────────────────────────── trend signals (past-only) ─────────────────────────────
def trend_signals(yf, tirex):
    """All PAST-only trend arrays over weeks 0..ntot-1 (0 where the required lag is unavailable).

    Every entry trend[t] uses only y (and TiRex) at indices < t, so p'_t = TiRex_t + k*trend[t]
    is leak-free for a 1-step-ahead forecast of y_t (which knows y_{<t} and the TiRex point at t).
    """
    n = len(yf)
    r = yf - tirex                                   # TiRex residual (nan for t<MIN_CTX)
    out = {}

    mom1 = np.zeros(n)
    mom1[2:] = yf[1:n - 1] - yf[0:n - 2]             # y[t-1]-y[t-2]
    out["mom1"] = mom1

    mom2 = np.zeros(n)
    mom2[3:] = 0.5 * (yf[2:n - 1] - yf[0:n - 3])     # 0.5*(y[t-1]-y[t-3])
    out["mom2"] = mom2

    ols4 = np.zeros(n)                               # OLS slope over y[t-4..t-1], x=[0,1,2,3]
    for t in range(4, n):                            # denom sum((x-1.5)^2)=5.0
        ols4[t] = (-1.5 * yf[t - 4] - 0.5 * yf[t - 3] + 0.5 * yf[t - 2] + 1.5 * yf[t - 1]) / 5.0
    out["ols4"] = ols4

    for w, key in ((4, "posres4"), (8, "posres8")):
        pr = np.zeros(n)
        for t in range(1, n):
            u = r[max(0, t - w):t]
            u = u[np.isfinite(u)]
            pr[t] = float(np.mean(np.clip(u, 0.0, None))) if u.size else 0.0
        out[key] = pr
    return out


def corrected_point(tirex, trend, kappa, gate):
    te = np.maximum(trend, 0.0) if gate == "gated" else trend
    return tirex + kappa * te


# ─────────────── Tweedie pearson residual-scale span around an ARBITRARY point ───────────────
def build_span_point(S, p, point_series):
    """Per-block leak-free Tweedie pearson quantiles built around `point_series` (not TiRex).

    For each REFIT_K-block starting at bstart, the head is calibrated only on strictly-past weeks
    tr=[MIN_CTX, bstart-K_CAL): dispersion phi = mean((y-mu)^2 / mu^p) and standardized-residual
    quantiles Qz = quantile((y-mu)/mu^(p/2), FQ) with mu=point_series on tr. Origin quantiles
    q = mu_o + Qz*mu_o^(p/2), mu_o=point_series at the block origins. Returns QP indexed week-WSTART
    (same layout as scripts._exp_tweedie.build_span so rolling_cqr_bounds consumes it unchanged).
    """
    cap, yf, ntot = S["cap"], S["yf"], S["ntot"]
    W = np.arange(WSTART, ntot)
    QP = np.zeros((len(W), len(FQ)))
    for bstart in range(WSTART, ntot, REFIT_K):
        bend = min(bstart + REFIT_K, ntot)
        train_end = bstart - K_CAL
        tr = np.arange(MIN_CTX, train_end)
        mu_tr = point_series[tr]
        ytr = yf[tr]
        mu_tr_c = np.clip(mu_tr, EPS, None)
        phi = max(float(np.mean((ytr - mu_tr) ** 2 / np.power(mu_tr_c, p))), 1e-6)
        s_tr = np.power(mu_tr_c, p / 2.0)
        Qz = np.quantile((ytr - mu_tr) / s_tr, FQ)
        oi = np.arange(bstart, bend)
        mu_o = np.clip(point_series[oi], EPS, None)
        s_o = np.power(mu_o, p / 2.0)
        pe = mu_o[:, None] + Qz[None, :] * s_o[:, None]
        pe = np.clip(pe, 0.0, cap)
        pe.sort(axis=1)
        QP[oi - WSTART] = pe
    return QP


def eval_span(QP, S, origins, cap, cal_start):
    """Expanding-CQR bounds + per-origin WIS + median for a point-Tweedie span."""
    B = rolling_cqr_bounds(QP, S, origins, cap, cal_start, None)
    med = QP[origins - WSTART][:, MED_COL]
    w = wis_of(B, S["yf"][origins], med)
    return B, med, w


def full_metrics(w, B, y, ref_wis, n):
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    peak = y >= PEAK_Y
    p_dm, dbar = dm(w, ref_wis)
    return dict(
        wis=round(float(w.mean()), 4), dm_p=float(p_dm), dm_meandiff=round(float(dbar), 4),
        picp95=round(k / n, 4), k_of_n=f"{k}/{n}", cp95ci=[round(v, 4) for v in cp(k, n)],
        peak_picp95=round(float(covv[peak].mean()), 3), n_peak=int(peak.sum()),
        last34_wis=round(float(w[n - 34:].mean()), 4),
        mean_w95=round(float((hi95 - lo95).mean()), 3),
    )


def main():
    t0 = time.time()
    S = setup()
    ntot = S["ntot"]
    cap_full = S["cap"]
    cap_train = 2.0 * float(S["yf"][:269].max())          # train-only spotless cap
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S["yf"][origins]
    peak_mask = y >= PEAK_Y
    val_origins = np.arange(VAL_LO, VAL_HI)
    y_val = S["yf"][val_origins]
    cal = np.arange(T0 - K_CAL, T0)
    r_full = S["yf"] - S["tirex"]

    # ---- exact 2.4012 reference (per-origin WIS for DM) ----
    qy_ref = tirex_empirical_qy(S["tirex"], r_full, origins, cap_full)
    cqr_ref = cqr_offsets(tirex_empirical_qy(S["tirex"], r_full, cal, cap_full), S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap_full)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())
    ref_cov = (y >= ref_B[0.05][0]) & (y <= ref_B[0.05][1])
    ref_peak = round(float(ref_cov[peak_mask].mean()), 3)

    trends = trend_signals(S["yf"], S["tirex"])

    # ---- kappa=0 anchor (pure-TiRex Tweedie; must reproduce 2.2427) ----
    QP0 = build_span_point(S, P_STAR, S["tirex"])
    B0, med0, w0 = eval_span(QP0, S, origins, cap_full, CAL_START)
    _, _, w0_val = eval_span(QP0, S, val_origins, cap_full, VAL_CAL_START)
    anchor = full_metrics(w0, B0, y, ref_wis, n)
    anchor_val = round(float(w0_val.mean()), 4)

    print("=" * 96)
    print(f"REFERENCE fair baseline TiRex+CQR : WIS={ref_mean:.4f}  peakPICP95={ref_peak}  "
          f"last34={float(ref_wis[n-34:].mean()):.4f}")
    print(f"kappa=0 anchor (pure-TiRex Tweedie): WIS={anchor['wis']:.4f}  peakPICP95={anchor['peak_picp95']}  "
          f"PICP95={anchor['picp95']} ({anchor['k_of_n']})  last34={anchor['last34_wis']}  valWIS={anchor_val}")
    print(f"    (anchor should reproduce the verified 2.2427 / peak 0.87 / last34 2.6491)")
    print("=" * 96)
    print(f"{'config':>24s} | {'valWIS':>7s} | {'WIS':>7s} {'DMp':>9s} {'d%':>6s} "
          f"{'PICP95':>7s} {'peakP95':>7s} {'last34':>7s}")
    print("-" * 96)

    rows = []
    per_origin = {}                                       # config -> (w_test, val_wis, B)
    # kappa=0 anchor as an entry
    rows.append(dict(config="kappa0", trend="none", gate="none", kappa=0.0,
                     val_wis=anchor_val, **anchor))
    per_origin["kappa0"] = (w0, anchor_val, B0)
    print(f"{'kappa0':>24s} | {anchor_val:>7.4f} | {anchor['wis']:>7.4f} {anchor['dm_p']:>9.2e} "
          f"{100*(anchor['wis']-ref_mean)/ref_mean:>6.1f} {anchor['picp95']:>7.4f} "
          f"{anchor['peak_picp95']:>7.3f} {anchor['last34_wis']:>7.4f}")

    for tt in TREND_TYPES:
        for gate in GATES:
            for kap in KAPPAS:
                if kap == 0.0:
                    continue                              # anchor already logged
                pser = corrected_point(S["tirex"], trends[tt], kap, gate)
                QP = build_span_point(S, P_STAR, pser)
                B, med, w = eval_span(QP, S, origins, cap_full, CAL_START)
                _, _, wv = eval_span(QP, S, val_origins, cap_full, VAL_CAL_START)
                val_wis = round(float(wv.mean()), 4)
                m = full_metrics(w, B, y, ref_wis, n)
                name = f"{tt}_{gate}_k{kap}"
                rows.append(dict(config=name, trend=tt, gate=gate, kappa=kap,
                                 val_wis=val_wis, **m))
                per_origin[name] = (w, val_wis, B)
                sig = "*" if (m["wis"] < ref_mean and m["dm_p"] < 0.05) else " "
                print(f"{name:>24s} | {val_wis:>7.4f} | {m['wis']:>7.4f}{sig}{m['dm_p']:>8.2e} "
                      f"{100*(m['wis']-ref_mean)/ref_mean:>6.1f} {m['picp95']:>7.4f} "
                      f"{m['peak_picp95']:>7.3f} {m['last34_wis']:>7.4f}")

    # ───── HONEST headline: argmin pre-T0 val WIS across the WHOLE sweep (kappa=0 included) ─────
    headline = min(rows, key=lambda r: r["val_wis"])

    # ───── do-no-harm ONLINE blend: best CORRECTED (kappa>0, lowest val WIS) vs kappa=0 anchor ─
    corr_rows = [r for r in rows if r["kappa"] > 0.0]
    best_corr = min(corr_rows, key=lambda r: r["val_wis"])
    wc, wc_val_mean, Bc = per_origin[best_corr["config"]]
    # per-origin val WIS for the two configs (to seed the online decision buffer)
    pser_c = corrected_point(S["tirex"], trends[best_corr["trend"]], best_corr["kappa"], best_corr["gate"])
    QPc = build_span_point(S, P_STAR, pser_c)
    _, _, wc_val = eval_span(QPc, S, val_origins, cap_full, VAL_CAL_START)
    buffer = list(wc_val - w0_val)                        # negative => corrected better (on val)
    lo95_c, hi95_c = Bc[0.05]
    lo95_0, hi95_0 = B0[0.05]
    blend_w = np.empty(n)
    blend_lo = np.empty(n); blend_hi = np.empty(n)
    n_corr_used = 0
    for j in range(n):
        use_corr = float(np.mean(buffer)) < 0.0           # decision uses ONLY past ([165,t))
        if use_corr:
            blend_w[j] = wc[j]; blend_lo[j] = lo95_c[j]; blend_hi[j] = hi95_c[j]; n_corr_used += 1
        else:
            blend_w[j] = w0[j]; blend_lo[j] = lo95_0[j]; blend_hi[j] = hi95_0[j]
        buffer.append(wc[j] - w0[j])                       # append AFTER deciding (y_t now past)
    blend_cov = (y >= blend_lo) & (y <= blend_hi)
    blend_k = int(blend_cov.sum())
    blend_p, blend_d = dm(blend_w, ref_wis)
    blend = dict(config="donoharm_online", base=best_corr["config"], n_corr_used=n_corr_used,
                 wis=round(float(blend_w.mean()), 4), dm_p=float(blend_p),
                 dm_meandiff=round(float(blend_d), 4),
                 picp95=round(blend_k / n, 4), k_of_n=f"{blend_k}/{n}",
                 cp95ci=[round(v, 4) for v in cp(blend_k, n)],
                 peak_picp95=round(float(blend_cov[peak_mask].mean()), 3),
                 last34_wis=round(float(blend_w[n - 34:].mean()), 4))

    # ───── train-only cap cross-check on the headline ─────
    if headline["config"] == "kappa0":
        pser_h = S["tirex"]
    else:
        pser_h = corrected_point(S["tirex"], trends[headline["trend"]], headline["kappa"], headline["gate"])
    QPh = build_span_point(S, P_STAR, pser_h)
    Bht, medht, wht = eval_span(QPh, S, origins, cap_train, CAL_START)
    m_train = full_metrics(wht, Bht, y, ref_wis, n)
    cap_binds = not (abs(headline["wis"] - m_train["wis"]) < 1e-9 and headline["picp95"] == m_train["picp95"])

    beats_tweedie = bool(headline["wis"] < anchor["wis"])
    point_helps_peaks = bool((headline["peak_picp95"] > anchor["peak_picp95"]) and
                             (headline["wis"] <= anchor["wis"]))

    print("\n" + "=" * 96)
    print("HONEST HEADLINE (argmin pre-T0 val WIS, weeks 165..204):")
    h = headline
    print(f"  config        = {h['config']}  (trend={h['trend']} gate={h['gate']} kappa={h['kappa']})")
    print(f"  WIS           = {h['wis']:.4f}   (ref 2.4012 delta {100*(h['wis']-ref_mean)/ref_mean:+.1f}%; "
          f"anchor 2.2427 delta {100*(h['wis']-anchor['wis'])/anchor['wis']:+.2f}%)")
    print(f"  DM p vs 2.4012= {h['dm_p']:.3e}   (mean per-origin WIS diff {h['dm_meandiff']:+.4f})")
    print(f"  PICP95        = {h['picp95']:.4f}  ({h['k_of_n']})  CP95 CI {h['cp95ci']}")
    print(f"  PEAK PICP95   = {h['peak_picp95']:.3f}  (vs Tweedie anchor 0.87; ref {ref_peak}; n_peak={h['n_peak']})")
    print(f"  last34 WIS    = {h['last34_wis']:.4f}")
    print(f"  train-cap chk : WIS={m_train['wis']:.4f} PICP95={m_train['picp95']:.4f}  cap_binds={cap_binds} (False=spotless)")
    print(f"  beats Tweedie anchor (2.2427)? {beats_tweedie}   point-correction helps peaks (peak up & WIS not worse)? {point_helps_peaks}")

    print("\ndo-no-harm ONLINE blend (best corrected vs kappa=0, past-cumulative-WIS switch):")
    print(f"  base corrected= {blend['base']}   corrected used on {blend['n_corr_used']}/{n} test origins")
    print(f"  WIS           = {blend['wis']:.4f}   DM p vs 2.4012 = {blend['dm_p']:.3e}")
    print(f"  PICP95        = {blend['picp95']:.4f} ({blend['k_of_n']})   PEAK PICP95 = {blend['peak_picp95']}   "
          f"last34 = {blend['last34_wis']}")

    # verdict: any corrected config that beats the anchor WIS AND lifts peak coverage, DM-sig
    beat_anchor = [r for r in rows if r["kappa"] > 0 and r["wis"] < anchor["wis"]
                   and r["peak_picp95"] > anchor["peak_picp95"] and r["dm_p"] < 0.05]
    print("\nCORRECTED configs beating the anchor WIS AND lifting peak cov AND DM-sig vs 2.4012:")
    print(f"  {[r['config'] for r in beat_anchor] or 'NONE'}")

    out = {
        "reference_wis": round(ref_mean, 4), "reference_peak_picp95": ref_peak, "n": n,
        "p_star": P_STAR, "kappa0_anchor": {k: v for k, v in anchor.items()},
        "kappa0_val_wis": anchor_val,
        "headline": headline, "headline_train_cap": m_train, "cap_binds": cap_binds,
        "beats_tweedie_anchor": beats_tweedie, "point_helps_peaks": point_helps_peaks,
        "donoharm_online": blend,
        "corrected_beat_anchor_and_peak_and_dmsig": [r["config"] for r in beat_anchor],
        "rows": rows,
    }
    (ROOT / "scripts" / "_exp_trend_point.json").write_text(json.dumps(out, indent=2))
    print(f"\nelapsed {time.time()-t0:.1f}s  wrote scripts/_exp_trend_point.json")


if __name__ == "__main__":
    raise SystemExit(main())
