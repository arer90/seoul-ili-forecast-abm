#!/usr/bin/env python
"""OVERNIGHT EXPERIMENT — TimesFM+TiRex ENSEMBLE POINT to fix ramp under-prediction.

Diagnosis carried in from the Tweedie head: the ONE remaining limitation is peak
coverage (y>=50) ~0.87 — a SHARED failure whose root cause is POINT under-prediction on
the rising limb (the TiRex 1-step point is too low on ramps, so BOTH the Tweedie and SPCI
upper bounds fall short). The fix must be a BETTER POINT on ramps, which lifts BOTH WIS
and peak coverage.

Empirical precondition (weeks 52..336, verified before this script): on the 26 peak weeks
(y>=50) TiRex under-predicts by mean -2.76 while TimesFM under-predicts by only -0.51
(TimesFM higher on 18/26); on strong ramps (rising & y>=30) TiRex -8.38 vs TimesFM -6.73.
TimesFM is noisier overall (MAE 2.35 vs 1.96) but LESS biased on peaks -> an ensemble that
leans toward TimesFM on the rising limb should reduce peak under-prediction.

POINT models (both leak-free rolling 1-step; the point at origin t uses y<t only):
  * TiRex   = S["tirex"] (official pipeline refit == rolling 1-step, verified reproducible)
  * TimesFM = TimesFM-2.5 rolling 1-step over weeks [52,337), cached to scripts/_timesfm_pool.npz.
              Cross-checked against per_model_optimal/TimesFM-2.5.json refit_test_predictions
              (weeks 269..336): corr=1.0000, MAE=0.000 -> identical, context is correct.

ENSEMBLE point (the ONLY new knob, chosen ONLY on pre-T0 val origins [165,205)):
  (a) convex   : pt = w*TiRex + (1-w)*TimesFM,  w in {0,.2,.35,.5,.65,.8,1}
  (b) ramp-gate: pt = w_ramp*TiRex + (1-w_ramp)*TimesFM  when a PAST-ONLY trend signal says
                 'ramp' (slope = y[t-1]-y[t-1-L] > thr), else w_base*TiRex+(1-w_base)*TimesFM.
                 (w_base, w_ramp, L, thr) grid; ramp uses less TiRex (=more TimesFM) on the limb.

INTERVAL: the SAME Tweedie residual-scale head as the winning config, put on the ENSEMBLE
point: q = mu + Qz*mu^(p/2), p*=1.5 (a-priori, inherited — the only tuned scalar of the
Tweedie campaign, itself picked pre-T0), Qz = empirical PAST standardized-residual quantiles
of the ENSEMBLE point, mu^(p/2) heteroscedastic scaling. EXPANDING split-CQR (window=None)
seeded at week 165 (test) / 125 (val). Per-block head trained only on weeks strictly before
each origin (train_end = block_start - K_CAL). cap = 2*max(y_full) (== reference) and a
spotless train-only cap = 2*max(y_train) cross-check.

Reference = the exact 2.4012 fair baseline (TiRex point + empirical past-residual FLUSIGHT
quantiles + static CQR). Per-origin WIS DM-tested (HLN h=1). Selection of w / gate is argmin
pre-T0 val WIS on [165,205) — NEVER test. No live/pipeline or existing-script edits.
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

from scripts.dec_boosted_mech import (cqr_offsets, build_bounds_cqr, MED_COL,
                                       K_CAL, MIN_CTX, MAX_CONTEXT, PEAK_Y)
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup, dm, cp, wis_of
from scripts._exp_tweedie import build_span, WSTART
from scripts._exp_tweedie_cover import rolling_cqr_bounds, static_bounds

P_STAR = 1.5                       # a-priori inherited Tweedie var-power (pre-T0 selected)
CAL_START = T0 - K_CAL             # 165 (test expanding-CQR seed)
VAL_CAL_START = 125                # val expanding-CQR seed
VAL_LO, VAL_HI = T0 - K_CAL, T0    # [165,205) pre-T0 validation origins
W_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)   # w on TiRex; 0=pure TimesFM, 1=pure TiRex
TF_CACHE = ROOT / "scripts" / "_timesfm_pool.npz"

# ramp-gate grid (all past-only)
GATE_LAGS = (2, 3, 4)                       # slope horizon L
GATE_THR = (1.0, 2.0, 4.0, 8.0)             # slope threshold on y-units
GATE_WBASE = (0.65, 0.8, 1.0)               # TiRex weight off-ramp
GATE_WRAMP = (0.0, 0.2, 0.35, 0.5)          # TiRex weight on-ramp (lower => more TimesFM)


# ─────────────────────── TimesFM rolling pool (cached) ───────────────────────
def load_or_build_timesfm(yf: np.ndarray, ntot: int) -> np.ndarray:
    """Rolling 1-step TimesFM point aligned to y_full index (nan for weeks < MIN_CTX).

    Leak-free: forecast for week t uses context y_full[max(0,t-MAX_CONTEXT):t] only.
    """
    if TF_CACHE.exists():
        d = np.load(TF_CACHE)
        weeks, preds = d["weeks"], d["preds"]
    else:
        from simulation.models.timesfm_wrapper import TimesFMForecaster
        f = TimesFMForecaster(max_context=MAX_CONTEXT, max_horizon=4)
        weeks = np.arange(MIN_CTX, ntot)
        preds = np.empty(len(weeks))
        for k, t in enumerate(weeks):
            f.fit_series(yf[max(0, t - MAX_CONTEXT):t].astype("float32"))
            preds[k] = float(f.forecast(1)[0])
        np.savez(TF_CACHE, weeks=weeks, preds=preds, max_context=MAX_CONTEXT)
    tf = np.full(ntot, np.nan)
    tf[weeks.astype(int)] = preds
    return tf


def crosscheck_timesfm(tf: np.ndarray, ntr: int, ntot: int) -> dict:
    p = ROOT / "simulation/results/per_model_optimal/TimesFM-2.5.json"
    refit = np.asarray(json.loads(p.read_text())["refit_test_predictions"], dtype=float)
    roll = tf[ntr:ntot]
    ok = len(roll) == len(refit)
    corr = float(np.corrcoef(roll, refit)[0, 1]) if ok else float("nan")
    mae = float(np.abs(roll - refit).mean()) if ok else float("nan")
    return {"n": int(len(roll)), "first_roll": round(float(roll[0]), 4),
            "first_refit": round(float(refit[0]), 4), "corr": round(corr, 4),
            "mae_vs_refit": round(mae, 4), "wildly_off": bool((not ok) or corr < 0.9 or mae > 5)}


# ─────────────────────── ensemble point constructors ───────────────────────
def convex_point(tirex, tf, w):
    """pt = w*TiRex + (1-w)*TimesFM (nan preserved where either is nan)."""
    return w * tirex + (1.0 - w) * tf


def gate_point(yf, tirex, tf, wbase, wramp, lag, thr):
    """Rising-limb-gated blend. slope_t = y[t-1]-y[t-1-lag] uses y<t only (past-only).
    On ramp (slope>thr) use wramp*TiRex+(1-wramp)*TimesFM (more TimesFM); else wbase blend."""
    n = len(yf)
    wv = np.full(n, wbase, dtype=float)
    for t in range(n):
        j0, j1 = t - 1, t - 1 - lag
        if j1 >= 0 and np.isfinite(yf[j0]) and np.isfinite(yf[j1]):
            if (yf[j0] - yf[j1]) > thr:
                wv[t] = wramp
    return wv * tirex + (1.0 - wv) * tf


# ─────────────────────────── head + evaluation ───────────────────────────
def head_span(S, point, p):
    """Tweedie residual-scale (pearson) FLUSIGHT quantiles around `point`. Returns QP."""
    S2 = dict(S)
    S2["tirex"] = point
    _QG, QP, _ = build_span(S2, p, "tirex")
    return QP, S2


def eval_point(S, point, origins, cal_start, cap, p=P_STAR):
    QP, S2 = head_span(S, point, p)
    med = QP[origins - WSTART][:, MED_COL]
    B = rolling_cqr_bounds(QP, S2, origins, cap, cal_start, None)
    y = S["yf"][origins]
    w = wis_of(B, y, med)
    return w, B, med, QP, S2


def full_metrics(w, B, y, ref_wis, n):
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    p_dm, dbar = dm(w, ref_wis)
    peak = y >= PEAK_Y
    return dict(
        wis=round(float(w.mean()), 4), dm_p=float(p_dm), dm_meandiff=round(float(dbar), 4),
        picp95=round(k / n, 4), k=k, k_of_n=f"{k}/{n}", cp95ci=[round(v, 4) for v in cp(k, n)],
        last34_wis=round(float(w[n - 34:].mean()), 4),
        mean_w95=round(float((hi95 - lo95).mean()), 3),
        peak_picp95=round(float(covv[peak].mean()), 4), n_peak=int(peak.sum()),
        peak_k=int(covv[peak].sum()))


def point_peak_bias(point, yf, origins):
    """Mean signed point error on peak (y>=50) test origins (negative = under-predict)."""
    y = yf[origins]
    peak = y >= PEAK_Y
    e = point[origins] - y
    return round(float(e[peak].mean()), 4), round(float(np.abs(e[peak]).mean()), 4)


def main():
    t0 = time.time()
    S = setup()
    ntot = S["ntot"]
    ntr = ntot - 68
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S["yf"][origins]
    cal = np.arange(T0 - K_CAL, T0)
    r_full = S["yf"] - S["tirex"]
    cap_full = S["cap"]                                   # 2*max(y_full) == reference
    cap_train = 2.0 * float(S["yf"][:ntr].max())          # 2*max(y_train), spotless

    tirex = S["tirex"]
    tf = load_or_build_timesfm(S["yf"], ntot)
    cc = crosscheck_timesfm(tf, ntr, ntot)

    # ---- exact 2.4012 reference (per-origin WIS for DM) ----
    qy_ref = tirex_empirical_qy(tirex, r_full, origins, cap_full)
    cqr_ref = cqr_offsets(tirex_empirical_qy(tirex, r_full, cal, cap_full), S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap_full)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())
    ref_k = int(((y >= ref_B[0.05][0]) & (y <= ref_B[0.05][1])).sum())
    ref_peak = ((y >= PEAK_Y) & (y >= ref_B[0.05][0]) & (y <= ref_B[0.05][1]))
    ref_peakcov = float(ref_peak.sum() / (y >= PEAK_Y).sum())

    val_origins = np.arange(VAL_LO, VAL_HI)
    y_val = S["yf"][val_origins]

    print("=" * 92)
    print(f"REFERENCE fair baseline TiRex+CQR: WIS={ref_mean:.4f}  PICP95={ref_k/n:.4f} ({ref_k}/{n})  "
          f"peakPICP95={ref_peakcov:.3f}  last34={float(ref_wis[n-34:].mean()):.4f}")
    print(f"TimesFM rolling cross-check vs refit_test_predictions: corr={cc['corr']} "
          f"MAE={cc['mae_vs_refit']} first(roll/refit)={cc['first_roll']}/{cc['first_refit']} "
          f"wildly_off={cc['wildly_off']}")
    print("=" * 92)

    # ── (a) convex sweep: honest w-selection on pre-T0 val ─────────────────
    print(f"\n(a) CONVEX  w*TiRex+(1-w)*TimesFM   Tweedie p={P_STAR} expanding-CQR   "
          f"[selection = argmin val WIS on {VAL_LO}..{VAL_HI-1}]")
    hdr = (f"{'w':>5s} | {'valWIS':>7s} | {'WIS':>7s} {'DMp':>8s} {'d%':>6s} {'PICP95':>7s} {'k/N':>7s} "
           f"{'pkP95':>6s} {'l34':>7s} {'pkBias':>7s}")
    print(hdr); print("-" * len(hdr))
    conv_rows = []
    for w in W_GRID:
        pt = convex_point(tirex, tf, w)
        wv, _, _, _, _ = eval_point(S, pt, val_origins, VAL_CAL_START, cap_full)
        val_wis = float(wv.mean())
        wt, Bt, medt, _, _ = eval_point(S, pt, origins, CAL_START, cap_full)
        m = full_metrics(wt, Bt, y, ref_wis, n)
        pb, pmae = point_peak_bias(pt, S["yf"], origins)
        m.update(w=w, val_wis=round(val_wis, 4), peak_bias=pb, peak_mae=pmae, kind="convex")
        conv_rows.append(m)
        dpct = 100 * (m["wis"] - ref_mean) / ref_mean
        sig = "*" if (m["wis"] < ref_mean and m["dm_p"] < 0.05) else " "
        print(f"{w:>5.2f} | {val_wis:>7.4f} | {m['wis']:>7.4f}{sig} {m['dm_p']:>8.2e} {dpct:>6.1f} "
              f"{m['picp95']:>7.4f} {m['k_of_n']:>7s} {m['peak_picp95']:>6.3f} {m['last34_wis']:>7.4f} {pb:>7.2f}")
    conv_pick = min(conv_rows, key=lambda r: r["val_wis"])

    # ── (b) ramp-gate sweep: honest selection on pre-T0 val ────────────────
    print(f"\n(b) RAMP-GATE  (past-only slope=y[t-1]-y[t-1-L]>thr -> more TimesFM)   "
          f"[selection = argmin val WIS on {VAL_LO}..{VAL_HI-1}]")
    gate_rows = []
    for lag in GATE_LAGS:
        for thr in GATE_THR:
            for wb in GATE_WBASE:
                for wr in GATE_WRAMP:
                    if wr >= wb:
                        continue  # ramp must use MORE TimesFM (less TiRex) than base
                    pt = gate_point(S["yf"], tirex, tf, wb, wr, lag, thr)
                    wv, _, _, _, _ = eval_point(S, pt, val_origins, VAL_CAL_START, cap_full)
                    gate_rows.append(dict(kind="gate", lag=lag, thr=thr, wbase=wb, wramp=wr,
                                          val_wis=round(float(wv.mean()), 4), _pt=pt))
    gate_pick = min(gate_rows, key=lambda r: r["val_wis"])
    # evaluate the honestly-picked gate on test
    gpt = gate_pick.pop("_pt")
    wt, Bt, medt, _, _ = eval_point(S, gpt, origins, CAL_START, cap_full)
    mg = full_metrics(wt, Bt, y, ref_wis, n)
    pb, pmae = point_peak_bias(gpt, S["yf"], origins)
    mg.update(gate_pick, peak_bias=pb, peak_mae=pmae)
    for r in gate_rows:
        r.pop("_pt", None)
    print(f"  best-val gate: lag={gate_pick['lag']} thr={gate_pick['thr']} wbase={gate_pick['wbase']} "
          f"wramp={gate_pick['wramp']}  valWIS={gate_pick['val_wis']}")
    dpct = 100 * (mg["wis"] - ref_mean) / ref_mean
    print(f"    TEST: WIS={mg['wis']} (d%{dpct:+.1f}) DMp={mg['dm_p']:.2e} PICP95={mg['picp95']} {mg['k_of_n']} "
          f"pkP95={mg['peak_picp95']} last34={mg['last34_wis']} pkBias={pb}")

    # ── HEADLINE: pick the honest val-winner across BOTH families ──────────
    cand = [{**conv_pick}, {**mg}]
    headline = min(cand, key=lambda r: r["val_wis"])

    # spotless train-only cap re-check on headline
    if headline["kind"] == "convex":
        hpt = convex_point(tirex, tf, headline["w"])
    else:
        hpt = gate_point(S["yf"], tirex, tf, headline["wbase"], headline["wramp"],
                         headline["lag"], headline["thr"])
    wtr, Btr, medtr, _, _ = eval_point(S, hpt, origins, CAL_START, cap_train)
    m_train = full_metrics(wtr, Btr, y, ref_wis, n)
    cap_binds = not (abs(headline["wis"] - m_train["wis"]) < 1e-9 and
                     headline["picp95"] == m_train["picp95"])

    # expand==static at first origin sanity
    QPh, S2h = head_span(S, hpt, P_STAR)
    B_static = static_bounds(QPh, S2h, origins, cap_full, CAL_START)
    B_full = rolling_cqr_bounds(QPh, S2h, origins, cap_full, CAL_START, None)
    first_match = bool(np.isclose(B_full[0.05][0][0], B_static[0.05][0][0]) and
                       np.isclose(B_full[0.05][1][0], B_static[0.05][1][0]))

    bars = dict(beats_wis=bool(headline["wis"] < ref_mean),
                dm_sig=bool(headline["dm_p"] < 0.05),
                picp_in_band=bool(0.93 <= headline["picp95"] <= 0.96),
                last34_lt_272=bool(headline["last34_wis"] < 2.72),
                peak_ge_tweedie=bool(headline["peak_picp95"] >= 0.87))
    point_helps_peaks = bool(headline["peak_bias"] > -2.762 and
                             headline["peak_picp95"] >= 0.87)

    if headline["kind"] == "convex":
        best_config = f"ens_convex_w{headline['w']}_tweedie_p{P_STAR}_expand"
    else:
        best_config = (f"ens_gate_L{headline['lag']}_thr{headline['thr']}_wb{headline['wbase']}"
                       f"_wr{headline['wramp']}_tweedie_p{P_STAR}_expand")

    print("\n" + "=" * 92)
    print(f"HEADLINE (honest val-winner across convex+gate): {best_config}")
    print(f"  WIS          = {headline['wis']:.4f}   (reference {ref_mean:.4f}; "
          f"delta {100*(headline['wis']-ref_mean)/ref_mean:+.1f}%)")
    print(f"  DM p vs 2.4012 = {headline['dm_p']:.3e}   (mean per-origin WIS diff = {headline['dm_meandiff']:+.4f})")
    print(f"  PICP95       = {headline['picp95']:.4f}   ({headline['k_of_n']})   CP95 CI = {headline['cp95ci']}")
    print(f"  PEAK PICP95  = {headline['peak_picp95']:.4f}  ({headline['peak_k']}/{headline['n_peak']})   "
          f"(Tweedie 0.87; reference {ref_peakcov:.3f})")
    print(f"  peak pt bias = {headline['peak_bias']:+.3f}  (TiRex peak bias -2.762; less-negative = higher on ramps)")
    print(f"  last34 WIS   = {headline['last34_wis']:.4f}   mean-W95 = {headline['mean_w95']:.3f}")
    print(f"  BARS: {bars}  point_helps_peaks={point_helps_peaks}")
    print(f"  LEAK-FREE: expand==static@first={first_match}  "
          f"train-cap WIS={m_train['wis']} PICP95={m_train['picp95']} cap_binds={cap_binds} (False=spotless)")

    out = {
        "reference_wis": round(ref_mean, 4), "n": n,
        "reference_picp95": round(ref_k / n, 4), "reference_peak_picp95": round(ref_peakcov, 3),
        "p_star": P_STAR, "timesfm_crosscheck": cc,
        "convex_rows": conv_rows, "convex_pick": conv_pick,
        "gate_rows_top": sorted(gate_rows, key=lambda r: r["val_wis"])[:10], "gate_pick": mg,
        "headline_config": best_config, "headline": {k: v for k, v in headline.items()},
        "headline_train_cap": m_train, "bars": bars, "point_helps_peaks": point_helps_peaks,
        "cap_binds": cap_binds, "expand_eq_static_first_origin": first_match,
        "tweedie_ref_peak_picp95": 0.87,
    }
    (ROOT / "scripts" / "_exp_timesfm_ens.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nelapsed {time.time()-t0:.1f}s  wrote scripts/_exp_timesfm_ens.json")
    return out


if __name__ == "__main__":
    raise SystemExit(main() and 0)
