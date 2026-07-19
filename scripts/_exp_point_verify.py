#!/usr/bin/env python
"""ADVERSARIAL INDEPENDENT VERIFICATION of the 'best peak-point candidate'.

Target under scrutiny (scripts/_exp_timesfm_ens.py headline):
  ens_convex_w1.0_tweedie_p1.5_expand  -- the TimesFM+TiRex ensemble whose HONEST
  pre-T0 val-selection DEGENERATES to w=1.0 (pure TiRex), i.e. it collapses onto the
  standalone Tweedie champion. Claimed:
     WIS 2.2427, DM p vs 2.4012 = 3.354e-6, PICP95 0.9318 (123/132),
     PEAK PICP95 (y>=50) 0.8696, last34 2.6491, beats_tweedie=False, point_helps_peaks=False.

This script re-derives EVERYTHING independently (reference, head, expanding-CQR eval,
val-selection, DM, bootstrap, leak perturbations) and tries hard to REFUTE the claim.
It writes NO live/pipeline code and edits no existing script. Reusable numerical
primitives (setup / build_span / rolling_cqr_bounds / tirex_empirical_qy / wis_of / dm)
are imported as the machinery-under-test; the evaluation, selection and adversarial
perturbations are all reconstructed here.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dec_boosted_mech import (cqr_offsets, build_bounds_cqr, MED_COL, K_CAL,
                                      MIN_CTX, MAX_CONTEXT, PEAK_Y)
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup, dm, cp, wis_of
from scripts._exp_tweedie import build_span, WSTART

P_STAR = 1.5
CAL_START = T0 - K_CAL              # 165 test expanding seed
VAL_CAL_START = 125                 # val expanding seed
VAL_LO, VAL_HI = T0 - K_CAL, T0     # [165,205)
W_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
GATE_LAGS = (2, 3, 4); GATE_THR = (1.0, 2.0, 4.0, 8.0)
GATE_WBASE = (0.65, 0.8, 1.0); GATE_WRAMP = (0.0, 0.2, 0.35, 0.5)
TF_CACHE = ROOT / "scripts" / "_timesfm_pool.npz"

CLAIM = dict(wis=2.2427, dm_p=3.354e-6, picp95=0.9318, peak_picp95=0.8696, last34=2.6491)
TWEEDIE_REF = 2.2427               # standalone Tweedie champion (the incumbent headline)


# ---- import the exact ensemble/head helpers we are auditing (as callables) ----
from scripts._exp_tweedie_cover import rolling_cqr_bounds, static_bounds


def head_span_on(S, point, p=P_STAR):
    """Tweedie residual-scale (pearson) FLUSIGHT quantile matrix around `point`."""
    S2 = dict(S); S2["tirex"] = point
    _QG, QP, _ = build_span(S2, p, "tirex")
    return QP, S2


def eval_point(S, point, origins, cal_start, cap, p=P_STAR):
    QP, S2 = head_span_on(S, point, p)
    med = QP[origins - WSTART][:, MED_COL]
    B = rolling_cqr_bounds(QP, S2, origins, cap, cal_start, None)
    y = S["yf"][origins]
    return wis_of(B, y, med), B, med


def convex_point(tirex, tf, w):
    return w * tirex + (1.0 - w) * tf


def gate_point(yf, tirex, tf, wbase, wramp, lag, thr):
    n = len(yf); wv = np.full(n, wbase, float)
    for t in range(n):
        j0, j1 = t - 1, t - 1 - lag
        if j1 >= 0 and np.isfinite(yf[j0]) and np.isfinite(yf[j1]) and (yf[j0] - yf[j1]) > thr:
            wv[t] = wramp
    return wv * tirex + (1.0 - wv) * tf


def load_tf():
    d = np.load(TF_CACHE)
    tf = np.full(337, np.nan)
    tf[d["weeks"].astype(int)] = d["preds"]
    return tf


def rolling_surrogate(yf, ntot, fn):
    """Reproduce load_or_build_timesfm's context slicing EXACTLY, with a deterministic
    surrogate fn(context)->float in place of TimesFM. Used only for the leak-slice audit."""
    weeks = np.arange(MIN_CTX, ntot)
    out = np.full(ntot, np.nan)
    for t in weeks:
        ctx = yf[max(0, t - MAX_CONTEXT):t]
        out[t] = fn(ctx)
    return out


def moving_block_boot(d, L, B=10000, seed=0):
    """Two-sided moving-block bootstrap p-value for H0: mean(d)=0, plus 95% CI of mean.
    d = per-origin difference (candidate - reference); negative mean => candidate better."""
    rng = np.random.default_rng(seed)
    n = len(d)
    if np.allclose(d, 0.0):
        return 1.0, (0.0, 0.0)
    nblocks = int(np.ceil(n / L))
    starts_pool = np.arange(0, n - L + 1)
    means = np.empty(B)
    for b in range(B):
        st = rng.choice(starts_pool, size=nblocks, replace=True)
        idx = (st[:, None] + np.arange(L)[None, :]).ravel()[:n]
        means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    # recenter for H0 test
    c = means - means.mean()
    obs = d.mean()
    p = 2.0 * min((c <= -abs(obs)).mean(), (c >= abs(obs)).mean())
    return float(min(p, 1.0)), (float(lo), float(hi))


def peak_cov(B, y):
    lo, hi = B[0.05]
    covv = (y >= lo) & (y <= hi)
    pk = y >= PEAK_Y
    return int(covv[pk].sum()), int(pk.sum()), float(covv[pk].mean())


def main():
    t0 = time.time()
    S = setup()
    ntot = S["ntot"]; ntr = ntot - 68
    origins = np.arange(T0, ntot); n = len(origins)
    y = S["yf"][origins]
    cal = np.arange(T0 - K_CAL, T0)
    r_full = S["yf"] - S["tirex"]
    cap_full = S["cap"]
    cap_train = 2.0 * float(S["yf"][:ntr].max())
    tirex = S["tirex"]
    tf = load_tf()
    R = {}

    # ============ (1) REPRODUCE reference 2.4012 and the headline (w=1.0) ============
    qy_ref = tirex_empirical_qy(tirex, r_full, origins, cap_full)
    cqr_ref = cqr_offsets(tirex_empirical_qy(tirex, r_full, cal, cap_full), S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap_full)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())

    hpt = convex_point(tirex, tf, 1.0)                      # w=1.0 headline point
    identical_to_tirex = bool(np.allclose(hpt[125:], tirex[125:], equal_nan=True))
    w_head, B_head, med_head = eval_point(S, hpt, origins, CAL_START, cap_full)
    wis_head = float(w_head.mean())
    lo95, hi95 = B_head[0.05]
    covv = (y >= lo95) & (y <= hi95); k = int(covv.sum())
    p_dm, dbar = dm(w_head, ref_wis)
    pk_k, pk_n, pk_cov = peak_cov(B_head, y)
    last34 = float(w_head[n - 34:].mean())

    R["reference_wis"] = round(ref_mean, 4)
    R["headline"] = dict(wis=round(wis_head, 4), dm_p=p_dm, dm_meandiff=round(dbar, 4),
                         picp95=round(k / n, 4), k_of_n=f"{k}/{n}", cp95ci=list(cp(k, n)),
                         peak_picp95=round(pk_cov, 4), peak=f"{pk_k}/{pk_n}",
                         last34=round(last34, 4), identical_to_tirex_point=identical_to_tirex)
    repro = dict(
        wis=abs(wis_head - CLAIM["wis"]) < 5e-4,
        dm_p=abs(p_dm - CLAIM["dm_p"]) < 5e-7 or (p_dm < 1e-4 and CLAIM["dm_p"] < 1e-4),
        picp95=abs(k / n - CLAIM["picp95"]) < 5e-4,
        peak=abs(pk_cov - CLAIM["peak_picp95"]) < 1e-3,
        last34=abs(last34 - CLAIM["last34"]) < 5e-4,
    )
    R["reproduces_claim"] = repro
    R["reproduces_all"] = all(repro.values())

    # cross-check the headline WIS array is IDENTICAL to a from-scratch Tweedie-champion
    # (pure tirex point, same head, same expanding CQR) -> proves 'ensemble' == Tweedie
    w_tw, B_tw, _ = eval_point(S, tirex.copy(), origins, CAL_START, cap_full)
    R["headline_equals_tweedie_bitwise"] = bool(np.array_equal(w_head, w_tw))

    # ============ (2a) LEAK AUDIT: TimesFM context slicing is strictly past ============
    yf0 = S["yf"].copy()
    base = rolling_surrogate(yf0, ntot, lambda c: float(c[-1]))    # last-value surrogate
    y260 = yf0.copy(); y260[260] += 100.0
    p260 = rolling_surrogate(y260, ntot, lambda c: float(c[-1]))
    diff260 = np.where(~np.isclose(base, p260, equal_nan=True))[0]
    first_change_260 = int(diff260.min()) if len(diff260) else None
    y336 = yf0.copy(); y336[336] += 100.0
    p336 = rolling_surrogate(y336, ntot, lambda c: float(c[-1]))
    diff336 = np.where(~np.isclose(base, p336, equal_nan=True))[0]
    R["leak_timesfm_slice"] = dict(
        first_change_after_perturb_y260=first_change_260,
        expect_261=(first_change_260 == 261),
        n_changes_after_perturb_y336_terminal=int(len(diff336)),
        expect_terminal_none=(len(diff336) == 0))

    # ============ (2b) headline is INDEPENDENT of TimesFM (w=1 => tf weight 0) ============
    rng = np.random.default_rng(7)
    tf_garbage = tf + rng.normal(0, 50, size=tf.shape)            # corrupt entire pool
    hpt_g = convex_point(tirex, tf_garbage, 1.0)
    w_head_g, _, _ = eval_point(S, hpt_g, origins, CAL_START, cap_full)
    R["headline_independent_of_timesfm"] = bool(np.array_equal(w_head, w_head_g))

    # ============ (2c) val-selection isolation: perturb TEST weeks -> val WIS identical ==
    val_origins = np.arange(VAL_LO, VAL_HI)
    y_val = S["yf"][val_origins]

    def convex_val_wis(Sx, tirex_x, tf_x, w):
        pt = convex_point(tirex_x, tf_x, w)
        QP, S2 = head_span_on(Sx, pt)
        med = QP[val_origins - WSTART][:, MED_COL]
        B = rolling_cqr_bounds(QP, S2, val_origins, cap_full, VAL_CAL_START, None)
        return float(wis_of(B, y_val, med).mean())

    val_base = {w: convex_val_wis(S, tirex, tf, w) for w in W_GRID}
    # perturb a test week (t=260, >=T0) in a COPY of S and recompute val sweep
    S_pert = dict(S); yfp = S["yf"].copy(); yfp[260] += 100.0; yfp[336] += 100.0
    S_pert["yf"] = yfp
    # tirex/tf unchanged (point models); only S['yf'] feeds the head's phi/Qz training.
    # head training for weeks<205 uses strictly-past y only -> must be identical.
    tirex_p = tirex.copy()  # tirex point unaffected by y perturbation (it's a fixed pool)
    val_pert = {w: convex_val_wis(S_pert, tirex_p, tf, w) for w in W_GRID}
    val_isolated = all(abs(val_base[w] - val_pert[w]) < 1e-12 for w in W_GRID)
    R["leak_val_selection_isolation"] = dict(
        val_wis_bit_identical_after_test_perturb=bool(val_isolated),
        val_wis_by_w={str(w): round(v, 4) for w, v in val_base.items()},
        argmin_val_w=float(min(val_base, key=val_base.get)))

    # ============ (2d) honest selection genuinely picks w=1.0 (convex + gate) ============
    # convex test metrics
    conv = []
    for w in W_GRID:
        pt = convex_point(tirex, tf, w)
        wt, Bt, _ = eval_point(S, pt, origins, CAL_START, cap_full)
        pkk, pkn, pkc = peak_cov(Bt, y)
        pdm, _ = dm(wt, ref_wis)
        conv.append(dict(w=w, val_wis=round(val_base[w], 4), test_wis=round(float(wt.mean()), 4),
                         dm_p_vs_ref=pdm, peak_picp95=round(pkc, 4),
                         dm_p_vs_tweedie=dm(wt, w_tw)[0]))
    conv_pick = min(conv, key=lambda r: r["val_wis"])
    # gate sweep (val-selected)
    gate = []
    for lag in GATE_LAGS:
        for thr in GATE_THR:
            for wb in GATE_WBASE:
                for wr in GATE_WRAMP:
                    if wr >= wb:
                        continue
                    pt = gate_point(S["yf"], tirex, tf, wb, wr, lag, thr)
                    QP, S2 = head_span_on(S, pt)
                    med = QP[val_origins - WSTART][:, MED_COL]
                    Bv = rolling_cqr_bounds(QP, S2, val_origins, cap_full, VAL_CAL_START, None)
                    gate.append(dict(lag=lag, thr=thr, wbase=wb, wramp=wr,
                                     val_wis=round(float(wis_of(Bv, y_val, med).mean()), 4)))
    gate_pick = min(gate, key=lambda r: r["val_wis"])
    gpt = gate_point(S["yf"], tirex, tf, gate_pick["wbase"], gate_pick["wramp"],
                     gate_pick["lag"], gate_pick["thr"])
    wg, Bg, _ = eval_point(S, gpt, origins, CAL_START, cap_full)
    gk, gn, gc = peak_cov(Bg, y)
    gate_test = dict(**gate_pick, test_wis=round(float(wg.mean()), 4),
                     dm_p_vs_ref=dm(wg, ref_wis)[0], dm_p_vs_tweedie=dm(wg, w_tw)[0],
                     peak_picp95=round(gc, 4))
    overall_val_winner = "convex_w%.2f" % conv_pick["w"] if conv_pick["val_wis"] <= gate_pick["val_wis"] \
        else "gate"
    R["selection"] = dict(convex=conv, convex_pick=conv_pick,
                          gate_pick_test=gate_test, overall_val_winner=overall_val_winner,
                          honest_pick_is_w1=bool(conv_pick["w"] == 1.0 and
                                                 conv_pick["val_wis"] <= gate_pick["val_wis"]))

    # ============ (2e) train-only cap gives same/better ============
    w_head_tc, B_tc, _ = eval_point(S, hpt, origins, CAL_START, cap_train)
    ltc, htc = B_tc[0.05]; ktc = int(((y >= ltc) & (y <= htc)).sum())
    wis_tc = float(w_head_tc.mean())
    # leak-safe criterion: the LEAK-FREE (train-only) cap must give SAME-OR-BETTER numbers
    # (WIS not higher, PICP not lower). If so, the reported full-cap number is not propped
    # up by peeking at the test-set maximum.
    R["leak_cap"] = dict(cap_full=round(cap_full, 2), cap_train=round(cap_train, 2),
                         wis_full=round(wis_head, 4), wis_train=round(wis_tc, 4),
                         picp95_full=round(k / n, 4), picp95_train=round(ktc / n, 4),
                         train_cap_same_or_better=bool(wis_tc <= wis_head + 1e-9 and ktc >= k),
                         cap_leak_concern=bool(wis_tc > wis_head + 1e-9 or ktc < k))

    # ============ (3) DM ROBUSTNESS: HLN + moving-block bootstrap ============
    d_ref = w_head - ref_wis          # headline - reference (expect <0 => headline better)
    d_tw = w_head - w_tw              # headline - Tweedie (should be all zeros => tie)
    # best genuine ensemble (w=0.8) vs Tweedie
    pt08 = convex_point(tirex, tf, 0.8)
    w08, _, _ = eval_point(S, pt08, origins, CAL_START, cap_full)
    d_08 = w08 - w_tw                 # w=0.8 - Tweedie (expect >0 => worse)
    boot = {}
    for L in (1, 4, 8, 12):
        p_r, ci_r = moving_block_boot(d_ref, L, seed=100 + L)
        p_t, ci_t = moving_block_boot(d_tw, L, seed=200 + L)
        p_8, ci_8 = moving_block_boot(d_08, L, seed=300 + L)
        boot[f"L{L}"] = dict(
            vs_ref=dict(mean_diff=round(float(d_ref.mean()), 4), p=round(p_r, 5),
                        ci95=[round(ci_r[0], 4), round(ci_r[1], 4)], sig_better=bool(ci_r[1] < 0)),
            vs_tweedie=dict(mean_diff=round(float(d_tw.mean()), 6), p=round(p_t, 5),
                            ci95=[round(ci_t[0], 6), round(ci_t[1], 6)],
                            tie=bool(abs(d_tw.mean()) < 1e-9)),
            w08_vs_tweedie=dict(mean_diff=round(float(d_08.mean()), 4), p=round(p_8, 5),
                                ci95=[round(ci_8[0], 4), round(ci_8[1], 4)],
                                better_than_tweedie=bool(ci_8[1] < 0)))
    hln_ref, _ = dm(w_head, ref_wis)
    hln_tw, _ = dm(w_head, w_tw)
    R["dm_robustness"] = dict(hln_p_vs_ref=hln_ref, hln_p_vs_tweedie=hln_tw, bootstrap=boot,
                              w08_test_wis=round(float(w08.mean()), 4))

    # ============ (4) does the point genuinely raise peak coverage without inflating WIS? ==
    # search all leak-free candidates (convex grid + gate pick) for peak PICP95 > 0.87 with WIS<=Tweedie
    cand_peak = []
    for r in conv:
        cand_peak.append(dict(name="convex_w%.2f" % r["w"], test_wis=r["test_wis"],
                              peak_picp95=r["peak_picp95"]))
    cand_peak.append(dict(name="gate_valpick", test_wis=gate_test["test_wis"],
                          peak_picp95=gate_test["peak_picp95"]))
    raises_peak_no_wis_cost = [c for c in cand_peak
                               if c["peak_picp95"] > 0.87 and c["test_wis"] <= TWEEDIE_REF + 1e-9]
    R["peak_analysis"] = dict(tweedie_peak_picp95=round(pk_cov, 4),
                              candidates=cand_peak,
                              any_leakfree_raises_peak_without_wis_cost=bool(raises_peak_no_wis_cost),
                              winners=raises_peak_no_wis_cost)

    # ============ VERDICT ============
    genuine_improvement = bool(
        (any(r["test_wis"] < TWEEDIE_REF - 1e-6 and r["dm_p_vs_tweedie"] < 0.05 for r in conv)) or
        bool(raises_peak_no_wis_cost))
    R["VERDICT"] = dict(
        reproduces_headline=R["reproduces_all"],
        headline_is_tweedie=R["headline_equals_tweedie_bitwise"],
        leak_free=bool(R["leak_timesfm_slice"]["expect_261"] and
                       R["leak_timesfm_slice"]["expect_terminal_none"] and
                       R["headline_independent_of_timesfm"] and
                       R["leak_val_selection_isolation"]["val_wis_bit_identical_after_test_perturb"] and
                       not R["leak_cap"]["cap_leak_concern"]),
        beats_tweedie_on_wis_or_peak=genuine_improvement,
        tweedie_remains_headline=bool(not genuine_improvement))

    (ROOT / "scripts" / "_exp_point_verify.json").write_text(json.dumps(R, indent=2, default=float))
    print(json.dumps(R, indent=2, default=float))
    print(f"\nelapsed {time.time()-t0:.1f}s  wrote scripts/_exp_point_verify.json")
    return R


if __name__ == "__main__":
    raise SystemExit(main() and 0)
