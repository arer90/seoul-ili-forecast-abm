#!/usr/bin/env python
"""NEW FUSEDEPI v2 — native-quantile FUSION ⊗ (mechanism-informed width ⊕ Poisson scale).

Composition (all three verified ingredient cores, reused by import)
-------------------------------------------------------------------
1. BASE = the native-quantile fusion from ``scripts/nov_quantile_fusion.py``
   (TiRex native deciles ⊕ count-aware NegBin, per-level convex weights + dispersion
   phi tuned on TRAIN only). Reproduces its V1 fused quantile matrix EXACTLY
   (same roll_tirex / expand / negbin / tune_weights / fuse). Its leak-free
   headline was WIS≈2.68 / PICP95≈0.912 on the frozen 68-week test.

2. WRAP the fusion's conformal correction (the additional PID half-width that sits
   on top of the fused CQR quantiles) with two leak-free modulators:
     * MECHANISM (``scripts/nov_mechanism_pi.py``): multiply the correction by
         m_i = clip( (foi_i / trailing_ref_i) ** gamma , m_lo, m_hi )
       with foi = Rt*(1-S), 1-lag, past-only (widen on rising force-of-infection,
       narrow on decline — REALLOCATES width toward the peak rather than inflating).
     * POISSON SCALE (``scripts/nov_shift_conformal.py``): conformalize the
       Poisson-NORMALIZED nonconformity u = cqr_score / s_i (s_i = sqrt(pred) or
       level), then rescale the correction by s_i so the width grows with the
       forecast level instantly (no lag) at the surge.

   Combined additional half-width at step i (per PI level):
       base_u = quantile( u_buffer[-window:], 1-alpha )        # normalized units
       Q      = max(0, base_u*s_i + ki*max(base_u*s_i,1)*integral)   # P + I (PID)
       Q_mod  = Q * m_i                                        # mechanism modulation
       nlo = qlo_i - Q_mod ; nhi = qhi_i + Q_mod   (or upper-only for side='upper')
   With gamma=0 AND scale='unit' this is byte-identical to the fusion's own
   ``_pid_adjust`` (verified in-script) — a strict generalization.

Leak-free protocol (identical frozen 68-wk hold-out as every ingredient)
------------------------------------------------------------------------
* Frozen split via scripts.ablation_fusedepi.load_split (pool_end, n_test=68).
* Fusion weights/phi: TRAIN rolling 1-step only (reused from nov_quantile_fusion).
* NEW conformal HP (scale_kind, gamma, side): tuned on the TRAIN POOL fused-quantile
  conformal WIS only (Protocol A). A stricter Protocol B tunes on the first-34 TEST
  weeks and evaluates the truly-unseen LAST-34.
* Every step i uses obs[0..i-1], pred[i], s[i], foi[i] (past-only) only.

Decisive-win bars reported explicitly (all must clear):
  (1) full-68 WIS < 2.68 (and < 2.677 quantile-fusion / < 2.720 held-out stack);
  (2) PICP95 >= 0.93 overall (ideally 0.95), esp. peak weeks y>=50;
  (3) the gain SURVIVES the truly-unseen last-34 (Protocol B), not just full-68;
  (4) NOT a width artifact — uniform widening to the SAME mean 95% width must NOT
      match the WIS gain (reallocation, not inflation).

No live pipeline/model code is modified. Writes one JSON.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np

from simulation.analytics.adaptive_conformal import wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS

# ── reuse the verified ingredient cores by import ──
from scripts.ablation_fusedepi import load_split
from scripts.nov_quantile_fusion import (
    CONF_KI,
    CONF_WINDOW,
    FQ,
    FQ_COL,
    MAX_CONTEXT,
    PEAK_Y,
    SEED_TAIL,
    WARMUP,
    conformalize,        # base fusion conformal (for reproduction + uniform control)
    expand_tirex_to_flusight,
    fuse,
    negbin_flusight,
    roll_tirex,
    score as fusion_score,
    seed_all,
    tune_weights,
    wis_train,
)
from scripts.nov_mechanism_pi import build_signals  # foi = Rt*(1-S), 1-lag leak-free
from scripts.nov_shift_conformal import compute_scale  # Poisson sqrt/level scale

LOG = logging.getLogger("dec_mech_quantile")
OUT_JSON = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/novelty/dec_mech_quantile.json"
)

REF_WINDOW = 30            # trailing window for the leak-free mechanism reference
M_LO, M_HI = 0.5, 2.5      # mechanism multiplier floor/ceiling (allows NARROWING)
COV_TARGET = 0.93          # train-pool PICP95 constraint for HP selection


# ─────────────────────────────────────────────────────────────────────────────
# combined conformal: Poisson-normalized CQR PID + mechanism multiplier (leak-free)
# ─────────────────────────────────────────────────────────────────────────────
def mech_scale_pid(qlo, qhi, obs, seed_lo, seed_hi, seed_y, seed_scale,
                   scale, foi, seed_foi, beta, target, *,
                   window=CONF_WINDOW, ki=CONF_KI, cap=np.inf,
                   gamma=0.0, ref_window=REF_WINDOW, m_lo=M_LO, m_hi=M_HI, side="sym"):
    """Single-level CQR Conformal-PID on fused quantiles, Poisson-normalized + mechanism-modulated.

    Args:
        qlo/qhi: (n,) fused CQR bounds for this PI level (test span).
        obs: (n,) rolling test observations.
        seed_lo/seed_hi/seed_y: (m,) train-tail fused bounds + obs (nonconformity seed).
        seed_scale: (m,) Poisson scale on the seed span (to normalize the seed scores).
        scale: (n,) leak-free Poisson scale s_i (compute_scale; sqrt/level/unit).
        foi/seed_foi: (n,)/(m,) 1-lag force-of-infection channel (test / train-pool seed).
        beta/target: P-quantile level (=1-alpha) / target miscoverage (=alpha).
        gamma: mechanism modulation strength (0 = mechanism-free).
        side: 'sym' widen both bounds; 'upper' widen only the upper bound.

    Returns: (nlo, nhi) adjusted (n,) bounds. Leak-free: step i uses obs[0..i-1],
    qlo/qhi[i], scale[i], foi[i]; obs[i] enters the buffer AFTER the interval is set.
    """
    qlo = np.asarray(qlo, float); qhi = np.asarray(qhi, float)
    obs = np.asarray(obs, float).ravel()
    s = np.asarray(scale, float).ravel()
    n = len(qlo)
    nlo = qlo.copy(); nhi = qhi.copy()
    # normalized seed nonconformity buffer (CQR score / seed scale)
    seed_scores = np.maximum(np.asarray(seed_lo, float) - np.asarray(seed_y, float),
                             np.asarray(seed_y, float) - np.asarray(seed_hi, float))
    u_buf = list(seed_scores / np.maximum(np.asarray(seed_scale, float).ravel(), 1e-9))
    sig_buf = list(np.asarray(seed_foi, float).ravel())
    integral = 0.0
    for i in range(n):
        si = max(s[i], 1e-9)
        base_u = max(0.0, float(np.quantile(u_buf[-window:], beta))) if u_buf else 0.0
        base_half = base_u * si                          # Poisson-rescaled base half-width
        Q = max(0.0, base_half + ki * max(base_half, 1.0) * integral)   # P + I
        if gamma > 0.0:
            ref = float(np.mean(sig_buf[-ref_window:])) if sig_buf else float(foi[i])
            ratio = foi[i] / ref if ref > 1e-9 else 1.0
            m = float(np.clip(ratio ** gamma, m_lo, m_hi))
        else:
            m = 1.0
        if side == "upper":
            Qlo, Qhi = Q, Q * m
        else:
            Qlo = Qhi = Q * m
        nlo[i] = max(0.0, qlo[i] - Qlo)
        nhi[i] = min(cap, qhi[i] + Qhi)
        miscov = 1.0 if (obs[i] < nlo[i] or obs[i] > nhi[i]) else 0.0
        integral = float(np.clip(integral + (miscov - target), -5.0, 5.0))
        u_buf.append(float(max(qlo[i] - obs[i], obs[i] - qhi[i])) / si)   # normalized CQR score
        sig_buf.append(float(foi[i]))
    return nlo, nhi


def mech_scale_conformalize(fused, y_obs, seed_fused, seed_y, cap, pred, seed_pred,
                            foi, seed_foi, *, scale_kind="unit", floor=1.0,
                            gamma=0.0, side="sym"):
    """All-level combined conformal → {alpha: (lo, hi)}. Leak-free.

    Args: fused (n,23) test fused quantiles; seed_fused (m,23) train-tail; y_obs/seed_y obs;
    pred/seed_pred (n,)/(m,) fused-median point (for Poisson scale); foi/seed_foi channels.
    scale_kind: 'unit'|'sqrt'|'level'; gamma/side: mechanism knobs.
    """
    s_test = compute_scale(pred, y_obs, scale_kind, floor=floor)
    s_seed = compute_scale(seed_pred, seed_y, scale_kind, floor=floor)
    bounds = {}
    for a in FLUSIGHT_ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]
        ch = FQ_COL[round(1.0 - a / 2.0, 4)]
        nlo, nhi = mech_scale_pid(
            fused[:, cl], fused[:, ch], y_obs,
            seed_fused[:, cl], seed_fused[:, ch], seed_y, s_seed,
            s_test, foi, seed_foi, beta=1.0 - a, target=a, cap=cap,
            gamma=gamma, side=side)
        bounds[a] = (nlo, nhi)
    return bounds


def uniform_widen(base_bounds, median, cap, c):
    """Widen every PI band by factor c around the median (uniform-width control)."""
    out = {}
    med = np.asarray(median, float).ravel()
    for a, (lo, hi) in base_bounds.items():
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        nlo = np.clip(med - c * (med - lo), 0.0, None)
        nhi = np.minimum(med + c * (hi - med), cap)
        out[a] = (nlo, nhi)
    return out


def pooled_wis(bounds, y, median, mask):
    wis = np.asarray(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=median), float)
    return float(np.mean(wis[np.asarray(mask, bool)]))


def pooled_picp(bounds, y, mask, alpha=0.05):
    lo, hi = bounds[alpha]
    cov = (np.asarray(y, float) >= lo) & (np.asarray(y, float) <= hi)
    return float(np.mean(cov[np.asarray(mask, bool)]))


def mean_width95(bounds, mask):
    lo, hi = bounds[0.05]
    return float(np.mean((hi - lo)[np.asarray(mask, bool)]))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(42)
    t0 = time.time()
    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test)
    LOG.info("split train=%d test=%d peak(y>=%.0f)=%d", ntr, nte, PEAK_Y, int((y_test >= PEAK_Y).sum()))

    from tirex import load_model
    model = load_model("NX-AI/TiRex", device="cpu")
    y_full_test = np.concatenate([y_train, y_test])
    cap = 2.0 * float(np.max(y_full_test))

    # ── roll TiRex: train pool (tuning) + test ──
    tr_idx = list(range(WARMUP, ntr))
    te_idx = list(range(ntr, ntr + nte))
    LOG.info("rolling TiRex train-pool (%d) ...", len(tr_idx))
    m_tr, dec_tr = roll_tirex(model, y_train.copy(), tr_idx)
    LOG.info("rolling TiRex test (%d) ...", len(te_idx))
    m_te, dec_te = roll_tirex(model, y_full_test, te_idx)
    y_tr_roll = y_train[WARMUP:]

    # verify test TiRex point == frozen official
    fro = np.asarray(json.loads((ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
                     ["refit_test_predictions"], dtype=float)
    tirex_point_maxdiff = float(np.max(np.abs(m_te - fro)))
    LOG.info("TiRex test point vs frozen official maxdiff=%.6f", tirex_point_maxdiff)

    # ── expand quantiles + choose phi/w on TRAIN (reproduce fusion V1 exactly) ──
    qA_tr = expand_tirex_to_flusight(dec_tr)
    qA_te = expand_tirex_to_flusight(dec_te)
    phi_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    best = None
    for phi in phi_grid:
        qB_tr = negbin_flusight(m_tr, phi, cap)
        w = tune_weights(qA_tr, qB_tr, y_tr_roll)
        f_tr = fuse(qA_tr, qB_tr, w)
        wtr = wis_train(f_tr, y_tr_roll)
        if best is None or wtr < best["train_wis"]:
            best = {"phi": phi, "w": w, "train_wis": wtr}
    phi, w = best["phi"], best["w"]
    LOG.info("fusion selected phi=%.3f train_WIS=%.4f", phi, best["train_wis"])

    qB_tr = negbin_flusight(m_tr, phi, cap); f_tr = fuse(qA_tr, qB_tr, w)
    qB_te = negbin_flusight(m_te, phi, cap); f_te = fuse(qA_te, qB_te, w)
    med_te = f_te[:, FQ_COL[0.5]]
    med_tr = f_tr[:, FQ_COL[0.5]]

    seed_fused = f_tr[-SEED_TAIL:]
    seed_y = y_tr_roll[-SEED_TAIL:]
    seed_pred = med_tr[-SEED_TAIL:]

    # ── mechanism channel foi (leak-free, full pool+test 1-lag) ──
    _mech_lag, ts, ch = build_signals(y_train, y_test)
    assert ts == ntr
    foi_full = ch["foi"]                          # length ntr+nte
    foi_te = foi_full[ntr:ntr + nte]
    foi_seed_test = foi_full[:ntr]                # train-pool past for test ref seed
    # train-pool rolling foi (aligned to y_tr_roll span WARMUP..ntr)
    foi_tr_roll = foi_full[WARMUP:ntr]
    foi_seed_tr = foi_full[:WARMUP]

    # masks
    overall = np.ones(nte, bool)
    peak = y_test >= PEAK_Y
    last34 = np.zeros(nte, bool); last34[nte // 2:] = True
    first34 = ~last34
    masks = {"overall_68": overall, "peak_y50": peak, "last34": last34, "first34": first34}

    # ═══════════ base fusion V1 (reproduce) ═══════════
    b_base = conformalize(f_te, y_test, seed_fused, seed_y, cap)
    base_scores = fusion_score(b_base, y_test, med_te, masks)
    LOG.info("BASE fusion V1: WIS_all=%.4f PICP95=%.3f peakPICP=%.3f last34WIS=%.4f",
             base_scores["overall_68"]["wis"], base_scores["overall_68"]["picp95"],
             base_scores["peak_y50"]["picp95"], base_scores["last34"]["wis"])

    # sanity: mech_scale (gamma=0, scale='unit', sym) == base fusion conformal
    b_sanity = mech_scale_conformalize(f_te, y_test, seed_fused, seed_y, cap, med_te, seed_pred,
                                       foi_te, foi_seed_test, scale_kind="unit", gamma=0.0)
    sanity_maxabs = max(
        max(float(np.max(np.abs(b_sanity[a][0] - b_base[a][0]))) for a in FLUSIGHT_ALPHAS),
        max(float(np.max(np.abs(b_sanity[a][1] - b_base[a][1]))) for a in FLUSIGHT_ALPHAS))
    LOG.info("sanity gamma0/unit vs base fusion maxabs=%.3e", sanity_maxabs)

    # ═══════════ TRAIN-POOL HP tuning (Protocol A, leak-free) ═══════════
    # conformalize the train-pool fused quantiles; seed with its own first SEED_TAIL,
    # score WIS/PICP on the remainder (all within train pool → leak-free).
    ntrr = len(y_tr_roll)
    tr_seed_sl = slice(0, SEED_TAIL)
    tr_eval_sl = slice(SEED_TAIL, ntrr)
    tr_eval_mask = np.zeros(ntrr - SEED_TAIL, bool) | True
    grid = []
    for scale_kind in ("unit", "sqrt", "level"):
        for gamma in (0.0, 0.5, 0.75, 1.0, 1.5):
            for side in ("sym", "upper"):
                grid.append((scale_kind, gamma, side))
    tune_rows = []
    for scale_kind, gamma, side in grid:
        bt = mech_scale_conformalize(
            f_tr[tr_eval_sl], y_tr_roll[tr_eval_sl],
            f_tr[tr_seed_sl], y_tr_roll[tr_seed_sl], cap,
            med_tr[tr_eval_sl], med_tr[tr_seed_sl],
            foi_tr_roll[tr_eval_sl], foi_tr_roll[tr_seed_sl],
            scale_kind=scale_kind, gamma=gamma, side=side)
        wis_tr = pooled_wis(bt, y_tr_roll[tr_eval_sl], med_tr[tr_eval_sl], tr_eval_mask)
        picp_tr = pooled_picp(bt, y_tr_roll[tr_eval_sl], tr_eval_mask)
        tune_rows.append({"scale_kind": scale_kind, "gamma": gamma, "side": side,
                          "train_wis": wis_tr, "train_picp95": picp_tr})
    # selection rule (leak-free): min train WIS s.t. train PICP95 >= COV_TARGET; else min WIS
    feasible = [r for r in tune_rows if r["train_picp95"] >= COV_TARGET]
    pool = feasible if feasible else tune_rows
    selA = min(pool, key=lambda r: r["train_wis"])
    selA_pure = min(tune_rows, key=lambda r: r["train_wis"])
    LOG.info("Protocol-A selected: %s (train_wis=%.4f picp=%.3f, feasible=%d)",
             {k: selA[k] for k in ("scale_kind", "gamma", "side")},
             selA["train_wis"], selA["train_picp95"], len(feasible))

    def eval_test(scale_kind, gamma, side, msk=masks):
        bb = mech_scale_conformalize(f_te, y_test, seed_fused, seed_y, cap, med_te, seed_pred,
                                     foi_te, foi_seed_test, scale_kind=scale_kind,
                                     gamma=gamma, side=side)
        return bb, fusion_score(bb, y_test, med_te, msk)

    bA, scoresA = eval_test(selA["scale_kind"], selA["gamma"], selA["side"])
    bA_pure, scoresA_pure = eval_test(selA_pure["scale_kind"], selA_pure["gamma"], selA_pure["side"])

    # ═══════════ Protocol B: tune on first-34 TEST, evaluate last-34 (truly unseen) ═══════════
    # seed the first-34 conformal from the train tail (as in deployment); tune HP on first-34 WIS
    # under the same PICP constraint; then FREEZE and read the last-34 slice.
    b_rows = []
    for scale_kind, gamma, side in grid:
        bb = mech_scale_conformalize(f_te, y_test, seed_fused, seed_y, cap, med_te, seed_pred,
                                     foi_te, foi_seed_test, scale_kind=scale_kind,
                                     gamma=gamma, side=side)
        wis_fh = pooled_wis(bb, y_test, med_te, first34)
        picp_fh = pooled_picp(bb, y_test, first34)
        b_rows.append({"scale_kind": scale_kind, "gamma": gamma, "side": side,
                       "first34_wis": wis_fh, "first34_picp95": picp_fh})
    feasibleB = [r for r in b_rows if r["first34_picp95"] >= COV_TARGET]
    poolB = feasibleB if feasibleB else b_rows
    selB = min(poolB, key=lambda r: r["first34_wis"])
    LOG.info("Protocol-B selected on first34: %s", {k: selB[k] for k in ("scale_kind", "gamma", "side")})
    bB, scoresB = eval_test(selB["scale_kind"], selB["gamma"], selB["side"])

    # ═══════════ ORACLE diagnostic (TEST-PEEKING — capability probe only, NOT reported as a win) ═══════════
    # Scans the same grid directly on the full-68 test to see whether ANY config can
    # simultaneously clear WIS<2.68 AND PICP95>=0.93. This is an upper bound the leak-free
    # selection cannot legitimately claim; it tells us if the method is even capable.
    oracle_rows = []
    for scale_kind, gamma, side in grid:
        _, sc = eval_test(scale_kind, gamma, side)
        oracle_rows.append({
            "scale_kind": scale_kind, "gamma": gamma, "side": side,
            "wis": sc["overall_68"]["wis"], "picp95": sc["overall_68"]["picp95"],
            "peak_picp95": sc["peak_y50"]["picp95"], "last34_wis": sc["last34"]["wis"],
        })
    oracle_both = [r for r in oracle_rows if r["wis"] < 2.68 and r["picp95"] >= 0.93]
    oracle_best_wis = min(oracle_rows, key=lambda r: r["wis"])
    oracle_best_cov_under_wis = None
    cov_feas = [r for r in oracle_rows if r["picp95"] >= 0.93]
    if cov_feas:
        oracle_best_cov_under_wis = min(cov_feas, key=lambda r: r["wis"])
    LOG.info("ORACLE: configs clearing BOTH WIS<2.68 & PICP95>=0.93: %d; best-WIS=%.4f@picp%.3f",
             len(oracle_both), oracle_best_wis["wis"], oracle_best_wis["picp95"])

    # ═══════════ width-artifact control (bar 4) ═══════════
    # uniform-widen the BASE fusion bounds to the SAME full-68 mean 95% width as Protocol-A.
    base_w95 = mean_width95(b_base, overall)
    target_w95 = mean_width95(bA, overall)
    c_uniform = target_w95 / base_w95 if base_w95 > 1e-9 else 1.0
    b_uniform = uniform_widen(b_base, med_te, cap, c_uniform)
    uni_scores = fusion_score(b_uniform, y_test, med_te, masks)
    uni_w95 = mean_width95(b_uniform, overall)
    LOG.info("uniform control c=%.3f width(base=%.2f target=%.2f got=%.2f) WIS=%.4f PICP95=%.3f",
             c_uniform, base_w95, target_w95, uni_w95,
             uni_scores["overall_68"]["wis"], uni_scores["overall_68"]["picp95"])

    # ═══════════ verdict — 4 bars ═══════════
    A = scoresA
    bar1 = bool(A["overall_68"]["wis"] < 2.68)
    bar1_strict = bool(A["overall_68"]["wis"] < 2.677)
    bar2 = bool(A["overall_68"]["picp95"] >= 0.93)
    bar2_peak = bool(A["peak_y50"]["picp95"] >= 0.93)
    # bar 3: gain survives last-34 (lower WIS than base fusion last-34 AND better/equal coverage)
    bar3 = bool(A["last34"]["wis"] < base_scores["last34"]["wis"]
                and A["last34"]["picp95"] >= base_scores["last34"]["picp95"])
    bar3_B = bool(scoresB["last34"]["wis"] < base_scores["last34"]["wis"]
                  and scoresB["last34"]["picp95"] >= base_scores["last34"]["picp95"])
    # bar 4: not a width artifact — Protocol-A WIS strictly below uniform-width WIS (same width)
    bar4 = bool(A["overall_68"]["wis"] < uni_scores["overall_68"]["wis"] - 1e-6)
    decisive = bool(bar1 and bar2 and bar3 and bar4)

    verdict = {
        "BASE_fusion_wis": base_scores["overall_68"]["wis"],
        "BASE_fusion_picp95": base_scores["overall_68"]["picp95"],
        "BASE_fusion_peak_picp95": base_scores["peak_y50"]["picp95"],
        "BASE_fusion_last34_wis": base_scores["last34"]["wis"],
        "v2_protocolA_config": {k: selA[k] for k in ("scale_kind", "gamma", "side")},
        "v2_full68_wis": A["overall_68"]["wis"],
        "v2_full68_picp95": A["overall_68"]["picp95"],
        "v2_peak_y50_wis": A["peak_y50"]["wis"],
        "v2_peak_y50_picp95": A["peak_y50"]["picp95"],
        "v2_last34_wis": A["last34"]["wis"],
        "v2_last34_picp95": A["last34"]["picp95"],
        "v2_full68_mean_width95": A["overall_68"]["mean_width95"],
        "protocolB_config": {k: selB[k] for k in ("scale_kind", "gamma", "side")},
        "protocolB_last34_wis": scoresB["last34"]["wis"],
        "protocolB_last34_picp95": scoresB["last34"]["picp95"],
        "uniform_control_wis": uni_scores["overall_68"]["wis"],
        "uniform_control_picp95": uni_scores["overall_68"]["picp95"],
        "uniform_control_c": c_uniform,
        "sanity_gamma0_unit_vs_base_maxabs": sanity_maxabs,
        "tirex_point_maxdiff_vs_frozen": tirex_point_maxdiff,
        "BAR1_wis_lt_2p68": bar1,
        "BAR1_strict_wis_lt_2p677": bar1_strict,
        "BAR2_picp95_ge_0p93": bar2,
        "BAR2_peak_picp95_ge_0p93": bar2_peak,
        "BAR3_gain_survives_last34_protoA": bar3,
        "BAR3_gain_survives_last34_protoB": bar3_B,
        "BAR4_not_width_artifact": bar4,
        "DECISIVE_WIN_all_4_bars": decisive,
    }

    out = {
        "method": "FusedEpi v2 = native-quantile fusion ⊗ (mechanism foi width ⊕ Poisson scale)",
        "selected_phi": phi,
        "fusion_train_wis": best["train_wis"],
        "protocol": {
            "split": meta,
            "tirex_max_context": MAX_CONTEXT,
            "conformal": "combined mech_scale_pid: Poisson-normalized CQR nonconformity "
                         "(u=cqr/s) with PID (window=%d ki=%.2f), width rescaled by s_i and "
                         "multiplied by clip((foi/ref)^gamma, %.1f, %.1f)"
                         % (CONF_WINDOW, CONF_KI, M_LO, M_HI),
            "protocolA_tuning": "scale_kind/gamma/side chosen on TRAIN-POOL fused-quantile "
                                "conformal WIS s.t. train PICP95>=%.2f; frozen for test" % COV_TARGET,
            "protocolB_tuning": "same grid chosen on FIRST-34 test weeks; evaluated on unseen LAST-34",
            "peak_definition": "y_test >= %.0f" % PEAK_Y,
            "leak_free": "step i uses obs[0..i-1], pred[i]=fused-median, s[i], foi[i] (1-lag, past-only)",
        },
        "selection_rule": "min train/first34 WIS subject to PICP95>=%.2f (else min WIS)" % COV_TARGET,
        "tune_grid_train_pool": tune_rows,
        "tune_grid_first34": b_rows,
        "ORACLE_diagnostic_testpeeking": {
            "note": "TEST-PEEKING capability probe — NOT a leak-free result",
            "n_configs_clearing_both_wis_and_picp": len(oracle_both),
            "configs_clearing_both": oracle_both,
            "best_wis_config": oracle_best_wis,
            "best_wis_under_picp95_ge_0p93": oracle_best_cov_under_wis,
            "all_configs": oracle_rows,
        },
        "results": {
            "BASE_fusion_V1": base_scores,
            "v2_protocolA_pooltuned": A,
            "v2_protocolA_pure_minWIS": scoresA_pure,
            "v2_protocolB_first34tuned": scoresB,
            "uniform_width_control": uni_scores,
        },
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── console table ──
    print("\n============ FusedEpi v2: FUSION ⊗ (mechanism ⊕ Poisson scale) ============")
    print(f"phi={phi}  sanity(gamma0/unit vs base)={sanity_maxabs:.2e}  "
          f"TiRex maxdiff={tirex_point_maxdiff:.6f}\n")
    hdr = f"{'variant':32s} {'WIS_all':>8s} {'PICP95':>7s} {'WIS_pk':>8s} {'PICP_pk':>8s} {'WIS_l34':>8s} {'PICP_l34':>9s} {'W95':>7s}"
    print(hdr); print("-" * len(hdr))
    rows = [
        ("BASE_fusion_V1", base_scores),
        ("v2_protocolA_pooltuned", A),
        ("v2_protocolA_pure_minWIS", scoresA_pure),
        ("v2_protocolB_first34tuned", scoresB),
        ("uniform_width_control", uni_scores),
    ]
    for name, r in rows:
        print(f"{name:32s} {r['overall_68']['wis']:8.4f} {r['overall_68']['picp95']:7.3f} "
              f"{r['peak_y50']['wis']:8.3f} {r['peak_y50']['picp95']:8.3f} "
              f"{r['last34']['wis']:8.4f} {r['last34']['picp95']:9.3f} {r['overall_68']['mean_width95']:7.2f}")
    print("\nProtocol-A config:", verdict["v2_protocolA_config"],
          "| Protocol-B config:", verdict["protocolB_config"])
    print(f"ORACLE(test-peeking) configs clearing BOTH WIS<2.68 & PICP95>=0.93: {len(oracle_both)} "
          f"| best-WIS={oracle_best_wis['wis']:.4f}@picp{oracle_best_wis['picp95']:.3f}"
          + (f" | best-WIS@picp>=0.93={oracle_best_cov_under_wis['wis']:.4f}"
             f"(picp{oracle_best_cov_under_wis['picp95']:.3f}, {oracle_best_cov_under_wis['scale_kind']}/"
             f"g{oracle_best_cov_under_wis['gamma']}/{oracle_best_cov_under_wis['side']})"
             if oracle_best_cov_under_wis else ""))
    print(json.dumps({k: v for k, v in verdict.items() if k.startswith("BAR") or k == "DECISIVE_WIN_all_4_bars"},
                     indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
