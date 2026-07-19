#!/usr/bin/env python
"""METHOD-MATCHED FAIR BASELINE for TiRex-alone (standalone; touches NO live/pipeline code).

Goal
----
The prior apples-to-apples TiRex-alone (WIS 2.5626 / PICP95 0.833 over the same 132 origins)
used a SIMPLE symmetric empirical |resid|-quantile conformal, which is NOT matched to the
method's CQR/PID machinery. This script gives TiRex-alone the SAME conformal treatment:

  1. Build TiRex "quantiles" qy = TiRex_point + empirical PAST-residual quantile offsets
     (unconditional, expanding, strictly past-only) at the FluSight quantile levels FQ.
     This is the exact unconditional analog of the method's GBM conditional residual
     quantiles: same qy shape (n, 23), fed to the SAME downstream functions.
  2. Run the IDENTICAL build_bounds_cqr AND build_bounds_pid (imported from
     scripts.dec_boosted_mech) over the SAME 132 origins (weeks 205..336, T0=205),
     with NO GBM and NO mechanism (foi_mult=None).
  3. CQR seed calibrated on [T0-K_CAL, T0) exactly as the full method's seed, but with the
     empirical-offset TiRex quantiles instead of seed-GBM quantiles.

Reconciliation
--------------
Before the TiRex-alone path, this script also reproduces the full method's GBM
`cqr_static` and `pid_plain` (gamma=0) numbers using MY harness calling the SAME
imported build_bounds_* functions, so any harness discrepancy would surface against the
verified JSON (cqr_static 2.2688/0.9848 ; pid_plain 2.3080/0.9015).

Leak-free: every quantile offset at origin/cal week t uses residuals for weeks < t only
(expanding past). CQR seed is fixed pre-T0. PID adapts online on past y only.
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
    os.environ.setdefault(_v, "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (
    FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y,
    load_split, build_features, cqr_offsets,
    build_bounds_cqr, build_bounds_pid,
)
from scripts.dec_boosted_mech_multiorigin import T0, REFIT_K, CONFIGS, fit_gbm, bagged_qy
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds


def tirex_empirical_qy(tirex_full, r_full, idxs, cap):
    """qy = TiRex point + empirical PAST-residual quantile offsets (expanding, past-only).

    For each week t in idxs: offsets = quantile(r_full[MIN_CTX:t], FQ) using residuals
    strictly before t only. row = clip(TiRex[t] + offsets, 0, cap), monotone-sorted.
    This is the unconditional analog of the method's GBM conditional residual quantiles.
    """
    qy = np.zeros((len(idxs), len(FQ)), dtype=float)
    for k, t in enumerate(idxs):
        past = r_full[MIN_CTX:t]
        past = past[np.isfinite(past)]
        off = np.quantile(past, FQ)
        row = np.clip(tirex_full[t] + off, 0.0, cap)
        row.sort()
        qy[k] = row
    return qy


def simple_abs_resid_bounds(tirex_seq, tirex_full, r_full, origins, cap):
    """The OLD apples-to-apples baseline: symmetric empirical |resid|-quantile conformal.

    Per origin t, per alpha: half = quantile(|r_full[MIN_CTX:t]|, 1-alpha); interval is
    [TiRex - half, TiRex + half], median = TiRex point. Expanding past-only. Included only
    to sanity-check the harness reproduces the ~2.5626/0.833 verified anchor.
    """
    n = len(origins)
    bounds = {a: (np.zeros(n), np.zeros(n)) for a in FLUSIGHT_ALPHAS}
    for k, t in enumerate(origins):
        past = np.abs(r_full[MIN_CTX:t])
        past = past[np.isfinite(past)]
        for a in FLUSIGHT_ALPHAS:
            half = float(np.quantile(past, 1.0 - a))
            lo = float(np.clip(tirex_full[t] - half, 0.0, cap))
            hi = float(np.clip(tirex_full[t] + half, 0.0, cap))
            bounds[a][0][k] = lo
            bounds[a][1][k] = hi
    return bounds


def score_bounds(bounds, y_seq, med_seq):
    wis = np.asarray(wis_from_bounds(y_seq, bounds, FLUSIGHT_ALPHAS, median=med_seq), dtype=float)
    lo95, hi95 = bounds[0.05]
    cov = (y_seq >= lo95) & (y_seq <= hi95)
    return {
        "wis": round(float(wis.mean()), 4),
        "picp95": round(float(cov.mean()), 4),
        "k_of_n": f"{int(cov.sum())}/{len(y_seq)}",
        "w95": round(float((hi95 - lo95).mean()), 2),
    }


def main():
    t_start = time.time()
    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test)
    ntot = ntr + nte
    frozen = np.asarray(json.loads(
        (ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE)
    tirex_pool = d["tirex_pool"]
    y_full = np.concatenate([y_train, y_test])
    cap = 2.0 * float(np.max(y_full))
    tirex_full = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])  # idx == week
    feat_full, foi_lag = build_features(y_train, y_test, X_train, X_test, tirex_full)
    r_full = y_full - tirex_full  # residual (nan < MIN_CTX)

    origins = np.arange(T0, ntot)
    n = len(origins)
    y_seq = y_full[origins]
    tirex_seq = tirex_full[origins]
    cal_idx = np.arange(T0 - K_CAL, T0)

    out = {"n_origins": int(n), "T0": T0, "weeks": f"{T0}..{ntot-1}",
           "cap": cap, "tirex_test_maxdiff_vs_frozen": None}

    # ================= (A) HARNESS RECONCILIATION: reproduce GBM cqr_static / pid_plain =====
    # Identical construction to scripts/dec_boosted_mech_multiorigin.py, gamma=0 (no mechanism)
    qy_gbm = np.zeros((n, len(FQ)), dtype=float)
    for bstart in range(T0, ntot, REFIT_K):
        bend = min(bstart + REFIT_K, ntot)
        train_end = bstart - K_CAL
        tr_idx = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat_full[tr_idx], r_full[tr_idx], cfg) for cfg in CONFIGS.values()]
        oi = np.arange(bstart, bend)
        qy_gbm[oi - T0] = bagged_qy(gbm, feat_full[oi], tirex_full[oi], cap)
    seed_train = np.arange(MIN_CTX, T0 - K_CAL)
    seed_gbm = [fit_gbm(feat_full[seed_train], r_full[seed_train], cfg) for cfg in CONFIGS.values()]
    qy_cal_gbm = bagged_qy(seed_gbm, feat_full[cal_idx], tirex_full[cal_idx], cap)
    cqr_gbm = cqr_offsets(qy_cal_gbm, y_full[cal_idx])
    med_gbm = qy_gbm[:, MED_COL]
    recon = {
        "gbm_cqr_static": score_bounds(build_bounds_cqr(qy_gbm, cqr_gbm, cap), y_seq, med_gbm),
        "gbm_pid_plain": score_bounds(build_bounds_pid(qy_gbm, cqr_gbm, y_seq, cap), y_seq, med_gbm),
    }
    out["harness_reconciliation"] = recon
    out["reference_json"] = {"cqr_static": [2.2688, 0.9848], "pid_plain": [2.3080, 0.9015]}

    # ================= (B) METHOD-MATCHED TiRex-alone (empirical offsets, IDENTICAL funcs) ===
    qy_te_alone = tirex_empirical_qy(tirex_full, r_full, origins, cap)
    qy_cal_alone = tirex_empirical_qy(tirex_full, r_full, cal_idx, cap)
    cqr_alone = cqr_offsets(qy_cal_alone, y_full[cal_idx])
    med_alone = qy_te_alone[:, MED_COL]

    b_cqr_alone = build_bounds_cqr(qy_te_alone, cqr_alone, cap)               # no foi_mult
    b_pid_alone = build_bounds_pid(qy_te_alone, cqr_alone, y_seq, cap)        # no foi_mult
    tirex_cqr = score_bounds(b_cqr_alone, y_seq, med_alone)
    tirex_pid = score_bounds(b_pid_alone, y_seq, med_alone)
    out["tirex_alone_method_matched"] = {"cqr": tirex_cqr, "pid": tirex_pid}

    # ================= (C) OLD apples baseline reproduction (sanity) ========================
    b_simple = simple_abs_resid_bounds(tirex_seq, tirex_full, r_full, origins, cap)
    out["tirex_alone_simple_absresid"] = score_bounds(b_simple, y_seq, tirex_seq)
    out["reference_simple_apples"] = {"wis": 2.5626, "picp95": 0.833}

    # ================= (D) VERDICT: does full method beat method-matched TiRex-alone? =======
    FULL_PID_MECH = 2.2194   # verified pid_mech g=1.0 WIS
    FULL_CQR_MECH = 2.2221   # verified cqr_mech g=0.75 WIS
    best_full = min(FULL_PID_MECH, FULL_CQR_MECH)
    best_fair = min(tirex_cqr["wis"], tirex_pid["wis"])
    margin_pct = 100.0 * (best_fair - best_full) / best_fair
    out["verdict"] = {
        "full_pid_mech_wis": FULL_PID_MECH,
        "full_cqr_mech_wis": FULL_CQR_MECH,
        "best_full_wis": best_full,
        "tirex_alone_cqr_wis": tirex_cqr["wis"],
        "tirex_alone_pid_wis": tirex_pid["wis"],
        "best_method_matched_fair_wis": best_fair,
        "method_beats_fair_baseline": bool(best_full < best_fair),
        "margin_pct_vs_best_fair": round(margin_pct, 2),
    }
    out["elapsed_sec"] = round(time.time() - t_start, 1)

    outp = ROOT / "scripts" / "_verify_fairbase.json"
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    raise SystemExit(main())
