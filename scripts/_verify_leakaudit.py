#!/usr/bin/env python
"""LEAK-FREE AUDIT probes for dec_boosted_mech_multiorigin.py.

Reproduces the 132-origin pipeline. The expensive per-block bagged GBM stack is
built ONCE per y_full (build_qy); bounds for any (engine,gamma) are derived
cheaply (derive_bounds). We then perturb FUTURE y and confirm earlier-origin
bounds are bit-identical (no leak). Prints JSON only. Touches no live/pipeline
or dec_boosted_mech* code.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (
    FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y,
    load_split, build_features, cqr_offsets, foi_multipliers,
    build_bounds_cqr, build_bounds_pid,
)
from scripts.dec_boosted_mech_multiorigin import fit_gbm, bagged_qy, CONFIGS, T0, REFIT_K
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

X_train, y_train, X_test, y_test, META = load_split()
NTR, NTE = len(y_train), len(y_test)
NTOT = NTR + NTE
FROZEN = np.asarray(json.loads(
    (ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
    ["refit_test_predictions"], dtype=float)
_d = np.load(D.TIREX_CACHE)
TIREX_POOL = _d["tirex_pool"]
Y_FULL_BASE = np.concatenate([y_train, y_test])
CAP_BASE = 2.0 * float(np.max(Y_FULL_BASE))
# tirex_full is INDEPENDENT of any y perturbation (pool cached, test=frozen json).
TIREX_FULL = np.concatenate([np.full(MIN_CTX, np.nan), TIREX_POOL, FROZEN])
ORIGINS = np.arange(T0, NTOT)


def build_qy(y_full, configs, cap):
    """Expensive part: per-block bagged GBM quantiles + CQR seed (mirrors main())."""
    feat_full, foi_lag = build_features(
        y_full[:NTR], y_full[NTR:], X_train, X_test, TIREX_FULL)
    cfgs = [CONFIGS[k] for k in configs]
    r_full = y_full - TIREX_FULL
    qy_seq = np.zeros((len(ORIGINS), len(FQ)), dtype=float)
    for bstart in range(T0, NTOT, REFIT_K):
        bend = min(bstart + REFIT_K, NTOT)
        train_end = bstart - K_CAL
        tr_idx = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat_full[tr_idx], r_full[tr_idx], cfg) for cfg in cfgs]
        oi = np.arange(bstart, bend)
        qy_seq[oi - T0] = bagged_qy(gbm, feat_full[oi], TIREX_FULL[oi], cap)
    seed_train = np.arange(MIN_CTX, T0 - K_CAL)
    seed_gbm = [fit_gbm(feat_full[seed_train], r_full[seed_train], cfg) for cfg in cfgs]
    cal_idx = np.arange(T0 - K_CAL, T0)
    qy_cal = bagged_qy(seed_gbm, feat_full[cal_idx], TIREX_FULL[cal_idx], cap)
    cqr_seed = cqr_offsets(qy_cal, y_full[cal_idx])
    return {"qy_seq": qy_seq, "cqr_seed": cqr_seed, "foi_lag": foi_lag}


def derive_bounds(pack, y_full, engine, gamma, cap):
    """Cheap: bounds {alpha:(lo,hi)} from a prebuilt qy pack for one (engine,gamma)."""
    qy_seq = pack["qy_seq"]; cqr_seed = pack["cqr_seed"]; foi_lag = pack["foi_lag"]
    y_seq = y_full[ORIGINS]
    foi_seq = foi_lag[ORIGINS]
    foi_seed = foi_lag[MIN_CTX:T0]
    mult = foi_multipliers(foi_seq, foi_seed, gamma) if gamma > 0 else None
    if engine == "cqr":
        b = build_bounds_cqr(qy_seq, cqr_seed, cap, foi_mult=mult)
    else:
        b = build_bounds_pid(qy_seq, cqr_seed, y_seq, cap, foi_mult=mult)
    med_seq = qy_seq[:, MED_COL]
    return b, y_seq, med_seq


def pooled(b, y_seq, med_seq):
    wis = np.asarray(wis_from_bounds(y_seq, b, FLUSIGHT_ALPHAS, median=med_seq), float)
    lo95, hi95 = b[0.05]
    cov = (y_seq >= lo95) & (y_seq <= hi95)
    return round(float(wis.mean()), 4), round(float(cov.mean()), 4), f"{int(cov.sum())}/{len(y_seq)}"


def bounds_stack(b):
    rows = []
    for a in FLUSIGHT_ALPHAS:
        lo, hi = b[a]
        rows.append(np.asarray(lo, float)); rows.append(np.asarray(hi, float))
    return np.vstack(rows)  # (2K, n)


ENG_G = [("pid", 1.0), ("cqr", 0.75)]


def main():
    t0 = time.time()
    out = {}

    # ---- index-bounds audit for every refit block (pure arithmetic) ----
    blocks = []
    min_gap = 10**9
    for bstart in range(T0, NTOT, REFIT_K):
        bend = min(bstart + REFIT_K, NTOT)
        train_end = bstart - K_CAL
        max_train_idx = train_end - 1
        gap = bstart - max_train_idx
        min_gap = min(min_gap, gap)
        blocks.append({"bstart": int(bstart), "bend": int(bend), "train_end": int(train_end),
                       "max_train_idx": int(max_train_idx), "first_eval_origin": int(bstart),
                       "gap_weeks": int(gap),
                       "train_before_all_served": bool(max_train_idx < bstart)})
    out["index_audit"] = {
        "n_blocks": len(blocks),
        "min_gap_weeks_train_to_first_served": int(min_gap),
        "all_train_end_strictly_before_bstart": all(b["train_before_all_served"] for b in blocks),
        "first_block": blocks[0], "last_block": blocks[-1],
        "seed_gbm_train_range": f"[{MIN_CTX},{T0 - K_CAL})",
        "cqr_seed_cal_range": f"[{T0 - K_CAL},{T0})",
        "cqr_cal_last_week": int(T0 - 1), "first_eval_origin_week": int(T0),
        "cqr_cal_strictly_before_first_origin": bool((T0 - 1) < T0),
    }

    # ---- reconcile pooled metrics vs stored JSON (full 6-config, qy built once) ----
    allcfg = list(CONFIGS.keys())
    pack6 = build_qy(Y_FULL_BASE, allcfg, CAP_BASE)
    recon = {}
    for eng, g in ENG_G:
        b, ys, ms = derive_bounds(pack6, Y_FULL_BASE, eng, g, CAP_BASE)
        w, p, kn = pooled(b, ys, ms)
        recon[f"{eng}_g{g}"] = {"wis": w, "picp95": p, "k_of_n": kn}
    stored = json.load(open(ROOT / "scripts/_dec_boosted_mech_multiorigin.json"))
    out["reconcile_full6cfg"] = {
        "computed": recon,
        "stored_pid_mech_g1.0": {k: stored["gammas"]["pid_mech_g1.0"]["all"][k] for k in ("wis", "picp95", "k_of_n")},
        "stored_cqr_mech_g0.75": {k: stored["gammas"]["cqr_mech_g0.75"]["all"][k] for k in ("wis", "picp95", "k_of_n")},
        "match_pid": bool(recon["pid_g1.0"]["wis"] == stored["gammas"]["pid_mech_g1.0"]["all"]["wis"]
                          and recon["pid_g1.0"]["k_of_n"] == stored["gammas"]["pid_mech_g1.0"]["all"]["k_of_n"]),
        "match_cqr": bool(recon["cqr_g0.75"]["wis"] == stored["gammas"]["cqr_mech_g0.75"]["all"]["wis"]
                          and recon["cqr_g0.75"]["k_of_n"] == stored["gammas"]["cqr_mech_g0.75"]["all"]["k_of_n"]),
    }

    # ---- PERTURBATION PROBES (1 config; cap pinned to isolate the online path) ----
    probe_cfg = ["default"]
    pack_base = build_qy(Y_FULL_BASE, probe_cfg, CAP_BASE)
    jA, jB = 270, NTOT - 1                     # middle future origin; last origin
    yA = Y_FULL_BASE.copy(); yA[jA] = 0.5 * yA[jA]     # perturb DOWN so 2*max(y) (cap) unaffected
    yB = Y_FULL_BASE.copy(); yB[jB] = 0.5 * yB[jB]
    pack_A = build_qy(yA, probe_cfg, CAP_BASE)
    pack_B = build_qy(yB, probe_cfg, CAP_BASE)

    probes = {}
    for eng, g in ENG_G:
        base_b, _, _ = derive_bounds(pack_base, Y_FULL_BASE, eng, g, CAP_BASE)
        base_s = bounds_stack(base_b)
        # Probe A
        bA, _, _ = derive_bounds(pack_A, yA, eng, g, CAP_BASE)
        sA = bounds_stack(bA)
        earlier = ORIGINS < jA
        later = ORIGINS >= jA
        max_earlier = float(np.max(np.abs(sA[:, earlier] - base_s[:, earlier])))
        changed_later = int(np.sum(np.any(sA[:, later] != base_s[:, later], axis=0)))
        probes[f"{eng}_g{g}_probeA_perturb_week{jA}"] = {
            "n_earlier_origins": int(earlier.sum()),
            "max_abs_bound_diff_earlier": max_earlier,
            "earlier_bounds_bit_identical": bool(max_earlier == 0.0),
            "n_later_origins_changed": changed_later, "n_later_origins": int(later.sum()),
            "nonvacuous_some_later_changed": bool(changed_later > 0),
        }
        # Probe B
        bB, _, _ = derive_bounds(pack_B, yB, eng, g, CAP_BASE)
        sB = bounds_stack(bB)
        max_all = float(np.max(np.abs(sB - base_s)))
        probes[f"{eng}_g{g}_probeB_perturb_last_week{jB}"] = {
            "max_abs_bound_diff_all_origins": max_all,
            "all_bounds_bit_identical": bool(max_all == 0.0),
        }
    out["perturbation_probes"] = probes

    out["cap_note"] = {
        "cap_formula": "2.0*max(y_full) — y_full includes the 68 test weeks",
        "cap_base": CAP_BASE, "argmax_week": int(np.argmax(Y_FULL_BASE)),
        "argmax_in_test_range": bool(int(np.argmax(Y_FULL_BASE)) >= NTR),
    }
    out["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
