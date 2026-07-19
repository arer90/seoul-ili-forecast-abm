#!/usr/bin/env python
"""ABLATION DECOMPOSITION + SIGNIFICANCE over the SAME 132 rolling 1-step origins.

Standalone verifier (NO live/pipeline code touched, does NOT modify dec_boosted_mech*.py).
Replicates the EXACT multi-origin setup of scripts.dec_boosted_mech_multiorigin (T0=205,
REFIT_K=4, bagged 6-cap GBM, fixed CQR seed on [T0-K_CAL,T0)) so the reconstructed
per-origin WIS arrays reproduce the verified means, then decomposes the WIS gain into:

  CQR ladder:  (A) TiRex-alone method-matched CQR (no GBM)
               (B) +GBM residual  = cqr_static (gamma=0)
               (C) +mechanism     = cqr_mech best gamma (0.75)
  PID ladder:  (A) TiRex-alone method-matched PID (no GBM)
               (B) +GBM residual  = pid_plain (gamma=0)
               (C) +mechanism     = pid_mech best gamma (1.0)

deltas: gbm_delta = WIS(A)-WIS(B) ; mech_delta = WIS(B)-WIS(C).
Significance of the MECHANISM step (B->C) on the paired per-origin WIS arrays:
  * Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction (h=1)
  * DM with Newey-West HAC variance (robustness to serial correlation of origins)
  * paired bootstrap 10k over the 132 origins (iid) + moving-block bootstrap (L=6)
  * Wilcoxon signed-rank (nonparametric cross-check)

Every printed number traces to this run. Prints JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (
    FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y,
    load_split, build_features, predict_qy, cqr_offsets, foi_multipliers,
    build_bounds_cqr, build_bounds_pid,
)
from scripts.dec_boosted_mech_multiorigin import CONFIGS, T0, REFIT_K, fit_gbm, bagged_qy
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

RNG = np.random.default_rng(42)
CQR_BEST_GAMMA = 0.75
PID_BEST_GAMMA = 1.0


# ───────────────────────── significance machinery ─────────────────────────
def dm_hln(d: np.ndarray, h: int = 1):
    """Diebold-Mariano stat with Harvey-Leybourne-Newbold (1997) small-sample
    correction. d = per-origin loss differential (loss_B - loss_C). h-1 autocovs."""
    d = np.asarray(d, float)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.mean(dc * dc)
    acov = gamma0
    for k in range(1, h):
        acov += 2.0 * np.mean(dc[k:] * dc[:-k])
    var_dbar = acov / n
    dm = dbar / np.sqrt(var_dbar)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = dm * corr
    p = 2.0 * stats.t.sf(abs(dm_star), df=n - 1)
    return {"dbar": float(dbar), "dm": float(dm), "dm_hln": float(dm_star),
            "df": n - 1, "p_value": float(p)}


def dm_hac(d: np.ndarray, lag: int | None = None):
    """DM with Newey-West HAC long-run variance (Bartlett). Robust to serial
    correlation of adjacent rolling origins."""
    d = np.asarray(d, float)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))  # Newey-West 1994 rule
    g0 = np.mean(dc * dc)
    lrv = g0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * np.mean(dc[k:] * dc[:-k])
    var_dbar = lrv / n
    dm = dbar / np.sqrt(var_dbar)
    p = 2.0 * stats.norm.sf(abs(dm))
    return {"dbar": float(dbar), "nw_lag": int(lag), "dm_hac": float(dm),
            "p_value": float(p)}


def paired_bootstrap(d: np.ndarray, B: int = 10000, block: int = 1):
    """Bootstrap the mean of the per-origin differential d. block=1 -> iid paired
    bootstrap; block>1 -> moving-block bootstrap (temporal dependence aware).
    Two-sided p = 2*min(P(mean<=0),P(mean>=0)) from the resampled means; plus a
    centered-null p (shift d to mean 0, count |resample mean| >= |observed|)."""
    d = np.asarray(d, float)
    n = len(d)
    obs = d.mean()
    means = np.empty(B, float)
    if block <= 1:
        idx = RNG.integers(0, n, size=(B, n))
        means = d[idx].mean(axis=1)
    else:
        nblocks = int(np.ceil(n / block))
        starts_all = RNG.integers(0, n - block + 1, size=(B, nblocks))
        for b in range(B):
            picks = np.concatenate([np.arange(s, s + block) for s in starts_all[b]])[:n]
            means[b] = d[picks].mean()
    p_sign = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    p_sign = min(1.0, float(p_sign))
    # centered null (percentile-t style, unstudentized)
    means_c = means - obs
    p_null = float((np.abs(means_c) >= abs(obs)).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"observed_mean_delta": float(obs), "boot_ci95": [float(lo), float(hi)],
            "p_sign": float(p_sign), "p_centered_null": p_null,
            "frac_boot_le_0": float((means <= 0).mean()), "B": B, "block": block}


# ───────────────────────── main ─────────────────────────
def main():
    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test)
    ntot = ntr + nte
    frozen = np.asarray(json.loads(
        (ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
        ["refit_test_predictions"], dtype=float)
    dcache = np.load(D.TIREX_CACHE)
    tirex_pool = dcache["tirex_pool"]
    y_full = np.concatenate([y_train, y_test])
    cap = 2.0 * float(np.max(y_full))
    tirex_full = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    feat_full, foi_lag = build_features(y_train, y_test, X_train, X_test, tirex_full)

    origins = np.arange(T0, ntot)
    n = len(origins)
    y_seq = y_full[origins]
    foi_seq = foi_lag[origins]
    foi_seed = foi_lag[MIN_CTX:T0]

    # ---- bagged GBM residual quantiles: periodic refit, past-only (EXACT replica) ----
    r_full = y_full - tirex_full
    qy_seq = np.zeros((n, len(FQ)), dtype=float)
    for bstart in range(T0, ntot, REFIT_K):
        bend = min(bstart + REFIT_K, ntot)
        train_end = bstart - K_CAL
        tr_idx = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat_full[tr_idx], r_full[tr_idx], cfg) for cfg in CONFIGS.values()]
        oi = np.arange(bstart, bend)
        qy_seq[oi - T0] = bagged_qy(gbm, feat_full[oi], tirex_full[oi], cap)

    # ---- CQR seed for the GBM ladder on [T0-K_CAL,T0) (EXACT replica) ----
    seed_train = np.arange(MIN_CTX, T0 - K_CAL)
    seed_gbm = [fit_gbm(feat_full[seed_train], r_full[seed_train], cfg) for cfg in CONFIGS.values()]
    cal_idx = np.arange(T0 - K_CAL, T0)
    qy_cal = bagged_qy(seed_gbm, feat_full[cal_idx], tirex_full[cal_idx], cap)
    cqr_seed = cqr_offsets(qy_cal, y_full[cal_idx])
    med_seq = qy_seq[:, MED_COL]

    # ---- TiRex-alone method-matched: degenerate quantiles all = TiRex ----
    tirex_seq = tirex_full[origins]
    qy_tirex_seq = np.repeat(tirex_seq[:, None], len(FQ), axis=1)
    qy_tirex_cal = np.repeat(tirex_full[cal_idx][:, None], len(FQ), axis=1)
    cqr_seed_tirex = cqr_offsets(qy_tirex_cal, y_full[cal_idx])

    def wis_arr(bounds, median):
        return np.asarray(wis_from_bounds(y_seq, bounds, FLUSIGHT_ALPHAS, median=median), float)

    def picp95(bounds):
        lo, hi = bounds[0.05]
        cov = (y_seq >= lo) & (y_seq <= hi)
        return float(cov.mean()), f"{int(cov.sum())}/{n}"

    # ===== CQR ladder =====
    bA_cqr = build_bounds_cqr(qy_tirex_seq, cqr_seed_tirex, cap)               # A: TiRex-alone CQR
    bB_cqr = build_bounds_cqr(qy_seq, cqr_seed, cap)                           # B: cqr_static
    multC = foi_multipliers(foi_seq, foi_seed, CQR_BEST_GAMMA)
    bC_cqr = build_bounds_cqr(qy_seq, cqr_seed, cap, foi_mult=multC)           # C: cqr_mech g0.75
    wA_cqr = wis_arr(bA_cqr, tirex_seq)
    wB_cqr = wis_arr(bB_cqr, med_seq)
    wC_cqr = wis_arr(bC_cqr, med_seq)

    # ===== PID ladder =====
    bA_pid = build_bounds_pid(qy_tirex_seq, cqr_seed_tirex, y_seq, cap)        # A: TiRex-alone PID
    bB_pid = build_bounds_pid(qy_seq, cqr_seed, y_seq, cap)                    # B: pid_plain
    multCp = foi_multipliers(foi_seq, foi_seed, PID_BEST_GAMMA)
    bC_pid = build_bounds_pid(qy_seq, cqr_seed, y_seq, cap, foi_mult=multCp)   # C: pid_mech g1.0
    wA_pid = wis_arr(bA_pid, tirex_seq)
    wB_pid = wis_arr(bB_pid, med_seq)
    wC_pid = wis_arr(bC_pid, med_seq)

    def rung(w, b):
        p, kn = picp95(b)
        return {"wis": round(float(w.mean()), 4), "picp95": round(p, 4), "k_of_n": kn}

    ladders = {
        "cqr": {
            "A_tirex_alone": rung(wA_cqr, bA_cqr),
            "B_plus_gbm_static": rung(wB_cqr, bB_cqr),
            "C_plus_mech_g0.75": rung(wC_cqr, bC_cqr),
            "gbm_delta": round(float(wA_cqr.mean() - wB_cqr.mean()), 4),
            "mech_delta": round(float(wB_cqr.mean() - wC_cqr.mean()), 4),
        },
        "pid": {
            "A_tirex_alone": rung(wA_pid, bA_pid),
            "B_plus_gbm_plain": rung(wB_pid, bB_pid),
            "C_plus_mech_g1.0": rung(wC_pid, bC_pid),
            "gbm_delta": round(float(wA_pid.mean() - wB_pid.mean()), 4),
            "mech_delta": round(float(wB_pid.mean() - wC_pid.mean()), 4),
        },
    }

    # ===== significance of the MECHANISM step (B->C), d = wis_B - wis_C =====
    d_cqr = wB_cqr - wC_cqr
    d_pid = wB_pid - wC_pid

    def sig_battery(d):
        return {
            "n_origins": len(d),
            "n_nonzero_diff": int((np.abs(d) > 1e-9).sum()),
            "dm_hln_h1": dm_hln(d, h=1),
            "dm_hac_nw": dm_hac(d),
            "bootstrap_iid_10k": paired_bootstrap(d, B=10000, block=1),
            "bootstrap_block6_10k": paired_bootstrap(d, B=10000, block=6),
            "wilcoxon": {k: float(v) for k, v in zip(
                ("stat", "p_value"),
                stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                               correction=False)
                if int((np.abs(d) > 1e-12).sum()) > 0 else (float("nan"), float("nan")))},
        }

    sig = {"cqr_mech": sig_battery(d_cqr), "pid_mech": sig_battery(d_pid)}

    # ===== reconciliation vs the verified apples-to-apples TiRex-alone =====
    recon = {
        "tirex_alone_cqr_wis_here": round(float(wA_cqr.mean()), 4),
        "tirex_alone_cqr_picp95_here": round(picp95(bA_cqr)[0], 4),
        "verified_apples_tirex_alone_wis": 2.5626,
        "verified_apples_tirex_alone_picp95": 0.833,
        "note": "rung-A is fixed-seed method-matched CQR (same machinery as B/C); the "
                "apples anchor 2.5626 used its own empirical conformal — compare, do not assume equal.",
    }

    out = {
        "setup": {"n_origins": int(n), "T0": T0, "refit_k": REFIT_K,
                  "n_configs": len(CONFIGS), "cqr_best_gamma": CQR_BEST_GAMMA,
                  "pid_best_gamma": PID_BEST_GAMMA},
        "ladders": ladders,
        "significance_mechanism_step": sig,
        "reconciliation": recon,
        "verified_targets": {
            "cqr_static": 2.2688, "cqr_mech_g0.75": 2.2221,
            "pid_plain": 2.308, "pid_mech_g1.0": 2.2194,
        },
    }
    # save per-origin arrays for auditability
    np.savez(ROOT / "scripts" / "_verify_ablation_wis.npz",
             wA_cqr=wA_cqr, wB_cqr=wB_cqr, wC_cqr=wC_cqr,
             wA_pid=wA_pid, wB_pid=wB_pid, wC_pid=wC_pid, y_seq=y_seq)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (ROOT / "scripts" / "_verify_ablation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
