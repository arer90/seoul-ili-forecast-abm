#!/usr/bin/env python
"""DEFINITIVE test of the coverage bar: rolling multi-origin evaluation (synthesis
fix #3 — "the real cure for the n=68 resolution-floor"). Breaks the 1.47pp coverage
quantization by evaluating the bagged mechanism-informed conformal method over ~130
leak-free 1-step origins instead of a single frozen 68-week hold-out.

Leak-free by construction:
  * TiRex 1-step base is available for every week 52..336 (tirex_pool ⊕ tirex_test_roll).
  * The whole origin range [T0,336] is treated as ONE sequence; per-origin GBM quantiles
    come from a bagged 6-capacity ensemble REFIT every K origins on that block's PAST only
    (train_end = block_start - K_CAL, strictly before every calibration week it serves).
  * Conformal-PID + foi-width run once over the sequence — PID adapts online using past
    y only (identical machinery to the frozen eval), foi multiplier is 1-lag past-only.
  * CQR seed calibrated on [T0-K_CAL, T0) from a seed-GBM trained on [52, T0-K_CAL).

Reports pooled WIS + pooled PICP95 (k/N) with Clopper-Pearson CI for each gamma, so the
question "does mechanism-informed conformal ROBUSTLY clear 0.93 once coverage is
resolvable?" gets a definitive, honest answer. Does NOT touch live/pipeline code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (
    FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y,
    load_split, build_features, predict_qy, cqr_offsets, foi_multipliers,
    build_bounds_cqr, build_bounds_pid,
)
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

CONFIGS = {
    "default": dict(lr=0.05, it=300, leaves=8, msl=15, depth=3),
    "weak":    dict(lr=0.03, it=150, leaves=4, msl=25, depth=2),
    "strong":  dict(lr=0.08, it=500, leaves=16, msl=10, depth=4),
    "shallow": dict(lr=0.05, it=300, leaves=4, msl=20, depth=2),
    "deep":    dict(lr=0.05, it=400, leaves=31, msl=8, depth=None),
    "lowlr":   dict(lr=0.02, it=600, leaves=8, msl=15, depth=3),
}
T0 = 205          # first eval origin (GBM gets ~113 train weeks at first refit)
REFIT_K = 4       # refit the bagged GBM every REFIT_K origins
GAMMAS = (0.0, 0.75, 1.0, 1.25, 1.5, 2.0)


def fit_gbm(Xtr, r_tr, cfg):
    models = {}
    for q in FQ:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=float(q), learning_rate=cfg["lr"],
            max_iter=cfg["it"], max_leaf_nodes=cfg["leaves"],
            min_samples_leaf=cfg["msl"], l2_regularization=1.0,
            max_depth=cfg["depth"], random_state=42)
        m.fit(Xtr, r_tr)
        models[round(float(q), 4)] = m
    return models


def bagged_qy(gbm_dicts, X, tirex, cap):
    stack = np.stack([predict_qy(g, X, tirex, cap) for g in gbm_dicts], axis=0)
    return np.sort(stack.mean(axis=0), axis=1)


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test)
    ntot = ntr + nte
    frozen = np.asarray(json.loads(
        (ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE)
    tirex_pool = d["tirex_pool"]                       # weeks 52..268
    y_full = np.concatenate([y_train, y_test])
    cap = 2.0 * float(np.max(y_full))
    tirex_full = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])  # idx=week
    feat_full, foi_lag = build_features(y_train, y_test, X_train, X_test, tirex_full)

    origins = np.arange(T0, ntot)                       # eval weeks
    n = len(origins)
    y_seq = y_full[origins]
    foi_seq = foi_lag[origins]
    foi_seed = foi_lag[MIN_CTX:T0]
    tirex_seq = tirex_full[origins]

    # ---- assemble per-origin bagged GBM quantiles (periodic refit, past-only) ----
    r_full = y_full - tirex_full                        # residual target (nan<52)
    qy_seq = np.zeros((n, len(FQ)), dtype=float)
    refit_starts = list(range(T0, ntot, REFIT_K))
    for bstart in refit_starts:
        bend = min(bstart + REFIT_K, ntot)
        train_end = bstart - K_CAL                      # strictly before every cal week in block
        tr_idx = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat_full[tr_idx], r_full[tr_idx], cfg) for cfg in CONFIGS.values()]
        oi = np.arange(bstart, bend)
        qy_seq[oi - T0] = bagged_qy(gbm, feat_full[oi], tirex_full[oi], cap)

    # ---- CQR seed on [T0-K_CAL, T0) from a seed-GBM trained on [52, T0-K_CAL) ----
    seed_train = np.arange(MIN_CTX, T0 - K_CAL)
    seed_gbm = [fit_gbm(feat_full[seed_train], r_full[seed_train], cfg) for cfg in CONFIGS.values()]
    cal_idx = np.arange(T0 - K_CAL, T0)
    qy_cal = bagged_qy(seed_gbm, feat_full[cal_idx], tirex_full[cal_idx], cap)
    cqr_seed = cqr_offsets(qy_cal, y_full[cal_idx])

    peak = y_seq >= PEAK_Y
    last_season = np.zeros(n, bool); last_season[n - 34:] = True
    med_seq = qy_seq[:, MED_COL]

    def evaluate(engine, gamma):
        mult = foi_multipliers(foi_seq, foi_seed, gamma) if gamma > 0 else None
        if engine == "cqr":
            b = build_bounds_cqr(qy_seq, cqr_seed, cap, foi_mult=mult)
        else:
            b = build_bounds_pid(qy_seq, cqr_seed, y_seq, cap, foi_mult=mult)
        wis = np.asarray(wis_from_bounds(y_seq, b, FLUSIGHT_ALPHAS, median=med_seq), dtype=float)
        lo95, hi95 = b[0.05]
        cov = (y_seq >= lo95) & (y_seq <= hi95)
        w95 = hi95 - lo95
        out = {}
        for mk, m in {"all": np.ones(n, bool), "peak": peak, "last34": last_season}.items():
            k = int(cov[m].sum()); nn = int(m.sum())
            cp = clopper_pearson(k, nn)
            out[mk] = {"wis": round(float(wis[m].mean()), 4),
                       "picp95": round(k / nn, 4), "k_of_n": f"{k}/{nn}",
                       "cp95ci": [round(cp[0], 3), round(cp[1], 3)],
                       "w95": round(float(w95[m].mean()), 2)}
        return out

    print(f"=== Rolling multi-origin eval: {n} origins (weeks {T0}..{ntot-1}), "
          f"refit every {REFIT_K}, bagged 6-cap ===")
    print(f"    peak(y>=50) origins: {int(peak.sum())} | last-34: 34 | n_total: {n}")
    print(f"    baselines (leak-free anchors): TiRex-alone frozen WIS≈2.951, "
          f"native-quantile 2.677, held-out stack 2.720\n")
    results = {"n_origins": int(n), "T0": T0, "refit_k": REFIT_K,
               "n_peak": int(peak.sum()), "gammas": {}}
    header = f"{'engine':>10s} {'g':>4s} | {'ALL wis':>8s} {'picp95':>7s} {'k/N':>7s} {'CP95ci':>14s} {'w95':>6s} | {'peak p95':>9s} | {'last34 wis':>10s}"
    print(header); print("-" * len(header))
    for engine in ("pid", "cqr"):
        for g in GAMMAS:
            if engine == "cqr" and g == 0.0:
                key = "cqr_static"
            elif engine == "pid" and g == 0.0:
                key = "pid_plain"
            else:
                key = f"{engine}_mech_g{g}"
            r = evaluate(engine, g)
            results["gammas"][key] = r
            a = r["all"]
            bar2 = "✓" if a["picp95"] >= 0.93 else "✗"
            print(f"{engine:>10s} {g:>4.2f} | {a['wis']:>8.4f} {a['picp95']:>7.4f}{bar2} "
                  f"{a['k_of_n']:>7s} {str(a['cp95ci']):>14s} {a['w95']:>6.2f} | "
                  f"{r['peak']['picp95']:>9.3f} | {r['last34']['wis']:>10.4f}")

    outp = ROOT / "scripts" / "_dec_boosted_mech_multiorigin.json"
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {outp}")
    # honest verdict
    pid_ok = [k for k, v in results["gammas"].items()
              if k.startswith("pid_mech") and v["all"]["picp95"] >= 0.93
              and v["all"]["wis"] < 2.68 and v["all"]["cp95ci"][0] >= 0.90]
    print("\n=== VERDICT ===")
    print(f"pid_mech gammas with picp95>=0.93 AND wis<2.68 AND CP-lower>=0.90: {pid_ok or 'NONE'}")


if __name__ == "__main__":
    raise SystemExit(main())
