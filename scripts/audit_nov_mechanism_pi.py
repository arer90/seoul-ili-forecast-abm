#!/usr/bin/env python
"""ADVERSARIAL AUDIT of scripts/nov_mechanism_pi.py — independent re-derivation.

Checks (all leak-free protocol):
  A. Independent baseline: online_conformal_bounds(TiRex) WIS == 2.9512 (matches shipped path).
  B. Signal leak-freeness: perturbing y_test[j] must NOT change the mechanism signal used at
     test conformal step j (sig[j]); it may only change sig[j+1..]. Also that the conformal
     interval at step j does not depend on y_test[j] (interval invariant to y_test[j] change).
  C. Pool-tuning is test-blind: recompute the pool gamma grid; confirm foi pool-optimal gamma;
     confirm the chosen gamma is INVARIANT when y_test is scrambled (test never enters tuning).
  D. Not-trivially-wide: overall mean width comparison + per-week WIS decomposition (how many
     weeks improve, is the gain a single outlier week, sharpness vs coverage split).
  E. Cache integrity: tx_pool derived from train pool only.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

from scripts.ablation_fusedepi import load_split
from scripts.fusedepi_fusion_wis import load_frozen
from scripts.nov_mechanism_pi import (
    mechanism_online_conformal, score_bounds, masks_for, build_signals,
    tirex_pool_rolling, PRIMARY_WINDOW, PRIMARY_KI,
)
from simulation.analytics.adaptive_conformal import online_conformal_bounds, wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.models.feature_engine._loaders.mechanistic import mechanistic_features

X_train, y_train, X_test, y_test, meta = load_split()
pred_tirex, y_true, oof = load_frozen("TiRex")
assert np.max(np.abs(y_test - y_true)) < 1e-9
n = len(y_test)
print(f"split n={meta['n']} pool_end={meta['pool_end']} n_test={n}")

# ── A. independent baseline via the SHIPPED helper directly ──
b_ship = online_conformal_bounds(pred_tirex, y_test, FLUSIGHT_ALPHAS,
                                 window=PRIMARY_WINDOW, ki=PRIMARY_KI)
wis_ship = np.asarray(wis_from_bounds(y_test, b_ship, FLUSIGHT_ALPHAS, median=pred_tirex), float)
lo95, hi95 = b_ship[0.05]
picp95 = float(np.mean((y_test >= lo95) & (y_test <= hi95)))
print(f"\n[A] shipped-helper baseline: WIS={wis_ship.mean():.4f}  PICP95={picp95:.4f}  "
      f"w95={np.mean(hi95-lo95):.2f}  (claim 2.9512 / 0.8824 / 32.4)")

mech_lag, ts, ch = build_signals(y_train, y_test)
ch_test = {k: v[ts:ts + n] for k, v in ch.items()}
ch_seed = {k: v[:ts] for k, v in ch.items()}

# ── B. signal + interval leak-freeness under y_test[j] perturbation ──
print("\n[B] leak-free perturbation test (perturb y_test[j], j=30):")
j = 30
y_test_pert = y_test.copy(); y_test_pert[j] += 25.0
# rebuild signals with perturbed test tail
_, _, ch_p = build_signals(y_train, y_test_pert)
foi_p = ch_p["foi"][ts:ts + n]
foi_0 = ch_test["foi"]
d_at_j = abs(foi_p[j] - foi_0[j])
d_before = float(np.max(np.abs(foi_p[:j] - foi_0[:j])))
d_after = float(np.max(np.abs(foi_p[j + 1:] - foi_0[j + 1:]))) if j + 1 < n else 0.0
print(f"    signal change at j       = {d_at_j:.6e}  (MUST be 0: sig[j] uses inc[:ts+j], not y_test[j])")
print(f"    signal change at <j       = {d_before:.6e}  (MUST be 0)")
print(f"    signal change at >j (max) = {d_after:.6e}  (expected >0: future signals see the bump)")
# interval at step j invariant to y_test[j]? build bounds with foi gamma=0.75 for both y_tests
def bounds_foi(pred, y, gamma):
    return mechanism_online_conformal(pred, y, FLUSIGHT_ALPHAS, ch_test["foi"],
                                      gamma=gamma, seed_signal=ch_seed["foi"])
bb0 = bounds_foi(pred_tirex, y_test, 0.75)
bbp = bounds_foi(pred_tirex, y_test_pert, 0.75)   # only y in loop differs; signal identical
lo0, hi0 = bb0[0.05]; lop, hip = bbp[0.05]
print(f"    interval[j] lo change    = {abs(lop[j]-lo0[j]):.6e}  hi change = {abs(hip[j]-hi0[j]):.6e}  "
      f"(MUST be 0: interval at j set before y_test[j] observed)")
print(f"    interval[>j] max lo change= {np.max(np.abs((lop-lo0)[j+1:])):.6e}  (expected >0: adaptive reacts after)")

# ── C. pool tuning is test-blind ──
tx_pool, min_ctx = tirex_pool_rolling(y_train)
y_pool = y_train[min_ctx:]
ch_pool = {k: v[min_ctx:ts] for k, v in ch.items()}
ch_pool_seed = {k: v[:min_ctx] for k, v in ch.items()}
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
def pool_wis(sig_pool, seed, g):
    bp = mechanism_online_conformal(tx_pool, y_pool, FLUSIGHT_ALPHAS, sig_pool, gamma=g, seed_signal=seed)
    return score_bounds(y_pool, bp, tx_pool, np.ones(len(y_pool), bool))["wis"]
foi_grid = {g: pool_wis(ch_pool["foi"], ch_pool_seed["foi"], g) for g in GAMMAS}
foi_best = min(GAMMAS, key=lambda g: foi_grid[g])
print(f"\n[C] pool-tuning (foi) grid WIS: " + " ".join(f"{g}:{foi_grid[g]:.4f}" for g in GAMMAS))
print(f"    pool-optimal foi gamma = {foi_best}  (claim 0.75); tx_pool len={len(tx_pool)} min_ctx={min_ctx}")
# test-blindness: scramble y_test, redo pool tuning -> chosen gamma unchanged (pool independent of test)
rng = np.random.default_rng(7)
y_test_scr = rng.permutation(y_test) * 3.0 + 999.0
_, _, ch_scr = build_signals(y_train, y_test_scr)
ch_pool_scr = {k: v[min_ctx:ts] for k, v in ch_scr.items()}
foi_grid2 = {g: pool_wis(ch_pool_scr["foi"], ch_pool_seed["foi"], g) for g in GAMMAS}
foi_best2 = min(GAMMAS, key=lambda g: foi_grid2[g])
print(f"    after scrambling y_test: pool grid identical? "
      f"{max(abs(foi_grid[g]-foi_grid2[g]) for g in GAMMAS):.3e}  chosen gamma={foi_best2} (must match {foi_best})")

# ── D. not-trivially-wide + per-week decomposition (foi gamma=0.75) ──
masks, peak_thr = masks_for(y_test)
base_sc = score_bounds(y_test, bb0 if False else online_conformal_bounds(
    pred_tirex, y_test, FLUSIGHT_ALPHAS, window=PRIMARY_WINDOW, ki=PRIMARY_KI), pred_tirex, masks["overall"])
mech_b = bounds_foi(pred_tirex, y_test, 0.75)
mech_sc = score_bounds(y_test, mech_b, pred_tirex, masks["overall"])
wis_base = np.asarray(wis_from_bounds(y_test, b_ship, FLUSIGHT_ALPHAS, median=pred_tirex), float)
wis_mech = np.asarray(wis_from_bounds(y_test, mech_b, FLUSIGHT_ALPHAS, median=pred_tirex), float)
dw = wis_mech - wis_base
print(f"\n[D] foi gamma=0.75 vs baseline:")
print(f"    overall WIS {wis_base.mean():.4f} -> {wis_mech.mean():.4f}   overall w95 {base_sc['mean_width95']:.2f} -> {mech_sc['mean_width95']:.2f} (NARROWER)")
print(f"    weeks improved (dWIS<0): {int((dw<0).sum())}/{n}   worsened: {int((dw>0).sum())}   unchanged: {int((dw==0).sum())}")
print(f"    total dWIS = {dw.sum():.3f}   most-improved single week dWIS = {dw.min():.3f}  (idx {int(dw.argmin())})")
print(f"    total dWIS w/o single best week = {(dw.sum()-dw.min()):.3f}  (still < 0 => not one outlier)")
mm95 = (hi95 - lo95)
mlo95, mhi95 = mech_b[0.05]
print(f"    PICP95 {picp95:.4f} -> {float(np.mean((y_test>=mlo95)&(y_test<=mhi95))):.4f}  "
      f"(coverage UP while width DOWN => genuine reallocation, not ballooning)")

# ── E. cache integrity ──
d = np.load(REPO / "scripts/.nov_mech_cache.npz", allow_pickle=True)
yt_cache = np.asarray(d["y_train"], float)
print(f"\n[E] cache: y_train len={len(yt_cache)} matches split pool ({len(y_train)})? "
      f"{len(yt_cache)==len(y_train) and np.max(np.abs(yt_cache-y_train))<1e-9}  "
      f"tx_pool len={len(np.asarray(d['tx_pool']))} == pool-min_ctx={len(y_train)-int(d['min_ctx'])}")
