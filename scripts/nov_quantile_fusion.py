#!/usr/bin/env python
"""NOVELTY — DIRECT QUANTILE FUSION (TiRex native quantiles ⊕ count-aware NegBin).

Idea
----
Every prior fusion in this project fuses a POINT base + a residual correction, then
conformalizes a single point. Experiment 3 showed that path never beats TiRex-alone on
WIS (TiRex point + online conformal = 2.951; FusedEpi point = 3.011; naive fusions
worse) and all methods UNDER-cover (~0.88 vs 0.95 nominal, peaks missed).

This script tests a genuinely different object: fuse the two *quantile functions*
directly, per quantile level, with a learned convex combination.

    q_fused(tau) = w(tau) * q_TiRex(tau) + (1 - w(tau)) * q_NegBin(tau)

- q_TiRex(tau): TiRex's NATIVE 9-decile output (deciles 0.1..0.9), expanded to the 23
  FluSight quantiles by monotone probit inter/extrapolation. TiRex quantiles are SHARP
  but too narrow (80% PI width ~7.8 vs residual std ~4.9 -> under-covers).
- q_NegBin(tau): a count-aware Negative-Binomial quantile head centred on TiRex's own
  mean (the strongest point), variance mu + phi*mu^2. Being count-aware, it widens
  automatically at high mu (peaks) and is right-skewed -> proactive peak widening that a
  reactive conformal wrapper cannot supply ahead of a miss.
- w(tau) in [0,1] learned PER QUANTILE LEVEL by minimising pinball loss on TRAIN rolling
  1-step predictions only (no test peeking). The count-dispersion phi is likewise chosen
  on train by min train WIS. Weights/phi are then FROZEN for the 68-week test.

After fusion the 23 quantiles are rearranged to be monotone (Chernozhukov 2010), then
each central PI is corrected by the project's own adaptive Conformal-PID
(simulation.analytics.adaptive_conformal._pid_adjust), seeded leak-free from train-tail
nonconformity scores and updated only from PAST test observations.

Leak-free guarantees
--------------------
* Frozen run_data split via scripts.ablation_fusedepi.load_split (pool_end=269, n_test=68).
* TiRex rolled with max_context=512 -> reproduces the OFFICIAL frozen TiRex point exactly
  (verified maxdiff 0.000 vs per_model_optimal/TiRex.json refit_test_predictions).
* Every tunable (w(tau), phi, conformal seed) is fit on the train pool / train tail only.
* Conformal step i uses obs[0..i-1] only.

Baselines to beat (68-wk test, from experiment 3 / scripts/fusedepi_fusion_wis.py):
  TiRex-alone WIS 2.951 / PICP95 0.882 ; FusedEpi point 3.011 ; equal 3.274 ;
  inverse-OOF 3.228 ; Protocol-B invRMSE stack 2.7205 (scored on last-34 only) ;
  FusedEpi native head 3.278 / PICP95 0.735.

Reuses the sanctioned helpers: FLUSIGHT_ALPHAS/QUANTILES, wis_from_bounds,
online_conformal_bounds (for the anchor line), adaptive conformal _pid_adjust.
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

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.stats import nbinom, norm

from simulation.analytics.adaptive_conformal import (
    _pid_adjust,
    online_conformal_bounds,
    wis_from_bounds,
)
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS, FLUSIGHT_QUANTILES
from scripts.ablation_fusedepi import load_split

LOG = logging.getLogger("nov_quantile_fusion")
OUT_JSON = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/novelty/quantile_fusion.json"
)

TIREX_DECILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
FQ = np.asarray(FLUSIGHT_QUANTILES, dtype=float)              # 23 levels
FQ_COL = {round(float(q), 4): i for i, q in enumerate(FQ)}    # level -> column
MAX_CONTEXT = 512
WARMUP = 60                 # train rolling start index (>= any TiRex min context need)
CONF_WINDOW = 30            # matches experiment-3 online conformal window
CONF_KI = 0.2              # matches experiment-3 online conformal ki
SEED_TAIL = 60              # train-tail weeks used to seed conformal nonconformity buffer
PEAK_Y = 50.0


def seed_all(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


# ─────────────────────────── TiRex native rolling ───────────────────────────
def roll_tirex(model, y_full: np.ndarray, idxs, max_context: int = MAX_CONTEXT):
    """Rolling 1-step TiRex over idxs. Returns (means[len], deciles[len,9]).

    Leak-free: forecast for index t uses context y_full[:t] only.
    """
    import torch
    means = np.empty(len(idxs), dtype=float)
    dec = np.empty((len(idxs), 9), dtype=float)
    with torch.no_grad():
        for k, t in enumerate(idxs):
            ctx = torch.tensor(y_full[max(0, t - max_context):t], dtype=torch.float32).unsqueeze(0)
            q, mean = model.forecast(context=ctx, prediction_length=1)
            means[k] = float(np.asarray(mean).ravel()[0])
            dec[k] = np.asarray(q, dtype=float).ravel()
    return means, dec


# ───────────────────── quantile expansion / fusion helpers ───────────────────
def expand_tirex_to_flusight(deciles: np.ndarray) -> np.ndarray:
    """Monotone probit inter/extrapolation of 9 TiRex deciles to the 23 FluSight levels.

    Interior (0.1..0.9): piecewise-linear in probit space. Tails: linear extrapolation
    using the outermost decile slope in probit space (Gaussian-tail assumption), then
    non-negativity + per-row monotone sort.
    Args: deciles (n,9) at levels 0.1..0.9. Returns: (n, 23) at FLUSIGHT_QUANTILES.
    """
    zd = norm.ppf(TIREX_DECILES)                 # probit of deciles
    zt = norm.ppf(FQ)                            # probit of target 23 levels
    n = deciles.shape[0]
    out = np.empty((n, len(FQ)), dtype=float)
    slope_lo = (deciles[:, 1] - deciles[:, 0]) / (zd[1] - zd[0])   # decile 0.1->0.2
    slope_hi = (deciles[:, 8] - deciles[:, 7]) / (zd[8] - zd[7])   # decile 0.8->0.9
    for j, z in enumerate(zt):
        if z <= zd[0]:
            out[:, j] = deciles[:, 0] + slope_lo * (z - zd[0])
        elif z >= zd[-1]:
            out[:, j] = deciles[:, 8] + slope_hi * (z - zd[-1])
        else:
            # vectorised piecewise-linear interp per row
            out[:, j] = [np.interp(z, zd, deciles[i]) for i in range(n)]
    out = np.clip(out, 0.0, None)
    out.sort(axis=1)                             # enforce monotone across levels
    return out


def negbin_flusight(mu: np.ndarray, phi: float, cap: float) -> np.ndarray:
    """Count-aware NegBin quantiles at the 23 FluSight levels. var = mu + phi*mu^2.

    Args: mu (n,) count mean (TiRex mean); phi dispersion (>0); cap upper clip.
    Returns: (n,23).
    """
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    r = 1.0 / max(phi, 1e-4)
    p = r / (r + mu)
    out = np.empty((len(mu), len(FQ)), dtype=float)
    for j, q in enumerate(FQ):
        out[:, j] = np.clip(np.asarray(nbinom.ppf(q, r, p), dtype=float), 0.0, cap)
    return out


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def tune_weights(qA: np.ndarray, qB: np.ndarray, y: np.ndarray, grid=None) -> np.ndarray:
    """Per-level convex weight w(tau) in [0,1] minimising pinball on train. Returns (23,)."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 41)
    w = np.empty(len(FQ), dtype=float)
    for j, tau in enumerate(FQ):
        losses = [pinball(y, g * qA[:, j] + (1.0 - g) * qB[:, j], tau) for g in grid]
        w[j] = float(grid[int(np.argmin(losses))])
    return w


def fuse(qA: np.ndarray, qB: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Convex per-level fusion + per-row monotone rearrangement."""
    f = w[None, :] * qA + (1.0 - w[None, :]) * qB
    f = np.clip(f, 0.0, None)
    f.sort(axis=1)
    return f


def wis_train(fused: np.ndarray, y: np.ndarray) -> float:
    """Un-conformalised WIS of a (n,23) fused quantile matrix (for train phi selection)."""
    bounds = {}
    for a in FLUSIGHT_ALPHAS:
        lo = fused[:, FQ_COL[round(a / 2.0, 4)]]
        hi = fused[:, FQ_COL[round(1.0 - a / 2.0, 4)]]
        bounds[a] = (lo, hi)
    med = fused[:, FQ_COL[0.5]]
    return float(np.mean(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=med)))


# ─────────────────────────── conformal + scoring ────────────────────────────
def conformalize(fused: np.ndarray, y_obs: np.ndarray, seed_fused: np.ndarray,
                 seed_y: np.ndarray, cap: float) -> dict:
    """Adaptive Conformal-PID per central level on fused quantiles. Leak-free.

    Args: fused (n,23) test fused quantiles; y_obs (n,) rolling test obs; seed_* train-tail
    fused quantiles + y (nonconformity seed); cap upper clip.
    Returns: {alpha: (lo(n), hi(n))}.
    """
    bounds = {}
    for a in FLUSIGHT_ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]
        ch = FQ_COL[round(1.0 - a / 2.0, 4)]
        qlo = fused[:, cl].copy()
        qhi = fused[:, ch].copy()
        seed_lo = seed_fused[:, cl]
        seed_hi = seed_fused[:, ch]
        init_scores = np.maximum(seed_lo - seed_y, seed_y - seed_hi)   # CQR nonconformity
        nlo, nhi = _pid_adjust(qlo, qhi, y_obs, init_scores, beta=1.0 - a, target=a,
                               window=CONF_WINDOW, ki=CONF_KI, cap=cap)
        bounds[a] = (nlo, nhi)
    return bounds


def score(bounds: dict, y: np.ndarray, median: np.ndarray, masks: dict) -> dict:
    wis_arr = np.asarray(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=median), dtype=float)
    lo95, hi95 = bounds[0.05]
    cov95 = (y >= lo95) & (y <= hi95)
    lo50, hi50 = bounds[0.50]
    cov50 = (y >= lo50) & (y <= hi50)
    out = {}
    for mk, m in masks.items():
        m = np.asarray(m, bool)
        out[mk] = {
            "wis": float(np.mean(wis_arr[m])),
            "picp95": float(np.mean(cov95[m])),
            "picp50": float(np.mean(cov50[m])),
            "mean_width95": float(np.mean((hi95 - lo95)[m])),
            "n": int(m.sum()),
        }
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(42)
    t0 = time.time()
    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test)
    LOG.info("split train=%d test=%d peak(y>=%.0f)=%d", ntr, nte, PEAK_Y, int((y_test >= PEAK_Y).sum()))

    from tirex import load_model
    model = load_model("NX-AI/TiRex", device="cpu")
    y_full_train = y_train.copy()
    y_full_test = np.concatenate([y_train, y_test])
    cap = 2.0 * float(np.max(y_full_test))

    # ── roll TiRex: train (for tuning) and test ──
    tr_idx = list(range(WARMUP, ntr))
    te_idx = list(range(ntr, ntr + nte))
    LOG.info("rolling TiRex train (%d) ...", len(tr_idx))
    m_tr, dec_tr = roll_tirex(model, y_full_train, tr_idx)
    LOG.info("rolling TiRex test (%d) ...", len(te_idx))
    m_te, dec_te = roll_tirex(model, y_full_test, te_idx)
    y_tr_roll = y_train[WARMUP:]

    # verify test TiRex point == frozen official
    fro = np.asarray(json.loads((ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
                     ["refit_test_predictions"], dtype=float)
    tirex_point_maxdiff = float(np.max(np.abs(m_te - fro)))
    LOG.info("TiRex test point vs frozen official maxdiff=%.6f", tirex_point_maxdiff)

    # ── expand both quantile sources to 23 FluSight levels ──
    qA_tr = expand_tirex_to_flusight(dec_tr)
    qA_te = expand_tirex_to_flusight(dec_te)

    # ── choose count dispersion phi on TRAIN by min train WIS; tune w per level per phi ──
    phi_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    best = None
    for phi in phi_grid:
        qB_tr = negbin_flusight(m_tr, phi, cap)
        w = tune_weights(qA_tr, qB_tr, y_tr_roll)
        f_tr = fuse(qA_tr, qB_tr, w)
        wtr = wis_train(f_tr, y_tr_roll)
        LOG.info("  phi=%.3f train_WIS=%.4f w_mean=%.3f", phi, wtr, float(w.mean()))
        if best is None or wtr < best["train_wis"]:
            best = {"phi": phi, "w": w, "train_wis": wtr}
    phi = best["phi"]
    w = best["w"]
    LOG.info("selected phi=%.3f train_WIS=%.4f", phi, best["train_wis"])

    # ── build train fused (for conformal seed) and test fused with frozen phi,w ──
    qB_tr = negbin_flusight(m_tr, phi, cap)
    f_tr = fuse(qA_tr, qB_tr, w)
    qB_te = negbin_flusight(m_te, phi, cap)
    f_te = fuse(qA_te, qB_te, w)

    seed_fused = f_tr[-SEED_TAIL:]
    seed_y = y_tr_roll[-SEED_TAIL:]

    # masks
    overall = np.ones(nte, dtype=bool)
    peak = y_test >= PEAK_Y
    peak25 = y_test >= float(np.quantile(y_test, 0.75))
    last34 = np.zeros(nte, dtype=bool); last34[nte // 2:] = True
    masks = {"overall_68": overall, "peak_y50": peak, "peak_top25pct": peak25, "last34": last34}

    med_fused = f_te[:, FQ_COL[0.5]]

    # ═══════════════ variants (all leak-free) ═══════════════
    results = {}

    # V1 MAIN: learned per-quantile fusion + adaptive conformal, fused median
    b1 = conformalize(f_te, y_test, seed_fused, seed_y, cap)
    results["V1_learned_fusion_pid"] = score(b1, y_test, med_fused, masks)

    # V1b: same but median = TiRex mean (best point) instead of fused median
    results["V1b_learned_fusion_pid_tirexmedian"] = score(b1, y_test, m_te, masks)

    # V2: TiRex-native-only quantiles (w=1) + adaptive conformal  (isolates fusion)
    f_te_A = fuse(qA_te, qB_te, np.ones(len(FQ)))
    seed_A = fuse(qA_tr, qB_tr, np.ones(len(FQ)))[-SEED_TAIL:]
    b2 = conformalize(f_te_A, y_test, seed_A, seed_y, cap)
    results["V2_tirex_native_only_pid"] = score(b2, y_test, f_te_A[:, FQ_COL[0.5]], masks)

    # V3: NegBin-only quantiles (w=0) + adaptive conformal  (isolates fusion)
    f_te_B = fuse(qA_te, qB_te, np.zeros(len(FQ)))
    seed_B = fuse(qA_tr, qB_tr, np.zeros(len(FQ)))[-SEED_TAIL:]
    b3 = conformalize(f_te_B, y_test, seed_B, seed_y, cap)
    results["V3_negbin_only_pid"] = score(b3, y_test, f_te_B[:, FQ_COL[0.5]], masks)

    # V4: learned fusion, NO conformal (raw calibration of the fused quantiles)
    b4 = {a: (f_te[:, FQ_COL[round(a / 2.0, 4)]], f_te[:, FQ_COL[round(1.0 - a / 2.0, 4)]])
          for a in FLUSIGHT_ALPHAS}
    results["V4_learned_fusion_raw_noconformal"] = score(b4, y_test, med_fused, masks)

    # ═══════════════ external anchors recomputed in-script ═══════════════
    # A0: TiRex frozen point + online conformal (== experiment-3 TiRex-alone 2.951)
    b_anchor = online_conformal_bounds(m_te, y_test, FLUSIGHT_ALPHAS,
                                       window=CONF_WINDOW, ki=CONF_KI)
    results["A0_tirex_point_online_conformal"] = score(b_anchor, y_test, m_te, masks)

    # A1: TiRex point + adaptive PID conformal seeded from train residual halfwidths
    #     (TiRex-alone under THE SAME adaptive conformal my method uses -> fair isolate)
    from simulation.analytics.adaptive_conformal import adaptive_conformal_bounds
    tr_res = y_tr_roll - m_tr
    hw = {a: float(np.quantile(np.abs(tr_res), 1.0 - a)) for a in FLUSIGHT_ALPHAS}
    b_a1 = adaptive_conformal_bounds(m_te, hw, tr_res[-SEED_TAIL:], y_test, FLUSIGHT_ALPHAS,
                                     window=CONF_WINDOW, ki=CONF_KI, cap=cap)
    results["A1_tirex_point_adaptive_pid"] = score(b_a1, y_test, m_te, masks)

    # ── verdict ──
    def wis(v, mk="overall_68"):
        return results[v][mk]["wis"]

    def picp(v, mk="overall_68"):
        return results[v][mk]["picp95"]

    main_wis = wis("V1_learned_fusion_pid")
    main_picp = picp("V1_learned_fusion_pid")
    tirex_alone = 2.951
    stack_b = 2.7205
    verdict = {
        "target_tirex_alone_wis": tirex_alone,
        "target_stack_B_wis_last34": stack_b,
        "main_variant": "V1_learned_fusion_pid",
        "main_overall_wis": main_wis,
        "main_overall_picp95": main_picp,
        "main_peak_y50_wis": wis("V1_learned_fusion_pid", "peak_y50"),
        "main_peak_y50_picp95": picp("V1_learned_fusion_pid", "peak_y50"),
        "main_last34_wis": wis("V1_learned_fusion_pid", "last34"),
        "beats_tirex_alone_overall_wis": bool(main_wis < tirex_alone),
        "beats_stack_B_on_last34": bool(wis("V1_learned_fusion_pid", "last34") < stack_b),
        "improves_picp_toward_95_vs_tirex": bool(
            abs(main_picp - 0.95) < abs(0.882 - 0.95)),
        "picp_improved_without_inflating_wis": bool(
            main_picp >= picp("A0_tirex_point_online_conformal") and main_wis <= tirex_alone),
        "fusion_beats_tirex_native_only": bool(main_wis < wis("V2_tirex_native_only_pid")),
        "fusion_beats_negbin_only": bool(main_wis < wis("V3_negbin_only_pid")),
        "tirex_point_maxdiff_vs_frozen": tirex_point_maxdiff,
    }

    out = {
        "method": "direct per-quantile convex fusion (TiRex native deciles ⊕ count-aware NegBin) + adaptive Conformal-PID",
        "selected_phi": phi,
        "selected_train_wis": best["train_wis"],
        "weights_per_flusight_quantile": {str(round(float(q), 4)): round(float(wj), 3)
                                          for q, wj in zip(FQ, w)},
        "protocol": {
            "split": meta,
            "tirex_max_context": MAX_CONTEXT,
            "conformal": f"adaptive Conformal-PID (_pid_adjust) window={CONF_WINDOW} ki={CONF_KI}, "
                         f"seeded from last {SEED_TAIL} train weeks; leak-free rolling obs",
            "tuning": "w(tau) per FluSight quantile by pinball on train rolling 1-step; "
                      "phi by min train WIS; both frozen for test. No test peeking.",
            "wis": "simulation.analytics.adaptive_conformal.wis_from_bounds over FLUSIGHT_ALPHAS",
            "peak_y50_definition": f"y_test >= {PEAK_Y}",
        },
        "results": results,
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── console table ──
    print("\n================= DIRECT QUANTILE FUSION =================")
    print(f"selected phi={phi}  train_WIS={best['train_wis']:.4f}  "
          f"TiRex point maxdiff vs frozen={tirex_point_maxdiff:.6f}\n")
    hdr = f"{'variant':40s} {'WIS_all':>8s} {'PICP95':>7s} {'WIS_peak':>9s} {'PICP_pk':>8s} {'WIS_l34':>8s}"
    print(hdr)
    print("-" * len(hdr))
    order = ["A0_tirex_point_online_conformal", "A1_tirex_point_adaptive_pid",
             "V2_tirex_native_only_pid", "V3_negbin_only_pid",
             "V4_learned_fusion_raw_noconformal", "V1_learned_fusion_pid",
             "V1b_learned_fusion_pid_tirexmedian"]
    for v in order:
        r = results[v]
        print(f"{v:40s} {r['overall_68']['wis']:8.4f} {r['overall_68']['picp95']:7.3f} "
              f"{r['peak_y50']['wis']:9.4f} {r['peak_y50']['picp95']:8.3f} {r['last34']['wis']:8.4f}")
    print("\nTARGETS: TiRex-alone overall WIS 2.951 / PICP95 0.882 ; Protocol-B stack last34 WIS 2.7205")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
