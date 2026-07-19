#!/usr/bin/env python
"""SPCI BAGGED — remove the single hyperparameter. beta_expanding meets all four constraints
for EVERY QRF leaf size in {12,14,16,20,24} (DM p in [0.0045,0.0494]); only the msl choice is
fragile. Bagging the conditional residual-quantile grids across an a-priori leaf-size set
{12,16,20,24} yields a parameter-free forecaster (no leaf to select, matching the codebase's
existing 6-config GBM bagging). Reported: bagged SPCI-beta + parameter-free expanding conformal
vs the exact 2.4012 baseline; SEED0 and leaf-set sensitivity; a symmetric-interval cross-check.

Leak-free: per refit block train_end = bstart-K_CAL; expanding conformal offset(t) over origins
s<t only (seeded pre-T0); cap = 2*max(y_train). Paired per-origin DM (HLN h=1). REAL numbers.
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
    os.environ.setdefault(_v, "3")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._exp_spci import (
    ALPHAS, setup, reference_wis, block_qrf_resid_grid, _yq_from_grid,
    bounds_symmetric, bounds_beta, median_from_grid, dm, cp, wis_of,
)
from scripts._exp_spci_final import conformity, calibrate_expanding, metrics, slice_bounds, VAL_LO
from scripts.dec_boosted_mech_multiorigin import T0

IDX0 = 110                                    # widest grid start (reused across seed0)


def bagged_grid(S, msl_set):
    """Average conditional resid-quantile grids over an a-priori leaf-size set (parameter-free)."""
    feat, r = S["feat"], S["r"]
    idx_all = np.arange(IDX0, S["ntot"])
    acc = None
    for msl in msl_set:
        kw = dict(n_estimators=400, min_samples_leaf=msl, max_features=0.6)
        g = block_qrf_resid_grid(feat, r, idx_all, kw)
        acc = g if acc is None else acc + g
    return idx_all, acc / len(msl_set)


def evaluate_bag(S, idx_all, grid, seed0, engine, ref_wis):
    cap, yf, tirex = S["cap"], S["yf"], S["tirex"]
    tir = tirex[idx_all]
    yqfq, yqg = _yq_from_grid(grid, tir, cap)
    Braw = bounds_beta(yqg) if engine == "beta" else bounds_symmetric(yqfq)
    med_all = median_from_grid(yqg, tir, True)
    y_all = yf[idx_all]
    E = conformity(Braw, y_all)
    B, ev_idx, order, ev_pos = calibrate_expanding(Braw, E, idx_all, VAL_LO, cap, seed0)
    med_ev = med_all[order][ev_pos]; y_ev = y_all[order][ev_pos]
    tst_m = ev_idx >= T0
    return metrics(slice_bounds(B, tst_m), y_ev[tst_m], med_ev[tst_m], ref_wis)


def show(tag, m, refm):
    meets = (m["wis"] < refm and m["dm_p"] < 0.05 and 0.93 <= m["picp95"] <= 0.96 and m["last34_wis"] < 2.72)
    print(f"{tag:>26s} | WIS={m['wis']:.4f} DMp={m['dm_p']:.4f} d%={100*(m['wis']-refm)/refm:+.1f} "
          f"PICP95={m['picp95']:.4f} {m['k_of_n']:>7s} CP={m['cp95ci']} W95={m['w95']:.2f} "
          f"l34={m['last34_wis']:.4f}  MEETS={meets}")
    return meets


def main():
    t0 = time.time()
    S = setup()
    ref_wis, _ = reference_wis(S, np.arange(T0, S["ntot"]))
    refm = float(ref_wis.mean()); n = len(ref_wis)
    ref_l34 = float(ref_wis[np.arange(n) >= n - 34].mean())
    print(f"=== SPCI BAGGED | fair baseline WIS={refm:.4f} PICP95=0.9545 last34={ref_l34:.4f} ===")
    print("    TARGET: WIS<2.4012 & DM p<0.05 & PICP95∈[0.93,0.96] & last34<2.72\n")

    MAIN_SET = (12, 16, 20, 24)
    idx_all, grid = bagged_grid(S, MAIN_SET)

    print(f"--- headline: bagged QRF leaf-set {MAIN_SET}, beta+expanding conformal, seed0=125 ---")
    head = evaluate_bag(S, idx_all, grid, 125, "beta", ref_wis)
    head_meets = show("bag_beta_expanding", head, refm)

    print("\n--- SEED0 sensitivity (headline bag, beta) ---")
    seed_rows = []
    for seed0 in (110, 125, 140, 155):
        m = evaluate_bag(S, idx_all, grid, seed0, "beta", ref_wis)
        seed_rows.append({"seed0": seed0, **{k: v for k, v in m.items() if k != "_w"}})
        show(f"seed0={seed0}", m, refm)

    print("\n--- leaf-set sensitivity (beta, seed0=125) ---")
    set_rows = []
    for ms in [(10, 14, 18, 22), (12, 16, 20, 24), (14, 18, 22, 26), (12, 18, 24), (10, 15, 20, 25, 30)]:
        ia, g = bagged_grid(S, ms)
        m = evaluate_bag(S, ia, g, 125, "beta", ref_wis)
        set_rows.append({"leaf_set": list(ms), **{k: v for k, v in m.items() if k != "_w"}})
        show(f"set={ms}", m, refm)

    print("\n--- engine cross-check (seed0=125, main set) ---")
    sym = evaluate_bag(S, idx_all, grid, 125, "sym", ref_wis)
    show("bag_sym_expanding", sym, refm)

    out = {
        "fair_baseline": {"wis": 2.4012, "picp95": 0.9545, "last34_wis": round(ref_l34, 4)},
        "headline": {"config": f"bagged QRF leaf-set{list(MAIN_SET)} beta_expanding seed0=125",
                     **{k: v for k, v in head.items() if k != "_w"}, "meets_all": bool(head_meets)},
        "seed0_sensitivity": seed_rows,
        "leaf_set_sensitivity": set_rows,
        "sym_crosscheck": {k: v for k, v in sym.items() if k != "_w"},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (ROOT / "scripts" / "_exp_spci_bag.json").write_text(json.dumps(out, indent=2))
    print(f"\nHEADLINE meets_all={head_meets}")
    print(f"wrote scripts/_exp_spci_bag.json  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
