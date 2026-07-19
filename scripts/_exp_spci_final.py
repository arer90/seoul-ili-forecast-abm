#!/usr/bin/env python
"""SPCI FINAL — the headline leak-free forecaster + rigorous leak-free hyperparameter selection.

METHOD (fixed a priori = SPCI, Xu & Xie 2023):
  1. TiRex 1-step point base.
  2. CONDITIONAL residual quantile Q(tau|x_t) via a Quantile Random Forest (Meinshausen 2006),
     refit every 5 origins on strictly past pairs (train_end = bstart - K_CAL).
     Features: r_{t-1..t-10}, rolling |resid| mean 4/8/13, rolling signed resid 4/8, TiRex level,
     epiweek fourier h1.  Target r_t = y_t - TiRex_t.
  3. SPCI width-optimal beta-search: shortest 1-alpha interval [Q(beta*), Q(1-alpha+beta*)].
  4. PARAMETER-FREE expanding split-conformal: offset_a(t) = (1-a)(1+1/m)-quantile of interval
     conformity over ALL prior origins s<t (seeded pre-T0). No width knob, no test tuning.

The ONE hyperparameter (QRF min_samples_leaf) is chosen by LEAK-FREE pre-T0 validation: argmin
mean WIS over validation origins [155,205) under the identical past-only protocol; the test
origins [205,336) are never consulted in selection. Cap = 2*max(y_train) (train-only, spotless).

Reports the selected config's REAL test metrics vs the exact 2.4012 fair baseline (paired
per-origin WIS DM, HLN h=1), a robustness table across leaf sizes, and a SEED0 sensitivity check.
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
from scripts.dec_boosted_mech_multiorigin import T0

VAL_LO, VAL_HI = 155, 205                     # pre-T0 validation origins (leak-free selection)


def conformity(Braw, y):
    return {a: np.maximum(Braw[a][0] - y, y - Braw[a][1]) for a in ALPHAS}


def calibrate_expanding(Braw, E, idx_all, eval_lo, cap, seed0):
    """Expanding split-conformal bounds for every origin >= eval_lo. offset_a(t)=quantile of
    conformity over origins [seed0, t). Returns bounds dict aligned to eval origins (sorted)."""
    order = np.argsort(idx_all); idx = idx_all[order]
    Braw = {a: (Braw[a][0][order], Braw[a][1][order]) for a in ALPHAS}
    E = {a: E[a][order] for a in ALPHAS}
    seed_pos = int(np.searchsorted(idx, seed0))
    eval_pos = np.where(idx >= eval_lo)[0]
    out = {a: (np.zeros(len(eval_pos)), np.zeros(len(eval_pos))) for a in ALPHAS}
    for oi, p in enumerate(eval_pos):
        m = p - seed_pos
        for a in ALPHAS:
            sc = E[a][seed_pos:p]
            q = min(1.0, (1.0 - a) * (1.0 + 1.0 / max(m, 1)))
            off = max(0.0, float(np.quantile(sc, q))) if m > 0 else 0.0
            out[a][0][oi] = np.clip(Braw[a][0][p] - off, 0.0, cap)
            out[a][1][oi] = np.clip(Braw[a][1][p] + off, 0.0, cap)
    return out, idx[eval_pos], order, eval_pos


def metrics(B, y, med, ref_wis=None):
    n = len(y); w = wis_of(B, y, med)
    lo, hi = B[0.05]; cov = (y >= lo) & (y <= hi); k = int(cov.sum())
    last34 = np.arange(n) >= n - 34
    d = {"wis": round(float(w.mean()), 4), "picp95": round(k / n, 4), "k_of_n": f"{k}/{n}",
         "cp95ci": list(cp(k, n)), "w95": round(float((hi - lo).mean()), 2),
         "last34_wis": round(float(w[last34].mean()), 4), "_w": w}
    if ref_wis is not None:
        p, dbar = dm(w, ref_wis); d["dm_p"] = round(p, 4); d["dm_mean_diff"] = round(dbar, 4)
    return d


def build(S, msl, seed0, engine="beta"):
    """Return dict of per-origin calibrated bounds/medians for eval origins >= VAL_LO."""
    cap, yf, feat, r, tirex = S["cap"], S["yf"], S["feat"], S["r"], S["tirex"]
    qrf_kw = dict(n_estimators=400, min_samples_leaf=msl, max_features=0.6)
    idx_all = np.arange(seed0, S["ntot"])
    grid = block_qrf_resid_grid(feat, r, idx_all, qrf_kw)
    tir = tirex[idx_all]
    yqfq, yqg = _yq_from_grid(grid, tir, cap)
    Braw = bounds_beta(yqg) if engine == "beta" else bounds_symmetric(yqfq)
    med_all = median_from_grid(yqg, tir, True)
    y_all = yf[idx_all]
    E = conformity(Braw, y_all)
    B, ev_idx, order, ev_pos = calibrate_expanding(Braw, E, idx_all, VAL_LO, cap, seed0)
    med_ev = med_all[order][ev_pos]; y_ev = y_all[order][ev_pos]
    return dict(B=B, idx=ev_idx, med=med_ev, y=y_ev)


def slice_bounds(B, mask):
    return {a: (B[a][0][mask], B[a][1][mask]) for a in ALPHAS}


def main():
    t0 = time.time()
    S = setup()
    test_origins = np.arange(T0, S["ntot"])
    ref_wis, _ = reference_wis(S, test_origins)
    refm = float(ref_wis.mean()); n = len(ref_wis)
    ref_l34 = float(ref_wis[np.arange(n) >= n - 34].mean())
    print(f"=== SPCI FINAL | fair baseline WIS={refm:.4f}  PICP95=0.9545  last34={ref_l34:.4f} ===")
    print("    TARGET: WIS<2.4012 & DM p<0.05 & PICP95∈[0.93,0.96] & last34<2.72\n")

    LEAVES = [8, 10, 12, 14, 16, 20, 24]
    SEED0 = 125
    print("--- leak-free pre-T0 validation selection (argmin WIS over origins [155,205)) + test readout ---")
    hdr = (f"{'msl':>4s} | {'valWIS':>7s} || {'tstWIS':>7s} {'DMp':>7s} {'PICP95':>7s} {'k/N':>7s} "
           f"{'CP95ci':>16s} {'W95':>6s} {'last34':>7s} {'MEETS':>6s}")
    print(hdr); print("-" * len(hdr))
    table = []
    for msl in LEAVES:
        r = build(S, msl, SEED0, engine="beta")
        idx = r["idx"]
        val_m = idx < T0; tst_m = idx >= T0
        val_wis = float(wis_of(slice_bounds(r["B"], val_m), r["y"][val_m], r["med"][val_m]).mean())
        tst = metrics(slice_bounds(r["B"], tst_m), r["y"][tst_m], r["med"][tst_m], ref_wis)
        meets = (tst["wis"] < refm and tst["dm_p"] < 0.05 and 0.93 <= tst["picp95"] <= 0.96 and tst["last34_wis"] < 2.72)
        tst["meets_all"] = bool(meets); tst["msl"] = msl; tst["val_wis"] = round(val_wis, 4)
        table.append(tst)
        print(f"{msl:>4d} | {val_wis:>7.4f} || {tst['wis']:>7.4f} {tst['dm_p']:>7.4f} {tst['picp95']:>7.4f} "
              f"{tst['k_of_n']:>7s} {str(tst['cp95ci']):>16s} {tst['w95']:>6.2f} {tst['last34_wis']:>7.4f} {str(meets):>6s}")

    sel = min(table, key=lambda d: d["val_wis"])
    print(f"\n>>> leak-free selected msl={sel['msl']} (min val WIS={sel['val_wis']})")
    print(f"    TEST: WIS={sel['wis']} (DM p={sel['dm_p']}, mean diff={sel['dm_mean_diff']})  "
          f"PICP95={sel['picp95']} {sel['k_of_n']} CP95ci={sel['cp95ci']}  W95={sel['w95']}  "
          f"last34={sel['last34_wis']}  MEETS_ALL={sel['meets_all']}")

    # ---- SEED0 sensitivity for the selected msl ----
    print(f"\n--- SEED0 sensitivity (selected msl={sel['msl']}, beta_expanding) ---")
    seed_rows = []
    for seed0 in (110, 125, 140, 155):
        r = build(S, sel["msl"], seed0, engine="beta")
        tst_m = r["idx"] >= T0
        tst = metrics(slice_bounds(r["B"], tst_m), r["y"][tst_m], r["med"][tst_m], ref_wis)
        seed_rows.append({"seed0": seed0, **{k: v for k, v in tst.items() if k != "_w"}})
        print(f"  seed0={seed0:>3d} | WIS={tst['wis']:.4f} DMp={tst['dm_p']:.4f} PICP95={tst['picp95']:.4f} "
              f"{tst['k_of_n']} last34={tst['last34_wis']:.4f}")

    out = {
        "fair_baseline": {"wis": 2.4012, "picp95": 0.9545, "last34_wis": round(ref_l34, 4)},
        "seed0": SEED0, "leaves": LEAVES,
        "selection": {"rule": "argmin mean WIS over pre-T0 validation origins [155,205)",
                      "selected_msl": sel["msl"], "val_wis": sel["val_wis"]},
        "selected_test": {k: v for k, v in sel.items() if k != "_w"},
        "leaf_table": [{k: v for k, v in d.items() if k != "_w"} for d in table],
        "seed0_sensitivity": seed_rows,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (ROOT / "scripts" / "_exp_spci_final.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/_exp_spci_final.json  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
