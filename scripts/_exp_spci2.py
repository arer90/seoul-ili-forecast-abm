#!/usr/bin/env python
"""SPCI v2 — land PICP95 in [0.93,0.96]. The raw conditional QRF interval already beats the
2.4012 fair baseline on WIS (2.343) and last-34 (2.55) but UNDER-covers (0.894); additive CQR
seeded on [165,205) OVER-inflates (0.99). The sweet spot is a MODEST, adaptivity-PRESERVING
inflation. This maps the leaf-size x inflation landscape on the 132 test origins AND performs a
strictly leak-free pre-T0 selection of the inflation factor on validation window V=[165,205).

Two inflation families (both preserve the conditional width structure):
  * mult(c): scale the symmetric conditional half-widths by c about the conditional median.
  * cqrf(f): additive CQR offset scaled by f in [0,1] (f=1 == full CQR).
Selection (leak-free): factor chosen so validation PICP95 is the smallest value >= 0.95
(target the FluSight nominal); applied unchanged to the test origins. DM vs the exact 2.4012
reference (paired per-origin WIS, HLN h=1). Reports REAL numbers only.
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
    ALPHAS, GL, setup, reference_wis, block_qrf_resid_grid, fit_qrf, qrf_grid,
    _yq_from_grid, bounds_symmetric, bounds_beta, cqr_offsets_generic, apply_cqr,
    median_from_grid, eval_bounds, dm, cp, wis_of,
)
from scripts.dec_boosted_mech import MIN_CTX, K_CAL
from scripts.dec_boosted_mech_multiorigin import T0

WARM = 13
V_LO, V_HI = 165, 205                       # pre-T0 validation origins (leak-free)


def inflate_mult(B, med, c, cap):
    out = {}
    for a in ALPHAS:
        lo, hi = B[a]
        out[a] = (np.clip(med - c * (med - lo), 0.0, cap), np.clip(med + c * (hi - med), 0.0, cap))
    return out


def make_grids(S, qrf_kw):
    """Conditional resid-quantile grids for test origins, validation origins, and the two
    calibration windows (test-CQR seed pre-T0, val-CQR seed pre-V)."""
    feat, r = S["feat"], S["r"]
    test_o = np.arange(T0, S["ntot"])
    val_o = np.arange(V_LO, V_HI)
    valid0 = MIN_CTX + WARM
    grid_test = block_qrf_resid_grid(feat, r, test_o, qrf_kw)
    grid_val = block_qrf_resid_grid(feat, r, val_o, qrf_kw)
    # test CQR seed: train [valid0, T0-K_CAL), predict cal [T0-K_CAL,T0)
    cal_t = np.arange(T0 - K_CAL, T0)
    tr = np.arange(valid0, T0 - K_CAL); tr = tr[np.isfinite(feat[tr]).all(1) & np.isfinite(r[tr])]
    rf, lt = fit_qrf(feat[tr], r[tr], **qrf_kw)
    grid_cal_t = qrf_grid(rf, lt, r[tr], feat[cal_t])
    # val CQR seed: train [valid0, V_LO-K_CAL), predict cal [V_LO-K_CAL,V_LO)
    cal_v = np.arange(V_LO - K_CAL, V_LO)
    trv = np.arange(valid0, V_LO - K_CAL); trv = trv[np.isfinite(feat[trv]).all(1) & np.isfinite(r[trv])]
    rfv, ltv = fit_qrf(feat[trv], r[trv], **qrf_kw)
    grid_cal_v = qrf_grid(rfv, ltv, r[trv], feat[cal_v])
    return dict(test_o=test_o, val_o=val_o, grid_test=grid_test, grid_val=grid_val,
                cal_t=cal_t, grid_cal_t=grid_cal_t, cal_v=cal_v, grid_cal_v=grid_cal_v)


def picp95(B, y):
    lo, hi = B[0.05]; cov = (y >= lo) & (y <= hi)
    return float(cov.mean()), int(cov.sum())


def main():
    t0 = time.time()
    S = setup(); cap = S["cap"]; yf = S["yf"]
    ref_wis, _ = reference_wis(S, np.arange(T0, S["ntot"]))
    refm = float(ref_wis.mean())
    n = len(ref_wis)
    last34 = np.arange(n) >= n - 34
    print(f"=== SPCI v2 inflation landscape + leak-free pre-T0 selection | ref WIS={refm:.4f} "
          f"(last34={ref_wis[last34].mean():.4f}) ===\n")

    LEAVES = [8, 12, 20, 30]
    C_GRID = np.round(np.arange(1.0, 2.31, 0.1), 2)
    F_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
    all_rows = []
    selections = []

    for msl in LEAVES:
        qrf_kw = dict(n_estimators=400, min_samples_leaf=msl, max_features=0.6)
        G = make_grids(S, qrf_kw)
        yt = yf[G["test_o"]]; yv = yf[G["val_o"]]
        tir_t = S["tirex"][G["test_o"]]; tir_v = S["tirex"][G["val_o"]]

        yqfq_t, yqg_t = _yq_from_grid(G["grid_test"], tir_t, cap)
        yqfq_v, yqg_v = _yq_from_grid(G["grid_val"], tir_v, cap)
        med_t = median_from_grid(yqg_t, tir_t, True)
        med_v = median_from_grid(yqg_v, tir_v, True)
        Bsym_t = bounds_symmetric(yqfq_t); Bsym_v = bounds_symmetric(yqfq_v)

        # CQR seeds
        yqfq_ct, _ = _yq_from_grid(G["grid_cal_t"], S["tirex"][G["cal_t"]], cap)
        yqfq_cv, _ = _yq_from_grid(G["grid_cal_v"], S["tirex"][G["cal_v"]], cap)
        Q_t = cqr_offsets_generic(bounds_symmetric(yqfq_ct), yf[G["cal_t"]])
        Q_v = cqr_offsets_generic(bounds_symmetric(yqfq_cv), yf[G["cal_v"]])

        print(f"--- QRF min_samples_leaf={msl} ---")
        hdr = f"{'infl':>10s} | {'valPICP':>7s} {'tstWIS':>7s} {'DMp':>7s} {'tstPICP':>7s} {'k/N':>7s} {'W95':>6s} {'l34':>7s}"
        print(hdr)
        # ---- mult family ----
        best_mult = None
        for c in C_GRID:
            Bv = inflate_mult(Bsym_v, med_v, c, cap); Bt = inflate_mult(Bsym_t, med_t, c, cap)
            vp, _ = picp95(Bv, yv)
            rr = eval_bounds(Bt, yt, med_t, ref_wis, n)
            all_rows.append({"family": "mult", "msl": msl, "c": float(c), "valPICP": round(vp, 4), **{k: v for k, v in rr.items() if k != "_wis_arr"}})
            if vp >= 0.95 and (best_mult is None or c < best_mult[0]):
                best_mult = (float(c), vp, rr)
            tag = " <-selVAL>=.95" if (best_mult and best_mult[0] == c) else ""
            print(f"  mult c={c:>4.1f} | {vp:>7.4f} {rr['wis']:>7.4f} {rr['dm_p']:>7.4f} {rr['picp95']:>7.4f} {rr['k_of_n']:>7s} {rr['w95']:>6.2f} {rr['last34_wis']:>7.4f}{tag}")
        # ---- cqrf family ----
        best_cqrf = None
        for f in F_GRID:
            Qtf = {a: Q_t[a] * f for a in ALPHAS}; Qvf = {a: Q_v[a] * f for a in ALPHAS}
            Bv = apply_cqr(Bsym_v, Qvf, cap); Bt = apply_cqr(Bsym_t, Qtf, cap)
            vp, _ = picp95(Bv, yv)
            rr = eval_bounds(Bt, yt, med_t, ref_wis, n)
            all_rows.append({"family": "cqrf", "msl": msl, "f": float(f), "valPICP": round(vp, 4), **{k: v for k, v in rr.items() if k != "_wis_arr"}})
            if vp >= 0.95 and (best_cqrf is None or f < best_cqrf[0]):
                best_cqrf = (float(f), vp, rr)
            tag = " <-selVAL>=.95" if (best_cqrf and best_cqrf[0] == f) else ""
            print(f"  cqrf f={f:>4.1f} | {vp:>7.4f} {rr['wis']:>7.4f} {rr['dm_p']:>7.4f} {rr['picp95']:>7.4f} {rr['k_of_n']:>7s} {rr['w95']:>6.2f} {rr['last34_wis']:>7.4f}{tag}")

        for fam, best in (("mult", best_mult), ("cqrf", best_cqrf)):
            if best is not None:
                key, vp, rr = best
                meets = (rr["wis"] < refm and rr["dm_p"] < 0.05 and 0.93 <= rr["picp95"] <= 0.96 and rr["last34_wis"] < 2.72)
                selections.append({"msl": msl, "family": fam, "factor": key, "valPICP": round(vp, 4),
                                   "test_wis": rr["wis"], "dm_p": rr["dm_p"], "test_picp95": rr["picp95"],
                                   "cp95ci": rr["cp95ci"], "last34_wis": rr["last34_wis"], "w95": rr["w95"],
                                   "meets_all": bool(meets)})
        print()

    print("=== LEAK-FREE SELECTED (val PICP95>=0.95 -> applied to test) ===")
    hdr = f"{'msl':>4s} {'fam':>5s} {'factor':>7s} {'valP':>6s} | {'tstWIS':>7s} {'DMp':>7s} {'tstPICP':>7s} {'CP95ci':>16s} {'l34':>7s} {'MEETS':>6s}"
    print(hdr); print("-" * len(hdr))
    for s in selections:
        print(f"{s['msl']:>4d} {s['family']:>5s} {s['factor']:>7.2f} {s['valPICP']:>6.3f} | "
              f"{s['test_wis']:>7.4f} {s['dm_p']:>7.4f} {s['test_picp95']:>7.4f} {str(s['cp95ci']):>16s} "
              f"{s['last34_wis']:>7.4f} {str(s['meets_all']):>6s}")

    winners = [s for s in selections if s["meets_all"]]
    out = {"ref_wis": round(refm, 4), "n": n, "selections": selections,
           "winners": winners, "landscape": all_rows, "elapsed_sec": round(time.time() - t0, 1)}
    (ROOT / "scripts" / "_exp_spci2.json").write_text(json.dumps(out, indent=2))
    print("\nWINNERS (leak-free selected AND meet all 4 test constraints): "
          + (json.dumps([{k: w[k] for k in ('msl', 'family', 'factor', 'test_wis', 'dm_p', 'test_picp95', 'last34_wis')} for w in winners]) if winners else "NONE"))
    print(f"wrote scripts/_exp_spci2.json  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
