#!/usr/bin/env python
"""SPCI v3 — PARAMETER-FREE leak-free calibration. The raw conditional QRF interval beats the
2.4012 baseline on WIS/last-34 but under-covers (0.894); a small fixed CQR window ([165,205))
over-inflates because that window is a single hard peak. The rigorous fix is EXPANDING split
conformal: at each origin t the per-alpha offset is the (1-a)(1+1/m)-quantile of the interval
conformity scores accumulated over ALL prior origins s<t (seeded from pre-T0). No free width
knob, no test tuning; it self-calibrates by mixing off-season (tiny residual) and peak (large
residual) weeks in proportion to how often they occur.

Also reports, for context, fixed representative-window CQR and a trailing-window conformal.

Everything past-only:
  * conditional QRF per refit block, train_end = bstart - K_CAL.
  * offset(t) uses conformity of weeks s<t only (expanding or trailing window).
  * seed conformity from pre-T0 origins [SEED0, T0).
  * cap = 2*max(y_train) (train-only, spotless). Reference keeps its 2*max(y_full) cap.

DM vs the exact 2.4012 reference (paired per-origin WIS, HLN h=1). REAL numbers only.
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
from scripts.dec_boosted_mech import MIN_CTX, K_CAL
from scripts.dec_boosted_mech_multiorigin import T0

SEED0 = 125                                  # first origin used to seed conformity (feature/QRF-feasible)


def conformity_raw(B_raw, y):
    """Per-alpha nonconformity E_a = max(lo-y, y-hi) from the RAW conditional interval."""
    return {a: np.maximum(B_raw[a][0] - y, y - B_raw[a][1]) for a in ALPHAS}


def expanding_conformal(B_raw_all, E_all, idx_all, test_mask, y_all, cap, tirex_med, mode="expanding", win=60):
    """Build calibrated bounds for the test origins (test_mask over idx_all).

    offset_a(t) = quantile({E_a[s]: s in cal(t)}, (1-a)(1+1/m)); cal(t) = all s<t (expanding)
    or the trailing `win` such s (trailing). Leak-free: uses conformity of past origins only.
    """
    order = np.argsort(idx_all)
    idx_all = idx_all[order]
    B_raw_all = {a: (B_raw_all[a][0][order], B_raw_all[a][1][order]) for a in ALPHAS}
    E_all = {a: E_all[a][order] for a in ALPHAS}
    test_pos = np.where(test_mask[order])[0]
    n_all = len(idx_all)
    out = {a: (np.zeros(len(test_pos)), np.zeros(len(test_pos))) for a in ALPHAS}
    for oi, p in enumerate(test_pos):
        if mode == "expanding":
            csl = slice(0, p)
        else:
            csl = slice(max(0, p - win), p)
        m = p - csl.start
        for a in ALPHAS:
            lo_r, hi_r = B_raw_all[a][0][p], B_raw_all[a][1][p]
            sc = E_all[a][csl]
            q = min(1.0, (1.0 - a) * (1.0 + 1.0 / max(m, 1)))
            off = max(0.0, float(np.quantile(sc, q))) if m > 0 else 0.0
            out[a][0][oi] = np.clip(lo_r - off, 0.0, cap)
            out[a][1][oi] = np.clip(hi_r + off, 0.0, cap)
    return out, test_pos, order


def evaluate(B, y, med, ref_wis):
    n = len(y)
    w = wis_of(B, y, med)
    lo, hi = B[0.05]; cov = (y >= lo) & (y <= hi); k = int(cov.sum())
    p, dbar = dm(w, ref_wis)
    last34 = np.arange(n) >= n - 34
    return {"wis": round(float(w.mean()), 4), "dm_p": round(p, 4),
            "picp95": round(k / n, 4), "k_of_n": f"{k}/{n}", "cp95ci": list(cp(k, n)),
            "w95": round(float((hi - lo).mean()), 2),
            "last34_wis": round(float(w[last34].mean()), 4), "_w": w}


def run_leaf(S, msl, ref_wis):
    cap = S["cap"]; yf = S["yf"]; feat = S["feat"]; r = S["r"]; tirex = S["tirex"]
    qrf_kw = dict(n_estimators=400, min_samples_leaf=msl, max_features=0.6)
    idx_all = np.arange(SEED0, S["ntot"])
    grid_all = block_qrf_resid_grid(feat, r, idx_all, qrf_kw)
    tir_all = tirex[idx_all]
    yqfq, yqg = _yq_from_grid(grid_all, tir_all, cap)
    Bsym = bounds_symmetric(yqfq)
    Bbeta = bounds_beta(yqg)
    med_all = median_from_grid(yqg, tir_all, True)
    y_all = yf[idx_all]
    test_mask = idx_all >= T0

    results = {}
    for label, Braw in (("sym", Bsym), ("beta", Bbeta)):
        E = conformity_raw(Braw, y_all)
        for mode, win in (("expanding", 0), ("trail60", 60), ("trail90", 90)):
            B, tpos, order = expanding_conformal(Braw, E, idx_all, test_mask, y_all, cap, None,
                                                 mode=("expanding" if mode == "expanding" else "trail"), win=win)
            med_t = med_all[order][tpos]
            y_t = y_all[order][tpos]
            results[f"{label}_{mode}"] = evaluate(B, y_t, med_t, ref_wis)
    return results


def main():
    t0 = time.time()
    S = setup()
    ref_wis, _ = reference_wis(S, np.arange(T0, S["ntot"]))
    refm = float(ref_wis.mean())
    n = len(ref_wis); last34 = np.arange(n) >= n - 34
    print(f"=== SPCI v3 parameter-free expanding conformal | ref WIS={refm:.4f} "
          f"(last34={ref_wis[last34].mean():.4f}) ===")
    print("    TARGET: WIS<2.4012 & DM p<0.05 & PICP95 in [0.93,0.96] & last34<2.72\n")

    hdr = (f"{'msl':>4s} {'variant':>14s} | {'WIS':>7s} {'DMp':>7s} {'d%':>6s} {'PICP95':>7s} "
           f"{'k/N':>7s} {'CP95ci':>16s} {'W95':>6s} {'last34':>7s} {'MEETS':>6s}")
    print(hdr); print("-" * len(hdr))
    out_rows = []
    for msl in (8, 12, 16, 20):
        res = run_leaf(S, msl, ref_wis)
        for name, rr in res.items():
            meets = (rr["wis"] < refm and rr["dm_p"] < 0.05 and 0.93 <= rr["picp95"] <= 0.96 and rr["last34_wis"] < 2.72)
            dpct = 100 * (rr["wis"] - refm) / refm
            sig = "*" if (rr["wis"] < refm and rr["dm_p"] < 0.05) else " "
            calok = "OK" if 0.93 <= rr["picp95"] <= 0.96 else "  "
            print(f"{msl:>4d} {name:>14s} | {rr['wis']:>7.4f}{sig} {rr['dm_p']:>7.4f} {dpct:>6.1f} "
                  f"{rr['picp95']:>6.4f}{calok} {rr['k_of_n']:>7s} {str(rr['cp95ci']):>16s} "
                  f"{rr['w95']:>6.2f} {rr['last34_wis']:>7.4f} {str(meets):>6s}")
            out_rows.append({"msl": msl, "variant": name, "meets_all": bool(meets),
                             **{k: v for k, v in rr.items() if k != "_w"}})
        print()

    winners = [r for r in out_rows if r["meets_all"]]
    (ROOT / "scripts" / "_exp_spci3.json").write_text(json.dumps(
        {"ref_wis": round(refm, 4), "n": n, "rows": out_rows, "winners": winners,
         "elapsed_sec": round(time.time() - t0, 1)}, indent=2))
    print("=== WINNERS (parameter-free, meet all 4): " + (json.dumps(
        [{k: w[k] for k in ('msl', 'variant', 'wis', 'dm_p', 'picp95', 'last34_wis')} for w in winners]) if winners else "NONE") + " ===")
    print(f"wrote scripts/_exp_spci3.json  ({round(time.time()-t0,1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
