#!/usr/bin/env python
"""ADVERSARIAL AUDIT of nov_shift_conformal.py (independent, standalone).

Checks, all with the SAME frozen 68-week hold-out split + official frozen point:
  A. Independent WIS: recompute WIS from the (lo,hi) bounds with a from-scratch
     Bracher-2021 implementation and compare to wis_from_bounds (must match).
  B. Leak-free perturbation test: corrupt y[i..n-1] (the FUTURE at step i) and
     confirm interval[i] is byte-identical for the sqrt scale (interval[i] may
     depend on pred[i] and y[0..i-1] only, never y[i..]).
  C. Trivial-widening test: does the sqrt WIS win survive when we FORCE the
     additive baseline to the SAME mean 95% width (uniform inflation)? If a mere
     uniform widening of the additive baseline to the sqrt width already wins,
     the "win" is just width; if it does NOT, the win is genuine reallocation.
  D. WIS decomposition (dispersion / underprediction / overprediction) additive
     vs sqrt, overall and peak — to see WHERE the WIS change comes from.
No live code touched. Reuses the prototype's own functions + the deployed wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from simulation.analytics.adaptive_conformal import online_conformal_bounds, wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from fusedepi_fusion_wis import load_frozen
from nov_shift_conformal import compute_scale, shift_aware_conformal_bounds

ALPHAS = FLUSIGHT_ALPHAS


def wis_independent(y, bounds, alphas, median):
    """From-scratch WIS (Bracher 2021): 1/(K+.5)[.5|y-m| + Σ (α/2) IS_α]."""
    y = np.asarray(y, float); m = np.asarray(median, float)
    acc = 0.5 * np.abs(y - m)
    K = 0
    for a in alphas:
        if a not in bounds:
            continue
        lo, hi = bounds[a]
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        IS = (hi - lo) + (2.0 / a) * (lo - y) * (y < lo) + (2.0 / a) * (y - hi) * (y > hi)
        acc = acc + (a / 2.0) * IS
        K += 1
    return acc / (K + 0.5)


def wis_decomp(y, bounds, alphas, median):
    """Decompose mean WIS into dispersion / underpred / overpred components."""
    y = np.asarray(y, float); m = np.asarray(median, float)
    disp = np.zeros(len(y)); under = np.zeros(len(y)); over = np.zeros(len(y))
    med_term = 0.5 * np.abs(y - m)
    K = 0
    for a in alphas:
        if a not in bounds:
            continue
        lo, hi = bounds[a]
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        disp += (a / 2.0) * (hi - lo)
        under += (a / 2.0) * (2.0 / a) * (lo - y) * (y < lo)
        over += (a / 2.0) * (2.0 / a) * (y - hi) * (y > hi)
        K += 1
    denom = (K + 0.5)
    return {
        "median": float(np.mean(med_term) / denom),
        "dispersion": float(np.mean(disp) / denom),
        "underprediction": float(np.mean(under) / denom),
        "overprediction": float(np.mean(over) / denom),
    }


def main():
    for model in ("TiRex", "FusedEpi"):
        pred, y, _ = load_frozen(model)
        pred = np.asarray(pred, float).ravel(); y = np.asarray(y, float).ravel()
        n = len(y)
        peak = y >= float(np.quantile(y, 0.75))

        base = online_conformal_bounds(pred, y, ALPHAS, window=30, ki=0.2)
        sqrt_scale = compute_scale(pred, y, "sqrt", floor=1.0)
        sqrt = shift_aware_conformal_bounds(pred, y, ALPHAS, sqrt_scale, window=30, ki=0.2)

        print("=" * 78); print(model, f"(n={n}, n_peak={int(peak.sum())})")

        # --- A. independent WIS ---
        for name, bnd in (("additive", base), ("sqrt", sqrt)):
            w_lib = np.asarray(wis_from_bounds(y, bnd, ALPHAS, median=pred), float)
            w_ind = wis_independent(y, bnd, ALPHAS, pred)
            print(f"  A. {name:8s} WIS lib={np.mean(w_lib):.6f} indep={np.mean(w_ind):.6f} "
                  f"max|Δ|={np.max(np.abs(w_lib-w_ind)):.2e}")

        # --- B. leak-free perturbation on the sqrt scale + full bounds ---
        # corrupt the future (y[i:]) at a mid index and re-derive interval[:i]
        i0 = 40
        y_corrupt = y.copy()
        rng = np.random.default_rng(0)
        y_corrupt[i0:] = y[i0:] + rng.normal(0, 500, size=n - i0)  # violent future corruption
        sc_c = compute_scale(pred, y_corrupt, "sqrt", floor=1.0)
        sqrt_c = shift_aware_conformal_bounds(pred, y_corrupt, ALPHAS, sqrt_scale=None) \
            if False else shift_aware_conformal_bounds(pred, y_corrupt, ALPHAS, sc_c, window=30, ki=0.2)
        # scale must be identical up to i0 (sqrt uses only pred), and every interval
        # bound at index < i0 must be identical (uses only pred[.] and y[0..-1])
        scale_gap = float(np.max(np.abs(sqrt_scale[:i0] - sc_c[:i0])))
        lo0, hi0 = sqrt[0.05]; lo0c, hi0c = sqrt_c[0.05]
        bound_gap = float(max(np.max(np.abs(lo0[:i0] - lo0c[:i0])),
                              np.max(np.abs(hi0[:i0] - hi0c[:i0]))))
        # sanity: futures DID change something at/after i0 (else corruption was a no-op)
        after_gap = float(np.max(np.abs(hi0[i0:] - hi0c[i0:])))
        print(f"  B. leak-free: future-corruption max|Δscale[:{i0}]|={scale_gap:.2e} "
              f"max|Δbound95[:{i0}]|={bound_gap:.2e} (post-{i0} Δ={after_gap:.2f} => corruption active)")

        # --- C. trivial-widening control: inflate additive to the sqrt mean-width ---
        lo_a, hi_a = base[0.05]
        wa = np.mean(hi_a - lo_a); ws = np.mean(hi0 - lo0)
        # uniform multiplicative widening of EVERY additive level to match total width ratio
        ratio = ws / wa
        base_wide = {}
        for a in ALPHAS:
            la, ha = base[a]
            mid = 0.5 * (la + ha)
            half = 0.5 * (ha - la) * ratio
            base_wide[a] = (np.maximum(0.0, mid - half), mid + half)
        w_base = float(np.mean(wis_from_bounds(y, base, ALPHAS, median=pred)))
        w_bwide = float(np.mean(wis_from_bounds(y, base_wide, ALPHAS, median=pred)))
        w_sqrt = float(np.mean(wis_from_bounds(y, sqrt, ALPHAS, median=pred)))
        cov = lambda b: float(np.mean((y >= b[0.05][0]) & (y <= b[0.05][1])))
        print(f"  C. trivial-widening control (match 95% width ratio={ratio:.3f}):")
        print(f"       additive       WIS={w_base:.4f} w95={wa:.2f} PICP95={cov(base):.4f}")
        print(f"       additive*ratio WIS={w_bwide:.4f} w95={np.mean(base_wide[0.05][1]-base_wide[0.05][0]):.2f} PICP95={cov(base_wide):.4f}")
        print(f"       sqrt(regime)   WIS={w_sqrt:.4f} w95={ws:.2f} PICP95={cov(sqrt):.4f}")

        # --- D. WIS decomposition ---
        for name, bnd, msk in (("additive/all", base, np.ones(n, bool)),
                               ("sqrt/all", sqrt, np.ones(n, bool)),
                               ("additive/peak", base, peak),
                               ("sqrt/peak", sqrt, peak)):
            bm = {a: (bnd[a][0][msk], bnd[a][1][msk]) for a in ALPHAS}
            dd = wis_decomp(y[msk], bm, ALPHAS, pred[msk])
            tot = sum(dd.values())
            print(f"  D. {name:14s} WIS={tot:.4f}  disp={dd['dispersion']:.3f} "
                  f"under={dd['underprediction']:.3f} over={dd['overprediction']:.3f} med={dd['median']:.3f}")


if __name__ == "__main__":
    main()
