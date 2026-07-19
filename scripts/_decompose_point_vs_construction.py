#!/usr/bin/env python
"""DECOMPOSE the 2.71(TiRex+Tweedie) vs 2.80(FusedEpi+Tweedie) gap into POINT-effect vs
CONSTRUCTION-effect, leak-free. From ONE FusedEpi fit per window we extract:
   base  = the internal TiRex zero-shot point  (fit's tx_tr / _tirex_roll)
   fused = base + alpha*corr                    (the FusedEpi fused point)
Then apply the IDENTICAL campaign Tweedie construction (X.tweedie_qy + X.expanding_cqr_bounds,
p=1.5 fixed a-priori) to BOTH point series at the window's origins.

  V1 = TiRex base  + campaign-Tweedie   (should reproduce campaign 2.713/68wk, 2.2427/132)
  V2 = Fused point + campaign-Tweedie   (same construction, only the point swapped)
  -> V1 vs V2 = PURE POINT effect (does the fusion mean-correction hurt Tweedie?)
  -> V2 vs V3(=2.80 fused_epi.py live branch, from _fused_negbin_vs_tweedie) = CONSTRUCTION effect

Leak audit: base=zero-shot rolling (context y[:t]); corr fit on pool only (in-sample on pool,
OOS on test — the pool in-sample tightens fused intervals, i.e. biases fused to look BETTER, so a
fused-WORSE result is conservative). Perturbation test on V2 included.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts._exp_crosscountry as X                       # tweedie_qy, expanding_cqr_bounds, wis_of, dm
from simulation.models.fused_epi import FusedEpiForecaster
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.pipeline.data import run_data
from simulation.pipeline.config import PipelineConfig
from scripts.ablation_fusedepi import _resolve_eval_features

MIN_CTX = X.MIN_CTX; K_CAL = X.K_CAL; P_FIX = 1.5
A95 = min(FLUSIGHT_ALPHAS, key=lambda a: abs(a - 0.05))     # 0.05 -> 95% interval
PEAK_Q = 0.90


def points_for(m, y_all, X_eval, N):
    """base (TiRex zero-shot) and fused (base+alpha*corr) over [MIN_CTX, N), from a fitted m."""
    idxs = list(range(MIN_CTX, N))
    base = np.full(N, np.nan)
    base[MIN_CTX:] = np.asarray(m._tirex_roll(y_all, idxs), float)      # zero-shot, leak-free per origin
    Xf = m._corr_features(X_eval[MIN_CTX:N], y_all, N - MIN_CTX, y_all[MIN_CTX:N])
    corr = np.asarray(m._corr.predict(Xf), float)
    fused = np.full(N, np.nan)
    fused[MIN_CTX:] = np.clip(base[MIN_CTX:] + m._alpha * corr, 0.0, None)
    return base, fused


def eval_pt(y_all, pt_series, origins, cap, p):
    qy = X.tweedie_qy(y_all, pt_series, origins, p, cap)
    B = X.expanding_cqr_bounds(qy, y_all[origins], cap)
    w = X.wis_of(B, y_all[origins], qy[:, X.MED_COL])
    lo, hi = B[A95]; y_te = y_all[origins]
    cov = (y_te >= lo) & (y_te <= hi)
    peak = y_te >= np.quantile(y_all, PEAK_Q)
    return dict(wis=round(float(w.mean()), 4), picp95=round(float(cov.mean()), 4),
                k_n=f"{int(cov.sum())}/{len(cov)}",
                peak95=round(float(cov[peak].mean()), 4) if peak.any() else None,
                n_peak=int(peak.sum()), last34=round(float(w[-34:].mean()), 4)), w


def main():
    t0 = time.time()
    data = run_data(PipelineConfig())
    X_all = np.asarray(data["X_all"], float); y_all = np.asarray(data["y_all"], float).ravel()
    X_eval, _c, _ = _resolve_eval_features(X_all, list(data["feature_cols"]), eval_basic=True)
    N = len(y_all)
    print(f"[data] N={N}", flush=True)

    fits = {}
    for pool_end in (269, 205):
        m = FusedEpiForecaster(pi_method="negbin"); m.fit(X_eval[:pool_end], y_all[:pool_end])
        fits[pool_end] = m
        print(f"[fit] pool_end={pool_end} alpha={m._alpha:.3f}", flush=True)

    base_full, _ = points_for(fits[269], y_all, X_eval, N)              # base = pool-independent zero-shot
    out = []
    for pool_end, label in [(269, "A_thesis_68wk"), (205, "B_robust_132origin")]:
        _, fused_full = points_for(fits[pool_end], y_all, X_eval, N)
        origins = np.arange(pool_end, N)
        cap = 2.0 * float(np.max(y_all[:pool_end]))                     # train-only cap
        v1, w1 = eval_pt(y_all, base_full, origins, cap, P_FIX)         # TiRex + campaign
        v2, w2 = eval_pt(y_all, fused_full, origins, cap, P_FIX)        # Fused + campaign
        p_dm, dbar = X.dm(w2, w1)                                       # >0 => base(V1) better
        rec = {"window": label, "test_span": [int(pool_end), int(N)], "n": len(origins),
               "V1_tirex_campaignTweedie": v1, "V2_fused_campaignTweedie": v2,
               "dm_p_fused_vs_tirex": round(p_dm, 5),
               "mean_wis_diff_fused_minus_tirex": round(dbar, 4),
               "point_effect_verdict": ("fusion HURTS Tweedie (V2>V1)" if v2["wis"] > v1["wis"]
                                        else "fusion helps/ties")}
        out.append(rec)
        print(f"\n=== {label}  origins[{pool_end},{N})  n={len(origins)} ===", flush=True)
        print(f"  V1 TiRex+campaignTweedie : WIS {v1['wis']:.4f}  PICP95 {v1['picp95']} ({v1['k_n']})  "
              f"peak95 {v1['peak95']}  last34 {v1['last34']:.4f}", flush=True)
        print(f"  V2 Fused+campaignTweedie : WIS {v2['wis']:.4f}  PICP95 {v2['picp95']} ({v2['k_n']})  "
              f"peak95 {v2['peak95']}  last34 {v2['last34']:.4f}", flush=True)
        print(f"  POINT effect: fused-minus-tirex meanΔ={dbar:+.4f}  DM p={p_dm:.4f}  -> {rec['point_effect_verdict']}", flush=True)

    # ---- leak audit on V2 (fused+campaign, window B): perturb a FUTURE y, bounds at earlier origins must be identical
    m = fits[205]; _, fused_full = points_for(m, y_all, X_eval, N)
    origins = np.arange(205, N); cap = 2.0 * float(np.max(y_all[:205]))
    B0 = X.expanding_cqr_bounds(X.tweedie_qy(y_all, fused_full, origins, P_FIX, cap), y_all[origins], cap)
    yp = y_all.copy(); tamper = len(origins) - 5; yp[origins[tamper]] += 999.0
    Bp = X.expanding_cqr_bounds(X.tweedie_qy(yp, fused_full, origins, P_FIX, cap), yp[origins], cap)
    lo0, hi0 = B0[A95]; lop, hip = Bp[A95]
    first_change = next((i for i in range(len(origins)) if abs(lo0[i]-lop[i])>1e-9 or abs(hi0[i]-hip[i])>1e-9), None)
    leak_ok = (first_change is None) or (first_change > tamper)
    print(f"\n[leak audit V2] tampered origin idx {tamper}; first changed bound idx {first_change} "
          f"-> {'LEAK-FREE' if leak_ok else 'LEAK!'}", flush=True)

    payload = {"decomposition": out, "leak_free_V2": bool(leak_ok), "first_change_idx": first_change,
               "tamper_idx": tamper, "p_fixed": P_FIX,
               "note": "V3(fused+fused_epi.py live)=2.800/68wk, 2.282/132 from _fused_negbin_vs_tweedie.json"}
    (ROOT / "scripts" / "_decompose_point_vs_construction.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s -> scripts/_decompose_point_vs_construction.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
