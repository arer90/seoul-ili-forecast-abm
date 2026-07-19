#!/usr/bin/env python
"""Probe: (1) reproduce the 2.4012 fair baseline reference over 132 origins,
(2) time TimesFM rolling 1-step over a handful of weeks to judge feasibility."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (load_split, build_features, cqr_offsets, build_bounds_cqr,
                                       MED_COL, MIN_CTX, K_CAL, MAX_CONTEXT)
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

ALPHAS = list(FLUSIGHT_ALPHAS)

def main():
    Xtr, ytr, Xte, yte, meta = load_split()
    ntr, nte = len(ytr), len(yte); ntot = ntr + nte
    print(f"split: ntr={ntr} nte={nte} ntot={ntot}  meta={ {k:meta.get(k) for k in ('n','pool_end','test_start','test_end','n_test')} }")
    frozen = np.asarray(json.loads((ROOT/"simulation/results/per_model_optimal/TiRex.json").read_text())
                        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE); tirex_pool = d["tirex_pool"]
    print("TIREX_CACHE keys:", list(d.keys()), "tirex_pool len:", len(tirex_pool))
    yf = np.concatenate([ytr, yte]); cap = 2.0*float(yf.max())
    tirex = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    r_full = yf - tirex
    origins = np.arange(T0, ntot); n = len(origins); y = yf[origins]
    cal_idx = np.arange(T0-K_CAL, T0)
    qy_ref = tirex_empirical_qy(tirex, r_full, origins, cap)
    qy_ref_cal = tirex_empirical_qy(tirex, r_full, cal_idx, cap)
    cqr_ref = cqr_offsets(qy_ref_cal, yf[cal_idx])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_med = qy_ref[:, MED_COL]
    ref_wis = np.asarray(wis_from_bounds(y, ref_B, ALPHAS, median=ref_med), dtype=float)
    lo95, hi95 = ref_B[0.05]; cov = (y>=lo95)&(y<=hi95)
    print(f"REFERENCE fair baseline: n={n}  WIS={ref_wis.mean():.4f}  PICP95={cov.mean():.4f}  k/N={int(cov.sum())}/{n}  cap={cap:.2f}")

    # ---- TimesFM rolling speed test ----
    from simulation.models.timesfm_wrapper import TimesFMForecaster
    import timesfm
    t0 = time.time()
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=MAX_CONTEXT, max_horizon=8,
                                     normalize_inputs=True, infer_is_positive=True,
                                     use_continuous_quantile_head=True, fix_quantile_crossing=True))
    print(f"[timesfm] load+compile: {time.time()-t0:.1f}s")
    test_weeks = list(range(205, 215))
    preds = []
    tt = time.time()
    for t in test_weeks:
        ctx = np.asarray(yf[max(0,t-MAX_CONTEXT):t], dtype=np.float32)
        point, quant = m.forecast(horizon=1, inputs=[ctx])
        preds.append(float(np.asarray(point[0]).ravel()[0]))
    dt = time.time()-tt
    print(f"[timesfm] 10 rolling 1-step forecasts: {dt:.2f}s ({dt/10*1000:.0f} ms/step)")
    print("  weeks 205..214 TimesFM point:", [round(p,3) for p in preds])
    print("  weeks 205..214 TiRex point:  ", [round(float(tirex[t]),3) for t in test_weeks])
    print("  weeks 205..214 actual y:     ", [round(float(yf[t]),3) for t in test_weeks])
    # extrapolate full 52..336 = 285 forecasts
    print(f"  est full 285-week roll: {dt/10*285:.0f}s")

    # cross-check cached TimesFM test-68 vs a fresh roll on week 269
    tfm_json = json.loads((ROOT/"simulation/results/per_model_optimal/TimesFM-2.5.json").read_text())["refit_test_predictions"]
    ctx = np.asarray(yf[max(0,269-MAX_CONTEXT):269], dtype=np.float32)
    p269, _ = m.forecast(horizon=1, inputs=[ctx])
    print(f"  week269 fresh TimesFM roll={float(np.asarray(p269[0]).ravel()[0]):.4f}  cached_json[0]={tfm_json[0]:.4f}")

if __name__ == "__main__":
    raise SystemExit(main())
