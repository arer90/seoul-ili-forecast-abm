#!/usr/bin/env python
"""Fast deliverable path for the TiRex+TimesFM ensemble experiment: reference (2.4012),
harness sanity (pure-TiRex bagged-GBM+CQR), leak-free do-no-harm w-selection on PAST
origins [171,205), and the FINAL ensemble candidate over 132 test origins. Skips the
expensive TEST-origin oracle scan (context only). Line-buffered; writes JSON immediately."""
from __future__ import annotations
import os, sys, json, time, functools
print = functools.partial(print, flush=True)
# more threads for HistGBM (override the module defaults BEFORE import)
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "6")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "6")
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts._exp_tirex_timesfm_ensemble import (load_state, eval_candidate, reference_wis,
                                                 dm, cp)
from scripts.dec_boosted_mech import PEAK_Y
from scripts.dec_boosted_mech_multiorigin import T0

def main():
    t0 = time.time()
    S = load_state(); ntot = S["ntot"]; tirex = S["tirex"]; tfm = S["tfm"]; yf = S["yf"]
    test_origins = np.arange(T0, ntot); nT = len(test_origins); y_test = yf[test_origins]

    ref_wis, ref_B, _ = reference_wis(S, test_origins)
    ref_lo, ref_hi = ref_B[0.05]; ref_cov = (y_test>=ref_lo)&(y_test<=ref_hi)
    print(f"REFERENCE fair TiRex+CQR: WIS={ref_wis.mean():.4f} PICP95={ref_cov.mean():.4f} k/N={int(ref_cov.sum())}/{nT}")

    # point accuracy diagnostic
    for nm, p in [("TiRex", tirex), ("TimesFM", tfm)]:
        e = y_test - p[test_origins]
        print(f"  point {nm:9s} MAE={np.abs(e).mean():.4f} RMSE={np.sqrt((e**2).mean()):.4f}")

    # harness sanity (pure-TiRex bagged-GBM+CQR ~ 2.2765/0.985/DMp0.057)
    pure = eval_candidate(S, tirex, T0, ntot); p_pure,_ = dm(pure["wis"], ref_wis)
    print(f"SANITY pure-TiRex bagged-GBM+CQR: WIS={pure['wis'].mean():.4f} PICP95={pure['cov'].mean():.4f} DMp={p_pure:.4f} (expect ~2.2765/0.985/0.057)")

    # do-no-harm w-selection on PAST origins [171,205)
    A_GRID = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.0]
    SEL_LO, SEL_HI = T0-34, T0
    print(f"--- do-no-harm selection on PAST origins [{SEL_LO},{SEL_HI}) ---")
    sel = {}
    for a in A_GRID:
        base = a*tirex + (1-a)*tfm
        c = eval_candidate(S, base, SEL_LO, SEL_HI); sel[a] = float(c["wis"].mean())
        print(f"  a={a:.1f} past-WIS={sel[a]:.4f} PICP95={c['cov'].mean():.3f}")
    base_wis = sel[1.0]; a_star = min(sel, key=lambda k: sel[k]); gain = base_wis - sel[a_star]
    adopt = a_star if (a_star != 1.0 and gain >= 0.05) else 1.0
    print(f"  pure past-WIS={base_wis:.4f} best a={a_star} (gain={gain:.4f}) -> ADOPT a={adopt}")

    # final candidate over 132 test origins
    base_star = adopt*tirex + (1-adopt)*tfm
    cand = eval_candidate(S, base_star, T0, ntot)
    w = cand["wis"]; cov = cand["cov"]; k = int(cov.sum()); p,dbar = dm(w, ref_wis)
    last34 = np.zeros(nT, bool); last34[nT-34:] = True; peak = y_test >= PEAK_Y
    print(f"=== FINAL ensemble candidate (a={adopt}) over {nT} test origins ===")
    print(f"  WIS={w.mean():.4f} (ref 2.4012, dbar={dbar:+.4f})")
    print(f"  DM p vs 2.4012 = {p:.4f} {'SIG(<0.05)' if p<0.05 else 'NOT sig'}")
    print(f"  PICP95={cov.mean():.4f} k/N={k}/{nT} CP95ci={cp(k,nT)} {'CAL[0.93,0.96]' if 0.93<=cov.mean()<=0.96 else 'OUT'}")
    print(f"  last34 WIS={w[last34].mean():.4f} {'<2.72 OK' if w[last34].mean()<2.72 else 'FAIL'}")
    print(f"  mean W95={cand['w95'].mean():.2f} peak PICP95={cov[peak].mean():.3f} (n={int(peak.sum())})")

    out = {"reference_wis": round(float(ref_wis.mean()),4), "reference_picp95": round(float(ref_cov.mean()),4),
           "pure_tirex_gbm": {"wis": round(float(pure["wis"].mean()),4), "picp95": round(float(pure["cov"].mean()),4), "dm_p": round(p_pure,4)},
           "selection": {"grid": A_GRID, "past_window": [int(SEL_LO),int(SEL_HI)],
                         "past_wis": {str(a): round(v,4) for a,v in sel.items()}, "a_star": a_star, "gain": round(gain,4), "adopted_a": adopt},
           "final": {"a": adopt, "wis": round(float(w.mean()),4), "dm_p": round(p,4),
                     "picp95": round(float(cov.mean()),4), "k_of_n": f"{k}/{nT}", "cp95ci": list(cp(k,nT)),
                     "last34_wis": round(float(w[last34].mean()),4), "mean_w95": round(float(cand["w95"].mean()),2),
                     "peak_picp95": round(float(cov[peak].mean()),3)},
           "elapsed_sec": round(time.time()-t0,1)}
    (ROOT/"scripts"/"_exp_fast.json").write_text(json.dumps(out, indent=2))
    print(f"wrote scripts/_exp_fast.json ({out['elapsed_sec']}s)")

if __name__ == "__main__":
    raise SystemExit(main())
