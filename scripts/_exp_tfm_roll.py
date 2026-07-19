#!/usr/bin/env python
"""Roll TimesFM-2.5 1-step over weeks 52..336 (leak-free: forecast for week t uses yf[:t]
only; zero-shot so no weight updates). Cache to npz. Verify the test-68 slice (weeks
269..336) reproduces the official cached refit_test_predictions byte-for-byte."""
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
from scripts.dec_boosted_mech import load_split, MIN_CTX, MAX_CONTEXT
CACHE = D.SCRATCH / "exp_timesfm_roll.npz"

def main():
    Xtr, ytr, Xte, yte, meta = load_split()
    ntr, nte = len(ytr), len(yte); ntot = ntr + nte
    yf = np.concatenate([ytr, yte])
    import timesfm
    t0 = time.time()
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    # horizon 1 only; max_context matches TiRex roll (512). All our ctx < 512 anyway.
    m.compile(timesfm.ForecastConfig(max_context=MAX_CONTEXT, max_horizon=8,
                                     normalize_inputs=True, infer_is_positive=True,
                                     use_continuous_quantile_head=True, fix_quantile_crossing=True))
    print(f"[timesfm] load+compile {time.time()-t0:.1f}s")
    idxs = list(range(MIN_CTX, ntot))          # weeks 52..336
    tt = time.time()
    tfm = np.empty(len(idxs), dtype=float)
    for k, t in enumerate(idxs):
        ctx = np.asarray(yf[max(0, t-MAX_CONTEXT):t], dtype=np.float32)
        point, _q = m.forecast(horizon=1, inputs=[ctx])
        tfm[k] = float(np.asarray(point[0]).ravel()[0])
    print(f"[timesfm] rolled {len(idxs)} weeks in {time.time()-tt:.1f}s")
    # tfm_full aligned to week index (nan < MIN_CTX)
    tfm_full = np.concatenate([np.full(MIN_CTX, np.nan), tfm])
    assert len(tfm_full) == ntot
    # verify test-68 vs cached json
    cached = np.asarray(json.loads(
        (ROOT/"simulation/results/per_model_optimal/TimesFM-2.5.json").read_text())
        ["refit_test_predictions"], dtype=float)
    test_roll = tfm_full[ntr:ntot]
    maxdiff = float(np.max(np.abs(test_roll - cached)))
    print(f"[verify] test-68 roll vs cached json: maxdiff={maxdiff:.6e}")
    np.savez(CACHE, tfm_full=tfm_full, test_maxdiff=maxdiff, weeks_lo=MIN_CTX, ntot=ntot)
    print(f"wrote {CACHE}")

if __name__ == "__main__":
    raise SystemExit(main())
