"""CANONICAL thesis protocol, ONE consistent environment (user's methodological demand):
  split = 242 train + 27 val + 68 test (docx: pool_end=269, test=[269,337))
  1) fit on TRAIN [0,242); select Tweedie p on VAL [242,269) by WIS (test never touched)
  2) refit on TRAIN+VAL [0,269); evaluate BOTH negbin+PID and Tweedie(p*) on TEST [269,337) ONLY
Same model pool, same test set, for both interval methods -> a fair head-to-head (no window/pool confound).
"""
from __future__ import annotations
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from simulation.models.fused_epi import FusedEpiForecaster
from simulation.analytics.hub_metrics import FLUSIGHT_QUANTILES, FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds
from simulation.pipeline.data import run_data
from simulation.pipeline.config import PipelineConfig
from scripts.ablation_fusedepi import _resolve_eval_features

FQ = [float(round(q, 4)) for q in FLUSIGHT_QUANTILES]
N_TRAIN, N_VAL = 242, 27
POOL = N_TRAIN + N_VAL                 # 269
P_GRID = (1.1, 1.3, 1.5, 1.7, 1.9)


def wis_arr(qd, y):
    B = {}
    for a in FLUSIGHT_ALPHAS:
        lo = round(a / 2, 4); hi = round(1 - a / 2, 4)
        if lo in qd and hi in qd:
            B[a] = (np.asarray(qd[lo], float), np.asarray(qd[hi], float))
    return np.asarray(wis_from_bounds(y, B, list(FLUSIGHT_ALPHAS), median=np.asarray(qd[0.5], float)), float)


def picp(qd, y, lo, hi):
    return float(((y >= np.asarray(qd[lo])) & (y <= np.asarray(qd[hi]))).mean())


def dm(wa, wb):
    d = wa - wb; n = len(d); v = np.var(d, ddof=1) / n
    if v <= 0: return 1.0, float(d.mean())
    st = d.mean() / np.sqrt(v) * np.sqrt((n - 1) / n)
    return float(2 * (1 - stats.t.cdf(abs(st), df=n - 1))), float(d.mean())


def main():
    t0 = time.time()
    data = run_data(PipelineConfig())
    X_all = np.asarray(data["X_all"], float); y_all = np.asarray(data["y_all"], float).ravel()
    X_eval, _c, _ = _resolve_eval_features(X_all, list(data["feature_cols"]), eval_basic=True)
    N = len(y_all)
    print(f"[data] N={N}  split = {N_TRAIN} train + {N_VAL} val + {N-POOL} test  (pool_end={POOL})", flush=True)

    # ---- 1) p-selection on VAL [242,269) using a model fit on TRAIN [0,242) ONLY (test untouched) ----
    m_tr = FusedEpiForecaster(pi_method="tweedie"); m_tr.fit(X_eval[:N_TRAIN], y_all[:N_TRAIN])
    Xv, yv = X_eval[N_TRAIN:POOL], y_all[N_TRAIN:POOL]
    val_wis = {}
    for p in P_GRID:
        m_tr.tweedie_p = float(p)
        val_wis[p] = round(float(wis_arr(m_tr.predict_quantiles(Xv, y_observed=yv, levels=FQ), yv).mean()), 4)
    p_star = min(val_wis, key=val_wis.get)
    print(f"[val p-selection on 242:269] WIS by p = {val_wis}  -> p* = {p_star}", flush=True)

    # ---- 2) refit on TRAIN+VAL [0,269); evaluate on TEST [269,337) ONLY ----
    m = FusedEpiForecaster(pi_method="negbin"); m.fit(X_eval[:POOL], y_all[:POOL])
    Xte, yte = X_eval[POOL:N], y_all[POOL:N]
    res = {}
    for meth, pp in [("negbin", None), ("tweedie", p_star)]:
        m.pi_method = meth
        if pp is not None: m.tweedie_p = float(pp)
        qd = m.predict_quantiles(Xte, y_observed=yte, levels=FQ)
        w = wis_arr(qd, yte)
        res[meth] = dict(wis=round(float(w.mean()), 4), picp95=round(picp(qd, yte, 0.025, 0.975), 4),
                         picp50=round(picp(qd, yte, 0.25, 0.75), 4), wisarr=w)
    dp, dbar = dm(res["tweedie"]["wisarr"], res["negbin"]["wisarr"])   # >0 => tweedie worse
    out = {
        "protocol": "CANONICAL: fit train[0,242), p* on val[242,269), refit train+val[0,269), eval test[269,337)",
        "p_star_val_selected": p_star, "val_wis_by_p": val_wis, "n_test": int(len(yte)),
        "negbin_pid": {k: v for k, v in res["negbin"].items() if k != "wisarr"},
        "tweedie": {k: v for k, v in res["tweedie"].items() if k != "wisarr"},
        "dm_p_tweedie_vs_negbin": round(dp, 4), "mean_diff_tw_minus_nb": round(dbar, 4),
        "lower_wis_wins": "negbin+PID" if res["negbin"]["wis"] < res["tweedie"]["wis"] else "tweedie",
    }
    (ROOT / "scripts" / "_canonical_split_verify.json").write_text(
        json.dumps({k: v for k, v in out.items()}, indent=2))
    print(f"\n=== CANONICAL TEST[269,337) (n={out['n_test']}), same model pool[0,269), same test ===", flush=True)
    print(f"  negbin+PID : WIS {out['negbin_pid']['wis']}  PICP95 {out['negbin_pid']['picp95']}  PICP50 {out['negbin_pid']['picp50']}", flush=True)
    print(f"  tweedie(p*={p_star}) : WIS {out['tweedie']['wis']}  PICP95 {out['tweedie']['picp95']}  PICP50 {out['tweedie']['picp50']}", flush=True)
    print(f"  DM p (tw vs nb) = {out['dm_p_tweedie_vs_negbin']}  meanΔ(tw-nb)={out['mean_diff_tw_minus_nb']:+.4f}  -> LOWER WIS: {out['lower_wis_wins']}", flush=True)
    print(f"\n[done] {time.time()-t0:.0f}s -> scripts/_canonical_split_verify.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
