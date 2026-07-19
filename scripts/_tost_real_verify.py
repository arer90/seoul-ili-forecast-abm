#!/usr/bin/env python
"""DECISIVE real-data check before the swap-vs-keep risk decision. Re-fits FusedEpi on the REAL data
(proves the numbers are real + reproducible), computes PER-ORIGIN WIS for negbin+PID vs Tweedie on both
windows, then runs the rigorous tests the 3-AI panel demanded:
  * paired mean WIS diff d = tweedie - negbin (>0 => tweedie worse), t-based 95% CI, moving-block bootstrap 95% CI
  * DM(HLN h=1) two-sided p
  * TOST equivalence at margins delta in {0.05, 0.10, 0.15} WIS units: equivalent if the 90% CI of mean(d)
    is fully inside [-delta, +delta]
  * reproducibility: prints summary WIS to compare against the prior run (negbin 2.7439/2.3287, tweedie 2.8001/2.2824)
No live-code edits (reads the live model, pi_method toggled).
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


def wis_arr(qd, y):
    B = {}
    for a in FLUSIGHT_ALPHAS:
        lo = round(a / 2, 4); hi = round(1 - a / 2, 4)
        if lo in qd and hi in qd:
            B[a] = (np.asarray(qd[lo], float), np.asarray(qd[hi], float))
    return np.asarray(wis_from_bounds(y, B, list(FLUSIGHT_ALPHAS), median=np.asarray(qd[0.5], float)), float)


def dm_hln(d):
    n = len(d); v = np.var(d, ddof=1) / n
    if v <= 0: return 1.0
    st = d.mean() / np.sqrt(v) * np.sqrt((n - 1) / n)      # HLN h=1 correct factor sqrt((n-1)/n)
    return float(2 * (1 - stats.t.cdf(abs(st), df=n - 1)))


def block_boot_ci(d, L=6, nb=10000, seed=0):
    rng = np.random.RandomState(seed); n = len(d); k = n // L + 1; means = np.empty(nb)
    for b in range(nb):
        idx = (rng.randint(0, n - L + 1, size=k)[:, None] + np.arange(L)).ravel()[:n]
        means[b] = d[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), means


def tost(d, delta, alpha=0.05):
    """Two one-sided tests: equivalent if 90% (1-2a) CI of mean(d) inside [-delta,+delta]."""
    n = len(d); se = np.std(d, ddof=1) / np.sqrt(n); m = d.mean()
    tcrit = stats.t.ppf(1 - alpha, df=n - 1)
    lo = m - tcrit * se; hi = m + tcrit * se                # 90% CI
    p_lower = 1 - stats.t.cdf((m - (-delta)) / se, df=n - 1)     # H0: mean <= -delta
    p_upper = stats.t.cdf((m - delta) / se, df=n - 1)            # H0: mean >= +delta
    p_tost = max(p_lower, p_upper)
    return dict(margin=delta, ci90=[round(lo, 4), round(hi, 4)], p_tost=round(float(p_tost), 4),
                equivalent=bool(lo > -delta and hi < delta))


def main():
    t0 = time.time()
    data = run_data(PipelineConfig())
    X_all = np.asarray(data["X_all"], float); y_all = np.asarray(data["y_all"], float).ravel()
    X_eval, _c, _ = _resolve_eval_features(X_all, list(data["feature_cols"]), eval_basic=True)
    N = len(y_all)
    print(f"[data] N={N} (real run_data)", flush=True)
    prior = {"A_thesis_68wk": {"negbin": 2.7439, "tweedie": 2.8001},
             "B_robust_132origin": {"negbin": 2.3287, "tweedie": 2.2824}}
    out = []
    for pool_end, label in [(269, "A_thesis_68wk"), (205, "B_robust_132origin")]:
        m = FusedEpiForecaster(pi_method="negbin"); m.fit(X_eval[:pool_end], y_all[:pool_end])
        Xte, yte = X_eval[pool_end:N], y_all[pool_end:N]
        w = {}
        for meth in ("negbin", "tweedie"):
            m.pi_method = meth
            w[meth] = wis_arr(m.predict_quantiles(Xte, y_observed=yte, levels=FQ), yte)
        d = w["tweedie"] - w["negbin"]                       # per-origin; >0 => tweedie worse
        blo, bhi, _ = block_boot_ci(d)
        se = np.std(d, ddof=1) / np.sqrt(len(d))
        tlo, thi = d.mean() - stats.t.ppf(0.975, len(d) - 1) * se, d.mean() + stats.t.ppf(0.975, len(d) - 1) * se
        rec = {
            "window": label, "n": int(len(d)),
            "wis_negbin": round(float(w["negbin"].mean()), 4), "wis_tweedie": round(float(w["tweedie"].mean()), 4),
            "reproduces_prior": {k: (round(float(w[k].mean()), 4), prior[label][k],
                                     abs(float(w[k].mean()) - prior[label][k]) < 1e-3) for k in ("negbin", "tweedie")},
            "mean_diff_tw_minus_nb": round(float(d.mean()), 4),
            "t_ci95": [round(tlo, 4), round(thi, 4)], "boot_ci95": [round(blo, 4), round(bhi, 4)],
            "dm_hln_p": round(dm_hln(d), 4),
            "tost": [tost(d, δ) for δ in (0.05, 0.10, 0.15)],
        }
        out.append(rec)
        print(f"\n=== {label}  n={rec['n']} ===", flush=True)
        print(f"  WIS negbin {rec['wis_negbin']} (prior {prior[label]['negbin']}, match={rec['reproduces_prior']['negbin'][2]}) | "
              f"tweedie {rec['wis_tweedie']} (prior {prior[label]['tweedie']}, match={rec['reproduces_prior']['tweedie'][2]})", flush=True)
        print(f"  mean diff (tw-nb) {rec['mean_diff_tw_minus_nb']:+.4f}  t-CI95 {rec['t_ci95']}  boot-CI95 {rec['boot_ci95']}  DM p={rec['dm_hln_p']}", flush=True)
        for tt in rec["tost"]:
            print(f"    TOST δ={tt['margin']}: 90%CI {tt['ci90']} inside[-{tt['margin']},{tt['margin']}]? equivalent={tt['equivalent']} (p_tost={tt['p_tost']})", flush=True)
    (ROOT / "scripts" / "_tost_real_verify.json").write_text(json.dumps(out, indent=2))
    allmatch = all(r["reproduces_prior"][k][2] for r in out for k in ("negbin", "tweedie"))
    print(f"\n[reproducibility] all 4 WIS reproduce prior run within 1e-3: {allmatch}", flush=True)
    print(f"[done] {time.time()-t0:.0f}s -> scripts/_tost_real_verify.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
