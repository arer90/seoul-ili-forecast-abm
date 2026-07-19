#!/usr/bin/env python
"""HONEST head-to-head: FusedEpi's NegBin+PID interval vs the new Tweedie interval, on the SAME
FusedEpi fused point, leak-free rolling 1-step. Two eval windows:
  A) thesis 68-week hold-out  [269, 337)
  B) robust 132-origin span   [205, 337)   (thesis TEST split, more power)
Both intervals feed y_observed (past obs only). fit ONCE per window; toggle pi_method (fit is
identical, only predict_quantiles differs). Reports WIS, PICP{95,80,50}, peak-regime coverage,
last-34 WIS, and DM(HLN) p-value NegBin-vs-Tweedie. No live-code edits (reads the live model).
"""
from __future__ import annotations
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
PEAK_Y = 50.0                                   # epidemic peak regime (ILI level), campaign convention


def wis_and_arr(qd, y):
    B = {}
    for a in FLUSIGHT_ALPHAS:
        lo = round(a / 2, 4); hi = round(1 - a / 2, 4)
        if lo in qd and hi in qd:
            B[a] = (np.asarray(qd[lo], float), np.asarray(qd[hi], float))
    w = np.asarray(wis_from_bounds(y, B, list(FLUSIGHT_ALPHAS), median=np.asarray(qd[0.5], float)), float)
    return w


def picp(qd, y, lo_q, hi_q):
    lo = np.asarray(qd[lo_q], float); hi = np.asarray(qd[hi_q], float)
    return float(((y >= lo) & (y <= hi)).mean())


def dm_hln(wa, wb):
    d = wa - wb; n = len(d); v = np.var(d, ddof=1) / n
    if v <= 0: return 1.0, float(d.mean())
    st = d.mean() / np.sqrt(v) * np.sqrt((n + 1) / n)
    return float(2 * (1 - stats.t.cdf(abs(st), df=n - 1))), float(d.mean())


def eval_window(X_eval, y_all, pool_end, test_end, label):
    Xtr, ytr = X_eval[:pool_end], y_all[:pool_end]
    Xte, yte = X_eval[pool_end:test_end], y_all[pool_end:test_end]
    m = FusedEpiForecaster(pi_method="negbin"); m.fit(Xtr, ytr)     # fit once (fit is pi-method-agnostic)
    out = {"window": label, "test_span": [int(pool_end), int(test_end)], "n": int(len(yte)),
           "n_peak": int((yte >= PEAK_Y).sum())}
    per = {}
    for method in ["negbin", "tweedie"]:
        m.pi_method = method
        qd = m.predict_quantiles(Xte, y_observed=yte, levels=FQ)
        w = wis_and_arr(qd, yte)
        peak = yte >= PEAK_Y
        per[method] = dict(
            wis=round(float(w.mean()), 4),
            picp95=round(picp(qd, yte, 0.025, 0.975), 4),
            picp80=round(picp(qd, yte, 0.10, 0.90), 4),
            picp50=round(picp(qd, yte, 0.25, 0.75), 4),
            peak_picp95=round(float((((yte >= np.asarray(qd[0.025])) & (yte <= np.asarray(qd[0.975])))[peak]).mean()) if peak.any() else float("nan"), 4),
            last34_wis=round(float(w[-34:].mean()), 4),
        )
        per[method + "_wisarr"] = w
    hln_p, dbar = dm_hln(per["negbin_wisarr"], per["tweedie_wisarr"])   # >0 => tweedie better
    out["negbin"] = per["negbin"]; out["tweedie"] = per["tweedie"]
    out["dm_hln_p_negbin_vs_tweedie"] = round(hln_p, 5)
    out["mean_wis_diff_negbin_minus_tweedie"] = round(dbar, 4)
    out["verdict"] = ("tweedie lower WIS, DM-sig" if per["tweedie"]["wis"] < per["negbin"]["wis"] and hln_p < 0.05
                      else "negbin lower WIS, DM-sig" if per["negbin"]["wis"] < per["tweedie"]["wis"] and hln_p < 0.05
                      else "statistical TIE (DM p>=0.05)")
    return out


def main():
    t0 = time.time()
    data = run_data(PipelineConfig())
    X_all = np.asarray(data["X_all"], float); y_all = np.asarray(data["y_all"], float).ravel()
    X_eval, _cols, _ = _resolve_eval_features(X_all, list(data["feature_cols"]), eval_basic=True)
    N = len(y_all)
    print(f"[data] N={N}  pool_end(frozen)={int(data['pool_end'])}  n_test={int(data['n_test'])}", flush=True)
    results = []
    for pool_end, label in [(269, "A_thesis_68wk"), (205, "B_robust_132origin")]:
        r = eval_window(X_eval, y_all, pool_end, N, label)
        results.append(r)
        print(f"\n=== {label}  test[{pool_end},{N})  n={r['n']} (peak {r['n_peak']}) ===", flush=True)
        for meth in ["negbin", "tweedie"]:
            d = r[meth]
            print(f"  {meth:8s}: WIS {d['wis']:.4f}  PICP95 {d['picp95']:.3f}  PICP80 {d['picp80']:.3f}  "
                  f"PICP50 {d['picp50']:.3f}  peak95 {d['peak_picp95']}  last34 {d['last34_wis']:.4f}", flush=True)
        print(f"  DM(HLN) negbin-vs-tweedie p={r['dm_hln_p_negbin_vs_tweedie']}  "
              f"(meanΔ nb−tw={r['mean_wis_diff_negbin_minus_tweedie']:+.4f})  -> {r['verdict']}", flush=True)
    clean = [{k: v for k, v in r.items() if not k.endswith("_wisarr")} for r in results]
    (ROOT / "scripts" / "_fused_negbin_vs_tweedie.json").write_text(json.dumps(clean, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s  -> scripts/_fused_negbin_vs_tweedie.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
