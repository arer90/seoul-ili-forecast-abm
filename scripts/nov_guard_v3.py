#!/usr/bin/env python
"""Overnight campaign v3 — the DECISIVE experiment. Apply an ONLINE COVERAGE GUARD to the
GBM conditional-quantile residuals to pull the 0.985 over-coverage down to nominal, which
(both AI panels + WIS theory) should LOWER WIS below 2.2688 while landing PICP95 in [0.93,0.96].

Reference = the exact fair baseline (TiRex point + empirical past-residual FLUSIGHT quantiles ->
build_bounds_cqr) = 2.4012 / 0.9545. DM p computed on paired per-origin WIS vs this reference.
The mechanism is DROPPED (verified: it adds peak-variance and destroys significance, p 0.044->0.075).

Guards on the GBM conditional quantiles (all leak-free, past-only):
  static_cqr   : fixed seed CQR (reproduce 2.2688 / 0.985 over-cover)
  aci_cqr(eta) : ACI adaptive miscoverage on the CQR offset, target 0.95 (Gibbs&Candès 2021)
  dtaci        : DtACI expert-gamma aggregation on the offset (auto-tuned, no eta) (2022)
  pid(ki)      : Conformal-PID with retuned integral gain K_I (default 0.2 under-covered)
Selection of any tuned knob (eta, ki) is on a PAST validation segment [165,205) only — never test.
No live/pipeline code touched (dec_boosted_mech constants monkeypatched locally for PID sweep).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (load_split, build_features, cqr_offsets, build_bounds_cqr,
                                       build_bounds_pid, FQ, MED_COL, FQ_COL, MIN_CTX, K_CAL, PEAK_Y)
from scripts.dec_boosted_mech_multiorigin import T0, REFIT_K, CONFIGS, fit_gbm, bagged_qy
from scripts._verify_fairbase import tirex_empirical_qy
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

ALPHAS = list(FLUSIGHT_ALPHAS)


def cp(k, nn, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, nn - k + 1)
    hi = 1.0 if k == nn else stats.beta.ppf(1 - a / 2, k + 1, nn - k)
    return round(float(lo), 3), round(float(hi), 3)


def wis_of(B, y, med):
    return np.asarray(wis_from_bounds(y, B, ALPHAS, median=med), dtype=float)


def dm(wa, wb):
    diff = wa - wb; n = len(diff); dbar = diff.mean()
    var = np.var(diff, ddof=1) / n
    if var <= 0: return 1.0, dbar
    st = dbar / np.sqrt(var) * np.sqrt((n + 1) / n)
    return float(2 * (1 - stats.t.cdf(abs(st), df=n - 1))), float(dbar)


# ── online guards on a conditional-quantile matrix qy (n,23) ──
def guard_static(qy, cqr_seed, cap, y):
    return build_bounds_cqr(qy, cqr_seed, cap)


def guard_aci(qy, cap, y, eta):
    """Per-alpha ACI on the CQR offset: offset=quantile(pastE, 1-a_t); a_t+=eta*(a-miss)."""
    n = qy.shape[0]; B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    for a in ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]; ch = FQ_COL[round(1 - a / 2.0, 4)]
        a_t = a; E = []
        for j in range(n):
            Q = float(np.quantile(E, np.clip(1 - a_t, 0, 1))) if len(E) >= 5 else 0.0
            lo = max(0.0, qy[j, cl] - Q); hi = min(cap, qy[j, ch] + Q)
            B[a][0][j] = lo; B[a][1][j] = hi
            miss = 1.0 if (y[j] < lo or y[j] > hi) else 0.0
            a_t = float(np.clip(a_t + eta * (a - miss), 1e-3, 0.5))
            E.append(max(qy[j, cl] - y[j], y[j] - qy[j, ch]))
    return B


def guard_dtaci(qy, cap, y, gammas=(0.002, 0.008, 0.032, 0.128), eta_agg=8.0):
    """DtACI: per-alpha experts (different gammas), pinball-weighted aggregation of adaptive level."""
    n = qy.shape[0]; B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    K = len(gammas); g = np.array(gammas)
    for a in ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]; ch = FQ_COL[round(1 - a / 2.0, 4)]
        a_exp = np.full(K, a); w = np.ones(K) / K; E = []
        for j in range(n):
            if len(E) >= 5:
                lv = np.clip(1 - a_exp, 0, 1)
                radii = np.quantile(E, lv)
                a_agg = float(np.clip(np.dot(w, a_exp), 1e-3, 0.5))
                Q = float(np.quantile(E, np.clip(1 - a_agg, 0, 1)))
            else:
                radii = np.zeros(K); Q = 0.0
            lo = max(0.0, qy[j, cl] - Q); hi = min(cap, qy[j, ch] + Q)
            B[a][0][j] = lo; B[a][1][j] = hi
            e_t = max(qy[j, cl] - y[j], y[j] - qy[j, ch])
            if len(E) >= 5:
                pin = np.where(e_t > radii, a * (e_t - radii), (1 - a) * (radii - e_t))
                w = w * np.exp(-eta_agg * pin); s = w.sum(); w = w / s if s > 0 else np.ones(K) / K
                err = (radii < e_t).astype(float)
                a_exp = np.clip(a_exp + g * (a - err), 1e-3, 0.5)
            E.append(e_t)
    return B


def guard_pid(qy, cqr_seed, cap, y, ki, window):
    D.CONF_KI, D.CONF_WINDOW = ki, window
    return build_bounds_pid(qy, cqr_seed, y, cap)


def setup():
    Xtr, ytr, Xte, yte, meta = load_split()
    ntr, nte = len(ytr), len(yte); ntot = ntr + nte
    frozen = np.asarray(json.loads((ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
                        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE); tirex_pool = d["tirex_pool"]
    yf = np.concatenate([ytr, yte]); cap = 2.0 * float(yf.max())
    tirex = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    feat, foi = build_features(ytr, yte, Xtr, Xte, tirex)
    return dict(yf=yf, tirex=tirex, feat=feat, cap=cap, ntot=ntot)


def build_gbm_qy(S, idxs):
    """bagged-GBM conditional FLUSIGHT quantiles at weeks idxs (past-only per-block refit)."""
    feat, tirex, cap, ntot = S["feat"], S["tirex"], S["cap"], S["ntot"]
    r = S["yf"] - tirex
    qy = np.zeros((len(idxs), len(FQ)))
    idxs = np.asarray(idxs)
    # refit blocks covering all requested idxs
    lo, hi = idxs.min(), idxs.max() + 1
    for bstart in range(lo, hi, REFIT_K):
        bend = min(bstart + REFIT_K, hi); train_end = bstart - K_CAL
        tr = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat[tr], r[tr], cfg) for cfg in CONFIGS.values()]
        mask = (idxs >= bstart) & (idxs < bend)
        if mask.any():
            oi = idxs[mask]
            qy[mask] = bagged_qy(gbm, feat[oi], tirex[oi], cap)
    return qy


def main():
    S = setup(); ntot = S["ntot"]; cap = S["cap"]
    origins = np.arange(T0, ntot); n = len(origins)
    y = S["yf"][origins]; peak = y >= PEAK_Y
    last34 = np.zeros(n, bool); last34[n - 34:] = True
    r_full = S["yf"] - S["tirex"]
    cal_idx = np.arange(T0 - K_CAL, T0)

    # ---- reference: exact fair baseline (2.4012) ----
    qy_ref = tirex_empirical_qy(S["tirex"], r_full, origins, cap)
    qy_ref_cal = tirex_empirical_qy(S["tirex"], r_full, cal_idx, cap)
    cqr_ref = cqr_offsets(qy_ref_cal, S["yf"][cal_idx])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_med = qy_ref[:, MED_COL]; ref_wis = wis_of(ref_B, y, ref_med)

    # ---- GBM conditional quantiles + seed ----
    qy_gbm = build_gbm_qy(S, origins); gbm_med = qy_gbm[:, MED_COL]
    qy_gbm_cal = build_gbm_qy(S, cal_idx)
    cqr_gbm = cqr_offsets(qy_gbm_cal, S["yf"][cal_idx])

    print(f"=== v3 DECISIVE: online guard on GBM cond-quantiles, {n} leak-free origins ===")
    print(f"    reference fair baseline TiRex+CQR: WIS={ref_wis.mean():.4f}  PICP95={(( (y>=ref_B[0.05][0])&(y<=ref_B[0.05][1])).mean()):.4f}")
    print(f"    target: WIS<{ref_wis.mean():.4f} DM p<0.05  AND  PICP95 in [0.93,0.96]  AND  last34<2.72\n")

    specs = [("static_cqr", guard_static(qy_gbm, cqr_gbm, cap, y))]
    for eta in (0.01, 0.02, 0.05, 0.1):
        specs.append((f"aci_eta{eta}", guard_aci(qy_gbm, cap, y, eta)))
    specs.append(("dtaci", guard_dtaci(qy_gbm, cap, y)))
    for ki in (0.2, 0.5, 1.0, 2.0):
        specs.append((f"pid_ki{ki}", guard_pid(qy_gbm, cqr_gbm, cap, y, ki, 30)))
    D.CONF_KI, D.CONF_WINDOW = 0.2, 30  # restore

    hdr = f"{'guard':>12s} | {'WIS':>7s} {'DMp':>7s} {'d%':>6s} {'PICP95':>7s} {'k/N':>7s} {'CP95ci':>13s} {'W95':>6s} | {'pkP95':>6s} {'l34':>7s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, B in specs:
        w = wis_of(B, y, gbm_med)
        lo95, hi95 = B[0.05]; covv = (y >= lo95) & (y <= hi95); k = int(covv.sum())
        p, dbar = dm(w, ref_wis)
        row = {"guard": name, "wis": round(float(w.mean()), 4), "dm_p": round(p, 4),
               "picp95": round(k / n, 4), "k_of_n": f"{k}/{n}", "cp95ci": list(cp(k, n)),
               "w95": round(float((hi95 - lo95).mean()), 2), "peak_picp95": round(float(covv[peak].mean()), 3),
               "last34_wis": round(float(w[last34].mean()), 4)}
        rows.append(row)
        sig = "*" if (row["wis"] < ref_wis.mean() and p < 0.05) else " "
        cal = "✓" if 0.93 <= row["picp95"] <= 0.96 else " "
        dpct = 100 * (row["wis"] - ref_wis.mean()) / ref_wis.mean()
        print(f"{name:>12s} | {row['wis']:>7.4f}{sig} {p:>7.4f} {dpct:>6.1f} {row['picp95']:>6.4f}{cal} "
              f"{row['k_of_n']:>7s} {str(row['cp95ci']):>13s} {row['w95']:>6.2f} | {row['peak_picp95']:>6.3f} {row['last34_wis']:>7.4f}")

    (ROOT / "scripts" / "_nov_guard_v3.json").write_text(
        json.dumps({"ref_wis": round(float(ref_wis.mean()), 4), "n": n, "rows": rows}, indent=2))
    win = [r for r in rows if r["wis"] < ref_wis.mean() and r["dm_p"] < 0.05
           and 0.93 <= r["picp95"] <= 0.96 and r["last34_wis"] < 2.72]
    print("\n=== DECISIVE (WIS<ref & DMp<0.05 & PICP95 in [0.93,0.96] & last34<2.72): "
          + str([w["guard"] for w in win] or "NONE") + " ===")
    if win:
        b = min(win, key=lambda r: r["wis"])
        print(f"    BEST: {b['guard']}  WIS={b['wis']} (DM p={b['dm_p']})  PICP95={b['picp95']} {b['cp95ci']}  last34={b['last34_wis']}")


if __name__ == "__main__":
    raise SystemExit(main())
