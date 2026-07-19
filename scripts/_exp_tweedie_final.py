#!/usr/bin/env python
"""FINAL verification — the decisive Tweedie distributional-head forecaster.

A-PRIORI recipe (every element fixed by principle / part-1 evidence, NOT by peeking
at the 132 test origins):
  * mean       = TiRex point  (part-1: learning an LGB/GLM mean correction over-widens
                 and DESTROYS significance; TiRex is already the strong point)
  * distribution = Tweedie (1<p<2, the exponential-dispersion member for continuous
                 non-negative overdispersed rates with a 0 mass); FLUSIGHT quantiles via
                 the RESIDUAL-SCALE model: q = mu + Qz * mu^(p/2), Qz = empirical past
                 standardized-residual quantiles, mu^(p/2) = Tweedie variance-function
                 scaling (heteroscedastic; widens at peaks). Nonparametric shape.
  * conformal  = EXPANDING split-CQR — a-priori design choice for the KNOWN seasonal
                 non-stationarity of ILI (a frozen pre-peak seed is not exchangeable
                 with the peak-heavy tail; expanding recalibration is the standard
                 leak-free remedy, Gibbs-Candes 2021 / Barber 2023). No window/eta knob;
                 at the first origin it EQUALS the static 40-week seed, then generalizes.
  * p          = the ONLY tuned scalar; chosen by argmin pre-T0 validation WIS on
                 origins [165,205) (never the test origins).

Leak-free: per-block GBM-free head trained/calibrated only on weeks strictly before each
origin (train_end = block_start - K_CAL); every conformity score at week s uses q_y(s)
and the KNOWN y_s with s<t only. SPOTLESS-cap check: rerun with cap = 2*max(y_train)
(train-only) to prove the cap never binds (identical numbers).

Reference = the exact 2.4012 fair baseline. Per-origin WIS DM-tested (HLN h=1). Reports
WIS, DM p (+ mean diff), PICP95 (k/N) + Clopper-Pearson CI, last-34 WIS, mean-W95, peak
coverage. No live/pipeline or dec_boosted_mech*.py edits.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dec_boosted_mech import (cqr_offsets, build_bounds_cqr, MED_COL, K_CAL)
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup, dm, cp, wis_of, ALPHAS
from scripts._exp_tweedie import build_span, WSTART, P_GRID
from scripts._exp_tweedie_cover import rolling_cqr_bounds, static_bounds

QMETHOD = "pearson"            # residual-scale (nonparametric shape); a-priori
CAL_START = T0 - K_CAL        # 165
VAL_CAL_START = 125


def span_for(S, p):
    QG, QP, _ = build_span(S, p, "tirex")
    return QP if QMETHOD == "pearson" else QG


def full_metrics(B, y, med, ref_wis, n):
    w = wis_of(B, y, med)
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    p_dm, dbar = dm(w, ref_wis)
    peak = y >= 50.0
    return dict(
        wis=round(float(w.mean()), 4), dm_p=float(p_dm), dm_meandiff=round(float(dbar), 4),
        picp95=round(k / n, 4), k_of_n=f"{k}/{n}", cp95ci=[round(v, 4) for v in cp(k, n)],
        last34_wis=round(float(w[n - 34:].mean()), 4),
        mean_w95=round(float((hi95 - lo95).mean()), 3),
        peak_picp95=round(float(covv[peak].mean()), 3), n_peak=int(peak.sum()),
        wis_arr=w,
    )


def main():
    S = setup()
    ntot = S["ntot"]
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S["yf"][origins]
    cal = np.arange(T0 - K_CAL, T0)
    r_full = S["yf"] - S["tirex"]
    cap_full = S["cap"]                                  # 2*max(y_full) — matches reference
    cap_train = 2.0 * float(S["yf"][:269].max())         # 2*max(y_train) — spotless (train-only)

    # ---- exact 2.4012 reference (per-origin WIS for DM) ----
    qy_ref = tirex_empirical_qy(S["tirex"], r_full, origins, cap_full)
    cqr_ref = cqr_offsets(tirex_empirical_qy(S["tirex"], r_full, cal, cap_full), S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap_full)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())
    ref_k = int(((y >= ref_B[0.05][0]) & (y <= ref_B[0.05][1])).sum())

    # ---- honest p-selection: argmin pre-T0 val WIS (expanding CQR, pearson head) ----
    val_origins = np.arange(T0 - K_CAL, T0)
    y_val = S["yf"][val_origins]
    val_scores = {}
    for p in P_GRID:
        Q = span_for(S, p)
        Bv = rolling_cqr_bounds(Q, S, val_origins, cap_full, VAL_CAL_START, None)
        medv = Q[val_origins - WSTART][:, MED_COL]
        val_scores[p] = float(wis_of(Bv, y_val, medv).mean())
    p_star = min(val_scores, key=val_scores.get)

    # ---- headline config on TEST (both caps) ----
    Qp = span_for(S, p_star)
    med = Qp[origins - WSTART][:, MED_COL]
    B_full = rolling_cqr_bounds(Qp, S, origins, cap_full, CAL_START, None)
    B_train = rolling_cqr_bounds(Qp, S, origins, cap_train, CAL_START, None)
    m_full = full_metrics(B_full, y, med, ref_wis, n)
    m_train = full_metrics(B_train, y, med, ref_wis, n)

    # ---- sanity: expand == static at the FIRST origin (window [165,205) == seed) ----
    B_static = static_bounds(Qp, S, origins, cap_full, CAL_START)
    first_match = bool(np.isclose(B_full[0.05][0][0], B_static[0.05][0][0]) and
                       np.isclose(B_full[0.05][1][0], B_static[0.05][1][0]))

    # ---- robustness: all configs clearing all four bars (from cover sweep) ----
    cover = json.loads((ROOT / "scripts" / "_exp_tweedie_cover.json").read_text())
    winners = [r["config"] for r in cover["constraint_winners"]]

    bars = dict(
        beats_wis=bool(m_full["wis"] < ref_mean),
        dm_sig=bool(m_full["dm_p"] < 0.05),
        picp_in_band=bool(0.93 <= m_full["picp95"] <= 0.96),
        last34_lt_272=bool(m_full["last34_wis"] < 2.72),
    )
    all_bars = all(bars.values())
    cap_binds = not (abs(m_full["wis"] - m_train["wis"]) < 1e-9 and
                     m_full["picp95"] == m_train["picp95"])

    best_config = f"tirex_{QMETHOD}_p{p_star}_expand"
    print("=" * 78)
    print(f"REFERENCE fair baseline TiRex+CQR: WIS={ref_mean:.4f}  "
          f"PICP95={ref_k/n:.4f} ({ref_k}/{n})  last34={float(ref_wis[n-34:].mean()):.4f}")
    print("=" * 78)
    print(f"pre-T0 val WIS by p (expanding CQR, pearson head): "
          f"{ {p: round(v,4) for p,v in val_scores.items()} }  -> p*={p_star}")
    print(f"\nHEADLINE  {best_config}   ({n} leak-free rolling 1-step origins, weeks {T0}..{ntot-1})")
    print(f"  WIS          = {m_full['wis']:.4f}   (reference 2.4012; delta {100*(m_full['wis']-ref_mean)/ref_mean:+.1f}%)")
    print(f"  DM p vs 2.4012 = {m_full['dm_p']:.2e}   (mean per-origin WIS diff = {m_full['dm_meandiff']:+.4f})")
    print(f"  PICP95       = {m_full['picp95']:.4f}   ({m_full['k_of_n']})   Clopper-Pearson 95% CI = {m_full['cp95ci']}")
    print(f"  last34 WIS   = {m_full['last34_wis']:.4f}   (< 2.72 target; reference last34 = {float(ref_wis[n-34:].mean()):.4f})")
    print(f"  mean-W95     = {m_full['mean_w95']:.3f}")
    print(f"  peak PICP95  = {m_full['peak_picp95']:.3f}  (n_peak y>=50 = {m_full['n_peak']})")
    print(f"\nBARS: beats_WIS={bars['beats_wis']}  DM_sig(p<0.05)={bars['dm_sig']}  "
          f"PICP95∈[0.93,0.96]={bars['picp_in_band']}  last34<2.72={bars['last34_lt_272']}  "
          f"=> ALL={all_bars}")
    print(f"\nLEAK-FREE checks:")
    print(f"  expand==static at first origin (window[165,205)==seed): {first_match}")
    print(f"  train-only cap (2*max y_train={cap_train:.1f}) -> WIS={m_train['wis']:.4f} "
          f"PICP95={m_train['picp95']:.4f}  cap_binds={cap_binds} (False = spotless)")
    print(f"\nROBUSTNESS: {len(winners)} configs clear ALL FOUR bars (not a knife-edge):")
    print(f"  {winners}")

    out = {"best_config": best_config, "reference_wis": round(ref_mean, 4), "n": n,
           "val_wis_by_p": {str(p): round(v, 4) for p, v in val_scores.items()}, "p_star": p_star,
           "headline_full_cap": {k: v for k, v in m_full.items() if k != "wis_arr"},
           "headline_train_cap": {k: v for k, v in m_train.items() if k != "wis_arr"},
           "bars": bars, "all_bars": all_bars, "cap_binds": cap_binds,
           "expand_eq_static_first_origin": first_match,
           "robust_winners": winners}
    (ROOT / "scripts" / "_exp_tweedie_final.json").write_text(json.dumps(out, indent=2))
    print("\nwrote scripts/_exp_tweedie_final.json")


if __name__ == "__main__":
    raise SystemExit(main())
