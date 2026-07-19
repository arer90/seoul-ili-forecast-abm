#!/usr/bin/env python
"""TASK A — RIGOROUS same-process FusedEpi vs Tweedie on the IDENTICAL 68-week hold-out.

WHY THIS EXISTS
---------------
The earlier headline "Tweedie 2.713 vs FusedEpi 3.278" was NOT apples-to-apples:
  (1) different interval protocol  — FusedEpi = static empirical residual-quantile PI
      (pi_source=r9_leakfree, symmetric-refit-B); Tweedie = rolling/expanding split-CQR.
  (2) different WIS convention      — the pipeline's ``weighted_interval_score_empirical``
      weights the median dispersion term by 1.0, whereas ``wis_from_bounds`` (used by the
      Tweedie scripts) uses the CORRECT Bracher-2021 median weight 0.5. So even the SCORER
      differed. That alone shifts FusedEpi 3.29 -> 3.12 on the very same intervals.

THE FIX (this script)
---------------------
Evaluate BOTH forecasters on the *identical* 68 hold-out weeks (269..336, the last 68 of
the 337-week series = FusedEpi's frozen refit_test window), the *identical* observed y, the
*identical* FluSight alpha grid, and the *identical* scorer ``wis_from_bounds``. Each model
contributes ONLY its own leak-free predictive quantiles; nothing else differs. That is the
truly-identical evaluation.

FusedEpi predictive quantiles (reconstructed EXACTLY from its stored artifacts):
  * point  = FusedEpi.json ``refit_test_predictions`` (68).
  * PI     = its own k11 half-widths from ``val_metrics.insample_residuals`` (the leak-free
             r9_leakfree residual the pipeline itself used). Two native variants scored:
       FE_static : symmetric residual-quantile band (== the thesis pi_source=r9_leakfree PI
                   that reports WIS 3.2784 / PICP95 0.735 under the pipeline's own WIS).
       FE_adapt  : the pipeline's DEFAULT adaptive PID conformal (MPH_ADAPTIVE_CONFORMAL=1,
                   ``adaptive_conformal_bounds``) — FusedEpi's BEST / best-calibrated PI.
  Tweedie is compared against BOTH, so it must beat FusedEpi's strongest interval too.

Tweedie predictive quantiles (the deployed champion, unchanged):
  * point  = TiRex 1-step rolling.
  * PI     = residual-scale Tweedie skeleton q=mu+Qz*mu^(p/2), p=1.5 (selected pre-205 on
             val [165,205), leak-free w.r.t. the 269..336 window) + expanding split-CQR
             seeded on the strictly-past conformity [165,t) — i.e. seeded pre-269.

Everything leak-free: every FusedEpi/TiRex residual, every Tweedie skeleton block, and every
CQR conformity score at week t uses only weeks < t. No live/pipeline or existing-script edits.

OUTPUT: per-origin WIS for both; mean WIS; DM (HLN h=1) p; PICP95 (+Clopper-Pearson CI);
a WIS-component decomposition (sharpness / under-penalty / over-penalty / median) that shows
HOW Tweedie improves; a peak (y>=50) vs off-season split; a point-vs-head attribution; and a
robustness battery over sub-windows, sliding windows, and a moving-block bootstrap of ΔWIS.
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

from scripts.nov_guard_v3 import setup, dm, cp                       # setup->S, dm=HLN, cp=Clopper-Pearson
from scripts._exp_tweedie import build_span, WSTART, P_GRID          # champion Tweedie skeleton
from scripts._exp_tweedie_cover import rolling_cqr_bounds            # expanding split-CQR
from scripts.dec_boosted_mech import MED_COL, K_CAL
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS, k11_pi_widths_from_residuals
from simulation.analytics.adaptive_conformal import wis_from_bounds, adaptive_conformal_bounds
from simulation.analytics.diagnostics import weighted_interval_score_empirical

A = list(FLUSIGHT_ALPHAS)                        # K=11 FluSight alpha levels
PEAK_Y = 50.0
FE_JSON = ROOT / "simulation/results/per_model_optimal/FusedEpi.json"
OUT_JSON = ROOT / "scripts" / "_rig_fusedepi_cmp.json"
P_STAR = 1.5                                      # deployed champion (pre-205 val-selected)


# ─────────────────────────── WIS + decomposition ────────────────────────────
def wis_components(y, bounds: dict, alphas, median):
    """Per-origin WIS AND its additive components (Bracher 2021), identical to
    ``wis_from_bounds`` in total.

    Args:
        y: observed (n,). bounds: {alpha:(lo,hi)} arrays (n,). alphas: K levels.
        median: (n,) point/median used for the dispersion term.

    Returns:
        dict of (n,) arrays: total, median_term, sharpness, underpen, overpen.
        sharpness = Σ (α/2)(hi−lo); underpen = Σ max(y−hi,0) (y ABOVE the band =
        under-prediction); overpen = Σ max(lo−y,0); all /(K+0.5). Sums to total.
    """
    y = np.asarray(y, float).ravel()
    median = np.asarray(median, float).ravel()
    ks = [a for a in alphas if a in bounds]
    K = len(ks)
    med_term = 0.5 * np.abs(y - median)
    sharp = np.zeros_like(y); under = np.zeros_like(y); over = np.zeros_like(y)
    for a in ks:
        lo, hi = bounds[a]
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        sharp += (a / 2.0) * (hi - lo)
        under += np.maximum(y - hi, 0.0)          # (α/2)(2/α)·1[y>hi]·(y-hi) = (y-hi)+
        over += np.maximum(lo - y, 0.0)
    d = K + 0.5
    return {"total": (med_term + sharp + under + over) / d,
            "median_term": med_term / d, "sharpness": sharp / d,
            "underpen": under / d, "overpen": over / d}


def picp_and_ci(y, bounds, alpha=0.05):
    lo, hi = bounds[alpha]
    cov = (y >= lo) & (y <= hi)
    k = int(cov.sum()); n = len(y)
    return k, n, round(k / n, 4), [round(v, 4) for v in cp(k, n)], cov, float((hi - lo).mean())


def moving_block_bootstrap(delta, block=4, reps=5000, seed=42):
    """Bootstrap distribution of MEAN(delta) via overlapping moving blocks (Kunsch 1989).

    Args: delta (T,) per-origin loss differential; block weeks; reps; seed.
    Returns: (mean, ci_lo, ci_hi, p_two_sided) — 95% percentile CI + 2-sided p that the
    bootstrap mean crosses 0.
    """
    rng = np.random.default_rng(seed)
    T = len(delta); nb = int(np.ceil(T / block)); pool = np.arange(T - block + 1)
    means = np.empty(reps)
    for b in range(reps):
        s = rng.choice(pool, size=nb, replace=True)
        means[b] = np.concatenate([delta[i:i + block] for i in s])[:T].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float(min(1.0, 2.0 * min((means >= 0).mean(), (means <= 0).mean())))
    return float(delta.mean()), float(lo), float(hi), p


# ───────────────────────────────── build ────────────────────────────────────
def main():
    S = setup()
    yf, ntot, cap = S["yf"], S["ntot"], S["cap"]
    tirex = S["tirex"]

    fe = json.loads(FE_JSON.read_text())
    pred_fe = np.asarray(fe["refit_test_predictions"], float)
    res_fe = np.asarray(fe["val_metrics"]["insample_residuals"], float)
    res_fe = res_fe[np.isfinite(res_fe)]
    n68 = len(pred_fe)
    test_start = ntot - n68                                    # 269
    o68 = np.arange(test_start, ntot)                          # weeks 269..336
    y = yf[o68]

    # ── alignment guard: the stored 68-pt FusedEpi point must line up with yf[269:337] ──
    sse = float(np.sum((pred_fe - y) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    r2_fe = 1.0 - sse / sst
    assert n68 == 68 and abs(r2_fe - 0.9357) < 0.01, \
        f"alignment FAIL: n68={n68} r2={r2_fe:.4f} (expect 68 / ~0.9357)"

    # ── FusedEpi native quantiles (2 variants), scored with wis_from_bounds ──
    k11 = k11_pi_widths_from_residuals(np.abs(res_fe), tuple(A))
    B_fe_static = {a: (pred_fe - k11[a], pred_fe + k11[a]) for a in A}     # symmetric residual band
    B_fe_adapt = adaptive_conformal_bounds(pred_fe, k11, res_fe, y, A)     # pipeline default PID

    # ── Tweedie native quantiles: champion skeleton + expanding CQR seeded pre-269 ──
    _, QP, _ = build_span(S, P_STAR, "tirex")
    med_tw = QP[o68 - WSTART][:, MED_COL]
    cal_start = 205 - K_CAL                                # 165: expanding CQR seed start (pre-269)
    B_tw = rolling_cqr_bounds(QP, S, o68, cap, cal_start, None)

    # provenance: reproduce the thesis SSOT WIS (median-weight-1 convention) for the record
    ssot_fe_wis = float(np.mean(weighted_interval_score_empirical(y, pred_fe, res_fe, alphas=A)))

    # ── per-origin WIS + components under the SINGLE shared scorer ──
    C_fe_s = wis_components(y, B_fe_static, A, pred_fe)
    C_fe_a = wis_components(y, B_fe_adapt, A, pred_fe)
    C_tw = wis_components(y, B_tw, A, med_tw)
    w_fe_s, w_fe_a, w_tw = C_fe_s["total"], C_fe_a["total"], C_tw["total"]
    # sanity: components reconstruct wis_from_bounds exactly
    assert np.allclose(w_fe_s, wis_from_bounds(y, B_fe_static, A, median=pred_fe))
    assert np.allclose(w_tw, wis_from_bounds(y, B_tw, A, median=med_tw))

    def summarize(name, w, B, C):
        k, n, picp, ci, cov, w95 = picp_and_ci(y, B)
        peak = y >= PEAK_Y
        return {"model": name, "wis": round(float(w.mean()), 4), "picp95": picp,
                "picp95_ci": ci, "k_of_n": f"{k}/{n}", "mean_w95": round(w95, 3),
                "peak_picp95": round(float(cov[peak].mean()), 4), "n_peak": int(peak.sum()),
                "sharpness": round(float(C["sharpness"].mean()), 4),
                "underpen": round(float(C["underpen"].mean()), 4),
                "overpen": round(float(C["overpen"].mean()), 4),
                "median_term": round(float(C["median_term"].mean()), 4)}

    rows = [summarize("FusedEpi_static", w_fe_s, B_fe_static, C_fe_s),
            summarize("FusedEpi_adaptivePID", w_fe_a, B_fe_adapt, C_fe_a),
            summarize("Tweedie", w_tw, B_tw, C_tw)]

    # ── DM (HLN h=1): Tweedie vs each FusedEpi variant (negative meandiff => Tweedie lower) ──
    dm_out = {}
    for tag, w_fe in (("vs_static", w_fe_s), ("vs_adaptivePID", w_fe_a)):
        p, dbar = dm(w_tw, w_fe)
        dm_out[tag] = {"dm_p_hln": float(p), "mean_wis_diff_tw_minus_fe": round(float(dbar), 4),
                       "tweedie_lower": bool(w_tw.mean() < w_fe.mean()),
                       "significant_beats": bool(w_tw.mean() < w_fe.mean() and p < 0.05)}

    # ── DECOMPOSITION: peak (y>=50) vs off-season (y<50) — HOW/WHY ──
    peak = y >= PEAK_Y; off = ~peak
    strata = {}
    for sname, mask in (("peak_y>=50", peak), ("offseason_y<50", off)):
        m = mask
        strata[sname] = {"n": int(m.sum())}
        for name, w, C, B in (("FusedEpi_static", w_fe_s, C_fe_s, B_fe_static),
                              ("FusedEpi_adaptivePID", w_fe_a, C_fe_a, B_fe_adapt),
                              ("Tweedie", w_tw, C_tw, B_tw)):
            lo, hi = B[0.05]
            strata[sname][name] = {
                "wis": round(float(w[m].mean()), 4),
                "sharpness": round(float(C["sharpness"][m].mean()), 4),
                "underpen": round(float(C["underpen"][m].mean()), 4),
                "overpen": round(float(C["overpen"][m].mean()), 4),
                "picp95": round(float(((y >= lo) & (y <= hi))[m].mean()), 4)}
        strata[sname]["tweedie_minus_static_wis"] = round(
            float(w_tw[m].mean() - w_fe_s[m].mean()), 4)
        strata[sname]["tweedie_minus_adaptive_wis"] = round(
            float(w_tw[m].mean() - w_fe_a[m].mean()), 4)

    # ── ATTRIBUTION: point vs head. Wrap BOTH points with the IDENTICAL static k11 head. ──
    #    TiRex leak-free in-sample residual = yf[52:269]-tirex[52:269] (all pre-269, leak-free).
    r_tirex_pre = (yf[52:test_start] - tirex[52:test_start])
    r_tirex_pre = r_tirex_pre[np.isfinite(r_tirex_pre)]
    k11_tx = k11_pi_widths_from_residuals(np.abs(r_tirex_pre), tuple(A))
    pt_tirex = tirex[o68]
    B_tx_static = {a: (pt_tirex - k11_tx[a], pt_tirex + k11_tx[a]) for a in A}
    w_tx_static = wis_from_bounds(y, B_tx_static, A, median=pt_tirex)
    _, _, picp_tx, _, _, w95_tx = picp_and_ci(y, B_tx_static)
    attribution = {
        "note": ("Identical static residual-quantile head on BOTH points isolates POINT "
                 "quality; Tweedie-native then adds the heteroscedastic head + expanding CQR."),
        "FusedEpi_point_static_head": {"wis": round(float(w_fe_s.mean()), 4), "picp95": rows[0]["picp95"]},
        "TiRex_point_static_head": {"wis": round(float(w_tx_static.mean()), 4),
                                    "picp95": picp_tx, "mean_w95": round(w95_tx, 3)},
        "Tweedie_full_head": {"wis": round(float(w_tw.mean()), 4), "picp95": rows[2]["picp95"]},
        "point_gain_tirex_vs_fused": round(float(w_fe_s.mean() - w_tx_static.mean()), 4),
        "head_gain_tweedie_vs_static": round(float(w_tx_static.mean() - w_tw.mean()), 4)}

    # ── ROBUSTNESS 1: named sub-windows ──
    subwin = {}
    idx = np.arange(n68)
    windows = {"full_68": idx, "off_season_y<50": np.where(off)[0], "peak_y>=50": np.where(peak)[0],
               "first_34": idx[:34], "last_34": idx[-34:], "last_20": idx[-20:]}
    for wn, ii in windows.items():
        if len(ii) < 2:
            continue
        rec = {"n": int(len(ii)),
               "wis_tweedie": round(float(w_tw[ii].mean()), 4),
               "wis_fe_static": round(float(w_fe_s[ii].mean()), 4),
               "wis_fe_adaptive": round(float(w_fe_a[ii].mean()), 4)}
        ps, _ = dm(w_tw[ii], w_fe_s[ii]); pa, _ = dm(w_tw[ii], w_fe_a[ii])
        rec["dm_p_vs_static"] = float(ps); rec["dm_p_vs_adaptive"] = float(pa)
        rec["tweedie_beats_static"] = bool(w_tw[ii].mean() < w_fe_s[ii].mean())
        rec["tweedie_beats_adaptive"] = bool(w_tw[ii].mean() < w_fe_a[ii].mean())
        subwin[wn] = rec

    # ── ROBUSTNESS 2: sliding contiguous windows (L=34, step=1) ──
    L = 34; slide = []
    for s in range(0, n68 - L + 1):
        ii = idx[s:s + L]
        slide.append((float(w_tw[ii].mean()), float(w_fe_s[ii].mean()), float(w_fe_a[ii].mean())))
    slide = np.asarray(slide)
    sliding = {"window_len": L, "n_windows": len(slide),
               "frac_tweedie_beats_static": round(float((slide[:, 0] < slide[:, 1]).mean()), 3),
               "frac_tweedie_beats_adaptive": round(float((slide[:, 0] < slide[:, 2]).mean()), 3),
               "worst_case_tweedie_minus_static": round(float((slide[:, 0] - slide[:, 1]).max()), 4),
               "worst_case_tweedie_minus_adaptive": round(float((slide[:, 0] - slide[:, 2]).max()), 4)}

    # ── ROBUSTNESS 3: moving-block bootstrap of ΔWIS = WIS_FE − WIS_Tweedie ──
    boot = {}
    for tag, w_fe in (("vs_static", w_fe_s), ("vs_adaptivePID", w_fe_a)):
        mean, lo, hi, p = moving_block_bootstrap(w_fe - w_tw)
        boot[tag] = {"mean_delta_wis_fe_minus_tw": round(mean, 4),
                     "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                     "ci_excludes_zero": bool(lo > 0 or hi < 0),
                     "tweedie_decisively_lower": bool(lo > 0), "boot_p_two_sided": round(p, 4)}

    beats_target = bool(dm_out["vs_static"]["significant_beats"]
                        and w_tw.mean() < w_fe_a.mean())

    out = {
        "eval": {"weeks": f"{test_start}..{ntot - 1}", "n_test": n68, "y_peak_thr": PEAK_Y,
                 "scorer": "wis_from_bounds (Bracher 2021, median weight 0.5) — SAME for both",
                 "alphas_K": len(A), "cap": round(cap, 2),
                 "fusedepi_point_r2_on_yf": round(r2_fe, 4),
                 "tweedie_p_star": P_STAR, "tweedie_cqr": "expanding split-CQR seeded [165,t) (pre-269)"},
        "provenance_note": {
            "ssot_fusedepi_wis_reported": 3.2784,
            "ssot_reproduced_weighted_interval_score_empirical": round(ssot_fe_wis, 4),
            "why_differs_from_wis_from_bounds": (
                "weighted_interval_score_empirical uses median dispersion weight 1.0; "
                "wis_from_bounds uses the correct Bracher-2021 weight 0.5. Same intervals, "
                "different median weighting -> FE_static drops 3.29 -> 3.12 under the shared "
                "scorer. Both FusedEpi and Tweedie are scored with wis_from_bounds here.")},
        "headline_table": rows,
        "dm_hln": dm_out,
        "decomposition_peak_vs_offseason": strata,
        "point_vs_head_attribution": attribution,
        "robustness_subwindows": subwin,
        "robustness_sliding_L34": sliding,
        "robustness_moving_block_bootstrap": boot,
        "beats_target_tweedie_dm_beats_fusedepi": beats_target,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ───────────────────────────── console ─────────────────────────────
    print("=" * 84)
    print(f"RIGOROUS same-eval FusedEpi vs Tweedie — identical 68 weeks {test_start}..{ntot-1}, "
          f"identical wis_from_bounds")
    print("=" * 84)
    print(f"provenance: SSOT FusedEpi WIS 3.2784 == weighted_interval_score_empirical "
          f"{ssot_fe_wis:.4f} (median-wt 1.0);\n            under the SHARED wis_from_bounds "
          f"(median-wt 0.5) the SAME static PI = {rows[0]['wis']:.4f}.")
    print(f"\n{'model':>22s} | {'WIS':>7s} {'PICP95':>7s} {'peakP95':>7s} {'W95':>7s} | "
          f"{'sharp':>6s} {'under':>6s} {'over':>6s} {'med':>6s}")
    print("-" * 84)
    for r in rows:
        print(f"{r['model']:>22s} | {r['wis']:>7.4f} {r['picp95']:>7.4f} {r['peak_picp95']:>7.4f} "
              f"{r['mean_w95']:>7.3f} | {r['sharpness']:>6.3f} {r['underpen']:>6.3f} "
              f"{r['overpen']:>6.3f} {r['median_term']:>6.3f}")
    print("\nDM (HLN h=1) Tweedie vs FusedEpi:")
    for tag, d in dm_out.items():
        print(f"  {tag:>16s}: meanΔ(tw-fe)={d['mean_wis_diff_tw_minus_fe']:+.4f}  "
              f"DM p={d['dm_p_hln']:.3e}  sig_beats={d['significant_beats']}")
    print("\nPEAK (y>=50) vs OFF-SEASON decomposition (mean WIS + components):")
    for sn, sd in strata.items():
        print(f"  [{sn}] n={sd['n']}")
        for mdl in ("FusedEpi_static", "FusedEpi_adaptivePID", "Tweedie"):
            m = sd[mdl]
            print(f"      {mdl:>22s}: WIS={m['wis']:>7.4f}  sharp={m['sharpness']:>6.3f}  "
                  f"under={m['underpen']:>6.3f}  over={m['overpen']:>6.3f}  PICP95={m['picp95']:.3f}")
        print(f"      Δ Tweedie−static={sd['tweedie_minus_static_wis']:+.4f}   "
              f"Δ Tweedie−adaptive={sd['tweedie_minus_adaptive_wis']:+.4f}")
    print("\nPOINT vs HEAD attribution (identical static head on both points):")
    print(f"  FusedEpi point + static head: WIS {attribution['FusedEpi_point_static_head']['wis']}")
    print(f"  TiRex    point + static head: WIS {attribution['TiRex_point_static_head']['wis']}  "
          f"(point gain vs Fused = {attribution['point_gain_tirex_vs_fused']:+.4f})")
    print(f"  Tweedie full head            : WIS {attribution['Tweedie_full_head']['wis']}  "
          f"(head gain vs static = {attribution['head_gain_tweedie_vs_static']:+.4f})")
    print("\nROBUSTNESS — named sub-windows (Tweedie WIS / FE_static / FE_adapt):")
    for wn, r in subwin.items():
        print(f"  {wn:>16s} n={r['n']:>3d}: {r['wis_tweedie']:>7.4f} / {r['wis_fe_static']:>7.4f} "
              f"/ {r['wis_fe_adaptive']:>7.4f}   beats_static={r['tweedie_beats_static']} "
              f"beats_adapt={r['tweedie_beats_adaptive']}")
    print(f"\nROBUSTNESS — sliding L={sliding['window_len']} ({sliding['n_windows']} windows): "
          f"Tweedie beats static in {sliding['frac_tweedie_beats_static']*100:.0f}% / "
          f"adaptive in {sliding['frac_tweedie_beats_adaptive']*100:.0f}%")
    print("ROBUSTNESS — moving-block bootstrap ΔWIS=WIS_FE−WIS_Tweedie (block=4, reps=5000):")
    for tag, b in boot.items():
        print(f"  {tag:>16s}: meanΔ={b['mean_delta_wis_fe_minus_tw']:+.4f}  "
              f"95%CI=[{b['ci95_lo']:+.4f},{b['ci95_hi']:+.4f}]  "
              f"decisive={b['tweedie_decisively_lower']}")
    print(f"\nBEATS TARGET (Tweedie DM-beats FusedEpi static AND lower than adaptive): {beats_target}")
    print(f"\nwrote {OUT_JSON}")
    return out


if __name__ == "__main__":
    main()
