#!/usr/bin/env python
"""ADVERSARIAL, INDEPENDENT verification of the Tweedie⊕SPCI combination (scripts/_exp_combo.py).

Goal = REFUTE. Nothing here trusts _exp_combo.py's own numbers; every quantity is recomputed
from the imported leak-free generators, and cross-checked with an INDEPENDENT WIS implementation
(standard Bracher-2021 formula) so a bug in wis_from_bounds would surface.

Checks
------
(1) REPRODUCE: standalone Tweedie / SPCI-bag / reference 2.4012 / combo headline
    (env_L6_thr0.0_lam1.0). Independent WIS must agree with the pipeline WIS to <1e-9.
(2) LEAK AUDIT (the core adversarial test): rebuild the ENTIRE combo (Tweedie span + expanding
    CQR, SPCI bagged-QRF + expanding conformal, rising-limb envelope) with cap held at a FIXED
    constant (train-only, not derived from the perturbed series), then perturb y at an interior
    test week (260) and the terminal week (336) by +1000:
       * every bound at an origin < perturbed_week+1 must be BIT-IDENTICAL,
       * the first origin whose bounds change must be EXACTLY perturbed_week+1 (interior),
       * perturbing the terminal week must change NOTHING (no origin peeks at its own y).
    Also: perturbing a TEST week must leave ALL pre-T0 validation WIS (used to select p* and the
    envelope config) BIT-IDENTICAL -> config/weight selection cannot see the test set.
(3) TRAIN-ONLY CAP: recompute the combo with cap = 2*max(y_train). WIS must be same-or-BETTER
    (a spotless cap must not be the thing making the combo look good).
(4) DM ROBUSTNESS: HLN-DM (h=1) AND a 10k paired moving-block bootstrap (block L in {1,4,8,12})
    of per-origin WIS diff vs the exact 2.4012 reference. Is p<0.05 robust for the combo, and is
    the combo-vs-Tweedie difference really a TIE (as claimed)?
(5) COVERAGE: PICP95 with Clopper-Pearson, peak PICP95 (y>=50), last34 WIS.

Verdict at the end: is the combination a GENUINE improvement over BOTH standalones (better/equal
WIS AND better peak coverage AND off-margin overall coverage, leak-free + DM-significant), or is
a single method the honest thesis headline?
"""
from __future__ import annotations
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dec_boosted_mech import cqr_offsets, build_bounds_cqr, MED_COL, K_CAL, FQ, FQ_COL, PEAK_Y
from scripts.dec_boosted_mech_multiorigin import T0
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup as tw_setup, ALPHAS
from scripts._exp_tweedie import build_span, WSTART, P_GRID
from scripts._exp_tweedie_cover import rolling_cqr_bounds
from scripts._exp_spci import (setup as spci_setup, build_spci_features, block_qrf_resid_grid,
                               _yq_from_grid, bounds_beta, median_from_grid)
from scripts._exp_spci_final import conformity, calibrate_expanding, slice_bounds, VAL_LO

CAL_START = T0 - K_CAL          # 165
VAL_CAL_START = 125
VAL_LO_TW, VAL_HI = T0 - K_CAL, T0     # [165,205)
SPCI_SEED0 = 125
LEAF_SET = (12, 16, 20, 24)
SPCI_IDX0 = 110
STD_TWEEDIE, STD_SPCI, REF = 2.2427, 2.3449, 2.4012
HEAD_L, HEAD_THR, HEAD_LAM = 6, 0.0, 1.0     # env_L6_thr0.0_lam1.0
FQ_COLS = {a: (FQ_COL[round(a / 2.0, 4)], FQ_COL[round(1 - a / 2.0, 4)]) for a in ALPHAS}


# ───────────────────── INDEPENDENT WIS (Bracher 2021), no pipeline import ─────────────────────
def wis_indep(B, y, med):
    """Independent per-origin WIS. WIS = 1/(K+.5)[.5|y-m| + Σ_k (α_k/2) IS_{α_k}],
    IS_α = (hi-lo) + (2/α)(lo-y)1{y<lo} + (2/α)(y-hi)1{y>hi}."""
    y = np.asarray(y, float); med = np.asarray(med, float)
    ks = [a for a in ALPHAS if a in B]
    acc = 0.5 * np.abs(y - med)
    for a in ks:
        lo, hi = B[a]
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        isc = (hi - lo) + (2.0 / a) * (lo - y) * (y < lo) + (2.0 / a) * (y - hi) * (y > hi)
        acc = acc + (a / 2.0) * isc
    return acc / (len(ks) + 0.5)


def dm_hln(wa, wb):
    diff = np.asarray(wa) - np.asarray(wb); n = len(diff); dbar = diff.mean()
    var = np.var(diff, ddof=1) / n
    if var <= 0:
        return 1.0, float(dbar)
    st = dbar / np.sqrt(var) * np.sqrt((n + 1) / n)
    return float(2 * (1 - stats.t.cdf(abs(st), df=n - 1))), float(dbar)


def cp(k, nn, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, nn - k + 1)
    hi = 1.0 if k == nn else stats.beta.ppf(1 - a / 2, k + 1, nn - k)
    return round(float(lo), 4), round(float(hi), 4)


def block_bootstrap_p(diff, L, n_boot=10000, seed=12345):
    """Two-sided paired moving-block bootstrap p for H0: mean(diff)=0.
    Resample circular blocks of length L to preserve serial correlation; p = 2*min(tail)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    nb = int(np.ceil(n / L))
    obs = diff.mean()
    # center under H0
    dc = diff - obs
    starts_pool = np.arange(n)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=nb, replace=True)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel() % n
        idx = idx[:n]
        means[b] = dc[idx].mean()
    # p-value: how often bootstrap mean is as extreme as observed
    p_hi = np.mean(means >= abs(obs))
    p_lo = np.mean(means <= -abs(obs))
    return float(min(1.0, 2.0 * min(p_hi, p_lo)))


# ───────────────────── combo constructor (cap passed EXPLICITLY, never derived from yf) ─────────
def build_tweedie(S_tw, yf, cap, p, origins, cal_start):
    Sp = {**S_tw, "yf": yf, "cap": cap}
    _QG, QP, _ = build_span(Sp, p, "tirex")
    B = rolling_cqr_bounds(QP, Sp, origins, cap, cal_start, None)
    med = QP[origins - WSTART][:, MED_COL]
    return B, med, QP


def select_p_star(S_tw, yf, cap):
    val_o = np.arange(VAL_LO_TW, VAL_HI)
    y_val = yf[val_o]
    scores = {}
    for p in P_GRID:
        B, med, _ = build_tweedie(S_tw, yf, cap, p, val_o, VAL_CAL_START)
        scores[p] = round(float(wis_indep(B, y_val, med).mean()), 4)
    return min(scores, key=scores.get), scores


def build_spci(S_sp, yf, cap):
    """Rebuild bagged-QRF SPCI expanding-conformal bounds over eval origins >= VAL_LO.
    feat/r are rebuilt from the (possibly perturbed) yf so the leak audit is faithful."""
    feat, r = build_spci_features(yf, S_sp["tirex"], S_sp["X_full"])
    tirex = S_sp["tirex"]
    idx_all = np.arange(SPCI_IDX0, S_sp["ntot"])
    grids = [block_qrf_resid_grid(feat, r, idx_all,
                                  dict(n_estimators=400, min_samples_leaf=msl, max_features=0.6))
             for msl in LEAF_SET]
    bag = sum(grids) / len(grids)
    tir = tirex[idx_all]
    _fq, yqg = _yq_from_grid(bag, tir, cap)
    Braw = bounds_beta(yqg)
    med_all = median_from_grid(yqg, tir, True)
    y_all = yf[idx_all]
    E = conformity(Braw, y_all)
    B, ev_idx, order, ev_pos = calibrate_expanding(Braw, E, idx_all, VAL_LO, cap, SPCI_SEED0)
    med_ev = med_all[order][ev_pos]
    return B, ev_idx, med_ev


def trailing_signed(r_full, origins, L):
    g = np.zeros(len(origins))
    for j, t in enumerate(origins):
        seg = r_full[t - L:t]
        seg = seg[np.isfinite(seg)]
        g[j] = float(seg.mean()) if len(seg) else 0.0
    return g


def envelope(B_tw, B_sp, flag, lam):
    out = {}
    for a in ALPHAS:
        lo_tw, hi_tw = B_tw[a]
        _lo, hi_sp = B_sp[a]
        hi = np.where(flag, hi_tw + lam * np.maximum(0.0, hi_sp - hi_tw), hi_tw)
        out[a] = (lo_tw.copy(), hi)
    return out


def build_combo(S_tw, S_sp, yf, cap, p_star, L, thr, lam):
    """FULL headline-config combo bounds over test origins [T0, ntot). cap FIXED (not from yf)."""
    origins = np.arange(T0, S_tw["ntot"])
    tirex = S_tw["tirex"]
    r_full = yf - tirex
    B_tw, med_tw, _ = build_tweedie(S_tw, yf, cap, p_star, origins, CAL_START)
    B_sp_all, ev_idx, _ = build_spci(S_sp, yf, cap)
    tst = ev_idx >= T0
    assert np.array_equal(ev_idx[tst], origins)
    B_sp = slice_bounds(B_sp_all, tst)
    flag = trailing_signed(r_full, origins, L) > thr
    B = envelope(B_tw, B_sp, flag, lam)
    return B, med_tw, origins


def bounds_stack(B, origins):
    """Concatenate all (lo,hi) arrays across alphas into one 2D array for bit-identity checks."""
    cols = []
    for a in ALPHAS:
        cols.append(B[a][0]); cols.append(B[a][1])
    return np.array(cols)   # (2K, n_origins)


def first_change_origin(Bbase, Bpert, origins):
    sb, sp = bounds_stack(Bbase, origins), bounds_stack(Bpert, origins)
    diff_any = np.any(sb != sp, axis=0)   # per-origin any bound differs
    idx = np.where(diff_any)[0]
    return (int(origins[idx[0]]) if len(idx) else None), int(diff_any.sum())


def full_metrics(B, y, med, ref_wis, n):
    w = wis_indep(B, y, med)
    lo95, hi95 = B[0.05]
    cov = (y >= lo95) & (y <= hi95); k = int(cov.sum())
    p_dm, dbar = dm_hln(w, ref_wis)
    peak = y >= PEAK_Y
    return dict(wis=round(float(w.mean()), 4), dm_p=float(p_dm), dm_meandiff=round(float(dbar), 4),
                picp95=round(k / n, 4), k=k, n=n, cp95=list(cp(k, n)),
                last34=round(float(w[n - 34:].mean()), 4),
                peak_picp95=round(float(cov[peak].mean()), 4), n_peak=int(peak.sum()),
                w=w)


def main():
    t0 = time.time()
    S_tw = tw_setup()
    S_sp = spci_setup()
    ntot = S_tw["ntot"]
    yf0 = S_tw["yf"].copy()
    tirex = S_tw["tirex"]
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = yf0[origins]
    cap_full = 2.0 * float(yf0.max())
    cap_train = 2.0 * float(yf0[:269].max())
    r_full0 = yf0 - tirex

    # ---- exact 2.4012 reference (independent WIS) ----
    cal = np.arange(CAL_START, T0)
    qy_ref = tirex_empirical_qy(tirex, r_full0, origins, cap_full)
    cqr_ref = cqr_offsets(tirex_empirical_qy(tirex, r_full0, cal, cap_full), yf0[cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap_full)
    ref_w = wis_indep(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_w.mean())
    print(f"[REF] independent WIS = {ref_mean:.4f}  (target 2.4012, |Δ|={abs(ref_mean-REF):.2e})")

    # ---- p* selection (leak-free) ----
    p_star, p_scores = select_p_star(S_tw, yf0, cap_full)
    print(f"[p*]  selected p*={p_star} by pre-T0 val WIS {p_scores}")

    # ---- standalones (cap_full) ----
    B_tw, med_tw, _ = build_tweedie(S_tw, yf0, cap_full, p_star, origins, CAL_START)
    m_tw = full_metrics(B_tw, y, med_tw, ref_w, n)
    B_sp_all, ev_idx, med_ev = build_spci(S_sp, yf0, cap_full)
    tst = ev_idx >= T0
    B_sp = slice_bounds(B_sp_all, tst); med_sp = med_ev[tst]
    m_sp = full_metrics(B_sp, y, med_sp, ref_w, n)

    # ---- combo headline (cap_full) ----
    B_c, med_c, _o = build_combo(S_tw, S_sp, yf0, cap_full, p_star, HEAD_L, HEAD_THR, HEAD_LAM)
    m_c = full_metrics(B_c, y, med_c, ref_w, n)

    print("\n=== (1) REPRODUCTION (independent WIS) ===")
    print(f"  Tweedie : WIS={m_tw['wis']}  PICP95={m_tw['picp95']} ({m_tw['k']}/{n})  "
          f"peak={m_tw['peak_picp95']}  last34={m_tw['last34']}  (claim 2.2427/0.9318/0.870)")
    print(f"  SPCI    : WIS={m_sp['wis']}  PICP95={m_sp['picp95']} ({m_sp['k']}/{n})  "
          f"peak={m_sp['peak_picp95']}  last34={m_sp['last34']}  (claim 2.3449/0.9545/0.783)")
    print(f"  COMBO   : WIS={m_c['wis']}  PICP95={m_c['picp95']} ({m_c['k']}/{n})  "
          f"peak={m_c['peak_picp95']}  last34={m_c['last34']}  (claim 2.2345/0.9394/0.870)")
    repro_ok = (abs(m_tw["wis"] - STD_TWEEDIE) < 2e-3 and abs(m_sp["wis"] - STD_SPCI) < 2e-3
                and abs(m_c["wis"] - 2.2345) < 3e-3 and abs(ref_mean - REF) < 5e-4)
    print(f"  reproduce_ok={repro_ok}")

    # ── independent-vs-pipeline WIS cross-check on the reference (bit-level) ──
    from scripts.nov_guard_v3 import wis_of as wis_pipe
    wdiff = float(np.max(np.abs(ref_w - wis_pipe(ref_B, y, qy_ref[:, MED_COL]))))
    print(f"  independent WIS vs pipeline wis_from_bounds max|Δ| = {wdiff:.2e} (must be ~0)")

    # ---- (2) LEAK AUDIT ----
    print("\n=== (2) LEAK AUDIT (cap FIXED = train-only 2*max(ytr); perturb future y by +1000) ===")
    capL = cap_train
    B_base, med_base, _ = build_combo(S_tw, S_sp, yf0, capL, p_star, HEAD_L, HEAD_THR, HEAD_LAM)
    leak = {}
    for wk in (260, 336):
        yf_p = yf0.copy(); yf_p[wk] += 1000.0
        B_p, _m, _o = build_combo(S_tw, S_sp, yf_p, capL, p_star, HEAD_L, HEAD_THR, HEAD_LAM)
        fc, nchg = first_change_origin(B_base, B_p, origins)
        expected = wk + 1 if wk + 1 <= ntot - 1 else None
        ok = (fc == expected)
        leak[wk] = dict(first_change=fc, expected=expected, n_changed=nchg, ok=bool(ok))
        # bit-identity of all origins strictly before wk+1
        pre = origins < (wk + 1)
        pre_ident = bool(np.array_equal(bounds_stack(B_base, origins)[:, pre],
                                        bounds_stack(B_p, origins)[:, pre]))
        leak[wk]["pre_bitidentical"] = pre_ident
        print(f"  perturb y[{wk}]+1000: first_change_origin={fc} (expected {expected})  "
              f"n_changed_origins={nchg}  pre-window bit-identical={pre_ident}  OK={ok}")

    # selection invariance: perturb a TEST week, val WIS (p* + envelope) must be identical
    yf_t = yf0.copy(); yf_t[260] += 1000.0
    _, p_scores_pert = select_p_star(S_tw, yf_t, capL)
    _, p_scores_base = select_p_star(S_tw, yf0, capL)
    sel_inv = (p_scores_pert == p_scores_base)
    print(f"  pre-T0 val p*-scores invariant to y[260] perturbation: {sel_inv}")
    print(f"    base={p_scores_base}  pert={p_scores_pert}")

    # ---- (3) TRAIN-ONLY CAP ----
    print("\n=== (3) TRAIN-ONLY CAP (combo recomputed with cap=2*max(ytr)) ===")
    m_c_train = full_metrics(B_base, y, med_base, ref_w, n)
    print(f"  combo WIS  cap_full={m_c['wis']}   cap_train={m_c_train['wis']}   "
          f"(train-only same-or-better: {m_c_train['wis'] <= m_c['wis'] + 1e-9})")

    # ---- (4) DM ROBUSTNESS ----
    print("\n=== (4) DM ROBUSTNESS vs exact 2.4012 reference ===")
    diff_c = m_c["w"] - ref_w
    p_hln_c, d_c = dm_hln(m_c["w"], ref_w)
    print(f"  combo vs REF: HLN-DM p={p_hln_c:.3e}  meandiff={d_c:+.4f}")
    boot = {}
    for L in (1, 4, 8, 12):
        boot[L] = block_bootstrap_p(diff_c, L)
    print("  combo vs REF moving-block bootstrap p (L=1/4/8/12): "
          + "  ".join(f"L{L}={boot[L]:.4f}" for L in (1, 4, 8, 12)))
    # combo vs Tweedie (claim: TIE) and vs SPCI (claim: decisive)
    p_ct, d_ct = dm_hln(m_c["w"], m_tw["w"])
    p_cs, d_cs = dm_hln(m_c["w"], m_sp["w"])
    boot_ct = {L: block_bootstrap_p(m_c["w"] - m_tw["w"], L) for L in (1, 4, 8, 12)}
    boot_cs = {L: block_bootstrap_p(m_c["w"] - m_sp["w"], L) for L in (1, 4, 8, 12)}
    print(f"  combo vs TWEEDIE: HLN-DM p={p_ct:.4f} meandiff={d_ct:+.4f}  "
          f"boot p={ {L: round(boot_ct[L],3) for L in (1,4,8,12)} }  (claim: TIE)")
    print(f"  combo vs SPCI   : HLN-DM p={p_cs:.4f} meandiff={d_cs:+.4f}  "
          f"boot p={ {L: round(boot_cs[L],3) for L in (1,4,8,12)} }  (claim: DECISIVE)")

    # ---- (5) COVERAGE ----
    print("\n=== (5) COVERAGE ===")
    print(f"  combo PICP95={m_c['picp95']} ({m_c['k']}/{n})  CP95={m_c['cp95']}  "
          f"in[0.93,0.96]={0.93 <= m_c['picp95'] <= 0.96}")
    print(f"  combo peak PICP95(y>=50)={m_c['peak_picp95']} ({m_c['n_peak']} peaks)  "
          f"Tweedie peak={m_tw['peak_picp95']}  SPCI peak={m_sp['peak_picp95']}")
    print(f"  combo last34 WIS={m_c['last34']}  (<2.72: {m_c['last34'] < 2.72})  "
          f"Tweedie last34={m_tw['last34']}  SPCI last34={m_sp['last34']}")

    # ---- VERDICT ----
    leak_ok = all(v["ok"] and v["pre_bitidentical"] for v in leak.values()) and sel_inv
    dm_robust = (p_hln_c < 0.05 and all(boot[L] < 0.05 for L in (1, 4, 8, 12)))
    combo_beats_tw_wis = m_c["wis"] <= m_tw["wis"] + 1e-9
    combo_beats_sp_wis = m_c["wis"] < m_sp["wis"]
    tie_vs_tw = p_ct >= 0.05
    decisive_vs_sp = p_cs < 0.05
    combo_peak_better_than_both = (m_c["peak_picp95"] > m_tw["peak_picp95"] + 1e-9
                                   and m_c["peak_picp95"] > m_sp["peak_picp95"] + 1e-9)
    genuine_dominance = (combo_beats_tw_wis and combo_beats_sp_wis
                         and combo_peak_better_than_both
                         and 0.93 <= m_c["picp95"] <= 0.96 and dm_robust and leak_ok)

    print("\n" + "=" * 92)
    print("VERDICT")
    print(f"  reproduce_ok           : {repro_ok}")
    print(f"  leak_free (perturb+sel): {leak_ok}")
    print(f"  DM vs ref robust <0.05 : {dm_robust}  (HLN {p_hln_c:.2e}; boot {[round(boot[L],4) for L in (1,4,8,12)]})")
    print(f"  combo WIS <= Tweedie   : {combo_beats_tw_wis}  ({m_c['wis']} vs {m_tw['wis']})")
    print(f"  combo WIS <  SPCI      : {combo_beats_sp_wis}  ({m_c['wis']} vs {m_sp['wis']})")
    print(f"  combo vs Tweedie = TIE : {tie_vs_tw}  (HLN p={p_ct:.3f})")
    print(f"  combo vs SPCI decisive : {decisive_vs_sp}  (HLN p={p_cs:.3f})")
    print(f"  combo peak > BOTH      : {combo_peak_better_than_both}  "
          f"(combo {m_c['peak_picp95']} / tw {m_tw['peak_picp95']} / sp {m_sp['peak_picp95']})")
    print(f"  >>> GENUINE DOMINANCE OVER BOTH (incl. peak) : {genuine_dominance}")
    print("=" * 92)

    out = dict(
        reference_wis_indep=round(ref_mean, 4), p_star=p_star, p_scores=p_scores,
        wis_indep_vs_pipeline_maxdiff=wdiff,
        standalones=dict(tweedie={k: v for k, v in m_tw.items() if k != "w"},
                         spci={k: v for k, v in m_sp.items() if k != "w"}),
        combo={k: v for k, v in m_c.items() if k != "w"},
        combo_train_cap_wis=m_c_train["wis"],
        reproduce_ok=bool(repro_ok),
        leak_audit=leak, selection_invariant=bool(sel_inv),
        dm=dict(combo_vs_ref_hln=p_hln_c, combo_vs_ref_boot=boot,
                combo_vs_tweedie_hln=p_ct, combo_vs_tweedie_meandiff=d_ct, combo_vs_tweedie_boot=boot_ct,
                combo_vs_spci_hln=p_cs, combo_vs_spci_meandiff=d_cs, combo_vs_spci_boot=boot_cs),
        verdict=dict(reproduce_ok=bool(repro_ok), leak_free=bool(leak_ok),
                     dm_vs_ref_robust=bool(dm_robust),
                     combo_wis_le_tweedie=bool(combo_beats_tw_wis),
                     combo_wis_lt_spci=bool(combo_beats_sp_wis),
                     tie_vs_tweedie=bool(tie_vs_tw), decisive_vs_spci=bool(decisive_vs_sp),
                     combo_peak_better_than_both=bool(combo_peak_better_than_both),
                     genuine_dominance_over_both=bool(genuine_dominance)),
        elapsed_sec=round(time.time() - t0, 1),
    )
    (ROOT / "scripts" / "_exp_combo_verify.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote scripts/_exp_combo_verify.json  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
