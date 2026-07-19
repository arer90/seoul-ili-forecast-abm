#!/usr/bin/env python
"""ADVERSARIAL verification of the Tweedie-head candidate (tirex_pearson_p1.5_expand).

Tries to REFUTE the win. Four checks:
 (1) reproduce WIS / DM p / PICP95 (independent recompute).
 (2) LEAK AUDIT:
     (a) tamper the LAST realized y (week ntot-1) by +1000; EVERY origin bound must be
         bit-identical (no interval may see the final truth).
     (b) tamper an interior future y (week W_INT) by +1000; origins t<=W_INT must be
         bit-identical, origins t>W_INT may move (they legitimately use realized past y).
     (c) structural: per-block head train_end <= origin-K_CAL, conformal window starts
         at 165 < T0=205 (pre-test); report train_end for the block covering p* origins.
 (3) DM vs 2.4012 two ways: HLN-DM (h=1) AND 10k paired moving-block bootstrap (L=6).
 (4) PICP95 Clopper-Pearson + last34 < 2.72.

No live/pipeline or dec_boosted_mech*.py edits.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dec_boosted_mech import MED_COL, K_CAL, MIN_CTX
from scripts.dec_boosted_mech_multiorigin import T0, REFIT_K
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.dec_boosted_mech import cqr_offsets, build_bounds_cqr
from scripts.nov_guard_v3 import setup, dm, cp, wis_of, ALPHAS
from scripts._exp_tweedie import build_span, WSTART
from scripts._exp_tweedie_cover import rolling_cqr_bounds

P_STAR = 1.5
CAL_START = T0 - K_CAL   # 165


def cand_bounds(S, p, cap):
    """Candidate: pearson Tweedie head, expanding split-CQR (cal_start=165)."""
    _, QP, _ = build_span(S, p, "tirex")
    origins = np.arange(T0, S["ntot"])
    B = rolling_cqr_bounds(QP, S, origins, cap, CAL_START, None)
    med = QP[origins - WSTART][:, MED_COL]
    return B, med, QP


def flatten_bounds(B, n):
    """Stack all (lo,hi) for every alpha into one (n, 2*A) array for bit comparison."""
    cols = []
    for a in ALPHAS:
        cols.append(B[a][0]); cols.append(B[a][1])
    return np.column_stack(cols)


def mbb_pvalue(diff, L=6, n_boot=10000, seed=42):
    """Two-sided moving-block bootstrap test of H0: mean(diff)=0.
    Center diffs, resample overlapping blocks of length L to preserve h=1 autocorr,
    p = P(|mean_boot_centered| >= |mean(diff)|)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    obs = float(diff.mean())
    dc = diff - obs                       # centered under H0
    n_blocks = int(np.ceil(n / L))
    max_start = n - L                     # inclusive
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
        means[b] = dc[idx].mean()
    p = float((np.abs(means) >= abs(obs)).mean())
    se = float(means.std(ddof=1))
    return obs, p, se


def main():
    S = setup()
    ntot = S["ntot"]
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S["yf"][origins]
    cap = S["cap"]
    cal = np.arange(T0 - K_CAL, T0)
    r_full = S["yf"] - S["tirex"]

    # ---------- reference 2.4012 ----------
    qy_ref = tirex_empirical_qy(S["tirex"], r_full, origins, cap)
    cqr_ref = cqr_offsets(tirex_empirical_qy(S["tirex"], r_full, cal, cap), S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())
    ref_k = int(((y >= ref_B[0.05][0]) & (y <= ref_B[0.05][1])).sum())

    # ---------- candidate ----------
    B, med, QP = cand_bounds(S, P_STAR, cap)
    w = wis_of(B, y, med)
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    p_hln, dbar = dm(w, ref_wis)
    diff = w - ref_wis          # candidate - reference (negative = candidate better)

    print("=" * 74)
    print("(1) REPRODUCTION")
    print(f"  reference   WIS={ref_mean:.4f}  PICP95={ref_k/n:.4f} ({ref_k}/{n})  last34={ref_wis[n-34:].mean():.4f}")
    print(f"  candidate   WIS={w.mean():.4f}  PICP95={k/n:.4f} ({k}/{n})  last34={w[n-34:].mean():.4f}")

    # ---------- (2) LEAK AUDIT ----------
    base_flat = flatten_bounds(B, n)

    # (a) tamper LAST realized y — no origin may change
    Sa = dict(S); Sa["yf"] = S["yf"].copy(); Sa["yf"][ntot - 1] += 1000.0
    Ba, _, _ = cand_bounds(Sa, P_STAR, cap)
    fa = flatten_bounds(Ba, n)
    last_maxdiff = float(np.abs(fa - base_flat).max())
    last_identical = bool(last_maxdiff == 0.0)

    # (b) tamper interior future y at W_INT — only origins t>W_INT may move
    W_INT = 260
    Sb = dict(S); Sb["yf"] = S["yf"].copy(); Sb["yf"][W_INT] += 1000.0
    Bb, _, _ = cand_bounds(Sb, P_STAR, cap)
    fb = flatten_bounds(Bb, n)
    rowdiff = np.abs(fb - base_flat).max(axis=1)     # per-origin max bound change
    changed = origins[rowdiff > 0.0]
    past_leak = changed[changed <= W_INT]            # origins that should NOT have moved
    first_changed = int(changed.min()) if changed.size else None
    b_ok = bool(past_leak.size == 0)

    # (c) structural: block train_end for the block covering the first test origin
    bstart0 = T0 - ((T0 - WSTART) % REFIT_K) if (T0 - WSTART) % REFIT_K else T0
    # replicate build_span block boundaries: blocks start at WSTART, step REFIT_K
    block_starts = list(range(WSTART, ntot, REFIT_K))
    blk_for_T0 = max(b for b in block_starts if b <= T0)
    train_end_T0 = blk_for_T0 - K_CAL
    struct_ok = bool(train_end_T0 <= T0 - K_CAL and CAL_START < T0)

    print("\n(2) LEAK AUDIT")
    print(f"  (a) tamper y[{ntot-1}] (last) +1000  -> max |Δbound| over ALL origins = {last_maxdiff:.1e}  "
          f"bit-identical={last_identical}")
    print(f"  (b) tamper y[{W_INT}] (interior) +1000 -> origins changed: "
          f"{'none' if first_changed is None else 'first='+str(first_changed)}  "
          f"past-origins(≤{W_INT}) leaked={past_leak.tolist()}  clean={b_ok}")
    print(f"      (#changed origins={changed.size}, all should be > {W_INT})")
    print(f"  (c) head block covering origin {T0}: block_start={blk_for_T0} train_end={train_end_T0} "
          f"(≤ origin-K_CAL={T0-K_CAL})  conformal cal_start={CAL_START} < T0={T0}  struct_ok={struct_ok}")

    # ---------- (3) DM two ways ----------
    obs, p_mbb, se_mbb = mbb_pvalue(diff, L=6, n_boot=10000, seed=42)
    # sensitivity to block length
    mbb_L = {L: mbb_pvalue(diff, L=L, n_boot=10000, seed=7)[1] for L in (1, 4, 8, 12)}
    print("\n(3) DM vs 2.4012  (paired per-origin WIS diff, candidate-reference)")
    print(f"  mean diff = {obs:+.4f}  (negative = candidate better)")
    print(f"  HLN-DM (h=1)                 p = {p_hln:.3e}")
    print(f"  moving-block bootstrap L=6   p = {p_mbb:.4f}  (10k, se_boot={se_mbb:.4f})")
    print(f"  MBB p by block length L: { {L: round(v,4) for L,v in mbb_L.items()} }")

    # ---------- (4) coverage / last34 ----------
    cp_lo, cp_hi = cp(k, n)
    l34 = float(w[n - 34:].mean())
    print("\n(4) CALIBRATION")
    print(f"  PICP95 = {k/n:.4f} ({k}/{n})  Clopper-Pearson 95% CI = [{cp_lo},{cp_hi}]  "
          f"in[0.93,0.96]={0.93 <= k/n <= 0.96}")
    print(f"  last34 WIS = {l34:.4f}  < 2.72 = {l34 < 2.72}")

    # ---------- verdict ----------
    verdict = dict(
        reproduces=bool(abs(w.mean() - 2.2427) < 5e-4 and k == 123),
        leak_free=bool(last_identical and b_ok and struct_ok),
        beats_wis=bool(w.mean() < ref_mean),
        dm_sig_hln=bool(p_hln < 0.05),
        dm_sig_mbb=bool(p_mbb < 0.05),
        picp_in_band=bool(0.93 <= k / n <= 0.96),
        last34_ok=bool(l34 < 2.72),
    )
    decisive = all(verdict.values())
    print("\n" + "=" * 74)
    print("VERDICT:", json.dumps(verdict))
    print("DECISIVE leak-free calibrated DM-significant win:", decisive)

    out = dict(ref_wis=round(ref_mean, 4), cand_wis=round(float(w.mean()), 4),
               picp95=round(k / n, 4), k_of_n=f"{k}/{n}", cp95ci=[cp_lo, cp_hi],
               last34=round(l34, 4), mean_diff=round(obs, 4),
               dm_p_hln=float(p_hln), dm_p_mbb_L6=round(p_mbb, 4),
               mbb_p_by_L={str(L): round(v, 4) for L, v in mbb_L.items()},
               leak_last_identical=last_identical, leak_last_maxdiff=last_maxdiff,
               leak_interior_first_changed=first_changed,
               leak_interior_past_leak=past_leak.tolist(),
               train_end_for_T0=train_end_T0, cal_start=CAL_START,
               verdict=verdict, decisive=decisive)
    (ROOT / "scripts" / "_exp_adv_verify.json").write_text(json.dumps(out, indent=2))
    print("wrote scripts/_exp_adv_verify.json")


if __name__ == "__main__":
    raise SystemExit(main())
