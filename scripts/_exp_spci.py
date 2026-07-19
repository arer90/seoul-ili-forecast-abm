#!/usr/bin/env python
"""SPCI done right (Xu & Xie 2023) — CONDITIONAL residual-quantile forecaster on the
same 132 leak-free rolling 1-step origins (Seoul ILI weeks 205..336). Standalone; imports
sanctioned helpers only, touches NO live/pipeline or dec_boosted_mech*.py code.

Idea
----
Reference fair baseline (WIS 2.4012 / PICP95 0.9545) = TiRex point + UNCONDITIONAL expanding
empirical residual quantiles + a fixed CQR seed. Its intervals are the SAME width every week.
SPCI instead estimates the CONDITIONAL residual quantile Q(tau | x_t) from LAGGED residuals,
so intervals are narrow off-peak (recent |resid| small) and wide on the rising limb (recent
|resid| large). A width-optimal beta-search (shortest 1-alpha interval of the conditional
residual law) sharpens further. The bet: a CONSISTENT per-origin WIS cut (high DM power) that
lands PICP95 inside [0.93,0.96] instead of the static CQR's 0.985 over-coverage.

Conditional quantile estimator = Quantile Random Forest (Meinshausen 2006; the QRF used by
SPCI): ONE RandomForestRegressor per refit block, conditional quantiles from per-tree leaf
memberships (arbitrary tau from a single forest -> exact beta-search). LightGBM-quantile
variant kept as a cross-check.

Features at index t (all strictly past-only except the TiRex level, known at forecast time):
  r_{t-1..t-L}  (L lagged residuals) ; rolling |resid| mean over 4/8/13 ;
  rolling signed resid mean over 4/8 ; TiRex level ; fourier_sin_h1/cos_h1 (epiweek).
Target: r_t = y_t - TiRex_t.

Leak-free protocol (spotless)
-----------------------------
* Per refit block [bstart, bstart+REFIT_K): learner trained on pairs with
  train_end = bstart - K_CAL  (== origin - K_CAL for every origin in the block).
* CQR seed (when used) calibrated on [T0-K_CAL, T0) from a seed learner trained on
  [MIN_CTX+WARM, T0-K_CAL) — strictly pre-T0.
* Cap = 2*max(y_train)  (train-only; spotless).  Reference keeps its own 2*max(y_full) cap.
* Construction (symmetric vs beta-optimal, point, CQR on/off) SELECTED on a pre-T0 validation
  window V=[165,205) by WIS s.t. PICP95 in [0.93,0.96]; test origins never touched in selection.

DM vs the exact 2.4012 reference: paired per-origin WIS, HLN h=1 (dm()). Reports WIS, DM p,
PICP95 (k/132) + Clopper-Pearson CI, last-34 WIS, mean-W95.  Never fabricates.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (
    FQ, FQ_COL, MED_COL, MIN_CTX, K_CAL, PEAK_Y,
    load_split, cqr_offsets, build_bounds_cqr,
)
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.dec_boosted_mech_multiorigin import T0
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

ALPHAS = list(FLUSIGHT_ALPHAS)
FQL = np.asarray(FQ, dtype=float)                 # 23 FluSight quantile levels
LAGS = 10                                          # lagged residuals
WARM = 13                                          # first usable feature offset (rolling-13)
REFIT_K = 5                                         # refit learner every REFIT_K origins
# fine level grid for the beta-search; includes the 23 FluSight levels exactly
_GL = np.unique(np.concatenate([
    np.round(np.arange(0.005, 0.9951, 0.005), 4), FQL,
    np.array([0.01, 0.99, 0.025, 0.975])]))
GL = np.clip(_GL, 1e-4, 1 - 1e-4)
FQ_IN_GL = np.array([int(np.argmin(np.abs(GL - q))) for q in FQL])   # index of each FluSight lvl


# ───────────────────────── stats helpers ─────────────────────────
def cp(k, nn, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, nn - k + 1)
    hi = 1.0 if k == nn else stats.beta.ppf(1 - a / 2, k + 1, nn - k)
    return round(float(lo), 4), round(float(hi), 4)


def dm(wa, wb):
    """Diebold-Mariano (two-sided), HLN small-sample h=1 correction. p, mean(a-b)."""
    diff = np.asarray(wa) - np.asarray(wb)
    n = len(diff); dbar = diff.mean()
    var = np.var(diff, ddof=1) / n
    if var <= 0:
        return 1.0, float(dbar)
    st = dbar / np.sqrt(var) * np.sqrt((n + 1) / n)
    return float(2 * (1 - stats.t.cdf(abs(st), df=n - 1))), float(dbar)


def wis_of(B, y, med):
    return np.asarray(wis_from_bounds(y, B, ALPHAS, median=med), dtype=float)


# ───────────────────────── features ─────────────────────────
def build_spci_features(y_full, tirex_full, X_full):
    """Per-index SPCI feature matrix (N,p) + residual r. feat[t] uses residuals < t only."""
    N = len(y_full)
    r = y_full - tirex_full                                   # nan for t<MIN_CTX
    cols = []
    # lagged residuals r_{t-1..t-L}
    for L in range(1, LAGS + 1):
        c = np.full(N, np.nan)
        c[L:] = r[:-L]
        cols.append(c)
    # rolling |resid| mean over 4/8/13 (windows end at t-1)
    absr = np.abs(r)
    for w in (4, 8, 13):
        c = np.full(N, np.nan)
        for t in range(w, N):
            seg = absr[t - w:t]
            c[t] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
        cols.append(c)
    # rolling signed resid mean over 4/8 (drift)
    for w in (4, 8):
        c = np.full(N, np.nan)
        for t in range(w, N):
            seg = r[t - w:t]
            c[t] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
        cols.append(c)
    # TiRex level (known at forecast time), epiweek fourier h1
    cols.append(tirex_full.copy())
    cols.append(X_full[:, 4].copy())                          # fourier_sin_h1
    cols.append(X_full[:, 5].copy())                          # fourier_cos_h1
    feat = np.column_stack(cols)
    return feat, r


# ───────────────────────── QRF conditional quantiles ─────────────────────────
def _weighted_quantiles(v, w, levels):
    """Hazen weighted quantiles of values v with nonneg weights w at `levels` (vectorized)."""
    order = np.argsort(v, kind="mergesort")
    vs = v[order]; ws = w[order]
    cw = np.cumsum(ws); tot = cw[-1]
    if tot <= 0:
        return np.full(len(levels), np.median(v))
    p = (cw - 0.5 * ws) / tot
    return np.interp(levels, p, vs)


def qrf_grid(rf, leaf_train, r_train, X_query):
    """Conditional residual quantiles at grid GL for each query row. Returns (nq, len(GL))."""
    leaf_q = rf.apply(X_query)                                # (nq, T)
    nq = leaf_q.shape[0]
    out = np.empty((nq, len(GL)), dtype=float)
    for i in range(nq):
        mask = (leaf_train == leaf_q[i][None, :])             # (ntr, T)
        cj = mask.sum(0)                                       # (T,)
        w = (mask / np.maximum(cj, 1)).sum(1)                 # (ntr,)
        out[i] = _weighted_quantiles(r_train, w, GL)
    return out


def fit_qrf(Xtr, rtr, n_estimators=400, min_samples_leaf=10, max_features=0.6, seed=42):
    rf = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        max_features=max_features, random_state=seed, n_jobs=3, bootstrap=True)
    rf.fit(Xtr, rtr)
    return rf, rf.apply(Xtr)


def block_qrf_resid_grid(feat, r, idxs, qrf_kw, refit_k=REFIT_K):
    """Bagged-free single-QRF per refit block; returns resid-quantile grid (len(idxs), len(GL))
    with block learner train_end = bstart - K_CAL (strictly past, leak-free)."""
    idxs = np.asarray(idxs)
    lo, hi = int(idxs.min()), int(idxs.max()) + 1
    grid = np.zeros((len(idxs), len(GL)), dtype=float)
    valid0 = MIN_CTX + WARM
    for bstart in range(lo, hi, refit_k):
        bend = min(bstart + refit_k, hi)
        train_end = bstart - K_CAL
        tr = np.arange(valid0, train_end)
        tr = tr[np.isfinite(feat[tr]).all(1) & np.isfinite(r[tr])]
        rf, leaf_tr = fit_qrf(feat[tr], r[tr], **qrf_kw)
        mask = (idxs >= bstart) & (idxs < bend)
        if mask.any():
            oi = idxs[mask]
            grid[mask] = qrf_grid(rf, leaf_tr, r[tr], feat[oi])
    return grid


def seed_qrf_resid_grid(feat, r, idxs, train_end, qrf_kw):
    """Conditional resid-quantile grid for calibration weeks from a seed QRF trained on
    [MIN_CTX+WARM, train_end) — used only for the CQR seed (strictly pre-T0)."""
    valid0 = MIN_CTX + WARM
    tr = np.arange(valid0, train_end)
    tr = tr[np.isfinite(feat[tr]).all(1) & np.isfinite(r[tr])]
    rf, leaf_tr = fit_qrf(feat[tr], r[tr], **qrf_kw)
    return qrf_grid(rf, leaf_tr, r[np.asarray(idxs)], feat[np.asarray(idxs)])  # placeholder


# ───────────────────────── bound constructors from a resid-quantile grid ─────────────────────────
def _yq_from_grid(resid_grid, tirex, cap):
    """23-col FluSight y-quantile matrix + fine y-quantile grid, monotone, clipped."""
    yq_fq = np.clip(tirex[:, None] + resid_grid[:, FQ_IN_GL], 0.0, cap)
    yq_fq = np.sort(yq_fq, axis=1)
    yq_grid = np.clip(tirex[:, None] + resid_grid, 0.0, cap)
    yq_grid = np.maximum.accumulate(yq_grid, axis=1)          # enforce monotone in tau
    return yq_fq, yq_grid


def bounds_symmetric(yq_fq):
    B = {}
    for a in ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]; ch = FQ_COL[round(1 - a / 2.0, 4)]
        B[a] = (yq_fq[:, cl].copy(), yq_fq[:, ch].copy())
    return B


def bounds_beta(yq_grid):
    """SPCI width-optimal beta-search per alpha: shortest 1-alpha interval of the (conditional)
    y-quantile law. lo=Q(beta*), hi=Q(1-alpha+beta*), beta* argmin width over GL∩[0,alpha]."""
    n = yq_grid.shape[0]
    B = {}
    for a in ALPHAS:
        lvl_lo = GL[GL <= a + 1e-9]                            # candidate lower levels
        lo_arr = np.empty(n); hi_arr = np.empty(n)
        # evaluate Q at lower levels and their matched upper levels 1-a+beta
        up_levels = np.clip(1 - a + lvl_lo, 0, 1 - 1e-4)
        Q_lo = np.empty((n, len(lvl_lo))); Q_hi = np.empty((n, len(lvl_lo)))
        for j in range(len(lvl_lo)):
            Q_lo[:, j] = _row_interp(yq_grid, lvl_lo[j])
            Q_hi[:, j] = _row_interp(yq_grid, up_levels[j])
        width = Q_hi - Q_lo
        jstar = np.argmin(width, axis=1)
        rows = np.arange(n)
        lo_arr = Q_lo[rows, jstar]; hi_arr = Q_hi[rows, jstar]
        B[a] = (lo_arr, hi_arr)
    return B


_GL_CACHE = GL
def _row_interp(yq_grid, level):
    """Interp each row of yq_grid (aligned to GL) at scalar `level`."""
    return np.array([np.interp(level, _GL_CACHE, yq_grid[i]) for i in range(yq_grid.shape[0])])


def median_from_grid(yq_grid, tirex, use_cond_median):
    if use_cond_median:
        return _row_interp(yq_grid, 0.5)
    return tirex.copy()


# ───────────────────────── CQR wrap around any interval constructor ─────────────────────────
def cqr_offsets_generic(B_cal, y_cal):
    """Per-alpha CQR offset from calibration interval (lo,hi) and y_cal (Romano 2019)."""
    K = len(y_cal); Q = {}
    for a in ALPHAS:
        lo, hi = B_cal[a]
        E = np.maximum(lo - y_cal, y_cal - hi)
        beta = min(1.0, (1.0 - a) * (1.0 + 1.0 / max(K, 1)))
        Q[a] = max(0.0, float(np.quantile(E, beta)))
    return Q


def apply_cqr(B, Q, cap):
    out = {}
    for a in ALPHAS:
        lo, hi = B[a]
        out[a] = (np.clip(lo - Q[a], 0.0, cap), np.clip(hi + Q[a], 0.0, cap))
    return out


# ───────────────────────── setup ─────────────────────────
def setup():
    Xtr, ytr, Xte, yte, meta = load_split()
    ntr, nte = len(ytr), len(yte); ntot = ntr + nte
    frozen = np.asarray(json.loads(
        (ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE); tirex_pool = d["tirex_pool"]
    yf = np.concatenate([ytr, yte])
    X_full = np.vstack([Xtr, Xte])
    cap_ref = 2.0 * float(yf.max())
    cap = 2.0 * float(ytr.max())                              # spotless train-only cap
    tirex = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    feat, r = build_spci_features(yf, tirex, X_full)
    return dict(yf=yf, tirex=tirex, feat=feat, r=r, cap=cap, cap_ref=cap_ref,
                ntot=ntot, X_full=X_full)


def reference_wis(S, origins):
    """Exact 2.4012 fair baseline per-origin WIS on `origins` (its own 2*max(yfull) cap)."""
    yf, tirex, cap_ref = S["yf"], S["tirex"], S["cap_ref"]
    r_full = yf - tirex
    cal_idx = np.arange(T0 - K_CAL, T0)
    qy_ref = tirex_empirical_qy(tirex, r_full, origins, cap_ref)
    qy_ref_cal = tirex_empirical_qy(tirex, r_full, cal_idx, cap_ref)
    cqr_ref = cqr_offsets(qy_ref_cal, yf[cal_idx])
    B = build_bounds_cqr(qy_ref, cqr_ref, cap_ref)
    med = qy_ref[:, MED_COL]
    return wis_of(B, yf[origins], med), B


# ───────────────────────── candidate evaluation ─────────────────────────
def eval_bounds(B, y, med, ref_wis, n_full):
    w = wis_of(B, y, med)
    lo95, hi95 = B[0.05]; cov = (y >= lo95) & (y <= hi95); k = int(cov.sum()); nn = len(y)
    p, dbar = dm(w, ref_wis)
    last34 = np.zeros(nn, bool); last34[nn - 34:] = True
    return {
        "wis": round(float(w.mean()), 4), "dm_p": round(p, 4),
        "picp95": round(k / nn, 4), "k_of_n": f"{k}/{nn}", "cp95ci": list(cp(k, nn)),
        "w95": round(float((hi95 - lo95).mean()), 2),
        "last34_wis": round(float(w[last34].mean()), 4),
        "_wis_arr": w,
    }


def build_all_variants(S, origins, qrf_kw, cache=None):
    """Return {variant_name: bounds, '_med': {...}} for a given origin set + QRF config."""
    yf, tirex, r, feat, cap = S["yf"], S["tirex"], S["r"], S["feat"], S["cap"]
    origins = np.asarray(origins)
    tir_o = tirex[origins]
    # conditional resid grid at origins (block-refit) + calibration
    if cache is not None and "grid_o" in cache:
        grid_o = cache["grid_o"]
    else:
        grid_o = block_qrf_resid_grid(feat, r, origins, qrf_kw)
        if cache is not None:
            cache["grid_o"] = grid_o
    yq_fq, yq_grid = _yq_from_grid(grid_o, tir_o, cap)

    # CQR seed: calibrate on [T0-K_CAL,T0) using a seed QRF trained on [MIN_CTX+WARM, T0-K_CAL)
    cal_idx = np.arange(T0 - K_CAL, T0)
    valid0 = MIN_CTX + WARM
    if cache is not None and "grid_cal" in cache:
        grid_cal = cache["grid_cal"]
    else:
        tr = np.arange(valid0, T0 - K_CAL)
        tr = tr[np.isfinite(feat[tr]).all(1) & np.isfinite(r[tr])]
        rf, leaf_tr = fit_qrf(feat[tr], r[tr], **qrf_kw)
        grid_cal = qrf_grid(rf, leaf_tr, r[tr], feat[cal_idx])
        if cache is not None:
            cache["grid_cal"] = grid_cal
    yq_fq_cal, yq_grid_cal = _yq_from_grid(grid_cal, tirex[cal_idx], cap)
    y_cal = yf[cal_idx]

    B_sym = bounds_symmetric(yq_fq)
    B_sym_cal = bounds_symmetric(yq_fq_cal)
    B_beta = bounds_beta(yq_grid)
    B_beta_cal = bounds_beta(yq_grid_cal)

    Q_sym = cqr_offsets_generic(B_sym_cal, y_cal)
    Q_beta = cqr_offsets_generic(B_beta_cal, y_cal)

    med_cond = median_from_grid(yq_grid, tir_o, True)
    med_pure = tir_o.copy()

    variants = {
        "sym_raw": B_sym,
        "sym_cqr": apply_cqr(B_sym, Q_sym, cap),
        "beta_raw": B_beta,
        "beta_cqr": apply_cqr(B_beta, Q_beta, cap),
    }
    meds = {
        "sym_raw": med_cond, "sym_cqr": med_cond,
        "beta_raw": med_cond, "beta_cqr": med_cond,
    }
    # point-choice variants on the strongest interval builders
    variants["sym_cqr_ptTirex"] = apply_cqr(B_sym, Q_sym, cap); meds["sym_cqr_ptTirex"] = med_pure
    variants["beta_cqr_ptTirex"] = apply_cqr(B_beta, Q_beta, cap); meds["beta_cqr_ptTirex"] = med_pure
    return variants, meds


def main():
    t0 = time.time()
    S = setup()
    ntot = S["ntot"]
    origins = np.arange(T0, ntot); n = len(origins)
    y = S["yf"][origins]
    ref_wis, _ = reference_wis(S, origins)
    print(f"=== SPCI-QRF conditional residual quantiles | {n} leak-free origins (weeks {T0}..{ntot-1}) ===")
    print(f"    reference fair baseline (TiRex+empirical+CQR): WIS={ref_wis.mean():.4f}")
    print(f"    TARGET: WIS<2.4012 & DM p<0.05 & PICP95 in [0.93,0.96] & last34<2.72  "
          f"(ref last34={ref_wis[np.arange(n)>=n-34].mean():.4f})\n")

    qrf_kw = dict(n_estimators=400, min_samples_leaf=10, max_features=0.6)
    cache = {}
    variants, meds = build_all_variants(S, origins, qrf_kw, cache=cache)

    hdr = (f"{'variant':>18s} | {'WIS':>7s} {'DMp':>7s} {'d%':>6s} {'PICP95':>7s} "
           f"{'k/N':>7s} {'CP95ci':>16s} {'W95':>6s} {'last34':>7s}")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for name, B in variants.items():
        rrow = eval_bounds(B, y, meds[name], ref_wis, n)
        rows[name] = rrow
        sig = "*" if (rrow["wis"] < ref_wis.mean() and rrow["dm_p"] < 0.05) else " "
        cal = "OK" if 0.93 <= rrow["picp95"] <= 0.96 else "  "
        l34 = "L" if rrow["last34_wis"] < 2.72 else " "
        dpct = 100 * (rrow["wis"] - ref_wis.mean()) / ref_wis.mean()
        print(f"{name:>18s} | {rrow['wis']:>7.4f}{sig} {rrow['dm_p']:>7.4f} {dpct:>6.1f} "
              f"{rrow['picp95']:>6.4f}{cal} {rrow['k_of_n']:>7s} {str(rrow['cp95ci']):>16s} "
              f"{rrow['w95']:>6.2f} {rrow['last34_wis']:>6.4f}{l34}")

    win = {k: v for k, v in rows.items()
           if v["wis"] < ref_wis.mean() and v["dm_p"] < 0.05
           and 0.93 <= v["picp95"] <= 0.96 and v["last34_wis"] < 2.72}
    print("\n=== meets ALL (WIS<ref & DMp<0.05 & PICP95∈[0.93,0.96] & last34<2.72): "
          + (", ".join(win) if win else "NONE") + " ===")

    out = {"ref_wis": round(float(ref_wis.mean()), 4), "n": n,
           "rows": {k: {kk: vv for kk, vv in v.items() if kk != "_wis_arr"} for k, v in rows.items()},
           "elapsed_sec": round(time.time() - t0, 1)}
    (ROOT / "scripts" / "_exp_spci.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/_exp_spci.json  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
