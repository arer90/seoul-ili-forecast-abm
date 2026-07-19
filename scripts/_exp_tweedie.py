#!/usr/bin/env python
"""OVERNIGHT EXPERIMENT — distributional (Tweedie) head for overdispersed ILI.

Goal: a COHERENT count/rate distribution around the TiRex point, whose FLUSIGHT
quantiles handle peaks better than the stitched pinball-GBM+CQR head (which
over-widened to PICP95 0.985 at WIS 2.2765, DM p=0.0572 vs the 2.4012 fair base).

y here is a CONTINUOUS non-negative ILI rate (0.81..100.7, non-integer), so a
discrete NegBin is a distributional mismatch; the correct member of the
exponential-dispersion family for continuous non-negative overdispersed data with
a point mass at 0 is the TWEEDIE (1<p<2, compound Poisson-Gamma). We fit the mean
three ways, all TiRex-anchored via a log-offset/init_score so the head models the
dispersion + residual structure AROUND TiRex:

  mean_source:
    tirex : mu = TiRex point (no learned mean correction)
    lgb   : LightGBM objective='tweedie' (var_power=p), init_score=log(TiRex)
            -> mu = TiRex * exp(f(x))   (flexible multiplicative correction)
    glm   : statsmodels Tweedie GLM (log link, var_power=p), offset=log(TiRex)
            -> mu = TiRex * exp(x'beta) (linear correction)

  quantile method (from the fitted predictive distribution):
    gamma   : Tweedie(1<p<2) positive part ~ Gamma; match mean=mu, var=phi*mu^p
              -> shape=mu^(2-p)/phi, scale=phi*mu^(p-1); q=Gamma.ppf(FQ) (closed form)
    pearson : residual-scale model using the Tweedie variance function.
              z=(y-mu)/mu^(p/2) empirical (past) quantiles Qz -> q=mu+Qz*mu^(p/2)
              (heteroscedastic, widens as mu^(p/2) at peaks; nonparametric shape)

Then CONFORMALIZE with the SAME CQR machinery as the fair baseline: per-alpha offset
Q_a from a calibration seed on weeks [165,205) (K_CAL=40), applied statically to the
132 test origins. Leak-free: per-block GBM/GLM/Qz/phi all trained on strictly PAST
weeks (train_end = block_start - K_CAL); refit every REFIT_K=4 origins; cap=2*max(y).

Selection is honest & pre-T0: the reported headline config is the argmin-WIS config on
a PAST validation window [165,205) (CQR sub-seed [125,165)) — never the 132 test origins.
The full sweep is printed for transparency. Reference = the exact 2.4012 fair baseline
(tirex_empirical_qy + build_bounds_cqr); every candidate's per-origin WIS is DM-tested
(HLN h=1) against it. No live/pipeline or dec_boosted_mech*.py edits.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
import lightgbm as lgb
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D  # noqa: F401  (TIREX_CACHE path)
from scripts.dec_boosted_mech import (cqr_offsets, build_bounds_cqr,
                                       FQ, MED_COL, MIN_CTX, K_CAL)
from scripts.dec_boosted_mech_multiorigin import T0, REFIT_K
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup, dm, cp, wis_of

EPS = 1e-6
WSTART = 125                 # earliest week we build quantiles for (val sub-seed start)
P_GRID = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
MEAN_SOURCES = ("tirex", "lgb", "glm")
QMETHODS = ("gamma", "pearson")
# validation window (pre-T0, honest selection): origins [165,205), sub-seed [125,165)
VAL_LO, VAL_HI = T0 - K_CAL, T0            # [165,205)
VAL_CAL = np.arange(VAL_LO - K_CAL, VAL_LO)  # [125,165)


# ───────────────────────────── mean models ──────────────────────────────
def fit_lgb(Xtr, ytr, base_tr, p):
    init = np.log(np.clip(base_tr, EPS, None))
    ds = lgb.Dataset(Xtr, label=ytr, init_score=init, free_raw_data=False)
    params = dict(objective="tweedie", tweedie_variance_power=float(p),
                  learning_rate=0.05, num_leaves=8, min_data_in_leaf=15,
                  max_depth=3, lambda_l2=1.0, verbosity=-1, num_threads=2,
                  seed=42, feature_pre_filter=False)
    return lgb.train(params, ds, num_boost_round=300)


def predmu_lgb(m, X, base):
    raw = np.asarray(m.predict(X, raw_score=True), dtype=float)   # margin f(x), no init
    return np.exp(np.log(np.clip(base, EPS, None)) + raw)


def fit_glm(Xtr, ytr, base_tr, p):
    Xd = sm.add_constant(Xtr, has_constant="add")
    off = np.log(np.clip(base_tr, EPS, None))
    try:
        fam = sm.families.Tweedie(link=sm.families.links.Log(), var_power=float(p))
        res = sm.GLM(ytr, Xd, family=fam, offset=off).fit(maxiter=100)
        if not np.all(np.isfinite(np.asarray(res.params, dtype=float))):
            return None
        return res
    except Exception:
        return None


def predmu_glm(res, X, base):
    Xd = sm.add_constant(np.atleast_2d(X), has_constant="add")
    off = np.log(np.clip(np.atleast_1d(base), EPS, None))
    return np.asarray(res.predict(Xd, offset=off), dtype=float)


# ───────────────────────── build quantiles over a span ──────────────────
def build_span(S, p, mean_source):
    """Per-block leak-free FLUSIGHT quantiles for weeks [WSTART, ntot).

    Returns (QG, QP): (Wspan, 23) gamma-parametric and pearson-residual-scale
    quantile matrices, indexed by week-WSTART. Every block trains only on
    weeks [MIN_CTX, block_start - K_CAL) (strictly past).
    """
    feat, tirex, cap, yf, ntot = S["feat"], S["tirex"], S["cap"], S["yf"], S["ntot"]
    W = np.arange(WSTART, ntot)
    QG = np.zeros((len(W), len(FQ)))
    QP = np.zeros((len(W), len(FQ)))
    glm_fail = 0
    for bstart in range(WSTART, ntot, REFIT_K):
        bend = min(bstart + REFIT_K, ntot)
        train_end = bstart - K_CAL
        tr = np.arange(MIN_CTX, train_end)
        Xtr, ytr, base_tr = feat[tr], yf[tr], tirex[tr]

        if mean_source == "tirex":
            mu_tr = base_tr

            def pred(idx):
                return tirex[idx]
        elif mean_source == "lgb":
            m = fit_lgb(Xtr, ytr, base_tr, p)
            mu_tr = predmu_lgb(m, Xtr, base_tr)

            def pred(idx, _m=m):
                return predmu_lgb(_m, feat[idx], tirex[idx])
        else:  # glm
            res = fit_glm(Xtr, ytr, base_tr, p)
            if res is None:
                glm_fail += 1
                mu_tr = base_tr

                def pred(idx):
                    return tirex[idx]
            else:
                mu_tr = predmu_glm(res, Xtr, base_tr)

                def pred(idx, _r=res):
                    return predmu_glm(_r, feat[idx], tirex[idx])

        mu_tr_c = np.clip(mu_tr, EPS, None)
        # Pearson dispersion phi = mean( (y-mu)^2 / mu^p )  (Tweedie variance function)
        phi = float(np.mean((ytr - mu_tr) ** 2 / np.power(mu_tr_c, p)))
        phi = max(phi, 1e-6)
        s_tr = np.power(mu_tr_c, p / 2.0)
        z = (ytr - mu_tr) / s_tr
        Qz = np.quantile(z, FQ)

        oi = np.arange(bstart, bend)
        mu_o = np.clip(pred(oi), EPS, None)
        # gamma parametric (Tweedie positive-part match)
        shape = np.power(mu_o, 2.0 - p) / phi
        scale = phi * np.power(mu_o, p - 1.0)
        g = stats.gamma.ppf(FQ[None, :], a=shape[:, None], scale=scale[:, None])
        g = np.clip(g, 0.0, cap)
        g.sort(axis=1)
        # pearson residual-scale (heteroscedastic mu^(p/2))
        s_o = np.power(mu_o, p / 2.0)
        pe = mu_o[:, None] + Qz[None, :] * s_o[:, None]
        pe = np.clip(pe, 0.0, cap)
        pe.sort(axis=1)

        sel = oi - WSTART
        QG[sel] = g
        QP[sel] = pe
    return QG, QP, glm_fail


# ───────────────────────────── evaluation ───────────────────────────────
def eval_wis(Q, origins, cal, S):
    cap = S["cap"]
    qy = Q[origins - WSTART]
    qy_cal = Q[cal - WSTART]
    cqr = cqr_offsets(qy_cal, S["yf"][cal])
    B = build_bounds_cqr(qy, cqr, cap)
    med = qy[:, MED_COL]
    y = S["yf"][origins]
    w = wis_of(B, y, med)
    return w, B


def metrics(w, B, y, n, ref_wis):
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    p_dm, dbar = dm(w, ref_wis)
    return dict(wis=round(float(w.mean()), 4), dm_p=round(float(p_dm), 4),
                picp95=round(k / n, 4), k=k, cp=cp(k, n),
                w95=round(float((hi95 - lo95).mean()), 2),
                last34=round(float(w[n - 34:].mean()), 4))


def main():
    t0 = time.time()
    S = setup()
    ntot = S["ntot"]
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S["yf"][origins]
    cal = np.arange(T0 - K_CAL, T0)          # test CQR seed [165,205)
    r_full = S["yf"] - S["tirex"]

    # reference fair baseline (2.4012)
    qy_ref = tirex_empirical_qy(S["tirex"], r_full, origins, S["cap"])
    qy_ref_cal = tirex_empirical_qy(S["tirex"], r_full, cal, S["cap"])
    cqr_ref = cqr_offsets(qy_ref_cal, S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, S["cap"])
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())

    val_origins = np.arange(VAL_LO, VAL_HI)
    y_val = S["yf"][val_origins]

    print(f"=== Tweedie distributional head — {n} leak-free origins (weeks {T0}..{ntot-1}) ===")
    print(f"    REFERENCE fair baseline TiRex+CQR: WIS={ref_mean:.4f}  "
          f"PICP95={(( (y>=ref_B[0.05][0])&(y<=ref_B[0.05][1])).mean()):.4f}  "
          f"last34={float(ref_wis[n-34:].mean()):.4f}")
    print(f"    TARGET: WIS<{ref_mean:.4f} & DM p<0.05 & PICP95 in [0.93,0.96] & last34<2.72\n")

    hdr = (f"{'config':>22s} | {'WIS':>7s} {'DMp':>7s} {'d%':>6s} {'PICP95':>7s} "
           f"{'k/N':>7s} {'CP95ci':>15s} {'W95':>6s} {'l34':>7s} | {'valWIS':>7s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for ms in MEAN_SOURCES:
        for p in P_GRID:
            QG, QP, gfail = build_span(S, p, ms)
            for qm, Q in (("gamma", QG), ("pearson", QP)):
                # test
                w, B = eval_wis(Q, origins, cal, S)
                m = metrics(w, B, y, n, ref_wis)
                # honest pre-T0 validation WIS
                wv, _ = eval_wis(Q, val_origins, VAL_CAL, S)
                m["val_wis"] = round(float(wv.mean()), 4)
                name = f"{ms}_{qm}_p{p}"
                m["config"] = name
                m["glm_fail"] = gfail if ms == "glm" else 0
                rows.append(m)
                sig = "*" if (m["wis"] < ref_mean and m["dm_p"] < 0.05) else " "
                calm = "✓" if 0.93 <= m["picp95"] <= 0.96 else " "
                dpct = 100 * (m["wis"] - ref_mean) / ref_mean
                print(f"{name:>22s} | {m['wis']:>7.4f}{sig} {m['dm_p']:>7.4f} {dpct:>6.1f} "
                      f"{m['picp95']:>6.4f}{calm} {str(m['k'])+'/'+str(n):>7s} "
                      f"{str(m['cp']):>15s} {m['w95']:>6.2f} {m['last34']:>7.4f} | {m['val_wis']:>7.4f}")

    # ---- honest headline: argmin VAL WIS (pre-T0, never test) ----
    headline = min(rows, key=lambda r: r["val_wis"])
    # transparency: post-hoc best test WIS meeting all constraints
    ok = [r for r in rows if r["wis"] < ref_mean and r["dm_p"] < 0.05
          and 0.93 <= r["picp95"] <= 0.96 and r["last34"] < 2.72]
    best_test = min(ok, key=lambda r: r["wis"]) if ok else None
    sig_only = [r for r in rows if r["wis"] < ref_mean and r["dm_p"] < 0.05]

    out = {"ref_wis": round(ref_mean, 4), "n": n,
           "headline_val_selected": headline, "constraint_winners": ok,
           "sig_beats_baseline": [r["config"] for r in sig_only], "rows": rows}
    (ROOT / "scripts" / "_exp_tweedie.json").write_text(json.dumps(out, indent=2))

    print("\n--- HONEST headline (argmin pre-T0 val WIS, weeks 165..204) ---")
    h = headline
    print(f"    {h['config']}: TEST WIS={h['wis']} (DM p={h['dm_p']}) PICP95={h['picp95']} "
          f"{h['cp']} last34={h['last34']} valWIS={h['val_wis']}")
    print(f"    beats & significant & calibrated & last34<2.72: "
          f"{bool(h['wis']<ref_mean and h['dm_p']<0.05 and 0.93<=h['picp95']<=0.96 and h['last34']<2.72)}")
    print(f"\n--- configs beating 2.4012 with DM p<0.05 (any coverage): "
          f"{[r['config'] for r in sig_only] or 'NONE'}")
    print(f"--- ALL constraints (WIS<ref & DMp<0.05 & PICP95∈[0.93,0.96] & last34<2.72): "
          f"{[r['config'] for r in ok] or 'NONE'}")
    if best_test:
        print(f"    post-hoc best (transparency, NOT the honest pick): {best_test['config']} "
              f"WIS={best_test['wis']} DMp={best_test['dm_p']} PICP95={best_test['picp95']} "
              f"last34={best_test['last34']}")
    print(f"\nelapsed {time.time()-t0:.1f}s  wrote scripts/_exp_tweedie.json")


if __name__ == "__main__":
    raise SystemExit(main())
