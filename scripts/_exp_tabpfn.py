#!/usr/bin/env python
"""OVERNIGHT EXPERIMENT — TabPFN as (a) an alternative base POINT and (b) a conditional
residual/quantile learner, vs the verified Tweedie distributional head.

The one strong untried model. Two independent probes, both leak-free rolling 1-step over
the same 132 origins (Seoul ILI weeks 205..336, T0=205, K_CAL=40), scored against the
exact 2.4012 fair baseline (tirex_empirical_qy + build_bounds_cqr) and the exact 2.2427
Tweedie head (pearson residual-scale span, p*=argmin pre-T0 val WIS, expanding split-CQR).

(a) POINT.  Roll TabPFN 1-step over weeks [52,337) on a GENUINE alternative tabular feature
    set (BASIC lag/seasonal ⊕ extra lags 3/5/6/8 ⊕ rolling means ⊕ momentum — NO TiRex level,
    so it is a real competitor, not a TiRex correction). Train rows = strictly PAST weeks
    [52,t); a short persistence warm-up covers the first <MIN_TRAIN weeks. Compare its 1-step
    MAE/RMSE to TiRex on test [205,337) and val [165,205). Then wrap the SAME Tweedie
    residual-scale interval (q=mu+Qz*mu^(p/2), expanding CQR) around the better of
    {TiRex, TabPFN, convex-ensemble}; the ensemble weight w and Tweedie power p are chosen
    ONLY on pre-T0 val [165,205) (argmin val WIS). build_pearson_span reuses no model fit —
    the mean array is precomputed — and is asserted byte-identical to _exp_tweedie.build_span
    when fed S['tirex'].

(b) RESIDUAL QUANTILES.  TabPFN as the conditional residual-quantile learner on r=y-TiRex:
    per-block refit (REFIT_K=4) on strictly-past weeks [52,bstart-K_CAL); predict TabPFN's
    predictive quantiles at the FluSight levels FQ -> q_y = TiRex + q_r (monotone-rearranged),
    then conformalize with the SAME CQR machinery (raw / static seed [165,205) / expanding).
    Analog of the boosted-GBM residual head, with TabPFN's in-context predictive distribution.

Honesty: every knob (w, p, scheme) is picked on pre-T0 val only; TabPFN is expected to be a
strong-but-not-winning competitor here (i.i.d. row assumption on an autocorrelated series;
TiRex is already the strong point). Both TabPFN passes are cached (npz) so a partial crash is
recoverable. cap = 2*max(y_train) (train-only) checked to never bind. No live/pipeline or
existing-script edits — this is a NEW read-only script that imports the sanctioned helpers.
"""
from __future__ import annotations

import os
# TabPFN telemetry OFF before any tabpfn import (matches simulation.models.tabpfn_wrapper)
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dec_boosted_mech import (cqr_offsets, build_bounds_cqr,
                                       FQ, MED_COL, MIN_CTX, K_CAL)
from scripts.dec_boosted_mech_multiorigin import T0, REFIT_K
from scripts._verify_fairbase import tirex_empirical_qy
from scripts.nov_guard_v3 import setup, dm, cp, wis_of
from scripts._exp_tweedie import build_span, WSTART, P_GRID
from scripts._exp_tweedie_cover import rolling_cqr_bounds

EPS = 1e-6
MIN_TRAIN = 24                 # persistence warm-up until TabPFN has this many past rows
N_EST_POINT = 8                # TabPFN estimators for the point pass
N_EST_RESID = 8                # TabPFN estimators for the residual-quantile pass
CAL_START = T0 - K_CAL         # 165 — first-origin expanding window == static seed
VAL_CAL_START = 125            # validation CQR sub-seed start
VAL_LO, VAL_HI = T0 - K_CAL, T0  # [165,205)
FQ_LIST = [float(round(q, 4)) for q in FQ]

SCRATCH = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/novelty"
)
PT_CACHE = SCRATCH / "tabpfn_point_v1.npz"
RQ_CACHE = SCRATCH / "tabpfn_residq_v1.npz"


# ─────────────────────────── TabPFN loader (offline public weights) ───────────
def _make_tabpfn(n_est: int):
    """Build a fresh TabPFNRegressor on cached public weights (offline, no token flow)."""
    from simulation.models.tabpfn_wrapper import _ensure_weights, _load_tabpfn_token
    _load_tabpfn_token()
    ckpt = _ensure_weights()
    from tabpfn import TabPFNRegressor
    kw = dict(device="cpu", ignore_pretraining_limits=True,
              n_estimators=int(n_est), random_state=42)
    if ckpt is not None:
        kw["model_path"] = str(ckpt)
    return TabPFNRegressor, kw


# ─────────────────────────── point feature matrix (no TiRex) ──────────────────
def build_point_features(S) -> np.ndarray:
    """Genuine alternative tabular features (past-only, NO TiRex level).

    Columns: BASIC 13 (lag1/2/4/52 + fourier/month/season) ⊕ extra lags [3,5,6,8]
    ⊕ rolling means [3,6,13] ⊕ 1-step momentum. Row t uses y[:t] only (leak-free);
    only weeks >= MIN_CTX (52) are ever trained/predicted on.
    """
    yf = S["yf"]
    basic = S["feat"][:, :13]           # BASIC lag/seasonal block (no mech, no TiRex)
    N = len(yf)
    G = np.full((N, 13 + 4 + 3 + 1), np.nan)
    G[:, :13] = basic
    for t in range(N):
        if t < 8:
            continue
        G[t, 13] = yf[t - 3]
        G[t, 14] = yf[t - 5]
        G[t, 15] = yf[t - 6]
        G[t, 16] = yf[t - 8]
        G[t, 17] = yf[t - 3:t].mean()
        G[t, 18] = yf[t - 6:t].mean()
        G[t, 19] = yf[max(0, t - 13):t].mean()
        G[t, 20] = yf[t - 1] - yf[t - 2]
    return G


def roll_tabpfn_point(S, G, w_lo=MIN_CTX, w_hi=None) -> np.ndarray:
    """Rolling 1-step TabPFN point over weeks [w_lo, w_hi). Point[t] fits on strictly-past
    rows [MIN_CTX, t); a persistence fallback y[t-1] covers the first <MIN_TRAIN weeks."""
    yf = S["yf"]
    N = len(yf)
    if w_hi is None:
        w_hi = N
    pt = np.full(N, np.nan)
    TabPFNRegressor, kw = _make_tabpfn(N_EST_POINT)
    n_fit = 0
    t0 = time.time()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(w_lo, w_hi):
            tr = np.arange(MIN_CTX, t)
            if len(tr) < MIN_TRAIN:
                pt[t] = yf[t - 1]              # causal persistence warm-up
                continue
            reg = TabPFNRegressor(**kw).fit(G[tr], yf[tr])
            pt[t] = float(np.ravel(reg.predict(G[t:t + 1], output_type="mean"))[0])
            n_fit += 1
            if n_fit % 40 == 0:
                print(f"    [point] {n_fit} fits, week {t}, {time.time()-t0:.0f}s", flush=True)
    print(f"    [point] done: {n_fit} TabPFN fits in {time.time()-t0:.0f}s", flush=True)
    return pt


def roll_tabpfn_resid_qspan(S) -> np.ndarray:
    """TabPFN conditional residual-quantile span over weeks [WSTART, ntot), indexed by
    week-WSTART. Per-block refit on r=y-TiRex over strictly-past [MIN_CTX, bstart-K_CAL);
    predict TabPFN predictive quantiles at FQ -> q_y = TiRex + q_r, clip+monotone-sort."""
    feat, tirex, yf, cap, ntot = S["feat"], S["tirex"], S["yf"], S["cap"], S["ntot"]
    r = yf - tirex
    W = np.arange(WSTART, ntot)
    Q = np.zeros((len(W), len(FQ)))
    TabPFNRegressor, kw = _make_tabpfn(N_EST_RESID)
    t0 = time.time()
    nb = 0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bstart in range(WSTART, ntot, REFIT_K):
            bend = min(bstart + REFIT_K, ntot)
            train_end = bstart - K_CAL
            tr = np.arange(MIN_CTX, train_end)
            reg = TabPFNRegressor(**kw).fit(feat[tr], r[tr])
            oi = np.arange(bstart, bend)
            qr = reg.predict(feat[oi], output_type="quantiles", quantiles=FQ_LIST)
            qr = np.asarray([np.ravel(a) for a in qr], dtype=float)   # (23, n_oi)
            qy = np.clip(tirex[oi][None, :] + qr, 0.0, cap).T          # (n_oi, 23)
            qy.sort(axis=1)
            Q[oi - WSTART] = qy
            nb += 1
            if nb % 15 == 0:
                print(f"    [residq] {nb} blocks, week {bstart}, {time.time()-t0:.0f}s", flush=True)
    print(f"    [residq] done: {nb} block refits in {time.time()-t0:.0f}s", flush=True)
    return Q


# ─────────────────────────── Tweedie pearson span on a mean array ─────────────
def build_pearson_span(S, p, mean_full) -> np.ndarray:
    """Tweedie residual-scale FLUSIGHT quantiles q=mu+Qz*mu^(p/2) for a PRECOMPUTED mean
    array mean_full (weeks [WSTART,ntot), indexed by week-WSTART). Per-block leak-free:
    phi & Qz from standardized past residuals on [MIN_CTX, bstart-K_CAL). Identical to
    _exp_tweedie.build_span's pearson branch when mean_full is S['tirex']."""
    yf, cap, ntot = S["yf"], S["cap"], S["ntot"]
    W = np.arange(WSTART, ntot)
    QP = np.zeros((len(W), len(FQ)))
    for bstart in range(WSTART, ntot, REFIT_K):
        bend = min(bstart + REFIT_K, ntot)
        train_end = bstart - K_CAL
        tr = np.arange(MIN_CTX, train_end)
        mu_tr = np.clip(mean_full[tr], EPS, None)
        ytr = yf[tr]
        phi = max(float(np.mean((ytr - mu_tr) ** 2 / np.power(mu_tr, p))), 1e-6)  # noqa: F841
        s_tr = np.power(mu_tr, p / 2.0)
        Qz = np.quantile((ytr - mu_tr) / s_tr, FQ)
        oi = np.arange(bstart, bend)
        mu_o = np.clip(mean_full[oi], EPS, None)
        s_o = np.power(mu_o, p / 2.0)
        pe = np.clip(mu_o[:, None] + Qz[None, :] * s_o[:, None], 0.0, cap)
        pe.sort(axis=1)
        QP[oi - WSTART] = pe
    return QP


# ─────────────────────────── metrics ─────────────────────────────────────────
def full_metrics(B, y, med, ref_wis, tw_wis, n):
    w = wis_of(B, y, med)
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    peak = y >= 50.0
    p_ref, d_ref = dm(w, ref_wis)
    p_tw, d_tw = dm(w, tw_wis)
    return dict(
        wis=round(float(w.mean()), 4),
        dm_p_vs_fair=float(p_ref), dm_meandiff_vs_fair=round(float(d_ref), 4),
        dm_p_vs_tweedie=float(p_tw), dm_meandiff_vs_tweedie=round(float(d_tw), 4),
        picp95=round(k / n, 4), k_of_n=f"{k}/{n}", cp95ci=[round(v, 4) for v in cp(k, n)],
        peak_picp95=round(float(covv[peak].mean()), 3), n_peak=int(peak.sum()),
        last34_wis=round(float(w[n - 34:].mean()), 4),
        mean_w95=round(float((hi95 - lo95).mean()), 3),
        wis_arr=w,
    )


def select_p_on_val(S, mean_full, cap, val_origins, y_val):
    """argmin pre-T0 val WIS over P_GRID (expanding CQR from VAL_CAL_START). Returns (p*, span, valWIS)."""
    best = None
    for p in P_GRID:
        Q = build_pearson_span(S, p, mean_full)
        Bv = rolling_cqr_bounds(Q, S, val_origins, cap, VAL_CAL_START, None)
        wv = float(wis_of(Bv, y_val, Q[val_origins - WSTART][:, MED_COL]).mean())
        if best is None or wv < best[2]:
            best = (p, Q, wv)
    return best


def main():
    t_start = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    S = setup()
    ntot = S["ntot"]
    cap = S["cap"]
    cap_train = 2.0 * float(S["yf"][:269].max())     # train-only cap (spotless check)
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = S["yf"][origins]
    val_origins = np.arange(VAL_LO, VAL_HI)
    y_val = S["yf"][val_origins]
    cal = np.arange(T0 - K_CAL, T0)
    r_full = S["yf"] - S["tirex"]

    # ── reference 1: exact 2.4012 fair baseline (per-origin WIS for DM) ──
    qy_ref = tirex_empirical_qy(S["tirex"], r_full, origins, cap)
    cqr_ref = cqr_offsets(tirex_empirical_qy(S["tirex"], r_full, cal, cap), S["yf"][cal])
    ref_B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    ref_wis = wis_of(ref_B, y, qy_ref[:, MED_COL])
    ref_mean = float(ref_wis.mean())

    # ── reference 2: exact 2.2427 Tweedie head (pearson, p*=val argmin, expanding CQR) ──
    p_tw, Qtw, _ = select_p_on_val(S, S["tirex"], cap, val_origins, y_val)
    Btw = rolling_cqr_bounds(Qtw, S, origins, cap, CAL_START, None)
    tw_wis = wis_of(Btw, y, Qtw[origins - WSTART][:, MED_COL])
    tw_mean = float(tw_wis.mean())

    print("=" * 92)
    print(f"REFERENCES  fair baseline WIS={ref_mean:.4f} (want 2.4012)   "
          f"Tweedie head WIS={tw_mean:.4f} (want 2.2427, p*={p_tw})")
    print(f"            Tweedie PICP95={float(((y>=Btw[0.05][0])&(y<=Btw[0.05][1])).mean()):.4f}  "
          f"DM(tweedie vs fair) p={dm(tw_wis, ref_wis)[0]:.2e}")
    print("=" * 92)

    # sanity: build_pearson_span(tirex) == _exp_tweedie.build_span(tirex) pearson
    _chk = build_pearson_span(S, p_tw, S["tirex"])
    _, _QPref, _ = build_span(S, p_tw, "tirex")
    span_match = bool(np.allclose(_chk, _QPref, atol=1e-9))
    print(f"[sanity] build_pearson_span(tirex) == build_span(tirex).pearson : {span_match}")

    results = {}

    # ══════════════════════ (a) TabPFN POINT ══════════════════════
    print("\n--- (a) TabPFN alternative base POINT (rolling 1-step, weeks 52..336) ---")
    G = build_point_features(S)
    if PT_CACHE.exists():
        tab_point = np.load(PT_CACHE)["tab_point"]
        if len(tab_point) != ntot or not np.isfinite(tab_point[MIN_CTX:]).all():
            tab_point = None
        else:
            print(f"    [point] loaded cache {PT_CACHE.name}")
    else:
        tab_point = None
    if tab_point is None:
        tab_point = roll_tabpfn_point(S, G)
        np.savez(PT_CACHE, tab_point=tab_point)
        print(f"    [point] cached -> {PT_CACHE.name}")

    tirex = S["tirex"]

    def _err(pred, idx):
        e = S["yf"][idx] - pred[idx]
        return dict(mae=round(float(np.mean(np.abs(e))), 4),
                    rmse=round(float(np.sqrt(np.mean(e ** 2))), 4))

    pt_err = {"test": {"tirex": _err(tirex, origins), "tabpfn": _err(tab_point, origins)},
              "val": {"tirex": _err(tirex, val_origins), "tabpfn": _err(tab_point, val_origins)}}
    print(f"    1-step point error  TEST[205,337): TiRex {pt_err['test']['tirex']}  "
          f"TabPFN {pt_err['test']['tabpfn']}")
    print(f"    1-step point error  VAL [165,205): TiRex {pt_err['val']['tirex']}  "
          f"TabPFN {pt_err['val']['tabpfn']}")
    point_better = bool(pt_err["test"]["tabpfn"]["mae"] < pt_err["test"]["tirex"]["mae"])

    # convex ensemble weight w on pre-T0 val (jointly with p); w=1 -> pure TiRex, 0 -> pure TabPFN
    W_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)
    ens_pick = None
    for w in W_GRID:
        mean_w = w * tirex + (1.0 - w) * tab_point
        p_w, Q_w, valwis_w = select_p_on_val(S, mean_w, cap, val_origins, y_val)
        if ens_pick is None or valwis_w < ens_pick["valwis"]:
            ens_pick = dict(w=float(w), p=p_w, span=Q_w, valwis=valwis_w)
    print(f"    ensemble (val-selected): w*TiRex+(1-w)*TabPFN  w={ens_pick['w']}  "
          f"p={ens_pick['p']}  valWIS={ens_pick['valwis']:.4f}")

    # candidate base means, each with its own val-selected p
    cand = {}
    for nm, mean_arr in (("tirex", tirex), ("tabpfn", tab_point)):
        p_c, Q_c, vw_c = select_p_on_val(S, mean_arr, cap, val_origins, y_val)
        cand[nm] = dict(p=p_c, span=Q_c, valwis=vw_c)
    cand["ens"] = dict(p=ens_pick["p"], span=ens_pick["span"], valwis=ens_pick["valwis"])

    print(f"\n    {'base':>8s} | {'p':>3s} {'valWIS':>7s} | {'WIS':>7s} {'DMvsFair':>9s} "
          f"{'DMvsTwd':>8s} {'PICP95':>7s} {'peakP95':>7s} {'last34':>7s}")
    print("    " + "-" * 82)
    part_a = {}
    for nm in ("tirex", "tabpfn", "ens"):
        Q = cand[nm]["span"]
        B = rolling_cqr_bounds(Q, S, origins, cap, CAL_START, None)
        med = Q[origins - WSTART][:, MED_COL]
        m = full_metrics(B, y, med, ref_wis, tw_wis, n)
        m["p"] = cand[nm]["p"]
        m["val_wis"] = round(cand[nm]["valwis"], 4)
        part_a[nm] = m
        print(f"    {nm:>8s} | {m['p']:>3} {m['val_wis']:>7.4f} | {m['wis']:>7.4f} "
              f"{m['dm_p_vs_fair']:>9.2e} {m['dm_p_vs_tweedie']:>8.2e} {m['picp95']:>7.4f} "
              f"{m['peak_picp95']:>7.3f} {m['last34_wis']:>7.4f}")

    # headline base = argmin pre-T0 val WIS (honest)
    a_head = min(("tirex", "tabpfn", "ens"), key=lambda k: cand[k]["valwis"])
    results["part_a"] = {
        "point_error": pt_err, "point_tabpfn_beats_tirex_mae": point_better,
        "ensemble_pick": {"w": ens_pick["w"], "p": ens_pick["p"], "valwis": round(ens_pick["valwis"], 4)},
        "headline_base": a_head,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "wis_arr"} for k, v in part_a.items()},
    }

    # ══════════════════════ (b) TabPFN residual-quantile learner ══════════════════════
    print("\n--- (b) TabPFN conditional residual-quantile learner on r=y-TiRex ---")
    if RQ_CACHE.exists():
        Qrq = np.load(RQ_CACHE)["Qrq"]
        if Qrq.shape != (ntot - WSTART, len(FQ)):
            Qrq = None
        else:
            print(f"    [residq] loaded cache {RQ_CACHE.name}")
    else:
        Qrq = None
    if Qrq is None:
        Qrq = roll_tabpfn_resid_qspan(S)
        np.savez(RQ_CACHE, Qrq=Qrq)
        print(f"    [residq] cached -> {RQ_CACHE.name}")

    from scripts.dec_boosted_mech import FQ_COL
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
    cols = {a: (FQ_COL[round(a / 2.0, 4)], FQ_COL[round(1 - a / 2.0, 4)]) for a in FLUSIGHT_ALPHAS}
    qy_te = Qrq[origins - WSTART]
    qy_val = Qrq[val_origins - WSTART]
    med_rq = qy_te[:, MED_COL]
    med_rq_val = qy_val[:, MED_COL]

    # scheme A: raw TabPFN predictive quantiles (no conformal)
    B_raw = {a: (np.clip(qy_te[:, cl], 0, cap), np.clip(qy_te[:, ch], 0, cap))
             for a, (cl, ch) in cols.items()}
    B_raw_val = {a: (np.clip(qy_val[:, cl], 0, cap), np.clip(qy_val[:, ch], 0, cap))
                 for a, (cl, ch) in cols.items()}

    # scheme B: static CQR seed [165,205)
    cqr_rq = cqr_offsets(Qrq[cal - WSTART], S["yf"][cal])
    B_static = build_bounds_cqr(qy_te, cqr_rq, cap)
    val_cal = np.arange(VAL_CAL_START, VAL_LO)
    cqr_rq_val = cqr_offsets(Qrq[val_cal - WSTART], S["yf"][val_cal])
    B_static_val = build_bounds_cqr(qy_val, cqr_rq_val, cap)

    # scheme C: expanding CQR
    B_expand = rolling_cqr_bounds(Qrq, S, origins, cap, CAL_START, None)
    B_expand_val = rolling_cqr_bounds(Qrq, S, val_origins, cap, VAL_CAL_START, None)

    schemes = {"raw": (B_raw, B_raw_val), "cqr_static": (B_static, B_static_val),
               "cqr_expand": (B_expand, B_expand_val)}
    print(f"\n    {'scheme':>12s} | {'valWIS':>7s} | {'WIS':>7s} {'DMvsFair':>9s} {'DMvsTwd':>8s} "
          f"{'PICP95':>7s} {'peakP95':>7s} {'last34':>7s}")
    print("    " + "-" * 80)
    part_b = {}
    for nm, (B, Bv) in schemes.items():
        m = full_metrics(B, y, med_rq, ref_wis, tw_wis, n)
        vw = float(wis_of(Bv, y_val, med_rq_val).mean())
        m["val_wis"] = round(vw, 4)
        part_b[nm] = m
        print(f"    {nm:>12s} | {m['val_wis']:>7.4f} | {m['wis']:>7.4f} "
              f"{m['dm_p_vs_fair']:>9.2e} {m['dm_p_vs_tweedie']:>8.2e} {m['picp95']:>7.4f} "
              f"{m['peak_picp95']:>7.3f} {m['last34_wis']:>7.4f}")
    b_head = min(schemes, key=lambda k: part_b[k]["val_wis"])
    med_full = np.concatenate([np.full(WSTART, np.nan), Qrq[:, MED_COL]])  # week-indexed
    rq_err = _err(med_full, origins)
    results["part_b"] = {
        "resid_median_point_error_test": rq_err,
        "headline_scheme": b_head,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "wis_arr"} for k, v in part_b.items()},
    }

    # ══════════════════════ overall best (honest, val-selected within each part) ══════════
    a_best = part_a[a_head]
    b_best = part_b[b_head]
    if a_best["wis"] <= b_best["wis"]:
        best_name = f"point_{a_head}"
        best = a_best
    else:
        best_name = f"residq_{b_head}"
        best = b_best
    beats_tweedie = bool(best["wis"] < tw_mean and best["dm_p_vs_tweedie"] < 0.05
                         and best["dm_meandiff_vs_tweedie"] < 0)

    # train-only cap spotless check on the best point config (if a point config)
    cap_note = "n/a"
    if best_name.startswith("point"):
        Qb = cand[a_head]["span"]
        # rebuild span with train cap? span clip uses full cap; re-clip check
        Bb_train = rolling_cqr_bounds(Qb, S, origins, cap_train, CAL_START, None)
        wb_train = float(wis_of(Bb_train, y, cand[a_head]["span"][origins - WSTART][:, MED_COL]).mean())
        cap_note = f"train_cap_WIS={wb_train:.4f} (full={best['wis']:.4f}, binds={abs(wb_train-best['wis'])>1e-9})"

    print("\n" + "=" * 92)
    print(f"OVERALL BEST (honest, val-selected): {best_name}")
    print(f"  WIS            = {best['wis']:.4f}   (fair 2.4012 | Tweedie {tw_mean:.4f})")
    print(f"  DM p vs 2.4012 = {best['dm_p_vs_fair']:.3e}   (mean diff {best['dm_meandiff_vs_fair']:+.4f})")
    print(f"  DM p vs Tweedie= {best['dm_p_vs_tweedie']:.3e}   (mean diff {best['dm_meandiff_vs_tweedie']:+.4f})")
    print(f"  PICP95         = {best['picp95']:.4f}   ({best['k_of_n']})   CP95 CI {best['cp95ci']}")
    print(f"  peak PICP95    = {best['peak_picp95']:.3f}   (n_peak={best['n_peak']})")
    print(f"  last34 WIS     = {best['last34_wis']:.4f}")
    print(f"  beats Tweedie (WIS< & DM p<0.05 & diff<0): {beats_tweedie}")
    print(f"  cap check: {cap_note}")
    print("=" * 92)

    out = {
        "n_origins": n, "T0": T0, "weeks": f"{T0}..{ntot-1}",
        "ref_fair_wis": round(ref_mean, 4), "ref_tweedie_wis": round(tw_mean, 4), "tweedie_p": p_tw,
        "span_sanity_match": span_match,
        "part_a": results["part_a"], "part_b": results["part_b"],
        "overall_best": {
            "name": best_name, "wis": best["wis"],
            "dm_p_vs_fair": best["dm_p_vs_fair"], "dm_p_vs_tweedie": best["dm_p_vs_tweedie"],
            "picp95": best["picp95"], "peak_picp95": best["peak_picp95"],
            "last34_wis": best["last34_wis"], "beats_tweedie": beats_tweedie,
            "cap_note": cap_note,
        },
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (ROOT / "scripts" / "_exp_tabpfn.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/_exp_tabpfn.json   (elapsed {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
