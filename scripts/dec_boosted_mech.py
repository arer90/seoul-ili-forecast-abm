#!/usr/bin/env python
"""NEW FUSEDEPI v2 — TiRex base ⊕ boosted quantile-GBM residual learner (CQR) ⊕
mechanism-informed (foi) width modulation.

Motivation
----------
The verified leak-free ceilings on this frozen 68-week hold-out are:
  * direct per-quantile fusion (TiRex native deciles ⊕ NegBin) + adaptive PID:
        WIS 2.677 / PICP95 0.912                     (scripts/nov_quantile_fusion.py)
  * shift-aware / mechanism-informed conformal WIDTH lifts PEAK PICP95 0.88→0.94
        (timing-shuffle-controlled)                  (scripts/nov_mechanism_pi.py)
  * held-out invRMSE stack (last-34)  WIS 2.720      (scripts/fusedepi_fusion_wis.py)

None simultaneously reach WIS<2.68 AND PICP95>=0.93 (esp. peak y>=50). The peak
misses are UNDER-predictions (y > hi) on the rising limb: covering them removes a
huge (2/alpha)*(y-hi) interval penalty — a single lever that lowers WIS AND raises
coverage at once. That is exactly where a CONDITIONAL, mechanism-aware width helps.

Method (this script)
--------------------
1. BASE = TiRex point forecast. Test point = OFFICIAL frozen refit_test_predictions
   (byte-identical to the exp-3 baselines). Pool point = TiRex rolled 1-step on the
   train pool with max_context=512 (verified to reproduce the frozen test point to
   ~0), giving leak-free pool residuals r = y - TiRex.
2. STRONGER RESIDUAL LEARNER = per-quantile Gradient-Boosted trees
   (HistGradientBoostingRegressor, loss="quantile") fit on the residual r, using
   lagged + mechanism features [BASIC lag/seasonal ⊕ 1-lag Rt, S/N, foi ⊕ TiRex
   level]. Its 0.5-quantile IS the boosted point correction. Fit on TRAIN only.
   Conditional residual quantiles q_r(tau|x) → raw y-quantiles q_y = TiRex + q_r,
   monotone-rearranged (Chernozhukov 2010).
3. CQR (Romano-Patterson-Candès 2019): per FluSight alpha, conformity
   E = max(q_y_lo - y, y - q_y_hi) on a held-out CALIBRATION tail of the pool;
   Q_a = (1-a)(1+1/K)-quantile(E); widen [q_y_lo - Q, q_y_hi + Q]. Leak-free (pool).
4. MECHANISM-INFORMED WIDTH MODULATION: after CQR, scale the UPPER half by
   m_i = clip((foi_i/ref_i)^gamma, m_lo, m_hi) — widen on rising force-of-infection
   (peak onset), narrow on decline; ref = leak-free trailing foi mean (seeded from
   pool). gamma chosen on a POOL validation split (never the test).

Model selection (leak-free): 4 conformal strategies {cqr_static, cqr_mech, pid,
pid_mech} are scored on the POOL validation tail; the argmin-WIS strategy is frozen
and applied to the test. gamma likewise pool-tuned.

Leak-free / honesty
-------------------
* Frozen split via scripts.ablation_fusedepi.load_split (pool_end=269, n_test=68).
* Every tunable (GBM, CQR Q, gamma, strategy pick) uses train-pool only.
* Adaptive-PID uses only PAST test obs (obs[0..i-1]); the last-34 (Protocol B) is
  scored with a model whose PID merely warmed up on the first-34 — no test tuning.
* Reports BOTH full-68 and truly-unseen last-34, overall + peak (y>=50, top-25%).
* WIDTH-ARTIFACT CONTROL: the anchor (TiRex point + online conformal) is uniformly
  widened to the SAME mean 95% width and to the SAME PICP95 — if that does NOT
  match the method's WIS, the gain is allocation/sharpness, not raw width.

Reuses the sanctioned helpers: FLUSIGHT_ALPHAS/QUANTILES, wis_from_bounds,
online_conformal_bounds, adaptive Conformal-PID _pid_adjust, mechanistic_features.
No live pipeline/model code is modified. Writes one JSON.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from simulation.analytics.adaptive_conformal import _pid_adjust, online_conformal_bounds, wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS, FLUSIGHT_QUANTILES
from simulation.models.feature_engine._loaders.mechanistic import mechanistic_features
from scripts.ablation_fusedepi import load_split

LOG = logging.getLogger("dec_boosted_mech")
SCRATCH = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/novelty"
)
OUT_JSON = SCRATCH / "dec_boosted_mech.json"
TIREX_CACHE = SCRATCH / "dec_tirex_pool512.npz"

FQ = np.asarray(FLUSIGHT_QUANTILES, dtype=float)
FQ_COL = {round(float(q), 4): i for i, q in enumerate(FQ)}
MED_COL = FQ_COL[0.5]
MAX_CONTEXT = 512
MIN_CTX = 52            # pool rolling start (matches FusedEpi min_ctx)
PEAK_Y = 50.0
K_CAL = 40             # CQR calibration tail (pool weeks) for the FINAL/test fit
K_VAL = 34             # pool validation tail for strategy/gamma selection (mirror last-34)
CONF_WINDOW = 30
CONF_KI = 0.2
REF_WINDOW = 30        # trailing window for the leak-free foi reference
M_LO, M_HI = 0.5, 3.0  # foi multiplier floor/ceiling (allows narrowing -> reallocation)
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def seed_all(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


# ─────────────────────────── TiRex rolling (mean) ───────────────────────────
def roll_tirex_mean(model, y_full: np.ndarray, idxs, max_context: int = MAX_CONTEXT) -> np.ndarray:
    """Rolling 1-step TiRex mean over idxs. Forecast for index t uses y_full[:t] only."""
    import torch
    out = np.empty(len(idxs), dtype=float)
    with torch.no_grad():
        for k, t in enumerate(idxs):
            ctx = torch.tensor(y_full[max(0, t - max_context):t], dtype=torch.float32).unsqueeze(0)
            _q, mean = model.forecast(context=ctx, prediction_length=1)
            out[k] = float(np.asarray(mean).ravel()[0])
    return out


# ─────────────────────────── feature construction ───────────────────────────
def build_features(y_train: np.ndarray, y_test: np.ndarray, X_train: np.ndarray,
                   X_test: np.ndarray, tirex_full: np.ndarray) -> tuple:
    """Assemble per-index feature matrix (BASIC ⊕ 1-lag mechanism ⊕ TiRex level).

    Returns (feat_full (N,p), foi_lag (N,)). feat[t] is causal/past-only except the
    TiRex level at t (available at prediction time). mechanistic_features is causal;
    1-lag makes feature[t] depend on incidence[:t] only (== FusedEpi guard).
    """
    y_full = np.concatenate([y_train, y_test])
    X_full = np.vstack([X_train, X_test])
    mech = mechanistic_features(y_full)                       # (N,3) [rt, s_frac, foi]
    mech_lag = np.vstack([mech[:1], mech[:-1]])               # 1-lag (leak-free)
    feat = np.hstack([X_full, mech_lag, tirex_full[:, None]])
    return feat, mech_lag[:, 2]


# ─────────────────────────── quantile-GBM residual learner ───────────────────
def fit_residual_quantile_gbm(Xtr: np.ndarray, r_tr: np.ndarray) -> dict:
    """Fit one HistGBM quantile regressor per FluSight level on residual r. Train-only.

    Returns {tau: fitted_model}. Conservative small-data settings.
    """
    models = {}
    for q in FQ:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=float(q), learning_rate=0.05,
            max_iter=300, max_leaf_nodes=8, min_samples_leaf=15,
            l2_regularization=1.0, max_depth=3, random_state=42,
        )
        m.fit(Xtr, r_tr)
        models[round(float(q), 4)] = m
    return models


def predict_qy(models: dict, X: np.ndarray, tirex: np.ndarray, cap: float) -> np.ndarray:
    """Raw y-quantile matrix (n,23) = TiRex + conditional residual quantiles, rearranged."""
    n = len(X)
    qy = np.empty((n, len(FQ)), dtype=float)
    for j, q in enumerate(FQ):
        qr = np.asarray(models[round(float(q), 4)].predict(X), dtype=float)
        qy[:, j] = tirex + qr
    qy = np.clip(qy, 0.0, cap)
    qy.sort(axis=1)                                           # monotone rearrangement
    return qy


# ─────────────────────────── conformal engines ───────────────────────────────
def cqr_offsets(qy_cal: np.ndarray, y_cal: np.ndarray) -> dict:
    """CQR per-alpha nonconformity offset Q_a from the calibration split (leak-free)."""
    K = len(y_cal)
    Q = {}
    scores = {}
    for a in FLUSIGHT_ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]
        ch = FQ_COL[round(1.0 - a / 2.0, 4)]
        E = np.maximum(qy_cal[:, cl] - y_cal, y_cal - qy_cal[:, ch])
        beta = min(1.0, (1.0 - a) * (1.0 + 1.0 / max(K, 1)))
        Q[a] = max(0.0, float(np.quantile(E, beta)))
        scores[a] = E
    return {"Q": Q, "scores": scores}


def foi_multipliers(foi_test: np.ndarray, foi_seed: np.ndarray, gamma: float,
                    ref_window: int = REF_WINDOW) -> np.ndarray:
    """Leak-free per-step foi multiplier m_i = clip((foi_i/trailing_ref)^gamma, M_LO, M_HI)."""
    n = len(foi_test)
    m = np.ones(n, dtype=float)
    if gamma <= 0.0:
        return m
    buf = list(np.asarray(foi_seed, dtype=float).ravel())
    for i in range(n):
        ref = float(np.mean(buf[-ref_window:])) if buf else float(foi_test[i])
        ratio = foi_test[i] / ref if ref > 1e-9 else 1.0
        m[i] = float(np.clip(ratio ** gamma, M_LO, M_HI))
        buf.append(float(foi_test[i]))
    return m


def build_bounds_cqr(qy: np.ndarray, cqr: dict, cap: float, foi_mult=None) -> dict:
    """Static CQR bounds (+ optional foi UPPER modulation). {alpha:(lo,hi)}."""
    med = qy[:, MED_COL]
    bounds = {}
    for a in FLUSIGHT_ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]
        ch = FQ_COL[round(1.0 - a / 2.0, 4)]
        Q = cqr["Q"][a]
        lo = np.clip(qy[:, cl] - Q, 0.0, cap)
        hi = np.clip(qy[:, ch] + Q, 0.0, cap)
        if foi_mult is not None:
            hi = np.clip(med + (hi - med) * foi_mult, 0.0, cap)
            lo = np.minimum(lo, hi)
        bounds[a] = (lo, hi)
    return bounds


def build_bounds_pid(qy: np.ndarray, cqr: dict, y_obs: np.ndarray, cap: float,
                     foi_mult=None) -> dict:
    """Adaptive Conformal-PID on the GBM quantiles, seeded from CQR cal scores (leak-free)."""
    med = qy[:, MED_COL]
    bounds = {}
    for a in FLUSIGHT_ALPHAS:
        cl = FQ_COL[round(a / 2.0, 4)]
        ch = FQ_COL[round(1.0 - a / 2.0, 4)]
        nlo, nhi = _pid_adjust(qy[:, cl], qy[:, ch], y_obs, cqr["scores"][a],
                               beta=1.0 - a, target=a, window=CONF_WINDOW, ki=CONF_KI, cap=cap)
        if foi_mult is not None:
            nhi = np.clip(med + (nhi - med) * foi_mult, 0.0, cap)
            nlo = np.minimum(nlo, nhi)
        bounds[a] = (nlo, nhi)
    return bounds


# ─────────────────────────── scoring ─────────────────────────────────────────
def score(bounds: dict, y: np.ndarray, median: np.ndarray, masks: dict) -> dict:
    wis_arr = np.asarray(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=median), dtype=float)
    lo95, hi95 = bounds[0.05]
    cov95 = (y >= lo95) & (y <= hi95)
    lo50, hi50 = bounds[0.50]
    cov50 = (y >= lo50) & (y <= hi50)
    out = {}
    for mk, m in masks.items():
        m = np.asarray(m, bool)
        if m.sum() == 0:
            continue
        out[mk] = {
            "wis": float(np.mean(wis_arr[m])),
            "picp95": float(np.mean(cov95[m])),
            "picp50": float(np.mean(cov50[m])),
            "mean_width95": float(np.mean((hi95 - lo95)[m])),
            "n": int(m.sum()),
        }
    return out


def wis_overall(bounds: dict, y: np.ndarray, median: np.ndarray) -> float:
    wis_arr = np.asarray(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=median), dtype=float)
    return float(np.mean(wis_arr))


# ─────────────────────────── main ────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(42)
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test)
    n_peak = int((y_test >= PEAK_Y).sum())
    LOG.info("split train=%d test=%d peak(y>=%.0f)=%d", ntr, nte, PEAK_Y, n_peak)

    # ── frozen official TiRex test point ──
    frozen = np.asarray(json.loads(
        (ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
        ["refit_test_predictions"], dtype=float)

    # ── TiRex pool rolling (max_context=512), cached; verify test roll == frozen ──
    from tirex import load_model
    model = load_model("NX-AI/TiRex", device="cpu")
    y_full = np.concatenate([y_train, y_test])
    cap = 2.0 * float(np.max(y_full))

    if TIREX_CACHE.exists():
        d = np.load(TIREX_CACHE)
        tirex_pool = d["tirex_pool"]
        tirex_test_roll = d["tirex_test_roll"]
        if len(tirex_pool) != ntr - MIN_CTX or len(tirex_test_roll) != nte:
            tirex_pool = tirex_test_roll = None
    else:
        tirex_pool = tirex_test_roll = None
    if tirex_pool is None:
        LOG.info("rolling TiRex pool (%d) + test (%d) ...", ntr - MIN_CTX, nte)
        tirex_pool = roll_tirex_mean(model, y_train, list(range(MIN_CTX, ntr)))
        tirex_test_roll = roll_tirex_mean(model, y_full, list(range(ntr, ntr + nte)))
        np.savez(TIREX_CACHE, tirex_pool=tirex_pool, tirex_test_roll=tirex_test_roll)
    tirex_maxdiff = float(np.max(np.abs(tirex_test_roll - frozen)))
    LOG.info("TiRex test roll vs frozen maxdiff=%.6f", tirex_maxdiff)

    # test point = frozen official (exact); pool uses rolled 512 (consistent regime)
    tirex_test = frozen
    tirex_full = np.concatenate([
        np.full(MIN_CTX, np.nan), tirex_pool, tirex_test])          # aligned to y_full idx

    # ── features ──
    feat_full, foi_lag = build_features(y_train, y_test, X_train, X_test, tirex_full)
    pool_slc = slice(MIN_CTX, ntr)                                  # usable pool indices
    test_slc = slice(ntr, ntr + nte)
    Xp = feat_full[pool_slc]; yp = y_train[MIN_CTX:]; tp = tirex_pool
    Xt = feat_full[test_slc]; foi_te = foi_lag[test_slc]
    foi_pool = foi_lag[pool_slc]
    rp = yp - tp                                                    # pool residuals
    npool = len(yp)

    # masks
    overall = np.ones(nte, dtype=bool)
    peak50 = y_test >= PEAK_Y
    peak25 = y_test >= float(np.quantile(y_test, 0.75))
    last34 = np.zeros(nte, dtype=bool); last34[nte - 34:] = True
    last34_peak = last34 & (y_test >= float(np.quantile(y_test[nte - 34:], 0.75)))
    masks = {"overall_68": overall, "peak_y50": peak50, "peak_top25pct": peak25,
             "last34": last34, "last34_peak": last34_peak}

    # ════════════════ leak-free selection on the POOL validation tail ════════════════
    # tuning fit: GBM on pool[:-(K_VAL+K_CAL)], CQR on the K_CAL before val, eval on last K_VAL
    v_cut = npool - K_VAL
    c_cut = v_cut - K_CAL
    LOG.info("pool selection: train=%d cal=%d val=%d", c_cut, K_CAL, K_VAL)
    gbm_tune = fit_residual_quantile_gbm(Xp[:c_cut], rp[:c_cut])
    qy_cal_v = predict_qy(gbm_tune, Xp[c_cut:v_cut], tp[c_cut:v_cut], cap)
    cqr_v = cqr_offsets(qy_cal_v, yp[c_cut:v_cut])
    qy_val = predict_qy(gbm_tune, Xp[v_cut:], tp[v_cut:], cap)
    y_val = yp[v_cut:]
    foi_val = foi_pool[v_cut:]
    foi_val_seed = foi_pool[:v_cut]
    med_val = qy_val[:, MED_COL]

    def val_gamma_pick(engine: str) -> tuple:
        best = None
        for g in GAMMAS:
            mult = foi_multipliers(foi_val, foi_val_seed, g) if g > 0 else None
            if engine == "cqr":
                b = build_bounds_cqr(qy_val, cqr_v, cap, foi_mult=mult)
            else:
                b = build_bounds_pid(qy_val, cqr_v, y_val, cap, foi_mult=mult)
            w = wis_overall(b, y_val, med_val)
            if best is None or w < best[1]:
                best = (g, w)
        return best

    # strategy pool-val WIS
    b_cqr_static_v = build_bounds_cqr(qy_val, cqr_v, cap)
    wis_cqr_static = wis_overall(b_cqr_static_v, y_val, med_val)
    g_cqr, wis_cqr_mech = val_gamma_pick("cqr")
    b_pid_v = build_bounds_pid(qy_val, cqr_v, y_val, cap)
    wis_pid = wis_overall(b_pid_v, y_val, med_val)
    g_pid, wis_pid_mech = val_gamma_pick("pid")
    pool_val = {
        "cqr_static": {"wis": wis_cqr_static, "gamma": 0.0},
        "cqr_mech": {"wis": wis_cqr_mech, "gamma": g_cqr},
        "pid": {"wis": wis_pid, "gamma": 0.0},
        "pid_mech": {"wis": wis_pid_mech, "gamma": g_pid},
    }
    selected = min(pool_val, key=lambda k: pool_val[k]["wis"])
    LOG.info("pool-val WIS: %s -> selected=%s",
             {k: round(v["wis"], 4) for k, v in pool_val.items()}, selected)

    # ════════════════ FINAL fit for TEST (pool proper-train + cal, no test tuning) ════
    f_cut = npool - K_CAL
    gbm_final = fit_residual_quantile_gbm(Xp[:f_cut], rp[:f_cut])
    qy_cal_f = predict_qy(gbm_final, Xp[f_cut:], tp[f_cut:], cap)
    cqr_f = cqr_offsets(qy_cal_f, yp[f_cut:])
    qy_te = predict_qy(gbm_final, Xt, tirex_test, cap)
    med_te = qy_te[:, MED_COL]

    mult_cqr = foi_multipliers(foi_te, foi_pool, g_cqr) if g_cqr > 0 else None
    mult_pid = foi_multipliers(foi_te, foi_pool, g_pid) if g_pid > 0 else None

    variants = {}
    variants["V0_tirex_online_conformal"] = online_conformal_bounds(
        tirex_test, y_test, FLUSIGHT_ALPHAS, window=CONF_WINDOW, ki=CONF_KI)
    variants["V1_cqr_static"] = build_bounds_cqr(qy_te, cqr_f, cap)
    variants["V2_cqr_mech"] = build_bounds_cqr(qy_te, cqr_f, cap, foi_mult=mult_cqr)
    variants["V3_pid"] = build_bounds_pid(qy_te, cqr_f, y_test, cap)
    variants["V4_pid_mech"] = build_bounds_pid(qy_te, cqr_f, y_test, cap, foi_mult=mult_pid)

    med_of = {"V0_tirex_online_conformal": tirex_test}
    results = {}
    for name, b in variants.items():
        med = med_of.get(name, med_te)
        results[name] = score(b, y_test, med, masks)

    # selected strategy -> the MAIN model
    strat_to_variant = {"cqr_static": "V1_cqr_static", "cqr_mech": "V2_cqr_mech",
                        "pid": "V3_pid", "pid_mech": "V4_pid_mech"}
    main_variant = strat_to_variant[selected]
    main_bounds = variants[main_variant]
    main_med = med_te
    main_scores = results[main_variant]

    # ════════════════ WIDTH-ARTIFACT CONTROLS (uniform widening of the anchor) ════════
    anchor = variants["V0_tirex_online_conformal"]
    anchor_w95 = float(np.mean(anchor[0.05][1] - anchor[0.05][0]))
    main_w95 = main_scores["overall_68"]["mean_width95"]

    def uniform_scale_anchor(c: float) -> dict:
        out = {}
        for a in FLUSIGHT_ALPHAS:
            lo, hi = anchor[a]
            half = (hi - lo) / 2.0
            mid = (hi + lo) / 2.0
            out[a] = (np.clip(mid - c * half, 0.0, cap), np.clip(mid + c * half, 0.0, cap))
        return out

    # (a) match mean width95
    c_width = main_w95 / anchor_w95 if anchor_w95 > 1e-9 else 1.0
    ctrl_width = uniform_scale_anchor(c_width)
    ctrl_width_scores = score(ctrl_width, y_test, tirex_test, masks)
    # (b) match overall PICP95 (grid search c)
    target_picp = main_scores["overall_68"]["picp95"]
    c_grid = np.linspace(1.0, 6.0, 101)
    c_pick, best_gap = 1.0, 1e9
    for c in c_grid:
        p = float(np.mean((y_test >= uniform_scale_anchor(c)[0.05][0]) &
                          (y_test <= uniform_scale_anchor(c)[0.05][1])))
        if p >= target_picp - 1e-9 and abs(p - target_picp) < best_gap:
            c_pick, best_gap = c, abs(p - target_picp)
    ctrl_picp = uniform_scale_anchor(c_pick)
    ctrl_picp_scores = score(ctrl_picp, y_test, tirex_test, masks)

    # ════════════════ verdict — the 4 decisive-win bars ════════════════
    full = main_scores["overall_68"]
    l34 = main_scores["last34"]
    bar1 = bool(full["wis"] < 2.68)
    bar2 = bool(full["picp95"] >= 0.93)
    # bar3: last-34 gain survives — beat the held-out stack 2.720 AND beat anchor last-34
    anchor_l34_wis = results["V0_tirex_online_conformal"]["last34"]["wis"]
    bar3 = bool(l34["wis"] < 2.720 and l34["wis"] < anchor_l34_wis)
    # bar4: not a width artifact — uniform widening to same width does NOT match the WIS
    bar4 = bool(full["wis"] < ctrl_width_scores["overall_68"]["wis"] - 1e-9)
    decisive = bool(bar1 and bar2 and bar3 and bar4)

    verdict = {
        "selected_strategy": selected,
        "selected_variant": main_variant,
        "selected_gamma": pool_val[selected]["gamma"],
        "full68_wis": full["wis"],
        "full68_picp95": full["picp95"],
        "peak_y50_wis": main_scores.get("peak_y50", {}).get("wis"),
        "peak_y50_picp95": main_scores.get("peak_y50", {}).get("picp95"),
        "last34_wis": l34["wis"],
        "last34_picp95": l34["picp95"],
        "anchor_last34_wis": anchor_l34_wis,
        "width_control_wis_at_equal_width": ctrl_width_scores["overall_68"]["wis"],
        "width_control_picp95_at_equal_width": ctrl_width_scores["overall_68"]["picp95"],
        "picp_control_wis_at_equal_coverage": ctrl_picp_scores["overall_68"]["wis"],
        "bar1_full68_wis_lt_2p68": bar1,
        "bar2_picp95_ge_0p93": bar2,
        "bar3_survives_last34": bar3,
        "bar4_not_width_artifact": bar4,
        "DECISIVE_WIN_all_4": decisive,
        "tirex_test_maxdiff_vs_frozen": tirex_maxdiff,
    }

    out = {
        "method": "TiRex base ⊕ boosted quantile-GBM residual (CQR) ⊕ mechanism-informed foi width modulation",
        "protocol": {
            "split": {k: meta[k] for k in ("n", "pool_end", "test_start", "test_end", "n_test")},
            "tirex": f"test=frozen official refit_test_predictions; pool rolled max_context={MAX_CONTEXT} "
                     f"(maxdiff vs frozen={tirex_maxdiff:.2e})",
            "residual_learner": "HistGradientBoostingRegressor(loss=quantile) per FluSight level on r=y-TiRex; "
                                "features=BASIC lag/seasonal ⊕ 1-lag [rt,s,foi] ⊕ TiRex level",
            "cqr": f"per-alpha CQR offset on pool calibration tail (K_CAL={K_CAL})",
            "mechanism": f"upper half ×clip((foi/trailing_ref)^gamma,{M_LO},{M_HI}); gamma pool-val tuned",
            "selection": f"strategy+gamma argmin WIS on pool validation tail (K_VAL={K_VAL}); test never tuned",
            "leak_free": "GBM/CQR/gamma/strategy all pool-only; PID uses past test obs only; "
                         "last-34 scored with no test tuning",
        },
        "n_peak_y50": n_peak,
        "pool_val_selection": pool_val,
        "results": results,
        "width_artifact_control": {
            "anchor_mean_width95": anchor_w95,
            "main_mean_width95": main_w95,
            "scale_to_equal_width": c_width,
            "control_equal_width": ctrl_width_scores["overall_68"],
            "scale_to_equal_picp95": c_pick,
            "control_equal_picp95": ctrl_picp_scores["overall_68"],
        },
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── console table ──
    print("\n================= NEW FUSEDEPI v2 (boosted-GBM + mechanism) =================")
    print(f"selected strategy={selected} ({main_variant}) gamma={pool_val[selected]['gamma']}  "
          f"TiRex maxdiff vs frozen={tirex_maxdiff:.2e}  n_peak(y>=50)={n_peak}\n")
    hdr = (f"{'variant':28s} {'WIS_all':>8s} {'PICP95':>7s} {'WIS_pk50':>9s} {'PICP_pk50':>10s} "
           f"{'WIS_l34':>8s} {'PICP_l34':>9s} {'W95':>7s}")
    print(hdr); print("-" * len(hdr))
    for v in ["V0_tirex_online_conformal", "V1_cqr_static", "V2_cqr_mech", "V3_pid", "V4_pid_mech"]:
        r = results[v]
        pk = r.get("peak_y50", {})
        print(f"{v:28s} {r['overall_68']['wis']:8.4f} {r['overall_68']['picp95']:7.3f} "
              f"{pk.get('wis', float('nan')):9.4f} {pk.get('picp95', float('nan')):10.3f} "
              f"{r['last34']['wis']:8.4f} {r['last34']['picp95']:9.3f} "
              f"{r['overall_68']['mean_width95']:7.2f}")
    print("\nWIDTH-ARTIFACT CONTROL (uniform widening of TiRex+online-conformal anchor):")
    print(f"  equal-width (c={c_width:.2f}): WIS={ctrl_width_scores['overall_68']['wis']:.4f} "
          f"PICP95={ctrl_width_scores['overall_68']['picp95']:.3f} "
          f"(main WIS={full['wis']:.4f} at W95={main_w95:.2f})")
    print(f"  equal-PICP95 (c={c_pick:.2f}): WIS={ctrl_picp_scores['overall_68']['wis']:.4f} "
          f"PICP95={ctrl_picp_scores['overall_68']['picp95']:.3f}")
    print("\nTARGETS: full-68 WIS<2.68 ; PICP95>=0.93 ; last-34 WIS<2.720 ; beat equal-width control")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
