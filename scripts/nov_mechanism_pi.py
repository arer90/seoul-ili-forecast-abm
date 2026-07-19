#!/usr/bin/env python
"""NOVELTY CANDIDATE — MECHANISM-INFORMED PREDICTION INTERVALS.

Question
--------
The generic online-conformal wrapper (experiment 3) is PURELY REACTIVE: at week i
its half-width Q[i] is a rolling quantile of *past* |residuals| + a PID integral.
It therefore LAGS the epidemic surge — the interval only widens AFTER the surge has
already produced large residuals, which is exactly why the 5 extreme peak weeks are
missed and PICP95 sits at ~0.88 instead of 0.95.

Mechanism-informed idea: FusedEpi already computes a leak-free, 1-lag SEIR/renewal
state — reproduction-number proxy ``rt``, susceptible fraction ``s_frac`` and force
of infection ``foi = rt*(1-s)`` — purely from past ILI incidence (causal, past-only).
On the TRAIN POOL these mechanism channels predict the MAGNITUDE of TiRex's forecast
error (leak-free corr(|resid|, foi_lag)=0.40 vs rt=0.16). We use that LEADING signal
to MODULATE the conformal half-width: widen when the current mechanism state is above
its recent trailing level (rising force of infection = high uncertainty), narrow when
below (decline/plateau = low uncertainty). The multiplier is a scale-free ratio
``(foi/ref)^gamma`` centred on a leak-free trailing reference, so it REALLOCATES width
across the season rather than uniformly inflating it.

Does mechanism-informed uncertainty beat the mechanism-free online conformal on
coverage (PICP95, esp. at the peak) or WIS, WITHOUT inflating WIS?

Design (leak-free, same frozen 68-week split & protocol as ablation_fusedepi.py)
--------------------------------------------------------------------------------
* Point predictions: OFFICIAL frozen TiRex ``refit_test_predictions`` (== the exact
  baseline that scores WIS 2.951 / PICP95 0.882 under the generic wrapper). The point
  forecast is IDENTICAL for baseline vs mechanism method — only the interval differs,
  so any gap is attributable purely to the mechanism-informed width.
* Mechanism state: ``mechanistic_features`` on the full pool+test incidence, 1-lag
  (``vstack([mech[:1], mech[:-1]])`` — exactly as FusedEpi._mech_features), sliced to
  the test span. feature[t] depends only on incidence[:t] → causal, leak-free.
* Conformal engine: the SAME ``online_conformal_bounds`` rolling logic (window=30,
  ki=0.2), re-implemented with an extra per-step mechanism multiplier so the base
  (non-mechanism) path is byte-identical to the shipped helper (verified in-script).
* Hyperparameter gamma (modulation strength) is tuned ONLY on the TRAIN POOL TiRex
  rolling residuals (min_ctx..pool_end) — never on the test. We also report a
  parameter-free gamma=1.0 and a first-half-test-tuned / second-half-evaluated
  protocol as leak-free robustness checks.
* Metrics: WIS via ``wis_from_bounds`` (identical to R10 per_model_eval); PICP95 =
  empirical 95% PI coverage. Reported OVERALL (68) and at the PEAK (y>=75th pct == the
  17 top-incidence weeks, and the alternate y>=50 extreme regime).

No live pipeline/model code is modified. Writes one JSON.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

from scripts.ablation_fusedepi import load_split
from scripts.fusedepi_fusion_wis import load_frozen
from simulation.analytics.adaptive_conformal import online_conformal_bounds, wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.models.feature_engine._loaders.mechanistic import mechanistic_features

OUT = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/novelty/mechanism_pi.json"
)
CACHE = REPO / "scripts/.nov_mech_cache.npz"

PRIMARY_WINDOW = 30
PRIMARY_KI = 0.2
REF_WINDOW = 30            # trailing window for the leak-free mechanism reference
M_LO, M_HI = 0.5, 2.5      # multiplier floor/ceiling (allows NARROWING -> reallocation)


# ─────────────────────────────────────────────────────────────────────────────
# mechanism-informed online conformal
# ─────────────────────────────────────────────────────────────────────────────
def mechanism_online_conformal(pred, y_observed, alphas, signal, *,
                               window=PRIMARY_WINDOW, ki=PRIMARY_KI, cap=None,
                               gamma=0.0, ref_window=REF_WINDOW, seed_signal=None,
                               m_lo=M_LO, m_hi=M_HI, side="sym"):
    """Online conformal (leak-free) with a per-step MECHANISM half-width multiplier.

    With gamma==0 this is byte-identical to ``online_conformal_bounds`` (the base
    reactive wrapper). With gamma>0 the half-width at step i is scaled by
        m[i] = clip( (signal[i]/ref[i]) ** gamma , m_lo, m_hi )
    where ref[i] = trailing mean of the past mechanism buffer (seeded, leak-free).
    ``signal`` is the 1-lag mechanism channel (foi/rt/1-s), itself past-only.

    Args:
        pred, y_observed: (n,) test point predictions & rolling observations.
        signal: (n,) 1-lag mechanism uncertainty channel (>=0).
        gamma: modulation strength (0 = mechanism-free baseline).
        seed_signal: past mechanism values (train pool) to seed the trailing ref.
        side: "sym" widen both bounds; "upper" widen only the upper bound.
    Returns: {alpha: (lo, hi)} — leak-free (y[i] appended only after interval set).
    """
    pred = np.asarray(pred, np.float64).ravel()
    y = np.asarray(y_observed, np.float64).ravel()
    sig = np.asarray(signal, np.float64).ravel()
    n = len(y)
    if cap is None:
        cap = 2.0 * float(max(np.nanmax(pred) if pred.size else 0.0,
                              np.nanmax(y) if y.size else 0.0, 1.0))
    seed = list(np.asarray(seed_signal, np.float64).ravel()) if seed_signal is not None else []
    out = {}
    for a in alphas:
        buf = []
        sig_buf = list(seed)
        integral = 0.0
        lo = np.empty(n); hi = np.empty(n)
        for i in range(n):
            if len(buf) >= 3:
                q = max(0.0, float(np.quantile(buf[-window:], 1.0 - a)))
            else:
                q = float(np.max(np.abs(buf))) if buf else 0.0
            Q = max(0.0, q + ki * max(q, 1.0) * integral)      # base reactive half-width
            # ── mechanism multiplier (leak-free trailing reference) ──
            if gamma > 0.0:
                ref = float(np.mean(sig_buf[-ref_window:])) if sig_buf else float(sig[i])
                ratio = sig[i] / ref if ref > 1e-9 else 1.0
                m = float(np.clip(ratio ** gamma, m_lo, m_hi))
            else:
                m = 1.0
            if side == "upper":
                Qlo, Qhi = Q, Q * m
            else:
                Qlo = Qhi = Q * m
            lo[i] = max(0.0, pred[i] - Qlo)
            hi[i] = min(cap, pred[i] + Qhi)
            miscov = 1.0 if (y[i] < lo[i] or y[i] > hi[i]) else 0.0
            integral = float(np.clip(integral + (miscov - a), -5.0, 5.0))
            buf.append(float(abs(y[i] - pred[i])))
            sig_buf.append(float(sig[i]))
        out[a] = (lo, hi)
    return out


def score_bounds(y, bounds, pred, mask):
    """WIS + coverage on a subset mask (leak-free scoring)."""
    wis = np.asarray(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=pred), float)
    m = np.asarray(mask, bool)

    def picp(alpha):
        lo, hi = bounds[alpha]
        return float(np.mean(((y >= lo) & (y <= hi))[m]))

    lo95, hi95 = bounds[0.05]
    lo80, hi80 = bounds[0.20]
    return {
        "wis": float(np.mean(wis[m])),
        "picp95": picp(0.05),
        "picp80": picp(0.20),
        "picp50": picp(0.50),
        "mean_width95": float(np.mean((hi95 - lo95)[m])),
        "mean_width80": float(np.mean((hi80 - lo80)[m])),
        "n": int(m.sum()),
    }


def masks_for(y):
    y = np.asarray(y, float)
    peak_thr = float(np.quantile(y, 0.75))
    return {
        "overall": np.ones(len(y), bool),
        "peak_top25pct": y >= peak_thr,
        "peak_ge50": y >= 50.0,
    }, peak_thr


# ─────────────────────────────────────────────────────────────────────────────
def build_signals(y_train, y_test):
    """Full-sequence 1-lag mechanism channels, sliced to pool & test (leak-free)."""
    y_full = np.concatenate([y_train, y_test])
    mech = mechanistic_features(y_full)                         # (N,3) [rt, s_frac, foi]
    mech_lag = np.vstack([mech[:1], mech[:-1]])                 # 1-lag (== FusedEpi)
    ts = len(y_train)
    level_lag = np.concatenate([y_full[:1], y_full[:-1]])       # 1-lag incidence LEVEL
    rng = np.random.default_rng(42)
    foi = mech_lag[:, 2]
    foi_shuf = foi.copy(); rng.shuffle(foi_shuf)                # timing-destroyed control
    ch = {  # name -> full-sequence 1-lag channel (sliced to pool/test downstream)
        "foi": foi,
        "rt": mech_lag[:, 0],
        "onemS": 1.0 - mech_lag[:, 1],  # 1 - susceptible fraction (depletion)
        "level": level_lag,             # heteroskedastic competitor (NOT mechanism)
        "foi_shuffled": foi_shuf,       # CONTROL: same values, timing destroyed
    }
    return mech_lag, ts, ch


def tirex_pool_rolling(y_train):
    """TiRex rolling 1-step on the train pool (for leak-free gamma tuning). Cached."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if int(d["min_ctx"]) and len(d["tx_pool"]) and int(len(d["y_train"])) == len(y_train):
            return np.asarray(d["tx_pool"], float), int(d["min_ctx"])
    from simulation.models.fused_epi import FusedEpiForecaster
    from tirex import load_model
    fe = FusedEpiForecaster()
    fe._tx = load_model(fe.repo_id, device="cpu")
    min_ctx = fe.min_ctx
    idxs = list(range(min_ctx, len(y_train)))
    tx_pool = np.asarray(fe._tirex_roll(y_train, idxs), float)
    try:  # self-managing warm cache (leak-free: pool TiRex rolling only)
        np.savez(CACHE, tx_pool=tx_pool, min_ctx=min_ctx, y_train=y_train)
    except Exception:
        pass
    return tx_pool, min_ctx


def main() -> dict:
    X_train, y_train, X_test, y_test, meta = load_split()
    pred_tirex, y_true, oof = load_frozen("TiRex")
    assert np.max(np.abs(y_test - y_true)) < 1e-9, "frozen y_true != split y_test"
    pred_fe, _, _ = load_frozen("FusedEpi")

    mech_lag, ts, ch = build_signals(y_train, y_test)
    # test-span mechanism channels
    ch_test = {k: v[ts:ts + len(y_test)] for k, v in ch.items()}
    # seed = train-pool mechanism values (past, leak-free) for the trailing ref
    ch_seed = {k: v[:ts] for k, v in ch.items()}

    masks, peak_thr = masks_for(y_test)

    # ── 1) verify base path == shipped helper (gamma=0) ──
    b_ship = online_conformal_bounds(pred_tirex, y_test, FLUSIGHT_ALPHAS,
                                     window=PRIMARY_WINDOW, ki=PRIMARY_KI)
    b_base = mechanism_online_conformal(pred_tirex, y_test, FLUSIGHT_ALPHAS,
                                        ch_test["foi"], gamma=0.0)
    ident = max(float(np.max(np.abs(b_ship[a][0] - b_base[a][0])))
                for a in FLUSIGHT_ALPHAS)
    ident = max(ident, max(float(np.max(np.abs(b_ship[a][1] - b_base[a][1])))
                           for a in FLUSIGHT_ALPHAS))
    base_scores = {mk: score_bounds(y_test, b_base, pred_tirex, mm)
                   for mk, mm in masks.items()}

    # ── 2) leak-free gamma tuning on TRAIN POOL (TiRex rolling residuals) ──
    tx_pool, min_ctx = tirex_pool_rolling(y_train)
    y_pool = y_train[min_ctx:]                     # aligned with tx_pool
    pool_masks = {"overall": np.ones(len(y_pool), bool)}
    # pool mechanism channels aligned with tx_pool span (indices min_ctx..pool_end)
    ch_pool = {k: v[min_ctx:ts] for k, v in ch.items()}
    ch_pool_seed = {k: v[:min_ctx] for k, v in ch.items()}

    GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    SIGNALS = ["foi", "rt", "onemS"]
    pool_grid = {}
    for sname in SIGNALS:
        pool_grid[sname] = {}
        for g in GAMMAS:
            bp = mechanism_online_conformal(
                tx_pool, y_pool, FLUSIGHT_ALPHAS, ch_pool[sname],
                gamma=g, seed_signal=ch_pool_seed[sname])
            sc = score_bounds(y_pool, bp, tx_pool, pool_masks["overall"])
            pool_grid[sname][g] = {"wis": sc["wis"], "picp95": sc["picp95"],
                                   "mean_width95": sc["mean_width95"]}
    # pick (signal, gamma) minimizing POOL WIS
    best = min(
        ((s, g) for s in SIGNALS for g in GAMMAS),
        key=lambda sg: pool_grid[sg[0]][sg[1]]["wis"],
    )
    best_signal, best_gamma = best[0], float(best[1])
    # also the pool-WIS-optimal gamma for the a-priori mechanism channel (foi)
    foi_best_gamma = float(min(GAMMAS, key=lambda g: pool_grid["foi"][g]["wis"]))

    # ── 3) evaluate on TEST: mechanism-informed vs mechanism-free ──
    def eval_config(signal_name, gamma, side="sym", pred=pred_tirex,
                    sig_test=None, sig_seed=None, y=y_test, ms=masks):
        st = ch_test[signal_name] if sig_test is None else sig_test
        sd = ch_seed[signal_name] if sig_seed is None else sig_seed
        bb = mechanism_online_conformal(pred, y, FLUSIGHT_ALPHAS, st,
                                        gamma=gamma, seed_signal=sd, side=side)
        return {mk: score_bounds(y, bb, pred, mm) for mk, mm in ms.items()}

    test_pool_tuned = eval_config(best_signal, best_gamma)          # gold: pool-tuned
    test_foi_pooltuned = eval_config("foi", foi_best_gamma)         # a-priori channel, pool-tuned
    test_foi_g1 = eval_config("foi", 1.0)                          # parameter-free
    test_foi_upper = eval_config("foi", best_gamma if best_signal == "foi" else foi_best_gamma,
                                 side="upper")                      # asym (upper only)

    # ── 3b) CONTROLS: is the gain from the mechanism TIMING, or just a free width knob? ──
    #   For each control channel, pool-tune gamma exactly like the mechanism channel, eval test.
    controls = {}
    for cname in ("level", "foi_shuffled"):
        cpool = ch[cname][min_ctx:ts]
        cpool_seed = ch[cname][:min_ctx]
        cgrid = {}
        for g in GAMMAS:
            bp = mechanism_online_conformal(tx_pool, y_pool, FLUSIGHT_ALPHAS, cpool,
                                            gamma=g, seed_signal=cpool_seed)
            cgrid[g] = score_bounds(y_pool, bp, tx_pool, pool_masks["overall"])["wis"]
        cg = float(min(GAMMAS, key=lambda g: cgrid[g]))
        controls[cname] = {
            "pool_optimal_gamma": cg,
            "test": eval_config(cname, cg),
        }

    # ── 4) leak-free robustness: tune gamma on FIRST HALF of test, eval on SECOND HALF ──
    cut = len(y_test) // 2
    fh = slice(0, cut); sh = slice(cut, len(y_test))
    fh_mask = np.zeros(len(y_test), bool); fh_mask[fh] = True
    sh_mask = np.zeros(len(y_test), bool); sh_mask[sh] = True
    # tune gamma on first half WIS (foi channel)
    fh_grid = {}
    for g in GAMMAS:
        bb = mechanism_online_conformal(pred_tirex, y_test, FLUSIGHT_ALPHAS,
                                        ch_test["foi"], gamma=g, seed_signal=ch_seed["foi"])
        fh_grid[g] = score_bounds(y_test, bb, pred_tirex, fh_mask)["wis"]
    fh_best_gamma = float(min(GAMMAS, key=lambda g: fh_grid[g]))
    bb_fh = mechanism_online_conformal(pred_tirex, y_test, FLUSIGHT_ALPHAS,
                                       ch_test["foi"], gamma=fh_best_gamma, seed_signal=ch_seed["foi"])
    bb_base_full = b_base
    sh_peak = sh_mask & (y_test >= float(np.quantile(y_test[sh], 0.75)))
    secondhalf = {
        "gamma_tuned_on_first_half": fh_best_gamma,
        "mechanism_secondhalf": {
            "overall": score_bounds(y_test, bb_fh, pred_tirex, sh_mask),
            "peak": score_bounds(y_test, bb_fh, pred_tirex, sh_peak),
        },
        "baseline_secondhalf": {
            "overall": score_bounds(y_test, bb_base_full, pred_tirex, sh_mask),
            "peak": score_bounds(y_test, bb_base_full, pred_tirex, sh_peak),
        },
    }

    # ── 5) does it also help the FusedEpi point? (generality) ──
    fe_base = {mk: score_bounds(y_test, mechanism_online_conformal(
        pred_fe, y_test, FLUSIGHT_ALPHAS, ch_test["foi"], gamma=0.0), pred_fe, mm)
        for mk, mm in masks.items()}
    fe_mech = eval_config("foi", foi_best_gamma, pred=pred_fe)

    # ── verdict ──
    base_o = base_scores["overall"]["wis"]
    base_p95 = base_scores["overall"]["picp95"]
    base_peak_p95 = base_scores["peak_top25pct"]["picp95"]
    m_o = test_foi_pooltuned["overall"]["wis"]
    m_p95 = test_foi_pooltuned["overall"]["picp95"]
    m_peak_p95 = test_foi_pooltuned["peak_top25pct"]["picp95"]

    beats_tirex_wis = bool(m_o < base_o)          # base_o == TiRex-alone 2.951
    beats_stack_wis = bool(m_o < 2.720)           # held-out-tuned inverse-RMSE stack
    improves_cov = bool(
        (m_p95 > base_p95 + 1e-9 or m_peak_p95 > base_peak_p95 + 1e-9)
        and m_o <= base_o + 0.02           # coverage lifted WITHOUT inflating WIS
    )

    verdict = {
        "base_path_identical_to_shipped_helper_maxabs": ident,
        "baseline_tirex_wis": base_o,
        "baseline_tirex_picp95": base_p95,
        "baseline_tirex_peak_picp95": base_peak_p95,
        "mechanism_pooltuned_signal": best_signal,
        "mechanism_pooltuned_gamma": best_gamma,
        "foi_pool_optimal_gamma": foi_best_gamma,
        "mechanism_foi_pooltuned_wis": m_o,
        "mechanism_foi_pooltuned_picp95": m_p95,
        "mechanism_foi_pooltuned_peak_picp95": m_peak_p95,
        "delta_wis_vs_tirex": float(m_o - base_o),
        "delta_picp95_vs_tirex": float(m_p95 - base_p95),
        "delta_peak_picp95_vs_tirex": float(m_peak_p95 - base_peak_p95),
        "beats_tirex_alone_wis": beats_tirex_wis,
        "beats_heldout_stack_wis_2p72": beats_stack_wis,
        "improves_coverage_without_inflating_wis": improves_cov,
        "foi_upper_asym_peak_picp95": test_foi_upper["peak_top25pct"]["picp95"],
        "foi_upper_asym_overall_wis": test_foi_upper["overall"]["wis"],
        "control_level_wis": controls["level"]["test"]["overall"]["wis"],
        "control_foishuffled_wis": controls["foi_shuffled"]["test"]["overall"]["wis"],
        "mechanism_beats_shuffled_control": bool(m_o < controls["foi_shuffled"]["test"]["overall"]["wis"] - 1e-9),
    }

    result = {
        "candidate": "mechanism_informed_prediction_intervals",
        "question": (
            "Does SEIR/renewal-state (foi=rt*(1-s), 1-lag leak-free) modulation of the "
            "conformal half-width beat the mechanism-free online conformal on coverage "
            "or WIS, without inflating WIS, and beat TiRex-alone?"
        ),
        "protocol": {
            "split": {k: meta[k] for k in ("n", "pool_end", "test_start", "test_end", "n_test")},
            "point_source": "OFFICIAL frozen TiRex refit_test_predictions (identical point for base & mechanism)",
            "conformal": f"online_conformal_bounds logic (window={PRIMARY_WINDOW}, ki={PRIMARY_KI}); "
                         f"gamma=0 verified byte-identical to shipped helper (maxabs={ident:.2e})",
            "mechanism": "mechanistic_features(full pool+test incidence) 1-lag == FusedEpi._mech_features; "
                         "multiplier m=clip((sig/trailing_ref)^gamma, %.1f, %.1f), ref seeded with train-pool" % (M_LO, M_HI),
            "gamma_tuning": "leak-free: pool-tuned on TiRex rolling residuals (min_ctx..pool_end); "
                            "also parameter-free gamma=1 and first-half/second-half protocol",
            "peak_definition": {"peak_top25pct_threshold": peak_thr,
                                "n_peak": int(masks["peak_top25pct"].sum()),
                                "n_peak_ge50": int(masks["peak_ge50"].sum())},
        },
        "n_test": int(len(y_test)),
        "tirex_oof_wis": oof,
        "baseline_mechfree": base_scores,
        "pool_gamma_grid": pool_grid,
        "pool_best": {"signal": best_signal, "gamma": best_gamma, "foi_gamma": foi_best_gamma},
        "test_mechanism_pooltuned_bestsignal": test_pool_tuned,
        "test_mechanism_foi_pooltuned": test_foi_pooltuned,
        "test_mechanism_foi_parameterfree_g1": test_foi_g1,
        "test_mechanism_foi_upper_asym": test_foi_upper,
        "secondhalf_heldout": secondhalf,
        "fusedepi_point_base": fe_base,
        "fusedepi_point_mechanism": fe_mech,
        "controls_timing_vs_freeknob": controls,
        "verdict": verdict,
        "caveats": [
            "Point forecast is IDENTICAL (frozen TiRex) for baseline and mechanism; only the "
            "interval half-width differs, isolating the mechanism-informed uncertainty contribution.",
            "gamma is tuned on the TRAIN POOL only (TiRex rolling residuals, min_ctx..pool_end) — "
            "the 68-week test is never used to choose gamma. A parameter-free gamma=1 and a "
            "first-half-tuned/second-half-evaluated protocol are reported as leak-free robustness.",
            "The multiplier ALLOWS narrowing (floor %.1f) so width is REALLOCATED across the season "
            "(widen on rising force-of-infection, narrow on decline) rather than uniformly inflated; "
            "mean_width columns document this." % M_LO,
            "Mechanism channels are causal past-only (feature[t] uses incidence[:t]) and 1-lag, "
            "matching FusedEpi's leak-free guard exactly.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2, ensure_ascii=False))
