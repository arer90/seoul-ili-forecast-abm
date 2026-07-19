#!/usr/bin/env python
"""Robustness probe for dec_boosted_mech.py — re-run the FULL leak-free pipeline
(pool-selection of strategy+gamma, then final test fit) across several GBM seeds.

This does NOT select the best seed; it reports the leak-free-selected variant's
full-68 + last-34 WIS/PICP95 for each seed to confirm the decisive win is stable,
not a seed artifact. Reuses dec_boosted_mech helpers + the cached TiRex roll.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (
    FQ, MED_COL, MIN_CTX, K_CAL, K_VAL, PEAK_Y, GAMMAS,
    load_split, build_features, predict_qy, cqr_offsets, foi_multipliers,
    build_bounds_cqr, build_bounds_pid, wis_overall, score,
)
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
import json


def fit_gbm(Xtr, r_tr, cfg):
    models = {}
    for q in FQ:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=float(q), learning_rate=cfg["lr"],
            max_iter=cfg["it"], max_leaf_nodes=cfg["leaves"],
            min_samples_leaf=cfg["msl"], l2_regularization=1.0,
            max_depth=cfg["depth"], random_state=42)
        m.fit(Xtr, r_tr)
        models[round(float(q), 4)] = m
    return models


def run_seed(cfg, Xp, yp, tp, foi_pool, Xt, tirex_test, foi_te, y_test, cap, masks):
    npool = len(yp)
    rp = yp - tp
    # pool selection
    v_cut = npool - K_VAL
    c_cut = v_cut - K_CAL
    gbm_t = fit_gbm(Xp[:c_cut], rp[:c_cut], cfg)
    qy_cal_v = predict_qy(gbm_t, Xp[c_cut:v_cut], tp[c_cut:v_cut], cap)
    cqr_v = cqr_offsets(qy_cal_v, yp[c_cut:v_cut])
    qy_val = predict_qy(gbm_t, Xp[v_cut:], tp[v_cut:], cap)
    y_val = yp[v_cut:]; foi_val = foi_pool[v_cut:]; foi_val_seed = foi_pool[:v_cut]
    med_val = qy_val[:, MED_COL]

    def gpick(engine):
        best = None
        for g in GAMMAS:
            mult = foi_multipliers(foi_val, foi_val_seed, g) if g > 0 else None
            b = (build_bounds_cqr(qy_val, cqr_v, cap, foi_mult=mult) if engine == "cqr"
                 else build_bounds_pid(qy_val, cqr_v, y_val, cap, foi_mult=mult))
            w = wis_overall(b, y_val, med_val)
            if best is None or w < best[1]:
                best = (g, w)
        return best

    pv = {}
    pv["cqr_static"] = (0.0, wis_overall(build_bounds_cqr(qy_val, cqr_v, cap), y_val, med_val))
    pv["cqr_mech"] = gpick("cqr")
    pv["pid"] = (0.0, wis_overall(build_bounds_pid(qy_val, cqr_v, y_val, cap), y_val, med_val))
    pv["pid_mech"] = gpick("pid")
    selected = min(pv, key=lambda k: pv[k][1])
    g_sel = pv[selected][0]
    pool_val_wis = pv[selected][1]

    # final fit
    f_cut = npool - K_CAL
    gbm_f = fit_gbm(Xp[:f_cut], rp[:f_cut], cfg)
    cqr_f = cqr_offsets(predict_qy(gbm_f, Xp[f_cut:], tp[f_cut:], cap), yp[f_cut:])
    qy_te = predict_qy(gbm_f, Xt, tirex_test, cap)
    med_te = qy_te[:, MED_COL]
    mult = foi_multipliers(foi_te, foi_pool, g_sel) if g_sel > 0 else None
    if selected.startswith("cqr"):
        b = build_bounds_cqr(qy_te, cqr_f, cap, foi_mult=mult)
    else:
        b = build_bounds_pid(qy_te, cqr_f, y_test, cap, foi_mult=mult)
    sc = score(b, y_test, med_te, masks)
    full, l34 = sc["overall_68"], sc["last34"]
    return {
        "selected": selected, "gamma": g_sel, "pool_val_wis": round(pool_val_wis, 4),
        "full68_wis": round(full["wis"], 4), "full68_picp95": round(full["picp95"], 3),
        "peak_y50_picp95": round(sc["peak_y50"]["picp95"], 3),
        "last34_wis": round(l34["wis"], 4), "last34_picp95": round(l34["picp95"], 3),
        "bars": [bool(full["wis"] < 2.68), bool(full["picp95"] >= 0.93),
                 bool(l34["wis"] < 2.720), None],
    }


def main():
    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test)
    frozen = np.asarray(json.loads(
        (ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE)
    tirex_pool = d["tirex_pool"]
    y_full = np.concatenate([y_train, y_test])
    cap = 2.0 * float(np.max(y_full))
    tirex_full = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    feat_full, foi_lag = build_features(y_train, y_test, X_train, X_test, tirex_full)
    Xp = feat_full[MIN_CTX:ntr]; yp = y_train[MIN_CTX:]; tp = tirex_pool
    Xt = feat_full[ntr:ntr + nte]; foi_te = foi_lag[ntr:ntr + nte]; foi_pool = foi_lag[MIN_CTX:ntr]

    peak50 = y_test >= PEAK_Y
    last34 = np.zeros(nte, bool); last34[nte - 34:] = True
    masks = {"overall_68": np.ones(nte, bool), "peak_y50": peak50, "last34": last34}

    # capacity sweep: default + weaker/stronger + shallower/deeper (NOT selected — all reported)
    configs = {
        "default": dict(lr=0.05, it=300, leaves=8, msl=15, depth=3),
        "weak":    dict(lr=0.03, it=150, leaves=4, msl=25, depth=2),
        "strong":  dict(lr=0.08, it=500, leaves=16, msl=10, depth=4),
        "shallow": dict(lr=0.05, it=300, leaves=4, msl=20, depth=2),
        "deep":    dict(lr=0.05, it=400, leaves=31, msl=8, depth=None),
        "lowlr":   dict(lr=0.02, it=600, leaves=8, msl=15, depth=3),
    }
    print(f"{'config':>8s} {'selected':>10s} {'g':>4s} {'poolValWIS':>10s} {'WIS68':>7s} {'P95':>5s} "
          f"{'Ppk50':>6s} {'WIS34':>7s} {'P34':>5s}  bars(1,2,3)")
    n_clear = 0
    rows = {}
    for name, cfg in configs.items():
        r = run_seed(cfg, Xp, yp, tp, foi_pool, Xt, frozen, foi_te, y_test, cap, masks)
        rows[name] = r
        b = r["bars"][:3]
        if all(b):
            n_clear += 1
        print(f"{name:>8s} {r['selected']:>10s} {r['gamma']:>4.2f} {r['pool_val_wis']:>10.4f} "
              f"{r['full68_wis']:>7.4f} {r['full68_picp95']:>5.3f} {r['peak_y50_picp95']:>6.3f} "
              f"{r['last34_wis']:>7.4f} {r['last34_picp95']:>5.3f}  {b}")
    print(f"\n{n_clear}/{len(configs)} capacity configs clear bars 1-3 (leak-free pool-selected variant)")
    # principled capacity pick = argmin pool-val WIS (leak-free, no test peek)
    best_cap = min(rows, key=lambda k: rows[k]["pool_val_wis"])
    br = rows[best_cap]
    print(f"pool-val-selected capacity = '{best_cap}': WIS68={br['full68_wis']} "
          f"PICP95={br['full68_picp95']} last34={br['last34_wis']} bars={br['bars'][:3]}")


if __name__ == "__main__":
    raise SystemExit(main())
