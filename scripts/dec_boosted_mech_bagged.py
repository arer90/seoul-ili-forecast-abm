#!/usr/bin/env python
"""Decisive-win iteration: BAG the residual-GBM across capacities + COVERAGE-AWARE
leak-free selection — the two highest-leverage fixes the adversarial verifier named
for turning the boosted-mech knife-edge (bar-2 coverage) into a robust win.

Fix 1 (bagging): the whole "not decisive" verdict hinged on a 0.912<->0.941 coverage
   variance ACROSS 6 GBM capacities. Averaging the quantile predictions over all 6
   removes the "which capacity" degree of freedom entirely (no capacity is selected).
Fix 2 (coverage-aware selection): the original pool-val objective was pure WIS
   (near-flat, ignores coverage). Here we keep only strategy+gamma combos whose
   POOL-VAL PICP95 >= cov_target (buffer above 0.93 to absorb the n=68 quantization
   gap), then argmin WIS. Fully leak-free (val = pool tail, no test peek).

Reports both selection rules + Clopper-Pearson CI on the final test PICP95, so the
claim is honest whatever the outcome. Deterministic; reuses the frozen TiRex roll.
Does NOT modify any live/pipeline code — imports helpers from dec_boosted_mech only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (
    FQ, MED_COL, MIN_CTX, K_CAL, K_VAL, PEAK_Y, GAMMAS,
    load_split, build_features, predict_qy, cqr_offsets, foi_multipliers,
    build_bounds_cqr, build_bounds_pid, wis_overall, score,
)

# same 6 capacities the verifier probed — we BAG over all of them (none selected)
CONFIGS = {
    "default": dict(lr=0.05, it=300, leaves=8, msl=15, depth=3),
    "weak":    dict(lr=0.03, it=150, leaves=4, msl=25, depth=2),
    "strong":  dict(lr=0.08, it=500, leaves=16, msl=10, depth=4),
    "shallow": dict(lr=0.05, it=300, leaves=4, msl=20, depth=2),
    "deep":    dict(lr=0.05, it=400, leaves=31, msl=8, depth=None),
    "lowlr":   dict(lr=0.02, it=600, leaves=8, msl=15, depth=3),
}


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


def bagged_qy(gbm_dicts, X, tirex, cap):
    """Mean quantile matrix across the fitted capacity ensemble (bar-preserving:
    averaging per-level quantiles keeps monotonicity since all are sorted the same)."""
    stack = np.stack([predict_qy(g, X, tirex, cap) for g in gbm_dicts], axis=0)
    qy = stack.mean(axis=0)
    return np.sort(qy, axis=1)  # enforce non-crossing after averaging


def val_picp95(bounds, y_val):
    lo, hi = bounds[0.05]
    return float(np.mean((y_val >= lo) & (y_val <= hi)))


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def run(cov_target: float):
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

    npool = len(yp)
    rp = yp - tp
    v_cut = npool - K_VAL
    c_cut = v_cut - K_CAL

    # ---- pool selection with BAGGED quantiles ----
    gbm_sel = [fit_gbm(Xp[:c_cut], rp[:c_cut], cfg) for cfg in CONFIGS.values()]
    qy_cal_v = bagged_qy(gbm_sel, Xp[c_cut:v_cut], tp[c_cut:v_cut], cap)
    cqr_v = cqr_offsets(qy_cal_v, yp[c_cut:v_cut])
    qy_val = bagged_qy(gbm_sel, Xp[v_cut:], tp[v_cut:], cap)
    y_val = yp[v_cut:]; foi_val = foi_pool[v_cut:]; foi_val_seed = foi_pool[:v_cut]
    med_val = qy_val[:, MED_COL]

    combos = []  # (name, gamma, engine)
    for g in [0.0]:
        combos.append(("cqr_static", g, "cqr"))
        combos.append(("pid", g, "pid"))
    for g in GAMMAS:
        if g > 0:
            combos.append(("cqr_mech", g, "cqr"))
            combos.append(("pid_mech", g, "pid"))

    scored = []
    for name, g, engine in combos:
        mult = foi_multipliers(foi_val, foi_val_seed, g) if g > 0 else None
        b = (build_bounds_cqr(qy_val, cqr_v, cap, foi_mult=mult) if engine == "cqr"
             else build_bounds_pid(qy_val, cqr_v, y_val, cap, foi_mult=mult))
        scored.append({"name": name, "g": g, "engine": engine,
                       "val_wis": wis_overall(b, y_val, med_val),
                       "val_picp95": val_picp95(b, y_val)})

    pure_wis = min(scored, key=lambda s: s["val_wis"])
    eligible = [s for s in scored if s["val_picp95"] >= cov_target]
    cov_aware = (min(eligible, key=lambda s: s["val_wis"]) if eligible else pure_wis)

    # ---- fit the FINAL bagged ensemble ONCE, reuse for all combo evals ----
    f_cut = npool - K_CAL
    gbm_f = [fit_gbm(Xp[:f_cut], rp[:f_cut], cfg) for cfg in CONFIGS.values()]
    cqr_f_shared = cqr_offsets(bagged_qy(gbm_f, Xp[f_cut:], tp[f_cut:], cap), yp[f_cut:])
    qy_te_shared = bagged_qy(gbm_f, Xt, frozen, cap)

    def test_cov_of(sel):
        """test full68 WIS + PICP95(k/68) for a given strategy+gamma — for the
        knife-edge spread check over the near-tied val combos."""
        med_te = qy_te_shared[:, MED_COL]
        mult = foi_multipliers(foi_te, foi_pool, sel["g"]) if sel["g"] > 0 else None
        if sel["engine"] == "cqr":
            b = build_bounds_cqr(qy_te_shared, cqr_f_shared, cap, foi_mult=mult)
        else:
            b = build_bounds_pid(qy_te_shared, cqr_f_shared, y_test, cap, foi_mult=mult)
        sc = score(b, y_test, med_te, masks)["overall_68"]
        k = int(round(sc["picp95"] * sc["n"]))
        return {"name": sel["name"], "g": sel["g"], "val_wis": round(sel["val_wis"], 4),
                "test_wis": round(sc["wis"], 4), "test_k_of_68": f"{k}/{sc['n']}",
                "test_picp95": round(sc["picp95"], 4)}

    tied = sorted(scored, key=lambda s: s["val_wis"])[:8]
    spread = [test_cov_of(s) for s in tied]

    def final_eval(sel):
        cqr_f = cqr_f_shared
        qy_te = qy_te_shared
        med_te = qy_te[:, MED_COL]
        mult = foi_multipliers(foi_te, foi_pool, sel["g"]) if sel["g"] > 0 else None
        if sel["engine"] == "cqr":
            b = build_bounds_cqr(qy_te, cqr_f, cap, foi_mult=mult)
        else:
            b = build_bounds_pid(qy_te, cqr_f, y_test, cap, foi_mult=mult)
        sc = score(b, y_test, med_te, masks)
        full, l34 = sc["overall_68"], sc["last34"]
        k = int(round(full["picp95"] * full["n"]))
        cp = clopper_pearson(k, full["n"])
        return {
            "selected": f"{sel['name']} (g={sel['g']})",
            "val_wis": round(sel["val_wis"], 4), "val_picp95": round(sel["val_picp95"], 3),
            "full68_wis": round(full["wis"], 4), "full68_picp95": round(full["picp95"], 4),
            "full68_k_of_n": f"{k}/{full['n']}",
            "full68_picp95_CI": [round(cp[0], 3), round(cp[1], 3)],
            "peak_y50_picp95": round(sc["peak_y50"]["picp95"], 3),
            "last34_wis": round(l34["wis"], 4), "last34_picp95": round(l34["picp95"], 3),
            "mean_width95": round(full["mean_width95"], 2),
            "bars": {"1_wis68<2.68": bool(full["wis"] < 2.68),
                     "2_picp95>=0.93": bool(full["picp95"] >= 0.93),
                     "3_last34<2.72": bool(l34["wis"] < 2.720)},
        }

    return {
        "cov_target": cov_target,
        "pure_wis_selection": final_eval(pure_wis),
        "coverage_aware_selection": final_eval(cov_aware),
        "n_eligible_combos": len(eligible),
        "val_grid_top": sorted(scored, key=lambda s: s["val_wis"])[:6],
        "tied_combo_test_spread": spread,
    }


def main():
    out = {}
    for tgt in (0.95, 0.97):
        r = run(tgt)
        out[f"cov_target_{tgt}"] = r
        print(f"\n===== BAGGED (6-capacity) · coverage-aware target={tgt} =====")
        for key in ("pure_wis_selection", "coverage_aware_selection"):
            e = r[key]
            allbars = all(e["bars"].values())
            print(f"  [{key}] {e['selected']}")
            print(f"    val: WIS={e['val_wis']} PICP95={e['val_picp95']}  ->  "
                  f"TEST full68 WIS={e['full68_wis']} PICP95={e['full68_picp95']} "
                  f"({e['full68_k_of_n']}, CP95%CI={e['full68_picp95_CI']}) "
                  f"peak={e['peak_y50_picp95']} | last34 WIS={e['last34_wis']} "
                  f"PICP95={e['last34_picp95']} | W95={e['mean_width95']}")
            print(f"    bars {e['bars']}  ->  ALL-3: {allbars}")
        print("    --- knife-edge check: test coverage of the near-tied val combos ---")
        ks = set()
        for s in r["tied_combo_test_spread"]:
            ks.add(s["test_k_of_68"])
            print(f"      {s['name']:>11s} g={s['g']:<4} valWIS={s['val_wis']:.4f} -> "
                  f"testWIS={s['test_wis']:.4f} PICP95={s['test_picp95']} ({s['test_k_of_68']})")
        print(f"    => distinct test k/68 among tied combos: {sorted(ks)} "
              f"{'(STRADDLES 0.93 → knife-edge relocated)' if any('63' in k for k in ks) and any('64' in k for k in ks) else ''}")
    outp = ROOT / "scripts" / "_dec_boosted_mech_bagged.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    raise SystemExit(main())
