#!/usr/bin/env python
"""COVERAGE-BRACKET RESOLUTION (standalone, leak-free).

Question: over the 132 rolling origins (weeks 205..336), pid_mech UNDERcovers (0.924)
and cqr OVERcovers (0.985). Can a LEAK-FREE-SELECTABLE construction land PICP95 in the
nominal [0.94, 0.96] with WIS < 2.68?

Design (leak-free):
  * TEST bounds reproduce the multiorigin run EXACTLY (T0=205, refit every 4, bagged
    6-cap GBM, cqr seed on [165,205) from seed-GBM on [52,165)). Reconciled against
    scripts/_dec_boosted_mech_multiorigin.json before anything is trusted.
  * VALIDATION tail = a *separate, strictly-earlier* self-contained construction on
    origins [165,205): its own cqr seed on [125,165) from a seed-GBM on [52,125), its
    own PID warmup and foi seed. Nothing it uses touches the test window [205,336].
    The scalar hyper-parameter (blend w / scale c) is picked ONLY on this past tail to
    target 0.95, then FROZEN and applied to the test.

Approaches:
  (a) convex blend of pid & cqr 95%+all-alpha bounds, weight w picked on val tail.
  (b) coverage-targeted multiplicative half-width scaling of a single base engine,
      factor c picked on val tail to target 0.95.
  (c) Pareto of the 12 existing pid/cqr configs + does ANY single leak-free-selectable
      config land in [0.94,0.96] with WIS<2.68.
  (d) ROBUSTNESS: online-adaptive blend (w chosen from PAST TEST coverage only, like
      PID itself) — keeps ~all origins, still leak-free.

Writes JSON to stdout + scripts/_verify_covbracket.json. Does NOT modify any live/
pipeline code or the existing dec_boosted_mech*.py scripts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (
    FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y,
    load_split, build_features, cqr_offsets, foi_multipliers,
    build_bounds_cqr, build_bounds_pid,
)
from scripts.dec_boosted_mech_multiorigin import CONFIGS, fit_gbm, bagged_qy
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

ALPHAS = list(FLUSIGHT_ALPHAS)
T0_TEST = 205
T0_VAL = 165          # validation origins [165, 205)
REFIT_K = 4
# canonical bracket endpoints (best-WIS config in each family, from the verified grid):
PID_GAMMA = 1.0       # pid_mech g=1.0  -> verified WIS 2.2194 / PICP95 0.9242
CQR_GAMMA = 0.75      # cqr_mech g=0.75 -> verified WIS 2.2221 / PICP95 0.9848
QY_CACHE = D.SCRATCH / "covbracket_qy_full.npz"
NOMINAL_LO, NOMINAL_HI = 0.94, 0.96
WIS_MAX = 2.68


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return round(lo, 4), round(hi, 4)


def build_all_qy(feat_full, tirex_full, r_full, cap, ntot):
    """Bagged GBM quantiles for every origin in [T0_VAL, ntot). Blocks align with the
    T0_TEST=205 run (205-165 divisible by REFIT_K) so origins>=205 are byte-identical."""
    origins = np.arange(T0_VAL, ntot)
    n = len(origins)
    qy = np.zeros((n, len(FQ)), dtype=float)
    for bstart in range(T0_VAL, ntot, REFIT_K):
        bend = min(bstart + REFIT_K, ntot)
        train_end = bstart - K_CAL
        tr_idx = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat_full[tr_idx], r_full[tr_idx], cfg) for cfg in CONFIGS.values()]
        oi = np.arange(bstart, bend)
        qy[oi - T0_VAL] = bagged_qy(gbm, feat_full[oi], tirex_full[oi], cap)
    return origins, qy


def seed_cqr(feat_full, tirex_full, r_full, y_full, cap, cal_start, cal_end):
    """cqr offsets from a seed-GBM trained on [MIN_CTX, cal_start) calibrated on
    [cal_start, cal_end). Strictly past-only w.r.t. every origin >= cal_end."""
    seed_train = np.arange(MIN_CTX, cal_start)
    seed_gbm = [fit_gbm(feat_full[seed_train], r_full[seed_train], cfg) for cfg in CONFIGS.values()]
    cal_idx = np.arange(cal_start, cal_end)
    qy_cal = bagged_qy(seed_gbm, feat_full[cal_idx], tirex_full[cal_idx], cap)
    return cqr_offsets(qy_cal, y_full[cal_idx])


def scores_of(bounds, y, med):
    wis = np.asarray(wis_from_bounds(y, bounds, ALPHAS, median=med), dtype=float)
    lo95, hi95 = bounds[0.05]
    cov = (y >= lo95) & (y <= hi95)
    return wis, cov, (hi95 - lo95)


def summarize(bounds, y, med, mask_dict):
    wis, cov, w95 = scores_of(bounds, y, med)
    out = {}
    for mk, m in mask_dict.items():
        m = np.asarray(m, bool)
        nn = int(m.sum())
        if nn == 0:
            continue
        k = int(cov[m].sum())
        out[mk] = {"wis": round(float(wis[m].mean()), 4),
                   "picp95": round(k / nn, 4), "k_of_n": f"{k}/{nn}",
                   "cp95ci": list(clopper_pearson(k, nn)),
                   "w95": round(float(w95[m].mean()), 2)}
    return out


def blend_bounds(bpid, bcqr, w):
    """Convex blend w*pid + (1-w)*cqr on every alpha's (lo,hi)."""
    out = {}
    for a in ALPHAS:
        plo, phi = bpid[a]
        clo, chi = bcqr[a]
        out[a] = (w * plo + (1 - w) * clo, w * phi + (1 - w) * chi)
    return out


def scale_bounds(bounds, med, c, cap):
    """Multiplicative half-width scaling around the shared median (all alphas)."""
    out = {}
    for a in ALPHAS:
        lo, hi = bounds[a]
        nlo = np.clip(med - c * (med - lo), 0.0, cap)
        nhi = np.clip(med + c * (hi - med), 0.0, cap)
        out[a] = (nlo, nhi)
    return out


def main():
    t0 = time.time()
    X_train, y_train, X_test, y_test, meta = load_split()
    ntr, nte = len(y_train), len(y_test)
    ntot = ntr + nte
    frozen = np.asarray(json.loads(
        (ROOT / "simulation/results/per_model_optimal/TiRex.json").read_text())
        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE)
    tirex_pool = d["tirex_pool"]
    y_full = np.concatenate([y_train, y_test])
    cap = 2.0 * float(np.max(y_full))
    tirex_full = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    feat_full, foi_lag = build_features(y_train, y_test, X_train, X_test, tirex_full)
    r_full = y_full - tirex_full

    # ---- bagged GBM quantiles for [T0_VAL, ntot) (cached) ----
    if QY_CACHE.exists():
        z = np.load(QY_CACHE)
        origins_all, qy_all = z["origins"], z["qy"]
        if len(origins_all) != ntot - T0_VAL:
            origins_all = None
    else:
        origins_all = None
    if origins_all is None:
        origins_all, qy_all = build_all_qy(feat_full, tirex_full, r_full, cap, ntot)
        np.savez(QY_CACHE, origins=origins_all, qy=qy_all)

    def slc(a, b):
        return slice(a - T0_VAL, b - T0_VAL)

    # ================= TEST construction (reproduce multiorigin exactly) =================
    test_sl = slc(T0_TEST, ntot)
    qy_te = qy_all[test_sl]
    y_te = y_full[origins_all[test_sl]]
    foi_te = foi_lag[origins_all[test_sl]]
    med_te = qy_te[:, MED_COL]
    n_te = len(y_te)
    cqr_test = seed_cqr(feat_full, tirex_full, r_full, y_full, cap,
                        T0_TEST - K_CAL, T0_TEST)          # cal [165,205)
    foi_seed_te = foi_lag[MIN_CTX:T0_TEST]
    mult_pid_te = foi_multipliers(foi_te, foi_seed_te, PID_GAMMA)
    mult_cqr_te = foi_multipliers(foi_te, foi_seed_te, CQR_GAMMA)
    b_pid_te = build_bounds_pid(qy_te, cqr_test, y_te, cap, foi_mult=mult_pid_te)
    b_cqr_te = build_bounds_cqr(qy_te, cqr_test, cap, foi_mult=mult_cqr_te)
    # also the g=0 ablation bounds for the Pareto / reconciliation
    b_pid_plain_te = build_bounds_pid(qy_te, cqr_test, y_te, cap, foi_mult=None)
    b_cqr_static_te = build_bounds_cqr(qy_te, cqr_test, cap, foi_mult=None)

    peak = y_te >= PEAK_Y
    last34 = np.zeros(n_te, bool); last34[n_te - 34:] = True
    masks = {"all": np.ones(n_te, bool), "peak": peak, "last34": last34}

    recon = {
        "pid_mech_g1.0": summarize(b_pid_te, y_te, med_te, masks)["all"],
        "cqr_mech_g0.75": summarize(b_cqr_te, y_te, med_te, masks)["all"],
        "pid_plain": summarize(b_pid_plain_te, y_te, med_te, masks)["all"],
        "cqr_static": summarize(b_cqr_static_te, y_te, med_te, masks)["all"],
    }
    # verified anchors (from _dec_boosted_mech_multiorigin.json)
    verified = {"pid_mech_g1.0": (2.2194, 0.9242), "cqr_mech_g0.75": (2.2221, 0.9848),
                "pid_plain": (2.308, 0.9015), "cqr_static": (2.2688, 0.9848)}
    recon_ok = all(abs(recon[k]["wis"] - verified[k][0]) < 5e-3
                   and abs(recon[k]["picp95"] - verified[k][1]) < 5e-3 for k in verified)

    # ================= VALIDATION construction ([165,205), self-contained) ==============
    val_sl = slc(T0_VAL, T0_TEST)
    qy_va = qy_all[val_sl]
    y_va = y_full[origins_all[val_sl]]
    foi_va = foi_lag[origins_all[val_sl]]
    med_va = qy_va[:, MED_COL]
    n_va = len(y_va)
    cqr_val = seed_cqr(feat_full, tirex_full, r_full, y_full, cap,
                       T0_VAL - K_CAL, T0_VAL)             # cal [125,165)
    foi_seed_va = foi_lag[MIN_CTX:T0_VAL]
    mult_pid_va = foi_multipliers(foi_va, foi_seed_va, PID_GAMMA)
    mult_cqr_va = foi_multipliers(foi_va, foi_seed_va, CQR_GAMMA)
    b_pid_va = build_bounds_pid(qy_va, cqr_val, y_va, cap, foi_mult=mult_pid_va)
    b_cqr_va = build_bounds_cqr(qy_va, cqr_val, cap, foi_mult=mult_cqr_va)

    def picp(bounds, y):
        lo, hi = bounds[0.05]
        return float(np.mean((y >= lo) & (y <= hi)))

    def wis_mean(bounds, y, med):
        return float(np.mean(wis_from_bounds(y, bounds, ALPHAS, median=med)))

    val_pid = {"picp95": round(picp(b_pid_va, y_va), 4), "wis": round(wis_mean(b_pid_va, y_va, med_va), 4)}
    val_cqr = {"picp95": round(picp(b_cqr_va, y_va), 4), "wis": round(wis_mean(b_cqr_va, y_va, med_va), 4)}

    # ---------------- (a) convex blend, w picked on val tail to target 0.95 ----------------
    w_grid = np.linspace(0.0, 1.0, 101)
    blend_scan = []
    best = None
    for w in w_grid:
        bv = blend_bounds(b_pid_va, b_cqr_va, w)
        pv, wv = picp(bv, y_va), wis_mean(bv, y_va, med_va)
        blend_scan.append((round(float(w), 3), round(pv, 4), round(wv, 4)))
        key = (abs(pv - 0.95), wv)                 # target 0.95, tie-break lower val WIS
        if best is None or key < best[0]:
            best = (key, float(w), pv, wv)
    w_star = best[1]
    b_blend_te = blend_bounds(b_pid_te, b_cqr_te, w_star)
    blend_test = summarize(b_blend_te, y_te, med_te, masks)
    approach_a = {
        "w_star": round(w_star, 3),
        "val_picp95_at_w": round(best[2], 4), "val_wis_at_w": round(best[3], 4),
        "test": blend_test,
        "in_bracket": bool(NOMINAL_LO <= blend_test["all"]["picp95"] <= NOMINAL_HI),
        "wis_ok": bool(blend_test["all"]["wis"] < WIS_MAX),
    }

    # ---------------- (b) coverage-targeted multiplicative scaling ----------------
    c_grid = np.linspace(0.4, 2.5, 211)

    def pick_c(bval, medval, bte, medte):
        best_c = None
        for c in c_grid:
            bs = scale_bounds(bval, medval, c, cap)
            pv = picp(bs, y_va)
            key = (abs(pv - 0.95), c)               # target 0.95, tie-break smaller c (sharper)
            if best_c is None or key < best_c[0]:
                best_c = (key, float(c), pv)
        cstar = best_c[1]
        bte_scaled = scale_bounds(bte, medte, cstar, cap)
        s = summarize(bte_scaled, y_te, med_te, masks)
        return cstar, round(best_c[2], 4), s

    c_pid, valp_pid, s_pid = pick_c(b_pid_va, med_va, b_pid_te, med_te)      # inflate the undercoverer
    c_cqr, valp_cqr, s_cqr = pick_c(b_cqr_va, med_va, b_cqr_te, med_te)      # deflate the overcoverer
    approach_b = {
        "pid_base": {"c_star": round(c_pid, 3), "val_picp95_at_c": valp_pid, "test": s_pid,
                     "in_bracket": bool(NOMINAL_LO <= s_pid["all"]["picp95"] <= NOMINAL_HI),
                     "wis_ok": bool(s_pid["all"]["wis"] < WIS_MAX)},
        "cqr_base": {"c_star": round(c_cqr, 3), "val_picp95_at_c": valp_cqr, "test": s_cqr,
                     "in_bracket": bool(NOMINAL_LO <= s_cqr["all"]["picp95"] <= NOMINAL_HI),
                     "wis_ok": bool(s_cqr["all"]["wis"] < WIS_MAX)},
    }

    # ---------------- (c) Pareto of existing 12 configs + bracket check ----------------
    grid_json = json.loads((ROOT / "scripts" / "_dec_boosted_mech_multiorigin.json").read_text())["gammas"]
    pareto = []
    for name, r in grid_json.items():
        a = r["all"]
        pareto.append({"config": name, "picp95": a["picp95"], "wis": a["wis"],
                       "in_bracket": bool(NOMINAL_LO <= a["picp95"] <= NOMINAL_HI and a["wis"] < WIS_MAX)})
    pareto.sort(key=lambda x: x["picp95"])
    any_single_in_bracket = any(p["in_bracket"] for p in pareto)

    # ---------------- (d) ROBUSTNESS: online-adaptive blend (past test coverage only) ----------------
    warm = 20
    w_online = np.full(n_te, np.nan)
    for i in range(n_te):
        if i < warm:
            w_online[i] = 0.5
            continue
        # choose w minimizing |past-coverage - 0.95| over test origins [0,i); tie-break lower past WIS
        yb = y_te[:i]; mb = med_te[:i]
        bestk = None
        for w in w_grid:
            bb = blend_bounds({a: (b_pid_te[a][0][:i], b_pid_te[a][1][:i]) for a in ALPHAS},
                              {a: (b_cqr_te[a][0][:i], b_cqr_te[a][1][:i]) for a in ALPHAS}, w)
            lo, hi = bb[0.05]
            pv = float(np.mean((yb >= lo) & (yb <= hi)))
            wv = float(np.mean(wis_from_bounds(yb, bb, ALPHAS, median=mb)))
            key = (abs(pv - 0.95), wv)
            if bestk is None or key < bestk[0]:
                bestk = (key, float(w))
        w_online[i] = bestk[1]
    # apply per-origin online w
    b_online = {}
    for a in ALPHAS:
        plo, phi = b_pid_te[a]; clo, chi = b_cqr_te[a]
        b_online[a] = (w_online * plo + (1 - w_online) * clo,
                       w_online * phi + (1 - w_online) * chi)
    eval_mask = np.zeros(n_te, bool); eval_mask[warm:] = True
    masks_online = {"all_post_warm": eval_mask,
                    "peak_post_warm": peak & eval_mask,
                    "last34": last34}
    online_summ = summarize(b_online, y_te, med_te, masks_online)
    approach_d = {
        "warmup": warm, "n_eval": int(eval_mask.sum()),
        "test": online_summ,
        "in_bracket": bool(NOMINAL_LO <= online_summ["all_post_warm"]["picp95"] <= NOMINAL_HI),
        "wis_ok": bool(online_summ["all_post_warm"]["wis"] < WIS_MAX),
        "w_online_mean": round(float(np.nanmean(w_online[warm:])), 3),
    }

    # ---------------- best achievable + verdict ----------------
    candidates = []
    ba = approach_a
    candidates.append(("a_blend_valtail", ba["test"]["all"]["picp95"], ba["test"]["all"]["wis"],
                       ba["in_bracket"] and ba["wis_ok"], f"w={ba['w_star']}"))
    for tag in ("pid_base", "cqr_base"):
        bb = approach_b[tag]
        candidates.append((f"b_scale_{tag}", bb["test"]["all"]["picp95"], bb["test"]["all"]["wis"],
                           bb["in_bracket"] and bb["wis_ok"], f"c={bb['c_star']}"))
    candidates.append(("d_online_blend", online_summ["all_post_warm"]["picp95"],
                       online_summ["all_post_warm"]["wis"], approach_d["in_bracket"] and approach_d["wis_ok"],
                       f"w_mean={approach_d['w_online_mean']} (n={approach_d['n_eval']})"))

    in_bracket_hits = [c for c in candidates if c[3]]
    # best = prefer in-bracket; among those min WIS; else min |picp-0.95|
    if in_bracket_hits:
        best_cfg = min(in_bracket_hits, key=lambda c: c[2])
    else:
        best_cfg = min(candidates, key=lambda c: (abs(c[1] - 0.95), c[2]))
    nominal_reachable = bool(in_bracket_hits) or any_single_in_bracket

    out = {
        "reconciliation": {"computed": recon, "verified": {k: {"wis": v[0], "picp95": v[1]} for k, v in verified.items()},
                           "match": recon_ok},
        "n_test_origins": n_te, "n_val_origins": n_va,
        "bracket_resolution_note": (f"{n_te} origins -> step={1/n_te:.4f}; "
                                    f"[0.94,0.96] == k in {{{int(np.ceil(NOMINAL_LO*n_te))}.."
                                    f"{int(np.floor(NOMINAL_HI*n_te))}}} = "
                                    f"{[k for k in range(n_te+1) if NOMINAL_LO<=k/n_te<=NOMINAL_HI]}"),
        "val_endpoints": {"pid_mech_g1.0": val_pid, "cqr_mech_g0.75": val_cqr},
        "approach_a_blend": approach_a,
        "approach_b_scale": approach_b,
        "approach_c_pareto": {"configs_sorted_by_picp95": pareto,
                              "any_single_config_in_bracket": any_single_in_bracket},
        "approach_d_online_blend": approach_d,
        "best_config": {"tag": best_cfg[0], "picp95": best_cfg[1], "wis": best_cfg[2],
                        "in_bracket_and_wis_ok": best_cfg[3], "param": best_cfg[4]},
        "nominal_0p95_leakfree_reachable": nominal_reachable,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    outp = ROOT / "scripts" / "_verify_covbracket.json"
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
