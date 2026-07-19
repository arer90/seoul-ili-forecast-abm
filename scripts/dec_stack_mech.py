#!/usr/bin/env python
"""NEW FUSEDEPI v2 — held-out-tuned inverse-RMSE POINT stack ⊕ MECHANISM-INFORMED PI.

Idea
----
Two verified ingredients are combined into one candidate:

  (A) POINT forecast = the held-out-tuned INVERSE-RMSE stack of the 4 strong base
      models {TiRex, NegBinGLM, TabPFN, ARIMA}. Weights w_i ∝ 1/RMSE_i are learned
      on the FIRST 34 test weeks only (Protocol B, pseudo-validation) and FROZEN.
      This is the EXACT stack that scored WIS 2.7205 on the last-34 in
      scripts/fusedepi_fusion_wis.py (beating FusedEpi's own point) — reproduced
      here as the gamma=0 reference.

  (B) INTERVAL = the mechanism-informed conformal width + asymmetric peak-widener
      from scripts/nov_mechanism_pi.py. The half-width at week i is scaled by
      m[i] = clip((foi[i]/trailing_ref)^gamma, 0.5, 2.5), where foi = rt*(1-s) is
      the leak-free 1-lag renewal force-of-infection (rises AHEAD of the surge).
      side="upper" widens only the UPPER bound (all 95% misses are under-predictions
      on the rising limb) = the asymmetric peak-widener.

Question
--------
Does a STRONGER point (the stack, 2.720 vs FusedEpi) + mechanism-informed coverage
clear ALL FOUR decisive bars?
  (1) full-68 WIS < 2.68           (below quantile-fusion 2.677 AND the stack 2.720)
  (2) PICP95 >= 0.93 (ideally .95), esp. peak weeks y>=50
  (3) the gain SURVIVES the truly-unseen last-34 (Protocol B), not just full-68
  (4) NOT a width artifact: uniform widening to the SAME mean width must NOT match
      the WIS gain.

Leak-free protocol
------------------
* Frozen run_data split (pool_end=269, n_test=68) via scripts.ablation_fusedepi.load_split.
* POINT: base preds are OFFICIAL frozen refit_test_predictions; stack weights learned
  on FIRST 34 test weeks only, frozen for all 68.
* GAMMA (mechanism strength): tuned ONLY on the TRAIN POOL (TiRex rolling residuals,
  min_ctx..pool_end) — the 68-week test never chooses gamma. Also reported: a
  parameter-free gamma=1.0, and a first-34-tuned gamma for the pure Protocol-B block.
* foi channel is causal past-only 1-lag (feature[t] uses incidence[:t]) == FusedEpi guard.
* Reports BOTH full-68 (first-34 in-sample for the point weights — disclosed) AND the
  truly-unseen last-34 (Protocol B: neither weights nor gamma saw it).

No live pipeline/model code modified. Reuses the verified helpers from
nov_mechanism_pi.py / fusedepi_fusion_wis.py. Writes one JSON.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

from scripts.ablation_fusedepi import load_split
from scripts.fusedepi_fusion_wis import load_frozen
from scripts.nov_mechanism_pi import (
    build_signals,
    mechanism_online_conformal,
    score_bounds,
    tirex_pool_rolling,
)
from simulation.analytics.adaptive_conformal import online_conformal_bounds, wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS

OUT = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/novelty/dec_stack_mech.json"
)

BASE_MODELS = ["TiRex", "NegBinGLM", "TabPFN", "ARIMA"]
WINDOW = 30
KI = 0.2
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
PEAK_Y = 50.0


def invrmse_stack_point(preds: dict, y_true: np.ndarray, cut: int) -> tuple[np.ndarray, np.ndarray]:
    """Held-out-tuned inverse-RMSE POINT stack (Protocol B, EXACT 2.720 recipe).

    weights_i ∝ 1/RMSE_i where RMSE_i is computed on the FIRST `cut` test weeks only;
    applied (frozen) to the full 68-week base prediction matrix.

    Returns: (stack_point (n,), weights (4,)).
    """
    X = np.column_stack([preds[m] for m in BASE_MODELS])           # (n, 4)
    Xf, yf = X[:cut], y_true[:cut]
    fit_rmse = np.array([np.sqrt(np.mean((yf - Xf[:, i]) ** 2)) for i in range(len(BASE_MODELS))])
    inv_w = (1.0 / fit_rmse) / np.sum(1.0 / fit_rmse)
    return X @ inv_w, inv_w


def uniform_widen(bounds: dict, pred: np.ndarray, factor: float) -> dict:
    """Multiply every half-width (both bounds) by a constant `factor` (width-artifact control).

    A time-UNIFORM widener: no reallocation across weeks. Used to test whether the
    mechanism gain is merely 'more width' vs 'width in the right weeks'.
    """
    pred = np.asarray(pred, float).ravel()
    out = {}
    for a, (lo, hi) in bounds.items():
        half_lo = pred - lo
        half_hi = hi - pred
        out[a] = (np.maximum(0.0, pred - factor * half_lo), pred + factor * half_hi)
    return out


def match_width_factor(base_bounds: dict, mech_bounds: dict, pred: np.ndarray,
                       mask: np.ndarray) -> float:
    """Scalar c so uniformly-widened base matches mech mean 95%-width on `mask`."""
    m = np.asarray(mask, bool)
    lo_b, hi_b = base_bounds[0.05]
    lo_m, hi_m = mech_bounds[0.05]
    w_base = float(np.mean((hi_b - lo_b)[m]))
    w_mech = float(np.mean((hi_m - lo_m)[m]))
    return w_mech / w_base if w_base > 1e-9 else 1.0


def tune_gamma_pool(tx_pool, y_pool, foi_pool, foi_pool_seed, side) -> tuple[float, dict]:
    """Leak-free gamma tuning on the TRAIN POOL (min pool WIS). Returns (best_gamma, grid)."""
    grid = {}
    for g in GAMMAS:
        bp = mechanism_online_conformal(tx_pool, y_pool, FLUSIGHT_ALPHAS, foi_pool,
                                        gamma=g, seed_signal=foi_pool_seed, side=side)
        grid[g] = float(score_bounds(y_pool, bp, tx_pool, np.ones(len(y_pool), bool))["wis"])
    best = float(min(GAMMAS, key=lambda g: grid[g]))
    return best, grid


def tune_gamma_first34(stack_point, y_test, foi_test, foi_seed, first_mask, side) -> tuple[float, dict]:
    """Leak-free Protocol-B gamma tuning: min WIS on the FIRST 34 test weeks (stack residuals)."""
    grid = {}
    for g in GAMMAS:
        bb = mechanism_online_conformal(stack_point, y_test, FLUSIGHT_ALPHAS, foi_test,
                                        gamma=g, seed_signal=foi_seed, side=side)
        grid[g] = float(score_bounds(y_test, bb, stack_point, first_mask)["wis"])
    best = float(min(GAMMAS, key=lambda g: grid[g]))
    return best, grid


def main() -> dict:
    X_train, y_train, X_test, y_test, meta = load_split()
    n = len(y_test)
    cut = n // 2                                                   # 34

    # ── frozen base predictions + shared y_true ──
    preds, oof = {}, {}
    y_true = None
    for m in BASE_MODELS + ["FusedEpi"]:
        p, yt, ow = load_frozen(m)
        preds[m] = p
        oof[m] = ow
        if y_true is None:
            y_true = yt
        else:
            assert np.max(np.abs(yt - y_true)) < 1e-9, f"y_true mismatch {m}"
    assert np.max(np.abs(y_true - y_test)) < 1e-9, "frozen y_true != split y_test"

    # ── (A) POINT: held-out-tuned inverse-RMSE stack (2.720 recipe) ──
    stack_point, inv_w = invrmse_stack_point(preds, y_true, cut)

    # ── mechanism signal: leak-free 1-lag foi = rt*(1-s), sliced to test & seed ──
    mech_lag, ts, ch = build_signals(y_train, y_test)
    foi_test = ch["foi"][ts:ts + n]
    foi_seed = ch["foi"][:ts]

    # ── masks ──
    peak_thr = float(np.quantile(y_test, 0.75))
    last_mask = np.zeros(n, bool); last_mask[cut:] = True
    first_mask = np.zeros(n, bool); first_mask[:cut] = True
    last_peak_thr = float(np.quantile(y_test[cut:], 0.75))
    masks_full = {
        "overall_68": np.ones(n, bool),
        "peak_ge50": y_test >= PEAK_Y,
        "peak_top25pct": y_test >= peak_thr,
    }
    masks_last34 = {
        "last34_overall": last_mask,
        "last34_peak_ge50": last_mask & (y_test >= PEAK_Y),
        "last34_peak_top25pct": last_mask & (y_test >= last_peak_thr),
    }
    all_masks = {**masks_full, **masks_last34}

    def eval_all(bounds, pred):
        return {mk: score_bounds(y_test, bounds, pred, mv) for mk, mv in all_masks.items()}

    # ── gamma=0 reference (stack point + plain online conformal) ──
    b_ship = online_conformal_bounds(stack_point, y_test, FLUSIGHT_ALPHAS, window=WINDOW, ki=KI)
    b_base = mechanism_online_conformal(stack_point, y_test, FLUSIGHT_ALPHAS, foi_test, gamma=0.0)
    base_ident = max(
        max(float(np.max(np.abs(b_ship[a][0] - b_base[a][0]))) for a in FLUSIGHT_ALPHAS),
        max(float(np.max(np.abs(b_ship[a][1] - b_base[a][1]))) for a in FLUSIGHT_ALPHAS),
    )
    base_scores = eval_all(b_base, stack_point)

    # ── (B) gamma tuning: leak-free on TRAIN POOL (TiRex rolling residuals) ──
    tx_pool, min_ctx = tirex_pool_rolling(y_train)
    y_pool = y_train[min_ctx:]
    foi_pool = ch["foi"][min_ctx:ts]
    foi_pool_seed = ch["foi"][:min_ctx]
    gamma_pool_sym, pool_grid_sym = tune_gamma_pool(tx_pool, y_pool, foi_pool, foi_pool_seed, "sym")
    gamma_pool_up, pool_grid_up = tune_gamma_pool(tx_pool, y_pool, foi_pool, foi_pool_seed, "upper")

    # ── build the candidate variants (all leak-free gamma) ──
    def mech_bounds(gamma, side):
        return mechanism_online_conformal(stack_point, y_test, FLUSIGHT_ALPHAS, foi_test,
                                          gamma=gamma, seed_signal=foi_seed, side=side)

    variants = {
        "v2_sym_pooltuned":  (gamma_pool_sym, "sym"),
        "v2_upper_pooltuned": (gamma_pool_up, "upper"),   # ← mandated: mechanism width + asym peak-widener
        "v2_sym_g1":         (1.0, "sym"),
        "v2_upper_g1":       (1.0, "upper"),
    }
    variant_scores = {}
    variant_bounds = {}
    for name, (g, side) in variants.items():
        bb = mech_bounds(g, side)
        variant_bounds[name] = bb
        variant_scores[name] = {"gamma": g, "side": side, "scores": eval_all(bb, stack_point)}

    # ── PRIMARY v2 = the mandated config (mechanism width + asymmetric peak-widener) ──
    PRIMARY = "v2_upper_pooltuned"
    prim_bounds = variant_bounds[PRIMARY]
    prim_scores = variant_scores[PRIMARY]["scores"]
    prim_gamma = variant_scores[PRIMARY]["gamma"]

    # ── (4) width-artifact control: uniformly widen base to PRIMARY's full-68 mean 95% width ──
    c_full = match_width_factor(b_base, prim_bounds, stack_point, masks_full["overall_68"])
    b_uniform = uniform_widen(b_base, stack_point, c_full)
    uniform_scores = eval_all(b_uniform, stack_point)

    # ── pure Protocol-B block: gamma tuned on FIRST 34 (stack), evaluated on LAST 34 ──
    gB_up, gridB_up = tune_gamma_first34(stack_point, y_test, foi_test, foi_seed, first_mask, "upper")
    gB_sym, gridB_sym = tune_gamma_first34(stack_point, y_test, foi_test, foi_seed, first_mask, "sym")
    protoB = {}
    for tag, (g, side) in {"upper_first34tuned": (gB_up, "upper"),
                           "sym_first34tuned": (gB_sym, "sym")}.items():
        bb = mech_bounds(g, side)
        protoB[tag] = {
            "gamma": g, "side": side,
            "last34_overall": score_bounds(y_test, bb, stack_point, masks_last34["last34_overall"]),
            "last34_peak_ge50": score_bounds(y_test, bb, stack_point, masks_last34["last34_peak_ge50"]),
            "last34_peak_top25pct": score_bounds(y_test, bb, stack_point, masks_last34["last34_peak_top25pct"]),
        }

    # ── FusedEpi native official reference point under same wrapper (for context) ──
    fe_base = eval_all(online_conformal_bounds(preds["FusedEpi"], y_test, FLUSIGHT_ALPHAS,
                                               window=WINDOW, ki=KI), preds["FusedEpi"])

    # ─────────────────────────── DECISIVE-WIN BARS ───────────────────────────
    p_full_wis = prim_scores["overall_68"]["wis"]
    p_full_picp = prim_scores["overall_68"]["picp95"]
    p_peak50_picp = prim_scores["peak_ge50"]["picp95"]
    p_peak25_picp = prim_scores["peak_top25pct"]["picp95"]
    p_l34_wis = prim_scores["last34_overall"]["wis"]
    p_l34_picp = prim_scores["last34_overall"]["picp95"]
    p_l34_peak50_picp = prim_scores["last34_peak_ge50"]["picp95"]

    base_full_wis = base_scores["overall_68"]["wis"]
    base_l34_wis = base_scores["last34_overall"]["wis"]
    uniform_full_wis = uniform_scores["overall_68"]["wis"]

    # gain from mechanism (vs stack+plain conformal); width-artifact gain (uniform, same width)
    gain_mech = base_full_wis - p_full_wis
    gain_uniform = base_full_wis - uniform_full_wis

    bar1 = bool(p_full_wis < 2.68)
    bar2 = bool(p_full_picp >= 0.93 and p_peak50_picp >= 0.93)
    bar3_survives = bool(p_l34_wis < 2.68 and p_l34_picp >= 0.93 and p_l34_peak50_picp >= 0.93)
    # bar4 cleared if mechanism WIS is materially better than uniform-widen-to-same-width
    # (i.e. uniform widening does NOT reproduce the WIS gain — reallocation matters)
    bar4_not_width_artifact = bool(p_full_wis < uniform_full_wis - 1e-6)

    decisive_win = bool(bar1 and bar2 and bar3_survives and bar4_not_width_artifact)

    verdict = {
        "PRIMARY_variant": PRIMARY,
        "primary_gamma": prim_gamma,
        "primary_side": "upper",
        "reference_stack_gamma0_full68_wis": base_full_wis,
        "reference_stack_gamma0_last34_wis": base_l34_wis,
        "bar1_full68_wis_lt_2p68": {"value": p_full_wis, "cleared": bar1},
        "bar2_picp95_ge_0p93": {
            "overall_picp95": p_full_picp,
            "peak_ge50_picp95": p_peak50_picp,
            "peak_top25pct_picp95": p_peak25_picp,
            "cleared": bar2,
        },
        "bar3_survives_last34": {
            "last34_wis": p_l34_wis,
            "last34_picp95": p_l34_picp,
            "last34_peak_ge50_picp95": p_l34_peak50_picp,
            "cleared": bar3_survives,
        },
        "bar4_not_width_artifact": {
            "mechanism_full68_wis": p_full_wis,
            "uniform_widen_same_width_full68_wis": uniform_full_wis,
            "uniform_widen_factor": c_full,
            "gain_from_mechanism": gain_mech,
            "gain_from_uniform_widen": gain_uniform,
            "cleared": bar4_not_width_artifact,
        },
        "DECISIVE_WIN_all_4_bars": decisive_win,
    }

    result = {
        "candidate": "FusedEpi_v2 = invRMSE-stack POINT ⊕ mechanism-informed conformal (upper asym)",
        "question": "Does the stronger stack point + mechanism-informed coverage clear all 4 decisive bars?",
        "protocol": {
            "split": {k: meta[k] for k in ("n", "pool_end", "test_start", "test_end", "n_test")},
            "point": "held-out-tuned inverse-RMSE stack of {TiRex,NegBinGLM,TabPFN,ARIMA}; "
                     "weights ∝ 1/RMSE on FIRST 34 test weeks, frozen (== 2.720 recipe)",
            "interval": f"mechanism_online_conformal (window={WINDOW}, ki={KI}); "
                        f"m=clip((foi/trailing_ref)^gamma,0.5,2.5); side=upper asym peak-widener; "
                        f"gamma0 byte-identical to shipped helper (maxabs={base_ident:.2e})",
            "gamma_tuning": "leak-free TRAIN POOL (TiRex rolling min_ctx..pool_end) min WIS; "
                            "also parameter-free gamma=1.0 and first-34-tuned (Protocol B)",
            "foi": "mechanistic_features 1-lag == FusedEpi._mech_features (causal past-only)",
            "peak_definition": {"peak_top25pct_threshold": peak_thr, "peak_ge50": PEAK_Y,
                                "n_peak_ge50": int(masks_full["peak_ge50"].sum()),
                                "n_peak_top25pct": int(masks_full["peak_top25pct"].sum()),
                                "last34_peak_top25pct_threshold": last_peak_thr,
                                "n_last34_peak_ge50": int(masks_last34["last34_peak_ge50"].sum())},
        },
        "n_test": n,
        "stack_weights": {m: round(float(w), 4) for m, w in zip(BASE_MODELS, inv_w)},
        "base_oof_wis": oof,
        "gamma_pool_grid_sym": pool_grid_sym,
        "gamma_pool_grid_upper": pool_grid_up,
        "gamma_pool_best": {"sym": gamma_pool_sym, "upper": gamma_pool_up},
        "reference_stack_gamma0": base_scores,
        "variants": variant_scores,
        "width_artifact_control_uniform": uniform_scores,
        "protocolB_first34tuned": protoB,
        "fusedepi_point_gamma0_reference": fe_base,
        "verdict": verdict,
        "caveats": [
            "POINT = held-out (first-34) tuned inverse-RMSE stack, frozen for all 68 — the exact "
            "recipe that scored WIS 2.7205 on last-34 in fusedepi_fusion_wis.py (reproduced as gamma=0).",
            "gamma tuned on the TRAIN POOL only (TiRex rolling residuals) — the 68-wk test never "
            "chooses gamma. A parameter-free gamma=1 and a first-34-tuned Protocol-B gamma are also reported.",
            "full-68 numbers include the first-34 weeks that the POINT weights were tuned on (disclosed); "
            "the protocolB_first34tuned + last34_* masks are the truly-unseen evaluation.",
            "bar4: uniform_widen scales base (gamma=0) half-widths by a constant to MATCH the mechanism "
            "full-68 mean 95% width, isolating temporal REALLOCATION from raw width.",
            "foi channel is causal past-only 1-lag (incidence[:t]); mechanism ALLOWS narrowing (floor 0.5).",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── console summary ──
    print("\n============== FUSEDEPI v2: invRMSE-stack ⊕ mechanism-informed PI ==============")
    print(f"stack weights (1/RMSE, first-34): "
          f"{ {m: round(float(w),3) for m,w in zip(BASE_MODELS, inv_w)} }")
    print(f"gamma0 base identical to shipped helper: maxabs={base_ident:.2e}")
    print(f"pool-tuned gamma: sym={gamma_pool_sym}  upper={gamma_pool_up}\n")
    hdr = (f"{'variant':22s} {'g':>4s} {'side':>5s} | "
           f"{'WIS68':>7s} {'P95':>6s} {'Ppk50':>6s} {'Ppk25':>6s} | "
           f"{'WISl34':>7s} {'P95l34':>7s} {'Ppk50l34':>8s}")
    print(hdr); print("-" * len(hdr))
    rows = [("stack_gamma0", 0.0, "sym", base_scores)]
    for name, v in variant_scores.items():
        rows.append((name, v["gamma"], v["side"], v["scores"]))
    rows.append(("uniform_widen_ctrl", c_full, "sym", uniform_scores))
    for name, g, side, sc in rows:
        print(f"{name:22s} {g:4.2f} {side:>5s} | "
              f"{sc['overall_68']['wis']:7.4f} {sc['overall_68']['picp95']:6.3f} "
              f"{sc['peak_ge50']['picp95']:6.3f} {sc['peak_top25pct']['picp95']:6.3f} | "
              f"{sc['last34_overall']['wis']:7.4f} {sc['last34_overall']['picp95']:7.3f} "
              f"{sc['last34_peak_ge50']['picp95']:8.3f}")
    print("\nProtocol-B (gamma tuned on FIRST 34, evaluated on LAST 34):")
    for tag, d in protoB.items():
        print(f"  {tag:20s} g={d['gamma']:.2f} side={d['side']:>5s} | "
              f"WISl34={d['last34_overall']['wis']:.4f} P95l34={d['last34_overall']['picp95']:.3f} "
              f"Ppk50l34={d['last34_peak_ge50']['picp95']:.3f}")
    print("\n---- DECISIVE-WIN BARS (PRIMARY = v2_upper_pooltuned) ----")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    return result


if __name__ == "__main__":
    main()
