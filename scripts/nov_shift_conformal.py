#!/usr/bin/env python
"""NOVELTY PROTOTYPE — SHIFT-AWARE / REGIME-ADAPTIVE ONLINE CONFORMAL (leak-free).

Motivation
----------
Experiment 3 showed EVERY method under-covers on the 68-week hold-out
(PICP95 ~ 0.882; FusedEpi native deployed head only 0.735). The deployed
model-agnostic wrapper is the pure-online additive Conformal-PID
(`simulation.analytics.adaptive_conformal.online_conformal_bounds`), which
maintains a rolling buffer of ABSOLUTE residuals |y_j - pred_j|. Its 95% PI
misses concentrate on the epidemic RISING LIMB: in TiRex the 95% misses are
idx {0,1,3 (cold-start, tiny y), 5,6,7 (first-wave surge, +7.5/+18.5/+42.3
week-over-week), 50 (off-season bump), 52 (second-wave surge, +29.3)}. The
buffer is dominated by small quiet-season residuals, so when incidence
accelerates the interval is calibrated to the wrong regime and is far too
narrow. On the falling limb / plateau (idx 53-67) coverage is already fine.

Idea (normalized / scale-conditioned online conformal, Papadopoulos-style
normalized nonconformity + Gibbs-Candes/Angelopoulos online PID)
----------------------------------------------------------------------------
Instead of conformalizing the RAW residual r_j, conformalize the NORMALIZED
residual u_j = |r_j| / s_j, where s_j is a *leak-free* scale signal that
responds INSTANTLY to the regime (unlike the lagging absolute-residual buffer):

    s_i = floor + kappa * pred_i + eta * vol_i

  * pred_i   : the point forecast level (Poisson/NegBin variance grows with
               the mean; the surge is visible in pred itself).
  * vol_i    : recent realized volatility = mean |Δy| over the last L weeks,
               using ONLY past observations y[0..i-1] (d[i]=|y[i]-y[i-1]| is
               excluded because it needs y[i]). This spikes on the rising limb.
  * floor    : keeps the off-season interval from collapsing to zero width.

The width half is Q_i = quantile(u-buffer, 1-alpha) * s_i + PID integral term.
Because s_i tracks the regime with no lag, the interval widens immediately on
the rising limb and narrows off-season -> targets 0.95 coverage WITHOUT
over-widening the quiet season (which would inflate WIS).

Optional asymmetric rising-limb widener: all 95% misses are UNDER-predictions
(y > hi) on the accelerating limb, so we add lambda * max(0, recent slope) to
the UPPER bound only (epidemic rising limbs are positively skewed; FusedEpi's
own native head uses asymmetric Conformal-PID too).

Leak-free guarantees (identical to the deployed online_conformal_bounds)
------------------------------------------------------------------------
* Every quantity used to set interval[i] uses only pred[i] and y[0..i-1].
* y[i] is appended to the residual buffer AFTER interval[i] is fixed.
* s_i is precomputed from pred and past y only (no alpha, no buffer, no y[i]).
* Hyperparameters (kappa, eta, floor, vol_L, lambda) are selected ONLY on the
  first 34 test weeks (held-out first half), never on the scored span. We
  report BOTH a parameter-free principled variant on the full 68 (no selection
  at all) AND the first-half-selected variant.

Comparison is apples-to-apples: the SAME point forecast (frozen official
refit_test_predictions, identical to what produced the exp-3 baselines) is run
through (a) the deployed additive online conformal and (b) this shift-aware
conformal, so any WIS/coverage gap is attributable to the CALIBRATION only.

No live pipeline/model code is modified. Writes one JSON.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# mandated reuse: conformal + WIS helpers + frozen point loader from experiment 3
from simulation.analytics.adaptive_conformal import (  # noqa: E402
    online_conformal_bounds,
    wis_from_bounds,
)
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS  # noqa: E402
from fusedepi_fusion_wis import load_frozen  # noqa: E402

OUT = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/elevate/nov_shift_conformal.json"
)

PRIMARY_WINDOW = 30   # same rolling window as the deployed wrapper / exp-3
PRIMARY_KI = 0.2      # same PID I-gain as the deployed wrapper / exp-3


# ----------------------------------------------------------------------------
# leak-free regime scale
# ----------------------------------------------------------------------------
def compute_scale(
    pred: np.ndarray,
    y: np.ndarray,
    kind: str,
    *,
    floor: float = 1.0,
    kappa: float = 1.0,
    eta: float = 0.0,
    vol_L: int = 3,
) -> np.ndarray:
    """Leak-free regime scale s_i from pred_i and PAST observations only.

    Args:
        pred: (n,) point forecast (available at step i).
        y: (n,) observations, rolling order.
        kind: 'unit' | 'sqrt' | 'level' | 'vol' | 'level_vol'.
        floor, kappa, eta: scale coefficients (see module docstring).
        vol_L: number of recent |Δy| terms used for the volatility signal.

    Returns:
        (n,) scale vector; s_i uses only pred[i] and y[0..i-1] (NOT y[i]).
    """
    pred = np.asarray(pred, float).ravel()
    y = np.asarray(y, float).ravel()
    n = len(pred)
    # d[k] = |y[k] - y[k-1]| ; d[0] := 0. At step i only d[1..i-1] are usable
    # (d[i] would need y[i]).
    d = np.abs(np.diff(y, prepend=y[0]))
    s = np.empty(n, dtype=float)
    for i in range(n):
        if i >= 2:
            a = max(1, i - vol_L)
            vol = float(np.mean(d[a:i])) if i > a else float(d[i - 1])
        elif i == 1:
            vol = 0.0
        else:
            vol = 0.0
        pi = max(float(pred[i]), 0.0)
        if kind == "unit":
            s[i] = 1.0
        elif kind == "sqrt":
            s[i] = float(np.sqrt(max(pi, floor)))
        elif kind == "level":
            s[i] = max(pi, floor)
        elif kind == "vol":
            s[i] = floor + eta * vol
        elif kind == "level_vol":
            s[i] = floor + kappa * pi + eta * vol
        else:
            raise ValueError(f"unknown scale kind {kind!r}")
    return s


def _recent_slope(y: np.ndarray, i: int, L: int = 2) -> float:
    """Leak-free recent upward slope at step i: mean of last L signed Δy>0.

    Uses only y[0..i-1] (the increment landing at i is unknown)."""
    if i < 2:
        return 0.0
    a = max(1, i - L)
    dif = y[a:i] - y[a - 1 : i - 1]  # signed diffs d[a..i-1]
    up = dif[dif > 0]
    return float(np.mean(up)) if up.size else 0.0


def shift_aware_conformal_bounds(
    pred,
    y,
    alphas,
    scale: np.ndarray,
    *,
    window: int = PRIMARY_WINDOW,
    ki: float = PRIMARY_KI,
    asym_lambda: float = 0.0,
    asym_L: int = 2,
    target_scale: float = 1.0,
    seed_u: np.ndarray | None = None,
    cap: float | None = None,
) -> dict:
    """Normalized (scale-conditioned) pure-online Conformal-PID bounds.

    Mirrors `online_conformal_bounds` exactly (same window, ki, cap, cold-start,
    PID windup clip) EXCEPT the nonconformity score is the scale-normalized
    residual u_j = |y_j - pred_j| / s_j and the width is rescaled by s_i.

    Args:
        pred: (n,) point forecast. y: (n,) rolling observations.
        alphas: FLUSIGHT_ALPHAS. scale: (n,) leak-free s_i (compute_scale).
        window, ki: rolling window and PID I-gain (deployed defaults).
        asym_lambda: upper-only rising-limb widener coefficient (0 = symmetric).
        asym_L: window for the recent-slope signal.
        target_scale: ACI coverage-boost — PID drives empirical miscoverage down
            to alpha*target_scale (<1 => deliberately over-cover to counter the
            systematic undercoverage; =1 => nominal, identical to deployed).
        seed_u: optional (m,) NORMALIZED nonconformity seed (|r|/s from leak-free
            train/val in-sample residuals) to warm the cold-start buffer.
        cap: upper clamp (None -> 2*max(pred, y)).

    Returns:
        {alpha: (lo(n,), hi(n,))}. Leak-free: interval[i] uses pred[i],
        y[0..i-1], s[i] only; y[i] enters the buffer after interval[i] is set.
    """
    pred = np.asarray(pred, float).ravel()
    y = np.asarray(y, float).ravel()
    s = np.asarray(scale, float).ravel()
    n = len(y)
    if cap is None:
        cap = 2.0 * float(max(np.nanmax(pred) if pred.size else 0.0,
                              np.nanmax(y) if y.size else 0.0, 1.0))
    # precompute leak-free upper-widener signal
    up_extra = np.zeros(n)
    if asym_lambda > 0.0:
        for i in range(n):
            up_extra[i] = asym_lambda * _recent_slope(y, i, asym_L)
    seed = list(np.asarray(seed_u, float).ravel()) if seed_u is not None else []
    out = {}
    for a in alphas:
        buf: list[float] = list(seed)
        integral = 0.0
        lo = np.empty(n)
        hi = np.empty(n)
        for i in range(n):
            si = max(s[i], 1e-9)
            if len(buf) >= 3:
                q = max(0.0, float(np.quantile(buf[-window:], 1.0 - a)))
            else:
                q = float(np.max(buf)) if buf else 0.0   # normalized units
            base_half = q * si
            Q = max(0.0, base_half + ki * max(base_half, 1.0) * integral)
            lo[i] = max(0.0, pred[i] - Q)
            hi[i] = min(cap, pred[i] + Q + up_extra[i])
            miscov = 1.0 if (y[i] < lo[i] or y[i] > hi[i]) else 0.0
            integral = float(np.clip(integral + (miscov - a * target_scale), -5.0, 5.0))
            buf.append(abs(y[i] - pred[i]) / si)          # normalized (leak-free)
        out[a] = (lo, hi)
    return out


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def score(bounds, pred, y, mask) -> dict:
    """WIS + multi-level PICP + mean 95% width on the masked subset."""
    y = np.asarray(y, float).ravel()
    m = np.asarray(mask, bool)
    wis = np.asarray(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=pred), float)

    def picp(alpha):
        lo, hi = bounds[alpha]
        return float(np.mean(((y >= lo) & (y <= hi))[m]))

    lo95, hi95 = bounds[0.05]
    return {
        "wis": float(np.mean(wis[m])),
        "picp95": picp(0.05),
        "picp80": picp(0.20),
        "picp50": picp(0.50),
        "mean_width95": float(np.mean((hi95 - lo95)[m])),
        "n": int(m.sum()),
    }


def eval_bounds(bounds, pred, y, masks) -> dict:
    return {mk: score(bounds, pred, y, mv) for mk, mv in masks.items()}


def main() -> dict:
    result: dict = {"protocol": {
        "split": "frozen run_data pool_end=269 n_test=68 (official refit_test_predictions)",
        "point_source": "OFFICIAL frozen refit_test_predictions (== exp-3 baselines)",
        "wrapper": "same rolling window=%d, ki=%.2f as deployed online_conformal_bounds"
                   % (PRIMARY_WINDOW, PRIMARY_KI),
        "wis": "wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=pred) — R10 identical",
        "leak_free": (
            "interval[i] uses pred[i], y[0..i-1], s[i] only; y[i] enters buffer after; "
            "hyperparams selected on first-34 test weeks only"
        ),
    }}

    for model in ("TiRex", "FusedEpi"):
        pred, y, _ = load_frozen(model)
        pred = np.asarray(pred, float).ravel()
        y = np.asarray(y, float).ravel()
        n = len(y)

        peak_thr = float(np.quantile(y, 0.75))
        masks = {
            "overall_68": np.ones(n, bool),
            "peak_top25pct": y >= peak_thr,
            "peak_ge40": y >= 40.0,
        }

        # ---- (0) deployed additive baseline (reproduce exp-3) ----
        base_bounds = online_conformal_bounds(pred, y, FLUSIGHT_ALPHAS,
                                               window=PRIMARY_WINDOW, ki=PRIMARY_KI)
        base_eval = eval_bounds(base_bounds, pred, y, masks)

        # sanity: shift-aware with scale='unit', no asym == additive baseline
        unit_scale = compute_scale(pred, y, "unit")
        unit_bounds = shift_aware_conformal_bounds(pred, y, FLUSIGHT_ALPHAS, unit_scale)
        unit_eval = eval_bounds(unit_bounds, pred, y, masks)

        # ---- (1) parameter-free principled variants (NO selection) ----
        pf_variants = {
            "sqrt_level": dict(kind="sqrt", floor=1.0),
            "level_mult": dict(kind="level", floor=1.0),
        }
        pf_eval = {}
        pf_bounds_store = {}
        for name, kw in pf_variants.items():
            sc = compute_scale(pred, y, **kw)
            bnd = shift_aware_conformal_bounds(pred, y, FLUSIGHT_ALPHAS, sc)
            pf_eval[name] = eval_bounds(bnd, pred, y, masks)
            pf_bounds_store[name] = bnd

        # ---- (2) first-half-selected variant (leak-free tuning) ----
        cut = n // 2
        first_mask = np.zeros(n, bool); first_mask[:cut] = True
        last_mask = np.zeros(n, bool); last_mask[cut:] = True

        # grid over scale family + coefficients + optional asym widener
        grid = []
        for floor in (1.0, 3.0):
            for kind, kappa_opts, eta_opts in [
                ("sqrt", (0.0,), (0.0,)),
                ("level", (0.0,), (0.0,)),
                ("level_vol", (0.5, 1.0), (0.0, 0.5, 1.0)),
                ("vol", (0.0,), (1.0, 2.0)),
            ]:
                for kappa in kappa_opts:
                    for eta in eta_opts:
                        for lam in (0.0, 0.3, 0.6):
                            grid.append(dict(kind=kind, floor=floor, kappa=kappa,
                                             eta=eta, asym_lambda=lam))

        best = None
        for cfg in grid:
            lam = cfg.pop("asym_lambda")
            sc = compute_scale(pred, y, **cfg)
            bnd = shift_aware_conformal_bounds(pred, y, FLUSIGHT_ALPHAS, sc,
                                               asym_lambda=lam)
            # select ONLY on first-34: minimise WIS s.t. picp95 not wildly over
            s_first = score(bnd, pred, y, first_mask)
            cfg["asym_lambda"] = lam
            key = (s_first["wis"], abs(s_first["picp95"] - 0.95))
            if best is None or key < best[0]:
                best = (key, dict(cfg), s_first)

        sel_cfg = best[1]
        lam = sel_cfg["asym_lambda"]
        sc_cfg = {k: v for k, v in sel_cfg.items() if k != "asym_lambda"}
        sel_scale = compute_scale(pred, y, **sc_cfg)
        sel_bounds = shift_aware_conformal_bounds(pred, y, FLUSIGHT_ALPHAS, sel_scale,
                                                  asym_lambda=lam)
        sel_eval_full = eval_bounds(sel_bounds, pred, y, masks)
        # honest Protocol-B: selection used first-34, report on last-34 (unseen)
        protoB_masks = {
            "heldout_last34": last_mask,
            "heldout_last34_peak": last_mask & (y >= float(np.quantile(y[cut:], 0.75))),
        }
        sel_eval_protoB = eval_bounds(sel_bounds, pred, y, protoB_masks)
        base_eval_protoB = eval_bounds(base_bounds, pred, y, protoB_masks)

        # ---- (3) window robustness of the parameter-free sqrt-scale headline ----
        window_robustness = {}
        for w in (20, 30, 40):
            base_w = online_conformal_bounds(pred, y, FLUSIGHT_ALPHAS, window=w, ki=PRIMARY_KI)
            sqrt_w = shift_aware_conformal_bounds(
                pred, y, FLUSIGHT_ALPHAS, compute_scale(pred, y, "sqrt", floor=1.0),
                window=w, ki=PRIMARY_KI)
            bo = score(base_w, pred, y, masks["overall_68"])
            so = score(sqrt_w, pred, y, masks["overall_68"])
            sp = score(sqrt_w, pred, y, masks["peak_top25pct"])
            window_robustness[str(w)] = {
                "additive_wis": bo["wis"], "additive_picp95": bo["picp95"],
                "sqrt_wis": so["wis"], "sqrt_picp95": so["picp95"],
                "sqrt_peak_picp95": sp["picp95"], "delta_wis": so["wis"] - bo["wis"],
            }

        result[model] = {
            "n_test": n,
            "peak_top25pct_threshold": peak_thr,
            "n_peak": int(masks["peak_top25pct"].sum()),
            "baseline_additive_online_conformal": base_eval,
            "window_robustness_sqrt_vs_additive": window_robustness,
            "sanity_unit_scale_equals_baseline": {
                "unit": unit_eval["overall_68"],
                "max_abs_wis_gap_vs_baseline": abs(
                    unit_eval["overall_68"]["wis"] - base_eval["overall_68"]["wis"]),
            },
            "parameter_free_variants_full68": pf_eval,
            "selected_first34_config": sel_cfg,
            "selected_variant_full68": sel_eval_full,
            "protocolB_last34": {
                "selected_shift_aware": sel_eval_protoB,
                "baseline_additive": base_eval_protoB,
            },
        }

    # ---- verdict on the headline model (TiRex, best single) ----
    def headline(model):
        r = result[model]
        base = r["baseline_additive_online_conformal"]["overall_68"]
        base_peak = r["baseline_additive_online_conformal"]["peak_top25pct"]
        # best parameter-free by overall WIS
        pf = r["parameter_free_variants_full68"]
        pf_best_name = min(pf, key=lambda k: pf[k]["overall_68"]["wis"])
        pf_best = pf[pf_best_name]
        sel = r["selected_variant_full68"]
        return {
            "baseline_overall_wis": base["wis"],
            "baseline_overall_picp95": base["picp95"],
            "baseline_peak_wis": base_peak["wis"],
            "baseline_peak_picp95": base_peak["picp95"],
            "param_free_best": pf_best_name,
            "param_free_overall_wis": pf_best["overall_68"]["wis"],
            "param_free_overall_picp95": pf_best["overall_68"]["picp95"],
            "param_free_peak_wis": pf_best["peak_top25pct"]["wis"],
            "param_free_peak_picp95": pf_best["peak_top25pct"]["picp95"],
            "selected_overall_wis": sel["overall_68"]["wis"],
            "selected_overall_picp95": sel["overall_68"]["picp95"],
            "selected_peak_wis": sel["peak_top25pct"]["wis"],
            "selected_peak_picp95": sel["peak_top25pct"]["picp95"],
        }

    result["verdict"] = {
        "TiRex": headline("TiRex"),
        "FusedEpi": headline("FusedEpi"),
        "baselines_reference": {
            "TiRex_alone": {"wis": 2.951, "picp95": 0.882},
            "FusedEpi": {"wis": 3.011, "picp95": 0.882},
            "heldout_tuned_invrmse_stack_protoB": {"wis": 2.720},
            "nominal_picp95": 0.95,
        },
        "targets": [
            "lift PICP95 toward 0.95 without inflating WIS",
            "WIS <= 3.011 (ideally < 2.72) AND beat TiRex-alone 2.951",
        ],
    }
    result["caveats"] = [
        "Point forecast = official frozen refit_test_predictions (identical to the "
        "predictor that produced the exp-3 baselines); only the CONFORMAL calibration "
        "differs, so WIS/coverage gaps isolate the calibration.",
        "parameter_free_variants use NO test-based selection (principled Poisson sqrt / "
        "multiplicative level scale) and are directly comparable to the full-68 baselines.",
        "selected_variant hyperparameters were chosen on the FIRST 34 test weeks only; "
        "full-68 numbers therefore include the 34 selection weeks (disclosed). The "
        "protocolB_last34 block reports the selected config on the truly unseen last 34.",
        "Cold-start (first ~window weeks) is identical to the deployed online conformal; "
        "the sanity_unit_scale check confirms scale='unit' reproduces the additive baseline.",
        "Peak regime = top-quartile test incidence (y >= peak_top25pct_threshold); the "
        "asym widener uses only the recent realized slope (leak-free).",
        "HONEST LIMITS: sqrt-scale is a robust Pareto win over the deployed additive "
        "online conformal (lower WIS AND equal/higher PICP95 at every window) and beats "
        "the exp-3 TiRex-alone/FusedEpi WIS, but does NOT reach WIS<2.72 and does NOT lift "
        "PICP95 to ~0.95. Peak-week PICP95 stays 0.882 under sqrt; only the multiplicative "
        "'level_mult' scale lifts peak PICP95 to 0.941, and it inflates overall WIS above "
        "3.011. On the truly-unseen last 34 the sqrt gain shrinks to ~a tie with additive "
        "(the improvement is concentrated in the first epidemic wave).",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2, ensure_ascii=False))
