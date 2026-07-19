#!/usr/bin/env python
"""NOVELTY CANDIDATE — REGIME-GATED / CONDITIONAL FUSION.

Motivation
----------
`scripts/fusedepi_multiseason.py` showed the TabPFN residual correction inside
FusedEpi helps ONLY in large epidemics and HURTS off-season.  On the frozen
68-week hold-out test the deployed scalar fusion (FusedEpi, WIS 3.011) is
therefore WORSE than pure TiRex-alone (2.951): the constant residual weight pays
an off-season tax that the peak benefit does not repay.

Idea
----
Make the residual weight REGIME-CONDITIONAL instead of a single scalar:

    point_t = TiRex_t + alpha_t * TabPFN_residual_t
    alpha_t = A * sigmoid((m_t - c) / s)

where m_t is a LEAK-FREE predicted-epidemic-magnitude signal (the TiRex 1-step
forecast for week t, or the lagged observation).  Off-season (m_t low) →
alpha_t≈0 → pure TiRex.  Epidemic (m_t high) → alpha_t≈A → the residual kicks in.
The gate parameters (A, c, s) and the magnitude feature are LEARNED ON TRAIN ONLY
(a leak-free within-train validation tail, scored through the SAME online
conformal WIS objective).  A=0 is in the grid, so if the residual never helps on
train the model collapses to pure TiRex (do-no-harm).

Protocol (leak-free, identical to experiment 3 / ablation_fusedepi.py)
----------------------------------------------------------------------
* Split:   run_data frozen split via load_split() — train pool [0:269], test
           [269:337] (n_test=68).  Test is NEVER touched during fitting or gate
           learning.
* Base+corr: live FusedEpi machinery (subclass of FusedEpiForecaster) exposes the
           internal TiRex 1-step roll (base) and the TabPFN residual (corr)
           SEPARATELY on the rolling test span (y_observed fed back — leak-free).
* Gate:    learned on the last `gate_val_weeks` of the train pool.  A separate
           TabPFN is fit on residuals BEFORE that tail and predicts OOF residuals
           on the tail (leak-free); (A,c,s) chosen by online-conformal WIS on the
           tail.  Deployed corr is refit on ALL train residuals.
* Scoring: simulation.analytics.adaptive_conformal.online_conformal_bounds
           (window=30, ki=0.2) + wis_from_bounds over FLUSIGHT_ALPHAS — the SAME
           model-agnostic wrapper applied to every point series, so any WIS gap
           is attributable to the POINT (fusion) mechanism, not the conformal
           machinery.  PICP95 = empirical coverage of the 95% PI (alpha=0.05).
* Reported: overall (68) and peak (y >= 75th pct of test y, 17 weeks).

References for the reader
-------------------------
* Deployed frozen TiRex + FusedEpi test predictions are ALSO scored through the
  wrapper to reproduce the experiment-3 headline baselines (2.951 / 3.011) and
  validate this harness byte-for-byte.
* gate=0 (internal TiRex) and scalar-alpha (live FusedEpi) points are reported so
  the regime gate is seen to interpolate between them.
* A DEPLOYED-ANCHORED variant adds the gated residual on top of the frozen
  deployed TiRex point (whose gate=0 IS the 2.951 headline), giving an airtight
  "beats 2.951?" read (internal vs deployed TiRex agree to RMSE 0.62, corr 0.9998).

No live pipeline/model code is modified.  Writes one JSON.
"""
from __future__ import annotations

import gc
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
os.environ.setdefault("OPTUNA_ISOLATE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd

from simulation.analytics.adaptive_conformal import online_conformal_bounds, wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.models.fused_epi import FusedEpiForecaster

# Reuse the frozen-split loader + seeding so the split/feature space is identical.
from scripts.ablation_fusedepi import load_split, seed_all

LOG = logging.getLogger("nov_regime_gated")
OUT_JSON = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/nov/regime_gated.json"
)

WINDOW = 30
KI = 0.2
OPT_DIR = ROOT / "simulation/results/per_model_optimal"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def wrapper_score(point: np.ndarray, y: np.ndarray, masks: dict) -> dict:
    """Score a point series through the common online-conformal wrapper.

    Args:
        point: (n,) point predictions in rolling order.
        y: (n,) observations in rolling order.
        masks: {name: bool-mask} scoring subsets (overall / peak).
    Returns: {mask_name: {wis, picp95, picp80, picp50, mean_width95, n}}.
    """
    point = np.asarray(point, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    bounds = online_conformal_bounds(point, y, FLUSIGHT_ALPHAS, window=WINDOW, ki=KI)
    wis_arr = np.asarray(wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=point), dtype=float)
    lo95, hi95 = bounds[0.05]
    cov95 = (y >= lo95) & (y <= hi95)

    def _picp(alpha, m):
        lo, hi = bounds[alpha]
        return float(np.mean(((y >= lo) & (y <= hi))[m]))

    out = {}
    for mk, m in masks.items():
        m = np.asarray(m, dtype=bool)
        out[mk] = {
            "wis": float(np.mean(wis_arr[m])),
            "picp95": _picp(0.05, m),
            "picp80": _picp(0.20, m),
            "picp50": _picp(0.50, m),
            "mean_width95": float(np.mean((hi95 - lo95)[m])),
            "n": int(np.sum(m)),
        }
    return out


class RegimeGatedFusedEpi(FusedEpiForecaster):
    """FusedEpi with a per-step REGIME GATE on the TabPFN residual.

    The scalar do-no-harm weight is replaced by alpha_t = A*sigmoid((m_t-c)/s),
    with m_t a leak-free predicted-magnitude signal. (A,c,s) and the magnitude
    feature are learned on a within-train validation tail (leak-free), scored
    through the online-conformal WIS. Exposes base/corr/magnitude components so
    the caller combines them under any gate.

    Caller responsibility: rolling 1-step eval — pass y_observed = the scoring
    slab (past→present order). Test is never touched during fit.
    """

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            gate_val_weeks: int = 104, **kwargs) -> "RegimeGatedFusedEpi":
        from tirex import load_model

        y = np.asarray(y_train, dtype=float).ravel()
        X = np.asarray(X_train, dtype=float)
        self._train_y = y
        self._y_max = float(np.max(y)) if y.size else 0.0
        n = len(y)
        if self._tx is None:
            self._tx = load_model(self.repo_id, device="cpu")

        # 1) TiRex base roll over train → residuals
        tr_idx = list(range(self.min_ctx, n))
        tx_tr = self._tirex_roll(y, tr_idx)
        resid = y[self.min_ctx:] - tx_tr
        yf = y[self.min_ctx:]

        # 2) features = lag/seasonal + mechanistic(1-lag)
        mech = self._mech_features(y)
        Xf_all = np.hstack([X, mech])[self.min_ctx:]

        # 3) mc do-no-harm (reuse FusedEpi) with the cal tail K
        K = max(10, int(len(yf) * self.cal_frac))
        self._mc_keep = self._select_mc_keep(Xf_all, resid, K)
        Xf = Xf_all[:, self._mc_keep] if self._mc_keep is not None else Xf_all

        # 4) SCALAR do-no-harm alpha (reference / FusedEpi reconstruction)
        corr_pt = self._tab()
        corr_pt.fit(Xf[:-K], resid[:-K])
        corr_cal = np.asarray(corr_pt.predict(Xf[-K:]), dtype=float)
        alpha_size = float(np.clip(n / self.n_ref, self.alpha_min, 1.0))
        base_err = float(np.mean(resid[-K:] ** 2))
        corr_err = float(np.mean((resid[-K:] - corr_cal) ** 2))
        harm = float(np.clip(base_err / (corr_err + 1e-9), 0.0, 1.0))
        self._alpha = alpha_size * harm

        # 5) GATE learning on a leak-free within-train validation tail
        V = int(min(gate_val_weeks, len(yf) - 20))
        if V < 20:
            raise RuntimeError(f"train too short for gate-val tail: len(yf)={len(yf)}")
        gcorr = self._tab()
        gcorr.fit(Xf[:-V], resid[:-V])                       # pre-tail proper-train
        corr_gval = np.asarray(gcorr.predict(Xf[-V:]), dtype=float)   # OOF residual on tail
        tx_gval = tx_tr[-V:]
        y_gval = yf[-V:]
        lag_obs_full = y[self.min_ctx - 1: n - 1]            # y_{t-1} aligned to yf
        m_variants = {
            "tirex_pred": tx_gval,
            "lag_obs": lag_obs_full[-V:],
        }
        learned = {
            fname: self._learn_gate(tx_gval, corr_gval, y_gval, m_gval, tx_tr)
            for fname, m_gval in m_variants.items()
        }
        best_feat = min(learned, key=lambda f: learned[f]["wis"])
        self.gate_feature = best_feat
        self._gate_params = learned[best_feat]
        self._gate_learned_all = learned
        self._gate_val_weeks = V
        # Constant-weight control (no sigmoid gate): best CONSTANT A on the same
        # train tail — isolates "apply more residual" from "gate by regime".
        self._const_params = self._learn_constant(tx_gval, corr_gval, y_gval)
        self._gate_val_arrays = {
            "tx": tx_gval.tolist(), "corr": corr_gval.tolist(),
            "y": y_gval.tolist(), "lag_obs": m_variants["lag_obs"].tolist(),
        }

        # 6) deploy corr = refit on ALL train residuals
        self._corr = self._tab()
        self._corr.fit(Xf, resid)
        self._fitted = True
        LOG.info(
            "gate feature=%s params=%s (train-tail WIS=%.4f); scalar alpha0=%.4f",
            self.gate_feature,
            {k: round(v, 4) for k, v in self._gate_params.items() if k != "wis"},
            self._gate_params["wis"], self._alpha,
        )
        return self

    def _learn_gate(self, tx, corr, y, m, mref) -> dict:
        """Grid-search (A,c,s) minimizing online-conformal WIS on the train tail.

        Args:
            tx: (V,) TiRex forecast on the tail. corr: (V,) OOF TabPFN residual.
            y: (V,) observations on the tail. m: (V,) magnitude gate feature.
            mref: (>=V,) magnitude reference distribution (whole train TiRex roll)
                  used to place threshold/scale candidates on the right scale.
        Returns: dict(A, c, s, wis) — the WIS-minimizing gate on the train tail.
        """
        mref = np.asarray(mref, dtype=float).ravel()
        c_cands = [float(np.quantile(mref, q)) for q in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)]
        iqr = float(np.quantile(mref, 0.75) - np.quantile(mref, 0.25))
        scale0 = iqr if iqr > 1e-6 else (float(np.std(mref)) if np.std(mref) > 1e-6 else 1.0)
        s_cands = [max(1e-3, f * scale0) for f in (0.05, 0.10, 0.20, 0.40, 0.80)]
        A_cands = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        best = None
        for A in A_cands:
            for c in c_cands:
                for s in s_cands:
                    alpha = A * _sigmoid((m - c) / s)
                    point = np.clip(tx + alpha * corr, 0.0, None)
                    b = online_conformal_bounds(point, y, FLUSIGHT_ALPHAS, window=WINDOW, ki=KI)
                    wis = float(np.mean(wis_from_bounds(y, b, FLUSIGHT_ALPHAS, median=point)))
                    if (best is None) or (wis < best["wis"]) or (
                        abs(wis - best["wis"]) < 1e-9 and A < best["A"]
                    ):
                        best = {"A": float(A), "c": float(c), "s": float(s), "wis": float(wis)}
        return best

    def _learn_constant(self, tx, corr, y) -> dict:
        """Best CONSTANT residual weight A on the train tail (no sigmoid gate).

        Control for the regime gate: alpha_t = A (constant) for all t. A=0 in the
        grid recovers pure TiRex. Returns dict(A, wis).
        """
        A_cands = np.round(np.arange(0.0, 1.55, 0.05), 3)
        best = None
        for A in A_cands:
            point = np.clip(tx + float(A) * corr, 0.0, None)
            b = online_conformal_bounds(point, y, FLUSIGHT_ALPHAS, window=WINDOW, ki=KI)
            wis = float(np.mean(wis_from_bounds(y, b, FLUSIGHT_ALPHAS, median=point)))
            if best is None or wis < best["wis"] or (
                abs(wis - best["wis"]) < 1e-9 and float(A) < best["A"]
            ):
                best = {"A": float(A), "wis": float(wis)}
        return best

    def predict_components(self, X_test: np.ndarray, y_observed) -> tuple:
        """Return (base, corr, m) on the rolling test span (leak-free).

        base: internal TiRex 1-step roll. corr: TabPFN residual on test features.
        m: magnitude gate feature (TiRex forecast or lagged observation).
        """
        X = np.asarray(X_test, dtype=float)
        n_test = len(X)
        obs = np.asarray(y_observed, dtype=float).ravel() if y_observed is not None else None
        base = np.empty(n_test, dtype=float)
        for i in range(n_test):
            hist = np.concatenate([self._train_y, obs[:i]]) if (obs is not None and i > 0) else self._train_y
            base[i] = self._tirex_1step(hist)
        y_full = np.concatenate([self._train_y, obs]) if obs is not None else self._train_y
        Xf = self._corr_features(X, y_full, n_test, obs)
        corr = np.asarray(self._corr.predict(Xf), dtype=float)
        if self.gate_feature == "lag_obs":
            m = np.empty(n_test, dtype=float)
            for i in range(n_test):
                m[i] = self._train_y[-1] if (obs is None or i == 0) else obs[i - 1]
        else:
            m = base.copy()
        return base, corr, m

    def alpha_from_m(self, m: np.ndarray) -> np.ndarray:
        A, c, s = self._gate_params["A"], self._gate_params["c"], self._gate_params["s"]
        return A * _sigmoid((np.asarray(m, dtype=float) - c) / s)


def load_frozen_point(model: str) -> tuple:
    """Deployed frozen refit_test_predictions + y_true from the R9 artifacts."""
    j = json.loads((OPT_DIR / f"{model}.json").read_text())
    pred = np.asarray(j["refit_test_predictions"], dtype=float).ravel()
    csv = pd.read_csv(ROOT / f"simulation/results/csv/predictions_{model}.csv").sort_values("idx")
    y = csv["y_true"].to_numpy(float)
    return pred, y


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(42)
    t0 = time.time()

    # ── frozen split (test never touched during fit) ──
    X_train, y_train, X_test, y_test, split_meta = load_split()
    n_test = len(y_test)
    LOG.info("split: train_pool=%d test=%d basic_features=%d",
             len(y_train), n_test, X_train.shape[1])

    # peak regime = top-quartile test weeks (matches experiment 3)
    peak_thr = float(np.quantile(y_test, 0.75))
    masks = {
        "overall_68": np.ones(n_test, dtype=bool),
        "peak_top25pct": y_test >= peak_thr,
    }

    # ── deployed frozen baselines (reproduce experiment-3 headline + validate harness) ──
    tirex_frozen, y_frozen = load_frozen_point("TiRex")
    fused_frozen, y_frozen2 = load_frozen_point("FusedEpi")
    assert np.max(np.abs(y_frozen - y_test)) < 1e-9, "frozen TiRex y_true != load_split y_test"
    assert np.max(np.abs(y_frozen2 - y_test)) < 1e-9, "frozen FusedEpi y_true != load_split y_test"
    ref = {
        "deployed_TiRex_frozen": wrapper_score(tirex_frozen, y_test, masks),
        "deployed_FusedEpi_frozen": wrapper_score(fused_frozen, y_test, masks),
    }

    # ── fit the regime-gated model (leak-free; test untouched) ──
    seed_all(42)
    model = RegimeGatedFusedEpi()
    model.fit(X_train, y_train)

    # rolling test components (y_observed fed back — leak-free)
    base, corr, m = model.predict_components(X_test, y_observed=y_test)
    alpha_t = model.alpha_from_m(m)
    cap = 2.0 * model._y_max if model._y_max > 0 else np.inf

    point_gate0 = np.clip(base, 0.0, cap)                          # internal TiRex (gate off)
    point_scalar = np.clip(base + model._alpha * corr, 0.0, cap)   # live FusedEpi scalar
    point_gated = np.clip(base + alpha_t * corr, 0.0, cap)         # REGIME-GATED (internal base)

    # deployed-anchored variant: gate the residual on top of the 2.951 frozen TiRex
    if model.gate_feature == "lag_obs":
        m_dep = m  # lag_obs independent of the base
    else:
        m_dep = tirex_frozen  # predicted magnitude on the deployed scale
    alpha_dep = model.alpha_from_m(m_dep)
    point_gated_dep = np.clip(tirex_frozen + alpha_dep * corr, 0.0, cap)

    # Constant-weight controls (best CONSTANT A learned on train tail; no gate)
    A_const = model._const_params["A"]
    point_const_int = np.clip(base + A_const * corr, 0.0, cap)
    point_const_dep = np.clip(tirex_frozen + A_const * corr, 0.0, cap)

    live = {
        "internal_TiRex_gate0": wrapper_score(point_gate0, y_test, masks),
        "live_FusedEpi_scalar": wrapper_score(point_scalar, y_test, masks),
        "const_weight_internal": wrapper_score(point_const_int, y_test, masks),
        "const_weight_deployed_anchored": wrapper_score(point_const_dep, y_test, masks),
        "regime_gated_internal": wrapper_score(point_gated, y_test, masks),
        "regime_gated_deployed_anchored": wrapper_score(point_gated_dep, y_test, masks),
    }

    # gate diagnostics
    alpha_on = float(np.mean(alpha_t[masks["peak_top25pct"]]))
    alpha_off = float(np.mean(alpha_t[~masks["peak_top25pct"]]))
    internal_vs_deployed_tirex_rmse = float(np.sqrt(np.mean((point_gate0 - tirex_frozen) ** 2)))

    # ── verdict vs the task baselines ──
    tirex_alone_wis = ref["deployed_TiRex_frozen"]["overall_68"]["wis"]  # 2.951 headline
    fused_wis = ref["deployed_FusedEpi_frozen"]["overall_68"]["wis"]     # 3.011
    rg_int = live["regime_gated_internal"]["overall_68"]["wis"]
    rg_dep = live["regime_gated_deployed_anchored"]["overall_68"]["wis"]
    best_rg = min(rg_int, rg_dep)
    best_rg_name = "regime_gated_internal" if rg_int <= rg_dep else "regime_gated_deployed_anchored"

    verdict = {
        "baselines": {
            "tirex_alone_2951": tirex_alone_wis,
            "fusedepi_3011": fused_wis,
            "held_out_tuned_invrmse_stack_2720": 2.720,
        },
        "regime_gated_internal_wis": rg_int,
        "regime_gated_deployed_anchored_wis": rg_dep,
        "best_regime_gated_wis": best_rg,
        "best_regime_gated_variant": best_rg_name,
        "beats_tirex_alone": bool(best_rg < tirex_alone_wis),
        "beats_fusedepi": bool(best_rg < fused_wis),
        "beats_invrmse_stack_2720": bool(best_rg < 2.720),
        "delta_vs_tirex_alone": float(best_rg - tirex_alone_wis),
        "mechanism_decomposition": {
            "note": "does the SIGMOID GATE add anything over a best CONSTANT residual weight?",
            "const_weight_A": float(A_const),
            "internal_base": {
                "gate0_tirex": live["internal_TiRex_gate0"]["overall_68"]["wis"],
                "const_weight": live["const_weight_internal"]["overall_68"]["wis"],
                "regime_gated": rg_int,
                "gate_minus_const": float(rg_int - live["const_weight_internal"]["overall_68"]["wis"]),
            },
            "deployed_base": {
                "gate0_tirex_2951": tirex_alone_wis,
                "const_weight": live["const_weight_deployed_anchored"]["overall_68"]["wis"],
                "regime_gated": rg_dep,
                "gate_minus_const": float(rg_dep - live["const_weight_deployed_anchored"]["overall_68"]["wis"]),
            },
        },
        "picp95_best_rg_overall": (
            live[best_rg_name]["overall_68"]["picp95"]
        ),
        "picp95_best_rg_peak": (
            live[best_rg_name]["peak_top25pct"]["picp95"]
        ),
    }

    result = {
        "candidate": "regime-gated / conditional fusion (alpha_t = A*sigmoid((m_t-c)/s))",
        "protocol": {
            "split": "run_data frozen split via load_split(); train[0:%d] test[%d:%d] n_test=%d"
                     % (split_meta["pool_end"], split_meta["test_start"], split_meta["test_end"], n_test),
            "base_corr": "live FusedEpi subclass: internal TiRex 1-step roll (base) + TabPFN residual (corr), rolling y_observed",
            "gate_learning": "within-train validation tail (last %d weeks): OOF TabPFN residual + online-conformal WIS grid over (A,c,s) and {tirex_pred,lag_obs}; A=0 in grid (do-no-harm)"
                             % model._gate_val_weeks,
            "conformal": "online_conformal_bounds(window=%d, ki=%.2f) — SAME wrapper on every point series" % (WINDOW, KI),
            "wis": "wis_from_bounds(y, bounds, FLUSIGHT_ALPHAS, median=point)",
            "picp95": "empirical coverage of the 95% PI (alpha=0.05)",
            "peak_definition": {"threshold_y": peak_thr, "n_peak": int(masks["peak_top25pct"].sum())},
            "leak_free": "test span untouched during fit and gate learning; rolling step i uses only past obs",
        },
        "gate": {
            "selected_feature": model.gate_feature,
            "params": {k: model._gate_params[k] for k in ("A", "c", "s")},
            "train_tail_wis": model._gate_params["wis"],
            "scalar_alpha0": float(model._alpha),
            "mean_alpha_peak": alpha_on,
            "mean_alpha_offpeak": alpha_off,
            "learned_all_features": {
                f: {k: round(v, 5) for k, v in p.items()} for f, p in model._gate_learned_all.items()
            },
            "best_constant_weight_control": model._const_params,
            "internal_vs_deployed_tirex_rmse": internal_vs_deployed_tirex_rmse,
        },
        "reference_baselines_through_wrapper": ref,
        "regime_gated_results": live,
        "verdict": verdict,
        "caveats": [
            "The regime-gated point is built on FusedEpi's INTERNAL TiRex roll; its gate=0 = "
            "internal_TiRex_gate0 (reported). The deployed-anchored variant adds the SAME gated "
            "residual on top of the frozen deployed TiRex whose gate=0 IS the 2.951 headline "
            "(internal vs deployed TiRex agree to RMSE %.2f) — an airtight beats-2.951 read."
            % internal_vs_deployed_tirex_rmse,
            "The conformal wrapper is applied IDENTICALLY to every point series, so WIS gaps "
            "isolate the POINT/fusion mechanism from the interval machinery.",
            "Gate (A,c,s) and the magnitude feature are chosen by online-conformal WIS on a "
            "within-train tail only; A=0 is in the grid so a non-helpful residual collapses to "
            "pure TiRex (do-no-harm).",
            "PICP95 is largely set by the shared online-conformal wrapper (all methods ~0.88); "
            "a better peak point can only nudge it via tighter residuals.",
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", OUT_JSON)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    del model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
