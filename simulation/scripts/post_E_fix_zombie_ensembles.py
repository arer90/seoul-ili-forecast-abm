"""Post-E P0#1 fix + follow-up — regenerate Ensemble-Stacking / Blending
predictions AS TWO DISTINCT ENSEMBLE SCHEMES (Wolpert 1992 differentiation).

Context:
  Both CSVs previously held constant y_pred=41.75 (Ridge meta-learner
  intercept-only fallback). The first fix produced *identical* Stacking and
  Blending outputs (both used RidgeCV+convex-combination), which reviewers
  will flag as duplicate entries.

Definition split (follow-up):
  - Ensemble-Stacking : non-negative RidgeCV weighted meta-learner on all
                         qualified base models (val R² ≥ 0.3), weights
                         learned adaptively.
  - Ensemble-Blending : equal-weight mean of top-10 models by val R² — a
                         simpler, non-adaptive baseline. Well-known
                         heuristic ensemble (Sagi & Rokach 2018 survey
                         calls this "simple averaging with selection").

Pipeline:
  1. Load val (41 weeks) + test (69 weeks) predictions for every base model
     (excludes any Ensemble-* to avoid meta-on-meta leakage).
  2. Drop NEGATIVE_CONTROL models (TabularDNN).
  3. Keep val R² ≥ 0.3 → qualified pool.
  4. Stacking : RidgeCV + positive projection + sum=1 rescale on full pool.
  5. Blending : pick top-10 by val R², equal-weight mean (w_i=1/10).
  6. Clip test predictions at 0 (no upper cap — base preds are in-range).

Run:
    .venv/Scripts/python.exe -m simulation.scripts.post_E_fix_zombie_ensembles
"""
from __future__ import annotations

import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("post_E.zombie_fix")

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "simulation" / "results" / "csv"

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
R2_FLOOR = 0.3

try:
    from simulation.models.registry import NEGATIVE_CONTROL as _NC
except Exception:
    _NC = set()


def _val_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


def _load_base_splits() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return (val_pred_by_model, test_pred_by_model, val_actual, test_actual)."""
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    val_actual = None
    test_actual = None
    for fp in sorted(glob.glob(str(CSV / "predictions_*.csv"))):
        name = os.path.basename(fp).replace("predictions_", "").replace(".csv", "")
        if name.startswith("Ensemble"):
            continue  # meta-on-meta 차단
        if name in _NC:
            log.info("  skip NEGATIVE_CONTROL: %s", name)
            continue
        df = pd.read_csv(fp)
        v = df[df["split"] == "val"].sort_values("idx")
        t = df[df["split"] == "test"].sort_values("idx")
        if len(v) == 0 or len(t) == 0:
            continue
        if val_actual is None:
            val_actual = v["y_true"].to_numpy(dtype=float)
            test_actual = t["y_true"].to_numpy(dtype=float)
        val_preds[name] = v["y_pred"].to_numpy(dtype=float)
        test_preds[name] = t["y_pred"].to_numpy(dtype=float)
    return val_preds, test_preds, val_actual, test_actual


def _ridge_positive(val_X: np.ndarray, val_y: np.ndarray, test_X: np.ndarray,
                    label: str) -> tuple[np.ndarray, RidgeCV, np.ndarray]:
    """Fit positive-projected RidgeCV on val meta-features, predict on test.

    Clipping note: original Stacking/Blending used ``2.5 * y_train_max`` which
    collapses on this dataset because val has no winter peak (val_max=16.7,
    test_max=100.7). We only floor at 0 — Ridge + positive-projection keeps
    amplification bounded as long as base predictions are in-range.
    """
    mdl = RidgeCV(alphas=ALPHAS, fit_intercept=False,
                  scoring="neg_root_mean_squared_error")
    mdl.fit(val_X, val_y)
    coefs = np.asarray(mdl.coef_, dtype=float)
    # Convex combination: clip to non-negative and rescale to sum=1.
    # Previous scheme preserved ||coef||_1 which inflates test predictions
    # when val has no peak (val_max=16.7, test_max=100.7): amplification
    # factor multiplies directly into extrapolation. sum=1 gives a proper
    # weighted-average that scales with base predictions themselves.
    coefs_pos = np.clip(coefs, a_min=0.0, a_max=None)
    s = float(coefs_pos.sum())
    if s > 1e-8:
        coefs_pos = coefs_pos / s
    else:
        coefs_pos = np.ones_like(coefs_pos) / max(len(coefs_pos), 1)
    mdl.coef_ = coefs_pos
    mdl.intercept_ = 0.0
    pred_raw = mdl.predict(test_X)
    pred = np.clip(pred_raw, 0.0, None)
    n_neg = int(np.sum(pred_raw < 0))
    nz = int(np.sum(coefs_pos > 1e-4))
    log.info("  [%s] α=%.3f, nonzero_weights=%d/%d, max_weight=%.3f, neg_clipped=%d",
             label, float(mdl.alpha_), nz, val_X.shape[1],
             float(coefs_pos.max()), n_neg)
    return pred, mdl, coefs_pos


def regenerate() -> None:
    val_preds, test_preds, val_actual, test_actual = _load_base_splits()
    log.info("loaded %d base models (val=%d, test=%d)",
             len(val_preds), len(val_actual), len(test_actual))

    r2 = {k: _val_r2(val_actual, v) for k, v in val_preds.items()}
    qualified = sorted([k for k, v in r2.items() if v >= R2_FLOOR])
    excluded = [(k, r2[k]) for k in val_preds if k not in qualified]
    if len(qualified) < 2:
        log.warning("R²>=%.2f 통과 %d개 (<2) → floor 해제", R2_FLOOR, len(qualified))
        qualified = sorted(val_preds.keys())
        excluded = []
    log.info("qualified=%d, excluded=%d (R²<%.2f)", len(qualified), len(excluded), R2_FLOOR)
    for k, v in sorted(excluded, key=lambda x: x[1])[:5]:
        log.info("  excluded: %s (R²=%.3f)", k, v)

    val_X = np.column_stack([val_preds[k] for k in qualified])
    test_X = np.column_stack([test_preds[k] for k in qualified])

    # ── Ensemble-Stacking : RidgeCV weighted meta-learner ──────────────
    pred_s, mdl_s, coefs_s = _ridge_positive(val_X, val_actual, test_X, "Stacking")
    val_pred_s = np.clip(mdl_s.predict(val_X), 0.0, None)

    # ── Ensemble-Blending : top-10 equal-weight average ────────────────
    top_k = 10
    r2_pairs = sorted(r2.items(), key=lambda kv: kv[1], reverse=True)
    top_names = [k for k, _ in r2_pairs[:top_k] if k in qualified]
    # If some of the top-10 are not in qualified (shouldn't happen since sorted on all),
    # fall back to the best of qualified until we have top_k or exhaust.
    if len(top_names) < top_k:
        for k, _ in r2_pairs:
            if k in qualified and k not in top_names:
                top_names.append(k)
            if len(top_names) >= top_k:
                break
    log.info("  [Blending] top-%d by val R²: %s", len(top_names),
             [(k, round(r2[k], 3)) for k in top_names])
    val_X_b = np.column_stack([val_preds[k] for k in top_names])
    test_X_b = np.column_stack([test_preds[k] for k in top_names])
    w_b = np.ones(len(top_names)) / len(top_names)
    pred_b = np.clip(test_X_b @ w_b, 0.0, None)
    val_pred_b = np.clip(val_X_b @ w_b, 0.0, None)

    # Write both
    for label, fname, val_pred, test_pred, member_info in [
        ("Stacking", "predictions_Ensemble-Stacking.csv", val_pred_s, pred_s,
         {"method": "RidgeCV weighted convex combination",
          "n_members": int(len(qualified)),
          "weights_nonzero": int(np.sum(coefs_s > 1e-4))}),
        ("Blending", "predictions_Ensemble-Blending.csv", val_pred_b, pred_b,
         {"method": f"top-{len(top_names)} equal-weight mean",
          "n_members": int(len(top_names)),
          "members": top_names}),
    ]:
        out_rows = []
        for i, (yt, yp) in enumerate(zip(val_actual, val_pred)):
            out_rows.append({"split": "val", "idx": i, "y_true": float(yt), "y_pred": float(yp)})
        for i, (yt, yp) in enumerate(zip(test_actual, test_pred)):
            out_rows.append({"split": "test", "idx": i + len(val_actual),
                             "y_true": float(yt), "y_pred": float(yp)})
        out = pd.DataFrame(out_rows)
        out_path = CSV / fname
        out.to_csv(out_path, index=False)
        uniq = out[out["split"] == "test"]["y_pred"].nunique()
        log.info("wrote %s — %s, test y_pred unique=%d, var=%.2f",
                 out_path.name, member_info, uniq, float(np.var(test_pred)))

    # Divergence sanity check
    diff = np.abs(pred_s - pred_b)
    log.info("Stacking vs Blending divergence: max|Δ|=%.3f, mean|Δ|=%.3f, corr=%.3f",
             float(diff.max()), float(diff.mean()),
             float(np.corrcoef(pred_s, pred_b)[0, 1]))


if __name__ == "__main__":
    regenerate()
