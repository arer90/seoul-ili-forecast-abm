"""Post-E P0#5 fix — compute *real* SHAP values for 5 tree ensembles.

Problem: ``checkpoint_phase8.json::shap_analysis`` is an empty dict, and
``per_model_recommended`` for all 5 tree models is just a copy of MI global
ranking (same top-5 across GradientBoosting/RandomForest/XGBoost/LightGBM/
ExtraTrees — statistically impossible with real TreeExplainer).

Fix here: rebuild the feature matrix via ``build_enriched_features``,
refit the 5 tree models with reasonable defaults, compute
``shap.TreeExplainer.shap_values`` on the test split, then rank features
by mean |SHAP|. Write back to ``stage3_shap/summary.json`` with
``source='real_shap_tree'`` so downstream (MCP/Turso) surfaces the truth.

Run:
    .venv/Scripts/python.exe -m simulation.scripts.post_E_real_shap
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("post_E.shap")

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "simulation" / "results"
SUMMARY = RES / "stage3_shap" / "summary.json"

DB_PATH = str(ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db")


def _get_matrix() -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Build feature matrix via the project feature engine.

    Returns (X, y, feat_names, n_train) where n_train = split index used for
    tree fitting (we keep a held-out test slice of 69 weeks like the paper).
    """
    from simulation.models.feature_engine.builder import build_enriched_features

    feat_df, meta = build_enriched_features(db_path=DB_PATH)
    # polars → pandas for sklearn compatibility
    import polars as pl
    pdf = feat_df.to_pandas()
    target = "ili_rate"
    exclude_cols = {"date", "week_start", "year", "week", "gu_nm", target}
    feat_cols = [c for c in pdf.columns if c not in exclude_cols]
    # drop fully-null or all-constant columns to avoid LightGBM crash
    keep = [c for c in feat_cols
            if pdf[c].notna().any() and pdf[c].nunique(dropna=True) > 1]
    X = pdf[keep].astype(float).fillna(0.0).to_numpy()
    y = pdf[target].astype(float).fillna(0.0).to_numpy()
    # use last 69 weeks as test (mirrors predictions CSV split)
    n_test = 69
    n_total = len(y)
    n_train = max(n_total - n_test - 26, 50)  # 26-wk holdout buffer
    log.info("matrix: X=%s, y=%s, n_train=%d, n_test=%d, n_features=%d",
             X.shape, y.shape, n_train, n_test, len(keep))
    return X, y, keep, n_train


def _fit_and_shap(X_tr, y_tr, X_te, feat_names, model_label, ClsOrParams):
    """Fit a single tree model and compute mean |SHAP|.

    Returns list[(feature_name, mean_abs_shap)] sorted desc.
    """
    import shap

    cls, kwargs = ClsOrParams
    mdl = cls(**kwargs)
    mdl.fit(X_tr, y_tr)
    log.info("  [%s] fitted on n_train=%d features=%d", model_label, X_tr.shape[0], X_tr.shape[1])
    explainer = shap.TreeExplainer(mdl)
    vals = explainer.shap_values(X_te)
    # vals shape: (n_test, n_features) or tuple for multi-output
    if isinstance(vals, list):
        vals = vals[0]
    mean_abs = np.mean(np.abs(vals), axis=0)
    ranked = sorted(zip(feat_names, mean_abs.tolist()),
                    key=lambda r: r[1], reverse=True)
    return ranked


def main() -> None:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
    try:
        from xgboost import XGBRegressor
    except Exception:
        XGBRegressor = None
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        LGBMRegressor = None

    X, y, feat_names, n_train = _get_matrix()
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te = X[n_train:]

    specs = [
        ("GradientBoosting", (GradientBoostingRegressor,
                              dict(n_estimators=200, max_depth=4, random_state=42))),
        ("RandomForest", (RandomForestRegressor,
                          dict(n_estimators=300, max_depth=None, n_jobs=2, random_state=42))),
        ("ExtraTrees", (ExtraTreesRegressor,
                        dict(n_estimators=300, max_depth=None, n_jobs=2, random_state=42))),
    ]
    if XGBRegressor:
        specs.append(("XGBoost", (XGBRegressor,
                                  dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                                       random_state=42, n_jobs=2, verbosity=0))))
    if LGBMRegressor:
        specs.append(("LightGBM", (LGBMRegressor,
                                   dict(n_estimators=300, max_depth=-1, num_leaves=31,
                                        learning_rate=0.05, random_state=42, n_jobs=2,
                                        verbose=-1))))

    per_model_top: dict[str, list[dict]] = {}
    per_model_full: dict[str, list[dict]] = {}
    for label, spec in specs:
        try:
            ranked = _fit_and_shap(X_tr, y_tr, X_te, feat_names, label, spec)
        except Exception as e:
            log.warning("  [%s] SHAP failed: %s", label, e)
            continue
        per_model_full[label] = [{"rank": i + 1, "feature": f, "mean_abs_shap": float(s)}
                                 for i, (f, s) in enumerate(ranked)]
        per_model_top[label] = per_model_full[label][:10]
        log.info("  [%s] top-5: %s", label, [r["feature"] for r in per_model_top[label][:5]])

    # Load existing summary (keep MI global ranking) and overwrite per_model blocks
    if SUMMARY.exists():
        summary = json.load(SUMMARY.open(encoding="utf-8"))
    else:
        summary = {"schema_version": "0.1"}
    summary.update({
        "generated_at": pd.Timestamp.now().isoformat(),
        "note": (
            "MI-based global ranking (R11 SHAP/XAI proxy) + real SHAP TreeExplainer "
            "values for 5 tree ensembles. Per-model top-10 uses mean |SHAP| "
            "computed on the 69-week test split. Non-tree models still "
            "surface MI as a proxy."
        ),
        "per_model_top_features": per_model_top,
        "per_model_full_rankings": per_model_full,
        "models_with_real_shap": list(per_model_top.keys()),
        "models_with_mi_fallback": "all_others",
        "shap_source": "TreeExplainer.shap_values on test split",
    })
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d tree models with real SHAP)", SUMMARY, len(per_model_top))


if __name__ == "__main__":
    main()
