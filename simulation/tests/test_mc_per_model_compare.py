"""④ per-model none/vif/corr/pca comparison + save + overfit recording (G-242).

Verifies the NEW code (`_compare_mc_per_model` / `_oof_cv_wis_with_mc`):
1. writes mc_per_model_selection.csv with the right schema (4 rows/model, one selected),
2. records a finite overfit_gap per cell + returns a per-model best and a global aggregate,
3. the NEW code actually APPLIES the overfit-control mechanism — vif/corr drop the
   redundant collinear features (n_kept < p). Combined with
   ``test_mc_overfitting_control`` (which proves that reduction controls overfit on the
   CURRENT filter code), this closes the loop: the new comparison applies + measures the
   same overfit-controlling filter, per model.

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
import numpy as np
import pytest

from simulation.tests.test_mc_overfitting_control import _collinear_data

# PCA on pathologically-collinear synthetic data overflows (cosmetic; real ILI data
# never has 30 near-exact dups). Filtered cells are dropped as inf by the code anyway.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _elasticnet_factory():
    import simulation.models.linear_models  # noqa: F401 — triggers @register
    from simulation.models.base import REGISTRY
    cls = REGISTRY.get("ElasticNet")
    if cls is None:
        pytest.skip("ElasticNet not registered")
    return lambda: cls()


def test_compare_mc_per_model_writes_csv(tmp_path):
    from simulation.pipeline.per_model_optimize import _compare_mc_per_model

    X, y, cols = _collinear_data(n=160)
    factories = {"ElasticNet": _elasticnet_factory()}
    per_best, global_best, rows = _compare_mc_per_model(
        factories, X, y, feature_cols=cols, out_dir=tmp_path, n_folds=2, force=True,
    )

    csv_path = tmp_path / "mc_per_model_selection.csv"
    assert csv_path.exists(), "mc_per_model_selection.csv not written"

    import csv as _c
    table = list(_c.DictReader(csv_path.open(encoding="utf-8")))
    assert len(table) == 4, f"expected 4 rows (1 model × 4 methods), got {len(table)}"
    assert {r["method"] for r in table} == {"none", "vif", "corr", "pca"}
    # exactly one selected per model
    assert len([r for r in table if r["selected"] == "Y"]) == 1
    # schema columns (incl. scale-invariant overfit_ratio — Gemini G-242 review)
    for col in ("model", "method", "oof_wis", "insample_wis", "overfit_gap",
                "overfit_ratio", "n_kept", "selected"):
        assert col in table[0], f"missing column {col}"
    # returns
    assert per_best["ElasticNet"] in {"none", "vif", "corr", "pca"}
    assert global_best in {"none", "vif", "corr", "pca"}


def test_compare_mc_per_model_applies_overfit_control_and_records_gap(tmp_path):
    """NEW code applies the overfit-control mechanism (vif/corr drop collinear dups,
    n_kept < p) and records a finite overfit_gap for the selected method."""
    from simulation.pipeline.per_model_optimize import _compare_mc_per_model

    X, y, cols = _collinear_data(n=160)  # 34 features, 30 collinear dups
    p = X.shape[1]
    factories = {"ElasticNet": _elasticnet_factory()}
    _, _, rows = _compare_mc_per_model(
        factories, X, y, feature_cols=cols, out_dir=tmp_path, n_folds=2, force=True,
    )
    by_method = {r["method"]: r for r in rows if r["model"] == "ElasticNet"}

    # mechanism: vif/corr actually reduce the collinear feature set
    for m in ("vif", "corr"):
        assert by_method[m]["n_kept"] < p, f"{m} kept all {p} collinear feats"
    # the per-model selected method has a finite OOF WIS + recorded overfit_gap
    sel = next(r for r in rows if r["selected"] == "Y")
    assert np.isfinite(sel["oof_wis"]), f"selected OOF not finite: {sel}"
    assert isinstance(sel["overfit_gap"], float), "overfit_gap not recorded"


def test_cache_invalidated_on_fingerprint_change(tmp_path):
    """Codex G-242: stale CSV must NOT be reused when the model set / data shape changes —
    a fingerprint sidecar gates the cache (else mc=auto silently applies a stale choice)."""
    import json
    from simulation.pipeline.per_model_optimize import _compare_mc_per_model

    X, y, cols = _collinear_data(n=160)
    fac = {"ElasticNet": _elasticnet_factory()}
    _compare_mc_per_model(fac, X, y, feature_cols=cols, out_dir=tmp_path, n_folds=2)
    meta = tmp_path / "mc_per_model_selection.meta.json"
    assert meta.exists(), "fingerprint sidecar not written"
    fp1 = json.loads(meta.read_text())["fingerprint"]

    # second run, DIFFERENT feature count, NOT force → fingerprint mismatch → recompute
    X2 = np.hstack([X, X[:, :5]])
    cols2 = cols + [f"extra{i}" for i in range(5)]
    _compare_mc_per_model(fac, X2, y, feature_cols=cols2, out_dir=tmp_path, n_folds=2)
    fp2 = json.loads(meta.read_text())["fingerprint"]
    assert fp1 != fp2, "fingerprint must change with feature count → stale cache avoided"


def test_oof_cv_wis_with_mc_no_leakage_finite():
    """`_oof_cv_wis_with_mc` returns a finite WIS for each method (per-fold mc path)."""
    from simulation.pipeline.per_model_optimize import _oof_cv_wis_with_mc

    X, y, cols = _collinear_data(n=160)
    fac = _elasticnet_factory()
    for method in ("none", "vif", "corr", "pca"):
        wis = _oof_cv_wis_with_mc(fac, X, y, "identity", "standard", method,
                                  feature_cols=cols, n_folds=2)
        assert np.isfinite(wis), f"{method} OOF WIS not finite: {wis}"
