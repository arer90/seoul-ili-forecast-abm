"""G-294: the Stage-2 feature guard must score OOF under the SAME hierarchical preproc
Stage-1 chose — not via the flat name-by-name path.

Bug (reproduced by `test_flat_oof_raises_on_hier_marker`): Stage-1 records its chosen preproc
as a mode MARKER (`best["transform"] = "HIER_<mode>"`), with the real Optuna params in
`best["preproc_optuna_params"]`. The Stage-2 guard routed that marker into the flat
`_oof_cv_wis(transform_name="HIER_*", ...)`, which calls `_apply_single_y_transform("HIER_*")`
→ `raise ValueError("Unknown Y transform: HIER_*")`. The fold loop does not wrap that call, so
it propagated out of `_oof_cv_wis` and was caught by the guard's blanket `except` → the guard
ALWAYS fell back to the stability subset ("SUBSET(guard 실패: ValueError)"). Both the nested
1-SE size-path AND the binary "각 단계 개선 보장" margin guard were therefore silently dead for
every HIER-preproc model (no crash / no leak / no report damage — just a nullified invariant).

Fix: route the guard's OOF to `_oof_cv_wis_hier(..., best["preproc_optuna_params"],
return_folds=...)`, which REPLAYS the frozen hierarchical preproc per fold (same FixedTrial
path as Stage-1), and add `return_folds` so the nested size-path keeps its per-fold list.

macOS: run PER-FILE (single-process pytest segfaults at LightGBM CQR — OpenMP conflict).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _skewed_data(n=180, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    # positive, right-skewed target → identity vs log1p genuinely matters
    y = np.abs(3.0 * X[:, 0] - 2.0 * X[:, 1]) + rng.gamma(2.0, 1.5, n)
    return X, y


def _elasticnet_factory():
    import simulation.models.linear_models  # noqa: F401 — register
    from simulation.models.base import REGISTRY
    cls = REGISTRY.get("ElasticNet")
    if cls is None:
        pytest.skip("ElasticNet not registered")
    return lambda: cls()


def _capture_frozen_params(fac, X, y, cols):
    """Mirror test_d4: run a short preproc study, return one trial's frozen params."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from simulation.pipeline.per_model_optimize import _evaluate_config_hierarchical

    captured = {}

    def _obj(t):
        c = _evaluate_config_hierarchical(
            fac, X[:120], y[:120], X[120:], y[120:], optuna_trial=t,
            feature_cols=cols, sigma_for_wis=1.0)
        captured[(c.get("transform"), c.get("scaler"))] = dict(t.params)
        return c.get("wis", 1e9)

    optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=3)
    ).optimize(_obj, n_trials=10, show_progress_bar=False)
    if not captured:
        pytest.skip("synthetic preproc study captured no params")
    # any one valid frozen-params dict (the marker key is "HIER_<mode>")
    (marker_tf, _marker_sc), params = next(iter(captured.items()))
    assert str(marker_tf).startswith("HIER_"), (
        f"Stage-1 transform should be a HIER marker, got {marker_tf!r}")
    return params, str(marker_tf), str(_marker_sc)


def test_flat_oof_raises_on_hier_marker():
    """Documents the bug: the flat OOF path cannot consume a 'HIER_*' marker → it raises,
    which is exactly why the guard's `except` swallowed it into an always-subset fallback."""
    from simulation.pipeline.per_model_optimize import _oof_cv_wis

    X, y = _skewed_data()
    fac = _elasticnet_factory()
    cols = [f"f{i}" for i in range(X.shape[1])]

    with pytest.raises(ValueError, match="Unknown Y transform"):
        _oof_cv_wis(fac, X, y, "HIER_individual", "HIER_none",
                    feature_cols=cols, n_folds=2)


def test_hier_replay_return_folds_shape_and_backcompat():
    """Fix surface: _oof_cv_wis_hier now supports return_folds (for the nested size-path),
    and the default scalar return is preserved for existing callers."""
    from simulation.pipeline._inline_optuna_3stage import _oof_cv_wis_hier

    X, y = _skewed_data()
    fac = _elasticnet_factory()
    cols = [f"f{i}" for i in range(X.shape[1])]
    params, _tf_marker, _sc_marker = _capture_frozen_params(fac, X, y, cols)

    # return_folds=False → scalar (back-compat: line 611 + existing tests)
    scalar = _oof_cv_wis_hier(fac, X, y, params, feature_cols=cols, n_folds=2)
    assert np.isscalar(scalar) or isinstance(scalar, float)
    assert np.isfinite(scalar), "HIER replay OOF must be finite for a valid frozen preproc"

    # return_folds=True → (agg, per_fold_list) mirroring flat _oof_cv_wis
    agg, folds = _oof_cv_wis_hier(
        fac, X, y, params, feature_cols=cols, n_folds=2, return_folds=True)
    assert np.isfinite(agg)
    assert isinstance(folds, list) and len(folds) >= 1
    assert all(np.isfinite(f) for f in folds)
    # the aggregate from the two calls must agree (same frozen preproc, deterministic replay)
    assert abs(agg - scalar) < 1e-9


def test_guard_invariant_restored_full_vs_subset_both_finite():
    """The actual user-facing invariant: with a HIER preproc, the guard can now compute a
    FINITE full-feature OOF and a FINITE subset OOF, so feature_guard_keep actually decides
    (subset vs full) instead of the dead always-subset fallback."""
    from simulation.pipeline._inline_optuna_3stage import _oof_cv_wis_hier
    from simulation.pipeline.feature_select_corr1se import feature_guard_keep

    X, y = _skewed_data()
    fac = _elasticnet_factory()
    cols = [f"f{i}" for i in range(X.shape[1])]
    params, _tf_marker, _sc_marker = _capture_frozen_params(fac, X, y, cols)

    oof_full = _oof_cv_wis_hier(
        fac, X, y, params, feature_indices=None, feature_cols=cols, n_folds=2)
    oof_sel = _oof_cv_wis_hier(
        fac, X, y, params, feature_indices=[0, 1, 2, 3], feature_cols=cols, n_folds=2)

    assert np.isfinite(oof_full) and np.isfinite(oof_sel), (
        f"guard OOF must be finite under HIER replay (was raising before): "
        f"full={oof_full} sel={oof_sel}")
    # the guard's decision function runs and returns a real bool (not crashing on inf)
    decision = feature_guard_keep(oof_full, oof_sel, 0.02, prefer_subset=True)
    assert isinstance(decision, (bool, np.bool_))
