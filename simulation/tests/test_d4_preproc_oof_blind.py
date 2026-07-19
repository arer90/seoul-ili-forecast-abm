"""D4: the per-model preproc-Optuna OOF objective is preproc-BLIND (bug demonstration).

`_stage1_preproc_optuna_inline._preproc_objective` returns, under best_by=oof_cv (the live
default), `_oof_cv_wis(transform="identity", scaler="robust", ...)` — FIXED — regardless of
the trial's SAMPLED transform/scaler. This test shows `_oof_cv_wis` genuinely VARIES with
the transform/scaler, so hardcoding identity/robust makes the objective unable to tell
preprocs apart → every trial scores the same → arbitrary (trial-0) selection.

D4 conclusion: OOF-CV is the RIGHT validation (vs val-single, G-132). The CONTENT is the
bug — it must score the SAMPLED preproc per fold, not a fixed one.

macOS: run PER-FILE.
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


def test_oof_varies_with_preproc_so_fixed_objective_is_blind():
    from simulation.pipeline.per_model_optimize import _oof_cv_wis

    X, y = _skewed_data()
    fac = _elasticnet_factory()
    cols = [f"f{i}" for i in range(X.shape[1])]

    wis_identity = _oof_cv_wis(fac, X, y, "identity", "robust", feature_cols=cols, n_folds=2)
    wis_log1p = _oof_cv_wis(fac, X, y, "log1p", "standard", feature_cols=cols, n_folds=2)

    assert np.isfinite(wis_identity) and np.isfinite(wis_log1p)
    # preproc genuinely moves the OOF score:
    assert abs(wis_identity - wis_log1p) > 1e-6, (
        f"preproc should matter but OOF identical → cannot demonstrate the bug: "
        f"identity={wis_identity:.4f} log1p={wis_log1p:.4f}")
    # ⇒ an objective that ALWAYS scores identity/robust is blind to this → arbitrary pick.


def test_oof_hier_replay_varies_with_sampled_preproc():
    """D4 FIX: `_oof_cv_wis_hier` (per-fold FixedTrial replay) gives DIFFERENT OOF for
    different sampled preprocs — so the fixed objective can now distinguish configs
    (under the bug all trials scored the same fixed identity/robust OOF)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from simulation.pipeline.per_model_optimize import _evaluate_config_hierarchical
    from simulation.pipeline._inline_optuna_3stage import _oof_cv_wis_hier

    X, y = _skewed_data()
    fac = _elasticnet_factory()
    cols = [f"f{i}" for i in range(X.shape[1])]

    # capture params for ≥2 DISTINCT sampled preprocs
    seen = {}

    def _obj(t):
        c = _evaluate_config_hierarchical(
            fac, X[:120], y[:120], X[120:], y[120:], optuna_trial=t,
            feature_cols=cols, sigma_for_wis=1.0)
        seen[(c.get("transform"), c.get("scaler"))] = dict(t.params)
        return c.get("wis", 1e9)

    optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=3)
    ).optimize(_obj, n_trials=14, show_progress_bar=False)
    if len(seen) < 2:
        pytest.skip("synthetic run did not sample ≥2 distinct preprocs")

    p0, p1 = list(seen.values())[:2]
    oof0 = _oof_cv_wis_hier(fac, X, y, p0, feature_cols=cols, n_folds=2)
    oof1 = _oof_cv_wis_hier(fac, X, y, p1, feature_cols=cols, n_folds=2)
    assert np.isfinite(oof0) and np.isfinite(oof1)
    assert abs(oof0 - oof1) > 1e-9, (
        f"replayed OOF must differ across distinct preprocs (fix works): {oof0} vs {oof1}")
