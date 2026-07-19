"""C (refit-HIER data leakage): the hierarchical-preproc REFIT path
(`_refit_and_predict_test`) must fit its Y-transform + X-scaler on the TRAIN pool ONLY —
never on the held-out test slab. The user asked to *verify* there is no leakage; this pins
it as a permanent regression test.

Proof of no-leakage (behavioral): hold the train pool fixed, vary the test slab wildly
(second slab scaled ×99) → the fitted preproc state (`hier_y_state`, fit on the pool) must be
BYTE-IDENTICAL. If the refit ever peeked at test to fit the transform/scaler, the state would
move with the test data. A companion test confirms the refit still *uses* test X for the
prediction (so the invariance is genuine blindness-during-fit, not test being ignored).

macOS: run PER-FILE.
"""
import numpy as np
import optuna
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _factory():
    import simulation.models.linear_models  # noqa: F401  (register)
    from simulation.models.base import REGISTRY
    return lambda: REGISTRY.get("ElasticNet")()


def _frozen_hier_params(fac, X, y, cols):
    """Capture a concrete hierarchical preproc param set from one Optuna trial."""
    import simulation.pipeline.per_model_optimize as m
    cap = {}

    def _obj(t):
        m._evaluate_config_hierarchical(
            fac, X[:90], y[:90], X[90:120], y[90:120],
            optuna_trial=t, feature_cols=cols, sigma_for_wis=1.0)
        cap["p"] = dict(t.params)
        return 0.0

    optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=3)
    ).optimize(_obj, n_trials=1)
    return cap["p"]


def _data(seed=0, n=200, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # Strong + positive linear signal so a regularized ElasticNet keeps non-zero coefs
    # (the companion sanity test needs predictions that actually track test X); positive
    # so the hierarchical log1p/sqrt y-transforms stay valid.
    y = 20.0 + 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.gamma(1.0, 0.5, n)
    return X, y, [f"f{i}" for i in range(p)]


def _refit(fac, params, cols, pool_X, pool_y, test_X, test_y):
    import simulation.pipeline.per_model_optimize as m
    return m._refit_and_predict_test(
        fac, transform_name="HIER_x", scaler_name="HIER",
        X_train_pool=pool_X, y_train_pool=pool_y,
        X_test=test_X, y_test=test_y, feature_cols=cols,
        hier_frozen_params=params, return_fitted_model=True)


def test_refit_hier_preproc_is_blind_to_test_slab():
    """Same train pool + two very different test slabs → fitted preproc state identical."""
    X, y, cols = _data()
    fac = _factory()
    params = _frozen_hier_params(fac, X, y, cols)
    pool_X, pool_y = X[:160], y[:160]

    r1 = _refit(fac, params, cols, pool_X, pool_y, X[160:180], y[160:180])
    r2 = _refit(fac, params, cols, pool_X, pool_y, X[180:200], y[180:200] * 99.0)

    s1 = r1.get("_artifact_state", {}).get("hier_y_state")
    s2 = r2.get("_artifact_state", {}).get("hier_y_state")
    assert s1 is not None and s2 is not None, "hier_y_state missing — refit path changed?"
    assert s1 == s2, (
        f"LEAKAGE: preproc fit moved with the test slab → test peeked during fit.\n"
        f"  s1={s1}\n  s2={s2}")


def test_refit_hier_still_uses_test_x_for_prediction():
    """Sanity: the invariance above is fit-blindness, NOT the refit ignoring test entirely —
    different test X must yield different predictions."""
    X, y, cols = _data()
    fac = _factory()
    params = _frozen_hier_params(fac, X, y, cols)
    pool_X, pool_y = X[:160], y[:160]

    r1 = _refit(fac, params, cols, pool_X, pool_y, X[160:180], y[160:180])
    r2 = _refit(fac, params, cols, pool_X, pool_y, X[160:180] + 5.0, y[160:180])

    p1 = np.asarray(r1.get("predictions"), float).ravel()
    p2 = np.asarray(r2.get("predictions"), float).ravel()
    assert p1.shape == p2.shape and p1.size > 0
    assert not np.allclose(p1, p2), (
        "refit ignores test X (predictions identical for shifted test) — not a real model")
