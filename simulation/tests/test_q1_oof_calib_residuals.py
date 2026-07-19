"""Q1 (codex+gemini 수렴, 사용자 "y_train-pred_train는 validation이지?" catch):
the empirical-WIS PI must be calibrated from OUT-OF-SAMPLE residuals (prior-fold OOF), not
in-sample train residuals (which are optimistic → PI too narrow → over-confident). The WF-CV
loops thread each fold's OOS val residuals into the NEXT fold's PI calibration (fold 0 falls
back to in-sample). This pins the eval-level contract: when calib_residuals is supplied, it —
not the model's in-sample train residuals — drives the PI.

macOS: run PER-FILE.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _factory():
    import simulation.models.linear_models  # noqa: F401
    from simulation.models.base import REGISTRY
    return lambda: REGISTRY.get("ElasticNet")()


def _data(n_train=90, n_val=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_val, 4))
    y = 5.0 + 2.0 * X[:, 0] + rng.gamma(1.0, 0.5, n_train + n_val)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:], [f"f{i}" for i in range(4)]


def _run(calib):
    from simulation.pipeline.per_model_optimize import _evaluate_config
    Xtr, ytr, Xva, yva, cols = _data()
    return _evaluate_config(
        _factory(), Xtr, ytr, Xva, yva,
        transform_name="identity", scaler_name="none",
        feature_cols=cols, calib_residuals=calib)


def test_calib_residuals_drive_the_pi():
    """Different calibration-residual spreads → different WIS (the PI is calibrated from the
    supplied OOF residuals, not the model's in-sample train residuals)."""
    wis_narrow = _run(np.full(40, 0.05))["wis"]
    wis_wide = _run(np.full(40, 10.0))["wis"]
    assert abs(wis_narrow - wis_wide) > 1e-6, (
        f"calib_residuals ignored: narrow={wis_narrow} wide={wis_wide}")


def test_eval_returns_val_residuals_for_threading():
    """The eval exposes its OOS val residuals so the CV loop can feed them to the next fold."""
    res = _run(None)
    assert "_val_residuals" in res, "eval must return _val_residuals for prior-fold threading"
    vr = np.asarray(res["_val_residuals"], float).ravel()
    assert vr.size == 40 and np.all(np.isfinite(vr))


def test_no_calib_falls_back_to_insample():
    """fold 0 (no prior residuals) must still produce a finite WIS via in-sample fallback."""
    res = _run(None)
    assert np.isfinite(res["wis"]) and res["wis"] > 0


def test_oof_loop_threads_prior_fold_residuals():
    """_oof_cv_wis_hier must thread fold k-1's OOS residuals into fold k (not all in-sample).
    Spy: capture the calib_residuals each fold receives — fold 0 None, fold ≥1 an array."""
    import simulation.pipeline.per_model_optimize as P
    import simulation.pipeline._inline_optuna_3stage as I
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    seen = []
    orig = P._evaluate_config_hierarchical

    def _spy(*a, **k):
        seen.append(k.get("calib_residuals"))
        return orig(*a, **k)

    import numpy as _np
    rng = _np.random.default_rng(1)
    X = rng.normal(size=(160, 4))
    y = 5.0 + 2.0 * X[:, 0] + rng.gamma(1.0, 0.5, 160)
    fac = _factory()
    # need frozen hier params
    cap = {}

    def _o(t):
        P._evaluate_config_hierarchical(fac, X[:90], y[:90], X[90:120], y[90:120],
                                        optuna_trial=t, feature_cols=[f"f{i}" for i in range(4)])
        cap["p"] = dict(t.params)
        return 0.0
    optuna.create_study(direction="minimize").optimize(_o, n_trials=1)

    import unittest.mock as mock
    with mock.patch.object(P, "_evaluate_config_hierarchical", _spy):
        I._oof_cv_wis_hier(fac, X, y, cap["p"],
                           feature_cols=[f"f{i}" for i in range(4)], n_folds=3)

    assert len(seen) >= 2, f"expected ≥2 folds, saw {len(seen)}"
    assert seen[0] is None, f"fold 0 should have no prior residuals, got {type(seen[0])}"
    later = [s for s in seen[1:] if s is not None]
    assert later, "folds ≥1 must receive prior-fold OOS residuals (none threaded)"
