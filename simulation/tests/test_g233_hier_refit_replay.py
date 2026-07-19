"""G-233 hierarchical-preproc refit replay — regression tests.

Bug (pre-fix): the Stage-1 hierarchical preproc Optuna stores its choice on the best cell as
a MODE MARKER (``transform="HIER_categorical"``, ``scaler="HIER_individual"``), not a flat
transform/scaler name. ``optimize_one_model`` then called the refit consumers
(``_refit_and_predict_test`` / ``_refit_and_predict_real`` / ``_oof_cv_metrics``) with those
markers, and ``_refit_and_predict_test`` applies the y-transform BY NAME
(``_apply_single_y_transform``) → ``ValueError: Unknown Y transform: HIER_categorical`` →
caught as "test refit exception" → **empty test_metrics** → the model could not be
test-evaluated or championed.

Fix: the best cell now carries ``hier_frozen_params`` (the Optuna ``trial.params``). The refit
consumers accept it and REPLAY the exact sampled hierarchical preproc, re-fit on the full
(train+val) pool — identical to the D4 ``_oof_cv_wis_hier`` FixedTrial replay — and the
ChampionArtifact persists ``hier_y_state`` so Pinf inference (serving) reproduces the same
inverse (state-based, picklable).

These tests assert: (a) no "error" + finite metrics + non-empty predictions for every
hierarchical mode, (b) the persisted artifact reproduces the refit predictions, (c) a HIER
marker WITHOUT frozen_params degrades to identity instead of crashing, (d)
``apply_y_preproc_inverse_only`` covers every METRIC_Y_TRANSFORMS member.

Run per-file (macOS OpenMP):  .venv/bin/python -m pytest simulation/tests/test_g233_hier_refit_replay.py -q
Or standalone:               .venv/bin/python simulation/tests/test_g233_hier_refit_replay.py
"""
from __future__ import annotations

import numpy as np
import optuna

from simulation.pipeline.per_model_optimize import _refit_and_predict_test
from simulation.pipeline.preproc_optuna_hierarchical import (
    METRIC_Y_TRANSFORMS,
    apply_y_preproc_inverse_only,
    suggest_y_preproc,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Feature names chosen so _categorize_feature_groups yields >=2 groups (for x_mode="group").
_FEATURE_COLS = ["ili_rate_lag1", "ili_rate_lag2", "temp_avg", "gt_flu", "rt_mobility"]


class _StubForecaster:
    """Minimal BaseForecaster-compatible stub: deterministic Ridge under the hood."""

    def fit(self, X, y, **_kw):
        from sklearn.linear_model import Ridge
        self._m = Ridge(alpha=1.0).fit(np.asarray(X), np.asarray(y).ravel())
        return self

    def predict(self, X, **_kw):
        return np.asarray(self._m.predict(np.asarray(X)), dtype=np.float64)


def _factory():
    return _StubForecaster()


def _make_data(n_pool=120, n_test=24, seed=42):
    rng = np.random.RandomState(seed)
    p = len(_FEATURE_COLS)
    X_pool = rng.randn(n_pool, p)
    X_test = rng.randn(n_test, p)
    beta = rng.randn(p) * 0.6
    # strictly-positive y so log/sqrt/boxcox/anscombe are all well-defined
    y_pool = np.maximum(0.05, X_pool @ beta + rng.randn(n_pool) * 0.3 + 4.0)
    y_test = np.maximum(0.05, X_test @ beta + rng.randn(n_test) * 0.3 + 4.0)
    return X_pool, y_pool, X_test, y_test


def _run(hier_params, *, return_fitted_model=False):
    X_pool, y_pool, X_test, y_test = _make_data()
    sigma = max(float(np.std(y_pool)), 1e-3)
    return _refit_and_predict_test(
        _factory,
        transform_name="HIER_" + str(hier_params.get("y_mode", "?")),
        scaler_name="HIER_" + str(hier_params.get("x_mode", "?")),
        X_train_pool=X_pool, y_train_pool=y_pool,
        X_test=X_test, y_test=y_test,
        feature_indices=None,
        sigma_for_wis=sigma,
        feature_cols=_FEATURE_COLS,
        return_fitted_model=return_fitted_model,
        hier_frozen_params=hier_params,
        hier_max_chain_length=2,
    )


def _assert_good(res, label):
    assert isinstance(res, dict), f"{label}: not a dict"
    assert "error" not in res, f"{label}: refit returned error: {res.get('error')}"
    for k in ("wis", "mae", "r2"):
        v = res.get(k)
        assert v is not None and np.isfinite(v), f"{label}: metric {k} not finite ({v})"
    preds = res.get("predictions")
    assert preds is not None and len(preds) == 24, f"{label}: predictions wrong ({preds})"
    assert np.all(np.isfinite(np.asarray(preds, dtype=np.float64))), f"{label}: non-finite preds"


# ── (a) every hierarchical mode refits without the "Unknown Y transform" crash ──

def test_refit_hier_individual_log1p_robust():
    res = _run({"y_mode": "individual", "y_individual": "log1p",
                "x_mode": "individual", "x_individual": "robust"})
    _assert_good(res, "individual/log1p + individual/robust")


def test_refit_hier_x_group():
    res = _run({"y_mode": "none",
                "x_mode": "group",
                "x_group_lag_ili": "standard",
                "x_group_weather": "robust",
                "x_group_search_trend": "quantile",
                "x_group_mobility_rt": "standard"})
    _assert_good(res, "x_mode=group ColumnTransformer")


def test_refit_hier_y_group_chain():
    res = _run({"y_mode": "group", "y_group_n": 2,
                "y_group_0": "log1p", "y_group_1": "sqrt",
                "x_mode": "none"})
    _assert_good(res, "y_mode=group chain")


def test_refit_hier_categorical_boxcox():
    res = _run({"y_mode": "categorical", "y_categorical": "boxcox", "x_mode": "none"})
    _assert_good(res, "y categorical boxcox")


def test_refit_hier_vst_anscombe():
    res = _run({"y_mode": "individual", "y_individual": "anscombe", "x_mode": "none"})
    _assert_good(res, "y individual anscombe (VST)")


# ── (b) the persisted ChampionArtifact reproduces the refit predictions ──

def test_refit_hier_artifact_roundtrip():
    import pickle
    from simulation.utils.model_artifact import make_artifact

    res = _run({"y_mode": "individual", "y_individual": "log1p",
                "x_mode": "group",
                "x_group_lag_ili": "robust", "x_group_weather": "standard",
                "x_group_search_trend": "quantile", "x_group_mobility_rt": "robust"},
               return_fitted_model=True)
    _assert_good(res, "artifact roundtrip refit")
    model = res["_fitted_model"]
    st = res["_artifact_state"]
    assert st.get("hier_y_state") is not None, "artifact_state missing hier_y_state"

    art = make_artifact(
        model=model,
        transform_name=st.get("transform_name", "identity"),
        transform_inv_obj=st.get("transform_inv_obj"),
        fitted_scaler=st.get("fitted_scaler"),
        feature_indices=st.get("feature_indices"),
        hier_y_state=st.get("hier_y_state"),
        model_name="StubForecaster",
    )
    art2 = pickle.loads(art.to_pickle_bytes())  # pickle survives (no closures)

    _, _, X_test, _ = _make_data()
    got = np.asarray(art2.predict(X_test), dtype=np.float64)
    want = np.asarray(res["predictions"], dtype=np.float64)
    assert got.shape == want.shape, f"artifact pred shape {got.shape} vs {want.shape}"
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6), (
        f"artifact predictions diverge from refit: max|Δ|={np.max(np.abs(got - want)):.3e}")


# ── (c) HIER marker without frozen_params degrades to identity (no crash) ──

def test_hier_marker_without_params_degrades():
    X_pool, y_pool, X_test, y_test = _make_data()
    res = _refit_and_predict_test(
        _factory,
        transform_name="HIER_categorical", scaler_name="HIER_individual",
        X_train_pool=X_pool, y_train_pool=y_pool,
        X_test=X_test, y_test=y_test,
        feature_cols=_FEATURE_COLS,
        # NOTE: no hier_frozen_params → must NOT raise "Unknown Y transform"
    )
    assert "error" not in res or "Unknown Y transform" not in str(res.get("error")), (
        f"HIER marker without params should degrade, got: {res.get('error')}")
    _assert_good(res, "HIER marker degrade-to-identity")


# ── (d) state-based inverse covers every METRIC_Y_TRANSFORMS member ──

def test_apply_inverse_covers_all_metric_y():
    rng = np.random.RandomState(0)
    y = np.maximum(0.05, rng.lognormal(1.5, 0.4, size=80))
    for t in METRIC_Y_TRANSFORMS:
        trial = optuna.trial.FixedTrial({"y_mode": "individual", "y_individual": t})
        y_t, closure_inv, state = suggest_y_preproc(trial, y)
        # state-based inverse must not raise and must match the closure inverse
        state_inv = apply_y_preproc_inverse_only(y_t, state)
        clos = np.asarray(closure_inv(y_t), dtype=np.float64)
        assert np.all(np.isfinite(state_inv)), f"{t}: state inverse non-finite"
        assert np.allclose(state_inv, clos, rtol=1e-4, atol=1e-5), (
            f"{t}: state inverse != closure inverse (max|Δ|={np.max(np.abs(state_inv - clos)):.3e})")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)
