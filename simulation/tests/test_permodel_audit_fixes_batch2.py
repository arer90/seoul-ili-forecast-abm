"""Per-model audit fixes — Batch 2 (prediction/transform logic, burn-zone — round-trip TDD).

G-298: trees clamped predictions with np.maximum(pred,0) in TRANSFORMED y-space, before the
       phase-13 inverse. Under a median-centered transform (mcmc_robust/laplace) that floored
       legitimate sub-median predictions (transformed<0) up to the median after the affine
       inverse → trough/quiet-season upward bias. Removing the wrong-space clamp is safe because
       trees are bounded by their training leaf range, so the inverse maps to [min_y, max_y] ≥ 0.

macOS: run PER-FILE.
"""
import numpy as np
import pytest


@pytest.mark.parametrize("transform", ["mcmc_robust", "laplace"])
def test_g298_tree_predict_preserves_sub_median_under_centered_transform(transform):
    """The clamp-removal lets trees predict below the training median (bug fix), while the
    train-range bound keeps original-unit predictions ≥ 0."""
    xgb = pytest.importorskip("xgboost")
    from simulation.models.tree_models import XGBoostForecaster
    from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform

    rng = np.random.RandomState(0)
    n = 140
    X = rng.normal(size=(n, 4))
    y = 12.0 + 9.0 * X[:, 0] + rng.normal(scale=0.4, size=n)
    y = np.clip(y, 0.0, None)
    med = float(np.median(y))

    # forward centered transform → sub-median maps to NEGATIVE transformed values
    y_t, inv, _ = _apply_single_y_transform(y, transform)
    assert y_t.min() < 0, "centered transform must produce negative (sub-median) targets"

    # fit a raw xgboost on transformed y (fast; bypass the Optuna fit())
    raw = xgb.XGBRegressor(n_estimators=60, max_depth=3, random_state=0).fit(X, y_t)
    f = XGBoostForecaster.__new__(XGBoostForecaster)
    f._model = raw
    f._fitted = True

    pred_t = f.predict(X)
    # HALF 1 (model): predict must NOT floor at 0 in transformed space → sub-median preserved
    assert pred_t.min() < 0, (
        f"transformed predictions must include negatives (sub-median); a transformed-space "
        f"clamp would have floored them at 0. min={pred_t.min()}")

    # HALF 2 (pipeline): ILI≥0 domain floor applied in ORIGINAL units (sanitize nonneg=True),
    #   exactly as _evaluate_config*/_refit_and_predict_* now do (G-298).
    from simulation.models.safety import sanitize_predictions
    pred = sanitize_predictions(np.asarray(inv(pred_t), dtype=float), nonneg=True)

    # the floored-to-median bug is gone: predictions reach below the training median
    assert pred.min() < med, (
        f"predictions must reach below the training median {med:.3f}; got min {pred.min():.3f} "
        f"(the old transformed-space clamp pinned trough weeks at the median)")
    # non-negativity holds in ORIGINAL units (pipeline floor catches GBM's slight sub-min extrapolation)
    assert pred.min() >= 0.0, f"predictions must stay ≥ 0 in original units; got {pred.min()}"

    # CONTRAST: the OLD transformed-space clamp would have pinned trough weeks at/above the median
    old_pred = np.asarray(inv(np.maximum(pred_t, 0.0)), dtype=float)
    assert old_pred.min() >= med - 1e-6, (
        f"sanity: the old transformed clamp should have pinned the trough at the median "
        f"(min {old_pred.min():.3f} vs median {med:.3f}) — confirms the bug it caused")


def test_g298_tree_predict_source_has_no_transformed_clamp():
    """All three active tree predict() bodies no longer wrap in np.maximum(..., 0)."""
    import inspect
    from simulation.models.tree_models import (
        XGBoostForecaster, LightGBMForecaster, RandomForestForecaster,
    )
    for cls in (XGBoostForecaster, LightGBMForecaster, RandomForestForecaster):
        src = inspect.getsource(cls.predict)
        assert "np.maximum(self._model.predict" not in src, (
            f"{cls.__name__}.predict still floors in transformed space")


def test_g298_pipeline_floors_original_units_at_all_inverse_sites():
    """Every original-units inverse site applies the ILI≥0 domain floor (selection == eval)."""
    import inspect
    from simulation.pipeline import per_model_optimize as P
    # selection (flat + hier) use np.maximum(..., 0.0); refit/real use sanitize(nonneg=True)
    src_flat = inspect.getsource(P._evaluate_config)
    src_hier = inspect.getsource(P._evaluate_config_hierarchical)
    src_test = inspect.getsource(P._refit_and_predict_test)
    src_real = inspect.getsource(P._refit_and_predict_real)
    assert "transform_inv(y_pred_t), dtype=np.float64), 0.0)" in src_flat, "flat OOF must floor ≥0"
    assert "inv_y_fn(y_pred_t)).ravel(), 0.0)" in src_hier, "hier OOF must floor ≥0"
    assert "sanitize_predictions(y_pred, nonneg=True)" in src_test, "refit must floor ≥0"
    assert "nonneg=True" in src_real, "rolling refit must floor ≥0"


def test_g299_feature_names_threaded_symmetrically():
    """selection (already wired) AND refit AND rolling all pass feature_names so the
    OverseasTransfer encoder is ON in eval/deploy exactly as it was during selection."""
    import inspect
    from simulation.pipeline import per_model_optimize as P
    src_test = inspect.getsource(P._refit_and_predict_test)
    src_real = inspect.getsource(P._refit_and_predict_real)
    assert "model.fit(X_tr_s, y_tr_t, feature_names=feat_names_use)" in src_test, (
        "reported/deploy refit must thread feature_names (encoder parity with selection)")
    assert "feature_names=_feat_names_step" in src_real, (
        "rolling refit must thread feature_names")


def test_g298_overseas_predict_no_transformed_lower_floor():
    """OverseasTransfer.predict keeps only the UPPER cap (np.minimum); the wrong-space lower
    0.0 floor is gone (original-units ≥0 handled by the pipeline)."""
    import inspect
    from simulation.models.overseas_transfer import OverseasTransferForecaster
    src = inspect.getsource(OverseasTransferForecaster.predict)
    assert "np.minimum(pred, 2.0 * float(_ymax))" in src, "must keep upper cap"
    assert "np.clip(pred, 0.0, 2.0" not in src, "must drop the transformed-space lower floor"


# ── #5 (G-300): force y_mode=none for models that own their y-transform ──
def test_g300_internal_y_transform_membership():
    from simulation.pipeline.preproc_optuna_hierarchical import model_applies_internal_y_transform
    # 2026-06-21 transform-fix(완료): 내부 y-transform 을 대부분 제거(데이터-주도 preproc 가 y 변환
    #   담당). NB/pf 의 옛 내부 log1p/softplus 는 un-hardcode/transformation=None 으로 제거 → 현재
    #   **hhh4-equivalent 만 내부변환 유지**. NB/SARIMA peak-외삽 폭발은 META_MODELS(G-331)+inverse
    #   cap(G-328) 으로 별도 통제(내부변환 플래그 아님).
    for m in ["hhh4-equivalent"]:
        assert model_applies_internal_y_transform(m), f"{m} must be flagged"
    for m in ["NegBinGLM", "NegBinGLM-V7", "PoissonAutoreg", "NegBinGLM-Glum", "GLARMA",
              "GAM-Spline", "N-BEATS", "N-HiTS", "TiDE"]:   # transform-fix 로 내부변환 제거됨
        assert not model_applies_internal_y_transform(m), f"{m} 내부변환 제거됨(transform-fix)"
    for m in ["XGBoost", "ElasticNet", "TabPFN", "SVR-RBF", "DLinear", None, ""]:
        assert not model_applies_internal_y_transform(m), f"{m} must NOT be flagged"


def test_g303_all_force_y_models_are_active():
    """Every force_y model must actually be in the 53 active lineup (else the fix is moot)."""
    from simulation.models.registry import verify_registry_coverage, CATEGORY_MODELS
    verify_registry_coverage(force_import=True)
    active = set()
    for mm in CATEGORY_MODELS.values():
        active |= set(mm)
    from simulation.pipeline.preproc_optuna_hierarchical import _INTERNAL_Y_TRANSFORM_MODELS
    for m in _INTERNAL_Y_TRANSFORM_MODELS:
        assert m in active, f"force_y model {m} is not in the active lineup"


def test_g303_linear_kernel_models_keep_g275_direct_use_floor():
    """G-303 decision: the in-model ILI≥0 floor is RETAINED for SVR/ElasticNet/KRR/BayesianRidge —
    removing it broke G-275's direct-use contract (linear extrapolation → −47.8). The centered-
    transform trough-bias is a documented should-fix; ≥0 in phase-13/inference is handled by the
    4-site pipeline floor + the artifact floor. This test pins the retention so it isn't re-removed
    without also handling direct callers."""
    import inspect
    from simulation.models.linear_models import (
        SVRLinearForecaster, SVRRBFForecaster, ElasticNetForecaster, KRRForecaster,
    )
    from simulation.models.epi_models import BayesianRidgeForecaster
    for cls in (SVRLinearForecaster, SVRRBFForecaster, ElasticNetForecaster, KRRForecaster):
        src = inspect.getsource(cls.predict)
        assert "np.maximum(" in src and "scaler_y.inverse_transform" in src, (
            f"{cls.__name__} must keep its direct-use ILI≥0 floor (G-275 contract)")
    brsrc = inspect.getsource(BayesianRidgeForecaster.predict)
    assert "return np.maximum(pred, 0)" in brsrc, "BayesianRidge must keep its direct-use ILI≥0 floor"


def test_g303_champion_artifact_floors_inference_in_original_units():
    """ChampionArtifact.predict applies the ILI≥0 floor (closes the inference negative-ILI leak)."""
    import inspect
    from simulation.utils.model_artifact import ChampionArtifact
    src = inspect.getsource(ChampionArtifact.predict)
    assert "np.maximum(" in src and "inverse_transform_target" in src, (
        "artifact predict must floor inverse_transform_target output ≥ 0")


def test_g300_force_y_identity_returns_identity_and_records_none():
    import optuna
    import numpy as np
    from simulation.pipeline.preproc_optuna_hierarchical import suggest_y_preproc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = np.array([1.0, 5, 12, 30, 8, 3, 20, 2, 15, 9], dtype=float)
    cap = {}

    def obj(trial):
        yt, inv, st = suggest_y_preproc(trial, y, force_y_identity=True)
        cap["st"] = st
        cap["params"] = dict(trial.params)
        cap["yt"] = yt
        cap["inv"] = np.asarray(inv(yt), dtype=float)
        return 0.0

    optuna.create_study().optimize(obj, n_trials=1, show_progress_bar=False)
    assert cap["st"]["y_mode"] == "none"
    assert np.allclose(cap["yt"], y), "forced identity forward must be y itself"
    assert np.allclose(cap["inv"], y), "forced identity inverse must be y itself"
    assert cap["params"].get("y_mode") == "none", "y_mode must be recorded for FixedTrial replay"


def test_g300_replay_without_flag_reproduces_identity():
    """Frozen y_mode='none' (from the forced search) replays to identity WITHOUT the flag —
    so the refit/OOF FixedTrial replay reproduces the forced choice (no plumbing to replay)."""
    import optuna
    import numpy as np
    from simulation.pipeline.preproc_optuna_hierarchical import suggest_y_preproc
    y = np.array([1.0, 5, 12, 30, 8, 3, 20, 2, 15, 9], dtype=float)
    t = optuna.trial.FixedTrial({"y_mode": "none"})
    yt, inv, st = suggest_y_preproc(t, y)   # NOTE: force_y_identity NOT passed (replay)
    assert st["y_mode"] == "none"
    assert np.allclose(yt, y) and np.allclose(np.asarray(inv(yt), dtype=float), y)


# ── #10 (G-301): force x_mode=none for USES_FEATURES=False models (skip wasted x-search) ──
def test_g301_force_x_identity_returns_none_and_records():
    import optuna
    import numpy as np
    from simulation.pipeline.preproc_optuna_hierarchical import suggest_x_scaler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xtr = np.random.RandomState(0).normal(size=(20, 6))
    Xte = np.random.RandomState(1).normal(size=(5, 6))
    cap = {}

    def obj(trial):
        Xtr_s, Xte_s, sc, st = suggest_x_scaler(trial, Xtr, Xte, force_x_identity=True)
        cap["st"] = st
        cap["params"] = dict(trial.params)
        cap["passthrough"] = np.allclose(Xtr_s, Xtr) and np.allclose(Xte_s, Xte) and sc is None
        return 0.0

    optuna.create_study().optimize(obj, n_trials=1, show_progress_bar=False)
    assert cap["st"]["x_mode"] == "none"
    assert cap["passthrough"], "force_x_identity must passthrough X (no scaler)"
    assert cap["params"].get("x_mode") == "none", "x_mode must be recorded for replay"


def test_g301_replay_without_flag_reproduces_x_none():
    import optuna
    import numpy as np
    from simulation.pipeline.preproc_optuna_hierarchical import suggest_x_scaler
    Xtr = np.random.RandomState(0).normal(size=(20, 6))
    Xte = np.random.RandomState(1).normal(size=(5, 6))
    t = optuna.trial.FixedTrial({"x_mode": "none"})
    Xtr_s, Xte_s, sc, st = suggest_x_scaler(t, Xtr, Xte)   # force flag NOT passed (replay)
    assert st["x_mode"] == "none"
    assert np.allclose(Xtr_s, Xtr) and sc is None


def test_g301_uses_features_false_only_for_foundation_x_ignorers():
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    for m in ("TimesFM-2.5", "TiRex"):
        c = REGISTRY.get(m)
        if c is None:
            import pytest
            pytest.skip(f"{m} not registered")
        assert getattr(c, "USES_FEATURES", True) is False, f"{m} must be USES_FEATURES=False"
    for m in ("XGBoost", "ElasticNet"):
        c = REGISTRY.get(m)
        assert getattr(c, "USES_FEATURES", True) is True, f"{m} must use features"


# ── G-303 should-fixes: 6-model bias, SVR-Linear C, SARIMA guard ──
def test_g303_restrict_centered_y_membership_and_pool():
    from simulation.pipeline.preproc_optuna_hierarchical import (
        model_floors_at_transformed_zero, _NONCENTERED_STABLE_Y)
    for m in ["SVR-Linear", "SVR-RBF", "ElasticNet", "KRR", "BayesianRidge"]:
        assert model_floors_at_transformed_zero(m), m
    for m in ["XGBoost", "NegBinGLM", "TimesFM-2.5", None, ""]:
        assert not model_floors_at_transformed_zero(m), m
    assert "laplace" not in _NONCENTERED_STABLE_Y and "mcmc_robust" not in _NONCENTERED_STABLE_Y
    # G-335: fourth_root 추가 (flat-grid 7-transform). 비-centered stable Y 집합.
    assert set(_NONCENTERED_STABLE_Y) == {"log1p", "sqrt", "asinh", "fourth_root"}


def test_g303_restrict_centered_y_excludes_centered_transforms():
    import optuna
    import numpy as np
    from simulation.pipeline.preproc_optuna_hierarchical import suggest_y_preproc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = np.abs(np.random.RandomState(0).normal(5, 3, 80)) + 1.0
    seen = set()

    def obj(t):
        _, _, st = suggest_y_preproc(t, y, restrict_centered_y=True)
        if st.get("y_individual"):
            seen.add(st["y_individual"])
        return 0.0

    optuna.create_study().optimize(obj, n_trials=40, show_progress_bar=False)
    assert "laplace" not in seen and "mcmc_robust" not in seen, f"centered leaked: {seen}"


def test_g303_svr_linear_has_c_search():
    import inspect
    from simulation.models.linear_models import SVRLinearForecaster
    src = inspect.getsource(SVRLinearForecaster.fit)
    assert "optuna" in src.lower() and 'suggest_float("C"' in src, "SVR-Linear must search C"


def test_g303_sarima_converged_guard():
    import inspect
    from simulation.models.ts_models import SARIMAForecaster
    src = inspect.getsource(SARIMAForecaster.fit_series)   # .fit delegates to fit_series
    assert "best_conv_fit" in src and "mle_retvals" in src, "SARIMA must prefer converged fit (G-290 parity)"
