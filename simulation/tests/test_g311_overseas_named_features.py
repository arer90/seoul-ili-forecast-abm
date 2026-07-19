"""G-311 (2026-06-18): OverseasTransfer must run with name-preserving mc ("none").

Root cause: OverseasTransfer's transfer encoder resolves ili_rate_lag1-4 BY NAME
(overseas_transfer.py:656-668). When per-model mc picks "pca", the 398 features are
renamed to anonymous PCs → the name lookup fails → transfer silently degrades to
feature-only (the "slow phantom": full neural cost, zero transfer effect). The dispatch
loop now forces mc="none" for models flagged by model_requires_named_features.

macOS: run PER-FILE.
"""
from simulation.pipeline.preproc_optuna_hierarchical import model_requires_named_features


def test_g311_overseas_requires_named_features():
    assert model_requires_named_features("OverseasTransfer") is True


def test_g311_normal_models_do_not():
    for m in ("SVR-RBF", "ElasticNet", "XGBoost", "TimesNet", "NegBinGLM"):
        assert model_requires_named_features(m) is False, f"{m} should not force named-features"


def test_g311_none_and_unknown_safe():
    assert model_requires_named_features(None) is False
    assert model_requires_named_features("") is False
    assert model_requires_named_features("NotARealModel") is False


def test_g311_dispatch_guard_wired():
    """The dispatch loop imports + applies the guard (source-level wiring check)."""
    import inspect
    from simulation.pipeline import per_model_optimize as pmo
    src = inspect.getsource(pmo)
    assert "model_requires_named_features(mname)" in src, "guard must be applied in dispatch loop"
    assert "mc forced 'none'" in src, "guard must force mc=none for named-feature models"
