"""Per-model audit fixes — Batch 1 (structural, low-risk).

G-295: kernel feature-floor gate was DEAD — it gated on REGISTRY meta.category=='kernel',
       but KRR/SVR-Linear/SVR-RBF carry meta.category=='linear' (only the CATEGORY_MODELS
       family is 'kernel'). The fix resolves the floor against CATEGORY_MODELS family.
G-296: N-BEATS/N-HiTS/TiDE registry classes are pf_models.Pf* wrappers with NO Optuna —
       their per_model_trials entries are inert (config-honesty annotation; no behavior change).
G-289 parity (GAT): GAT.predict now applies apply_extrapolation_cap like GCN, and GAT.fit
       sets _y_train_max.

macOS: run PER-FILE.
"""
import numpy as np
import pytest


def test_g295_kernel_floor_gate_uses_category_family_not_meta_category():
    """Documents the bug + fix: kernel models' meta.category is 'linear' (old gate dead),
    but the CATEGORY_MODELS family is 'kernel' (new gate fires)."""
    from simulation.models.registry import verify_registry_coverage, CATEGORY_MODELS
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY

    for m in ("KRR", "SVR-Linear", "SVR-RBF"):
        cls = REGISTRY.get(m)
        if cls is None:
            pytest.skip(f"{m} not registered")
        meta_cat = getattr(getattr(cls, "meta", None), "category", "")
        fam = next((f for f, mm in CATEGORY_MODELS.items() if m in mm), "")
        # the OLD gate (meta.category=='kernel') would have been DEAD:
        assert meta_cat != "kernel", f"{m} meta.category unexpectedly 'kernel' ({meta_cat!r})"
        # the NEW gate (CATEGORY family=='kernel') FIRES:
        assert fam == "kernel", f"{m} CATEGORY family must be 'kernel', got {fam!r}"


def test_g295_floor_source_gates_on_family():
    """The live source now gates the kernel floor on _fam_fp (family), not _cat_fp (meta.category)."""
    import inspect
    from simulation.pipeline import per_model_optimize as P
    src = inspect.getsource(P.optimize_one_model)
    assert '_fam_fp == "kernel"' in src, "kernel floor must gate on CATEGORY family (_fam_fp)"
    assert 'CATEGORY_MODELS as _CATM_FP' in src, "family lookup must be wired"


def test_g296_pf_models_have_no_optuna_and_dead_entries_removed():
    """N-BEATS/N-HiTS/TiDE registry classes are Pf* wrappers (no Optuna) → their misleading
    per_model_trials entries were REMOVED (the dict no longer implies HP tuning for them)."""
    import inspect
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY

    for m in ("N-BEATS", "N-HiTS", "TiDE"):
        cls = REGISTRY.get(m)
        if cls is None:
            pytest.skip(f"{m} not registered")
        assert cls.__module__.endswith("pf_models"), (
            f"{m} expected pf_models wrapper, got {cls.__module__}")
        try:
            src = inspect.getsource(cls)
        except Exception:
            src = ""
        assert "optuna" not in src.lower(), f"{m} Pf wrapper unexpectedly references optuna"

    # G-296: the dead/misleading entries are gone from the live config dict
    from simulation.pipeline.config import OptunaConfig
    trials = OptunaConfig().per_model_trials
    for m in ("N-BEATS", "N-HiTS", "TiDE"):
        assert m not in trials, f"{m} dead per_model_trials entry must be removed (Pf=no Optuna)"
    # sanity: models that DO have Optuna keep their entries
    assert trials.get("XGBoost") == 50 and trials.get("Mamba") == 40


def test_g289_gat_predict_applies_extrapolation_cap():
    """GAT.predict now caps to y_train_max×1.5 (GCN parity) instead of only flooring at 0."""
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    cls = REGISTRY.get("GAT")
    if cls is None:
        pytest.skip("GAT not registered")
    import inspect
    fit_src = inspect.getsource(cls.fit)
    pred_src = inspect.getsource(cls.predict)
    assert "_y_train_max" in fit_src, "GAT.fit must record _y_train_max (G-289 parity)"
    assert "apply_extrapolation_cap" in pred_src, "GAT.predict must apply the extrapolation cap"

    # functional: a fitted-stub GAT caps a runaway prediction to y_train_max*1.5
    from sklearn.preprocessing import StandardScaler

    class _StubModel:
        def parameters(self):
            import torch
            yield torch.zeros(1)
        def eval(self):
            return self
        def __call__(self, x):
            import torch
            # absurd over-prediction in scaled space
            return torch.full((x.shape[0],), 50.0)

    torch = pytest.importorskip("torch")
    g = cls.__new__(cls)
    g._fitted = True
    g._model = _StubModel()
    g._scaler_X = StandardScaler().fit(np.random.RandomState(0).normal(size=(40, 5)))
    ytr = np.linspace(1.0, 30.0, 40)        # train max = 30
    g._scaler_y = StandardScaler().fit(ytr.reshape(-1, 1))
    g._y_train_max = float(ytr.max())
    out = g.predict(np.random.RandomState(1).normal(size=(8, 5)))
    cap = max(g._y_train_max * 1.5, 100.0)
    assert np.all(out <= cap + 1e-6), f"GAT prediction not capped: max={out.max()} cap={cap}"
    assert np.all(out >= 0.0), "GAT prediction must stay non-negative"
