"""G-236 regression — per-model subprocess isolation covers ALL individual
model categories (not just DL), so in-process OMP/BLAS thread accumulation
can't crash a full run.

Root cause (2026-05-29): `_SUBPROCESS_CATEGORIES = {"dl", "modern_ts"}` only
isolated DL; tree/linear/ts/epi/physics ran in-process → after ~30 models the
accumulated OMP threads hit `OMP: System error #22 (pthread_create EINVAL)` →
SIGSEGV (observed: full run 73min, CQR-LightGBM 30/56). ("modern_ts" was dead —
no model.meta.category equals it.)

Fix: isolate every *individual* model category; only "meta" (ensembles, which
combine OOF predictions in-process) stays in-process.
"""
from __future__ import annotations


# ── fast decision-logic guard (no subprocess) ────────────────────────────────
def test_subprocess_categories_cover_all_individual_categories():
    """The isolation set must contain every category an INDIVIDUAL model can
    carry (dl/tree/linear/ts/epi/physics). 'meta' (ensemble) is excluded."""
    from simulation.models.runner import _SUBPROCESS_CATEGORIES
    required = {"dl", "tree", "linear", "ts", "epi", "physics"}
    missing = required - set(_SUBPROCESS_CATEGORIES)
    assert not missing, (
        f"categories not isolated → in-process OMP accumulation risk: {sorted(missing)}"
    )
    assert "meta" not in _SUBPROCESS_CATEGORIES, (
        "ensemble ('meta') must stay in-process (combines OOF in-memory)."
    )
    # 'modern_ts' (dead, hyphen/underscore mismatch) should not linger.
    assert "modern_ts" not in _SUBPROCESS_CATEGORIES, "drop dead 'modern_ts'."


def test_should_use_subprocess_decision_per_category():
    """Live registry: tree/linear/ts/epi models isolate; ensembles don't;
    macOS PyG/MPS override (if present) forced in-process."""
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    from simulation.models.runner import _should_use_subprocess, _get_inprocess_override

    override = _get_inprocess_override()
    isolate_expected = ["XGBoost", "LightGBM", "CQR-LightGBM", "KRR",
                        "ElasticNet", "ARIMA", "EpiEstim", "DNN"]
    for name in isolate_expected:
        cls = REGISTRY.get(name)
        if cls is None:
            continue
        want = name not in override  # override whitelist forced in-process
        got = _should_use_subprocess(cls.meta.category, name)
        assert got == want, (
            f"{name} (cat={cls.meta.category}) isolate={got}, want {want}"
        )

    for name in ("Ensemble-NNLS", "Ensemble-BMA"):
        cls = REGISTRY.get(name)
        if cls is None:
            continue
        assert _should_use_subprocess(cls.meta.category, name) is False, (
            f"{name} (ensemble) must stay in-process."
        )


# ── subprocess-worker compatibility (spawns a real subprocess; slower) ────────
def test_worker_runs_non_dl_tree_model():
    """The generic worker must run a non-DL (tree) model end-to-end — proves
    expanding isolation to tree/linear/ts/epi doesn't trade the OMP crash for a
    subprocess-incompatibility crash."""
    import numpy as np
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    from simulation.models.runner import _run_model_in_subprocess

    cls = REGISTRY.get("XGBoost")
    assert cls is not None, "XGBoost not registered"
    model = cls()
    rng = np.random.default_rng(0)
    n, p = 120, 8
    X = rng.normal(size=(n, p)); y = rng.normal(size=n)
    res = _run_model_in_subprocess(
        model, X[:80], y[:80], X[80:100], X[100:],
        y_val_len=20, y_test_len=20, is_ts=False, name="XGBoost",
        timeout=120, save_dir="", stall_timeout=90, poll_interval=5,
        feature_names=[f"f{i}" for i in range(p)],
    )
    assert isinstance(res, dict) and "val_pred" in res and "test_pred" in res, (
        f"subprocess worker did not return predictions for tree model: {res!r}"
    )
    assert len(res["val_pred"]) == 20 and len(res["test_pred"]) == 20


# ── every registered model's meta.category must be isolated (or meta) ─────────
def test_every_registered_model_category_is_isolated():
    """No registered model may carry a meta.category outside
    _SUBPROCESS_CATEGORIES ∪ {"meta"} — else it silently runs in-process and
    can re-trigger the G-236 OMP accumulation crash. Catches a future model
    introducing a new coarse category without updating the isolation set."""
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    from simulation.models.runner import _SUBPROCESS_CATEGORIES
    allowed = set(_SUBPROCESS_CATEGORIES) | {"meta"}
    offenders = {n: c.meta.category for n, c in REGISTRY.get_all().items()
                 if c.meta.category not in allowed}
    assert not offenders, (
        f"models with meta.category outside {sorted(allowed)} → would run "
        f"in-process (OMP accumulation risk): {offenders}"
    )


# ── macOS in-process override must reference REGISTERED names (G-236 후속) ─────
def test_darwin_override_names_are_registered():
    """_INPROCESS_OVERRIDE_DARWIN must list CURRENT registered model names.
    Regression guard for the stale GE-DNN/GE-GAT → GCN/GAT rename that left the
    PyG/MPS-unsafe models unprotected (Codex 교차검증 2026-05-29)."""
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    from simulation.models.runner import _INPROCESS_OVERRIDE_DARWIN
    stale = [n for n in _INPROCESS_OVERRIDE_DARWIN if REGISTRY.get(n) is None]
    assert not stale, (
        f"override lists unregistered (stale) names → match nothing: {stale}"
    )
    # PyG/MPS-unsafe models, if registered, MUST be in the override.
    for pyg in ("GAT", "GCN"):
        if REGISTRY.get(pyg) is not None:
            assert pyg in _INPROCESS_OVERRIDE_DARWIN, (
                f"{pyg} (PyG/MPS-unsafe) missing from darwin override → would "
                f"run in subprocess on macOS (fork/MPS SIGSEGV risk)."
            )


if __name__ == "__main__":
    test_subprocess_categories_cover_all_individual_categories()
    print("PASS  categories")
    test_should_use_subprocess_decision_per_category()
    print("PASS  decision")
    test_worker_runs_non_dl_tree_model()
    print("PASS  worker non-DL")
    test_every_registered_model_category_is_isolated()
    print("PASS  every-model isolated")
    test_darwin_override_names_are_registered()
    print("PASS  darwin override registered")
    print("=== ALL PASS ===")
