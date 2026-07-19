"""TDD smoke test — Registry v4 integrity (2026-05-24).

Verifies 53-model CATEGORY_MODELS, PAPER_PRIMARY_11, renamed models,
and ALL_MODELS/_scenarios.py sync.  Run from project root with:

    # without pytest (standalone):
    NUMBA_DISABLE_JIT=1 .venv/bin/python simulation/tests/test_registry_v4.py
    # with pytest if available:
    NUMBA_DISABLE_JIT=1 .venv/bin/python -m pytest simulation/tests/test_registry_v4.py -v

All 8 tests must PASS for the v4 sprint to be considered clean.
"""
try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
    # minimal stub so function-body assert statements work standalone
    class _FakePytest:
        @staticmethod
        def raises(*a, **kw):
            class _CM:
                def __enter__(self): return self
                def __exit__(self, *a): return False
            return _CM()
    pytest = _FakePytest()  # type: ignore[assignment]

from simulation.models.registry import (
    CATEGORY_MODELS,
    PAPER_PRIMARY_11,
    EXTRA_MODELS,
)
from simulation.cli._scenarios import ALL_MODELS

# ── helpers ─────────────────────────────────────────────────────────────────

ALL_CATEGORY_MODELS: set[str] = {
    m for models in CATEGORY_MODELS.values() for m in models
}

# PAPER_PRIMARY_11 is tuple[tuple[str,str],...] — extract names only
PAPER_PRIMARY_NAMES: set[str] = {name for name, _ in PAPER_PRIMARY_11}

# ALL_MODELS (from _scenarios.py) is dict[str, list[str]]
ALL_MODELS_FLAT: set[str] = {m for lst in ALL_MODELS.values() for m in lst}

# ── Test 1: CATEGORY_MODELS is internally consistent ────────────────────────

def test_category_models_count():
    """The active lineup must agree with the registry's own coverage check.

    2026-07-19: this asserted a hardcoded 53. The lineup has been deliberately
    reduced since — 53 → 49 (G-319f NegBinGLM-V7, G-323 EARS×3) → 48 (G-347
    NegBinGLM-Glum, GLARMA) and on to the present count — each step recorded in
    ``DEFER_MODELS`` with its reason. A literal count turns every one of those
    documented decisions into a test failure while proving nothing about the
    registry's integrity.

    ``verify_registry_coverage()`` is the live SSOT: it checks that every active
    model is registered and reports what is registered but not active. Assert
    that instead, plus the structural properties a bare count cannot see.
    """
    from simulation.models.registry import verify_registry_coverage

    report = verify_registry_coverage()
    if not report["ok"]:
        # A model whose module cannot import is never registered, so coverage is
        # legitimately incomplete on a machine without the heavy optional stack
        # (torch / torch-geometric gate GCN and OverseasTransfer). That is the
        # environment being thin, not the registry being wrong — distinguish the
        # two instead of failing CI for an install it deliberately does not do.
        try:
            import torch  # noqa: F401
            has_torch = True
        except ImportError:
            has_torch = False
        if not has_torch:
            pytest.skip(f"torch absent — registry cannot be complete: {report['missing']}")
    assert report["ok"], f"registry coverage failed: {report}"

    total = sum(len(v) for v in CATEGORY_MODELS.values())
    assert total == report["total_expected"], (
        f"CATEGORY_MODELS holds {total} but coverage expects "
        f"{report['total_expected']} — the two SSOTs disagree"
    )
    # No model may appear in two categories, and none may be silently empty.
    seen, dupes = set(), []
    for cat, models in CATEGORY_MODELS.items():
        assert models, f"category {cat!r} is empty"
        for m in models:
            if m in seen:
                dupes.append(m)
            seen.add(m)
    assert not dupes, f"model(s) in more than one category: {dupes}"


# ── Test 2: PAPER_PRIMARY_11 models all exist in CATEGORY_MODELS ────────────

def test_paper_primary_all_in_category():
    """Every PAPER_PRIMARY_11 model must appear in CATEGORY_MODELS."""
    missing = PAPER_PRIMARY_NAMES - ALL_CATEGORY_MODELS
    assert not missing, (
        f"PAPER_PRIMARY_11 models not in CATEGORY_MODELS: {sorted(missing)}"
    )


# ── Test 3: stale model names are gone from CATEGORY_MODELS ─────────────────

# Names that were genuinely removed/renamed and must NOT reappear in
# CATEGORY_MODELS. (2026-05-29 정정: TabularDNN / DNN-Conformal / Ensemble-BMA /
# Ensemble-Adaptive 은 현역 active 멤버이므로 stale 목록에서 제외 — 이전 테스트
# 가 활성 모델을 stale 로 잘못 분류했던 것.)
STALE_NAMES = {
    "GE-DNN", "GE-DNN-GAT",          # renamed to GCN / GAT
    "TinyMLP",                        # deregistered (2026-05-26 prune)
    "DNN-Attention-Anchored",         # anchor models removed
    "DNN-Res-Anchored",
    "DNN-Stacked-Anchored",
    "DNN-Conformer-Anchored",
    "DeepAR-pf", "RNN-pf",           # pf suffix removed (DeepAR/RNN → DEFER extras)
    "N-BEATS-pf", "N-HiTS-pf", "TiDE-pf", "TFT-pf",
    "Ensemble-Stacking",             # removed from ensemble
    "Ensemble-Temporal",
    "Ensemble-Blending",
}

def test_no_stale_model_names():
    """None of the removed/renamed models should appear in CATEGORY_MODELS."""
    found = STALE_NAMES & ALL_CATEGORY_MODELS
    assert not found, f"Stale model names still in CATEGORY_MODELS: {sorted(found)}"


# ── Test 4: renamed models present under new names ───────────────────────────

def test_renamed_models_present():
    """GCN, GAT (renamed from GE-DNN/GE-DNN-GAT) must be in CATEGORY_MODELS.
    (2026-05-29 정정: DeepAR / RNN 은 DEFER extras 로 이동했으므로 active
    CATEGORY_MODELS 필수 목록에서 제외.)"""
    required = {"GCN", "GAT"}
    missing = required - ALL_CATEGORY_MODELS
    assert not missing, f"Renamed models missing from CATEGORY_MODELS: {sorted(missing)}"


# ── Test 5: PAPER_PRIMARY_11 has exactly 11 entries ─────────────────────────

def test_paper_primary_structural():
    """PAPER_PRIMARY_11 = tuple of (name, path) pairs, no duplicates.
    INTENTIONALLY EMPTY pending post-training refreeze (registry.py 사용자
    2026-05-12: "학습 후 새 PaperPRIMARY 정의") — assert structure, not count."""
    assert isinstance(PAPER_PRIMARY_11, tuple)
    names = [n for n, _ in PAPER_PRIMARY_11]
    assert len(names) == len(set(names)), f"duplicate names: {names}"


# ── Test 6: ARIMA is an active ts model (not a naive baseline) ───────────────

def test_arima_is_active_ts_model():
    """ARIMA 은 CATEGORY_MODELS['ts'] 의 active 모델. (2026-05-29 정정: 이전
    테스트는 ARIMA 를 EXTRA_MODELS['baselines'] 에서 찾았으나 baselines 는
    naive 기준선 ar1/persistence/climatology 만 — ARIMA 는 정식 ts 모델.)"""
    assert "ARIMA" in CATEGORY_MODELS.get("ts", []), (
        f"ARIMA not in CATEGORY_MODELS['ts']. Got: {CATEGORY_MODELS.get('ts')}"
    )
    baselines = EXTRA_MODELS.get("baselines", [])
    assert set(baselines) == {"ar1", "persistence", "climatology"}, (
        f"baselines drifted: {baselines}"
    )


# ── Test 7: ALL_MODELS (_scenarios.py) matches CATEGORY_MODELS ──────────────

def test_all_models_sync():
    """ALL_MODELS in _scenarios.py must match CATEGORY_MODELS exactly."""
    only_in_all = ALL_MODELS_FLAT - ALL_CATEGORY_MODELS
    only_in_cat = ALL_CATEGORY_MODELS - ALL_MODELS_FLAT
    assert not only_in_all and not only_in_cat, (
        f"Mismatch between ALL_MODELS and CATEGORY_MODELS.\n"
        f"  Only in ALL_MODELS:      {sorted(only_in_all)}\n"
        f"  Only in CATEGORY_MODELS: {sorted(only_in_cat)}"
    )


# ── Test 8: no 'anchor' category in CATEGORY_MODELS ─────────────────────────

def test_no_anchor_category():
    """The 'anchor' model category must not exist in CATEGORY_MODELS."""
    assert "anchor" not in CATEGORY_MODELS, (
        "'anchor' category still present in CATEGORY_MODELS"
    )


# ── Standalone runner (no pytest) ────────────────────────────────────────────

if __name__ == "__main__":
    _tests = [
        test_category_models_count,
        test_paper_primary_all_in_category,
        test_no_stale_model_names,
        test_renamed_models_present,
        test_paper_primary_structural,
        test_arima_is_active_ts_model,
        test_all_models_sync,
        test_no_anchor_category,
    ]
    print("=== Registry v4 TDD (8 tests) ===")
    _all_pass = True
    for _fn in _tests:
        try:
            _fn()
            print(f"  PASS  {_fn.__name__}")
        except AssertionError as _e:
            _all_pass = False
            print(f"  FAIL  {_fn.__name__}")
            print(f"        → {_e}")
    print()
    print("=== ALL PASS ===" if _all_pass else "=== FAILURES FOUND ===")
    raise SystemExit(0 if _all_pass else 1)
