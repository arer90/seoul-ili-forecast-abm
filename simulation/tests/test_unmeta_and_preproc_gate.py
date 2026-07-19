"""PART B + PART C (transform-fix reconciliation, 2026-06-21).

PART B — un-META: with each model's internal y-transform removed (PART A), NegBinGLM / SARIMA /
  PoissonAutoreg no longer carry a hardcoded transform, so they can (and should) run the data-driven
  preproc transform search instead of being pinned to identity×none via META_MODELS. They are
  therefore removed from META_MODELS. Intrinsic-link / persistence / ensemble / EARS models stay.

PART C ★ no-op blocker — preproc gate: _INTERNAL_Y_TRANSFORM_MODELS / model_applies_internal_y_transform
  force y_mode="none" (identity) for their members, which would make PART A a NO-OP (preproc could
  never search a transform for them). PoissonAutoreg / N-BEATS / N-HiTS / TiDE (and NegBinGLM / SARIMA
  / GAM-Spline, already absent on this base) must be ABSENT so the transform search actually runs.
  hhh4-equivalent stays (its integer-rounded NB log-link genuinely breaks under an external transform).
"""
from __future__ import annotations

import pytest


# ── PART B: META_MODELS membership ──────────────────────────────────────────────

def _get_meta_models() -> set:
    """Extract the literal META_MODELS set from per_model_optimize source (no full pipeline run)."""
    import ast
    import inspect
    from simulation.pipeline import per_model_optimize as pmo

    src = inspect.getsource(pmo)
    tree = ast.parse(src)
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "META_MODELS":
                    if isinstance(node.value, ast.Set):
                        found = {
                            el.value for el in node.value.elts
                            if isinstance(el, ast.Constant) and isinstance(el.value, str)
                        }
    assert found is not None, "META_MODELS literal set not found in per_model_optimize"
    return found


@pytest.mark.xfail(reason="G-331(2026-06-21): SARIMA·NegBinGLM 을 peak-외삽 폭발 안전 위해 META_MODELS "
                          "로 RE-ADD — PART B un-META 목표가 안전을 위해 부분 반전됨. 완전 un-META "
                          "(STABLE_Y transform+G-328 cap 으로 폭발 통제 가능한지)는 post-run A/B 대기. "
                          "이 테스트는 그 목표를 추적하는 것이므로 현재는 expected-fail.",
                   strict=False)
def test_unmeta_removed_models_absent():
    meta = _get_meta_models()
    for m in ("NegBinGLM", "SARIMA", "PoissonAutoreg"):
        assert m not in meta, (
            f"{m} must be REMOVED from META_MODELS — its internal transform is gone (PART A), so "
            f"the data-driven preproc transform search should run for it")


def test_unmeta_intrinsic_and_ensembles_retained():
    meta = _get_meta_models()
    # intrinsic-link / count / renewal models keep identity (true NB / count / renewal structure)
    for m in ("NegBinGLM-Glum", "GLARMA", "hhh4-equivalent", "EpiEstim", "Wallinga-Teunis"):
        assert m in meta, f"{m} must stay META (intrinsic count/renewal link)"
    # statistical-ts persistence / univariate baselines stay
    for m in ("ARIMA", "SARIMAX", "Theta", "FluSight-Baseline"):
        assert m in meta, f"{m} must stay META"
    # all ensembles + EARS stay
    for m in ("Ensemble-NNLS", "Ensemble-BMA", "Ensemble-Adaptive",
              "EARS-C1", "EARS-C2", "EARS-C3"):
        assert m in meta, f"{m} must stay META"


# ── PART C: preproc internal-y-transform gate ───────────────────────────────────

def test_gate_removed_models_search_transforms():
    from simulation.pipeline.preproc_optuna_hierarchical import (
        _INTERNAL_Y_TRANSFORM_MODELS, model_applies_internal_y_transform,
    )
    # these must NOT force identity → preproc transform search runs (else PART A is a no-op)
    for m in ("PoissonAutoreg", "N-BEATS", "N-HiTS", "TiDE",
              "NegBinGLM", "SARIMA", "GAM-Spline"):
        assert m not in _INTERNAL_Y_TRANSFORM_MODELS, (
            f"{m} must be ABSENT from _INTERNAL_Y_TRANSFORM_MODELS (transform now data-driven)")
        assert not model_applies_internal_y_transform(m), (
            f"model_applies_internal_y_transform({m}) must be False — else preproc forces identity "
            f"and PART A's un-hardcode is a no-op")


def test_gate_keeps_hhh4():
    from simulation.pipeline.preproc_optuna_hierarchical import (
        _INTERNAL_Y_TRANSFORM_MODELS, model_applies_internal_y_transform,
    )
    assert "hhh4-equivalent" in _INTERNAL_Y_TRANSFORM_MODELS, (
        "hhh4-equivalent must STAY in the gate (integer-rounded NB log-link breaks under an "
        "external transform)")
    assert model_applies_internal_y_transform("hhh4-equivalent")
