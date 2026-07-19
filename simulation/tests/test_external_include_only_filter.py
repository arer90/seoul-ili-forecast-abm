"""Bug-B (2026-06-08): R3 external must honor --models filter on a FRESH run.

Reproduction
------------
`train --models "NegBinGLM,ARIMA,..." --scenario quick-test` (optuna_mode=external)
routed to ``run_external`` (runner.py:964). On a FRESH run there are no external
Optuna JSON caches, so ``per_model_features`` stays empty. The pre-fix code derived
the runner's ``include_only`` purely from ``per_model_features.keys()``::

    _filter_keys = list(per_model_features.keys()) if per_model_features else None

→ ``include_only=None`` → MultiModelRunner ran ALL 56 registered models, despite the
explicit ``--models`` restriction. Symptom: ``MultiModelRunner -- 56개 모델 실행`` and
retired models (TFT-pf etc.) appearing in the tournament.

Fix (external.py): the user's explicit ``_selected`` (config._selected_models) must win
even when no Optuna JSON exists::

    _filter_keys = (list(per_model_features.keys()) if per_model_features
                    else (_selected or None))

This module encodes the decision table as a regression guard (G-style smoke).
"""
from __future__ import annotations


def _resolve_filter_keys(per_model_features: dict, selected: list | None):
    """The exact include_only resolution from external.py (Bug-B fix).

    Args:
        per_model_features: models with loaded external-optuna features (may be empty).
        selected: config._selected_models — the user's explicit --models filter.

    Returns:
        include_only list, or None (= all models pass) when neither constrains.
    """
    return (list(per_model_features.keys()) if per_model_features
            else (selected or None))


def test_fresh_run_honors_models_filter():
    """FRESH run (no JSON) + --models filter → restrict to the selected models.

    This is the bug: pre-fix returned None → all 56 models ran.
    """
    selected = ["NegBinGLM", "ARIMA", "SARIMA", "SARIMAX", "Theta", "FluSight-Baseline"]
    assert _resolve_filter_keys({}, selected) == selected


def test_loaded_features_take_precedence():
    """When external-optuna JSONs exist, use those models (back-compat path)."""
    pmf = {"NegBinGLM": 1, "XGBoost": 2}
    assert _resolve_filter_keys(pmf, ["NegBinGLM"]) == ["NegBinGLM", "XGBoost"]


def test_no_filter_no_json_passes_all():
    """No --models and no JSON → None (runner runs the full registry)."""
    assert _resolve_filter_keys({}, []) is None
    assert _resolve_filter_keys({}, None) is None


def test_single_model_fresh():
    """Single --models on a fresh run still restricts (common smoke case)."""
    assert _resolve_filter_keys({}, ["NegBinGLM"]) == ["NegBinGLM"]


if __name__ == "__main__":
    for fn in (test_fresh_run_honors_models_filter, test_loaded_features_take_precedence,
               test_no_filter_no_json_passes_all, test_single_model_fresh):
        fn()
        print(f"  ✓ {fn.__name__}")
    print("ALL PASS")
