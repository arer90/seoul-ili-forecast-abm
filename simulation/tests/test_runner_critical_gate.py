"""Guard for the critical-phase fail-loud gate (G-237, 2026-05-30).

After a 1-token NameError silently voided P1 (real_forecaster / champion gate) for a 10h run,
the orchestrator now records critical-phase errors (P1/R9/R10) and surfaces them
loudly instead of unconditionally printing "Pipeline Complete!". This unit-tests
the pure detection helper that drives the banner + report["CRITICAL_FAILURES"].

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
from simulation.pipeline.runner import _collect_critical_failures


def test_clean_run_has_no_critical_failures():
    all_results = {
        "real_eval": {"best_model": "Ensemble-BMA", "best_r2": 0.89},
        "per_model_optimize": {"n_models": 12},
        "per_model_eval": {"metrics_csv": "x.csv"},
    }
    assert _collect_critical_failures(all_results) == []


def test_detects_champion_gate_failure():
    fails = _collect_critical_failures(
        {"real_eval": {"error": "NameError: name 'metrics' is not defined", "critical": True}}
    )
    assert len(fails) == 1
    assert "P1" in fails[0][0]
    assert "NameError" in fails[0][1]


def test_detects_multiple_and_skips_ok_and_noncritical():
    fails = _collect_critical_failures({
        "real_eval": {"error": "e1"},
        "per_model_optimize": {"error": "e2"},
        "per_model_eval": {"metrics_csv": "ok"},            # no error → not flagged
        "feature_importance": {"error": "shap failed"},     # non-critical → ignored
    })
    labels = [f[0] for f in fails]
    assert len(fails) == 2
    assert "P1 real_forecaster operational forecast (real)" in labels
    assert "R9 per_model_optimize HP-optimize" in labels
    assert "R10 per_model_eval SSOT eval" not in labels


def test_missing_phases_are_safe():
    assert _collect_critical_failures({}) == []
