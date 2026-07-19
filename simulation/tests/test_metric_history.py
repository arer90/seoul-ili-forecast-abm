"""Guard for the unified model×134×phase metric-history writer (G-238, 2026-05-30).

Audit (G-237 follow-up) found: the 134-key SSOT (phase_eval_r8) is computed at
phases 4-18 but was only persisted nested in-memory — no queryable cross-phase
"model × 134-metric × phase" comparison artifact existed. `_write_metric_history`
walks all_results, finds every nested phase_eval_r8, and writes a long-format CSV.

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
import csv

from simulation.pipeline.runner import _write_metric_history, repair_metric_history


def test_metric_history_walk_heterogeneous_nesting(tmp_path):
    """Finds phase_eval_r8 under different per-phase shapes; attributes phase/model/slab."""
    all_results = {
        "baseline": {"model_results": {
            "lightgbm": {"r2": 0.5,
                         "phase_eval_r8": {"r2": 0.85, "wis": 3.4, "mae": 2.1, "_skipped": False}},
            "xgboost": {"phase_eval_r8": {"r2": 0.80, "wis": 3.6, "mae": 2.3}},
        }},
        "real_eval": {"metrics": {  # different container key
            "Ensemble-BMA": {"phase_eval_r8": {"r2": 0.89, "wis": 8.4}},
        }},
        "wfcv": {"wf_results": {  # yet another container
            "lightgbm": {"phase_eval_r8": {"r2": 0.78, "wis": 3.5}},
        }},
        "feature_importance": {"shap": [1, 2, 3]},  # no phase_eval_r8 → ignored
    }
    out = _write_metric_history(all_results, str(tmp_path))
    assert out is not None and out.exists()

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    phases = {r["phase"] for r in rows}
    models = {r["model"] for r in rows}
    metrics = {r["metric"] for r in rows}

    assert phases == {"baseline", "real_eval", "wfcv"}          # only phases with r8
    assert {"lightgbm", "xgboost", "Ensemble-BMA"} <= models    # model = key above r8
    assert "_skipped" not in metrics                            # underscore keys excluded

    # slab mapped per phase (metric_rubric)
    bma = next(r for r in rows if r["model"] == "Ensemble-BMA")
    assert bma["slab"] == "real"
    lgb_base = [r for r in rows if r["model"] == "lightgbm" and r["phase"] == "baseline"]
    assert {r["metric"] for r in lgb_base} == {"r2", "wis", "mae"}
    assert all(r["slab"] == "test" for r in lgb_base)


def test_metric_history_empty_returns_none(tmp_path):
    """No phase_eval_r8 anywhere → None (no file)."""
    assert _write_metric_history({"feature_importance": {"x": 1}}, str(tmp_path)) is None


def test_container_key_does_not_become_model(tmp_path):
    """Phase-13 nesting {model: {test_metrics: {phase_eval_r8}}} must NOT label
    every model 'test_metrics' (the bug that hid 13 models' optimized metrics)."""
    all_results = {
        "per_model_optimize": {
            "ARIMA": {"test_metrics": {"phase_eval_r8": {"phase_id": "phase13_refit_ARIMA", "r2": -0.4, "wis": 15.0}}},
            "NegBinGLM-V7": {"test_metrics": {"phase_eval_r8": {"phase_id": "phase13_refit_NegBinGLM-V7", "r2": 0.93, "wis": 3.2}}},
        }
    }
    out = _write_metric_history(all_results, str(tmp_path))
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    models = {r["model"] for r in rows}
    assert "test_metrics" not in models
    assert {"ARIMA", "NegBinGLM-V7"} <= models
    wis = {r["model"]: r["value"] for r in rows if r["metric"] == "wis"}
    assert wis["NegBinGLM-V7"] == "3.2"


def test_repair_rekeys_existing_csv(tmp_path):
    """repair_metric_history fixes pre-guard files in place, idempotently."""
    p = tmp_path / "mh.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "model", "slab", "metric", "value"])
        w.writeheader()
        w.writerows([
            {"phase": "per_model_optimize", "model": "test_metrics", "slab": "oof_cv", "metric": "phase_id", "value": "phase13_refit_ARIMA"},
            {"phase": "per_model_optimize", "model": "test_metrics", "slab": "oof_cv", "metric": "wis", "value": "15.0"},
            {"phase": "per_model_optimize", "model": "test_metrics", "slab": "oof_cv", "metric": "phase_id", "value": "phase13_refit_KRR"},
            {"phase": "per_model_optimize", "model": "test_metrics", "slab": "oof_cv", "metric": "wis", "value": "4.0"},
        ])
    assert repair_metric_history(p) == 4
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert {r["model"] for r in rows} == {"ARIMA", "KRR"}
    assert repair_metric_history(p) == 0  # idempotent
