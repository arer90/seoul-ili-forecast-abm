"""Smoke + regression guard for the P1 (real_forecaster) champion gate (real-slab eval).

2026-05-30 incident (G-237): ``run_real_eval`` crashed at the summary-assembly step
with ``NameError: name 'metrics' is not defined`` — a stale bare ``metrics`` where
the per-model accumulator is ``results`` (the metrics module is imported as ``M``).
A broad ``except`` in the caller swallowed it and the 10h run silently produced NO
champion gate and NO ``summary.json``. These tests would have caught it in ms.

macOS: run PER-FILE (single-process pytest segfaults at LightGBM; see memory
``test-suite-execution``)::

    .venv/bin/python -m pytest simulation/tests/test_phase12_smoke.py -q
"""
import ast
import inspect

import numpy as np
import pytest


def _synthetic_inputs(n_in: int = 60, n_real: int = 8, n_feat: int = 5):
    """Minimal phase1/all_results that drive run_real_eval to the summary step.

    ``best_name`` is set (via wf_results) so the 1072 path is exercised; the model
    name is unregistered so the rolling-origin refit fails gracefully (guarded) —
    a truthy ``best_name`` alone is enough to trip the NameError.
    """
    rng = np.random.default_rng(42)
    X_all = rng.normal(size=(n_in, n_feat))
    y_all = rng.uniform(5.0, 25.0, size=n_in)
    real_X = rng.normal(size=(n_real, n_feat))
    real_y = rng.uniform(20.0, 40.0, size=n_real)  # distribution-shifted, like the slab
    real_dates = np.array(
        [np.datetime64("2025-01-06") + np.timedelta64(7 * i, "D") for i in range(n_real)]
    )
    dates_in = np.array(
        [np.datetime64("2023-11-06") + np.timedelta64(7 * i, "D") for i in range(n_in)]
    )
    phase1 = {
        "real_X": real_X, "real_y": real_y, "real_dates": real_dates,
        "X_all": X_all, "y_all": y_all,
        "feature_cols": [f"f{i}" for i in range(n_feat)],
        "dates": dates_in,
    }
    all_results = {
        "wfcv": {
            "wf_results": {"MockModel": {"overall_metrics": {"r2": 0.5}}},
            "oof_predictions": {"MockModel": (y_all + rng.normal(0, 1.0, n_in)).tolist()},
        }
    }
    return phase1, all_results


def test_phase12_champion_gate_writes_summary(tmp_path):
    """run_real_eval must NOT raise NameError and MUST write summary.json."""
    from types import SimpleNamespace
    import simulation.pipeline.real_eval as p12

    phase1, all_results = _synthetic_inputs()
    config = SimpleNamespace(
        save_dir=str(tmp_path),
        split=SimpleNamespace(real_eval_enabled=True),
    )
    try:
        p12.run_real_eval(phase1, all_results, config)
    except NameError as e:  # the 2026-05-30 bug
        pytest.fail(f"champion gate NameError (regression): {e}")
    except Exception:
        # Later-stage artifacts on synthetic data may legitimately fail; we only
        # guard the NameError + the gate's primary output (summary.json), which is
        # written immediately after the previously-broken line.
        pass

    summary = tmp_path / "real_eval" / "summary.json"
    assert summary.exists(), "champion gate did not write summary.json"


def test_phase12_no_bare_metrics_name():
    """Static guard: ``metrics`` must not be LOADed as a bare name in run_real_eval
    (it is imported as ``M``; the accumulator is ``results``)."""
    import simulation.pipeline.real_eval as p12

    src = inspect.getsource(p12.run_real_eval)
    tree = ast.parse(src)
    loaded = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    assert "metrics" not in loaded, (
        "bare 'metrics' load in run_real_eval — use 'results' "
        "(G-237 champion-gate NameError regression)"
    )
