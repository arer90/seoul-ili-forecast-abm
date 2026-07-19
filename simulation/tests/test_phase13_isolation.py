"""R9 (per_model_optimize) per-model subprocess isolation (G-236/G-249 gap fix, 2026-06-10).

The WF-CV (R4) path isolates every model category in a subprocess; R9 (per_model_optimize) did not,
so an IRLS-heavy ``epi`` model (GLARMA) fitting in a process already polluted by
torch+lightgbm libomp aborted the whole run with ``OMP: Error #179
pthread_mutex_init``. That abort is a PROCESS abort — ``try/except`` can't contain
it. These tests prove ``run_isolated`` contains process aborts / timeouts /
stalls at the process boundary (parent survives) and that an isolated model probe
returns the SAME result as the in-process probe (no behaviour change when on).

Run (macOS, per-file to avoid the OpenMP/LightGBM single-process segfault):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      .venv/bin/python -m pytest simulation/tests/test_phase13_isolation.py -x -q
"""
from __future__ import annotations

import numpy as np


# ── module-level worker targets (imported by the child via "module:func") ─────
def _worker_echo(payload: dict) -> dict:
    x = payload.get("x", 0)
    return {"got": x, "doubled": x * 2, "msg": payload.get("msg", "")}


def _worker_hard_exit(payload: dict) -> dict:
    """Simulate an OMP #179 / SIGSEGV abort: die immediately, emit NO result."""
    import os
    os._exit(139)            # 139 == 128 + SIGSEGV; no result file written


def _worker_sleep(payload: dict) -> dict:
    import time
    time.sleep(payload.get("secs", 30))
    return {"slept": True}


def _worker_raises(payload: dict) -> dict:
    raise ValueError("boom-from-worker")


def _worker_progress(payload: dict) -> dict:
    """Actively logs (grows the captured log) every 0.2s for `duration` s, then returns —
    simulates a slow-but-progressing mc-probe / Optuna stage (G-260)."""
    import time as _t, sys as _s
    dur = float(payload.get("duration", 3.0))
    t0 = _t.time(); i = 0
    while _t.time() - t0 < dur:
        print(f"[progress] step {i}", flush=True); _s.stdout.flush()
        i += 1
        _t.sleep(0.2)
    return {"steps": i, "ok": True}


# ── G-260: progress-aware hard cap ─────────────────────────────────────────────
def test_run_isolated_extends_while_progressing(monkeypatch):
    """A child still actively logging past the soft `timeout` is NOT killed — it is extended
    up to timeout×MAX_EXTEND and completes. (Old hard cap would kill it at `timeout`.)"""
    monkeypatch.setenv("MPH_ISOLATE_MAX_EXTEND", "30")
    from simulation.pipeline._phase13_isolation import run_isolated
    res = run_isolated(
        "simulation.tests.test_phase13_isolation:_worker_progress",
        {"duration": 2.5}, timeout=0.6, stall_timeout=1.5, poll_interval=0.2, label="prog")
    assert res.get("ok") is True and res.get("steps", 0) >= 5, res  # finished, not timeout-killed


def test_run_isolated_absolute_ceiling_still_kills(monkeypatch):
    """Even while progressing, timeout×MAX_EXTEND is an absolute runaway ceiling → killed."""
    monkeypatch.setenv("MPH_ISOLATE_MAX_EXTEND", "2")
    from simulation.pipeline._phase13_isolation import run_isolated
    res = run_isolated(
        "simulation.tests.test_phase13_isolation:_worker_progress",
        {"duration": 20.0}, timeout=0.6, stall_timeout=5.0, poll_interval=0.2, label="ceil")
    assert res.get("__crashed__") is True, res  # killed at ~1.2s (0.6×2), no result


# ── generic runner: green path ────────────────────────────────────────────────
def test_run_isolated_returns_dict():
    from simulation.pipeline._phase13_isolation import run_isolated
    res = run_isolated(
        "simulation.tests.test_phase13_isolation:_worker_echo",
        {"x": 21, "msg": "hi"}, timeout=60, stall_timeout=60, label="echo",
    )
    assert res.get("doubled") == 42 and res.get("got") == 21 and res.get("msg") == "hi"
    assert "__crashed__" not in res and "__worker_error__" not in res


# ── generic runner: contains a PROCESS abort (the actual OMP #179 class) ───────
def test_run_isolated_contains_process_abort():
    from simulation.pipeline._phase13_isolation import run_isolated
    res = run_isolated(
        "simulation.tests.test_phase13_isolation:_worker_hard_exit",
        {}, timeout=60, stall_timeout=60, label="abort",
    )
    # Parent MUST survive and report containment, not raise.
    assert res.get("__crashed__") is True
    assert res.get("reason") in ("no_result", "exit")


# ── generic runner: contains timeout / stall ──────────────────────────────────
def test_run_isolated_contains_timeout():
    from simulation.pipeline._phase13_isolation import run_isolated
    res = run_isolated(
        "simulation.tests.test_phase13_isolation:_worker_sleep",
        {"secs": 30}, timeout=2, stall_timeout=2, poll_interval=0.5, label="sleep",
    )
    assert res.get("__crashed__") is True
    assert res.get("reason") in ("timeout", "stall")


# ── generic runner: a Python error comes back as DATA (not a parent raise) ─────
def test_run_isolated_returns_worker_error_as_data():
    from simulation.pipeline._phase13_isolation import run_isolated
    res = run_isolated(
        "simulation.tests.test_phase13_isolation:_worker_raises",
        {}, timeout=60, stall_timeout=60, label="raise",
    )
    assert "__worker_error__" in res and "boom-from-worker" in res["__worker_error__"]
    assert res.get("__crashed__") is not True


# ── gate ──────────────────────────────────────────────────────────────────────
def test_isolation_gate_default_on(monkeypatch):
    from simulation.pipeline._phase13_isolation import phase13_isolation_enabled
    monkeypatch.delenv("MPH_PHASE13_ISOLATE", raising=False)
    assert phase13_isolation_enabled() is True
    monkeypatch.setenv("MPH_PHASE13_ISOLATE", "0")
    assert phase13_isolation_enabled() is False


# ── determinism: isolated mc-probe == in-process mc-probe (no behaviour change) ─
def test_mc_probe_isolated_equals_inprocess():
    """A model's per-model mc probe must return the SAME selection + OOF WIS
    whether run in-process or in an isolated subprocess (proves isolation is a
    transparent transport, not a behaviour change)."""
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    from simulation.pipeline.per_model_optimize import _probe_one_model
    from simulation.pipeline._phase13_isolation import run_isolated

    name = "ElasticNet"
    if REGISTRY.get(name) is None:
        import pytest
        pytest.skip(f"{name} not registered")

    rng = np.random.default_rng(0)
    n, p = 140, 6
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 1.5 - X[:, 1] * 0.7 + rng.normal(scale=0.3, size=n) + 5.0
    fcols = [f"f{i}" for i in range(p)]

    inproc = _probe_one_model(
        name, X, y, transform_name="identity", scaler_name="standard",
        feature_cols=fcols, n_folds=2,
    )
    isolated = run_isolated(
        "simulation.pipeline.per_model_optimize:_mc_probe_worker",
        {"mname": name, "X_train": X, "y_train": y,
         "transform_name": "identity", "scaler_name": "standard",
         "feature_cols": fcols, "n_folds": 2},
        timeout=300, stall_timeout=120, label=name,
    )
    assert "__crashed__" not in isolated, f"isolated probe crashed: {isolated}"
    assert isolated["best"] == inproc["best"], (
        f"selection differs: inproc={inproc['best']} isolated={isolated['best']}"
    )
    for m in ("none", "vif", "corr", "pca"):
        a = inproc["cells"][m]["oof_wis"]
        b = isolated["cells"][m]["oof_wis"]
        if np.isfinite(a) and np.isfinite(b):
            assert abs(a - b) <= 1e-6 + 1e-3 * abs(a), (
                f"{name}/{m} oof_wis drift: inproc={a} isolated={b}"
            )


if __name__ == "__main__":
    test_run_isolated_returns_dict(); print("PASS  returns dict")
    test_run_isolated_contains_process_abort(); print("PASS  contains abort")
    test_run_isolated_contains_timeout(); print("PASS  contains timeout")
    test_run_isolated_returns_worker_error_as_data(); print("PASS  worker error as data")
    test_mc_probe_isolated_equals_inprocess(); print("PASS  probe determinism")
    print("=== ALL PASS ===")
