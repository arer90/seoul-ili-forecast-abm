"""G-317 (2026-06-18): isolate monitor emits a periodic heartbeat → ALL models traceable.

Most models log during fit, but during a LONG single step (foundation inference, SVR inner
Optuna with show_progress_bar=False) the log goes silent — so we couldn't trace what the model
was doing. run_isolated now writes a heartbeat to the MAIN log every _HEARTBEAT s
(elapsed / log-silence / CPU%), separate from child.log so it does not mask a real stall.

macOS: run PER-FILE.
"""
import inspect
import subprocess
import sys
import time

from simulation.pipeline._phase13_isolation import _child_cpu_pct, _child_cpu_active


def test_g317_cpu_pct_busy_is_high():
    p = subprocess.Popen([sys.executable, "-c", "x=0\nwhile True:\n x+=1"])
    try:
        time.sleep(0.5)
        assert _child_cpu_pct(p.pid) > 15.0
    finally:
        p.kill(); p.wait()


def test_g317_cpu_pct_idle_is_low():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        time.sleep(0.3)
        assert 0.0 <= _child_cpu_pct(p.pid) < 15.0
    finally:
        p.kill(); p.wait()


def test_g317_cpu_pct_dead_is_negative():
    """Unmeasurable pid → -1.0 (callers treat as inactive)."""
    assert _child_cpu_pct(2_000_000_000) == -1.0
    assert _child_cpu_active(2_000_000_000) is False  # -1.0 → not active


def test_g317_heartbeat_wired_in_run_isolated():
    """run_isolated has the heartbeat block + interval (source-level wiring check)."""
    from simulation.pipeline import _phase13_isolation as iso
    src = inspect.getsource(iso.run_isolated)
    assert "last_heartbeat" in src and "_HEARTBEAT" in src, "heartbeat state must exist"
    assert "[R9-isolate]" in src and "진행 elapsed" in src, "heartbeat must log to main log"
    assert "MPH_ISOLATE_HEARTBEAT" in src, "heartbeat interval must be env-configurable"
