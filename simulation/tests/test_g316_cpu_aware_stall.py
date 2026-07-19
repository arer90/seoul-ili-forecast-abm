"""G-316 (2026-06-18): isolate stall-guard must treat CPU activity as a liveness signal.

The stall-guard killed a child when its child.log was silent for stall_timeout — but a
foundation model (TabPFN inference) or big libsvm computes LONG without logging, so log
growth alone false-killed it (band-aid G-312 = 3× stall). _child_cpu_active lets a CPU-busy
child extend the stall window; only a child idle in BOTH log AND CPU genuinely stalls. The
absolute ceiling (timeout × MAX_EXTEND) still bounds a CPU-busy runaway.

macOS: run PER-FILE.
"""
import subprocess
import sys
import time

from simulation.pipeline._phase13_isolation import _child_cpu_active


def test_g316_busy_process_is_active():
    """A CPU-spinning child (computes, no log) is reported ACTIVE → stall window extends."""
    p = subprocess.Popen([sys.executable, "-c", "x=0\nwhile True:\n x+=1"])
    try:
        time.sleep(0.5)  # let it spin up
        assert _child_cpu_active(p.pid, threshold=15.0) is True
    finally:
        p.kill()
        p.wait()


def test_g316_idle_process_not_active():
    """A sleeping child (no log AND no CPU) is NOT active → genuine stall, may be killed."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        time.sleep(0.3)
        assert _child_cpu_active(p.pid, threshold=15.0) is False
    finally:
        p.kill()
        p.wait()


def test_g316_dead_pid_is_conservative_false():
    """Unmeasurable / non-existent pid → False (let stall fire; never keep alive forever)."""
    assert _child_cpu_active(2_000_000_000) is False
    assert _child_cpu_active(-1) is False
