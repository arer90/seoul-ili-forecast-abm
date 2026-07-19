"""R9 (per_model_optimize) per-model subprocess isolation — generic crash-contained runner.

Why this exists (G-236 / G-249 lineage, 2026-06-10)
---------------------------------------------------
The WF-CV path (``MultiModelRunner``) already isolates every individual model
category in a subprocess (``_SUBPROCESS_CATEGORIES`` incl. ``epi``) so in-process
OpenMP/BLAS thread accumulation can't crash a run (G-236). **R9**
(``per_model_optimize``) BYPASSED that discipline: the multicollinearity probe
(``_compare_mc_per_model``) and the main ``optimize_one_model`` loop call
``factory() -> model.fit()`` directly in the parent process. After torch (DL) +
lightgbm (tree) + statsmodels load their *own* libomp into one long-lived
process, an IRLS-heavy ``epi`` model (GLARMA: two sequential NegBin GLM.fit)
tips the already-polluted OMP runtime →

    OMP: Error #179: Function pthread_mutex_init failed: System error #22

That ``#179`` abort is a **process-level abort**, not a Python exception — a
``try/except`` around the fit can NOT contain it. The only robust containment is
the process boundary: run each model's work in a *fresh* child that loads only
that one model's stack (no toxic torch+lightgbm+statsmodels mix), and if the
child dies anyway, the parent survives and continues.

Design (D-4 deep module): one small interface, rich implementation.

    run_isolated("pkg.mod:func", payload_dict) -> dict

The target function takes ONE picklable ``dict`` and returns a picklable ``dict``.
A clean run returns that dict verbatim. A child that aborts (OMP #179 / SIGSEGV /
OOM-kill), times out, or stalls returns a **containment marker**
``{"__crashed__": True, ...}`` instead — the parent NEVER raises and NEVER dies.

This module imports only stdlib (+ optional reuse of runner's OS-aware tmpdir),
so it has NO dependency on ``per_model_optimize`` — the domain worker functions
live there and are addressed by qualified name, keeping the dependency one-way
(``per_model_optimize`` → this module) with no import cycle.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import logging

log = logging.getLogger(__name__)


# ── gate ──────────────────────────────────────────────────────────────────────
def phase13_isolation_enabled() -> bool:
    """Whether R9 (per_model_optimize) model fits run in per-model subprocesses (default ON).

    Returns:
        True unless ``MPH_PHASE13_ISOLATE=0``. Default-on because the in-process
        OMP-accumulation crash (G-236/G-249) is real and reproducible; the opt-out
        exists only for debugging a single model in-process.

    Side effects: none (reads os.environ).
    """
    return os.environ.get("MPH_PHASE13_ISOLATE", "1") != "0"


# ── generic worker script (runs in the child) ────────────────────────────────
# Loads {target, payload} from args.pkl, imports the target by "module:func",
# calls func(payload), pickles the result. A Python-level exception inside func
# is returned as DATA ({"__worker_error__": ...}) so the caller can distinguish a
# clean domain failure from a process abort (which leaves NO result file at all).
_WORKER_SCRIPT = r'''
import os, sys, pickle, traceback, importlib

args_path, result_path = sys.argv[1], sys.argv[2]

def _emit(obj):
    tmp = result_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, result_path)

try:
    with open(args_path, "rb") as f:
        spec = pickle.load(f)
    target = spec["target"]          # "package.module:function"
    payload = spec["payload"]
    mod_name, _, fn_name = target.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    out = fn(payload)
    if not isinstance(out, dict):
        out = {"__worker_error__": f"worker returned {type(out).__name__}, expected dict"}
    _emit(out)
except BaseException as e:               # noqa: BLE001 — capture EVERYTHING returnable
    try:
        _emit({"__worker_error__": repr(e), "__traceback__": traceback.format_exc()})
    except Exception:
        pass
finally:
    # Hard exit guarantees the OS reclaims this child's OMP/BLAS thread teams and
    # all memory — no atexit/GC that could re-touch a corrupted OMP runtime.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
'''


def _fast_tmpdir(prefix: str) -> str:
    """OS-aware temp dir, reusing runner's tracked /dev/shm chooser when available."""
    try:
        from simulation.models.runner import _get_fast_tmpdir, _track_tmpdir
        return _track_tmpdir(_get_fast_tmpdir(prefix=prefix))
    except Exception:
        import tempfile
        return tempfile.mkdtemp(prefix=prefix)


def _child_env() -> dict:
    """Parent env + explicit device pin (so the child picks the same accelerator)."""
    env = os.environ.copy()
    if "MPH_DEVICE" not in env and env.get("MPH_FORCE_CPU") != "1":
        try:
            from simulation.models.base import device_str as _dstr
            env["MPH_DEVICE"] = _dstr()
        except Exception:
            pass
    return env


def _child_cpu_pct(pid: int, sample: float = 0.3) -> float:
    """Sum CPU% of ``pid`` + descendants over ``sample`` s. ``-1.0`` on any error.

    Shared by the isolate stall-guard (liveness, G-316) and the heartbeat (traceability, G-317).
    A child computing WITHOUT writing its log (TabPFN inference, libsvm SMO) still shows CPU.
    ``-1.0`` = unmeasurable → callers treat as inactive (conservative).

    Performance: blocks ~``sample`` s. Side effects: none (read-only psutil).
    """
    try:
        import psutil
        targets = [psutil.Process(pid)]
        try:
            targets += targets[0].children(recursive=True)
        except psutil.Error:
            pass
        for q in targets:
            try:
                q.cpu_percent(None)      # prime — first call returns 0.0 / since-creation
            except psutil.Error:
                pass
        time.sleep(sample)
        total = 0.0
        for q in targets:
            try:
                total += q.cpu_percent(None)
            except psutil.Error:
                pass
        return total
    except Exception:
        return -1.0


def _child_cpu_active(pid: int, threshold: float = 15.0, sample: float = 0.3) -> bool:
    """True if ``pid`` (incl. descendants) uses > ``threshold`` % CPU (G-316 liveness signal).

    A child that computes WITHOUT writing its log (TabPFN inference, libsvm SMO) is NOT stalled
    even though log growth is silent. Error (``-1.0``) → ``False`` (conservative — stall fires).
    """
    return _child_cpu_pct(pid, sample) > threshold


def run_isolated(
    target_qual: str,
    payload: dict,
    *,
    timeout: float = 1800.0,
    stall_timeout: float = 300.0,
    poll_interval: float = 3.0,
    label: str = "model",
) -> dict:
    """Run ``target_qual(payload)`` in a fresh spawn subprocess; contain any crash.

    Args:
        target_qual: ``"package.module:function"`` — the function takes ONE
            picklable ``dict`` (``payload``) and returns a picklable ``dict``. It
            is re-imported in the child (fresh interpreter), so module-level state
            (REGISTRY, GLOBAL config) re-initialises from inherited env vars.
        payload: picklable arguments for the target (numpy arrays / str / list /
            dict — NOT lambdas; reconstruct callables inside the target from names).
        timeout: hard wall-clock cap (s). Exceeding it kills the child → crash marker.
        stall_timeout: kill if the child's log is silent this long (s) — guards a
            hung fit that never returns.
        poll_interval: parent poll cadence (s).
        label: short name for logs (the model name).

    Returns:
        The target's ``dict`` verbatim on a clean run. Otherwise a containment
        marker — the parent NEVER raises:
          * ``{"__crashed__": True, "rc": int|None, "reason": "exit"|"timeout"|
             "stall"|"no_result", "stderr_tail": str}`` — child aborted (OMP #179 /
             SIGSEGV / OOM-kill), timed out, stalled, or left no result file.
          * ``{"__worker_error__": str, "__traceback__": str}`` — the target raised
             a normal Python exception (returned as data by the child).

    Performance: ~30-60ms spawn overhead per call + the target's own runtime.
    Side effects: writes args/result pickles + a log file under a temp dir
        (auto-tracked for atexit cleanup); spawns and reaps one child process.
    Caller responsibility: ``payload`` must be picklable; the target must return a
        picklable dict; check for ``__crashed__`` / ``__worker_error__`` keys.
    """
    tmp_dir = _fast_tmpdir(prefix=f"mph_p13_{label}_")
    args_path = str(Path(tmp_dir) / "args.pkl")
    result_path = str(Path(tmp_dir) / "result.pkl")
    worker_path = str(Path(tmp_dir) / "_worker.py")
    log_path = str(Path(tmp_dir) / "child.log")

    with open(worker_path, "w", encoding="utf-8") as f:
        f.write(_WORKER_SCRIPT)
    with open(args_path, "wb") as f:
        pickle.dump({"target": target_qual, "payload": payload}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    _logf = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, worker_path, args_path, result_path],
            cwd=repo_root, stdout=_logf, stderr=_logf, env=_child_env(),
        )
    except Exception as e:  # spawn itself failed — degrade to crash marker
        _logf.close()
        log.warning("  [R9-isolate] %s: spawn failed → %r", label, e)
        return {"__crashed__": True, "rc": None, "reason": "spawn", "stderr_tail": repr(e)}

    start = time.time()
    last_size = 0
    last_activity = start
    last_heartbeat = start             # G-317: 주기적 진행 heartbeat → 모든 격리 모델 추적가능
    try:
        _HEARTBEAT = max(15.0, float(os.environ.get("MPH_ISOLATE_HEARTBEAT", "60")))
    except (TypeError, ValueError):
        _HEARTBEAT = 60.0
    reason: Optional[str] = None
    # G-260: how far past the soft `timeout` an ACTIVELY-progressing child may run before the
    # absolute ceiling kills it (env MPH_ISOLATE_MAX_EXTEND, default 3×). Stall still kills idle.
    try:
        _MAX_EXTEND = max(1.0, float(os.environ.get("MPH_ISOLATE_MAX_EXTEND", "3")))
    except (TypeError, ValueError):
        _MAX_EXTEND = 3.0
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        now = time.time()
        # G-317 (2026-06-18, user): periodic heartbeat to the MAIN log so EVERY isolated model is
        #   traceable even when it is silent during a long step (foundation inference, inner
        #   Optuna). Separate from child.log (the stall signal) → does NOT mask a real stall.
        if now - last_heartbeat >= _HEARTBEAT:
            last_heartbeat = now
            _cpu = _child_cpu_pct(proc.pid)
            log.info("  [R9-isolate] %s 진행 elapsed=%.0fs log_silent=%.0fs CPU=%s",
                     label, now - start, now - last_activity,
                     f"{_cpu:.0f}%" if _cpu >= 0 else "?")
        try:
            cur = os.path.getsize(log_path)
            if cur > last_size:
                last_size = cur
                last_activity = now
        except OSError:
            pass
        if now - last_activity > stall_timeout:
            # G-316 (2026-06-18, user): log-silence ≠ stall. A foundation model (TabPFN
            #   inference) or big libsvm computes LONG without writing child.log → log growth
            #   alone false-killed it (band-aid G-312 = 3× stall). If the child is STILL using
            #   CPU it IS progressing → reset the stall window; only a child idle in BOTH log
            #   AND CPU genuinely stalls. The absolute ceiling below (timeout × MAX_EXTEND)
            #   still bounds a CPU-busy runaway loop.
            if _child_cpu_active(proc.pid):
                last_activity = now     # CPU active = progressing → extend, don't kill
            else:
                reason = "stall"        # no log AND no CPU = genuine stall → fallback
                break
        # G-260 (user 2026-06-12): progress-aware hard cap. Don't kill work that is still
        # actively progressing — the log growing means OOF folds / Optuna trials are completing.
        # The hard `timeout` is now a SOFT target: past it we only stop if the child has also
        # gone (semi-)idle; while it keeps logging we extend up to `timeout × MAX_EXTEND`
        # (a generous absolute ceiling that guards a runaway-but-active loop). So: 진행 중이면
        # 추가시간, 정체면 fallback. (Previously a tight hard cap killed mid-progress mc-probes.)
        _recently_active = (now - last_activity) < stall_timeout * 0.5
        if now - start > timeout and not _recently_active:
            reason = "timeout"          # past soft cap AND no recent progress → stop
            break
        if now - start > timeout * _MAX_EXTEND:
            reason = "timeout_max"      # absolute ceiling (runaway guard) regardless of activity
            break
        time.sleep(poll_interval)

    if reason is not None:                       # stall / timeout → kill + reap
        try:
            proc.kill()
            proc.wait(timeout=30)
        except Exception:
            pass
        _logf.close()
        log.warning("  [R9-isolate] %s: %s after %.0fs → killed (contained)",
                    label, reason, time.time() - start)
        return {"__crashed__": True, "rc": proc.returncode, "reason": reason,
                "stderr_tail": _tail(log_path)}

    _logf.close()
    rc = proc.returncode

    # Clean exit: a result file should exist (worker always _emit()s before os._exit).
    if os.path.exists(result_path):
        try:
            with open(result_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning("  [R9-isolate] %s: result unpickle failed → %r", label, e)
            return {"__crashed__": True, "rc": rc, "reason": "result_unpickle",
                    "stderr_tail": _tail(log_path)}

    # No result file = the child aborted BEFORE _emit (OMP #179 / SIGSEGV / OOM-kill).
    log.warning("  [R9-isolate] %s: no result (rc=%s) — process abort contained "
                "(OMP/SIGSEGV/OOM). stderr: %s", label, rc, _tail(log_path, 400))
    return {"__crashed__": True, "rc": rc, "reason": "no_result",
            "stderr_tail": _tail(log_path)}


def _tail(path: str, n: int = 800) -> str:
    try:
        data = Path(path).read_text(encoding="utf-8", errors="replace")
        return data[-n:]
    except Exception:
        return ""
