"""
Optuna ASK-TELL + per-trial subprocess + tmp-file IPC.
=======================================================

User-requested pattern (정확히 사용자가 의도한 대로):

  1. Parent holds the Optuna ``study`` (TPE sampler, all in-memory state).
  2. For each trial:
     a. parent calls ``trial = study.ask()``  → sampler suggests params
     b. parent serializes ``(params, eval_args)`` to a **temp pickle file**
     c. parent spawns ``subprocess.Popen`` with a small worker script
     d. worker loads the pickle, runs the user's eval function
        (= one ``model.fit + score``), writes ``(value, error?)`` to a
        second temp pickle, and **exits** — which guarantees the OS
        reclaims all allocated memory (PyTorch tensors, MPS, libomp
        thread-pool, etc.) regardless of how heavy the trial was.
     e. parent reads back the temp pickle → ``study.tell(trial, value)``
     f. temp files are deleted; loop continues.

Net effect (precisely the user's question):
  • Each trial gets a *fresh process* → no memory leaks accumulate
    over the study (the ~50-trial mark where the previous setup OOMed
    is now irrelevant).
  • CPU pressure stays at "one trial at a time" — the OS happily reuses
    cores between subprocesses.
  • Crash containment — a SIGSEGV / OOM / Optuna prune kills only that
    trial's subprocess; the parent records FAIL and goes to next.
  • No RDB dependency — pickle file IPC is enough.

Sampler state guarantees: TPE / CMA-ES samplers all keep their state
inside ``study`` (in-memory). ``study.ask()`` reads it; ``study.tell()``
updates it. The subprocess *only* runs the eval; sampler state never
crosses process boundaries.

Public API
----------

    from simulation.models._optuna_subprocess import optimize_with_isolation

    study = optuna.create_study(direction="minimize")
    optimize_with_isolation(
        study=study,
        eval_fn=my_eval_function,        # (params: dict) → float
        param_space=my_param_suggester,  # (trial) → dict
        n_trials=50,
        model_name="GE-DNN-GAT",
        timeout_per_trial=600,
    )

The ``eval_fn`` MUST be importable by the child process — pass a
fully-qualified module-level function, not a lambda or closure.

OS-aware
--------
For models in ``runner._INPROCESS_OVERRIDE_*`` (e.g. PyG/MPS-fork-unsafe
on macOS), this falls back to plain in-process ``study.optimize``.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Worker script — written to a temp .py and run via `python <script>`
# ──────────────────────────────────────────────────────────────────────
_TRIAL_WORKER_SCRIPT = r'''
"""One-shot Optuna trial worker. Reads params, calls eval_fn, writes value."""
import sys, os, pickle, traceback, time, gc

in_path  = sys.argv[1]   # input pickle: {eval_fn_qual, params, repo_root}
out_path = sys.argv[2]   # output pickle: {value, error?}

try:
    with open(in_path, "rb") as f:
        payload = pickle.load(f)
    eval_fn_qual = payload["eval_fn_qual"]      # "module.path:func_name"
    params       = payload["params"]
    repo_root    = payload.get("repo_root", os.getcwd())
    extra_kwargs = payload.get("extra_kwargs", {}) or {}

    sys.path.insert(0, repo_root)

    # Resolve eval_fn
    mod_path, func_name = eval_fn_qual.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(mod_path)
    eval_fn = getattr(mod, func_name)

    print(f"[trial-worker] params={params}", flush=True)
    t0 = time.time()
    value = eval_fn(params, **extra_kwargs)
    dt = time.time() - t0
    print(f"[trial-worker] value={value} (took {dt:.1f}s)", flush=True)

    with open(out_path + ".tmp", "wb") as f:
        pickle.dump({"value": float(value), "elapsed": dt}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(out_path + ".tmp", out_path)
except Exception as e:
    err = {"error": f"{type(e).__name__}: {e}",
           "traceback": traceback.format_exc()}
    try:
        with open(out_path, "wb") as f:
            pickle.dump(err, f)
    except Exception:
        pass
    print(f"[trial-worker] FAIL: {err['error']}", file=sys.stderr, flush=True)
    sys.exit(1)

# Best-effort GPU/MPS cleanup — process exit reclaims OS-side anyway,
# but explicitly empty cuda cache so any peer subprocess that's still
# alive sees free VRAM (NVIDIA driver doesn't reclaim on dirty exit).
try:
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
except Exception:
    pass

# Hard exit — skip Python destructors that may segfault during torch cleanup
os._exit(0)
'''


def _write_worker_script(tmpdir: Path) -> Path:
    p = tmpdir / "_optuna_trial_worker.py"
    p.write_text(_TRIAL_WORKER_SCRIPT)
    return p


def _is_in_process_only(model_name: str) -> bool:
    """Mirror runner.py's per-OS in-process whitelist."""
    try:
        from simulation.models.runner import _get_inprocess_override
        return model_name in _get_inprocess_override()
    except Exception:
        return False


def optimize_with_isolation(
    study: Any,                              # optuna.Study
    eval_fn: Callable[[dict], float],        # (params) → metric (lower better)
    param_space: Callable[[Any], dict],      # (trial) → params dict
    n_trials: int,
    *,
    model_name: str = "<unknown>",
    eval_fn_qual: Optional[str] = None,      # "pkg.mod:func"; required for subprocess
    timeout_per_trial: int = 600,
    isolate_trials: bool = True,
    extra_kwargs: Optional[dict] = None,
    repo_root: Optional[Path] = None,
    show_progress: bool = True,
) -> dict:
    """ASK-TELL Optuna driver with per-trial subprocess + tmp-file IPC.

    Returns ``{n_completed, n_failed, n_pruned, elapsed_sec}``.
    """
    extra_kwargs = extra_kwargs or {}
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)

    # In-process fallback for models that can't survive subprocess fork
    use_subprocess = isolate_trials and not _is_in_process_only(model_name)
    if not use_subprocess:
        log.info(f"  [optuna-iso] {model_name}: in-process "
                  f"(reason: {'isolate_trials=False' if not isolate_trials else 'OS whitelist'})")

        def _objective(trial):
            params = param_space(trial)
            return eval_fn(params, **extra_kwargs)
        t0 = time.time()
        n_before = len(study.trials)
        study.optimize(_objective, n_trials=n_trials, gc_after_trial=True,
                         show_progress_bar=show_progress)
        return {
            "n_completed": len(study.trials) - n_before,
            "n_failed":    0,
            "elapsed_sec": time.time() - t0,
            "mode":        "in-process",
        }

    if not eval_fn_qual:
        # Subprocess needs a fully-qualified function name to import in the
        # child. Closures / lambdas can't be unpickled there.
        log.warning(f"  [optuna-iso] {model_name}: eval_fn_qual not given → "
                      f"falling back to in-process")
        return optimize_with_isolation(
            study=study, eval_fn=eval_fn, param_space=param_space,
            n_trials=n_trials, model_name=model_name,
            isolate_trials=False, extra_kwargs=extra_kwargs,
            repo_root=repo_root, show_progress=show_progress)

    log.info(f"  [optuna-iso] {model_name}: per-trial subprocess "
              f"(n_trials={n_trials}, timeout={timeout_per_trial}s, "
              f"eval_fn={eval_fn_qual})")

    # 2026-04-26: OS-aware fast tmpdir (Linux /dev/shm 자동) — runner._get_fast_tmpdir 위임
    # 2026-04-27: atexit cleanup 추적 (crash/SIGINT 시 자동 회수)
    # Linux 운영 시 trial 격리 IO 50% ↓; macOS/Windows 는 default 그대로
    try:
        from simulation.models.runner import _get_fast_tmpdir, _track_tmpdir
        td_path = _track_tmpdir(_get_fast_tmpdir(prefix=f"optuna_iso_{model_name}_"))
        # context manager 호환 위해 cleanup 명시적
        import shutil as _shutil
        class _TempDirCM:
            def __init__(self, p): self.p = p
            def __enter__(self): return self.p
            def __exit__(self, *a):
                try: _shutil.rmtree(self.p, ignore_errors=True)
                except Exception: pass
        _ctx = _TempDirCM(td_path)
    except Exception:
        # fallback — 기존 default tempfile
        _ctx = tempfile.TemporaryDirectory(prefix=f"optuna_iso_{model_name}_")
    with _ctx as td:
        tmpdir = Path(td)
        worker_py = _write_worker_script(tmpdir)
        py = sys.executable

        n_completed = 0
        n_failed = 0
        t_global = time.time()
        for i in range(n_trials):
            trial = study.ask()
            try:
                params = param_space(trial)
            except optuna.TrialPruned as e:
                # 2026-05-12 Codex BUG fix: TrialPruned (예: param_budget 초과) 가
                # broad exception 으로 잡혀서 FAIL 처리되던 문제. PRUNED 로 명시.
                log.info(f"  [optuna-iso] {model_name} trial {i+1}: "
                         f"PRUNED at param_space (over-budget?): {e}")
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            except Exception as e:
                log.warning(f"  [optuna-iso] {model_name} trial {i+1}: "
                              f"param_space() crash: {e}")
                study.tell(trial, state="FAIL")
                n_failed += 1
                continue

            in_path = tmpdir / f"trial_{i:03d}_in.pkl"
            out_path = tmpdir / f"trial_{i:03d}_out.pkl"
            payload = {
                "eval_fn_qual": eval_fn_qual,
                "params":       params,
                "repo_root":    str(repo_root),
                "extra_kwargs": extra_kwargs,
            }
            try:
                with in_path.open("wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                log.warning(f"  [optuna-iso] {model_name} trial {i+1}: "
                              f"pickle params failed: {e}")
                study.tell(trial, state="FAIL")
                n_failed += 1
                continue

            t0 = time.time()
            try:
                rc = subprocess.call(
                    [py, str(worker_py), str(in_path), str(out_path)],
                    timeout=timeout_per_trial + 30,
                )
            except subprocess.TimeoutExpired:
                log.warning(f"  [optuna-iso] {model_name} trial {i+1}: "
                              f"timeout {timeout_per_trial}s")
                study.tell(trial, state="FAIL")
                n_failed += 1
                continue
            except Exception as e:
                log.warning(f"  [optuna-iso] {model_name} trial {i+1}: "
                              f"subprocess error: {e}")
                study.tell(trial, state="FAIL")
                n_failed += 1
                continue
            dt = time.time() - t0

            # Read result
            if not out_path.exists():
                log.warning(f"  [optuna-iso] {model_name} trial {i+1}: "
                              f"no output file (rc={rc}, {dt:.1f}s)")
                study.tell(trial, state="FAIL")
                n_failed += 1
                continue
            try:
                with out_path.open("rb") as f:
                    res = pickle.load(f)
            except Exception as e:
                log.warning(f"  [optuna-iso] {model_name} trial {i+1}: "
                              f"unpickle failed: {e}")
                study.tell(trial, state="FAIL")
                n_failed += 1
                continue

            if "error" in res:
                log.warning(f"  [optuna-iso] {model_name} trial {i+1}: "
                              f"worker FAIL: {res['error']}")
                study.tell(trial, state="FAIL")
                n_failed += 1
            else:
                value = float(res["value"])
                study.tell(trial, value)
                n_completed += 1
                if show_progress and (i + 1) % 5 == 0:
                    log.info(f"  [optuna-iso] {model_name} progress: "
                              f"{i+1}/{n_trials}, best={study.best_value:.4f}, "
                              f"avg {dt:.0f}s/trial")

            # Cleanup tmp files immediately to keep /tmp small
            for p in (in_path, out_path):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        elapsed = time.time() - t_global
        log.info(f"  [optuna-iso] {model_name}: done — "
                  f"{n_completed} ok / {n_failed} fail / {elapsed:.1f}s "
                  f"(best={study.best_value if n_completed else float('nan'):.4f})")
        return {
            "n_completed": n_completed,
            "n_failed":    n_failed,
            "elapsed_sec": elapsed,
            "mode":        "subprocess_per_trial",
        }


__all__ = ["optimize_with_isolation"]
