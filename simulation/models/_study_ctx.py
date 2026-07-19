"""G-14H (2026-06-21, codex): content-hash for Optuna study names → safe warm-start.

Persistent Optuna studies (storage=sqlite, load_if_exists=True) used FIXED names ({model}_v1),
so across runs they reused trials from incompatible contexts (different code/data) — stale
warm-start that undermined #13's per-trial seeding (G-13F). Appending this content hash to the
study name means a study is reused ONLY when the context fingerprint matches; any code change
(git commit / dirty tree) or schema bump yields a NEW study (= recompute from scratch). Same
context → safe reuse (the warm-start speed benefit is preserved).

Single source of truth (D-2): both _optuna_torch.run_optuna_loop and dl_models DNN/TCN import
``study_ctx_hash`` from here. Computed once per process and cached.

The fingerprint is intentionally git-commit-based (captures every meaningful code change incl the
#13/#12 fixes). Finer-grained payloads (feature columns, HP search space, data-table fingerprint
via simulation/utils/db_fingerprint.py) are a possible refinement; the git+dirty signal already
guarantees post-fix studies are distinct from pre-fix ones, which is the load-bearing property.
"""
from __future__ import annotations

import hashlib
import subprocess

_SCHEMA = "v2"          # bump when the study payload/contract changes
_CACHE: str | None = None


def study_ctx_hash() -> str:
    """Return a stable 12-char hex content hash of the current code/schema context.

    Returns:
        12-char lowercase hex digest. Stable within a process (cached); identical across runs
        iff git HEAD + dirty-state + schema are identical. Falls back to ``git=na`` (still a valid
        hash, just code-change-blind) if git is unavailable — never raises.

    Performance: 2 short ``git`` subprocess calls on first use (~ms), cached thereafter.
    Side effects: none (read-only git introspection).
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    parts = [f"schema={_SCHEMA}"]
    try:
        _gc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, encoding="utf-8", timeout=5).stdout.strip()
        _dirty = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True, encoding="utf-8", timeout=5).stdout.strip()
        parts.append(f"git={_gc or 'na'}{'+dirty' if _dirty else ''}")
    except Exception:
        parts.append("git=na")
    _CACHE = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return _CACHE
