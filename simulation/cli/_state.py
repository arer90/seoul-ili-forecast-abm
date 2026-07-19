"""Resume state helpers — shared by large CLI handlers.

Phase C2 partial (2026-05-12): extracted from __main__.py:1150-1184 so
large handlers (cmd_train_all, cmd_run_all, etc.) can be split into
their own cli/* modules without depending on __main__.py-level state.

Design (D-4 deep module):
    Small interface (4 functions: path / load / save / clear)
    NaN-safe: load on missing/corrupt JSON returns empty dict
    Crash-safe save: tempfile → atomic os.replace

API:
    state_path(name)          → Path (checkpoint JSON file)
    load_state(name)          → dict (empty if missing/corrupt)
    save_state(name, state)   → None (atomic write)
    clear_state(name)         → None (idempotent unlink)

Performance: O(1) for path / clear; O(state JSON size) for load / save.
Side effects: filesystem writes under DATA_DIR/results/.
Caller responsibility: name must be unique per phase (e.g. "run_all", "train").

Note: leading-underscore prefix preserved (`_state_path` etc.) so
__main__.py can re-import as `from simulation.cli._state import ...`
without breaking existing callers within the module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _state_path(name: str) -> Path:
    """Return checkpoint JSON path under simulation/data/results/."""
    from simulation.database.config import DATA_DIR

    p = Path(DATA_DIR) / "results" / f".{name}_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state(name: str) -> dict:
    """Read checkpoint JSON. Empty dict on missing or corrupt."""
    p = _state_path(name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(name: str, state: dict) -> None:
    """Atomic write of checkpoint JSON (tempfile → os.replace)."""
    p = _state_path(name)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _clear_state(name: str) -> None:
    """Idempotent removal of checkpoint JSON."""
    p = _state_path(name)
    if p.exists():
        p.unlink()


__all__ = [
    "_state_path",
    "_load_state",
    "_save_state",
    "_clear_state",
]
