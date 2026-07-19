"""Central path resolution — no more hardcoded drive letters.

Storage root priority (first match wins):
    1. Env var `MPH_OUTPUT_ROOT` — the explicit override. Set this on any
       OS to redirect heavy artifacts to a different disk (external SSD,
       second HDD, network mount). Example:
           export MPH_OUTPUT_ROOT=/Volumes/ExternalSSD/mph
           export MPH_OUTPUT_ROOT=D:/mph_outputs     (Windows)
           export MPH_OUTPUT_ROOT=/mnt/data/mph       (Linux)
    2. Project-local `simulation/results/` — the default. Works everywhere
       without configuration; requires the main drive to have enough space.

The E:/MPH_results pattern used historically on the original Windows
machine is NO longer hardcoded. If you want E: again, set:
    set MPH_OUTPUT_ROOT=E:/mph      (cmd)
    $env:MPH_OUTPUT_ROOT = "E:/mph" (PowerShell)

API:
    from simulation.utils.paths import (
        get_output_root,   # base storage dir (user's `MPH_OUTPUT_ROOT` or project-local)
        get_results_dir,   # base / results/
        get_cache_dir,     # base / cache/
        get_optuna_dir,    # base / results/  (Optuna feat-sel JSONs)
        get_models_pt_dir, # base / results/models_pt/  (or project-local checkpoints_history)
        resolve_path,      # generic: prefer env-rooted, fall back to project-local
    )

Design rule:
    Callers SHOULD NOT construct `Path("E:/MPH_results")` or any other
    OS-specific absolute path. Use these helpers. If the helpers are too
    narrow for your use case, extend them here rather than hardcoding.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

__all__ = [
    "get_output_root",
    "get_results_dir",
    "get_cache_dir",
    "get_optuna_dir",
    "get_models_pt_dir",
    "resolve_path",
    "PROJECT_ROOT",
]


def _find_project_root() -> Path:
    """Locate repo root by walking up from this file until finding pyproject.toml."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fallback: three levels up from simulation/utils/paths.py
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _find_project_root()


@lru_cache(maxsize=1)
def get_output_root() -> Path:
    """Base storage directory for heavy artifacts (results, cache, checkpoints).

    Returns:
        - `Path($MPH_OUTPUT_ROOT)` if env var is set and non-empty. The
          directory is created if missing. Symlinks and relative paths work.
        - Otherwise `<repo>/simulation/` — project-local default. This keeps
          results next to the code for analysts who don't need a separate disk.
    """
    from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
    override = _GCFG.paths.output_root
    if override:
        p = Path(override).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    # Default: project-local — results live next to the code.
    return PROJECT_ROOT / "simulation"


def get_results_dir() -> Path:
    """The `results/` subdirectory under the output root. Auto-created."""
    p = get_output_root() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cache_dir() -> Path:
    """The `cache/` subdirectory. Used by feature_engine parquet cache."""
    p = get_output_root() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_optuna_dir() -> Path:
    """Where `optuna_feat_sel_*.json` lives.

    Defaults to the same as `get_results_dir()`. Kept as a separate function
    so a future split is a one-line change without touching call sites.
    """
    return get_results_dir()


def get_models_pt_dir() -> Path:
    """Where trained `.pt` checkpoints live.

    When `MPH_OUTPUT_ROOT` is set: `<root>/results/models_pt/` (historical
    layout, lots of artifacts off the main drive).
    When project-local: `<repo>/simulation/checkpoints_history/` (the
    in-tree 54 existing checkpoints).
    """
    from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
    if _GCFG.paths.output_root:
        p = get_results_dir() / "models_pt"
    else:
        p = PROJECT_ROOT / "simulation" / "checkpoints_history"
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(
    *parts: str | Path,
    under: str = "results",
    create: bool = False,
) -> Path:
    """Generic resolver. Joins `parts` under a named output subdirectory.

    Examples:
        resolve_path("optuna_feat_sel_lightgbm.json")
            → $OUTPUT_ROOT/results/optuna_feat_sel_lightgbm.json
        resolve_path("feature_cache.parquet", under="cache", create=True)
            → $OUTPUT_ROOT/cache/feature_cache.parquet   (cache/ dir ensured)
    """
    base = {
        "results": get_results_dir(),
        "cache": get_cache_dir(),
        "models_pt": get_models_pt_dir(),
        "root": get_output_root(),
    }.get(under, get_output_root() / under)

    if create:
        base.mkdir(parents=True, exist_ok=True)

    return base.joinpath(*map(str, parts))
