"""
Reproducibility audit log.
==========================

Captures the metadata a reviewer needs to reproduce a run: git hash,
DB checksum, dependency versions, Python info, OS info, RNG seed,
config hash. Written once per run by EvalLogger.log_audit().

Use:
    from simulation.utils.audit_log import capture_audit
    metadata = capture_audit(config)
    eval_logger.log_audit(metadata)
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _git_hash(short: bool = True) -> str:
    """Current git HEAD short hash, or '?' if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode().strip() or "?"
    except Exception:
        return "?"


def _git_dirty() -> bool:
    """True if working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return len(out.decode().strip()) > 0
    except Exception:
        return False


def _db_checksum(db_path: Optional[Path]) -> dict:
    if db_path is None or not Path(db_path).exists():
        return {"db_path": str(db_path), "exists": False}
    p = Path(db_path)
    stat = p.stat()
    # Compute SHA-256 of first 1MB + last 1MB (cheap fingerprint, not full hash)
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(1024 * 1024))
        if stat.st_size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, 2)
            h.update(f.read(1024 * 1024))
    return {
        "db_path": str(p),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256_fingerprint_first_last_1MB": h.hexdigest(),
    }


def _key_dep_versions() -> dict:
    """Versions of the packages that materially affect results."""
    deps = {}
    for mod in ("numpy", "polars", "pandas", "torch", "sklearn",
                "lightgbm", "xgboost", "scipy", "statsmodels", "duckdb"):
        try:
            m = __import__(mod)
            deps[mod] = getattr(m, "__version__", "?")
        except ImportError:
            deps[mod] = "<not installed>"
        except Exception as e:
            deps[mod] = f"<error: {e}>"
    return deps


def _platform_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "?",
    }


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "backend": "cuda",
                "device": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "cuda_version": getattr(torch.version, "cuda", "?"),
            }
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return {"backend": "mps", "device": "Apple Silicon (Metal)",
                    "torch_version": torch.__version__}
        return {"backend": "cpu", "torch_version": torch.__version__}
    except ImportError:
        return {"backend": "<no torch>"}


def _seed_info(config) -> dict:
    return {
        "PYTHONHASHSEED": __import__("os").environ.get("PYTHONHASHSEED", "<unset>"),
        "config_seed": _safe(lambda: getattr(config, "seed", None), None),
        "torch_deterministic": _safe(lambda: __import__("torch").are_deterministic_algorithms_enabled(), None),
    }


def capture_audit(config) -> dict:
    """One-shot audit capture. Returns a dict suitable for JSON serialization."""
    db_path = _safe(lambda: config.data.db_path, None)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scenario": _safe(lambda: getattr(config, "scenario", "?"), "?"),
        "preset": _safe(lambda: getattr(config, "preset", "?"), "?"),
        "git": {
            "hash": _git_hash(short=True),
            "dirty": _git_dirty(),
        },
        "db": _db_checksum(db_path) if db_path else {"db_path": None},
        "deps": _key_dep_versions(),
        "platform": _platform_info(),
        "gpu": _gpu_info(),
        "seeds": _seed_info(config),
        "split": {
            "paper_cutoff_week": _safe(lambda: config.split.paper_cutoff_week, None),
            "in_sample_test_ratio": _safe(lambda: config.split.in_sample_test_ratio, None),
            "in_sample_val_ratio": _safe(lambda: config.split.in_sample_val_ratio, None),
            "weather_mode": _safe(lambda: config.split.real_weather_mode, "?"),
            "covid_mode": _safe(lambda: config.split.covid_inclusion_mode, "?"),
            "conformal_method": _safe(lambda: config.split.real_conformal_method, "?"),
            "ensemble_method": _safe(lambda: config.split.ensemble_method, "?"),
            "horizons": list(_safe(lambda: config.split.real_horizons, (1,))),
        },
        "training": {
            "batch_size": _safe(lambda: config.training.batch_size, None),
            "epochs": _safe(lambda: config.training.epochs, None),
            "early_stopping_patience": _safe(lambda: config.training.early_stopping_patience, None),
        },
        "optuna": {
            "mode": _safe(lambda: config.optuna.mode, "?"),
            "trials": _safe(lambda: config.optuna.trials, None),
            "strategy": _safe(lambda: config.optuna.strategy, "?"),
        },
    }
