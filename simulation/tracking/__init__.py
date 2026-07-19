"""Experiment tracking layer for MPH Infection Simulation.

Thin wrapper around MLflow (SQLite backend by default) so pipeline code can
log metrics/artifacts/params without caring where the backend lives.

Usage:
    from simulation.tracking import experiment, log_metric, log_params, log_artifact

    with experiment("phase6_wfcv", scenario="full_light"):
        log_params({"train_ratio": 0.7, "wfcv_step": 1})
        for fold_id, metrics in results.items():
            log_metric(f"fold_{fold_id}_rmse", metrics["rmse"])
        log_artifact("simulation/results/diagnostics_report.json")

Configuration:
    Default tracking URI = sqlite:///simulation/data/mlflow.db
    Override via env: MPH_MLFLOW_URI

Fallback behavior:
    If MLflow is not installed OR tracking URI fails, every call becomes a
    no-op with a debug-log entry. Pipeline code never has to try/except.
"""
from __future__ import annotations

import contextlib
import logging
import math
from pathlib import Path
from typing import Any, Iterator, Optional

log = logging.getLogger(__name__)

_DEFAULT_URI = f"sqlite:///{Path('simulation/data/mlflow.db').absolute()}"
_DEFAULT_EXPERIMENT = "mph-infection-simulation"

try:
    import mlflow as _mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
    _mlflow = None


def _uri() -> str:
    from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
    return GLOBAL.ops.mlflow_uri or _DEFAULT_URI


def _setup_once():
    """Initialize MLflow tracking URI once per process. Idempotent."""
    if not _HAS_MLFLOW:
        return False
    uri = _uri()
    try:
        current = _mlflow.get_tracking_uri()
        if current != uri:
            _mlflow.set_tracking_uri(uri)
    except Exception as e:
        log.debug(f"mlflow setup failed: {e}")
        return False
    return True


@contextlib.contextmanager
def experiment(
    name: str = _DEFAULT_EXPERIMENT,
    *,
    run_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    nested: bool = False,
    **run_kwargs: Any,
) -> Iterator[Optional[Any]]:
    """Context manager that scopes all logging calls into one MLflow run.

    Safe to nest via ``nested=True``. Yields the mlflow Run object or None
    if MLflow is unavailable (in which case inner log_* calls are no-ops).
    """
    if not _setup_once():
        yield None
        return

    try:
        exp = _mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = _mlflow.create_experiment(name)
        else:
            exp_id = exp.experiment_id
    except Exception as e:
        log.debug(f"mlflow experiment lookup failed: {e}")
        yield None
        return

    try:
        with _mlflow.start_run(
            experiment_id=exp_id,
            run_name=run_name,
            tags=tags,
            nested=nested,
            **run_kwargs,
        ) as run:
            yield run
    except Exception as e:
        log.warning(f"mlflow start_run failed ({e}); continuing without tracking")
        yield None


def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
    """Log a scalar metric. No-op if MLflow unavailable or no active run."""
    if not _HAS_MLFLOW:
        return
    try:
        _mlflow.log_metric(key, float(value), step=step)
    except Exception as e:
        log.debug(f"mlflow.log_metric({key}) skipped: {e}")


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """Log a dict of metrics in one call. NaN/Inf silently dropped."""
    if not _HAS_MLFLOW:
        return
    try:
        safe = {
            k: float(v) for k, v in metrics.items()
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
        }
        if safe:
            _mlflow.log_metrics(safe, step=step)
    except Exception as e:
        log.debug(f"mlflow.log_metrics skipped: {e}")


def log_params(params: dict[str, Any]) -> None:
    """Log hyperparameters / scenario settings."""
    if not _HAS_MLFLOW:
        return
    try:
        # MLflow params must be str — coerce
        safe = {k: str(v)[:500] for k, v in params.items()}
        _mlflow.log_params(safe)
    except Exception as e:
        log.debug(f"mlflow.log_params skipped: {e}")


def log_artifact(path: str | Path, artifact_path: Optional[str] = None) -> None:
    """Log a file artifact (JSON, PNG, etc.)."""
    if not _HAS_MLFLOW:
        return
    p = Path(path)
    if not p.exists():
        log.debug(f"mlflow.log_artifact: {p} does not exist")
        return
    try:
        _mlflow.log_artifact(str(p), artifact_path=artifact_path)
    except Exception as e:
        log.debug(f"mlflow.log_artifact({p.name}) skipped: {e}")


def set_tag(key: str, value: str) -> None:
    """Set a tag on the current run."""
    if not _HAS_MLFLOW:
        return
    try:
        _mlflow.set_tag(key, value)
    except Exception as e:
        log.debug(f"mlflow.set_tag({key}) skipped: {e}")


def tracking_info() -> dict:
    """Diagnostic: return current MLflow configuration."""
    return {
        "mlflow_available": _HAS_MLFLOW,
        "mlflow_version": getattr(_mlflow, "__version__", None) if _HAS_MLFLOW else None,
        "tracking_uri": _uri(),
        "backend_path": _uri().replace("sqlite:///", ""),
    }
