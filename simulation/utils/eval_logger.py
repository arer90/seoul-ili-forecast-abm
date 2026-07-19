"""
Structured evaluation logger (append-only JSONL).
==================================================

Every metric computed by every phase, for every model, gets a single
line in a per-run JSONL file. This produces:

  • Reproducibility: complete record of what was computed when.
  • Audit trail: re-derive any leaderboard from the JSONL.
  • Streaming: open the file at any time during a run to see progress.
  • Multi-run aggregation: glob all JSONLs to compare configurations.

File layout:

  simulation/results/eval_logs/
    ├── {run_id}.jsonl              ← this run's records (append-only)
    ├── {run_id}_audit.json         ← reproducibility metadata
    └── INDEX.csv                    ← all runs roll-up (run_id × config × score)

Each JSONL line:

    {"ts": "2026-04-25T15:42:30.124", "phase": "phase11",
     "model": "XGBoost", "metric": "wis", "value": 3.124,
     "horizon": 1, "slab": "test", "regime": "post-covid",
     "n": 68, "config_hash": "a1b2..."}

Use:

    from simulation.utils.eval_logger import EvalLogger
    el = EvalLogger.from_config(config)
    el.log(phase="phase11", model="XGBoost", metric="wis", value=3.124,
           horizon=1, slab="test", n=68)
    el.log_phase_start("phase11")
    el.log_phase_end("phase11", elapsed=124.5, status="ok")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterable

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

log = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
           f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _config_hash(config) -> str:
    """Stable 8-char hash of the relevant config fields. Used to identify
    runs sharing a configuration."""
    try:
        relevant = {
            "preset": getattr(config, "preset", "?"),
            "scenario": getattr(config, "scenario", "?"),
            "paper_cutoff_week": getattr(config.split, "paper_cutoff_week", None),
            "weather_mode": getattr(config.split, "real_weather_mode", "?"),
            "covid_mode": getattr(config.split, "covid_inclusion_mode", "?"),
            "ensemble_method": getattr(config.split, "ensemble_method", "?"),
            "conformal_method": getattr(config.split, "real_conformal_method", "?"),
            "horizons": list(getattr(config.split, "real_horizons", (1,))),
        }
        s = json.dumps(relevant, sort_keys=True, default=str)
    except Exception:
        s = str(config)
    return hashlib.sha256(s.encode()).hexdigest()[:8]


class EvalLogger:
    """Append-only JSONL evaluation logger.

    Methods:
      log(phase, model, metric, value, **dims)  → one JSONL line
      log_phase_start(phase, **dims)
      log_phase_end(phase, elapsed, status, **dims)
      log_audit(metadata)                       → companion JSON file
      checkpoint()                              → flush buffered writes
    """

    def __init__(self, run_id: str, log_dir: Optional[Path] = None,
                 config_hash: str = "?"):
        self.run_id = run_id
        self.config_hash = config_hash
        self.log_dir = Path(log_dir) if log_dir else (get_results_dir() / "eval_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"{run_id}.jsonl"
        self.audit_path = self.log_dir / f"{run_id}_audit.json"
        # Open append-mode handle; we let the OS buffer; checkpoint() flushes
        self._fh = self.jsonl_path.open("a", encoding="utf-8")
        self._n_written = 0
        log.info(f"  [EvalLogger] run_id={run_id} → {self.jsonl_path}")

    @classmethod
    def from_config(cls, config) -> "EvalLogger":
        # Build run_id from timestamp + config hash
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ch = _config_hash(config)
        run_id = f"{ts}_{ch}"
        return cls(run_id=run_id, config_hash=ch)

    def _line(self, **fields) -> dict:
        d = {
            "ts": _utcnow_iso(),
            "run_id": self.run_id,
            "config_hash": self.config_hash,
        }
        d.update(fields)
        return d

    def log(self, phase: str, model: str, metric: str,
            value: Any, **dims) -> None:
        """Record one (phase, model, metric, value) tuple."""
        rec = self._line(phase=phase, model=model, metric=metric, value=value)
        rec.update(dims)
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._n_written += 1

    def log_phase_start(self, phase: str, **dims) -> None:
        self.log(phase=phase, model="<phase>", metric="phase_start",
                 value=time.time(), **dims)

    def log_phase_end(self, phase: str, elapsed: float, status: str = "ok",
                      **dims) -> None:
        self.log(phase=phase, model="<phase>", metric="phase_end",
                 value=elapsed, status=status, **dims)

    def log_metrics_dict(self, phase: str, model: str,
                          metrics: dict, **dims) -> None:
        """Convenience: log every key/value in a metrics dict as a separate line."""
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                self.log(phase=phase, model=model, metric=k, value=v, **dims)
            elif isinstance(v, dict) and len(v) <= 20:
                # Flatten one level (e.g., peak_week: {abs_weeks, hit, ...})
                for kk, vv in v.items():
                    if isinstance(vv, (int, float, str, bool)) or vv is None:
                        self.log(phase=phase, model=model,
                                 metric=f"{k}.{kk}", value=vv, **dims)

    def log_audit(self, metadata: dict) -> None:
        """Write companion audit JSON (one-shot per run)."""
        full = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "audit_written_at": _utcnow_iso(),
        }
        full.update(metadata)
        self.audit_path.write_text(json.dumps(full, indent=2, default=str))
        log.info(f"  [EvalLogger] audit → {self.audit_path}")

    def checkpoint(self) -> None:
        """Flush OS buffer so partial reads are consistent."""
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception:
            pass

    def close(self) -> None:
        self.checkpoint()
        try:
            self._fh.close()
        except Exception:
            pass
        log.info(f"  [EvalLogger] closed: {self._n_written} records → {self.jsonl_path}")

    def __enter__(self) -> "EvalLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ─── Roll-up: aggregate JSONL across runs into a CSV index ──────────────

def build_run_index(log_dir: Optional[Path] = None) -> Path:
    """Scan all *.jsonl files and produce INDEX.csv with one row per run.

    Each row: run_id, config_hash, n_records, started_at, ended_at,
              best_model_by_wis, best_wis, etc.
    """
    log_dir = Path(log_dir) if log_dir else (get_results_dir() / "eval_logs")
    runs = []
    for f in sorted(log_dir.glob("*.jsonl")):
        records = []
        try:
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
        if not records:
            continue
        wis = [r for r in records
               if r.get("metric") == "wis" and r.get("slab") == "test"]
        starts = [r for r in records if r.get("metric") == "phase_start"]
        ends = [r for r in records if r.get("metric") == "phase_end"]
        run = {
            "run_id": records[0].get("run_id"),
            "config_hash": records[0].get("config_hash"),
            "n_records": len(records),
            "n_wis_test": len(wis),
            "started_at": (starts[0]["ts"] if starts else records[0]["ts"]),
            "ended_at": (ends[-1]["ts"] if ends else records[-1]["ts"]),
        }
        if wis:
            best = min(wis, key=lambda r: float(r.get("value", float("inf")))
                       if isinstance(r.get("value"), (int, float)) else float("inf"))
            run["best_model"] = best.get("model")
            run["best_wis"] = best.get("value")
        runs.append(run)

    import csv
    out = log_dir / "INDEX.csv"
    if not runs:
        out.write_text("run_id,config_hash,n_records,started_at,ended_at\n")
        return out
    cols = sorted({k for r in runs for k in r})
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow(r)
    return out
