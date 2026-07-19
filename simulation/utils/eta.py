"""
Pre-flight ETA + warnings banner — set expectations BEFORE long work starts.
=============================================================================

Why
---
An R4 walk-forward CV or R9 per-model optimisation can run for
hours. Without a heads-up the user can't tell whether the process is alive
and whether they should background it. This module emits:

  • A *pre-flight banner* once per ``simulation <cmd>`` invocation, after
    argparse and before any heavy work, listing:
        – ETA (low / typical / high) inferred from the command + flags
        – Output directories that will be written
        – Hardware / data warnings (RAM, disk, DB freshness)
        – Tip for backgrounding when ETA > 30 min

  • A *phase banner* at each R/P phase start, with the per-phase ETA and
    per-phase progress.  Bound through ``runner.py``.

The estimates are deliberately wide ranges (low / high) — we'd rather be
honest than precise. They are calibrated against actual measured wall-time
on the project (see ``simulation/results/eval_logs/INDEX.csv``).

CLI integration
---------------
``simulation/__main__.py`` calls ``print_preflight_banner(args)`` right
after argparse parses successfully. ``runner.py`` calls
``print_phase_banner(phase_n, total_phases, name)`` from inside its phase
loop. Doctor recommendations now include the ETA tag too (handled in
``doctor.py``).
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# ETA value object
# ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ETA:
    """Wall-time estimate. low/typical/high are seconds."""
    low: int
    typical: int
    high: int
    note: str = ""

    @property
    def human(self) -> str:
        """Best human-readable string for the typical case."""
        return f"{_fmt_seconds(self.low)} – {_fmt_seconds(self.high)}"

    @property
    def human_typical(self) -> str:
        return _fmt_seconds(self.typical)

    @property
    def is_long(self) -> bool:
        return self.typical >= 30 * 60   # ≥ 30 min suggests background it

    @property
    def is_quick(self) -> bool:
        return self.high < 30            # < 30s is "instant"


def _fmt_seconds(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h{m:02d}m" if m else f"{h}h"


# ─────────────────────────────────────────────────────────────────
# Command-level ETA registry
#   These are calibrated for: M-series Mac, n_jobs ≤ 4, Optuna trials
#   range 20-100, 13-50 model registry. CPU-only baseline; MPS shaves
#   ~10-20% off DL-heavy stages.
# ─────────────────────────────────────────────────────────────────
COMMAND_ETAS: dict[str, ETA] = {
    # Instant / interactive
    "db-init":          ETA(1, 2, 5),
    "db-status":        ETA(1, 2, 5),
    "db-optimize":      ETA(5, 30, 120, "VACUUM is the slow part"),
    "doctor":           ETA(2, 5, 15),
    "verify-audit":     ETA(2, 8, 30),
    "freeze-paper-primary": ETA(5, 15, 60),
    "list-models":      ETA(1, 2, 5),

    # IO-bound
    "extract-pdf":      ETA(60, 120, 300, "first run only; idempotent"),
    "import-external":  ETA(10, 60, 180, "scans simulation/data/external/"),
    "maintain":         ETA(5, 30, 120),

    # Network-bound (depends on KDCA / KMA / KOSIS API responsiveness)
    "collect:default":  ETA(120, 300, 900, "incremental; needs internet"),
    "collect:backfill": ETA(900, 1800, 3600, "365-day backfill on cold DB"),

    # Large pipeline runs (CPU-bound)
    "bootstrap":        ETA(180, 300, 600, "schema + import + PDF + verify"),
    "train:dry-run":    ETA(5, 10, 30),
    "train:lite":       ETA(1800, 3000, 5400, "fewer features, fewer epochs"),
    "train:full":       ETA(3600, 7200, 14400, "phases 1-13"),
    "train:full+optim": ETA(7200, 10800, 18000, "+ per-model-optimize (R9)"),
    "train-all":        ETA(14400, 28800, 64800, "every scenario, sequential"),
    "run-all":          ETA(18000, 36000, 72000, "bootstrap + collect + train-all"),
    "orchestrate":      ETA(30, 120, 600),

    # Inference
    "predict-real":     ETA(20, 60, 180, "build features + load every champion"),

    # Simulation
    "sim":              ETA(30, 120, 600, "deterministic ODE; depends on --days"),
    "mcp-server":       ETA(0, 0, 0, "long-running stdio loop; no fixed ETA"),
}


# ─────────────────────────────────────────────────────────────────
# Phase-level ETA (inside `train --scenario full`)
# ─────────────────────────────────────────────────────────────────
# LABEL = R/P 라벨 → (의미이름, ETA). SSOT = simulation.pipeline.phases. 배너 "▶ <의미이름>".
# 사용자 2026-06-08 "번호를 다 없애" → 디스플레이 de-number. 빔(옛 2-3)·은퇴(옛 ar_correction)는
# dispatch·매핑·이 표에서 완전 제거. dict 키 = R/P 라벨(숫자 폐기).
# test_phase_label_sync.py 가 라벨↔phases 레지스트리 동기화를 강제(드리프트 차단).
# Keyed by R/P label — phase numbers removed (SSOT = simulation.pipeline.phases).
PHASE_ETAS: dict[str, tuple[str, ETA]] = {
    "R1":   ("data",               ETA(5, 60, 300, "data + FE; cache hit if --no-cache absent")),
    "R2":   ("baseline",           ETA(180, 1800, 3600, "BASIC anchor; DL Optuna lives here")),
    "R3":   ("external",           ETA(180, 1800, 3600, "external optuna; inert if optuna.mode=none")),
    "R4":   ("wfcv",               ETA(900, 1800, 3600, "walk-forward CV — champion 비교 패널")),
    "R5":   ("diagnostics",        ETA(5, 30, 120, "residual diagnostics")),
    "R6":   ("dm_test",            ETA(5, 30, 60, "Diebold-Mariano pairwise (regime-split)")),
    "R7":   ("intervals",          ETA(15, 60, 180, "PI + conformal")),
    "R8":   ("scoring",            ETA(15, 60, 180, "composite diagnostic — champion 아님")),
    "P1":   ("real_forecaster",    ETA(120, 600, 1800, "operational forecast on real, rolling-origin")),
    "R9":   ("per_model_optimize", ETA(900, 1800, 5400, "transform×scaler×model — heaviest")),
    "R10":  ("per_model_eval",     ETA(60, 300, 900, "129 metrics per model")),
    "R11":  ("shap",               ETA(60, 600, 1800, "SHAP + XAI; depth ∝ n_features")),
    "R12":  ("comprehensive_eval", ETA(30, 120, 300, "aggregator + figures")),
    "Pinf": ("inference",          ETA(15, 60, 180, "optional — 별도 CLI")),
    "Pov":  ("overseas",           ETA(30, 120, 600, "optional — 별도 CLI")),
}


# ─────────────────────────────────────────────────────────────────
# Resolve which ETA bucket a CLI invocation falls into
# ─────────────────────────────────────────────────────────────────
def _resolve_command_eta(args) -> tuple[str, ETA]:
    cmd = getattr(args, "command", None)
    if cmd is None:
        return ("?", ETA(0, 0, 0))

    # Drill into multi-mode commands
    if cmd == "train":
        if getattr(args, "dry_run", False):
            return ("train --dry-run", COMMAND_ETAS["train:dry-run"])
        scenario = getattr(args, "scenario", None) or "full"
        per_optim = bool(getattr(args, "per_model_optimize", False))
        if scenario == "lite":
            return ("train --scenario lite", COMMAND_ETAS["train:lite"])
        if per_optim:
            return ("train --scenario full --per-model-optimize",
                    COMMAND_ETAS["train:full+optim"])
        return ("train --scenario " + scenario, COMMAND_ETAS["train:full"])
    if cmd == "collect":
        bd = getattr(args, "backfill_days", None)
        if bd and bd >= 90:
            return (f"collect --backfill-days {bd}",
                    COMMAND_ETAS["collect:backfill"])
        return ("collect (incremental)", COMMAND_ETAS["collect:default"])

    # Direct lookup
    if cmd in COMMAND_ETAS:
        return (cmd, COMMAND_ETAS[cmd])

    return (cmd, ETA(0, 0, 0))


# ─────────────────────────────────────────────────────────────────
# Output dirs the command will write to (best-effort)
# ─────────────────────────────────────────────────────────────────
def _expected_outputs(args) -> list[str]:
    cmd = getattr(args, "command", None)
    out: list[str] = []
    if cmd == "train":
        out.append("simulation/results/")
        out.append("simulation/results/checkpoints/")
        if getattr(args, "per_model_optimize", False):
            out.append("simulation/results/per_model_optimal/")
            out.append("models/<name>.pt + champion_log.json")
        out.append("simulation/logs/training_log_*.log")
    elif cmd == "predict-real":
        out.append("simulation/results/inference/<ts>/")
    elif cmd == "collect":
        out.append("simulation/data/db/epi_real_seoul.db (writes)")
        out.append("simulation/data/collected/")
    elif cmd == "bootstrap":
        out.append("simulation/data/db/epi_real_seoul.db (creates)")
    elif cmd == "doctor":
        out.append("(nothing; --save-report optional)")
    elif cmd == "sim":
        out.append("simulation/results/sim_state_*.npz (with --out)")
    return out


# ─────────────────────────────────────────────────────────────────
# Quick environment warnings (subset of doctor.py, fast)
# ─────────────────────────────────────────────────────────────────
def _quick_warnings(args, eta: ETA) -> list[str]:
    warns: list[str] = []
    cmd = getattr(args, "command", None)

    # Skip warnings for instant commands
    if eta.is_quick:
        return warns

    # RAM
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        if avail_gb < 4 and eta.typical >= 600:
            warns.append(f"low free RAM ({avail_gb:.1f} GB) — close other apps "
                          f"or use --scenario lite")
    except ImportError:
        pass

    # Disk
    try:
        free_gb = shutil.disk_usage(os.getcwd()).free / 1e9
        if free_gb < 5:
            warns.append(f"low disk space ({free_gb:.1f} GB free) — checkpoints "
                          f"and Optuna SQLite can grow large")
    except Exception:
        pass

    # DB freshness for training
    if cmd in ("train", "train-all", "predict-real"):
        try:
            from simulation.database.config import DB_PATH
            db_path = Path(DB_PATH)
            if db_path.exists():
                age_days = (time.time() - db_path.stat().st_mtime) / 86400
                if age_days > 14:
                    warns.append(f"DB last updated {age_days:.0f} days ago — "
                                  f"`simulation collect` recommended first")
            else:
                warns.append("DB not found — run `simulation bootstrap` first")
        except Exception:
            pass

    # No internet for collect
    if cmd == "collect":
        warns.append("collect needs internet (KDCA/KMA/KOSIS APIs)")

    # Lonely background-run hint
    if eta.is_long and sys.stdout.isatty():
        warns.append(f"long run (~{eta.human_typical}); recommend running in "
                      f"background — see tip below")

    # Active training already running?
    if cmd in ("train", "train-all", "run-all"):
        try:
            import subprocess as _sp
            ps = _sp.run(["ps", "-axo", "pid,command"],
                          capture_output=True, text=True, timeout=2)
            other = [
                line for line in ps.stdout.splitlines()
                if "simulation" in line and "train" in line
                and str(os.getpid()) not in line
                and "grep" not in line
            ]
            if other:
                warns.append(f"another `simulation train` process is already "
                              f"running ({len(other)} match(es)) — they will "
                              f"compete for CPU/RAM")
        except Exception:
            pass

    return warns


# ─────────────────────────────────────────────────────────────────
# Pre-flight banner — print before dispatch
# ─────────────────────────────────────────────────────────────────
_BANNER_PRINTED = False  # don't double-print on imports


def print_preflight_banner(args, *, force: bool = False) -> Optional[ETA]:
    """Pretty banner with ETA + outputs + warnings. Returns the resolved ETA
    so the caller can short-circuit if needed. Idempotent within a process."""
    global _BANNER_PRINTED
    if _BANNER_PRINTED and not force:
        return None
    _BANNER_PRINTED = True

    cmd = getattr(args, "command", None)
    if cmd is None:
        return None

    label, eta = _resolve_command_eta(args)
    outputs = _expected_outputs(args)
    warns = _quick_warnings(args, eta)

    # Don't print for deeply trivial commands
    if eta.is_quick and not warns:
        return eta

    bar = "─" * 70
    print()
    print(f"┌{bar}┐")
    print(f"│  ▶ {label}".ljust(72) + "│")
    if eta.typical > 0:
        eta_str = f"  ⏱ ETA: {eta.human}"
        if eta.note:
            eta_str += f"  ({eta.note})"
        print(f"│{eta_str}".ljust(72) + "│")
    if outputs:
        print(f"│  📁 Output:".ljust(72) + "│")
        for o in outputs:
            print(f"│       • {o}".ljust(72) + "│")
    if warns:
        print(f"│{bar}│")
        print(f"│  ⚠️  Warnings ({len(warns)}):".ljust(72) + "│")
        for w in warns:
            wrapped = _wrap(w, 60)
            for i, line in enumerate(wrapped):
                prefix = "       • " if i == 0 else "         "
                print(f"│{prefix}{line}".ljust(72) + "│")
    if eta.is_long:
        print(f"│{bar}│")
        print(f"│  💡 Tip: long run — for background execution use:".ljust(72) + "│")
        print(f"│        nohup .venv/bin/python -m simulation {cmd} ... &".ljust(72) + "│")
        print(f"│        tail -f simulation/logs/{cmd}_*.log".ljust(72) + "│")
    print(f"└{bar}┘")
    print()
    return eta


# ─────────────────────────────────────────────────────────────────
# Per-phase banner (called from runner.py at each phase boundary)
# ─────────────────────────────────────────────────────────────────
def print_phase_banner(phase_label: str, *, total: int = 16,
                        prev_elapsed_sec: Optional[float] = None) -> None:
    """Compact one-line banner at each R/P phase start (keyed by R/P label).
    ``prev_elapsed_sec`` is the wall-time consumed by the *preceding* phase,
    if known — printed for context."""
    info = PHASE_ETAS.get(phase_label)
    if info is None:
        return
    name, eta = info
    bar = "─" * 70

    print()
    print(f"  ╭{bar}╮")
    # 번호 없이 의미이름만 (사용자 2026-06-08 "번호를 다 없애" — 디스플레이 de-number).
    head = (f"  │ ▶ {name}")
    print(head.ljust(74) + "│")
    if eta.typical > 0:
        eta_str = f"  │   ⏱ phase ETA: {eta.human}"
        if eta.note:
            eta_str += f"  ({eta.note})"
        print(eta_str.ljust(74) + "│")
    if prev_elapsed_sec is not None and prev_elapsed_sec >= 1:
        print((f"  │   ⌛ previous phase: {_fmt_seconds(int(prev_elapsed_sec))}")
              .ljust(74) + "│")
    print(f"  ╰{bar}╯")


def get_command_eta(args) -> ETA:
    """Public lookup — used by doctor.py to attach ETA tags to recommendations."""
    _, eta = _resolve_command_eta(args)
    return eta


def get_command_eta_by_label(label: str) -> Optional[ETA]:
    """Lookup by registry key, e.g. ``train:full+optim``."""
    return COMMAND_ETAS.get(label)


def _wrap(s: str, width: int) -> list[str]:
    """Tiny word-wrap so a long warning doesn't overflow the banner box."""
    out: list[str] = []
    line = ""
    for w in s.split():
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


__all__ = [
    "ETA", "COMMAND_ETAS", "PHASE_ETAS",
    "print_preflight_banner", "print_phase_banner",
    "get_command_eta", "get_command_eta_by_label",
]
