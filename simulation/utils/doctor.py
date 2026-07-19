"""
simulation doctor — end-to-end environment + project diagnostics.
==================================================================

Checks (in order, all non-blocking unless ``--strict``):

  A. System         OS, kernel, arch, Python version, CPU cores, RAM, disk
  B. Accelerator    CUDA (Linux/Win) or MPS (Apple silicon) detection
  C. Packages       numpy<2.2 / polars / sklearn / xgboost / torch / optuna / …
  D. Env vars       Apple OMP guards (libomp fork-safety)
  E. Project layout simulation/data/db/, models/, results/, logs/, .venv
  F. DB ↔ models    sentinel weekly grid, weather rows, commuter shape,
                    REGISTRY size, champion count
  G. Code self-test R1..R12 + P-track pipeline modules + champion_log +
                    model_artifact import
  H. Pipeline-ready Optuna caches, sidecars, in-sample boundary date,
                    feature-engine cache mtime vs DB mtime
  I. Recommendations Hardware-aware tuning hints (n_jobs, epochs, scenario,
                     OMP env vars). With ``--auto`` the safe ones are
                     applied automatically (mkdir, set environ, write hint
                     file).

Output:

  - Human-readable report on stdout (colourised tags ✓ / ⚠ / ✗)
  - Optional JSON manifest at ``--save-report PATH`` (machine-readable)
  - Exit code: 0 if no FAIL, 1 if any FAIL, 2 if invocation error

CLI:
    python -m simulation doctor                # report only
    python -m simulation doctor --auto          # apply safe fixes
    python -m simulation doctor --verbose       # include all OK lines
    python -m simulation doctor --save-report doctor.json
    python -m simulation doctor --strict        # exit 1 on any WARN too
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Result accumulator
# ─────────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    status: str           # "ok" | "warn" | "fail"
    message: str
    section: str = "misc"
    fix_hint: Optional[str] = None
    auto_fixable: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class DoctorReport:
    started_at: str
    elapsed_sec: float = 0.0
    results: list[CheckResult] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
    auto_fixes_applied: list[str] = field(default_factory=list)
    n_ok: int = 0
    n_warn: int = 0
    n_fail: int = 0

    def add(self, r: CheckResult) -> None:
        self.results.append(r)
        if r.status == "ok":
            self.n_ok += 1
        elif r.status == "warn":
            self.n_warn += 1
        else:
            self.n_fail += 1

    def to_json(self) -> dict:
        return {
            "started_at": self.started_at,
            "elapsed_sec": self.elapsed_sec,
            "summary": {"ok": self.n_ok, "warn": self.n_warn, "fail": self.n_fail},
            "results": [
                {
                    "section": r.section,
                    "name":    r.name,
                    "status":  r.status,
                    "message": r.message,
                    "fix_hint": r.fix_hint,
                    "auto_fixable": r.auto_fixable,
                    "extra":   r.extra,
                }
                for r in self.results
            ],
            "recommendations": self.recommendations,
            "auto_fixes_applied": self.auto_fixes_applied,
        }


# ─────────────────────────────────────────────────────────────────
# Section A — System
# ─────────────────────────────────────────────────────────────────
def _check_system(rep: DoctorReport) -> dict:
    """OS, Python, CPU, RAM, disk."""
    info: dict = {}
    info["os"] = f"{platform.system()} {platform.release()}"
    info["arch"] = platform.machine()
    info["python"] = platform.python_version()
    info["python_exe"] = sys.executable

    rep.add(CheckResult("os", "ok",
                        f"{info['os']} ({info['arch']})",
                        section="system"))

    py = sys.version_info
    if py >= (3, 12):
        rep.add(CheckResult("python", "ok",
                            f"Python {info['python']} (>= 3.12 OK)",
                            section="system"))
    elif py >= (3, 10):
        rep.add(CheckResult("python", "warn",
                            f"Python {info['python']} (project validated on 3.12)",
                            section="system",
                            fix_hint="upgrade interpreter to Python 3.12"))
    else:
        rep.add(CheckResult("python", "fail",
                            f"Python {info['python']} below minimum (need >=3.10)",
                            section="system",
                            fix_hint="install Python 3.12 (uv python install 3.12)"))

    # CPU cores
    try:
        import os as _os
        cores_logical = _os.cpu_count() or 1
        info["cpu_cores"] = cores_logical
        rep.add(CheckResult("cpu_cores", "ok",
                            f"{cores_logical} logical cores",
                            section="system"))
    except Exception as e:
        rep.add(CheckResult("cpu_cores", "warn",
                            f"could not detect CPU cores: {e}",
                            section="system"))

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = vm.total / 1e9
        avail_gb = vm.available / 1e9
        info["ram_total_gb"] = round(total_gb, 1)
        info["ram_avail_gb"] = round(avail_gb, 1)
        rep.add(CheckResult("ram_total", "ok",
                            f"Total RAM: {total_gb:.1f} GB",
                            section="system"))
        if avail_gb >= 4.0:
            rep.add(CheckResult("ram_avail", "ok",
                                f"Available: {avail_gb:.1f} GB",
                                section="system"))
        else:
            rep.add(CheckResult("ram_avail", "warn",
                                f"Available RAM low: {avail_gb:.1f} GB",
                                section="system",
                                fix_hint="close memory-heavy apps before "
                                         "training; consider --scenario lite"))
    except ImportError:
        rep.add(CheckResult("ram", "warn",
                            "psutil not installed — RAM check skipped",
                            section="system",
                            fix_hint="uv pip install psutil"))

    # Disk
    try:
        repo_root = Path.cwd()
        usage = shutil.disk_usage(str(repo_root))
        free_gb = usage.free / 1e9
        info["disk_free_gb"] = round(free_gb, 1)
        if free_gb >= 5:
            rep.add(CheckResult("disk_free", "ok",
                                f"Disk free at {repo_root}: {free_gb:.1f} GB",
                                section="system"))
        else:
            rep.add(CheckResult("disk_free", "warn",
                                f"Low disk: {free_gb:.1f} GB free",
                                section="system",
                                fix_hint="free up disk; .venv + DB + results need ~5GB"))
    except Exception as e:
        rep.add(CheckResult("disk_free", "warn",
                            f"disk usage check failed: {e}",
                            section="system"))
    return info


# ─────────────────────────────────────────────────────────────────
# Section B — Accelerator
# ─────────────────────────────────────────────────────────────────
def _check_subprocess_strategy(rep: DoctorReport) -> dict:
    """OS 별 subprocess 격리 전략 표시."""
    info: dict = {}
    try:
        from simulation.models.runner import get_subprocess_strategy_summary
        s = get_subprocess_strategy_summary()
        info = s
        plat = s.get("platform", "?")
        n_override = len(s.get("in_process_override", []))
        notes = s.get("notes", "")
        rep.add(CheckResult("subprocess_strategy", "ok",
                            f"{plat}: {notes} ({n_override} model(s) in-process)",
                            section="env",
                            extra=s))
        if n_override:
            rep.add(CheckResult("subprocess_inprocess_list", "ok",
                                f"in-process forced: "
                                f"{', '.join(s['in_process_override'])}",
                                section="env"))
    except Exception as e:
        rep.add(CheckResult("subprocess_strategy", "warn",
                            f"could not read subprocess strategy: {e}",
                            section="env"))
    return info


def _check_accelerator(rep: DoctorReport, sysinfo: dict) -> dict:
    info: dict = {"cuda": False, "mps": False, "device": "cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["device"] = "cuda"
            try:
                info["cuda_device"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
            except Exception:
                pass
            rep.add(CheckResult("accelerator", "ok",
                                f"CUDA: {info.get('cuda_device','?')} "
                                f"(CUDA {info.get('cuda_version','?')})",
                                section="accelerator"))
        elif (hasattr(torch.backends, "mps")
              and torch.backends.mps.is_available()):
            info["mps"] = True
            info["device"] = "mps"
            rep.add(CheckResult("accelerator", "ok",
                                "Apple MPS available (Metal Performance Shaders)",
                                section="accelerator"))
        else:
            rep.add(CheckResult("accelerator", "warn",
                                "CPU mode (no CUDA, no MPS) — DL models will be slower",
                                section="accelerator",
                                fix_hint="optional; CPU is fine for n=345 weeks"))
    except ImportError:
        rep.add(CheckResult("accelerator", "warn",
                            "PyTorch not installed — DL models will fail",
                            section="accelerator",
                            fix_hint="uv pip install torch"))
    return info


# ─────────────────────────────────────────────────────────────────
# Section C — Packages
# ─────────────────────────────────────────────────────────────────
def _check_packages(rep: DoctorReport) -> dict:
    """Critical packages with version constraints from ENGINEERING_PRINCIPLES.md."""
    info: dict = {}

    # name → (import_name, pip_name, constraint_op, constraint_version, hard?)
    REQUIRED = [
        ("numpy",        "numpy",       "<",  "2.2",  True),  # SHAP/numba compat
        ("polars",       "polars",      ">=", "1.0",  True),
        ("pandas",       "pandas",      ">=", "2.0",  True),
        ("sklearn",      "scikit-learn",">=", "1.3",  True),
        ("scipy",        "scipy",       ">=", "1.10", True),
        ("xgboost",      "xgboost",     ">=", "1.7",  True),
        ("lightgbm",     "lightgbm",    ">=", "3.3",  False),
        ("statsmodels",  "statsmodels", ">=", "0.14", True),
        ("optuna",       "optuna",      ">=", "3.0",  True),
        ("shap",         "shap",        ">=", "0.42", False),
        ("torch",        "torch",       ">=", "2.0",  False),
        ("duckdb",       "duckdb",      ">=", "0.10", False),
        ("psutil",       "psutil",      ">=", "5.9",  False),
    ]

    def _cmp(a: str, op: str, b: str) -> bool:
        from packaging.version import Version
        va, vb = Version(a), Version(b)
        return {">=": va >= vb, "<=": va <= vb,
                ">":  va >  vb, "<":  va <  vb,
                "==": va == vb}.get(op, True)

    for imp, pip_name, op, target, hard in REQUIRED:
        try:
            mod = __import__(imp.replace("-", "_"))
            ver = getattr(mod, "__version__", "?")
            try:
                ok = _cmp(ver, op, target)
            except Exception:
                ok = True
            if ok:
                rep.add(CheckResult(pip_name, "ok",
                                    f"{pip_name} {ver}",
                                    section="packages"))
                info[pip_name] = ver
            else:
                rep.add(CheckResult(pip_name,
                                    "fail" if hard else "warn",
                                    f"{pip_name} {ver} (need {op}{target})",
                                    section="packages",
                                    fix_hint=f"uv pip install '{pip_name}{op}{target}'"))
                info[pip_name] = ver
        except ImportError:
            rep.add(CheckResult(pip_name,
                                "fail" if hard else "warn",
                                f"{pip_name} not installed",
                                section="packages",
                                fix_hint=f"uv pip install {pip_name}",
                                auto_fixable=False))   # don't auto-pip-install
            info[pip_name] = None

    return info


# ─────────────────────────────────────────────────────────────────
# Section D — Environment vars (Apple libomp fork-safety guards)
# ─────────────────────────────────────────────────────────────────
def _check_env_vars(rep: DoctorReport, sysinfo: dict, auto: bool) -> dict:
    info: dict = {}
    is_darwin = sys.platform == "darwin"
    if not is_darwin:
        rep.add(CheckResult("omp_env", "ok",
                            "non-darwin: OMP guards not required",
                            section="env"))
        return info

    # The exact set enforced in __main__.py top-of-file
    REQUIRED_DARWIN = {
        "KMP_DUPLICATE_LIB_OK":      "TRUE",
        "OMP_NUM_THREADS":           "1",
        "OPENBLAS_NUM_THREADS":      "1",
        "MKL_NUM_THREADS":           "1",
        "VECLIB_MAXIMUM_THREADS":    "1",
        "XGBOOST_NUM_THREADS":       "1",
        "NUMEXPR_NUM_THREADS":       "1",
    }
    missing: list[str] = []
    for k, v in REQUIRED_DARWIN.items():
        cur = os.environ.get(k)
        info[k] = cur
        if cur is None:
            missing.append(k)
            if auto:
                os.environ[k] = v
                rep.auto_fixes_applied.append(f"set ${k}={v} (process-local)")

    if not missing:
        rep.add(CheckResult("omp_env", "ok",
                            "Apple libomp guards present "
                            "(KMP_DUPLICATE_LIB_OK + 6 thread caps)",
                            section="env"))
    else:
        status = "ok" if auto else "warn"
        msg = (f"applied {len(missing)} OMP guards in-process"
               if auto else
               f"{len(missing)} env var(s) missing: {', '.join(missing)}")
        rep.add(CheckResult("omp_env", status, msg,
                            section="env",
                            auto_fixable=True,
                            fix_hint=("simulation/__main__.py already sets these "
                                       "at runtime; missing only matters for "
                                       "subprocesses that bypass __main__.")))
    return info


# ─────────────────────────────────────────────────────────────────
# Section E — Project layout
# ─────────────────────────────────────────────────────────────────
def _check_project(rep: DoctorReport, auto: bool) -> dict:
    info: dict = {}

    repo_root = Path.cwd()
    info["repo_root"] = str(repo_root)

    REQUIRED_DIRS = [
        repo_root / "simulation",
        repo_root / "simulation" / "data",
        repo_root / "simulation" / "data" / "db",
        repo_root / "simulation" / "results",
        repo_root / "simulation" / "logs",
        repo_root / "models",
    ]
    for d in REQUIRED_DIRS:
        if d.exists():
            rep.add(CheckResult(f"dir:{d.name}", "ok",
                                f"{d.relative_to(repo_root)}",
                                section="project"))
        else:
            if auto:
                try:
                    d.mkdir(parents=True, exist_ok=True)
                    rep.auto_fixes_applied.append(f"mkdir -p {d}")
                    rep.add(CheckResult(f"dir:{d.name}", "ok",
                                        f"{d.relative_to(repo_root)} (auto-created)",
                                        section="project"))
                except Exception as e:
                    rep.add(CheckResult(f"dir:{d.name}", "fail",
                                        f"could not create: {e}",
                                        section="project"))
            else:
                rep.add(CheckResult(f"dir:{d.name}", "warn",
                                    f"missing: {d.relative_to(repo_root)}",
                                    section="project",
                                    auto_fixable=True,
                                    fix_hint=f"mkdir -p {d}"))

    # Writability
    for label, path in [("results", repo_root / "simulation" / "results"),
                         ("logs",    repo_root / "simulation" / "logs")]:
        try:
            test = path / f".doctor_test_{int(time.time())}.tmp"
            test.write_text("ok")
            test.unlink()
            rep.add(CheckResult(f"writable:{label}", "ok",
                                f"{label}/ writable",
                                section="project"))
        except Exception as e:
            rep.add(CheckResult(f"writable:{label}", "fail",
                                f"{label}/ NOT writable: {e}",
                                section="project"))

    # .venv
    venv_paths = [
        repo_root / ".venv" / "bin" / "python3",       # POSIX
        repo_root / ".venv" / "Scripts" / "python.exe",  # Windows
    ]
    venv = next((p for p in venv_paths if p.exists()), None)
    if venv is not None:
        info["venv_python"] = str(venv)
        rep.add(CheckResult("venv", "ok",
                            f".venv detected: {venv.relative_to(repo_root)}",
                            section="project"))
    else:
        rep.add(CheckResult("venv", "warn",
                            ".venv not found at standard path",
                            section="project",
                            fix_hint="uv venv && uv pip install -e ."))
    return info


# ─────────────────────────────────────────────────────────────────
# Section F — DB ↔ models
# ─────────────────────────────────────────────────────────────────
def _check_db_and_models(rep: DoctorReport, auto: bool) -> dict:
    info: dict = {}

    try:
        from simulation.database.config import DB_PATH
        db_path = Path(DB_PATH)
    except Exception as e:
        rep.add(CheckResult("db_path", "fail",
                            f"could not import DB_PATH: {e}",
                            section="db"))
        return info

    info["db_path"] = str(db_path)
    if not db_path.exists():
        rep.add(CheckResult("db_file", "fail",
                            f"DB not found: {db_path}",
                            section="db",
                            fix_hint=".venv/bin/python -m simulation bootstrap"))
        return info

    size_mb = db_path.stat().st_size / 1e6
    age_days = (time.time() - db_path.stat().st_mtime) / 86400
    info["db_size_mb"] = round(size_mb, 1)
    info["db_age_days"] = round(age_days, 1)
    rep.add(CheckResult("db_file", "ok",
                        f"{db_path.name} ({size_mb:.1f} MB, "
                        f"mtime {age_days:.1f} d ago)",
                        section="db"))

    # Quick read
    try:
        from simulation.database import safe_connect
        conn = safe_connect(str(db_path))
        cur = conn.cursor()
        n_tables = cur.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        info["n_tables"] = n_tables
        rep.add(CheckResult("db_tables", "ok",
                            f"{n_tables} tables",
                            section="db"))

        # Critical tables for the train pipeline
        for tbl in ["sentinel_influenza", "weather_historical",
                     "commuter_matrix", "weekly_disease"]:
            try:
                n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                info[f"rows:{tbl}"] = n
                if n > 0:
                    rep.add(CheckResult(f"table:{tbl}", "ok",
                                        f"{tbl}: {n:,} rows",
                                        section="db"))
                else:
                    rep.add(CheckResult(f"table:{tbl}", "warn",
                                        f"{tbl}: empty",
                                        section="db",
                                        fix_hint=".venv/bin/python -m simulation collect"))
            except Exception as e:
                rep.add(CheckResult(f"table:{tbl}", "warn",
                                    f"{tbl}: {e}",
                                    section="db"))
        conn.close()
    except Exception as e:
        rep.add(CheckResult("db_read", "fail",
                            f"DB read failed: {e}",
                            section="db"))

    # DB freshness
    if age_days > 14:
        rep.add(CheckResult("db_freshness", "warn",
                            f"DB last updated {age_days:.0f} days ago",
                            section="db",
                            fix_hint=".venv/bin/python -m simulation collect "
                                     "--groups all"))
    else:
        rep.add(CheckResult("db_freshness", "ok",
                            f"DB fresh ({age_days:.1f} days)",
                            section="db"))
    return info


# ─────────────────────────────────────────────────────────────────
# Section G — Code self-test
# ─────────────────────────────────────────────────────────────────
def _check_code(rep: DoctorReport) -> dict:
    info: dict = {"phases_ok": [], "phases_fail": []}

    PHASES = [
        ("phase1_data",                "simulation.pipeline.data"),
        ("phase4_baseline",            "simulation.pipeline.baseline"),
        ("phase5_external",            "simulation.pipeline.external"),
        ("phase7_diagnostics",         "simulation.pipeline.diagnostics"),
        ("phase9_dm_test",             "simulation.pipeline.dm_test"),
        ("phase10_intervals",           "simulation.pipeline.intervals"),
        ("phase6_wfcv",                "simulation.pipeline.wfcv"),
        ("phase15_xai",         "simulation.pipeline.xai"),
        ("phase11_scoring",             "simulation.pipeline.scoring"),
        ("real_eval",          "simulation.pipeline.real_eval"),
        ("per_model_eval",     "simulation.pipeline.per_model_eval"),
        ("phase13_per_model_optimize", "simulation.pipeline.per_model_optimize"),
        ("comprehensive_eval", "simulation.pipeline.comprehensive_eval"),
        ("inference",          "simulation.pipeline.inference"),
        ("champion_log",               "simulation.utils.champion_log"),
        ("model_artifact",             "simulation.utils.model_artifact"),
        ("eval_logger",                "simulation.utils.eval_logger"),
        ("audit_log",                  "simulation.utils.audit_log"),
    ]
    import importlib
    for short, full in PHASES:
        try:
            importlib.import_module(full)
            info["phases_ok"].append(short)
            rep.add(CheckResult(short, "ok",
                                "import OK",
                                section="code"))
        except Exception as e:
            info["phases_fail"].append((short, str(e)))
            rep.add(CheckResult(short, "fail",
                                f"import FAIL: {e}",
                                section="code"))
    # Functional artifact roundtrip (tiny synthetic regression)
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from simulation.pipeline.per_model_optimize import (
            _refit_and_predict_test)
        from simulation.utils.model_artifact import (
            make_artifact, load_artifact)
        np.random.seed(0)
        Xtr = np.random.randn(40, 5); ytr = np.random.rand(40) + 1
        Xte = np.random.randn(8, 5);  yte = np.random.rand(8) + 1
        res = _refit_and_predict_test(
            lambda: Ridge(alpha=1.0),
            transform_name="log1p", scaler_name="robust",
            X_train_pool=Xtr, y_train_pool=ytr,
            X_test=Xte, y_test=yte,
            return_fitted_model=True)
        if "error" in res:
            rep.add(CheckResult("artifact_roundtrip", "warn",
                                f"refit error: {res['error']}",
                                section="code"))
        else:
            state = res.pop("_artifact_state", {})
            art = make_artifact(model=res.pop("_fitted_model"),
                                 transform_name=state["transform_name"],
                                 transform_inv_obj=state["transform_inv_obj"],
                                 fitted_scaler=state["fitted_scaler"],
                                 feature_indices=state["feature_indices"])
            yp = art.predict(Xte)
            yp0 = np.asarray(res["predictions"])
            diff = float(np.max(np.abs(yp - yp0)))
            if diff < 1e-9:
                rep.add(CheckResult("artifact_roundtrip", "ok",
                                    f"ChampionArtifact.predict ↔ refit "
                                    f"diff = {diff:.2e}",
                                    section="code"))
            else:
                rep.add(CheckResult("artifact_roundtrip", "warn",
                                    f"diff = {diff:.2e} > 1e-9",
                                    section="code"))
    except Exception as e:
        rep.add(CheckResult("artifact_roundtrip", "warn",
                            f"smoke test failed: {e}",
                            section="code"))
    return info


# ─────────────────────────────────────────────────────────────────
# Section H — Pipeline-ready (caches, sidecars, champions)
# ─────────────────────────────────────────────────────────────────
def _check_pipeline_ready(rep: DoctorReport) -> dict:
    info: dict = {}

    repo_root = Path.cwd()

    # Champion log
    log_path = repo_root / "models" / "champion_log.json"
    if log_path.exists():
        try:
            j = json.loads(log_path.read_text())
            n_champs = len(j)
            info["n_champions"] = n_champs
            artifact_count = sum(
                1 for k, rec in j.items()
                if isinstance(rec, dict)
                and (rec.get("current") or {}).get("config", {}).get("artifact")
                    == "ChampionArtifact"
            )
            info["n_artifact_champions"] = artifact_count
            if n_champs > 0:
                rep.add(CheckResult("champions", "ok",
                                    f"{n_champs} champion(s) "
                                    f"({artifact_count} artifact-format, "
                                    f"{n_champs - artifact_count} legacy)",
                                    section="pipeline"))
                if artifact_count < n_champs:
                    rep.add(CheckResult("champions_legacy", "warn",
                                        f"{n_champs - artifact_count} legacy "
                                        f"champions need R9 per_model_optimize re-fit "
                                        f"to upgrade to ChampionArtifact",
                                        section="pipeline",
                                        fix_hint=".venv/bin/python -m simulation "
                                                  "train --per-model-optimize"))
            else:
                rep.add(CheckResult("champions", "warn",
                                    "champion_log.json empty",
                                    section="pipeline",
                                    fix_hint=".venv/bin/python -m simulation "
                                              "train --per-model-optimize"))
        except Exception as e:
            rep.add(CheckResult("champions", "warn",
                                f"champion_log.json unreadable: {e}",
                                section="pipeline"))
    else:
        rep.add(CheckResult("champions", "warn",
                            "no champion_log.json yet",
                            section="pipeline",
                            fix_hint=".venv/bin/python -m simulation train "
                                      "--per-model-optimize"))

    # Optuna caches
    opt_dir = repo_root / "simulation" / "results" / "optuna_per_model"
    if opt_dir.exists():
        n = sum(1 for _ in opt_dir.glob("optuna_*.json"))
        info["optuna_files"] = n
        if n > 0:
            rep.add(CheckResult("optuna_cache", "ok",
                                f"{n} per-model Optuna study files",
                                section="pipeline"))
        else:
            rep.add(CheckResult("optuna_cache", "warn",
                                "Optuna cache dir empty",
                                section="pipeline"))
    else:
        rep.add(CheckResult("optuna_cache", "warn",
                            "no optuna_per_model/ yet",
                            section="pipeline",
                            fix_hint="will be auto-created on first --optuna-mode "
                                      "all run"))

    # Feature engine cache
    fe_cache = repo_root / "simulation" / "results" / "feature_cache.parquet"
    if fe_cache.exists():
        cache_age = (time.time() - fe_cache.stat().st_mtime) / 86400
        info["fe_cache_age_days"] = round(cache_age, 1)
        try:
            from simulation.database.config import DB_PATH
            db_age = (time.time() - Path(DB_PATH).stat().st_mtime) / 86400
            info["db_age_days"] = round(db_age, 1)
            if cache_age < db_age:
                rep.add(CheckResult("fe_cache_freshness", "warn",
                                    f"feature_cache older than DB "
                                    f"({cache_age:.1f}d cache vs {db_age:.1f}d DB) "
                                    f"— feature_engine will reuse stale cache",
                                    section="pipeline",
                                    fix_hint="add --no-cache to next train run"))
            else:
                rep.add(CheckResult("fe_cache_freshness", "ok",
                                    f"feature_cache fresh ({cache_age:.1f} d)",
                                    section="pipeline"))
        except Exception:
            rep.add(CheckResult("fe_cache", "ok",
                                f"feature_cache present ({cache_age:.1f} d old)",
                                section="pipeline"))
    return info


# ─────────────────────────────────────────────────────────────────
# Section I — Recommendations (hardware-aware)
# ─────────────────────────────────────────────────────────────────
def _build_recommendations(rep: DoctorReport, sysinfo: dict, accel: dict,
                            db_info: dict, pipeline_info: dict) -> None:
    """Emit ranked tuning hints based on collected info."""
    recs: list[dict] = []

    # Cores → n_jobs
    cores = sysinfo.get("cpu_cores") or 1
    if cores <= 2:
        recs.append({
            "priority": "high", "category": "performance",
            "issue": f"only {cores} CPU core(s) detected",
            "action": "use --n-jobs 1 (already enforced by config)",
            "cli": None,
        })
    elif cores <= 4:
        recs.append({
            "priority": "med", "category": "performance",
            "issue": f"{cores} cores — leave 1 free for OS",
            "action": f"max recommended n_jobs = {cores - 1}",
            "cli": None,
        })

    # RAM → scenario / batch
    ram = sysinfo.get("ram_total_gb") or 8.0
    if ram < 8:
        recs.append({
            "priority": "high", "category": "performance",
            "issue": f"low RAM ({ram} GB) — full scenario may OOM",
            "action": "use --scenario lite or --models <subset>",
            "cli": ".venv/bin/python -m simulation train --scenario lite",
        })
    elif ram < 16:
        recs.append({
            "priority": "low", "category": "performance",
            "issue": f"{ram} GB RAM — full pipeline runs but tight",
            "action": "consider closing browser/IDE during run",
            "cli": None,
        })

    # Apple silicon ARM
    if sysinfo.get("arch") == "arm64":
        recs.append({
            "priority": "med", "category": "platform",
            "issue": "Apple Silicon detected",
            "action": "OMP guards in __main__.py prevent libomp fork crashes "
                      "(KMP_DUPLICATE_LIB_OK + thread caps); already wired",
            "cli": None,
        })

    # MPS / CUDA / CPU
    if accel.get("device") == "cpu":
        recs.append({
            "priority": "low", "category": "performance",
            "issue": "no GPU/MPS — DL models will be slow",
            "action": "n=345 weeks small enough that CPU is fine; or skip DL "
                      "with --models <non-DL list>",
            "cli": None,
        })

    # DB freshness
    age_days = db_info.get("db_age_days")
    if age_days and age_days > 14:
        recs.append({
            "priority": "high", "category": "data",
            "issue": f"DB last updated {age_days} days ago",
            "action": "refresh from KDCA/KMA before training",
            "cli": ".venv/bin/python -m simulation collect --groups all",
        })

    # Champions empty → first-time hint
    if pipeline_info.get("n_champions", 0) == 0:
        recs.append({
            "priority": "med", "category": "deployment",
            "issue": "no champion .pt artifacts yet",
            "action": "run R9 per_model_optimize to populate models/<name>.pt with ChampionArtifact "
                      "bundles, then `predict-real` for new-data inference",
            "cli": ".venv/bin/python -m simulation train --per-model-optimize",
        })

    # Legacy bare-model .pt
    if (pipeline_info.get("n_artifact_champions", 0)
            < pipeline_info.get("n_champions", 0)):
        n_legacy = (pipeline_info.get("n_champions", 0)
                    - pipeline_info.get("n_artifact_champions", 0))
        recs.append({
            "priority": "high", "category": "deployment",
            "issue": f"{n_legacy} champion(s) are legacy bare-model .pt — "
                      f"missing fitted scaler & transform_state",
            "action": "inference (Pinf) will use identity transform on these "
                      "(WARN logged); re-run R9 per_model_optimize to upgrade",
            "cli": ".venv/bin/python -m simulation train --per-model-optimize",
        })

    # Recommended training command
    if (db_info.get("db_size_mb", 0) > 100
        and pipeline_info.get("n_champions", 0) == 0
        and (sysinfo.get("ram_total_gb") or 0) >= 16):
        recs.append({
            "priority": "info", "category": "next-step",
            "issue": "first run on a populated DB",
            "action": "thesis-grade default (HWP §3 + champion-challenger):",
            "cli": (".venv/bin/python -m simulation train --force "
                     "--scenario full --weather-mode hybrid "
                     "--conformal-method aci --ensemble-method stacking "
                     "--covid-mode indicator --per-model-optimize"),
        })

    # Attach ETA tags to recommendations whose CLI is a known command.
    try:
        from simulation.utils.eta import get_command_eta_by_label
        # Map CLI suffix → ETA registry key
        for r in recs:
            cli = r.get("cli") or ""
            eta_key = None
            if "predict-real" in cli:
                eta_key = "predict-real"
            elif "--per-model-optimize" in cli:
                eta_key = "train:full+optim"
            elif "train --scenario full" in cli:
                eta_key = "train:full"
            elif "train --scenario lite" in cli:
                eta_key = "train:lite"
            elif "train-all" in cli:
                eta_key = "train-all"
            elif "collect" in cli and "--backfill" in cli:
                eta_key = "collect:backfill"
            elif cli.endswith("collect") or " collect " in cli:
                eta_key = "collect:default"
            elif "bootstrap" in cli:
                eta_key = "bootstrap"
            elif "doctor" in cli:
                eta_key = "doctor"
            elif "db-optimize" in cli:
                eta_key = "db-optimize"
            if eta_key:
                eta = get_command_eta_by_label(eta_key)
                if eta and eta.typical > 0:
                    r["eta"] = eta.human
                    r["eta_typical_sec"] = eta.typical
    except Exception as _e:
        log.debug(f"  [doctor] ETA tagging skipped: {_e}")

    rep.recommendations = recs


# ─────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────
def run_doctor(*, auto: bool = False, verbose: bool = False,
               save_report: Optional[Path] = None,
               strict: bool = False) -> tuple[int, DoctorReport]:
    """Run full diagnostic. Returns (exit_code, report)."""
    from datetime import datetime, timezone
    t0 = time.time()
    started_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rep = DoctorReport(started_at=started_iso)

    print("\n" + "=" * 70)
    print("  simulation doctor — environment + project diagnostics")
    print("=" * 70)
    if auto:
        print("  Mode: --auto (safe fixes applied automatically)")
    print()

    # Run sections
    print("  --- A. System ---")
    sysinfo = _check_system(rep)
    print("\n  --- B. Accelerator ---")
    accel = _check_accelerator(rep, sysinfo)
    print("\n  --- C. Packages ---")
    _check_packages(rep)
    print("\n  --- D. Env vars (Apple OMP guards) ---")
    _check_env_vars(rep, sysinfo, auto)
    print("\n  --- D2. Subprocess isolation strategy (OS-aware) ---")
    _check_subprocess_strategy(rep)
    print("\n  --- E. Project layout ---")
    _check_project(rep, auto)
    print("\n  --- F. DB ↔ models ---")
    db_info = _check_db_and_models(rep, auto)
    print("\n  --- G. Code self-test ---")
    _check_code(rep)
    print("\n  --- H. Pipeline-ready (caches / champions) ---")
    pipeline_info = _check_pipeline_ready(rep)

    # Recommendations
    _build_recommendations(rep, sysinfo, accel, db_info, pipeline_info)
    rep.elapsed_sec = time.time() - t0

    # ── Print results table ──
    GLYPH = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}
    last_section = None
    for r in rep.results:
        if not verbose and r.status == "ok":
            continue
        if r.section != last_section:
            last_section = r.section
        print(f"  {GLYPH.get(r.status, '[?]   ')} "
              f"[{r.section:<10}] {r.name:<24}  {r.message}")
        if r.fix_hint and r.status != "ok":
            print(f"           ↳ fix: {r.fix_hint}")

    # ── Recommendations block ──
    if rep.recommendations:
        print("\n" + "=" * 70)
        print(f"  Recommendations ({len(rep.recommendations)})")
        print("=" * 70)
        prio_order = {"high": 0, "med": 1, "low": 2, "info": 3}
        for rec in sorted(rep.recommendations,
                            key=lambda r: prio_order.get(r["priority"], 9)):
            print(f"\n  [{rec['priority'].upper():<4}] [{rec['category']}] "
                  f"{rec['issue']}")
            print(f"          → {rec['action']}")
            if rec.get("cli"):
                print(f"          $ {rec['cli']}")
            if rec.get("eta"):
                print(f"          ⏱ ETA: {rec['eta']}")

    # ── Auto-fix summary ──
    if auto and rep.auto_fixes_applied:
        print("\n" + "=" * 70)
        print(f"  Auto-fixes applied ({len(rep.auto_fixes_applied)})")
        print("=" * 70)
        for fx in rep.auto_fixes_applied:
            print(f"  ✓ {fx}")
        print("\n  Note: env-var fixes are process-local. To make them permanent")
        print("  for new shells, add them to ~/.zshrc / ~/.bashrc.")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print(f"  Result: {rep.n_ok} OK, {rep.n_warn} WARN, {rep.n_fail} FAIL  "
          f"({rep.elapsed_sec:.1f}s)")
    if rep.n_fail == 0 and (not strict or rep.n_warn == 0):
        print("  >> System ready.")
    elif rep.n_fail == 0:
        print("  >> System usable but has warnings (strict mode flagged).")
    else:
        print("  >> Address the FAILs above before training.")
    print("=" * 70)

    if save_report is not None:
        save_report = Path(save_report)
        save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report.write_text(json.dumps(rep.to_json(), indent=2, default=str))
        print(f"\n  Report written: {save_report}")

    if rep.n_fail > 0:
        return 1, rep
    if strict and rep.n_warn > 0:
        return 1, rep
    return 0, rep


__all__ = ["run_doctor", "DoctorReport", "CheckResult"]
