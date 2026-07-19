#!/usr/bin/env python3
"""
simulation -- Unified CLI entry point
=====================================
Usage:
    uv run -m simulation --help
    uv run -m simulation train [--dry-run] [--config FILE]
    uv run -m simulation db-init
    uv run -m simulation db-status
    uv run -m simulation collect [--groups E,D,B]
    uv run -m simulation collect --list
    uv run -m simulation train --dry-run

Covers the full lifecycle: DB init → data collection → training → evaluation
"""
import sys as _sys
# Ensure UTF-8 output on Windows cp949 terminals
if _sys.platform == "win32" and hasattr(_sys, "stdout") and hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import sys
import os

# Apple Silicon libomp fork-safety 문제:
# NegBinGLM(statsmodels/BLAS) 이 OMP 스레드풀 생성 → XGBoost 가 같은 프로세스에서
# OMP 새로 초기화 시 `OMP Error #179: pthread_mutex_init failed (EINVAL)` → SIGSEGV (139).
# 해결: 전 파이프라인에서 OMP/BLAS 를 단일 스레드로 강제.
# KMP_DUPLICATE_LIB_OK=TRUE 는 중복 로드는 허용하나 mutex 충돌은 못 막아서 부족.
# 속도 손실: XGBoost/LightGBM 약 2-3× 느려지지만, 프로세스 생존 > 속도.
# 반드시 numpy/sklearn/xgboost import 전에 설정해야 효력 있음 — 이 위치가 유일하게 안전.
if _sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("XGBOOST_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # 2026-04-26: MPS 미지원 op (timesnet 의 fft.rfft 등) 가 자동 CPU
    # fallback 하도록 활성화. 없으면 학습 도중 NotImplementedError 로
    # crash. 이걸 setdefault 로 두면 외부에서 명시 override 가능.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ════════════════════════════════════════════════════════════════
# 2026-04-26: MPH_PRESET 자동 환경변수 묶음 (사용자 요청)
# ----------------------------------------------------------------
# 분산된 9개 환경변수 (OPTUNA_OBJECTIVE/VERBOSE/SAMPLER/ISOLATE +
#   MPH_FAST_TMPDIR/DEVICE/FORCE_CPU + PYTORCH_ENABLE_MPS_FALLBACK 등)
# 를 한 번에 묶어서 적용.
#   MPH_PRESET=production  → WIS objective + best sampler + ISOLATE + verbose
#   MPH_PRESET=development → RMSE + tpe-mv + 비격리 + 빠른 iteration
#   MPH_PRESET=debug       → DEBUG verbosity + random sampler (재현성)
#   MPH_PRESET=safe        → 가장 보수적 (MPS fallback + CPU + 격리)
# 명시적 env 가 이미 있으면 setdefault 라 override 안 함.
# ════════════════════════════════════════════════════════════════
_preset = os.environ.get("MPH_PRESET", "").lower().strip()
if _preset == "production":
    os.environ.setdefault("OPTUNA_OBJECTIVE", "wis")    # FluSight 표준
    os.environ.setdefault("OPTUNA_VERBOSE", "1")        # per-trial log
    os.environ.setdefault("OPTUNA_SAMPLER", "best")     # per-model best
    os.environ.setdefault("OPTUNA_ISOLATE", "1")        # 메모리 누수 방어
    os.environ.setdefault("MPH_FAST_TMPDIR", "1")       # Linux /dev/shm
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
elif _preset == "development":
    os.environ.setdefault("OPTUNA_OBJECTIVE", "rmse")   # 빠른 iteration
    os.environ.setdefault("OPTUNA_VERBOSE", "1")
    os.environ.setdefault("OPTUNA_SAMPLER", "tpe-mv")
    os.environ.setdefault("OPTUNA_ISOLATE", "0")        # 빠른 디버깅
    os.environ.setdefault("MPH_FAST_TMPDIR", "1")
elif _preset == "debug":
    os.environ.setdefault("OPTUNA_OBJECTIVE", "rmse")
    os.environ.setdefault("OPTUNA_VERBOSE", "2")        # DEBUG level
    os.environ.setdefault("OPTUNA_SAMPLER", "random")   # 재현 단순
    os.environ.setdefault("OPTUNA_ISOLATE", "0")
    os.environ.setdefault("MPH_FAST_TMPDIR", "0")
elif _preset == "safe":
    os.environ.setdefault("OPTUNA_OBJECTIVE", "rmse")
    os.environ.setdefault("OPTUNA_VERBOSE", "1")
    os.environ.setdefault("OPTUNA_SAMPLER", "tpe")      # 가장 보수
    os.environ.setdefault("OPTUNA_ISOLATE", "1")
    os.environ.setdefault("MPH_FORCE_CPU", "1")          # MPS 회피
    os.environ.setdefault("MPH_FAST_TMPDIR", "0")
elif _preset:
    print(f"[WARN] MPH_PRESET='{_preset}' 알 수 없음. "
          f"production/development/debug/safe 중 선택", file=_sys.stderr)

# Preset 적용 결과 가시화 (선택된 경우만)
if _preset:
    print(f"[MPH_PRESET={_preset}] 적용된 환경변수:", file=_sys.stderr)
    for _k in ("OPTUNA_OBJECTIVE", "OPTUNA_VERBOSE", "OPTUNA_SAMPLER",
               "OPTUNA_ISOLATE", "MPH_FAST_TMPDIR", "MPH_FORCE_CPU",
               "PYTORCH_ENABLE_MPS_FALLBACK"):
        _v = os.environ.get(_k)
        if _v is not None:
            print(f"  {_k}={_v}", file=_sys.stderr)
import argparse
import logging
import atexit
import signal
import traceback
from datetime import datetime
from pathlib import Path as _LogPath


# ════════════════════════════════════════════════════════════════
# 2026-04-28: Pre-flight cache staleness check
# ─────────────────────────────────────────────────────────────────
# Issue: 코드 patch 적용 후 학습 process 가 옛 .pyc 캐시 사용 →
#        cap=200 patch 무력화 (DNN-Optuna 502/30, 617/30 폭주)
# Fix:   학습 시작 시 .py 와 __pycache__ mtime 비교 → stale 시 자동 정리
#        + WARNING 출력
# ════════════════════════════════════════════════════════════════

def _check_pycache_staleness(threshold_seconds: int = 60) -> None:
    """학습 시작 시 .pyc 가 .py 보다 오래됐는지 체크 + 자동 정리.

    threshold_seconds: .pyc 가 .py 보다 N 초 이상 더 오래되면 stale
                       (compile 타이밍 차이 고려, default 60s)
    """
    import shutil
    from pathlib import Path
    sim_dir = Path(__file__).parent
    stale_count = 0
    sample = []
    for py_file in sim_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        pyc_dir = py_file.parent / "__pycache__"
        if not pyc_dir.exists():
            continue
        # 해당 .py 의 .pyc 찾기
        for pyc in pyc_dir.glob(f"{py_file.stem}.cpython-*.pyc"):
            try:
                py_mtime = py_file.stat().st_mtime
                pyc_mtime = pyc.stat().st_mtime
                if py_mtime - pyc_mtime > threshold_seconds:
                    stale_count += 1
                    if len(sample) < 3:
                        sample.append(py_file.relative_to(sim_dir.parent))
                    break
            except OSError:
                pass

    if stale_count > 0:
        msg = (
            f"\n⚠ [STALE CACHE WARNING] {stale_count} .py files newer than "
            f".pyc (코드 patch 적용 후 학습 시 사용 안 될 수 있음).\n"
            f"   샘플: {', '.join(str(s) for s in sample)}\n"
            f"   해결: bash run_resume_phase12.sh --clean --no-restart\n"
        )
        print(msg, file=_sys.stderr)
        # 자동 정리 (선택, env MPH_CLEAN_PYCACHE=1)
        if os.environ.get("MPH_CLEAN_PYCACHE", "0") == "1":
            cleared = 0
            for cache_dir in sim_dir.rglob("__pycache__"):
                if cache_dir.is_dir():
                    try:
                        shutil.rmtree(cache_dir)
                        cleared += 1
                    except OSError:
                        pass
            print(f"   [MPH_CLEAN_PYCACHE=1] {cleared} __pycache__ 디렉토리 정리됨\n",
                  file=_sys.stderr)


# train 명령 진입 시에만 체크 (다른 명령 — collect, db-status 등은 skip)
if len(_sys.argv) > 1 and _sys.argv[1] in ("train", "train-all"):
    try:
        _check_pycache_staleness()
    except Exception as _ce:
        print(f"  [pre-flight] cache check failed: {_ce}", file=_sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("simulation")


def _resume_type(value):
    """`--resume-from` argument type — R/P label ("R9") or semantic name.

    Phase numbers were removed when the pipeline moved to R/P labels, so "13" and the
    like now raise. "0" alone survives as the "run every phase" sentinel, because
    run_pipeline.sh passes `--resume-from 0` on an ordinary full run.

    lazy import (CLI startup 경량 유지): runner.resolve_resume_from 으로 index 변환.
    """
    from simulation.pipeline.runner import resolve_resume_from
    return resolve_resume_from(value)


# ════════════════════════════════════════════════════════════════
# 2026-04-27: 견고화 — uncaught exception + signal handlers
# ─────────────────────────────────────────────────────────────────
# - sys.excepthook : 잡히지 않은 예외를 traceback 과 함께 로깅
# - SIGINT/SIGTERM : Ctrl+C / kill 시 graceful shutdown
#                    + 임시파일 cleanup + atexit 트리거
# 학습 process 에 영향 없음 (다음 학습부터 적용).
# ════════════════════════════════════════════════════════════════

def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    """잡히지 않은 예외를 stdout 대신 logger 로 기록 (traceback 포함)."""
    if issubclass(exc_type, KeyboardInterrupt):
        # KeyboardInterrupt 는 정상 종료로 취급 (signal handler 가 처리)
        _sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    log.critical("UNCAUGHT EXCEPTION — process 종료",
                  exc_info=(exc_type, exc_value, exc_tb))
    # traceback 도 명시적으로 로그
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log.critical(f"Traceback:\n{tb_str}")

_sys.excepthook = _log_uncaught_exception


_SHUTDOWN_REQUESTED = {"flag": False}

def _graceful_shutdown(signum, frame):
    """SIGINT (Ctrl+C) / SIGTERM (kill) 시 graceful 종료.

    - 한 번 누르면 → 정리 후 종료 시작
    - 두 번 누르면 → 즉시 강제 종료 (sys.exit(130))
    - atexit handlers 가 stdout 복원 + tmpfile 정리 실행
    """
    sig_name = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}.get(
        signum, f"SIG{signum}")
    if _SHUTDOWN_REQUESTED["flag"]:
        # 두 번째 신호 → 강제
        log.critical(f"⚠ {sig_name} 두 번 수신 → 강제 종료")
        _sys.exit(130)
    _SHUTDOWN_REQUESTED["flag"] = True
    log.warning(f"⚠ {sig_name} 수신 → graceful shutdown 시작 "
                "(다시 한 번 누르면 강제 종료)")
    # SystemExit 발생 → atexit handlers 작동
    _sys.exit(130)

# Signal handlers 등록 (main thread 에서만 작동)
try:
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)
except (ValueError, OSError):
    # background thread 또는 unsupported OS — 무시
    pass


# ============================================================
# : Cross-platform auto-logging
# ------------------------------------------------------------
# Replaces PowerShell `Tee-Object` / bash `tee` / cmd `>>` tricks.
# Every CLI invocation writes to `simulation/logs/{command}_{YYYYMMDD_HHMMSS}.log`
# in addition to stdout/stderr, unless `--no-log-file` is given.
# Cross-platform: uses pathlib + Python's own logging FileHandler, no shell
# redirection needed. Works identically on Windows, Linux, macOS.
# ============================================================

_LOG_FILE_HANDLE = None           # open file handle (utf-8, line-buffered)
_ORIGINAL_STDOUT = None           # saved for restore on atexit
_ORIGINAL_STDERR = None
_TEE_LOG_HANDLER = None           # dedicated StreamHandler attached in configure


class _TeeStream:
    """Cross-platform stdout/stderr fan-out.

    Writes every chunk to both the original stream (terminal) and the log
    file handle. Supports ``print()``, ``sys.stderr.write()``, tracebacks,
    and anything that duck-types to a text stream. ``fileno()`` falls
    through to the original so libraries that need a real file descriptor
    (``tqdm``, ``rich``, C-extensions) keep working.
    """

    def __init__(self, original, file_handle):
        self._original = original
        self._file = file_handle

    def write(self, data):
        # Terminal first so the user still sees output even if the log
        # file handle has been closed / disk is full.
        try:
            n = self._original.write(data)
        except Exception:
            n = len(data) if isinstance(data, str) else 0
        try:
            if self._file is not None and not self._file.closed:
                self._file.write(data)
        except Exception:
            pass
        return n

    def flush(self):
        for s in (self._original, self._file):
            try:
                if s is not None and not getattr(s, "closed", False):
                    s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._original.fileno()

    def getvalue(self):
        # 0-C: pytest's capture/logging plugins probe ``getvalue``
        # on whatever sys.stdout/stderr pointed at when their snapshot was
        # taken — even after cleanup restores the original stream. Returning
        # an empty string keeps teardown quiet without silently forwarding
        # arbitrary attribute lookups to the wrapped stream.
        return ""

    def writable(self):
        try:
            return self._original.writable()
        except Exception:
            return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def __getattr__(self, name):
        # Delegate unknown attributes (buffer, encoding, errors, ...) to
        # the original stream. Called only when normal lookup fails.
        # Raise a clean AttributeError if the original also lacks the attr
        # so probes see the standard Python signal rather than a confusing
        # chained error.
        try:
            return getattr(self._original, name)
        except AttributeError:
            raise AttributeError(
                f"_TeeStream: underlying stream has no attribute {name!r}"
            )


def _default_log_dir() -> "_LogPath":
    """Resolve log dir: env override → simulation/logs (repo-relative)."""
    env = os.environ.get("SIM_LOG_DIR")
    if env:
        return _LogPath(env)
    # __file__ -> simulation/__main__.py  →  parent = simulation/
    return _LogPath(__file__).resolve().parent / "logs"


def _auto_log_file(command: str) -> "_LogPath":
    """Allocate a timestamped log path for a given CLI command."""
    log_dir = _default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_cmd = (command or "unknown").replace("/", "_").replace("\\", "_")
    return log_dir / f"{safe_cmd}_{stamp}.log"


def _configure_file_logging(path: "_LogPath") -> None:
    """Tee stdout/stderr into the log file, with no shell redirection.

    Every ``print()``, log record (via the existing StreamHandler that
    ``basicConfig`` installed), and traceback lands in the same file.
    The FileHandler approach used in Stage 1 only captured logger calls
    and missed the 150+ ``print()`` sites in ``__main__.py`` / pipeline
    runner — which is why ``train --dry-run`` produced a 107-byte log
    file containing only the banner. This tee fixes that.
    """
    global _LOG_FILE_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _TEE_LOG_HANDLER
    # utf-8 explicitly so Korean log lines survive on Windows cp949 defaults.
    # buffering=1 (line-buffered) keeps the log readable while running.
    fh = open(path, "w", encoding="utf-8", errors="replace", buffering=1)
    _LOG_FILE_HANDLE = fh
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    sys.stdout = _TeeStream(sys.stdout, fh)
    sys.stderr = _TeeStream(sys.stderr, fh)
    # The StreamHandler that ``logging.basicConfig`` installed has already
    # captured the OLD sys.stderr; rebind it to the tee so log.info()
    # records also land in the file via the same handler pipeline.
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            try:
                h.stream = sys.stderr
            except Exception:
                pass
    # 0-C: pytest's logging plugin mutates the root logger level and
    # attaches its own LogCaptureHandler (WARNING by default), which swallows
    # log.info records before they reach basicConfig's StreamHandler. Force
    # the root/simulation loggers back to INFO AND attach our own dedicated
    # StreamHandler pointing at the tee'd stderr, so every record lands in
    # the log file regardless of the harness.
    try:
        logging.getLogger().setLevel(logging.INFO)
        log.setLevel(logging.INFO)
    except Exception:
        pass
    try:
        _h = logging.StreamHandler(stream=sys.stderr)
        _h.setLevel(logging.INFO)
        _h.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(_h)
        _TEE_LOG_HANDLER = _h
    except Exception:
        _TEE_LOG_HANDLER = None
    atexit.register(_cleanup_file_logging)


def _cleanup_file_logging() -> None:
    """Restore stdout/stderr and close the log file on interpreter exit."""
    global _LOG_FILE_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _TEE_LOG_HANDLER
    # Flush + detach our dedicated handler FIRST so later logging.shutdown()
    # does not try to write through the soon-to-close file handle.
    try:
        if _TEE_LOG_HANDLER is not None:
            try:
                _TEE_LOG_HANDLER.flush()
            except Exception:
                pass
            try:
                logging.getLogger().removeHandler(_TEE_LOG_HANDLER)
            except Exception:
                pass
            _TEE_LOG_HANDLER = None
    except Exception:
        pass
    try:
        if _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        if _ORIGINAL_STDERR is not None:
            sys.stderr = _ORIGINAL_STDERR
    except Exception:
        pass
    try:
        if _LOG_FILE_HANDLE is not None and not _LOG_FILE_HANDLE.closed:
            _LOG_FILE_HANDLE.flush()
            _LOG_FILE_HANDLE.close()
    except Exception:
        pass
    _LOG_FILE_HANDLE = None
    _ORIGINAL_STDOUT = None
    _ORIGINAL_STDERR = None


def _emit_log_banner(command: str, path: "_LogPath") -> None:
    """Print a single human-readable line so the user can tail the file."""
    try:
        rel = path.relative_to(_LogPath.cwd())
        shown = str(rel)
    except Exception:
        shown = str(path)
    log.info("[log] command=%s file=%s", command, shown)

# ============================================================
# Scenario Presets + Model Registry
# ============================================================
# Phase C2 fix (Day 9, 2026-05-12): SCENARIOS + ALL_MODELS extracted to
# simulation/cli/_scenarios.py — referenced by both __main__.py (build_parser
# choices) and cli/training_commands.py (cmd_train / cmd_train_all dispatch).
from simulation.cli._scenarios import ALL_MODELS, SCENARIOS




def build_parser():
    p = argparse.ArgumentParser(
        prog="simulation",
        description="MPH Infection Simulation Pipeline -- full lifecycle CLI",
    )
    # : cross-platform auto-logging (opt-out only).
    # Each invocation writes simulation/logs/{command}_{timestamp}.log.
    p.add_argument("--no-log-file", action="store_true", dest="no_log_file",
                   help="Disable auto log file (stdout/stderr only)")
    p.add_argument("--log-file", default=None, dest="log_file",
                   help="Explicit log file path (overrides auto-generated)")
    sub = p.add_subparsers(dest="command", help="Available commands")

    # --- db-init ---
    sub_init = sub.add_parser("db-init", help="Initialize DB schema (idempotent)")
    sub_init.add_argument("--db-path", dest="db_path", default=None,
                          help="Custom DB path")

    # --- db-status ---
    sub_status = sub.add_parser("db-status", help="Show table row counts")
    sub_status.add_argument("--db-path", dest="db_path", default=None)

    # --- collect ---
    sub_collect = sub.add_parser("collect", help="Run data collection")
    sub_collect.add_argument("--groups", default=None, nargs="+",
                             help="Group letters: `E D B` (space) or `E,D,B` (comma) "
                                  "or mixed `E,D B` — all accepted. (default: all)")
    sub_collect.add_argument("--list", action="store_true", dest="list_groups",
                             help="List available collection groups")
    sub_collect.add_argument("--force", action="store_true",
                             help="Force re-collection even if data is fresh")
    sub_collect.add_argument("--backfill-days", type=int, default=None,
                             dest="backfill_days",
                             help="Sweep the last N days for every time-windowed "
                                  "collector (B, C, CM, D, E, S). Default = 1 "
                                  "(incremental). Use --backfill-days 365+ to "
                                  "rebuild the full DB on a fresh install.")

    # --- train ---
    sub_train = sub.add_parser("train", help="Run training pipeline (phases 1-9)")
    sub_train.add_argument("--config", default=None, help="YAML config file")
    sub_train.add_argument("--dry-run", action="store_true", help="Pre-flight check only")
    sub_train.add_argument("--scenario", "-s", choices=list(SCENARIOS.keys()),
                           help="Scenario preset (baseline, full, dl-only, etc.)")
    sub_train.add_argument("--preset",
                           choices=["aggressive", "moderate", "conservative",
                                    "boxcox", "robust"],
                           default=None,
                           help="Target transform preset (overrides scenario)")
    sub_train.add_argument("--optuna-mode", choices=["none", "external", "inline", "all"],
                           default=None)
    sub_train.add_argument("--optuna-trials", type=int, default=None)
    sub_train.add_argument("--optuna-strategy", dest="optuna_strategy",
                           choices=["mandatory_only", "feature_only",
                                    "hp_then_feature", "joint"],
                           default=None,
                           help="Optuna search strategy (default: joint)")
    sub_train.add_argument("--epochs", type=int, default=None,
                           help="Training epochs (final fit)")
    # Stage 3 full_light: decouple Optuna inline trial epochs from final fit
    sub_train.add_argument("--inline-epochs", type=int, default=None,
                           dest="inline_epochs",
                           help="Optuna inline-trial epochs (default inherits --epochs)")
    sub_train.add_argument("--early-stopping-patience", type=int, default=None,
                           dest="early_stopping_patience",
                           help="Early-stopping patience for NN training")
    sub_train.add_argument("--resume-from", type=_resume_type, default=None,
                           help="Resume from an R/P label (e.g. R9) or a semantic name "
                                "(e.g. per_model_optimize). Labels are the SSOT in "
                                "simulation/pipeline/phases.py. Phase numbers were removed "
                                "and are rejected; the sole exception is 0, which means "
                                "'run every phase'.")
    sub_train.add_argument("--models", default=None,
                           help="Comma-separated model filter (e.g. LSTM,XGBoost)")
    sub_train.add_argument("--paper-cutoff-week", type=int, default=None,
                           dest="paper_cutoff_week",
                           help="HWP §3 in-sample boundary (week count). "
                                "Default 337 (= 269 train+val + 68 test). "
                                "Weeks past this become real forecast slab.")
    sub_train.add_argument("--in-sample-end", default=None,
                           dest="in_sample_end",
                           help="ISO date override for in-sample boundary "
                                "(takes priority over --paper-cutoff-week). "
                                "Example: 2026-02-09")
    sub_train.add_argument("--no-real-eval", action="store_true",
                           dest="no_real_eval",
                           help="Skip P1 (real forecast slab evaluation).")
    sub_train.add_argument("--weather-mode",
                           choices=["oracle", "observed", "climatology", "hybrid"],
                           default=None,
                           dest="weather_mode",
                           help="P1 PF-risk handling on real slab. "
                                "'oracle' (perfect-foresight ceiling, NOT op-achievable), "
                                "'climatology' (woy-mean fallback), "
                                "'hybrid' (KMA fcst + climatology, RECOMMENDED).")
    sub_train.add_argument("--covid-mode",
                           choices=["include", "exclude", "indicator"],
                           default=None,
                           dest="covid_inclusion_mode",
                           help="R1 COVID-era 3-way sensitivity: "
                                "'include' (legacy), 'exclude' (drop 2020-03 → 2022-12), "
                                "'indicator' (add covid_era covariate).")
    sub_train.add_argument("--conformal-method",
                           choices=["split", "aci", "agaci"],
                           default=None,
                           dest="real_conformal_method",
                           help="P1 conformal PI method. "
                                "'split' (default, exchangeability assumed), "
                                "'aci' (Gibbs & Candès 2021, online α update), "
                                "'agaci' (Zaffran 2022, multi-γ aggregation).")
    sub_train.add_argument("--oof-folds", type=int, default=None, dest="oof_folds",
                           help="OOF WF-CV fold 수 (학위논문 제출용 기본 5 = paper-grade). "
                                "우선순위: --oof-folds > MPH_OOF_FOLDS env > 기본 5. "
                                "미지정 시 5 (config default). 3 = service 빠른 모드.")
    sub_train.add_argument("--ensemble-method",
                           choices=["nnls", "bma", "stacking", "median"],
                           default=None,
                           dest="ensemble_method",
                           help="Ensemble combination method. "
                                "'stacking' (Yao 2018 CRPS-stacking, RECOMMENDED), "
                                "'median' (Sherratt 2023 baseline), "
                                "'nnls'/'bma' (legacy).")
    sub_train.add_argument("--per-model-optimize", action="store_true",
                           dest="per_model_optimize",
                           help="R9: optimize EACH model individually "
                                "(transform × scaler × feature × HP grid). "
                                "Heavy — adds hours.")
    sub_train.add_argument("--no-comprehensive-eval", action="store_true",
                           dest="no_comprehensive_eval",
                           help="Skip R12 (comprehensive aggregator + "
                                "per-model deep-dives + figures + audit).")
    sub_train.add_argument("--sweep", default=None, dest="sweep_spec",
                           help="Multi-config sweep. Format: 'dim1:v1,v2;dim2:v3,v4'. "
                                "Dims: covid (include/exclude/indicator), "
                                "weather (oracle/climatology/hybrid), "
                                "conformal (split/aci/agaci), "
                                "ensemble (nnls/bma/stacking/median). "
                                "Runs pipeline once per Cartesian product and "
                                "aggregates into simulation/results/sweeps/<ts>/.")
    sub_train.add_argument("--lite", action="store_true",
                           help="Lite mode (fewer features)")
    sub_train.add_argument("--force", action="store_true")
    sub_train.add_argument("--no-cache", action="store_true")
    sub_train.add_argument("--skip-feature-optuna", action="store_true",
                           dest="skip_feature_optuna",
                           help="Skip the scenario's auto-rerun of Feature-Optuna "
                                "(use existing optuna_feat_sel_*.json as-is). "
                                "Safety valve for cases where a model like GAM "
                                "times out and blocks Feature-Optuna progress.")
    sub_train.add_argument("--auto-collect", action="store_true", dest="auto_collect",
                           help="Auto-run `collect` before training if DB is stale")
    sub_train.add_argument("--collect-groups", default="all", dest="collect_groups",
                           help="Groups to pass to auto-collect (default: all)")
    sub_train.add_argument("--stale-days", type=float, default=7.0, dest="stale_days",
                           help="DB considered stale after N days (default: 7)")
    sub_train.add_argument("--list-models", action="store_true",
                           help="List all available models")
    sub_train.add_argument("--list-scenarios", action="store_true",
                           help="List scenario presets")
    sub_train.add_argument("--export-config", metavar="FILE",
                           help="Export default config to YAML")

    # --- train-all (scenario sweep) ---
    sub_sweep = sub.add_parser(
        "train-all",
        help="Run every training scenario sequentially (force + no-cache by default)",
    )
    sub_sweep.add_argument("--scenarios", default=None,
                           help="Comma-separated subset (default: all executable scenarios)")
    sub_sweep.add_argument("--skip", default=None,
                           help="Comma-separated scenarios to skip")
    sub_sweep.add_argument("--no-force", action="store_true", dest="no_force",
                           help="Do not pass --force/--no-cache to each run (default: force fresh)")
    sub_sweep.add_argument("--continue-on-error", action="store_true",
                           help="Keep going if a scenario fails")
    sub_sweep.add_argument("--dry-run", action="store_true",
                           help="Print what would run without executing")
    sub_sweep.add_argument("--restart", action="store_true",
                           help="Ignore previous resume state and rerun everything")

    # --- extract-pdf ---
    sub_pdf = sub.add_parser("extract-pdf",
                             help="Extract annual report PDF into DB")
    sub_pdf.add_argument("--pdf", default=None, help="Path to PDF file")
    sub_pdf.add_argument("--force", action="store_true",
                         help="Re-extract even if data already exists")
    sub_pdf.add_argument("--source-tag", default=None,
                         help="Override DB source tag")

    # --- import-external ---
    sub_ext = sub.add_parser("import-external",
                             help="Import external data (WHO FluNet, commuter matrix, KOSIS)")
    sub_ext.add_argument("--scan", action="store_true",
                         help="Scan available files (no writes)")
    sub_ext.add_argument("--all", action="store_true", dest="all_",
                         help="Import everything (FluNet + commuter + KOSIS + registry)")
    sub_ext.add_argument("--flunet", action="store_true")
    sub_ext.add_argument("--commuter", action="store_true")
    sub_ext.add_argument("--kosis-gender", action="store_true", dest="kosis_gender")
    sub_ext.add_argument("--kosis-registry", action="store_true", dest="kosis_registry")

    # --- maintain ---
    sub_maint = sub.add_parser("maintain",
                               help="Run DB data quality fixes + coverage report")
    sub_maint.add_argument("--no-fix", action="store_true",
                           help="Skip fixes, report only")

    # --- bootstrap ---
    sub_boot = sub.add_parser(
        "bootstrap",
        help="Build DB from empty state: schema -> external import -> PDF -> maintain -> verify",
    )
    sub_boot.add_argument("--skip-pdf", action="store_true",
                          help="Skip PDF extraction (slow, ~2 min)")
    sub_boot.add_argument("--skip-maintain", action="store_true",
                          help="Skip data quality maintenance pass")
    sub_boot.add_argument("--vacuum", action="store_true",
                          help="Run VACUUM + ANALYZE at the end")

    # --- db-optimize ---
    sub_opt = sub.add_parser(
        "db-optimize",
        help="WAL checkpoint + ANALYZE (fast). Use --vacuum for full rewrite.",
    )
    sub_opt.add_argument("--vacuum", action="store_true",
                         help="Include VACUUM (rewrites entire file)")

    # --- run-all (end-to-end lifecycle) ---
    sub_runall = sub.add_parser(
        "run-all",
        help="Full lifecycle: bootstrap -> collect -> db-optimize -> train-all",
    )
    sub_runall.add_argument("--skip-bootstrap", action="store_true",
                            help="Skip DB bootstrap (assume DB already built)")
    sub_runall.add_argument("--skip-collect", action="store_true",
                            help="Skip live data collection (no internet needed)")
    sub_runall.add_argument("--skip-optimize", action="store_true",
                            help="Skip WAL checkpoint + ANALYZE")
    sub_runall.add_argument("--skip-train", action="store_true",
                            help="Skip training sweep (DB build only)")
    sub_runall.add_argument("--collect-groups", default="all",
                            help="Collector groups passed to `collect --groups` (default: all)")
    sub_runall.add_argument("--backfill-days", type=int, default=None,
                            dest="backfill_days",
                            help="Propagate to `collect --backfill-days`. "
                                 "Use 365+ to rebuild the full DB from scratch.")
    sub_runall.add_argument("--scenarios", default=None,
                            help="Scenario subset for train-all (default: every executable scenario)")
    sub_runall.add_argument("--skip-scenarios", default=None,
                            help="Scenarios to exclude from train-all")
    sub_runall.add_argument("--vacuum", action="store_true",
                            help="Run VACUUM + ANALYZE in db-optimize stage")
    sub_runall.add_argument("--no-force", action="store_true",
                            help="Do not force-refresh Optuna caches in train-all")
    sub_runall.add_argument("--continue-on-error", action="store_true",
                            help="Keep going if a train-all scenario fails")
    sub_runall.add_argument("--dry-run", action="store_true",
                            help="Print planned stages without executing")
    sub_runall.add_argument("--restart", action="store_true",
                            help="Ignore previous resume state and rerun every stage")

    # ── subcommands ────────────────────────────────────────────
    sub_mig = sub.add_parser(
        "db-migrate-",
        help="Apply schema migration (adds model_registry, run_ledger, "
             "scenario, verifier_audit, rt_estimates, nowcast_results).",
    )
    sub_mig.add_argument("--verbose", action="store_true", default=True,
                         help="Print each created table / column")

    sub_aud = sub.add_parser(
        "verify-audit",
        help="Run AST verifier scan on simulation/ and print FORBIDDEN_PATTERNS violations.",
    )
    sub_aud.add_argument("--path", default="simulation",
                         help="Root path to scan (default: simulation)")
    sub_aud.add_argument("--fail-on-warn", action="store_true",
                         help="Exit non-zero on warnings (default: only on fails)")

    sub_freeze = sub.add_parser(
        "freeze-paper-primary",
        help="Persist PAPER_PRIMARY_11 snapshot with SHA-256 hashes into model_registry.",
    )
    sub_freeze.add_argument("--freeze", action="store_true",
                            help="Set frozen_at=now() (commits to paper's model set)")
    sub_freeze.add_argument("--verify", action="store_true",
                            help="Compare current SHAs against frozen ones")

    sub_orch = sub.add_parser(
        "orchestrate",
        help="Run 3-stage tournament ensemble on existing OOF predictions.",
    )
    sub_orch.add_argument("--oof-json", required=True,
                          help="Path to JSON with {model_name: [preds...], 'y_true': [...]}")
    sub_orch.add_argument("--categories-json", required=True,
                          help="Path to JSON with {model_name: category}")
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    sub_orch.add_argument("--out", default=str(get_results_dir() / "tournament_trace.json"),
                          help="Output trace path")
    sub_orch.add_argument("--top-k", type=int, default=2)
    sub_orch.add_argument("--caruana-steps", type=int, default=50)

    # Stage 5 — Metapop SEIR-V-D simulator
    sub_sim = sub.add_parser(
        "sim",
        help="Run the 25-gu metapop SEIR-V-D simulator under a named scenario.",
    )
    sub_sim.add_argument("--scenario", default=None,
                         help="Scenario name; see `sim --list-scenarios`")
    sub_sim.add_argument("--list-scenarios", action="store_true",
                         help="List registered scenarios and exit")
    sub_sim.add_argument("--days", type=int, default=None,
                         help="Override days (default: scenario-defined, ~200)")
    sub_sim.add_argument("--seed-infected", type=float, default=10.0,
                         help="Initial infectious seed in seed-district")
    sub_sim.add_argument("--seed-district", default="강남구",
                         help="Gu where initial I is seeded")
    sub_sim.add_argument("--use-db", action="store_true",
                         help="Load populations + mobility from epi_real_seoul.db "
                              "(default: synthetic 25-gu uniform-mixing)")
    sub_sim.add_argument("--out", default=None,
                         help="Optional .npz path to save state/incidence/gate")
    # Codex non-bio review #5 (sprint 2026-05-06): user-controllable RNG seed.
    # Previously seed=42 was hardcoded in scenarios.py:84 / io.py:282 /
    # parameters.py:114 — paper §재현성 약점.
    sub_sim.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility (default: 42, "
                              "matches the legacy hardcoded value)")
    # Codex non-bio review #8 (sprint 2026-05-06): explicit gate-bypass flag.
    # Previously `MetapopSEIRVD.run(run_validator=False)` could silently
    # skip the validity gate with no audit trail.
    sub_sim.add_argument("--allow-gate-bypass", action="store_true",
                         help="Allow the epi-validity gate to be bypassed "
                              "(run_validator=False). NOT RECOMMENDED — gate "
                              "violations silently swallowed. Debug only.")
    # M1 forecast→ABM connection: anchor the simulation to the live real forecast.
    sub_sim.add_argument("--anchor-forecast", nargs="?", const="", default=None,
                         metavar="MODEL",
                         help="Forecast→ABM: drive the ABM by anchoring its seasonal "
                              "forcing to the LIVE real_eval forecast. Optional MODEL "
                              "(default = operational best_model).")
    sub_sim.add_argument("--n-agents", type=int, default=37_500,
                         help="Agent count for --anchor-forecast (default 37500).")

    # Stage 6a — ARIA MCP server (JSON-RPC 2.0 ndjson over stdio)
    sub_mcp = sub.add_parser(
        "mcp-server",
        help="Run the ARIA epi MCP server (JSON-RPC ndjson over stdio).",
    )
    sub_mcp.add_argument("--artifacts-dir", default=None,
                         help="Override EPI_ARTIFACTS_DIR (where forecast / "
                              "DM / SHAP / RAG manifests live).")
    sub_mcp.add_argument("--list-tools", action="store_true",
                         help="Print tool schemas (JSON) and exit without "
                              "entering the stdio loop. Useful for smoke "
                              "tests and doc generation.")

    # P4 — ARIA multi-agent advisory layer (on-path crew + fail-loud gate)
    sub_aria = sub.add_parser(
        "aria",
        help="Run the multi-agent ARIA advisory crew (Retriever→Analyst→Verifier) "
             "with a fail-loud grounding gate.",
    )
    sub_aria.add_argument("--query", "-q", default=None,
                          help="Epidemiology-advisory question (KO/EN). "
                               "Required unless --refresh.")
    sub_aria.add_argument("--root", default=None,
                          help="Results root for grounding facts "
                               "(default: active simulation/results).")
    sub_aria.add_argument("--mock", action="store_true",
                          help="Offline structural check (no Ollama). Output is "
                               "labelled and NOT deliverable advisory.")
    sub_aria.add_argument("--host", default="http://127.0.0.1:11434",
                          help="Ollama daemon URL.")
    sub_aria.add_argument("--deep", action="store_true",
                          help="Use the 6 specialist research agents (S2).")
    sub_aria.add_argument("--stream", action="store_true",
                          help="Stream specialist progress as it arrives (S2).")
    sub_aria.add_argument("--refresh", action="store_true",
                          help="Refresh few-shot exemplars from verified history "
                               "and exit (S3); no --query needed.")

    # phase-a / phase-b subparsers removed 2026-05-26 (Sprint B B4):
    # MPH_MULTICOLLINEARITY=auto (G-234, R9) 가 4-method 자동 비교 wire.
    # 별도 phase-a / phase-b CLI 불필요. Worker scripts already archived
    # to simulation/scripts/_archive/.

    # --- overseas-validate (P3: overseas generalization) ---
    sub_ov = sub.add_parser(
        "overseas-validate",
        help="P3 해외 검증 — 서울 champion 모델을 해외 국가에 적용. P2 이후 실행.",
    )
    sub_ov.add_argument("--countries", nargs="+",
                        default=["JP", "US"],
                        help="검증 대상 국가 (default: JP US — JP: JIHS 2023+, US: Delphi wILI%)")
    sub_ov.add_argument("--test-weeks", type=int, default=52, dest="test_weeks",
                        help="검증 주 수 (default: 52)")
    sub_ov.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="계획만 출력, 실행 안 함")

    # --- predict-real (P2: inference using saved champion .pt) ---
    sub_pr = sub.add_parser(
        "predict-real",
        help="Predict on NEW data using saved champion .pt artifacts "
             "(model + fitted scaler + transform state). P2.",
    )
    sub_pr.add_argument("--models", default=None,
                        help="Comma-separated champion subset (e.g. "
                             "XGBoost,LightGBM,NegBinGLM). Default: every "
                             "champion in models/champion_log.json.")
    sub_pr.add_argument("--start-date", default=None, dest="start_date",
                        help="ISO date / week-start for the inference window "
                             "lower bound (inclusive). If omitted, uses the "
                             "feature matrix's last test slab + real slab.")
    sub_pr.add_argument("--end-date", default=None, dest="end_date",
                        help="ISO date / week-start upper bound (inclusive). "
                             "Default: feature matrix end.")
    sub_pr.add_argument("--weeks-ahead", type=int, default=None,
                        dest="weeks_ahead",
                        help="Convenience: predict the LAST N weeks of the "
                             "feature matrix (overrides --start/--end-date).")
    sub_pr.add_argument("--with-actuals", action="store_true",
                        dest="with_actuals",
                        help="If actual ILI rate exists in the window, "
                             "compute WIS/MAE/RMSE/R² and write "
                             "inference_metrics.json.")
    sub_pr.add_argument("--out-dir", default=None, dest="out_dir",
                        help="Override output directory "
                             "(default: simulation/results/inference/<ts>/).")
    sub_pr.add_argument("--models-dir", default="models", dest="models_dir",
                        help="Champion .pt directory (default: models/).")
    sub_pr.add_argument("--list-champions", action="store_true",
                        dest="list_champions",
                        help="List champions in models/champion_log.json and exit.")

    # --- prune (disk footprint optimizer) ---
    sub_pr2 = sub.add_parser(
        "prune",
        help="Reclaim disk: trash non-champion .pt + Optuna VACUUM + "
             "old eval_logs/inference outputs. Reversible (→ _trash/) "
             "unless --purge.",
    )
    sub_pr2.add_argument("--models", action="store_true",
                         dest="prune_models",
                         help="Trash non-champion .pt files (legacy + "
                              "failed-attempt + archived-previous).")
    sub_pr2.add_argument("--optuna-vacuum", action="store_true",
                         dest="optuna_vacuum",
                         help="VACUUM optuna_feature_selection.db (typical "
                              "100MB → 10-20MB).")
    sub_pr2.add_argument("--eval-logs", action="store_true",
                         dest="prune_eval_logs",
                         help="Trash eval_logs/*.jsonl older than --keep-days.")
    sub_pr2.add_argument("--inference-results", action="store_true",
                         dest="prune_inference",
                         help="Trash inference/<ts>/ older than 14 days.")
    sub_pr2.add_argument("--all", action="store_true",
                         dest="prune_all",
                         help="Apply all four (models + optuna-vacuum + "
                              "eval-logs + inference-results).")
    sub_pr2.add_argument("--keep-days", type=int, default=7,
                         dest="keep_days",
                         help="Eval-logs cutoff (default: 7 days).")
    sub_pr2.add_argument("--purge", action="store_true",
                         help="PERMANENT delete; skip _trash/ "
                              "(default: trash for reversibility).")
    sub_pr2.add_argument("--dry-run", action="store_true",
                         dest="dry_run",
                         help="Preview only; no changes.")

    # --- list-models (paper/extra/all tiered model registry view) ---
    sub_lm = sub.add_parser(
        "list-models",
        help="List models by tier (paper-primary 11 / extras 55 / negative / all). "
             "Shows source file + champion status + tier label.",
    )
    sub_lm.add_argument("--tier", default="all",
                         choices=["all", "paper", "extra", "negative"],
                         help="Which tier to list (default: all).")
    sub_lm.add_argument("--with-champion-status", action="store_true",
                         dest="with_champion_status",
                         help="Cross-reference models/champion_log.json — "
                              "show which models have a current champion.")

    # --- rehydrate (register legacy .pt files as champion entries) ---
    sub_rh = sub.add_parser(
        "rehydrate",
        help="Scan models/*.pt and register legacy bare-model files (not "
             "yet in champion_log.json) as champions, importing test_WIS/MAE "
             "from post_E_eval.json when available. Idempotent.",
    )
    sub_rh.add_argument("--dry-run", action="store_true",
                         dest="dry_run",
                         help="Preview only; no writes.")
    sub_rh.add_argument("--force", action="store_true",
                         help="Overwrite existing champion_log entries with "
                              "rehydrated metrics (default: skip if exists).")
    sub_rh.add_argument("--eval-source", default="post_E",
                         choices=["post_E", "none"],
                         dest="eval_source",
                         help="Where to pull metrics from (default: post_E_eval.json).")

    # --- feature-importance (Optuna selection + SHAP integrated) ---
    sub_fi = sub.add_parser(
        "feature-importance",
        help="Generate Optuna feature-selection frequency + SHAP per-model "
             "figures + heatmap (model × feature) + mandatory vs optional "
             "distribution. Uses optuna_feat_freq_*.json + R11 (shap) output.",
    )
    sub_fi.add_argument("--models", default=None,
                         help="Comma-separated subset (default: all champions).")
    sub_fi.add_argument("--top-k", type=int, default=30,
                         dest="top_k",
                         help="Top-K features to plot (default 30).")
    sub_fi.add_argument("--no-shap", action="store_true",
                         dest="no_shap",
                         help="Skip SHAP overlay (Optuna freq only).")
    sub_fi.add_argument("--out-dir", default=None, dest="out_dir",
                         help="Custom output directory.")

    # --- visualize (per-model figures + slab tables + Optuna from .pt) ---
    sub_vis = sub.add_parser(
        "visualize",
        help="Generate per-model figures (timeseries with train/val/test/real "
             "bands, residuals, Optuna trial history) + per-slab metric tables "
             "from saved ChampionArtifact .pt files. No re-training required.",
    )
    sub_vis.add_argument("--models", default=None,
                         help="Comma-separated subset (e.g. XGBoost,LightGBM). "
                              "Default: every champion in models/champion_log.json.")
    sub_vis.add_argument("--no-residuals", action="store_true",
                         dest="no_residuals",
                         help="Skip residual diagnostic plots (still saves "
                              "timeseries + per-model markdown report).")
    sub_vis.add_argument("--no-optuna", action="store_true",
                         dest="no_optuna",
                         help="Skip Optuna trial history figures "
                              "(default: included).")
    sub_vis.add_argument("--out-dir", default=None, dest="out_dir",
                         help="Custom output directory "
                              "(default: simulation/results/visualizations/<ts>/).")

    # --- auto-update (weekly maintenance loop) ---
    sub_au = sub.add_parser(
        "auto-update",
        help="Weekly cycle: collect new data → champion-challenger refit "
             "→ horizon-stratified forecast (h=1 KPI). Idempotent; safe "
             "for cron / launchd / Task Scheduler.",
    )
    sub_au.add_argument("--forecast-only", action="store_true",
                         dest="forecast_only",
                         help="Skip collect + refit; just predict with current "
                              "champions. Use for ad-hoc forecast refresh.")
    sub_au.add_argument("--weeks-ahead", type=int, default=4,
                         dest="weeks_ahead",
                         help="Forecast horizons h=1..N (default: 4). "
                              "h=1 is the operational KPI.")
    sub_au.add_argument("--min-db-age-hours", type=float, default=24.0,
                         dest="min_db_age_hours",
                         help="Skip collect if DB age < N hours (default: 24).")
    sub_au.add_argument("--min-refit-days", type=float, default=7.0,
                         dest="min_refit_days",
                         help="Skip refit if last champion promo < N days ago "
                              "(default: 7).")
    sub_au.add_argument("--force-refit", action="store_true",
                         dest="force_refit",
                         help="Bypass --min-refit-days and refit anyway.")
    sub_au.add_argument("--force-collect", action="store_true",
                         dest="force_collect",
                         help="Bypass --min-db-age-hours and re-collect anyway.")
    sub_au.add_argument("--no-actuals", action="store_true",
                         dest="no_actuals",
                         help="Skip actual ILI rate comparison "
                              "(default: include if available).")
    sub_au.add_argument("--dry-run", action="store_true",
                         dest="dry_run",
                         help="Print plan; don't run anything.")

    # --- doctor (system + project diagnostics with optional auto-fix) ---
    sub_doc = sub.add_parser(
        "doctor",
        help="Diagnose OS / Python / packages / DB / models / pipeline-readiness "
             "and emit hardware-aware recommendations. Use --auto to apply "
             "safe fixes (mkdir, set OMP env, ...).",
    )
    sub_doc.add_argument("--auto", action="store_true",
                         help="Apply safe fixes (mkdir missing dirs, set OMP "
                              "env vars in-process). Never installs packages "
                              "or writes to ~/.zshrc.")
    sub_doc.add_argument("--verbose", action="store_true",
                         help="Print every check (default: hide OK lines).")
    sub_doc.add_argument("--save-report", default=None, dest="save_report",
                         help="Write machine-readable JSON manifest to PATH.")
    sub_doc.add_argument("--strict", action="store_true",
                         help="Exit 1 on WARN as well as FAIL "
                              "(default: only FAIL → 1).")

    return p


# ============================================================
# Phase C2 partial (2026-05-12): resume state helpers extracted to
# simulation/cli/_state.py for shared use by extracted cmd_* handlers.
from simulation.cli._state import (
    _clear_state,
    _load_state,
    _save_state,
    _state_path,
)


# Phase C2 partial (2026-05-12): cmd handlers extracted to simulation/cli/*
# — re-imported here to preserve dispatch table.
from simulation.cli.db_commands import (
    cmd_db_init,
    cmd_db_migrate_v22,
    cmd_db_optimize,
    cmd_db_status,
)
from simulation.cli.maintenance_commands import (
    cmd_auto_update,
    cmd_doctor,
    cmd_maintain,
    cmd_prune,
)
from simulation.cli.sim_commands import (
    cmd_mcp_server,
    cmd_sim,
)
from simulation.cli.aria_commands import cmd_aria
from simulation.cli.data_commands import (
    cmd_import_external,
    cmd_orchestrate,
)
from simulation.cli.inference_commands import (
    cmd_predict_real,
)
from simulation.cli.pipeline_commands import (
    cmd_bootstrap,
    cmd_overseas_validate,
)
from simulation.cli.training_commands import (
    cmd_collect,
    cmd_run_all,
    cmd_train,
    cmd_train_all,
)
from simulation.cli.utility_commands import (
    cmd_extract_pdf,
    cmd_feature_importance,
    cmd_freeze_paper_primary,
    cmd_list_models,
    cmd_rehydrate,
    cmd_verify_audit,
    cmd_visualize,
)


# cmd_collect / cmd_train / cmd_train_all / cmd_run_all
# + _OPTUNA_FEAT_MODEL_MAP / _map_models_to_optuna_keys / _rerun_feature_optuna
# moved to simulation/cli/training_commands.py (Phase C2 partial 5차)


# cmd_extract_pdf moved to simulation/cli/utility_commands.py


# cmd_import_external moved to simulation/cli/data_commands.py


# cmd_maintain moved to simulation/cli/maintenance_commands.py


# cmd_bootstrap moved to simulation/cli/pipeline_commands.py


# cmd_db_optimize / cmd_db_migrate_v22 moved to simulation/cli/db_commands.py
# (imported above near other cmd_db_* handlers).


# cmd_verify_audit / cmd_freeze_paper_primary moved to simulation/cli/utility_commands.py


# cmd_orchestrate moved to simulation/cli/data_commands.py


# cmd_sim / cmd_mcp_server moved to simulation/cli/sim_commands.py


# cmd_feature_importance / cmd_rehydrate / cmd_visualize moved to simulation/cli/utility_commands.py


# cmd_auto_update / cmd_prune / cmd_doctor moved to simulation/cli/maintenance_commands.py


# cmd_predict_real moved to simulation/cli/inference_commands.py


def main():
    parser = build_parser()
    args = parser.parse_args()

    # : cross-platform auto-logging — attach FileHandler before dispatch
    # so every command's stdout/stderr also lands in a timestamped file.
    # Opt-out: `--no-log-file`.  Override path: `--log-file <path>`.
    if args.command is not None and not getattr(args, "no_log_file", False):
        try:
            explicit = getattr(args, "log_file", None)
            log_path = _LogPath(explicit) if explicit else _auto_log_file(args.command)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _configure_file_logging(log_path)
            _emit_log_banner(args.command, log_path)
        except Exception as e:  # log init is never fatal
            log.warning("[log] auto-logging init failed: %s", e)

    if args.command is None:
        parser.print_help()
        print("\nExamples:")
        print("  # --- empty-state bootstrap ---")
        print("  python -m simulation bootstrap         # Build DB from empty state")
        print("  python -m simulation bootstrap --skip-pdf")
        print("  python -m simulation bootstrap --vacuum")
        print("")
        print("  # --- DB management ---")
        print("  python -m simulation db-init           # Schema only (idempotent)")
        print("  python -m simulation db-status         # Row counts + verify_schema")
        print("  python -m simulation db-optimize       # WAL checkpoint + ANALYZE")
        print("  python -m simulation db-optimize --vacuum")
        print("")
        print("  # --- ingestion ---")
        print("  python -m simulation import-external --scan")
        print("  python -m simulation import-external --all")
        print("  python -m simulation import-external --flunet --commuter")
        print("  python -m simulation extract-pdf [--force]")
        print("  python -m simulation maintain          # Quality fixes")
        print("  python -m simulation collect --list")
        print("  python -m simulation collect --groups E,D,B")
        print("")
        print("  # --- training ---")
        print("  python -m simulation train --dry-run")
        print("  python -m simulation train --scenario full --force --no-cache")
        print("  python -m simulation train --models LSTM,XGBoost")
        print("  python -m simulation train-all                 # every scenario, fresh")
        print("  python -m simulation train-all --dry-run")
        print("  python -m simulation train-all --scenarios baseline,full")
        print("")
        print("  # --- end-to-end lifecycle ---")
        print("  python -m simulation run-all                    # bootstrap+collect+optimize+train-all")
        print("  python -m simulation run-all --dry-run")
        print("  python -m simulation run-all --skip-collect     # no internet needed")
        print("  python -m simulation run-all --skip-bootstrap --skip-collect  # train-only")
        return

    commands = {
        "db-init": cmd_db_init,
        "db-status": cmd_db_status,
        "db-optimize": cmd_db_optimize,
        "bootstrap": cmd_bootstrap,
        "collect": cmd_collect,
        "train": cmd_train,
        "train-all": cmd_train_all,
        "run-all": cmd_run_all,
        "extract-pdf": cmd_extract_pdf,
        "import-external": cmd_import_external,
        "maintain": cmd_maintain,
        # 
        "db-migrate-": cmd_db_migrate_v22,
        "verify-audit": cmd_verify_audit,
        "freeze-paper-primary": cmd_freeze_paper_primary,
        "orchestrate": cmd_orchestrate,
        # Stage 5 — metapop simulator
        "sim": cmd_sim,
        # Stage 6a — ARIA MCP server
        "mcp-server": cmd_mcp_server,
        # P4 — ARIA multi-agent advisory layer (on-path crew + fail-loud gate)
        "aria": cmd_aria,
        # P2 — inference using saved champion .pt artifacts
        "predict-real": cmd_predict_real,
        # System + project doctor (env / DB / models / pipeline + auto-fix)
        "doctor": cmd_doctor,
        # Disk footprint optimizer (.pt prune + Optuna VACUUM + stale logs)
        "prune": cmd_prune,
        # Weekly maintenance: collect + champion refit + horizon-stratified forecast
        "auto-update": cmd_auto_update,
        # Visualization from saved .pt artifacts (no re-training)
        "visualize": cmd_visualize,
        # Register legacy bare-model .pt files as champion entries
        "rehydrate": cmd_rehydrate,
        # Tier-grouped model registry view (paper / extra / all)
        "list-models": cmd_list_models,
        # Optuna selection + SHAP integrated importance figures
        "feature-importance": cmd_feature_importance,
        # P3: 해외 국가 검증. P2 이후 실행.
        "overseas-validate": cmd_overseas_validate,
    }
    # phase-a / phase-b removed 2026-05-26 (Sprint B B4):
    # MPH_MULTICOLLINEARITY=auto (G-234) 가 4-method 자동 비교 wire.

    fn = commands.get(args.command)
    if fn:
        # Pre-flight banner: ETA, output dirs, hardware/data warnings, and
        # a backgrounding tip if the run is long. Skipped silently for
        # very fast commands when no warnings exist (see eta.py for the
        # bypass logic). Doctor's own banner is preferred for `doctor`.
        if args.command != "doctor":
            try:
                from simulation.utils.eta import print_preflight_banner
                print_preflight_banner(args)
            except Exception as _e:
                log.debug(f"[preflight] banner skipped: {_e}")
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
