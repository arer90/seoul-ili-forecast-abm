"""
post_E_orchestrator.py
======================
E 학습 완료 후 자동 실행되는 체인:

  1. E 종료 대기 (training_log 의 최신 마커 + PID 체크)
  2. checkpoint / sidecar 백업
  3. (retired) Chronos-MultiCountry fill-in — G-261: no-op (Chronos 제거)
  4. post_E_comprehensive_eval.py   (WIS/CRPS/DM/bootstrap)
  5. post_E_export_r_csvs.py        (R scripts 입력용 CSV 덤프)
  6. r3_5_npi_ablation.py           (SEIR-V2 κ=0 vs κ fit)
  7. [optional] Rscript r_verification/run_all.R  (ADF/KPSS/Box/WIS-R/NB/Rt/ITS/TBATS)

Usage:
  .venv/Scripts/python.exe -m simulation.scripts.post_E_orchestrator \
      [--skip-fill] [--skip-loso] [--skip-r] [--dry-run]

실행 전제:
  - E 학습 (training_log_YYYYMMDD_HHMMSS.log) 이 시작됐을 것
  - (G-261 2026-06-13: Chronos-MultiCountry fill 단계 retire — no-op)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RES = ROOT / "simulation" / "results"
LOG_DIR = ROOT / "simulation" / "logs"
BACKUP_DIR = RES / "backup_pre_post_E"


# ── 1) E 종료 감지 ──────────────────────────────────────────
def _latest_training_log() -> Optional[Path]:
    logs = sorted(LOG_DIR.glob("training_log_*.log"))
    return logs[-1] if logs else None


def _is_e_running(log_path: Path) -> bool:
    """최신 훈련 로그의 꼬리를 읽어 종료 마커가 있는지 확인.

    종료 마커 후보:
      - 'All phases completed'
      - 'Phase 9'  followed by elapsed
      - 'SystemExit' / error traceback
    """
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")[-5000:]
    except Exception:
        return False
    done_markers = (
        "All phases completed",
        "Phase 9 elapsed",
        "pipeline finished",
        "Pipeline Complete!",
        "DONE, saved to",
    )
    if any(m in text for m in done_markers):
        return False
    # Recent alive tick?
    if "alive..." in text[-2000:] or "subprocess 격리 모드로 실행" in text[-2000:]:
        return True
    # Stale fallback: mtime check (fresh within 5min means still running)
    try:
        age = time.time() - log_path.stat().st_mtime
        return age < 300
    except Exception:
        return False


def wait_for_e(poll_interval: int = 60, max_wait_hours: float = 6.0) -> bool:
    """E 종료를 기다린다. True=종료, False=타임아웃."""
    max_poll = int(max_wait_hours * 3600 / poll_interval)
    for i in range(max_poll):
        log_path = _latest_training_log()
        if log_path and not _is_e_running(log_path):
            log.info(f"[wait] E 종료 감지: {log_path.name}")
            return True
        if i % 5 == 0 and log_path:
            log.info(f"[wait] ({i*poll_interval}s elapsed) still running: {log_path.name}")
        time.sleep(poll_interval)
    return False


# ── 2) 백업 ─────────────────────────────────────────────────
def backup_checkpoints() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"pre_post_E_{stamp}"
    dst.mkdir(exist_ok=True)

    for src in [
        RES / "checkpoints",
        RES / "phase4_baseline_sidecar.pkl",
        RES / "csv",
    ]:
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst / src.name, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst / src.name)
    log.info(f"[backup] {dst}")
    return dst


# ── 3~6) Step runners ───────────────────────────────────────
def _run(cmd: list[str], cwd: Optional[Path] = None, check: bool = False) -> int:
    log.info(f"[run] {' '.join(cmd)}")
    try:
        rc = subprocess.run(cmd, cwd=str(cwd or ROOT), check=check).returncode
    except subprocess.CalledProcessError as e:
        rc = e.returncode
    log.info(f"[run] rc={rc}")
    return rc


def fill_chronos_multicountry() -> int:
    # G-261 (2026-06-13): Chronos-MultiCountry retire — Chronos 전 변형 제거.
    #   더 이상 학습 불가(모델 등록 제거) → 이 fill 단계는 no-op. foundation 은 TimesFM-2.5/TiRex.
    log.info("[fill] Chronos-MultiCountry retired (G-261) — fill 단계 skip (no-op)")
    return 0


def run_comprehensive_eval() -> int:
    return _run([PY, "-m", "simulation.scripts.post_E_comprehensive_eval"])


def run_export_r_csvs() -> int:
    return _run([PY, "-m", "simulation.scripts.post_E_export_r_csvs"])


def run_r3_5_npi_ablation() -> int:
    return _run([PY, "-m", "simulation.scripts.r3_5_npi_ablation"])


def run_r_verification() -> int:
    r_dir = ROOT / "simulation" / "r_verification"
    if not r_dir.exists():
        log.warning("[r] r_verification/ missing, skip")
        return 1
    if shutil.which("Rscript") is None:
        log.warning("[r] Rscript not on PATH, skip")
        return 1
    return _run(["Rscript", "run_all.R"], cwd=r_dir)


# ── main orchestration ─────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-wait", action="store_true",
                   help="Skip E-ended check (assume already done)")
    p.add_argument("--skip-fill", action="store_true",
                   help="Skip (retired) Chronos-MultiCountry fill-in [G-261 no-op]")
    p.add_argument("--skip-r", action="store_true",
                   help="Skip Rscript run_all.R")
    p.add_argument("--skip-r35", action="store_true",
                   help="Skip r3_5 NPI ablation")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned steps, do nothing")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    steps: list[tuple[str, bool]] = [
        ("1. wait for E",              not args.skip_wait),
        ("2. backup checkpoints",       True),
        ("3. fill Chronos-MultiCountry (retired, no-op)", not args.skip_fill),
        ("4. comprehensive eval",       True),
        ("5. export R CSVs",            True),
        ("6. r3_5 NPI ablation",        not args.skip_r35),
        ("7. Rscript run_all.R",        not args.skip_r),
    ]
    log.info("\n=== post-E orchestrator plan ===")
    for name, enabled in steps:
        log.info(f"  {'[X]' if enabled else '[ ]'} {name}")
    if args.dry_run:
        log.info("[dry-run] exit")
        return 0

    rc_map: dict[str, int] = {}

    if not args.skip_wait:
        if not wait_for_e():
            log.error("[wait] timeout waiting for E — abort")
            return 2

    backup_checkpoints()

    if not args.skip_fill:
        rc_map["fill"] = fill_chronos_multicountry()
        if rc_map["fill"] != 0:
            log.warning(f"[fill] rc={rc_map['fill']} — continue anyway")

    rc_map["eval"] = run_comprehensive_eval()
    rc_map["export"] = run_export_r_csvs()

    if not args.skip_r35:
        rc_map["r35"] = run_r3_5_npi_ablation()

    if not args.skip_r:
        rc_map["r_verify"] = run_r_verification()

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rc": rc_map,
        "ok": all(v in (0, 1) for v in rc_map.values()),  # 1=SKIP, 0=OK
    }
    summary_path = RES / "post_E_orchestrator_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"\n=== post-E orchestrator summary ===\n{json.dumps(summary, indent=2)}")
    log.info(f"[summary] -> {summary_path}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
