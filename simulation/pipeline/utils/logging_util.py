"""
Structured Logging
===================
stdout + 파일 동시 출력. 선택적 JSON 구조화 로깅.
"""
import io
import sys
import time
import logging
from pathlib import Path


def setup_logging(log_dir: str = "results", structured: bool = False) -> Path:
    """로깅 초기화. 로그 파일 경로 반환."""
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"training_log_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 기존 핸들러 제거
    for h in root.handlers[:]:
        root.removeHandler(h)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _phase_meaning_name(phase_ref) -> str:
    """R/P 라벨/의미이름 → 의미이름 (SSOT = simulation.pipeline.phases). 매핑 부재 시 ''.

    lazy import 로 circular 회피. 옛 phase 번호는 더 이상 허용 안 됨(라벨/이름만).
    """
    try:
        from simulation.pipeline import phases
        return phases.name_of(phase_ref)
    except Exception:
        pass
    return ""


def phase_banner(phase_num: int, title: str):
    """Phase 시작 배너 — 의미이름 모듈 + title (번호 없음).

    사용자 2026-06-08 "번호를 다 없애" → 배너는 의미이름만 (예: 'wfcv: Walk-Forward CV').
    phase_ref 는 R/P 라벨/의미이름(simulation.pipeline.phases SSOT).
    """
    log = logging.getLogger(__name__)
    name = _phase_meaning_name(phase_num)
    log.info("")
    log.info("  " + "=" * 58)
    log.info(f"    {name + ': ' if name else ''}{title}")
    log.info("  " + "=" * 58)
    log.info("")


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"
