"""Auto-logging tee regression test (Stage 3 follow-up).

Stage 1 installed a FileHandler that only captured `logging.*` calls,
so `print` statements never reached the log file. A `train --dry-run`
produced a 107-byte file containing only the banner. The Stage 3 tee
upgrade captures stdout + stderr + logger output all in one file.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def fresh_log_path():
    f = tempfile.NamedTemporaryFile(prefix="tee_reg_", suffix=".log", delete=False)
    f.close()
    p = Path(f.name)
    yield p
    try:
        os.unlink(p)
    except OSError:
        pass


def test_tee_captures_print_log_and_stderr(fresh_log_path):
    """`print()`, `log.info()`, and direct `sys.stderr.write()` must all
    appear in the log file."""
    from simulation import __main__ as m

    m._configure_file_logging(fresh_log_path)
    try:
        print("captured-print-line")
        m.log.info("captured-log-info")
        m.log.warning("captured-log-warning")
        sys.stderr.write("captured-stderr\n")
    finally:
        m._cleanup_file_logging()

    contents = fresh_log_path.read_text(encoding="utf-8")
    assert "captured-print-line" in contents, (
        "print() output missing from log — tee regression"
    )
    assert "captured-log-info" in contents, "log.info missing from log"
    assert "captured-log-warning" in contents, "log.warning missing from log"
    assert "captured-stderr" in contents, "direct stderr write missing from log"


def test_tee_preserves_utf8_korean(fresh_log_path):
    """UTF-8 Korean text must round-trip cleanly through the tee."""
    from simulation import __main__ as m

    m._configure_file_logging(fresh_log_path)
    try:
        print("한글 테스트 — 확인용 라인")
        m.log.info("한글 로그 메시지")
    finally:
        m._cleanup_file_logging()

    contents = fresh_log_path.read_text(encoding="utf-8")
    assert "한글 테스트" in contents
    assert "한글 로그 메시지" in contents


def test_tee_restores_stdout_on_cleanup(fresh_log_path):
    """After cleanup, sys.stdout must be the original stream (no leak)."""
    from simulation import __main__ as m

    original = sys.stdout
    m._configure_file_logging(fresh_log_path)
    assert sys.stdout is not original, "tee did not wrap stdout"
    m._cleanup_file_logging()
    assert sys.stdout is original, "cleanup did not restore stdout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
