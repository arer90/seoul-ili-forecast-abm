#!/usr/bin/env python3
"""Double-fork daemonizer for MPH training launch (macOS, no setsid).

Pattern:
    1. First fork → parent exits, child continues.
    2. os.setsid() → new session leader (no controlling tty).
    3. Second fork → first child exits, grandchild orphan → PPID=1 (launchd).
    4. dup2 stdio → log file, /dev/null for stdin.
    5. os.execvp(cmd) → grandchild reparented to launchd, no Claude session tie.

Memory: training-detach-launch.md — MPH training dies on Claude session teardown
unless launched into its own session via double-fork (PPID=1, own pgid).

Usage:
    .venv/bin/python scripts/double_fork_launch.py <log_path> <cmd> [args...]

Example:
    .venv/bin/python scripts/double_fork_launch.py /tmp/train.log \\
        bash run_resume_phase12.sh --force
"""
import os
import sys


def daemonize(log_path: str, cmd_args: list[str], cwd: str) -> None:
    if os.fork() > 0:
        os._exit(0)

    os.setsid()

    if os.fork() > 0:
        os._exit(0)

    os.chdir(cwd)

    sys.stdout.flush()
    sys.stderr.flush()
    with open("/dev/null", "rb", 0) as f_in:
        os.dup2(f_in.fileno(), 0)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    os.execvp(cmd_args[0], cmd_args)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <log_path> <cmd> [args...]\n")
        sys.exit(2)
    log_path = sys.argv[1]
    cmd_args = sys.argv[2:]
    cwd = os.getcwd()
    daemonize(log_path, cmd_args, cwd)
