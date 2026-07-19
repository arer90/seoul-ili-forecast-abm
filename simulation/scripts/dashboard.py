"""Optuna dashboard launcher (Tier 1 #4).

브라우저로 학습 진행 실시간 모니터링.
http://localhost:8080 에서 trial 분포 / pruning rate / best HP 시각화.

사용:
    .venv/bin/python -m simulation.scripts.dashboard
    .venv/bin/python -m simulation.scripts.dashboard --port 8081 --db optuna_study.db

ENGINEERING_PRINCIPLES.md §원칙 #4 (KISS): 단일 명령으로 dashboard 실행.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
DEFAULT_DB = get_results_dir() / "optuna_feature_selection.db"
ALT_DB = get_results_dir() / "optuna_study.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB),
                    help=f"Optuna study DB (default: {DEFAULT_DB})")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--alt", action="store_true",
                    help=f"Use {ALT_DB.name} instead")
    args = ap.parse_args()

    db = Path(ALT_DB if args.alt else args.db)
    if not db.exists():
        print(f"✗ DB 없음: {db}")
        # Fallback chain
        for candidate in [DEFAULT_DB, ALT_DB]:
            if candidate.exists():
                print(f"  fallback → {candidate}")
                db = candidate
                break
        else:
            return 1

    storage = f"sqlite:///{db}"
    print(f"Optuna dashboard 시작")
    print(f"  DB:      {db}")
    print(f"  Port:    {args.port}")
    print(f"  URL:     http://localhost:{args.port}")
    print()
    print("Ctrl+C 로 중단")
    print()

    try:
        return subprocess.call([
            sys.executable, "-m", "optuna_dashboard",
            storage, "--port", str(args.port),
        ])
    except KeyboardInterrupt:
        print("\n중단됨")
        return 0
    except FileNotFoundError:
        print("✗ optuna-dashboard 미설치")
        print("  설치: uv pip install optuna-dashboard")
        return 1


if __name__ == "__main__":
    sys.exit(main())
