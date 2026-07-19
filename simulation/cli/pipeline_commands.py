"""Training pipeline CLI commands — extracted from __main__.py.

Phase C2 partial (2026-05-12 cont.): training-pipeline handlers. Starting
with cmd_bootstrap (empty-DB → production). cmd_collect / cmd_train /
cmd_train_all / cmd_run_all remain inline for now (state machine +
inter-handler closures need careful refactor — separate session).
"""
from __future__ import annotations

import logging
import sys


log = logging.getLogger(__name__)


def cmd_bootstrap(args) -> None:
    """`python -m simulation bootstrap` — empty DB → production multi-step.

    Order:
      1. init_db()           -- create schema (idempotent)
      2. import-external     -- WHO FluNet, commuter, KOSIS gender/registry
      3. extract-pdf         -- 감염병감시연보 (district + monthly)
      4. maintain            -- data quality fixes + disease_master
      5. verify_schema + quick_check + row counts
      6. (optional) VACUUM + ANALYZE
    """
    from simulation.collectors import import_external as ie
    from simulation.collectors.extract_pdf import (
        DEFAULT_SOURCE_TAG, extract_pdf, find_pdf,
    )
    from simulation.database import (
        DB_PATH, checkpoint_wal, get_table_shapes, init_db, quick_check,
        vacuum_analyze, verify_schema,
    )

    db = str(DB_PATH)
    print("=" * 64)
    print("  simulation bootstrap -- empty DB -> production")
    print("=" * 64)

    # --- 1. Schema ---
    print("\n[1/5] init_db() : creating schema (idempotent)")
    conn = init_db(db)
    conn.close()

    # --- 2. External imports ---
    print("\n[2/5] import-external --all")
    try:
        total = ie.import_all(db)
        print(f"       imported {total} rows")
    except FileNotFoundError as e:
        print(f"       skipped: {e}")

    # --- 3. PDF extraction ---
    if not getattr(args, "skip_pdf", False):
        print("\n[3/5] extract-pdf")
        try:
            pdf_path = find_pdf(None)
            result = extract_pdf(pdf_path, db, source_tag=DEFAULT_SOURCE_TAG, force=False)
            if result.get("skipped"):
                print("       skipped (data already present)")
            else:
                print(f"       district={result['district']}, monthly={result['monthly']}")
        except FileNotFoundError as e:
            print(f"       skipped: {e}")
        except RuntimeError as e:
            # Missing optional dep (pdfplumber) -- don't kill bootstrap
            print(f"       skipped: {e}")
        except Exception as e:
            print(f"       skipped: unexpected {type(e).__name__}: {e}")
    else:
        print("\n[3/5] extract-pdf : SKIPPED (--skip-pdf)")

    # --- 4. Maintenance ---
    if not getattr(args, "skip_maintain", False):
        print("\n[4/5] maintain : data quality + disease_master")
        try:
            from simulation.database.maintain import run_maintenance
            run_maintenance(fix=True, report=True)
        except Exception as e:
            print(f"       warning: {e}")
    else:
        print("\n[4/5] maintain : SKIPPED (--skip-maintain)")

    # --- 5. Verification ---
    print("\n[5/5] verification")
    qc = quick_check(db)
    print(f"       quick_check      : {qc}")
    v = verify_schema(db_path=db)
    print(f"       verify_schema    : ok={v['ok']}, missing={v['missing']}")
    shapes = get_table_shapes(db)
    total_rows = sum(c for c in shapes.values() if c > 0)
    print(f"       tables           : {len(shapes)}")
    print(f"       total rows       : {total_rows:,}")

    for t in ("weekly_disease", "kosis_disease_gender", "commuter_matrix",
              "who_flunet", "seoul_annual_report_district",
              "seoul_annual_report_monthly"):
        c = shapes.get(t)
        if isinstance(c, int):
            print(f"         {t:34s} {c:>12,}")

    # --- 6. Optional VACUUM ---
    if getattr(args, "vacuum", False):
        print("\n[+] VACUUM + ANALYZE (this takes a minute on ~680MB)")
        vacuum_analyze(db)
    else:
        checkpoint_wal(db)

    print("\n✓ bootstrap complete")


# cmd_phase_a / cmd_phase_b removed 2026-05-26 (Sprint B B4, Gemini MD audit):
# MPH_MULTICOLLINEARITY=auto (G-234, R9 per_model_optimize) 가 4-method 자동 비교 wire.
# 별도 phase-a / phase-b CLI 불필요. Worker scripts (_phase_a_worker.sh,
# _phase_b_worker.sh, launch_phase.sh) 모두 simulation/scripts/_archive/ 에.


def cmd_overseas_validate(args) -> None:
    """`python -m simulation overseas-validate` — Pov(overseas): 해외 국가 외부 검증.

    파이프라인 공식 스텝 Pov(overseas, 구 phase18). Pinf(inference) real inference 완료 후 실행.
    서울 전 모델(66/67개)을 JP/US/KR에 적용, 동일 320 피처·38 지표 기준.
    KR = 내부 baseline 교차검증, JP/US = 외부 generalizability 검증.

    Args:
        args.countries: list[str] — 검증 대상 국가 (default: ['US', 'JP', 'DE', 'FR', 'KR'])
        args.test_weeks: int — 검증 주 수 (default: 52)
        args.dry_run: bool — 계획만 출력, 실행 안 함

    Side effects: simulation/results/overseas_validation/ 저장 (Pov overseas 산출).
    """
    import subprocess, sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    countries = getattr(args, "countries", None) or ["US", "JP", "DE", "FR", "HK", "KR"]
    test_weeks = getattr(args, "test_weeks", 52)

    cmd = [
        sys.executable, "-m", "simulation.pipeline.overseas",
        "--countries", *countries,
        "--test-weeks", str(test_weeks),
    ]
    if getattr(args, "dry_run", False):
        print(f"[overseas-validate] DRY RUN — would execute: {' '.join(cmd)}")
        return
    print(f"[overseas-validate] Pov(overseas) 실행: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0:
        print(f"[overseas-validate] 종료코드 {result.returncode}")
        sys.exit(result.returncode)


__all__ = [
    "cmd_bootstrap",
    "cmd_overseas_validate",
]
