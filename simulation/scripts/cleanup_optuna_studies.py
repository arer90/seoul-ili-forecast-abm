"""Optuna study 정리 — 누적 폭주 study 삭제 (G-143, G-151, G-161).

V2 학습 후 누적 trial:
  DNN-Optuna_v1:    960 trials (정상 30-40 의 24-32×)
  TCN-Optuna_v1:    410 trials (정상 30-40 의 10-13×)

G-161 (2026-05-02) 후속: optuna_feature_selection.db 가 608MB 로 폭증
  (run_optuna_feature_selection.py 의 누적). 기존 cleanup 은 optuna_study.db
  만 처리 → feature_selection.db 무시 → 디스크/메모리 압박. 이번 fix:
  --db 가 list 받아 두 DB 모두 처리 + VACUUM 으로 실제 디스크 회수.

원인 (G-143): WF-CV 3-fold × R9(per_model_optimize) × 은퇴(구 Phase 8 AR) 등 multiple stages 에서
  같은 study_name (`DNN-Optuna_v1`) 으로 study.optimize() 반복 호출 →
  storage SQLite 에 trial 무한 누적 → search space 오염 → V2 mini test 의
  raw R²=0.27 vs default HP R²=0.87 (Optuna 가 오히려 해침).

해결:
  1) 누적 trial 모두 삭제 (해당 study)
  2) study_name 자체도 삭제 → 다음 학습 시 fresh start
  3) 정상 30-40 trial 의 다른 model 들은 보존
  4) VACUUM 으로 SQLite 페이지 회수 → 디스크 실제 축소

사용:
    .venv/bin/python -m simulation.scripts.cleanup_optuna_studies
    .venv/bin/python -m simulation.scripts.cleanup_optuna_studies --dry-run
    .venv/bin/python -m simulation.scripts.cleanup_optuna_studies --threshold 100
    # G-161: 단일 DB 만 처리하고 싶으면
    .venv/bin/python -m simulation.scripts.cleanup_optuna_studies --db simulation/results/optuna_study.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

# G-161 (2026-05-02): default 로 두 DB 모두 처리
DEFAULT_DBS = [
    str(get_results_dir() / "optuna_study.db"),            # R9 per_model_optimize / DL Optuna
    str(get_results_dir() / "optuna_feature_selection.db"),  # run_optuna_feature_selection.py
]


def cleanup_db(
    db_path: Path,
    threshold: int,
    force_list: list[str],
    dry_run: bool,
    do_vacuum: bool = True,
) -> dict:
    """단일 Optuna SQLite DB 정리 — 폭주 study 삭제 + VACUUM (G-143/G-151/G-161, D-4).

    누적 trial 폭주 study (threshold ≥ N) + force-list study 모두 삭제. 7 child
    table 의 foreign-key cascade 처리 후 study row 삭제. VACUUM 으로 free pages
    회수 (실제 디스크 크기 축소). schema mismatch (Optuna 표준 X) DB 는 graceful skip.

    Args:
        db_path: SQLite DB path. 존재 안 하면 graceful skip.
        threshold: trial 수 임계값. ≥ 이상 study 가 삭제 candidate.
                   default 100 (G-143 normal study = 30-40, 폭주 study = 200-1000+).
        force_list: threshold 무관 강제 삭제할 study_name list.
                    예: `["DNN-Optuna_v1", "TCN-Optuna_v1"]` (G-143 known offenders).
        dry_run: True 시 실제 삭제 없이 candidate 출력만.
        do_vacuum: True (default) 시 삭제 후 VACUUM 으로 free page 회수.
                   False 시 row 만 삭제 (DB 크기 변동 X).

    Returns:
        dict (성공 시):
          - db: str — DB path
          - n_studies: int — 전체 study 수 (삭제 전)
          - n_candidates: int — 삭제 candidate 수
          - n_trials_deleted: int — 삭제된 trial 수 (실제 실행 시)
          - n_trials_would_delete: int — dry-run 시
          - size_before / size_after: int (bytes) — VACUUM 전후
          - saved_bytes: int — 절약 디스크
          - dry_run: bool — dry-run 여부
        dict (skip 시):
          - db: str, skipped: True, reason: str

    Raises:
        절대 raise X — schema mismatch / OperationalError 모두 graceful 처리.

    Performance:
        - VACUUM 시 O(DB size) — 73MB DB → ~5초.
        - cascade DELETE 시 O(n_trials × 7 tables) — 1000 trial → ~2초.

    Side effects:
        - sqlite3 connect + DELETE + commit + VACUUM (실제 실행 시).
        - stdout print: 진행 메시지.

    Caller responsibility:
        - dry_run=True 로 먼저 candidate 확인 권장.
        - VACUUM 시 다른 process 가 같은 DB 접근 X 보장 (lock 충돌 방지).

    Example:
        >>> from pathlib import Path
        >>> r = cleanup_db(Path("simulation/results/optuna_feature_selection.db"),
        ...                threshold=100, force_list=["DNN-Optuna_v1"],
        ...                dry_run=False, do_vacuum=True)
        >>> r["saved_bytes"] / 1024 / 1024
        607.6  # MB

    See: G-143 (trial 폭주 root cause), G-151 (cap 50/call fix),
         G-161 (feature_selection.db 확장).
    """
    size_before = db_path.stat().st_size if db_path.exists() else 0
    if not db_path.exists():
        print(f"  DB 없음: {db_path} — skip")
        return {"db": str(db_path), "skipped": True}

    print(f"\n──────────────────────────────────────────────────")
    print(f"DB: {db_path}")
    print(f"  size: {size_before / 1024 / 1024:.1f} MB")

    from simulation.database import safe_connect  # G-116 (2026-05-29)
    conn = safe_connect(str(db_path))
    try:
        studies = conn.execute("SELECT study_id, study_name FROM studies").fetchall()
    except sqlite3.OperationalError as e:
        # G-161: feature_selection.db 가 다른 schema (자체 설계) 일 수 있음
        print(f"  ✗ 'studies' 테이블 없음 ({e}) — Optuna 표준 schema 아님, skip")
        conn.close()
        return {"db": str(db_path), "skipped": True, "reason": str(e)}

    print(f"  Total studies: {len(studies)}")

    candidates = []
    for sid, sname in studies:
        try:
            n = conn.execute("SELECT COUNT(*) FROM trials WHERE study_id=?",
                             (sid,)).fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
        if n >= threshold or sname in force_list:
            reason = f"≥{threshold}" if n >= threshold else "force-list"
            candidates.append((sid, sname, n, reason))

    if not candidates:
        print(f"  삭제 대상 없음")
        conn.close()
        return {"db": str(db_path), "n_studies": len(studies), "n_candidates": 0,
                "size_before": size_before, "size_after": size_before}

    print(f"  삭제 대상 ({len(candidates)}개):")
    total_trials = 0
    for sid, sname, n, reason in candidates:
        print(f"    {sname:<30s}  {n:>7d} trials  ({reason})")
        total_trials += n
    print(f"  → 총 {total_trials} trials")

    if dry_run:
        print(f"  --dry-run — 실제 삭제 안 함")
        conn.close()
        return {"db": str(db_path), "n_candidates": len(candidates),
                "n_trials_would_delete": total_trials, "dry_run": True}

    for sid, sname, n, reason in candidates:
        # Foreign key cascade
        for table in ["trial_values", "trial_params", "trial_user_attributes",
                      "trial_intermediate_values", "trial_system_attributes",
                      "trial_heartbeats"]:
            try:
                conn.execute(f"DELETE FROM {table} WHERE trial_id IN "
                             f"(SELECT trial_id FROM trials WHERE study_id=?)", (sid,))
            except sqlite3.OperationalError:
                pass
        conn.execute("DELETE FROM trials WHERE study_id=?", (sid,))
        for table in ["study_user_attributes", "study_system_attributes",
                      "study_directions"]:
            try:
                conn.execute(f"DELETE FROM {table} WHERE study_id=?", (sid,))
            except sqlite3.OperationalError:
                pass
        conn.execute("DELETE FROM studies WHERE study_id=?", (sid,))
        print(f"    ✓ {sname} ({n} trials) 삭제")

    conn.commit()

    # G-161: VACUUM 으로 실제 디스크 회수 (SQLite 의 free pages 회수)
    if do_vacuum:
        print(f"  VACUUM 진행...")
        try:
            conn.execute("VACUUM")
            conn.commit()
        except sqlite3.OperationalError as ve:
            print(f"  ⚠ VACUUM 실패 ({ve}) — 페이지 회수 안 됨")

    conn.close()
    size_after = db_path.stat().st_size
    saved = size_before - size_after
    print(f"  ✓ 완료 — {size_before/1024/1024:.1f} MB → {size_after/1024/1024:.1f} MB "
          f"(절약 {saved/1024/1024:.1f} MB)")
    return {"db": str(db_path), "n_candidates": len(candidates),
            "n_trials_deleted": total_trials,
            "size_before": size_before, "size_after": size_after,
            "saved_bytes": saved}


def main():
    """CLI entry — 두 Optuna DB 모두 정리 + VACUUM (G-161, D-4).

    `optuna_study.db` (R9 per_model_optimize / DL Optuna) + `optuna_feature_selection.db`
    (run_optuna_feature_selection.py) 두 DB 모두 default 처리. `--db` 로
    단일 DB 명시 가능.

    CLI args (parse_args):
        --db: 콤마구분 DB list. default = 두 DB 모두.
        --threshold: trial 수 임계값 (default 100, ≥ candidate).
        --dry-run: 실제 삭제 X (candidate 출력만).
        --no-vacuum: VACUUM skip (default 실행).
        --force-list: threshold 무관 강제 삭제 study_name (콤마구분).

    Returns: int — exit code (0 = 성공).

    Side effects:
        - sqlite3 connect / DELETE / commit / VACUUM
        - stdout: 진행 메시지 + 요약 (절약 MB)

    Example:
        # default (두 DB, threshold 100, VACUUM 실행)
        $ .venv/bin/python -m simulation.scripts.cleanup_optuna_studies
        # dry-run
        $ ... --dry-run
        # 단일 DB
        $ ... --db simulation/results/optuna_study.db

    See: G-143 (DNN-Optuna 960 trial 폭주), G-151 (cap 50/call),
         G-161 (feature_selection.db 607.6 MB → 0.1 MB 회수).
    """
    ap = argparse.ArgumentParser()
    # G-161: --db 가 list (콤마구분) 받음. 단일 DB 처리 시 명시.
    ap.add_argument("--db", default=",".join(DEFAULT_DBS),
                    help=f"콤마구분 DB list. default = 두 DB 모두 ({DEFAULT_DBS}).")
    ap.add_argument("--threshold", type=int, default=100,
                    help="trial 수 threshold (default: 100). 이상이면 삭제 candidate.")
    ap.add_argument("--dry-run", action="store_true", help="실제 삭제 안 함")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="VACUUM skip (default: 실행 — 페이지 회수)")
    ap.add_argument("--force-list", default="DNN-Optuna_v1,TCN-Optuna_v1",
                    help="threshold 무관 강제 삭제할 study_name")
    args = ap.parse_args()

    db_paths = [Path(p.strip()) for p in args.db.split(",") if p.strip()]
    force_list = [s.strip() for s in args.force_list.split(",")]

    print(f"=== Optuna Study 정리 (G-143, G-151, G-161) ===")
    print(f"Threshold: {args.threshold} trials")
    print(f"Force-list: {force_list}")
    print(f"DBs: {len(db_paths)}")
    for p in db_paths:
        print(f"  - {p}")

    results = []
    for db_path in db_paths:
        result = cleanup_db(
            db_path=db_path,
            threshold=args.threshold,
            force_list=force_list,
            dry_run=args.dry_run,
            do_vacuum=not args.no_vacuum,
        )
        results.append(result)

    # 요약
    print()
    print("══════════════════════════════════════════════════")
    print("요약")
    print("══════════════════════════════════════════════════")
    total_saved = 0
    for r in results:
        if r.get("skipped"):
            continue
        size_before = r.get("size_before", 0) / 1024 / 1024
        size_after = r.get("size_after", 0) / 1024 / 1024
        saved = r.get("saved_bytes", 0) / 1024 / 1024
        print(f"  {Path(r['db']).name}: {size_before:.1f} → {size_after:.1f} MB "
              f"(절약 {saved:.1f} MB, {r.get('n_candidates', 0)} studies)")
        total_saved += saved
    print(f"\n총 절약: {total_saved:.1f} MB")
    print(f"\n✓ 완료 — 다음 학습 시 fresh start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
