#!/usr/bin/env bash
# Reproducible full-run launcher (2026-05-29)
#
# 한 명령으로 "초기 상태 clean → 전체 파이프라인 detached 실행" 을 재현 가능하게 묶음.
# 이전엔 이 과정(result archive + detached launch)이 대화형 ad-hoc 단계였음 →
# 본 스크립트가 committed 단일 진입으로 고정 (재현성, ENGINEERING_PRINCIPLES.md 원칙 #5).
#
# 파이프라인 자체는 run_pipeline.sh (→ python -m simulation train --scenario full)
# 가 committed 코드 + seed 고정(np.random.seed(42)/torch.manual_seed(42)) 으로 수행 →
# 동일 clone + 동일 명령 = 동일 결과. 본 launcher 는 그 앞단(clean + detach)만 담당.
#
# 사용법:
#   bash scripts/launch_full_run.sh             # clean(archive) → detached 풀런
#   bash scripts/launch_full_run.sh --no-clean --resume-from R9   # 중단된 run 재개
#
#   ⚠ --no-clean 단독은 재개가 아니다. archive 단계만 건너뛸 뿐 resume_from 은 0 이므로,
#     기존 결과 report 가 있으면 runner 가 "Results already exist" 로 exit 1 한다.
#     재개하려면 반드시 --resume-from <label> 을 함께 준다 (label = R1..R12, P1;
#     SSOT = simulation/pipeline/phases.py).
#   bash scripts/launch_full_run.sh --dry-run   # 무엇을 할지 표시, 실행 X
#   bash scripts/launch_full_run.sh --fresh-fe  # FE 캐시 강제 삭제(FE 코드 변경 후). 기본=보존(R1 ~40소스 DB join 11분 절약)
#   bash scripts/launch_full_run.sh --help
#
# 안전: 이미 학습 프로세스가 살아있으면 중복실행/clean 을 거부(exit 1).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

NO_CLEAN=0; DRY=0; FRESH_FE=0; PASS_ARGS=()
for a in "$@"; do
  case "$a" in
    --no-clean) NO_CLEAN=1 ;;
    --dry-run)  DRY=1 ;;
    --fresh-fe) FRESH_FE=1 ;;   # FE 캐시 강제 삭제(FE 코드 변경 후). 기본=보존(DB mtime 자동 무효화)
    # 헤더 주석 블록 전체를 출력한다. 고정 행 범위(구: '2,21p')는 헤더를 한 줄만
    # 늘려도 조용히 잘렸다 — 첫 비-주석 행에서 멈추는 편이 드리프트하지 않는다.
    -h|--help)  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' \
                    "${BASH_SOURCE[0]}"; exit 0 ;;
    *) PASS_ARGS+=("$a") ;;   # G-315: forward 미인식 인자(예: --resume-from R9)를 run_pipeline.sh 로 전달
  esac
done

# ── 1) Guard: 이미 학습 중이면 거부 (live run 밑에서 clean/중복실행 방지) ──
LIVE=$(ps axo pid,command | grep -E "simulation train|run_pipeline\.sh" \
         | grep -v grep | grep -v "launch_full_run" || true)
if [ -n "$LIVE" ]; then
  echo "✗ 학습 프로세스가 이미 실행 중 — 중복 실행/clean 차단:" >&2
  echo "$LIVE" | sed 's/^/    /' >&2
  echo "  진행 중 run 을 기다리거나, 먼저 kill 후 재실행하세요." >&2
  exit 1
fi

# ── 2) Clean: result 초기 상태로 archive (reversible mv, --no-clean 이면 skip) ──
if [ "$NO_CLEAN" = "0" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  A="simulation/results/_archive_fullrun_$TS"
  echo "[clean] result 전체 archive → $A (reversible mv; rm 아님). DB(data/db)·FE cache(cache)는 results 밖이라 보존."
  if [ "$DRY" = "0" ]; then
    mkdir -p "$A"
    # G-320d (2026-06-19, 사용자): results/ 의 모든 산출(결과·csv·json·plots·real_eval·diagnostics·
    #   STATISTICAL_AUDIT·mc_*·eda·phase15_xai·optuna db 등)을 통째로 archive → 완전 fresh 시작 + stale 0.
    #   옛 selective glob 은 csv/·real_eval/·diagnostics_report.json·phase15_xai 를 누락 → stale 잔존 +
    #   runner.py:668(stale diagnostics_report.json → 0초 sys.exit) 사망 위험. 이전 archive 디렉토리만 제외.
    for _item in simulation/results/*; do
      [ -e "$_item" ] || continue                       # 빈 results/ 의 리터럴 glob 가드
      case "$(basename "$_item")" in
        _archive_*|_ARCHIVE_*) continue ;;              # 이전 archive 는 재이동 안 함
      esac
      mv "$_item" "$A/" 2>/dev/null || true
    done
    # G-310: champion .pt dir(results/ 밖 models/)도 archive — fresh champions 에서 시작.
    if [ -d models ]; then mv models "$A/models_champions" 2>/dev/null || true; mkdir -p models; fi
    # FE 캐시: 기본 보존(데이터 R1의 ~40 소스 DB join = ~11분 1회 비용; data.py:132 가 DB mtime>cache
    #   면 자동 재계산). FE 코드를 바꿨을 때만 --fresh-fe 로 강제 삭제.
    if [ "$FRESH_FE" = "1" ]; then
      find simulation/cache -name "*.parquet" -delete 2>/dev/null || true
      echo "  ✓ results 전체 archived + FE cache 삭제(--fresh-fe; 재계산)"
    else
      echo "  ✓ results 전체 archived (결과·csv·json·plots·real_eval·diagnostics 모두) — DB·FE cache 보존"
    fi
    mkdir -p simulation/results/checkpoints
  else
    echo "  (--dry-run) results/* 전체를 $A 로 mv 예정 (이전 archive·DB·FE cache 제외)"
  fi
else
  echo "[clean] skip (--no-clean): 기존 result 유지 → resume 동작"
fi

# ── 3) Detached launch (macOS 는 setsid 없음 → python start_new_session, PPID→1) ──
mkdir -p simulation/logs
LOG="simulation/logs/fullrun_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] detached full pipeline (run_pipeline.sh) → $LOG"
if [ "$DRY" = "1" ]; then
  echo "  (--dry-run) 실행 X. 실제: python start_new_session 로 'bash run_pipeline.sh' detached"
  exit 0
fi
.venv/bin/python - "$ROOT" "$LOG" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} <<'PY'
import subprocess, sys, os
root, log = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]   # G-315: forwarded run_pipeline.sh args (e.g. --resume-from R9)
logf = open(os.path.join(root, log), "w")
p = subprocess.Popen(
    ["bash", "run_pipeline.sh"] + extra,
    stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True,   # setsid in child → 세션 독립, launcher 종료 후 PPID→1 (세션 teardown 생존)
    cwd=root,
)
print(f"  ✓ LAUNCHED PID={p.pid}  (launcher 종료 후 PPID→1)")
PY
echo "[done] 모니터:"
echo "    ps -o pid,ppid,stat,etime -p <PID>        # 생존/경과 (PPID=1 확인)"
echo "    tail -f /tmp/training_resume_*.log        # phase 진행 (가장 최근)"
echo "    tail -f $LOG                              # launcher/preflight"
echo "    중단: kill <PID>"
