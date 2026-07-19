#!/usr/bin/env bash
# 전체 테스트 한 번에 검증 (사용자 2026-06-16: "TDD 하나씩만 말고 다 돌려서 매번 확인").
#   macOS 는 단일 프로세스서 LightGBM/CQR + OpenMP 충돌로 segfault → 각 파일을 별도
#   프로세스로 돌리고 결과를 집계한다. 한 명령 = 전체 검증 + 단일 PASS/FAIL 요약.
# 사용: bash scripts/run_all_tests.sh            (전체)
#       bash scripts/run_all_tests.sh g274 preproc   (이름에 키워드 매칭만)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PASS=0; FAIL=0; ERR=0; FAILED_FILES=()
echo "═══ 전체 테스트 (per-file, macOS segfault 회피) ═══"
shopt -s nullglob
# Both suites. This repository has two test directories (see docs/REPOSITORY_MAP.md),
# and globbing only simulation/tests/ silently skipped every guard in tests/ —
# including the champion leak-free and paper-integrity checks — while still
# printing "전체 GREEN".
FILES=(simulation/tests/test_*.py tests/test_*.py)
# 인자 있으면 키워드 필터
if [ "$#" -gt 0 ]; then
  SEL=(); for f in "${FILES[@]}"; do for k in "$@"; do [[ "$f" == *"$k"* ]] && { SEL+=("$f"); break; }; done; done
  FILES=("${SEL[@]}")
fi
for f in "${FILES[@]}"; do
  out=$(OMP_NUM_THREADS=1 "$PY" -m pytest "$f" -q 2>&1)
  rc=$?
  line=$(echo "$out" | grep -E "passed|failed|error|no tests" | tail -1)
  if [ "$rc" -eq 0 ]; then
    n=$(echo "$line" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+"); PASS=$((PASS + ${n:-0}))
    printf "  ✓ %-52s %s\n" "$(basename "$f")" "$line"
  else
    # segfault(rc=139) vs 실패 구분
    if echo "$out" | grep -qE "failed"; then nf=$(echo "$line" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+"); FAIL=$((FAIL + ${nf:-1})); np=$(echo "$line" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+"); PASS=$((PASS + ${np:-0}))
    else ERR=$((ERR+1)); fi
    FAILED_FILES+=("$(basename "$f") [rc=$rc] $line")
    printf "  ✗ %-52s %s\n" "$(basename "$f")" "${line:-rc=$rc (segfault/collect err)}"
  fi
done
echo "─────────────────────────────────────────────────────────"
echo "  파일 ${#FILES[@]}개 | PASS=$PASS  FAIL=$FAIL  파일오류=$ERR"
if [ "${#FAILED_FILES[@]}" -gt 0 ]; then
  echo "  ⚠ 문제 파일:"; for x in "${FAILED_FILES[@]}"; do echo "      $x"; done
  exit 1
fi
echo "  ✅ 전체 GREEN"
