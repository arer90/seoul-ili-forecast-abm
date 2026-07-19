#!/usr/bin/env bash
# Preflight check — 학습 시작 전 환경변수 / DB / cache 자동 검증
#
# G-158 (2026-05-02) 사건 후 추가:
#   학습 entry script (run_resume_phase12.sh, train_by_category.sh)
#   안에서 자동 호출되어, 필수 환경변수 / DB / cache / disk space 가
#   모두 갖춰진 상태에서만 학습 시작.
#
# 사용:
#   bash scripts/preflight_check.sh                   # 검증 + RC=0/1
#   bash scripts/preflight_check.sh --strict          # 1 개라도 실패 시 RC=1
#   bash scripts/preflight_check.sh --quiet           # 실패만 출력
#
# 호출 (학습 entry script 안에서):
#   if ! bash scripts/preflight_check.sh --strict; then
#       echo "✗ Preflight failed — 학습 시작 불가"
#       exit 1
#   fi

set -e
cd "$(dirname "$0")/.."

# Locate .venv — may live in main repo when this runs from a worktree.
WORKTREE_ROOT="$(pwd)"
VENV_PYTHON=""
if [ -x "$WORKTREE_ROOT/.venv/bin/python" ]; then
    VENV_PYTHON="$WORKTREE_ROOT/.venv/bin/python"
else
    COMMON_DIR=$(git rev-parse --git-common-dir 2>/dev/null || echo "")
    if [ -n "$COMMON_DIR" ]; then
        MAIN_ROOT=$(cd "$COMMON_DIR/.." && pwd)
        [ -x "$MAIN_ROOT/.venv/bin/python" ] && VENV_PYTHON="$MAIN_ROOT/.venv/bin/python"
    fi
fi

STRICT=0
QUIET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strict) STRICT=1; shift ;;
        --quiet)  QUIET=1; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

FAIL=0
WARN=0

log_ok()   { [ $QUIET -eq 0 ] && echo "  ✓ $1"; }
log_warn() { echo "  ⚠ $1"; WARN=$((WARN + 1)); }
log_fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

[ $QUIET -eq 0 ] && echo "════════════════════════════════════════════════════════"
[ $QUIET -eq 0 ] && echo "  Preflight Check — 학습 시작 전 검증"
[ $QUIET -eq 0 ] && echo "════════════════════════════════════════════════════════"
[ $QUIET -eq 0 ] && echo ""

# ────────────────────────────────────────────────────
# 1. 필수 환경변수 (G-158 + 그 외)
# ────────────────────────────────────────────────────
[ $QUIET -eq 0 ] && echo "[1] 필수 환경변수"

check_env() {
    local var="$1"; local expected="$2"; local gotcha="$3"
    local actual="${!var}"
    if [ -z "$actual" ]; then
        log_fail "$var unset (예상: $expected) — $gotcha"
    elif [ "$actual" != "$expected" ]; then
        log_warn "$var=$actual (권장: $expected) — $gotcha"
    else
        log_ok "$var=$actual"
    fi
}

# G-158: subprocess trial 격리 (메모리 누수 방지)
check_env "OPTUNA_ISOLATE" "1" "G-158: trial in-process → 메모리 누수"
# G-132: OOF best 강제
check_env "MPH_BEST_BY" "oof_cv" "G-132: val n=27 single trap"
# G-133, G-146: stable transforms
check_env "MPH_STABLE_TRANSFORMS" "1" "G-133/146: yeo_johnson Y → R²=−10³⁹"
# G-134 (deprecated 2026-05-26 Sprint 1): MPH_USE_3STAGE 검사 제거
# phase0a archive 됨 → Stage 1 disk path 사라짐, env switch 도 더 이상 효과 없음.
# Phase 12 가 Stage 2 (phase0b) 만 unconditional 로 로드.
# G-152: Lightning timeout
check_env "MPH_LIGHTNING_MAX_TIME_PER_MODEL" "1800" "G-152: Lightning final fit stuck 6-8h"

# 2026-06-05 (사용자 명시): 4-criteria/g175 filter 완전 제거 — champion = best-WIS.
# MPH_R2_FLOOR/MAPE_CEILING/WIS_CEILING/PICP95_FLOOR exact-match 검사 폐지 (canonical 없음).

# G-231 (2026-05-22): anchor blend 완전 제거 — MPH_PJ_ALPHA_*/MPH_PI_AUGMENT_* 검사 불필요
# MPH_PJ_ALPHA_LO/HI, MPH_PI_AUGMENT_LO/HI → 더 이상 사용 안 함
log_ok "Anchor blend 제거 확인 (G-231) — no alpha/augment env vars needed"

[ $QUIET -eq 0 ] && echo ""

# ────────────────────────────────────────────────────
# 2. DB / 데이터 자산
# ────────────────────────────────────────────────────
[ $QUIET -eq 0 ] && echo "[2] DB / 데이터 자산"

DB_PATH="simulation/data/db/epi_real_seoul.db"
if [ -f "$DB_PATH" ]; then
    DB_SIZE=$(ls -l "$DB_PATH" | awk '{print $5}')
    DB_SIZE_MB=$((DB_SIZE / 1024 / 1024))
    if [ $DB_SIZE_MB -lt 100 ]; then
        log_fail "DB 크기 ${DB_SIZE_MB}MB (예상: >100MB) — collect 안 됨?"
    else
        log_ok "DB: $DB_PATH (${DB_SIZE_MB}MB)"
    fi
else
    log_fail "DB 부재: $DB_PATH"
fi

# ────────────────────────────────────────────────────
# 3. Cache / Resume 자산 (Phase checkpoint, fe_cache, per_model_optimal)
# ────────────────────────────────────────────────────
[ $QUIET -eq 0 ] && echo ""
[ $QUIET -eq 0 ] && echo "[3] Resume 자산 (G-150)"

PHASE1_CKPT="simulation/results/checkpoints/checkpoint_phase1.json"
if [ -f "$PHASE1_CKPT" ]; then
    log_ok "Phase 1 checkpoint 보존 (resume 가능)"
else
    log_warn "Phase 1 checkpoint 없음 — fresh start (정상, 첫 학습 시)"
fi

FE_CACHE_COUNT=$(find simulation -maxdepth 4 -name "fe_cache_*.parquet" 2>/dev/null | wc -l | tr -d ' ')
[ "$FE_CACHE_COUNT" -gt 0 ] && log_ok "fe_cache: $FE_CACHE_COUNT files" \
    || log_warn "fe_cache 없음 — 첫 학습 시 정상"

PMO_COUNT=$(find simulation/results/per_model_optimal -maxdepth 1 -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
[ "$PMO_COUNT" -gt 0 ] && log_ok "per_model_optimal: $PMO_COUNT models" \
    || log_warn "per_model_optimal 비어있음 — 첫 학습 시 정상"

# ────────────────────────────────────────────────────
# 4. Optuna study DB
# ────────────────────────────────────────────────────
[ $QUIET -eq 0 ] && echo ""
[ $QUIET -eq 0 ] && echo "[4] Optuna study (G-143/151)"

STUDY_DB="simulation/results/optuna_study.db"
if [ -f "$STUDY_DB" ]; then
    STUDY_SIZE_MB=$(($(ls -l "$STUDY_DB" | awk '{print $5}') / 1024 / 1024))
    if [ $STUDY_SIZE_MB -gt 500 ]; then
        log_warn "Optuna DB ${STUDY_SIZE_MB}MB (>500MB) — cleanup_optuna_studies --threshold 100 권장"
    else
        log_ok "Optuna DB: ${STUDY_SIZE_MB}MB"
    fi
else
    log_ok "Optuna DB 없음 (fresh start)"
fi

# ────────────────────────────────────────────────────
# 5. Disk space (학습 ~5-10GB 필요)
# ────────────────────────────────────────────────────
[ $QUIET -eq 0 ] && echo ""
[ $QUIET -eq 0 ] && echo "[5] Disk space"

AVAIL_KB=$(df -k . | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ $AVAIL_GB -lt 5 ]; then
    log_fail "Disk free ${AVAIL_GB}GB (<5GB) — 학습 중단 위험"
elif [ $AVAIL_GB -lt 10 ]; then
    log_warn "Disk free ${AVAIL_GB}GB (<10GB)"
else
    log_ok "Disk free ${AVAIL_GB}GB"
fi

# ────────────────────────────────────────────────────
# 6. Python venv
# ────────────────────────────────────────────────────
[ $QUIET -eq 0 ] && echo ""
[ $QUIET -eq 0 ] && echo "[6] Python venv"

if [ -n "$VENV_PYTHON" ]; then
    PY_VER=$("$VENV_PYTHON" --version 2>&1)
    log_ok ".venv/bin/python: $PY_VER ($VENV_PYTHON)"
    # Optuna 설치 검증
    if "$VENV_PYTHON" -c "import optuna" 2>/dev/null; then
        OPTUNA_VER=$("$VENV_PYTHON" -c "import optuna; print(optuna.__version__)")
        log_ok "optuna: $OPTUNA_VER"
    else
        log_fail "optuna 설치 안 됨"
    fi
    # cloudpickle (G-154)
    "$VENV_PYTHON" -c "import cloudpickle" 2>/dev/null \
        && log_ok "cloudpickle 설치됨 (G-154)" \
        || log_warn "cloudpickle 없음 — Sequence 모델 pickle 실패 위험 (G-154)"
else
    log_fail ".venv/bin/python 없음 (worktree + main 모두 검색됨)"
fi

# ────────────────────────────────────────────────────
# 7. REGISTRY coverage (G-161 — CatBoost 같은 누락 영구 차단)
# ────────────────────────────────────────────────────
[ $QUIET -eq 0 ] && echo ""
[ $QUIET -eq 0 ] && echo "[7] REGISTRY coverage"

if [ -n "$VENV_PYTHON" ]; then
    REG_OUT=$("$VENV_PYTHON" -c "
from simulation.models.registry import verify_registry_coverage
r = verify_registry_coverage()
if r['ok']:
    print(f\"OK total={r['total_expected']} registered={r['total_registered']}\")
else:
    miss = ', '.join(f'[{c}]{m}' for c, m in r['missing'][:10])
    print(f\"MISSING n={len(r['missing'])} ({miss})\")
" 2>&1)
    if echo "$REG_OUT" | grep -q "^OK"; then
        log_ok "REGISTRY: $REG_OUT"
    else
        log_warn "REGISTRY: $REG_OUT"
    fi
fi

# ────────────────────────────────────────────────────
# 8. (제거됨 2026-06-05) Latest Policy SSOT verify — 4-criteria/g175 완전 제거로 obsolete.
#    champion = best-WIS; canonical 4-criteria 가 없으므로 drift 검사 대상 없음.
# ────────────────────────────────────────────────────

# ────────────────────────────────────────────────────
# 9. Stale "TODO manual" wiring (P2-1 — audit 2026-05-26)
# ────────────────────────────────────────────────────
# Detect manual-wiring TODOs that future refactor sprints might miss.
# `TODO manual` is the canonical marker used in package_*/apply.py for
# patches that require human intervention to wire (Rt clip, sanitize, etc.).
# Listing them keeps the operator aware before each training run.
[ $QUIET -eq 0 ] && echo ""
[ $QUIET -eq 0 ] && echo "[9] Stale TODO manual (audit drift)"

TODO_HITS=$(grep -rn "TODO manual\|TODO 수동" --include="*.py" simulation/ 2>/dev/null \
    | grep -v "_archive\|__pycache__" \
    | wc -l | tr -d ' ')

if [ "$TODO_HITS" -eq 0 ]; then
    log_ok "no stale 'TODO manual' wiring"
else
    log_warn "$TODO_HITS 'TODO manual' marker(s) — verify wiring or archive the source"
    [ $QUIET -eq 0 ] && grep -rn "TODO manual\|TODO 수동" --include="*.py" simulation/ 2>/dev/null \
        | grep -v "_archive\|__pycache__" | head -3 | sed 's/^/        /'
fi

# ────────────────────────────────────────────────────
# 결과
# ────────────────────────────────────────────────────
[ $QUIET -eq 0 ] && echo ""
[ $QUIET -eq 0 ] && echo "════════════════════════════════════════════════════════"
if [ $FAIL -gt 0 ]; then
    echo "  ✗ Preflight FAIL: $FAIL fails, $WARN warnings"
    echo "════════════════════════════════════════════════════════"
    exit 1
elif [ $WARN -gt 0 ] && [ $STRICT -eq 1 ]; then
    echo "  ⚠ Preflight WARN (strict): $WARN warnings → 학습 시작 불가"
    echo "════════════════════════════════════════════════════════"
    exit 1
elif [ $WARN -gt 0 ]; then
    [ $QUIET -eq 0 ] && echo "  ⚠ Preflight WARN: $WARN warnings (학습 진행 가능)"
    [ $QUIET -eq 0 ] && echo "════════════════════════════════════════════════════════"
    exit 0
else
    [ $QUIET -eq 0 ] && echo "  ✓ Preflight PASS — 학습 시작 가능"
    [ $QUIET -eq 0 ] && echo "════════════════════════════════════════════════════════"
    exit 0
fi
