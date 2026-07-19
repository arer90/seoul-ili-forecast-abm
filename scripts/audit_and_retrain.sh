#!/bin/bash
# 학습 완료 후 문제 모델 자동 탐지 + 선택적 재학습 + 비교 (2026-04-28)
#
# 워크플로우:
#   [1] audit_problem_models.py     → S0/S1/S2 문제 식별
#   [2] retrain_problem_models.py   → 문제 모델만 재학습 (patched code)
#   [3] compare_v1_v2.py            → v1 vs v2 비교 리포트
#   [4] (선택) --apply              → PROMOTE_V2 자동 적용
#
# 옵션:
#   bash scripts/audit_and_retrain.sh                 # 기본 (S0+S1, --regenerate-phase1)
#   bash scripts/audit_and_retrain.sh --include-s2    # S2 도 재학습
#   bash scripts/audit_and_retrain.sh --apply         # PROMOTE_V2 자동 적용
#   bash scripts/audit_and_retrain.sh --no-regen      # Phase 1 cache 사용 (빠름)
#   bash scripts/audit_and_retrain.sh --max-models 5  # 처음 5개만 (테스트)
#   bash scripts/audit_and_retrain.sh --audit-only    # audit 만 (재학습 X)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── 옵션 파싱 ──
INCLUDE_S2=0
APPLY=0
REGEN_PHASE1=1
AUDIT_ONLY=0
MAX_MODELS=""

for arg in "$@"; do
    case "$arg" in
        --include-s2) INCLUDE_S2=1 ;;
        --apply) APPLY=1 ;;
        --no-regen) REGEN_PHASE1=0 ;;
        --audit-only) AUDIT_ONLY=1 ;;
        --max-models=*) MAX_MODELS="${arg#--max-models=}" ;;
    esac
done

echo "════════════════════════════════════════════════════════"
echo "  Problem-model audit + selective retrain ($(date '+%Y-%m-%d %H:%M'))"
echo "════════════════════════════════════════════════════════"
echo "  Include S2: $INCLUDE_S2 | Apply: $APPLY | Regen Phase 1: $REGEN_PHASE1"
echo ""

# ── [1] Audit ────────────────────────────────────────
echo "[1/3] 문제 모델 탐지 (audit_problem_models.py)"
.venv/bin/python -m simulation.scripts.audit_problem_models

if [ "$AUDIT_ONLY" = "1" ]; then
    echo ""
    echo "  --audit-only 옵션 → 재학습 skip"
    echo "  결과: simulation/results/problem_models_audit.json"
    exit 0
fi

# 문제 모델 수 확인
if [ ! -f "simulation/results/problem_models_audit.json" ]; then
    echo "  ✗ audit JSON 생성 실패"
    exit 1
fi

N_S0=$(.venv/bin/python -c "
import json
d = json.load(open('simulation/results/problem_models_audit.json'))
print(d['n_S0_severe'])
")
N_S1=$(.venv/bin/python -c "
import json
d = json.load(open('simulation/results/problem_models_audit.json'))
print(d['n_S1_moderate'])
")
N_S2=$(.venv/bin/python -c "
import json
d = json.load(open('simulation/results/problem_models_audit.json'))
print(d['n_S2_mild'])
")
echo ""
echo "  S0=$N_S0 / S1=$N_S1 / S2=$N_S2"

if [ "$INCLUDE_S2" = "1" ]; then
    N_TARGET=$((N_S0 + N_S1 + N_S2))
else
    N_TARGET=$((N_S0 + N_S1))
fi

if [ "$N_TARGET" = "0" ]; then
    echo ""
    echo "  ✓ 모든 모델 OK — 재학습 불필요"
    exit 0
fi

# ── [2] Retrain ──────────────────────────────────────
echo ""
echo "[2/3] 선택적 재학습 (retrain_problem_models.py) — $N_TARGET 모델"

RETRAIN_ARGS=""
if [ "$INCLUDE_S2" = "1" ]; then
    RETRAIN_ARGS="$RETRAIN_ARGS --include-s2"
fi
if [ "$REGEN_PHASE1" = "1" ]; then
    RETRAIN_ARGS="$RETRAIN_ARGS --regenerate-phase1"
fi
if [ -n "$MAX_MODELS" ]; then
    RETRAIN_ARGS="$RETRAIN_ARGS --max-models $MAX_MODELS"
fi

.venv/bin/python -m simulation.scripts.retrain_problem_models $RETRAIN_ARGS

# ── [3] Compare ──────────────────────────────────────
echo ""
echo "[3/3] v1 vs v2 비교 리포트"

COMPARE_ARGS=""
if [ "$APPLY" = "1" ]; then
    COMPARE_ARGS="$COMPARE_ARGS --apply"
fi
.venv/bin/python -m simulation.scripts.compare_v1_v2 $COMPARE_ARGS

# ── [4] Statistical audit (TRIPOD+AI 정합) ────────────
echo ""
echo "[4/4] Statistical audit (TRIPOD+AI 통합 보고서)"
.venv/bin/python -m simulation.scripts.statistical_audit \
    --mode prediction --baseline persistence --n-boot 1000 \
    || echo "  ⚠ statistical_audit failed (continue)"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ Audit + Retrain + Compare + Statistical 완료"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  결과 파일:"
echo "    simulation/results/problem_models_audit.json     (audit 결과)"
echo "    simulation/results/per_model_optimal_v2/*.json   (재학습 결과)"
echo "    simulation/results/v1_vs_v2_comparison.md        (비교 리포트)"
echo "    simulation/results/v1_vs_v2_comparison.json"
echo "    simulation/results/STATISTICAL_AUDIT.md          (TRIPOD+AI 통합 검증)"
echo "    simulation/results/STATISTICAL_AUDIT.json"
if [ "$APPLY" = "1" ]; then
    echo "    simulation/results/per_model_optimal_v1_backup_*/ (v1 백업)"
fi
