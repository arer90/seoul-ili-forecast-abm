#!/usr/bin/env bash
# 카테고리별 학습 (A 모드, 사용자 명시 2026-05-02)
#
# 사용:
#   bash scripts/train_by_category.sh           # 모든 카테고리 순차
#   bash scripts/train_by_category.sh --cat tree    # 특정 카테고리만
#   bash scripts/train_by_category.sh --start tree  # tree 부터 끝까지 (resume)
#
# 카테고리:
#   tree       — XGBoost, LightGBM, RandomForest, GradientBoosting, CatBoost
#   linear     — ElasticNet, BayesianRidge, NegBinGLM, NegBinGLM-V7, PoissonAutoreg
#   kernel     — KRR, SVR-Linear, SVR-RBF
#   other      — GAM-Spline, GP-RBF-Periodic, BayesianMCMC
#   ts         — ARIMA, SARIMA, SARIMAX
#   dl-tabular — DNN, DNN-Optuna, TinyMLP, TabularDNN, TabularDNN-Lite, DNN-Conformal
#   dl-seq     — TCN, TCN-Optuna
#   modern-ts  — PatchTST, iTransformer, Mamba, TimesNet, N-BEATS, N-HiTS, TiDE, TFT, ...
#   anchor     — DNN-Conformer, DNN-Stacked-Anchored, DNN-Res-Anchored, DNN-Attention-Anchored
#   graph      — GE-DNN, GE-DNN-GAT
#   mech       — PINN-Lite, MP-PINN, Bayesian-SEIR, Metapop-SEIR, SEIR-V2-Forced, Rt-Augmented
#   foundation — Chronos-2, Chronos-2-FT, Chronos-2-FT-Real, Chronos-MultiCountry, ...
#   ensemble   — NNLS, BMA, Stacking, Blending, ResidualAR, Diversity, Temporal, Adaptive, ...

set -e

# 환경변수 (G-150~G-156 fix 적용)
export MPH_GROUPED_PREPROC=1
export MPH_BEST_BY=oof_cv
export MPH_STABLE_TRANSFORMS=1
export MPH_ADVANCED_FEATURES=1
export MPH_PRUNER=hyperband
export MPH_USE_3STAGE=1
export MPH_PJ_ALPHA_LO=0.0
export MPH_PJ_ALPHA_HI=1.5
export MPH_PI_AUGMENT_LO=0
export MPH_PI_AUGMENT_HI=3
# 2026-06-05 (사용자 명시): 4-criteria/g175 filter 완전 제거 — champion = best-WIS.
# MPH_R2_FLOOR/MAPE_CEILING/WIS_CEILING/PICP95_FLOOR export 폐지 (코드 미소비).
export MPH_LIGHTNING_MAX_TIME_PER_MODEL=900    # Q23 A (2026-05-03): 1800 → 900 (cap, 새 background task 부터 적용)
export MPH_OPTUNA_REMAINING_CAP=25              # Q23 A: default 25 명시 (G-151 cap)
export MPH_PRESET=production

# ════════════════════════════════════════════════════════════════
# G-158 fix (2026-05-02): Optuna trial 간 메모리 누수 방지
# ════════════════════════════════════════════════════════════════
# 문제: in-process Optuna → trial 끝나도 Python heap/Torch allocator 잔존
#       → Cat 1 (tree) 학습 1h 09m 만에 19.1% MEM 도달 (8.8% → 30%+ 추세)
# 원인: scripts 에 OPTUNA_ISOLATE=1 환경변수 누락 → _optuna_torch.py:416
#       `_isolate = (_os.environ.get("OPTUNA_ISOLATE", "0") == "1")` default OFF
# 해결: subprocess.Popen 으로 trial 격리 → child process 종료 시 OS 가
#       memory 100% 회수 (macOS MPS / Linux CUDA / Windows 동일)
# 검증: ps eww $PID | grep OPTUNA_ISOLATE 로 환경변수 누락 여부 확인
# ════════════════════════════════════════════════════════════════
export OPTUNA_ISOLATE=1                       # G-158: trial subprocess 격리

# 카테고리 → 모델 list (bash 3.2 호환 — case 문 사용)
# Phase C6 (2026-05-12): single source-of-truth = simulation.models.registry.
# 이전: 여기 case 문 + Python CATEGORY_MODELS 가 중복 → silent drift risk.
# 정정: Python CLI 한 번 호출 → all_models 변수에 캐싱 후 lookup.
#
# G-169 (2026-05-03) 정책 보존: Bayesian-SEIR / Metapop-SEIR 는 시뮬용 격리
# (`bayesian_seir.py:743` + `metapop_seir.py:624` REGISTRY.register 의도적 주석).
# 카테고리 list (CATEGORY_MODELS) 에서도 제거되어 false-positive missing 차단.
# Locate .venv (may live in main repo when this runs from a worktree)
VENV_PYTHON=""
if [ -x ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
else
    COMMON_DIR=$(git rev-parse --git-common-dir 2>/dev/null || echo "")
    if [ -n "$COMMON_DIR" ]; then
        MAIN_ROOT=$(cd "$COMMON_DIR/.." && pwd)
        [ -x "$MAIN_ROOT/.venv/bin/python" ] && VENV_PYTHON="$MAIN_ROOT/.venv/bin/python"
    fi
fi
if [ -z "$VENV_PYTHON" ]; then
    echo "✗ FAIL — .venv/bin/python 부재 (worktree + main 모두 검색됨)"
    exit 1
fi

ALL_MODELS_CACHE="$("$VENV_PYTHON" -m simulation.models.registry --all 2>/dev/null || true)"
if [ -z "$ALL_MODELS_CACHE" ]; then
    echo "✗ FAIL — registry CLI 호출 실패 (CATEGORY_MODELS 소스 부재)"
    exit 1
fi

get_models() {
    # bash assoc-array lookup against cached --all output
    echo "$ALL_MODELS_CACHE" | awk -F'=' -v cat="$1" '$1==cat {print $2}'
}

# 순서 = registry.py canonical (--list-categories)
CATS_ORDER=($("$VENV_PYTHON" -m simulation.models.registry --list-categories 2>/dev/null))
if [ ${#CATS_ORDER[@]} -eq 0 ]; then
    echo "✗ FAIL — registry CLI --list-categories 비어있음"
    exit 1
fi

# 인자 파싱
SINGLE_CAT=""
START_CAT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cat) SINGLE_CAT="$2"; shift 2 ;;
        --start) START_CAT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# 실행할 카테고리 결정
if [ -n "$SINGLE_CAT" ]; then
    if [ -z "$(get_models "$SINGLE_CAT")" ]; then
        echo "✗ Unknown category: $SINGLE_CAT"
        echo "Available: ${CATS_ORDER[*]}"
        exit 1
    fi
    CATS_TO_RUN=("$SINGLE_CAT")
elif [ -n "$START_CAT" ]; then
    # START_CAT 부터 끝까지
    FOUND=0
    CATS_TO_RUN=()
    for cat in "${CATS_ORDER[@]}"; do
        if [ "$cat" = "$START_CAT" ]; then FOUND=1; fi
        if [ $FOUND -eq 1 ]; then CATS_TO_RUN+=("$cat"); fi
    done
    if [ $FOUND -eq 0 ]; then
        echo "✗ Unknown start category: $START_CAT"
        exit 1
    fi
else
    CATS_TO_RUN=("${CATS_ORDER[@]}")
fi

echo "════════════════════════════════════════════════════════"
echo "  카테고리별 학습 (A 모드, 2026-05-02)"
echo "════════════════════════════════════════════════════════"
echo "  실행 카테고리: ${CATS_TO_RUN[@]}"
echo "  시작 시각: $(date)"
echo ""

# ════════════════════════════════════════════════════════
# G-158 (2026-05-02): Preflight 검증 — 환경변수 / DB / cache
# ════════════════════════════════════════════════════════
# OPTUNA_ISOLATE / MPH_BEST_BY 등 누락되면 학습 시작 차단.
# 우회: SKIP_PREFLIGHT=1
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    if ! bash scripts/preflight_check.sh; then
        echo ""
        echo "✗ Preflight check FAIL — 학습 시작 차단 (G-158)"
        echo "  우회: SKIP_PREFLIGHT=1 bash scripts/train_by_category.sh ..."
        exit 1
    fi
fi
echo ""

TOTAL_T0=$(date +%s)
for cat in "${CATS_TO_RUN[@]}"; do
    MODELS="$(get_models "$cat")"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  [$cat] 학습 시작 ($(date '+%H:%M'))"
    echo "  Models: $MODELS"
    echo "════════════════════════════════════════════════════════"

    LOG_FILE="${MPH_FAST_TMPDIR:-${TMPDIR:-/tmp}}"
    LOG_FILE="${LOG_FILE%/}/train_cat_${cat}_$(date +%Y%m%d_%H%M%S).log"
    T0=$(date +%s)

    .venv/bin/python -m simulation train \
        --resume-from R1 \
        --scenario full \
        --per-model-optimize \
        --weather-mode hybrid \
        --conformal-method aci \
        --ensemble-method stacking \
        --covid-mode indicator \
        --auto-collect \
        --stale-days 7 \
        --collect-groups all \
        --models "$MODELS" \
        > "$LOG_FILE" 2>&1
    RC=$?

    T1=$(date +%s)
    ELAPSED=$((T1 - T0))

    if [ $RC -eq 0 ]; then
        echo "  ✓ [$cat] 완료 ($((ELAPSED / 60))분, log=$LOG_FILE)"
    else
        echo "  ✗ [$cat] FAIL (rc=$RC, log=$LOG_FILE)"
        echo "  중단 — 사용자 확인 필요"
        exit 1
    fi
done

TOTAL_T1=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_T1 - TOTAL_T0))
echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ 모든 카테고리 학습 완료 (총 $((TOTAL_ELAPSED / 60))분)"
echo "════════════════════════════════════════════════════════"
echo "  완료 시각: $(date)"
echo ""
echo "  Audit chain 호출:"
echo "    bash scripts/audit_and_retrain.sh --no-regen"
