#!/bin/bash
# Phase 0 → 14 전체 파이프라인 운영 표준 entry (2026-05-26)
#
# 옵션 (Sprint 2 통합):
#   --force            기존 run_training.sh 흡수 — checkpoint clear + --resume-from 0
#   --clean            기존 clean_restart.sh 흡수 — kill + /tmp + __pycache__ + gc + restart
#     --no-restart     --clean 의 sub: 정리만 (학습 재시작 X)
#     --keep-cache     --clean 의 sub: __pycache__ 유지
#   --dry-run          환경 검증 + 실행 명령 표시, 학습은 시작 X
#   --smoke            빠른 기능 점검 (~30분, 3모델): 전체 phase 경로 + 2-3 CPU 모델 + trial 2
#                      + 수집 skip(기존 DB) + 평가 skip + 출력 격리(MPH_OUTPUT_ROOT) +
#                      누수분(eval_logs 등 하드코딩 경로) run 후 manifest-diff 자동정리.
#                      풀런(6-24h) 전 코드 경로 crash 체크용.
#   --skip-eval        학습 후 audit chain skip
#   --resume-from L    phase label L 부터 재개 (예: R9). label SSOT =
#                      simulation/pipeline/phases.py (R1..R12, P1). phase 번호는
#                      제거됨 — 13 같은 값은 argparse 가 거부한다. 예외는 0 뿐이며
#                      (= 전체 실행), 이 스크립트의 기본값이 바로 그 0 이다.
#   --help, -h         이 메시지
#
# Env vars (기존 호환):
#   SKIP_PREFLIGHT=1                 preflight check skip
#   MPH_SKIP_OPTUNA_CLEANUP=1        Optuna study cleanup skip
#   MPH_FORCE_PHASE1_REGEN=1         Phase 1 checkpoint 강제 재생성
#   SKIP_EVAL=1                      자동 평가 chain skip (= --skip-eval)
#
# 실행:
#   bash run_pipeline.sh                                  # 전체 실행
#   bash run_pipeline.sh --resume-from R9                 # R9 부터 재개
#   bash run_pipeline.sh --force                          # 강제 재학습
#   bash run_pipeline.sh --clean                          # 정리 후 재시작
#   bash run_pipeline.sh --clean --no-restart             # 정리만 (재시작 X)
#   bash run_pipeline.sh --dry-run                        # 검증만
#
#   (run_resume_phase12.sh 는 이 스크립트로 넘기는 back-compat shim 이다.
#    detached 실행은 scripts/launch_full_run.sh 를 쓴다.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ════════════════════════════════════════════════════════════════
# Arg parser (Sprint 2 — Prereq 3 spec)
# ════════════════════════════════════════════════════════════════

usage() {
    sed -n '/^# Phase 0/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

FORCE_MODE=0
CLEAN_MODE=0
NO_RESTART=0
KEEP_CACHE=0
DRY_RUN=0
SKIP_EVAL_FLAG=0
SMOKE_MODE=0
RESUME_FROM=0

while [ $# -gt 0 ]; do
    case "$1" in
        --force)          FORCE_MODE=1; RESUME_FROM=0; shift ;;
        --clean)          CLEAN_MODE=1; shift ;;
        --no-restart)     NO_RESTART=1; shift ;;
        --keep-cache)     KEEP_CACHE=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --smoke)          SMOKE_MODE=1; shift ;;
        --skip-eval)      SKIP_EVAL_FLAG=1; shift ;;
        --resume-from)    RESUME_FROM="$2"; shift 2 ;;
        --help|-h)        usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# ════════════════════════════════════════════════════════════════
# --clean handler — kill / /tmp / __pycache__ / gc / checkpoint 보존 확인
# (구 scripts/clean_restart.sh 의 5단계 logic 흡수, 2026-05-26)
# ════════════════════════════════════════════════════════════════
if [ "$CLEAN_MODE" = "1" ]; then
    echo "════════════════════════════════════════════════════════"
    echo "  학습 완전 정리 ($(date '+%Y-%m-%d %H:%M'))"
    echo "════════════════════════════════════════════════════════"

    # [1/5] Process kill
    echo ""
    echo "[1/5] 모든 학습 process 종료"
    PIDS=$(ps aux | grep -E "simulation train|run_resume|_worker.py|optuna_iso" | grep -v grep | awk '{print $2}')
    if [ -z "$PIDS" ]; then
        echo "  ✓ 진행 중 process 없음"
    else
        echo "  종료 대상 PIDs: $PIDS"
        for p in $PIDS; do kill -TERM "$p" 2>/dev/null || true; done
        sleep 5
        for p in $PIDS; do kill -9 "$p" 2>/dev/null || true; done
        sleep 2
        echo "  ✓ 종료 완료"
    fi

    # [2/5] /tmp 임시파일 정리
    echo ""
    echo "[2/5] 임시파일 정리"
    TMPDIR_BASE=$(dirname "${TMPDIR:-/tmp}")
    TMP_COUNT_BEFORE=$(ls -d /tmp/mph_* /tmp/optuna_iso_* 2>/dev/null | wc -l | tr -d ' ')
    rm -rf /tmp/mph_* /tmp/optuna_iso_* 2>/dev/null || true
    # macOS /var/folders/.../T 도
    if [ -d "$TMPDIR_BASE" ]; then
        find "$TMPDIR_BASE" -maxdepth 2 -type d \( -name "mph_*" -o -name "optuna_iso_*" \) \
            -exec rm -rf {} + 2>/dev/null || true
    fi
    echo "  ✓ /tmp/mph_*, optuna_iso_* 정리 ($TMP_COUNT_BEFORE 개)"

    # [3/5] Python __pycache__ 정리 (--keep-cache 아니면)
    echo ""
    echo "[3/5] Python __pycache__ 정리"
    if [ "$KEEP_CACHE" = "1" ]; then
        echo "  ⊘ skipped (--keep-cache)"
    else
        PYC_BEFORE=$(find simulation -name "__pycache__" -type d 2>/dev/null | wc -l | tr -d ' ')
        find simulation -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find scripts -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        echo "  ✓ __pycache__ 디렉토리 $PYC_BEFORE 개 삭제"
    fi

    # [4/5] Python gc + vm_stat
    echo ""
    echo "[4/5] 메모리 회수"
    .venv/bin/python -c "
import gc
for _ in range(3):
    gc.collect()
print('  ✓ Python gc.collect() ×3')
"
    vm_stat | head -5 | awk '
      /Pages free/ {free=$3}
      /Pages active/ {active=$3}
      /Pages inactive/ {inactive=$3}
      END {
        pg = 16384
        printf "  Free: %d GB / Active: %d GB / Inactive: %d GB\n",
            int(free*pg/1024/1024/1024),
            int(active*pg/1024/1024/1024),
            int(inactive*pg/1024/1024/1024)
      }' 2>/dev/null || true

    # [5/5] Checkpoint + Optuna DB 보존 확인
    echo ""
    echo "[5/5] 결과 보존 확인"
    PHASE_N=$(ls simulation/results/checkpoints/checkpoint_phase*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  Phase checkpoint: $PHASE_N / 14 보존"
    OPTUNA_DBS=$(find simulation/results -maxdepth 2 -name "*.db" -path "*optuna*" 2>/dev/null)
    for db in $OPTUNA_DBS; do
        SIZE=$(ls -la "$db" | awk '{print $5}')
        SIZE_MB=$(.venv/bin/python -c "print(f'{$SIZE/1024/1024:.1f}')")
        echo "  $(basename "$db"): ${SIZE_MB} MB (warm-start 보존)"
    done

    if [ "$NO_RESTART" = "1" ]; then
        echo ""
        echo "════════════════════════════════════════════════════════"
        echo "  ✓ 정리 완료 (--no-restart, 학습 재시작 안 함)"
        echo "════════════════════════════════════════════════════════"
        echo "  수동 시작: bash run_resume_phase12.sh"
        exit 0
    fi
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  정리 완료 — 학습 재시작 진입"
    echo "════════════════════════════════════════════════════════"
    # CLEAN_MODE 면서 NO_RESTART=0 이면 아래 normal flow 로 떨어짐.
fi

# ════════════════════════════════════════════════════════════════
# 환경 정리 (이전 설정 제거)
# ════════════════════════════════════════════════════════════════
unset MPH_MANDATORY_SET     # → default = full (321 features)
unset MPH_DROP_HIGH_VIF      # → default = 0 (lag features 보존)

# ── 핵심 설정 ──
export MPH_PRESET=production
export MPH_PRUNER=hyperband           # Hyperband (Li 2017)

# 2026-06-16 (재현성, Gemini 진단): PYTHONHASHSEED 를 python 기동 **전** 고정. runner.py:660 의
#   런타임 os.environ 설정은 no-op(인터프리터 이미 시작) → 매 run hash 랜덤 → TPE group/multivariate
#   sampler 비결정(안 건드린 XGBoost 도 transform 갈림). shell export 가 SSOT. 격리 subprocess 도 env 상속.
export PYTHONHASHSEED=42

# G-158 fix (2026-05-02): Optuna trial 메모리 격리
export OPTUNA_ISOLATE=1
export MPH_LIGHTNING_MAX_TIME_PER_MODEL=1800   # G-152: Lightning final fit timeout
export MPH_OVERSEAS_ENCODER_CACHE=1            # G-279 fix: OverseasTransfer encoder 1회 사전학습+캐시(수백× 가속; frozen=누수0)

# ── Native-thread 상한 (2026-06-03): 아침 main 이 phase0.5 Polars VIF 처리 중 SIGABRT
#    (.ips: polars_stream async_executor + libsystem_pthread + libomp + libtorch 공존 →
#    pthread_mutex_init System #22, CQR-LightGBM OMP #179 와 동일 뿌리). Polars/OpenMP 스레드풀
#    상한으로 다중 native-runtime 충돌 차단. n=349 소표본 → tree 단일스레드 영향 미미(factorial 검증).
export POLARS_MAX_THREADS=4
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# G-264 프라이버시 (사용자 2026-06-13 "외부 유출 안 했으면, 기능만"): TabPFN/HF 텔레메트리 완전 차단.
# TABPFN_DISABLE_TELEMETRY → telemetry_enabled() 가 PostHog/config-download 전에 False (외부 전송 0).
# 가중치 1회 다운로드(공개 모델, 사용자 데이터 X)는 허용 — HF_HUB_OFFLINE 은 set 안 함(가중치 필요).
export TABPFN_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_TELEMETRY=1

# 신규 grouped preprocessing + OOF best
export MPH_GROUPED_PREPROC=1
export MPH_BEST_BY=oof_cv
export MPH_STABLE_TRANSFORMS=1
# G-303 (2026-06-17, 검증 적발): hard linear-inverse 제한이 launcher 미설정으로 inert 였음
#   (STABLE+soft penalty 만 보호). 음수-폭발 가족(DNN −24.8/TCN/GCN)에 하한-안전 backstop 으로
#   per-model allow-list 기본값 설정. 외부 env 존중(resume launch 명령이 덮어씀). 챔피언(TabPFN/
#   NegBinGLM/ElasticNet) 비포함 — 전수표 흔들지 않게. N-BEATS/TiDE 는 G-300 force_y 라 무관.
export MPH_LINEAR_INVERSE_MODELS=${MPH_LINEAR_INVERSE_MODELS:-"DNN,TCN,GCN,DNN-Conformal,TabularDNN,DLinear"}
# Feature guard = NESTED size-path + 1-SE (사용자 채택 2026-06-01, 실측 binary-vs-nested).
#   binary{subset,full} → π ladder(0.8/0.6/0.4)+full nested 사다리에서 1-SE/parsimony per-model 선택.
#   실측: nested 우세 1(KRR test 8.59→4.26, −4.337)/동등 4/binary 우세 1(DNN 소폭). 비정규화 모델
#   (kernel 등) full-fallback 재앙 구제 = safety net. codex+Gemini 검증("유일하게 안전한 enrichment").
#   default(코드)=binary 유지 — 학습만 opt-in. MPH_FEAT_PATH=binary 로 회귀 가능.
export MPH_FEAT_PATH=nested

# ── LEAN phase-13 budget (다중 검토 합의 결론 C, 2026-06-02) ──────────────
#   full per-model 최적화는 n=242/p=401 에서 full 401 가 BASIC 대비 ~0.005 R² 추가 위해 ~12-30h =
#   Freedman's paradox 노이즈 채굴. lean 은 credible full-pool 후보만 공급(~1-3h) + 동일한
#   FAIR-COMPETITION II(phase14 test-slab head-to-head, champion=진짜 best-WIS) 출력 + null 확인.
#   활성: MPH_PHASE13_BUDGET=lean (env-only; 코드 무변경, full 로 회귀 자유).
if [ "${MPH_PHASE13_BUDGET:-}" = "lean" ]; then
  export MPH_FEAT_PATH=binary          # 2-size(subset/full) — nested 4-size ladder 생략
  export MPH_PREPROC_TRIALS=10         # preproc Optuna 30 → 10
  export MPH_HP_OPTUNA_TRIALS=15       # per-model HP Optuna → 15
  export MPH_MC_COMPARE=0              # mc 4-way probe 생략 (적용 mc=none 기본 유지, 시각화만 off)
  echo "  [run_pipeline] MPH_PHASE13_BUDGET=lean → binary/preproc10/hp15/mc-compare-off (~1-3h)"
fi

# 신규 advanced features
export MPH_ADVANCED_FEATURES=1

# 2026-05-26 (Sprint 1 hierarchical 단일화):
#   phase0a (Stage 1) archive 됨 — Phase 12 의 hierarchical 이 trial 마다
#   preproc 결정. 단 stage2_feature_optuna/ (phase0b 결과) 는 여전히 phase12 가
#   자동 로드 (mc filter 의 feature subset 으로 사용).
# (MPH_USE_3STAGE removed 2026-05-26 — no longer consumed; MPH_MODEL_AWARE_PREPROC
#  removed 2026-05-27 — sh export 만 있고 코드 grep 0 hits, vestigial flag)

# G-234 (2026-05-24): MPH_MULTICOLLINEARITY=auto → Phase 13 mc 4-method 자동 probe
# probe disable (2026-06-04): MPH_NO_MC_PROBE=1 → mc=none + probe 생략. 사유 = per-model mc
#   probe 의 ~400 in-process fit 이 macOS pthread key(512) 고갈 → OMP Error #179 크래시(~1h48m,
#   GE-GAT/GLARMA 구간). worker 는 OPTUNA_ISOLATE subprocess 라 OMP 안전 → probe 만 끄면 완주.
#   (auto/per-model probe 복귀 = flag 해제. 근본해결 = probe fit subprocess 격리 — 별도 작업.)
if [ "${MPH_NO_MC_PROBE:-}" = "1" ]; then
  export MPH_MULTICOLLINEARITY=none
  export MPH_MC_PER_MODEL=0
  export MPH_MC_COMPARE=0
  echo "  [run_pipeline] MPH_NO_MC_PROBE=1 → mc probe 생략 (mc=none, OMP #179 회피)"
else
  export MPH_MULTICOLLINEARITY=auto
fi

# G-233 (2026-05-24): hierarchical preproc Optuna
# 2026-05-28 사용자 명시 trial budget 통일 (preproc + feature + HP).
# G-302 (2026-06-17, budget 감사 + 사용자): preproc 100→60. STABLE 공간이 ~12셀(y 6 × x 2)이라
#   100 은 ~8× 과다(나머지는 중복 TPE 재표본). 60 = 늦은-plateau 챔피언층 cap, plateau-stop 은
#   조기 모델을 ~35서 종료. n_startup 은 floor 30 으로 고정(아래) → G-280 log1p 커버리지 ~96%
#   유지(98.5%서 소폭↓, 무손실 영역). 외부 env 존중(미설정 시 60).
export MPH_PREPROC_TRIALS=${MPH_PREPROC_TRIALS:-60}    # Stage 1 preproc Optuna (G-302: 100→60)
export MPH_OPTUNA_STARTUP_MIN=${MPH_OPTUNA_STARTUP_MIN:-30}   # G-302: startup floor 30 (coverage 유지; n_trials↓에도 random 탐색량 보존)
# G-297 (2026-06-17, budget 감사): MPH_FEATURE_OPTUNA_TRIALS 는 DEAD CONFIG — 어떤 python 도
#   읽지 않는다(grep 0). Stage-2 feature 선택은 Optuna 가 아니라 STABILITY selection
#   (select_features_stability, Meinshausen-Bühlmann 재표본). 돌지 않는 budget 을 암시하지
#   않도록 export 비활성(주석). 복원 금지 — 진짜 Stage-2 trial 노브는 존재하지 않음.
# export MPH_FEATURE_OPTUNA_TRIALS=20
export MPH_HP_OPTUNA_TRIALS=${MPH_HP_OPTUNA_TRIALS:-20}   # Stage 3 — model internal HP Optuna (외부 env 존중: lean 15 안 덮음; 미설정 시 20)
export MPH_HIER_MAX_CHAIN=2

# MPH_OPTUNA_TRIALS_JSON — model 별 HP budget override (JSON dict).
# 2026-05-28: 모든 model HP default=20 통일 (사용자 명시 "HP trial default 동일 반영").
# 비워두면 _optuna_budget.get_trials(name, default=20) 적용.
# export MPH_OPTUNA_TRIALS_JSON='{"XGBoost": 20, "LightGBM": 20, "CatBoost": 20, ...}'

# 2026-06-05 (사용자 명시): 4-criteria/g175 filter 완전 제거 — champion = best-WIS.
# MPH_R2_FLOOR/MAPE_CEILING/WIS_CEILING/PICP95_FLOOR export 폐지 (코드 미소비).

# ── audit Stage 1-4 (2026-05-27, 사용자 명시 "1,2,3까지") ──
# Per-metric bootstrap CI (paper-grade; 4-criteria/g175 제거 2026-06-05 — champion=best-WIS)
export MPH_CI_BOOTSTRAP_N=1000
export MPH_CI_BOOTSTRAP_SEED=42
# Stage 2.3 hierarchical champion gate (alert + lead_time + PICP CI lower)
export MPH_ALERT_F1_FLOOR=0.6
export MPH_LEAD_TIME_FLOOR=1.0
# Stage 3.3 audit chain (champion gate + MCS_90; g175 strict 제거 2026-06-05 — champion=best-WIS)
export MPH_AUDIT_USE_CHAMPION_GATE=1
export MPH_AUDIT_USE_MCS=1
# Stage 3.1 multi-seed stability — ⚠ 5-seed 학습은 아직 미배선(post-cascade): =1 이어도 phase13 은
#   단일 seed 42 + manifest 로그만(분기 per_model_optimize.py:2966-, "full multi-seed integration is
#   post-cascade"). 라벨 정직성(3자 감사 #5, 2026-06-18): =1 이 5-seed 를 암시하나 실제는 단일-seed →
#   misleading. =0 으로 정정(동작 동일=단일 seed, 오해 제거). 논문 재현성 절에 multi-seed 주장 금지.
#   실제 5-seed robustness 가 필요하면 get_seed_list 루프 배선 후 =1 (학습 5× — 별도 작업).
export MPH_MULTI_SEED_RUN=0

# ════════════════════════════════════════════════════════════════
# --smoke handler: 빠른 기능 점검 (전체 phase 경로, 최소 compute, 격리 출력)
# 사용자 명시 (2026-05-29): 풀런 전 "기능 안 깨지는지" ~몇 분 확인용.
# 격리: MPH_OUTPUT_ROOT 로 results/models/cache 를 throwaway dir 로 리디렉트
#       → 운영 simulation/results 무오염. DB(data/db)는 redirect 안 됨 → 기존 데이터 읽기(수집 skip).
# ════════════════════════════════════════════════════════════════
if [ "$SMOKE_MODE" = "1" ]; then
    # valid_test/ consolidation (2026-06-05): smoke 산출물을 안정 폴더 valid_test/smoke 로
    #   모아 real simulation/results 와 혼동 방지 (+ /tmp 흩어짐 제거). 매 run fresh.
    SMOKE_OUT="$SCRIPT_DIR/valid_test/smoke"
    rm -rf "$SMOKE_OUT"
    export MPH_OUTPUT_ROOT="$SMOKE_OUT"
    export MPH_PREPROC_TRIALS=2
    # export MPH_FEATURE_OPTUNA_TRIALS=2   # G-297: DEAD CONFIG (Stage-2 = stability, not Optuna)
    export MPH_HP_OPTUNA_TRIALS=2
    export MPH_MULTI_SEED_RUN=0          # 5-seed off (smoke = 단일)
    export MPH_SKIP_OPTUNA_CLEANUP=1     # 운영 optuna db 미접근
    SKIP_EVAL_FLAG=1                     # 자동 평가 chain skip
    SMOKE_MODELS="${MPH_SMOKE_MODELS:-ElasticNet,ARIMA,XGBoost}"
    # 무오염 백업망: 4 sub-component(eval_logs/stage2_feature_optuna/phase12_elife/
    # statistical_audit)는 2026-05-29 근본수정으로 get_results_dir() (MPH_OUTPUT_ROOT 존중)
    # 경유 → 더 이상 simulation/results 에 하드코딩 안 함. manifest-diff 는 미확인/신규 누수
    # 대비 backstop 으로 유지(사용자 명시). → run 전 스냅샷, run 후 신규 누수만 제거(report 블록).
    mkdir -p "$SMOKE_OUT"
    SMOKE_MANIFEST="$SMOKE_OUT/_prod_results_manifest.txt"
    find simulation/results -type f 2>/dev/null | sort > "$SMOKE_MANIFEST" || true
    echo "════════════════════════════════════════════════════════"
    echo "  SMOKE MODE — 기능 점검 (주요 산출물 격리 + 누수분 자동정리)"
    echo "  출력 격리:  $SMOKE_OUT"
    echo "  모델:       $SMOKE_MODELS (CPU 위주, trial 2, 단일 seed)"
    echo "  수집 skip(기존 DB) · 평가 chain skip · 예상 ~30분"
    echo "════════════════════════════════════════════════════════"
fi

# Optuna study 누적 정리 (G-143 fix) — 학습 시작 전 ≥100 trials 자동 삭제
if [ "${MPH_SKIP_OPTUNA_CLEANUP:-0}" != "1" ]; then
    echo "[$(date)] Optuna study 누적 정리 (≥100 trials)"
    .venv/bin/python -m simulation.scripts.cleanup_optuna_studies \
        --threshold 100 2>&1 | tail -10
fi

# Preflight 검증 (G-158)
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    if ! bash scripts/preflight_check.sh; then
        echo ""
        echo "✗ Preflight check FAIL — 학습 시작 차단 (G-158)"
        echo "  우회: SKIP_PREFLIGHT=1 bash run_resume_phase12.sh"
        exit 1
    fi
fi

# ════════════════════════════════════════════════════════════════
# --force handler: checkpoint clear (run_training.sh 흡수)
# ════════════════════════════════════════════════════════════════
if [ "$FORCE_MODE" = "1" ]; then
    echo "[$(date)] --force: Phase checkpoint 모두 삭제 + Phase 0 재학습"
    PHASE_CKPT_DIR="simulation/results/checkpoints"
    if [ -d "$PHASE_CKPT_DIR" ]; then
        BAK_TS="$(date +%Y%m%d_%H%M%S)"
        find "$PHASE_CKPT_DIR" -maxdepth 1 -name "checkpoint_phase*.json" \
            -exec mv {} {}.bak_${BAK_TS} \; 2>/dev/null || true
        echo "  ✓ Phase checkpoint 백업 + 무효화 (.bak_${BAK_TS})"
    fi
fi

# Phase 1 invalidation 정책 (G-150 fix, 2026-05-01)
if [ "${MPH_FORCE_PHASE1_REGEN:-0}" = "1" ]; then
    echo "[$(date)] MPH_FORCE_PHASE1_REGEN=1 → Phase 1 강제 재생성"
    if [ -f "simulation/results/checkpoints/checkpoint_phase1.json" ]; then
        mv simulation/results/checkpoints/checkpoint_phase1.json \
           simulation/results/checkpoints/checkpoint_phase1.json.bak_$(date +%Y%m%d_%H%M%S)
        echo "  Phase 1 checkpoint 백업 + 무효화"
    fi
    find simulation/results -maxdepth 3 -name "fe_cache_*.parquet" -delete 2>/dev/null
    echo "  fe_cache_*.parquet 정리"
else
    echo "[$(date)] Resume mode: Phase 1 checkpoint + fe_cache 보존 (G-150)"
fi

# ════════════════════════════════════════════════════════════════
# --dry-run handler: 검증 + 실행 명령 표시, 학습 시작 X
# ════════════════════════════════════════════════════════════════
if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  DRY RUN — 환경 검증 통과. 실제 학습 명령:"
    echo "════════════════════════════════════════════════════════"
    echo "  .venv/bin/python -m simulation train \\"
    echo "      --resume-from ${RESUME_FROM} \\"
    echo "      --scenario full \\"
    echo "      --per-model-optimize \\"
    echo "      --weather-mode hybrid \\"
    echo "      --conformal-method aci \\"
    echo "      --ensemble-method stacking \\"
    echo "      --covid-mode indicator \\"
    echo "      --auto-collect --stale-days 1 --collect-groups all"
    echo ""
    echo "  학습 시작 X (--dry-run)"
    exit 0
fi

# ════════════════════════════════════════════════════════════════
# 학습 실행
# ════════════════════════════════════════════════════════════════
# Must agree with paths.fast_tmp (simulation/config_global.py), which is the
# SSOT for the temp directory: MPH_FAST_TMPDIR, else tempfile.gettempdir().
# The readers of this log — progress.py, training_health.py and
# parse_per_model_resources.py — all discover it by globbing that directory, so
# a literal /tmp here would hide the log from them on macOS (private per-user
# TMPDIR) and on Windows.
MPH_TMP="${MPH_FAST_TMPDIR:-${TMPDIR:-/tmp}}"
LOG_FILE="${MPH_TMP%/}/training_resume_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] 학습 시작 → $LOG_FILE"
echo "[$(date)] 옵션: force=$FORCE_MODE clean=$CLEAN_MODE resume_from=$RESUME_FROM skip_eval=$SKIP_EVAL_FLAG"

if [ "$SMOKE_MODE" = "1" ]; then
    # 기능 smoke: 수집 skip(기존 DB) + 모델 부분집합 + lite + 격리 출력
    set +e
    .venv/bin/python -m simulation train \
        --resume-from 0 \
        --scenario full \
        --lite \
        --models "$SMOKE_MODELS" \
        --per-model-optimize \
        --weather-mode hybrid \
        --conformal-method aci \
        --ensemble-method stacking \
        --covid-mode indicator \
        > "$LOG_FILE" 2>&1
    TRAIN_RC=$?
    set -e
else
    # ── Collect 플래그 env-gate (2026-06-03): MPH_SKIP_COLLECT=1 → 수집 생략(기존 DB 사용).
    #    HIRA 가 ILI 무관 법정감염병 ~80종 전수 수집(1h+) → ILI 학습/디버깅 시 skip. 기본=수집(back-compat).
    COLLECT_ARGS="--auto-collect --stale-days 1 --collect-groups all"
    [ "${MPH_SKIP_COLLECT:-}" = "1" ] && { COLLECT_ARGS=""; echo "  [run_pipeline] MPH_SKIP_COLLECT=1 → 수집 생략(기존 DB 사용)"; }
    # Model subset env-gate (2026-06-04): MPH_MODELS="m1,m2,…" → --models 로 해당 모델만 학습
    #   (전체 phase 동일 실행, 모델집합만 제한). 빠른 부분완주용.
    # G-313 (2026-06-18): MPH_MODELS 미설정 시 활성 CATEGORY_MODELS(53)를 자동 적용. phase-13
    #   (per_model_optimize)는 --models 가 없으면 REGISTRY.get_all()=66(은퇴/deferred 13개 포함)을
    #   학습하므로 `--scenario full` 만으로는 53 으로 제한되지 않는다(메모리 검증). 기본을 활성
    #   라인업으로 못박아 cuts 가 재학습에 섞이는 사고를 차단.
    if [ -z "${MPH_MODELS:-}" ]; then
        MPH_MODELS=$(.venv/bin/python -c "from simulation.models.registry import CATEGORY_MODELS as c; print(','.join(m for v in c.values() for m in v))" 2>/dev/null)
        if [ -n "${MPH_MODELS:-}" ]; then
            echo "  [run_pipeline] G-313 MPH_MODELS 미설정 → 활성 CATEGORY_MODELS $(echo "$MPH_MODELS" | tr ',' '\n' | grep -c .)개 자동 적용"
        else
            echo "  [run_pipeline] ⚠ G-313 활성 리스트 추출 실패 → REGISTRY 전체(66) 학습됨"
        fi
    fi
    MODELS_ARG=""
    [ -n "${MPH_MODELS:-}" ] && { MODELS_ARG="--models ${MPH_MODELS}"; echo "  [run_pipeline] 학습 모델집합: $(echo "$MPH_MODELS" | tr ',' '\n' | grep -c .)개"; }
    .venv/bin/python -m simulation train \
        --resume-from "$RESUME_FROM" \
        --scenario full \
        $MODELS_ARG \
        --per-model-optimize \
        --weather-mode hybrid \
        --conformal-method aci \
        --ensemble-method stacking \
        --covid-mode indicator \
        $COLLECT_ARGS \
        > "$LOG_FILE" 2>&1
    TRAIN_RC=$?
fi

echo "[$(date)] 학습 완료 (rc=$TRAIN_RC)" >> "$LOG_FILE"

# --smoke 결과 요약 후 종료 (운영 평가 chain 진입 X)
if [ "$SMOKE_MODE" = "1" ]; then
    # 무오염 복원: MPH_OUTPUT_ROOT 미적용(하드코딩) sub-component 가 simulation/results 에
    # 새로 쓴 파일을 manifest diff 로 제거 (INDEX.csv 같은 append 수정분은 잔존 가능).
    LEAK_N=0
    if [ -f "$SMOKE_MANIFEST" ]; then
        find simulation/results -type f 2>/dev/null | sort > "${SMOKE_MANIFEST}.after" || true
        while IFS= read -r f; do
            if [ -n "$f" ]; then rm -f "$f"; LEAK_N=$((LEAK_N+1)); fi
        done < <(comm -13 "$SMOKE_MANIFEST" "${SMOKE_MANIFEST}.after")
    fi
    echo ""
    echo "════════════════════════════════════════════════════════"
    if [ "$TRAIN_RC" = "0" ]; then
        echo "  ✓ SMOKE PASS — 전체 phase 경로 정상 (rc=0)"
    else
        echo "  ✗ SMOKE FAIL — rc=$TRAIN_RC → 로그 마지막 50줄:"
        tail -50 "$LOG_FILE"
    fi
    echo "  격리 출력:  ${MPH_OUTPUT_ROOT}"
    echo "  운영 복원:  simulation/results 신규 누수 ${LEAK_N} 파일 제거 (manifest diff)"
    echo "  로그:       $LOG_FILE"
    echo "  정리:       rm -rf \"${MPH_OUTPUT_ROOT}\""
    echo "════════════════════════════════════════════════════════"
    exit "$TRAIN_RC"
fi

# ════════════════════════════════════════════════════════════════
# 자동 평가 chain (--skip-eval / SKIP_EVAL 둘 다 미적용 시)
# ════════════════════════════════════════════════════════════════
# G-361 (2026-06-25, 사용자): audit_and_retrain auto-chain 기본 비활성.
#   audit_problem_models 가 test_r2<0.8(=hold-out 68주 외삽 artifact)로 severity 분류 →
#   챔피언(FusedEpi, rel-WIS 1위)까지 "severe 재학습" 오분류 + 현재(이미 fix 적용) 코드로
#   재학습 = v2≈v1 중복 → 수시간 낭비. 파이프라인은 Pipeline Complete 에서 종료.
#   opt-in: MPH_AUTO_AUDIT_RETRAIN=1, 또는 수동 `bash scripts/audit_and_retrain.sh`.
if [ "$TRAIN_RC" = "0" ] && [ "${SKIP_EVAL:-0}" != "1" ] && [ "$SKIP_EVAL_FLAG" = "0" ] \
   && [ "${MPH_AUTO_AUDIT_RETRAIN:-0}" = "1" ]; then
    echo "[$(date)] 학습 성공 — 자동 평가 chain 시작 (MPH_AUTO_AUDIT_RETRAIN=1)" >> "$LOG_FILE"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  자동 평가 chain — audit + retrain + compare + statistical"
    echo "════════════════════════════════════════════════════════"
    bash scripts/audit_and_retrain.sh --no-regen 2>&1 | tee -a "$LOG_FILE"
    EVAL_RC=$?
    if [ "$EVAL_RC" = "0" ]; then
        echo "[$(date)] ✓ 자동 평가 완료" >> "$LOG_FILE"
        echo ""
        echo "════════════════════════════════════════════════════════"
        echo "  ✓ 학습 + 평가 chain 완료"
        echo "════════════════════════════════════════════════════════"
        echo "  결과:"
        echo "    학습 로그:        $LOG_FILE"
        echo "    Audit:            simulation/results/AUTO_AUDIT_LATEST.md"
        echo "    Compare v1↔v2:    simulation/results/v1_vs_v2_comparison.md"
        echo "    Statistical:      simulation/results/STATISTICAL_AUDIT.md"
    else
        echo "[$(date)] ⚠ 평가 chain 실패 (rc=$EVAL_RC)" >> "$LOG_FILE"
    fi
elif [ "$TRAIN_RC" != "0" ]; then
    echo "[$(date)] ✗ 학습 실패 (rc=$TRAIN_RC) — 평가 chain skip" >> "$LOG_FILE"
else
    echo "[$(date)] 평가 chain skip (--skip-eval / SKIP_EVAL=1 / MPH_AUTO_AUDIT_RETRAIN≠1) — Pipeline Complete 에서 종료" >> "$LOG_FILE"
fi

# ════════════════════════════════════════════════════════════════
# M4 (2026-06-06): web 데이터 재생성 (db→web 실시간 동기화).
#   학습 성공 + 비-smoke 시, live 산출(per_model_eval/real_eval/DB)에서 web
#   aggregate(trained-models·seir-init·choropleth)를 재생성 → web 동결/불일치 제거.
#   degrade-and-continue (web builder 실패가 학습 결과를 무효화하지 않음).
# ════════════════════════════════════════════════════════════════
if [ "$SMOKE_MODE" != "1" ] && [ "$TRAIN_RC" = "0" ]; then
    echo "[$(date)] web 데이터 재생성 (db→web sync)" >> "$LOG_FILE"
    .venv/bin/python web/scripts/refresh_web_data.py 2>&1 | tee -a "$LOG_FILE" \
        || echo "[$(date)] ⚠ web refresh 일부 실패 (계속 진행)" >> "$LOG_FILE"
fi
