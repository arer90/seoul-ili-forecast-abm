"""Global config — 단일 source-of-truth (ENGINEERING_PRINCIPLES.md §원칙 #4).

모든 env var / 기본 thresholds / paths / 핵심 user input 을 한 곳에 모음.
- Default 모두 정의 (없으면 안전한 default 사용)
- env var override 지원 (운영 toggle)
- Type-safe (dataclass frozen)
- Validation (잘못된 값이면 즉시 raise)
- 모든 모듈은 `from simulation.config_global import GLOBAL` 만 사용

사용 (현행 코드):
    from simulation.config_global import GLOBAL
    if GLOBAL.training.use_3stage:
        ...
    pruner = GLOBAL.optuna.pruner_name  # "hyperband"

env var override:
    MPH_BEST_BY=oof_cv .venv/bin/python -m simulation train ...
    → GLOBAL.training.best_by == "oof_cv"   (env 로 default override 가능)

호환성:
    각 모듈은 점진적으로 GLOBAL 로 마이그레이션. 기존 os.environ.get() 도 OK.
    GLOBAL 은 초기 import 시 1회 로드 (immutable, frozen).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ════════════════════════════════════════════════════════════════
# ENV var helpers (default + 타입 변환)
# ════════════════════════════════════════════════════════════════

def _env_bool(key: str, default: bool) -> bool:
    """환경변수 → bool. '1'/'true'/'yes' = True, 그 외 = default."""
    raw = os.environ.get(key, "").lower().strip()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(key: str, default: int, lo: int = -10**9, hi: int = 10**9) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        if v < lo or v > hi:
            return default
        return v
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float, lo: float = -1e18, hi: float = 1e18) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        if v < lo or v > hi:
            return default
        return v
    except (ValueError, TypeError):
        return default


def _env_str(key: str, default: str, choices: tuple[str, ...] | None = None) -> str:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    if choices and raw.lower() not in [c.lower() for c in choices]:
        return default
    return raw


def _env_set(key: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """comma-separated → tuple."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    return tuple(s.strip() for s in raw.split(",") if s.strip())


# ════════════════════════════════════════════════════════════════
# Section A — Training control
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TrainingConfig:
    """학습 흐름 / 3-Stage / preset / family-first 설정."""

    use_3stage: bool = field(default_factory=lambda: _env_bool("MPH_USE_3STAGE", True))
    """3-Stage Pipeline (preproc → feature → HP) 자동 로드."""

    model_aware_preproc: bool = field(
        default_factory=lambda: _env_bool("MPH_MODEL_AWARE_PREPROC", True)
    )
    """모델별 (transforms_y × scalers_x) menu 차별화."""

    grouped_preproc: bool = field(
        default_factory=lambda: _env_bool("MPH_GROUPED_PREPROC", True)
    )
    """ColumnTransformer 그룹별 fallback (Tier 1+2 안전장치)."""

    stable_transforms: bool = field(
        default_factory=lambda: _env_bool("MPH_STABLE_TRANSFORMS", True)
    )
    """Y target identity/log1p 만 — yeo_johnson 발산 방지 (G-133)."""

    advanced_features: bool = field(
        default_factory=lambda: _env_bool("MPH_ADVANCED_FEATURES", True)
    )
    """Hilbert/EMD/FFT 등 12 카테고리 + sanitize."""

    feature_family_first: bool = field(
        default_factory=lambda: _env_bool("MPH_FEATURE_FAMILY_FIRST", True)
    )
    """Family-first feature selection (G-138 — 2^321 → ~90 keys)."""

    drop_high_vif: bool = field(
        default_factory=lambda: _env_bool("MPH_DROP_HIGH_VIF", False)
    )
    """High-VIF lag features 제거 (NegBinGLM AR 신호 손실 우려 — default OFF)."""

    use_pca: bool = field(default_factory=lambda: _env_bool("MPH_USE_PCA", False))

    best_by: Literal["oof_cv", "val"] = field(
        default_factory=lambda: _env_str("MPH_BEST_BY", "oof_cv", ("oof_cv", "val"))
    )
    """Best 결정 기준 — n=27 val single 거절 (G-132)."""

    oof_folds: int = field(
        default_factory=lambda: _env_int("MPH_OOF_FOLDS", 5, lo=2, hi=10)
    )
    """OOF WF-CV fold 수 (2026-05-31 사용자 명시: 학위논문 제출용 기본 **5** = paper-grade;
    이전 3). 3-vs-5 TDD: transform 선택 동일, 5 가 약간 보수적. `--oof-folds` / `MPH_OOF_FOLDS`."""

    preset: Literal["conservative", "aggressive", "production"] = field(
        default_factory=lambda: _env_str(
            "MPH_PRESET", "production",
            ("conservative", "aggressive", "production"),
        )
    )

    preproc_optuna: bool = field(
        default_factory=lambda: _env_bool("MPH_PREPROC_OPTUNA", False)
    )
    """R9(per_model_optimize) 안에서 preproc Optuna 추가 (default OFF, 3-Stage 가 흡수)."""

    preproc_trials: int = field(
        default_factory=lambda: _env_int("MPH_PREPROC_TRIALS", 30, lo=1, hi=200)
    )

    multicollinearity: Literal["none", "vif", "corr", "pca", "auto"] = field(
        default_factory=lambda: _env_str(
            "MPH_MULTICOLLINEARITY", "none", ("none", "vif", "corr", "pca", "auto")
        )
    )
    """R9(per_model_optimize) multicollinearity filter 방법 (G-232). 이전 3곳 os.getenv("none") 중복.
    "auto" = R9 4-method 자동 비교 후 최적 선택 (G-234). _env_str 가 invalid 시
    default 로 떨어지므로 "auto" 를 choices 에 포함해야 silent drop 방지."""

    max_epochs_override: int = field(
        default_factory=lambda: _env_int("MPH_MAX_EPOCHS", 0, lo=0, hi=100000)
    )
    """DL 최대 epoch override. 0 = 모델별 default 사용 (이전 빈 문자열 sentinel 대체)."""

    # multicollinearity filter tuning — mc_filter_stage3._get_* helpers SSOT (Tier2 2026-05-28)
    vif_threshold: float = field(
        default_factory=lambda: _env_float("MPH_VIF_THRESHOLD", 10.0, lo=1.0, hi=1e6)
    )
    corr_threshold: float = field(
        default_factory=lambda: _env_float("MPH_CORR_THRESHOLD", 0.9, lo=0.0, hi=1.0)
    )
    pca_variance: float = field(
        default_factory=lambda: _env_float("MPH_PCA_VARIANCE", 0.95, lo=0.0, hi=1.0)
    )
    vif_max_iter: int = field(
        default_factory=lambda: _env_int("MPH_VIF_MAX_ITER", 50, lo=1, hi=10000)
    )

    # ── training-loop knobs (Tier2 2026-05-28 SSOT) ──
    advanced_enabled: str = field(
        default_factory=lambda: _env_str("MPH_ADVANCED_ENABLED", "")
    )
    """advanced feature 카테고리 whitelist (comma-sep, ""=all). builder.py SSOT."""

    lightning_max_time_per_model: int = field(
        default_factory=lambda: _env_int("MPH_LIGHTNING_MAX_TIME_PER_MODEL", 300, lo=10, hi=86400)
    )
    """Lightning final-fit per-model timeout 초 (G-152). run script 1800/900 override."""

    hier_max_chain: int = field(
        default_factory=lambda: _env_int("MPH_HIER_MAX_CHAIN", 2, lo=1, hi=10)
    )
    """hierarchical preproc chain 최대 길이 (_inline_optuna_3stage)."""

    fast_train: bool = field(default_factory=lambda: _env_bool("MPH_FAST_TRAIN", False))
    """빠른 학습 모드 (runner — 축소 trial/epoch)."""

    multi_seed_run: bool = field(default_factory=lambda: _env_bool("MPH_MULTI_SEED_RUN", False))
    """multi-seed 반복 학습 (analytics.multi_seed)."""

    seed_list: tuple[int, ...] = field(
        default_factory=lambda: tuple(
            int(x) for x in _env_set("MPH_SEED_LIST", ("13", "42", "137", "1729", "31415"))
        )
    )
    """multi-seed seed 목록 SSOT (env MPH_SEED_LIST=13,42,... override).
    default = arbitrary wide-magnitude values (external audit 2026-05-27 — not all prime).
    analytics.multi_seed.SEED_LIST_DEFAULT 가 이 필드를 참조 (단일 source)."""


# ════════════════════════════════════════════════════════════════
# Section B — Multi-criteria filter (R9 per_model_optimize best decision)
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class FilterConfig:
    """Filter / gate config. 2026-06-05 (사용자 명시): 4-criteria/g175 filter 완전 제거 —
    champion = 순수 best-WIS. r2_floor/mape_ceiling/wis_ceiling/picp95_floor/alpha_floor 삭제.
    r2_catastrophic_cutoff(Optuna pruning) + alert_threshold + ci_bootstrap_* 만 잔존."""

    r2_catastrophic_cutoff: float = field(
        default_factory=lambda: _env_float("MPH_R2_CATASTROPHIC_CUTOFF", -1e6, lo=-1e18, hi=0.0)
    )
    """Optuna trial pruning R² 폭주 cutoff (첫 fold R²<cutoff → prune).
    tree/linear/epi models 5곳 멀티라인 os.getenv 통일 (R8 re-audit 2026-05-28 — 이전
    single-line grep 이 멀티라인 getenv 를 놓침)."""

    full_eval_trajectory: bool = field(
        default_factory=lambda: _env_bool("MPH_FULL_EVAL_TRAJECTORY", True)
    )
    """134-metric evaluator 전체 계산 여부 (False=minimal skip). 이전 4곳 os.getenv("1") 중복.
    주의: MPH_FAST_METRIC=1 alias 는 read 측에서 추가 AND 조건으로 처리."""

    alert_threshold: float = field(
        default_factory=lambda: _env_float("MPH_ALERT_THRESHOLD", 8.6, lo=0.0, hi=1000.0)
    )
    """KDCA ILI alert threshold (2024-25 기본 8.6/1000). 이전 13개 phase 가
    evaluate_predictions_full(threshold=8.6) 하드코딩 → SSOT 통일.
    절기별 값은 real_eval._kdca_threshold_for 별도 유지 (season-aware)."""

    alert_f1_floor: float = field(
        default_factory=lambda: _env_float("MPH_ALERT_F1_FLOOR", 0.6, lo=0.0, hi=1.0)
    )
    """R10(per_model_eval) champion gate — alert F1 minimum (PAPER_TOP_3 metric)."""

    lead_time_floor: float = field(
        default_factory=lambda: _env_float("MPH_LEAD_TIME_FLOOR", 1.0, lo=0.0, hi=52.0)
    )
    """R10(per_model_eval) champion gate — lead-time weeks minimum (PAPER_TOP_3 metric)."""

    champion_min_count: int = field(
        default_factory=lambda: _env_int("MPH_CHAMPION_MIN_COUNT", 2, lo=1, hi=68)
    )
    """R10(per_model_eval) 최소 champion 후보 수 (이 미만이면 gate relax warning)."""

    ci_bootstrap_n: int = field(
        default_factory=lambda: _env_int("MPH_CI_BOOTSTRAP_N", 500, lo=0, hi=100_000)
    )
    """Per-metric bootstrap CI resample 횟수 (진단; champion=best-WIS)."""

    ci_bootstrap_seed: int = field(
        default_factory=lambda: _env_int("MPH_CI_BOOTSTRAP_SEED", 42)
    )
    """Per-metric bootstrap CI RNG seed (재현성, #5)."""

    phase14_research_mode: bool = field(
        default_factory=lambda: _env_bool("MPH_PHASE14_RESEARCH_MODE", False)
    )
    """R10(per_model_eval) research mode (2026-05-28 design A: R10 = research용)."""

    fast_metric: bool = field(default_factory=lambda: _env_bool("MPH_FAST_METRIC", False))
    """phase_evaluator fast path (MPH_FULL_EVAL_TRAJECTORY 의 legacy alias).
    read 측에서 full_eval_trajectory 와 OR 조건으로 결합."""

    use_val_test_gap: bool = field(default_factory=lambda: _env_bool("MPH_USE_VAL_TEST_GAP", False))
    """audit: val↔test gap 진단 사용 (G-156)."""


# ════════════════════════════════════════════════════════════════
# Section C — Optuna 표준
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class OptunaConfig:
    """Optuna sampler / pruner / objective / 격리 설정."""

    objective: Literal["wis", "rmse", "mae", "crps"] = field(
        default_factory=lambda: _env_str(
            "OPTUNA_OBJECTIVE", "wis", ("wis", "rmse", "mae", "crps")
        )
    )
    """FluSight 표준 = WIS."""

    sampler: Literal["best", "tpe-mv", "tpe", "cma", "gp", "nsga2"] = field(
        default_factory=lambda: _env_str(
            "OPTUNA_SAMPLER", "best", ("best", "tpe-mv", "tpe", "cma", "gp", "nsga2")
        )
    )
    """'best' = per-model 자동 (TPE-MV/CMA/GP/NSGA2)."""

    pruner_name: Literal["hyperband", "halving", "median", "none"] = field(
        default_factory=lambda: _env_str(
            "MPH_PRUNER", "hyperband", ("hyperband", "halving", "median", "none")
        )
    )
    """전역 override — 비어 있으면 모델별 default (`_optuna_pruners.py`)."""

    isolate: bool = field(default_factory=lambda: _env_bool("OPTUNA_ISOLATE", True))
    """Trial-level subprocess 격리 (메모리 누적 0)."""

    verbose: bool = field(default_factory=lambda: _env_bool("OPTUNA_VERBOSE", False))

    storage: str = field(
        default_factory=lambda: _env_str("MPH_OPTUNA_STORAGE", "")
    )
    """Optuna study DB URL (sqlite:// 등). 빈 문자열 = in-memory."""

    use_full: bool = field(
        default_factory=lambda: _env_bool("MPH_OPTUNA_USE_FULL", False)
    )

    force: bool = field(default_factory=lambda: _env_bool("MPH_OPTUNA_FORCE", False))

    remaining_cap: int = field(
        default_factory=lambda: _env_int("MPH_OPTUNA_REMAINING_CAP", 200, lo=0, hi=10000)
    )
    """Optuna warm-start trial cap per call (일반/tree/feature).
    주의 (R8 re-audit 2026-05-28): reader = `min(cap, N_TRIALS); if remaining>0: optimize`
    → **cap=0 = trial 0개(비활성)**, "no cap" 아님. uncapped 원하면 cap≥N_TRIALS.
    SSOT 통일: 이전 dead field default 0 → 실사용값 200 (dl/feature read 와 일치)."""

    dnn_remaining_cap: int = field(
        default_factory=lambda: _env_int("MPH_OPTUNA_DNN_REMAINING_CAP", 25, lo=0, hi=10000)
    )
    """DNN(torch) warm-start trial cap — trial 비용 큼 → 일반(200)보다 작게(25).
    이전 _optuna_torch 가 os.getenv default 25 로 하드코딩하던 값 (context-specific 보존)."""

    use_storage: bool = field(
        default_factory=lambda: _env_bool("MPH_OPTUNA_USE_STORAGE", True)
    )
    """Optuna study 영속 storage 사용 여부.
    SSOT 통일 (2026-05-28): 이전 MPH_OPTUNA_STORAGE 가 storage URL(`storage`)과
    bool flag("1"/"0") 로 이름 충돌 → flag 를 MPH_OPTUNA_USE_STORAGE 로 분리. False = in-memory."""

    # ── sampler / study 식별 (Tier2 2026-05-28 SSOT) ──
    hp_space: str = field(default_factory=lambda: _env_str("MPH_HP_SPACE", "full"))
    """DNN HP search space 크기 (_optuna_samplers): full/medium/small."""

    small_sample_cap: bool = field(
        default_factory=lambda: _env_bool("MPH_SMALL_SAMPLE_CAP", False)
    )
    """small-n 시 trial space 축소 (_optuna_samplers)."""

    phase_tag: str = field(default_factory=lambda: _env_str("MPH_OPTUNA_PHASE_TAG", ""))
    """study 이름 phase tag suffix (_optuna_torch)."""

    study_suffix: str = field(default_factory=lambda: _env_str("MPH_STUDY_SUFFIX", "v16"))
    """feature-selection study warm-start 버전 suffix (run_optuna_feature_selection)."""

    hp_trials_default: int = field(
        default_factory=lambda: _env_int("MPH_HP_OPTUNA_TRIALS", 0, lo=0, hi=100000)
    )
    """Stage 3 model-internal HP trials (_optuna_budget; 0 = caller/모델 default)."""


# ════════════════════════════════════════════════════════════════
# Section D — Package C (Speed + Accuracy 패치)
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PackageCConfig:
    """Package C 가속/정확도 옵션."""

    tier_budget: bool = field(
        default_factory=lambda: _env_bool("MPH_PC_TIER_BUDGET", True)
    )
    """Tier-stratified search budget (Tier C: 10 trials cap)."""

    autocast: bool = field(default_factory=lambda: _env_bool("MPH_PC_AUTOCAST", False))
    """Mixed precision (autocast) for DL 모델.
    R8 (2026-05-28): default True→False. CUDA fp16 autocast 는 GradScaler 통합 +
    학습 A/B 검증이 필요 (현재 학습 미실행) → env-gated OFF. MPH_PC_AUTOCAST=1 로 opt-in.
    배선 시 GradScaler 추가 필수 (package_c_autocast_ctx 는 ctx 만 제공)."""

    compile_models: bool = field(
        default_factory=lambda: _env_bool("MPH_PC_COMPILE", False)
    )
    """torch.compile wrap for DL 모델 (package_c_compile_helper 로 _train_loop 배선됨).
    R8 (2026-05-28): default True→False (env-gated, behavior-neutral merge).
    MPH_PC_COMPILE=1 로 opt-in 시 CUDA max-autotune / MPS reduce-overhead."""

    tier_c_extra: tuple[str, ...] = field(
        default_factory=lambda: _env_set("MPH_PC_TIER_C", ())
    )
    """추가 Tier C 모델 (default Tier C 외)."""

    softmin_temperature: float = field(default=0.5)
    """MCS-temperature stacking T 값."""

    mondrian_min_per_group: int = field(default=5)
    """Mondrian conformal 그룹 최소 sample 수."""

    # G-218: huber 영구 제거 (huber-loss-banned-20260520) — mse/mae/pinball 만
    loss_menu: Literal["mse", "mae", "pinball"] = field(
        default_factory=lambda: _env_str(
            "MPH_PC_LOSS", "mse", ("mse", "mae", "pinball")
        )
    )

    tier_a_trials: int = field(default=50)
    tier_b_trials: int = field(default=30)
    tier_c_trials: int = field(default=10)


# ════════════════════════════════════════════════════════════════
# Section E — Resources (디바이스 / n_jobs / 메모리)
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ResourceConfig:
    """Hardware / OS / 재현성 시드."""

    seed: int = field(default=42)
    """np.random.seed + torch.manual_seed."""

    n_jobs_max: int = field(default=2)
    """n_jobs 상한 — G-039, G-049 (절대 -1 금지)."""

    device_override: str = field(
        default_factory=lambda: _env_str(
            "MPH_DEVICE", "", ("", "cpu", "cuda", "mps")
        )
    )
    """비어 있으면 pick_device() 자동 (cuda > mps > cpu)."""

    force_cpu: bool = field(default_factory=lambda: _env_bool("MPH_FORCE_CPU", False))
    """디버깅용 CPU 강제."""

    mps_fallback: bool = field(
        default_factory=lambda: _env_bool("PYTORCH_ENABLE_MPS_FALLBACK", True)
    )
    """MPS 미지원 op → CPU 자동 fallback."""

    @property
    def device(self) -> str:
        """현재 device 결정 — pick_device() 와 일치."""
        if self.force_cpu:
            return "cpu"
        if self.device_override:
            return self.device_override
        try:
            from simulation.models.base import pick_device
            return pick_device()
        except Exception:
            return "cpu"


# ════════════════════════════════════════════════════════════════
# Section F — Paths (DB / 결과 / 캐시)
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PathConfig:
    """단일 경로 source-of-truth."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    """MPH_infection_simulation/ 루트."""

    @property
    def db(self) -> Path:
        return self.project_root / "simulation/data/db/epi_real_seoul.db"

    @property
    def data_root(self) -> Path:
        return self.project_root / "simulation/data"

    @property
    def output_root(self) -> str:
        """raw MPH_OUTPUT_ROOT (stripped, "" = project-local). 이 env key 의 SSOT —
        paths.py / config.py / sanity_check 가 이 property 를 참조 (live read)."""
        return _env_str("MPH_OUTPUT_ROOT", "")

    @property
    def results(self) -> Path:
        if self.output_root:
            return Path(self.output_root)
        return self.project_root / "simulation/results"

    @property
    def cache(self) -> Path:
        return self.project_root / "simulation/cache"

    @property
    def logs(self) -> Path:
        return self.project_root / "simulation/logs"

    @property
    def checkpoints_history(self) -> Path:
        return self.project_root / "simulation/checkpoints_history"

    @property
    def api_keys(self) -> Path:
        return self.project_root / "simulation/data/api_key.txt"

    @property
    def staging(self) -> Path:
        override = _env_str("MPH_STAGING", "")
        if override:
            return Path(override)
        return self.project_root / "simulation/_staging"

    @property
    def fast_tmp(self) -> Path:
        # SSOT (R8 2026-05-28): "/tmp" 리터럴 → tempfile.gettempdir() (Windows 비대응 #1).
        return Path(_env_str("MPH_FAST_TMPDIR", "") or tempfile.gettempdir())


# ════════════════════════════════════════════════════════════════
# Section G — Data split (HWP §3 정합)
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DataSplitConfig:
    """Train / val / test / real / paper_cutoff_week."""

    paper_cutoff_week: int = field(default=337)
    """In-sample = first 337 weeks (HWP §3 표준)."""

    n_train: int = field(default=242)
    n_features: int = field(default=320)
    """전체 feature 수 (DNN HP space param-budget 산정; _optuna_samplers SSOT)."""
    val_size: int = field(default=27)
    test_size: int = field(default=68)
    real_size: int = field(default=8)

    conformal_holdout_weeks: int = field(default=26)
    """ACI / CQR holdout (default for full_light, KUIRB §3)."""

    seoul_n_districts: int = field(default=25)
    """서울 25 자치구 — Metapop SEIR-V-D."""

    overseas_year_min: int = field(
        default_factory=lambda: _env_int("MPH_PHASE18_YEAR_MIN_OVERRIDE", 2019, lo=1990, hi=2100)
    )
    overseas_year_max: int = field(
        default_factory=lambda: _env_int("MPH_PHASE18_YEAR_MAX_OVERRIDE", 2025, lo=1990, hi=2100)
    )
    """Pov(overseas, 구 phase18) KR period 비교 연도 범위 (phase18_overseas SSOT)."""


# ════════════════════════════════════════════════════════════════
# Section H — Logging / 운영
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class OpsConfig:
    """로깅 / clean restart / mlflow."""

    warn_verbose: bool = field(
        default_factory=lambda: _env_bool("MPH_WARN_VERBOSE", False)
    )

    clean_pycache: bool = field(
        default_factory=lambda: _env_bool("MPH_CLEAN_PYCACHE", False)
    )

    mlflow_uri: str = field(default_factory=lambda: _env_str("MPH_MLFLOW_URI", ""))

    mandatory_features_set: str = field(
        default_factory=lambda: _env_str("MPH_MANDATORY_SET", "full")
    )
    """필수 feature set 선택 (run_optuna_feature_selection). default=full(321 features, back-compat).
    Tier2 2026-05-28: canonical "default"→"full" 정정 (read-site 실제 default 와 일치)."""

    # ── 운영 toggle (Tier2 2026-05-28 SSOT) ──
    # 주의: MPH_FAST_TMPDIR 는 paths.fast_tmp(경로) ↔ models.runner(`!="0"` bool) dual-purpose
    #       → bool 필드화 시 _env_bool("/tmp")=False 로 의미 역전. SKIP 유지 (paths.fast_tmp 가 SSOT).
    safe_mode_auto: bool = field(default_factory=lambda: _env_bool("MPH_SAFE_MODE_AUTO", True))
    """R9(per_model_optimize) safe-mode 자동 활성 (default ON)."""

    enable_phase6_reopt: bool = field(
        default_factory=lambda: _env_bool("MPH_ENABLE_PHASE6_REOPT", False)
    )
    """R4(WF-CV) 재-Optuna 활성 (runner)."""

    no_xai: bool = field(default_factory=lambda: _env_bool("MPH_NO_XAI", False))
    """XAI(SHAP) stage skip (runner)."""

    phase4_force_retrain: bool = field(
        default_factory=lambda: _env_bool("MPH_PHASE4_FORCE_RETRAIN", False)
    )
    """R2(baseline) 강제 재학습 (models.runner)."""

    force_redo_phase13: bool = field(
        default_factory=lambda: _env_bool("MPH_FORCE_REDO_PHASE13", False)
    )
    """R9(per_model_optimize) per-model 강제 재실행."""

    disable_eda_sidecar: bool = field(
        default_factory=lambda: _env_bool("MPH_DISABLE_EDA_SIDECAR", False)
    )
    """EDA sidecar 산출 비활성 (eda_writer)."""

    use_cqr: bool = field(default_factory=lambda: _env_bool("MPH_USE_CQR", False))
    """CQR PI 보정 적용 (apply_cqr_pi)."""

    phase18_loro: bool = field(default_factory=lambda: _env_bool("MPH_PHASE18_LORO", False))
    """Pov(overseas, 구 phase18) LORO(Leave-One-Region-Out) CV 활성."""

    audit_use_champion_gate: bool = field(
        default_factory=lambda: _env_bool("MPH_AUDIT_USE_CHAMPION_GATE", False)
    )
    audit_use_mcs: bool = field(
        default_factory=lambda: _env_bool("MPH_AUDIT_USE_MCS", False)
    )
    """audit_problem_models 게이트 toggle 3종 (run script 에서 =1 export)."""


# ════════════════════════════════════════════════════════════════
# 통합 Global config
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GlobalConfig:
    """전체 통합 — 모든 모듈이 이걸 import.

    사용:
        from simulation.config_global import GLOBAL
        if GLOBAL.training.use_3stage: ...
        pruner_name = GLOBAL.optuna.pruner_name
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    package_c: PackageCConfig = field(default_factory=PackageCConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    ops: OpsConfig = field(default_factory=OpsConfig)

    def summary(self) -> str:
        """human-readable 요약 (학습 시작 시 print 권장)."""
        lines = [
            "═══ MPH Global Config ═══",
            f"Training:",
            f"  3-Stage Pipeline:    {self.training.use_3stage}",
            f"  Model-aware preproc: {self.training.model_aware_preproc}",
            f"  Stable transforms Y: {self.training.stable_transforms}",
            f"  Best by:             {self.training.best_by}",
            f"  Preset:              {self.training.preset}",
            f"Champion: best-WIS (4-criteria/g175 제거 2026-06-05)",
            f"Optuna:",
            f"  Objective:           {self.optuna.objective}",
            f"  Sampler:             {self.optuna.sampler}",
            f"  Pruner (override):   {self.optuna.pruner_name}",
            f"  Isolate (subprocess):{self.optuna.isolate}",
            f"Package C:",
            f"  Tier budget:         {self.package_c.tier_budget}",
            f"  Autocast:            {self.package_c.autocast}",
            f"  torch.compile:       {self.package_c.compile_models}",
            f"  Loss menu:           {self.package_c.loss_menu}",
            f"Resources:",
            f"  Seed:                {self.resources.seed}",
            f"  Device:              {self.resources.device}",
            f"  n_jobs max:          {self.resources.n_jobs_max}",
            f"Paths:",
            f"  DB:                  {self.paths.db}",
            f"  Results:             {self.paths.results}",
            f"Data split:",
            f"  paper_cutoff_week:   {self.data_split.paper_cutoff_week}",
            f"  n_train/val/test:    {self.data_split.n_train}/{self.data_split.val_size}/{self.data_split.test_size}",
            f"  conformal_holdout:   {self.data_split.conformal_holdout_weeks}",
            "═════════════════════════",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Statistical constants — symmetric Gaussian PI 분위수 (SSOT, dynamic 2026-05-28)
# ════════════════════════════════════════════════════════════════
# 정규분포 Φ⁻¹ 양측 분위수 (대칭 예측구간 PI half-width 배수). 이전 1.96/1.282/0.6745
# 가 10+ 파일에 산재 → 단일 상수.
# 사용자 명시 (2026-05-28): 하드코딩 리터럴 transcription error 제거 위해 stdlib
# statistics.NormalDist 로 import-time 1회 계산 (정확값). scipy 미사용 — config_global
# 은 stdlib-only SSOT (safety.py device hot-path 가 import → 무거운 의존성 금지).
# behavior: 1.96 → 1.9599639845400534 (PI width Δ ≈ -0.0018%, 무시 가능).
# test 코드의 리터럴 1.96 = test 독립성(D-3) 위해 의도적 미치환.
from statistics import NormalDist as _NormalDist

_STD_NORM = _NormalDist()  # 표준정규 μ=0, σ=1


def z_two_sided(coverage: float) -> float:
    """양측 (1-α) PI 의 정규 분위수 Φ⁻¹(0.5 + coverage/2).

    Args:
        coverage: 양측 명목 커버리지 ∈ (0,1) — 예: 0.95 → 95% PI.

    Returns:
        z multiplier (예: 0.95 → 1.9599639845400534). transcription error 0.
    """
    return _STD_NORM.inv_cdf(0.5 + coverage / 2.0)


Z95: float = z_two_sided(0.95)   # Φ⁻¹(0.975) ≈ 1.95996
Z90: float = z_two_sided(0.90)   # Φ⁻¹(0.95)  ≈ 1.64485
Z80: float = z_two_sided(0.80)   # Φ⁻¹(0.90)  ≈ 1.28155
Z50: float = z_two_sided(0.50)   # Φ⁻¹(0.75)  ≈ 0.67449  (IQR-like)


# ════════════════════════════════════════════════════════════════
# Singleton — import 시 1회 로드
# ════════════════════════════════════════════════════════════════
GLOBAL = GlobalConfig()


# 알려진 모든 env var (참조용 — pre-commit hook 에서 검증 가능)
KNOWN_ENV_VARS: frozenset[str] = frozenset({
    # Training
    "MPH_USE_3STAGE", "MPH_MODEL_AWARE_PREPROC", "MPH_GROUPED_PREPROC",
    "MPH_STABLE_TRANSFORMS", "MPH_ADVANCED_FEATURES", "MPH_FEATURE_FAMILY_FIRST",
    "MPH_DROP_HIGH_VIF", "MPH_USE_PCA", "MPH_BEST_BY", "MPH_PRESET",
    "MPH_PREPROC_OPTUNA", "MPH_PREPROC_TRIALS",
    # Multi-criteria filter
    "MPH_R2_FLOOR", "MPH_R2_CATASTROPHIC_CUTOFF", "MPH_MAPE_CEILING", "MPH_WIS_CEILING",
    "MPH_PICP95_FLOOR", "MPH_ALPHA_FLOOR",
    # Optuna
    "OPTUNA_OBJECTIVE", "OPTUNA_SAMPLER", "MPH_PRUNER", "OPTUNA_ISOLATE",
    "OPTUNA_VERBOSE", "MPH_OPTUNA_STORAGE", "MPH_OPTUNA_USE_FULL",
    "MPH_OPTUNA_FORCE", "MPH_OPTUNA_REMAINING_CAP",
    # Package C
    "MPH_PC_TIER_BUDGET", "MPH_PC_AUTOCAST", "MPH_PC_COMPILE",
    "MPH_PC_TIER_C", "MPH_PC_LOSS",
    # Resources
    "MPH_DEVICE", "MPH_FORCE_CPU", "PYTORCH_ENABLE_MPS_FALLBACK",
    # Paths / Ops
    "MPH_OUTPUT_ROOT", "MPH_STAGING", "MPH_FAST_TMPDIR",
    "MPH_WARN_VERBOSE", "MPH_CLEAN_PYCACHE", "MPH_MLFLOW_URI",
    "MPH_MANDATORY_SET",
    # ── Tier2/3 SSOT 추가 (2026-05-28) ──
    # Training-loop
    "MPH_ADVANCED_ENABLED", "MPH_FAST_TRAIN", "MPH_MULTI_SEED_RUN", "MPH_SEED_LIST",
    "MPH_HIER_MAX_CHAIN", "MPH_LIGHTNING_MAX_TIME_PER_MODEL", "MPH_MAX_EPOCHS",
    "MPH_MULTICOLLINEARITY", "MPH_VIF_THRESHOLD", "MPH_CORR_THRESHOLD",
    "MPH_PCA_VARIANCE", "MPH_VIF_MAX_ITER",
    # Filter / champion gate
    "MPH_ALERT_THRESHOLD", "MPH_ALERT_F1_FLOOR", "MPH_LEAD_TIME_FLOOR",
    "MPH_CHAMPION_MIN_COUNT", "MPH_CI_BOOTSTRAP_N", "MPH_CI_BOOTSTRAP_SEED",
    "MPH_PHASE14_RESEARCH_MODE", "MPH_FAST_METRIC", "MPH_FULL_EVAL_TRAJECTORY",
    "MPH_USE_VAL_TEST_GAP",
    # Optuna
    "MPH_HP_SPACE", "MPH_SMALL_SAMPLE_CAP", "MPH_OPTUNA_PHASE_TAG", "MPH_STUDY_SUFFIX",
    "MPH_HP_OPTUNA_TRIALS", "MPH_OPTUNA_USE_STORAGE", "MPH_OPTUNA_DNN_REMAINING_CAP",
    # Data split (overseas)
    "MPH_PHASE18_YEAR_MIN_OVERRIDE", "MPH_PHASE18_YEAR_MAX_OVERRIDE",
    # Ops / phase toggles
    "MPH_SAFE_MODE_AUTO", "MPH_ENABLE_PHASE6_REOPT", "MPH_NO_XAI",
    "MPH_PHASE4_FORCE_RETRAIN", "MPH_FORCE_REDO_PHASE13", "MPH_DISABLE_EDA_SIDECAR",
    "MPH_USE_CQR", "MPH_PHASE18_LORO",
    "MPH_AUDIT_USE_CHAMPION_GATE", "MPH_AUDIT_USE_MCS",
})


__all__ = [
    "GLOBAL",
    "GlobalConfig",
    "TrainingConfig",
    "FilterConfig",
    "OptunaConfig",
    "PackageCConfig",
    "ResourceConfig",
    "PathConfig",
    "DataSplitConfig",
    "OpsConfig",
    "KNOWN_ENV_VARS",
]


if __name__ == "__main__":
    # 직접 실행 시 현재 설정 출력
    print(GLOBAL.summary())
