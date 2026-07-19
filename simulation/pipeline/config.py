"""
Pipeline Configuration for MPH Infection Simulation ========================================================
Complete config with CLI/YAML support, all sub-configs,
and dry-run mode for library/OS compatibility checking.
"""
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any
import logging

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

log = logging.getLogger(__name__)


def _safe_init(cls, d: dict):
    """Construct a dataclass from dict, ignoring unknown keys."""
    if not d:
        return cls()
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in valid_keys}
    return cls(**filtered)


@dataclass
class DataConfig:
    """R1 (data) settings."""
    db_path: str = ""
    cache_dir: str = ""
    use_fe_cache: bool = True
    leakage_corr_threshold: float = 0.95
    leakage_action: str = "warn"
    # F2: runner injects week_start vector from R1 (data) so that
    # phase9_dm_test / phase10_intervals can do calendar-accurate regime
    # splits instead of the 47/36/17 proportional fallback.
    dates: Optional[object] = None

    def __post_init__(self):
        from simulation.database.config import _PKG_ROOT, DB_PATH
        if not self.db_path:
            self.db_path = DB_PATH
        if not self.cache_dir:
            self.cache_dir = str(_PKG_ROOT / "cache")


@dataclass
class SplitConfig:
    """Train/Val/Test/Real split settings.

    KUIRB §3 (HWP 연구계획서) 정합 4-way 시간순 split:

        [   train    |  val  |  test  | (conformal_holdout) | real ]
                                      |                     |
                                paper_cutoff_week        end_of_data

    HWP §3 명시 수치 (data analysis period 2019~2025):
      • 전체 in-sample n = 약 337 (341 raw 에서 결측 ~4 제거)
      • train(+val): 약 269 (80%) — 모델 학습 + 파라미터 최적화
      • test       : 약 68  (20%) — 최종 성능 평가 (2024-25 시즌)
      • val        : train_pool 의 마지막 10% = 약 27 (DL early stopping)

    Real slab (HWP 범위 밖 — 추가 forecasting 검증):
      • idx [paper_cutoff_week, end_of_data) — 모델 학습/평가 절대 금지
      • P1 (real_forecaster, 구 real_eval) 에서만 final ensemble refit + 1-step
        rolling forecast 로 성능 보고
      • [stale Phase 8 → Phase 12 corrected 2026-05-20, codex audit cleanup]

    Field 의미:
    - paper_cutoff_week: in-sample 끝 인덱스 (HWP 기본값 337). int.
    - in_sample_test_ratio: 0.20 = test 슬랩이 in-sample 의 마지막 20%
    - in_sample_val_ratio:  0.10 = val 이 train_pool 의 마지막 10%
    - real_eval_enabled: True = real_X/y/dates persist + P1 (real_forecaster) trigger
    - in_sample_end / in_sample_start: 선택적 ISO 날짜 cutoff
        (paper_cutoff_week 보다 우선 — 명시되면 날짜로 자른다)

    legacy ratio-based split (paper_cutoff_week 도 None, in_sample_end 도
    None 일 때만 발동):
    - train_ratio / val_ratio / test_ratio
    - conformal_holdout_weeks (S0-1)
    """
    # Primary: HWP-aligned week-count split
    paper_cutoff_week: Optional[int] = 337    # HWP §3
    in_sample_test_ratio: float = 0.20        # HWP: 68/337
    in_sample_val_ratio: float = 0.10         # val from train_pool
    real_eval_enabled: bool = True
    # Weather/PF-risk handling on the real slab (P1 / real_forecaster):
    #   "oracle":     PERFECT-FORESIGHT ORACLE — uses actual observed weather
    #                 / mobility / population / rt_* in X_real. NOT operationally
    #                 achievable; serves as upper-bound benchmark only.
    #   "climatology": replace each PF-risk column with the week-of-year mean
    #                 computed from in-sample data only. No foresight; lower-
    #                 bound benchmark.
    #   "hybrid":     use existing KMA `fcst_*` columns (already in matrix)
    #                 + climatology fallback for non-forecastable PF-risk
    #                 features. Closest to live-deployment performance —
    #                 RECOMMENDED HEADLINE MODE.
    # The legacy alias "observed" maps to "oracle" for back-compat.
    real_weather_mode: str = "oracle"  # {"oracle","climatology","hybrid","observed"(alias)}

    # Multi-horizon forecasting (FluSight-aligned: h ∈ {1,2,3,4}).
    # P1 (real_forecaster) reports a horizon-decay table on the test slab (n=68) where
    # power exists; real slab stays at h=1 for the operational nowcast.
    real_horizons: tuple = (1, 2, 3, 4)

    # COVID-era inclusion sensitivity (S2-E).
    #   "include":   2020-22 in train as-is (legacy default)
    #   "exclude":   drop 2020-03 → 2022-12 from training
    #   "indicator": include + add COVID-era binary indicator covariate
    covid_inclusion_mode: str = "include"  # {"include","exclude","indicator"}

    # ACI (Adaptive Conformal Inference) for time-series PI on real slab.
    # Default falls back to standard split-conformal; set "aci" or "agaci"
    # for non-exchangeability adjustment (Gibbs & Candès 2021 / Zaffran 2022).
    real_conformal_method: str = "split"  # {"split","aci","agaci"}
    aci_gamma: float = 0.05               # ACI step size

    # Stacking method for ensemble combination (S2-C).
    #   "nnls":      legacy (negative least squares on OOF predictions)
    #   "bma":       legacy Bayesian Model Averaging (M-open flawed)
    #   "stacking":  Yao et al. 2018 stacking on CRPS (RECOMMENDED)
    #   "median":    equal-weighted median ensemble (Sherratt 2023 baseline)
    ensemble_method: str = "stacking"

    # Optional date-anchored override (advanced)
    in_sample_end: Optional[str] = None       # e.g. "2026-02-09"
    in_sample_start: Optional[str] = None     # e.g. "2019-09-02" (no-op default)

    # Legacy ratio-based split (fallback)
    train_ratio: float = 0.7
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    use_validation: bool = True
    conformal_holdout_weeks: int = 0

@dataclass
class OptunaConfig:
    """Optuna feature selection settings."""
    mode: str = "none"              # "none", "external", "inline", "all"
    trials: int = 100               # n_trials for Optuna
    n_trials: int = 100             # alias (backward compat)
    cv_folds: int = 3
    timeout_seconds: int = 3600
    strategy: str = "hp_then_feature"  # "mandatory_only","feature_only","joint","hp_then_feature" []
    epochs_per_trial: int = 100
    external_json_dir: Optional[str] = None

    # 2026-04-28: 실측 기반 trial 예산 재할당 (15h → 7h 목표)
    # 근거 (실측 plateau / 2026-04-27 학습 logs):
    #   - GE-GAT: plateau trial 3, target 40 → 70+ 누적, 6.7h ⚠
    #   - GE-DNN:     plateau trial 6, target 40 → 19분 (정상)
    #   - DNN-Optuna: plateau trial ~10, target 50 → 무한 누적 폭주
    # 변경:
    #   - 무거운 DL 모델 (GAT/foundation): 40 → 15-20 (-50%)
    #   - 가벼운 모델 (Linear/Tree): 20-50 → 20-30 (유지)
    #   - 옛 codebase 전용 deprecated 모델 (DNN-Optuna/TCN-Optuna): 50 → 30
    # 환경변수 `MPH_FAST_TRAIN=1` 시 추가 50% cut (smoke 검증용).
    per_model_trials: dict = field(default_factory=lambda: {
        # Tree (HP 차원 높음, robust) — 50 으로 cut
        "XGBoost": 50,
        "LightGBM": 50,
        # DL Tier A (38 HP 표준 적용 모델) — 40
        # 2026-06-15 (3-LLM 검증 + 사용자 승인): 음성-R² deep 은 HP trial↑로 0 못 넘음
        #   (외삽-평탄 = 구조/평가 artifact) → 40→25 무손실 cut. 양성-R² deep(Mamba·
        #   PatchTST·iTransformer·TimesNet)은 진짜 신호라 40 유지(챔피언 무손상).
        # G-296 (2026-06-17, budget 감사 + 사용자): N-BEATS·N-HiTS·TiDE 는 여기서 의도적으로 제외.
        #   REGISTRY 클래스가 pf_models.Pf{NBeats,NHiTS,TiDE}Forecaster (-pf 정책 2026-05-12: Pf
        #   wrapper 가 이름 점유, Optuna 내장 custom impl nbeats/nhits/tide.py 는 register 차단)라
        #   fit() 에 Optuna 가 ZERO → static-config + Lightning EarlyStopping(patience=10)만 돈다.
        #   "per-model HP tuning" 으로 보고/주장 금지. 옛 25/40/40 entry 는 존재하지 않는 search 를
        #   암시해 가짜 "trial↑ 무효 → 40→25" 결정을 낳았으므로 삭제(get_trials_for 는 self.trials
        #   default 반환, 역시 미소비 = 동작 무변경). (TFT/RNN/DeepAR 도 같은 Pf-static, 비활성.)
        "TabularDNN": 25,             # 40→25 (test R²−0.58, trial↑ 무효)
        "TabularDNN-Lite": 40,
        "Mamba": 40,
        "PatchTST": 40,
        "TFT": 40,
        "GE-DNN": 30,                  # plateau ~6, 30 충분
        "TCN": 30,
        "iTransformer": 40,
        "TimesNet": 40,
        # 무거운 모델 — 대폭 cut (4분/trial × 40 = 2.7h, 너무 김)
        "GE-GAT": 15,              # 40→15, plateau ~3 실측
        # G-261 (2026-06-13): Chronos-2-FT 제거 — Chronos retire (foundation = TimesFM-2.5 + TiRex).
        # Deprecated (옛 코드 50 trial) — cut
        "DNN-Optuna": 30,              # 50→30 (plateau ~10)
        "TCN-Optuna": 30,              # 50→30
        # 작은 모델 (HP 차원 낮음)
        "RandomForest": 30,
        "SVR-Linear": 20,
        "SVR-RBF": 20,
        "KRR": 20,
        "ElasticNet": 20,
    })

    def get_trials_for(self, model_name: str) -> int:
        """모델별 trial 수 조회 (미지정 시 기본 trials 사용)."""
        return self.per_model_trials.get(model_name, self.trials)


@dataclass
class TrainingConfig:
    """Model training settings."""
    run_ensembles: bool = True
    save_models: bool = True              # : .pt 저장 기본 ON
    save_history: bool = True             # : 에폭별 val_loss JSON 저장
    save_dir_plots: str = ""              # : plots 저장 루트 (""=save_dir)
    model_dir: str = "./models"
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class WFCVConfig:
    """Walk-Forward Cross-Validation settings."""
    min_train_weeks: int = 120
    step_size: int = 1
    retune_every: int = 50
    # C-step: PAPER_PRIMARY_11 only mode
    #   True 이면 registry.PAPER_PRIMARY_11 과 factory 이름의 교집합만 WF-CV
    #   step_size_paper_primary 로 fold 수를 대폭 감축 (n=343, min_train=120
    #   → step=4 면 (343-120)/4 ≈ 55 fold 로 압축)
    paper_primary_only: bool = False
    step_size_paper_primary: int = 4

@dataclass
class MemoryConfig:
    """Memory optimization settings."""
    use_float32: bool = True
    use_sparse: bool = False
    max_memory_gb: float = 8.0
    min_free_mb: int = 800
    gc_after_each_fold: bool = False


@dataclass
class OutputConfig:
    """Output / logging settings."""
    log_dir: str = ""
    structured_logging: bool = True
    report_name: str = "diagnostics_report.json"
    save_plots: bool = True

    def __post_init__(self):
        if not self.log_dir:
            from simulation.database.config import _PKG_ROOT
            self.log_dir = str(_PKG_ROOT / "logs")


@dataclass
class ScoringConfig:
    """Composite scoring weights (must sum to 1.0)."""
    w_r2: float = 0.40
    w_rmse_rank: float = 0.20
    w_dm_win_rate: float = 0.15
    w_stability: float = 0.15
    w_conformal: float = 0.10


@dataclass
class EpiValidityConfig:
    """Stage 4 — epi-validity gate configuration.

 The gate runs after R4 (wfcv / walk-forward CV) and produces a
 per-model report of literature-range / sequence / compartment /
 seasonal-peak / outbreak-alignment violations.

 `enabled`
 Master switch. Default True — running the gate is cheap and
 the output is purely informational by default.
 `strict_exclude`
 If True, models that fail any check are flagged
 ``exclude_from_ensemble=True`` in the report so the tournament
 will drop them. Default False (flag only — matches Stage 4
 design note "flag 만 저장, 강제 제외는 opt-in").
 Range overrides mirror the Stage-4 ranges in
 ``simulation/verifier/epi_validity.py`` and let callers relax the
 gate for exploratory runs without editing the module-level
 constants.
 """
    enabled: bool = True
    strict_exclude: bool = False
    # Rt sequence thresholds
    rt_lo: float = 0.3
    rt_hi: float = 5.0
    rt_delta_cap: float = 1.5
    # Compartment conservation tolerance (|S+E+I+R+V+D − N|/N)
    compartment_tol: float = 1e-6
    # Outbreak-alignment peak-window tolerance in ISO weeks
    outbreak_tolerance_weeks: int = 2


@dataclass
class FeatureConfig:
    """Feature inclusion flags for build_enriched_features."""
    include_weather: bool = True
    include_vaccination: bool = True
    include_sentinel_ari: bool = True
    include_weekly_disease: bool = True
    include_population_district: bool = True
    include_subway: bool = True
    include_bus: bool = True
    include_hotspot: bool = True
    include_school: bool = True
    include_hospitals: bool = True
    include_employment: bool = True
    include_hira: bool = True
    include_hourly_population: bool = True
    include_weather_forecast: bool = True
    # rename: 실제 로더는 monthly_* 이므로 플래그명도 맞춘다.
    include_monthly_subway_hourly: bool = True
    include_monthly_bus_hourly: bool = True
    include_sentinel_sari: bool = True
    include_sentinel_hfmd: bool = True
    include_sentinel_enterovirus: bool = True
    include_hira_inpat: bool = True
    include_hira_region: bool = True
    include_childhood_vax: bool = True
    include_dong_population: bool = True
    include_emp_residence: bool = True
    include_rt_road: bool = True
    include_rt_subway_crowd: bool = True
    include_rt_pop_detail: bool = True
    # 2026-04-17: include_rt_bike 제거 (따릉이 피처 제외 지시)
    include_rt_pop_forecast: bool = True
    include_rt_spatial: bool = True
    include_rt_temporal: bool = True
    include_google_trends: bool = True
    include_school_closure: bool = True

@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wfcv: WFCVConfig = field(default_factory=WFCVConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    epi_validity: EpiValidityConfig = field(default_factory=EpiValidityConfig)
    preset: str = "aggressive"
    # Stage-3 scenario tag — canonical map lives in
    # `simulation.__main__.SCENARIOS`; stored here so the YAML round-trip
    # preserves which scenario was applied.
    scenario: Optional[str] = None
    save_dir: str = ""  # resolved at runtime
    log_level: str = "INFO"
    resume_from_phase: int = 0
    dry_run: bool = False
    force_overwrite: bool = False
    no_cache: bool = False

    def __post_init__(self):
        # : MPH_OUTPUT_ROOT 환경 변수로 results/models/cache 루트 리디렉트.
        #   Windows C: 디스크 포화 시 E: 로 분산하려는 용도 (1.8TB 여유).
        #   env 가 없으면 기존 동작(simulation/results, ./models)을 유지.
        from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
        _output_root = _GCFG.paths.output_root
        if not self.save_dir:
            if _output_root:
                self.save_dir = str(Path(_output_root) / "results")
            else:
                from simulation.database.config import _PKG_ROOT
                self.save_dir = str(_PKG_ROOT / "results")
        # model_dir 기본값 (./models) 에만 루트를 덮어쓴다 (YAML 명시 경로는 존중)
        if _output_root and self.training.model_dir in ("./models", "models", ""):
            self.training.model_dir = str(Path(_output_root) / "results" / "models_pt")
        if _output_root and (not self.data.cache_dir or self.data.cache_dir.endswith("cache")):
            self.data.cache_dir = str(Path(_output_root) / "cache")

    # --- Directory helpers ---
    def get_save_dir(self) -> Path:
        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_dir(self) -> Path:
        path = Path(self.training.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cache_dir(self) -> Path:
        path = Path(self.data.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    # --- Serialization ---
    def to_dict(self) -> dict:
        """Convert entire config to serializable dict."""
        # F2: DataConfig.dates is a numpy array injected by runner;
        # exclude from serialization so YAML/JSON dumps stay small + valid.
        data_dict = asdict(self.data)
        data_dict.pop("dates", None)
        # G-219 (2026-05-20): _selected_models 는 CLI 가 동적으로 주입.
        # YAML/JSON report 에서 user filter 가 누락되던 버그 — 명시 mirror.
        _selected = getattr(self, "_selected_models", None)
        return {
            "data": data_dict,
            "split": asdict(self.split),
            "optuna": asdict(self.optuna),
            "training": asdict(self.training),
            "wfcv": asdict(self.wfcv),
            "memory": asdict(self.memory),
            "output": asdict(self.output),
            "scoring": asdict(self.scoring),
            "features": asdict(self.features),
            "epi_validity": asdict(self.epi_validity),
            # G-219: user --models filter mirror (None or list[str])
            "_selected_models": list(_selected) if _selected else None,
            "preset": self.preset,
            "scenario": self.scenario,
            "save_dir": self.save_dir,
            "log_level": self.log_level,
            "resume_from_phase": self.resume_from_phase,
            "dry_run": self.dry_run,
            "force_overwrite": self.force_overwrite,
            "no_cache": self.no_cache,
        }

    def save_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        log.info(f"Config saved: {path}")
    # --- Factory methods ---
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dict, tolerant of extra YAML keys."""
        # Handle output.save_dir -> top-level save_dir
        output_d = d.get("output", {})
        save_dir = d.get("save_dir", output_d.get("save_dir", ""))

        return cls(
            data=_safe_init(DataConfig, d.get("data", {})),
            split=_safe_init(SplitConfig, d.get("split", {})),
            optuna=_safe_init(OptunaConfig, d.get("optuna", {})),
            training=_safe_init(TrainingConfig, d.get("training", {})),
            wfcv=_safe_init(WFCVConfig, d.get("wfcv", {})),
            memory=_safe_init(MemoryConfig, d.get("memory", {})),
            output=_safe_init(OutputConfig, output_d),
            scoring=_safe_init(ScoringConfig, d.get("scoring", {})),
            features=_safe_init(FeatureConfig, d.get("features", {})),
            epi_validity=_safe_init(EpiValidityConfig, d.get("epi_validity", {})),
            preset=d.get("preset", "aggressive"),
            scenario=d.get("scenario"),
            save_dir=save_dir,
            log_level=d.get("log_level", "INFO"),
            resume_from_phase=d.get("resume_from_phase", 0),
            dry_run=d.get("dry_run", False),
            force_overwrite=d.get("force_overwrite", False),
            no_cache=d.get("no_cache", False),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        log.info(f"Config loaded from YAML: {path}")
        return cls.from_dict(d)
    @classmethod
    def from_cli(cls, args) -> "PipelineConfig":
        """Create config from argparse Namespace.

        Priority: CLI args > YAML file > defaults.
        """
        # 1) Start from YAML or defaults
        if args.config:
            config = cls.from_yaml(args.config)
        else:
            config = cls()

        # 2) Override with CLI args (only if explicitly provided)
        if args.preset:
            config.preset = args.preset
        # Stage-3: --scenario tag only. Canonical scenario expansion
        # happens in simulation.__main__.cmd_train (SCENARIOS dict), which
        # forwards individual flags (--optuna-mode, --optuna-strategy, ...) to
        # this parser. Recording the scenario name here preserves it across
        # YAML round-trips without duplicating the expansion table.
        if getattr(args, "scenario", None):
            config.scenario = args.scenario

        if args.optuna_mode:
            config.optuna.mode = args.optuna_mode
        if args.optuna_trials is not None:
            config.optuna.trials = args.optuna_trials
            config.optuna.n_trials = args.optuna_trials
        if args.optuna_strategy:
            config.optuna.strategy = args.optuna_strategy
        if args.epochs is not None:
            config.training.epochs = args.epochs
            # Back-compat: --epochs alone still sets both. --inline-epochs
            # (applied below) overrides optuna.epochs_per_trial if provided.
            config.optuna.epochs_per_trial = args.epochs
        if getattr(args, "inline_epochs", None) is not None:
            # Stage 3 full_light: Optuna inline trials use a shorter fit
            # (e.g. 50 ep) while the final-fit epochs can stay at 200.
            config.optuna.epochs_per_trial = args.inline_epochs
        if getattr(args, "early_stopping_patience", None) is not None:
            config.training.early_stopping_patience = args.early_stopping_patience
        if args.train_ratio is not None:
            config.split.train_ratio = args.train_ratio
            config.split.test_ratio = round(1.0 - args.train_ratio, 2)
        # HWP §3 4-way split overrides
        if getattr(args, "paper_cutoff_week", None) is not None:
            config.split.paper_cutoff_week = int(args.paper_cutoff_week)
        if getattr(args, "in_sample_end", None):
            config.split.in_sample_end = str(args.in_sample_end)
        if getattr(args, "no_real_eval", False):
            config.split.real_eval_enabled = False
        if getattr(args, "weather_mode", None):
            config.split.real_weather_mode = str(args.weather_mode)
        if getattr(args, "covid_inclusion_mode", None):
            config.split.covid_inclusion_mode = str(args.covid_inclusion_mode)
        if getattr(args, "real_conformal_method", None):
            config.split.real_conformal_method = str(args.real_conformal_method)
        if getattr(args, "ensemble_method", None):
            config.split.ensemble_method = str(args.ensemble_method)
        if getattr(args, "per_model_optimize", False):
            # Mirror onto both config and config.split for redundancy
            config.per_model_optimize = True
            try:
                config.split.per_model_optimize = True
            except Exception:
                pass
        if getattr(args, "no_comprehensive_eval", False):
            config.no_comprehensive_eval = True

        if args.wf_step is not None:
            config.wfcv.step_size = args.wf_step
        if args.wf_retune is not None:
            config.wfcv.retune_every = args.wf_retune
        # C-step
        if getattr(args, "paper_primary_only", False):
            config.wfcv.paper_primary_only = True

        if args.resume_from is not None:
            config.resume_from_phase = args.resume_from

        if args.save_dir:
            config.save_dir = args.save_dir

        if hasattr(args, 'dry_run') and args.dry_run:
            config.dry_run = True
        if hasattr(args, 'force_overwrite') and args.force_overwrite:
            config.force_overwrite = True
        if hasattr(args, 'no_cache') and args.no_cache:
            config.no_cache = True
            config.data.use_fe_cache = False

        # Stage 4 — epi-validity gate CLI overrides
        if getattr(args, "epi_validity_disable", False):
            config.epi_validity.enabled = False
        if getattr(args, "epi_validity_strict", False):
            config.epi_validity.strict_exclude = True

        if hasattr(args, 'lite') and args.lite:
            # Lite mode: fewer trials, smaller WF-CV
            config.optuna.trials = min(config.optuna.trials, 20)
            config.optuna.n_trials = config.optuna.trials
            config.wfcv.step_size = max(config.wfcv.step_size, 4)
            config.training.epochs = min(config.training.epochs, 30)

        return config