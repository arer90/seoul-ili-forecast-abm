#!/usr/bin/env python3
r"""R9 per_model_optimize — Optuna Feature Selection (actual cmd_train entry).

⚠ 2026-05-27 명명 정정 (PHASE_AUDIT §S9): 본 file = **actual R9 per_model_optimize feature-선택 entry**
(cmd_train auto-path, NOT `pipeline/phase3_feature_optuna.py` orphan).

cmd_train 호출 path:
  `bash run_resume_phase12.sh --force` → `python -m simulation train --scenario full`
  → `cli/training_commands.py:cmd_train` → `_rerun_feature_optuna` L450
  → subprocess: `python -m simulation.tools.run_optuna_feature_selection`
  → 결과: `simulation/results/stage2_feature_optuna/<model>.json`
  → `pipeline/phase12._optimize_one_model` L968-991 가 load

2026-05-27 사용자 명시 budget (cli/_scenarios.py "full" scenario):
  - scope: representative (9 model) → **individual (53 model 각각)**
  - trials: 25 → **20 per model dedicated**

전략:
  A. feature_only    : HP 기본값 고정, 피처만 Optuna 탐색
  B. joint           : HP + 피처를 하나의 study에서 동시 탐색
  C. hp_then_feature : stage1 HP Optuna(전체 피처) → stage2 피처 Optuna(확정HP) × N회
  D. mandatory_only  : 필수 38개 피처만 사용 (baseline, Optuna 없음)

사용법:
  .venv\Scripts\python.exe simulation\tools\run_optuna_feature_selection.py --model lightgbm --n-trials 50
  .venv\Scripts\python.exe simulation\tools\run_optuna_feature_selection.py --model all --n-trials 100
  .venv\Scripts\python.exe simulation\tools\run_optuna_feature_selection.py --model dnn --strategy hp_then_feature --n-rounds 10
  .venv\Scripts\python.exe simulation\tools\run_optuna_feature_selection.py --model all --strategy all --n-trials 50

인자:
  --model      lightgbm | xgboost | elasticnet | dnn | all (기본: all)
  --strategy   feature_only | joint | hp_then_feature | all (기본: all)
  --n-trials   Optuna trial 수 (기본: 50)
  --n-rounds   hp_then_feature stage2 반복 횟수 (기본: 10)
  --resume     기존 study에서 이어서 실행
  --cv-folds   Walk-Forward CV fold 수 (기본: 3)
  --composite  복합 교차변수 생성 활성화 (기본: True)
"""
import io, sys, os, gc, json, time, warnings, logging, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Warning filter 정리 (2026-04-26 사용자 요청) ──
# 이전: warnings.filterwarnings("ignore") 로 모든 warning 무시 → numerical/convergence 도 묻혀 디버깅 불가
# 변경: 클래스별 분류 — 무해한 것만 ignore, 중요한 것 (수렴/수치) 은 보이게
# 환경변수 MPH_WARN_VERBOSE=1 시 모두 보이게 (디버깅 모드)
if GLOBAL.ops.warn_verbose:
    warnings.filterwarnings("default")    # 모두 보이게
else:
    # 중요한 것 (보이게)
    warnings.filterwarnings("default", category=RuntimeWarning)        # NaN/inf, divide by zero
    warnings.filterwarnings("default", category=UserWarning)            # 기본
    # 무해한 것 (무시)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    # sklearn ConvergenceWarning 은 정확히 따로 — 작은 데이터에서 항상 발생하지만 무시 가능 신호
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("once", category=ConvergenceWarning)    # 한 번만 보고
    except Exception:
        pass

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)  # 일반 log.info 억제

sys.path.insert(0, ".")
import optuna
from optuna.samplers import TPESampler
# 2026-05-13 (사용자 명시): 모든 phase HyperbandPruner 통일.
# get_pruner_for_stage("stage2") → HyperbandPruner(min_resource=1, max=10, rf=3).
from simulation.models._optuna_pruners import get_pruner_for_stage as _make_pruner

# ── Optuna 로그 verbosity ──
# OPTUNA_VERBOSE=1 → INFO (모든 trial print: "Trial X finished with value=...")
# OPTUNA_VERBOSE=2 → DEBUG (sampler 내부도 print)
# 기본값 = WARNING (조용함, ProgressLine 한 줄만 갱신)
_verb = os.environ.get("OPTUNA_VERBOSE", "0")
if _verb == "2":
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
elif _verb == "1":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_default_handler()
    optuna.logging.enable_propagation()
else:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════
# 제자리 진행 표시 (\r 덮어쓰기)
# ══════════════════════════════════════════════════════════

def _fmt_time(seconds):
    """초 → 'm:ss' 또는 'h:mm:ss'."""
    if seconds < 0:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class ProgressLine:
    """한 줄 제자리 진행 표시기.

    사용법:
        p = ProgressLine("LightGBM feature_only")
        p.update(trial=5, total=50, best_rmse=4.32)
        p.finish("RMSE=4.32, 피처 42개")
    """

    def __init__(self, label: str):
        self.label = label
        self.t0 = time.time()
        self._last_len = 0

    def update(self, trial: int, total: int, best_rmse: float = None,
               extra: str = ""):
        elapsed = time.time() - self.t0
        # ETA 계산
        if trial > 0:
            eta = elapsed / trial * (total - trial)
            eta_str = _fmt_time(eta)
        else:
            eta_str = "?"
        pct = trial / total * 100 if total > 0 else 0

        rmse_str = f" best={best_rmse:.3f}" if best_rmse and best_rmse < 1000 else ""
        extra_str = f" {extra}" if extra else ""

        line = (f"\r  ▶ {self.label} | "
                f"{trial}/{total} ({pct:.0f}%) | "
                f"경과 {_fmt_time(elapsed)} | "
                f"남은 ~{eta_str}"
                f"{rmse_str}{extra_str}")

        # 이전 출력보다 짧으면 공백으로 덮기
        pad = max(0, self._last_len - len(line))
        sys.stdout.write(line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def finish(self, summary: str):
        elapsed = time.time() - self.t0
        line = f"\r  ✓ {self.label} | {_fmt_time(elapsed)} | {summary}"
        pad = max(0, self._last_len - len(line))
        sys.stdout.write(line + " " * pad + "\n")
        sys.stdout.flush()
        self._last_len = 0


def _print(msg: str):
    """줄바꿈 출력 (결과 요약용)."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

# Storage root: set MPH_OUTPUT_ROOT env var to redirect to an external disk.
# Default = project-local `simulation/results/`.
from simulation.utils.paths import get_results_dir
SAVE_DIR = get_results_dir()
STUDY_DB = str(SAVE_DIR / "optuna_feature_selection.db")
# Cat 2 (Codex finding, 2026-05-12): study suffix env-gated.
# Categorical 변경 시 (Cat 0 + 11/7/6 mapper fix) 새 study 시작 필요.
# Default = v16 (Cat 0/1/2/3 적용 후 fresh).
# Override: MPH_STUDY_SUFFIX=v15 (옛 study warm-start) / v17 (다음 변경).
_STUDY_SUFFIX = GLOBAL.optuna.study_suffix
# Cat 3 (ANO option, 2026-05-12): storage URL env-gated.
# Default = sqlite (현재). PostgreSQL 도입 시 env: MPH_OPTUNA_STORAGE_URL.
# 예: postgresql://user:pass@host:5432/optuna_db
_STORAGE_URL = os.environ.get("MPH_OPTUNA_STORAGE_URL", f"sqlite:///{STUDY_DB}")
# 2026-04-20: auto-detect DB path so the script works regardless of CWD.
# The old relative literal "data/db/epi_real_seoul.db" only resolved when CWD was the
# project root; subprocess callers from cmd_train may invoke from elsewhere.
_DB_CANDIDATES = [
    "simulation/data/db/epi_real_seoul.db",
    "data/db/epi_real_seoul.db",
    str(Path(__file__).resolve().parent.parent.parent / "simulation/data/db/epi_real_seoul.db"),
]
DB = next((p for p in _DB_CANDIDATES if Path(p).exists()), _DB_CANDIDATES[0])


# ══════════════════════════════════════════════════════════
# 1. Mandatory Features — 다중공선성 완화를 위한 3-tier set
# ══════════════════════════════════════════════════════════
#
# 환경변수 `MPH_MANDATORY_SET` 으로 선택:
#   "full"  (default, back-compat) — 37 features (이전 38 - season_idx)
#   "core"  — 12 features (강한 colilnear set 만 유지, 나머지 제거)
#   "lean"  — 7 features (가장 fundamental 만, OLS-friendly)
#
# 선택 근거 (lag/rolling 그룹 안에서):
#   - lag: lag1 (자기상관 핵심), lag4 (월간), lag12 (분기)
#         → lag2/3 은 lag1 과 corr 0.95+, lag6/8 은 lag4 와 0.9+
#   - rmean: rmean8 (단기), rmean26 (반년) 만 유지
#         → rmean4 ↔ lag2 corr 0.98, rmean13 ↔ rmean8 0.96
#   - rstd/rmin/rmax: rmin/rmax 는 rstd 와 강하게 상관 → rstd 만 유지
#   - sin/cos pairs: p52, p26 만 유지 (p13/p6_5 는 harmonic, 정보 중복)
# ══════════════════════════════════════════════════════════

# Full set (기존 — back-compat) — 37 + advanced derived = 49 features
_MANDATORY_FULL = {
    # 시계열 자기상관 (ablation ΔR²=-0.96, 핵심) — 26개
    "ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag3", "ili_rate_lag4",
    "ili_rate_lag6", "ili_rate_lag8", "ili_rate_lag12",
    "ili_rate_rmean4", "ili_rate_rmean8", "ili_rate_rmean13", "ili_rate_rmean26",
    "ili_rate_rstd4", "ili_rate_rstd8", "ili_rate_rstd13", "ili_rate_rstd26",
    "ili_rate_rmin4", "ili_rate_rmin8", "ili_rate_rmin13", "ili_rate_rmin26",
    "ili_rate_rmax4", "ili_rate_rmax8", "ili_rate_rmax13", "ili_rate_rmax26",
    "ili_rate_diff1", "ili_rate_diff2", "ili_rate_diff4",
    # 계절성 — 10개
    "sin_p52", "cos_p52", "sin_p26", "cos_p26",
    "sin_p13", "cos_p13", "sin_p6_5", "cos_p6_5",
    "sin_month", "cos_month",
    # 시즌 인코딩 — 1개 (season_idx 제거됨, 2026-04-26)
    "season_norm",
    # ── Advanced derived (2026-04-28) — 가장 안정한 12 개 mandatory 후보 ──
    "ili_rate_stl_trend_w104",         # STL trend (low-freq)
    "ili_rate_stl_seasonal_w104",      # STL seasonal
    "ili_rate_savgol_smooth_w17",      # Savitzky-Golay smoothed
    "ili_rate_savgol_deriv_w17",       # local 1차 미분
    "ili_rate_hampel_cleaned_w13",     # outlier 제거된 신호
    "ili_rate_hilbert_amp_w26",        # 진폭 (epi cycle 강도)
    "ili_rate_hilbert_phase_w26",      # 위상 (epi cycle 단계)
    "ili_rate_perment_w26",            # 복잡도 (predictability ↔)
    "ili_rate_spec_ent_w52",           # 주파수 엔트로피
    "ili_rate_hjorth_mob_w13",         # mobility (variance ratio)
    "ili_rate_catch22_f7_acf1_w26",    # ACF lag-1
    "ili_rate_imf1_w26",               # 가장 highest-freq IMF
}

# Core set — 다중공선성 완화 (12 features, VIF 평균 < 50 기대)
_MANDATORY_CORE = {
    # 시계열 자기상관 (lag 핵심 3 개 만)
    "ili_rate_lag1",        # 자기상관 가장 강함
    "ili_rate_lag4",        # 월간 의존성
    "ili_rate_lag12",       # 분기 의존성
    # Rolling: 단기 + 장기 1쌍씩
    "ili_rate_rmean8",      # 2 개월 평균
    "ili_rate_rmean26",     # 반년 평균
    "ili_rate_rstd8",       # 2 개월 변동성
    # Diff: 가속도 핵심 1개
    "ili_rate_diff1",       # 1 주 변화
    # 계절성 — 주기 핵심 2 쌍
    "sin_p52", "cos_p52",   # 연 주기 (인플루엔자 가장 중요)
    "sin_p26", "cos_p26",   # 반년 주기
    # 시즌
    "season_norm",
}

# Lean set — OLS-friendly (7 features, VIF 평균 < 10 목표)
_MANDATORY_LEAN = {
    "ili_rate_lag1",        # 자기상관 본질
    "ili_rate_rmean26",     # 장기 평균
    "ili_rate_diff1",       # 1 주 변화
    "sin_p52", "cos_p52",   # 연 주기
    "season_norm",
    # 위 6 개 외에 1 개 더 — 하지만 모두 강하게 상관됨, 6 으로 마무리
}


def _select_mandatory_set() -> set:
    """env `MPH_MANDATORY_SET` 따라 set 선택. default=full (back-compat)."""
    choice = GLOBAL.ops.mandatory_features_set.lower().strip()
    if choice == "core":
        return _MANDATORY_CORE
    elif choice == "lean":
        return _MANDATORY_LEAN
    elif choice in ("full", ""):
        return _MANDATORY_FULL
    else:
        log.warning(f"  Unknown MPH_MANDATORY_SET='{choice}', falling back to full")
        return _MANDATORY_FULL


# 모듈 초기화 시점에 1회 결정 (env 변경 후 reload 필요)
MANDATORY_FEATURES_EXACT = _select_mandatory_set()


def _is_mandatory(col_name: str) -> bool:
    return col_name in MANDATORY_FEATURES_EXACT


# ══════════════════════════════════════════════════════════
# 2. Data Loading + 피처 필터링
# ══════════════════════════════════════════════════════════

_DATA_CACHE = None

def _filter_useless_features(X_df, feature_cols, y):
    """Zero/near-zero variance + 타겟 무상관 + 다중공선성 피처 자동 제거.

    제거 기준:
      1) std == 0 (상수)
      2) 최빈값 비율 > 98% AND |r(y)| < 0.05
      3) [NEW] |corr(다른 feature)| ≥ 0.999 — perfect collinearity (mandatory 보호)
      4) [NEW] (env MPH_DROP_HIGH_VIF=1) VIF > 100 + non-mandatory → 제거
      5) 필수(mandatory) 피처는 절대 제거 안 함 (1, 3, 4 의 mandatory 보호)
    """
    exclude = set()
    n = len(y)

    for i, col in enumerate(feature_cols):
        if _is_mandatory(col):
            continue
        vals = X_df[:, i]
        std = float(np.std(vals))

        # (1) 상수 피처
        if std < 1e-10:
            exclude.add(col)
            continue

        # (2) 거의 상수 + 타겟 무상관
        unique_vals, counts = np.unique(vals, return_counts=True)
        most_common_ratio = float(counts.max()) / n
        if most_common_ratio > 0.98:
            r = abs(float(np.corrcoef(vals, y)[0, 1])) if std > 0 else 0
            if r < 0.05:
                exclude.add(col)

    # ── (3) [NEW 2026-04-26] Perfect collinearity (corr ≥ 0.999) 자동 제거 ──
    # mandatory 보호 (mandatory ↔ non-mandatory 일 때만 non-mandatory 제거).
    # mandatory ↔ mandatory 는 경고만 (사용자 결정 필요).
    perf_threshold = 0.999
    keep_mask = np.array([c not in exclude for c in feature_cols])
    if keep_mask.sum() > 1:
        try:
            X_keep = X_df[:, keep_mask]
            keep_names = [c for c, k in zip(feature_cols, keep_mask) if k]
            corr = np.corrcoef(X_keep.T)
            np.fill_diagonal(corr, 0.0)
            n_keep = len(keep_names)
            for i in range(n_keep):
                a = keep_names[i]
                if a in exclude:
                    continue
                for j in range(i + 1, n_keep):
                    b = keep_names[j]
                    if b in exclude:
                        continue
                    r = corr[i, j]
                    if not np.isfinite(r):
                        continue
                    if abs(r) >= perf_threshold:
                        a_mand = a in MANDATORY_FEATURES_EXACT
                        b_mand = b in MANDATORY_FEATURES_EXACT
                        if a_mand and b_mand:
                            # 둘 다 mandatory: 경고만 (사용자 사전 결정 필요)
                            log.warning(
                                f"  ⚠ Perfect collinearity (mandatory↔mandatory): "
                                f"{a} ↔ {b} (|r|={abs(r):.4f}) — "
                                f"MANDATORY_FEATURES_EXACT 검토 필요"
                            )
                        elif a_mand and not b_mand:
                            exclude.add(b)
                            log.info(f"  [collinearity] drop {b} (|r|={abs(r):.4f} with mandatory {a})")
                        elif b_mand and not a_mand:
                            exclude.add(a)
                            log.info(f"  [collinearity] drop {a} (|r|={abs(r):.4f} with mandatory {b})")
                        else:
                            # 둘 다 non-mandatory: 알파벳 순 두 번째 제거
                            keep_one, drop_one = sorted([a, b])
                            exclude.add(drop_one)
                            log.info(f"  [collinearity] drop {drop_one} (|r|={abs(r):.4f} with {keep_one})")
        except Exception as _ce:
            log.debug(f"  collinearity check skipped: {_ce}")

    # ── (4) [NEW 2026-04-26] env MPH_DROP_HIGH_VIF=1 시 VIF>100 non-mandatory 제거 ──
    if GLOBAL.training.drop_high_vif:
        try:
            keep_mask2 = np.array([c not in exclude for c in feature_cols])
            X_keep2 = X_df[:, keep_mask2]
            keep_names2 = [c for c, k in zip(feature_cols, keep_mask2) if k]
            for j, name in enumerate(keep_names2):
                if name in MANDATORY_FEATURES_EXACT:
                    continue
                others = [k for k in range(X_keep2.shape[1]) if k != j]
                if not others:
                    continue
                Xo = X_keep2[:, others]
                y_j = X_keep2[:, j]
                try:
                    coef, *_ = np.linalg.lstsq(Xo, y_j, rcond=None)
                    pred = Xo @ coef
                    ss_res = float(np.sum((y_j - pred) ** 2))
                    ss_tot = float(np.sum((y_j - y_j.mean()) ** 2)) + 1e-12
                    r2 = 1.0 - ss_res / ss_tot
                    vif = 1.0 / max(1e-12, 1.0 - r2)
                    if vif > 100.0:
                        exclude.add(name)
                        log.info(f"  [VIF] drop {name} (VIF={vif:.1f})")
                except Exception:
                    continue
        except Exception as _ve:
            log.debug(f"  VIF check skipped: {_ve}")

    return exclude


def load_data():
    """feature_engine에서 데이터 로드 + 무의미 피처 자동 제거 (캐싱)."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    from simulation.models.feature_engine import build_enriched_features
    feat_df, meta = build_enriched_features(DB)
    pdf = feat_df.to_pandas()
    target = "ili_rate"
    y = pdf[target].values.astype(np.float64)
    drop_cols = [target, "week_start", "gu_nm"]
    all_feature_cols = [c for c in pdf.columns
                        if c not in drop_cols
                        and pd.api.types.is_numeric_dtype(pdf[c])]
    X_all = pdf[all_feature_cols].fillna(0).values.astype(np.float64)

    # 무의미 피처 자동 제거
    exclude = _filter_useless_features(X_all, all_feature_cols, y)
    if exclude:
        keep_idx = [i for i, c in enumerate(all_feature_cols) if c not in exclude]
        feature_cols = [all_feature_cols[i] for i in keep_idx]
        X = X_all[:, keep_idx]
        _print(f"  피처 필터링: {len(all_feature_cols)} → {len(feature_cols)}개 "
               f"(제거 {len(exclude)}: zero-var/near-zero/무상관)")
    else:
        feature_cols = all_feature_cols
        X = X_all

    # ── PCA 옵션 (2026-04-27): MPH_USE_PCA=K → top-K components 변환 ──
    # mandatory features 보존 + non-mandatory 만 PCA 변환 → orthogonal
    # 다중공선성 완벽 해결 (κ → 5.17), VIF=0.
    # 단점: feature interpretation 손실 (SHAP/논문 어려움).
    _pca_k = int(os.environ.get("MPH_USE_PCA", "0") or 0)
    if _pca_k > 0:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            mand_idx = [i for i, c in enumerate(feature_cols)
                         if c in MANDATORY_FEATURES_EXACT]
            non_mand_idx = [i for i, c in enumerate(feature_cols)
                              if c not in MANDATORY_FEATURES_EXACT]
            if len(non_mand_idx) > _pca_k:
                X_mand = X[:, mand_idx]
                X_non = X[:, non_mand_idx]
                # standardize 후 PCA
                scaler = StandardScaler()
                X_non_std = scaler.fit_transform(X_non)
                pca = PCA(n_components=_pca_k, random_state=42)
                X_non_pca = pca.fit_transform(X_non_std)
                # 합치기: mandatory 그대로 + PCA components
                X = np.concatenate([X_mand, X_non_pca], axis=1)
                feature_cols = (
                    [feature_cols[i] for i in mand_idx]
                    + [f"pca_{k}" for k in range(_pca_k)]
                )
                cum_var = float(pca.explained_variance_ratio_.cumsum()[-1])
                _print(f"  🔬 PCA top-{_pca_k}: non-mandatory {len(non_mand_idx)} → {_pca_k} "
                       f"(cum var {cum_var*100:.1f}%, mandatory {len(mand_idx)} 유지)")
            else:
                _print(f"  PCA skipped (non-mandatory {len(non_mand_idx)} ≤ K={_pca_k})")
        except Exception as _pe:
            log.warning(f"  PCA 실패 → 원본 유지: {_pe}")

    _DATA_CACHE = (X, y, feature_cols)
    return X, y, feature_cols


# ══════════════════════════════════════════════════════════
# 3. Composite Feature Generation
# ══════════════════════════════════════════════════════════

COMPOSITE_CANDIDATES = [
    # ── A. 기상 interaction (2-way) ──
    ("temp_x_humidity", "temp_avg", "humidity", "multiply"),
    ("temp_x_wind", "temp_avg", "wind_speed", "multiply"),
    ("humidity_x_wind", "humidity", "wind_speed", "multiply"),
    ("temp_x_rain", "temp_avg", "rn_day", "multiply"),
    ("cold_dry", "temp_avg", "humidity", "cold_dry_index"),
    # 이진 threshold
    ("temp_below5", "temp_avg", None, "threshold_below_5"),
    ("temp_below0", "temp_avg", None, "threshold_below_0"),
    ("humidity_above80", "humidity", None, "threshold_above_80"),
    ("humidity_below40", "humidity", None, "threshold_below_40"),

    # ── B. lag × 환경 (2-way): ILI 자기상관 × 외부 driver ──
    ("lag1_x_temp", "ili_rate_lag1", "temp_avg", "multiply"),
    ("lag1_x_humidity", "ili_rate_lag1", "humidity", "multiply"),
    ("lag1_x_vax", "ili_rate_lag1", "vax_rate", "multiply"),
    ("lag1_x_subway", "ili_rate_lag1", "subway_total", "multiply"),
    ("lag1_x_bus", "ili_rate_lag1", "bus_total", "multiply"),
    ("lag1_x_pop", "ili_rate_lag1", "pop_inflow", "multiply"),
    ("lag1_x_hotspot", "ili_rate_lag1", "hotspot_congestion", "multiply"),
    ("lag1_x_ari", "ili_rate_lag1", "ari_total", "multiply"),
    ("lag1_x_closure", "ili_rate_lag1", "sch_closure_lag1", "multiply"),
    ("lag2_x_temp", "ili_rate_lag2", "temp_avg", "multiply"),
    ("lag2_x_subway", "ili_rate_lag2", "subway_total", "multiply"),
    ("lag4_x_temp", "ili_rate_lag4", "temp_avg", "multiply"),

    # ── C. lag 자기상호작용 (2-way): 추세/모멘텀 캡처 ──
    ("lag1_sq", "ili_rate_lag1", "ili_rate_lag1", "multiply"),
    ("lag1_x_lag2", "ili_rate_lag1", "ili_rate_lag2", "multiply"),
    ("lag1_x_lag4", "ili_rate_lag1", "ili_rate_lag4", "multiply"),
    ("lag2_x_lag4", "ili_rate_lag2", "ili_rate_lag4", "multiply"),
    ("diff1_x_lag1", "ili_rate_diff1", "ili_rate_lag1", "multiply"),
    ("diff1_x_diff2", "ili_rate_diff1", "ili_rate_diff2", "multiply"),

    # ── D. 비율 피처 (2-way): 상대 변화율 ──
    ("lag1_div_lag4", "ili_rate_lag1", "ili_rate_lag4", "ratio"),
    ("lag1_div_lag12", "ili_rate_lag1", "ili_rate_lag12", "ratio"),
    ("rmean4_div_rmean12", "ili_rate_rmean4", "ili_rate_rmean12", "ratio"),
    ("rmean4_div_rmean26", "ili_rate_rmean4", "ili_rate_rmean26", "ratio"),
    ("lag1_div_rmean4", "ili_rate_lag1", "ili_rate_rmean4", "ratio"),
    ("lag1_div_rmean12", "ili_rate_lag1", "ili_rate_rmean12", "ratio"),
    ("sari_div_ili", "sari_count", "ili_rate_lag1", "ratio"),
    ("ari_div_ili", "ari_total", "ili_rate_lag1", "ratio"),

    # ── E. 교통 interaction (2-way) ──
    ("pop_x_transport", "pop_inflow", "subway_total", "multiply"),
    ("subway_x_rush", "subway_total", "sub_rush_ratio", "multiply"),
    ("bus_x_subway", "bus_total", "subway_total", "multiply"),
    ("subway_x_temp", "subway_total", "temp_avg", "multiply"),
    ("pop_x_temp", "pop_inflow", "temp_avg", "multiply"),

    # ── F. Google Trends × 환경 (2-way) ──
    ("gt_fever_x_lag1", "gt_fever_lag1", "ili_rate_lag1", "multiply"),
    ("gt_flu_x_temp", "gt_flu_lag1", "temp_avg", "multiply"),
    ("gt_flu_x_lag1", "gt_flu_lag1", "ili_rate_lag1", "multiply"),
    ("gt_cold_x_lag1", "gt_cold_lag1", "ili_rate_lag1", "multiply"),

    # ── G. 계절성 × ILI (2-way): 위상-강도 캡처 ──
    ("sin52_x_lag1", "sin_p52", "ili_rate_lag1", "multiply"),
    ("cos52_x_lag1", "cos_p52", "ili_rate_lag1", "multiply"),
    ("sin52_x_temp", "sin_p52", "temp_avg", "multiply"),
    ("cos52_x_temp", "cos_p52", "temp_avg", "multiply"),
    ("sin26_x_lag1", "sin_p26", "ili_rate_lag1", "multiply"),

    # ── H. rolling stat interaction (2-way) ──
    ("rmean4_x_temp", "ili_rate_rmean4", "temp_avg", "multiply"),
    ("rstd4_x_lag1", "ili_rate_rstd4", "ili_rate_lag1", "multiply"),
    ("rmean4_x_rmean12", "ili_rate_rmean4", "ili_rate_rmean12", "multiply"),

    # ── I. 역학 × 역학 (2-way) ──
    ("ari_x_temp", "ari_total", "temp_avg", "multiply"),
    ("vax_x_temp", "vax_rate", "temp_avg", "multiply"),
    ("closure_x_temp", "sch_closure_lag1", "temp_avg", "multiply"),
]

COMPOSITE_3WAY = [
    # ── 3-way: 전파 dynamics (환경 × 접촉 × 감수성) ──
    ("cold_crowd_ili", ["temp_avg", "subway_total", "ili_rate_lag1"], "multiply3"),
    ("dry_pop_ili", ["humidity", "pop_inflow", "ili_rate_lag1"], "cold_dry_pop"),
    ("vax_cold_ili", ["vax_rate", "temp_avg", "ili_rate_lag1"], "vax_cold"),
    ("closure_cold_ili", ["sch_closure_lag1", "temp_avg", "ili_rate_lag1"], "multiply3"),
    ("subway_rush_ili", ["subway_total", "sub_rush_ratio", "ili_rate_lag1"], "multiply3"),
    ("pop_hotspot_ili", ["pop_inflow", "hotspot_congestion", "ili_rate_lag1"], "multiply3"),
    ("lag1_diff_rmean", ["ili_rate_lag1", "ili_rate_diff1", "ili_rate_rmean4"], "multiply3"),
    ("season_cold_ili", ["sin_p52", "temp_avg", "ili_rate_lag1"], "multiply3"),
    ("gt_cold_ili", ["gt_flu_lag1", "temp_avg", "ili_rate_lag1"], "multiply3"),
    # ── 3-way: 기상 복합 ──
    ("cold_dry_wind", ["temp_avg", "humidity", "wind_speed"], "multiply3"),
    ("cold_humid_ili", ["temp_avg", "humidity", "ili_rate_lag1"], "multiply3"),
    # ── 3-way: 교통 복합 ──
    ("subway_bus_ili", ["subway_total", "bus_total", "ili_rate_lag1"], "multiply3"),
    ("transport_cold_ili", ["subway_total", "temp_avg", "ili_rate_lag1"], "multiply3"),
    # ── 3-way: lag 모멘텀 ──
    ("lag1_lag2_lag4", ["ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag4"], "multiply3"),
    ("lag1_diff1_temp", ["ili_rate_lag1", "ili_rate_diff1", "temp_avg"], "multiply3"),
    # ── 3-way: 계절 × 환경 × ILI ──
    ("sin52_cold_ili", ["sin_p52", "temp_avg", "ili_rate_lag1"], "multiply3"),
    ("cos52_humid_ili", ["cos_p52", "humidity", "ili_rate_lag1"], "multiply3"),
    # ── 3-way: Google Trends ──
    ("gt_flu_cold_ili", ["gt_flu_lag1", "temp_avg", "ili_rate_lag1"], "multiply3"),
    ("gt_fever_sub_ili", ["gt_fever_lag1", "subway_total", "ili_rate_lag1"], "multiply3"),

    # ── 4-way: 전파 복합 모델 (β × S × I × contact) 근사 ──
    ("full_transmission", ["temp_avg", "subway_total", "vax_rate", "ili_rate_lag1"], "multiply_n"),
    ("env_contact_ili", ["temp_avg", "humidity", "subway_total", "ili_rate_lag1"], "multiply_n"),
    ("cold_dry_crowd_ili", ["temp_avg", "humidity", "pop_inflow", "ili_rate_lag1"], "multiply_n"),
    ("season_env_ili", ["sin_p52", "temp_avg", "humidity", "ili_rate_lag1"], "multiply_n"),
    ("gt_env_contact_ili", ["gt_flu_lag1", "temp_avg", "subway_total", "ili_rate_lag1"], "multiply_n"),

    # ── 5-way: 종합 전파 지표 (환경+접촉+감수성+추세+계절) ──
    ("mega_transmission", ["temp_avg", "humidity", "subway_total", "ili_rate_lag1", "sin_p52"], "multiply_n"),
    ("full_epi_driver", ["temp_avg", "vax_rate", "pop_inflow", "ili_rate_lag1", "ili_rate_diff1"], "multiply_n"),
]


def find_col_index(feature_cols, pattern):
    for i, c in enumerate(feature_cols):
        if c == pattern or c.startswith(pattern):
            return i
    return None


def generate_composite_features(X, feature_cols, trial):
    """Optuna trial에서 선택된 복합변수 생성."""
    new_features = []
    new_names = []

    for name, pat_a, pat_b, op in COMPOSITE_CANDIDATES:
        use = trial.suggest_categorical(f"comp_{name}", [True, False])
        if not use:
            continue
        idx_a = find_col_index(feature_cols, pat_a)
        if idx_a is None:
            continue

        if op == "multiply":
            idx_b = find_col_index(feature_cols, pat_b)
            if idx_b is None:
                continue
            feat = X[:, idx_a] * X[:, idx_b]
        elif op == "ratio":
            idx_b = find_col_index(feature_cols, pat_b)
            if idx_b is None:
                continue
            feat = X[:, idx_a] / (X[:, idx_b] + 1e-10)
        elif op == "threshold_below_5":
            feat = (X[:, idx_a] < 5).astype(np.float64)
        elif op == "threshold_below_0":
            feat = (X[:, idx_a] < 0).astype(np.float64)
        elif op == "threshold_above_80":
            feat = (X[:, idx_a] > 80).astype(np.float64)
        elif op == "threshold_below_40":
            feat = (X[:, idx_a] < 40).astype(np.float64)
        elif op == "cold_dry_index":
            idx_b = find_col_index(feature_cols, pat_b)
            if idx_b is None:
                continue
            feat = np.maximum(5 - X[:, idx_a], 0) * np.maximum(80 - X[:, idx_b], 0)
        else:
            continue
        feat = np.nan_to_num(feat, nan=0, posinf=0, neginf=0)
        new_features.append(feat)
        new_names.append(f"comp_{name}")

    for name, feat_patterns, op in COMPOSITE_3WAY:
        use = trial.suggest_categorical(f"comp_{name}", [True, False])
        if not use:
            continue
        indices = [find_col_index(feature_cols, p) for p in feat_patterns]
        if any(idx is None for idx in indices):
            continue
        if op == "multiply3":
            feat = X[:, indices[0]] * X[:, indices[1]] * X[:, indices[2]]
        elif op == "multiply_n":
            feat = np.ones(X.shape[0])
            for idx in indices:
                feat = feat * X[:, idx]
        elif op == "cold_dry_pop":
            feat = np.maximum(80 - X[:, indices[0]], 0) * X[:, indices[1]] * X[:, indices[2]]
        elif op == "vax_cold":
            feat = np.maximum(1 - X[:, indices[0]], 0) * np.maximum(5 - X[:, indices[1]], 0) * X[:, indices[2]]
        else:
            continue
        feat = np.nan_to_num(feat, nan=0, posinf=0, neginf=0)
        new_features.append(feat)
        new_names.append(f"comp_{name}")

    if new_features:
        return np.column_stack([X] + [f.reshape(-1, 1) for f in new_features]), new_names
    return X, []


# ══════════════════════════════════════════════════════════
# 4. Model Factories — 전체 모델 지원
# ══════════════════════════════════════════════════════════

# ── Scope별 모델 목록 ──
# quick(4):          핵심 대표 4개 — 빠른 테스트용
# representative(9): 계열별 대표 9개 — 기본값, 균형
# individual(15):    전체 15개 모델 각각 — 가장 정확, 시간 오래
MODELS_QUICK = [
    "lightgbm", "xgboost", "elasticnet", "dnn",
]
MODELS_REPRESENTATIVE = [
    # Tree: lightgbm, xgboost (+ randomforest 독립 — tree지만 앙상블 구조 다름)
    "lightgbm", "xgboost", "randomforest",
    # Linear: elasticnet (+ svr_rbf 독립 — 커널 비선형)
    "elasticnet", "svr_rbf",
    # DL
    "dnn",
    # Epi/Bayesian: gp, bayesianridge, gam (3계열 대표)
    "gp_rbf_periodic", "bayesianridge", "gam",
]
MODELS_INDIVIDUAL = [
    # ── Tree ──
    "lightgbm", "xgboost", "randomforest", "gradientboosting",
    # ── Linear ──
    "elasticnet", "svr_rbf", "svr_linear", "krr",
    # ── DL ──
    "dnn",
    # ── Modern TS (DNN proxy 평가, 독립 Optuna study) ──
    "tcn", "nbeats", "nhits", "tft", "patchtst",
    "itransformer", "tide", "mamba", "timesnet",
    # ── Graph + Tabular ──
    "ge_dnn", "tabular_dnn",
    # ── Epi/Bayesian ──
    "gp_rbf_periodic", "bayesianridge", "negbinglm",
    "bayesianmcmc", "poissonautoreg", "gam",
]
ALL_MODELS = MODELS_INDIVIDUAL  # 하위 호환

# ── DNN proxy로 평가되는 모델 키 목록 ──
_DNN_PROXY_MODELS = {
    "dnn", "tcn", "nbeats", "nhits", "tft", "patchtst",
    "itransformer", "tide", "mamba", "timesnet",
    "ge_dnn", "tabular_dnn",
}

# ══════════════════════════════════════════════════════════
# 학습 파이프라인 모델명 → Optuna 피처 선택 키
# ── 2가지 모드 ──
#   category:   카테고리별 proxy 공유 (DL→dnn, 빠름)
#   individual: 모델별 독립 Optuna (느리지만 모델 특성 반영)
#               → 매핑 없는 Modern TS 모델은 MI fallback
# ══════════════════════════════════════════════════════════

# ── 공통 (Tree + Linear + Epi/Bayesian): 둘 다 동일 ──
_COMMON_KEY_MAP = {
    # Tree
    "LightGBM": "lightgbm",
    "XGBoost": "xgboost",
    "RandomForest": "randomforest",
    "GradientBoosting": "gradientboosting",
    # Linear
    "ElasticNet": "elasticnet",
    "SVR-RBF": "svr_rbf",
    "SVR-Linear": "svr_linear",
    "KRR": "krr",
    # DNN 변종 (아키텍처 동일 → 항상 공유)
    "DNN": "dnn", "DNN-Optuna": "dnn", "DNN-Conformal": "dnn",
    "TabularDNN": "tabular_dnn",
    # Graph
    "GE-DNN": "ge_dnn",
    # Epi/Bayesian
    "GP-RBF-Periodic": "gp_rbf_periodic",
    "BayesianRidge": "bayesianridge",
    "NegBinGLM": "negbinglm",
    "BayesianMCMC": "bayesianmcmc",
    "PoissonAutoreg": "poissonautoreg",
    "GAM": "gam",
}

# ── category 모드: Modern TS → DNN proxy 공유 (빠름) ──
_CATEGORY_KEY_MAP = {
    **_COMMON_KEY_MAP,
    "TCN": "dnn", "TCN-Optuna": "dnn",
    "N-BEATS": "dnn", "N-HiTS": "dnn",
    "TFT": "dnn", "PatchTST": "dnn",
    "iTransformer": "dnn", "TiDE": "dnn",
    "Mamba": "dnn", "TimesNet": "dnn",
    "GE-DNN": "dnn", "TabularDNN": "dnn",
}

# ── individual 모드: Modern TS → 각각 독립 키 (DNN proxy로 평가하되 별도 study) ──
_INDIVIDUAL_KEY_MAP = {
    **_COMMON_KEY_MAP,
    "TCN": "tcn", "TCN-Optuna": "tcn",
    "N-BEATS": "nbeats", "N-HiTS": "nhits",
    "TFT": "tft", "PatchTST": "patchtst",
    "iTransformer": "itransformer", "TiDE": "tide",
    "Mamba": "mamba", "TimesNet": "timesnet",
    "GE-DNN": "ge_dnn", "TabularDNN": "tabular_dnn",
}

# 기본값: individual (모델별 독립 선택)
TRAIN_MODEL_TO_OPTUNA_KEY = _INDIVIDUAL_KEY_MAP


def set_optuna_scope(scope: str):
    """optuna_scope 설정: 'category' 또는 'individual'."""
    global TRAIN_MODEL_TO_OPTUNA_KEY
    if scope == "category":
        TRAIN_MODEL_TO_OPTUNA_KEY = _CATEGORY_KEY_MAP
    else:
        TRAIN_MODEL_TO_OPTUNA_KEY = _INDIVIDUAL_KEY_MAP

# ── Scope별 fallback 매핑 ──
# quick(4): 나머지 모델 → 4개 대표 중 하나
QUICK_MAP = {
    "lightgbm": "lightgbm", "xgboost": "xgboost",
    "randomforest": "lightgbm", "gradientboosting": "lightgbm",
    "elasticnet": "elasticnet",
    "svr_rbf": "elasticnet", "svr_linear": "elasticnet", "krr": "elasticnet",
    "dnn": "dnn",
    "ge_dnn": "dnn",                        # GE-DNN → DNN proxy
    "tabular_dnn": "dnn",                   # TabularDNN → DNN proxy
    "gp_rbf_periodic": "elasticnet",
    "bayesianridge": "elasticnet",
    "negbinglm": "elasticnet",
    "bayesianmcmc": "elasticnet",
    "poissonautoreg": "elasticnet",
    "gam": "elasticnet",
}

# representative(9): 나머지 모델 → 9개 대표 중 하나
REPRESENTATIVE_MAP = {
    "lightgbm": "lightgbm", "xgboost": "xgboost",
    "randomforest": "randomforest",
    "gradientboosting": "lightgbm",         # GB → LightGBM (같은 boosting)
    "elasticnet": "elasticnet",
    "svr_rbf": "svr_rbf",
    "svr_linear": "elasticnet",             # SVR-Linear → ElasticNet (같은 linear)
    "krr": "svr_rbf",                       # KRR → SVR-RBF (같은 커널)
    "dnn": "dnn",
    "ge_dnn": "dnn",                        # GE-DNN → DNN proxy
    "tabular_dnn": "dnn",                   # TabularDNN → DNN proxy
    "gp_rbf_periodic": "gp_rbf_periodic",
    "bayesianridge": "bayesianridge",
    "negbinglm": "bayesianridge",           # NegBin → BayesianRidge (같은 통계)
    "bayesianmcmc": "bayesianridge",        # MCMC → BayesianRidge
    "poissonautoreg": "bayesianridge",      # Poisson → BayesianRidge
    "gam": "gam",
}


def _resolve_proxy(model_type):
    """Modern TS → DNN proxy 해석. 나머지는 그대로 반환."""
    if model_type in _DNN_PROXY_MODELS:
        return "dnn"
    return model_type


def _default_hp(model_type):
    """모델별 기본 하이퍼파라미터."""
    model_type = _resolve_proxy(model_type)
    if model_type == "lightgbm":
        return {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.1, "reg_lambda": 1.0,
                "min_child_samples": 20, "random_state": 42, "n_jobs": 2, "verbose": -1}
    elif model_type == "xgboost":
        return {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.1, "reg_lambda": 1.0,
                "min_child_weight": 5, "random_state": 42, "n_jobs": 2, "verbosity": 0}
    elif model_type == "randomforest":
        return {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5,
                "random_state": 42, "n_jobs": 2}
    elif model_type == "gradientboosting":
        return {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5,
                "random_state": 42}
    elif model_type == "elasticnet":
        return {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 2000, "random_state": 42}
    elif model_type in ("svr_rbf", "svr_linear"):
        kernel = "rbf" if model_type == "svr_rbf" else "linear"
        return {"kernel": kernel, "C": 1.0, "epsilon": 0.1}
    elif model_type == "krr":
        return {"alpha": 1.0, "kernel": "rbf", "gamma": None}
    elif model_type == "dnn":
        # 표준 키 (suggest_tabular_dnn_hp 의 default 영역) + back-compat 옛 키.
        return {
            # ── 표준 (priority) ──
            "n_layers": 2,
            "hidden_dims": [128, 64],
            "dropouts": [0.25, 0.15],
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "activation": "relu",
            "norm": "none",
            "init": "default",
            "optimizer": "adamw",
            "loss": "mse",  # G-218: huber 영구 제거 (huber-loss-banned-20260520)
            "lr_schedule": "cosine_restart",
            "gradient_clip": 1.0,
            "warmup_epochs": 0,
            "momentum": 0.9,
            "nesterov": True,
            "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
            "layer_type": "linear",
            "use_attention": False, "use_fm": False,
            "use_bias": True, "skip_connection": False,
            # ── back-compat (옛 _train_dnn fallback) ──
            "h1": 128, "h2": 64, "d1": 0.25, "d2": 0.15, "use_h3": False,
            "wd": 1e-4, "bs": 32,
        }
    elif model_type == "gp_rbf_periodic":
        return {"alpha": 1e-2, "n_restarts": 3}
    elif model_type == "bayesianridge":
        return {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6}
    elif model_type == "negbinglm":
        return {"alpha": 1.0, "max_iter": 200}
    elif model_type == "bayesianmcmc":
        return {"n_samples": 500, "n_warmup": 200}
    elif model_type == "poissonautoreg":
        return {"max_iter": 200}
    elif model_type == "gam":
        return {"n_splines": 20, "lam": 0.6}
    return {}


def _suggest_hp(model_type, trial, prefix=""):
    """Optuna trial에서 HP 제안."""
    model_type = _resolve_proxy(model_type)
    p = prefix
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int(f"{p}lgb_n_est", 50, 300),
            "max_depth": trial.suggest_int(f"{p}lgb_depth", 2, 6),
            "learning_rate": trial.suggest_float(f"{p}lgb_lr", 0.01, 0.1, log=True),
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": trial.suggest_float(f"{p}lgb_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float(f"{p}lgb_lambda", 0.01, 10.0, log=True),
            "min_child_samples": trial.suggest_int(f"{p}lgb_min_child", 10, 50),
            "random_state": 42, "n_jobs": 2, "verbose": -1,
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int(f"{p}xgb_n_est", 50, 300),
            "max_depth": trial.suggest_int(f"{p}xgb_depth", 2, 6),
            "learning_rate": trial.suggest_float(f"{p}xgb_lr", 0.01, 0.1, log=True),
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": trial.suggest_float(f"{p}xgb_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float(f"{p}xgb_lambda", 0.01, 10.0, log=True),
            "min_child_weight": 5, "random_state": 42, "n_jobs": 2, "verbosity": 0,
        }
    elif model_type == "randomforest":
        return {
            "n_estimators": trial.suggest_int(f"{p}rf_n_est", 50, 300),
            "max_depth": trial.suggest_int(f"{p}rf_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int(f"{p}rf_min_leaf", 2, 20),
            "random_state": 42, "n_jobs": 2,
        }
    elif model_type == "gradientboosting":
        return {
            "n_estimators": trial.suggest_int(f"{p}gb_n_est", 50, 300),
            "max_depth": trial.suggest_int(f"{p}gb_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int(f"{p}gb_min_leaf", 2, 20),
            "random_state": 42,
        }
    elif model_type == "elasticnet":
        return {
            "alpha": trial.suggest_float(f"{p}en_alpha", 0.001, 10.0, log=True),
            "l1_ratio": trial.suggest_float(f"{p}en_l1", 0.1, 0.9),
            "max_iter": 2000, "random_state": 42,
        }
    elif model_type == "svr_rbf":
        return {
            "kernel": "rbf",
            "C": trial.suggest_float(f"{p}svr_C", 0.01, 100.0, log=True),
            "epsilon": trial.suggest_float(f"{p}svr_eps", 0.01, 1.0, log=True),
            "gamma": "scale",
        }
    elif model_type == "svr_linear":
        return {
            "kernel": "linear",
            "C": trial.suggest_float(f"{p}svrl_C", 0.01, 100.0, log=True),
            "epsilon": trial.suggest_float(f"{p}svrl_eps", 0.01, 1.0, log=True),
        }
    elif model_type == "krr":
        return {
            "alpha": trial.suggest_float(f"{p}krr_alpha", 0.01, 10.0, log=True),
            "kernel": "rbf",
            "gamma": trial.suggest_float(f"{p}krr_gamma", 1e-4, 1.0, log=True),
        }
    elif model_type == "dnn":
        # 표준 38 HP 공간 — simulation.models._optuna_samplers.suggest_tabular_dnn_hp 위임.
        # n_layers 1~12, hidden 2~9999 (log), lr 1e-5~1e-2 (log),
        # weight_decay 1e-6~1e-3 (log), dropout 0.1~0.5,
        # 11 activations / 8 optimizers / 7 norms / 6 inits / 4 layer_types,
        # gradient_clip / loss / lr_schedule / warmup_epochs / momentum / betas / eps.
        # ⚠ prefix 무시 — 표준 함수는 fixed `td_*` 키 사용. feature `use_*` 와 충돌 없음.
        from simulation.models._optuna_samplers import suggest_tabular_dnn_hp
        hp = suggest_tabular_dnn_hp(trial)
        # back-compat 키 추가 — _train_dnn 의 옛 fallback 경로 호환.
        hp["wd"] = hp["weight_decay"]
        hp["bs"] = hp["batch_size"]
        return hp
    elif model_type == "gp_rbf_periodic":
        return {
            "alpha": trial.suggest_float(f"{p}gp_alpha", 1e-4, 1.0, log=True),
            "n_restarts": trial.suggest_int(f"{p}gp_restarts", 1, 5),
        }
    elif model_type == "bayesianridge":
        return {
            "alpha_1": trial.suggest_float(f"{p}br_a1", 1e-8, 1e-3, log=True),
            "alpha_2": trial.suggest_float(f"{p}br_a2", 1e-8, 1e-3, log=True),
            "lambda_1": trial.suggest_float(f"{p}br_l1", 1e-8, 1e-3, log=True),
            "lambda_2": trial.suggest_float(f"{p}br_l2", 1e-8, 1e-3, log=True),
        }
    elif model_type == "negbinglm":
        return {
            "alpha": trial.suggest_float(f"{p}nb_alpha", 0.1, 10.0, log=True),
            "max_iter": 200,
        }
    elif model_type == "gam":
        return {
            "n_splines": trial.suggest_int(f"{p}gam_splines", 10, 40),
            "lam": trial.suggest_float(f"{p}gam_lam", 0.01, 10.0, log=True),
        }
    # bayesianmcmc, poissonautoreg → HP 탐색 공간 제한적 → 기본값 반환
    return _default_hp(model_type)


def _safe_init(cls, hp):
    """cls.__init__이 받을 수 있는 파라미터만 필터링해서 인스턴스 생성."""
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    # **kwargs가 있으면 전부 통과 (LightGBM/XGBoost 등)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return cls(**hp)
    filtered = {k: v for k, v in hp.items() if k in valid_keys}
    dropped = set(hp.keys()) - valid_keys
    if dropped:
        log.debug(f"  [_safe_init] {cls.__name__}: 지원하지 않는 파라미터 제외 → {dropped}")
    return cls(**filtered)


def _build_model(model_type, hp):
    """HP 딕셔너리로 모델 생성."""
    model_type = _resolve_proxy(model_type)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        return _safe_init(LGBMRegressor, hp)
    elif model_type == "xgboost":
        from xgboost import XGBRegressor
        return _safe_init(XGBRegressor, hp)
    elif model_type == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        return _safe_init(RandomForestRegressor, hp)
    elif model_type == "gradientboosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return _safe_init(GradientBoostingRegressor, hp)
    elif model_type == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return Pipeline([("scaler", StandardScaler()), ("en", _safe_init(ElasticNet, hp))])
    elif model_type in ("svr_rbf", "svr_linear"):
        from sklearn.svm import SVR
        return Pipeline([("scaler", StandardScaler()), ("svr", _safe_init(SVR, hp))])
    elif model_type == "krr":
        from sklearn.kernel_ridge import KernelRidge
        return Pipeline([("scaler", StandardScaler()), ("krr", _safe_init(KernelRidge, hp))])
    elif model_type == "bayesianridge":
        from sklearn.linear_model import BayesianRidge
        return Pipeline([("scaler", StandardScaler()), ("br", _safe_init(BayesianRidge, hp))])
    elif model_type == "gp_rbf_periodic":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        alpha = hp.get("alpha", 1e-2)
        kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=0.5)
        return Pipeline([("scaler", StandardScaler()),
                         ("gp", GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                                          n_restarts_optimizer=hp.get("n_restarts", 3),
                                                          random_state=42))])
    elif model_type == "negbinglm":
        # statsmodels NegBin → sklearn 인터페이스 래핑이 어려워 Ridge로 대체 평가
        from sklearn.linear_model import Ridge
        return Pipeline([("scaler", StandardScaler()),
                         ("ridge", Ridge(alpha=hp.get("alpha", 1.0)))])
    elif model_type == "gam":
        # G-159 fix (2026-05-02): n_splines / terms / topk 모두 명시 필수.
        # 이전 (BUG): `LinearGAM(lam=lam)` 만 → n_splines 미사용,
        #   terms 미명시, topk 없음 → 49 features × 20 splines = 980 basis
        #   vs n_train=150 → singular matrix → 모든 trial score=100.0 sentinel.
        # 수정: epi_models.GAMForecaster 직접 사용 (topk=10, terms 명시,
        #   n_splines/lam Optuna search 정상 동작, log1p Y, scaler 모두 포함).
        # mini test 검증: GAMForecaster (splines=39, lam=8.84) 직접 호출 OK.
        from simulation.models.epi_models import GAMForecaster
        return GAMForecaster(
            n_splines=hp.get("n_splines", 20),
            lam=hp.get("lam", 0.6),
            topk=10,
        )
    elif model_type in ("bayesianmcmc", "poissonautoreg"):
        # MCMC/Poisson은 sklearn 인터페이스가 없으므로 BayesianRidge로 대체 평가
        from sklearn.linear_model import BayesianRidge
        return Pipeline([("scaler", StandardScaler()), ("br", BayesianRidge())])
    return None


def _get_activation(name: str):
    """이름 → nn.Module 매핑.

    2026-05-12 Codex BUG fix: sampler 가 11 종 suggest 하는데 mapper 가 7 종만
    인식 → tanh/softplus/prelu/celu 가 silent fallback to ReLU 였음.
    이제 sampler 11 종 모두 정확 매핑 (사용자 "categorical 보존" 요구).
    """
    import torch.nn as nn
    _MAP = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
        "mish": nn.Mish,
        "elu": nn.ELU,
        "swish": nn.SiLU,         # SiLU = Swish
        "tanh": nn.Tanh,          # NEW (Day 12 fix)
        "softplus": nn.Softplus,  # NEW
        "prelu": lambda: nn.PReLU(),  # NEW
        "celu": nn.CELU,          # NEW
    }
    cls = _MAP.get(name, nn.ReLU)
    return cls()


def _get_norm(name: str, dim: int):
    """이름 → normalization layer.

    2026-05-12 Codex BUG fix: sampler 가 7 종 suggest 하는데 mapper 가 3 종만
    (layer/batch/Identity) 인식. 이제 7 종 모두 정확 매핑.
    """
    import torch.nn as nn
    if name == "layer":
        return nn.LayerNorm(dim)
    elif name == "batch":
        return nn.BatchNorm1d(dim)
    elif name == "group":
        # GroupNorm: groups 개수 자동 (gcd(dim, 8) 이상)
        groups = max(1, min(8, dim))
        while dim % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, dim)
    elif name == "instance":
        # G-237 (2026-05-30): InstanceNorm1d treats dim-0 of a 2-D (batch, features)
        # tensor as the channel axis → it compares batch_size vs num_features and raises
        # "expected input's size at dim=0 to match num_features (H), but got: B"
        # (the "16 vs 32" fold failures). It is also undefined for length-1 tabular
        # sequences. Tabular input has no spatial axis, so instance-norm degenerates to
        # a no-op → return Identity (as weight/spectral already do).
        return nn.Identity()
    elif name == "weight":
        # weight_norm 은 wrapper — Linear 에 적용해야 함, 단독 Identity 반환
        # (실제 weight_norm 적용은 model build 시 별도)
        return nn.Identity()
    elif name == "spectral":
        # spectral_norm 도 wrapper, 단독은 Identity
        return nn.Identity()
    else:
        return nn.Identity()


def _apply_init(model, init_type: str):
    """가중치 초기화 전략 적용.

    2026-05-12 Codex BUG fix: sampler 가 6 종 (kaiming_uniform/_normal, xavier_*,
    orthogonal, trunc_normal) suggest 하는데 mapper 가 3 종 (kaiming/xavier/lecun)
    만 인식. back-compat 유지 + 정확 매핑 추가.
    """
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Exact match (sampler 11 종)
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif init_type == "trunc_normal":
                nn.init.trunc_normal_(m.weight, std=0.02)
            # Back-compat (옛 alias)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "lecun":
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            # "default" → PyTorch 기본 init 유지
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ══════════════════════════════════════════════════════════
# DNN 고급 building blocks (표준 38-HP suggester 의 layer_type 등 지원)
# ══════════════════════════════════════════════════════════

def _make_block_layers(in_dim, h_dim, drop, norm_name, act_name, use_bias):
    """공통 inner: Linear → Norm → Activation → Dropout 시퀀스."""
    import torch.nn as nn
    layers = [nn.Linear(in_dim, h_dim, bias=use_bias),
              _get_norm(norm_name, h_dim),
              _get_activation(act_name)]
    if drop > 0:
        layers.append(nn.Dropout(drop))
    return layers


def _build_dnn(n_features, hp):
    """HP 딕셔너리로 DNN 생성 — 표준 38 HP 의 모든 architecture knob 지원.

    Architecture knobs (`hp` 키):
      - n_layers, hidden_dims, dropouts (per-layer)
      - activation, norm, init (legacy)
      - **layer_type**: linear / residual / dense_block / highway   ← NEW
      - **use_bias**: bool                                          ← NEW
      - **use_attention**: bool — final block 뒤에 self-attention   ← NEW
      - **use_fm**: bool — input 단에 Factorization Machine layer   ← NEW
      - **skip_connection**: bool — input → output direct skip      ← NEW

    Back-compat:
      - 옛 키 (h1/h2/d1/d2/use_h3) 자동 변환.
      - 새 knob 누락 시 기본값 = linear / no attention / no fm / bias / no skip
        → 이전 Sequential 모델과 bit-exact (architecture).
    """
    import torch
    import torch.nn as nn

    hidden_dims = hp.get("hidden_dims", [128, 64])
    drop_list   = hp.get("dropouts",    [0.25, 0.15])
    act_name    = hp.get("activation",  "relu")
    norm_name   = hp.get("norm",        "none")
    init_name   = hp.get("init",        "default")
    layer_type  = hp.get("layer_type",  "linear")
    use_bias    = bool(hp.get("use_bias", True))
    use_attn    = bool(hp.get("use_attention", False))
    use_fm      = bool(hp.get("use_fm", False))
    use_skip    = bool(hp.get("skip_connection", False))

    # backward compat: 옛 스타일 (h1/h2/use_h3)
    if "hidden_dims" not in hp and "h1" in hp:
        h1 = hp.get("h1", 128); h2 = hp.get("h2", 64)
        d1 = hp.get("d1", 0.25); d2 = hp.get("d2", 0.15)
        if hp.get("use_h3", False):
            hidden_dims = [h1, h2, max(16, h2 // 2)]
            drop_list   = [d1, d2, d2 * 0.5]
        else:
            hidden_dims = [h1, h2]; drop_list = [d1, d2]

    # dropouts 길이 맞춤
    while len(drop_list) < len(hidden_dims):
        drop_list.append(drop_list[-1] if drop_list else 0.2)

    # ── layer_type=linear + 모든 advanced 옵션 OFF → 옛 Sequential path ──
    # (back-compat: 정확히 옛 모델 구조 재현)
    if (layer_type == "linear" and not use_attn and not use_fm and not use_skip):
        layers = []
        in_dim = n_features
        for h_dim, drop in zip(hidden_dims, drop_list):
            layers.extend(_make_block_layers(in_dim, h_dim, drop,
                                             norm_name, act_name, use_bias))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1, bias=use_bias))
        model = nn.Sequential(*layers)
        _apply_init(model, init_name)
        return model

    # ── 고급 path: layer_type / attention / fm / skip 사용 시 ──
    # 동적 nn.Module 정의 (closure 로 hp 캡처)

    class _ResBlock(nn.Module):
        """Linear→Norm→Act→Drop→Linear → + skip(in→out)."""
        def __init__(self, in_d, out_d, drop):
            super().__init__()
            self.l1 = nn.Linear(in_d, out_d, bias=use_bias)
            self.norm1 = _get_norm(norm_name, out_d)
            self.act1 = _get_activation(act_name)
            self.drop1 = nn.Dropout(drop) if drop > 0 else nn.Identity()
            self.l2 = nn.Linear(out_d, out_d, bias=use_bias)
            self.skip_proj = (nn.Linear(in_d, out_d, bias=False)
                              if in_d != out_d else nn.Identity())
        def forward(self, x):
            h = self.l1(x); h = self.norm1(h); h = self.act1(h); h = self.drop1(h)
            h = self.l2(h)
            return h + self.skip_proj(x)

    class _DenseBlock(nn.Module):
        """DenseNet: out = concat(input, Linear(input)). out_dim ≥ in_dim 필요."""
        def __init__(self, in_d, out_d, drop):
            super().__init__()
            self.delta = max(1, out_d - in_d)
            self.l1 = nn.Linear(in_d, self.delta, bias=use_bias)
            self.norm1 = _get_norm(norm_name, self.delta)
            self.act1 = _get_activation(act_name)
            self.drop1 = nn.Dropout(drop) if drop > 0 else nn.Identity()
            self.out_dim = in_d + self.delta
        def forward(self, x):
            h = self.l1(x); h = self.norm1(h); h = self.act1(h); h = self.drop1(h)
            return torch.cat([x, h], dim=-1)

    class _HighwayBlock(nn.Module):
        """Highway: y = T*H(x) + (1-T)*x where T = sigmoid(WT*x)."""
        def __init__(self, in_d, out_d, drop):
            super().__init__()
            self.proj = (nn.Linear(in_d, out_d, bias=use_bias)
                         if in_d != out_d else nn.Identity())
            self.h_lin = nn.Linear(out_d, out_d, bias=use_bias)
            self.t_lin = nn.Linear(out_d, out_d, bias=True)  # gate always biased
            self.norm1 = _get_norm(norm_name, out_d)
            self.act1 = _get_activation(act_name)
            self.drop1 = nn.Dropout(drop) if drop > 0 else nn.Identity()
        def forward(self, x):
            x = self.proj(x)
            h = self.h_lin(x); h = self.norm1(h); h = self.act1(h); h = self.drop1(h)
            t = torch.sigmoid(self.t_lin(x))
            return t * h + (1.0 - t) * x

    class _LinearBlock(nn.Module):
        """linear: 옛 Sequential 블록 (Linear→Norm→Act→Drop) 의 Module 래퍼."""
        def __init__(self, in_d, out_d, drop):
            super().__init__()
            self.layers = nn.Sequential(
                *_make_block_layers(in_d, out_d, drop, norm_name, act_name, use_bias)
            )
        def forward(self, x):
            return self.layers(x)

    class _FMLayer(nn.Module):
        """Factorization Machine: pairwise feature interactions.

        out = concat(x, fm_features), where fm = 0.5 * sum_k ((xV)^2 - x²V²) over fm_dim.
        fm_dim 은 내부 hidden dimension (16 fixed for stability).
        """
        def __init__(self, in_d, fm_dim=16):
            super().__init__()
            self.V = nn.Parameter(torch.randn(in_d, fm_dim) * 0.01)
            self.out_dim = in_d + fm_dim
        def forward(self, x):
            xv = x @ self.V                         # (B, fm_dim)
            x2v2 = (x ** 2) @ (self.V ** 2)         # (B, fm_dim)
            fm = 0.5 * (xv ** 2 - x2v2)
            return torch.cat([x, fm], dim=-1)

    class _SelfAttention(nn.Module):
        """1D self-attention over feature dim — features 를 1-step sequence 로 처리."""
        def __init__(self, dim):
            super().__init__()
            # head 수: dim 의 약수 중 [1, 2, 4, 8] 에서 max
            n_heads = 1
            for h in [8, 4, 2, 1]:
                if dim % h == 0:
                    n_heads = h; break
            self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)
        def forward(self, x):
            x_seq = x.unsqueeze(1)                 # (B, 1, dim)
            out, _ = self.attn(x_seq, x_seq, x_seq)
            return self.norm((x_seq + out).squeeze(1))

    class _AdvancedDNN(nn.Module):
        """Advanced wrapper — FM frontend → blocks → (attention) → head + (skip)."""
        def __init__(self):
            super().__init__()
            in_dim = n_features
            # FM frontend
            self.fm = _FMLayer(n_features, fm_dim=16) if use_fm else None
            if self.fm is not None:
                in_dim = self.fm.out_dim

            blocks = []
            for h_dim, drop in zip(hidden_dims, drop_list):
                if layer_type == "residual":
                    blocks.append(_ResBlock(in_dim, h_dim, drop));     in_dim = h_dim
                elif layer_type == "dense_block":
                    b = _DenseBlock(in_dim, h_dim, drop)
                    blocks.append(b);                                   in_dim = b.out_dim
                elif layer_type == "highway":
                    blocks.append(_HighwayBlock(in_dim, h_dim, drop)); in_dim = h_dim
                else:  # "linear" (왔을 수도 있음, advanced path 안에서)
                    blocks.append(_LinearBlock(in_dim, h_dim, drop));  in_dim = h_dim
            self.blocks = nn.ModuleList(blocks)

            # Self-attention 후처리 (final block 뒤)
            self.attn = _SelfAttention(in_dim) if use_attn else None

            # Head
            self.head = nn.Linear(in_dim, 1, bias=use_bias)

            # Direct input → output skip
            self.skip_proj = (nn.Linear(n_features, 1, bias=False)
                              if use_skip else None)

        def forward(self, x):
            x_in = x
            if self.fm is not None:
                x = self.fm(x)
            for blk in self.blocks:
                x = blk(x)
            if self.attn is not None:
                x = self.attn(x)
            out = self.head(x)
            if self.skip_proj is not None:
                out = out + self.skip_proj(x_in)
            return out

    model = _AdvancedDNN()
    _apply_init(model, init_name)
    return model


def _train_dnn(X_train, y_train, X_test, hp, seed=42):
    """DNN 학습 + 예측. 피처 선택용 경량 모드 (y 스케일링 포함)."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    # y 스케일링 — 수렴 안정화 핵심
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train)) + 1e-8
    y_tr_scaled = (y_train - y_mean) / y_std

    n_feat = X_tr.shape[1]
    val_size = max(10, int(len(X_tr) * 0.15))
    X_tr_s, X_val = X_tr[:-val_size], X_tr[-val_size:]
    y_tr_s, y_val = y_tr_scaled[:-val_size], y_tr_scaled[-val_size:]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_dnn(n_feat, hp)

    lr = hp.get("lr", 1e-3)
    # weight_decay (standard) → wd (legacy) fallback
    wd = hp.get("weight_decay", hp.get("wd", 1e-4))
    # batch_size (standard) → bs (legacy) fallback
    bs = hp.get("batch_size", hp.get("bs", 32))

    # 표준 38-HP suggester 의 추가 knob (back-compat default)
    momentum = hp.get("momentum", 0.9)
    nesterov = hp.get("nesterov", True)
    beta1    = hp.get("beta1", 0.9)
    beta2    = hp.get("beta2", 0.999)
    eps      = hp.get("eps", 1e-8)
    grad_clip = hp.get("gradient_clip", 1.0)
    loss_name = hp.get("loss", "mse")  # G-218: default huber → mse (huber-loss-banned-20260520)
    sched_name = hp.get("lr_schedule", "cosine_restart")
    warmup_epochs = int(hp.get("warmup_epochs", 0))

    # ── Optimizer (8 종 지원) ──
    optim_name = hp.get("optimizer", "adamw")
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd,
                                     betas=(beta1, beta2), eps=eps)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                      betas=(beta1, beta2), eps=eps)
    elif optim_name == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=wd,
                                      betas=(beta1, beta2), eps=eps)
    elif optim_name == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=wd,
                                      betas=(beta1, beta2), eps=eps)
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd,
                                        momentum=momentum, eps=eps)
    elif optim_name in ("sgd", "sgd_momentum"):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                                    momentum=momentum, nesterov=bool(nesterov))
    elif optim_name in ("lamb", "ranger"):
        # PyTorch 미지원 → AdamW 로 fallback (메시지는 verbose 줄임).
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                      betas=(beta1, beta2), eps=eps)
    else:  # 기타 → AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                      betas=(beta1, beta2), eps=eps)

    # ── LR scheduler (7 종 지원) ──
    if sched_name in ("cosine", "cosine_restart"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=lr * 0.01
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif sched_name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    elif sched_name == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=lr * 0.1, max_lr=lr, step_size_up=20,
            mode="triangular2", cycle_momentum=False,
        )
    elif sched_name == "warmup_cosine":
        # 간이 warmup → cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=120, eta_min=lr * 0.01
        )
    elif sched_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    else:  # "none" 또는 미지원
        scheduler = None

    # ── Loss (4 종 지원) ──
    # G-218: huber + smooth_l1 (HuberLoss equivalent) 영구 제거 (huber-loss-banned-20260520)
    if loss_name == "mae":
        loss_fn = nn.L1Loss()
    elif loss_name == "logcosh":
        # logcosh = log(cosh(x)). PyTorch 내장 없음 → 구현
        def _logcosh(pred, target):
            d = pred - target
            return torch.mean(d + torch.nn.functional.softplus(-2.0 * d) - np.log(2.0))
        loss_fn = _logcosh
    elif loss_name == "quantile":
        # q=0.5 → MAE 와 동일. 회귀에서는 단순화.
        loss_fn = nn.L1Loss()
    else:  # "mse" (default — G-218 안전 fallback)
        loss_fn = nn.MSELoss()

    X_t = torch.FloatTensor(X_tr_s)
    y_t = torch.FloatTensor(y_tr_s).unsqueeze(1)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val).unsqueeze(1)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15           # 피처선택용 (경량)
    max_epochs = 120        # 피처선택용 (경량)
    best_state = None

    try:
        for epoch in range(max_epochs):
            # ── Linear warmup (선택) ──
            if warmup_epochs and epoch < warmup_epochs:
                warmup_scale = (epoch + 1) / max(1, warmup_epochs)
                for g in optimizer.param_groups:
                    g["lr"] = lr * warmup_scale

            model.train()
            indices = torch.randperm(len(X_t))
            for start in range(0, len(X_t), bs):
                idx = indices[start:min(start + bs, len(X_t))]
                optimizer.zero_grad()
                pred_t = model(X_t[idx])
                loss = loss_fn(pred_t, y_t[idx])
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = float(loss_fn(val_pred, y_v).item())

            # ── Scheduler step (epoch-level) ──
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    try:
                        scheduler.step()
                    except Exception:
                        pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.FloatTensor(X_te)).numpy().ravel()
        # y 역변환
        pred = pred_scaled * y_std + y_mean
        # G-159 (2026-05-02): NaN/inf/-inf → 0.0 sanitize (값 없을 때만).
        # 음수 prediction 보존 (사용자 명시 의도). nonneg=False default.
        from simulation.models.base import sanitize_predictions
        return sanitize_predictions(pred)
    finally:
        # ── 메모리 해제: torch 텐서 + 모델 + 옵티마이저 + 스케줄러 ──
        try:
            del model, optimizer, X_t, y_t, X_v, y_v, best_state
        except Exception:
            pass
        if scheduler is not None:
            try:
                del scheduler
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _fit_predict(model_type, hp, X_train, y_train, X_test):
    """모델 학습 + 예측 (공통 인터페이스).

    G-159 (2026-05-02): 모든 prediction 에 sanitize_predictions 적용 →
    **NaN/None/±inf 만** 0.0 으로 치환 (사용자 명시 "값이 없을 경우에만").
    음수 prediction 등 정상 값은 그대로 보존 → 모델 진단/디버깅 가능.
    log1p inverse 발산 (G-146), GAM singular matrix, DL prediction
    overflow 같은 numerical issue 가 downstream WIS/PI 로 전파되는
    것 차단.
    """
    from simulation.models.base import sanitize_predictions
    model_type = _resolve_proxy(model_type)
    if model_type == "dnn":
        pred = _train_dnn(X_train, y_train, X_test, hp)
    else:
        model = _build_model(model_type, hp)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        del model
    # G-159: invalid sentinel (NaN/inf/-inf) 만 0.0, 음수 등 정상 값 보존.
    return sanitize_predictions(pred)  # nonneg=False default


# ══════════════════════════════════════════════════════════
# 5. Walk-Forward CV 공통 평가 함수
# ══════════════════════════════════════════════════════════

_TRIAL_TIMEOUT = 120  # DNN 등 느린 모델: trial 1개 최대 120초


def _safe_best_value(study):
    """완료된 trial이 없으면 None 반환 (ValueError 방지)."""
    try:
        return study.best_value
    except (ValueError, AttributeError):
        return None


def _make_trial_logger(label: str, n_trials: int):
    """Optuna trial 마다 한 줄 print 하는 callback 생성.

    OPTUNA_VERBOSE=1 또는 OPTUNA_VERBOSE=trial 일 때만 활성화.
    출력 형식: `[label] Trial 7/50: value=+1.2345 (best=+0.8765)  obj=wis`.
    """
    verbose = os.environ.get("OPTUNA_VERBOSE", "0")
    objective = os.environ.get("OPTUNA_OBJECTIVE", "rmse").lower()
    if verbose not in ("1", "2", "trial"):
        return None  # disabled

    def _cb(study, trial):
        if trial.value is None:
            return
        try:
            best = study.best_value
        except Exception:
            best = float("inf")
        # 줄 끝에서 \r 덮어쓰지 않고 newline 으로 (logs/CI-friendly)
        sys.stdout.write(
            f"  [{label}] Trial {trial.number+1:>3d}/{n_trials}: "
            f"value={trial.value:+.4f} (best={best:+.4f})  obj={objective}\n"
        )
        sys.stdout.flush()
    return _cb

def _compute_score(y_true, y_pred, residuals_train, objective: str) -> float:
    """객관함수 계산 (MINIMIZE 방향) — RMSE / WIS / MAE / Huber / R² 지원.

    - rmse  : 표준 sqrt(MSE).  (default, back-compat)
    - wis   : Weighted Interval Score (Bracher 2021, FluSight 표준).
              점예측 + train residual std 로 PI 생성 → q05/25/50/75/95 pinball
              평균 + interval span. 보건역학적 가치 ↑.
    - mae   : 평균절대오차.
    - huber : Huber(delta=1.0) — train loss 와 align.
    - r2    : -R² (음수화 → MINIMIZE 일관).
    """
    obj = (objective or "rmse").lower()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if obj == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    elif obj == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    # G-218: huber objective 영구 제거 (huber-loss-banned-20260520) — rmse/mae 만
    elif obj == "r2":
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
        return -(1.0 - ss_res / ss_tot)  # 음수화 (MINIMIZE)
    elif obj == "wis":
        # Train residual 표준편차 → Gaussian PI 근사.
        sigma = float(np.std(residuals_train)) if len(residuals_train) >= 5 else None
        if sigma is None or sigma < 1e-6:
            # fallback: AR(1) diff std 또는 RMSE 기본.
            sigma = max(1.0, float(np.std(np.diff(y_true))) if len(y_true) >= 2 else 1.0)

        # Bracher 2021 — K=3 quantile pairs (α=0.5/0.2/0.05 → 25/75, 10/90, 5/95).
        alphas = [0.5, 0.2, 0.05]
        wis_total = 0.0
        n_obs = len(y_true)
        for alpha in alphas:
            from scipy.stats import norm as _norm
            z = _norm.ppf(1.0 - alpha / 2.0)
            q_lo = y_pred - z * sigma
            q_hi = y_pred + z * sigma
            # Interval Score (alpha) = (q_hi - q_lo) + 2/α * max(0, q_lo - y) + 2/α * max(0, y - q_hi)
            span = q_hi - q_lo
            penalty_lo = (2.0 / alpha) * np.maximum(0.0, q_lo - y_true)
            penalty_hi = (2.0 / alpha) * np.maximum(0.0, y_true - q_hi)
            wis_total += np.sum(span + penalty_lo + penalty_hi) * (alpha / 2.0)
        # |y - median| (q50 = pred 사용)
        wis_total += np.sum(np.abs(y_true - y_pred)) * 0.5
        # 정규화: K + 0.5 = 3.5 (3 alphas + median weight 0.5)
        wis_avg = wis_total / (n_obs * (len(alphas) + 0.5))
        return float(wis_avg)
    else:
        # 알 수 없는 objective → RMSE fallback.
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _wf_cv_score(X_sel, y, model_type, hp, n_folds=3, trial=None):
    """Walk-Forward CV 점수 계산 (MINIMIZE).

    객관함수 = `OPTUNA_OBJECTIVE` 환경변수 (rmse/wis/mae/huber/r2).
    기본값 = `rmse` (back-compat).
    `wis` (사용자 결정 — 보건역학 가치) 권장.
    trial 있으면 fold-level pruning 지원.

    2026-04-28 critical fix: HWP §3 정합 — test 68 보호.
    이전: n=len(y)=337 → fold 3 가 test 영역 (270-337) 일부 평가
        → HP 가 test 에 overfit 위험 (snooping)
    변경: n = train_pool_end (337 - 68 = 269) — Optuna 는 train_pool 만 봄
        test 68 은 R10 per_model_eval final eval 에만, real 8 은 P1 real_forecaster 에만.
    환경변수 `MPH_OPTUNA_USE_FULL=1` 로 옛 동작 (back-compat) 가능.
    """
    objective = os.environ.get("OPTUNA_OBJECTIVE", "rmse").lower()
    n_full = len(y)

    # 2026-04-28: test 보호 — train_pool 까지만 Optuna 에 노출
    # 2026-04-29 (취소): MPH_OPTUNA_TRAIN_ONLY 옵션 제거 (사용자 요청).
    #   val 은 fold 의 일부로 자연스럽게 사용 (HWP §3 default 유지).
    if GLOBAL.optuna.use_full:
        n = n_full   # 옛 동작 (test fold 에 들어감)
    else:
        # HWP §3 default: test_ratio = 0.20 → n_test = ceil(n * 0.20)
        # train_pool = n - n_test (이 안에서 WF-CV — val 은 fold 의 일부)
        import math
        n_test = math.ceil(n_full * 0.20)
        n = n_full - n_test    # train_pool_end

    min_train = max(150, n // 3)
    fold_size = (n - min_train) // n_folds
    scores = []
    t0_trial = time.time()

    # G-159 fix (2026-05-02): silent fail 흔적 제거.
    # 이전: scores.append(100.0) — 100 sentinel 가 정상 RMSE=100 과 구분 안 됨,
    #       `log.debug` 가 INFO 레벨 로그에서 silent → GAM 같은 model_type↔hp
    #       mismatch bug 가 6h+ 동안 발견 안 됨.
    # 수정: float("inf") sentinel + log.warning (exception type/message 명시).
    #       Optuna 가 inf trial 을 fail 로 인식해 best 후보에서 자동 제외.

    for fold in range(n_folds):
        # trial별 타임아웃 체크
        if time.time() - t0_trial > _TRIAL_TIMEOUT:
            log.warning(f"  [{model_type}] Fold {fold} TIMEOUT (>{_TRIAL_TIMEOUT}s) "
                        f"— hp={hp}")
            scores.append(float("inf"))
            continue

        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n) if fold < n_folds - 1 else n
        if test_end <= train_end or len(y[train_end:test_end]) < 3:
            continue

        try:
            pred = _fit_predict(model_type, hp,
                                X_sel[:train_end], y[:train_end],
                                X_sel[train_end:test_end])
            # WIS: train residual std 가 필요 → in-sample fit 의 residual 사용.
            # 단순화: train 의 last-fold residual = train pred - train y (없으면 diff).
            if objective == "wis":
                # in-sample residual: train 의 마지막 fold_size 만큼 hold-out 으로 추정.
                k = max(10, min(fold_size, train_end // 4))
                try:
                    pred_in = _fit_predict(model_type, hp,
                                           X_sel[:train_end - k], y[:train_end - k],
                                           X_sel[train_end - k:train_end])
                    residuals_train = y[train_end - k:train_end] - pred_in
                except Exception as _re:
                    log.debug(f"  [{model_type}] in-sample residual failed: "
                              f"{type(_re).__name__}: {_re}")
                    residuals_train = np.diff(y[:train_end])
            else:
                residuals_train = np.array([])

            score = _compute_score(y[train_end:test_end], pred,
                                   residuals_train, objective)
            # G-159 fix: NaN/Inf prediction 명시적 차단 (silent 100.0 대신).
            if not np.isfinite(score):
                log.warning(f"  [{model_type}] Fold {fold} non-finite score={score} "
                            f"— hp={hp}, pred range=[{np.nanmin(pred):.2f}, {np.nanmax(pred):.2f}]")
                scores.append(float("inf"))
            else:
                scores.append(score)

            if trial is not None:
                trial.report(np.mean(scores), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # G-159 fix: log.debug → log.warning. exception type 명시 → 다음
            # bug 가 즉시 보이게.
            log.warning(f"  [{model_type}] Fold {fold} FAILED: "
                        f"{type(e).__name__}: {e} — hp={hp}")
            scores.append(float("inf"))
        gc.collect()

    if not scores:
        return float("inf")
    return float(np.mean(scores))


def _select_features_from_trial(trial, feature_cols):
    """trial에서 피처 선택 마스크 생성.

    [2026-04-29] 2가지 모드:

    1. Individual mode (기존, default):
       모든 non-mandatory feature 마다 use_<col> bool search.
       Search space = 2^(n_features - n_mandatory).

    2. Family-first mode (MPH_FEATURE_FAMILY_FIRST=1):
       Stage 1: 각 family on/off (예: family_weather=on, family_search_trend=off)
       Stage 2: enabled family 안에서만 individual feature search
       Search space 가 family 단위로 더 효율적으로 탐색됨 (TPE marginal 추정 향상).
       Mandatory features 의 family 는 자동 enabled.
    """
    family_first = GLOBAL.training.feature_family_first
    selected_mask = np.zeros(len(feature_cols), dtype=bool)
    n_mandatory = 0

    if family_first:
        # ── Stage 1: family on/off ──
        try:
            from simulation.models.grouped_preprocessor import classify_feature
        except ImportError:
            classify_feature = lambda c: "other"

        # 각 column 의 family + mandatory 여부 미리 계산
        col_family = [classify_feature(c) for c in feature_cols]
        col_mandatory = [_is_mandatory(c) for c in feature_cols]

        # Mandatory features 가 속한 family 는 항상 enabled (Stage 1 skip)
        mandatory_families = {col_family[i]
                               for i in range(len(feature_cols)) if col_mandatory[i]}
        all_families = set(col_family)

        family_enabled = {}
        for fam in sorted(all_families):
            if fam in mandatory_families:
                family_enabled[fam] = True   # 항상 on (mandatory 보호)
            else:
                family_enabled[fam] = trial.suggest_categorical(
                    f"family_{fam}", [True, False]
                )

        # ── Stage 2: enabled family 안에서 individual ──
        for i, col in enumerate(feature_cols):
            if col_mandatory[i]:
                selected_mask[i] = True
                n_mandatory += 1
            else:
                fam = col_family[i]
                if family_enabled.get(fam, False):
                    # Family enabled → individual selection
                    selected_mask[i] = trial.suggest_categorical(
                        f"use_{col}", [True, False]
                    )
                else:
                    # Family disabled → 전체 skip
                    selected_mask[i] = False
    else:
        # 기존 individual-only 모드
        for i, col in enumerate(feature_cols):
            if _is_mandatory(col):
                selected_mask[i] = True
                n_mandatory += 1
            else:
                selected_mask[i] = trial.suggest_categorical(
                    f"use_{col}", [True, False]
                )

    n_selected = selected_mask.sum()
    if n_selected < max(5, n_mandatory + 3):
        return None, 0
    return selected_mask, n_mandatory


def _create_study(study_name: str, resume: bool):
    """Optuna study 생성 공통 헬퍼 (중복 제거).

    - resume=False: 기존 study 삭제 후 새로 생성
    - resume=True: 기존 study 이어서 실행

    Cat 3 (2026-05-12): MPH_OPTUNA_STORAGE_URL env 우선 (PostgreSQL 등).
    """
    storage = _STORAGE_URL
    if not resume:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception:
            pass
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=10),
        pruner=_make_pruner("stage2"),
        load_if_exists=True,
    )
    return study


# ══════════════════════════════════════════════════════════
# 6. Strategy A: Feature Only (HP 고정, 피처만 탐색)
# ══════════════════════════════════════════════════════════

def run_strategy_feature_only(X, y, feature_cols, model_type,
                              n_trials, cv_folds, use_composite, resume):
    """Strategy A: HP 기본값 고정, 피처만 Optuna 탐색."""
    hp = _default_hp(model_type)
    study = _create_study(f"feat_only_{model_type}_{_STUDY_SUFFIX}", resume)

    prog = ProgressLine(f"{model_type} feature_only")
    _completed = [0]

    def objective(trial):
        mask, _ = _select_features_from_trial(trial, feature_cols)
        if mask is None:
            return float("inf")
        X_sel = X[:, mask]

        if use_composite:
            sel_names = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
            X_sel, _ = generate_composite_features(X_sel, sel_names, trial)

        score = _wf_cv_score(X_sel, y, model_type, hp, cv_folds, trial)
        penalty = 0.001 * mask.sum()
        _completed[0] += 1
        best = _safe_best_value(study)
        prog.update(_completed[0], n_trials, best)
        return score + penalty

    t0 = time.time()
    _logger_cb = _make_trial_logger(f"{model_type}/feature_only", n_trials)
    _cbs = [_logger_cb] if _logger_cb else []
    # 2026-04-28 v2: cap=200, 재학습 자유 (skip 안 함)
    # 정책:
    #   - 한 호출 max 200 trial 추가 (cap)
    #   - existing 수 무관하게 항상 추가 (계속 재학습 가능)
    #   - 사용자가 학습 stop 까지 누적 무한 가능
    _existing = len(study.trials)
    _MAX_PER_CALL = GLOBAL.optuna.remaining_cap
    _remaining = min(_MAX_PER_CALL, int(n_trials))
    if _existing > 0:
        _print(f"  🔁 {model_type}/feature_only: existing {_existing} + {_remaining} 추가")
    if _remaining > 0:
        study.optimize(objective, n_trials=_remaining, show_progress_bar=False,
                       gc_after_trial=True, callbacks=_cbs)
    elapsed = time.time() - t0

    if _safe_best_value(study) is None:
        prog.finish("FAIL — 완료된 trial 없음")
        return None

    result = _extract_result(study, feature_cols, model_type, "feature_only", hp, elapsed)
    prog.finish(f"RMSE={result['best_rmse']:.4f}, 피처 {result['n_features_selected']}개")
    return result


# ══════════════════════════════════════════════════════════
# 7. Strategy B: Joint (HP + 피처 동시 탐색)
# ══════════════════════════════════════════════════════════

def run_strategy_joint(X, y, feature_cols, model_type,
                       n_trials, cv_folds, use_composite, resume):
    """Strategy B: HP + 피처를 하나의 study에서 동시 탐색."""
    study = _create_study(f"joint_{model_type}_{_STUDY_SUFFIX}", resume)

    prog = ProgressLine(f"{model_type} joint")
    _completed = [0]

    def objective(trial):
        hp = _suggest_hp(model_type, trial)
        mask, _ = _select_features_from_trial(trial, feature_cols)
        if mask is None:
            return float("inf")
        X_sel = X[:, mask]

        if use_composite:
            sel_names = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
            X_sel, _ = generate_composite_features(X_sel, sel_names, trial)

        score = _wf_cv_score(X_sel, y, model_type, hp, cv_folds, trial)
        penalty = 0.001 * mask.sum()
        _completed[0] += 1
        best = _safe_best_value(study)
        prog.update(_completed[0], n_trials, best)
        return score + penalty

    t0 = time.time()
    _logger_cb = _make_trial_logger(f"{model_type}/joint", n_trials)
    _cbs = [_logger_cb] if _logger_cb else []
    # 2026-04-28 v2: cap=200, 재학습 자유
    _existing_j = len(study.trials)
    _MAX_PER_CALL = GLOBAL.optuna.remaining_cap
    _remaining_j = min(_MAX_PER_CALL, int(n_trials))
    if _existing_j > 0:
        _print(f"  🔁 {model_type}/joint: existing {_existing_j} + {_remaining_j} 추가")
    if _remaining_j > 0:
        study.optimize(objective, n_trials=_remaining_j, show_progress_bar=False,
                       gc_after_trial=True, callbacks=_cbs)
    elapsed = time.time() - t0

    best_val = _safe_best_value(study)
    if best_val is None:
        prog.finish("FAIL — 완료된 trial 없음")
        return None

    best_hp = {}
    for k, v in study.best_trial.params.items():
        if not k.startswith("use_") and not k.startswith("comp_"):
            best_hp[k] = v

    result = _extract_result(study, feature_cols, model_type, "joint", best_hp, elapsed)
    prog.finish(f"RMSE={result['best_rmse']:.4f}, 피처 {result['n_features_selected']}개")
    return result


# ══════════════════════════════════════════════════════════
# 8. Strategy C: HP → Feature (2단계 순차 최적화)
# ══════════════════════════════════════════════════════════

def run_strategy_hp_then_feature(X, y, feature_cols, model_type,
                                 n_trials, n_rounds, cv_folds,
                                 use_composite, resume):
    """Strategy C: stage1 HP Optuna → stage2 피처 Optuna × N회 → 합의."""
    t0_total = time.time()
    storage = _STORAGE_URL  # Cat 3 (2026-05-12): env-gated, stage2 라운드별 study

    # ── stage 1: HP Optuna (전체 피처 사용) ──
    hp_study = _create_study(f"hp_{model_type}_{_STUDY_SUFFIX}", resume)

    prog_hp = ProgressLine(f"{model_type} hp_then_feat [P1:HP]")
    _hp_done = [0]

    def hp_objective(trial):
        hp = _suggest_hp(model_type, trial)
        score = _wf_cv_score(X, y, model_type, hp, cv_folds, trial)
        _hp_done[0] += 1
        best = _safe_best_value(hp_study)
        prog_hp.update(_hp_done[0], n_trials, best)
        return score

    _logger_cb = _make_trial_logger(f"{model_type}/hp_then_feat[P1:HP]", n_trials)
    _cbs = [_logger_cb] if _logger_cb else []
    # 2026-04-28 v2: cap=200, 재학습 자유
    _existing_hp = len(hp_study.trials)
    _MAX_PER_CALL = GLOBAL.optuna.remaining_cap
    _remaining_hp = min(_MAX_PER_CALL, int(n_trials))
    if _existing_hp > 0:
        _print(f"  🔁 {model_type}/hp_then_feat[P1]: existing {_existing_hp} + {_remaining_hp} 추가")
    if _remaining_hp > 0:
        hp_study.optimize(hp_objective, n_trials=_remaining_hp, show_progress_bar=False,
                          gc_after_trial=True, callbacks=_cbs)

    hp_rmse = _safe_best_value(hp_study)
    if hp_rmse is None:
        prog_hp.finish("FAIL — HP 완료 trial 없음")
        return None

    best_hp_raw = hp_study.best_trial.params
    best_hp = _reconstruct_hp(model_type, best_hp_raw)
    prog_hp.finish(f"HP RMSE={hp_rmse:.4f}")

    # ── stage 2: Feature Optuna × N회 (확정 HP 사용) ──
    feature_counter = Counter()
    round_results = []
    feat_trials = max(10, n_trials // 2)
    total_feat_work = n_rounds * feat_trials

    prog_feat = ProgressLine(f"{model_type} hp_then_feat [P2:피처×{n_rounds}]")
    _feat_done = [0]

    for rnd in range(n_rounds):
        seed = 42 + rnd
        feat_study_name = f"feat_r{rnd}_{model_type}_{_STUDY_SUFFIX}"

        if not resume:
            try:
                optuna.delete_study(study_name=feat_study_name, storage=storage)
            except Exception:
                pass

        feat_study = optuna.create_study(
            study_name=feat_study_name, storage=storage, direction="minimize",
            sampler=TPESampler(seed=seed, n_startup_trials=max(5, n_trials // 5)),
            pruner=_make_pruner("stage2"),
            load_if_exists=True,
        )

        def feat_objective(trial):
            mask, _ = _select_features_from_trial(trial, feature_cols)
            if mask is None:
                return float("inf")
            X_sel = X[:, mask]

            if use_composite:
                sel_names = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
                X_sel, _ = generate_composite_features(X_sel, sel_names, trial)

            score = _wf_cv_score(X_sel, y, model_type, best_hp, cv_folds, trial)
            penalty = 0.001 * mask.sum()
            _feat_done[0] += 1
            prog_feat.update(_feat_done[0], total_feat_work,
                             extra=f"R{rnd+1}/{n_rounds}")
            return score + penalty

        _logger_cb = _make_trial_logger(
            f"{model_type}/hp_then_feat[P2:R{rnd+1}/{n_rounds}]", feat_trials)
        _cbs = [_logger_cb] if _logger_cb else []
        # 2026-04-28 v2: cap=200, 재학습 자유
        _existing_p2 = len(feat_study.trials)
        _MAX_PER_CALL = GLOBAL.optuna.remaining_cap
        _remaining_p2 = min(_MAX_PER_CALL, int(feat_trials))
        if _remaining_p2 > 0:
            feat_study.optimize(feat_objective, n_trials=_remaining_p2,
                                show_progress_bar=False, gc_after_trial=True,
                                callbacks=_cbs)

        rnd_best = _safe_best_value(feat_study)
        if rnd_best is None:
            continue  # 이 라운드 스킵

        best_trial = feat_study.best_trial
        rnd_features = []
        for col in feature_cols:
            if _is_mandatory(col):
                rnd_features.append(col)
                feature_counter[col] += 1
            elif best_trial.params.get(f"use_{col}", False):
                rnd_features.append(col)
                feature_counter[col] += 1

        round_results.append({
            "round": rnd,
            "rmse": rnd_best,
            "n_features": len(rnd_features),
            "features": sorted(rnd_features),
        })

    prog_feat.finish(f"{n_rounds}회 완료")

    # ── 합의(Consensus) 피처: 60% 이상 라운드에서 선택된 것 ──
    threshold = max(1, int(n_rounds * 0.6))
    consensus_features = sorted([
        col for col, cnt in feature_counter.items() if cnt >= threshold
    ])

    for col in feature_cols:
        if _is_mandatory(col) and col not in consensus_features:
            consensus_features.append(col)
    consensus_features = sorted(set(consensus_features))

    # ── 합의 피처로 최종 RMSE 측정 ──
    consensus_mask = np.array([col in consensus_features for col in feature_cols])
    X_consensus = X[:, consensus_mask]
    final_rmse = _wf_cv_score(X_consensus, y, model_type, best_hp, cv_folds)

    elapsed = time.time() - t0_total  # 전체 경과 시간

    # 결과 구성
    mandatory_features = [c for c in feature_cols if _is_mandatory(c)]
    optuna_selected = [c for c in consensus_features if not _is_mandatory(c)]

    result = {
        "model_type": model_type,
        "strategy": "hp_then_feature",
        "best_rmse": round(final_rmse, 4),
        "hp_phase_rmse": round(hp_rmse, 4),
        "n_features_total": len(feature_cols),
        "n_features_selected": len(consensus_features),
        "n_mandatory": len(mandatory_features),
        "n_optuna_selected": len(optuna_selected),
        "mandatory_features": mandatory_features,
        "selected_features": consensus_features,
        "composite_features": [],  # consensus에서는 복합변수 빈도 추적 어려움 → 제외
        "best_hp": _make_serializable(best_hp),
        "best_params": _make_serializable(best_hp),
        "n_trials_hp": len(hp_study.trials),
        "n_rounds": n_rounds,
        "consensus_threshold": f"{threshold}/{n_rounds}",
        "round_results": round_results,
        "feature_frequency": {col: cnt / n_rounds for col, cnt in feature_counter.most_common()},
        "elapsed_min": round(elapsed / 60, 1),
    }
    return result


def _reconstruct_hp(model_type, params):
    """Optuna params 딕셔너리를 모델 HP 딕셔너리로 재구성."""
    if model_type == "lightgbm":
        return {
            "n_estimators": params.get("lgb_n_est", 200),
            "max_depth": params.get("lgb_depth", 4),
            "learning_rate": params.get("lgb_lr", 0.05),
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": params.get("lgb_alpha", 0.1),
            "reg_lambda": params.get("lgb_lambda", 1.0),
            "min_child_samples": params.get("lgb_min_child", 20),
            "random_state": 42, "n_jobs": 2, "verbose": -1,
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": params.get("xgb_n_est", 200),
            "max_depth": params.get("xgb_depth", 4),
            "learning_rate": params.get("xgb_lr", 0.05),
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": params.get("xgb_alpha", 0.1),
            "reg_lambda": params.get("xgb_lambda", 1.0),
            "min_child_weight": 5, "random_state": 42, "n_jobs": 2, "verbosity": 0,
        }
    elif model_type in ("randomforest", "gradientboosting"):
        return {
            "n_estimators": params.get("rf_n_est", 200),
            "max_depth": params.get("rf_depth", 6),
            "min_samples_leaf": params.get("rf_min_leaf", 5),
            "random_state": 42, "n_jobs": 2,
        }
    elif model_type == "elasticnet":
        return {
            "alpha": params.get("en_alpha", 0.1),
            "l1_ratio": params.get("en_l1", 0.5),
            "max_iter": 2000, "random_state": 42,
        }
    elif model_type == "svr_rbf":
        return {
            "kernel": "rbf", "gamma": "scale",
            "C": params.get("svr_C", 1.0),
            "epsilon": params.get("svr_eps", 0.1),
        }
    elif model_type == "svr_linear":
        return {
            "kernel": "linear",
            "C": params.get("svrl_C", 1.0),
            "epsilon": params.get("svrl_eps", 0.1),
        }
    elif model_type == "krr":
        return {
            "alpha": params.get("krr_alpha", 1.0),
            "kernel": "rbf",
            "gamma": params.get("krr_gamma", None),
        }
    elif model_type == "dnn":
        # 새 표준 38-HP suggester (`td_*` 키) 우선, 없으면 옛 키 fallback.
        # `td_*` 가 하나라도 있으면 새 형식으로 간주.
        if any(k.startswith("td_") for k in params):
            n_layers = int(params.get("td_n_layers", 2))
            hidden_dims = [int(params.get(f"td_h{i}", 128)) for i in range(n_layers)]
            dropouts    = [float(params.get(f"td_d{i}", 0.2)) for i in range(n_layers)]
            return {
                # standard keys
                "n_layers":     n_layers,
                "hidden_dims":  hidden_dims,
                "dropouts":     dropouts,
                "lr":           params.get("td_lr", 1e-3),
                "weight_decay": params.get("td_l2", 1e-4),
                "batch_size":   params.get("td_bs", 32),
                "activation":   params.get("td_act", "relu"),
                "optimizer":    params.get("td_opt", "adamw"),
                "norm":         params.get("td_norm", "none"),
                "init":         params.get("td_init", "default"),
                "layer_type":   params.get("td_layer_type", "linear"),
                "use_attention": bool(params.get("td_attn", False)),
                "use_fm":       bool(params.get("td_fm", False)),
                "use_bias":     bool(params.get("td_bias", True)),
                "skip_connection": bool(params.get("td_skip", False)),
                "lr_schedule":  params.get("td_lr_sched", "cosine"),
                "gradient_clip": params.get("td_grad_clip", 1.0),
                "loss":         params.get("td_loss", "mse"),  # G-218: default huber → mse (huber-loss-banned-20260520)
                "momentum":     params.get("td_momentum", 0.9),
                "nesterov":     bool(params.get("td_nest", True)),
                "beta1":        params.get("td_beta1", 0.9),
                "beta2":        params.get("td_beta2", 0.999),
                "eps":          params.get("td_eps", 1e-8),
                "warmup_epochs": int(params.get("td_warmup", 0)),
                # back-compat keys
                "wd": params.get("td_l2", 1e-4),
                "bs": params.get("td_bs", 32),
            }
        # 옛 키 (dnn_* prefix)
        return {
            "h1": params.get("dnn_h1", 128),
            "h2": params.get("dnn_h2", 64),
            "d1": params.get("dnn_d1", 0.25),
            "d2": params.get("dnn_d2", 0.15),
            "use_h3": params.get("dnn_h3", False),
            "lr": params.get("dnn_lr", 1e-3),
            "wd": params.get("dnn_wd", 1e-4),
            "bs": params.get("dnn_bs", 32),
            "weight_decay": params.get("dnn_wd", 1e-4),
            "batch_size":  params.get("dnn_bs", 32),
        }
    elif model_type == "gp_rbf_periodic":
        return {
            "alpha": params.get("gp_alpha", 1e-2),
            "n_restarts": params.get("gp_restarts", 3),
        }
    elif model_type == "bayesianridge":
        return {
            "alpha_1": params.get("br_a1", 1e-6),
            "alpha_2": params.get("br_a2", 1e-6),
            "lambda_1": params.get("br_l1", 1e-6),
            "lambda_2": params.get("br_l2", 1e-6),
        }
    elif model_type == "negbinglm":
        return {"alpha": params.get("nb_alpha", 1.0), "max_iter": 200}
    elif model_type == "gam":
        return {
            "n_splines": params.get("gam_splines", 20),
            "lam": params.get("gam_lam", 0.6),
        }
    return _default_hp(model_type)


# ══════════════════════════════════════════════════════════
# 9. Result Extraction (Strategy A, B 공통)
# ══════════════════════════════════════════════════════════

def _make_serializable(obj):
    """numpy/bool_ 등 JSON 직렬화 불가 타입 변환."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _extract_result(study, feature_cols, model_type, strategy, hp, elapsed):
    """Study에서 결과 추출 (Strategy A, B 공통)."""
    best = study.best_trial
    mandatory_features = [col for col in feature_cols if _is_mandatory(col)]
    optuna_selected = [col for col in feature_cols
                       if best.params.get(f"use_{col}", False)]
    selected_features = sorted(set(mandatory_features + optuna_selected))
    composite_features = [k for k, v in best.params.items()
                          if k.startswith("comp_") and v]

    # 피처 선택 빈도
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    feature_freq = {}
    if n_complete > 0:
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            for col in feature_cols:
                if _is_mandatory(col):
                    feature_freq[col] = feature_freq.get(col, 0) + 1
                elif trial.params.get(f"use_{col}", False):
                    feature_freq[col] = feature_freq.get(col, 0) + 1
        feature_freq = {col: cnt / n_complete for col, cnt in
                        sorted(feature_freq.items(), key=lambda x: -x[1])}

    # 결과 로깅은 호출부의 ProgressLine.finish()에서 처리

    result = {
        "model_type": model_type,
        "strategy": strategy,
        "best_rmse": round(study.best_value, 4),
        "n_features_total": len(feature_cols),
        "n_features_selected": len(selected_features),
        "n_mandatory": len(mandatory_features),
        "n_optuna_selected": len(optuna_selected),
        "mandatory_features": mandatory_features,
        "selected_features": selected_features,
        "composite_features": composite_features,
        "best_hp": _make_serializable(hp),
        "best_params": _make_serializable(hp),
        "n_trials_total": len(study.trials),
        "feature_frequency": feature_freq,
        "elapsed_min": round(elapsed / 60, 1),
    }
    return result


# ══════════════════════════════════════════════════════════
# 9b. Strategy D: Mandatory Only (Baseline, Optuna 없음)
# ══════════════════════════════════════════════════════════

def run_strategy_mandatory_only(X, y, feature_cols, model_type, cv_folds):
    """Strategy D: 필수 38개 피처만 사용, HP 기본값. Baseline 비교용."""
    prog = ProgressLine(f"{model_type} baseline")
    prog.update(0, 1)

    t0 = time.time()
    hp = _default_hp(model_type)

    mandatory_mask = np.array([_is_mandatory(col) for col in feature_cols])
    mandatory_features = [col for col, m in zip(feature_cols, mandatory_mask) if m]
    n_mandatory = len(mandatory_features)

    if n_mandatory == 0:
        prog.finish("ERROR: 필수 피처 없음!")
        return None

    X_mandatory = X[:, mandatory_mask]
    rmse = _wf_cv_score(X_mandatory, y, model_type, hp, cv_folds)

    elapsed = time.time() - t0
    prog.finish(f"RMSE={rmse:.4f}, 피처 {n_mandatory}개")

    result = {
        "model_type": model_type,
        "strategy": "mandatory_only",
        "best_rmse": round(rmse, 4),
        "n_features_total": len(feature_cols),
        "n_features_selected": n_mandatory,
        "n_mandatory": n_mandatory,
        "n_optuna_selected": 0,
        "mandatory_features": mandatory_features,
        "selected_features": mandatory_features,
        "composite_features": [],
        "best_hp": _make_serializable(hp),
        "best_params": _make_serializable(hp),
        "n_trials_total": 0,
        "feature_frequency": {col: 1.0 for col in mandatory_features},
        "elapsed_min": round(elapsed / 60, 1),
    }
    return result


# ══════════════════════════════════════════════════════════
# 10. 전략 비교 + 추천
# ══════════════════════════════════════════════════════════

def compare_and_recommend(strategy_results, model_type):
    """전략별 결과 비교, 최적 전략 추천, JSON 저장."""
    _print(f"\n  ┌─ {model_type} 전략 비교 ─────────────────────")
    for name, r in strategy_results.items():
        _print(f"  │ {name:20s} RMSE={r['best_rmse']:.4f}  "
               f"피처 {r['n_features_selected']:3d}  {r['elapsed_min']:.1f}분")

    baseline_rmse = strategy_results.get("mandatory_only", {}).get("best_rmse")
    if baseline_rmse:
        for name, r in strategy_results.items():
            if name == "mandatory_only":
                continue
            delta = baseline_rmse - r["best_rmse"]
            pct = delta / baseline_rmse * 100 if baseline_rmse > 0 else 0
            sign = "↓" if delta > 0 else "↑"
            _print(f"  │   vs baseline: {name} RMSE {sign}{abs(delta):.4f} ({abs(pct):.1f}%)")

    # 최적 전략: RMSE 최소 (mandatory_only 제외)
    non_baseline = {k: v for k, v in strategy_results.items() if k != "mandatory_only"}
    if non_baseline:
        best_strategy = min(non_baseline, key=lambda k: non_baseline[k]["best_rmse"])
    else:
        best_strategy = min(strategy_results, key=lambda k: strategy_results[k]["best_rmse"])
    best_result = strategy_results[best_strategy]

    if baseline_rmse and best_result["best_rmse"] >= baseline_rmse:
        _print(f"  │ ⚠ 모든 전략이 baseline 이하 → mandatory_only 추천")
        best_strategy = "mandatory_only"
        best_result = strategy_results["mandatory_only"]

    _print(f"  └─ ★ 추천: {best_strategy} (RMSE={best_result['best_rmse']:.4f}, "
           f"피처 {best_result['n_features_selected']}개)")

    # ── JSON 저장 ──
    best_result["recommended"] = True
    best_result["all_strategies"] = {
        name: {"rmse": r["best_rmse"],
               "n_features": r["n_features_selected"],
               "elapsed_min": r["elapsed_min"]}
        for name, r in strategy_results.items()
    }

    out_path = SAVE_DIR / f"optuna_feat_sel_{model_type}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(best_result), f, indent=2, ensure_ascii=False)

    all_path = SAVE_DIR / f"optuna_all_strategies_{model_type}.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(strategy_results), f, indent=2, ensure_ascii=False)

    if best_result.get("feature_frequency"):
        freq_path = SAVE_DIR / f"optuna_feat_freq_{model_type}.json"
        with open(freq_path, "w", encoding="utf-8") as f:
            json.dump(best_result["feature_frequency"], f, indent=2, ensure_ascii=False)

    return best_strategy, best_result


# ══════════════════════════════════════════════════════════
# 11. Main
# ══════════════════════════════════════════════════════════

def run_all_strategies(model_type, args):
    """단일 모델에 대해 선택된 전략들을 실행하고 비교."""
    objective = os.environ.get("OPTUNA_OBJECTIVE", "rmse").lower()
    _print(f"\n{'='*55}")
    _print(f"  {model_type} | 전략: {args.strategy} | Trials: {args.n_trials}")
    _print(f"  Objective: {objective.upper()}  "
           f"({'WIS = 보건역학 표준 (FluSight)' if objective == 'wis' else 'MINIMIZE'})")
    _print(f"{'='*55}")

    X, y, feature_cols = load_data()
    _print(f"  Data: {len(y)} samples, {len(feature_cols)} features")

    strategies_to_run = []
    if args.strategy == "all":
        strategies_to_run = ["mandatory_only", "feature_only", "joint", "hp_then_feature"]
    else:
        strategies_to_run = [args.strategy]

    strategy_results = {}

    for strat in strategies_to_run:
        try:
            if strat == "mandatory_only":
                result = run_strategy_mandatory_only(
                    X, y, feature_cols, model_type, args.cv_folds)
            elif strat == "feature_only":
                result = run_strategy_feature_only(
                    X, y, feature_cols, model_type,
                    args.n_trials, args.cv_folds, not args.no_composite, args.resume)
            elif strat == "joint":
                result = run_strategy_joint(
                    X, y, feature_cols, model_type,
                    args.n_trials, args.cv_folds, not args.no_composite, args.resume)
            elif strat == "hp_then_feature":
                result = run_strategy_hp_then_feature(
                    X, y, feature_cols, model_type,
                    args.n_trials, args.n_rounds, args.cv_folds,
                    not args.no_composite, args.resume)
            else:
                continue

            if result is None:
                continue
            strategy_results[strat] = result
        except Exception as e:
            # 2026-04-27: 견고화 — 전체 traceback + context 로깅
            import traceback
            _tb = traceback.format_exc()
            _print(f"  ✗ {strat} 실패: {type(e).__name__}: {e}")
            _print(f"    Last frame: {_tb.splitlines()[-2] if len(_tb.splitlines()) >= 2 else 'N/A'}")
            # 전체 traceback 은 log 에만 (console 은 간결)
            log.error(f"[{model_type}/{strat}] Exception {type(e).__name__}: {e}\n"
                      f"Traceback:\n{_tb}")

    if not strategy_results:
        _print(f"  ✗ {model_type}: 모든 전략 실패!")
        return None, None

    # 전략이 1개면 바로 저장, 여러 개면 비교 후 추천
    if len(strategy_results) == 1:
        strat_name = list(strategy_results.keys())[0]
        result = strategy_results[strat_name]
        result["recommended"] = True
        out_path = SAVE_DIR / f"optuna_feat_sel_{model_type}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(result), f, indent=2, ensure_ascii=False)
        return strat_name, result
    else:
        return compare_and_recommend(strategy_results, model_type)


def main():
    parser = argparse.ArgumentParser(description="Optuna Feature Selection — Multi-Strategy")
    # G-170 (2026-05-03): 콤마구분 list 지원 ("xgboost,lightgbm,randomforest").
    # 이전: choices=ALL_MODELS + ["all"] 단일만 → 카테고리별 학습 시 "all" 강제 → bottleneck.
    # 이제: 콤마구분 시 valid set 와 교집합만 처리 (invalid key 는 warn + skip).
    parser.add_argument("--model", default="all",
                        help=f"콤마구분 list 가능. 'all' 또는 {ALL_MODELS}")
    parser.add_argument("--scope", default="representative",
                        choices=["quick", "representative", "individual"],
                        help="quick: 대표 4개, representative: 대표 9개 (기본), "
                             "individual: 전체 15개 각각")
    parser.add_argument("--strategy", default="all",
                        choices=["mandatory_only", "feature_only", "joint", "hp_then_feature", "all"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=10,
                        help="hp_then_feature stage2 반복 횟수 (기본: 10)")
    # 2026-04-28: default = resume (학습 이어서)
    # `MPH_OPTUNA_FORCE=1` 또는 `--no-resume` 으로 강제 fresh start
    parser.add_argument("--resume", action="store_true", default=True,
                         help="기존 study 이어서 실행 (default ON, 2026-04-28~)")
    parser.add_argument("--no-resume", action="store_true",
                         help="강제 fresh start (기존 Optuna DB 삭제)")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--no-composite", action="store_true")
    parser.add_argument("--optuna-scope", default="individual",
                        choices=["category", "individual"],
                        help="category=카테고리별 proxy 공유 (DL→dnn, 빠름), "
                             "individual=모델별 독립 선택 (기본)")
    args = parser.parse_args()

    # optuna-scope 적용
    set_optuna_scope(args.optuna_scope)

    if args.model == "all":
        if args.scope == "quick":
            models = MODELS_QUICK
        elif args.scope == "individual":
            models = MODELS_INDIVIDUAL
        else:
            models = MODELS_REPRESENTATIVE
    else:
        # G-170 (2026-05-03): 콤마구분 list 지원 — 카테고리별 학습 bottleneck 차단.
        _requested = [m.strip().lower() for m in args.model.split(",") if m.strip()]
        _valid = set(ALL_MODELS)
        models = [m for m in _requested if m in _valid]
        _skipped = [m for m in _requested if m not in _valid]
        if _skipped:
            _print(f"  ⚠ --model 에 invalid key skip: {_skipped} "
                   f"(valid = {sorted(_valid)})")
        if not models:
            _print(f"  ✗ --model={args.model} 매칭 없음 → all 로 fallback")
            models = (MODELS_QUICK if args.scope == "quick" else
                      MODELS_INDIVIDUAL if args.scope == "individual" else
                      MODELS_REPRESENTATIVE)
        else:
            _print(f"  ✓ --model 필터 활성: {models}")

    # 2026-04-28: default 가 resume — 학습 끊어져도 이어서 진행
    # fresh start 조건:
    #   1) `--no-resume` flag
    #   2) `MPH_OPTUNA_FORCE=1` 환경변수
    _force_fresh = (args.no_resume or GLOBAL.optuna.force)
    if _force_fresh and Path(STUDY_DB).exists():
        try:
            Path(STUDY_DB).unlink()
            _print("  🗑 기존 Optuna DB 삭제 (fresh start — --no-resume 또는 MPH_OPTUNA_FORCE=1)")
        except Exception:
            pass
    elif Path(STUDY_DB).exists():
        # default: resume — 기존 trial 들 자동 활용
        try:
            sz = Path(STUDY_DB).stat().st_size / (1024 * 1024)
            _print(f"  🔁 기존 Optuna DB 재사용 (resume, {sz:.1f}MB) — fresh 원하면 --no-resume")
        except Exception:
            pass

    _print(f"\n  Scope: {args.scope} ({len(models)}개 모델)")
    _print(f"  전략: {args.strategy} | Trials: {args.n_trials} | "
           f"CV: {args.cv_folds}fold | Rounds: {args.n_rounds}")
    _print(f"  모델: {', '.join(models)}")
    _print("")

    all_recommendations = {}
    for i, model_type in enumerate(models):
        _print(f"  [{i+1}/{len(models)}] {model_type}")
        best_strat, best_result = run_all_strategies(model_type, args)
        if best_strat:
            all_recommendations[model_type] = {
                "best_strategy": best_strat,
                "best_rmse": best_result["best_rmse"],
                "n_features": best_result["n_features_selected"],
            }

    # ── 최종 요약 (간결하게) ──
    _print(f"\n{'='*55}")
    _print(f"  최종 요약")
    _print(f"{'='*55}")
    for model_type, rec in all_recommendations.items():
        _print(f"  ★ {model_type:20s} {rec['best_strategy']:18s} "
               f"RMSE={rec['best_rmse']:.4f}  피처 {rec['n_features']}개")

    # 추천 요약 저장
    summary_path = SAVE_DIR / "optuna_recommendation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(all_recommendations), f, indent=2, ensure_ascii=False)
    _print(f"\n  저장: {summary_path}")
    _print("\nFEATURE_SELECTION_DONE")


# ══════════════════════════════════════════════════════════
# 12. Inline API — run_full_diagnostics.py에서 import하여 사용
# ══════════════════════════════════════════════════════════

def run_inline_for_model(model_name: str, X_data=None, y_data=None,
                         feat_cols=None, n_trials: int = 30,
                         cv_folds: int = 3, strategy: str = "hp_then_feature",
                         force_refresh: bool = False):
    """학습 파이프라인 내부에서 호출하는 경량 Optuna 피처 선택.

    Args:
        model_name: 학습 파이프라인 모델명 (예: "LightGBM", "DNN-Optuna")
        X_data, y_data, feat_cols: 데이터 (None이면 load_data() 사용)
        n_trials: Optuna trial 수 (기본 30, 사전 실행보다 적게)
        cv_folds: Walk-Forward CV fold 수
        strategy: "hp_then_feature" | "feature_only" | "joint" | "all" | "mandatory_only"
        force_refresh: True면 기존 JSON 캐시 무시, 전부 새로 실행

    Returns:
        선택된 피처 이름 리스트 (실패 시 None)
    """
    # 모델명 → optuna key 변환
    optuna_key = TRAIN_MODEL_TO_OPTUNA_KEY.get(model_name)
    if not optuna_key:
        # 직접 매핑 안 되면 소문자로 시도
        optuna_key = model_name.lower().replace("-", "_")
        if optuna_key not in MODELS_INDIVIDUAL:
            return None

    # 2026-04-28: default resume — 최초 실행도 학습 끊어지면 이어서.
    # MPH_OPTUNA_FORCE=1 일 때만 fresh start.
    _inline_resume = (not GLOBAL.optuna.force and not force_refresh)

    # 데이터 로드
    if X_data is None or y_data is None or feat_cols is None:
        X, y, feat_cols = load_data()
    else:
        X, y = X_data, y_data

    # 이미 JSON 결과가 있는지 확인 (force_refresh면 건너뜀)
    out_path = SAVE_DIR / f"optuna_feat_sel_{optuna_key}.json"
    if not force_refresh and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("selected_features", [])
            if features:
                _print(f"  ✓ {model_name}: 캐시 사용 ({len(features)}개 피처)")
                return features
        except Exception:
            pass
    elif force_refresh and out_path.exists():
        _print(f"  🔄 {model_name}: 캐시 무시 (force_refresh), 새로 실행")

    _print(f"  ▶ {model_name} ({optuna_key}): inline Optuna 피처 선택 ({strategy}, {n_trials} trials)")

    try:
        # ── strategy="all": 4가지 전략 모두 실행 → baseline 대비 비교 → 최적 추천 ──
        if strategy == "all":
            strategy_results = {}
            strategies_order = ["mandatory_only", "feature_only", "joint", "hp_then_feature"]
            for strat in strategies_order:
                try:
                    if strat == "mandatory_only":
                        r = run_strategy_mandatory_only(X, y, feat_cols, optuna_key, cv_folds)
                    elif strat == "feature_only":
                        r = run_strategy_feature_only(
                            X, y, feat_cols, optuna_key,
                            n_trials=n_trials, cv_folds=cv_folds,
                            use_composite=False, resume=_inline_resume)
                    elif strat == "joint":
                        r = run_strategy_joint(
                            X, y, feat_cols, optuna_key,
                            n_trials=n_trials, cv_folds=cv_folds,
                            use_composite=False, resume=_inline_resume)
                    elif strat == "hp_then_feature":
                        r = run_strategy_hp_then_feature(
                            X, y, feat_cols, optuna_key,
                            n_trials=n_trials, n_rounds=3,
                            cv_folds=cv_folds, use_composite=False, resume=_inline_resume)
                    else:
                        continue
                    if r is not None:
                        strategy_results[strat] = r
                except Exception as e:
                    _print(f"  ✗ {model_name}/{strat} 실패: {e}")

            if not strategy_results:
                _print(f"  ✗ {model_name}: 모든 전략 실패")
                return None

            # compare_and_recommend로 baseline 대비 비교 + 최적 추천 + JSON 저장
            best_strategy, best_result = compare_and_recommend(strategy_results, optuna_key)
            if best_result is None:
                return None

            features = best_result.get("selected_features", [])
            _print(f"  ✓ {model_name}: ★{best_strategy} → {len(features)}개 피처 "
                   f"(RMSE={best_result['best_rmse']:.4f})")
            return features

        # ── 단일 전략 실행 ──
        if strategy == "hp_then_feature":
            result = run_strategy_hp_then_feature(
                X, y, feat_cols, optuna_key,
                n_trials=n_trials, n_rounds=3,  # 경량: 3회
                cv_folds=cv_folds, use_composite=False, resume=_inline_resume)
        elif strategy == "joint":
            result = run_strategy_joint(
                X, y, feat_cols, optuna_key,
                n_trials=n_trials, cv_folds=cv_folds,
                use_composite=False, resume=_inline_resume)
        elif strategy == "mandatory_only":
            result = run_strategy_mandatory_only(X, y, feat_cols, optuna_key, cv_folds)
        else:  # feature_only
            result = run_strategy_feature_only(
                X, y, feat_cols, optuna_key,
                n_trials=n_trials, cv_folds=cv_folds,
                use_composite=False, resume=_inline_resume)

        if result is None:
            _print(f"  ✗ {model_name}: inline Optuna 실패")
            return None

        # ── baseline 비교 기록 (단일 전략에서도 mandatory_only 대비 개선 측정) ──
        if strategy != "mandatory_only":
            try:
                baseline = run_strategy_mandatory_only(X, y, feat_cols, optuna_key, cv_folds)
                if baseline:
                    bl_rmse = baseline["best_rmse"]
                    opt_rmse = result["best_rmse"]
                    delta = bl_rmse - opt_rmse
                    pct = delta / bl_rmse * 100 if bl_rmse > 0 else 0
                    result["baseline_rmse"] = round(bl_rmse, 4)
                    result["baseline_delta"] = round(delta, 4)
                    result["baseline_improvement_pct"] = round(pct, 1)
                    sign = "↓" if delta > 0 else "↑"
                    _print(f"  📊 baseline 대비: RMSE {sign}{abs(delta):.4f} ({abs(pct):.1f}%)")

                    # baseline보다 나빠졌으면 mandatory_only 결과 사용
                    if opt_rmse >= bl_rmse:
                        _print(f"  ⚠ Optuna({strategy})가 baseline 이하 → mandatory_only 피처 사용")
                        result = baseline
            except Exception as e:
                _print(f"  ⚠ baseline 비교 실패 (무시): {e}")

        # JSON 저장 (다음 실행 시 캐시)
        result["recommended"] = True
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(result), f, indent=2, ensure_ascii=False)

        features = result.get("selected_features", [])
        _print(f"  ✓ {model_name}: {len(features)}개 피처 선택 (RMSE={result['best_rmse']:.4f})")
        return features

    except Exception as e:
        _print(f"  ✗ {model_name}: inline Optuna 에러 — {e}")
        return None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        err_msg = traceback.format_exc()
        _print(f"\n  ✗ FATAL ERROR: {exc}")
        err_path = SAVE_DIR / "optuna_error.log"
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(err_msg)
        _print(f"  에러 로그: {err_path}")
        sys.exit(1)
