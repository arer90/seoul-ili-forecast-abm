"""P3 — Overseas Cross-Country ILI Validation.

P3-1: 각국 ILI 시계열 예측 (US / JP / DE / FR / KR-baseline)
P3-3: Cross-country R²/RMSE/MAPE/WIS 비교 표 + HTML 리포트

Feature set (7개, 전국 공통):
    ili_lag1~4, sin_week, cos_week, year_trend

Design (D-4 deep module):
    Single entry-point run() — feature engineering / model loop /
    metric computation / HTML report 모두 캡슐화.

Public API:
    run(countries, model_names, out_dir, quick) → pd.DataFrame

Performance:
    16 models × 5 countries = 80 runs.
    각 run: 10 Optuna trials × small dataset → ~1-5분/run → 총 1-2h.
    quick=True → 5 trials, 빠른 검증.

Side effects:
    - DB read (overseas_ili) → read-only
    - writes to out_dir (CSV, HTML)

Caller responsibility:
    - DB path 유효 (epi_real_seoul.db)
    - out_dir 생성 권한 있어야 함
    - OPTUNA_ISOLATE=1 환경변수 권장 (subprocess 격리)

See: G-116 (safe_connect), G-159 (sanitize_predictions), G-166 (_validate_shapes).
"""
from __future__ import annotations
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

import json
import logging
import os
import warnings
from pathlib import Path
from sqlite3 import Connection as _Conn
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── DB 경로 (프로젝트 기준) ──────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "epi_real_seoul.db"

# ── 국가별 ILI 소스 ──────────────────────────────────────────────────────────
# P3 cross-country 비교: KR (main pipeline KDCA sentinel) 동일 기간 적용.
# 사용자 명시 (2026-05-26): "P3 에서는 KR 의 동일 기간으로 해줘"
# KDCA sentinel_influenza season_start ∈ [2019, 2025] (n=2450 rows, 350 weekly × 7 ages).
# 모든 비교 국가가 동일 기간 사용 → fair cross-country 비교.
#
# Sprint E4 caveat (Gemini P3 review): DE Bundesländer (rki_bundesland) sub-national 데이터는
# 자체 2020 부터 시작 (다른 region: 2019). DE regional 분석 시 한 시즌 짧음.
# Strict equal period 원하면 env MPH_PHASE18_YEAR_MIN_OVERRIDE=2020 사용.
#
# ⚠️ POSITIVITY% LEGACY NOTE (2026-05-27, §8 corrigendum):
# 본 COUNTRY_SOURCES 는 WHO FluNet positivity% 기반 LEGACY cohort (KR/DE/FR/HK).
# Positivity% = lab confirmation rate, NOT ILI consultation rate.
# TRUE ILI 분석 (KR↔BE/NO/JP, eLife 107767 표준) 은 별도 모듈 사용:
#   from simulation.pipeline.true_ili_cohort import (
#       get_cohort_ia, get_cohort_ib, load_country_ili,
#   )
# 두 cohort 모두 paper 에 보고:
#   - LEGACY (이 파일): §7 결과 — positivity dynamics (surveillance system 동질화 효과)
#   - TRUE ILI (phase18_true_ili_cohort.py): §8 결과 — ILI consultation dynamics
_KR_PERIOD_YEAR_MIN = GLOBAL.data_split.overseas_year_min
_KR_PERIOD_YEAR_MAX = GLOBAL.data_split.overseas_year_max

COUNTRY_SOURCES: dict[str, dict] = {
    "US": {"source": "delphi_national", "unit": "wILI%",
           "year_min": _KR_PERIOD_YEAR_MIN, "year_max": _KR_PERIOD_YEAR_MAX},
    "JP": {"source": "japan_jihs",      "unit": "per_clinic",
           "year_min": _KR_PERIOD_YEAR_MIN, "year_max": _KR_PERIOD_YEAR_MAX},
    "DE": {"source": "who_flunet",      "unit": "positivity%",
           "year_min": _KR_PERIOD_YEAR_MIN, "year_max": _KR_PERIOD_YEAR_MAX},
    "FR": {"source": "who_flunet",      "unit": "positivity%",
           "year_min": _KR_PERIOD_YEAR_MIN, "year_max": _KR_PERIOD_YEAR_MAX},
    "HK": {"source": "who_flunet",      "unit": "positivity%",
           "year_min": _KR_PERIOD_YEAR_MIN, "year_max": _KR_PERIOD_YEAR_MAX},
    "KR": {"source": "who_flunet",      "unit": "positivity%",
           "year_min": _KR_PERIOD_YEAR_MIN, "year_max": _KR_PERIOD_YEAR_MAX},
}

# ── Sub-national regional sources (Sprint γ Item 7 full impl, 2026-05-26) ───
# 사용자 명시: "P3에서는 지역구별 ili 의 내용이 있으면 좋겠어. 있는 것들만이라도"
# 가용 region: USA 10 (HHS), DEU 17 (Bundesland), FRA 22 (regional), JPN 47 (prefecture), HKG 1.
# KR 은 sub-national ILI 데이터 X → national fallback.
# DB col=country uses ISO-3 (USA/JPN/DEU/FRA/HKG) — short code (US/JP/DE/FR/HK) 와 매핑 필요.
REGIONAL_SOURCES: dict[str, Optional[dict]] = {
    "US": {"country_iso3": "USA", "source": "delphi_hhs",
           "unit": "wILI%", "n_regions_expected": 10},
    "DE": {"country_iso3": "DEU", "source": "rki_bundesland",
           "unit": "positivity%", "n_regions_expected": 17},
    "FR": {"country_iso3": "FRA", "source": "sentiweb_fr_regional",
           "unit": "incidence_rate", "n_regions_expected": 22},
    "JP": {"country_iso3": "JPN", "source": "jihs_prefecture",
           "unit": "per_clinic", "n_regions_expected": 47},
    "HK": {"country_iso3": "HKG", "source": "hk_chp",
           "unit": "positivity%", "n_regions_expected": 1},
    # KR: 사용자 명시 (2026-05-26) "KR 에서는 national 로만으로도 해줘".
    # sub-national ILI 데이터 (sentinel_influenza_by_gu) 가 DB 에 없으므로
    # regional 모드에서 자동으로 national fallback 수행 — caller logic 참조.
    "KR": None,
}

# ── feature 컬럼 (모든 국가 동일) ────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "ili_lag1", "ili_lag2", "ili_lag3", "ili_lag4",
    "sin_week", "cos_week",
    "year_trend",
]

# ── 평가 대상 모델 (GCN/GAT/SEIR-V2-Forced/SARIMAX/TFT 제외, 앙상블은 별도) ──
# CATEGORY_MODELS 기준 50개 portable 중 비앙상블 40개 전체 포함
# Ensemble은 기반 모델 예측이 완료된 후 별도 실행 (현재 제외)
DEFAULT_MODELS: list[str] = [
    # Tree (5)
    "XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "CatBoost",
    # Linear (5)
    "ElasticNet", "BayesianRidge", "NegBinGLM", "NegBinGLM-V7", "PoissonAutoreg",
    # Kernel (3)
    "KRR", "SVR-Linear", "SVR-RBF",
    # Other / Statistical (3)
    "GAM-Spline", "GP-RBF-Periodic", "BayesianMCMC",
    # TS (2)
    "ARIMA", "SARIMA",
    # DL Tabular (3)
    "DNN", "DNN-Optuna", "TabularDNN-Lite",
    # DL Seq (2)
    "TCN", "TCN-Optuna",
    # Modern-TS (9) — neuralforecast / pytorch-forecasting 기반
    "N-BEATS", "N-HiTS", "TiDE", "DeepAR", "RNN",
    "PatchTST", "iTransformer", "TimesNet", "Mamba",
    # Mech (3)
    "PINN-Lite", "MP-PINN", "Rt-Augmented",
    # Foundation (2) — TimesFM-2.5 (Chronos 대체, G-261: transformers-free → 메인 env 호환)
    "TimesFM-2.5", "OverseasTransfer",
]

# Foundation zero-shot path (feature matrix 미사용, y_train 시계열만) — G-261: Chronos→TimesFM-2.5.
# chronos 는 transformers<5 강제로 메인 env(mlx-lm/ARIA = transformers>=5) 와 충돌 → 제거.
_FOUNDATION_ZEROSHOT: set[str] = {"TimesFM-2.5"}

# 최소 학습 데이터 (주) — 국가별 체크
_MIN_TRAIN_WEEKS = 52  # 1년 이상 학습 데이터 필요

# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_ili_series(con: _Conn, country: str) -> pd.DataFrame:
    """overseas_ili에서 국가별 주간 ILI 시계열 로드.

    Args:
        con: SQLite connection (read-only 권장).
        country: 2자리 국가코드 (US/JP/DE/FR/KR).

    Returns:
        DataFrame[year, week_no, ili_rate] — 결측 없는 행만, time-sorted.

    Raises:
        ValueError: country 없거나 non-null rows < 10.
    """
    cfg = COUNTRY_SOURCES[country]
    source = cfg["source"]
    year_min = cfg["year_min"]
    year_max = cfg.get("year_max", 9999)   # 모든 나라가 KR 과 동일 기간 (2019-2025)

    rows = con.execute(
        """
        SELECT year, week_no, ili_rate
        FROM   overseas_ili
        WHERE  country = ? AND source = ?
               AND ili_rate IS NOT NULL
               AND year >= ?
               AND year <= ?
        ORDER  BY year, week_no
        """,
        (country, source, year_min, year_max),
    ).fetchall()

    if len(rows) < 10:
        raise ValueError(
            f"[phase18] {country}/{source}: non-null rows={len(rows)} < 10. 데이터 부족."
        )

    df = pd.DataFrame(rows, columns=["year", "week_no", "ili_rate"])
    df = df.drop_duplicates(["year", "week_no"]).sort_values(["year", "week_no"]).reset_index(drop=True)
    log.info("[phase18] %s/%s: %d rows (%.0f~%.0f)", country, source, len(df),
             df["year"].min(), df["year"].max())
    return df


def _get_regions_for_country(con: _Conn, country_short: str) -> list[str]:
    """Sub-national region list for one country (R8.4 — gu-level scaffolding 실제 구현).

    Args:
        con: SQLite connection.
        country_short: 2-letter short code (US/JP/DE/FR/HK/KR).

    Returns:
        list of region names. Empty list if REGIONAL_SOURCES[country_short] is None
        (KR or unsupported — caller falls back to national).

    Performance: O(1) — small distinct query (~10-50 regions per country).
    """
    cfg = REGIONAL_SOURCES.get(country_short)
    if cfg is None:
        return []
    rows = con.execute(
        """
        SELECT DISTINCT region
        FROM   overseas_ili_regional
        WHERE  country = ? AND source = ?
               AND ili_rate IS NOT NULL
               AND year >= ? AND year <= ?
        ORDER  BY region
        """,
        (cfg["country_iso3"], cfg["source"],
         _KR_PERIOD_YEAR_MIN, _KR_PERIOD_YEAR_MAX),
    ).fetchall()
    return [r[0] for r in rows]


def _fetch_ili_series_regional(
    con: _Conn, country_short: str, region: str
) -> pd.DataFrame:
    """Sub-national ILI series for (country, region) — KR 동일 기간 (2019-2025).

    Args:
        con: SQLite connection.
        country_short: 2-letter short code (US/JP/DE/FR/HK).
        region: region name (e.g. 'HHS1', 'Bayern', 'Aichi').

    Returns:
        DataFrame[year, week_no, ili_rate] — 결측 없는 행만, time-sorted.

    Raises:
        ValueError: country_short 가 REGIONAL_SOURCES 에 없거나 non-null rows < 10.

    Caller responsibility: 호출 전 _get_regions_for_country() 로 가용 region 확인.
    """
    cfg = REGIONAL_SOURCES.get(country_short)
    if cfg is None:
        raise ValueError(
            f"[phase18] {country_short} 는 sub-national 데이터 없음 — "
            "caller 가 national fallback 처리해야 함"
        )
    rows = con.execute(
        """
        SELECT year, week_no, ili_rate
        FROM   overseas_ili_regional
        WHERE  country = ? AND source = ? AND region = ?
               AND ili_rate IS NOT NULL
               AND year >= ? AND year <= ?
        ORDER  BY year, week_no
        """,
        (cfg["country_iso3"], cfg["source"], region,
         _KR_PERIOD_YEAR_MIN, _KR_PERIOD_YEAR_MAX),
    ).fetchall()
    if len(rows) < 10:
        raise ValueError(
            f"[phase18] {country_short}/{region}: non-null rows={len(rows)} < 10"
        )
    df = pd.DataFrame(rows, columns=["year", "week_no", "ili_rate"])
    df = (df.drop_duplicates(["year", "week_no"])
            .sort_values(["year", "week_no"])
            .reset_index(drop=True))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """ILI lag1~4 + 계절성 + 추세 feature engineering.

    Args:
        df: DataFrame[year, week_no, ili_rate] — 시간순 정렬 가정.

    Returns:
        DataFrame[year, week_no, ili_rate, ili_lag1~4, sin_week, cos_week, year_trend]
        결측 행(첫 4행) dropna 후 반환.

    Performance: O(n), ~0MB overhead (in-place shift).
    Side effects: None.
    Caller responsibility: df는 중복 없이 시간순 정렬 필요.
    """
    df = df.copy()

    # autoregressive lags (1~4주)
    for lag in range(1, 5):
        df[f"ili_lag{lag}"] = df["ili_rate"].shift(lag)

    # 계절성 (cyclical encoding)
    df["sin_week"] = np.sin(2 * np.pi * df["week_no"] / 52.18)
    df["cos_week"] = np.cos(2 * np.pi * df["week_no"] / 52.18)

    # 선형 추세
    year_min = df["year"].min()
    df["year_trend"] = (df["year"] - year_min) + (df["week_no"] - 1) / 52.0

    df = df.dropna(subset=FEATURE_COLS + ["ili_rate"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────

def _split_train_test(df: pd.DataFrame, test_weeks: int = 52) -> tuple[pd.DataFrame, pd.DataFrame]:
    """최근 test_weeks 주를 test set으로 분리.

    Args:
        df: feature-engineered DataFrame.
        test_weeks: test set 크기 (기본 52주).

    Returns:
        (df_train, df_test)

    Raises:
        ValueError: train rows < _MIN_TRAIN_WEEKS 시.
    """
    n = len(df)
    if n <= test_weeks + _MIN_TRAIN_WEEKS:
        # JP처럼 데이터 짧은 경우: 최소 50% train 보장
        test_weeks = max(12, n // 4)
        log.warning("[phase18] 데이터 짧음(%d rows) → test_weeks=%d 로 축소", n, test_weeks)

    df_test  = df.iloc[-test_weeks:].copy()
    df_train = df.iloc[:-test_weeks].copy()

    if len(df_train) < _MIN_TRAIN_WEEKS:
        raise ValueError(
            f"train rows={len(df_train)} < {_MIN_TRAIN_WEEKS}. 국가 데이터 부족."
        )

    log.info("[phase18] split: train=%d, test=%d", len(df_train), len(df_test))
    return df_train, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 4. 단일 모델 실행
# ─────────────────────────────────────────────────────────────────────────────

def _set_trial_budget(model_name: str, n_trials: int) -> None:
    """MPH_OPTUNA_TRIALS_JSON에 model_name: n_trials 추가.

    Caller responsibility: run() 시작 시 호출 (전체 모델 한번에).
    """
    raw = os.environ.get("MPH_OPTUNA_TRIALS_JSON", "{}")
    try:
        budget = json.loads(raw)
    except Exception:
        budget = {}
    budget[model_name] = n_trials
    os.environ["MPH_OPTUNA_TRIALS_JSON"] = json.dumps(budget)


def _run_one_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    country: str = "unknown",
    region: str = "national",
) -> dict:
    """단일 모델 학습 + 예측 + 메트릭 계산.

    Args:
        model_name: CATEGORY_MODELS에 등록된 이름.
        X_train: (n_train, 7) feature matrix.
        y_train: (n_train,) ILI rate.
        X_test: (n_test, 7) feature matrix.
        y_test: (n_test,) ILI rate (ground truth).

    Returns:
        dict with keys: model, y_pred (list), metrics (dict[str,float]), error (str|None).

    Performance: 모델별 상이 — tree 1-30s, DL 30-120s, foundation(TimesFM) 별도.
    Side effects: None (모델 파일 저장 안 함).
    """
    from simulation.models.base import sanitize_predictions
    from simulation.pipeline.metric_eval import compute_full_metrics

    result: dict = {
        "model": model_name,
        "y_pred": None,
        "metrics": {},
        "error": None,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    try:
        # ── 모델 클래스 로드 ──
        from simulation.models.registry import CATEGORY_MODELS
        model_cls = None
        for cat_models in CATEGORY_MODELS.values():
            if model_name in cat_models:
                # 동적 import: 카테고리별 모듈
                model_cls = _import_model_class(model_name)
                break

        if model_cls is None:
            raise ImportError(f"모델 클래스 없음: {model_name}")

        model = model_cls()

        # min_data 체크
        min_data = getattr(getattr(model, "meta", None), "min_data", 40)
        if len(y_train) < min_data:
            raise ValueError(
                f"train_rows={len(y_train)} < min_data={min_data} for {model_name}"
            )

        # ── 학습 + 예측 ──
        model.fit(X_train, y_train)
        y_pred_raw = model.predict(X_test)
        y_pred = sanitize_predictions(y_pred_raw)

        # sigma (잔차 기반 PI proxy)
        y_train_pred = model.predict(X_train)
        residuals = y_train - sanitize_predictions(y_train_pred)
        sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0
        sigma = max(sigma, 1e-6)

        # ── 메트릭 계산 ──
        # audit Stage 1.1 (cascade #2, 2026-05-27): country-level viral_positivity
        # 미산출 (단순 KDCA fallback q70 자동). 향후: WHO FluNet country-level
        # positivity loader → metric_eval 의 KDCA primary path 활성화.
        metrics = compute_full_metrics(
            y_test=y_test,
            y_pred=y_pred,
            sigma_for_wis=sigma,
            y_train_pool=y_train,
            viral_positivity_train=None,  # country별 산출은 후속 sub-task
        )

        result["y_pred"] = y_pred.tolist()
        result["metrics"] = metrics

        # R8.2 (2026-05-26): full 134-key SSOT eval on overseas (P3) predictions.
        # Trajectory: P3 overseas country-specific predictions → cross-country comparison.
        # Provides paper_top{2,3,5,10}_complete on per-country granularity (g175_*_pass 제거 2026-06-05).
        try:
            from simulation.pipeline.phase_evaluator import evaluate_predictions_full
            _y_arr = np.asarray(y_test, dtype=np.float64)
            _p_arr = np.asarray(y_pred, dtype=np.float64)
            _mask = np.isfinite(_y_arr) & np.isfinite(_p_arr)
            if _mask.sum() >= 5:
                full_r8 = evaluate_predictions_full(
                    y_test=_y_arr[_mask],
                    y_pred=_p_arr[_mask],
                    residuals=(_y_arr[_mask] - _p_arr[_mask]),
                    sigma=sigma,
                    y_train_pool=np.asarray(y_train, dtype=np.float64),
                    threshold=GLOBAL.filter.alert_threshold,
                    phase_id=f"Pov_{country}_{region}_{model_name}",
                    enable_bootstrap_ci=False,
                )
                result["phase_eval_r8"] = full_r8
        except Exception as _e:
            result["phase_eval_r8_err"] = str(_e)

        log.info("[phase18] %s: R²=%.3f MAPE=%.1f%% WIS=%.4f",
                 model_name, metrics.get("r2", float("nan")),
                 metrics.get("mape", float("nan")),
                 metrics.get("wis", float("nan")))

    except Exception as exc:
        log.warning("[phase18] %s FAILED: %s", model_name, exc)
        result["error"] = str(exc)

    return result


def _import_model_class(model_name: str):
    """모델 이름 → 클래스 동적 import.

    Args:
        model_name: 등록 이름 (예: 'XGBoost', 'ARIMA').

    Returns:
        model class (BaseForecaster 서브클래스).

    Raises:
        ImportError: 모델을 찾을 수 없는 경우.

    Performance: O(1) — 모듈은 Python 캐시됨.
    """
    _MAP: dict[str, tuple[str, str]] = {
        # ── Tree ──────────────────────────────────────────────────────────────
        "XGBoost":          ("simulation.models.tree_models",    "XGBoostForecaster"),
        "LightGBM":         ("simulation.models.tree_models",    "LightGBMForecaster"),
        "RandomForest":     ("simulation.models.tree_models",    "RandomForestForecaster"),
        # GradientBoosting 제거 2026-05-28 (R8 re-audit): GradientBoostingForecaster
        # 가 tree_models 에서 삭제됨 (Sprint D1 MERGE-drop) → dead reference 였음.
        "CatBoost":         ("simulation.models.tree_models",    "CatBoostForecaster"),
        # ── Linear ────────────────────────────────────────────────────────────
        "ElasticNet":       ("simulation.models.linear_models",  "ElasticNetForecaster"),
        "BayesianRidge":    ("simulation.models.epi_models",     "BayesianRidgeForecaster"),
        "NegBinGLM":        ("simulation.models.negbin_glm",     "NegBinGLMForecaster"),
        "NegBinGLM-V7":     ("simulation.models.negbin_glm",     "NegBinGLMForecaster"),   # alias
        "PoissonAutoreg":   ("simulation.models.epi_models",     "PoissonAutoregForecaster"),
        # ── Kernel ────────────────────────────────────────────────────────────
        "SVR-Linear":       ("simulation.models.linear_models",  "SVRLinearForecaster"),
        "SVR-RBF":          ("simulation.models.linear_models",  "SVRRBFForecaster"),
        "KRR":              ("simulation.models.linear_models",  "KRRForecaster"),
        # ── Other / Statistical ───────────────────────────────────────────────
        "GAM-Spline":       ("simulation.models.epi_models",     "GAMForecaster"),
        "GP-RBF-Periodic":  ("simulation.models.epi_models",     "GaussianProcessForecaster"),
        "BayesianMCMC":     ("simulation.models.epi_models",     "BayesianMCMCForecaster"),
        # ── TS ────────────────────────────────────────────────────────────────
        "ARIMA":            ("simulation.models.ts_models",      "ARIMAForecaster"),
        "SARIMA":           ("simulation.models.ts_models",      "SARIMAForecaster"),
        # ── DL Tabular ────────────────────────────────────────────────────────
        "DNN":              ("simulation.models.dl_models",      "DNNForecaster"),
        "DNN-Optuna":       ("simulation.models.dl_models",      "OptunaDNNForecaster"),
        "TabularDNN-Lite":  ("simulation.models.dl_models",      "TabularDNNLiteForecaster"),
        # ── DL Seq ────────────────────────────────────────────────────────────
        "TCN":              ("simulation.models.dl_models",      "TCNForecaster"),
        "TCN-Optuna":       ("simulation.models.dl_models",      "OptunaTCNForecaster"),
        # ── Modern-TS (simulation.models.modern_ts) ───────────────────────────
        "N-BEATS":          ("simulation.models.modern_ts",      "NBEATSForecaster"),
        "N-HiTS":           ("simulation.models.modern_ts",      "NHiTSForecaster"),
        "TiDE":             ("simulation.models.modern_ts",      "TiDEForecaster"),
        "DeepAR":           ("simulation.models.modern_ts",      "PfDeepARForecaster"),
        "RNN":              ("simulation.models.modern_ts",      "PfRNNForecaster"),
        "PatchTST":         ("simulation.models.modern_ts",      "PatchTSTForecaster"),
        "iTransformer":     ("simulation.models.modern_ts",      "iTransformerForecaster"),
        "TimesNet":         ("simulation.models.modern_ts",      "TimesNetForecaster"),
        "Mamba":            ("simulation.models.modern_ts",      "MambaForecaster"),
        # ── Mech ──────────────────────────────────────────────────────────────
        "PINN-Lite":        ("simulation.models.pinn_model",     "SimplifiedPINNForecaster"),
        "MP-PINN":          ("simulation.models.pinn_model",     "PINNForecaster"),
        "Rt-Augmented":     ("simulation.models.rt_estimator",   "RtForecaster"),
        # ── Foundation ────────────────────────────────────────────────────────
        # TimesFM-2.5 는 _FOUNDATION_ZEROSHOT → _run_foundation_zeroshot() 로 처리 (factory 불필요, G-261).
        "OverseasTransfer": ("simulation.models.overseas_transfer","OverseasTransferForecaster"),
    }

    if model_name not in _MAP:
        raise ImportError(f"_import_model_class: '{model_name}' 미등록 모델")

    module_path, class_name = _MAP[model_name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f"{module_path}.{class_name} 없음")
    return cls


# ─────────────────────────────────────────────────────────────────────────────
# 5. Foundation (TimesFM-2.5) zero-shot (feature-free, 별도 처리) — G-261: Chronos 대체
# ─────────────────────────────────────────────────────────────────────────────

def _run_foundation_zeroshot(
    model_name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_steps: int,
    *,
    country: str = "unknown",
    region: str = "national",
) -> dict:
    """Foundation zero-shot forecasting (feature matrix 불필요) — G-261: Chronos→TimesFM-2.5.

    model_name: TimesFM-2.5 (transformers-free, 메인 env 네이티브). chronos 는 transformers<5
    강제로 메인 env(mlx-lm/ARIA) 와 충돌하여 제거됨.

    Args:
        y_train: 학습 ILI 시계열 (n_train,).
        y_test: 실제 ILI (n_test,) — 메트릭 계산용.
        n_steps: 예측 horizon (= len(y_test)).

    Returns:
        dict: y_pred (list), metrics (dict), error (str|None).

    Performance: ~10-30s (모델 로드 1회, zero-shot — fit 없음).
    Side effects: TimesFM checkpoint 다운로드 (최초 1회, ~/.cache/huggingface/).
    """
    from simulation.models.base import sanitize_predictions
    from simulation.pipeline.metric_eval import compute_full_metrics

    result: dict = {"model": model_name, "y_pred": None, "metrics": {}, "error": None,
                    "n_train": len(y_train), "n_test": len(y_test)}
    try:
        from simulation.models.timesfm_wrapper import TimesFMForecaster

        # zero-shot: fit_series(y_train) → forecast(n_steps) 직접 다단계 (Chronos 와 동일 convention)
        f = TimesFMForecaster()
        f.fit_series(np.asarray(y_train, dtype=np.float32))
        y_pred = sanitize_predictions(np.asarray(f.forecast(n_steps), dtype=np.float64))

        sigma = float(np.std(y_train)) if len(y_train) > 1 else 1.0
        # audit Stage 1.1 (cascade #2, 2026-05-27): KDCA fallback q70 (foundation zero-shot path)
        metrics = compute_full_metrics(
            y_test=y_test, y_pred=y_pred,
            sigma_for_wis=max(sigma, 1e-6), y_train_pool=y_train,
            viral_positivity_train=None,  # country별 산출은 후속 sub-task
        )
        result["y_pred"] = y_pred.tolist()
        result["metrics"] = metrics

        # R8.2 (2026-05-26): full 134-key SSOT eval on foundation zero-shot predictions.
        try:
            from simulation.pipeline.phase_evaluator import evaluate_predictions_full
            _y_arr = np.asarray(y_test, dtype=np.float64)
            _p_arr = np.asarray(y_pred, dtype=np.float64)
            _mask = np.isfinite(_y_arr) & np.isfinite(_p_arr)
            if _mask.sum() >= 5:
                full_r8 = evaluate_predictions_full(
                    y_test=_y_arr[_mask],
                    y_pred=_p_arr[_mask],
                    residuals=(_y_arr[_mask] - _p_arr[_mask]),
                    sigma=max(sigma, 1e-6),
                    y_train_pool=np.asarray(y_train, dtype=np.float64),
                    threshold=GLOBAL.filter.alert_threshold,
                    phase_id=f"Pov_{country}_{region}_foundation_{model_name}",
                    enable_bootstrap_ci=False,
                )
                result["phase_eval_r8"] = full_r8
        except Exception as _e:
            result["phase_eval_r8_err"] = str(_e)

        log.info("[phase18] %s zero-shot: R²=%.3f", model_name, metrics.get("r2", float("nan")))

    except Exception as exc:
        log.warning("[phase18] foundation(%s) FAILED: %s", model_name, exc)
        result["error"] = str(exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. HTML 리포트 생성 (C-3)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_html_report(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    country_info: dict,
) -> Path:
    """Cross-country 비교 HTML 리포트 생성.

    Args:
        metrics_df: DataFrame[country, model, r2, rmse, mape, wis, pi95_coverage, n_train, n_test, error].
        out_dir: 출력 디렉토리.
        country_info: 국가별 설명 dict (source, unit, n_rows).

    Returns:
        생성된 HTML 파일 경로.

    Performance: ~100ms (pandas to_html + f-string).
    Side effects: HTML 파일 1개 write.
    """
    html_path = out_dir / "overseas_validation_report.html"

    # 국가별 컬러
    country_colors = {"US": "#1f77b4", "JP": "#d62728", "DE": "#2ca02c",
                      "FR": "#9467bd", "KR": "#8c564b"}

    # ── Pivot: model × country → R² ──
    pivot_r2 = metrics_df.pivot_table(
        index="model", columns="country", values="r2", aggfunc="first"
    ).round(3)

    pivot_mape = metrics_df.pivot_table(
        index="model", columns="country", values="mape", aggfunc="first"
    ).round(1)

    pivot_wis = metrics_df.pivot_table(
        index="model", columns="country", values="wis", aggfunc="first"
    ).round(4)

    pivot_pi95 = metrics_df.pivot_table(
        index="model", columns="country", values="pi95_coverage", aggfunc="first"
    ).round(3)

    def _color_r2(val):
        if pd.isna(val):
            return "background-color:#f0f0f0; color:#999"
        if val >= 0.8:
            return "background-color:#2ca02c; color:white"
        if val >= 0.6:
            return "background-color:#98df8a; color:#333"
        if val >= 0.3:
            return "background-color:#ffdd57; color:#333"
        if val >= 0.0:
            return "background-color:#ff9999; color:#333"
        return "background-color:#cc3333; color:white"

    # ── summary stats ──
    success_rate = (metrics_df["error"].isna()).mean() * 100
    mean_r2_by_country = metrics_df.groupby("country")["r2"].mean().round(3)
    best_model = metrics_df.groupby("model")["r2"].mean().idxmax() if len(metrics_df) > 0 else "N/A"

    # ── country info table ──
    info_rows = ""
    for c, info in country_info.items():
        color = country_colors.get(c, "#333")
        info_rows += (
            f"<tr><td style='color:{color};font-weight:bold'>{c}</td>"
            f"<td>{info.get('source','')}</td><td>{info.get('unit','')}</td>"
            f"<td>{info.get('n_rows','-')}</td>"
            f"<td>{info.get('year_range','')}</td>"
            f"<td style='color:green'>{info.get('n_train','-')}</td>"
            f"<td style='color:#c00'>{info.get('n_test','-')}</td></tr>\n"
        )

    # R² 테이블 HTML
    def _pivot_to_html(pivot, title, fmt=None):
        rows_html = ""
        for model in pivot.index:
            rows_html += f"<tr><td><b>{model}</b></td>"
            for country in pivot.columns:
                val = pivot.loc[model, country]
                style = _color_r2(val) if "R²" in title else ""
                cell_val = f"{val:.3f}" if not pd.isna(val) else "—"
                if fmt == "pct":
                    cell_val = f"{val:.1f}%" if not pd.isna(val) else "—"
                rows_html += f"<td style='{style}'>{cell_val}</td>"
            rows_html += "</tr>\n"

        cols_html = "".join(
            "<th style='color:{}'>{}</th>".format(country_colors.get(c, "#333"), c)
            for c in pivot.columns
        )
        return f"""
        <h3>{title}</h3>
        <table class='metric-table'>
          <thead><tr><th>Model</th>{cols_html}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Overseas ILI Validation — C-1/C-3</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 32px; background: #f8f9fa; color: #222; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 8px; }}
  h2 {{ color: #16213e; margin-top: 32px; }}
  h3 {{ color: #0f3460; margin-top: 24px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .stat {{ display: inline-block; margin: 8px 16px; text-align: center; }}
  .stat-val {{ font-size: 2em; font-weight: bold; color: #4361ee; }}
  .stat-lbl {{ font-size: .85em; color: #666; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: center; }}
  thead {{ background: #1a1a2e; color: white; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .metric-table td:first-child {{ text-align: left; font-family: monospace; }}
  .info-table th {{ background: #4361ee; }}
  .badge-good {{ background: #2ca02c; color: white; border-radius: 4px; padding: 2px 8px; }}
  .badge-ok {{ background: #ffdd57; color: #333; border-radius: 4px; padding: 2px 8px; }}
  .badge-bad {{ background: #d62728; color: white; border-radius: 4px; padding: 2px 8px; }}
</style>
</head>
<body>
<h1>🌍 Overseas Cross-Country ILI Validation</h1>
<p style="color:#666">P3-1 (예측 생성) + P3-3 (논문 cross-country 비교) &nbsp;|&nbsp;
   생성일: <b>2026-05-25</b> &nbsp;|&nbsp;
   Feature: ILI lag1~4 + sin/cos_week + year_trend (7개)</p>

<div class="card">
  <div class="stat"><div class="stat-val">{int(success_rate)}%</div><div class="stat-lbl">모델 성공률</div></div>
  <div class="stat"><div class="stat-val">{len(metrics_df["model"].unique())}</div><div class="stat-lbl">평가 모델 수</div></div>
  <div class="stat"><div class="stat-val">{len(metrics_df["country"].unique())}</div><div class="stat-lbl">대상 국가 수</div></div>
  <div class="stat"><div class="stat-val">{best_model}</div><div class="stat-lbl">평균 R² 1위 모델</div></div>
</div>

<div class="card">
<h2>1. 국가별 데이터 현황</h2>
<table class="info-table">
  <thead><tr><th>국가</th><th>Source</th><th>ILI 단위</th><th>전체 rows</th>
              <th>기간</th><th>Train</th><th>Test</th></tr></thead>
  <tbody>{info_rows}</tbody>
</table>
<p style="font-size:.85em;color:#888">
JP: 2023-2026만 가능 (JIHS 2022 이전 미공개). DE/FR: WHO FluNet 양성률. US: Delphi wILI%.
</p>
</div>

<div class="card">
<h2>2. 국가별 평균 R²</h2>
{"".join("<span class='stat'><div class='stat-val' style='color:{}'>{:.3f}</div><div class='stat-lbl'>{} mean R²</div></span>".format(country_colors.get(c,"#333"),v,c) for c,v in mean_r2_by_country.items())}
</div>

<div class="card">
<h2>3. Metric Tables</h2>
{_pivot_to_html(pivot_r2, "R² (높을수록 좋음 ↑) — scale-invariant, cross-country 비교 ✅")}
{_pivot_to_html(pivot_mape, "MAPE % (낮을수록 좋음 ↓) — scale-invariant, cross-country 비교 ✅", fmt="pct")}
{_pivot_to_html(pivot_wis, "WIS (낮을수록 좋음 ↓) — ⚠ <b>unit-dependent</b>: within-country 비교만 의미. KR/HK/DE/FR positivity%, US wILI%, JP per_clinic 단위 다름 (R8.4 Gemini P0 disclaimer)")}
{_pivot_to_html(pivot_pi95, "PICP95 (0.90+ 권장 ↑) — proportion, cross-country 비교 ✅")}
</div>

<div class="card">
<h2>4. 오류/실패 모델</h2>
{"<p style='color:green'>모든 모델 성공</p>" if metrics_df["error"].isna().all() else
 metrics_df[metrics_df["error"].notna()][["country","model","error","n_train"]].to_html(index=False, classes="metric-table")}
</div>

<p style="color:#aaa;font-size:.8em;margin-top:32px">
MPH Infection Simulation · P3 Overseas Validation · 2026-05-25
</p>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    log.info("[phase18] HTML 리포트: %s", html_path)
    return html_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 실행 함수 (Public API)
# ─────────────────────────────────────────────────────────────────────────────

def run(
    countries: Optional[list[str]] = None,
    model_names: Optional[list[str]] = None,
    out_dir: Optional[Path] = None,
    quick: bool = False,
    db_path: Optional[Path] = None,
    test_weeks: int = 52,
    granularity: str = "national",
    run_comparison: bool = True,
) -> pd.DataFrame:
    """C-1 + C-3 해외 ILI 검증 실행.

    Args:
        countries: 대상 국가 코드 리스트 (기본: US/JP/DE/FR/KR).
        model_names: 평가 모델 이름 리스트 (기본: DEFAULT_MODELS).
        out_dir: 출력 디렉토리 (기본: simulation/results/overseas_validation/).
        quick: True면 Optuna 5 trials (빠른 검증), False면 10 trials.
        db_path: SQLite DB 경로 (기본: epi_real_seoul.db).
        test_weeks: test set 크기 (기본 52주).
        granularity (Sprint γ Item 7, 2026-05-26): "national" | "regional" | "gu-level".
            national  — 국가별 단일 ILI 시계열 (현재 default + 유일하게 fully 작동)
            regional  — sub-national (예: US states from Delphi, DE Bundesländer
                        from RKI, JP prefecture from JIHS). overseas_ili_regional
                        테이블 의존. 일부 국가만 데이터 가용.
            gu-level  — 국내 (KR) 자치구 단위 + 해외 sub-national 매칭. KR baseline
                        은 sentinel_influenza_by_gu 의존 (테이블 존재 시).
            현재 minimal scaffolding 상태 — regional/gu-level 은 NotImplementedError stub.
            full implementation 은 별도 sprint (audit MD §item 7 권장 phased plan).
        run_comparison (Sprint γ Item 3, 2026-05-26): True면 run() 끝에
            generate_international_comparison.py + generate_mc_comparison_csv.py
            를 후처리로 실행 (KR + 해외 비교 CSV/HTML 생성). non-fatal — 실패 시
            warning 만 로그 + return value 동일.

    Returns:
        DataFrame[country, model, r2, rmse, mape, wis, pi95_coverage,
                  n_train, n_test, error, source, unit, granularity]
        — 국가 × 모델 전체 결과. (granularity column 은 Sprint γ Item 7 신규.)

    Performance:
        16 models × 5 countries × 10 Optuna trials ≈ 60-120분 (CPU).
        quick=True: ≈ 20-40분.
        granularity != "national" 은 stub → 즉시 ValueError.
    Side effects:
        - out_dir 에 predictions_<country>_<model>.csv (80개)
        - overseas_metrics_all.csv
        - overseas_validation_report.html
        - (run_comparison=True 시) international_comparison.{html,csv} +
          mc_comparison_metrics.csv (generate_*.py 출력)
    """
    # Sprint γ Item 7 + R8.4 (2026-05-26): granularity gate.
    # 'national' (default) + 'regional' (sub-national) 모두 구현.
    # 'gu-level' = KR 자치구 단위 — DB 에 sentinel_influenza_by_gu 미존재 → stub 유지.
    if granularity not in ("national", "regional", "gu-level"):
        raise ValueError(
            f"granularity={granularity!r} invalid; expected one of "
            "'national' | 'regional' | 'gu-level'"
        )
    if granularity == "gu-level":
        raise NotImplementedError(
            "P3 gu-level (KR 자치구 단위) not yet implemented. "
            "KR sub-national ILI 데이터 (sentinel_influenza_by_gu) 가 DB 에 없음. "
            "Sprint γ Item 7 scaffolding 만 존재. 'regional' (overseas sub-national) 사용."
        )
    from simulation.database import safe_connect

    # ── 파라미터 기본값 ──
    countries    = countries or list(COUNTRY_SOURCES.keys())
    model_names  = model_names or DEFAULT_MODELS
    out_dir      = out_dir or (Path(__file__).parent.parent / "results" / "overseas_validation")
    db_path      = db_path or _DB_PATH
    n_trials     = 5 if quick else 10

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("[phase18] 시작 — countries=%s, models=%d, trials=%d",
             countries, len(model_names), n_trials)

    # ── Optuna 예산 설정 ──
    for mname in model_names:
        if mname not in _FOUNDATION_ZEROSHOT:  # foundation(TimesFM) zero-shot 은 Optuna 미사용
            _set_trial_budget(mname, n_trials)

    # ── 데이터 로드 + Feature Engineering ──
    # R8.4 (2026-05-26): country_data 키 = (country, region) tuple — regional 지원.
    # National 모드: region="national" 마커. Regional 모드: 실제 region 명.
    country_data: dict[tuple[str, str], dict] = {}
    country_info: dict[tuple[str, str], dict] = {}

    def _load_one(country: str, region: str, df_raw: pd.DataFrame, source: str, unit: str) -> None:
        """Build features + split + cache into country_data[(country, region)]."""
        df_feat = _build_features(df_raw)
        df_train, df_test = _split_train_test(df_feat, test_weeks=test_weeks)
        X_train = df_train[FEATURE_COLS].values.astype(np.float32)
        y_train = df_train["ili_rate"].values.astype(np.float32)
        X_test  = df_test[FEATURE_COLS].values.astype(np.float32)
        y_test  = df_test["ili_rate"].values.astype(np.float32)
        country_data[(country, region)] = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test,   "y_test": y_test,
            "df_test": df_test,
        }
        country_info[(country, region)] = {
            "source": source, "unit": unit,
            "n_rows": len(df_feat),
            "year_range": f"{df_feat['year'].min()}-{df_feat['year'].max()}",
            "n_train": len(df_train),
            "n_test": len(df_test),
        }
        log.info("[phase18] %s/%s: train=%d, test=%d, ILI range=[%.2f,%.2f]",
                 country, region, len(y_train), len(y_test), y_test.min(), y_test.max())

    with safe_connect(str(db_path)) as con:
        for country in countries:
            if granularity == "regional":
                # R8.4: gu-level/regional — 가용 sub-national region 별로 학습.
                regional_cfg = REGIONAL_SOURCES.get(country)
                if regional_cfg is None:
                    # KR 같은 sub-national 데이터 없는 나라 → national fallback.
                    log.info("[phase18] %s: regional 데이터 없음 → national fallback", country)
                    try:
                        df_raw = _fetch_ili_series(con, country)
                        cfg_nat = COUNTRY_SOURCES[country]
                        _load_one(country, "national", df_raw,
                                  cfg_nat["source"], cfg_nat["unit"])
                    except Exception as exc:
                        log.error("[phase18] %s national fallback 실패: %s", country, exc)
                else:
                    regions = _get_regions_for_country(con, country)
                    log.info("[phase18] %s: %d sub-national regions (%s)",
                             country, len(regions), regional_cfg["source"])
                    for region in regions:
                        try:
                            df_raw = _fetch_ili_series_regional(con, country, region)
                            _load_one(country, region, df_raw,
                                      regional_cfg["source"], regional_cfg["unit"])
                        except Exception as exc:
                            log.warning("[phase18] %s/%s 로드 실패: %s",
                                        country, region, exc)
            else:
                # National (default, 기존 path)
                try:
                    df_raw = _fetch_ili_series(con, country)
                    cfg = COUNTRY_SOURCES[country]
                    _load_one(country, "national", df_raw, cfg["source"], cfg["unit"])
                except Exception as exc:
                    log.error("[phase18] %s 데이터 로드 실패: %s", country, exc)

    if not country_data:
        raise RuntimeError("[phase18] 모든 국가 데이터 로드 실패")

    # ── 모델 × (국가, 지역) 루프 ──
    # R8.4: (country, region) tuple 키 — regional 모드 시 region 별 학습.
    all_records: list[dict] = []

    for (country, region), cdata in country_data.items():
        X_train = cdata["X_train"]
        y_train = cdata["y_train"]
        X_test  = cdata["X_test"]
        y_test  = cdata["y_test"]
        df_test = cdata["df_test"]
        cinfo   = country_info[(country, region)]

        for model_name in model_names:
            log.info("[phase18] ── %s/%s × %s ──", country, region, model_name)

            # Chronos: 별도 zero-shot path (R8.4 P0 fix: country/region 전달 → phase_id 고유성)
            if model_name in _FOUNDATION_ZEROSHOT:
                res = _run_foundation_zeroshot(model_name, y_train, y_test,
                                            n_steps=len(y_test),
                                            country=country, region=region)
            else:
                res = _run_one_model(model_name, X_train, y_train, X_test, y_test,
                                     country=country, region=region)

            # ── 예측 CSV 저장 (region 포함 파일명) ──
            if res["y_pred"] is not None:
                pred_df = df_test[["year", "week_no", "ili_rate"]].copy()
                pred_df["y_pred"]  = res["y_pred"]
                pred_df["model"]   = model_name
                pred_df["country"] = country
                pred_df["region"]  = region
                _region_safe = region.replace("/", "_").replace(" ", "_")
                _model_safe  = model_name.replace("/", "_")
                csv_path = out_dir / f"predictions_{country}_{_region_safe}_{_model_safe}.csv"
                pred_df.to_csv(csv_path, index=False)

            # ── 메트릭 레코드 (region 컬럼 추가) ──
            m = res.get("metrics", {})
            record = {
                "country": country,
                "region": region,
                "model": model_name,
                "r2": m.get("r2", float("nan")),
                "rmse": m.get("rmse", float("nan")),
                "mae": m.get("mae", float("nan")),
                "mape": m.get("mape", float("nan")),
                "wis": m.get("wis", float("nan")),
                "pi95_coverage": m.get("pi95_coverage", float("nan")),
                "n_train": res.get("n_train", 0),
                "n_test": res.get("n_test", 0),
                "error": res.get("error"),
                "source": cinfo["source"],
                "unit": cinfo["unit"],
            }
            all_records.append(record)

    # ── 결과 DataFrame + granularity column (Sprint γ Item 7) ──
    # granularity 컬럼은 CSV write 이전에 추가 (Codex audit 2026-05-26 P1 fix).
    results_df = pd.DataFrame(all_records)
    results_df["granularity"] = granularity   # 현재는 "national" 만 — stub 게이트 통과 후 도달

    # Sprint E3 (2026-05-26, Gemini P3 P0): BH-FDR 적용 범위 fix.
    # 기존 phase_evaluator 의 BH-FDR family = 3 (DM tests within model).
    # 실제 P3 multiple testing family = (country × region × model) ~3,528 cells.
    # post-hoc 적용: results_df 의 p-value column 들 에 BH-FDR 일괄 적용.
    try:
        from simulation.analytics.metrics import adjust_pvalues
        _p_cols = [c for c in results_df.columns if c.startswith("dm_p_value") or
                   c.startswith("dm_p_vs_") or "_p_value" in c]
        for _pc in _p_cols:
            _vals = results_df[_pc].dropna().tolist()
            if len(_vals) >= 2:
                bh = adjust_pvalues(_vals, method="fdr_bh")
                _adj_col = f"{_pc}_bh_phase15"
                _adj_iter = iter(bh["p_adj"])
                results_df[_adj_col] = results_df[_pc].apply(
                    lambda v: next(_adj_iter) if pd.notna(v) else float("nan")
                )
                log.info(
                    f"[phase18] BH-FDR (P3 family n={len(_vals)}) applied to {_pc} → {_adj_col}"
                )
    except Exception as _bhe:
        log.warning(f"[phase18] P3 BH-FDR post-hoc failed: {_bhe}")

    metrics_csv = out_dir / "overseas_metrics_all.csv"
    results_df.to_csv(metrics_csv, index=False)
    log.info("[phase18] 메트릭 CSV: %s", metrics_csv)

    # ── KR baseline 추가 (per_model_optimal) ──
    kr_baseline = _load_kr_baseline(model_names)
    if not kr_baseline.empty:
        baseline_csv = out_dir / "kr_baseline_metrics.csv"
        kr_baseline.to_csv(baseline_csv, index=False)
        log.info("[phase18] KR baseline CSV: %s", baseline_csv)

    # ── HTML 리포트 ──
    _generate_html_report(results_df, out_dir, country_info)

    # ── Sprint E5 (R7 큰 변경, opt-in): LORO Cross-Validation ──
    # 사용자 명시 (Sprint E5): "LORO CV — per-country leave-one-region-out"
    # Gemini P3 P1: per-country LORO 가 진정한 sub-national generalization 평가.
    # 각 나라 (≥3 regions) 마다 region N-1 train + 1 held-out test 반복.
    # HK 는 1 region 만 → exclude.
    # opt-in via env MPH_PHASE18_LORO=1 (default off — cost).
    # 현재는 scaffolding — actual cross-validation logic 은 별도 sprint.
    if GLOBAL.ops.phase18_loro:
        log.warning(
            "[E5] MPH_PHASE18_LORO=1 — Leave-One-Region-Out CV is scaffolded "
            "but NOT YET IMPLEMENTED. See Sprint E5 (2026-05-26) for next-sprint "
            "implementation plan. Excludes HK (1 region)."
        )
        # TODO: per-country (US 10 / DE 17 / FR 22 / JP 47) LORO loop
        # TODO: train on N-1 regions, predict held-out region, aggregate per-country
        # TODO: results to out_dir / "loro_results.csv"

    # ── Sprint γ Item 3 (2026-05-26): post-P3 comparison reports ──
    # generate_international_comparison.py + generate_mc_comparison_csv.py 후처리.
    # 사용자 의도: KR + 해외 + (향후) gu-level 동일 환경 비교분석.
    # 둘 다 non-fatal — 실패해도 P3 자체 결과 (results_df) 영향 없음.
    if run_comparison:
        _run_comparison_postprocess(out_dir)

    # ── 요약 출력 ──
    _print_summary(results_df)

    return results_df


def _run_comparison_postprocess(out_dir: Path) -> None:
    """Sprint γ Item 3 post-process — generate_international_comparison +
    generate_mc_comparison_csv subprocess 호출. 둘 다 non-fatal.

    Args:
        out_dir: P3 출력 디렉토리. 두 generator 의 결과도 같은 dir 에 합쳐짐.

    Performance: 각 generator 가 DB read + analysis (수십 초 ~ 수 분).
    Side effects: out_dir 에 international_comparison.{html,csv} +
                  mc_comparison_metrics.csv 추가.
    """
    import subprocess
    import sys
    for module, label in [
        ("simulation.scripts.generate_international_comparison", "international comparison"),
        ("simulation.scripts.generate_mc_comparison_csv", "mc comparison"),
    ]:
        try:
            log.info("[phase18] post-process: %s (%s)", label, module)
            result = subprocess.run(
                [sys.executable, "-m", module,
                 "--output-dir", str(out_dir)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                log.info("[phase18] %s 완료", label)
            else:
                log.warning("[phase18] %s 실패 (rc=%d): %s",
                            label, result.returncode, (result.stderr or "")[:300])
        except Exception as e:
            log.warning("[phase18] %s skipped — %s: %s", label, type(e).__name__, e)


def _load_kr_baseline(model_names: list[str]) -> pd.DataFrame:
    """per_model_optimal 에서 KR full-feature test_metrics 로드 (C-3 비교 baseline).

    Args:
        model_names: 조회할 모델 이름 리스트.

    Returns:
        DataFrame[model, r2, rmse, mape, wis, pi95_coverage, n, source='kr_full_feature'].
        없으면 empty DataFrame.

    Performance: O(n_models) file reads.
    Side effects: None (read-only).
    """
    opt_dir = get_results_dir() / "per_model_optimal"  # SSOT MPH_OUTPUT_ROOT (was Path(__file__).parent.parent/results)
    records = []
    for mname in model_names:
        p = opt_dir / f"{mname}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            m = d.get("test_metrics", {})
            records.append({
                "model": mname,
                "r2": m.get("r2", float("nan")),
                "rmse": m.get("rmse", float("nan")),
                "mae": m.get("mae", float("nan")),
                "mape": m.get("mape", float("nan")),
                "wis": m.get("wis", float("nan")),
                "pi95_coverage": m.get("pi95_coverage", float("nan")),
                "n": m.get("n", float("nan")),
                "source": "kr_full_feature",
            })
        except Exception:
            pass
    return pd.DataFrame(records) if records else pd.DataFrame()


def _print_summary(results_df: pd.DataFrame) -> None:
    """콘솔 요약 출력."""
    print("\n" + "="*60)
    print("P3 — Overseas ILI Validation 완료")
    print("="*60)
    success = results_df["error"].isna().sum()
    fail    = results_df["error"].notna().sum()
    print(f"성공: {success} / 실패: {fail} / 전체: {len(results_df)}")
    print()

    for country in results_df["country"].unique():
        sub = results_df[results_df["country"] == country].copy()
        best = sub.loc[sub["r2"].idxmax()] if sub["r2"].notna().any() else None
        print(f"[{country}] n_models={len(sub)}, "
              f"mean_R²={sub['r2'].mean():.3f}, "
              f"best={best['model'] if best is not None else 'N/A'}"
              f"(R²={best['r2']:.3f})" if best is not None else "")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="P3: Overseas ILI cross-country validation")
    p.add_argument("--countries", nargs="+", default=None,
                   help="국가 코드 (US JP DE FR KR). 기본: 전체")
    p.add_argument("--models", nargs="+", default=None,
                   help="모델 이름 목록. 기본: DEFAULT_MODELS")
    p.add_argument("--quick", action="store_true",
                   help="빠른 검증 (Optuna 5 trials)")
    p.add_argument("--test-weeks", type=int, default=52,
                   help="Test set 크기 (주). 기본: 52")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="출력 디렉토리")
    # Sprint γ Item 7 (2026-05-26): granularity flag
    p.add_argument("--granularity", choices=["national", "regional", "gu-level"],
                   default="national",
                   help="비교 단위 — national (default, 작동) / regional / gu-level "
                        "(둘 다 minimal scaffolding stub — NotImplementedError)")
    # Sprint γ Item 3 (2026-05-26): comparison post-process toggle
    p.add_argument("--no-comparison", action="store_true",
                   help="generate_international_comparison + generate_mc_comparison_csv "
                        "후처리 skip (기본: 자동 실행)")
    return p.parse_args(argv)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = _parse_args()
    run(
        countries=args.countries,
        model_names=args.models,
        out_dir=args.out_dir,
        quick=args.quick,
        test_weeks=args.test_weeks,
        granularity=args.granularity,             # Sprint γ Item 7
        run_comparison=not args.no_comparison,    # Sprint γ Item 3
    )
