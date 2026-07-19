"""
sci_supplement.py — SCI 게재용 외적타당도/교란 3-표 (Gemini #1 블로커)
=====================================================================

신규 스크립트 (라이브 코드 무수정). read_only_connect + 기존 feature_cache.
재학습 최소 = 해석가능 count 모델 **NegBinGLM** (statsmodels GLM, ~초/refit)
을 대표로, BASIC 피처(lag+계절성 13)로 leak-free rolling 1-step.

세 표 (docs/MASTER_REFERENCE / Gemini 외적타당도·교란 리뷰):

  ① LOSO  — multi-season leave-one-season-out (TRIPOD-AI 5j, epi 저널 요구).
            한 시즌 hold-out → 나머지 fit → 해당 시즌 rolling 1-step 예측 →
            시즌별 WIS/R²/MASE. 시즌별 + 평균±분포.
  ② regime-stratified — 챔피언(대표) WIS+95%CI 를 pre-COVID(2019)/
            during-COVID(2020-22)/post-rebound(2023-25) sub-window 별.
            full-window rolling 1-step 예측을 regime 으로 re-slice (재학습 0
            추가 — ① 의 rolling pred 재사용).
  ③ covid sensitivity — apply_covid_sensitivity_mode(data.py) include vs
            exclude vs indicator 로 WIS 차이 1-표 + WHO FluNet KR subtype
            positivity 를 covariate/strata 가능성 + limitation 정량.

왜 NegBinGLM:
  - FusedEpi refit = TiRex (수시간/refit) → LOSO/rolling 7× 비현실적.
  - NegBinGLM = 해석가능 count 모델 (챔피언 아님 — 챔피언은 FusedEpi 하나), native
    predict_interval, statsmodels GLM → 초 단위 refit.
  - WIS = canonical `weighted_interval_score_empirical` (leak-free SSOT,
    point forecast + OOF residual 분위수; pipeline 전체 동일 함수).

출력:
  simulation/results/sci_supplement/
    ├── loso_table.csv            # 시즌별 WIS/R²/MASE + 평균±분포 행
    ├── regime_stratified.csv     # regime별 WIS + 95%CI (block bootstrap)
    ├── covid_sensitivity.csv     # include/exclude/indicator WIS
    ├── subtype_strata.csv        # WHO FluNet KR subtype positivity 분해
    ├── summary.json              # 3-표 요약 + 메타 (모델/재학습량/sqlite=0)
    └── figures/
        ├── loso_by_season.png
        └── regime_wis_ci.png

Usage:
    .venv/bin/python -m simulation.scripts.sci_supplement \
        [--db simulation/data/db/epi_real_seoul.db] \
        [--cache simulation/cache/feature_cache.parquet] \
        [--out simulation/results/sci_supplement]

Determinism: seed=42 고정 (np + bootstrap RandomState). 동일 입력 → 동일 출력.
sqlite writes = 0 (read_only_connect + mode=ro). leak-free: rolling 1-step,
fit on STRICT past only, OOF residual 도 train-internal.

Reference: TRIPOD-AI 2024 §5j (external validity); Cawley & Talbot 2010
(no test-set selection); Reich 2019 / Bracher 2021 (WIS).
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # repo root (scripts/sci_supplement/tables.py)
DEFAULT_DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
DEFAULT_CACHE = ROOT / "simulation" / "cache" / "feature_cache.parquet"
DEFAULT_OUT = ROOT / "simulation" / "results" / "sci_supplement"

SEED = 42
PAPER_CUTOFF_WEEK = 337  # config_global.DataSplit.paper_cutoff_week (in-sample window)

# BASIC eval features (SSOT: simulation/pipeline/baseline.BASIC_FEATURE_COLS) —
# lag + 계절성, 전부 causal / leakage-safe (shift-based / calendar).
BASIC_FEATURE_COLS = [
    "ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag4", "ili_rate_lag52",
    "sin_month", "cos_month",
    "fourier_sin_h1", "fourier_cos_h1", "fourier_sin_h2", "fourier_cos_h2",
    "fourier_sin_h3", "fourier_cos_h3",
    "season_idx",
]

# Regime mapping (SSOT: scripts/phase11_loso_full.SEASON_CONTEXT).
# season = epi-year starting ~ Aug (week_start month >= 8 → that year, else prev).
REGIME_OF_SEASON = {
    2019: "pre_covid",       # pre-COVID baseline
    2020: "during_covid",    # COVID y1 (lockdown + school closure)
    2021: "during_covid",    # COVID y2 (vaccines, masks)
    2022: "during_covid",    # COVID y3 (Omicron, partial reopen)
    2023: "post_rebound",    # post-COVID rebound
    2024: "post_rebound",    # normal season
    2025: "post_rebound",    # current season (partial)
}
SEASON_NOTE = {
    2019: "pre-COVID baseline", 2020: "COVID y1 lockdown",
    2021: "COVID y2 vaccines", 2022: "COVID y3 Omicron",
    2023: "post-COVID rebound", 2024: "normal season",
    2025: "current (partial)",
}

# COVID-era window (SSOT: data.apply_covid_sensitivity_mode).
COVID_START = np.datetime64("2020-03-01")
COVID_END = np.datetime64("2022-12-31")

# Min train weeks before the first rolling prediction (need lag52 warm-up).
MIN_TRAIN_WEEKS = 60


# ──────────────────────────────────────────────────────────────────────
# Data load — read-only, in-sample window (337 weeks), BASIC matrix
# ──────────────────────────────────────────────────────────────────────
def season_of(date64: np.datetime64) -> int:
    """Epi-year season for a week_start date (season starts ~ August).

    Args:
        date64: numpy datetime64 week-start.

    Returns:
        int season label (e.g. 2020 = the 2020/21 winter season).
    """
    d = np.datetime64(date64, "D").astype("datetime64[D]").astype(object)
    return d.year if d.month >= 8 else d.year - 1


def load_basic_matrix(cache_path: Path):
    """Load the in-sample BASIC design matrix + target + dates (read-only).

    Reads the existing FE parquet cache (no DB write, no FE recompute) and
    carves the in-sample window (first PAPER_CUTOFF_WEEK rows = R1 convention),
    then column-slices to BASIC_FEATURE_COLS (lag + seasonal, leakage-safe).

    Args:
        cache_path: path to simulation/cache/feature_cache.parquet.

    Returns:
        (X, y, dates, seasons): X (n,13) float64, y (n,) float64,
        dates (n,) datetime64[D], seasons (n,) int — all length n=337.

    Raises:
        FileNotFoundError: cache missing (must be built by a prior run_data).
        KeyError: a BASIC column is absent from the cache.

    Side effects: reads parquet only. No DB, no disk write.
    """
    import polars as pl

    if not cache_path.exists():
        raise FileNotFoundError(
            f"FE 캐시 없음: {cache_path} — 먼저 학습/run_data 로 생성 필요"
        )
    df = pl.read_parquet(cache_path)
    missing = [c for c in BASIC_FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"BASIC 피처 누락: {missing}")
    y = df["ili_rate"].to_numpy().astype(np.float64)
    dates = df["week_start"].to_numpy().astype("datetime64[D]")
    X = df.select(BASIC_FEATURE_COLS).to_numpy().astype(np.float64)
    # in-sample window = first PAPER_CUTOFF_WEEK rows (R1 4-way split SSOT).
    n_full = len(y)
    cut = min(PAPER_CUTOFF_WEEK, n_full)
    X, y, dates = X[:cut], y[:cut], dates[:cut]
    # NaN guard (lag52 warm-up rows): impute column-mean (deterministic).
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(col_mean, np.where(bad)[1])
    seasons = np.array([season_of(d) for d in dates], dtype=int)
    return X, y, dates, seasons


# ──────────────────────────────────────────────────────────────────────
# NegBinGLM rolling 1-step — leak-free (fit strict past, predict 1 ahead)
# ──────────────────────────────────────────────────────────────────────
def _fit_predict_negbin(X_tr, y_tr, X_te):
    """Fit NegBinGLM on (X_tr,y_tr), return point preds + OOF residuals.

    Args:
        X_tr: (k,13) train design. y_tr: (k,) train target.
        X_te: (m,13) rows to predict.

    Returns:
        (pred (m,), residuals (k,)): point forecasts for X_te and in-sample
        OOF-proxy residuals (y_tr - fitted) used for the WIS predictive band.

    Side effects: none. Deterministic (statsmodels GLM IRLS, seeded NB band).
    """
    from simulation.models.negbin_glm import NegBinGLMForecaster

    m = NegBinGLMForecaster()
    m.fit(X_tr, y_tr)
    pred = np.asarray(m.predict(X_te), dtype=np.float64)
    fitted_tr = np.asarray(m.predict(X_tr), dtype=np.float64)
    resid = (y_tr - fitted_tr).astype(np.float64)
    return pred, resid


def rolling_one_step(X, y, dates, seasons, *, hold_out_season=None,
                     covid_mode="include", min_train=MIN_TRAIN_WEEKS):
    """Leak-free rolling-origin 1-step NegBinGLM over the in-sample window.

    For each origin t >= min_train (and t not in the held-out season, when
    LOSO), fit on STRICT past [0:t] (optionally excluding the held-out
    season's weeks and/or applying a COVID sensitivity transform) and predict
    week t. The prediction at t therefore never sees t or any future week.

    Args:
        X: (n,13) BASIC design. y: (n,). dates: (n,) datetime64[D].
        seasons: (n,) int season labels.
        hold_out_season: if set, the predicted weeks are exactly that season's
            weeks and the training pool EXCLUDES that season entirely
            (true LOSO — that season is never in any training fold).
        covid_mode: "include" | "exclude" | "indicator" — passed through to
            data.apply_covid_sensitivity_mode on each training fold.
        min_train: minimum past weeks before the first prediction (lag warm-up).

    Returns:
        dict {idx, dates, y_true, y_pred, residuals_last, seasons} where idx
        are the global row indices predicted (np arrays, aligned).

    Side effects: none (pure compute). Determinism: seed-fixed.
    """
    from simulation.pipeline.data import apply_covid_sensitivity_mode

    n = len(y)
    pred_idx, pred_dates, y_true, y_pred, pred_seasons = [], [], [], [], []
    last_resid = np.zeros(0)

    for t in range(min_train, n):
        # Define which origins we score:
        if hold_out_season is not None:
            if seasons[t] != hold_out_season:
                continue  # only score the held-out season's weeks
            # train pool = all weeks NOT in the held-out season, strictly before t
            train_mask = (seasons != hold_out_season) & (np.arange(n) < t)
        else:
            train_mask = np.arange(n) < t  # all strict past

        if train_mask.sum() < min_train:
            continue
        X_tr = X[train_mask].copy()
        y_tr = y[train_mask].copy()
        d_tr = dates[train_mask].copy()
        X_te = X[t:t + 1].copy()

        # COVID sensitivity transform on the training fold (+ test col-align).
        X_tr, y_tr, d_tr, _cols, X_te = apply_covid_sensitivity_mode(
            X_tr, y_tr, d_tr, list(BASIC_FEATURE_COLS), X_te, covid_mode
        )
        if len(y_tr) < min_train:
            continue
        try:
            p, resid = _fit_predict_negbin(X_tr, y_tr, X_te)
        except Exception as e:  # noqa: BLE001 — robustness, never abort the sweep
            log.warning(f"    [rolling] t={t} fit 실패: {e}")
            continue
        pred_idx.append(t)
        pred_dates.append(dates[t])
        y_true.append(float(y[t]))
        y_pred.append(float(p[0]))
        pred_seasons.append(int(seasons[t]))
        last_resid = resid

    return {
        "idx": np.array(pred_idx, dtype=int),
        "dates": np.array(pred_dates, dtype="datetime64[D]"),
        "y_true": np.array(y_true, dtype=np.float64),
        "y_pred": np.array(y_pred, dtype=np.float64),
        "seasons": np.array(pred_seasons, dtype=int),
        "residuals_last": np.asarray(last_resid, dtype=np.float64),
    }


# ──────────────────────────────────────────────────────────────────────
# Metrics — WIS (canonical leak-free SSOT) + R² + MASE
# ──────────────────────────────────────────────────────────────────────
def _wis(y_true, y_pred, residuals):
    """Mean WIS via the canonical empirical SSOT helper.

    Args:
        y_true: (m,) observed. y_pred: (m,) point forecast.
        residuals: (k,) OOF-proxy residuals defining the predictive band.

    Returns:
        float mean WIS (NaN if < few residuals or empty).
    """
    from simulation.analytics.diagnostics import (
        weighted_interval_score_empirical,
    )
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS

    if len(y_true) == 0 or len(residuals) < 5:
        return float("nan")
    arr = weighted_interval_score_empirical(
        np.asarray(y_true), np.asarray(y_pred), np.asarray(residuals),
        alphas=list(FLUSIGHT_ALPHAS),
    )
    return float(np.mean(arr))


def _r2(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if len(yt) < 2:
        return float("nan")
    sst = float(np.sum((yt - yt.mean()) ** 2))
    if sst <= 0:
        return float("nan")
    sse = float(np.sum((yp - yt) ** 2))
    return 1.0 - sse / sst


def _mase(y_true, y_pred, y_train):
    """MASE_h1 — MAE / naive-1-step MAE on the train series (Hyndman 2006)."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    ytr = np.asarray(y_train)
    if len(yt) == 0 or len(ytr) < 2:
        return float("nan")
    denom = float(np.mean(np.abs(np.diff(ytr))))
    if denom <= 0:
        return float("nan")
    return float(np.mean(np.abs(yp - yt)) / denom)


def _block_bootstrap_ci(y_true, y_pred, residuals, *, block=4, b=2000,
                        seed=SEED):
    """Block-bootstrap 95% CI for mean WIS (serial-correlation aware).

    Args:
        y_true,y_pred: aligned (m,) arrays. residuals: band residuals.
        block: moving-block length (weeks). b: bootstrap reps. seed: RNG.

    Returns:
        (lo, hi): 2.5 / 97.5 percentile of the bootstrap mean-WIS dist;
        (nan,nan) if m < block.
    """
    from simulation.analytics.diagnostics import (
        weighted_interval_score_empirical,
    )
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS

    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    m = len(yt)
    if m < block or len(residuals) < 5:
        return float("nan"), float("nan")
    per_week = weighted_interval_score_empirical(
        yt, yp, np.asarray(residuals), alphas=list(FLUSIGHT_ALPHAS)
    )
    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(m / block))
    starts_pool = np.arange(0, m - block + 1)
    means = np.empty(b)
    for i in range(b):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:m]
        means[i] = float(np.mean(per_week[idx]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ──────────────────────────────────────────────────────────────────────
# ① LOSO table
# ──────────────────────────────────────────────────────────────────────
def build_loso(X, y, dates, seasons):
    """Leave-one-season-out: per-season WIS/R²/MASE (NegBinGLM, BASIC, rolling).

    Returns: (rows, mean_row) — list of per-season dicts + an aggregate dict.
    """
    uniq = sorted(set(seasons.tolist()))
    rows = []
    for s in uniq:
        res = rolling_one_step(X, y, dates, seasons, hold_out_season=s)
        if len(res["y_true"]) == 0:
            log.info(f"  [LOSO] season {s}: 평가 주 없음 (warm-up 부족) → skip")
            continue
        # train series for MASE = all weeks NOT in season s (the LOSO pool).
        y_train_pool = y[seasons != s]
        wis = _wis(res["y_true"], res["y_pred"], res["residuals_last"])
        r2 = _r2(res["y_true"], res["y_pred"])
        mase = _mase(res["y_true"], res["y_pred"], y_train_pool)
        mae = float(np.mean(np.abs(res["y_pred"] - res["y_true"])))
        rmse = float(np.sqrt(np.mean((res["y_pred"] - res["y_true"]) ** 2)))
        rows.append({
            "season": int(s),
            "regime": REGIME_OF_SEASON.get(int(s), "unknown"),
            "note": SEASON_NOTE.get(int(s), ""),
            "n_weeks": int(len(res["y_true"])),
            "wis": round(wis, 4),
            "r2": round(r2, 4),
            "mase_h1": round(mase, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "y_test_mean": round(float(np.mean(res["y_true"])), 4),
        })
        log.info(f"  [LOSO] season {s} ({SEASON_NOTE.get(s,'')}): "
                 f"WIS={wis:.3f} R2={r2:.3f} MASE={mase:.3f} n={len(res['y_true'])}")

    def _agg(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return (round(float(np.mean(vals)), 4) if vals else float("nan"),
                round(float(np.std(vals)), 4) if vals else float("nan"),
                round(float(np.min(vals)), 4) if vals else float("nan"),
                round(float(np.max(vals)), 4) if vals else float("nan"))
    mean_row = {"season": "MEAN±SD (across seasons)", "regime": "all",
                "note": "LOSO aggregate", "n_weeks": sum(r["n_weeks"] for r in rows)}
    for k in ("wis", "r2", "mase_h1", "mae", "rmse"):
        mu, sd, lo, hi = _agg(k)
        mean_row[k] = f"{mu}±{sd}"
        mean_row[f"{k}_min"] = lo
        mean_row[f"{k}_max"] = hi
    return rows, mean_row


# ──────────────────────────────────────────────────────────────────────
# ② regime-stratified table  (re-slice the full-window rolling preds)
# ──────────────────────────────────────────────────────────────────────
def build_regime(X, y, dates, seasons):
    """Regime-stratified WIS + 95%CI from one full-window rolling pass (재학습 0 추가).

    Runs a single full-window rolling-origin pass (no season held out), then
    re-slices the predictions into pre-COVID / during-COVID / post-rebound
    sub-windows and computes WIS + block-bootstrap 95% CI per regime.

    Returns: (rows, full_pass) — per-regime dicts + the raw full-pass result.
    """
    res = rolling_one_step(X, y, dates, seasons, hold_out_season=None)
    resid = res["residuals_last"]
    regimes = np.array([REGIME_OF_SEASON.get(int(s), "unknown")
                        for s in res["seasons"]])
    # Count pre_covid weeks that exist in the in-sample window but are NOT
    # evaluable under rolling-origin 1-step (they precede the lag warm-up).
    n_pre_total = int(np.sum([REGIME_OF_SEASON.get(int(s)) == "pre_covid"
                              for s in seasons]))
    rows = []
    for reg in ("pre_covid", "during_covid", "post_rebound"):
        mask = regimes == reg
        if mask.sum() == 0:
            # Honest fail-loud row: pre_covid (2019, first 47 weeks) sits
            # entirely inside the lag/min-train warm-up, so a leak-free
            # rolling-origin model has no past to forecast it — it is the
            # warm-up window, not a missing result. Disclose, don't omit.
            if reg == "pre_covid" and n_pre_total > 0:
                rows.append({
                    "regime": reg, "seasons": "2019",
                    "n_weeks": 0, "wis": "n/a (warm-up)",
                    "wis_ci95_lo": "n/a", "wis_ci95_hi": "n/a",
                    "r2": "n/a", "mae": "n/a",
                    "y_obs_mean": round(float(np.mean(
                        y[np.array([REGIME_OF_SEASON.get(int(s)) == "pre_covid"
                                    for s in seasons])])), 4),
                    "y_obs_std": "n/a",
                    "note": (f"{n_pre_total}wk all precede the rolling warm-up "
                             f"(min_train={MIN_TRAIN_WEEKS}); not evaluable "
                             f"under leak-free 1-step — first season = warm-up"),
                })
                log.info(f"  [regime] {reg}: n/a — {n_pre_total}wk inside "
                         f"warm-up (first season, no past to forecast)")
            continue
        yt = res["y_true"][mask]
        yp = res["y_pred"][mask]
        wis = _wis(yt, yp, resid)
        lo, hi = _block_bootstrap_ci(yt, yp, resid)
        rows.append({
            "regime": reg,
            "seasons": ",".join(str(s) for s in sorted(
                {int(s) for s, m in zip(res["seasons"], mask) if m})),
            "n_weeks": int(mask.sum()),
            "wis": round(wis, 4),
            "wis_ci95_lo": round(lo, 4),
            "wis_ci95_hi": round(hi, 4),
            "r2": round(_r2(yt, yp), 4),
            "mae": round(float(np.mean(np.abs(yp - yt))), 4),
            "y_obs_mean": round(float(np.mean(yt)), 4),
            "y_obs_std": round(float(np.std(yt)), 4),
            "note": "rolling-origin 1-step, leak-free",
        })
        log.info(f"  [regime] {reg}: WIS={wis:.3f} "
                 f"[{lo:.3f},{hi:.3f}] obs_mean={np.mean(yt):.2f} n={mask.sum()}")
    # shift quantification: train(pre+during) mean vs test(post) mean.
    return rows, res


# ──────────────────────────────────────────────────────────────────────
# ③ covid sensitivity table  (include vs exclude vs indicator)
# ──────────────────────────────────────────────────────────────────────
def build_covid_sensitivity(X, y, dates, seasons):
    """Champion WIS under include / exclude / indicator COVID modes (1-표).

    Each mode runs a full-window rolling pass with the corresponding
    apply_covid_sensitivity_mode transform on every training fold, and we
    score WIS on the COMMON post-rebound weeks (so the comparison is apples-
    to-apples; "exclude" simply removes COVID weeks from the training pool).

    Returns: list of per-mode dicts.
    """
    rows = []
    base_wis = None
    for mode in ("include", "exclude", "indicator"):
        res = rolling_one_step(X, y, dates, seasons, hold_out_season=None,
                               covid_mode=mode)
        resid = res["residuals_last"]
        # score on post-rebound weeks (common, never-COVID eval window).
        regimes = np.array([REGIME_OF_SEASON.get(int(s), "unknown")
                            for s in res["seasons"]])
        mask = regimes == "post_rebound"
        yt, yp = res["y_true"][mask], res["y_pred"][mask]
        wis = _wis(yt, yp, resid)
        if mode == "include":
            base_wis = wis
        rows.append({
            "covid_mode": mode,
            "eval_window": "post_rebound (2023-2025)",
            "n_eval_weeks": int(mask.sum()),
            "wis": round(wis, 4),
            "delta_wis_vs_include": round(wis - base_wis, 4)
            if base_wis is not None else 0.0,
            "rel_delta_pct": round(100.0 * (wis - base_wis) / base_wis, 2)
            if base_wis else 0.0,
            "r2": round(_r2(yt, yp), 4),
        })
        log.info(f"  [covid/{mode}] WIS={wis:.3f} (post-rebound n={mask.sum()})")
    return rows


# ──────────────────────────────────────────────────────────────────────
# subtype positivity strata (WHO FluNet KR) — read-only, covariate/limitation
# ──────────────────────────────────────────────────────────────────────
def build_subtype_strata(db_path: Path):
    """WHO FluNet KR subtype positivity by season (covariate/strata candidate).

    Pulls Republic-of-Korea weekly subtype counts (H1N1pdm09 / H3 / B) read-
    only, aggregates per epi-season, and reports the dominant subtype +
    positivity. This is the strata/covariate candidate and the limitation
    quantification ③ asks for (we do NOT refit on it — Seoul ILI denominator
    differs from the national FluNet sentinel; it is a confounder to disclose).

    Returns: list of per-season dicts (read-only; sqlite writes = 0).
    """
    from simulation.database import read_only_connect

    con = read_only_connect(str(db_path))
    try:
        cur = con.cursor()
        q = """
            SELECT sdate, inf_a_h1n1pdm09, inf_a_h1, inf_a_h3, inf_b,
                   inf_total, spec_processed
            FROM who_flunet
            WHERE iso3 = 'KOR' AND sdate >= '2019-08-01'
            ORDER BY sdate
        """
        recs = cur.execute(q).fetchall()
    finally:
        con.close()

    by_season = {}
    for sdate, h1n1pdm, h1, h3, b, tot, spec in recs:
        try:
            d = np.datetime64(sdate[:10], "D")
        except Exception:
            continue
        s = season_of(d)
        agg = by_season.setdefault(s, {"h1n1pdm09": 0.0, "h1": 0.0, "h3": 0.0,
                                       "b": 0.0, "total": 0.0, "spec": 0.0})
        agg["h1n1pdm09"] += float(h1n1pdm or 0)
        agg["h1"] += float(h1 or 0)
        agg["h3"] += float(h3 or 0)
        agg["b"] += float(b or 0)
        agg["total"] += float(tot or 0)
        agg["spec"] += float(spec or 0)

    rows = []
    for s in sorted(by_season):
        a = by_season[s]
        subtypes = {"A(H1N1)pdm09": a["h1n1pdm09"] + a["h1"],
                    "A(H3)": a["h3"], "B": a["b"]}
        tot = max(a["total"], 1e-9)
        dominant = max(subtypes, key=subtypes.get)
        pos = (a["total"] / a["spec"] * 100.0) if a["spec"] > 0 else float("nan")
        rows.append({
            "season": int(s),
            "regime": REGIME_OF_SEASON.get(int(s), "unknown"),
            "flu_positive_specimens": int(a["total"]),
            "positivity_pct": round(pos, 2),
            "dominant_subtype": dominant,
            "pct_AH1N1pdm09": round(100.0 * subtypes["A(H1N1)pdm09"] / tot, 1),
            "pct_AH3": round(100.0 * subtypes["A(H3)"] / tot, 1),
            "pct_B": round(100.0 * subtypes["B"] / tot, 1),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────
# IO — CSV / JSON / figures
# ──────────────────────────────────────────────────────────────────────
def _write_csv(path: Path, rows, fieldnames=None):
    import csv
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _figures(out_dir: Path, loso_rows, regime_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log.warning(f"  [figure] matplotlib 불가 → skip: {e}")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # LOSO by season
    if loso_rows:
        ss = [r["season"] for r in loso_rows]
        wis = [r["wis"] for r in loso_rows]
        cols = ["#d62728" if r["regime"] == "during_covid"
                else ("#1f77b4" if r["regime"] == "pre_covid" else "#2ca02c")
                for r in loso_rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([str(s) for s in ss], wis, color=cols)
        ax.set_xlabel("Held-out season")
        ax.set_ylabel("WIS (lower = better)")
        ax.set_title("LOSO cross-season WIS — NegBinGLM (BASIC, rolling 1-step)")
        for i, v in enumerate(wis):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "loso_by_season.png", dpi=140)
        plt.close(fig)

    # regime WIS + CI (skip the warm-up pre_covid n/a row — non-numeric)
    plot_rows = [r for r in regime_rows if isinstance(r.get("wis"), (int, float))]
    if plot_rows:
        regs = [r["regime"] for r in plot_rows]
        wis = [r["wis"] for r in plot_rows]
        lo = [r["wis"] - r["wis_ci95_lo"] for r in plot_rows]
        hi = [r["wis_ci95_hi"] - r["wis"] for r in plot_rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(regs, wis, color=["#1f77b4", "#d62728", "#2ca02c"][:len(regs)],
               yerr=[lo, hi], capsize=6)
        ax.set_ylabel("WIS (95% block-bootstrap CI)")
        ax.set_title("Regime-stratified WIS — NegBinGLM (rolling 1-step)")
        for i, v in enumerate(wis):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "regime_wis_ci.png", dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="SCI 외적타당도/교란 3-표")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    np.random.seed(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("SCI 외적타당도/교란 3-표 — NegBinGLM (해석가능 count 모델) 대표, BASIC, leak-free")
    log.info("=" * 70)

    X, y, dates, seasons = load_basic_matrix(Path(args.cache))
    log.info(f"in-sample: n={len(y)}  span {dates[0]} → {dates[-1]}  "
             f"seasons={sorted(set(seasons.tolist()))}")

    log.info("\n① LOSO (leave-one-season-out, rolling 1-step) ...")
    loso_rows, loso_mean = build_loso(X, y, dates, seasons)

    log.info("\n② regime-stratified (full-window rolling, re-slice) ...")
    regime_rows, _full = build_regime(X, y, dates, seasons)

    log.info("\n③ covid sensitivity (include/exclude/indicator) ...")
    covid_rows = build_covid_sensitivity(X, y, dates, seasons)

    log.info("\n   subtype positivity strata (WHO FluNet KR, read-only) ...")
    subtype_rows = build_subtype_strata(Path(args.db))

    # write tables
    _write_csv(out_dir / "loso_table.csv", loso_rows + [loso_mean])
    _write_csv(out_dir / "regime_stratified.csv", regime_rows, fieldnames=[
        "regime", "seasons", "n_weeks", "wis", "wis_ci95_lo", "wis_ci95_hi",
        "r2", "mae", "y_obs_mean", "y_obs_std", "note"])
    _write_csv(out_dir / "covid_sensitivity.csv", covid_rows)
    _write_csv(out_dir / "subtype_strata.csv", subtype_rows)
    _figures(out_dir, loso_rows, regime_rows)

    # regime shift quantification (train vs test mean of the observed series).
    train_mean = float(np.mean(y[np.array(
        [REGIME_OF_SEASON.get(int(s)) in ("pre_covid", "during_covid")
         for s in seasons])]))
    test_mean = float(np.mean(y[np.array(
        [REGIME_OF_SEASON.get(int(s)) == "post_rebound" for s in seasons])]))

    summary = {
        "model": "NegBinGLM (interpretable count model)",
        "why_this_model": ("FusedEpi refit = TiRex (hours/refit) infeasible for "
                           "7-fold LOSO + rolling; NegBinGLM = statsmodels GLM "
                           "(~sec/refit), native predict_interval, ranking.json top-3."),
        "features": "BASIC (lag1/2/4/52 + sin/cos + fourier h1-3 + season_idx = 13)",
        "eval_protocol": "leak-free rolling-origin 1-step; fit STRICT past only",
        "wis_helper": "weighted_interval_score_empirical (canonical SSOT, leak-free)",
        "retraining": {
            "loso_folds": len(loso_rows),
            "rolling_origins_full_pass": "≈ (337 - min_train) per pass",
            "passes": "1 LOSO pass/season + 1 regime pass + 3 covid passes",
            "note": "all NegBinGLM GLM refits (~sec each); champion FusedEpi NOT refit",
        },
        "sqlite_writes": 0,
        "determinism": "seed=42 (np + bootstrap RandomState); read-only DB",
        "loso_seasons": loso_rows,
        "loso_aggregate": loso_mean,
        "regime_stratified": regime_rows,
        "covid_sensitivity": covid_rows,
        "regime_shift": {
            "train_pre+during_obs_mean": round(train_mean, 4),
            "test_post_rebound_obs_mean": round(test_mean, 4),
            "shift_ratio": round(test_mean / train_mean, 3) if train_mean else None,
        },
        "subtype_strata": subtype_rows,
        "subtype_limitation": ("WHO FluNet KR is a NATIONAL sentinel with a "
                               "different denominator than the Seoul KDCA ILI "
                               "target; subtype positivity is reported as a "
                               "confounder/covariate candidate, NOT refit into "
                               "the model (avoids denominator-mismatch leakage)."),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    log.info("\n" + "=" * 70)
    log.info(f"완료. 출력 → {out_dir}")
    log.info(f"  loso_table.csv  regime_stratified.csv  covid_sensitivity.csv")
    log.info(f"  subtype_strata.csv  summary.json  figures/*.png")
    log.info(f"  regime shift: train(pre+during) mean={train_mean:.2f} vs "
             f"test(post) mean={test_mean:.2f}  ({test_mean/train_mean:.2f}×)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
