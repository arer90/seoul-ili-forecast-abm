#!/usr/bin/env python3
"""Production operational forecast — NegBinGLM full-data refit → future 1-step prediction.

Run from project root:
    python3 web/scripts/build_production_forecast.py

This script:
1. Identifies and confirms the web champion model (NegBinGLM, V6 RidgeCV+log1p,
   test R²=0.9085 from summary_metrics.csv — distinct from champion_log NegBinGLM-V7).
2. Loads full feature matrix (all rows = train+val+test in-sample) from cache.
3. Extracts BASIC feature subset (13 cols: lag + seasonal).
4. Refits NegBinGLM on all in-sample data.
5. Builds one synthetic future row for the next ISO week:
   - lag features = last observed ILI values
   - seasonality = target week's sin/cos/Fourier
   - weather/mobility = climatology (in-sample week-of-year mean) — no oracle leakage
6. Predicts (1-step), applies gate (_gate_forecast contract), computes conformal PI
   from in-sample OOF residuals (split-conformal half-width, Lei 2018).
7. Writes web/public/aggregates/ili-forecast.json and ili-forecast-models.json
   with the REAL future prediction (date > last observation).

Champion identity:
  - Web champion (rank-1 by test R²): NegBinGLM (V6 salvage = RidgeCV+log1p)
    test R²=0.9085, test RMSE=7.86, test MAE=4.90
  - champion_log.json rank-1 by WIS: NegBinGLM-V7 (true NB-GLM)
    WIS=16.17, R²=-0.41 — worse on test R² but lower WIS
  Decision: deploy NegBinGLM (V6) for web because summary_metrics shows it
  as rank-1 by test R² (0.9085 vs -0.41) and it is what the web already
  displays. Honest label is appended to the note field.

Constraints:
  - champion refit only (no Phase 13, no 53-model retrain)
  - DB access: read_only_connect (G-116/117)
  - Gate: finite ∧ nonneg ∧ ≤3×train_max ∧ |Δ|≤q99.5
  - Climatology weather only (no oracle/future-weather leakage)
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("build_production_forecast")

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
CACHE_PATH = PROJECT_ROOT / "valid_test" / "smoke" / "cache" / "feature_cache.parquet"
SUMMARY_CSV = PROJECT_ROOT / "simulation" / "results" / "csv" / "summary_metrics.csv"
PRED_DIR = PROJECT_ROOT / "simulation" / "results" / "csv"
ABM_PATH = PROJECT_ROOT / "web" / "public" / "aggregates" / "abm-scenarios.json"
ILI_LOCAL_PATH = PROJECT_ROOT / "web" / "public" / "aggregates" / "ili-local.json"
OUT_FORECAST = PROJECT_ROOT / "web" / "public" / "aggregates" / "ili-forecast.json"
OUT_MODELS = PROJECT_ROOT / "web" / "public" / "aggregates" / "ili-forecast-models.json"

# ── Web champion: resolved from the evaluation results, not hardcoded ──────────
# These were fixed constants naming NegBinGLM with metrics from before the
# log-link GLM correction, so rebuilding the dashboard reinstated a champion the
# pipeline no longer selects and quoted stale numbers alongside it. The champion
# now comes from per_model_eval's own designation column.
_CHAMPION_CSV = (PROJECT_ROOT / "simulation" / "results" / "per_model_eval"
                 / "per_model_metrics.csv")


def _resolve_web_champion() -> dict:
    """Read the champion and its test metrics out of R10's shipped table.

    Returns:
        {"name", "r2", "rmse", "mae", "version"} — `version` records where the
        designation came from so the dashboard can show its own provenance.

    Raises:
        RuntimeError: the table is missing or designates no champion. Failing
            here is deliberate: silently falling back to a hardcoded name is how
            the dashboard came to advertise a model the pipeline had dropped.

    Side effects: none — reads one CSV.
    """
    import csv as _csv

    if not _CHAMPION_CSV.exists():
        raise RuntimeError(
            f"{_CHAMPION_CSV} is absent — run R10 (per_model_eval) before "
            f"building the dashboard; the champion is not hardcoded any more."
        )
    with _CHAMPION_CSV.open(encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    def _num(v, default=float("nan")):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    picked = [r for r in rows if str(r.get("champion_best_wis", "")).strip().lower()
              in ("true", "1")]
    if not picked:
        raise RuntimeError(
            f"no row in {_CHAMPION_CSV.name} carries champion_best_wis=True"
        )
    r = picked[0]
    return {
        "name": r["model"],
        "r2": _num(r.get("r2") or r.get("test_r2")),
        "rmse": _num(r.get("rmse") or r.get("test_rmse")),
        "mae": _num(r.get("mae") or r.get("test_mae")),
        "version": f"champion_best_wis from {_CHAMPION_CSV.name} (leak-free OOF selection)",
    }


_CHAMP = _resolve_web_champion()
WEB_CHAMPION_NAME = _CHAMP["name"]
WEB_CHAMPION_TRUE_R2 = _CHAMP["r2"]
WEB_CHAMPION_RMSE = _CHAMP["rmse"]
WEB_CHAMPION_MAE = _CHAMP["mae"]
WEB_CHAMPION_VERSION = _CHAMP["version"]

# BASIC feature columns (SSOT: simulation/pipeline/baseline.py)
BASIC_FEATURE_COLS = [
    "ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag4", "ili_rate_lag52",
    "sin_month", "cos_month",
    "fourier_sin_h1", "fourier_cos_h1", "fourier_sin_h2", "fourier_cos_h2",
    "fourier_sin_h3", "fourier_cos_h3",
    "season_idx",
]

# Real-time road-traffic add-on (validated 2026-06: only external group that helps the
# Jan–Feb 2nd wave robustly across 3 winters AND never overfits — docs/
# REALTIME_FEATURE_COLLECT_AUDIT_20260609.md, web/scripts/test_nowcast_facts.py 5/5).
# Used at 1-week lag (week t uses week t-1's traffic = real-time available, no leakage).
RT_ROAD_PREFIX = "rt_road"
USE_RT_ROAD = True   # set False to revert to BASIC-only production forecast

# Per-family "current-week observed" prefixes — these get climatology substitution
_WEATHER_COLS = ("temp_avg", "temp_min", "humidity", "wind_speed", "rainfall",
                 "pressure", "sunshine", "temp_std")
_MOBILITY_PREFIXES = ("subway_", "bus_", "sub_h_", "bus_h_")
_POP_PREFIXES = ("pop_", "dong_", "hpop_")
_RT_PREFIX = "rt_"

SEOUL_GU = [
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load feature matrix from cache
# ─────────────────────────────────────────────────────────────────────────────

def _load_feature_matrix() -> tuple[np.ndarray, np.ndarray, list[str], list]:
    """Load full feature matrix from parquet cache.

    Returns:
        (X_all, y_all, feature_cols, week_starts)
        X_all: (n, p) float64 feature matrix — all in-sample rows
        y_all: (n,) float64 ILI rate target
        feature_cols: list of p feature names
        week_starts: list of n datetime objects (week start dates)

    Raises:
        FileNotFoundError: if cache parquet not found
    """
    import polars as pl

    if not CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {CACHE_PATH}\n"
            "Run the pipeline (phase 1) to generate it first."
        )
    df = pl.read_parquet(str(CACHE_PATH))
    y_col = "ili_rate" if "ili_rate" in df.columns else df.columns[0]
    feature_cols = [c for c in df.columns if c not in (y_col, "week_start")]
    X_all = df.select(feature_cols).to_numpy().astype(np.float64)
    y_all = df[y_col].to_numpy().astype(np.float64)

    week_starts: list = []
    if "week_start" in df.columns:
        week_starts = df["week_start"].to_list()

    log.info(f"Feature matrix loaded: {X_all.shape} rows×cols, "
             f"y range=[{y_all.min():.3f}, {y_all.max():.3f}]")
    return X_all, y_all, feature_cols, week_starts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Extract BASIC feature subset
# ─────────────────────────────────────────────────────────────────────────────

def _extract_basic_features(
    X_all: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    """Slice X_all to the BASIC feature columns (13 lag+seasonal).

    Returns:
        (X_basic, basic_cols_present, basic_indices_in_full)
        X_basic: (n, k) where k = number of BASIC cols found
        basic_cols_present: names of cols that were found
        basic_indices_in_full: column indices in the full feature_cols list
    """
    present = []
    indices = []
    for col in BASIC_FEATURE_COLS:
        if col in feature_cols:
            idx = feature_cols.index(col)
            present.append(col)
            indices.append(idx)
    if not present:
        raise ValueError(
            "No BASIC_FEATURE_COLS found in feature_cols. "
            f"Available sample: {feature_cols[:10]}"
        )
    X_basic = X_all[:, indices]
    log.info(f"BASIC feature subset: {len(present)}/{len(BASIC_FEATURE_COLS)} cols present: {present}")
    return X_basic, present, indices


def _rt_road_features(
    X_all: np.ndarray,
    feature_cols: list[str],
    train_n: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Real-time road-traffic columns, lagged 1 week (no leakage), NaN→train-mean.

    The forecast for week t uses week (t-1)'s road traffic, which is observed in real
    time at forecast time — so this is an honest nowcast feature, not oracle.  The
    "future" row (next week beyond the data) uses the LAST observed traffic = its 1-week
    lag, consistent with training.

    Args:
        X_all: (n, p) full feature matrix.
        feature_cols: length-p column names.
        train_n: number of rows to use for the NaN-imputation mean (in-sample).

    Returns:
        (rt_train, rt_future, rt_cols)
          rt_train: (n, k) lagged-1wk, NaN-filled — concat to BASIC for refit.
          rt_future: (1, k) last-observed, NaN-filled — concat to the synthetic future row.
          rt_cols: the k column names (empty list if no rt_road cols present).

    Side effects: none.
    """
    idx = [i for i, c in enumerate(feature_cols) if c.startswith(RT_ROAD_PREFIX)]
    if not idx:
        return np.zeros((len(X_all), 0)), np.zeros((1, 0)), []
    raw = X_all[:, idx].astype(np.float64)
    lagged = np.vstack([raw[0:1], raw[:-1]])           # week t ← week t-1 (real-time available)
    mu = np.nanmean(lagged[:train_n], axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    def _fill(M: np.ndarray) -> np.ndarray:
        M = M.copy()
        bad = np.where(~np.isfinite(M))
        M[bad] = np.take(mu, bad[1])
        return M

    rt_train = _fill(lagged)
    rt_future = _fill(raw[-1:].copy())                 # next week uses last observed (1wk lag)
    return rt_train, rt_future, [feature_cols[i] for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Refit NegBinGLM on all in-sample data
# ─────────────────────────────────────────────────────────────────────────────

def _refit_negbin_glm(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    """Refit NegBinGLM (V6 champion) on the entire in-sample dataset.

    Args:
        X_train: (n, k) BASIC feature matrix — all in-sample rows
        y_train: (n,) ILI rate target — all in-sample rows

    Returns:
        Fitted NegBinGLMForecaster instance from simulation.models.epi_models

    Side effects: logs fit stats (top-K, alpha, train R²)
    """
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))

    from simulation.models.epi_models import NegBinGLMForecaster
    model = NegBinGLMForecaster(topk=20)
    model.fit(X_train, y_train)

    # Compute and log in-sample train R²
    pred_train = model.predict(X_train)
    ss_res = float(np.sum((y_train - pred_train) ** 2))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    r2_insample = 1.0 - ss_res / max(ss_tot, 1e-9)
    log.info(f"NegBinGLM full-data refit: n={len(y_train)}, "
             f"in-sample R²={r2_insample:.4f}, y_max={y_train.max():.2f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Build synthetic future row
# ─────────────────────────────────────────────────────────────────────────────

def _iso_week_to_month(year: int, week: int) -> int:
    """Return month (1-12) for the Monday of ISO week (year, week)."""
    d = datetime.date.fromisocalendar(year, week, 1)
    return d.month


def _build_future_row(
    X_all: np.ndarray,
    y_all: np.ndarray,
    feature_cols: list[str],
    basic_cols: list[str],
    basic_indices: list[int],
    week_starts: list,
) -> tuple[np.ndarray, datetime.date, int, int]:
    """Build a synthetic feature row for the next ISO week beyond the last observation.

    Algorithm:
      1. lag features: copy last observed ILI values from y_all
      2. seasonality features (sin_month, cos_month, Fourier, season_idx):
         compute analytically for the target week
      3. weather/mobility/population/realtime = climatology (in-sample week-of-year mean)
         — no oracle leakage

    Args:
        X_all: full feature matrix (n, p)
        y_all: (n,) ILI rates
        feature_cols: full feature column names (length p)
        basic_cols: BASIC column names subset
        basic_indices: indices of basic_cols in feature_cols
        week_starts: list of datetime for each row

    Returns:
        (future_row, forecast_date, forecast_year, forecast_week)
        future_row: (1, k) array of BASIC features for next week
        forecast_date: datetime.date of next week's Monday
        forecast_year, forecast_week: ISO year, week of the forecast
    """
    n = len(y_all)

    # Determine last observation date
    if week_starts:
        last_dt = week_starts[-1]
        if hasattr(last_dt, 'date'):
            last_dt = last_dt.date()
        elif isinstance(last_dt, datetime.datetime):
            last_dt = last_dt.date()
        elif not isinstance(last_dt, datetime.date):
            last_dt = None
    else:
        last_dt = None

    if last_dt is None:
        # Fallback: compute from ILI season/week pattern
        last_dt = datetime.date(2026, 5, 24)  # known from cache inspection
        log.warning(f"week_starts empty — using fallback last_dt={last_dt}")

    # Next week = last_dt + 7 days
    forecast_date = last_dt + datetime.timedelta(weeks=1)
    fc_iso = forecast_date.isocalendar()
    fc_year, fc_week = fc_iso[0], fc_iso[1]
    log.info(f"Last observation: {last_dt} (ILI={y_all[-1]:.3f}/1k)")
    log.info(f"Forecast target: {forecast_date} = {fc_year}-W{fc_week:02d}")

    # Compute week-of-year for all in-sample rows (for climatology)
    in_sample_woy = np.zeros(n, dtype=np.int32)
    for i, ws in enumerate(week_starts[:n]):
        try:
            d = ws.date() if hasattr(ws, 'date') else ws
            if hasattr(d, 'isocalendar'):
                in_sample_woy[i] = d.isocalendar()[1]
            else:
                in_sample_woy[i] = 1
        except Exception:
            in_sample_woy[i] = 1

    # Build future row for BASIC features
    future_row = np.zeros((1, len(basic_cols)), dtype=np.float64)
    fc_month = forecast_date.month

    # Compute season_idx for future week (same logic as builder.py)
    # season_start = year of the September that started this flu season
    # Sep-Dec → season_start = fc_year, Jan-Aug → season_start = fc_year - 1
    if fc_month >= 9:
        fc_season_start = fc_year
    else:
        fc_season_start = fc_year - 1

    # Estimate season_norm: look up what season_idx values exist in data
    # and extrapolate by adding 1 per season year beyond the last seen
    season_idx_col_i = basic_cols.index("season_idx") if "season_idx" in basic_cols else -1

    for j, col in enumerate(basic_cols):
        if col == "ili_rate_lag1":
            future_row[0, j] = float(y_all[-1])
        elif col == "ili_rate_lag2":
            future_row[0, j] = float(y_all[-2]) if n >= 2 else float(y_all[-1])
        elif col == "ili_rate_lag4":
            future_row[0, j] = float(y_all[-4]) if n >= 4 else float(y_all[-1])
        elif col == "ili_rate_lag52":
            future_row[0, j] = float(y_all[-52]) if n >= 52 else float(y_all[-1])
        elif col == "sin_month":
            future_row[0, j] = math.sin(2 * math.pi * fc_month / 12.0)
        elif col == "cos_month":
            future_row[0, j] = math.cos(2 * math.pi * fc_month / 12.0)
        elif col.startswith("fourier_sin_h"):
            h = int(col[-1])  # harmonic index 1,2,3
            doy = forecast_date.timetuple().tm_yday
            future_row[0, j] = math.sin(2 * math.pi * h * doy / 365.0)
        elif col.startswith("fourier_cos_h"):
            h = int(col[-1])
            doy = forecast_date.timetuple().tm_yday
            future_row[0, j] = math.cos(2 * math.pi * h * doy / 365.0)
        elif col == "season_idx":
            # Use the last in-sample season_idx col value + 1 if we crossed a season boundary
            col_full_idx = basic_indices[j]
            last_season_idx = float(X_all[-1, col_full_idx])
            # If we're still in the same flu season as the last row, keep same
            # If not, increment by 1
            last_row_month = last_dt.month if last_dt else 5
            if fc_month >= 9 and last_row_month < 9:
                # crossed a season boundary (last row was pre-Sep, future is post-Sep)
                future_row[0, j] = last_season_idx + 1.0
            else:
                future_row[0, j] = last_season_idx
        else:
            # Fallback: climatology by week-of-year
            col_full_idx = basic_indices[j]
            mask = (in_sample_woy == fc_week)
            if mask.any():
                future_row[0, j] = float(np.nanmean(X_all[mask, col_full_idx]))
            else:
                # use adjacent weeks ±1
                for dw in [1, -1, 2, -2, 3]:
                    w2 = fc_week + dw
                    if 1 <= w2 <= 53:
                        mask2 = (in_sample_woy == w2)
                        if mask2.any():
                            future_row[0, j] = float(np.nanmean(X_all[mask2, col_full_idx]))
                            break
                # else stays 0

    log.info(f"Synthetic future row built: lag1={future_row[0, 0]:.3f}, "
             f"sin_month={future_row[0, basic_cols.index('sin_month')]:.3f}, "
             f"fourier_sin_h1={future_row[0, basic_cols.index('fourier_sin_h1')]:.3f}")

    return future_row, forecast_date, fc_year, fc_week


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Gate (mirrors real_eval._gate_forecast)
# ─────────────────────────────────────────────────────────────────────────────

def _gate_forecast(
    pred: np.ndarray,
    y_train: np.ndarray,
    fallback: float | None = None,
    k: float = 3.0,
) -> dict:
    """Hard reject-and-replace contract gate (mirrors real_eval._gate_forecast).

    Args:
        pred: (h,) candidate forecast values
        y_train: in-sample history for computing caps
        fallback: replacement value when gate trips. If None, pred is returned
            with replaced=False.
        k: train-max multiplier (3.0 = lenient)

    Returns:
        {pred, replaced, n_violations, reason}
    """
    out = {"pred": np.asarray(pred, dtype=np.float64),
           "replaced": False, "n_violations": 0, "reason": "ok"}
    try:
        p = np.asarray(pred, dtype=np.float64)
        yt = np.asarray(y_train, dtype=np.float64)
        yt = yt[np.isfinite(yt)]
        if yt.size == 0:
            return out
        cap = k * float(np.max(yt))
        dcap = (float(np.quantile(np.abs(np.diff(yt)), 0.995))
                if yt.size > 2 else float("inf"))
        viol: list[str] = []
        if not np.all(np.isfinite(p)):
            viol.append("non-finite")
        if np.any(p < 0):
            viol.append("negative")
        if np.any(p > cap):
            viol.append(f"exceeds {k:g}×train_max ({float(np.nanmax(p)):.1f}>{cap:.1f})")
        if p.size > 1 and float(np.nanmax(np.abs(np.diff(p)))) > dcap:
            viol.append(f"|Δ|>{dcap:.1f} (max {float(np.nanmax(np.abs(np.diff(p)))):.1f})")
        out["n_violations"] = len(viol)
        out["reason"] = "; ".join(viol) if viol else "ok"
        if viol and fallback is not None:
            out["pred"] = np.array([float(fallback)], dtype=np.float64)
            out["replaced"] = True
    except Exception as e:
        out["reason"] = f"gate-error: {type(e).__name__}: {e}"
    return out


def _surge_aware_bound(
    raw: float,
    recent_real: np.ndarray,
    train_max: float,
    k_level: float = 3.0,
) -> tuple[float, bool, str]:
    """Trajectory-relative bound: clamp single-week false explosions, allow real surges.

    Evidence (web/scripts/pandemic_vs_explosion_sweep.py): alpha tuning does NOT fix the
    log1p/expm1 explosion — at EVERY alpha the seasonal peak forecast blows up 3-4× and a
    synthetic surge extrapolates to millions.  The structural fix is therefore NOT a higher
    alpha nor a hard 2×train_max cap (which would also kill a genuine pandemic), but a bound
    relative to the RECENT TRAJECTORY:

      • plateau/decline + model jumps high  → single-week amplification (false) → tight clamp.
      • sustained 3-week rise (≥50%)         → plausible real surge → relaxed bound AND
        surge_detected=True so the caller DEFERS to the mechanistic SEIR/ABM engine (the ML
        lag model cannot reliably extrapolate beyond its seasonal training range).

    Args:
        raw: raw 1-step model forecast (ILI per 1k).
        recent_real: last ≥4 observed ILI values, oldest→newest.
        train_max: max ILI in training (absolute backstop only).
        k_level: absolute backstop = k_level × train_max.

    Returns:
        (gated, surge_detected, reason).  surge_detected=True ⇒ caller should route to the
        mechanistic engine and raise a surveillance alert.

    Role (docs/PANDEMIC_MODE_DESIGN_20260610.md): this is a **2차 안전망**, NOT the primary
    pandemic trigger.  The 1st-tier trigger is the external-news mode gate (resolve_mode):
    KDCA 위기경보 경계↑ = hard PANDEMIC, DON novel / news spike = WATCH.  surge_detected here
    only ORs into WATCH as a backup for missed news — because trajectory cannot distinguish a
    normal endemic winter surge from a true pandemic onset.  Keep-light; do not strengthen to a
    hard 2×cap (that would kill genuine surges and is alpha-invariantly unable to fix the
    in-range false explosion anyway — pandemic_vs_explosion_sweep.py).

    Side effects: none.
    """
    r = np.asarray(recent_real, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float(max(0.0, raw)), False, "no trajectory"
    last = float(r[-1])
    rising = r.size >= 4 and all(r[-i] > r[-i - 1] for i in range(1, 4))
    growth = last / max(float(r[-4]), 1e-6) if r.size >= 4 else 1.0
    sustained = bool(rising and growth >= 1.5)        # ≥50% rise sustained over 3 weeks
    if sustained:
        bound = max(last * 3.0, last + 30.0)
        surge, reason = True, f"sustained surge ({growth:.1f}× / 3wk) — DEFER to mechanistic engine + alert"
    else:
        bound = last * 1.5 + 5.0                       # reject 2.5× single-week jump (201→~127)
        surge, reason = False, "trajectory-bounded (false-explosion guard)"
    bound = min(bound, k_level * float(train_max))
    return float(np.clip(raw, 0.0, bound)), surge, reason


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Conformal PI (in-sample OOF residuals, Lei 2018 split-conformal)
# ─────────────────────────────────────────────────────────────────────────────

def _conformal_half_width(
    alpha: float = 0.05,
    pred_csv: Path | None = None,
    pred_level: float | None = None,
    regime_thr: float = 20.0,
) -> float:
    """Compute split-conformal PI half-width from true out-of-sample test residuals.

    pred_level (regime-aware, 2026-06): 주어지면 보정 잔차를 예측수준 regime(저<thr/고≥thr)으로
    나눠 forecast 와 **같은 regime 잔차만**으로 q 산출 — 같은 coverage 에 더 좁은 구간(피크 분산↑
    반영). TDD 증명(test_pi_calibration: per-regime 폭 63→53, cov 동일). 해당 regime 표본<8 이면
    전체 fallback.

    Uses test-set residuals from predictions_NegBinGLM.csv (split="test") as the
    calibration set. These are the only true out-of-sample residuals: the model was
    fitted on train+val and evaluated on the held-out test slab.

    The full-data refit produces in-sample residuals (≈0 on training rows) which
    would underestimate uncertainty. The test-set residuals (n=68) are the correct
    calibration set for split-conformal coverage.

    Formula: Lei 2018 / Vovk 2005 split-conformal ceiling:
        q_hat = |residual|_{⌈(n+1)(1-alpha)⌉ - 1}  (0-indexed, sorted ascending)

    Args:
        alpha: miscoverage level (0.05 = 95% PI)
        pred_csv: path to predictions_NegBinGLM.csv. Defaults to
            simulation/results/csv/predictions_NegBinGLM.csv.

    Returns:
        conformal half-width q_hat (rate-scale, ≥0)

    Raises:
        Nothing — falls back to test_rmse-based estimate if CSV unavailable.

    Side effects: none
    """
    import csv as _csv

    if pred_csv is None:
        pred_csv = PRED_DIR / "predictions_NegBinGLM.csv"

    pairs = []   # (|residual|, y_pred) — y_pred 로 regime 분리
    if pred_csv.is_file():
        try:
            with pred_csv.open(newline="", encoding="utf-8") as fh:
                for row in _csv.DictReader(fh):
                    if row.get("split") == "test" and row.get("y_true") and row.get("y_pred"):
                        yp = float(row["y_pred"])
                        res = abs(float(row["y_true"]) - yp)
                        if np.isfinite(res):
                            pairs.append((res, yp))
        except Exception as exc:
            log.warning(f"Could not read predictions CSV: {exc}")

    # regime-aware: forecast 와 같은 수준대(저/고) 잔차만 (표본<8 이면 전체)
    residuals = [r for r, _ in pairs]
    if pred_level is not None and pairs:
        hi = pred_level >= regime_thr
        same = [r for r, yp in pairs if (yp >= regime_thr) == hi]
        if len(same) >= 8:
            residuals = same

    if not residuals:
        # Fallback: use known test RMSE as approximate PI sigma
        log.warning(
            f"No test residuals from {pred_csv} — fallback to "
            f"1.96 × test_RMSE={WEB_CHAMPION_RMSE}"
        )
        return 1.96 * WEB_CHAMPION_RMSE

    arr = np.array(residuals, dtype=np.float64)
    n_res = len(arr)
    sorted_res = np.sort(arr)
    # Lei 2018 / Vovk 2005 split-conformal ceiling:
    k_idx = int(np.ceil((n_res + 1) * (1.0 - alpha))) - 1
    k_idx = max(0, min(k_idx, n_res - 1))
    q_hat = float(sorted_res[k_idx])
    log.info(
        f"Conformal PI 95%: n_test_residuals={n_res}, "
        f"q_hat={q_hat:.3f} (rate-scale half-width, test-slab OOS calibration)"
    )
    return q_hat


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Load ABM weights
# ─────────────────────────────────────────────────────────────────────────────

def _gu_source_summary() -> dict:
    """gu 분배의 선택 단계·신뢰·품질 요약(web 표면화용). 없으면 시뮬 default."""
    gw = ABM_PATH.parent / "gu-weights.json"
    if gw.is_file():
        try:
            d = json.loads(gw.read_text("utf-8"))
            t2 = next((x for x in d.get("ladder", []) if x.get("tier") == 2), {})
            return {"selected_tier": d.get("selected_tier"), "source": d.get("selected_source"),
                    "confidence": d.get("confidence"), "quality": t2.get("quality"),
                    "note": d.get("note", "")}
        except Exception:
            pass
    return {"selected_tier": 3, "source": "abm_sim", "confidence": "낮음(시뮬)", "quality": None}


def _load_abm_weights() -> dict[str, float]:
    """자치구 분배 가중 — gu-weights.json(품질-게이트 계단식: 실측ILI→endemic→ABM→균등) 우선.

    사용자 아이디어(2026-06): 진짜 구별 독감 부재 → 인플루엔자와 상관 높은 자치구별 감염병(COVID-19,
    +0.41) 실데이터 패턴을 1차 참고. build_gu_weights.py 가 gu-weights.json 생성. 미존재 시 종전
    ABM I_frac 단독으로 fallback(back-compat).
    """
    gw = ABM_PATH.parent / "gu-weights.json"
    if gw.is_file():
        try:
            d = json.loads(gw.read_text("utf-8"))
            w = d.get("weights") or d.get("weights_blend") or {}   # quality-gated ladder | (legacy blend)
            if w:
                log.info(f"gu 분배: {d.get('selected_tier','?')}차 {d.get('selected_source','')} "
                         f"· 신뢰 {d.get('confidence','?')}")
                return {g: float(x) for g, x in w.items()}
        except Exception as exc:
            log.warning(f"gu-weights.json load error: {exc} — ABM fallback")
    if not ABM_PATH.is_file():
        return {}
    try:
        abm = json.loads(ABM_PATH.read_text("utf-8"))
        gu_names: list[str] = abm.get("gu_names", [])
        i_frac: list[list[float]] = (
            abm.get("scenarios", {}).get("baseline", {}).get("I_frac", [])
        )
        if gu_names and i_frac:
            last_day = i_frac[-1]
            total = sum(last_day)
            if total > 0 and len(last_day) == len(gu_names):
                return {
                    gu: frac / total * len(gu_names)
                    for gu, frac in zip(gu_names, last_day)
                }
    except Exception as exc:
        log.warning(f"ABM weight load error: {exc}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Write ili-forecast.json (champion 1-step production forecast)
# ─────────────────────────────────────────────────────────────────────────────

def _write_ili_forecast(
    city_forecast: float,
    city_lo: float,
    city_hi: float,
    forecast_date: datetime.date,
    last_obs_date: datetime.date,
    gate_result: dict,
    conformal_q: float,
    abm_weights: dict[str, float],
    surge_detected: bool = False,
    surge_reason: str = "",
) -> None:
    """Write ili-forecast.json with the production future 1-step forecast.

    Args:
        city_forecast: gated point forecast (ILI per 1k)
        city_lo: 95% PI lower bound
        city_hi: 95% PI upper bound
        forecast_date: date of the forecast week (first day of target week)
        last_obs_date: last observed ILI date
        gate_result: dict from _gate_forecast
        conformal_q: conformal half-width used for PI
        abm_weights: per-gu weight dict from ABM I_frac

    Side effects: writes OUT_FORECAST to disk
    """
    generated_at = datetime.datetime.utcnow().isoformat() + "Z"
    forecast_at = forecast_date.isoformat() + "T00:00:00Z"
    observed_at = last_obs_date.isoformat() + "T00:00:00Z"

    gate_ok = gate_result["reason"] == "ok"
    note = (
        f"production 운영 forecast — 전체 refit→미래 1-step, climatology, gated. "
        f"Champion: {WEB_CHAMPION_NAME} ({WEB_CHAMPION_VERSION}). "
        f"Test R²={WEB_CHAMPION_TRUE_R2:.4f}, RMSE={WEB_CHAMPION_RMSE:.2f}, "
        f"MAE={WEB_CHAMPION_MAE:.2f}. "
        f"Gate: {gate_result['reason']}. "
        f"Conformal PI 95% half-width: {conformal_q:.3f}/1k (test-slab OOS 잔차 n=68, Lei 2018). "
        f"Weather/mobility: climatology (주차 평균, in-sample). "
        f"{'⚠ gate triggered — fallback used' if not gate_ok else 'gate passed'}"
    )

    gu_dict: dict[str, dict] = {}
    for gu in SEOUL_GU:
        w = abm_weights.get(gu, 1.0)
        gu_dict[gu] = {
            "ili":  round(city_forecast * w, 4),
            "lo":   round(max(0.0, city_lo * w), 4),
            "hi":   round(city_hi * w, 4),
        }

    payload = {
        "generated_at": generated_at,
        "observed_at": observed_at,
        "forecast_at": forecast_at,
        "source": "production-refit-forecast",
        "model": WEB_CHAMPION_NAME,
        "model_version": WEB_CHAMPION_VERSION,
        "horizon_weeks": 1,
        "city_forecast": round(city_forecast, 4),
        "city_lo": round(max(0.0, city_lo), 4),
        "city_hi": round(city_hi, 4),
        "conformal_q95": round(conformal_q, 4),
        "gate": {
            "passed": gate_ok,
            "n_violations": gate_result["n_violations"],
            "reason": gate_result["reason"],
            "replaced": gate_result["replaced"],
        },
        "surge": {
            "detected": bool(surge_detected),
            "reason": surge_reason,
            "action": ("DEFER to mechanistic SEIR/ABM engine — ML lag model cannot extrapolate "
                       "a large surge (pandemic_vs_explosion_sweep)" if surge_detected else "none"),
        },
        "metrics": {
            "test_r2": WEB_CHAMPION_TRUE_R2,
            "test_rmse": WEB_CHAMPION_RMSE,
            "test_mae": WEB_CHAMPION_MAE,
        },
        "note": note,
        "gu_source": _gu_source_summary(),
        "gu": gu_dict,
    }

    OUT_FORECAST.parent.mkdir(parents=True, exist_ok=True)
    OUT_FORECAST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        f"Wrote {OUT_FORECAST}: city_forecast={city_forecast:.4f}/1k, "
        f"PI=[{city_lo:.3f}, {city_hi:.3f}], forecast_at={forecast_at}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Write ili-forecast-models.json (multi-model, but with REAL champion forecast)
# ─────────────────────────────────────────────────────────────────────────────

def _write_ili_forecast_models(
    city_forecast: float,
    city_lo: float,
    city_hi: float,
    forecast_date: datetime.date,
    observed_at: str,
    conformal_q: float,
    abm_weights: dict[str, float],
) -> None:
    """Write ili-forecast-models.json with champion model production forecast.

    The champion's entry uses the REAL future forecast. Other top models from
    summary_metrics.csv retain the legacy persistence-anchored approach
    (they haven't been refitted) but are clearly labelled as such.

    Args:
        city_forecast: gated production forecast (champion, ILI per 1k)
        city_lo: 95% PI lower bound (conformal)
        city_hi: 95% PI upper bound (conformal)
        forecast_date: ISO date of forecast target week
        observed_at: ISO datetime string of last observation
        conformal_q: conformal PI half-width for the champion
        abm_weights: per-gu weight dict from ABM I_frac

    Side effects: writes OUT_MODELS to disk
    """
    import csv as _csv

    generated_at = datetime.datetime.utcnow().isoformat() + "Z"
    forecast_at = forecast_date.isoformat() + "T00:00:00Z"

    # Load all models from summary_metrics.csv for multi-model panel
    models_out: list[dict] = []

    if SUMMARY_CSV.is_file():
        with SUMMARY_CSV.open(newline="", encoding="utf-8") as fh:
            all_rows = list(_csv.DictReader(fh))
        # Rank by test_r2 (higher = better), take top 12
        ranked = sorted(
            all_rows,
            key=lambda r: float(r.get("test_r2") or "-inf"),
            reverse=True,
        )[:12]
    else:
        ranked = []

    for rank_i, row in enumerate(ranked):
        name = row["name"]
        category = row.get("category", "unknown")
        test_r2 = float(row.get("test_r2") or "nan")
        test_rmse = float(row.get("test_rmse") or "0")
        test_mae = float(row.get("test_mae") or "0")
        test_mape = float(row.get("test_mape") or "0")

        is_champion = (name == WEB_CHAMPION_NAME)

        if is_champion:
            # Champion gets the REAL production refit forecast
            this_city = city_forecast
            this_lo = max(0.0, city_lo)
            this_hi = city_hi
            this_rel_rmse = conformal_q / max(city_forecast, 1.0)
            forecast_source = "production-refit"
            pi_method = "conformal-oof-residuals"
        else:
            # Other models: load last test prediction for rel_rmse estimation,
            # but anchor forecast level to the same city_forecast as champion
            # (these haven't been refitted; only the champion has)
            rel_rmse = 0.25  # fallback
            pred_path = PRED_DIR / f"predictions_{name}.csv"
            if pred_path.is_file():
                try:
                    with pred_path.open(newline="", encoding="utf-8") as pfh:
                        pred_rows = list(_csv.DictReader(pfh))
                    test_preds = [
                        float(r["y_pred"])
                        for r in pred_rows
                        if r.get("split") == "test" and r.get("y_pred")
                    ]
                    test_mean = sum(test_preds) / len(test_preds) if test_preds else 0.0
                    if test_mean > 0 and test_rmse > 0:
                        rel_rmse = min(0.5, max(0.05, test_rmse / test_mean))
                except Exception as exc:
                    log.warning(f"  pred parse error for {name}: {exc}")
            this_city = round(city_forecast, 4)  # same level as champion refit
            this_lo = max(0.0, round(city_forecast * (1 - 2 * rel_rmse), 4))
            this_hi = round(city_forecast * (1 + 2 * rel_rmse), 4)
            this_rel_rmse = rel_rmse
            forecast_source = "champion-level-anchored"
            pi_method = "rel-rmse-approximation"

        gu_dict: dict[str, dict] = {}
        for gu in SEOUL_GU:
            w = abm_weights.get(gu, 1.0)
            gu_dict[gu] = {
                "ili": round(this_city * w, 4),
                "lo":  round(max(0.0, this_lo * w), 4),
                "hi":  round(this_hi * w, 4),
            }

        models_out.append({
            "name": name,
            "category": category,
            "rank": rank_i + 1,
            "is_champion": is_champion,
            "forecast_source": forecast_source,
            "pi_method": pi_method,
            "metrics": {
                "test_r2":   round(test_r2, 4),
                "test_rmse": round(test_rmse, 4),
                "test_mae":  round(test_mae, 4),
                "test_mape": round(test_mape, 2),
            },
            "city_forecast": round(this_city, 4),
            "city_lo": round(this_lo, 4),
            "city_hi": round(this_hi, 4),
            "gu": gu_dict,
        })
        log.info(
            f"  [{rank_i+1:2d}] {name:30s}  r2={test_r2:.4f}  "
            f"city=[{this_lo:.2f}, {this_city:.2f}, {this_hi:.2f}]  "
            f"src={forecast_source}"
        )

    payload = {
        "generated_at": generated_at,
        "observed_at": observed_at,
        "forecast_at": forecast_at,
        "source": "multi-model-production-forecast",
        "horizon_weeks": 1,
        "champion": WEB_CHAMPION_NAME,
        "champion_version": WEB_CHAMPION_VERSION,
        "champion_forecast": round(city_forecast, 4),
        "conformal_q95": round(conformal_q, 4),
        "note": (
            f"Champion ({WEB_CHAMPION_NAME}) = 전체 in-sample refit → 미래 1-step 예측 "
            f"(test R²={WEB_CHAMPION_TRUE_R2:.4f}). "
            f"Conformal PI: test-slab OOS 잔차 (Lei 2018, n=68, split-conformal). "
            f"기상/이동: climatology (주차 평균, oracle 없음). "
            f"나머지 모델 = champion 레벨 anchored + 상대-RMSE PI (재학습 미완). "
            f"Gate: finite ∧ nonneg ∧ ≤3×train_max ∧ |Δ|≤q99.5."
        ),
        "models": models_out,
    }

    OUT_MODELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        f"Wrote {OUT_MODELS}: {len(models_out)} models, "
        f"champion={WEB_CHAMPION_NAME}, city_forecast={city_forecast:.4f}/1k"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """Run the full production refit pipeline.

    Returns:
        0 on success, 1 on error.

    Side effects:
        Writes ili-forecast.json and ili-forecast-models.json.
    """
    log.info("=== Production NegBinGLM refit → future 1-step forecast ===")
    log.info(f"Champion: {WEB_CHAMPION_NAME} ({WEB_CHAMPION_VERSION})")
    log.info(f"  test R²={WEB_CHAMPION_TRUE_R2}, RMSE={WEB_CHAMPION_RMSE}, MAE={WEB_CHAMPION_MAE}")

    try:
        # Step 1: Load feature matrix
        X_all, y_all, feature_cols, week_starts = _load_feature_matrix()

        # Step 2: Extract BASIC features (+ validated Rt 실시간 도로교통 add-on, 1주 lag)
        X_basic, basic_cols, basic_indices = _extract_basic_features(X_all, feature_cols)
        rt_train, rt_future, rt_cols = (
            _rt_road_features(X_all, feature_cols, len(y_all)) if USE_RT_ROAD
            else (np.zeros((len(X_all), 0)), np.zeros((1, 0)), []))
        X_feat = np.hstack([X_basic, rt_train]) if rt_cols else X_basic
        log.info(f"Feature set: BASIC {X_basic.shape[1]} + Rt도로 {len(rt_cols)} (1주 lag) "
                 f"= {X_feat.shape[1]} cols")

        # Step 3: Refit NegBinGLM on all in-sample data (no train/test split)
        log.info(f"Refitting {WEB_CHAMPION_NAME} on ALL {len(y_all)} in-sample rows...")
        model = _refit_negbin_glm(X_feat, y_all)

        # Step 4: Build synthetic future row (BASIC) + append last-observed Rt도로 (1주 lag)
        future_basic, forecast_date, fc_year, fc_week = _build_future_row(
            X_all, y_all, feature_cols, basic_cols, basic_indices, week_starts
        )
        future_row = np.hstack([future_basic, rt_future]) if rt_cols else future_basic

        # Determine last obs date for observed_at
        if week_starts:
            last_dt = week_starts[-1]
            if hasattr(last_dt, 'date'):
                last_dt = last_dt.date()
            elif isinstance(last_dt, datetime.datetime):
                last_dt = last_dt.date()
        else:
            last_dt = datetime.date(2026, 5, 24)

        # Step 5: Predict raw
        raw_pred = model.predict(future_row)
        raw_city = float(raw_pred[0])
        log.info(f"Raw NegBinGLM 1-step prediction: {raw_city:.4f}/1k")

        # Step 6a: Gate — fallback = last observed ILI (persistence)
        persistence_fallback = float(y_all[-1])
        gate_result = _gate_forecast(raw_pred, y_all, fallback=persistence_fallback, k=3.0)
        city_forecast = float(gate_result["pred"][0])
        if gate_result["replaced"]:
            log.warning(
                f"Gate TRIPPED — raw={raw_city:.4f} violates: {gate_result['reason']}. "
                f"Replaced with persistence fallback={persistence_fallback:.4f}/1k"
            )
        else:
            log.info(f"Gate passed: city_forecast={city_forecast:.4f}/1k")

        # Step 6a': surge-aware trajectory bound — clamp log1p/expm1 false explosions
        # (alpha tuning does NOT fix them, pandemic_vs_explosion_sweep.py) WITHOUT killing a
        # real surge; surge_detected ⇒ defer to mechanistic SEIR/ABM engine + alert.
        city_forecast, surge_detected, surge_reason = _surge_aware_bound(
            city_forecast, y_all[-4:], float(np.nanmax(y_all)))
        log.info(f"Surge-aware bound: {city_forecast:.4f}/1k · surge_detected={surge_detected} "
                 f"({surge_reason})")

        # Step 6a'': 잔차 bias 보정 — 최근 6주 1-step 잔차 평균 차감 (codex/gemini 레버, TDD:
        # 상승기 MAE 9.5→7.8). 모델이 못 잡는 계통오차(상승기 과소/하강기 과대)를 데이터로 정정.
        recent_bias = 0.0
        try:
            from accuracy_calibration import recent_onestep_bias  # inline: circular import 회피
            recent_bias = recent_onestep_bias(X_basic, y_all, len(y_all) - 1, k=6)
            debiased = max(0.0, city_forecast - recent_bias)
            log.info(f"Bias correction: recent +1wk bias={recent_bias:+.3f}/1k → "
                     f"{city_forecast:.3f}→{debiased:.3f}")
            city_forecast = debiased
        except Exception as e:
            log.warning(f"Bias correction skipped: {type(e).__name__}: {e}")

        # Step 6b: Conformal PI from test-set OOS residuals — regime-aware(저/고) per pi_calibration
        conformal_q = _conformal_half_width(alpha=0.05, pred_level=city_forecast)
        city_lo = max(0.0, city_forecast - conformal_q)
        city_hi = city_forecast + conformal_q

        log.info(
            f"Final production forecast: {city_forecast:.4f}/1k "
            f"95% PI=[{city_lo:.4f}, {city_hi:.4f}] "
            f"(conformal q={conformal_q:.4f})"
        )
        log.info(f"Forecast week: {forecast_date} = {fc_year}-W{fc_week:02d}")
        log.info(f"Last observation: {last_dt} ILI={y_all[-1]:.4f}/1k")

        # Verify future date constraint
        if forecast_date <= last_dt:
            log.error(
                f"CRITICAL: forecast_date={forecast_date} ≤ last_obs={last_dt}. "
                "This is not a future forecast. Aborting."
            )
            return 1
        log.info(f"Future constraint verified: {forecast_date} > {last_dt}")

        # Step 7: Load ABM weights
        abm_weights = _load_abm_weights()

        # Step 8: Write ili-forecast.json
        _write_ili_forecast(
            city_forecast=city_forecast,
            city_lo=city_lo,
            city_hi=city_hi,
            forecast_date=forecast_date,
            last_obs_date=last_dt,
            gate_result=gate_result,
            conformal_q=conformal_q,
            abm_weights=abm_weights,
            surge_detected=surge_detected,
            surge_reason=surge_reason,
        )

        # Step 9: Write ili-forecast-models.json
        observed_at = last_dt.isoformat() + "T00:00:00Z"
        _write_ili_forecast_models(
            city_forecast=city_forecast,
            city_lo=city_lo,
            city_hi=city_hi,
            forecast_date=forecast_date,
            observed_at=observed_at,
            conformal_q=conformal_q,
            abm_weights=abm_weights,
        )

        log.info("=== Production forecast complete ===")
        log.info(f"Champion: {WEB_CHAMPION_NAME}")
        log.info(f"  Version: {WEB_CHAMPION_VERSION}")
        log.info(f"  Test R²: {WEB_CHAMPION_TRUE_R2} | RMSE: {WEB_CHAMPION_RMSE} | MAE: {WEB_CHAMPION_MAE}")
        log.info(f"  champion_log best-WIS: NegBinGLM-V7 (WIS=16.17, R²=-0.41) — different model")
        log.info(f"Forecast: {city_forecast:.4f}/1k for {forecast_date} ({fc_year}-W{fc_week:02d})")
        log.info(f"PI 95%: [{city_lo:.4f}, {city_hi:.4f}]  q={conformal_q:.4f}")
        log.info(f"Gate: {gate_result['reason']} (replaced={gate_result['replaced']})")

        return 0

    except Exception as exc:
        log.error(f"Production forecast failed: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
