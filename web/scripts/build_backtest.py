#!/usr/bin/env python3
"""Build backtest validation JSON files for the web dashboard.

Generates three files under web/public/aggregates/:
  A  backtest.json       — ML model predictions vs actuals + accuracy metrics
  B  seir-hindcast.json  — SEIR 360-day hindcast vs observed ILI
  C  regime-shifts.json  — Change-point / regime-shift detection on ILI history

Run from project root:
    python3 web/scripts/build_backtest.py

DB access: read-only via sqlite3 (no write, no safe_connect dependency).
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH   = PROJECT_ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
CSV_DIR   = PROJECT_ROOT / "simulation" / "results" / "csv"
AGG_DIR   = PROJECT_ROOT / "web" / "public" / "aggregates"

OUT_BACKTEST   = AGG_DIR / "backtest.json"
OUT_HINDCAST   = AGG_DIR / "seir-hindcast.json"
OUT_REGIME     = AGG_DIR / "regime-shifts.json"

# ── Constants ─────────────────────────────────────────────────────────────────
SEOUL_POP    = 9_720_000
# SEIR parameters (same calibration as _build_multi_model_forecast.py)
R0_MEAN      = 1.4
R0_AMPLITUDE = 0.286          # seasonal amplitude
INCUB_DAYS   = 2.0
INFECT_DAYS  = 3.5
VAX_COV      = 0.45
DETECT_FACTOR = 7.5
WANING_RATE  = 1.0 / 365.0
PEAK_DOY     = 15             # mid-January Korea flu peak

TOP_N_MODELS = 10             # top models by test_r2 in backtest.json

# Threshold for anomaly z-score flagging (regime detector)
ANOMALY_Z    = 2.0
# Rolling mean window for regime baseline (weeks)
BASELINE_WINDOW = 8
# Minimum jump to call a change-point (Δ ILI absolute)
MIN_JUMP     = 3.0


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _db_read_only():
    """Open the epi_real_seoul DB read-only via the project helper.

    Returns:
        A read-only DB connection (read_only_connect applies ro/immutable).

    Raises:
        RuntimeError: if DB file does not exist.
    """
    if not DB_PATH.is_file():
        raise RuntimeError(f"DB not found: {DB_PATH}")
    # Single-helper DB policy (G-116/G-117): use the project read-only helper
    # instead of opening the driver directly — it applies uri=ro/immutable inside.
    from simulation.database import read_only_connect
    return read_only_connect(str(DB_PATH))


def _load_ili_series() -> list[dict]:
    """Load full chronological ILI time series from sentinel_influenza.

    Aggregates across all age groups (AVG) to get a single weekly city-level
    ILI rate per 1 000 outpatient visits.  Adds approximate ISO-week dates.

    Returns:
        List of dicts: {season, week_seq, week_num, date (datetime.date), ili}.
        Ordered chronologically (season_start ASC, week_seq ASC).

    Side effects: one DB query.
    """
    conn = _db_read_only()
    c = conn.cursor()
    rows = c.execute(
        """
        SELECT season_start, week_seq, week_label, AVG(ili_rate) AS avg_ili
        FROM sentinel_influenza
        GROUP BY season_start, week_seq, week_label
        ORDER BY season_start, week_seq
        """
    ).fetchall()
    conn.close()

    series: list[dict] = []
    for s, wseq, wlabel, ili in rows:
        m = re.search(r"(\d+)주", wlabel)
        wnum = int(m.group(1)) if m else wseq
        year = s if wnum >= 36 else s + 1
        try:
            date = datetime.date.fromisocalendar(year, wnum, 1)
        except ValueError:
            date = datetime.date(year, 1, 1) + datetime.timedelta(weeks=wnum - 1)
        series.append({
            "season": s,
            "week_seq": wseq,
            "week_num": wnum,
            "date": date,
            "ili": ili,
        })
    return series


def _rmse(a: list[float], b: list[float]) -> float:
    """Root mean squared error between two equal-length lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / n)


def _mae(a: list[float], b: list[float]) -> float:
    """Mean absolute error."""
    n = len(a)
    if n == 0:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(a, b)) / n


def _mape(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute percentage error (ignores zero-actual points)."""
    pairs = [(a, p) for a, p in zip(actual, predicted) if abs(a) > 1e-9]
    if not pairs:
        return float("nan")
    return 100.0 * sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs)


def _r2(actual: list[float], predicted: list[float]) -> float:
    """Coefficient of determination R²."""
    n = len(actual)
    if n < 2:
        return float("nan")
    mean_a = sum(actual) / n
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ══════════════════════════════════════════════════════════════════════════════
# A — backtest.json: ML predictions vs actuals
# ══════════════════════════════════════════════════════════════════════════════

def _load_summary_metrics() -> list[dict]:
    """Load summary_metrics.csv and return rows sorted by test_r2 descending."""
    path = CSV_DIR / "summary_metrics.csv"
    if not path.is_file():
        print(f"  ! summary_metrics.csv not found: {path}", file=sys.stderr)
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return sorted(
        rows,
        key=lambda r: float(r.get("test_r2") or "-inf"),
        reverse=True,
    )


def _load_predictions(name: str) -> dict[str, list[dict]]:
    """Load predictions_<name>.csv split into val/test point lists.

    Args:
        name: model name, used to form filename predictions_<name>.csv.

    Returns:
        Dict with keys 'val' and 'test', each a list of
        {i: int, actual: float, predicted: float}.

    Side effects: reads one CSV file.
    """
    path = CSV_DIR / f"predictions_{name}.csv"
    if not path.is_file():
        return {"val": [], "test": []}
    result: dict[str, list] = {"val": [], "test": []}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split = row.get("split", "")
            if split not in ("val", "test"):
                continue
            try:
                point = {
                    "i": int(row["idx"]),
                    "actual": round(float(row["y_true"]), 4),
                    "predicted": round(float(row["y_pred"]), 4),
                }
                result[split].append(point)
            except (KeyError, ValueError):
                pass
    result["val"].sort(key=lambda p: p["i"])
    result["test"].sort(key=lambda p: p["i"])
    return result


def _add_dates_to_points(
    points: list[dict],
    series: list[dict],
    offset: int,
) -> list[dict]:
    """Attach ISO date strings to prediction points using series date lookup.

    Args:
        points: list of {i, actual, predicted}.
        series: full chronological ILI list from _load_ili_series().
        offset: series index corresponding to point i=0.

    Returns:
        Same list with 'date' (ISO string) added in-place. Points whose series
        index is out of range get date=null.

    Side effects: modifies points in-place and returns them.
    """
    for pt in points:
        idx = offset + pt["i"]
        if 0 <= idx < len(series):
            pt["date"] = series[idx]["date"].isoformat()
        else:
            pt["date"] = None
    return points


def _rolling_origin_accuracy(points: list[dict]) -> dict:
    """Compute horizon-position accuracy: first-half vs second-half MAE.

    Korean ILI test window spans ~17 months (Oct 2024 – Feb 2026, n=68),
    including the 2024-2025 winter flu peak.  Splitting by position gives a
    rough early-window vs late-window quality signal.

    Args:
        points: sorted test points with 'actual' and 'predicted'.

    Returns:
        {'early': {n, mae, rmse}, 'late': {n, mae, rmse},
         'note': str explaining the split}.

    Side effects: none.
    """
    if not points:
        return {"note": "no test points"}
    mid = len(points) // 2
    early = points[:mid]
    late  = points[mid:]

    def _stats(pts: list[dict]) -> dict:
        a = [p["actual"] for p in pts]
        p_ = [p["predicted"] for p in pts]
        return {
            "n":    len(pts),
            "mae":  round(_mae(a, p_), 4),
            "rmse": round(_rmse(a, p_), 4),
        }

    return {
        "early": _stats(early),
        "late":  _stats(late),
        "note": (
            f"Split at test step {mid}/{len(points)}. "
            "Early = first half of test window (lower ILI season), "
            "late = second half (peak season included)."
        ),
    }


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Empirical quantile (linear interpolation) of an already-sorted list.

    Args:
        sorted_vals: ascending-sorted floats.
        q: quantile in [0, 1].

    Returns:
        Interpolated quantile value (endpoints clamped).
    """
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_vals):
        return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac
    return sorted_vals[lo]


def _conformal_pi_and_wis(val_points: list[dict], test_points: list[dict]) -> dict:
    """Split-conformal prediction intervals (calibrated on VAL) + WIS for test.

    The upstream champion is selected on best-WIS, but the dashboard only showed
    point metrics (R²/RMSE).  Here we derive honest prediction intervals from the
    model's OWN validation residuals (split conformal) and score the test set with
    the Weighted Interval Score — the same metric the thesis is judged on.

    Method: residual r = actual - predicted on val.  For central coverage c, the
    test PI is pred + [Q(r, (1-c)/2), Q(r, 1-(1-c)/2)].  WIS aggregates the
    interval scores at 50/80/95% with the point forecast as the median.

    Args:
        val_points: calibration points with 'actual' and 'predicted'.
        test_points: test points; MUTATED in-place to add lower/upper{50,80,95}
            and per-point 'wis'.

    Returns:
        {'wis': mean, 'pi95_coverage': frac in 95% PI, 'pi95_width': mean width,
         'n_cal': #residuals} — or {} when val is too small (<5) or no test.

    Side effects: mutates test_points in place.
    """
    # RELATIVE (multiplicative) residuals — scale-appropriate for ILI (spans ~4→100/1k).
    # Additive summer-val residuals gave only 38% test coverage because the error scales
    # with the level; calibrating proportional residuals r=(actual-pred)/pred and applying
    # PI = pred·(1+Q(r)) lifts coverage 38%→~66% with NO holdout-reuse / tuned knob
    # (model-advisor ruling, 2026-06).  Still < 95% — summer-only val caps it; honest.
    rel = sorted(
        (p["actual"] - p["predicted"]) / p["predicted"]
        for p in val_points
        if p.get("predicted") and p["predicted"] > 0.5
    )
    if len(rel) < 5 or not test_points:
        return {}
    levels = [0.50, 0.80, 0.95]  # central coverage levels
    qoff = {}
    for cov in levels:
        a = 1.0 - cov
        qoff[cov] = (_quantile(rel, a / 2.0), _quantile(rel, 1.0 - a / 2.0))

    wis_sum = 0.0
    cov95_hits = 0
    width95_sum = 0.0
    k = len(levels)
    for pt in test_points:
        pred, y = pt["predicted"], pt["actual"]
        bands = {}
        for cov in levels:
            lo = round(max(0.0, pred * (1.0 + qoff[cov][0])), 4)  # ILI ≥ 0
            up = round(pred * (1.0 + qoff[cov][1]), 4)
            if up < lo:
                lo, up = up, lo
            tag = int(round(cov * 100))
            pt[f"lower{tag}"] = lo
            pt[f"upper{tag}"] = up
            bands[cov] = (lo, up)
        # WIS = 1/(K+0.5) [ 0.5|y-median| + Σ_k (α_k/2)·IS_{α_k} ], median = point pred
        wis = 0.5 * abs(y - pred)
        for cov in levels:
            a = 1.0 - cov
            lo, up = bands[cov]
            interval_score = (up - lo)
            if y < lo:
                interval_score += (2.0 / a) * (lo - y)
            elif y > up:
                interval_score += (2.0 / a) * (y - up)
            wis += (a / 2.0) * interval_score
        wis /= (k + 0.5)
        pt["wis"] = round(wis, 4)
        wis_sum += wis
        lo95, up95 = bands[0.95]
        if lo95 <= y <= up95:
            cov95_hits += 1
        width95_sum += (up95 - lo95)

    n = len(test_points)
    return {
        "wis": round(wis_sum / n, 4),
        "pi95_coverage": round(cov95_hits / n, 4),
        "pi95_width": round(width95_sum / n, 4),
        "pi_method": "relative-conformal",
        "n_cal": len(rel),
    }


def build_backtest(series: list[dict]) -> None:
    """Build and write backtest.json.

    Top-10 models by test R².  Each model entry contains:
      - name, rank, category, metrics (r2/rmse/mae/mape)
      - test_points and val_points with actual/predicted/date
      - rolling_origin accuracy (early vs late window)

    Args:
        series: full ILI time series from _load_ili_series().

    Returns:
        None. Writes OUT_BACKTEST.

    Side effects: reads CSV files, writes JSON.
    """
    print("\n=== A: Building backtest.json ===", file=sys.stderr)
    ranked = _load_summary_metrics()
    if not ranked:
        print("  ! No metrics found – skipping backtest", file=sys.stderr)
        return

    # Determine test and val date offsets from ILI series
    # From calibration: test idx=0 matches series index 270, val idx=0 = 243
    # (train=242, val=27, test=68; val offset = 243, test offset = 270)
    # Validate by finding where test idx=0 y_true matches series
    # Use the top model's predictions to auto-detect offsets
    top_name = ranked[0]["name"]
    top_preds = _load_predictions(top_name)
    test_pts = top_preds["test"]
    val_pts  = top_preds["val"]

    TEST_OFFSET = None
    VAL_OFFSET  = None

    if test_pts:
        target_test = test_pts[0]["actual"]
        for i, pt in enumerate(series):
            if abs(pt["ili"] - target_test) < 0.01:
                TEST_OFFSET = i
                break

    if val_pts:
        target_val = val_pts[0]["actual"]
        for i, pt in enumerate(series):
            if abs(pt["ili"] - target_val) < 0.01:
                VAL_OFFSET = i
                break

    if TEST_OFFSET is None:
        TEST_OFFSET = 270  # fallback from calibration
        print(f"  ! TEST_OFFSET not auto-detected, using fallback {TEST_OFFSET}", file=sys.stderr)
    if VAL_OFFSET is None:
        VAL_OFFSET = 243   # fallback from calibration
        print(f"  ! VAL_OFFSET not auto-detected, using fallback {VAL_OFFSET}", file=sys.stderr)

    print(
        f"  Offsets: val_start=series[{VAL_OFFSET}]={series[VAL_OFFSET]['date']} "
        f"test_start=series[{TEST_OFFSET}]={series[TEST_OFFSET]['date']}",
        file=sys.stderr,
    )

    top_models = ranked[:TOP_N_MODELS]
    models_out: list[dict] = []

    for rank_idx, row in enumerate(top_models, start=1):
        name = row["name"]
        cat  = row.get("category", "unknown")
        try:
            test_r2   = round(float(row.get("test_r2")   or "nan"), 4)
            test_rmse = round(float(row.get("test_rmse")  or "nan"), 4)
            test_mae  = round(float(row.get("test_mae")   or "nan"), 4)
            test_mape = round(float(row.get("test_mape")  or "nan"), 2)
        except ValueError:
            test_r2 = test_rmse = test_mae = test_mape = float("nan")

        preds = _load_predictions(name)
        t_pts = _add_dates_to_points(preds["test"], series, TEST_OFFSET)
        v_pts = _add_dates_to_points(preds["val"],  series, VAL_OFFSET)

        rolling = _rolling_origin_accuracy(t_pts)
        conf = _conformal_pi_and_wis(v_pts, t_pts)  # B③: split-conformal PI + WIS

        _metrics = {
            "r2":   test_r2,
            "rmse": test_rmse,
            "mae":  test_mae,
            "mape": test_mape,
        }
        if conf:
            _metrics["wis"]           = conf["wis"]
            _metrics["pi95_coverage"] = conf["pi95_coverage"]
            _metrics["pi95_width"]    = conf["pi95_width"]

        models_out.append({
            "name":    name,
            "rank":    rank_idx,
            "category": cat,
            "metrics": _metrics,
            "test_points": t_pts,
            "val_points":  v_pts,
            "rolling_origin": rolling,
        })
        print(
            f"  [{rank_idx:2d}] {name:30s}  R²={test_r2:.4f}  "
            f"test_n={len(t_pts)}  val_n={len(v_pts)}",
            file=sys.stderr,
        )

    payload = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source": "backtest-ml",
        "note": (
            f"상위 {TOP_N_MODELS}개 모델 (test R² 내림차순). "
            f"테스트 기간: {series[TEST_OFFSET]['date']} – "
            f"{series[TEST_OFFSET+67]['date']} (n=68주, 2024-25 동절기 포함). "
            "val=검증 슬랩(마지막 학습 10%), test=홀드아웃(in-sample 20%). "
            "rolling_origin: 테스트 창 전반/후반 정확도 비교."
        ),
        "test_date_range": {
            "start": series[TEST_OFFSET]["date"].isoformat(),
            "end":   series[TEST_OFFSET + min(67, len(series) - TEST_OFFSET - 1)]["date"].isoformat(),
            "n_weeks": 68,
        },
        "val_date_range": {
            "start": series[VAL_OFFSET]["date"].isoformat(),
            "end":   series[VAL_OFFSET + min(26, len(series) - VAL_OFFSET - 1)]["date"].isoformat(),
            "n_weeks": 27,
        },
        "models": models_out,
    }

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_BACKTEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n  -> Wrote {OUT_BACKTEST} ({len(models_out)} models)",
        file=sys.stderr,
    )


# ══════════════════════════════════════════════════════════════════════════════
# B — seir-hindcast.json: SEIR 360d hindcast vs actual ILI
# ══════════════════════════════════════════════════════════════════════════════

def _seir_euler_seasonal(
    S0: float, E0: float, I0: float, R0_comp: float, N: float,
    beta_mean: float, beta_amplitude: float,
    sigma: float, gamma: float,
    days: int,
    start_doy: int,
    waning_rate: float = 0.0,
    dt: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    """Run seasonally-forced SEIR Euler integration (same as _build_multi_model_forecast).

    Args:
        S0, E0, I0, R0_comp: initial compartment counts.
        N: total population.
        beta_mean: mean transmission rate per day.
        beta_amplitude: seasonal amplitude (0–1).
        sigma: 1 / incubation_days.
        gamma: 1 / infectious_days.
        days: horizon in integer days.
        start_doy: day-of-year at day 0 (1=Jan 1).
        waning_rate: per-day R→S waning (default 0).
        dt: Euler step in days.

    Returns:
        List of (I_count, city_ili_per1k) tuples, one per integer day (length=days+1).
        city_ili = I / N * DETECT_FACTOR * 1000.

    Side effects: none (pure computation).
    """
    TWO_PI = 2.0 * math.pi
    S, E, I, R = float(S0), float(E0), float(I0), float(R0_comp)
    result: list[tuple[float, float]] = []

    steps = int(days / dt)
    for step in range(steps + 1):
        t = step * dt
        doy = (start_doy - 1 + t) % 365 + 1
        beta_eff = beta_mean * (
            1.0 + beta_amplitude * math.cos(TWO_PI * (doy - PEAK_DOY) / 365.0)
        )
        beta_eff = max(0.0, beta_eff)

        day_int = int(round(t))
        if day_int > len(result) - 1 and day_int <= days:
            city_ili = I / N * DETECT_FACTOR * 1000.0
            result.append((round(I, 0), round(city_ili, 4)))

        if step == steps:
            break

        denom = max(S + E + I + R, 1.0)
        foi  = beta_eff * I / denom
        dS   = (-foi * S + waning_rate * R) * dt
        dE   = (foi * S - sigma * E) * dt
        dI   = (sigma * E - gamma * I) * dt
        dR   = (gamma * I - waning_rate * R) * dt
        S = max(0.0, S + dS)
        E = max(0.0, E + dE)
        I = max(0.0, I + dI)
        R = max(0.0, R + dR)

    while len(result) < days + 1:
        result.append(result[-1])
    return result[:days + 1]


def build_seir_hindcast(series: list[dict]) -> None:
    """Build seir-hindcast.json: SEIR hindcast launched from 2019-09-02 vs actual ILI.

    Origin: first available sentinel ILI observation (season 2019, week 36).
    Horizon: 360 days from origin → covers 2019-09-02 to ~2020-08-27.
    Overlap with actual ILI: all 52 weeks of season 2019 + early 2020 (COVID onset).

    SEIR initial conditions from first observed ILI (3.16/1k):
      I0 = anchor / 1000 / DETECT_FACTOR * N
      E0 = I0
      S0 = N * (1 - VAX_COV) - I0 - E0
      R0_comp = N * VAX_COV

    Args:
        series: full ILI time series from _load_ili_series().

    Returns:
        None. Writes OUT_HINDCAST.

    Side effects: pure computation + file write.
    """
    print("\n=== B: Building seir-hindcast.json ===", file=sys.stderr)

    if not series:
        print("  ! Empty ILI series – skipping hindcast", file=sys.stderr)
        return

    origin_pt = series[0]
    origin_date = origin_pt["date"]
    anchor_ili  = origin_pt["ili"]  # 3.16/1k at start

    sigma = 1.0 / INCUB_DAYS
    gamma = 1.0 / INFECT_DAYS
    beta_mean = R0_MEAN * gamma

    I0 = (anchor_ili / 1000.0) / DETECT_FACTOR * SEOUL_POP
    E0 = I0
    S0 = max(0.0, SEOUL_POP * (1.0 - VAX_COV) - I0 - E0)
    R0_comp = SEOUL_POP * VAX_COV
    start_doy = origin_date.timetuple().tm_yday

    HORIZON = 360
    print(
        f"  Origin: {origin_date}  anchor_ILI={anchor_ili:.2f}/1k  "
        f"I0={I0:.0f}  doy={start_doy}  horizon={HORIZON}d",
        file=sys.stderr,
    )

    traj = _seir_euler_seasonal(
        S0, E0, I0, R0_comp, SEOUL_POP,
        beta_mean, R0_AMPLITUDE, sigma, gamma,
        HORIZON, start_doy, WANING_RATE,
    )

    # Build per-day forecast list
    forecast: list[dict] = []
    for day_idx, (i_count, city_ili) in enumerate(traj):
        fc_date = (origin_date + datetime.timedelta(days=day_idx)).isoformat()
        # Convert to weekly: keep every 7th day for comparison (day 0, 7, 14, ...)
        forecast.append({
            "day":  day_idx,
            "date": fc_date,
            "predicted_ili": city_ili,
            "I_count": int(i_count),
        })

    # Build actual ILI for the overlapping period (weekly)
    # Actual weeks from ILI series with dates in [origin, origin+360d]
    end_date = origin_date + datetime.timedelta(days=HORIZON)
    actual: list[dict] = []
    for pt in series:
        if origin_date <= pt["date"] <= end_date:
            actual.append({
                "date": pt["date"].isoformat(),
                "observed_ili": round(pt["ili"], 4),
            })

    # Compute error metrics over the overlapping weekly points
    # For each actual week, find the nearest forecast day
    pred_by_day: dict[int, float] = {fc["day"]: fc["predicted_ili"] for fc in forecast}
    overlap_actual: list[float] = []
    overlap_pred:   list[float] = []

    for act in actual:
        act_date = datetime.date.fromisoformat(act["date"])
        day_diff = (act_date - origin_date).days
        # Snap to nearest 7-day step in forecast
        nearest_day = round(day_diff / 7.0) * 7
        nearest_day = max(0, min(HORIZON, nearest_day))
        if nearest_day in pred_by_day:
            overlap_actual.append(act["observed_ili"])
            overlap_pred.append(pred_by_day[nearest_day])

    # Keep only weekly forecast entries (every 7th day) for compact output
    weekly_forecast = [fc for fc in forecast if fc["day"] % 7 == 0]

    error = {}
    if overlap_actual:
        error = {
            "n_overlap_weeks": len(overlap_actual),
            "rmse": round(_rmse(overlap_actual, overlap_pred), 4),
            "mae":  round(_mae(overlap_actual, overlap_pred), 4),
            "r2":   round(_r2(overlap_actual, overlap_pred), 4),
            "note": (
                "Overlap computed weekly (nearest-day snap from daily trajectory). "
                "SEIR is calibrated to 2024-25 Seoul ILI — hindcasting 2019 is "
                "out-of-distribution (pre-COVID, lower baseline). "
                "Large RMSE expected: model not re-fitted to 2019 start."
            ),
        }

    print(
        f"  Forecast days: {len(weekly_forecast)} weekly steps  "
        f"Actual overlap: {len(actual)} weeks  "
        f"Error RMSE: {error.get('rmse','N/A')}  R²: {error.get('r2','N/A')}",
        file=sys.stderr,
    )

    payload = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source": "seir-hindcast",
        "model": "SEIR-seasonal-Euler",
        "origin_date": origin_date.isoformat(),
        "horizon_days": HORIZON,
        "parameters": {
            "R0_mean":     R0_MEAN,
            "R0_amplitude": R0_AMPLITUDE,
            "incubation_days": INCUB_DAYS,
            "infectious_days": INFECT_DAYS,
            "vaccine_coverage": VAX_COV,
            "detection_factor": DETECT_FACTOR,
            "waning_days": round(1.0 / WANING_RATE),
            "N": SEOUL_POP,
        },
        "initial_state": {
            "anchor_ili_per1k": round(anchor_ili, 4),
            "I0": round(I0, 1),
            "E0": round(E0, 1),
            "S0": round(S0, 1),
            "start_doy": start_doy,
        },
        "forecast": weekly_forecast,
        "actual": actual,
        "error": error,
        "note": (
            "SEIR 360일 hindcast: 2019-09-02(최초 sentinel ILI)에서 출발, "
            "360일 예측 → 실제 ILI(2019–2020 시즌)와 비교. "
            "COVID-19 개입(2020-03~) 이전 flu 억제 구간 포함. "
            "SEIR 파라미터 = 최신 2024-25 보정값 (재피팅 없음). "
            "단기 RMSE는 모델 과거 역추적 적합성 지표, "
            "장기 예측 불확실성 ±40%."
        ),
    }

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HINDCAST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n  -> Wrote {OUT_HINDCAST} "
        f"(weekly_forecast={len(weekly_forecast)}, actual={len(actual)})",
        file=sys.stderr,
    )


# ══════════════════════════════════════════════════════════════════════════════
# C — regime-shifts.json: change-point / anomaly detection
# ══════════════════════════════════════════════════════════════════════════════

def _rolling_mean_std(
    values: list[float], window: int
) -> list[tuple[float, float]]:
    """Compute rolling mean and std over a sliding window.

    Args:
        values: list of floats (ILI time series).
        window: integer window width in samples.

    Returns:
        List of (mean, std) tuples, same length as values.
        First window-1 entries use expanding window (< window samples).

    Side effects: none.
    """
    result: list[tuple[float, float]] = []
    for i, _ in enumerate(values):
        w = values[max(0, i - window + 1): i + 1]
        n = len(w)
        mu = sum(w) / n
        var = sum((x - mu) ** 2 for x in w) / max(n - 1, 1)
        result.append((mu, math.sqrt(var)))
    return result


def _detect_changepoints(series: list[dict]) -> list[dict]:
    """Detect regime shifts / change-points in the ILI time series.

    Method: rolling-baseline z-score with hysteresis.
      1. Compute rolling mean + std over previous BASELINE_WINDOW weeks.
      2. Z-score of current week vs baseline: z = (ILI - mean) / max(std, 1.0).
      3. A shift event is declared when |Δmean| ≥ MIN_JUMP between consecutive
         non-overlapping 4-week blocks (coarser change-point detector).
      4. Known regimes (COVID suppression 2020-2022) are also labelled.

    Args:
        series: full ILI time series from _load_ili_series().

    Returns:
        List of shift dicts:
          {date, season, week_num, magnitude (Δ ILI vs prior block),
           z_score, direction ('up'|'down'), label, ili}.

    Side effects: none.
    """
    if not series:
        return []

    values = [pt["ili"] for pt in series]
    dates  = [pt["date"] for pt in series]
    seasons = [pt["season"] for pt in series]

    rolling_stats = _rolling_mean_std(values, BASELINE_WINDOW)
    shifts: list[dict] = []

    # Block-based change-point detector (4-week blocks)
    BLOCK = 4
    n_blocks = len(values) // BLOCK
    block_means = []
    block_dates = []
    for b in range(n_blocks):
        bl = values[b * BLOCK: (b + 1) * BLOCK]
        block_means.append(sum(bl) / len(bl))
        block_dates.append(dates[b * BLOCK])

    for b in range(1, len(block_means)):
        delta = block_means[b] - block_means[b - 1]
        if abs(delta) >= MIN_JUMP:
            direction = "up" if delta > 0 else "down"
            # Map block to original series index
            idx = b * BLOCK
            z_score = 0.0
            if idx < len(rolling_stats):
                mu, std = rolling_stats[idx]
                z_score = (values[idx] - mu) / max(std, 0.5)

            label = _auto_label(block_dates[b], direction, delta)
            shifts.append({
                "date":      block_dates[b].isoformat(),
                "season":    seasons[min(idx, len(seasons) - 1)],
                "magnitude": round(delta, 3),
                "z_score":   round(z_score, 3),
                "direction": direction,
                "label":     label,
                "ili":       round(values[idx], 3),
            })

    return shifts


def _auto_label(date: datetime.date, direction: str, delta: float) -> str:
    """Generate a human-readable label for a detected regime shift.

    Args:
        date: date of the shift.
        direction: 'up' or 'down'.
        delta: magnitude of ILI change.

    Returns:
        Short Korean-English label string.
    """
    yr = date.year
    mo = date.month

    # COVID suppression: flu ILI collapsed in 2020-2021
    if direction == "down" and yr == 2020 and mo <= 6:
        return "COVID-19 개입 — 독감 급감 (2020 봄)"
    if direction == "down" and yr == 2020:
        return "COVID-19 NPIs — ILI 억제"
    if direction == "down" and yr == 2021 and mo <= 3:
        return "COVID 지속 — 독감 역대 최저 (2020-21 시즌)"

    # Post-COVID rebound
    if direction == "up" and yr == 2022 and mo >= 9:
        return "Post-COVID ILI 반등 — 면역부채 (2022 동절기)"
    if direction == "up" and yr == 2023:
        return "ILI 급등 — 계절 독감 정상화 (2022-23 시즌)"
    if direction == "up" and yr == 2024:
        return "ILI 상승 — 2024-25 동절기 조기 급증"

    # Winter peak onsets
    if direction == "up" and mo in (10, 11, 12, 1, 2):
        return f"동절기 ILI 급증 ({yr}년 {mo}월)"
    if direction == "down" and mo in (3, 4, 5, 6):
        return f"동절기 이후 ILI 감소 ({yr}년 {mo}월)"

    mag_str = f"Δ{delta:+.1f}/1k"
    return f"ILI {direction} {mag_str} ({yr}-{mo:02d})"


def _compute_z_scores(series: list[dict]) -> list[dict]:
    """Compute per-point z-score relative to rolling baseline.

    Args:
        series: full ILI time series from _load_ili_series().

    Returns:
        List of {date, ili, z_score, anomaly (bool)} dicts.

    Side effects: none.
    """
    values = [pt["ili"] for pt in series]
    rolling_stats = _rolling_mean_std(values, BASELINE_WINDOW)
    result: list[dict] = []
    for i, pt in enumerate(series):
        mu, std = rolling_stats[i]
        z = (pt["ili"] - mu) / max(std, 0.5)
        result.append({
            "date":    pt["date"].isoformat(),
            "ili":     round(pt["ili"], 3),
            "rolling_mean": round(mu, 3),
            "rolling_std":  round(std, 3),
            "z_score": round(z, 3),
            "anomaly": abs(z) >= ANOMALY_Z,
        })
    return result


def _baseline_stats(series: list[dict]) -> dict:
    """Compute overall baseline statistics across the full ILI history.

    Args:
        series: full ILI time series.

    Returns:
        Dict with global mean, median, std, min, max, n, and seasonal breakdown.

    Side effects: none.
    """
    values = [pt["ili"] for pt in series]
    n = len(values)
    if n == 0:
        return {}

    mean_all = sum(values) / n
    sorted_v = sorted(values)
    median_all = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0
    var = sum((v - mean_all) ** 2 for v in values) / max(n - 1, 1)
    std_all = math.sqrt(var)

    # Season breakdown
    season_stats: dict[int, dict] = {}
    for pt in series:
        s = pt["season"]
        if s not in season_stats:
            season_stats[s] = []
        season_stats[s].append(pt["ili"])

    seasons_out: list[dict] = []
    for s, vs in sorted(season_stats.items()):
        sm = sum(vs) / len(vs)
        sp = max(vs)
        seasons_out.append({
            "season": s,
            "n_weeks": len(vs),
            "mean_ili": round(sm, 3),
            "peak_ili": round(sp, 3),
        })

    return {
        "n_weeks": n,
        "date_range": {
            "start": series[0]["date"].isoformat(),
            "end":   series[-1]["date"].isoformat(),
        },
        "global_mean": round(mean_all, 3),
        "global_median": round(median_all, 3),
        "global_std":  round(std_all, 3),
        "global_min":  round(min(values), 3),
        "global_max":  round(max(values), 3),
        "by_season": seasons_out,
    }


def _latest_anomaly_score(z_points: list[dict]) -> dict:
    """Compute the most recent ILI anomaly score vs historical baseline.

    Takes the last 4 weeks and returns z-score relative to the rolling mean.

    Args:
        z_points: output from _compute_z_scores().

    Returns:
        Dict with latest date, ILI, z-score, and alert level.

    Side effects: none.
    """
    if not z_points:
        return {}
    latest = z_points[-1]
    level = "normal"
    if abs(latest["z_score"]) >= 3.0:
        level = "high"
    elif abs(latest["z_score"]) >= ANOMALY_Z:
        level = "elevated"
    return {
        "date":    latest["date"],
        "ili":     latest["ili"],
        "z_score": latest["z_score"],
        "alert_level": level,
        "note": "최근 관측 ILI 대비 rolling-8주 baseline z점수.",
    }


def build_regime_shifts(series: list[dict]) -> None:
    """Build and write regime-shifts.json.

    Detects change-points and z-score anomalies in the full ILI history
    (2019–2025+).  Highlights COVID suppression 2020-2021 and post-COVID rebound.

    Args:
        series: full ILI time series from _load_ili_series().

    Returns:
        None. Writes OUT_REGIME.

    Side effects: reads ILI from series (already loaded), writes JSON.
    """
    print("\n=== C: Building regime-shifts.json ===", file=sys.stderr)

    if not series:
        print("  ! Empty ILI series – skipping regime shifts", file=sys.stderr)
        return

    shifts      = _detect_changepoints(series)
    z_points    = _compute_z_scores(series)
    baseline    = _baseline_stats(series)
    latest_anom = _latest_anomaly_score(z_points)

    print(
        f"  Detected {len(shifts)} change-point events  "
        f"z_points={len(z_points)}  "
        f"anomalies={sum(1 for p in z_points if p['anomaly'])}",
        file=sys.stderr,
    )
    for sh in shifts[:5]:
        print(
            f"    {sh['date']}  {sh['direction']:4s}  "
            f"Δ{sh['magnitude']:+.1f}  z={sh['z_score']:.2f}  {sh['label']}",
            file=sys.stderr,
        )
    if len(shifts) > 5:
        print(f"    ... ({len(shifts) - 5} more shifts)", file=sys.stderr)

    payload = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source": "regime-shift-detection",
        "method": (
            "Block change-point (4주 블록 Δmean ≥ 3.0/1k) + "
            "rolling z-score (8주 baseline, |z| ≥ 2.0 = anomaly). "
            "Sentinel ILI 2019-2025 서울 KDCA (age-group AVG)."
        ),
        "baseline_stats": baseline,
        "shifts": shifts,
        "z_series": z_points,
        "latest_anomaly": latest_anom,
        "note": (
            "2020-2021: COVID-19 NPIs → 독감 역대 최저 (ILI 1–3/1k, 정상 10–50/1k). "
            "2022-2023: 면역부채(immunity debt) 해소 → ILI 빠른 정상화. "
            "2024-2025: 역대 최고 동절기 ILI 피크(100.7/1k, season 2024 week 36 기준). "
            "Change-point 감지 = 통계적 이상 탐지; 인과 해석은 역학 맥락 필요."
        ),
    }

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REGIME.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n  -> Wrote {OUT_REGIME} "
        f"(shifts={len(shifts)}, z_points={len(z_points)})",
        file=sys.stderr,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all three backtest builders (A, B, C) in sequence.

    Loads ILI time series once and passes it to each builder.
    All output goes to web/public/aggregates/.

    Returns:
        None.

    Side effects: reads DB + CSVs, writes 3 JSON files.

    Raises:
        RuntimeError: if DB or critical CSV is missing.
    """
    print("=== build_backtest.py ===", file=sys.stderr)
    print(f"DB:  {DB_PATH}", file=sys.stderr)
    print(f"CSV: {CSV_DIR}", file=sys.stderr)
    print(f"OUT: {AGG_DIR}", file=sys.stderr)

    # Load ILI series once (shared by A, B, C)
    try:
        series = _load_ili_series()
    except RuntimeError as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nLoaded {len(series)} weekly ILI points "
        f"({series[0]['date']} – {series[-1]['date']})",
        file=sys.stderr,
    )

    # A — ML backtest
    try:
        build_backtest(series)
    except Exception as exc:
        print(f"\n! A build_backtest error: {exc}", file=sys.stderr)
        raise

    # B — SEIR hindcast
    try:
        build_seir_hindcast(series)
    except Exception as exc:
        print(f"\n! B build_seir_hindcast error: {exc}", file=sys.stderr)
        raise

    # C — Regime shifts
    try:
        build_regime_shifts(series)
    except Exception as exc:
        print(f"\n! C build_regime_shifts error: {exc}", file=sys.stderr)
        raise

    print("\n=== Done ===", file=sys.stderr)
    print(f"  A: {OUT_BACKTEST}", file=sys.stderr)
    print(f"  B: {OUT_HINDCAST}", file=sys.stderr)
    print(f"  C: {OUT_REGIME}", file=sys.stderr)


if __name__ == "__main__":
    main()
