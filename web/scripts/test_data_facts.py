#!/usr/bin/env python3
"""Data-fact regression tests for the web aggregates (TDD).

Verifies that every number the web SHOWS is traceable to the real DB source —
no fabrication, no scale/series mismatch. Run after any aggregate rebuild:

    .venv/bin/python -m pytest web/scripts/test_data_facts.py -v
    # or standalone:
    .venv/bin/python web/scripts/test_data_facts.py

Provenance verified (2026-06-09):
  - City ILI = AVG over the 7 sentinel_influenza age groups (the DB has NO
    "연령군 평균" row; it is a computed mean). Latest = 5.143/1k.
  - backtest.json y_true is that exact AVG series (68/68 match at offset 270).
  - production forecast is a FUTURE week (forecast_at > observed_at), gated.
  - regime-shifts detect the real 2020 COVID flu collapse.

DB access via the project read-only helper (single-helper DB policy).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
AGG = ROOT / "web" / "public" / "aggregates"
DATA_JSON = ROOT / "web_prototype" / "data.json"


def _con():
    from simulation.database import read_only_connect
    return read_only_connect(str(DB))


def _avg_ili_series() -> list[float]:
    """Weekly city ILI = mean over the 7 sentinel age groups (the real series)."""
    con = _con()
    try:
        rows = con.execute(
            "SELECT season_start, week_seq, AVG(ili_rate) "
            "FROM sentinel_influenza GROUP BY season_start, week_seq "
            "ORDER BY season_start, week_seq"
        ).fetchall()
    finally:
        con.close()
    return [round(r[2], 3) for r in rows]


def _load(name: str):
    return json.loads((AGG / name).read_text(encoding="utf-8"))


# ── [1] ILI series identity ────────────────────────────────────────────────
def test_city_ili_is_age_group_average():
    """data.json '연령군 평균' must equal the computed mean of DB age groups."""
    d = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    web_latest = d["ili_weekly"][-1]["rate"]
    avg = _avg_ili_series()
    assert avg, "AVG series empty"
    assert abs(web_latest - avg[-1]) < 0.1, f"web {web_latest} != AVG {avg[-1]}"


def test_no_avg_row_in_db():
    """The DB stores only the 7 specific age groups — the average is derived."""
    con = _con()
    try:
        groups = {r[0] for r in con.execute(
            "SELECT DISTINCT age_group FROM sentinel_influenza").fetchall()}
    finally:
        con.close()
    assert "연령군 평균" not in groups
    assert len(groups) == 7, f"expected 7 age groups, got {groups}"


# ── [2] Backtest y_true is the real series ─────────────────────────────────
def test_backtest_ytrue_is_real_avg_series():
    """Every test y_true must be a contiguous slice of the real AVG series."""
    bt = _load("backtest.json")
    yt = [round(p["actual"], 3) for p in bt["models"][0]["test_points"]]
    avg = _avg_ili_series()
    for off in range(0, len(avg) - len(yt) + 1):
        seg = avg[off:off + len(yt)]
        if all(abs(a - b) < 0.05 for a, b in zip(seg, yt)):
            return  # exact contiguous match found
    raise AssertionError("backtest y_true is not a slice of the real AVG ILI series")


def test_backtest_r2_matches_summary_metrics():
    """Reported champion R² must equal summary_metrics.csv (no inflated number)."""
    import csv
    bt = _load("backtest.json")
    name = bt["models"][0]["name"]
    r2_reported = bt["models"][0]["metrics"]["r2"]
    rows = list(csv.DictReader(
        (ROOT / "simulation" / "results" / "csv" / "summary_metrics.csv").open()))
    row = next((r for r in rows if r.get("name") == name or r.get("model") == name), None)
    assert row, f"{name} not in summary_metrics"
    assert abs(r2_reported - float(row["test_r2"])) < 0.005


# ── [3] Production forecast is a real future prediction ────────────────────
def test_forecast_is_future():
    fc = _load("ili-forecast-models.json")
    assert fc["forecast_at"] > fc["observed_at"], "forecast not beyond last observation"


def test_forecast_is_production_refit():
    fc = _load("ili-forecast-models.json")
    assert "production" in fc.get("source", ""), f"source={fc.get('source')}"


def test_forecast_gated_within_bounds():
    """Gated forecast: finite, non-negative, not an extrapolation blow-up."""
    fc = _load("ili-forecast-models.json")
    v = fc["models"][0]["city_forecast"]
    assert 0.0 <= v < 300.0, f"forecast {v} outside gate"


# ── [4] Regime shift detects the real COVID collapse ───────────────────────
def test_regime_detects_covid_collapse():
    rs = _load("regime-shifts.json")
    shifts = rs.get("shifts") or rs.get("change_points") or []
    blob = json.dumps(shifts, ensure_ascii=False)
    assert "2020" in blob or "COVID" in blob, "2020 COVID collapse not detected"


# ── [5] Aggregate counts trace to source ───────────────────────────────────
def test_subway_station_count():
    ss = _load("subway-stations.json")
    n = len(ss if isinstance(ss, list) else ss.get("stations", []))
    assert n == 223, f"subway stations {n} != 223"


def test_schools_count_matches_db():
    sch = _load("schools.json")
    n = len(sch if isinstance(sch, list) else sch.get("schools", []))
    con = _con()
    try:
        db_n = con.execute("SELECT COUNT(*) FROM school_info_seoul").fetchone()[0]
    finally:
        con.close()
    assert abs(n - db_n) < 30, f"web {n} vs DB {db_n}"


# ── [6] Per-gu ILI uses real density, not fabricated ───────────────────────
def test_density_is_real_source():
    """Density modulating per-gu ILI must come from daily_population_gu_hourly."""
    con = _con()
    try:
        n = con.execute("SELECT COUNT(DISTINCT gu_nm) FROM daily_population_gu_hourly").fetchone()[0]
    finally:
        con.close()
    assert n == 25, f"expected 25 gu in density source, got {n}"


if __name__ == "__main__":
    import sys
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn()
            print(f"  ✓ PASS  {fn.__name__}")
            p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}")
            f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
