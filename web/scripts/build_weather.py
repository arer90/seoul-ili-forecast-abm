#!/usr/bin/env python3
"""Build the Seoul weather time-series overlay (panel + forecast) for the map.

Two real KMA sources:
  • weather_forecast  — single Seoul grid cell (nx=60, ny=127; NOT a spatial
                        grid), latest issue, hourly for ~72 h. Pivoted per
                        valid_at into TMP(기온)/POP(강수확률)/PTY/SKY/REH(습도)/
                        WSD(풍속)/PCP(강수량) → a slider-synced forecast panel.
  • weather_historical — Seoul station daily ta_avg/max/min, ws_avg(풍속),
                        hm_avg(습도), rn_day(강수); recent window → trend sparkline.

NOTE: spatial temperature/wind per gu come from rt_sdot_env (build_air_env.py),
since the forecast is a single point. Output (weather.json) consumed by Map3D.tsx
as a time panel — there is no spatial weather LAYER from this file. Reproducible
(DB read-only, no key).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT = ROOT / "web" / "public" / "aggregates" / "weather.json"

_FC_VARS = ["TMP", "POP", "PTY", "SKY", "REH", "WSD", "PCP"]
_HIST_DAYS = 120


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v  # keep categorical strings (PCP "강수없음", PTY codes) as-is


def build() -> dict:
    from simulation.database import read_only_connect
    con = read_only_connect(str(DB))
    try:
        issue = con.execute("SELECT MAX(issued_at) FROM weather_forecast").fetchone()[0]
        pivot: dict[str, dict] = defaultdict(dict)
        for var, valid, val in con.execute(
            "SELECT variable, valid_at, value FROM weather_forecast "
            "WHERE issued_at = ? AND variable IN (%s)" % ",".join("?" * len(_FC_VARS)),
            (issue, *_FC_VARS)):
            pivot[valid][var] = _f(val)
        forecast = [{"valid_at": v, **pivot[v]} for v in sorted(pivot)]
        hist = [
            {"date": d, "ta_avg": ta, "ta_max": tx, "ta_min": tn,
             "ws_avg": ws, "hm_avg": hm, "rn_day": rn}
            for d, ta, tx, tn, ws, hm, rn in con.execute(
                "SELECT obs_date, ta_avg, ta_max, ta_min, ws_avg, hm_avg, rn_day "
                "FROM weather_historical ORDER BY obs_date DESC LIMIT ?", (_HIST_DAYS,))
        ][::-1]
    finally:
        con.close()
    return {"issued_at": issue, "forecast": forecast, "historical": hist}


def main() -> int:
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    fc, hi = gj["forecast"], gj["historical"]
    t0 = fc[0] if fc else {}
    print(f"wrote {OUT.relative_to(ROOT)} (forecast {len(fc)}h @ issue {gj['issued_at']}, "
          f"historical {len(hi)} days)")
    print(f"  first forecast: {t0}")
    if hi:
        print(f"  latest obs: {hi[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
