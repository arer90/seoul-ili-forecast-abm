#!/usr/bin/env python3
"""Build the Seoul bus-stop overlay (all 11,253 stops) for the map.

Bus route GEOMETRY is not in the Seoul Open Data stop API, but the 11,253 stop
COORDINATES (busStopLocationXyInfo, fetched to seoul_bus_stops.csv) cover all
three requested bus options at once via the stop TYPE:
  ① 간선(trunk) corridor   = 중앙차로 stops (median-lane = BRT/trunk routes)
  ② 정류장 밀집도(density)  = heatmap of every stop
  ③ 전체(full) network      = every stop colored by type
The map renders a HeatmapLayer (②) + a ScatterplotLayer colored by type with the
trunk corridor (중앙차로) highlighted (①/③).

Reproducible from the committed CSV (public coordinates, no key). Output consumed
by Map3D.tsx.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "simulation" / "data" / "external" / "seoul_bus_stops.csv"
OUT = ROOT / "web" / "public" / "aggregates" / "bus-stops.json"

# stop-type → display color (RGB). 중앙차로 = trunk corridor (highlight red).
_TYPE_COLOR = {
    "중앙차로": [233, 30, 60], "일반차로": [80, 150, 230], "마을버스": [60, 190, 120],
    "가로변전일": [180, 180, 190], "가로변시간": [180, 180, 190],
}
_DEFAULT = [150, 150, 160]


def _ridership() -> dict[str, int]:
    """stops_no → daily boarding+alighting from daily_bus (current ridership).

    daily_bus.station_id matches the stop STOPS_NO (verified 709/718 overlap);
    only a sampled subset of stops appears, the rest get no ridership.
    """
    DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
    if not DB.exists():
        return {}
    from simulation.database import read_only_connect
    con = read_only_connect(str(DB))
    try:
        return {str(sid): int((r or 0) + (a or 0)) for sid, r, a in con.execute(
            "SELECT station_id, SUM(ride_cnt), SUM(alight_cnt) FROM daily_bus "
            "GROUP BY station_id")}
    finally:
        con.close()


def build() -> dict:
    ride = _ridership()
    stops = []
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lon, lat = float(r["lon"]), float(r["lat"])
            except (KeyError, ValueError):
                continue
            if not (126.7 < lon < 127.3 and 37.4 < lat < 37.75):
                continue
            t = (r.get("type") or "").strip()
            stops.append({"position": [round(lon, 6), round(lat, 6)],
                          "type": t, "trunk": t == "중앙차로",
                          "ridership": ride.get(r["stops_no"], 0),
                          "color": _TYPE_COLOR.get(t, _DEFAULT)})
    return {"stops": stops, "type_colors": _TYPE_COLOR}


def main() -> int:
    if not CSV.exists():
        print(f"! bus-stops CSV missing: {CSV} (fetch via busStopLocationXyInfo first)")
        return 1
    gj = build()
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    by = Counter(s["type"] for s in gj["stops"])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(gj['stops'])} stops; "
          f"trunk 중앙차로={by.get('중앙차로', 0)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
