#!/usr/bin/env python3
"""Build the Seoul Metro route-line geojson for the map's subway overlay.

The DB has subway RIDERSHIP by station/line name but no coordinates, so the route
geometry comes from station coordinates (lat/lon + line number) in
``simulation/data/external/seoul_subway_stations.csv`` (Seoul open-data 역사 좌표,
via yoon-gu/902efb6d gist). Stations are ordered along each line in the CSV, so a
LineString per line connects them in order. Output (FeatureCollection of
LineStrings, one per line, with the official Seoul Metro colors) is consumed by
``Map3D.tsx`` GeoJsonLayer.

Reproducible: reads the committed CSV (no network). Run:
    python web/scripts/build_subway_lines.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "simulation" / "data" / "external" / "seoul_subway_stations.csv"
OUT = ROOT / "web" / "public" / "aggregates" / "subway-lines.geojson"

# Official Seoul Metro line colors (서울교통공사 노선 색상).
LINE_COLORS = {
    "1": "#0052A4", "2": "#00A84D", "3": "#EF7C1C", "4": "#00A5DE",
    "5": "#996CAC", "6": "#CD7C2F", "7": "#747F00", "8": "#E6186C", "9": "#BDB092",
}


def build() -> dict:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    by_line: dict[str, list[list[float]]] = {}
    for r in rows:
        try:
            lon, lat = float(r["lon"]), float(r["lat"])
        except (KeyError, ValueError):
            continue
        by_line.setdefault(str(r["no_line"]).strip(), []).append([lon, lat])
    features = []
    for line, coords in sorted(by_line.items()):
        if len(coords) < 2:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "line": line, "name": f"{line}호선",
                "color": LINE_COLORS.get(line, "#888888"),
                "n_stations": len(coords),
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    return {"type": "FeatureCollection", "features": features}


def main() -> int:
    if not CSV.exists():
        print(f"! station CSV missing: {CSV}")
        return 1
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    n_st = sum(f["properties"]["n_stations"] for f in gj["features"])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(gj['features'])} lines, {n_st} stations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
