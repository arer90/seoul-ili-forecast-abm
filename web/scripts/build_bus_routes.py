#!/usr/bin/env python3
"""Build real Seoul bus ROUTE polylines for the map's bus-line overlay.

Bus route geometry comes from the Seoul Open Data ``masterRouteNode`` service
(노선 정류장마스터): each row is (RTE_ID 노선, CRTR_ID 노드, CRTR_SEQ 순번,
LNKG_LEN 링크길이). The node id CRTR_ID matches the stop id STOPS_NO in
``seoul_bus_stops.csv`` (verified 268/271 overlap on a sample), so ordering each
route's nodes by CRTR_SEQ and mapping CRTR_ID -> (lon, lat) yields a LineString
per route — the actual bus line along its stops.

The 191,215 route-node rows were fetched once (key-gated) to the committed cache
``seoul_bus_route_nodes.csv``; this builder reads that cache + the stop CSV and
needs NO network or key, so it is fully reproducible.

Three bus options are served at once:
  ① 간선(trunk)  = routes colored by trunk_frac (fraction of nodes on 중앙차로
                   median-lane corridors) — BRT/trunk routes light up.
  ② 밀집도        = stop heatmap (build_bus_stops.py).
  ③ 전체(full)    = every route polyline.
Output (FeatureCollection of LineStrings) consumed by Map3D.tsx GeoJsonLayer.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "simulation" / "data" / "external" / "seoul_bus_route_nodes.csv"
STOPS = ROOT / "simulation" / "data" / "external" / "seoul_bus_stops.csv"
OUT = ROOT / "web" / "public" / "aggregates" / "bus-routes.geojson"

#: Split a route's polyline wherever consecutive stops jump farther than this.
#: masterRouteNode's CRTR_SEQ is numbered per DIRECTION (상·하행 share seq 1..N)
#: with no direction column, so a plain seq sort places the up-end next to the
#: down-end (opposite sides of the city) and draws a false ~25 km chord. Seoul
#: bus stops sit ~440 m apart (median), p90 ~1.7 km, so any >2.5 km consecutive
#: hop is a direction-boundary/data artifact, not a real segment — break there.
SPLIT_KM = 2.5


def _haversine_km(a: list[float], b: list[float]) -> float:
    """Great-circle distance in km between [lon, lat] points a and b."""
    from math import asin, cos, radians, sin, sqrt
    dlat, dlon = radians(b[1] - a[1]), radians(b[0] - a[0])
    h = sin(dlat / 2) ** 2 + cos(radians(a[1])) * cos(radians(b[1])) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def _split_on_jumps(pts: list[list[float]]) -> list[list[list[float]]]:
    """Break an ordered point list into sub-lines at jumps > SPLIT_KM.

    Returns a list of LineString coordinate arrays (each length >= 2); drops
    singletons. Keeps verified-adjacent stop hops connected, severs the
    cross-city direction-boundary chords.
    """
    lines: list[list[list[float]]] = []
    cur = [pts[0]]
    for prev, nxt in zip(pts, pts[1:]):
        if _haversine_km(prev, nxt) > SPLIT_KM:
            if len(cur) >= 2:
                lines.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    if len(cur) >= 2:
        lines.append(cur)
    return lines


def _stop_maps() -> tuple[dict[str, list[float]], dict[str, str]]:
    """Return {stops_no: [lon, lat]} and {stops_no: type} from the stop CSV."""
    coord: dict[str, list[float]] = {}
    stype: dict[str, str] = {}
    with STOPS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lon, lat = float(r["lon"]), float(r["lat"])
            except (KeyError, ValueError):
                continue
            if not (126.7 < lon < 127.3 and 37.4 < lat < 37.75):
                continue
            coord[r["stops_no"]] = [round(lon, 6), round(lat, 6)]
            stype[r["stops_no"]] = (r.get("type") or "").strip()
    return coord, stype


def build() -> dict:
    coord, stype = _stop_maps()
    # group route-nodes by route, keep (seq, crtr)
    routes: dict[str, list[tuple[int, str]]] = {}
    with NODES.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                seq = int(r["seq"])
            except (KeyError, ValueError):
                continue
            routes.setdefault(r["rte_id"], []).append((seq, r["crtr_id"]))
    feats = []
    for rte, nodes in routes.items():
        nodes.sort()
        pts, n_trunk = [], 0
        for _, c in nodes:
            if c in coord:
                pts.append(coord[c])
                if stype.get(c) == "중앙차로":
                    n_trunk += 1
        if len(pts) < 2:
            continue
        lines = _split_on_jumps(pts)
        if not lines:
            continue
        kept = sum(len(ln) for ln in lines)
        trunk_frac = round(n_trunk / len(pts), 3)
        feats.append({
            "type": "Feature",
            "properties": {"rte_id": rte, "n": kept, "n_parts": len(lines),
                           "trunk_frac": trunk_frac, "trunk": trunk_frac >= 0.3},
            "geometry": {"type": "MultiLineString", "coordinates": lines},
        })
    return {"type": "FeatureCollection", "features": feats}


def main() -> int:
    if not NODES.exists():
        print(f"! route-node cache missing: {NODES} (fetch masterRouteNode first)")
        return 1
    if not STOPS.exists():
        print(f"! bus-stops CSV missing: {STOPS}")
        return 1
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    n_trunk = sum(1 for f in gj["features"] if f["properties"]["trunk"])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(gj['features'])} route lines, "
          f"{n_trunk} trunk 간선; {OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
