#!/usr/bin/env python3
"""Build the enriched Seoul subway STATION overlay for the map.

Each station (from seoul_subway_stations.csv coords) carries four real signals
so the map can size/color/animate them:
  ① 승하차 규모   ridership = SUM(ride_pasgr + alight_pasgr) over daily_subway,
                  name-normalized join → marker size.
  ② 환승역(교합점) lines[] from the CSV (a station listed on >1 호선) → transfer
                  flag + n_lines → highlight + size by interchange degree.
  ③ 러시아워 맥동 hourly[24] = SUM(ride_cnt+alight_cnt) per hour from
                  monthly_subway_hourly, normalized to the station's own peak →
                  pulse synced to the time slider.
  ④ 구 연령구성   age = {youth, adult, elderly} share of the gu the station sits
                  in (point-in-polygon vs seoul-gu.geojson × daily_population age
                  bands). PROXY: the district's residents, not the riders' ages.

Reproducible from the DB (read-only) + committed CSV/geojson; no key. Output
(subway-stations.json) consumed by Map3D.tsx.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
CSV = ROOT / "simulation" / "data" / "external" / "seoul_subway_stations.csv"
GEOJSON = ROOT / "web" / "public" / "seoul-gu.geojson"
OUT = ROOT / "web" / "public" / "aggregates" / "subway-stations.json"


def _norm(s: str) -> str:
    """Normalize a station name for cross-table joins (drop 역/(...)/digits)."""
    s = re.sub(r"\(.*?\)", "", s or "")
    s = re.sub(r"역$", "", s.strip())
    return re.sub(r"[0-9·.\s]", "", s).strip()


def _gu_lookup():
    """Return a fn position[lon,lat] -> gu name via point-in-polygon."""
    from shapely.geometry import Point, shape
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    polys = []
    for f in geo["features"]:
        p = f["properties"]
        name = p.get("name") or p.get("SIG_KOR_NM") or next(iter(p.values()))
        polys.append((name, shape(f["geometry"])))

    def which(lon: float, lat: float):
        pt = Point(lon, lat)
        for name, geom in polys:
            if geom.contains(pt):
                return name
        return None

    return which


def build() -> dict:
    from simulation.database import read_only_connect
    con = read_only_connect(str(DB))
    try:
        # ① ridership
        ride: dict[str, float] = defaultdict(float)
        for nm, r, a in con.execute(
            "SELECT station_nm, SUM(ride_pasgr), SUM(alight_pasgr) "
            "FROM daily_subway GROUP BY station_nm"):
            ride[_norm(nm)] += (r or 0) + (a or 0)
        # ③ hourly profile
        hourly: dict[str, list[float]] = defaultdict(lambda: [0.0] * 24)
        for nm, h, rc, ac in con.execute(
            "SELECT station_nm, hour, SUM(ride_cnt), SUM(alight_cnt) "
            "FROM monthly_subway_hourly GROUP BY station_nm, hour"):
            if h is not None and 0 <= int(h) < 24:
                hourly[_norm(nm)][int(h)] += (rc or 0) + (ac or 0)
        # ④ gu age shares (youth 0-19 / adult 20-59 / elderly 60+)
        gu_age: dict[str, dict] = {}
        for gu, y, ad, el in con.execute(
            "SELECT gu_nm, SUM(pop_0_9+pop_10_19), "
            "SUM(pop_20_29+pop_30_39+pop_40_49+pop_50_59), "
            "SUM(pop_60_69+pop_70plus) FROM daily_population_gu_hourly GROUP BY gu_nm"):
            tot = (y or 0) + (ad or 0) + (el or 0)
            if tot:
                gu_age[gu] = {"youth": round(y / tot, 3),
                              "adult": round(ad / tot, 3),
                              "elderly": round(el / tot, 3)}
    finally:
        con.close()

    which_gu = _gu_lookup()
    # CSV: name -> {coords, lines}
    by_name: dict[str, dict] = {}
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lon, lat = float(r["lon"]), float(r["lat"])
            except (KeyError, ValueError):
                continue
            d = by_name.setdefault(r["name"], {"pos": [round(lon, 6), round(lat, 6)],
                                               "lines": set()})
            d["lines"].add(str(r["no_line"]).strip())

    stations = []
    for name, d in by_name.items():
        key = _norm(name)
        prof = hourly.get(key)
        peak = max(prof) if prof and max(prof) > 0 else 0
        gu = which_gu(d["pos"][0], d["pos"][1])
        stations.append({
            "name": name, "position": d["pos"],
            "lines": sorted(d["lines"]), "n_lines": len(d["lines"]),
            "transfer": len(d["lines"]) > 1,
            "ridership": int(ride.get(key, 0)),
            "gu": gu, "age": gu_age.get(gu),
            "hourly": [round(v / peak, 3) for v in prof] if peak else None,
        })
    stations.sort(key=lambda s: -s["ridership"])
    return {"stations": stations}


def main() -> int:
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    st = gj["stations"]
    n_tr = sum(1 for s in st if s["transfer"])
    n_ride = sum(1 for s in st if s["ridership"] > 0)
    n_hr = sum(1 for s in st if s["hourly"])
    top = st[0] if st else {}
    print(f"wrote {OUT.relative_to(ROOT)} ({len(st)} stations; {n_tr} transfer; "
          f"{n_ride} with ridership; {n_hr} with hourly)")
    print(f"  busiest: {top.get('name')} ({top.get('ridership'):,}, "
          f"lines {top.get('lines')}, gu {top.get('gu')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
