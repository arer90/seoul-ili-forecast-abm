#!/usr/bin/env python3
"""Build the Seoul real-time POI overlay (population / traffic) for the map.

The Seoul 실시간도시데이터 feed reports 79 major places (POI codes) with live
crowding + age mix + road speed. None carry coordinates, but poi_metadata maps
area_cd → (lat, lon), so a single join yields placeable points:
  • rt_population_detail — congestion(여유..붐빔), ppltn_min/max, age rates
    (rate_0..rate_70 in %), resident/non-resident split → 인구집단 점 layer
    (size ∝ ppltn, color by congestion; flu-relevant where crowds + elderly).
  • rt_road_traffic — road_traffic_idx(원활/서행/정체) + speed → 도로교통.
  • rt_bike_status — city-wide 따릉이 totals only (NO per-station coords) → a
    panel stat, not a map layer.

⚠ These collectors last ran 2026-04-16 (≈2 months stale vs the live air/subway
feeds); collected_at is surfaced so the map can label the vintage honestly.
Reproducible (DB read-only, no key). Output consumed by Map3D.tsx.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT = ROOT / "web" / "public" / "aggregates" / "realtime-poi.json"

_AGE = ["rate_0", "rate_10", "rate_20", "rate_30", "rate_40", "rate_50", "rate_60", "rate_70"]


def _category(nm: str) -> str:
    """Classify a 실시간도시데이터 place by name → Seoul's area categories."""
    if "관광특구" in nm:
        return "관광특구"
    if "고궁" in nm or "문화유산" in nm or nm.endswith("궁") or "릉" in nm:
        return "고궁·문화유산"
    if "공원" in nm or "한강" in nm or "숲" in nm or "수목원" in nm:
        return "공원"
    if "시장" in nm or "상권" in nm or "거리" in nm or "로데오" in nm or "타운" in nm:
        return "발달상권"
    if nm.endswith("역") or "역(" in nm:
        return "역세권"
    return "기타"


def build() -> dict:
    from simulation.database import read_only_connect
    con = read_only_connect(str(DB))
    try:
        ts_pop = con.execute("SELECT MAX(collected_at) FROM rt_population_detail").fetchone()[0]
        ts_trf = con.execute("SELECT MAX(collected_at) FROM rt_road_traffic").fetchone()[0]
        # population (latest) + coords
        agecols = ", ".join(f"d.{c}" for c in _AGE)
        pop = {}
        for row in con.execute(
            f"SELECT d.area_cd, d.area_nm, p.lon, p.lat, d.congestion, d.ppltn_min, "
            f"d.ppltn_max, d.resnt_rate, d.non_resnt_rate, {agecols} "
            "FROM rt_population_detail d JOIN poi_metadata p ON d.area_cd = p.area_cd "
            "WHERE d.collected_at = ?", (ts_pop,)):
            cd, nm, lon, lat = row[0], row[1], row[2], row[3]
            if lon is None or lat is None:
                continue
            pop[cd] = {
                "area_nm": nm, "position": [round(lon, 6), round(lat, 6)],
                "category": _category(nm),
                "congestion": row[4], "ppltn_min": row[5], "ppltn_max": row[6],
                "resident": row[7], "non_resident": row[8],
                "ages": {a.replace("rate_", ""): row[9 + i] for i, a in enumerate(_AGE)},
            }
        # road traffic (latest) merged onto the same POIs
        for cd, idx, spd in con.execute(
            "SELECT area_cd, road_traffic_idx, road_traffic_spd FROM rt_road_traffic "
            "WHERE collected_at = ?", (ts_trf,)):
            if cd in pop:
                pop[cd]["traffic_idx"] = idx
                pop[cd]["traffic_spd"] = spd
        # population forecast: peak congestion/headcount over the next horizon
        from collections import defaultdict
        fc: dict[str, list] = defaultdict(list)
        for cd, congest, pmax, ftime in con.execute(
            "SELECT area_cd, fcst_congest, fcst_ppltn_max, fcst_time FROM rt_population_forecast "
            "WHERE collected_at = (SELECT MAX(collected_at) FROM rt_population_forecast)"):
            if cd in pop and pmax is not None:
                fc[cd].append((float(pmax), congest, ftime))
        for cd, vals in fc.items():
            pk = max(vals, key=lambda v: v[0])
            pop[cd]["fcst_peak_ppltn"] = int(pk[0])
            pop[cd]["fcst_peak_congest"] = pk[1]
            pop[cd]["fcst_peak_time"] = pk[2]
        # subway accumulated boarding/alighting at the POI areas (rt_subway_crowd)
        for cd, gton, gtoff in con.execute(
            "SELECT area_cd, acml_gton_max, acml_gtoff_max FROM rt_subway_crowd "
            "WHERE collected_at = (SELECT MAX(collected_at) FROM rt_subway_crowd)"):
            if cd in pop:
                pop[cd]["subway_on"] = gton
                pop[cd]["subway_off"] = gtoff
        # bike totals (panel stat)
        b = con.execute(
            "SELECT total_stations, total_bikes, avg_shared_pct, collected_at "
            "FROM rt_bike_status ORDER BY collected_at DESC LIMIT 1").fetchone()
        bike = ({"stations": b[0], "bikes": b[1], "shared_pct": b[2], "collected_at": b[3]}
                if b else None)
    finally:
        con.close()
    return {"pois": list(pop.values()), "bike": bike,
            "collected_at": ts_pop, "traffic_at": ts_trf}


def main() -> int:
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    n = len(gj["pois"])
    n_trf = sum(1 for p in gj["pois"] if p.get("traffic_idx"))
    busiest = max(gj["pois"], key=lambda p: p.get("ppltn_max") or 0, default={})
    print(f"wrote {OUT.relative_to(ROOT)} ({n} POIs, {n_trf} with traffic, "
          f"@ {gj['collected_at']}; bike {gj['bike']['bikes'] if gj['bike'] else '—'}대)")
    print(f"  busiest: {busiest.get('area_nm')} "
          f"({busiest.get('ppltn_min')}–{busiest.get('ppltn_max')}, {busiest.get('congestion')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
