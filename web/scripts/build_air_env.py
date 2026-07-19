#!/usr/bin/env python3
"""Build per-gu real-time air-quality + environment overlay for the map.

Two live Seoul sources, latest snapshot each:
  • rt_air_quality  — 25-gu 대기질: pm10, pm25, o3, no2, so2, co, khai_grade
                      (통합대기환경지수). Drives the air-pollution choropleth.
  • rt_sdot_env     — S-DoT IoT sensors: temperature, humidity, pm10/pm25,
                      uv_index, noise, wind_speed, wind_dir. Aggregated per gu
                      (circular mean for wind direction) → wind arrows + temp.

gu_code→gu_nm comes from daily_population_gu_hourly. Output (air-env.json) is
consumed by Map3D.tsx (GeoJsonLayer choropleth + wind IconLayer). Reproducible
from the DB (read-only); no key.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT = ROOT / "web" / "public" / "aggregates" / "air-env.json"

_AIR_COLS = ["pm10", "pm25", "o3", "no2", "so2", "co", "khai_grade"]
_SDOT_COLS = ["temperature", "humidity", "pm10", "pm25", "uv_index", "noise", "wind_speed"]

#: rt_sdot_env.cgg is the romanized gu name; map to the Korean gu names used by
#: the choropleth + the rest of the DB (the 25 Seoul 자치구 — stable reference).
_CGG_KR = {
    "Jongno-gu": "종로구", "Jung-gu": "중구", "Yongsan-gu": "용산구",
    "Seongdong-gu": "성동구", "Gwangjin-gu": "광진구", "Dongdaemun-gu": "동대문구",
    "Jungnang-gu": "중랑구", "Seongbuk-gu": "성북구", "Gangbuk-gu": "강북구",
    "Dobong-gu": "도봉구", "Nowon-gu": "노원구", "Eunpyeong-gu": "은평구",
    "Seodaemun-gu": "서대문구", "Mapo-gu": "마포구", "Yangcheon-gu": "양천구",
    "Gangseo-gu": "강서구", "Guro-gu": "구로구", "Geumcheon-gu": "금천구",
    "Yeongdeungpo-gu": "영등포구", "Dongjak-gu": "동작구", "Gwanak-gu": "관악구",
    "Seocho-gu": "서초구", "Gangnam-gu": "강남구", "Songpa-gu": "송파구",
    "Gangdong-gu": "강동구",
}


def _latest_air(con) -> dict:
    """Latest air-quality row per gu (location_nm)."""
    cols = ", ".join(_AIR_COLS)
    rows = con.execute(
        f"SELECT location_nm, {cols}, collected_at FROM rt_air_quality r "
        "WHERE collected_at = (SELECT MAX(collected_at) FROM rt_air_quality "
        "                      WHERE location_nm = r.location_nm)"
    ).fetchall()
    air = {}
    for row in rows:
        gu = row[0]
        air[gu] = {c: row[i + 1] for i, c in enumerate(_AIR_COLS)}
    return air


def _latest_env(con) -> dict:
    """Latest S-DoT sensors, aggregated per gu (means; circular mean wind_dir)."""
    cols = ", ".join(_SDOT_COLS)
    rows = con.execute(
        f"SELECT cgg, {cols}, wind_dir FROM rt_sdot_env "
        "WHERE collected_at = (SELECT MAX(collected_at) FROM rt_sdot_env)"
    ).fetchall()
    def _f(v):
        """Float or None (S-DoT writes '' for missing sensor readings)."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    acc: dict[str, dict] = {}
    for row in rows:
        gu = _CGG_KR.get(row[0])
        if not gu:
            continue
        a = acc.setdefault(gu, {c: [] for c in _SDOT_COLS} | {"_sin": [], "_cos": []})
        for i, c in enumerate(_SDOT_COLS):
            v = _f(row[i + 1])
            if v is not None:
                a[c].append(v)
        wd = _f(row[-1])
        if wd is not None:
            r = math.radians(wd)
            a["_sin"].append(math.sin(r))
            a["_cos"].append(math.cos(r))
    env = {}
    for gu, a in acc.items():
        e = {c: round(sum(a[c]) / len(a[c]), 1) for c in _SDOT_COLS if a[c]}
        if a["_sin"]:
            e["wind_dir"] = round(math.degrees(math.atan2(
                sum(a["_sin"]) / len(a["_sin"]), sum(a["_cos"]) / len(a["_cos"]))) % 360, 0)
        env[gu] = e
    return env


def build() -> dict:
    from simulation.database import read_only_connect
    con = read_only_connect(str(DB))
    try:
        air = _latest_air(con)
        env = _latest_env(con)
        ts = con.execute("SELECT MAX(collected_at) FROM rt_air_quality").fetchone()[0]
    finally:
        con.close()
    # national PM10 grade bands (환경부): good≤30, moderate≤80, bad≤150, vbad>150
    return {"air": air, "env": env, "collected_at": ts,
            "pm10_bands": [30, 80, 150], "pm25_bands": [15, 35, 75]}


def main() -> int:
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    na = len(gj["air"])
    ne = len(gj["env"])
    sample = next(iter(gj["air"].items()), ("—", {}))
    print(f"wrote {OUT.relative_to(ROOT)} ({na} gu air, {ne} gu env, "
          f"@ {gj['collected_at']})")
    print(f"  sample air {sample[0]}: {sample[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
