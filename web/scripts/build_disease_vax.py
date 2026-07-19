#!/usr/bin/env python3
"""Build per-gu influenza vaccination + notifiable-disease overlay for the map.

Two epidemiology choropleths keyed by the 25 Seoul 자치구:
  • vaccination_coverage — 인플루엔자 예방접종률(표준화율) %, latest ref_year,
    filtered to the 25 gu (the table also holds nationwide 시군구 + 보건소 권역
    rows like 서부/동부, which are dropped). Directly relevant to the ILI model.
  • seoul_disease_district — notifiable-disease 발생(계) cases per gu, summed
    over diseases for the latest year. NOTE the populated rows are largely city-
    aggregated (gu='서울시') and the flu entries are 신종/동물 인플루엔자, not
    seasonal flu, so per-gu coverage is best-effort; absent gu simply get no fill.

Reproducible (DB read-only, no key). Output (disease-vax.json) → Map3D.tsx
GeoJsonLayer choropleths.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT = ROOT / "web" / "public" / "aggregates" / "disease-vax.json"

#: The 25 Seoul 자치구 (filter — the source tables also carry nationwide rows).
SEOUL_GU = {
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구",
    "강동구",
}


def build() -> dict:
    from simulation.database import read_only_connect
    con = read_only_connect(str(DB))
    try:
        # vaccination: latest year, influenza, 25 gu only
        vyear = con.execute(
            "SELECT MAX(ref_year) FROM vaccination_coverage "
            "WHERE vaccine_nm LIKE '인플루엔자%'").fetchone()[0]
        vax = {}
        for gu, pct in con.execute(
            "SELECT gu_nm, coverage_pct FROM vaccination_coverage "
            "WHERE vaccine_nm LIKE '인플루엔자%' AND ref_year = ? AND coverage_pct IS NOT NULL",
            (vyear,)):
            if gu in SEOUL_GU:
                vax[gu] = round(float(pct), 1)
        # disease: latest year, 발생_계, per-gu sum over diseases (skip city total)
        dyear = con.execute(
            "SELECT MAX(year) FROM seoul_disease_district WHERE cases IS NOT NULL").fetchone()[0]
        disease = {}
        for gu, tot in con.execute(
            "SELECT gu_nm, SUM(cases) FROM seoul_disease_district "
            "WHERE category = '발생_계' AND year = ? AND cases IS NOT NULL "
            "GROUP BY gu_nm", (dyear,)):
            if gu in SEOUL_GU and tot:
                disease[gu] = int(tot)
    finally:
        con.close()
    return {"vax": vax, "vax_year": vyear, "vax_label": "인플루엔자 접종률(%)",
            "disease": disease, "disease_year": dyear,
            "disease_label": "법정감염병 발생(계)"}


def main() -> int:
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (vax {len(gj['vax'])} gu @ {gj['vax_year']}, "
          f"disease {len(gj['disease'])} gu @ {gj['disease_year']})")
    if gj["vax"]:
        lo = min(gj["vax"].items(), key=lambda x: x[1])
        hi = max(gj["vax"].items(), key=lambda x: x[1])
        print(f"  vax range: {lo[0]} {lo[1]}% .. {hi[0]} {hi[1]}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
