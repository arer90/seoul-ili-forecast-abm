#!/usr/bin/env python3
"""Generate hospitals.json for the Seoul map overlay.

Sources:
  - hospitals table: 404 rows, all Seoul, lat/lng present for all.
    Columns: inst_nm, gu_nm, clcd_nm, lat, lng, bed_cnt.
  - emergency_room_availability table: 53 Seoul ER institutions (NEDIS).
    No lat/lng stored → matched by name to hospitals table.

Strategy:
  1. Build a name→(lat, lng, gu_nm, clcd_nm, bed_cnt) lookup from hospitals table.
  2. Collect the 53 Seoul ER names from emergency_room_availability.
  3. For each ER name, attempt exact match, then whitespace-normalised match,
     then partial-keyword match against hospitals lookup.
  4. Mark matched hospitals as is_er=True.
  5. Include ALL hospitals (종합병원, 상급종합 + 병원 with beds) with real coords.
     Exclude 요양병원 and 정신병원 to keep the map readable.

Output: web/public/aggregates/hospitals.json
  [{"name":str, "gu":str, "lat":float, "lon":float, "is_er":bool,
    "beds":int|null, "type":str}, ...]

Reproducible: DB read-only, no RNG, no hardcoded coordinates.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT = ROOT / "web" / "public" / "aggregates" / "hospitals.json"

# Hospital types to include (exclude 요양병원 / 정신병원 for map readability)
INCLUDE_TYPES = {"상급종합", "종합병원", "병원"}

# ER keyword fingerprints for fuzzy matching (shorter canonical substrings)
_ER_KEYWORDS = [
    "강남세브란스", "보훈", "양지병원", "상계백병원", "서울아산",
    "여의도성모", "강남성심", "성애병원", "순천향", "은평성모",
    "동부병원", "동신병원", "청구성심", "한일병원", "안암병원",
]


def _normalise(s: str) -> str:
    """Strip legal entity prefixes and collapse whitespace."""
    s = re.sub(r"(학교법인|의료법인|재단법인|사회복지재단|의료재단)\s*", "", s)
    return re.sub(r"\s+", "", s).strip()


def build() -> list[dict]:
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()

    # --- 1. Load all hospitals with coords ---
    cur.execute(
        "SELECT inst_nm, gu_nm, clcd_nm, lat, lng, bed_cnt "
        "FROM hospitals WHERE lat IS NOT NULL AND lng IS NOT NULL"
    )
    all_hospitals = cur.fetchall()

    # Build lookup: normalised name → row
    hosp_lookup: dict[str, tuple] = {}
    for row in all_hospitals:
        key = _normalise(row[0])
        hosp_lookup[key] = row

    # --- 2. Collect Seoul ER names ---
    cur.execute(
        "SELECT DISTINCT hp_nm FROM emergency_room_availability "
        "WHERE sido_nm LIKE '%서울%'"
    )
    er_names_raw: list[str] = [r[0] for r in cur.fetchall()]
    conn.close()

    # --- 3. Match ER names to hospitals ---
    er_set: set[str] = set()  # inst_nm of matched ERs

    for er_nm in er_names_raw:
        # Exact match
        if er_nm in {r[0] for r in all_hospitals}:
            er_set.add(er_nm)
            continue
        # Normalised match
        er_norm = _normalise(er_nm)
        if er_norm in hosp_lookup:
            er_set.add(hosp_lookup[er_norm][0])
            continue
        # Partial keyword match
        matched = False
        for kw in _ER_KEYWORDS:
            if kw in er_nm:
                for row in all_hospitals:
                    if kw in row[0]:
                        er_set.add(row[0])
                        matched = True
                        break
            if matched:
                break
        if not matched:
            # Last resort: check if any hospital name is a substring of the ER name
            for row in all_hospitals:
                norm_h = _normalise(row[0])
                if len(norm_h) > 4 and norm_h in _normalise(er_nm):
                    er_set.add(row[0])
                    break

    # --- 4. Build output list ---
    result: list[dict] = []
    for inst_nm, gu_nm, clcd_nm, lat, lng, bed_cnt in all_hospitals:
        if clcd_nm not in INCLUDE_TYPES:
            continue
        result.append({
            "name": inst_nm,
            "gu": gu_nm or "",
            "lat": round(lat, 7),
            "lon": round(lng, 7),
            "is_er": inst_nm in er_set,
            "beds": int(bed_cnt) if bed_cnt else None,
            "type": clcd_nm or "",
        })

    return result


def main() -> int:
    hospitals = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hospitals, ensure_ascii=False, indent=None), encoding="utf-8")

    total = len(hospitals)
    er_count = sum(1 for h in hospitals if h["is_er"])
    by_type: dict[str, int] = {}
    for h in hospitals:
        by_type[h["type"]] = by_type.get(h["type"], 0) + 1

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  total hospitals: {total}")
    print(f"  emergency room (is_er=True): {er_count}")
    print(f"  by type: {by_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
