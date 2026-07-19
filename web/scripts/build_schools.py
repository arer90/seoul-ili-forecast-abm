#!/usr/bin/env python3
"""Build the Seoul school overlay for the map (gu-centroid approximation).

school_info (1,422 schools) carries gu_nm + school_kind + address but NO
coordinates, so each school is placed near its gu centroid via a deterministic
golden-angle spiral (even, reproducible spread within the district). This is an
APPROXIMATION — a school's dot marks its 자치구, not its exact address. Colored by
school kind (유치원/초/중/고/특수). Output (schools.json) → Map3D.tsx ScatterplotLayer.

Reproducible (DB read-only + committed gu geojson, no key, no RNG).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
GEOJSON = ROOT / "web" / "public" / "seoul-gu.geojson"
OUT = ROOT / "web" / "public" / "aggregates" / "schools.json"

#: school_kind → display color (RGB).
_KIND_COLOR = {
    "유치원": [255, 150, 200], "초등학교": [90, 200, 130], "중학교": [90, 150, 230],
    "고등학교": [240, 160, 70], "특수학교": [180, 110, 220],
}
_DEFAULT = [160, 160, 170]
_GOLDEN = 2.399963229728653  # golden angle (rad) — even non-overlapping spiral
_SPREAD = 0.014  # max radial offset (deg ≈ 1.5 km) from the gu centroid


def _gu_centroids() -> dict[str, list[float]]:
    from shapely.geometry import shape
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    out = {}
    for f in geo["features"]:
        nm = f["properties"].get("name") or next(iter(f["properties"].values()))
        c = shape(f["geometry"]).centroid
        out[nm] = [c.x, c.y]
    return out


def build() -> dict:
    from simulation.database import read_only_connect
    cents = _gu_centroids()
    con = read_only_connect(str(DB))
    try:
        rows = con.execute(
            "SELECT school_nm, school_kind, gu_nm FROM school_info "
            "WHERE gu_nm IS NOT NULL ORDER BY gu_nm, school_nm").fetchall()
    finally:
        con.close()
    by_gu: dict[str, list] = defaultdict(list)
    for nm, kind, gu in rows:
        if gu in cents:
            by_gu[gu].append((nm, kind))
    schools = []
    for gu, lst in by_gu.items():
        cx, cy = cents[gu]
        coslat = math.cos(math.radians(cy)) or 1.0
        n = len(lst)
        for i, (nm, kind) in enumerate(lst):
            r = _SPREAD * math.sqrt((i + 0.5) / n)
            theta = i * _GOLDEN
            schools.append({
                "name": nm, "kind": (kind or "").strip(), "gu": gu,
                "position": [round(cx + r * math.cos(theta) / coslat, 6),
                             round(cy + r * math.sin(theta), 6)],
                "color": _KIND_COLOR.get((kind or "").strip(), _DEFAULT),
            })
    return {"schools": schools, "kind_colors": _KIND_COLOR, "approx": "gu-centroid spiral"}


def main() -> int:
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    by = Counter(s["kind"] for s in gj["schools"])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(gj['schools'])} schools; {dict(by)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
