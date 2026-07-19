#!/usr/bin/env python3
"""Generate schools.json for the Seoul school map overlay.

Coordinate strategy (honest):
  1. PRIMARY — try DB columns lat / lon / latitude / longitude / x_coord /
     y_coord on school_info and school_info_seoul.  As of 2026-06-08 neither
     table carries coordinates; the NEIS API that originally populated them
     returns only address strings.
  2. FALLBACK — gu-centroid golden-angle spiral (same algorithm as the old
     build_schools.py).  Each dot marks the 자치구 district, NOT the building.
     The approx field in the output JSON is set to "gu-centroid spiral" so the
     map legend can display a disclaimer to the user.

If a future DB update adds real coordinates (e.g. after running a Kakao / NAVER
geocoding pass), this script will automatically prefer them without any code
change — just re-run it.

Usage:
    python web/scripts/gen_schools_json.py [--db PATH] [--out PATH]

Output: web/public/aggregates/schools.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
DEFAULT_OUT = ROOT / "web" / "public" / "aggregates" / "schools.json"
GEOJSON = ROOT / "web" / "public" / "seoul-gu.geojson"

# ── display colours (RGB) ────────────────────────────────────────────────────
_KIND_COLOR: dict[str, list[int]] = {
    "유치원":   [255, 150, 200],
    "초등학교": [90,  200, 130],
    "중학교":   [90,  150, 230],
    "고등학교": [240, 160,  70],
    "특수학교": [180, 110, 220],
}
_DEFAULT_COLOR = [160, 160, 170]

# golden angle (rad) — gives a non-overlapping spiral spread
_GOLDEN = 2.399963229728653
_SPREAD = 0.014  # max radial offset (deg ≈ 1.5 km) from gu centroid


# ── helpers ──────────────────────────────────────────────────────────────────

def _gu_centroids() -> dict[str, list[float]]:
    """Read gu centroids from the committed seoul-gu.geojson.

    Returns:
        dict mapping gu name (e.g. "강남구") to [lon, lat].

    Raises:
        FileNotFoundError: if seoul-gu.geojson is missing.
        ImportError: if shapely is not installed.
    """
    from shapely.geometry import shape  # type: ignore[import]
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {}
    for f in geo["features"]:
        nm = f["properties"].get("name") or next(iter(f["properties"].values()))
        c = shape(f["geometry"]).centroid
        out[nm] = [c.x, c.y]
    return out


def _spiral_positions(
    schools_by_gu: dict[str, list[tuple[str, str]]],
    centroids: dict[str, list[float]],
) -> list[dict]:
    """Place each school on a golden-angle spiral around its gu centroid.

    This is an approximation: dots mark the 자치구, not the real building.

    Args:
        schools_by_gu: {gu_name: [(school_name, school_kind), ...]}.
        centroids: {gu_name: [lon, lat]}.

    Returns:
        List of school dicts ready for the JSON output.
    """
    records: list[dict] = []
    for gu, lst in schools_by_gu.items():
        if gu not in centroids:
            continue
        cx, cy = centroids[gu]
        coslat = math.cos(math.radians(cy)) or 1.0
        n = len(lst)
        for i, (nm, kind) in enumerate(lst):
            r = _SPREAD * math.sqrt((i + 0.5) / n)
            theta = i * _GOLDEN
            records.append({
                "name": nm,
                "kind": (kind or "").strip(),
                "gu": gu,
                "position": [
                    round(cx + r * math.cos(theta) / coslat, 6),
                    round(cy + r * math.sin(theta), 6),
                ],
                "color": _KIND_COLOR.get((kind or "").strip(), _DEFAULT_COLOR),
            })
    return records


# ── coordinate detection ─────────────────────────────────────────────────────

_COORD_CANDIDATES = {
    "lat":       ("lat",       "lon"),
    "latitude":  ("latitude",  "longitude"),
    "lat_wgs84": ("lat_wgs84", "lon_wgs84"),
    "y_coord":   ("y_coord",   "x_coord"),
    "y_wgs84":   ("y_wgs84",   "x_wgs84"),
}

def _detect_coord_cols(cursor: sqlite3.Cursor, table: str) -> tuple[str, str] | None:
    """Return (lat_col, lon_col) if the table has recognisable coordinate columns.

    Args:
        cursor: open SQLite cursor.
        table: table name to inspect.

    Returns:
        (lat_col, lon_col) tuple or None if no coordinates found.
    """
    cols = {row[1].lower() for row in cursor.execute(f"PRAGMA table_info({table})")}
    for lat_name, (lat_c, lon_c) in _COORD_CANDIDATES.items():
        if lat_c in cols and lon_c in cols:
            return lat_c, lon_c
    return None


def _try_real_coords(
    db_path: str,
) -> tuple[list[dict] | None, str]:
    """Attempt to read real coordinates from DB school tables.

    Tries school_info first, then school_info_seoul.

    Args:
        db_path: path to epi_real_seoul.db.

    Returns:
        (records, source_description) where records is None if no coordinates
        were found.  source_description is a human-readable string for logging.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        for table, name_col, kind_col, gu_col in [
            ("school_info",       "school_nm",   "school_kind", "gu_nm"),
            ("school_info_seoul", "school_name", "school_type", "gu_name"),
        ]:
            coord_cols = _detect_coord_cols(cur, table)
            if coord_cols is None:
                continue
            lat_c, lon_c = coord_cols
            rows = cur.execute(
                f"SELECT {name_col}, {kind_col}, {gu_col}, {lat_c}, {lon_c} "
                f"FROM {table} "
                f"WHERE {gu_col} IS NOT NULL "
                f"  AND {lat_c} IS NOT NULL AND {lon_c} IS NOT NULL"
            ).fetchall()
            if not rows:
                continue
            records = []
            for nm, kind, gu, lat, lon in rows:
                if lat is None or lon is None:
                    continue
                records.append({
                    "name": (nm or "").strip(),
                    "kind": (kind or "").strip(),
                    "gu": (gu or "").strip(),
                    "position": [round(float(lon), 6), round(float(lat), 6)],
                    "color": _KIND_COLOR.get((kind or "").strip(), _DEFAULT_COLOR),
                })
            if records:
                return records, f"DB table={table}, cols=({lat_c},{lon_c}), n={len(records)}"
        return None, "no lat/lon columns found in school_info or school_info_seoul"
    finally:
        con.close()


# ── main ─────────────────────────────────────────────────────────────────────

def build(db_path: str, out_path: Path) -> int:
    """Build schools.json, preferring real DB coordinates over spiral fallback.

    Args:
        db_path: path to epi_real_seoul.db.
        out_path: output path for schools.json.

    Returns:
        Exit code (0 = success).
    """
    # 1. Try real DB coordinates first.
    records, coord_source = _try_real_coords(db_path)

    if records is not None:
        approx = "real-db"
        print(f"[gen_schools] Using REAL coordinates from DB ({coord_source}).")
    else:
        # 2. Honest fallback: gu-centroid spiral.
        print(
            f"[gen_schools] WARNING: {coord_source}.\n"
            "  School coordinates are NOT available in the DB.\n"
            "  Falling back to gu-centroid golden-angle spiral.\n"
            "  Each dot marks the 자치구 district, not the real building.\n"
            "  To get real coordinates, run a Kakao/NAVER geocoding pass on\n"
            "  the 'address' column and store results in school_info.lat/lon."
        )
        approx = "gu-centroid spiral"
        try:
            centroids = _gu_centroids()
        except ImportError:
            print("  [error] shapely not installed — pip install shapely", file=sys.stderr)
            return 1
        except FileNotFoundError as e:
            print(f"  [error] {e}", file=sys.stderr)
            return 1

        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT school_nm, school_kind, gu_nm FROM school_info "
                "WHERE gu_nm IS NOT NULL ORDER BY gu_nm, school_nm"
            ).fetchall()
        finally:
            con.close()

        by_gu: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for nm, kind, gu in rows:
            if gu in centroids:
                by_gu[gu].append((nm, kind or ""))
        records = _spiral_positions(by_gu, centroids)

    out = {
        "schools": records,
        "kind_colors": _KIND_COLOR,
        "approx": approx,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    by_kind = Counter(s["kind"] for s in records)
    print(
        f"[gen_schools] wrote {out_path.relative_to(ROOT)} "
        f"({len(records)} schools, approx={approx!r})\n"
        f"  by kind: {dict(by_kind)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db",  default=str(DEFAULT_DB), help="path to epi_real_seoul.db")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output schools.json path")
    args = p.parse_args(argv)
    return build(args.db, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
