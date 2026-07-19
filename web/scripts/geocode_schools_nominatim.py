#!/usr/bin/env python3
"""Geocode Seoul schools via Nominatim (OSM) and write schools.json.

Strategy:
  1. Query DB for school name + address (school_info table).
  2. For each school: try school_name search first, then road address fallback.
  3. Cache results in web/scripts/_geocode_cache.json (resume on interruption).
  4. Write web/public/aggregates/schools.json with real coordinates.

Rate limit: 1 req/sec per Nominatim policy.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT_PATH = ROOT / "web" / "public" / "aggregates" / "schools.json"
CACHE_PATH = Path(__file__).parent / "_geocode_cache.json"
GEOJSON = ROOT / "web" / "public" / "seoul-gu.geojson"

USER_AGENT = "MPH-school-geocoder/1.0 (major1106ai@gmail.com; academic non-commercial)"
SLEEP_SEC = 1.1  # nominatim policy: max 1 req/s

_KIND_COLOR: dict[str, list[int]] = {
    "유치원":   [255, 150, 200],
    "초등학교": [90,  200, 130],
    "중학교":   [90,  150, 230],
    "고등학교": [240, 160,  70],
    "특수학교": [180, 110, 220],
}
_DEFAULT_COLOR = [160, 160, 170]

# Golden-angle spiral fallback params
_GOLDEN = 2.399963229728653
_SPREAD = 0.014


def _nominatim_search(query: str, timeout: int = 8) -> tuple[float, float] | None:
    """Search Nominatim for a place string, return (lat, lon) or None.

    Args:
        query: search string (school name or address).
        timeout: HTTP timeout seconds.

    Returns:
        (lat, lon) tuple of floats, or None if not found / error.
    """
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": 1,
        "countrycodes": "kr",
        "addressdetails": "0",
    })
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def _gu_centroids() -> dict[str, list[float]]:
    """Read gu centroids from seoul-gu.geojson.

    Returns:
        {gu_name: [lon, lat]} dict.
    """
    from shapely.geometry import shape  # type: ignore[import]
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {}
    for f in geo["features"]:
        nm = f["properties"].get("name") or next(iter(f["properties"].values()))
        c = shape(f["geometry"]).centroid
        out[nm] = [c.x, c.y]
    return out


def _spiral_position(cx: float, cy: float, i: int, n: int) -> list[float]:
    coslat = math.cos(math.radians(cy)) or 1.0
    r = _SPREAD * math.sqrt((i + 0.5) / max(n, 1))
    theta = i * _GOLDEN
    return [
        round(cx + r * math.cos(theta) / coslat, 6),
        round(cy + r * math.sin(theta), 6),
    ]


def load_schools_from_db() -> list[tuple[str, str, str, str]]:
    """Load (name, kind, gu, address) rows from school_info.

    Returns:
        List of (name, kind, gu, address) tuples for geocoding targets.
    """
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT school_nm, school_kind, gu_nm, address "
            "FROM school_info "
            "WHERE address IS NOT NULL AND address != '' "
            "ORDER BY school_kind, gu_nm, school_nm"
        ).fetchall()
    finally:
        con.close()
    return [(r[0] or "", r[1] or "", r[2] or "", r[3] or "") for r in rows]


def geocode_all(schools: list[tuple[str, str, str, str]]) -> dict[str, list[float]]:
    """Geocode all schools using Nominatim with cache.

    Args:
        schools: list of (name, kind, gu, address) tuples.

    Returns:
        {school_name: [lon, lat]} dict for successfully geocoded schools.
    """
    # load existing cache
    cache: dict[str, list[float] | None] = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            print(f"[geocode] loaded cache: {len(cache)} entries")
        except Exception:
            cache = {}

    total = len(schools)
    done = 0
    hit = 0
    fail = 0

    for i, (name, kind, gu, address) in enumerate(schools):
        key = f"{name}|{address}"
        if key in cache:
            done += 1
            if cache[key] is not None:
                hit += 1
            continue

        # --- try 1: school name + 서울
        result = _nominatim_search(f"{name} 서울")
        time.sleep(SLEEP_SEC)

        if result is None:
            # --- try 2: road address
            result = _nominatim_search(address)
            time.sleep(SLEEP_SEC)

        if result is not None:
            lat, lon = result
            # sanity: Seoul bounding box 37.4~37.72 / 126.7~127.2
            if 37.4 <= lat <= 37.72 and 126.7 <= lon <= 127.2:
                cache[key] = [lon, lat]
                hit += 1
            else:
                cache[key] = None  # out-of-Seoul result
                fail += 1
        else:
            cache[key] = None
            fail += 1

        done += 1

        # save cache every 50 entries
        if done % 50 == 0:
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            pct = 100 * done / total
            print(f"[geocode] {done}/{total} ({pct:.0f}%) — hit={hit} fail={fail}")

    # final cache save
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"[geocode] done: {done} total, hit={hit}, fail={fail}")
    return {k.split("|")[0]: v for k, v in cache.items() if v is not None}


def build_json(
    schools: list[tuple[str, str, str, str]],
    coords: dict[str, list[float]],
) -> None:
    """Build and write schools.json.

    Args:
        schools: list of (name, kind, gu, address).
        coords: {school_name: [lon, lat]} for geocoded schools.
    """
    # load centroids for fallback
    try:
        centroids = _gu_centroids()
    except Exception:
        centroids = {}

    # count per-gu for spiral fallback indexing
    gu_counts: dict[str, int] = defaultdict(int)
    gu_idx: dict[str, int] = defaultdict(int)
    for nm, kind, gu, addr in schools:
        if nm not in coords:
            gu_counts[gu] += 1

    records: list[dict] = []
    geocoded_count = 0
    fallback_count = 0

    for nm, kind, gu, addr in schools:
        color = _KIND_COLOR.get(kind.strip(), _DEFAULT_COLOR)
        if nm in coords:
            records.append({
                "name": nm.strip(),
                "kind": kind.strip(),
                "gu": gu.strip(),
                "position": coords[nm],
                "color": color,
            })
            geocoded_count += 1
        elif gu in centroids:
            cx, cy = centroids[gu]
            idx = gu_idx[gu]
            n = gu_counts[gu]
            gu_idx[gu] += 1
            records.append({
                "name": nm.strip(),
                "kind": kind.strip(),
                "gu": gu.strip(),
                "position": _spiral_position(cx, cy, idx, n),
                "color": color,
                "approx_fallback": True,
            })
            fallback_count += 1

    approx = "real-geocoded" if geocoded_count > fallback_count else "mixed"

    out = {
        "schools": records,
        "kind_colors": _KIND_COLOR,
        "approx": approx,
        "geocoded_count": geocoded_count,
        "fallback_count": fallback_count,
        "source": "Nominatim/OSM",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    by_kind = Counter(s["kind"] for s in records)
    print(
        f"\n[schools.json] wrote {OUT_PATH.relative_to(ROOT)}\n"
        f"  total={len(records)}, geocoded={geocoded_count}, "
        f"spiral-fallback={fallback_count}\n"
        f"  approx={approx!r}\n"
        f"  by kind: {dict(by_kind)}"
    )


def main() -> None:
    schools = load_schools_from_db()
    print(f"[main] {len(schools)} schools loaded from DB")

    coords = geocode_all(schools)
    print(f"[main] {len(coords)} schools geocoded successfully")

    build_json(schools, coords)


if __name__ == "__main__":
    main()
