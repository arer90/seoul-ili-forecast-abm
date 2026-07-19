"""
build_seoul_boundary.py — union the 25 Seoul gu polygons into a single
outer-ring FeatureCollection for the Frame D map overlay.

Why this script exists:
  The 25-gu geojson is the ground truth for choropleth fills, but drawing
  every internal district border as the "Seoul boundary" is visually
  noisy — we only want the outer silhouette. shapely's ``unary_union``
  merges the 25 Polygons into a MultiPolygon / Polygon, after which we
  strip the interior rings (gu-to-gu seams) by taking only each exterior
  ring and snap near-coincident points to dedupe float noise.

Output:
  web/public/aggregates/seoul-boundary.geojson
  — FeatureCollection with one Feature (Polygon or MultiPolygon) +
    a ``name = 'Seoul'`` property so leaflet's bind tooltip can use it.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "web" / "public" / "seoul-gu.geojson"
    dst = repo_root / "web" / "public" / "aggregates" / "seoul-boundary.geojson"

    with src.open("r", encoding="utf-8") as fp:
        fc = json.load(fp)

    polygons = [shape(feat["geometry"]) for feat in fc["features"]]
    print(f"[build_seoul_boundary] loaded {len(polygons)} gu polygons")

    # Buffer(0) heals float imprecision at shared edges before union so
    # the resulting geometry is a clean single polygon instead of a
    # mosaic with hairline cracks.
    merged = unary_union([p.buffer(0) for p in polygons])
    print(f"[build_seoul_boundary] merged type = {merged.geom_type}")

    # Collapse to a single Polygon if the float-snap leaves us with a
    # MultiPolygon of tiny slivers; keep the largest piece.
    if merged.geom_type == "MultiPolygon":
        largest = max(merged.geoms, key=lambda g: g.area)
        total = sum(g.area for g in merged.geoms)
        frac = largest.area / total if total else 0.0
        print(
            f"[build_seoul_boundary] largest piece = {frac * 100:.3f}% of total area"
        )
        if frac > 0.98:
            merged = largest

    # Strip interior rings — we want the outer silhouette only.
    def outer_only(g):
        if g.geom_type == "Polygon":
            return {"type": "Polygon", "coordinates": [list(g.exterior.coords)]}
        if g.geom_type == "MultiPolygon":
            return {
                "type": "MultiPolygon",
                "coordinates": [[list(p.exterior.coords)] for p in g.geoms],
            }
        raise ValueError(f"unexpected geom {g.geom_type}")

    geom = outer_only(merged)

    out = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Seoul",
                    "kr_name": "서울특별시",
                    "source": "derived from seoul-gu.geojson (unary_union)",
                },
                "geometry": geom,
            }
        ],
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False)
    size = dst.stat().st_size
    print(f"[build_seoul_boundary] wrote {dst} ({size:,} bytes)")


if __name__ == "__main__":
    main()
