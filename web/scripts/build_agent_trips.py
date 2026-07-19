#!/usr/bin/env python3
"""Build real per-agent commute trips with route-snapping.

Agents travel along real Seoul transit infrastructure:
  - subway agents  → nearest station → subway network shortest path → dest station
  - bus agents     → nearest stop → bus-route polyline corridor → dest stop
  - walk agents    → a few waypoints (not straight 2-point line)

O-D is drawn from commuter_matrix.coupling (real KOSIS 25×25).
AM trip (home→work, 7-9h) and PM reverse trip (work→home, 18-20h) both generated.

Output: web/public/aggregates/agent-trips.json
  [{"path":[[lon,lat],...], "timestamps":[...], "group", "mode", "period", "color"}]

Reproducible (seed=42). ~600-900 agents.
"""
from __future__ import annotations

import heapq
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GEOJSON = ROOT / "web" / "public" / "seoul-gu.geojson"
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
SUBWAY_STATIONS = ROOT / "web" / "public" / "aggregates" / "subway-stations.json"
SUBWAY_LINES = ROOT / "web" / "public" / "aggregates" / "subway-lines.geojson"
BUS_ROUTES = ROOT / "web" / "public" / "aggregates" / "bus-routes.geojson"
BUS_STOPS = ROOT / "web" / "public" / "aggregates" / "bus-stops.json"
OUT = ROOT / "web" / "public" / "aggregates" / "agent-trips.json"

N_AGENTS = 750          # total agents before AM/PM doubling
# Cross-gu oversampling: the real KOSIS coupling is ~94% same-gu (walk).
# For animation visual interest we oversample cross-gu movers:
# ~60% cross-gu agents (transit-capable) + ~40% same/local (walk).
CROSS_GU_FRACTION = 0.60  # fraction of agents forced to cross-gu O-D
MAX_PATH_PTS = 35       # decimate path to at most this many points
TRAVEL_STEPS = 70       # animation steps per trip
AM_WINDOW = (0, 120)    # departure step range for AM (maps to 7-9h in animation)
PM_WINDOW = (180, 300)  # departure step range for PM (maps to 18-20h)

_GROUP = {0: "child", 1: "child", 2: "adult", 3: "adult",
          4: "adult", 5: "adult", 6: "elderly"}
_COLOR = {"child": [120, 200, 255], "adult": [255, 180, 80], "elderly": [255, 100, 140]}

# Mode colors (used as fallback if group color not enough distinction)
_MODE_COLOR = {
    "subway": [80, 160, 255],
    "bus":    [80, 220, 140],
    "walk":   [255, 200, 80],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _decimate(path: list[list[float]], max_pts: int) -> list[list[float]]:
    """Keep start, end, and evenly-spaced intermediate points up to max_pts."""
    if len(path) <= max_pts:
        return path
    step = (len(path) - 1) / (max_pts - 1)
    indices = {0, len(path) - 1}
    for i in range(1, max_pts - 1):
        indices.add(round(i * step))
    return [path[i] for i in sorted(indices)]


def _round_path(path: list[list[float]]) -> list[list[float]]:
    return [[round(p[0], 5), round(p[1], 5)] for p in path]


# ---------------------------------------------------------------------------
# GU centroids
# ---------------------------------------------------------------------------

def _load_centroids() -> tuple[list[str], dict[str, list[float]]]:
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    names: list[str] = []
    centroids: dict[str, list[float]] = {}
    for f in geo["features"]:
        p = f["properties"]
        name = p.get("name") or p.get("SIG_KOR_NM") or list(p.values())[0]
        names.append(name)
        coords = f["geometry"]["coordinates"]
        gtype = f["geometry"]["type"]
        # Compute centroid manually (avoid shapely dependency)
        if gtype == "Polygon":
            rings = [coords[0]]
        elif gtype == "MultiPolygon":
            rings = [poly[0] for poly in coords]
        else:
            rings = []
        all_pts: list[list[float]] = []
        for ring in rings:
            all_pts.extend(ring)
        if all_pts:
            cx = sum(pt[0] for pt in all_pts) / len(all_pts)
            cy = sum(pt[1] for pt in all_pts) / len(all_pts)
        else:
            cx, cy = 0.0, 0.0
        centroids[name] = [round(cx, 5), round(cy, 5)]
    return names, centroids


# ---------------------------------------------------------------------------
# Commuter O-D matrix from DB
# ---------------------------------------------------------------------------

def _load_od_matrix(gu_names: list[str]) -> np.ndarray:
    """Return (25, 25) matrix of coupling weights. Rows=origin, cols=dest."""
    from simulation.database import read_only_connect
    idx = {g: i for i, g in enumerate(gu_names)}
    n = len(gu_names)
    mat = np.zeros((n, n), dtype=float)
    con = read_only_connect(str(DB))
    try:
        rows = con.execute(
            "SELECT origin_gu, dest_gu, coupling FROM commuter_matrix"
        ).fetchall()
    finally:
        con.close()
    for origin_gu, dest_gu, coupling in rows:
        if origin_gu in idx and dest_gu in idx and coupling and coupling > 0:
            mat[idx[origin_gu], idx[dest_gu]] = float(coupling)
    # Normalize each row so probabilities sum to 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


# ---------------------------------------------------------------------------
# Subway graph
# ---------------------------------------------------------------------------

class SubwayNetwork:
    """Minimal subway graph: nodes=station index, edges=adjacent stations on same line."""

    def __init__(self) -> None:
        stations_raw = json.loads(SUBWAY_STATIONS.read_text(encoding="utf-8"))["stations"]
        lines_raw = json.loads(SUBWAY_LINES.read_text(encoding="utf-8"))["features"]

        self.stations: list[dict[str, Any]] = stations_raw  # list of station dicts
        self.n = len(stations_raw)

        # Map name → index
        self._name_idx: dict[str, int] = {s["name"]: i for i, s in enumerate(stations_raw)}
        self._pos: np.ndarray = np.array([s["position"] for s in stations_raw], dtype=float)

        # Build adjacency: edges between consecutive stations on the same line
        # subway-lines.geojson coords = ordered station positions along the line
        self._adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(self.n)}
        self._line_segs: dict[tuple[int, int], list[list[float]]] = {}  # (u,v) → path coords

        for feat in lines_raw:
            line_id = feat["properties"]["line"]
            coords = feat["geometry"]["coordinates"]  # ordered list of [lon, lat]
            # Match each coord to nearest station on this line
            line_stations = [s for s in stations_raw if line_id in s["lines"]]
            station_indices_ordered = self._match_coords_to_stations(coords, line_stations)
            # Add edges between consecutive matched stations
            for k in range(len(station_indices_ordered) - 1):
                u = station_indices_ordered[k]
                v = station_indices_ordered[k + 1]
                if u == v:
                    continue
                # Weight = km distance
                w = _haversine_km(
                    self._pos[u, 0], self._pos[u, 1],
                    self._pos[v, 0], self._pos[v, 1]
                )
                # Bidirectional
                self._adj[u].append((v, w))
                self._adj[v].append((u, w))
                # Store the line segment geometry (2-point segment between u,v)
                seg_u = self._pos[u].tolist()
                seg_v = self._pos[v].tolist()
                self._line_segs[(min(u, v), max(u, v))] = [seg_u, seg_v]

    def _match_coords_to_stations(
        self, coords: list[list[float]], line_stations: list[dict[str, Any]]
    ) -> list[int]:
        """Map ordered line coords to station indices (by nearest match)."""
        result = []
        for coord in coords:
            best_i = min(
                range(len(line_stations)),
                key=lambda k: (line_stations[k]["position"][0] - coord[0]) ** 2
                + (line_stations[k]["position"][1] - coord[1]) ** 2,
            )
            si = self._name_idx[line_stations[best_i]["name"]]
            result.append(si)
        # Deduplicate consecutive
        deduped = [result[0]]
        for x in result[1:]:
            if x != deduped[-1]:
                deduped.append(x)
        return deduped

    def nearest_station(self, lon: float, lat: float) -> int:
        """Index of closest station to given position."""
        dists = (self._pos[:, 0] - lon) ** 2 + (self._pos[:, 1] - lat) ** 2
        return int(np.argmin(dists))

    def shortest_path(self, src: int, dst: int) -> list[int] | None:
        """Dijkstra shortest path. Returns list of station indices or None."""
        if src == dst:
            return [src]
        dist: dict[int, float] = {src: 0.0}
        prev: dict[int, int] = {}
        heap = [(0.0, src)]
        visited: set[int] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                # Reconstruct
                path = []
                node = dst
                while node in prev:
                    path.append(node)
                    node = prev[node]
                path.append(src)
                return path[::-1]
            for v, w in self._adj[u]:
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return None

    def path_geometry(self, station_path: list[int]) -> list[list[float]]:
        """Return ordered [lon, lat] points following the station sequence."""
        pts: list[list[float]] = []
        for i, si in enumerate(station_path):
            pts.append(self._pos[si].tolist())
        return pts


# ---------------------------------------------------------------------------
# Bus routes
# ---------------------------------------------------------------------------

class BusNetwork:
    """Lightweight bus network: stops array + route polylines."""

    def __init__(self) -> None:
        stops_raw = json.loads(BUS_STOPS.read_text(encoding="utf-8"))["stops"]
        routes_raw = json.loads(BUS_ROUTES.read_text(encoding="utf-8"))["features"]

        self._stops_pos: np.ndarray = np.array(
            [s["position"] for s in stops_raw], dtype=float
        )  # (N, 2) lon/lat

        # Flatten MultiLineString routes into list of polylines
        self._routes: list[list[list[float]]] = []  # each route = list of [lon,lat]
        for feat in routes_raw:
            coords = feat["geometry"]["coordinates"]
            # MultiLineString: coords = list of segments
            flat: list[list[float]] = []
            for seg in coords:
                flat.extend(seg)
            if len(flat) >= 2:
                self._routes.append(flat)

        # Build a spatial index: for each stop, store its position
        # (no scipy KDTree needed — use numpy broadcasting for small N)

    def nearest_stop(self, lon: float, lat: float) -> np.ndarray:
        """Return [lon, lat] of nearest bus stop."""
        dists = (self._stops_pos[:, 0] - lon) ** 2 + (self._stops_pos[:, 1] - lat) ** 2
        return self._stops_pos[int(np.argmin(dists))].tolist()

    def best_route_corridor(
        self,
        home_lon: float, home_lat: float,
        work_lon: float, work_lat: float,
    ) -> list[list[float]] | None:
        """Find bus route whose polyline passes between home and work.

        Strategy: pick the route that minimises the sum of distances to its
        start-region and end-region, weighted by coverage of the O-D axis.
        Returns the clipped sub-polyline between the two nearest points on
        the route, or None if no suitable route found.
        """
        best_pts: list[list[float]] | None = None
        best_score = math.inf

        for route in self._routes:
            arr = np.array(route, dtype=float)
            # Distance from each route point to home and work
            d_home = np.sqrt(((arr[:, 0] - home_lon) ** 2 + (arr[:, 1] - home_lat) ** 2))
            d_work = np.sqrt(((arr[:, 0] - work_lon) ** 2 + (arr[:, 1] - work_lat) ** 2))
            i_home = int(np.argmin(d_home))
            i_work = int(np.argmin(d_work))

            # Must pass through two distinct points
            if i_home == i_work:
                continue

            # Score: sum of closest approach distances
            score = float(d_home[i_home]) + float(d_work[i_work])
            if score < best_score:
                best_score = score
                lo, hi = min(i_home, i_work), max(i_home, i_work)
                sub = route[lo: hi + 1]
                if len(sub) >= 2:
                    # Orient so we go home→work
                    if i_home > i_work:
                        sub = sub[::-1]
                    best_pts = sub

        return best_pts


# ---------------------------------------------------------------------------
# Walk path generator
# ---------------------------------------------------------------------------

def _walk_path(
    home: list[float], work: list[float], rng: random.Random
) -> list[list[float]]:
    """Generate a 4-6 waypoint walk path (not a straight 2-pt line)."""
    n_via = rng.randint(3, 5)
    pts = [home]
    for i in range(1, n_via):
        t = i / n_via
        # Interpolate + add random perpendicular jitter (proportional to distance)
        lon = home[0] + t * (work[0] - home[0])
        lat = home[1] + t * (work[1] - home[1])
        jitter_scale = 0.002  # ~200m in degree units
        lon += rng.gauss(0, jitter_scale * 0.5)
        lat += rng.gauss(0, jitter_scale * 0.5)
        pts.append([round(lon, 5), round(lat, 5)])
    pts.append(work)
    return pts


# ---------------------------------------------------------------------------
# Agent group → night population per gu
# ---------------------------------------------------------------------------

def _load_night_pop(gu_names: list[str]) -> np.ndarray:
    """Return (n_gu,) night population array using commuter_matrix.night_population."""
    from simulation.database import read_only_connect
    idx = {g: i for i, g in enumerate(gu_names)}
    n = len(gu_names)
    pop = np.zeros(n, dtype=float)
    con = read_only_connect(str(DB))
    try:
        rows = con.execute(
            "SELECT origin_gu, night_population FROM commuter_matrix "
            "GROUP BY origin_gu"
        ).fetchall()
    finally:
        con.close()
    for gu, night_pop in rows:
        if gu in idx and night_pop:
            pop[idx[gu]] = float(night_pop)
    if pop.sum() == 0:
        pop[:] = 1.0 / n
    else:
        pop /= pop.sum()
    return pop


# ---------------------------------------------------------------------------
# Mode assignment
# ---------------------------------------------------------------------------

def _assign_mode(
    home_gu: str, work_gu: str, group: str,
    gu_names: list[str], cents: dict[str, list[float]],
    rng: random.Random,
) -> str:
    """Assign transit mode based on O-D distance and age group."""
    h = cents.get(home_gu, [127.0, 37.5])
    w = cents.get(work_gu, [127.0, 37.5])
    dist_km = _haversine_km(h[0], h[1], w[0], w[1])

    # Children and elderly prefer walk / local
    if group == "child":
        return rng.choices(["walk", "bus"], weights=[0.6, 0.4])[0]
    if group == "elderly":
        return rng.choices(["walk", "bus", "subway"], weights=[0.4, 0.35, 0.25])[0]

    # Adults: distance-based
    if dist_km < 1.5:
        return rng.choices(["walk", "bus"], weights=[0.65, 0.35])[0]
    elif dist_km < 5.0:
        return rng.choices(["bus", "subway", "walk"], weights=[0.50, 0.35, 0.15])[0]
    else:
        return rng.choices(["subway", "bus"], weights=[0.60, 0.40])[0]


# ---------------------------------------------------------------------------
# Route-snapped path builders
# ---------------------------------------------------------------------------

def _build_subway_path(
    home: list[float], work: list[float],
    subway: SubwayNetwork,
) -> list[list[float]] | None:
    """Build subway path: home → nearest station → network route → dest station → work."""
    src = subway.nearest_station(home[0], home[1])
    dst = subway.nearest_station(work[0], work[1])
    if src == dst:
        return None

    station_path = subway.shortest_path(src, dst)
    if not station_path or len(station_path) < 2:
        return None

    route_pts = subway.path_geometry(station_path)
    # Prepend home → first station, append last station → work
    full = [home] + route_pts + [work]
    return _decimate(full, MAX_PATH_PTS)


def _build_bus_path(
    home: list[float], work: list[float],
    bus: BusNetwork,
) -> list[list[float]] | None:
    """Build bus path: home → nearest stop → route corridor → dest stop → work."""
    corridor = bus.best_route_corridor(home[0], home[1], work[0], work[1])
    if corridor is None or len(corridor) < 2:
        return None
    # Home → start of corridor → corridor → end of corridor → work
    full = [home] + corridor + [work]
    return _decimate(full, MAX_PATH_PTS)


def _build_walk_path(
    home: list[float], work: list[float],
    rng: random.Random,
) -> list[list[float]]:
    """Walk path with waypoints."""
    return _decimate(_walk_path(home, work, rng), MAX_PATH_PTS)


# ---------------------------------------------------------------------------
# Trip generation
# ---------------------------------------------------------------------------

def _make_timestamps(depart: int, n_pts: int) -> list[int]:
    """Assign timestamps spread evenly over TRAVEL_STEPS for n_pts waypoints."""
    if n_pts < 2:
        return [depart, depart + TRAVEL_STEPS]
    return [depart + round(i * TRAVEL_STEPS / (n_pts - 1)) for i in range(n_pts)]


def build() -> dict:
    """Build all trips. Returns dict with 'trips' list and 'groups' color map."""
    rng_np = np.random.default_rng(42)
    rng_py = random.Random(42)

    gu_names, cents = _load_centroids()
    night_pop = _load_night_pop(gu_names)
    od_mat = _load_od_matrix(gu_names)
    gu_idx = {g: i for i, g in enumerate(gu_names)}

    # Load transit networks
    print("  Loading subway network...", flush=True)
    subway = SubwayNetwork()
    print(f"    {subway.n} stations, {sum(len(v) for v in subway._adj.values()) // 2} edges")

    print("  Loading bus network...", flush=True)
    bus = BusNetwork()
    print(f"    {len(bus._routes)} routes, {len(bus._stops_pos)} stops")

    # Group weights: adult 60%, child 20%, elderly 20%
    group_weights = {
        "child":   0.20,
        "adult":   0.60,
        "elderly": 0.20,
    }

    trips: list[dict] = []
    n_attempted = 0
    n_subway = n_bus = n_walk = 0

    for _ in range(N_AGENTS):
        n_attempted += 1

        # --- Sample home gu by night population ---
        home_idx = int(rng_np.choice(len(gu_names), p=night_pop))
        home_gu = gu_names[home_idx]

        # --- Age group ---
        group = rng_py.choices(
            list(group_weights.keys()),
            weights=list(group_weights.values()),
        )[0]

        # --- Sample work gu by O-D coupling from home ---
        # Oversample cross-gu movers for visual transit animation interest.
        # Real KOSIS coupling is ~94% same-gu; we force CROSS_GU_FRACTION of
        # agents to pick a different dest gu (using real cross-gu weights).
        od_row = od_mat[home_idx].copy()
        if od_row.sum() <= 0:
            continue

        force_cross = rng_py.random() < CROSS_GU_FRACTION
        if force_cross:
            # Zero out self-coupling and re-normalize to get cross-gu distribution
            cross_row = od_row.copy()
            cross_row[home_idx] = 0.0
            if cross_row.sum() > 0:
                cross_row = cross_row / cross_row.sum()
                work_idx = int(rng_np.choice(len(gu_names), p=cross_row))
            else:
                # Fallback: uniform over other gus
                choices = [i for i in range(len(gu_names)) if i != home_idx]
                work_idx = rng_py.choice(choices)
        else:
            work_idx = int(rng_np.choice(len(gu_names), p=od_row))
        work_gu = gu_names[work_idx]

        # --- Children/elderly are more local (reduce cross-gu probability) ---
        if group in ("child", "elderly") and home_gu != work_gu:
            # 50% chance to reassign work to home gu (less aggressive than before)
            if rng_py.random() < 0.50:
                work_gu = home_gu
                work_idx = home_idx

        # Get centroids
        home_cent = cents.get(home_gu, [127.0, 37.5])
        work_cent = cents.get(work_gu, [127.0, 37.5])

        # Add a small jitter around centroid so agents don't all stack
        home_lon = home_cent[0] + rng_py.gauss(0, 0.008)
        home_lat = home_cent[1] + rng_py.gauss(0, 0.005)
        work_lon = work_cent[0] + rng_py.gauss(0, 0.008)
        work_lat = work_cent[1] + rng_py.gauss(0, 0.005)
        home_pt = [round(home_lon, 5), round(home_lat, 5)]
        work_pt = [round(work_lon, 5), round(work_lat, 5)]

        # --- Mode assignment ---
        if home_gu == work_gu:
            mode = "walk"
        else:
            mode = _assign_mode(home_gu, work_gu, group, gu_names, cents, rng_py)

        # --- Build AM path (home → work) ---
        am_path: list[list[float]] | None = None
        if mode == "subway":
            am_path = _build_subway_path(home_pt, work_pt, subway)
            if am_path is None:
                # Fallback to bus
                am_path = _build_bus_path(home_pt, work_pt, bus)
                if am_path is not None:
                    mode = "bus"
        if mode == "bus" and am_path is None:
            am_path = _build_bus_path(home_pt, work_pt, bus)
            if am_path is None:
                mode = "walk"
        if mode == "walk" or am_path is None:
            am_path = _build_walk_path(home_pt, work_pt, rng_py)
            mode = "walk"

        if am_path is None or len(am_path) < 2:
            continue

        # Ensure path points are valid
        am_path = _round_path(am_path)

        # PM path = reverse of AM path
        pm_path = am_path[::-1]

        color = _COLOR[group]

        # --- AM departure in 0-120 window (7-9h) ---
        am_depart = int(rng_np.integers(AM_WINDOW[0], AM_WINDOW[1]))
        am_ts = _make_timestamps(am_depart, len(am_path))

        # --- PM departure in 180-300 window (18-20h) ---
        pm_depart = int(rng_np.integers(PM_WINDOW[0], PM_WINDOW[1]))
        pm_ts = _make_timestamps(pm_depart, len(pm_path))

        trips.append({
            "path": am_path,
            "timestamps": am_ts,
            "group": group,
            "mode": mode,
            "period": "am",
            "color": color,
        })
        trips.append({
            "path": pm_path,
            "timestamps": pm_ts,
            "group": group,
            "mode": mode,
            "period": "pm",
            "color": color,
        })

        # Track mode counts (per agent, not trip)
        if mode == "subway":
            n_subway += 1
        elif mode == "bus":
            n_bus += 1
        else:
            n_walk += 1

    total_agents = n_subway + n_bus + n_walk
    print(f"  Generated {total_agents} agents × 2 periods = {len(trips)} trips")
    print(f"  Mode split — subway: {n_subway} ({100*n_subway/max(1,total_agents):.0f}%)"
          f"  bus: {n_bus} ({100*n_bus/max(1,total_agents):.0f}%)"
          f"  walk: {n_walk} ({100*n_walk/max(1,total_agents):.0f}%)")

    # Verify path points > 2
    pt_counts = [len(t["path"]) for t in trips]
    avg_pts = sum(pt_counts) / len(pt_counts) if pt_counts else 0
    multi_pt = sum(1 for c in pt_counts if c > 2)
    print(f"  Avg path points: {avg_pts:.1f}  |  paths with >2 pts: {multi_pt}/{len(trips)}")

    # Verify AM and PM both present
    am_cnt = sum(1 for t in trips if t["period"] == "am")
    pm_cnt = sum(1 for t in trips if t["period"] == "pm")
    print(f"  AM trips: {am_cnt}  |  PM trips: {pm_cnt}")

    # Sample subway route for verification
    subway_trips = [t for t in trips if t["mode"] == "subway" and t["period"] == "am"]
    if subway_trips:
        ex = subway_trips[0]
        print(f"  Sample subway AM path ({len(ex['path'])} pts): "
              f"{ex['path'][0]} → ... → {ex['path'][-1]}")

    return {"trips": trips, "groups": _COLOR}


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT))
    print("Building agent trips with real transit route-snapping...")
    gj = build()
    OUT.write_text(json.dumps(gj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    by_group: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    by_period: dict[str, int] = {}
    for t in gj["trips"]:
        by_group[t["group"]] = by_group.get(t["group"], 0) + 1
        by_mode[t["mode"]] = by_mode.get(t["mode"], 0) + 1
        by_period[t["period"]] = by_period.get(t["period"], 0) + 1
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"  total trips: {len(gj['trips'])} (each agent generates am+pm)")
    print(f"  by group:  {by_group}")
    print(f"  by mode:   {by_mode}")
    print(f"  by period: {by_period}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
