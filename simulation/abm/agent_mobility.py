"""Per-agent time-resolved location provider (branch A, first increment).

The full branch-A goal is per-agent movement along real subway/bus routes. This
module is the validated DATA FOUNDATION for it: the real
``daily_population_gu_hourly`` table carries population by hour, district AND age
band, so it directly encodes WHERE agents of each age are at each time of day —
the distribution a per-agent mover would sample from. The increment exposes that
age × time × district location distribution and validates its central feature:
working-age people commute INTO the business districts during the day while
children and the elderly stay residential. A per-agent kernel can then draw each
agent's daytime district from this checked distribution.

Additive (no agent_kernel change). The actual per-agent route sampling inside
``agent_kernel`` is the remaining future work. Never raises in the analysis layer.
Companion to ``mobility_timeresolved`` (district swing) and ``age_validation``.
"""
from __future__ import annotations

import numpy as np

# age-band population columns of daily_population_gu_hourly
_AGE_COLS = ["pop_0_9", "pop_10_19", "pop_20_29", "pop_30_39",
             "pop_40_49", "pop_50_59", "pop_60_69", "pop_70plus"]
_WORKING = {"pop_20_29", "pop_30_39", "pop_40_49", "pop_50_59"}
_BUSINESS_HUBS = ["중구", "종로구", "강남구", "서초구", "영등포구"]


def load_age_hub_shares(db_path: str, *, day_hours=(10, 17),
                        night_hours=(0, 6)) -> dict[str, dict]:
    """Per age band, the fraction of that band's population located in the
    business hubs during day vs night, and the day−night shift.

    Returns ``{age_col: {day_share, night_share, shift}}``. ``shift > 0`` ⇒ the
    band concentrates into the central business districts in the daytime (it
    commutes in). Never raises (DB error → ``{}``)."""
    try:
        from simulation.database import read_only_connect
        con = read_only_connect(db_path)
    except Exception:
        return {}
    out: dict[str, dict] = {}
    hub_ph = ",".join("?" * len(_BUSINESS_HUBS))
    try:
        for col in _AGE_COLS:
            q = (f"SELECT "
                 f" SUM(CASE WHEN hour BETWEEN ? AND ? AND gu_nm IN ({hub_ph}) THEN {col} END), "
                 f" SUM(CASE WHEN hour BETWEEN ? AND ? THEN {col} END), "
                 f" SUM(CASE WHEN hour BETWEEN ? AND ? AND gu_nm IN ({hub_ph}) THEN {col} END), "
                 f" SUM(CASE WHEN hour BETWEEN ? AND ? THEN {col} END) "
                 f"FROM daily_population_gu_hourly")
            params = [day_hours[0], day_hours[1], *_BUSINESS_HUBS,
                      day_hours[0], day_hours[1],
                      night_hours[0], night_hours[1], *_BUSINESS_HUBS,
                      night_hours[0], night_hours[1]]
            try:
                hd, td, hn, tn = con.execute(q, params).fetchone()
            except Exception:
                continue
            if not td or not tn:
                continue
            ds, ns = float(hd or 0) / td, float(hn or 0) / tn
            out[col] = {"day_share": round(ds, 4), "night_share": round(ns, 4),
                        "shift": round(ds - ns, 4)}
    finally:
        con.close()
    return out


# model age band (AGE_BAND_LABELS index) → daily_population_gu_hourly columns
_BAND_TO_COLS = {0: ["pop_0_9"], 1: ["pop_10_19"], 2: ["pop_20_29"], 3: ["pop_30_39"],
                 4: ["pop_40_49"], 5: ["pop_50_59"], 6: ["pop_60_69", "pop_70plus"]}


def load_daytime_location_dist(db_path: str, gu_names: list[str], *,
                               day_hours=(10, 17)) -> np.ndarray:
    """Real daytime location distribution by age band: a (7, |gu|) matrix where
    row ``b`` is the probability an age-band-``b`` person is in each district
    during the day, from ``daily_population_gu_hourly``. This is what a per-agent
    mover samples its daytime district from. Rows sum to 1 (uniform fallback if a
    band is missing). Never raises."""
    n_gu = len(gu_names)
    idx = {g: i for i, g in enumerate(gu_names)}
    dist = np.zeros((7, n_gu), dtype=np.float64)
    try:
        from simulation.database import read_only_connect
        con = read_only_connect(db_path)
    except Exception:
        dist[:] = 1.0 / n_gu
        return dist
    try:
        for b, cols in _BAND_TO_COLS.items():
            expr = " + ".join(cols)
            for gu, pop in con.execute(
                f"SELECT gu_nm, SUM({expr}) FROM daily_population_gu_hourly "
                f"WHERE hour BETWEEN ? AND ? GROUP BY gu_nm",
                (day_hours[0], day_hours[1]),
            ).fetchall():
                if gu in idx and pop:
                    dist[b, idx[gu]] = float(pop)
    except Exception:
        pass
    finally:
        con.close()
    for b in range(7):
        s = dist[b].sum()
        dist[b] = dist[b] / s if s > 0 else 1.0 / n_gu
    return dist


def assign_daytime_location(age_bands: np.ndarray, dist_matrix: np.ndarray, *,
                            seed: int = 42) -> np.ndarray:
    """Sample each agent's DAYTIME district from its age band's real location
    distribution — the per-agent routing draw. Returns a length-N int array of
    district indices. Never raises."""
    rng = np.random.default_rng(seed)
    bands = np.clip(np.asarray(age_bands, dtype=int), 0, 6)
    n_gu = dist_matrix.shape[1]
    out = np.empty(len(bands), dtype=np.int64)
    for b in np.unique(bands):
        mask = bands == b
        out[mask] = rng.choice(n_gu, size=int(mask.sum()), p=dist_matrix[b])
    return out


def validate_daytime_routing(db_path: str, gu_names: list[str], *,
                             n_per_band: int = 4000, seed: int = 0) -> dict:
    """Validate the per-agent daytime routing reproduces the real commute: sampled
    working-age agents land in the business hubs more than children/elderly do.
    Returns ``{working_hub_share, nonworking_hub_share, match, verdict}``. Never
    raises."""
    dist = load_daytime_location_dist(db_path, gu_names)
    hub_idx = [i for i, g in enumerate(gu_names) if g in _BUSINESS_HUBS]
    if not hub_idx or dist.shape[1] != len(gu_names):
        return {"error": "no business-hub districts in gu_names"}
    work_bands, non_bands = [2, 3, 4, 5], [0, 1, 6]
    bands = np.repeat(np.arange(7), n_per_band)
    loc = assign_daytime_location(bands, dist, seed=seed)
    in_hub = np.isin(loc, hub_idx)
    work_share = float(in_hub[np.isin(bands, work_bands)].mean())
    non_share = float(in_hub[np.isin(bands, non_bands)].mean())
    match = work_share > non_share
    verdict = (
        f"daytime in business hubs: working-age {work_share:.3f} vs non-working "
        f"{non_share:.3f}. "
        + ("✓ per-agent routing reproduces the real commute — working agents are "
           "sampled into the central business districts by day, children/elderly "
           "stay residential. A real-data per-agent daytime mover (FoI integration "
           "into agent_kernel = remaining core step)."
           if match else "✗ routing does not reproduce the commute gradient.")
    )
    return {"working_hub_share": round(work_share, 4),
            "nonworking_hub_share": round(non_share, 4),
            "match": bool(match), "verdict": verdict}


def validate_working_age_commute(db_path: str) -> dict:
    """Validate the age-specific commute signature: working-age bands shift INTO
    the business hubs in the daytime more than children/elderly do.

    Returns ``{working_shift, nonworking_shift, working_gt_nonworking,
    working_positive, match, verdict}``. ``match`` requires the working-age mean
    daytime hub-shift to be positive AND larger than the non-working mean — the
    pattern a per-agent age-aware mover must reproduce. Never raises."""
    shares = load_age_hub_shares(db_path)
    if not all(c in shares for c in _AGE_COLS):
        return {"error": f"age×hour population incomplete (have {len(shares)}/{len(_AGE_COLS)} bands)"}
    work = np.mean([shares[c]["shift"] for c in _AGE_COLS if c in _WORKING])
    nonwork = np.mean([shares[c]["shift"] for c in _AGE_COLS if c not in _WORKING])
    work_gt = bool(work > nonwork)
    work_pos = bool(work > 0)
    match = work_gt and work_pos
    verdict = (
        f"daytime business-hub shift: working-age {work:+.3f} vs non-working "
        f"{nonwork:+.3f}. "
        + ("✓ working-age commute signature confirmed — workers concentrate into "
           "the central business districts by day while children/elderly stay "
           "residential; a valid basis for age-aware per-agent daytime movement "
           "(full route sampling in agent_kernel = future work)."
           if match else
           "✗ no clear working-age commute gradient — age-aware per-agent routing "
           "is not supported by the data here.")
    )
    return {"working_shift": round(float(work), 4),
            "nonworking_shift": round(float(nonwork), 4),
            "working_gt_nonworking": work_gt, "working_positive": work_pos,
            "match": bool(match), "verdict": verdict}
