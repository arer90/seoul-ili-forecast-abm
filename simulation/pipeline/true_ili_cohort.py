"""
Pov (overseas track) — TRUE ILI Cohort Module (§8 corrigendum, 2026-05-27).

PURPOSE: Replace WHO FluNet positivity% cohort (legacy phase18_overseas.py)
with TRUE ILI consultation rate cohort for proper cross-country dynamics analysis.

CITATIONS:
- eLife 107767 (Wang 2025) — same-region forecast principle
- ECDC ERVISS — EU/EEA per 100k ILI standard
- CDC ILINet — US weighted ILI%
- KDCA Sentinel — KR ILI per 1,000 외래환자

USAGE:
    from simulation.pipeline.true_ili_cohort import (
        get_cohort_ia, get_cohort_ib,
        load_kr_sentinel_ili, load_country_ili,
    )

    # Cohort I-A: 4-country long-term (2019-2025)
    cohort_ia = get_cohort_ia()  # ["KR", "US", "IT", "JP"]

    # Cohort I-B: 28-country wide (2021-2025)
    cohort_ib = get_cohort_ib()  # ["KR", "AT", "BE", ..., "US", "JP"]

NOTE on legacy phase18_overseas.py:
    Original module uses WHO FluNet positivity% (lab confirmation, NOT ILI).
    This new module uses TRUE ILI sources only. Both can coexist:
    - legacy = §7 cohort (positivity-based, KR↔DE finding INVALID)
    - this   = §8 cohort (TRUE ILI, KR↔BE/JP finding valid)
"""
from __future__ import annotations

import logging
from pathlib import Path
from sqlite3 import Connection

from simulation.database import safe_connect  # G-116/G-117: 단일 진입점
from typing import Optional

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "epi_real_seoul.db"

# ── TRUE ILI Source Priority (positivity 제외) ──
# Higher priority = more authoritative ILI source per country.
ILI_SOURCE_PRIORITY = [
    "ecdc_erviss",       # EU/EEA per 100k ILI consultations (25 countries)
    "sentiweb_fr",       # FR per 100k ILI
    "influnet_it",       # IT per 100k ILI
    "japan_jihs",        # JP per clinic (ILI proxy)
    "japan_jihs_hist",   # JP historical
    "cdc_ilinet",        # US weighted ILI %
    "delphi_national",   # US weighted ILI % (longest, 1997-)
]

# Excluded (positivity-based, NOT ILI):
EXCLUDED_SOURCES = {
    "who_flunet",        # lab confirmation rate (% positive), NOT ILI
    "who_fluid",         # consultation rate but 2026 only (sparse)
    "cdc_flusurvnet",    # hospital admission /100k, NOT ILI strictly
}

# ── Cohort Definitions ──

# Cohort I-A: Long-term TRUE ILI (4 country, 2019-2025, 7yr)
# Each has dedicated native ILI source (no FluNet fallback).
COHORT_I_A = ["KR", "US", "IT", "JP"]

# Cohort I-B: Wide TRUE ILI (28 country, 2021-2025, 5yr)
# All ECDC ERVISS + KR + US + IT + JP. Excludes DE/GB/AU/CN/HK/SG/NL/SE (FluNet only).
def get_cohort_ib(conn: Optional[Connection] = None) -> list[str]:
    """Get Cohort I-B country list — 25 EU ERVISS + KR + US + JP (+ IT already in EU).

    Args:
        conn: optional DB connection.

    Returns:
        Sorted country code list (2-letter ISO).
    """
    if conn is None:
        conn = safe_connect(str(_DB_PATH))  # G-116/G-117
        own_conn = True
    else:
        own_conn = False
    try:
        eu_erviss = [r[0] for r in conn.execute(
            "SELECT DISTINCT country FROM overseas_ili WHERE source='ecdc_erviss'"
        ).fetchall()]
        cohort = sorted(set(eu_erviss + ["KR", "US", "JP"]))
        return cohort
    finally:
        if own_conn:
            conn.close()


def get_cohort_ia() -> list[str]:
    """Get Cohort I-A — long-term 4 country."""
    return list(COHORT_I_A)


# ── Data Loading ──

def load_kr_sentinel_ili(
    conn: Connection, year_min: int = 2019, year_max: int = 2025
) -> list[tuple[int, int, float]]:
    """Load KR national ILI from KDCA sentinel_influenza (per-age averaged).

    Args:
        conn: DB connection.
        year_min/max: period bounds.

    Returns:
        list[(season_start, week_seq, ili_rate)].

    Source: KDCA Sentinel — ILI per 1,000 외래환자, 7 age groups averaged.
    """
    rows = conn.execute(
        "SELECT season_start, week_seq, AVG(ili_rate) "
        "FROM sentinel_influenza WHERE ili_rate IS NOT NULL "
        "AND season_start BETWEEN ? AND ? "
        "GROUP BY season_start, week_seq ORDER BY season_start, week_seq",
        (year_min, year_max),
    ).fetchall()
    return [(y, w, v) for y, w, v in rows]


def load_country_ili(
    conn: Connection,
    country: str,
    year_min: int,
    year_max: int,
) -> tuple[list[tuple[int, int, float]], str]:
    """Load TRUE ILI series for given country (highest priority source).

    Args:
        conn: DB connection.
        country: 2-letter ISO country code.
        year_min/max: period bounds.

    Returns:
        (rows, source_name). Returns ([], "NONE") if no TRUE ILI source available.

    NOTE: KR returns KDCA sentinel (not from overseas_ili table).
    """
    if country == "KR":
        rows = load_kr_sentinel_ili(conn, year_min, year_max)
        return rows, "KDCA_sentinel"

    # For other countries: pick highest priority available
    sources = [r[0] for r in conn.execute(
        "SELECT DISTINCT source FROM overseas_ili WHERE country=? AND ili_rate IS NOT NULL",
        (country,)
    ).fetchall()]
    for src in ILI_SOURCE_PRIORITY:
        if src in sources:
            rows = conn.execute(
                "SELECT year, week_no, ili_rate FROM overseas_ili "
                "WHERE country=? AND source=? AND ili_rate IS NOT NULL "
                "AND year BETWEEN ? AND ? ORDER BY year, week_no",
                (country, src, year_min, year_max)
            ).fetchall()
            return [(y, w, v) for y, w, v in rows], src

    log.warning(f"[true_ili] {country}: no TRUE ILI source. Available: {sources}")
    return [], "NONE"


# ── Cohort Info ──

COHORT_INFO = {
    "I-A": {
        "name": "Long-Term TRUE ILI",
        "period": "2019-2025 (7yr)",
        "countries": COHORT_I_A,
        "description": "KR + US + IT + JP — long-term, native ILI per country, biological similarity",
        "expected_kr_dtw_top1": "US",
        "expected_kr_plv_top1": "JP",
    },
    "I-B": {
        "name": "Wide TRUE ILI",
        "period": "2021-2025 (5yr)",
        "countries": None,  # dynamic via get_cohort_ib()
        "description": "KR + 25 EU/EEA ERVISS + US + IT + JP — wide cohort, methodological similarity",
        "expected_kr_dtw_top1": "NO",
        "expected_kr_plv_top1": "BE",
    },
}


if __name__ == "__main__":
    # Quick smoke
    logging.basicConfig(level=logging.INFO)
    print(f"Cohort I-A: {get_cohort_ia()}")
    print(f"Cohort I-B: {get_cohort_ib()}")
    with safe_connect(str(_DB_PATH)) as conn:  # G-116/G-117
        kr_rows = load_kr_sentinel_ili(conn)
        print(f"KR sentinel: {len(kr_rows)} weeks (period 2019-2025)")
        for country in ["US", "DE", "JP", "BE", "IT"]:
            rows, src = load_country_ili(conn, country, 2021, 2025)
            print(f"  {country}: source={src}, n={len(rows)}")
