"""Group J — Population density by region (overseas countries).

Purpose
-------
Provides sub-national population density for overseas ILI rate interpretation.
High-density regions typically show faster ILI spread; density is used downstream
as a scaling feature for regional ILI rate normalization.

Sources
-------
  1. WorldBank EN.POP.DNST indicator (national level, all target countries)
     URL: https://api.worldbank.org/v2/country/{iso2}/indicator/EN.POP.DNST
     No API key required. Covers 2010-present.

  2. US Census Bureau ACS 5-year estimates (state-level population, no key)
     URL: https://api.census.gov/data/{year}/acs/acs5?get=NAME,B01003_001E&for=state:*
     Density derived via embedded state land areas (km²) from US Census Bureau.

  3. Japan 2020 census (prefecture-level) — static embedded values
     Source: 国勢調査 2020; Statistics Japan e-Stat.
     Population and land area for all 47 都道府県.
     Refreshed only when e-Stat API key is configured (ESTAT_API_KEY env var).

Output table: overseas_population_density
  (country TEXT, region TEXT, region_type TEXT,
   year INT, population REAL, area_km2 REAL, pop_density REAL,
   source TEXT, collected_at TEXT,
   PRIMARY KEY (country, region, year))

CLI:
  .venv/bin/python -m simulation.collectors.group_j_population_density
  .venv/bin/python -m simulation.collectors.group_j_population_density --years-back 5
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Connection as _Conn
from typing import Optional

# G-116/G-117: safe_connect via lazy import — avoids top-level circular import.
def _safe_connect_import():
    from simulation.database import safe_connect
    return safe_connect

log = logging.getLogger(__name__)

_DEFAULT_DB = Path("simulation/data/db/epi_real_seoul.db")
_TIMEOUT_S = 30

# WorldBank API endpoint (no key required, OData-style JSON)
_WB_API_URL = (
    "https://api.worldbank.org/v2/country/{iso2}/indicator/EN.POP.DNST"
    "?format=json&per_page=200&date={start}:{end}"
)

# US Census ACS 5-year state population (requires API key since mid-2024)
# Key env var: CENSUS_API_KEY — optional; static 2020 data used when absent.
_CENSUS_ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"
_CENSUS_KEY_ENV = "CENSUS_API_KEY"

# Target countries: ISO3 → ISO2 (WorldBank uses ISO2)
_TARGET_COUNTRIES: dict[str, str] = {
    "USA": "US", "JPN": "JP", "GBR": "GB", "DEU": "DE",
    "FRA": "FR", "NLD": "NL", "SWE": "SE", "KOR": "KR",
    # EU-6 added 2026-05-25: weather collected (1 capital each), need national density
    "AUT": "AT", "BEL": "BE", "ESP": "ES", "ITA": "IT", "POL": "PL", "ROU": "RO",
}

# ─────────────────────────────────────────────────────────────────────────────
# Static reference data
# ─────────────────────────────────────────────────────────────────────────────

# US state land areas (km²) — US Census Bureau, 2020.
_US_STATE_AREA_KM2: dict[str, float] = {
    "AL": 131171.0, "AK": 1477953.0, "AZ": 294207.0, "AR": 134771.0,
    "CA": 403466.0, "CO": 268431.0,  "CT": 12543.0,  "DE": 5047.0,
    "FL": 139670.0, "GA": 149976.0,  "HI": 16635.0,  "ID": 214045.0,
    "IL": 143793.0, "IN": 92789.0,   "IA": 144669.0, "KS": 211754.0,
    "KY": 102269.0, "LA": 111898.0,  "ME": 79883.0,  "MD": 25314.0,
    "MA": 20202.0,  "MI": 147122.0,  "MN": 206189.0, "MS": 121531.0,
    "MO": 178040.0, "MT": 376979.0,  "NE": 198973.0, "NV": 284332.0,
    "NH": 23187.0,  "NJ": 19047.0,   "NM": 314161.0, "NY": 122057.0,
    "NC": 125920.0, "ND": 178711.0,  "OH": 105829.0, "OK": 177660.0,
    "OR": 248608.0, "PA": 115883.0,  "RI": 2678.0,   "SC": 77983.0,
    "SD": 196540.0, "TN": 106798.0,  "TX": 676587.0, "UT": 212818.0,
    "VT": 23871.0,  "VA": 102279.0,  "WA": 172112.0, "WV": 62259.0,
    "WI": 140268.0, "WY": 251470.0,  "DC": 159.0,
}

# US state 2020 census populations — US Census Bureau PL 94-171 apportionment file.
# Tuple: (population_2020,).  Density = pop / _US_STATE_AREA_KM2[abbrev].
_US_STATE_POP_2020: dict[str, int] = {
    "AL": 5024279,  "AK": 733391,   "AZ": 7151502,  "AR": 3011524,
    "CA": 39538223, "CO": 5773714,  "CT": 3605944,  "DE": 989948,
    "FL": 21538187, "GA": 10711908, "HI": 1455271,  "ID": 1839106,
    "IL": 12812508, "IN": 6785528,  "IA": 3190369,  "KS": 2937880,
    "KY": 4505836,  "LA": 4657757,  "ME": 1362359,  "MD": 6177224,
    "MA": 7029917,  "MI": 10077331, "MN": 5706494,  "MS": 2961279,
    "MO": 6154913,  "MT": 1084225,  "NE": 1961504,  "NV": 3104614,
    "NH": 1377529,  "NJ": 9288994,  "NM": 2117522,  "NY": 20201249,
    "NC": 10439388, "ND": 779094,   "OH": 11799448, "OK": 3959353,
    "OR": 4237256,  "PA": 13002700, "RI": 1097379,  "SC": 5118425,
    "SD": 886667,   "TN": 6910840,  "TX": 29145505, "UT": 3271616,
    "VT": 643077,   "VA": 8631393,  "WA": 7705281,  "WV": 1793716,
    "WI": 5893718,  "WY": 576851,   "DC": 689545,
}

# US state name → 2-letter abbreviation (lowercase key)
_US_NAME_TO_ABBREV: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Japan prefecture population + land area — 2020 国勢調査 (Statistics Japan).
# Tuple: (population, area_km2).
# Source: e-Stat / 国土数値情報 2020.
_JP_PREFECTURE_2020: dict[str, tuple[int, float]] = {
    "Hokkaido":   (5224614,  83424.0), "Aomori":    (1237984,   9644.0),
    "Iwate":      (1210534,  15275.0), "Miyagi":    (2301996,   7282.0),
    "Akita":      ( 959502,  11638.0), "Yamagata":  (1068027,   9323.0),
    "Fukushima":  (1833152,  13784.0), "Ibaraki":   (2867009,   6098.0),
    "Tochigi":    (1933146,   6408.0), "Gunma":     (1939110,   6362.0),
    "Saitama":    (7344765,   3798.0), "Chiba":     (6284480,   5158.0),
    "Tokyo":      (13960000,  2194.0), "Kanagawa":  (9237337,   2416.0),
    "Niigata":    (2201272,  12584.0), "Toyama":    (1034814,   4248.0),
    "Ishikawa":   (1132526,   4186.0), "Fukui":     ( 766863,   4190.0),
    "Yamanashi":  ( 809974,   4465.0), "Nagano":    (2048011,  13562.0),
    "Gifu":       (1978742,  10621.0), "Shizuoka":  (3633202,   7777.0),
    "Aichi":      (7542415,   5173.0), "Mie":       (1770254,   5774.0),
    "Shiga":      (1413610,   4017.0), "Kyoto":     (2578087,   4612.0),
    "Osaka":      (8837685,   1905.0), "Hyogo":     (5465002,   8401.0),
    "Nara":       (1324473,   3691.0), "Wakayama":  ( 922584,   4725.0),
    "Tottori":    ( 553407,   3507.0), "Shimane":   ( 671126,   6708.0),
    "Okayama":    (1888432,   7114.0), "Hiroshima": (2799702,   8480.0),
    "Yamaguchi":  (1342059,   6113.0), "Tokushima": ( 719559,   4147.0),
    "Kagawa":     ( 950244,   1877.0), "Ehime":     (1334841,   5676.0),
    "Kochi":      ( 691527,   7104.0), "Fukuoka":   (5135214,   4987.0),
    "Saga":       ( 811442,   2441.0), "Nagasaki":  (1312317,   4132.0),
    "Kumamoto":   (1738301,   7409.0), "Oita":      (1123852,   6341.0),
    "Miyazaki":   (1069576,   7735.0), "Kagoshima": (1588256,   9188.0),
    "Okinawa":    (1467480,   2282.0),
}


def _resolve_db(db_path: Optional[str | Path]) -> Path:
    if db_path is not None:
        return Path(db_path)
    cwd = Path.cwd()
    for p in [_DEFAULT_DB, cwd / _DEFAULT_DB, cwd.parent / _DEFAULT_DB]:
        if p.exists():
            return p
    return _DEFAULT_DB


def _connect(db_path: Path) -> _Conn:
    safe_connect = _safe_connect_import()
    return safe_connect(str(db_path), timeout=60.0)


def _ensure_table(con: _Conn) -> None:
    """Create overseas_population_density if absent.

    Performance: O(1).
    Side effects: may create table.
    Caller responsibility: con must be open.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS overseas_population_density (
            country      TEXT    NOT NULL,   -- ISO3: 'USA', 'JPN', etc.
            region       TEXT    NOT NULL,   -- 'national', state abbrev, prefecture name
            region_type  TEXT,              -- 'national', 'state', 'prefecture', 'nuts2'
            year         INTEGER NOT NULL,
            population   REAL,              -- person count
            area_km2     REAL,              -- land area km²
            pop_density  REAL,              -- persons per km²
            source       TEXT,
            collected_at TEXT,
            PRIMARY KEY (country, region, year)
        )
    """)
    con.commit()


def _upsert(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into overseas_population_density.

    Args:
        con:  Open SQLite connection.
        rows: List of dicts matching overseas_population_density columns.

    Returns:
        (inserted, skipped) counts.

    Performance: batched 200 rows/tx.
    Side effects: writes to overseas_population_density; commits each batch.
    Caller responsibility: _ensure_table must have been called.
    """
    if not rows:
        return 0, 0
    sql = """
        INSERT INTO overseas_population_density
          (country, region, region_type, year, population, area_km2,
           pop_density, source, collected_at)
        VALUES
          (:country, :region, :region_type, :year, :population, :area_km2,
           :pop_density, :source, :collected_at)
        ON CONFLICT(country, region, year) DO UPDATE SET
          population   = COALESCE(excluded.population,  population),
          area_km2     = COALESCE(excluded.area_km2,    area_km2),
          pop_density  = COALESCE(excluded.pop_density, pop_density),
          source       = excluded.source,
          collected_at = excluded.collected_at
    """
    inserted = 0
    for i in range(0, len(rows), 200):
        chunk = rows[i:i + 200]
        cur = con.executemany(sql, chunk)
        inserted += cur.rowcount
        con.commit()
    return inserted, len(rows) - inserted


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: WorldBank national density
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_worldbank_national(years_back: int) -> list[dict]:
    """Fetch national population density from WorldBank EN.POP.DNST indicator.

    One API call fetches all 8 target countries for the full year range.
    Response format: [{page_metadata}, [{country, date, value, ...}, ...]]

    Args:
        years_back: Number of past years to retrieve (end = current year).

    Returns:
        List of row dicts for overseas_population_density:
          region='national', region_type='national', pop_density from WorldBank.

    Raises:
        RuntimeError: on HTTP error or unexpected JSON format.

    Performance: 1 HTTP request, ~0.5 s.
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    iso2_list = ";".join(_TARGET_COUNTRIES.values())
    end_year = datetime.now().year
    start_year = end_year - years_back
    url = _WB_API_URL.format(iso2=iso2_list, start=start_year, end=end_year)

    log.info("[group_j] WorldBank national density: %s", url)
    resp = requests.get(url, timeout=_TIMEOUT_S)
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError(f"WorldBank API unexpected format: {str(data)[:200]}")

    items = data[1] or []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build ISO2 → ISO3 reverse map for the region label
    iso2_to_iso3 = {v: k for k, v in _TARGET_COUNTRIES.items()}

    rows: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        iso2 = it.get("countryiso3code") or (it.get("country") or {}).get("id", "")
        # WorldBank sometimes uses iso2, sometimes iso3 in countryiso3code
        iso3 = iso2_to_iso3.get(iso2, iso2)
        if not iso3:
            continue
        try:
            year = int(str(it.get("date", "0"))[:4])
            value = it.get("value")
            if value is None:
                continue
            density = float(value)
        except (TypeError, ValueError):
            continue

        rows.append({
            "country":     iso3,
            "region":      "national",
            "region_type": "national",
            "year":        year,
            "population":  None,   # WorldBank indicator = density; pop not separate
            "area_km2":    None,
            "pop_density": density,
            "source":      "worldbank_EN.POP.DNST",
            "collected_at": now_iso,
        })

    log.info("[group_j] WorldBank: %d national density rows", len(rows))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: US Census ACS state population → density
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_us_state_density(census_key: Optional[str] = None) -> list[dict]:
    """Return US state population density from 2020 census embedded data.

    The embedded _US_STATE_POP_2020 + _US_STATE_AREA_KM2 dicts provide 2020
    census population and land area for all 50 states + DC.  density = pop/area.

    If CENSUS_API_KEY is provided, a live Census ACS 5-year refresh is attempted
    for the most recent finalized year (tried 2023 → 2022 → 2021).  Live rows
    are added in addition to the static 2020 baseline — they do not replace it.

    Args:
        census_key: Census Bureau API key (optional).  When None, only 2020
                    static data is returned.

    Returns:
        List of row dicts for overseas_population_density:
          country='USA', region=2-letter state abbrev, region_type='state'.
        Always returns at least 51 rows (2020 static).

    Raises:
        Nothing — Census API failures fall back to static data only.

    Performance: O(1) for static; ~0.5 s if Census API attempted.
    Side effects: None.
    Caller responsibility: None.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []

    # ── Always: static 2020 census ───────────────────────────────────────────
    for abbrev, pop in _US_STATE_POP_2020.items():
        area = _US_STATE_AREA_KM2.get(abbrev)
        if area is None or area <= 0:
            continue
        density = round(pop / area, 4)
        rows.append({
            "country":     "USA",
            "region":      abbrev,
            "region_type": "state",
            "year":        2020,
            "population":  float(pop),
            "area_km2":    area,
            "pop_density": density,
            "source":      "census_2020_static",
            "collected_at": now_iso,
        })
    log.info("[group_j] US 2020 census: %d state density rows (static)", len(rows))

    # ── Optional: Census ACS live refresh (requires key) ─────────────────────
    if not census_key:
        log.debug(
            "[group_j] CENSUS_API_KEY not set — using 2020 static US state data. "
            "Set env var CENSUS_API_KEY for annual ACS refresh."
        )
        return rows

    try:
        import requests
    except ImportError:
        log.warning("[group_j] requests not installed — Census ACS live skipped")
        return rows

    # ACS 5-year finalized with ~2-year lag; try 2023, 2022, 2021
    live_year = None
    live_data = None
    for year in [datetime.now().year - 2, datetime.now().year - 3, 2022]:
        url = _CENSUS_ACS_URL.format(year=year)
        try:
            resp = requests.get(
                url,
                params={"get": "NAME,B01003_001E", "for": "state:*",
                        "key": census_key},
                timeout=_TIMEOUT_S,
            )
            # Check content type before parsing
            ct = resp.headers.get("Content-Type", "")
            if "html" in ct.lower():
                log.warning("[group_j] Census ACS %d: HTML response (key invalid?)", year)
                continue
            resp.raise_for_status()
            live_data = resp.json()
            live_year = year
            break
        except Exception as e:
            log.warning("[group_j] Census ACS %d failed: %s", year, e)

    if live_data and live_year:
        header   = [str(h).strip() for h in live_data[0]]
        name_idx = header.index("NAME") if "NAME" in header else 0
        pop_idx  = header.index("B01003_001E") if "B01003_001E" in header else 1
        added = 0
        for row in live_data[1:]:
            try:
                state_name = str(row[name_idx]).lower().strip()
                population = float(row[pop_idx])
            except (IndexError, TypeError, ValueError):
                continue
            abbrev = _US_NAME_TO_ABBREV.get(state_name)
            area   = _US_STATE_AREA_KM2.get(abbrev or "")
            if abbrev is None or area is None or area <= 0:
                continue
            rows.append({
                "country":     "USA",
                "region":      abbrev,
                "region_type": "state",
                "year":        live_year,
                "population":  population,
                "area_km2":    area,
                "pop_density": round(population / area, 4),
                "source":      f"census_acs5_{live_year}",
                "collected_at": now_iso,
            })
            added += 1
        log.info("[group_j] Census ACS %d: %d live US state density rows", live_year, added)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: Japan prefecture (static 2020 + optional e-Stat refresh)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_jp_prefecture_density(estat_key: Optional[str] = None) -> list[dict]:
    """Return Japan prefecture population density from 2020 census embedded data.

    The embedded _JP_PREFECTURE_2020 dict contains 2020 国勢調査 values for
    all 47 都道府県.  If an e-Stat API key is provided (ESTAT_API_KEY env var),
    a live refresh is attempted for the most recent year; this is additive —
    the static 2020 data is always returned regardless.

    Args:
        estat_key: e-Stat API key (optional).  If None, only static data returned.

    Returns:
        List of row dicts for overseas_population_density:
          country='JPN', region=prefecture romaji name, region_type='prefecture',
          year=2020 (static) or most recent available (if e-Stat called).

    Raises:
        Nothing — e-Stat failures are logged and ignored.

    Performance: O(1) for static data; ~1 s if e-Stat called.
    Side effects: None.
    Caller responsibility: None.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []

    # Always include static 2020 values
    for pref, (pop, area) in _JP_PREFECTURE_2020.items():
        density = round(pop / area, 4) if area > 0 else None
        rows.append({
            "country":     "JPN",
            "region":      pref,
            "region_type": "prefecture",
            "year":        2020,
            "population":  float(pop),
            "area_km2":    area,
            "pop_density": density,
            "source":      "census_jp_2020_static",
            "collected_at": now_iso,
        })

    log.info("[group_j] Japan 2020 census: %d prefecture density rows (static)", len(rows))

    # Optional e-Stat live refresh (skeleton — expand when API confirmed)
    if estat_key:
        log.info("[group_j] e-Stat API key detected — live refresh not yet implemented. "
                 "Static 2020 data used.")
        # TODO: implement e-Stat statsDataId=0003410383 (population by prefecture) refresh

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    backfill_days: Optional[int] = None,
    db_path: Optional[str | Path] = None,
    years_back: int = 5,
    skip_worldbank: bool = False,
    skip_us: bool = False,
    skip_jp: bool = False,
) -> dict:
    """Collect population density for overseas countries at national + regional level.

    Sources attempted:
      1. WorldBank EN.POP.DNST  → national density, all 8 target countries
      2. US Census ACS 5-year  → state-level population, derive density
      3. Japan 2020 census      → prefecture-level static data (+ e-Stat if key set)

    All sources are non-fatal: failure of one does not abort others.

    Args:
        backfill_days: Orchestrator override. Converts to years_back = days//365+1.
        db_path:       Path to epi_real_seoul.db (default: standard project path).
        years_back:    Years of WorldBank history to retrieve (default 5).
                       US and Japan static/Census data are always for latest year.
        skip_worldbank: Skip WorldBank national density collection.
        skip_us:        Skip US Census ACS state density collection.
        skip_jp:        Skip Japan prefecture density collection.

    Returns:
        dict: {inserted, skipped, errors: list[str], sources_fetched: list[str]}

    Raises:
        Nothing — all errors caught and returned in `errors`.

    Performance: ~2-5 s (2-3 HTTP requests).
    Side effects: creates overseas_population_density if absent; upserts rows.
    Caller responsibility: `requests` must be installed; DB must exist.
    """
    if backfill_days is not None and backfill_days > 0:
        years_back = max(years_back, backfill_days // 365 + 1)

    resolved = _resolve_db(db_path)
    result: dict = {"inserted": 0, "skipped": 0, "errors": [], "sources_fetched": []}

    if not resolved.exists():
        result["errors"].append(f"DB not found: {resolved}")
        return result

    estat_key   = os.environ.get("ESTAT_API_KEY",   "").strip() or None
    census_key  = os.environ.get(_CENSUS_KEY_ENV,   "").strip() or None
    con = _connect(resolved)
    try:
        _ensure_table(con)
        all_rows: list[dict] = []

        if not skip_worldbank:
            try:
                wb_rows = _fetch_worldbank_national(years_back)
                all_rows.extend(wb_rows)
                result["sources_fetched"].append("worldbank_national")
            except Exception as e:
                log.error("[group_j] WorldBank national density failed: %s", e)
                result["errors"].append(f"worldbank: {e}")

        if not skip_us:
            try:
                us_rows = _fetch_us_state_density(census_key=census_key)
                all_rows.extend(us_rows)
                if us_rows:
                    result["sources_fetched"].append(
                        "census_acs_state" if census_key else "census_2020_static"
                    )
            except Exception as e:
                log.error("[group_j] US state density failed: %s", e)
                result["errors"].append(f"census_us: {e}")

        if not skip_jp:
            try:
                jp_rows = _fetch_jp_prefecture_density(estat_key=estat_key)
                all_rows.extend(jp_rows)
                if jp_rows:
                    result["sources_fetched"].append("jp_prefecture_2020")
            except Exception as e:
                log.error("[group_j] Japan prefecture density failed: %s", e)
                result["errors"].append(f"jp_prefecture: {e}")

        ins, skp = _upsert(con, all_rows)
        result["inserted"] = ins
        result["skipped"]  = skp

    finally:
        con.close()

    log.info(
        "[group_j] Done: inserted=%d skipped=%d errors=%d sources=%s",
        result["inserted"], result["skipped"],
        len(result["errors"]), result["sources_fetched"],
    )
    return result


if __name__ == "__main__":
    import argparse
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Overseas population density collector (WorldBank + Census + Japan)"
    )
    parser.add_argument("--years-back",      type=int, default=5,
                        help="WorldBank history years (default 5)")
    parser.add_argument("--skip-worldbank",  action="store_true")
    parser.add_argument("--skip-us",         action="store_true")
    parser.add_argument("--skip-jp",         action="store_true")
    parser.add_argument("--db",              default=None,
                        help="Path to epi_real_seoul.db")
    args = parser.parse_args()

    r = run(
        db_path=args.db,
        years_back=args.years_back,
        skip_worldbank=args.skip_worldbank,
        skip_us=args.skip_us,
        skip_jp=args.skip_jp,
    )
    print(json.dumps(r, indent=2, default=str))
    sys.exit(1 if r["errors"] else 0)
