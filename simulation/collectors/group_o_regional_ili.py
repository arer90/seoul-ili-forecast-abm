"""Group O — Regional ILI surveillance (US / Japan / Germany + stubs for UK, FR).

Purpose
-------
Extends overseas ILI from national-level (Group I: overseas_ili) to sub-national
regional resolution — the primary feature demanded for GU-level ILI forecasting:

  US sources (country='USA') — four complementary signals:
  1. Delphi FluView        → source='delphi_covidcast'      wili/ili%, 51 states, 2010-present
  2. Delphi NSSP           → source='nssp_ed_visits_flu'    % flu ED visits, 50 states, 2022-present
  3. CDC NHSN vdzy-6i9v   → source='nhsn_flu_admissions'   per-100k + count + level, 51, 2021-present
  4. CDC NWSS atcp-73re   → source='nwss_flu_a'            wastewater Flu-A 1-5 ordinal, 2023-present
     Note: CDC 6svj-q4zv (FluView ordinal) ARCHIVED 2024-10-16 → removed.

  Japan (country='JPN'):
  5. NIID IDWR API         → source='niid_prefecture'       DNS failure → graceful empty
     Status: id.niid.go.jp / api.jihs.go.jp unresolvable (2026-05-24).

  Germany (country='DEU'):
  6. RKI GitHub TSV        → source='rki_bundesland'        Inzidenz/100k, 16 Bundesländer

  France (country='FRA') — added 2026-05-25:
  8. Sentiweb FR regional → source='sentiweb_fr_regional'  ILI/100k, 22 regions, 1984-present

  Not yet implemented — DNS / access failures:
  - UK UKHSA (DNS failure), OECD SDMX (all 404), NL RIVM (metadata API broken)

API diagnostics (updated 2026-05-25):
  WORKING:  Delphi COVIDcast (fluview, nssp, nhsn signals)
            CDC NHSN Socrata vdzy-6i9v
            CDC NWSS Socrata atcp-73re
            RKI GitHub TSV (raw.githubusercontent.com)
            Sentiweb France regional (sentiweb.fr/datasets/incidence-REG-3.csv) ✅
  BROKEN:   OECD SDMX/Stat — all endpoints 404
            Japan NIID/JIHS — DNS resolution failure (id.niid.go.jp)
            UK UKHSA — DNS failure
            WHO xMart (xmart-api-public.who.int) — DNS failure (blocks FluNet+FluID)

Output table: overseas_ili_regional
  (source TEXT, country TEXT, region TEXT, year INT, week_no INT,
   ili_rate REAL,                 -- ILI% (NULL if source uses ordinal scale)
   activity_level_ordinal INT,    -- CDC ordinal 1-10 (NULL if source uses ILI%)
   n_providers INT, n_patients INT, n_ili INT,
   collected_at TEXT,
   PRIMARY KEY (source, country, region, year, week_no))

CLI:
  .venv/bin/python -m simulation.collectors.group_o_regional_ili
  .venv/bin/python -m simulation.collectors.group_o_regional_ili --years-back 3
  .venv/bin/python -m simulation.collectors.group_o_regional_ili --skip-delphi
"""
from __future__ import annotations

import json
import logging
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
_TIMEOUT_S = 120  # ZIP download may be slow (~1-2 MB)

# Sprint α Item 9 (2026-05-26): URL constants centralized in
# simulation/collectors/_endpoints.py
from simulation.collectors._endpoints import (
    _NIID_API_BASE,        # NIID IDWR weekly aggregate (disease_id=0018 = influenza)
    _DE_RKI_TSV_URL,       # RKI Bundesland weekly TSV (Influenzafaelle)
    _DELPHI_FLUVIEW_URL,   # Delphi FluView (CMU) — state-level ILI%
)

# All 51 US jurisdictions tracked by Delphi (2-letter abbreviations, lowercase for API).
_US_DELPHI_STATES: list[str] = [
    "ak", "al", "ar", "az", "ca", "co", "ct", "dc", "de", "fl",
    "ga", "hi", "ia", "id", "il", "in", "ks", "ky", "la", "ma",
    "md", "me", "mi", "mn", "mo", "ms", "mt", "nc", "nd", "ne",
    "nh", "nj", "nm", "nv", "ny", "oh", "ok", "or", "pa", "ri",
    "sc", "sd", "tn", "tx", "ut", "va", "vt", "wa", "wi", "wv", "wy",
]
# Note: Delphi also tracks "nat" (national), "hhs1"-"hhs10" (HHS regions), "cen1"-"cen9"
# (census divisions). We only collect state-level here.

# Delphi COVIDcast + CDC NHSN/NWSS — centralized in _endpoints
from simulation.collectors._endpoints import (
    _DELPHI_COVIDCAST_URL,
    _NHSN_HRD_URL,
    _NWSS_FLU_URL,
)

# Map "Very Low" / "Low" / "Moderate" / "High" / "Very High" → integer 1-5.
_LEVEL_LABEL_TO_INT: dict[str, int] = {
    "very low": 1, "low": 2, "moderate": 3, "high": 4, "very high": 5,
}

# ─────────────────────────────────────────────────────────────────────────────
# Static mappings
# ─────────────────────────────────────────────────────────────────────────────

# US state/territory full name → 2-letter abbreviation.
_US_STATE_ABBREV: dict[str, str] = {
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
    "american samoa": "AS", "guam": "GU", "northern mariana islands": "MP",
    "puerto rico": "PR", "virgin islands": "VI", "new york city": "NYC",
}

# Japan prefecture JP (romaji) → abbreviation.  NIID uses kanji; mapping added
# on parse so region column is consistent (ASCII-safe for downstream joins).
_JP_PREF_ROMAJI: dict[str, str] = {
    "北海道": "Hokkaido", "青森": "Aomori", "岩手": "Iwate", "宮城": "Miyagi",
    "秋田": "Akita", "山形": "Yamagata", "福島": "Fukushima", "茨城": "Ibaraki",
    "栃木": "Tochigi", "群馬": "Gunma", "埼玉": "Saitama", "千葉": "Chiba",
    "東京": "Tokyo", "神奈川": "Kanagawa", "新潟": "Niigata", "富山": "Toyama",
    "石川": "Ishikawa", "福井": "Fukui", "山梨": "Yamanashi", "長野": "Nagano",
    "岐阜": "Gifu", "静岡": "Shizuoka", "愛知": "Aichi", "三重": "Mie",
    "滋賀": "Shiga", "京都": "Kyoto", "大阪": "Osaka", "兵庫": "Hyogo",
    "奈良": "Nara", "和歌山": "Wakayama", "鳥取": "Tottori", "島根": "Shimane",
    "岡山": "Okayama", "広島": "Hiroshima", "山口": "Yamaguchi", "徳島": "Tokushima",
    "香川": "Kagawa", "愛媛": "Ehime", "高知": "Kochi", "福岡": "Fukuoka",
    "佐賀": "Saga", "長崎": "Nagasaki", "熊本": "Kumamoto", "大分": "Oita",
    "宮崎": "Miyazaki", "鹿児島": "Kagoshima", "沖縄": "Okinawa",
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
    """Create overseas_ili_regional if absent; add activity_level_ordinal column if missing.

    The activity_level_ordinal column was added in the 2026-05-24 revision to
    distinguish CDC ordinal activity levels (1-10) from continuous ILI rates.
    ALTER TABLE handles the case where the table was created by an earlier version.

    Performance: O(1) — DDL + one PRAGMA.
    Side effects: may create table; may ALTER TABLE to add column.
    Caller responsibility: con must be open.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS overseas_ili_regional (
            source                 TEXT    NOT NULL,
            country                TEXT    NOT NULL,
            region                 TEXT    NOT NULL,
            year                   INTEGER NOT NULL,
            week_no                INTEGER NOT NULL,
            ili_rate               REAL,
            activity_level_ordinal INTEGER,
            n_providers            INTEGER,
            n_patients             INTEGER,
            n_ili                  INTEGER,
            collected_at           TEXT,
            PRIMARY KEY (source, country, region, year, week_no)
        )
    """)
    # Back-compat: add column if table existed without it
    existing = {r[1] for r in con.execute("PRAGMA table_info(overseas_ili_regional)")}
    if "activity_level_ordinal" not in existing:
        con.execute("ALTER TABLE overseas_ili_regional ADD COLUMN activity_level_ordinal INTEGER")
    con.commit()


def _upsert(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into overseas_ili_regional.

    Args:
        con:  Open SQLite connection.
        rows: List of dicts matching overseas_ili_regional columns.
              Keys: source, country, region, year, week_no, ili_rate,
              activity_level_ordinal, n_providers, n_patients, n_ili, collected_at.

    Returns:
        (inserted, skipped) counts. skipped = len(rows) - inserted (ON CONFLICT rows).

    Performance: batched 500 rows/tx; O(n_rows) total.
    Side effects: writes to overseas_ili_regional; commits each batch.
    Caller responsibility: _ensure_table must have been called.
    """
    if not rows:
        return 0, 0
    sql = """
        INSERT INTO overseas_ili_regional
          (source, country, region, year, week_no,
           ili_rate, activity_level_ordinal,
           n_providers, n_patients, n_ili, collected_at)
        VALUES
          (:source, :country, :region, :year, :week_no,
           :ili_rate, :activity_level_ordinal,
           :n_providers, :n_patients, :n_ili, :collected_at)
        ON CONFLICT(source, country, region, year, week_no) DO UPDATE SET
          ili_rate               = COALESCE(excluded.ili_rate,
                                            ili_rate),
          activity_level_ordinal = COALESCE(excluded.activity_level_ordinal,
                                            activity_level_ordinal),
          n_providers            = COALESCE(excluded.n_providers, n_providers),
          n_patients             = COALESCE(excluded.n_patients,  n_patients),
          n_ili                  = COALESCE(excluded.n_ili,        n_ili),
          collected_at           = excluded.collected_at
    """
    inserted = 0
    for i in range(0, len(rows), 500):
        chunk = rows[i:i + 500]
        # Fill missing optional keys with None so executemany doesn't KeyError
        for r in chunk:
            r.setdefault("ili_rate", None)
            r.setdefault("activity_level_ordinal", None)
            r.setdefault("n_providers", None)
            r.setdefault("n_patients", None)
            r.setdefault("n_ili", None)
        cur = con.executemany(sql, chunk)
        inserted += cur.rowcount
        con.commit()
    return inserted, len(rows) - inserted


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: US Delphi COVIDcast state-level ILI% (PRIMARY source)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_us_state_ili_delphi(years_back: int) -> list[dict]:
    """Fetch actual state-level ILI% from Delphi COVIDcast FluView endpoint.

    Delphi COVIDcast (Carnegie Mellon) aggregates CDC FluView data and exposes
    it via a clean JSON API — the only public no-key endpoint returning actual
    ILI percentage (not just ordinal levels) at the US state level.

    Both weighted ILI (wili) and unweighted ILI (ili) are returned; wili is
    stored in ``ili_rate`` as the primary metric.  Provider, patient, and ILI
    patient counts are also stored.

    Endpoint: https://api.delphi.cmu.edu/epidata/fluview/
    Params:   regions=ak,al,...  (comma-sep lowercase 2-letter state codes)
              epiweeks=YYYYWW-YYYYWW (start-end range, inclusive)
    Response: {"result": 1, "epidata": [{
                "region": "ny", "epiweek": 202401,
                "wili": 5.23, "ili": 5.13,
                "num_providers": 234, "num_patients": 45678, "num_ili": 2345,
                ...
              }]}

    States are batched 10 at a time to respect API URL length limits.
    On HTTP error or JSON parse failure, the batch is skipped with a WARNING;
    a partial result may be returned.

    Args:
        years_back: Number of historical flu seasons to retrieve.
                    Translates to epiweek range start = (current_year - years_back)*100+01.

    Returns:
        List of row dicts for overseas_ili_regional:
          source='delphi_covidcast', country='USA',
          region=2-letter state abbrev (uppercase), year, week_no,
          ili_rate=wili (weighted ILI%), n_providers, n_patients, n_ili.

    Raises:
        Nothing — all exceptions caught and logged; returns partial results.

    Performance: ~10 batches × 1 HTTP request ≈ 10 s; ~3-5 K rows total.
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError:
        log.warning("[group_o] requests not installed — skipping Delphi COVIDcast")
        return []

    now = datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    cur_year = now.year
    start_year = cur_year - years_back

    # Delphi epiweek range: YYYYWW integer, range = "start-end"
    start_epiweek = start_year * 100 + 1
    end_epiweek   = cur_year  * 100 + 53  # generous upper bound; API clips to actual data
    epiweek_range = f"{start_epiweek}-{end_epiweek}"

    log.info(
        "[group_o] Delphi COVIDcast US state ILI%% epiweeks=%s (%d states, batches of 10) …",
        epiweek_range, len(_US_DELPHI_STATES),
    )

    rows: list[dict] = []
    batch_size = 10  # stay well under URL length limit

    for batch_start in range(0, len(_US_DELPHI_STATES), batch_size):
        batch = _US_DELPHI_STATES[batch_start:batch_start + batch_size]
        regions_param = ",".join(batch)

        try:
            resp = requests.get(
                _DELPHI_FLUVIEW_URL,
                params={"regions": regions_param, "epiweeks": epiweek_range},
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            log.warning("[group_o] Delphi batch %s failed: %s", regions_param, e)
            continue

        if payload.get("result") != 1:
            log.warning(
                "[group_o] Delphi batch %s: result=%s message=%s",
                regions_param, payload.get("result"), payload.get("message"),
            )
            continue

        for rec in payload.get("epidata") or []:
            epiweek_raw = rec.get("epiweek")
            try:
                epiweek = int(epiweek_raw)
                year    = epiweek // 100
                week    = epiweek % 100
            except (TypeError, ValueError):
                continue

            if year < start_year or week < 1 or week > 53:
                continue

            region = str(rec.get("region") or "").strip().upper()
            if not region or len(region) > 3:
                continue

            # wili = weighted ILI%; ili = unweighted. Prefer wili.
            wili = rec.get("wili")
            ili  = rec.get("ili")
            try:
                ili_rate = float(wili) if wili is not None else (
                           float(ili)  if ili  is not None else None)
            except (TypeError, ValueError):
                ili_rate = None

            def _safe_int_rec(v) -> Optional[int]:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None

            rows.append({
                "source":                 "delphi_covidcast",
                "country":                "USA",
                "region":                 region,
                "year":                   year,
                "week_no":                week,
                "ili_rate":               ili_rate,       # actual weighted ILI%
                "activity_level_ordinal": None,            # not applicable
                "n_providers":            _safe_int_rec(rec.get("num_providers")),
                "n_patients":             _safe_int_rec(rec.get("num_patients")),
                "n_ili":                  _safe_int_rec(rec.get("num_ili")),
                "collected_at":           now_iso,
            })

    log.info("[group_o] Delphi COVIDcast US: %d rows (year >= %d)", len(rows), start_year)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: US NSSP ED visits % (Delphi COVIDcast)
# Note: CDC dataset 6svj-q4zv was archived 2024-10-16 (returns 400) — removed.
# ─────────────────────────────────────────────────────────────────────────────

def _epiweek_range(years_back: int) -> str:
    """Build Delphi COVIDcast epiweek range string from current date.

    Args:
        years_back: Number of past years to include.

    Returns:
        String "YYYYWW-YYYYWW" covering [current_year - years_back, current_year].

    Performance: O(1).
    Side effects: None.
    Caller responsibility: years_back >= 1.
    """
    now = datetime.now()
    start = (now.year - years_back) * 100 + 1
    end   = now.year * 100 + 53
    return f"{start}-{end}"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: US Delphi HHS regional ILI% (10 regions, 1997-present)
# ─────────────────────────────────────────────────────────────────────────────

_US_HHS_REGIONS: list[str] = [
    "hhs1", "hhs2", "hhs3", "hhs4", "hhs5",
    "hhs6", "hhs7", "hhs8", "hhs9", "hhs10",
]


def _fetch_us_delphi_hhs_regions(years_back: int = 3) -> list[dict]:
    """Fetch US HHS-region-level weighted ILI% from Delphi COVIDcast FluView.

    HHS (Dept. of Health and Human Services) regions are 10 aggregated US
    multi-state areas — finer than national, coarser than individual states.
    Delphi COVIDcast exposes them as ``hhs1``–``hhs10`` via the FluView endpoint,
    going back to 1997-W40 for most regions.

    Endpoint: https://api.delphi.cmu.edu/epidata/fluview/
    Params:   regions=hhs1,...,hhs10   epiweeks=199740-<current>
    Response: same schema as national/state FluView.

    Args:
        years_back: Number of historical years to fetch (default 3).
                    Delphi supports back to 1997; set years_back=30 for full history.

    Returns:
        List of row dicts for overseas_ili_regional:
          source='delphi_hhs', country='USA',
          region='HHS1'..'HHS10' (uppercase for consistent joins),
          year, week_no,
          ili_rate=wili (weighted ILI%),
          n_providers, n_patients, n_ili.

    Raises:
        Nothing — HTTP/JSON failures logged; partial results returned.

    Performance: 1 HTTP request (all 10 regions in one call); ~3-5 s.
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError:
        log.warning("[group_o] requests not installed — skipping Delphi HHS regions")
        return []

    now     = datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    cur_year = now.year
    start_epiweek = (cur_year - years_back) * 100 + 1
    end_epiweek   = cur_year * 100 + 53
    epiweek_range = f"{start_epiweek}-{end_epiweek}"

    regions_str = ",".join(_US_HHS_REGIONS)
    log.info("[group_o] Delphi HHS regions epiweeks=%s …", epiweek_range)

    try:
        resp = requests.get(
            _DELPHI_FLUVIEW_URL,
            params={"regions": regions_str, "epiweeks": epiweek_range},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        log.warning("[group_o] Delphi HHS regions fetch failed: %s", e)
        return []

    if payload.get("result") != 1:
        log.warning(
            "[group_o] Delphi HHS result=%s message=%s",
            payload.get("result"), payload.get("message"),
        )
        return []

    rows: list[dict] = []
    for rec in payload.get("epidata") or []:
        try:
            epiweek = int(rec["epiweek"])
            year    = epiweek // 100
            week    = epiweek % 100
        except (KeyError, TypeError, ValueError):
            continue
        if week < 1 or week > 53:
            continue

        region_raw = str(rec.get("region", "")).upper()  # "hhs1" → "HHS1"

        wili = rec.get("wili")
        ili  = rec.get("ili")
        try:
            ili_rate = float(wili) if wili is not None else (
                       float(ili)  if ili  is not None else None)
        except (TypeError, ValueError):
            ili_rate = None

        def _si(v) -> Optional[int]:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        rows.append({
            "source":                 "delphi_hhs",
            "country":                "USA",
            "region":                 region_raw,      # HHS1..HHS10
            "year":                   year,
            "week_no":                week,
            "ili_rate":               ili_rate,        # weighted ILI%
            "activity_level_ordinal": None,
            "n_providers":            _si(rec.get("num_providers")),
            "n_patients":             _si(rec.get("num_patients")),
            "n_ili":                  _si(rec.get("num_ili")),
            "collected_at":           now_iso,
        })

    log.info("[group_o] Delphi HHS: %d rows (epiweeks %s)", len(rows), epiweek_range)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: US Delphi Census Division ILI% (9 divisions, 2010-present)
# ─────────────────────────────────────────────────────────────────────────────

_US_CENSUS_DIVISIONS: list[str] = [
    "cen1", "cen2", "cen3", "cen4", "cen5",
    "cen6", "cen7", "cen8", "cen9",
]
# cen1=New England, cen2=Mid-Atlantic, cen3=East North Central,
# cen4=West North Central, cen5=South Atlantic, cen6=East South Central,
# cen7=West South Central, cen8=Mountain, cen9=Pacific.


def _fetch_us_delphi_census_divisions(years_back: int = 3) -> list[dict]:
    """Fetch US Census Division-level weighted ILI% from Delphi COVIDcast FluView.

    Census divisions are 9 geographic groupings of US states (New England,
    Mid-Atlantic, etc.) — finer than national, coarser than HHS regions.
    Delphi exposes them as ``cen1``–``cen9`` via the FluView endpoint.
    Data availability: 2010W01 onwards (confirmed via API test 2026-05-25).

    Endpoint: https://api.delphi.cmu.edu/epidata/fluview/
    Params:   regions=cen1,...,cen9  epiweeks=201001-<current>

    Args:
        years_back: Number of historical years to fetch (default 3).
                    Set to 16+ for full history back to 2010.

    Returns:
        List of row dicts for overseas_ili_regional:
          source='delphi_census', country='USA',
          region='CEN1'..'CEN9' (uppercase), year, week_no,
          ili_rate=wili (weighted ILI%), n_providers, n_patients, n_ili.

    Raises:
        Nothing — failures logged; partial results returned.

    Performance: 1 HTTP request (all 9 divisions in one call); ~2-3 s.
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError:
        log.warning("[group_o] requests not installed — skipping Delphi census divisions")
        return []

    now     = datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    cur_year = now.year
    start_epiweek = (cur_year - years_back) * 100 + 1
    end_epiweek   = cur_year * 100 + 53
    epiweek_range = f"{start_epiweek}-{end_epiweek}"

    regions_str = ",".join(_US_CENSUS_DIVISIONS)
    log.info("[group_o] Delphi census divisions epiweeks=%s …", epiweek_range)

    try:
        resp = requests.get(
            _DELPHI_FLUVIEW_URL,
            params={"regions": regions_str, "epiweeks": epiweek_range},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        log.warning("[group_o] Delphi census divisions fetch failed: %s", e)
        return []

    if payload.get("result") != 1:
        log.warning(
            "[group_o] Delphi census div result=%s message=%s",
            payload.get("result"), payload.get("message"),
        )
        return []

    rows: list[dict] = []
    for rec in payload.get("epidata") or []:
        try:
            epiweek = int(rec["epiweek"])
            year    = epiweek // 100
            week    = epiweek % 100
        except (KeyError, TypeError, ValueError):
            continue
        if week < 1 or week > 53:
            continue

        region_raw = str(rec.get("region", "")).upper()  # "cen1" → "CEN1"

        wili = rec.get("wili")
        ili  = rec.get("ili")
        try:
            ili_rate = float(wili) if wili is not None else (
                       float(ili)  if ili  is not None else None)
        except (TypeError, ValueError):
            ili_rate = None

        def _si(v) -> Optional[int]:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        rows.append({
            "source":                 "delphi_census",
            "country":                "USA",
            "region":                 region_raw,      # CEN1..CEN9
            "year":                   year,
            "week_no":                week,
            "ili_rate":               ili_rate,        # weighted ILI%
            "activity_level_ordinal": None,
            "n_providers":            _si(rec.get("num_providers")),
            "n_patients":             _si(rec.get("num_patients")),
            "n_ili":                  _si(rec.get("num_ili")),
            "collected_at":           now_iso,
        })

    log.info("[group_o] Delphi census: %d rows (epiweeks %s)", len(rows), epiweek_range)
    return rows


def _fetch_us_nssp_ed_visits(years_back: int) -> list[dict]:
    """Fetch flu-specific ED visit percentage for all US states via Delphi COVIDcast.

    NSSP (National Syndromic Surveillance Program) tracks the fraction of
    emergency department visits where the diagnosis is influenza.  This is the
    best available public proxy for syndromic ILI surveillance at state level,
    updated weekly with ~1-week lag.

    Signal: ``pct_ed_visits_influenza`` — % of all ED visits diagnosed as influenza
    (smoothed variant ``smoothed_pct_ed_visits_influenza`` also available).
    Coverage: all 50 states + DC; a few states excluded by data-sharing agreement.
    Availability: 2022-10-01 onward.

    Args:
        years_back: Number of past years of data to retrieve.

    Returns:
        List of row dicts for overseas_ili_regional:
          source='nssp_ed_visits_flu', country='USA',
          region=2-letter state abbrev (uppercase), year, week_no,
          ili_rate=pct_ed_visits_influenza (%).

    Raises:
        RuntimeError: on HTTP error or JSON parse failure.

    Performance: 1 HTTP request (geo_value=* returns all states per epiweek batch).
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ew_range = _epiweek_range(years_back)
    log.info("[group_o] Delphi NSSP ED visits influenza (all states) epiweeks=%s …", ew_range)

    resp = requests.get(
        _DELPHI_COVIDCAST_URL,
        params={
            "data_source": "nssp",
            "signal":      "pct_ed_visits_influenza",
            "geo_type":    "state",
            "time_type":   "week",
            "geo_value":   "*",          # all states
            "time_values": ew_range,
        },
        timeout=_TIMEOUT_S,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("result") != 1:
        raise RuntimeError(
            f"Delphi NSSP: result={payload.get('result')} message={payload.get('message')}"
        )

    rows: list[dict] = []
    min_year = datetime.now().year - years_back
    for rec in payload.get("epidata") or []:
        try:
            epiweek = int(rec["time_value"])
            year    = epiweek // 100
            week    = epiweek % 100
        except (KeyError, TypeError, ValueError):
            continue
        if year < min_year or week < 1 or week > 53:
            continue

        region = str(rec.get("geo_value") or "").strip().upper()
        if not region:
            continue

        try:
            ili_rate = float(rec["value"]) if rec.get("value") is not None else None
        except (TypeError, ValueError):
            ili_rate = None

        rows.append({
            "source":                 "nssp_ed_visits_flu",
            "country":                "USA",
            "region":                 region,
            "year":                   year,
            "week_no":                week,
            "ili_rate":               ili_rate,
            "activity_level_ordinal": None,
            "collected_at":           now_iso,
        })

    log.info("[group_o] NSSP ED visits: %d rows", len(rows))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: US NHSN confirmed flu hospitalizations (CDC Socrata vdzy-6i9v)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_us_nhsn_hospitalizations(years_back: int) -> list[dict]:
    """Fetch confirmed influenza hospital admissions for all US states from CDC NHSN.

    NHSN (National Healthcare Safety Network) Hospital Respiratory Data reports
    weekly confirmed influenza admissions by jurisdiction.  Dataset vdzy-6i9v
    provides both raw admission counts and per-100k rates alongside a categorical
    level label.

    Columns used:
      jurisdiction               → region (2-letter state abbrev)
      weekendingdate             → year, week_no (via isocalendar)
      totalconfflunewadm         → n_ili (raw count)
      totalconfflunewadmper100k  → ili_rate (rate per 100,000 population)
      totalconfflunewadmper100klevel → activity_level_ordinal (1-5 via _LEVEL_LABEL_TO_INT)

    Args:
        years_back: Number of past years to retrieve.

    Returns:
        List of row dicts for overseas_ili_regional:
          source='nhsn_flu_admissions', country='USA',
          region=2-letter state abbrev (uppercase), year, week_no,
          ili_rate=per-100k admission rate, n_ili=raw count,
          activity_level_ordinal=1-5.

    Raises:
        RuntimeError: on HTTP error or JSON parse failure.

    Performance: 1-2 Socrata pages (~51 states × 52 weeks × years_back ≈ 8 K rows).
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    from datetime import date as _date

    now_iso  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    min_date = f"{datetime.now().year - years_back}-01-01"
    log.info("[group_o] CDC NHSN flu hospitalizations (vdzy-6i9v) from %s …", min_date)

    rows: list[dict] = []
    offset, page_size = 0, 50_000

    while True:
        resp = requests.get(
            _NHSN_HRD_URL,
            params={
                "$limit":  page_size,
                "$offset": offset,
                "$where":  f"weekendingdate >= '{min_date}'",
                "$order":  "weekendingdate DESC",
            },
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break

        for it in page:
            region = str(it.get("jurisdiction") or "").strip().upper()
            if not region:
                continue

            # Parse date → ISO week
            raw_date = str(it.get("weekendingdate") or "").strip()[:10]  # "YYYY-MM-DD"
            try:
                dt       = datetime.strptime(raw_date, "%Y-%m-%d")
                iso_cal  = dt.isocalendar()
                year     = iso_cal[0]
                week     = iso_cal[1]
            except ValueError:
                continue

            try:
                n_ili    = int(float(it["totalconfflunewadm"])) if it.get("totalconfflunewadm") else None
            except (TypeError, ValueError):
                n_ili    = None

            try:
                ili_rate = float(it["totalconfflunewadmper100k"]) if it.get("totalconfflunewadmper100k") else None
            except (TypeError, ValueError):
                ili_rate = None

            level_raw = str(it.get("totalconfflunewadmper100klevel") or "").strip().lower()
            ordinal   = _LEVEL_LABEL_TO_INT.get(level_raw)

            rows.append({
                "source":                 "nhsn_flu_admissions",
                "country":                "USA",
                "region":                 region,
                "year":                   year,
                "week_no":                week,
                "ili_rate":               ili_rate,   # per-100k admission rate
                "activity_level_ordinal": ordinal,    # 1=Very Low … 5=Very High
                "n_ili":                  n_ili,      # raw admission count
                "collected_at":           now_iso,
            })

        if len(page) < page_size:
            break
        offset += page_size

    log.info("[group_o] NHSN hospitalizations: %d rows (from %s)", len(rows), min_date)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: US NWSS wastewater Influenza A (CDC atcp-73re, state aggregated)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_us_wastewater_flu(years_back: int) -> list[dict]:
    """Fetch CDC NWSS wastewater Influenza A signals aggregated to US state level.

    The CDC National Wastewater Surveillance System (NWSS) tracks Influenza A
    virus RNA at ~1,500 sites nationally (dataset atcp-73re).  Each site has a
    weekly ordinal activity level (site_wval: 1.0=Very Low to 5.0=Very High).

    This function aggregates per-site data to state level by taking the mean
    site_wval across all sites in a state for each week.  Wastewater is a leading
    indicator (peaks 1-2 weeks before clinical ILI) and provides county-resolution
    coverage distinct from clinical surveillance.

    Availability: 2023-03-13 onward (Influenza A tracking start date).

    Args:
        years_back: Number of past years to retrieve (capped at 3 since data starts 2023).

    Returns:
        List of row dicts for overseas_ili_regional:
          source='nwss_flu_a', country='USA',
          region=2-letter state abbrev (uppercase), year, week_no,
          ili_rate=mean site_wval (1.0-5.0 continuous),
          activity_level_ordinal=rounded mean (1-5).

    Raises:
        RuntimeError: on HTTP error or JSON parse failure.

    Performance: 1-5 paginated Socrata requests (SoQL group-by reduces payload).
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    now_iso  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # NWSS Influenza A data starts 2023-03-13; cap years_back at 3
    min_date = f"{max(datetime.now().year - years_back, 2023)}-01-01"
    log.info("[group_o] CDC NWSS wastewater Flu-A (atcp-73re) grouped by state from %s …", min_date)

    # Use SoQL GROUP BY to aggregate at API level — avoids fetching 234K raw rows
    rows_agg: list[dict] = []
    offset, page_size = 0, 50_000

    while True:
        resp = requests.get(
            _NWSS_FLU_URL,
            params={
                "$select": (
                    "state_territory,week_end,"
                    "AVG(site_wval) as avg_wval,"
                    "COUNT(*) as n_sites"
                ),
                "$where": (
                    f"pathogen_target='Influenza A virus' "
                    f"AND week_end >= '{min_date}'"
                ),
                "$group":  "state_territory,week_end",
                "$order":  "week_end DESC",
                "$limit":  page_size,
                "$offset": offset,
            },
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        rows_agg.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    rows: list[dict] = []
    for it in rows_agg:
        state_raw = str(it.get("state_territory") or "").strip()
        abbrev    = _US_STATE_ABBREV.get(state_raw.lower())
        if abbrev is None:
            continue  # territories not in the map (skip silently)

        raw_date = str(it.get("week_end") or "").strip()[:10]
        try:
            dt      = datetime.strptime(raw_date, "%Y-%m-%d")
            iso_cal = dt.isocalendar()
            year    = iso_cal[0]
            week    = iso_cal[1]
        except ValueError:
            continue

        try:
            avg_wval = float(it["avg_wval"]) if it.get("avg_wval") is not None else None
        except (TypeError, ValueError):
            avg_wval = None

        if avg_wval is None:
            continue

        ordinal = int(round(avg_wval))
        ordinal = max(1, min(5, ordinal))  # clamp to 1-5

        rows.append({
            "source":                 "nwss_flu_a",
            "country":                "USA",
            "region":                 abbrev,
            "year":                   year,
            "week_no":                week,
            "ili_rate":               avg_wval,   # continuous 1.0-5.0 wastewater signal
            "activity_level_ordinal": ordinal,    # rounded to 1-5
            "collected_at":           now_iso,
        })

    log.info("[group_o] NWSS wastewater Flu-A: %d state-weeks (from %s)", len(rows), min_date)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: Japan JIHS (formerly NIID) prefecture ILI via weekly CSV
# ─────────────────────────────────────────────────────────────────────────────

# JIHS (Japan Institute for Health Security) — centralized in _endpoints
from simulation.collectors._endpoints import _JIHS_CSV_BASE, _JIHS_HIST_BASE

import re as _re


def _parse_jihs_csv(text: str, year: int, week: int) -> list[dict]:
    """Parse one JIHS teiten CSV into row dicts.

    CSV layout (English language file):
      Row 0: Table title
      Row 1: "{N}th week, {YYYY}" and data collection date
      Row 2: Empty
      Row 3: "Prefecture,Influenza(excld. avian influenza…),,RSV,,…"
             (disease headers; merged 2-column spans = Current week + per sentinel)
      Row 4: ",Current week,per sentinel,Current week,per sentinel,…"
      Row 5: "Total No.,…" (national aggregate)
      Rows 6–52: 47 prefectures

    Influenza "per sentinel" (col 2) = weekly cases per sentinel reporting station,
    normalised rate equivalent to ILI%.  Stored in ``ili_rate``.
    Influenza "Current week" (col 1) = absolute weekly case count.  Stored in ``n_ili``.

    Args:
        text:  Raw CSV text (decoded from response bytes).
        year:  ISO year (parsed by caller from URL).
        week:  ISO week number (parsed by caller from URL).

    Returns:
        List of row dicts; empty list on parse failure.

    Performance: O(n_prefectures) = O(47).
    Side effects: None.
    Caller responsibility: text must be non-empty.
    """
    import csv as _csv, io as _io
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        reader  = _csv.reader(_io.StringIO(text))
        rows_raw = [row for row in reader]
    except Exception:
        return []

    if len(rows_raw) < 6:
        return []

    # Find Influenza column index in disease header row (row 3)
    disease_row = rows_raw[3] if len(rows_raw) > 3 else []
    flu_col: Optional[int] = None
    for i, cell in enumerate(disease_row):
        if "Influenza" in cell and "avian" in cell.lower():
            flu_col = i
            break
    if flu_col is None:
        return []

    per_sentinel_col = flu_col + 1  # "per sentinel" immediately follows disease header

    rows: list[dict] = []
    for raw_row in rows_raw[5:]:  # skip title, week-info, empty, header x2
        if not raw_row or not raw_row[0].strip():
            continue
        pref = raw_row[0].strip()
        if pref in ("Total No.", "Prefecture"):
            continue

        # Influenza per sentinel (ILI rate proxy)
        try:
            ps_raw   = raw_row[per_sentinel_col].strip() if len(raw_row) > per_sentinel_col else "-"
            ili_rate = float(ps_raw) if ps_raw not in ("-", "") else None
        except (ValueError, IndexError):
            ili_rate = None

        # Absolute weekly case count
        try:
            wk_raw = raw_row[flu_col].strip() if len(raw_row) > flu_col else "-"
            n_ili  = int(float(wk_raw)) if wk_raw not in ("-", "") else None
        except (ValueError, IndexError):
            n_ili  = None

        rows.append({
            "source":                 "jihs_prefecture",
            "country":                "JPN",
            "region":                 pref,         # English romaji (Hokkaido, Tokyo, etc.)
            "year":                   year,
            "week_no":                week,
            "ili_rate":               ili_rate,     # cases per sentinel (normalised rate)
            "activity_level_ordinal": None,
            "n_providers":            None,
            "n_patients":             None,
            "n_ili":                  n_ili,        # absolute weekly count
            "collected_at":           now_iso,
        })
    return rows


def _fetch_jp_prefecture_ili(years_back: int) -> list[dict]:
    """Fetch Japan JIHS prefecture-level influenza sentinel data from weekly CSVs.

    JIHS (Japan Institute for Health Security, formerly NIID) publishes one CSV
    per week at a predictable URL with no authentication required.  This function
    iterates flu-season weeks for the requested history, fetching each in parallel.

    URL pattern:
      https://id-info.jihs.go.jp/en/surveillance/idwr/rapid/{YEAR}/{WW:02d}/teiten{WW:02d}.csv
    Source: ``teiten`` (weekly reported cases) — NOT ``teitenrui`` (cumulative).

    Flu season weeks (skipping summer): 1–20 (Jan–May) and 36–53 (Sep–Dec).
    Off-season weeks return 404 and are silently skipped.

    Args:
        years_back: Number of past flu seasons to retrieve (each season ≈ 38 active weeks).

    Returns:
        List of row dicts for overseas_ili_regional:
          source='jihs_prefecture', country='JPN',
          region=prefecture romaji name (English, e.g. "Tokyo", "Hokkaido"),
          year, week_no,
          ili_rate=cases per sentinel site (normalised ILI rate proxy),
          n_ili=absolute weekly case count.

    Raises:
        Nothing — per-week failures silently skipped; partial results returned.

    Performance: ~38 weeks/year × years_back requests, parallelised with 10 threads.
                 Typical: 2-8 s for years_back=3.
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except ImportError as e:
        log.warning("[group_o] requests/concurrent not available — skipping JIHS Japan: %s", e)
        return []

    now    = datetime.now()
    session = requests.Session()
    session.headers["User-Agent"] = "MPH-flu-collector/1.0"

    # Build list of (year, week) tuples to fetch (flu-season weeks only)
    targets: list[tuple[int, int]] = []
    min_year = now.year - years_back
    for yr in range(min_year, now.year + 1):
        max_week = now.isocalendar()[1] if yr == now.year else 53
        for wk in range(1, max_week + 1):
            # Skip summer (off-season) — JIHS typically doesn't publish weeks 21-35
            if 21 <= wk <= 35:
                continue
            targets.append((yr, wk))

    log.info("[group_o] JIHS prefecture ILI: fetching %d week-CSVs (years_back=%d) …",
             len(targets), years_back)

    def _fetch_one(yr: int, wk: int) -> list[dict]:
        # Three URL generations (confirmed 2026-05-25):
        #   yr ≤ 2014: 2-digit year subdir (1247 = 2012W47)
        #   2015-2022: 4-digit year subdir (201536 = 2015W36)
        #   2023+:     new JIHS public portal
        if yr <= 2014:
            url = f"{_JIHS_HIST_BASE}/idwr-e{yr}/{yr % 100:02d}{wk:02d}/teiten{wk:02d}.csv"
        elif yr <= 2022:
            url = f"{_JIHS_HIST_BASE}/idwr-e{yr}/{yr}{wk:02d}/teiten{wk:02d}.csv"
        else:
            url = f"{_JIHS_CSV_BASE}/{yr}/{wk:02d}/teiten{wk:02d}.csv"
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            return _parse_jihs_csv(r.content.decode("utf-8-sig", errors="replace"), yr, wk)
        except Exception:
            return []

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_one, yr, wk): (yr, wk) for yr, wk in targets}
        for fut in as_completed(futures):
            all_rows.extend(fut.result())

    if all_rows:
        log.info("[group_o] JIHS: %d Japan prefecture-week rows", len(all_rows))
    else:
        log.warning("[group_o] JIHS prefecture CSV: no rows returned (off-season or connectivity issue)")
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: Hong Kong CHP Flu Express (single CSV, national, 2014-present)
# ─────────────────────────────────────────────────────────────────────────────

_HK_FLUX_URL = "https://www.chp.gov.hk/files/misc/flux_data.csv"


def _fetch_hk_ili(years_back: int) -> list[dict]:
    """Fetch Hong Kong CHP Flu Express weekly ILI and flu surveillance data.

    Hong Kong Centre for Health Protection (CHP) publishes a single CSV file
    (flux_data.csv) containing all weekly flu metrics from 2014 to the present.
    Updated weekly during flu season.  No key or login required.

    Key columns used:
      Year, Week           → year, week_no
      ILI_FMC              → ili_rate (ILI rate at Family Medicine Clinics, %)
      ILI_PMP              → fallback ILI rate at Private Medical Practitioners
      H1, H3, B, AandB    → flu subtype lab counts → n_ili = AandB (total)
      Adm_All              → n_patients (total hospital admissions)

    Args:
        years_back: Number of past years to include (filters by Year column).

    Returns:
        List of row dicts for overseas_ili_regional:
          source='hk_chp', country='HKG',
          region='Hong Kong' (territory-wide; no sub-territory breakdown),
          year, week_no,
          ili_rate=ILI_FMC (or ILI_PMP fallback),
          n_ili=total flu A+B lab counts (AandB column),
          n_patients=total hospital admissions.

    Raises:
        RuntimeError: on HTTP error or CSV parse failure.

    Performance: 1 HTTP GET (~94 KB); ~646 rows total; filtered to ~52×years_back.
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    log.info("[group_o] HK CHP flux_data.csv …")
    resp = requests.get(_HK_FLUX_URL, timeout=_TIMEOUT_S)
    resp.raise_for_status()

    import csv as _csv, io as _io
    text    = resp.content.decode("utf-8-sig", errors="replace")
    reader  = _csv.DictReader(_io.StringIO(text))

    now_iso  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    min_year = datetime.now().year - years_back
    rows: list[dict] = []

    for rec in reader:
        try:
            year = int(rec.get("Year") or 0)
            week = int(rec.get("Week") or 0)
        except (TypeError, ValueError):
            continue
        if year < min_year or week < 1 or week > 53:
            continue

        def _f(key: str) -> Optional[float]:
            v = rec.get(key, "").strip()
            try:
                return float(v) if v else None
            except ValueError:
                return None

        def _i(key: str) -> Optional[int]:
            v = rec.get(key, "").strip()
            try:
                return int(float(v)) if v else None
            except ValueError:
                return None

        ili_fmc = _f("ILI_FMC")
        ili_pmp = _f("ILI_PMP")
        ili_rate = ili_fmc if ili_fmc is not None else ili_pmp

        rows.append({
            "source":                 "hk_chp",
            "country":                "HKG",
            "region":                 "Hong Kong",
            "year":                   year,
            "week_no":                week,
            "ili_rate":               ili_rate,    # ILI% at Family Medicine Clinics
            "activity_level_ordinal": None,
            "n_providers":            None,
            "n_patients":             _i("Adm_All"),   # total hospital admissions
            "n_ili":                  _i("AandB"),     # total flu A+B lab counts
            "collected_at":           now_iso,
        })

    log.info("[group_o] HK CHP: %d rows (year >= %d)", len(rows), min_year)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: Germany RKI Bundesland-level (IfSG confirmed influenza)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_de_bundesland_ili(years_back: int) -> list[dict]:
    """Download RKI confirmed influenza Inzidenz by Bundesland from GitHub Open Data.

    Source: robert-koch-institut/Influenzafaelle_in_Deutschland (TSV, weekly update)
    Columns used: Meldewoche (YYYY-Www), Region, Region_Id, Altersgruppe,
                  Fallzahl (case count), Inzidenz (per 100,000).
    Filter: Altersgruppe == "00+" (all-age aggregate only).
    Region_Id "00" = Deutschland (national summary); "01"-"16" = Bundesländer.

    The Inzidenz column (IfSG lab/clinical confirmed, per 100,000) is stored in
    ``ili_rate`` — the best available granular flu metric for Germany with a
    public no-key API.  Note this is *confirmed influenza* incidence, not
    self-reported ILI rate.

    Args:
        years_back: Filter to rows where parsed year >= (current year - years_back).

    Returns:
        List of row dicts for overseas_ili_regional:
          source='rki_bundesland', country='DEU',
          region=Bundesland name (German) or 'Deutschland' for national,
          year, week_no, ili_rate=Inzidenz per 100k, n_ili=Fallzahl.

    Raises:
        RuntimeError: on HTTP error or TSV parse failure.

    Performance: ~650 KB download (~0.5 s), ~18 K rows total; filter reduces to ~6 K.
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    log.info("[group_o] RKI Bundesland ILI TSV from GitHub …")
    resp = requests.get(_DE_RKI_TSV_URL, timeout=_TIMEOUT_S)
    resp.raise_for_status()

    # BOM-safe UTF-8; TSV uses \t separator
    text = resp.content.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise RuntimeError("RKI TSV is empty")

    header = [h.strip() for h in lines[0].split("\t")]
    # Expected headers (German): Meldewoche  Region  Region_Id  Altersgruppe  Fallzahl  Inzidenz
    # Build index map defensively (handles BOM, reorder)
    idx: dict[str, int] = {h.lstrip("﻿"): i for i, h in enumerate(header)}

    required = {"Meldewoche", "Region", "Altersgruppe", "Fallzahl", "Inzidenz"}
    missing = required - set(idx)
    if missing:
        raise RuntimeError(f"RKI TSV missing columns: {missing}. Got: {list(idx)}")

    min_year = datetime.now().year - years_back
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            continue

        # Altersgruppe filter: only "00+" (all ages) to avoid duplication
        alters = parts[idx["Altersgruppe"]].strip()
        if alters != "00+":
            continue

        # Parse ISO week: "2024-W03" → year=2024, week=3
        mw = parts[idx["Meldewoche"]].strip()
        try:
            yr_str, wk_str = mw.split("-W")
            year   = int(yr_str)
            week   = int(wk_str)
        except (ValueError, AttributeError):
            continue

        if year < min_year or week < 1 or week > 53:
            continue

        region = parts[idx["Region"]].strip()
        if not region:
            continue

        try:
            fallzahl  = int(float(parts[idx["Fallzahl"]].strip().replace(",", ".")))
        except (ValueError, IndexError):
            fallzahl  = None

        try:
            inzidenz  = float(parts[idx["Inzidenz"]].strip().replace(",", "."))
        except (ValueError, IndexError):
            inzidenz  = None

        rows.append({
            "source":                 "rki_bundesland",
            "country":                "DEU",
            "region":                 region,
            "year":                   year,
            "week_no":                week,
            "ili_rate":               inzidenz,   # per 100,000 confirmed influenza
            "activity_level_ordinal": None,
            "n_providers":            None,
            "n_patients":             None,
            "n_ili":                  fallzahl,   # absolute case count
            "collected_at":           now_iso,
        })

    log.info("[group_o] RKI Bundesland: %d rows (year >= %d)", len(rows), min_year)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector: France Sentiweb regional ILI (REG level, per 100k)
# ─────────────────────────────────────────────────────────────────────────────

_SENTIWEB_FR_REG_URL = "https://www.sentiweb.fr/datasets/incidence-REG-3.csv"


def _fetch_fr_sentiweb_regional(years_back: int = 3) -> list[dict]:
    """Fetch France Sentiweb regional ILI incidence (per 100k) from REG-3 CSV.

    Sentiweb publishes weekly ILI incidence per 100k at the administrative
    region level (22 old French regions, going back to 1984).  CSV starts with
    one '#' comment line containing JSON metadata, followed by a standard
    CSV header and data rows.

    Key columns used:
      week    — YYYYWW format (e.g. 202620 = 2026 week 20)
      inc100  — ILI incidence per 100,000 inhabitants
      geo_name — French region name (uppercase, e.g. 'ALSACE')

    Args:
        years_back: Number of past calendar years to include (default 3).

    Returns:
        List of row dicts for overseas_ili_regional:
          source='sentiweb_fr_regional', country='FRA', region=geo_name,
          ili_rate=inc100 (per 100k), other numeric fields None.

    Raises:
        RuntimeError: if HTTP fetch fails.

    Performance: 1 HTTP request, ~300 KB CSV.
    Side effects: None (pure fetch).
    Caller responsibility: `requests` must be installed.
    """
    try:
        import csv as _csv
        import io as _io
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    current_year = datetime.now().year
    min_year = current_year - (years_back - 1)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = requests.get(_SENTIWEB_FR_REG_URL, timeout=30,
                            headers={"Accept": "text/csv"})
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Sentiweb FR regional fetch failed: {e}") from e

    text = resp.content.decode("utf-8-sig")
    # Strip leading '#' comment lines (Sentiweb metadata JSON)
    data_lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if not data_lines:
        log.warning("[group_o] Sentiweb FR regional: CSV empty after stripping comments")
        return []

    reader = _csv.DictReader(_io.StringIO("\n".join(data_lines)))
    rows: list[dict] = []
    for r in reader:
        raw_week = str(r.get("week") or "").strip()
        if len(raw_week) < 6:
            continue
        try:
            year    = int(raw_week[:4])
            week_no = int(raw_week[4:])
        except ValueError:
            continue
        if year < min_year:
            continue

        inc100  = None
        raw_inc = str(r.get("inc100") or "").strip()
        try:
            v = float(raw_inc)
            inc100 = v if v == v else None  # NaN guard
        except ValueError:
            pass

        region = str(r.get("geo_name") or "").strip().upper()
        if not region or inc100 is None:
            continue

        rows.append({
            "source":                "sentiweb_fr_regional",
            "country":               "FRA",
            "region":                region,
            "year":                  year,
            "week_no":               week_no,
            "ili_rate":              inc100,
            "n_providers":           None,
            "n_patients":            None,
            "n_ili":                 None,
            "activity_level_ordinal": None,
            "collected_at":          now_iso,
        })

    log.info("[group_o] Sentiweb FR regional: %d rows (year >= %d)", len(rows), min_year)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    backfill_days: Optional[int] = None,
    db_path: Optional[str | Path] = None,
    years_back: int = 3,
    skip_delphi: bool = False,
    skip_nssp: bool = False,
    skip_nhsn: bool = False,
    skip_ww: bool = False,
    skip_hhs: bool = False,
    skip_census: bool = False,
    skip_jp: bool = False,
    skip_hk: bool = False,
    skip_de: bool = False,
    skip_fr: bool = False,
    skip_us: bool = False,   # legacy: if True overrides skip_delphi/nssp/nhsn/ww/hhs
) -> dict:
    """Collect regional-level ILI/hospitalization/wastewater for US states, JP, DE, FR.

    Sources attempted in order (all non-fatal — one failure does not abort others):

    US sources (country='USA'):
      1. Delphi COVIDcast FluView  → source='delphi_covidcast'      (actual ILI%, all 51 states)
      2. Delphi HHS regions        → source='delphi_hhs'            (actual ILI%, HHS1-10, 1997+)
      3. Delphi Census divisions   → source='delphi_census'         (actual ILI%, CEN1-9, 2010+)
      4. Delphi NSSP ED visits     → source='nssp_ed_visits_flu'    (% flu ED visits, 50 states)
      5. CDC NHSN hospitalizations → source='nhsn_flu_admissions'   (per-100k + count, 51)
      6. CDC NWSS wastewater Flu-A → source='nwss_flu_a'            (1-5 ordinal, state agg)

    Other countries:
      7. Japan JIHS prefecture     → source='jihs_prefecture'      (per-sentinel ILI, 47 pref, 2012+)
      8. Hong Kong CHP Flu Express → source='hk_chp'               (ILI%, flu subtypes, admissions)
      9. Germany RKI Bundesland    → source='rki_bundesland'        (Inzidenz/100k)

    Note: CDC dataset 6svj-q4zv (FluView ordinal) archived 2024-10-16 — removed.
    Note: id.niid.go.jp (old NIID API) DNS-dead — replaced by JIHS CSV (id-info.jihs.go.jp).

    Args:
        backfill_days: Orchestrator override → years_back = max(current, days//365+1).
        db_path:       Path to epi_real_seoul.db (default: standard project path).
        years_back:    Number of historical years to collect/filter (default 3).
        skip_delphi:   Skip Delphi FluView ILI% (primary US ILI source, 51 states).
        skip_hhs:      Skip Delphi HHS regional ILI% (HHS1-10, 1997-present).
        skip_census:   Skip Delphi Census division ILI% (CEN1-9, 2010-present).
        skip_nssp:     Skip Delphi NSSP ED visit % (syndromic ILI proxy).
        skip_nhsn:     Skip CDC NHSN hospitalization data.
        skip_ww:       Skip CDC NWSS wastewater Flu-A (leading indicator).
        skip_jp:       Skip Japan JIHS prefecture CSV collection (2010+, all years).
        skip_hk:       Skip Hong Kong CHP Flu Express.
        skip_de:       Skip Germany RKI Bundesland.
        skip_fr:       Skip France Sentiweb regional ILI.
        skip_us:       Legacy flag — if True, skips ALL US sources regardless of others.

    Returns:
        dict: {inserted, skipped, errors: list[str], sources_fetched: list[str]}

    Raises:
        Nothing — all errors are caught and returned in `errors`.

    Performance: ~30-60 s (Delphi fluview 10 batches + NSSP + NHSN + NWSS + DE TSV).
    Side effects: creates overseas_ili_regional if absent; upserts rows.
    Caller responsibility: `requests` must be installed; DB must exist.
    """
    if backfill_days is not None and backfill_days > 0:
        years_back = max(years_back, backfill_days // 365 + 1)

    # Legacy skip_us overrides individual flags
    if skip_us:
        skip_delphi = skip_nssp = skip_nhsn = skip_ww = skip_hhs = skip_census = True

    resolved = _resolve_db(db_path)
    result: dict = {"inserted": 0, "skipped": 0, "errors": [], "sources_fetched": []}

    if not resolved.exists():
        result["errors"].append(f"DB not found: {resolved}")
        return result

    con = _connect(resolved)
    try:
        _ensure_table(con)
        all_rows: list[dict] = []

        # ── US: Delphi FluView (actual ILI%, 51 states) ─────────────────────
        if not skip_delphi:
            try:
                rows = _fetch_us_state_ili_delphi(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("delphi_covidcast")
            except Exception as e:
                log.error("[group_o] Delphi COVIDcast ILI failed: %s", e)
                result["errors"].append(f"delphi_covidcast: {e}")

        # ── US: Delphi HHS regional ILI% (HHS1-10, 1997-present) ───────────
        if not skip_hhs:
            try:
                rows = _fetch_us_delphi_hhs_regions(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("delphi_hhs")
            except Exception as e:
                log.error("[group_o] Delphi HHS regions failed: %s", e)
                result["errors"].append(f"delphi_hhs: {e}")

        # ── US: Delphi Census divisions ILI% (CEN1-9, 2010-present) ────────
        if not skip_census:
            try:
                rows = _fetch_us_delphi_census_divisions(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("delphi_census")
            except Exception as e:
                log.error("[group_o] Delphi census divisions failed: %s", e)
                result["errors"].append(f"delphi_census: {e}")

        # ── US: NSSP ED visit % (syndromic, flu-specific) ───────────────────
        if not skip_nssp:
            try:
                rows = _fetch_us_nssp_ed_visits(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("nssp_ed_visits_flu")
            except Exception as e:
                log.error("[group_o] NSSP ED visits failed: %s", e)
                result["errors"].append(f"nssp_ed_visits: {e}")

        # ── US: NHSN hospitalizations (count + per-100k + level) ────────────
        if not skip_nhsn:
            try:
                rows = _fetch_us_nhsn_hospitalizations(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("nhsn_flu_admissions")
            except Exception as e:
                log.error("[group_o] NHSN hospitalizations failed: %s", e)
                result["errors"].append(f"nhsn_hospitalizations: {e}")

        # ── US: NWSS wastewater Flu-A (leading indicator) ───────────────────
        if not skip_ww:
            try:
                rows = _fetch_us_wastewater_flu(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("nwss_flu_a")
            except Exception as e:
                log.error("[group_o] NWSS wastewater failed: %s", e)
                result["errors"].append(f"nwss_wastewater: {e}")

        # ── Japan: JIHS prefecture CSV (47 prefectures, weekly) ─────────────
        if not skip_jp:
            try:
                rows = _fetch_jp_prefecture_ili(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("jihs_prefecture")
            except Exception as e:
                log.error("[group_o] Japan JIHS prefecture failed: %s", e)
                result["errors"].append(f"jp_prefecture: {e}")

        # ── Hong Kong: CHP Flu Express (ILI% + subtypes + admissions) ────────
        if not skip_hk:
            try:
                rows = _fetch_hk_ili(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("hk_chp")
            except Exception as e:
                log.error("[group_o] HK CHP failed: %s", e)
                result["errors"].append(f"hk_chp: {e}")

        # ── Germany: RKI Bundesland (confirmed influenza incidence) ──────────
        if not skip_de:
            try:
                rows = _fetch_de_bundesland_ili(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("rki_bundesland")
            except Exception as e:
                log.error("[group_o] RKI Bundesland failed: %s", e)
                result["errors"].append(f"de_bundesland: {e}")

        # ── France: Sentiweb regional ILI per 100k (22 regions, 1984-present) ─
        if not skip_fr:
            try:
                rows = _fetch_fr_sentiweb_regional(years_back)
                all_rows.extend(rows)
                if rows:
                    result["sources_fetched"].append("sentiweb_fr_regional")
            except Exception as e:
                log.error("[group_o] Sentiweb FR regional failed: %s", e)
                result["errors"].append(f"fr_sentiweb_regional: {e}")

        ins, skp = _upsert(con, all_rows)
        result["inserted"] = ins
        result["skipped"]  = skp

    finally:
        con.close()

    log.info(
        "[group_o] Done: inserted=%d skipped=%d errors=%d sources=%s",
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
        description="Regional ILI collector (US state + JP prefecture + DE Bundesland)"
    )
    parser.add_argument("--years-back", type=int, default=3,
                        help="Years of ILI history to collect/filter (default 3)")
    parser.add_argument("--skip-delphi", action="store_true",
                        help="Skip Delphi FluView ILI%% (primary US source, 51 states)")
    parser.add_argument("--skip-hhs",    action="store_true",
                        help="Skip Delphi HHS regional ILI%% (HHS1-10, 1997-present)")
    parser.add_argument("--skip-census", action="store_true",
                        help="Skip Delphi Census division ILI%% (CEN1-9, 2010-present)")
    parser.add_argument("--skip-nssp",   action="store_true",
                        help="Skip Delphi NSSP ED visit %% flu")
    parser.add_argument("--skip-nhsn",   action="store_true",
                        help="Skip CDC NHSN flu hospitalizations")
    parser.add_argument("--skip-ww",     action="store_true",
                        help="Skip CDC NWSS wastewater Flu-A")
    parser.add_argument("--skip-us",     action="store_true",
                        help="Skip ALL US sources (legacy; overrides individual skip flags)")
    parser.add_argument("--skip-de",     action="store_true",
                        help="Skip Germany RKI Bundesland collection")
    parser.add_argument("--skip-jp",     action="store_true",
                        help="Skip Japan JIHS prefecture CSV collection (2010+)")
    parser.add_argument("--skip-hk",     action="store_true",
                        help="Skip Hong Kong CHP Flu Express")
    parser.add_argument("--skip-fr",     action="store_true",
                        help="Skip France Sentiweb regional ILI")
    parser.add_argument("--db", default=None,
                        help="Path to epi_real_seoul.db")
    args = parser.parse_args()

    r = run(
        db_path=args.db,
        years_back=args.years_back,
        skip_delphi=args.skip_delphi,
        skip_hhs=args.skip_hhs,
        skip_census=args.skip_census,
        skip_nssp=args.skip_nssp,
        skip_nhsn=args.skip_nhsn,
        skip_ww=args.skip_ww,
        skip_us=args.skip_us,
        skip_jp=args.skip_jp,
        skip_hk=args.skip_hk,
        skip_de=args.skip_de,
        skip_fr=args.skip_fr,
    )
    print(json.dumps(r, indent=2, default=str))
    sys.exit(1 if r["errors"] else 0)
