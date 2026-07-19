"""WHO FluNet (VIW_FNT) + WHO FluID (VIW_FID_EPI) — global virological + EU ILI.

Sprint β Item 4 full migration (Codex analysis 2026-05-26): real bodies
moved here from group_i_overseas.py. The legacy module re-exports for
back-compat.
"""
from __future__ import annotations

import csv
import io
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from simulation.collectors._endpoints import (
    _WHO_BASE_URL,   # WHO FluNet xMart API
    _WHO_FLUID_URL,  # WHO FluID (EU sentinel ILI consultation rate)
)
from simulation.collectors.overseas._common import (
    _REQUEST_TIMEOUT_S,
    _safe_float,
    _safe_int,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Countries to fetch from WHO FluNet (ISO3 COUNTRY_CODE values).
# EU/US already covered by higher-resolution sources (ECDC/Delphi/JIHS);
# APAC block (AUS/CHN/HKG/SGP) is FluNet-only → positivity_pct used as ILI proxy.
WHO_TARGET_COUNTRIES = {
    # Legacy EU/US (FluNet positivity supplements higher-res ILI sources)
    "USA", "JPN", "GBR", "DEU", "FRA", "NLD", "SWE", "KOR",
    # APAC — added 2026-05-25 (confirmed accessible via xmart-api-public.who.int)
    # AU from 1996W52, CN from 2005W39, HK from ~2009, SG from 1997W05.
    "AUS", "CHN", "HKG", "SGP",
}

# Friendly ISO3 → ISO2 map (overseas_ili.country column uses ISO2)
_ISO3_TO_ISO2: dict[str, str] = {
    "USA": "US", "JPN": "JP", "GBR": "GB", "DEU": "DE",
    "FRA": "FR", "NLD": "NL", "SWE": "SE", "KOR": "KR",
    # APAC additions
    "AUS": "AU", "CHN": "CN", "HKG": "HK", "SGP": "SG",
}

# How many past years to fetch (default: 3 = current + 2 prior)
_DEFAULT_YEARS_BACK = 3

# EU-10 countries covered by WHO FluID (ISO3 → ISO2).
# ITA/ESP/BEL/AUT/POL/ROU add new countries; FRA/GBR/NLD/DEU supplement FluNet.
_FLUID_TARGET_ISO3: dict[str, str] = {
    "FRA": "FR", "GBR": "GB", "ITA": "IT", "ESP": "ES",
    "NLD": "NL", "BEL": "BE", "AUT": "AT", "POL": "PL",
    "ROU": "RO", "DEU": "DE",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector 1 — WHO FluNet
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_who_flunet(years_back: int = _DEFAULT_YEARS_BACK) -> list[dict]:
    """Fetch WHO FluNet virological surveillance for 12 target countries.

    Coverage (ISO3 → ISO2):
      Legacy : USA→US  JPN→JP  GBR→GB  DEU→DE  FRA→FR  NLD→NL  SWE→SE  KOR→KR
      APAC   : AUS→AU (1996+)  CHN→CN (2005+)  HKG→HK (~2009+)  SGP→SG (1997+)

    Column-name fix (2026-05-25): xMart VIW_FNT uses COUNTRY_CODE (not ISO_CODE3),
    ISO2 (not ISO_CODE2), ISO_WEEK (not MMWR_WEEKNO).  Previous code returned 0 rows.

    ILI proxy: FluNet has no direct ILI rate.  ``ili_rate`` stores positivity_pct
    (INF_ALL / (INF_ALL+INF_NEGATIVE) × 100) as the best available numeric signal.
    For APAC countries this is the only available ILI proxy; for JP/US it is
    supplemented by higher-resolution rows from JIHS/Delphi (different source keys).

    Args:
        years_back: Calendar years to request (e.g. 3 → ISO_YEAR ≥ current-2).

    Returns:
        List of row dicts with keys:
          source, country, year, week_no, ili_rate, specimen_positive,
          specimen_total, influenza_a, influenza_b, positivity_pct, collected_at.

    Raises:
        RuntimeError: if HTTP request fails after 2 retries.

    Performance: 1-2 HTTP requests, ~200 KB CSV, ~3 s network.
    Side effects: None (pure fetch).
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed — pip install requests") from e

    current_year = datetime.now().year
    min_year = current_year - (years_back - 1)

    # Build OData filter for target countries.
    # Actual column name confirmed 2026-05-25: COUNTRY_CODE (ISO3), NOT ISO_CODE3.
    iso3_list = ",".join(f"'{iso3}'" for iso3 in sorted(WHO_TARGET_COUNTRIES))
    odata_filter = (
        f"ISO_YEAR ge {min_year} and COUNTRY_CODE in ({iso3_list})"
    )
    params = {
        "$format": "csv",
        "$filter": odata_filter,
        # Confirmed column names 2026-05-25 (COUNTRY_CODE/ISO2/ISO_WEEK, NOT _CODE3/_CODE2/MMWR_WEEKNO)
        "$select": (
            "COUNTRY_CODE,ISO2,ISO_YEAR,ISO_WEEK,"
            "SPEC_PROCESSED_NB,INF_A,INF_B,INF_ALL,INF_NEGATIVE,ILI_ACTIVITY"
        ),
    }

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []

    for attempt in range(2):
        try:
            resp = requests.get(
                _WHO_BASE_URL,
                params=params,
                timeout=_REQUEST_TIMEOUT_S,
                # NOTE: do NOT send Accept: text/csv — xMart returns HTTP 406.
                # The $format=csv param is sufficient (same pattern as WHO FluID).
            )
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 1:
                raise RuntimeError(f"WHO FluNet fetch failed after 2 attempts: {e}") from e
            log.warning("[overseas.who] WHO FluNet attempt %d failed: %s — retrying", attempt + 1, e)
            time.sleep(5)

    text = resp.content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        # COUNTRY_CODE is the ISO3 field (confirmed 2026-05-25 metadata check)
        iso3 = r.get("COUNTRY_CODE", "").strip().upper()
        if iso3 not in WHO_TARGET_COUNTRIES:
            continue
        country = _ISO3_TO_ISO2.get(iso3, iso3[:2])
        year = _safe_int(r.get("ISO_YEAR"))
        week = _safe_int(r.get("ISO_WEEK"))  # ISO_WEEK, not MMWR_WEEKNO
        if year is None or week is None:
            continue

        spec_proc = _safe_int(r.get("SPEC_PROCESSED_NB"))
        inf_a     = _safe_int(r.get("INF_A"))
        inf_b     = _safe_int(r.get("INF_B"))
        inf_total = _safe_int(r.get("INF_ALL")) or 0
        # Keep raw Optional[int] for INF_NEGATIVE — None means "not reported by this country".
        # Some countries (CN, HK) submit only positives; INF_NEGATIVE field is empty/null.
        inf_neg_raw = _safe_int(r.get("INF_NEGATIVE"))  # None → not reported; 0 → truly 0

        # Positivity rate selection (confirmed 2026-05-25 via live API test):
        #   AU/SG: INF_NEGATIVE is populated → use (INF_ALL+INF_NEGATIVE) as denominator.
        #   CN/HK: INF_NEGATIVE is empty → use SPEC_PROCESSED_NB as denominator.
        positivity: Optional[float] = None
        denom_tested: int = 0
        if inf_neg_raw is not None:
            # Country reports negatives → reliable denominator
            denom_tested = inf_total + inf_neg_raw
            if denom_tested > 0:
                positivity = round(inf_total / denom_tested * 100.0, 2)
        elif spec_proc and spec_proc >= 10 and spec_proc >= inf_total:
            # Country only submits positives → fall back to SPEC_PROCESSED_NB
            denom_tested = spec_proc
            positivity = round(inf_total / spec_proc * 100.0, 2)

        # ili_rate: FluNet has no direct ILI consultation rate.
        # For APAC-only countries (AUS/CHN/HKG/SGP) positivity_pct is the only
        # numeric ILI proxy — store it in ili_rate so feature engineering picks it
        # up.  For countries with better ILI sources (JP→japan_jihs, US→delphi)
        # this row's ili_rate will be ignored by the model (different source key).
        ili_rate = positivity  # best available numeric proxy from FluNet

        rows.append({
            "source":            "who_flunet",
            "country":           country,
            "year":              year,
            "week_no":           week,
            "ili_rate":          ili_rate,        # flu positivity% as ILI proxy
            "specimen_positive": inf_total,
            "specimen_total":    denom_tested or spec_proc,  # denominator used for positivity
            "influenza_a":       inf_a,
            "influenza_b":       inf_b,
            "positivity_pct":    positivity,
            "collected_at":      now_iso,
        })

    log.info("[overseas.who] WHO FluNet: %d rows fetched (years ≥ %d)", len(rows), min_year)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector 2 — WHO FluID (EU ILI consultation rate)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_who_fluid_eu(years_back: int = _DEFAULT_YEARS_BACK) -> list[dict]:
    """Fetch WHO FluID ILI incidence for EU-10 countries (VIW_FID_EPI).

    VIW_FID_EPI column facts (confirmed 2026-05-24):
      COUNTRY_CODE = ISO3 code  (NOT ISO_ALPHA3 or ISO_3ALPHA)
      ISO_YEAR / ISO_WEEK       = year / week number
      ILI_CASE                  = raw ILI case count (sentinel GP network)
      ILI_OUTPATIENTS           = always empty in EU-10 reporting
      ILI_POP_COV               = sentinel network population coverage
      AGEGROUP_CODE             = '0TO4','5TO14','15TO64','65TO','All','UNKNOWN'

    ILI rate is derived as ILI_CASE / ILI_POP_COV × 100,000 (per-100k sentinel
    incidence), matching Sentiweb's inc100 metric.  Only AGEGROUP_CODE='All'
    rows are stored to avoid double-counting age-stratified sub-rows.

    NOTE: Do NOT send Accept: text/csv — the xMart endpoint rejects it (HTTP 406).
    The $format=csv param alone is sufficient.

    Args:
        years_back: Calendar years back from current year to fetch.

    Returns:
        List of row dicts for overseas_ili, source='who_fluid'.
        ili_rate = round(ILI_CASE / ILI_POP_COV × 100_000, 4)  (per 100k).
        specimen_positive = ILI_CASE (raw count).
        specimen_total    = ILI_POP_COV (sentinel pop coverage).

    Raises:
        RuntimeError: if all HTTP fetch attempts fail.

    Performance: 1-2 HTTP requests, ~1 MB CSV, ~8 s.
    Side effects: None (pure fetch).
    Caller responsibility: caller upserts rows into DB.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed — pip install requests") from e

    current_year = datetime.now().year
    min_year = current_year - (years_back - 1)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    iso3_list = ",".join(f"'{c}'" for c in sorted(_FLUID_TARGET_ISO3))

    # Confirmed working filter (2026-05-24 live test):
    # COUNTRY_CODE is the ISO3 field; AGEGROUP_CODE='All' returns aggregate rows only.
    _filter_attempts = [
        (
            f"COUNTRY_CODE in ({iso3_list}) "
            f"and ISO_YEAR ge {min_year} "
            f"and AGEGROUP_CODE eq 'All'"
        ),
        # Fallback: no country server-side filter (client-side country filter below)
        f"ISO_YEAR ge {min_year} and AGEGROUP_CODE eq 'All'",
    ]

    resp = None
    for attempt, odata_filter in enumerate(_filter_attempts):
        try:
            # NOTE: no Accept header — xMart returns HTTP 406 with Accept: text/csv
            r = requests.get(
                _WHO_FLUID_URL,
                params={"$format": "csv", "$filter": odata_filter},
                timeout=_REQUEST_TIMEOUT_S,
            )
            r.raise_for_status()
            resp = r
            log.info("[overseas.who] WHO FluID: fetched (filter variant %d, %d bytes)",
                     attempt + 1, len(r.content))
            break
        except Exception as e:
            log.warning(
                "[overseas.who] WHO FluID attempt %d failed: %s",
                attempt + 1, str(e)[:120],
            )
            time.sleep(3)

    if resp is None:
        raise RuntimeError("WHO FluID: all fetch attempts failed")

    text = resp.content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    rows: list[dict] = []
    for r in reader:
        iso3 = (r.get("COUNTRY_CODE") or "").strip().upper()
        if iso3 not in _FLUID_TARGET_ISO3:
            continue  # client-side filter (catches fallback attempt)

        country = _FLUID_TARGET_ISO3[iso3]

        year = _safe_int(r.get("ISO_YEAR"))
        week = _safe_int(r.get("ISO_WEEK"))
        if year is None or week is None or year < min_year:
            continue

        cases   = _safe_float(r.get("ILI_CASE"))
        pop_cov = _safe_float(r.get("ILI_POP_COV"))

        ili_rate: Optional[float] = None
        if cases is not None and pop_cov and pop_cov > 0.0:
            ili_rate = round(cases / pop_cov * 100_000.0, 4)  # per 100k sentinel pop

        rows.append({
            "source":            "who_fluid",
            "country":           country,
            "year":              year,
            "week_no":           week,
            "ili_rate":          ili_rate,
            "specimen_positive": _safe_int(cases),    # raw ILI case count
            "specimen_total":    _safe_int(pop_cov),  # sentinel pop coverage
            "influenza_a":       None,
            "influenza_b":       None,
            "positivity_pct":    None,
            "collected_at":      now_iso,
        })

    countries_found = {r["country"] for r in rows}
    log.info(
        "[overseas.who] WHO FluID: %d rows (%d countries: %s)",
        len(rows), len(countries_found), sorted(countries_found),
    )
    return rows


__all__ = [
    "WHO_TARGET_COUNTRIES", "_ISO3_TO_ISO2", "_DEFAULT_YEARS_BACK",
    "_FLUID_TARGET_ISO3",
    "_fetch_who_flunet", "_fetch_who_fluid_eu",
]
