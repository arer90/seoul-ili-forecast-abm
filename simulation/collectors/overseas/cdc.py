"""US CDC FluSurv-NET + Delphi FluView — US national ILI / hospitalization.

Sprint β Item 4 full migration (Codex analysis 2026-05-26): real bodies
moved here from group_i_overseas.py. The legacy module re-exports for
back-compat.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from simulation.collectors._endpoints import (
    _CDC_URL,            # CDC ILINet Socrata
    _DELPHI_FLUVIEW_URL,  # Delphi (CMU) FluView
)
from simulation.collectors.overseas._common import (
    _REQUEST_TIMEOUT_S,
    _safe_float,
    _safe_int,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector — CDC ILINet (US FluSurv-NET hospitalization)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_cdc_ilinet(weeks_back: int = 104, start_year: int = 2010) -> list[dict]:
    """Fetch US national FluSurv-NET hospitalization rate from CDC Socrata (kvib-3txy).

    NOTE: Dataset kvib-3txy was repurposed — it now contains FluSurv-NET weekly
    hospitalization rates (per 100,000 population) rather than ILI% from outpatient
    visits.  US national ILI% is covered by delphi_national (1997-present).
    This function fetches Overall/Age-Adjusted FluSurv-NET rows.

    Columns used: mmwr_year, mmwr_week, weekly_rate (hospitalizations/100k).
    Filter: site='Overall' AND age_group='Overall' AND sex='Overall'
            AND surveillance_network='FluSurv-NET' AND rate_type='Age-Adjusted'
    Source tag stored: 'cdc_flusurvnet' (distinct from legacy 'cdc_ilinet' rows).

    Args:
        weeks_back: Approximate number of past weeks to request (backward compat).
                    Ignored when start_year resolves to an earlier year.
        start_year: Earliest MMWR year to fetch (default 2010 — FluSurv-NET start).

    Returns:
        List of row dicts with keys matching overseas_ili schema.

    Raises:
        RuntimeError: if HTTP request fails.

    Performance: 1 HTTP request, ~50 KB JSON.
    Side effects: None.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    current_year = datetime.now().year
    # Fetch from the earlier of start_year or weeks_back-derived year
    min_year = min(start_year, current_year - max(1, weeks_back // 52))
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # kvib-3txy schema (as of 2026): mmwr_year/mmwr_week/weekly_rate/surveillance_network
    # Note: mmwr_year is stored as TEXT in this Socrata dataset → string comparison.
    params = {
        "$where": (
            f"mmwr_year >= '{min_year}' "
            "and site = 'Overall' "
            "and age_group = 'Overall' "
            "and sex = 'Overall' "
            "and surveillance_network = 'FluSurv-NET' "
            "and rate_type = 'Age-Adjusted'"
        ),
        "$limit":  "5000",
        "$select": "mmwr_year,mmwr_week,weekly_rate",
        "$order":  "mmwr_year DESC, mmwr_week DESC",
    }

    try:
        resp = requests.get(
            _CDC_URL,
            params=params,
            timeout=_REQUEST_TIMEOUT_S,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"CDC FluSurv-NET fetch failed: {e}") from e

    rows: list[dict] = []
    for r in data:
        year = _safe_int(r.get("mmwr_year"))
        week = _safe_int(r.get("mmwr_week"))
        rate = _safe_float(r.get("weekly_rate"))   # hospitalizations per 100k
        if year is None or week is None:
            continue
        rows.append({
            "source":            "cdc_flusurvnet",  # distinct from legacy cdc_ilinet rows
            "country":           "US",
            "year":              year,
            "week_no":           week,
            "ili_rate":          rate,   # hosp/100k (different scale from ILI%; noted in source)
            "specimen_positive": None,
            "specimen_total":    None,
            "influenza_a":       None,
            "influenza_b":       None,
            "positivity_pct":    None,
            "collected_at":      now_iso,
        })

    log.info("[overseas.cdc] CDC FluSurv-NET: %d rows fetched (year >= %d)", len(rows), min_year)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector — Delphi FluView (US national wILI%)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_delphi_national_us() -> list[dict]:
    """Fetch US national weighted ILI% from Delphi COVIDcast FluView (1997-present).

    Delphi (Carnegie Mellon) exposes CDC ILINet national data via a clean JSON
    API — no key required.  Region 'nat' returns the US national aggregate.
    wili (weighted ILI%) accounts for state-level provider density variation
    and is the official CDC benchmark metric; stored in ``ili_rate``.

    Endpoint: https://api.delphi.cmu.edu/epidata/fluview/
    Params:   regions=nat  epiweeks=199740-<current>
    Response: {"result": 1, "epidata": [{"epiweek": 201001, "wili": 1.91, ...}]}

    Returns:
        List of row dicts for overseas_ili:
          source='delphi_national', country='US',
          year, week_no, ili_rate=wili (%),
          specimen_positive=num_ili (patient count with ILI),
          specimen_total=num_patients (total patients seen).

    Raises:
        RuntimeError: on HTTP or JSON parse failure.

    Performance: 1 HTTP request, ~1,400 rows (~3 s).
    Side effects: None.
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur_year = datetime.now().year
    # Start from 1997-W40 (first week with Delphi national data)
    epiweek_range = f"199740-{cur_year * 100 + 53}"

    log.info("[overseas.cdc] Delphi FluView US national wILI%% epiweeks=%s …", epiweek_range)
    resp = requests.get(
        _DELPHI_FLUVIEW_URL,
        params={"regions": "nat", "epiweeks": epiweek_range},
        timeout=_REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("result") != 1:
        raise RuntimeError(
            f"Delphi national: result={payload.get('result')} "
            f"message={payload.get('message')}"
        )

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
            "source":            "delphi_national",
            "country":           "US",
            "year":              year,
            "week_no":           week,
            "ili_rate":          ili_rate,          # weighted ILI%
            "specimen_positive": _si(rec.get("num_ili")),      # ILI patient count
            "specimen_total":    _si(rec.get("num_patients")), # total patients seen
            "influenza_a":       None,
            "influenza_b":       None,
            "positivity_pct":    None,
            "collected_at":      now_iso,
        })

    log.info("[overseas.cdc] Delphi national US: %d rows (%s)", len(rows), epiweek_range)
    return rows


__all__ = ["_fetch_cdc_ilinet", "_fetch_delphi_national_us"]
