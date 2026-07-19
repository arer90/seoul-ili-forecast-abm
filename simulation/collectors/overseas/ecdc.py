"""EU ECDC ERVISS GitHub mirror — ILI consultation rate for 28 EU/EEA countries.

Sprint β Item 4 full migration (Codex analysis 2026-05-26): real body moved
here from group_i_overseas.py. The legacy module re-exports for back-compat.

NOTE: `_fetch_influnet_it` (Italy InfluNet) was previously grouped with ECDC
in the shim; it now lives in `overseas/influnet.py` as its own deep module.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone

from simulation.collectors._endpoints import _ECDC_ERVISS_URL
from simulation.collectors.overseas._common import (
    _REQUEST_TIMEOUT_S,
    _safe_float,
)

log = logging.getLogger(__name__)


# Country name (as in ECDC CSV) → ISO2
_ECDC_COUNTRY_MAP: dict[str, str] = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR",
    "Cyprus": "CY", "Czechia": "CZ", "Denmark": "DK", "Estonia": "EE",
    "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR",
    "Hungary": "HU", "Iceland": "IS", "Ireland": "IE", "Italy": "IT",
    "Latvia": "LV", "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT",
    "Netherlands": "NL", "Norway": "NO", "Poland": "PL", "Portugal": "PT",
    "Romania": "RO", "Slovakia": "SK", "Slovenia": "SI", "Spain": "ES",
}


def _fetch_ecdc_erviss_github() -> list[dict]:
    """Fetch ILI consultation rate for 28 EU/EEA countries from ECDC ERVISS GitHub.

    Source: EU-ECDC/Respiratory_viruses_weekly_data, ILIARIRates.csv.
    Coverage: 2021-W25 to present (~5 years); updated weekly.
    Accessible via raw.githubusercontent.com (no DNS issues unlike opendata.ecdc.europa.eu).

    CSV schema:
      survtype    — survey network type (ignored; we take all)
      countryname — country name in English (e.g. "Austria")
      yearweek    — ISO week string "YYYY-Www" (e.g. "2026-W14")
      indicator   — "ILIconsultationrate" or "ARIconsultationrate"
      age         — "total", "0-4", "5-14", "15-64", "65+"
      value       — ILI consultation rate per 100,000

    Only rows where indicator == "ILIconsultationrate" AND age == "total" are
    kept to avoid age-stratified double-counting.

    Returns:
        List of row dicts, source='ecdc_erviss'.
        ili_rate = value (per 100k consultation rate).
        All other specimen/influenza columns are None (ECDC ERVISS is ILI-only).

    Raises:
        RuntimeError: if HTTP fetch fails.

    Performance: 1 HTTP request, ~2.8 MB CSV, ~5 s.
    Side effects: None (pure fetch).
    Caller responsibility: caller upserts rows into DB.
    """
    try:
        from simulation.utils.http import http_get  # SSOT retry-session
    except ImportError as e:
        raise RuntimeError("requests not installed — pip install requests") from e

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = http_get(_ECDC_ERVISS_URL, timeout=_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"ECDC ERVISS GitHub fetch failed: {e}") from e

    log.info("[overseas.ecdc] ECDC ERVISS: fetched %d bytes", len(resp.content))

    text = resp.content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    rows: list[dict] = []
    skipped_unknown = 0
    for r in reader:
        # Only ILI total aggregate rows
        if r.get("indicator", "").strip() != "ILIconsultationrate":
            continue
        if r.get("age", "").strip() != "total":
            continue

        country_name = (r.get("countryname") or "").strip()
        iso2 = _ECDC_COUNTRY_MAP.get(country_name)
        if iso2 is None:
            skipped_unknown += 1
            continue

        # yearweek format: "YYYY-Www"  →  year=YYYY, week_no=WW
        raw_week = (r.get("yearweek") or "").strip()   # e.g. "2026-W14"
        try:
            year_part, w_part = raw_week.split("-W")
            year    = int(year_part)
            week_no = int(w_part)
        except (ValueError, AttributeError):
            continue

        ili_rate = _safe_float(r.get("value"))
        if ili_rate is None:
            continue

        rows.append({
            "source":            "ecdc_erviss",
            "country":           iso2,
            "year":              year,
            "week_no":           week_no,
            "ili_rate":          round(ili_rate, 4),
            "specimen_positive": None,
            "specimen_total":    None,
            "influenza_a":       None,
            "influenza_b":       None,
            "positivity_pct":    None,
            "collected_at":      now_iso,
        })

    countries_found = sorted({r["country"] for r in rows})
    if skipped_unknown:
        log.warning("[overseas.ecdc] ECDC ERVISS: %d rows skipped (unknown country name)", skipped_unknown)
    log.info(
        "[overseas.ecdc] ECDC ERVISS: %d rows, %d countries: %s",
        len(rows), len(countries_found), countries_found,
    )
    return rows


__all__ = ["_ECDC_COUNTRY_MAP", "_fetch_ecdc_erviss_github"]
