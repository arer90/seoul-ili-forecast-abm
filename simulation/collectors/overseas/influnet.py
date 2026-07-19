"""Italy InfluNet (fbranda mirror) — IT national ILI incidence 2003-present.

Sprint β Item 4 full migration (Codex analysis 2026-05-26): real body moved
here from group_i_overseas.py. Previously colocated with ECDC in the shallow
shim; now its own deep module per Codex Rank-2 plan (single HTTP fetch with no
shared infra besides _safe_int/_safe_float).
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone

from simulation.collectors._endpoints import _INFLUNET_IT_URL
from simulation.collectors.overseas._common import (
    _REQUEST_TIMEOUT_S,
    _safe_float,
    _safe_int,
)

log = logging.getLogger(__name__)


def _fetch_influnet_it(years_back: int = 3) -> list[dict]:
    """Fetch Italy national ILI incidence from fbranda/influnet GitHub.

    Source: github.com/fbranda/influnet, data-aggregated/epidemiological_data/national_cases.csv.
    Coverage: 2003-W42 to present (weekly sentinel GP network).

    CSV schema (relevant columns):
      year_week  — "YYYY-WW" (e.g. "2003-42")
      incidence  — ILI incidence per 100,000 inhabitants (total population)
      number_cases — raw ILI case count
      population   — sentinel coverage population

    Args:
        years_back: Calendar years back to include (filters year >= current - years_back + 1).
                    Pass a large value (e.g. 50) to fetch the full 2003-present history.

    Returns:
        List of row dicts, source='influnet_it', country='IT'.
        ili_rate = incidence (per 100k).
        specimen_positive = number_cases (raw sentinel count).
        specimen_total = population (sentinel coverage pop, not national pop).

    Raises:
        RuntimeError: if HTTP fetch fails.

    Performance: 1 HTTP request, ~65 KB CSV.
    Side effects: None (pure fetch).
    Caller responsibility: caller upserts rows into DB.
    """
    try:
        from simulation.utils.http import http_get  # SSOT retry-session
    except ImportError as e:
        raise RuntimeError("requests not installed — pip install requests") from e

    current_year = datetime.now().year
    min_year = current_year - (years_back - 1)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = http_get(_INFLUNET_IT_URL, timeout=_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"influnet IT fetch failed: {e}") from e

    log.info("[overseas.influnet] influnet IT: fetched %d bytes", len(resp.content))

    text = resp.content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    rows: list[dict] = []
    for r in reader:
        # year_week: "2003-42"  →  year=2003, week_no=42
        raw = (r.get("year_week") or "").strip().strip('"')
        if not raw or "-" not in raw:
            continue
        try:
            year_s, week_s = raw.split("-", 1)
            year    = int(year_s)
            week_no = int(week_s)
        except ValueError:
            continue

        if year < min_year:
            continue

        incidence = _safe_float(r.get("incidence"))
        if incidence is None:
            continue

        rows.append({
            "source":            "influnet_it",
            "country":           "IT",
            "year":              year,
            "week_no":           week_no,
            "ili_rate":          round(incidence, 4),
            "specimen_positive": _safe_int(r.get("number_cases")),
            "specimen_total":    _safe_int(r.get("population")),
            "influenza_a":       None,
            "influenza_b":       None,
            "positivity_pct":    None,
            "collected_at":      now_iso,
        })

    log.info("[overseas.influnet] influnet IT: %d rows (years ≥ %d)", len(rows), min_year)
    return rows


__all__ = ["_fetch_influnet_it"]
