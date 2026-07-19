"""France Sentiweb — GP sentinel ILI incidence (national CSV + regional JSON).

Sprint β Item 4 full migration (Codex analysis 2026-05-26): real bodies
moved here from group_i_overseas.py. The legacy module re-exports for
back-compat.

This module owns the regional ILI sub-pipeline:
- `_fetch_sentiweb_fr`            — national CSV → overseas_ili (via _upsert_rows)
- `_parse_sentiweb_json`          — shared JSON parser (regional + national variants)
- `collect_sentiweb_france`       — regional fallback chain → overseas_ili_regional
                                    (via _upsert_regional_ili_rows)

The regional table helpers (`_ensure_overseas_ili_regional_table`,
`_upsert_regional_ili_rows`) are in `overseas/_common.py` since they're shared
DDL primitives, but Sentiweb is currently the only producer.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Optional

from simulation.collectors._endpoints import (
    _SENTIWEB_FR_URL,
    _SENTIWEB_REGIONAL_ENDPOINTS,
    _SENTIWEB_NATIONAL_ENDPOINT,
)
from simulation.collectors.overseas._common import (
    _REQUEST_TIMEOUT_S,
    _connect,
    _ensure_overseas_ili_regional_table,
    _resolve_db,
    _retry_get,
    _safe_float,
    _safe_int,
    _upsert_regional_ili_rows,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector — Sentiweb national CSV (used by `run()`)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_sentiweb_fr(years_back: int = 3) -> list[dict]:
    """Fetch France Sentiweb ILI incidence (national, per 100k).

    Higher-resolution French ILI data from the GP sentinel network.  Metric is
    ILI incidence per 100,000 inhabitants (inc100), which differs from WHO
    FluID's % consultation rate.  Stored as-is in ili_rate; downstream models
    treat it as a raw signal regardless of unit.

    File: incidence-PAY-3.csv (France métropolitaine, PAY = pays level 3).
    Weekly since 2000.  Columns: semaine (YYYYWW), inc, taux, inc100, ...

    Args:
        years_back: Calendar years back from current year to fetch.

    Returns:
        List of row dicts, source='sentiweb_fr', country='FR'.
        ili_rate = inc100 (per 100k; note: different unit from WHO FluID %).

    Raises:
        RuntimeError: if HTTP fetch fails.

    Performance: 1 HTTP request, ~100 KB CSV.
    Side effects: None.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    current_year = datetime.now().year
    min_year = current_year - (years_back - 1)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = requests.get(
            _SENTIWEB_FR_URL,
            timeout=_REQUEST_TIMEOUT_S,
            headers={"Accept": "text/csv"},
        )
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Sentiweb FR fetch failed: {e}") from e

    text = resp.content.decode("utf-8-sig")

    # Sentiweb CSV starts with one or more '#' comment lines (HTML-escaped JSON
    # metadata) followed by the actual CSV header and data rows.  Strip them.
    data_lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if not data_lines:
        log.warning("[overseas.sentiweb] Sentiweb: CSV appears empty after stripping comments")
        return []

    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
    fieldnames: list[str] = list(reader.fieldnames or [])

    # Detect week and inc100 columns (accept 'week' or French 'semaine')
    lower_map = {f.lower().strip('"'): f for f in fieldnames}
    week_col  = next(
        (lower_map[k] for k in ("week", "semaine", "sem") if k in lower_map), None
    )
    inc100_col = next(
        (lower_map[k] for k in ("inc100", "taux100", "taux") if k in lower_map), None
    )
    if not week_col:
        log.warning("[overseas.sentiweb] Sentiweb: no week column in %s", fieldnames[:8])
        return []
    if not inc100_col:
        log.warning("[overseas.sentiweb] Sentiweb: no inc100 column in %s", fieldnames[:8])
        return []

    rows: list[dict] = []
    for r in reader:
        raw_week = str(r.get(week_col) or "").strip().strip('"')
        if len(raw_week) < 6:
            continue
        try:
            year    = int(raw_week[:4])
            week_no = int(raw_week[4:])
        except ValueError:
            continue

        if year < min_year:
            continue

        inc100 = _safe_float(r.get(inc100_col))
        if inc100 is None:
            continue

        rows.append({
            "source":            "sentiweb_fr",
            "country":           "FR",
            "year":              year,
            "week_no":           week_no,
            "ili_rate":          inc100,   # per 100k; stored as-is
            "specimen_positive": None,
            "specimen_total":    None,
            "influenza_a":       None,
            "influenza_b":       None,
            "positivity_pct":    None,
            "collected_at":      now_iso,
        })

    log.info("[overseas.sentiweb] Sentiweb FR: %d rows (years ≥ %d)", len(rows), min_year)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Regional pipeline (JSON fallback chain → overseas_ili_regional)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sentiweb_json(payload: list | dict, min_year: int,
                         now_iso: str) -> list[dict]:
    """Parse Sentiweb JSON response (regional or national) into row dicts.

    Handles two known schemas:
      A. List of records with keys: year, week, inc (or inc100), cod_reg (optional).
      B. Dict with 'obs' or 'data' key containing such a list.

    Args:
        payload:  Parsed JSON (list or dict).
        min_year: Earliest year to include.
        now_iso:  ISO timestamp string for collected_at.

    Returns:
        List of regional ILI row dicts (source='sentiweb_fr_regional').

    Performance: O(n_records).
    Side effects: None.
    """
    if isinstance(payload, dict):
        records = (
            payload.get("obs") or payload.get("data") or
            payload.get("records") or payload.get("results") or []
        )
    elif isinstance(payload, list):
        records = payload
    else:
        return []

    rows: list[dict] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        # Year / week
        year = _safe_int(r.get("year") or r.get("an") or r.get("annee"))
        week = _safe_int(
            r.get("week") or r.get("sem") or r.get("semaine") or r.get("wk")
        )
        if year is None or week is None or year < min_year:
            continue
        # ILI rate (per 100k)
        inc = _safe_float(
            r.get("inc100") or r.get("taux100") or r.get("inc") or r.get("taux")
        )
        if inc is None:
            continue
        # Region code (may be absent for national endpoint)
        region = str(
            r.get("cod_reg") or r.get("reg") or r.get("region") or "national"
        ).strip() or "national"

        rows.append({
            "source":      "sentiweb_fr_regional",
            "country":     "FR",
            "region":      region,
            "year":        year,
            "week_no":     week,
            "ili_rate":    round(inc, 4),
            "collected_at": now_iso,
        })
    return rows


def collect_sentiweb_france(
    db_path: str = "simulation/data/db/epi_real_seoul.db",
    start_date: str = "2010-01-01",
    end_date: str | None = None,
) -> dict:
    """Collect Sentiweb France regional ILI incidence into overseas_ili_regional.

    Tries JSON endpoints in order (regional → national JSON → national CSV).
    Falls back gracefully; never crashes.

    Args:
        db_path:    Path to epi_real_seoul.db.
        start_date: Earliest date to include (YYYY-MM-DD); only year is used.
        end_date:   Unused (Sentiweb API returns full history); kept for interface
                    uniformity.

    Returns:
        dict with keys: rows_inserted (int), errors (list[str]),
        source (str, URL actually used).

    Performance: 1 HTTP request, ~100-300 KB JSON/CSV.
    Side effects: writes overseas_ili_regional in epi_real_seoul.db.
    Caller responsibility: DB file must exist.
    """
    import requests  # noqa: F401 — checked here for early ImportError

    result: dict = {"rows_inserted": 0, "errors": [], "source": ""}
    try:
        min_year = int(start_date[:4])
    except (ValueError, TypeError):
        min_year = 2010
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved = _resolve_db(db_path)
    con = _connect(resolved)

    try:
        _ensure_overseas_ili_regional_table(con)
    except Exception as e:
        result["errors"].append(f"table setup: {e}")
        con.close()
        return result

    # ── Try JSON endpoints (regional first, then national) ──────────────────
    json_endpoints = [
        _SENTIWEB_REGIONAL_ENDPOINTS[0],
        _SENTIWEB_REGIONAL_ENDPOINTS[1],
        _SENTIWEB_NATIONAL_ENDPOINT,
    ]

    rows: list[dict] = []
    source_used = ""

    for url in json_endpoints:
        try:
            resp = _retry_get(url, timeout=60)
            payload = resp.json()
            rows = _parse_sentiweb_json(payload, min_year, now_iso)
            if rows:
                source_used = url
                log.info("[overseas.sentiweb] Sentiweb FR JSON: %d rows from %s", len(rows), url[:80])
                break
        except Exception as e:
            log.warning("[overseas.sentiweb] Sentiweb FR JSON endpoint %s failed: %s", url[:80], e)

    # ── Fallback: national CSV (PAY-3) ───────────────────────────────────────
    if not rows:
        csv_url = _SENTIWEB_REGIONAL_ENDPOINTS[2]
        try:
            resp = _retry_get(csv_url, timeout=60)
            text = resp.content.decode("utf-8-sig")
            data_lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
            reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
            lower_map = {f.lower().strip('"'): f for f in (reader.fieldnames or [])}
            week_col  = next((lower_map[k] for k in ("week", "semaine", "sem") if k in lower_map), None)
            inc_col   = next((lower_map[k] for k in ("inc100", "taux100", "taux", "inc") if k in lower_map), None)
            if week_col and inc_col:
                for r in reader:
                    raw = str(r.get(week_col) or "").strip().strip('"')
                    if len(raw) < 6:
                        continue
                    try:
                        year = int(raw[:4]); week = int(raw[4:])
                    except ValueError:
                        continue
                    if year < min_year:
                        continue
                    inc = _safe_float(r.get(inc_col))
                    if inc is None:
                        continue
                    rows.append({
                        "source": "sentiweb_fr_regional", "country": "FR",
                        "region": "national", "year": year, "week_no": week,
                        "ili_rate": round(inc, 4), "collected_at": now_iso,
                    })
                source_used = csv_url
                log.info("[overseas.sentiweb] Sentiweb FR CSV fallback: %d rows", len(rows))
        except Exception as e:
            msg = f"Sentiweb FR CSV fallback failed: {e}"
            log.warning("[overseas.sentiweb] %s", msg)
            result["errors"].append(msg)

    if not rows:
        msg = "Sentiweb FR: no data retrieved from any endpoint"
        log.warning("[overseas.sentiweb] %s", msg)
        result["errors"].append(msg)
        con.close()
        result["source"] = source_used
        return result

    try:
        ins, _ = _upsert_regional_ili_rows(con, rows)
        result["rows_inserted"] = ins
    except Exception as e:
        result["errors"].append(f"DB upsert: {e}")
    finally:
        con.close()

    result["source"] = source_used
    log.info("[overseas.sentiweb] collect_sentiweb_france: %d inserted", result["rows_inserted"])
    return result


__all__ = [
    "_fetch_sentiweb_fr",
    "_parse_sentiweb_json",
    "collect_sentiweb_france",
]
