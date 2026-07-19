"""Japan JIHS/NIID — national aggregate (from Group O prefecture data) +
historical CSVs (2012-2022 teiten backfill).

Sprint β Item 4 full migration (Codex analysis 2026-05-26): real bodies
moved here from group_i_overseas.py. The legacy module re-exports for
back-compat.

⚠ Cross-module data dependency: `_aggregate_jp_national_from_regional` reads
`overseas_ili_regional` populated by Group O (`group_o_regional_ili.py`).
Order in pipeline: Group O must run BEFORE Group I JP aggregate.
"""
from __future__ import annotations

import csv
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from sqlite3 import Connection as _Conn
from typing import Optional

from simulation.collectors._endpoints import _JIHS_HIST_BASE
from simulation.collectors.overseas._common import _safe_int  # noqa: F401 (future-use)

log = logging.getLogger(__name__)


# Minimum number of Japanese prefectures that must report in a given week
# for the national aggregate to be considered reliable.
_JP_MIN_PREF_REPORTING = 40


# ─────────────────────────────────────────────────────────────────────────────
# JP national aggregate (DB-internal, no network)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_jp_national_from_regional(con: _Conn) -> list[dict]:
    """Derive Japan national ILI from overseas_ili_regional jihs_prefecture aggregate.

    JIHS (Japan Institute for Health Security) publishes prefecture-level weekly
    sentinel data (47 prefectures, collected by group_o).  Averaging across
    prefectures with ≥ ``_JP_MIN_PREF_REPORTING`` reporters gives a robust
    national weekly ILI rate proxy on the same scale as the per-sentinel values.

    Source dependency: requires overseas_ili_regional to be populated by
    Group O (``group_o_regional_ili.py``).  If the table is empty, returns [].

    Duplication guard: skips week/year tuples already present in overseas_ili
    for source='japan_jihs', country='JP'.

    Args:
        con: Open SQLite connection (read-only queries only).

    Returns:
        List of row dicts for overseas_ili:
          source='japan_jihs', country='JP',
          year, week_no,
          ili_rate = mean(ili_rate across ≥40 prefectures),
          specimen_total = number of prefectures included in mean.

    Performance: O(n_jihs_rows) ≈ O(6000) rows — single SQL GROUP BY query.
    Side effects: None (read-only; caller upserts).
    Caller responsibility: overseas_ili_regional must exist.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Existing JP japan_jihs weeks — avoid re-inserting
    existing_q = con.execute(
        "SELECT year, week_no FROM overseas_ili "
        "WHERE country='JP' AND source='japan_jihs'"
    )
    existing: set[tuple[int, int]] = {(r[0], r[1]) for r in existing_q}

    # Aggregate jihs_prefecture → national weekly mean (≥40 prefectures only)
    try:
        agg_rows = con.execute("""
            SELECT year, week_no,
                   AVG(ili_rate)  AS nat_ili,
                   COUNT(*)       AS n_pref
            FROM   overseas_ili_regional
            WHERE  source   = 'jihs_prefecture'
              AND  country  = 'JPN'
              AND  ili_rate IS NOT NULL
            GROUP  BY year, week_no
            HAVING COUNT(*) >= ?
            ORDER  BY year, week_no
        """, (_JP_MIN_PREF_REPORTING,)).fetchall()
    except Exception as e:
        log.warning("[overseas.jihs] JP jihs aggregate query failed: %s", e)
        return []

    rows: list[dict] = []
    for year, week_no, nat_ili, n_pref in agg_rows:
        if (year, week_no) in existing:
            continue
        rows.append({
            "source":            "japan_jihs",
            "country":           "JP",
            "year":              int(year),
            "week_no":           int(week_no),
            "ili_rate":          float(nat_ili),
            "specimen_positive": None,
            "specimen_total":    int(n_pref),  # store n_prefectures for transparency
            "influenza_a":       None,
            "influenza_b":       None,
            "positivity_pct":    None,
            "collected_at":      now_iso,
        })

    log.info("[overseas.jihs] JP jihs national aggregate: %d new rows "
             "(total available=%d, already in DB=%d)",
             len(rows), len(agg_rows), len(existing))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collector — JIHS historical teiten CSVs (2012-2022)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_jihs_national_total(text: str, year: int, week: int) -> Optional[dict]:
    """Parse JIHS teiten CSV and extract the national 'Total No.' aggregate row.

    CSV layout (English-language file):
      Row 3: "Prefecture,Influenza(excld. avian influenza…),,…"  (disease headers)
      Row 4: ",Current week,per sentinel,…"
      Row 5: "Total No.,<abs_cases>,<per_sentinel>,…"            (NATIONAL AGGREGATE)
      Rows 6-52: 47 prefectures

    Args:
        text:  Raw decoded CSV text.
        year:  ISO year (from URL path).
        week:  ISO week (from URL path).

    Returns:
        Row dict for overseas_ili (source='japan_jihs_hist'), or None on failure.

    Performance: O(55) rows.
    Side effects: None.
    Caller responsibility: text must be non-empty.
    """
    try:
        reader   = csv.reader(io.StringIO(text))
        rows_raw = list(reader)
    except Exception:
        return None

    if len(rows_raw) < 6:
        return None

    # Find Influenza column in disease header row (row index 3)
    disease_row = rows_raw[3] if len(rows_raw) > 3 else []
    flu_col: Optional[int] = None
    for i, cell in enumerate(disease_row):
        if "Influenza" in cell and "avian" in cell.lower():
            flu_col = i
            break
    if flu_col is None:
        # Fallback: first cell containing "Influenza" at all
        for i, cell in enumerate(disease_row):
            if "Influenza" in cell:
                flu_col = i
                break
    if flu_col is None:
        return None

    per_sentinel_col = flu_col + 1  # "per sentinel" immediately follows

    # Find "Total No." row — usually index 5, search 5-8 for robustness
    total_row = None
    for r in rows_raw[5:9]:
        if r and r[0].strip() in ("Total No.", "Total", "合計"):
            total_row = r
            break
    if total_row is None:
        return None

    def _parse_cell(row: list, col: int) -> Optional[float]:
        try:
            v = row[col].strip() if len(row) > col else "-"
            return float(v) if v not in ("-", "", "—") else None
        except (ValueError, IndexError):
            return None

    ili_rate = _parse_cell(total_row, per_sentinel_col)
    n_cases  = _parse_cell(total_row, flu_col)

    if ili_rate is None and n_cases is None:
        return None

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "source":            "japan_jihs_hist",
        "country":           "JP",
        "year":              year,
        "week_no":           week,
        "ili_rate":          ili_rate,               # cases per sentinel (national)
        "specimen_positive": int(n_cases) if n_cases is not None else None,
        "specimen_total":    None,
        "influenza_a":       None,
        "influenza_b":       None,
        "positivity_pct":    None,
        "collected_at":      now_iso,
    }


def _fetch_jihs_national_historical(
    start_year: int = 2012,
    end_year: int = 2022,
) -> list[dict]:
    """Fetch JP national ILI from JIHS/NIID historical teiten CSVs (2012–2022).

    Fills the historical gap in overseas_ili before the current JIHS portal
    (id-info.jihs.go.jp) started.  Fetches one CSV per flu-season week and
    extracts the 'Total No.' national aggregate row.

    URL patterns (confirmed 2026-05-25 via HTTP 200 verification):
      yr ≤ 2014: {_JIHS_HIST_BASE}/idwr-e{yr}/{yr%100:02d}{wk:02d}/teiten{wk:02d}.csv
                 subdirectory uses 2-DIGIT year (e.g. 1247 for 2012W47)
      yr ≥ 2015: {_JIHS_HIST_BASE}/idwr-e{yr}/{yr}{wk:02d}/teiten{wk:02d}.csv
                 subdirectory uses 4-DIGIT year (e.g. 201536 for 2015W36)
    2010-2011: no data available (all weeks return 404).
    Off-season weeks (21-35) return 404 and are silently skipped.

    Args:
        start_year: First calendar year to fetch (inclusive). Default 2012.
        end_year:   Last calendar year to fetch (inclusive). Default 2022.

    Returns:
        List of row dicts for overseas_ili:
          source='japan_jihs_hist', country='JP', year, week_no,
          ili_rate=cases per sentinel (national 'Total No.' row),
          specimen_positive=absolute weekly case count.

    Raises:
        Nothing — per-week failures silently skipped; partial results returned.

    Performance: ~38 flu weeks × (end_year-start_year+1) parallel HTTP requests.
                 10 worker threads; 13 years ≈ ~30-90 s total network time.
    Side effects: None (pure fetch).
    Caller responsibility: `requests` must be installed.
    """
    try:
        import requests as _req
    except ImportError as e:
        log.warning("[overseas.jihs] requests not available — skipping JIHS historical: %s", e)
        return []

    session = _req.Session()
    session.headers["User-Agent"] = "MPH-flu-collector/1.0"

    # Build flu-season (year, week) targets (skip summer 21-35)
    targets: list[tuple[int, int]] = [
        (yr, wk)
        for yr in range(start_year, end_year + 1)
        for wk in range(1, 54)
        if not (21 <= wk <= 35)
    ]

    log.info(
        "[overseas.jihs] JIHS historical JP national: fetching %d week-CSVs (%d–%d) …",
        len(targets), start_year, end_year,
    )

    def _fetch_one(yr: int, wk: int) -> Optional[dict]:
        # 2-digit year in subdirectory for 2012-2014; 4-digit for 2015-2022.
        if yr <= 2014:
            subdir = f"{yr % 100:02d}{wk:02d}"
        else:
            subdir = f"{yr}{wk:02d}"
        url = f"{_JIHS_HIST_BASE}/idwr-e{yr}/{subdir}/teiten{wk:02d}.csv"
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return _parse_jihs_national_total(
                r.content.decode("utf-8-sig", errors="replace"), yr, wk
            )
        except Exception:
            return None

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_one, yr, wk): (yr, wk) for yr, wk in targets}
        for fut in as_completed(futures):
            row = fut.result()
            if row is not None:
                rows.append(row)

    log.info(
        "[overseas.jihs] JIHS historical JP: %d rows fetched (%d–%d)",
        len(rows), start_year, end_year,
    )
    return rows


__all__ = [
    "_JP_MIN_PREF_REPORTING",
    "_aggregate_jp_national_from_regional",
    "_parse_jihs_national_total",
    "_fetch_jihs_national_historical",
]
