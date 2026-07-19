"""Group I — Overseas ILI incremental update (weekly).

Sprint β Item 4 full migration (2026-05-26, Codex analysis):
This file was 2657 lines with 9 sources + 12 shared helpers inline. Bodies are
now split into per-source deep modules under `simulation/collectors/overseas/`
(D-4 compliant). This file is a thin facade that:

1. Re-exports all 34 names so `from simulation.collectors.group_i_overseas
   import X` keeps working for legacy callers.
2. Provides `run(...)` — the orchestrator's dynamic-dispatch entry point
   (`simulation.collectors.orchestrator.py:280-283` calls `module.run`).
3. Provides `collect_all_new_overseas(...)` — driver for the 4 newer
   regional/weather collectors.
4. Provides the `__main__` CLI (kept here so `python -m
   simulation.collectors.group_i_overseas` continues to work).

Sources (see per-module files for implementation + docstrings):
  1. WHO FluNet            — overseas/who._fetch_who_flunet
  2. CDC ILINet/FluSurv-NET — overseas/cdc._fetch_cdc_ilinet
  3. WHO FluID (EU-10)     — overseas/who._fetch_who_fluid_eu
  4. Sentiweb France       — overseas/sentiweb._fetch_sentiweb_fr
  5. ECDC ERVISS (EU/EEA)  — overseas/ecdc._fetch_ecdc_erviss_github
  6. Italy InfluNet        — overseas/influnet._fetch_influnet_it
  7. Delphi national US    — overseas/cdc._fetch_delphi_national_us
  8. JP jihs aggregate     — overseas/jihs._aggregate_jp_national_from_regional
  9. JIHS hist 2012-2022   — overseas/jihs._fetch_jihs_national_historical

  10. Sentiweb regional    — overseas/sentiweb.collect_sentiweb_france
  11. Open-Meteo weather   — overseas/openmeteo.collect_openmeteo_regional
  12. AU NNDSS             — overseas/nndss.collect_au_nndss
  13. Bright Sky weather   — overseas/brightsky.collect_brightsky_germany

CLI:
  python -m simulation.collectors.group_i_overseas
  python -m simulation.collectors.group_i_overseas --years-back 3
  python -m simulation.collectors.group_i_overseas --run-all-new
"""
from __future__ import annotations

import json
import logging
from datetime import datetime  # noqa: F401 (CLI usage)
from pathlib import Path
from typing import Optional

# ── Shared infrastructure (re-export) ──────────────────────────────────────
from simulation.collectors.overseas._common import (
    _DEFAULT_DB,
    _REQUEST_TIMEOUT_S,
    _safe_connect_import,
    _resolve_db,
    _connect,
    _safe_float,
    _safe_int,
    _retry_get,
    _ensure_overseas_ili_table,
    _ensure_overseas_ili_regional_table,
    _ensure_overseas_weather_regional_table,
    _ensure_overseas_flu_state_table,
    _upsert_rows,
    _upsert_regional_ili_rows,
    _upsert_weather_rows,
    _upsert_flu_state_rows,
)

# ── Per-source fetchers + collectors (re-export) ───────────────────────────
from simulation.collectors.overseas.who import (
    WHO_TARGET_COUNTRIES,
    _ISO3_TO_ISO2,
    _DEFAULT_YEARS_BACK,
    _FLUID_TARGET_ISO3,
    _fetch_who_flunet,
    _fetch_who_fluid_eu,
)
from simulation.collectors.overseas.cdc import (
    _fetch_cdc_ilinet,
    _fetch_delphi_national_us,
)
from simulation.collectors.overseas.jihs import (
    _JP_MIN_PREF_REPORTING,
    _aggregate_jp_national_from_regional,
    _parse_jihs_national_total,
    _fetch_jihs_national_historical,
)
from simulation.collectors.overseas.ecdc import (
    _ECDC_COUNTRY_MAP,
    _fetch_ecdc_erviss_github,
)
from simulation.collectors.overseas.influnet import _fetch_influnet_it
from simulation.collectors.overseas.sentiweb import (
    _fetch_sentiweb_fr,
    _parse_sentiweb_json,
    collect_sentiweb_france,
)
from simulation.collectors.overseas.openmeteo import (
    _OPENMETEO_LOCATIONS,
    _OPENMETEO_DAILY_VARS,
    _fetch_openmeteo_one_year,
    collect_openmeteo_regional,
)
from simulation.collectors.overseas.nndss import (
    _AU_STATES,
    _parse_nndss_excel,
    collect_au_nndss,
)
from simulation.collectors.overseas.brightsky import (
    _DE_BUNDESLAND_CITIES,
    _BRIGHTSKY_WINDOW_DAYS,
    _fetch_brightsky_window,
    collect_brightsky_germany,
)

# Backward-compat URL aliases (some callers still grep these)
from simulation.collectors._endpoints import (
    _WHO_BASE_URL,
    _WHO_FLUID_URL,
    _CDC_URL,
    _SENTIWEB_FR_URL,
    _ECDC_ERVISS_URL,
    _INFLUNET_IT_URL,
    _DELPHI_FLUVIEW_URL,
    _JIHS_HIST_BASE,
    _SENTIWEB_REGIONAL_ENDPOINTS,
    _SENTIWEB_NATIONAL_ENDPOINT,
    _OPENMETEO_ARCHIVE_URL,
    _NNDSS_URLS,
    _AIHW_CSV_URL,
    _BRIGHTSKY_URL,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — orchestrator dynamic-dispatch contract
# ─────────────────────────────────────────────────────────────────────────────

def run(
    backfill_days: Optional[int] = None,
    db_path: Optional[str | Path] = None,
    years_back: int = _DEFAULT_YEARS_BACK,
    skip_who: bool = False,
    skip_cdc: bool = False,
    skip_fluid: bool = False,
    skip_sentiweb: bool = False,
    skip_ecdc: bool = False,
    skip_influnet: bool = False,
    skip_delphi_us: bool = False,
    skip_jp_aggregate: bool = False,
    skip_jp_hist: bool = False,
) -> dict:
    """Run overseas ILI incremental update (9 sub-collectors).

    Called by `simulation.collectors.orchestrator` for Group I dispatch.
    `backfill_days` is accepted for orchestrator-contract uniformity but ignored
    — WHO FluNet/FluID data arrive with 1-2 week lag, so we always fetch the
    last `years_back` calendar years.

    Sources (see module docstring for the 9-source list).

    Args:
        backfill_days:     Passed by orchestrator; unused.
        db_path:           Path to epi_real_seoul.db.
        years_back:        Calendar years of data to fetch (sources 1–4, 6).
        skip_who/cdc/fluid/sentiweb/ecdc/influnet/delphi_us/jp_aggregate/jp_hist:
                          Per-source skip flags (default False).

    Returns:
        dict with keys: inserted, skipped, errors, sources_attempted.

    Performance: 5-7 HTTP requests + 1 DB read (~20 s total), O(15000) DB rows.
    Side effects: writes overseas_ili in epi_real_seoul.db.
    """
    resolved = _resolve_db(db_path)
    result: dict = {
        "inserted": 0,
        "skipped":  0,
        "errors":   [],
        "sources_attempted": [],
    }

    if not resolved.exists():
        msg = f"DB not found: {resolved}"
        log.error("[group_i] %s", msg)
        result["errors"].append(msg)
        return result

    con = _connect(resolved)
    try:
        _ensure_overseas_ili_table(con)

        # ── WHO FluNet ─────────────────────────────────────────────────────
        if not skip_who:
            result["sources_attempted"].append("who_flunet")
            try:
                who_rows = _fetch_who_flunet(years_back=years_back)
                ins, skp = _upsert_rows(con, who_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] WHO FluNet: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] WHO FluNet error: %s", e)
                result["errors"].append(f"who_flunet: {e}")

        # ── CDC ILINet ─────────────────────────────────────────────────────
        if not skip_cdc:
            result["sources_attempted"].append("cdc_ilinet")
            try:
                cdc_rows = _fetch_cdc_ilinet()
                ins, skp = _upsert_rows(con, cdc_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] CDC ILINet: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] CDC ILINet error: %s", e)
                result["errors"].append(f"cdc_ilinet: {e}")

        # ── WHO FluID (EU-10 ILI rate) ─────────────────────────────────────
        if not skip_fluid:
            result["sources_attempted"].append("who_fluid")
            try:
                fluid_rows = _fetch_who_fluid_eu(years_back=years_back)
                ins, skp = _upsert_rows(con, fluid_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] WHO FluID: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] WHO FluID error: %s", e)
                result["errors"].append(f"who_fluid: {e}")

        # ── Sentiweb France ────────────────────────────────────────────────
        if not skip_sentiweb:
            result["sources_attempted"].append("sentiweb_fr")
            try:
                sw_rows = _fetch_sentiweb_fr(years_back=years_back)
                ins, skp = _upsert_rows(con, sw_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] Sentiweb FR: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] Sentiweb FR error: %s", e)
                result["errors"].append(f"sentiweb_fr: {e}")

        # ── ECDC ERVISS GitHub (EU/EEA ILI, 2021-present) ─────────────────
        if not skip_ecdc:
            result["sources_attempted"].append("ecdc_erviss")
            try:
                ecdc_rows = _fetch_ecdc_erviss_github()
                ins, skp = _upsert_rows(con, ecdc_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] ECDC ERVISS: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] ECDC ERVISS error: %s", e)
                result["errors"].append(f"ecdc_erviss: {e}")

        # ── fbranda/influnet Italy (ILI 2003-present) ──────────────────────
        if not skip_influnet:
            result["sources_attempted"].append("influnet_it")
            try:
                it_rows = _fetch_influnet_it(years_back=years_back)
                ins, skp = _upsert_rows(con, it_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] influnet IT: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] influnet IT error: %s", e)
                result["errors"].append(f"influnet_it: {e}")

        # ── Delphi FluView US national wILI% (1997-present) ───────────────
        if not skip_delphi_us:
            result["sources_attempted"].append("delphi_national")
            try:
                delphi_rows = _fetch_delphi_national_us()
                ins, skp = _upsert_rows(con, delphi_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] Delphi national US: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] Delphi national US error: %s", e)
                result["errors"].append(f"delphi_national: {e}")

        # ── JP jihs prefecture→national aggregate (DB-internal, no network) ─
        if not skip_jp_aggregate:
            result["sources_attempted"].append("japan_jihs_aggregate")
            try:
                jp_rows = _aggregate_jp_national_from_regional(con)
                ins, skp = _upsert_rows(con, jp_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] JP jihs aggregate: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] JP jihs aggregate error: %s", e)
                result["errors"].append(f"japan_jihs_aggregate: {e}")

        # ── JIHS/NIID historical JP (2010-2022 backfill) ───────────────────
        if not skip_jp_hist:
            result["sources_attempted"].append("japan_jihs_hist")
            try:
                hist_rows = _fetch_jihs_national_historical(start_year=2012, end_year=2022)
                ins, skp = _upsert_rows(con, hist_rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_i] JIHS hist JP: inserted=%d skipped=%d", ins, skp)
            except Exception as e:
                log.error("[group_i] JIHS hist JP error: %s", e)
                result["errors"].append(f"japan_jihs_hist: {e}")

    finally:
        con.close()

    total = result["inserted"] + result["skipped"]
    log.info(
        "[group_i] Done: %d total rows processed (%d inserted / %d skipped / %d errors)",
        total, result["inserted"], result["skipped"], len(result["errors"]),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — collect all 4 new collectors (regional + weather + AU)
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_new_overseas(
    db_path: str = "simulation/data/db/epi_real_seoul.db",
    start_date: str = "2010-01-01",
    end_date: str | None = None,
) -> dict:
    """Run all 4 new overseas collectors in sequence.

    Order: Sentiweb FR → Open-Meteo weather → AU NNDSS → Bright Sky DE.
    Each collector's result is merged into the combined summary.

    Args:
        db_path:    Path to epi_real_seoul.db.
        start_date: Earliest date (YYYY-MM-DD); passed to each sub-collector.
        end_date:   Latest date (YYYY-MM-DD); passed to each sub-collector.

    Returns:
        dict: rows_inserted (int), errors (list[str]), by_collector (dict).

    Performance: sum of individual collectors (roughly 30-45 min for full history).
    Side effects: writes overseas_ili_regional, overseas_weather_regional,
                  overseas_flu_state.
    Caller responsibility: DB file must exist; rate-limit sleeps are internal.
    """
    combined: dict = {"rows_inserted": 0, "errors": [], "by_collector": {}}

    collectors = [
        ("sentiweb_france",      collect_sentiweb_france),
        ("openmeteo_regional",   collect_openmeteo_regional),
        ("au_nndss",             collect_au_nndss),
        ("brightsky_germany",    collect_brightsky_germany),
    ]

    for name, fn in collectors:
        log.info("[group_i] collect_all_new_overseas: starting %s", name)
        try:
            r = fn(db_path=db_path, start_date=start_date, end_date=end_date)
            combined["rows_inserted"] += r.get("rows_inserted", 0)
            combined["errors"].extend(r.get("errors", []))
            combined["by_collector"][name] = r
            log.info(
                "[group_i] %s done: inserted=%d errors=%d",
                name, r.get("rows_inserted", 0), len(r.get("errors", [])),
            )
        except Exception as e:
            msg = f"{name}: unexpected error: {e}"
            log.error("[group_i] %s", msg)
            combined["errors"].append(msg)
            combined["by_collector"][name] = {"rows_inserted": 0, "errors": [msg]}

    log.info(
        "[group_i] collect_all_new_overseas complete: total=%d errors=%d",
        combined["rows_inserted"], len(combined["errors"]),
    )
    return combined


__all__ = [
    # config
    "_DEFAULT_DB", "_REQUEST_TIMEOUT_S",
    # who
    "WHO_TARGET_COUNTRIES", "_ISO3_TO_ISO2", "_DEFAULT_YEARS_BACK",
    "_FLUID_TARGET_ISO3",
    "_fetch_who_flunet", "_fetch_who_fluid_eu",
    # cdc/delphi
    "_fetch_cdc_ilinet", "_fetch_delphi_national_us",
    # jp
    "_JP_MIN_PREF_REPORTING",
    "_aggregate_jp_national_from_regional",
    "_parse_jihs_national_total", "_fetch_jihs_national_historical",
    # ecdc
    "_ECDC_COUNTRY_MAP", "_fetch_ecdc_erviss_github",
    # it
    "_fetch_influnet_it",
    # sentiweb
    "_fetch_sentiweb_fr", "_parse_sentiweb_json", "collect_sentiweb_france",
    # openmeteo
    "_OPENMETEO_LOCATIONS", "_OPENMETEO_DAILY_VARS",
    "_fetch_openmeteo_one_year", "collect_openmeteo_regional",
    # au nndss
    "_AU_STATES", "_parse_nndss_excel", "collect_au_nndss",
    # brightsky
    "_DE_BUNDESLAND_CITIES", "_BRIGHTSKY_WINDOW_DAYS",
    "_fetch_brightsky_window", "collect_brightsky_germany",
    # common helpers
    "_safe_connect_import", "_resolve_db", "_connect",
    "_safe_float", "_safe_int", "_retry_get",
    "_ensure_overseas_ili_table",
    "_ensure_overseas_ili_regional_table",
    "_ensure_overseas_weather_regional_table",
    "_ensure_overseas_flu_state_table",
    "_upsert_rows", "_upsert_regional_ili_rows",
    "_upsert_weather_rows", "_upsert_flu_state_rows",
    # URLs (backward-compat aliases)
    "_WHO_BASE_URL", "_WHO_FLUID_URL", "_CDC_URL", "_SENTIWEB_FR_URL",
    "_ECDC_ERVISS_URL", "_INFLUNET_IT_URL", "_DELPHI_FLUVIEW_URL",
    "_JIHS_HIST_BASE", "_SENTIWEB_REGIONAL_ENDPOINTS", "_SENTIWEB_NATIONAL_ENDPOINT",
    "_OPENMETEO_ARCHIVE_URL", "_NNDSS_URLS", "_AIHW_CSV_URL", "_BRIGHTSKY_URL",
    # orchestrator entry
    "run", "collect_all_new_overseas",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Overseas ILI incremental update")
    parser.add_argument("--years-back",     type=int, default=_DEFAULT_YEARS_BACK,
                        help="Calendar years of data to fetch (all sources)")
    parser.add_argument("--skip-who",          action="store_true", help="Skip WHO FluNet")
    parser.add_argument("--skip-cdc",          action="store_true", help="Skip CDC ILINet")
    parser.add_argument("--skip-fluid",        action="store_true", help="Skip WHO FluID EU")
    parser.add_argument("--skip-sentiweb",     action="store_true", help="Skip Sentiweb France")
    parser.add_argument("--skip-ecdc",         action="store_true", help="Skip ECDC ERVISS GitHub")
    parser.add_argument("--skip-influnet",     action="store_true", help="Skip influnet Italy")
    parser.add_argument("--skip-delphi-us",    action="store_true", help="Skip Delphi national US wILI%%")
    parser.add_argument("--skip-jp-aggregate", action="store_true", help="Skip JP jihs prefecture→national aggregate")
    parser.add_argument("--skip-jp-hist",       action="store_true", help="Skip JIHS/NIID historical JP backfill 2010-2022")
    parser.add_argument("--db",                default=None,         help="DB path override")
    # ── New collector flags ──────────────────────────────────────────────────
    parser.add_argument("--run-sentiweb-regional", action="store_true",
                        help="Run Sentiweb France regional ILI collector")
    parser.add_argument("--run-openmeteo",          action="store_true",
                        help="Run Open-Meteo ERA5 regional weather collector")
    parser.add_argument("--run-au-nndss",           action="store_true",
                        help="Run AU NNDSS influenza state counts collector")
    parser.add_argument("--run-brightsky",          action="store_true",
                        help="Run Bright Sky Germany weather collector")
    parser.add_argument("--run-all-new",            action="store_true",
                        help="Run all 4 new overseas collectors (collect_all_new_overseas)")
    parser.add_argument("--start-date", default="2010-01-01",
                        help="Start date for new collectors (YYYY-MM-DD)")
    parser.add_argument("--end-date",   default=None,
                        help="End date for new collectors (YYYY-MM-DD, default today)")
    args = parser.parse_args()

    # ── Dispatch new collectors if requested ────────────────────────────────
    new_collector_requested = (
        args.run_sentiweb_regional or args.run_openmeteo or
        args.run_au_nndss or args.run_brightsky or args.run_all_new
    )

    if new_collector_requested:
        if args.run_all_new:
            result = collect_all_new_overseas(
                db_path=args.db or "simulation/data/db/epi_real_seoul.db",
                start_date=args.start_date,
                end_date=args.end_date,
            )
        else:
            combined: dict = {"rows_inserted": 0, "errors": [], "by_collector": {}}
            db = args.db or "simulation/data/db/epi_real_seoul.db"
            if args.run_sentiweb_regional:
                r = collect_sentiweb_france(db, args.start_date, args.end_date)
                combined["rows_inserted"] += r.get("rows_inserted", 0)
                combined["errors"].extend(r.get("errors", []))
                combined["by_collector"]["sentiweb_france"] = r
            if args.run_openmeteo:
                r = collect_openmeteo_regional(db, args.start_date, args.end_date)
                combined["rows_inserted"] += r.get("rows_inserted", 0)
                combined["errors"].extend(r.get("errors", []))
                combined["by_collector"]["openmeteo_regional"] = r
            if args.run_au_nndss:
                r = collect_au_nndss(db, args.start_date, args.end_date)
                combined["rows_inserted"] += r.get("rows_inserted", 0)
                combined["errors"].extend(r.get("errors", []))
                combined["by_collector"]["au_nndss"] = r
            if args.run_brightsky:
                r = collect_brightsky_germany(db, args.start_date, args.end_date)
                combined["rows_inserted"] += r.get("rows_inserted", 0)
                combined["errors"].extend(r.get("errors", []))
                combined["by_collector"]["brightsky_germany"] = r
            result = combined
        print(json.dumps(result, indent=2))
        sys.exit(1 if result["errors"] else 0)

    # ── Default: run original overseas ILI update ───────────────────────────
    result = run(
        db_path=args.db,
        years_back=args.years_back,
        skip_who=args.skip_who,
        skip_cdc=args.skip_cdc,
        skip_fluid=args.skip_fluid,
        skip_sentiweb=args.skip_sentiweb,
        skip_ecdc=args.skip_ecdc,
        skip_influnet=args.skip_influnet,
        skip_delphi_us=args.skip_delphi_us,
        skip_jp_aggregate=args.skip_jp_aggregate,
        skip_jp_hist=args.skip_jp_hist,
    )
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["errors"] else 0)
