"""
simulation.collectors.orchestrator
Unified data collection interface -- fully self-contained in simulation/.
Legacy collectors are stored in simulation/collectors/legacy/.
"""
import sys
import time
import logging
import importlib
import importlib.util
import inspect
from datetime import datetime, timedelta
from pathlib import Path

from simulation.database import init_db, get_table_shapes, DB_PATH
from simulation.database.storage import log_collection

log = logging.getLogger(__name__)

# -- Collector locations (now local to simulation/) --
_LEGACY_DIR = Path(__file__).resolve().parent / "legacy"

GROUP_INFO = {
    "A":  {"module": "group_a_realtime",  "desc": "Real-time population (Seoul API)", "est_sec": 300},
    "B":  {"module": "group_b_weather",   "desc": "Weather historical (KMA ASOS)",    "est_sec": 900},
    "C":  {"module": "group_c_daily",     "desc": "Employment data (daily)",          "est_sec": 600},
    "CM": {"module": "group_c_monthly",   "desc": "Transit hourly (monthly)",         "est_sec": 600},
    "D":  {"module": "group_d_weekly",    "desc": "Population/demographics (KOSIS)",  "est_sec": 300},
    "E":  {"module": "group_e_periodic",  "desc": "Weekly disease counts (KDCA)",     "est_sec": 600},
    "F":  {"module": "group_f_vaccine",   "desc": "Vaccination coverage",             "est_sec": 300},
    "G":  {"module": "group_g_gtrends",   "desc": "Google Trends (KR + US/JP/EU-10)", "est_sec": 900},
    "H":  {"module": "group_h_hira",      "desc": "HIRA health claims",               "est_sec": 300},
    "P":  {"module": "group_p_pubmed",    "desc": "PubMed articles",                  "est_sec": 120},
    "Q":  {"module": "group_q_emergency", "desc": "Emergency dispatch",               "est_sec": 120},
    "R":  {"module": "group_r_school",    "desc": "School closure/info",              "est_sec": 180},
    "S":  {"module": "group_s_sentinel",  "desc": "Sentinel surveillance",            "est_sec": 180},
    # Group I added 2026-05-24: active overseas ILI weekly update.
    # Sources: WHO FluNet (US/JP/GB/DE/FR/NL/SE/KR) + CDC ILINet (US rate) +
    #          WHO FluID EU-10 (ILI consultation %) + Sentiweb France (per 100k).
    # Upserts into overseas_ili (never deletes). Run after Group E for freshest data.
    "I":  {"module": "group_i_overseas",  "desc": "Overseas ILI (FluNet+FluID+CDC+Sentiweb)", "est_sec": 90},
    # Group K added 2026-05-24: KMA ASOS multi-station weather for Seoul GU coverage.
    # Extends Group B (stn 108 only) with 5 additional ASOS stations bracketing Seoul.
    # Requires KMA_API_KEY env var. Creates weather_gu_station_map lookup table.
    "K":  {"module": "group_k_weather_gu","desc": "GU-level weather (multi-station KMA)", "est_sec": 30},
    # Group N added 2026-05-24: weekly hospital ED burden index.
    # Aggregates emergency_room_availability + HIRA claims → ed_weekly_burden table.
    # Optional NEDIS OpenAPI (set NEDIS_API_KEY) for enhanced ED visit counts.
    "N":  {"module": "group_n_hospital",  "desc": "Hospital ED burden (NEDIS/HIRA)", "est_sec": 30},
    # Group O added 2026-05-24: sub-national ILI for overseas countries.
    # US: Delphi FluView ILI% + NSSP ED visits% + NHSN hospitalization + NWSS wastewater Flu-A.
    # DE: RKI Bundesland confirmed influenza incidence (TSV, GitHub).
    # JP: NIID prefecture (DNS failure → graceful empty).
    # Writes overseas_ili_regional (4 US sources + DE + JP stubs).
    # CDC 6svj-q4zv archived 2024-10-16 → removed.
    "O":  {"module": "group_o_regional_ili",    "desc": "Regional ILI+hosp+wastewater (US/DE/JP)", "est_sec": 120},
    # Group J added 2026-05-24: population density by region for ILI normalization.
    # WorldBank EN.POP.DNST (national, all targets) + Census ACS (US states) +
    # Japan 2020 census static (47 prefectures). Writes overseas_population_density.
    "J":  {"module": "group_j_population_density", "desc": "Population density (WorldBank + Census + JP)", "est_sec": 30},
    # Group W added 2026-05-24: overseas daily weather via Open-Meteo archive API.
    # Coverage: 51 US state capitals + 47 JP prefecture capitals + 10 EU capitals.
    # Free API, no key required. Parallel fetching (6 workers). ~2-3 min full run.
    # Writes overseas_weather (source, country, location, date → temp/precip/wind/humidity).
    "W":  {"module": "group_w_overseas_weather",   "desc": "Overseas weather (Open-Meteo, US+JP+EU)", "est_sec": 180},
    # Group T added 2026-05-24: intra-country regional commuter flow matrices.
    # US: Census ACS 2016-2020 county→county aggregated to state×state (51 codes).
    # DE: Bundesagentur 30.06.2020 BL×BL matrix (16 Bundesländer, ~260 KB).
    # JP: e-Stat Population Census 2020 (requires ESTAT_APP_ID; skipped if absent).
    # Writes commuter_flows (source, country, origin, destination, workers, year).
    # Run once (static reference year data); re-run only if source data updated.
    "T":  {"module": "group_t_commuter_flows",     "desc": "Commuter flows (Census/Bundesagentur, US+DE+JP)", "est_sec": 60},
}

# CM (monthly transit) added 2026-04-24 so `collect --groups all` actually
# populates monthly_subway_hourly / monthly_bus_hourly on a fresh install.
# Previously CM lived only as an unreachable module in legacy/ and the
# original 4.6M-row DB was reproducible only because cron kept it alive.
# Groups I/K/N added 2026-05-24: overseas ILI, GU weather, hospital burden.
# K/N run last (depend on H/Q being populated first).
DEFAULT_ORDER = ["E", "D", "S", "B", "C", "CM", "A", "F", "H", "R", "G", "P", "Q", "I", "K", "N", "O", "J", "W", "T"]


def _derive_collector_kwargs(group: str, backfill_days: int | None) -> dict:
    """Translate `--backfill-days N` into each collector's native parameter.

    Groups expose different period knobs (days, months, years, KDCA YYYYMM
    periods). This helper keeps that mapping in one place so CLI users see a
    single `--backfill-days` flag and the orchestrator does the arithmetic.

    Returns an empty dict when `backfill_days` is falsy — in that case every
    collector falls back to its own default (usually "today only" or
    "last 3 months"), preserving the old incremental behaviour.
    """
    if not backfill_days or backfill_days <= 0:
        return {}

    today = datetime.today()
    cutoff = today - timedelta(days=int(backfill_days))

    if group == "B":
        return {"historical_days": int(backfill_days)}
    if group == "C":
        return {"backfill_days": int(backfill_days)}
    if group == "CM":
        # months_back must cover the full backfill window; round up.
        return {"months_back": max(1, (int(backfill_days) + 29) // 30)}
    if group == "D":
        return {"start_year": cutoff.year, "end_year": today.year}
    if group == "E":
        # KDCA periods are semi-annual YYYYMM. Round to Jan/Dec bookends
        # so the E collectors scan the full calendar window.
        return {"start_prd": f"{cutoff.year}01", "end_prd": f"{today.year}12"}
    if group == "S":
        return {"start_year": cutoff.year, "end_year": today.year}
    # I, K, N, O, J, W: pass backfill_days directly; each collector converts internally.
    if group in ("I", "K", "N", "O", "J", "W"):
        return {"backfill_days": int(backfill_days)}
    # T: static reference data (Census/Bundesagentur 2020); backfill_days irrelevant.
    # A, F, G, H, P, Q, R: realtime / snapshot collectors with no time-range.
    return {}


def _call_with_supported_kwargs(fn, kwargs: dict):
    """Invoke `fn` passing only the kwargs it actually declares.

    Shields the orchestrator from TypeError when a collector's entry point
    doesn't accept the derived kwarg (e.g. group_a_realtime.run()).
    """
    if not kwargs:
        return fn()
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins / C-extensions: just try with kwargs, fall back to none.
        try:
            return fn(**kwargs)
        except TypeError:
            return fn()
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


_COLLECTORS_DIR = Path(__file__).resolve().parent


def _load_collector(group: str):
    """Import a collector module.

    Search order (first match wins):
      1. simulation/collectors/<module>.py  (new-style collectors: I, K, N, ...)
      2. simulation/collectors/legacy/<module>.py  (legacy groups A-S)
    """
    info = GROUP_INFO.get(group)
    if not info:
        raise ValueError(f"Unknown group: {group}")

    module_name = info["module"]

    # ── 1. New-style collector (in collectors/ directly) ─────────────────
    new_style_path = _COLLECTORS_DIR / f"{module_name}.py"
    if new_style_path.exists():
        try:
            mod = importlib.import_module(f"simulation.collectors.{module_name}")
            return mod
        except Exception as e:
            log.error("Failed to import collector %s (new-style): %s", group, e)
            return None

    # ── 2. Legacy collector (in collectors/legacy/) ───────────────────────
    module_path = _LEGACY_DIR / f"{module_name}.py"
    if not module_path.exists():
        log.warning("Collector not found: %s (checked %s and %s)",
                    group, new_style_path, module_path)
        return None

    # Add legacy dir to sys.path temporarily for internal cross-imports
    legacy_str = str(_LEGACY_DIR)
    added = False
    if legacy_str not in sys.path:
        sys.path.insert(0, legacy_str)
        added = True

    try:
        # Import as simulation.collectors.legacy.<module>
        mod = importlib.import_module(f"simulation.collectors.legacy.{module_name}")
        return mod
    except ImportError:
        # Fallback: direct import from legacy dir
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            log.error("Failed to import collector %s: %s", group, e)
            return None
    finally:
        if added and legacy_str in sys.path:
            sys.path.remove(legacy_str)

def list_groups() -> dict:
    """Return available collection groups with descriptions."""
    result = {}
    for g, info in GROUP_INFO.items():
        mod = info["module"]
        # Available if found in collectors/ (new-style) OR collectors/legacy/
        available = (
            (_COLLECTORS_DIR / f"{mod}.py").exists()
            or (_LEGACY_DIR / f"{mod}.py").exists()
        )
        result[g] = {
            "desc":      info["desc"],
            "est_sec":   info["est_sec"],
            "available": available,
        }
    return result


def run_collection(groups: list[str] | None = None, force: bool = False,
                   verbose: bool = True, timeout_per_group: int = 3600,
                   backfill_days: int | None = None):
    """
    Run data collection for specified groups (or all in DEFAULT_ORDER).

    When `backfill_days` is set, every time-windowed collector (B / C / CM /
    D / E / S) is asked to sweep the last N days. This is how fresh installs
    reach parity with the curated 4.6M-row DB — without it, each collector
    only pulls "today's delta" and it takes months to rebuild.
    """
    conn = init_db()
    conn.close()

    groups = groups or DEFAULT_ORDER
    total = len(groups)
    results = {"ok": [], "fail": [], "skip": []}
    for i, g in enumerate(groups, 1):
        info = GROUP_INFO.get(g)
        if not info:
            log.warning("Unknown group: %s, skipping", g)
            results["skip"].append(g)
            continue

        if verbose:
            print(f"\n[{i}/{total}] Group {g}: {info['desc']} ...")

        t0 = time.time()
        mod = _load_collector(g)
        if mod is None:
            results["skip"].append(g)
            if verbose:
                print(f"  -> SKIP (collector not available)")
            continue

        kwargs = _derive_collector_kwargs(g, backfill_days)
        # `sub_self_logged`: when the entry returns a dict, every sub-API has
        # already called log_collection() with its own status. Adding an
        # aggregate "group_x EMPTY" row on top is duplicate noise (sum of
        # OK-but-zero-row sub-APIs would otherwise produce a false EMPTY).
        sub_self_logged = False
        try:
            # Dispatch order (2026-04-17 fix):
            #   1. module-level collect()           -- legacy contract
            #   2. module-level main()              -- legacy contract
            #   3. collect_all_*() function         -- e.g. collect_all_sentinel (group S)
            #   4. GroupXCollector().run()          -- canonical class entry point
            #      (orchestrator previously called collect() which no class defined,
            #       silently crashing A2/A3/A4 since 2026-03-22)
            if hasattr(mod, "collect"):
                extra = dict(kwargs); extra["force"] = force
                rows, error = _call_with_supported_kwargs(mod.collect, extra)
            elif hasattr(mod, "main"):
                rows = _call_with_supported_kwargs(mod.main, kwargs)
                error = None
            elif hasattr(mod, "run") and callable(mod.run):
                # New-style collectors (group_i, group_w, group_g, etc.) expose
                # only a module-level run() returning a result dict.
                result = _call_with_supported_kwargs(mod.run, kwargs)
                if isinstance(result, dict):
                    rows = result.get("inserted", 0) + result.get("rows", 0)
                    sub_self_logged = True
                    error = (", ".join(result["errors"]) if result.get("errors") else None)
                elif isinstance(result, int):
                    rows = result
                    error = None
                else:
                    rows = 0
                    error = None
            else:
                cls_name = [n for n in dir(mod)
                            if "Collector" in n and n != "BaseCollector"]
                collect_all = [n for n in dir(mod) if n.startswith("collect_all")]
                if collect_all:
                    result = _call_with_supported_kwargs(
                        getattr(mod, collect_all[0]), kwargs,
                    )
                    if isinstance(result, dict):
                        rows = sum(v for v in result.values() if isinstance(v, int))
                        sub_self_logged = True
                    else:
                        rows = result
                    error = None
                elif cls_name:
                    collector = getattr(mod, cls_name[0])()
                    # Prefer run() (canonical); fall back to collect() / __call__.
                    entry = None
                    for attr in ("run", "collect", "__call__"):
                        if callable(getattr(collector, attr, None)):
                            entry = getattr(collector, attr)
                            break
                    if entry is None:
                        log.warning("Group %s: %s has no run()/collect() entry",
                                    g, cls_name[0])
                        results["skip"].append(g)
                        continue
                    result = _call_with_supported_kwargs(entry, kwargs)
                    # run() conventionally returns a dict {api_name: n_rows};
                    # collapse into an integer row count for the summary log.
                    if isinstance(result, dict):
                        rows = sum(v for v in result.values() if isinstance(v, int))
                        sub_self_logged = True
                    elif isinstance(result, int):
                        rows = result
                    else:
                        rows = len(result) if result else 0
                    error = None
                else:
                    log.warning("Group %s: no collect/main/run/Collector found", g)
                    results["skip"].append(g)
                    continue
            elapsed = time.time() - t0
            n = rows if isinstance(rows, int) else (len(rows) if rows else 0)
            # Suppress umbrella when sub-APIs already self-logged. Otherwise
            # treat 0-rows-but-no-error as OK (call succeeded; "no new data"
            # is recorded via rows_saved=0, not via a separate EMPTY status).
            if error:
                log_collection(g, info["module"], "FAIL", n,
                               elapsed=elapsed, error=error)
            elif not sub_self_logged:
                log_collection(g, info["module"], "OK", n,
                               elapsed=elapsed, error=None)

            if error:
                results["fail"].append(g)
                if verbose:
                    print(f"  -> FAIL ({elapsed:.1f}s): {error}")
            else:
                results["ok"].append(g)
                if verbose:
                    print(f"  -> OK ({n} rows, {elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            log.error("Group %s error: %s", g, e, exc_info=True)
            log_collection(
                g, info["module"], "ERROR", 0,
                elapsed=elapsed, error=str(e),
            )
            results["fail"].append(g)
            if verbose:
                print(f"  -> ERROR ({elapsed:.1f}s): {e}")

    # : emit the final summary via the logger (not bare print) so it
    # also lands in simulation/logs/collect_*.log under the auto-logging
    # FileHandler. `verbose` only controls whether it's ALSO shown on
    # stdout, which the StreamHandler already handles via propagation.
    summary = (
        f"Collection complete: {len(results['ok'])} OK, "
        f"{len(results['fail'])} FAIL, {len(results['skip'])} SKIP"
    )
    log.info("=" * 50)
    log.info(summary)
    if results["ok"]:
        log.info("  OK:   %s", ", ".join(results["ok"]))
    if results["fail"]:
        log.warning("  FAIL: %s", ", ".join(results["fail"]))
    if results["skip"]:
        log.info("  SKIP: %s", ", ".join(results["skip"]))


def run_collection_parallel(
    groups: list[str] | None = None,
    force: bool = False,
    max_workers: int = 4,
    verbose: bool = True,
    backfill_days: int | None = None,
) -> dict:
    """Run multiple collector groups in parallel via ThreadPoolExecutor.

    Each collector is IO-bound (HTTP polling), so threading gives near-linear
    speedup even with the GIL. SQLite WAL mode serializes writes, so
    concurrent inserts are safe (writers queue on the journal lock).

    Caveats:
      * max_workers=4 is conservative; raise if your network has capacity.
      * Some APIs have per-client rate limits that sibling threads may hit.
      * Order-dependent collectors (E depends on D for disease_master) must
        still run serially — this function does NOT respect order, use
        `run_collection` for strict ordering.

    Returns {"ok": [...], "fail": [...], "skip": [...], "elapsed_s": float}.
    """
    import concurrent.futures as cf

    conn = init_db()
    conn.close()

    groups = groups or DEFAULT_ORDER
    t_start = time.time()
    results = {"ok": [], "fail": [], "skip": []}

    def _one_group(g: str) -> tuple[str, str, str | None]:
        """Returns (group, status, error)."""
        info = GROUP_INFO.get(g)
        if not info:
            return (g, "SKIP", f"unknown group {g}")
        mod = _load_collector(g)
        if mod is None:
            return (g, "SKIP", "collector not available")
        t0 = time.time()
        kwargs = _derive_collector_kwargs(g, backfill_days)
        sub_self_logged = False
        try:
            # Same dispatch as run_collection — kept in sync.
            if hasattr(mod, "collect"):
                extra = dict(kwargs); extra["force"] = force
                rows, error = _call_with_supported_kwargs(mod.collect, extra)
            elif hasattr(mod, "main"):
                rows = _call_with_supported_kwargs(mod.main, kwargs)
                error = None
            elif hasattr(mod, "run") and callable(mod.run):
                # New-style collectors (group_i, group_w, group_g, etc.) expose
                # only a module-level run() returning a result dict.
                res = _call_with_supported_kwargs(mod.run, kwargs)
                if isinstance(res, dict):
                    rows = res.get("inserted", 0) + res.get("rows", 0)
                    sub_self_logged = True
                    error = (", ".join(res["errors"]) if res.get("errors") else None)
                elif isinstance(res, int):
                    rows = res
                    error = None
                else:
                    rows = 0
                    error = None
            else:
                cls_name = [n for n in dir(mod)
                            if "Collector" in n and n != "BaseCollector"]
                collect_all = [n for n in dir(mod) if n.startswith("collect_all")]
                if collect_all:
                    res = _call_with_supported_kwargs(
                        getattr(mod, collect_all[0]), kwargs,
                    )
                    if isinstance(res, dict):
                        rows = sum(v for v in res.values() if isinstance(v, int))
                        sub_self_logged = True
                    else:
                        rows = res
                    error = None
                elif cls_name:
                    collector = getattr(mod, cls_name[0])()
                    entry = None
                    for attr in ("run", "collect", "__call__"):
                        if callable(getattr(collector, attr, None)):
                            entry = getattr(collector, attr)
                            break
                    if entry is None:
                        return (g, "SKIP", f"{cls_name[0]} has no entry")
                    res = _call_with_supported_kwargs(entry, kwargs)
                    if isinstance(res, dict):
                        rows = sum(v for v in res.values() if isinstance(v, int))
                        sub_self_logged = True
                    elif isinstance(res, int):
                        rows = res
                    else:
                        rows = len(res) if res else 0
                    error = None
                else:
                    return (g, "SKIP", "no collect/main/run/Collector")
            elapsed = time.time() - t0
            n = rows if isinstance(rows, int) else (len(rows) if rows else 0)
            # Same suppression rule as serial run_collection (kept in sync).
            if error:
                log_collection(g, info["module"], "FAIL", n,
                               elapsed=elapsed, error=error)
            elif not sub_self_logged:
                log_collection(g, info["module"], "OK", n,
                               elapsed=elapsed, error=None)
            return (g, "OK" if not error else "FAIL", error)
        except Exception as e:
            elapsed = time.time() - t0
            log.error("Group %s error (parallel): %s", g, e, exc_info=True)
            log_collection(g, info["module"], "ERROR", 0,
                           elapsed=elapsed, error=str(e))
            return (g, "FAIL", str(e))

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one_group, g): g for g in groups}
        for fut in cf.as_completed(futures):
            g, status, err = fut.result()
            if status == "OK":
                results["ok"].append(g)
            elif status == "SKIP":
                results["skip"].append(g)
            else:
                results["fail"].append(g)
            if verbose:
                mark = "✓" if status == "OK" else ("·" if status == "SKIP" else "✗")
                print(f"  [{mark}] group {g}: {status}" +
                      (f" — {err[:60]}" if err else ""))

    elapsed_total = time.time() - t_start
    results["elapsed_s"] = round(elapsed_total, 2)
    results["max_workers"] = max_workers
    log.info("=" * 50)
    log.info(f"Collection (parallel, workers={max_workers}): "
             f"{len(results['ok'])} OK, {len(results['fail'])} FAIL, "
             f"{len(results['skip'])} SKIP in {elapsed_total:.1f}s")
    return results


def print_status(db_path: str | None = None):
    """Print table row counts and collection freshness."""
    shapes = get_table_shapes(db_path)
    if not shapes:
        print("No tables found. Run `db-init` first.")
        return

    print(f"\n{'Table':<35} {'Rows':>10}")
    print("-" * 47)
    total_rows = 0
    for table, cnt in sorted(shapes.items()):
        if table == "sqlite_sequence":
            continue
        status = f"{cnt:>10,}" if cnt >= 0 else "    ERROR"
        print(f"  {table:<33} {status}")
        if cnt > 0:
            total_rows += cnt
    print("-" * 47)
    print(f"  {'TOTAL':<33} {total_rows:>10,}")
    print(f"  Tables: {len(shapes)}")