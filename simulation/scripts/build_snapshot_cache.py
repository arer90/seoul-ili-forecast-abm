"""ABM/ARIA materialization snapshot builder — DB → static parquet.

Reads the historical epidemiological tables ONCE (via the lock-free
``read_only_connect``) and materializes them into
``simulation/cache/snapshot/*.parquet`` so that the ABM (Pillar 2) and the
ARIA LLM layer (Pillar 4) load a static snapshot at startup instead of
re-querying the SQLite DB on every run. A ``manifest.json`` records, per table,
the row count, the temporal coverage and the generation timestamp.

The companion helper :func:`load_snapshot` is the read side: it reads the
parquet snapshot when present and transparently falls back to a live
``read_only_connect`` query when the snapshot is missing — so callers get the
same DataFrame either way.

Design notes (honesty / project rules):
  * read path = ``simulation.database.read_only_connect`` ONLY — never a raw
    low-level DB handle (G-116/117 forbids the latter).
  * Existing code is NOT modified — this is a new, additive file.
  * The manifest ``generated_at`` is NOT ``datetime.now()`` baked at import.
    It is passed in (``--generated-at``) or, by default, derived from the data
    itself: ``max(collected_at)`` across the materialized tables (a stable,
    reproducible vintage stamp). The per-run wall-clock is recorded separately
    as ``built_at`` so the two concepts stay distinct.
  * Large tables (e.g. ``daily_population_gu_hourly`` ≈ 1.79 M rows) are
    projected to the columns the consumers actually need.
  * Overseas: WHO FluNet / overseas_ili are observations only — NO overseas
    geojson, NO overseas forecast is produced here (OverseasTransfer = phantom).

Usage:
    .venv/bin/python -m simulation.scripts.build_snapshot_cache
    .venv/bin/python -m simulation.scripts.build_snapshot_cache --no-figure
    .venv/bin/python -m simulation.scripts.build_snapshot_cache --generated-at 2026-06-26
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import polars as pl

from simulation.database import DB_PATH, read_only_connect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("snapshot_cache")

# ── Output location (single cache root, project rule #4) ──────────────────
# DB_PATH = <root>/simulation/data/db/epi_real_seoul.db → project root = parents[3]
_PROJECT_ROOT = Path(DB_PATH).resolve().parents[3]
SNAPSHOT_DIR = _PROJECT_ROOT / "simulation" / "cache" / "snapshot"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"


# ══════════════════════════════════════════════════════════════════════════
# Table specifications — the SSOT for what gets materialized
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SnapshotSpec:
    """One materialized snapshot table.

    Attributes:
        name: snapshot/table identifier (also the parquet stem and the
            ``load_snapshot(name)`` key).
        sql: SELECT statement (column-projected for large tables). Must read
            only — no writes.
        period_cols: ordered ``(min, max)`` column expressions used to compute
            the temporal coverage string for the manifest. Empty = no period.
        time_label: human label for the period (e.g. ``"season_start"``).
    """

    name: str
    sql: str
    period_cols: tuple[str, ...] = field(default_factory=tuple)
    time_label: str = ""


# Seoul ILI: age-band rows averaged to a single all-ages series, matching the
# established convention in simulation/abm/epi_proof.py:211
# (SELECT season_start, week_seq, AVG(ili_rate) ... GROUP BY season_start, week_seq).
_SPECS: list[SnapshotSpec] = [
    SnapshotSpec(
        name="sentinel_influenza",
        sql=(
            "SELECT season_start, week_seq, week_label, "
            "       AVG(ili_rate) AS ili_rate, "
            "       MAX(collected_at) AS collected_at "
            "FROM sentinel_influenza "
            "WHERE ili_rate IS NOT NULL AND ili_rate >= 0 "
            "GROUP BY season_start, week_seq, week_label "
            "ORDER BY season_start, week_seq"
        ),
        period_cols=("season_start",),
        time_label="season_start",
    ),
    # Mobility core: compact gu×gu coupling matrix (625 rows) — preferred over
    # the 1.79 M-row hourly panel for ABM metapop coupling.
    SnapshotSpec(
        name="commuter_matrix",
        sql=(
            "SELECT origin_gu, dest_gu, commuters, coupling, night_population, "
            "       source, MAX(collected_at) AS collected_at "
            "FROM commuter_matrix "
            "GROUP BY origin_gu, dest_gu, source "
            "ORDER BY origin_gu, dest_gu"
        ),
        period_cols=(),
        time_label="",
    ),
    # Hourly daytime population per gu — LARGE (≈1.79 M rows). Project to the
    # columns ABM age-structured WAIFW actually consumes (total + age bands).
    SnapshotSpec(
        name="daily_population_gu_hourly",
        sql=(
            "SELECT stdr_de, gu_code, gu_nm, hour, tot_pop, "
            "       pop_0_9, pop_10_19, pop_20_29, pop_30_39, "
            "       pop_40_49, pop_50_59, pop_60_69, pop_70plus "
            "FROM daily_population_gu_hourly "
            "ORDER BY stdr_de, gu_code, hour"
        ),
        period_cols=("stdr_de",),
        time_label="stdr_de",
    ),
    SnapshotSpec(
        name="vaccination_coverage",
        sql=(
            "SELECT ref_year, vaccine_nm, gu_nm, age_group, coverage_pct, "
            "       dose_cnt, target_pop, MAX(collected_at) AS collected_at "
            "FROM vaccination_coverage "
            "GROUP BY ref_year, vaccine_nm, gu_nm, age_group "
            "ORDER BY ref_year, vaccine_nm, gu_nm, age_group"
        ),
        period_cols=("ref_year",),
        time_label="ref_year",
    ),
    SnapshotSpec(
        name="sentinel_ari",
        sql=(
            "SELECT year, week_no, pathogen_group, pathogen_nm, count, "
            "       MAX(collected_at) AS collected_at "
            "FROM sentinel_ari "
            "GROUP BY year, week_no, pathogen_group, pathogen_nm "
            "ORDER BY year, week_no"
        ),
        period_cols=("year",),
        time_label="year",
    ),
    SnapshotSpec(
        name="sentinel_sari",
        sql=(
            "SELECT year, week_no, week_label, count, "
            "       MAX(collected_at) AS collected_at "
            "FROM sentinel_sari GROUP BY year, week_no, week_label "
            "ORDER BY year, week_no"
        ),
        period_cols=("year",),
        time_label="year",
    ),
    SnapshotSpec(
        name="sentinel_hfmd",
        sql=(
            "SELECT year, week_no, week_label, rate, "
            "       MAX(collected_at) AS collected_at "
            "FROM sentinel_hfmd GROUP BY year, week_no, week_label "
            "ORDER BY year, week_no"
        ),
        period_cols=("year",),
        time_label="year",
    ),
    SnapshotSpec(
        name="sentinel_enterovirus",
        sql=(
            "SELECT year, week_no, count, "
            "       MAX(collected_at) AS collected_at "
            "FROM sentinel_enterovirus GROUP BY year, week_no "
            "ORDER BY year, week_no"
        ),
        period_cols=("year",),
        time_label="year",
    ),
    # WHO FluNet — observations only, project to the influenza-relevant columns
    # used as overseas leading indicators (NO forecast produced).
    SnapshotSpec(
        name="who_flunet",
        sql=(
            "SELECT country, iso3, hemisphere, year, week_no, sdate, edate, "
            "       spec_processed, inf_a, inf_b, inf_total, inf_negative, "
            "       ili_activity, MAX(collected_at) AS collected_at "
            "FROM who_flunet "
            "GROUP BY country, year, week_no "
            "ORDER BY country, year, week_no"
        ),
        period_cols=("year",),
        time_label="year",
    ),
    SnapshotSpec(
        name="overseas_ili",
        sql=(
            "SELECT source, country, year, week_no, ili_rate, positivity_pct, "
            "       specimen_positive, specimen_total, influenza_a, influenza_b, "
            "       MAX(collected_at) AS collected_at "
            "FROM overseas_ili "
            "GROUP BY source, country, year, week_no "
            "ORDER BY source, country, year, week_no"
        ),
        period_cols=("year",),
        time_label="year",
    ),
]

_SPEC_BY_NAME = {s.name: s for s in _SPECS}


# ══════════════════════════════════════════════════════════════════════════
# Build side
# ══════════════════════════════════════════════════════════════════════════
def _query_to_df(con, sql: str) -> pl.DataFrame:
    """Run a read-only SELECT and return a polars DataFrame.

    Args:
        con: an open ``read_only_connect`` SQLite connection.
        sql: SELECT statement (no writes).

    Returns:
        polars DataFrame with one row per result row; empty (0-row) frame if the
        query matched nothing.

    Performance: O(rows) materialization; the largest spec
    (daily_population_gu_hourly) is ~1.79 M rows / ~150 MB peak.
    Side effects: none (read-only cursor).
    """
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    # infer_schema_length=None scans ALL rows so a column whose early rows are
    # NULL/small ints but later rows overflow the inferred dtype is typed
    # correctly (overseas_ili specimen counts hit this otherwise).
    return pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None)


def _period_string(df: pl.DataFrame, spec: SnapshotSpec) -> Optional[str]:
    """Compute a ``"<min>..<max>"`` coverage string for the manifest, or None."""
    if not spec.period_cols or df.height == 0:
        return None
    col = spec.period_cols[0]
    if col not in df.columns:
        return None
    lo = df[col].min()
    hi = df[col].max()
    return f"{lo}..{hi}"


def _resolve_generated_at(
    per_table_collected: dict[str, Optional[str]], override: Optional[str]
) -> str:
    """Resolve the manifest vintage stamp WITHOUT wall-clock ``datetime.now()``.

    Args:
        per_table_collected: per-table ``max(collected_at)`` (or None).
        override: explicit ``--generated-at`` value (wins if provided).

    Returns:
        The override if given; else the maximum ``collected_at`` seen across all
        materialized tables; else the literal ``"unknown"`` (no DB collected_at
        anywhere). Never the current wall-clock.
    """
    if override:
        return override
    stamps = [v for v in per_table_collected.values() if v]
    return max(stamps) if stamps else "unknown"


def build_snapshots(
    out_dir: Path = SNAPSHOT_DIR,
    generated_at: Optional[str] = None,
    *,
    make_figure: bool = True,
) -> dict:
    """Materialize every spec to parquet and write ``manifest.json``.

    Args:
        out_dir: snapshot output directory (created if absent).
        generated_at: explicit manifest vintage stamp; None → derived from
            ``max(collected_at)`` across tables (see :func:`_resolve_generated_at`).
        make_figure: also render a Korean-labelled coverage summary PNG.

    Returns:
        The manifest dict (also written to disk as ``manifest.json``).

    Performance: single read-only connection, one query per spec; dominated by
    the hourly-population scan (~few seconds, ~150 MB peak).
    Side effects: writes ``<out_dir>/<name>.parquet`` for every spec, plus
    ``manifest.json`` and (optionally) ``coverage_summary.png``.
    Caller responsibility: none — read-only, no DB writes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    import datetime as _dt

    built_at = _dt.datetime.now().isoformat(timespec="seconds")

    tables: dict[str, dict] = {}
    per_table_collected: dict[str, Optional[str]] = {}

    con = read_only_connect()
    try:
        for spec in _SPECS:
            try:
                df = _query_to_df(con, spec.sql)
            except Exception as exc:  # table missing / schema drift — record, continue
                log.warning("skip %s: %s", spec.name, exc)
                tables[spec.name] = {"status": "error", "error": str(exc)}
                continue

            parquet_path = out_dir / f"{spec.name}.parquet"
            df.write_parquet(parquet_path)

            collected = None
            if "collected_at" in df.columns and df.height > 0:
                collected = df["collected_at"].max()
            per_table_collected[spec.name] = collected

            tables[spec.name] = {
                "status": "ok",
                "rows": int(df.height),
                "columns": list(df.columns),
                "period": _period_string(df, spec),
                "time_axis": spec.time_label or None,
                "max_collected_at": collected,
                "file": parquet_path.name,
                "bytes": int(parquet_path.stat().st_size),
            }
            log.info(
                "wrote %s rows=%d -> %s (%d bytes)",
                spec.name,
                df.height,
                parquet_path.name,
                tables[spec.name]["bytes"],
            )
    finally:
        con.close()

    manifest = {
        "generated_at": _resolve_generated_at(per_table_collected, generated_at),
        "built_at": built_at,
        "snapshot_dir": str(out_dir),
        "db_path": str(DB_PATH),
        "n_tables": sum(1 for t in tables.values() if t.get("status") == "ok"),
        "tables": tables,
    }
    MANIFEST_PATH_local = out_dir / "manifest.json"
    MANIFEST_PATH_local.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("wrote manifest -> %s", MANIFEST_PATH_local)

    if make_figure:
        try:
            _render_coverage_figure(manifest, out_dir / "coverage_summary.png")
        except Exception as exc:  # figure is a convenience, never fatal
            log.warning("coverage figure skipped: %s", exc)

    return manifest


def _render_coverage_figure(manifest: dict, out_png: Path) -> None:
    """Render a Korean-labelled bar chart of rows-per-snapshot.

    Uses matplotlib Agg + a Korean font (AppleGothic → NanumGothic). Honest:
    bars are the actual materialized row counts from the manifest, no fabrication.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("AppleGothic", "NanumGothic", "Apple SD Gothic Neo"):
        if cand in available:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    ok = [(n, t) for n, t in manifest["tables"].items() if t.get("status") == "ok"]
    ok.sort(key=lambda kv: kv[1]["rows"], reverse=True)
    names = [n for n, _ in ok]
    rows = [t["rows"] for _, t in ok]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(names) + 1.5))
    bars = ax.barh(names, rows, color="#3b6ea5")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("행 수 (log scale)")
    ax.set_title(
        f"ABM/ARIA 스냅샷 캐시 커버리지  (vintage={manifest['generated_at']})"
    )
    for bar, r in zip(bars, rows):
        ax.text(
            bar.get_width() * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{r:,}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info("wrote coverage figure -> %s", out_png)


# ══════════════════════════════════════════════════════════════════════════
# Read side — snapshot-first, DB-fallback
# ══════════════════════════════════════════════════════════════════════════
def load_snapshot(
    name: str,
    *,
    snapshot_dir: Path = SNAPSHOT_DIR,
    as_pandas: bool = False,
):
    """Load a materialized snapshot table, falling back to a live DB query.

    Read path for ABM/ARIA: prefer the static parquet snapshot; if it is absent
    (snapshot never built) and ``name`` is a known spec, run that spec's SELECT
    against the DB via ``read_only_connect`` so the caller still gets data.

    Args:
        name: snapshot identifier (one of the built-in spec names, e.g.
            ``"sentinel_influenza"``, ``"commuter_matrix"``, ``"who_flunet"``).
        snapshot_dir: directory holding ``<name>.parquet``.
        as_pandas: return a pandas DataFrame instead of polars (sklearn bridge).

    Returns:
        polars.DataFrame (or pandas DataFrame if ``as_pandas``) with the snapshot
        rows. Empty frame if the table matched nothing.

    Raises:
        KeyError: ``name`` is neither a built parquet nor a known spec — there is
            no way to source it (fail loud, never silently return empty).

    Performance: parquet read is O(rows), memory-mapped by polars; DB fallback
    is one SELECT. Side effects: opens (and closes) a read-only DB fd only on
    the fallback path.
    """
    parquet_path = snapshot_dir / f"{name}.parquet"
    if parquet_path.exists():
        df = pl.read_parquet(parquet_path)
        return df.to_pandas() if as_pandas else df

    spec = _SPEC_BY_NAME.get(name)
    if spec is None:
        raise KeyError(
            f"unknown snapshot '{name}': no parquet at {parquet_path} and no spec "
            f"(known: {sorted(_SPEC_BY_NAME)})"
        )
    log.info("snapshot '%s' absent — falling back to live DB query", name)
    con = read_only_connect()
    try:
        df = _query_to_df(con, spec.sql)
    finally:
        con.close()
    return df.to_pandas() if as_pandas else df


def available_snapshots() -> list[str]:
    """Return the names of all known snapshot specs (build targets)."""
    return [s.name for s in _SPECS]


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build ABM/ARIA snapshot parquet cache.")
    ap.add_argument(
        "--out-dir",
        default=str(SNAPSHOT_DIR),
        help="snapshot output directory (default: simulation/cache/snapshot)",
    )
    ap.add_argument(
        "--generated-at",
        default=None,
        help="manifest vintage stamp; default = max(collected_at) across tables "
        "(NEVER wall-clock now)",
    )
    ap.add_argument(
        "--no-figure", action="store_true", help="skip the coverage summary PNG"
    )
    args = ap.parse_args(argv)

    manifest = build_snapshots(
        out_dir=Path(args.out_dir),
        generated_at=args.generated_at,
        make_figure=not args.no_figure,
    )

    ok = manifest["n_tables"]
    total = len(_SPECS)
    log.info(
        "DONE: %d/%d tables materialized (vintage=%s)",
        ok,
        total,
        manifest["generated_at"],
    )
    return 0 if ok > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
