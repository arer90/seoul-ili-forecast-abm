"""EDA report — comprehensive exploratory data analysis of epi_real_seoul.db.

Produces a complete audit of the 77-table DB:
  * Per-table row count, column schema, data-type distribution
  * NULL ratio per column (flags high-NULL fields for review)
  * Outlier / range check for numeric columns (IQR fences)
  * Temporal coverage for time-series tables (earliest / latest / gaps)
  * Disease / gu / week coverage grids (ILI target matrix)
  * Cross-table consistency (e.g. sentinel_influenza season counts)
  * Collection health (collector status breakdown by API + by group)

Outputs go to:
  simulation/results/eda/  (or $MPH_OUTPUT_ROOT/results/eda/)
    ├── index.html              — navigable report with all sections
    ├── summary.json            — machine-readable overview
    ├── tables/                 — per-table CSV dumps (cols, stats)
    │   ├── row_counts.csv
    │   ├── null_ratios.csv
    │   ├── numeric_summary.csv
    │   └── temporal_coverage.csv
    └── figures/                — PNG charts
        ├── rows_per_table.png
        ├── null_ratio_top20.png
        ├── collection_status_pie.png
        ├── ili_time_series.png
        ├── disease_coverage_heatmap.png
        └── commuter_matrix_heatmap.png

Usage:
    uv run python -m simulation.scripts.eda_report
    uv run python -m simulation.scripts.eda_report --quick   # skip heavy plots
    uv run python -m simulation.scripts.eda_report --open    # open HTML when done
"""
from __future__ import annotations

import argparse
import html
import json
import sqlite3
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

# matplotlib: non-interactive backend so this runs headless in CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import polars as pl

from simulation.database.config import DB_PATH
from simulation.utils.paths import get_results_dir


# ─── Output paths ──────────────────────────────────────────────────────

def _eda_root() -> Path:
    p = get_results_dir() / "eda"
    (p / "tables").mkdir(parents=True, exist_ok=True)
    (p / "figures").mkdir(parents=True, exist_ok=True)
    return p


# ─── Helpers ───────────────────────────────────────────────────────────

def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _column_info(conn: sqlite3.Connection, table: str) -> list[dict]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [
        {"cid": r[0], "name": r[1], "type": r[2], "notnull": bool(r[3]),
         "dflt_value": r[4], "pk": bool(r[5])}
        for r in cur.fetchall()
    ]


def _null_ratios(conn: sqlite3.Connection, table: str, cols: list[dict]) -> dict[str, float]:
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if n == 0:
        return {c["name"]: 0.0 for c in cols}
    out = {}
    for c in cols:
        nulls = conn.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{c["name"]}" IS NULL'
        ).fetchone()[0]
        out[c["name"]] = nulls / n
    return out


def _numeric_summary(conn: sqlite3.Connection, table: str, cols: list[dict]) -> list[dict]:
    """For REAL/INTEGER columns: min, median, mean, max, IQR, outlier count."""
    out = []
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if n == 0:
        return out
    for c in cols:
        ctype = c["type"].upper()
        if not any(t in ctype for t in ("INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE")):
            continue
        if c["pk"]:
            continue  # skip PK autoincrements
        try:
            stats = conn.execute(f"""
                SELECT MIN("{c['name']}"), MAX("{c['name']}"),
                       AVG("{c['name']}"),
                       (SELECT COUNT(*) FROM {table} WHERE "{c['name']}" IS NOT NULL)
                FROM {table}
            """).fetchone()
        except sqlite3.OperationalError:
            continue
        mn, mx, mean, n_nn = stats
        if n_nn == 0 or mn is None:
            continue
        # SQLite can return '' for MIN/MAX on mixed-type columns
        try:
            mn = float(mn); mx = float(mx)
            mean = float(mean) if mean is not None else 0.0
        except (TypeError, ValueError):
            continue
        # Median + IQR via Python (SQLite doesn't have PERCENTILE)
        # SQLite is type-permissive — a REAL column may contain '' strings or
        # text that coerces poorly. Coerce defensively, drop failures.
        raw = [r[0] for r in conn.execute(
            f'SELECT "{c["name"]}" FROM {table} WHERE "{c["name"]}" IS NOT NULL'
        )]
        if not raw:
            continue
        clean = []
        for v in raw:
            if v is None or v == "":
                continue
            try:
                clean.append(float(v))
            except (TypeError, ValueError):
                continue
        if not clean:
            continue
        vals_np = np.asarray(clean, dtype=float)
        q1, med, q3 = np.percentile(vals_np, [25, 50, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(np.sum((vals_np < lo) | (vals_np > hi)))
        out.append({
            "table": table, "column": c["name"], "type": c["type"],
            "n_nn": n_nn,
            "min": mn, "q1": float(q1), "median": float(med),
            "q3": float(q3), "max": mx, "mean": mean,
            "iqr": float(iqr),
            "outlier_count": outliers,
            "outlier_pct": round(outliers / n_nn * 100, 2),
        })
    return out


def _temporal_coverage(conn: sqlite3.Connection) -> list[dict]:
    """For tables with a date/time column, report earliest/latest/count."""
    # Heuristic: columns whose name matches these patterns
    date_patterns = ("date", "_de", "stdr_de", "week_start", "season_start",
                     "cal_date", "collected_at", "obs_date", "use_dt", "use_ymd", "use_ym")
    tables = _table_names(conn)
    out = []
    for t in tables:
        cols = _column_info(conn, t)
        for c in cols:
            name_lower = c["name"].lower()
            if any(p in name_lower for p in date_patterns):
                try:
                    row = conn.execute(f"""
                        SELECT MIN("{c['name']}"), MAX("{c['name']}"),
                               COUNT(DISTINCT "{c['name']}")
                        FROM {t}
                        WHERE "{c['name']}" IS NOT NULL
                    """).fetchone()
                except sqlite3.OperationalError:
                    continue
                mn, mx, distinct = row
                if mn is None:
                    continue
                out.append({
                    "table": t, "date_col": c["name"],
                    "earliest": str(mn), "latest": str(mx),
                    "distinct_dates": int(distinct),
                })
                break  # one date column per table
    return out


# ─── Section builders ──────────────────────────────────────────────────

def section_overview(conn: sqlite3.Connection) -> dict:
    """High-level DB stats."""
    tables = _table_names(conn)
    total_rows = 0
    per_table = []
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        total_rows += n
        per_table.append({"table": t, "rows": n})
    per_table.sort(key=lambda x: -x["rows"])
    size_mb = Path(str(DB_PATH)).stat().st_size / 1024**2
    return {
        "db_path": str(DB_PATH),
        "db_size_mb": round(size_mb, 1),
        "n_tables": len(tables),
        "total_rows": total_rows,
        "top_10_tables_by_rows": per_table[:10],
        "empty_tables": [r["table"] for r in per_table if r["rows"] == 0],
    }


def section_collection_status(conn: sqlite3.Connection) -> dict:
    """Breakdown of collection_log by status + API."""
    try:
        from simulation.collectors.status import (
            FAILURE_STATUSES, BENIGN_NO_PROGRESS, Status,
        )
    except ImportError:
        FAILURE_STATUSES = {"FAIL", "ERROR"}
        BENIGN_NO_PROGRESS = {"SKIP", "EMPTY"}

    rows = conn.execute("""
        SELECT api_name,
               SUM(CASE WHEN status='OK' THEN 1 ELSE 0 END) AS n_ok,
               SUM(CASE WHEN status IN ('SKIP','EMPTY') THEN 1 ELSE 0 END) AS n_skip,
               SUM(CASE WHEN status IN ('FAIL','ERROR','FAIL_IMPORT','FAIL_NO_SCHOOLS') THEN 1 ELSE 0 END) AS n_fail,
               SUM(COALESCE(rows_saved, 0)) AS total_rows,
               MAX(collected_at) AS last
        FROM collection_log
        GROUP BY api_name
        ORDER BY n_fail DESC, total_rows DESC
    """).fetchall()
    apis = [
        {"api_name": r[0], "n_ok": r[1], "n_skip": r[2], "n_fail": r[3],
         "total_rows": r[4], "last_seen": r[5]}
        for r in rows
    ]
    status_totals = conn.execute("""
        SELECT status, COUNT(*) FROM collection_log GROUP BY status
    """).fetchall()
    return {
        "total_api_calls": sum(r[1] for r in status_totals),
        "status_breakdown": {r[0]: r[1] for r in status_totals},
        "per_api": apis,
    }


def section_disease_coverage(conn: sqlite3.Connection) -> dict:
    """sentinel_influenza + weekly_disease + seoul_disease_district coverage."""
    out = {}
    for t, date_col in [
        ("sentinel_influenza", "season_start"),
        ("weekly_disease", "year"),
        ("seoul_disease_district", "year"),
    ]:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            if n == 0:
                out[t] = {"rows": 0, "note": "empty (run collectors)"}
                continue
            mn, mx = conn.execute(
                f"SELECT MIN({date_col}), MAX({date_col}) FROM {t}"
            ).fetchone()
            out[t] = {"rows": n, "earliest": str(mn), "latest": str(mx)}
        except sqlite3.OperationalError as e:
            out[t] = {"error": str(e)}
    return out


# ─── Figures ───────────────────────────────────────────────────────────

def fig_rows_per_table(conn: sqlite3.Connection, fig_dir: Path):
    tables = _table_names(conn)
    counts = []
    for t in tables:
        counts.append((t, conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]))
    counts.sort(key=lambda x: -x[1])
    counts = [(t, n) for t, n in counts if n > 0][:30]
    if not counts:
        return

    names = [c[0] for c in counts]
    vals = [c[1] for c in counts]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(names))))
    ax.barh(names[::-1], vals[::-1], color="#3b82f6")
    ax.set_xscale("log")
    ax.set_xlabel("rows (log scale)")
    ax.set_title(f"Top {len(names)} tables by row count")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "rows_per_table.png", dpi=110)
    plt.close(fig)


def fig_null_ratio_top20(conn: sqlite3.Connection, fig_dir: Path):
    """Top 20 columns by NULL %. Flags data-quality issues."""
    tables = _table_names(conn)
    all_rows = []
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if n < 10:
            continue
        for c in _column_info(conn, t):
            if c["pk"]:
                continue
            try:
                k = conn.execute(
                    f'SELECT COUNT(*) FROM {t} WHERE "{c["name"]}" IS NULL'
                ).fetchone()[0]
            except sqlite3.OperationalError:
                continue
            if k > 0:
                all_rows.append({
                    "table": t, "column": c["name"],
                    "null_pct": k / n * 100,
                })
    if not all_rows:
        return
    all_rows.sort(key=lambda r: -r["null_pct"])
    top = all_rows[:20]
    labels = [f"{r['table']}.{r['column']}" for r in top]
    vals = [r["null_pct"] for r in top]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#ef4444" if v > 50 else "#f59e0b" if v > 20 else "#84cc16" for v in vals]
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlim(0, 100)
    ax.set_xlabel("NULL %")
    ax.set_title("Top-20 columns by NULL ratio")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "null_ratio_top20.png", dpi=110)
    plt.close(fig)


def fig_collection_status_pie(conn: sqlite3.Connection, fig_dir: Path):
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM collection_log GROUP BY status"
    ).fetchall()
    if not rows:
        return
    labels = [r[0] for r in rows]
    sizes = [r[1] for r in rows]
    colors_map = {
        "OK": "#22c55e", "SKIP": "#60a5fa", "EMPTY": "#a78bfa",
        "FAIL": "#ef4444", "ERROR": "#dc2626",
        "FAIL_IMPORT": "#991b1b", "FAIL_NO_SCHOOLS": "#7f1d1d",
    }
    colors = [colors_map.get(l, "#94a3b8") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=[f"{l} ({n})" for l, n in zip(labels, sizes)],
           colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("collection_log status breakdown")
    fig.tight_layout()
    fig.savefig(fig_dir / "collection_status_pie.png", dpi=110)
    plt.close(fig)


def fig_ili_time_series(conn: sqlite3.Connection, fig_dir: Path):
    try:
        df = pl.read_database(
            "SELECT season_start, week_seq, ili_rate FROM sentinel_influenza "
            "WHERE ili_rate IS NOT NULL ORDER BY season_start, week_seq",
            conn,
        )
    except Exception:
        return
    if df.is_empty():
        return
    df = df.with_columns(
        t=pl.col("season_start").cast(pl.Float64) + pl.col("week_seq").cast(pl.Float64) / 53.0
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["t"].to_numpy(), df["ili_rate"].to_numpy(),
            color="#2563eb", lw=1.1)
    ax.set_xlabel("season (year + week/53)")
    ax.set_ylabel("ILI rate")
    ax.set_title(f"sentinel_influenza — ILI rate over time  "
                  f"(n={df.height} weeks)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "ili_time_series.png", dpi=110)
    plt.close(fig)


def fig_commuter_matrix_heatmap(conn: sqlite3.Connection, fig_dir: Path):
    try:
        df = pl.read_database(
            "SELECT origin, destination, flow FROM commuter_matrix", conn
        )
    except Exception:
        return
    if df.is_empty():
        return
    origins = sorted(df["origin"].unique().to_list())
    dests = sorted(df["destination"].unique().to_list())
    n_o, n_d = len(origins), len(dests)
    if n_o == 0 or n_d == 0:
        return
    mat = np.zeros((n_o, n_d))
    o_idx = {o: i for i, o in enumerate(origins)}
    d_idx = {d: i for i, d in enumerate(dests)}
    for o, d, f in zip(df["origin"].to_list(), df["destination"].to_list(),
                        df["flow"].to_list()):
        mat[o_idx[o], d_idx[d]] = f

    fig, ax = plt.subplots(figsize=(9, 8))
    # log-scale for readability
    with np.errstate(divide="ignore"):
        mat_log = np.log10(np.maximum(mat, 1))
    im = ax.imshow(mat_log, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_d))
    ax.set_xticklabels(dests, rotation=90, fontsize=7)
    ax.set_yticks(range(n_o))
    ax.set_yticklabels(origins, fontsize=7)
    ax.set_xlabel("destination (commute to)")
    ax.set_ylabel("origin (residence)")
    ax.set_title("Commuter matrix (log10 flow)")
    fig.colorbar(im, ax=ax, label="log10(flow)")
    fig.tight_layout()
    fig.savefig(fig_dir / "commuter_matrix_heatmap.png", dpi=110)
    plt.close(fig)


def fig_disease_coverage_heatmap(conn: sqlite3.Connection, fig_dir: Path):
    try:
        df = pl.read_database(
            "SELECT year, disease_nm, SUM(cases) AS n FROM weekly_disease "
            "WHERE cases IS NOT NULL GROUP BY year, disease_nm",
            conn,
        )
    except Exception:
        return
    if df.is_empty():
        return
    # top 25 diseases
    top = (df.group_by("disease_nm")
             .agg(pl.col("n").sum().alias("tot"))
             .sort("tot", descending=True)
             .head(25)["disease_nm"].to_list())
    df2 = df.filter(pl.col("disease_nm").is_in(top))
    pivoted = (df2.pivot(index="disease_nm", on="year", values="n")
                 .fill_null(0))
    years = sorted(c for c in pivoted.columns if c != "disease_nm")
    mat = pivoted.select(years).to_numpy()
    names = pivoted["disease_nm"].to_list()
    with np.errstate(divide="ignore"):
        mat_log = np.log10(np.maximum(mat, 1))

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mat_log, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("year")
    ax.set_title("weekly_disease — top 25 diseases × year (log10 cases)")
    fig.colorbar(im, ax=ax, label="log10(cases)")
    fig.tight_layout()
    fig.savefig(fig_dir / "disease_coverage_heatmap.png", dpi=110)
    plt.close(fig)


# ─── HTML index ────────────────────────────────────────────────────────

def _html_table(rows: list[dict], cols: list[str]) -> str:
    """Render a list of dicts as an HTML table."""
    if not rows:
        return "<p><em>(no rows)</em></p>"
    headers = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols
        ) + "</tr>"
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


def build_html(summary: dict, eda_dir: Path) -> Path:
    """Build index.html with all sections linked."""
    def a(target, label):
        return f'<a href="{target}">{html.escape(label)}</a>'

    ov = summary["overview"]
    coll = summary["collection"]
    disease = summary["disease_coverage"]
    temporal = summary["temporal_coverage"]

    # Inline figures
    figs = sorted(p.name for p in (eda_dir / "figures").glob("*.png"))
    fig_html = "".join(
        f'<figure><img src="figures/{f}" alt="{f}"/><figcaption>{f}</figcaption></figure>\n'
        for f in figs
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>EDA — {html.escape(Path(str(DB_PATH)).name)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2em;
         color: #0f172a; max-width: 1200px; }}
 h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: .4em; }}
 h2 {{ color: #1e3a8a; margin-top: 2em; border-bottom: 1px solid #cbd5e1;
       padding-bottom: .3em; }}
 h3 {{ color: #475569; }}
 table {{ border-collapse: collapse; margin: .5em 0; font-size: .9em; }}
 th, td {{ border: 1px solid #cbd5e1; padding: .3em .6em; text-align: left; }}
 th {{ background: #eff6ff; }}
 tr:nth-child(even) {{ background: #f8fafc; }}
 .metric {{ display: inline-block; padding: .6em 1em; margin: .4em;
            background: #eff6ff; border-left: 4px solid #2563eb;
            border-radius: 4px; }}
 .metric .v {{ font-size: 1.5em; font-weight: 600; color: #1e3a8a; }}
 .metric .k {{ color: #64748b; font-size: .85em; }}
 figure {{ margin: 1em 0; padding: 1em; background: #f8fafc;
           border-radius: 6px; }}
 figure img {{ max-width: 100%; display: block; border: 1px solid #cbd5e1; }}
 figcaption {{ color: #64748b; font-size: .85em; margin-top: .5em; }}
 code {{ background: #e2e8f0; padding: 1px 5px; border-radius: 3px; }}
 .nav a {{ margin-right: 1em; }}
 .warn {{ color: #b45309; }}
 .fail {{ color: #b91c1c; font-weight: 600; }}
 .ok {{ color: #166534; }}
</style>
</head>
<body>

<h1>EDA — epi_real_seoul.db</h1>
<p>Generated {html.escape(time.strftime("%Y-%m-%d %H:%M:%S"))}</p>

<p class="nav">
 {a('#overview', '개요')}
 {a('#collection', '수집 상태')}
 {a('#tables', '테이블별')}
 {a('#disease', '질병 coverage')}
 {a('#temporal', '시간 coverage')}
 {a('#figures', '그래프')}
 {a('tables/row_counts.csv', 'CSV 다운로드')}
</p>

<h2 id="overview">Overview</h2>
<div class="metric"><span class="v">{ov['n_tables']}</span><br><span class="k">tables</span></div>
<div class="metric"><span class="v">{ov['total_rows']:,}</span><br><span class="k">total rows</span></div>
<div class="metric"><span class="v">{ov['db_size_mb']} MB</span><br><span class="k">DB size</span></div>
<div class="metric"><span class="v">{len(ov['empty_tables'])}</span><br><span class="k">empty tables</span></div>

<h3>Top 10 tables by rows</h3>
{_html_table(ov['top_10_tables_by_rows'], ['table', 'rows'])}

<h3>Empty tables ({len(ov['empty_tables'])})</h3>
<p>{', '.join(html.escape(t) for t in ov['empty_tables']) or '<em>none</em>'}</p>

<h2 id="collection">Collection status</h2>
<p>Total API calls recorded: <strong>{coll['total_api_calls']:,}</strong></p>
<ul>
 {''.join(f'<li><strong>{html.escape(k)}</strong>: {v:,}</li>' for k, v in coll['status_breakdown'].items())}
</ul>
<h3>Per-API breakdown (top 30)</h3>
{_html_table(coll['per_api'][:30], ['api_name','n_ok','n_skip','n_fail','total_rows','last_seen'])}

<h2 id="disease">Disease coverage</h2>
<ul>
 {''.join(f'<li><strong>{html.escape(k)}</strong>: {html.escape(json.dumps(v, ensure_ascii=False))}</li>' for k, v in disease.items())}
</ul>

<h2 id="temporal">Temporal coverage (tables with date columns)</h2>
{_html_table(temporal, ['table','date_col','earliest','latest','distinct_dates'])}

<h2 id="figures">Figures</h2>
{fig_html}

<p style="margin-top:3em; color:#94a3b8; font-size: .8em;">
 Regenerate: <code>uv run python -m simulation.scripts.eda_report</code>
</p>
</body>
</html>
"""
    out = eda_dir / "index.html"
    out.write_text(html_doc, encoding="utf-8")
    return out


# ─── CSV dumps ─────────────────────────────────────────────────────────

def dump_csvs(conn: sqlite3.Connection, eda_dir: Path):
    td = eda_dir / "tables"

    # row counts
    tables = _table_names(conn)
    rows = []
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cols = _column_info(conn, t)
        rows.append({"table": t, "rows": n, "columns": len(cols)})
    pl.DataFrame(rows).sort("rows", descending=True).write_csv(td / "row_counts.csv")

    # null ratios (all columns of all non-empty tables)
    rows = []
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if n == 0:
            continue
        cols = _column_info(conn, t)
        for c in cols:
            try:
                k = conn.execute(
                    f'SELECT COUNT(*) FROM {t} WHERE "{c["name"]}" IS NULL'
                ).fetchone()[0]
            except sqlite3.OperationalError:
                continue
            rows.append({
                "table": t, "column": c["name"], "type": c["type"],
                "null_count": k, "null_pct": round(k / n * 100, 2),
                "n": n,
            })
    pl.DataFrame(rows).sort("null_pct", descending=True).write_csv(td / "null_ratios.csv")

    # numeric summary
    rows = []
    for t in tables:
        cols = _column_info(conn, t)
        rows.extend(_numeric_summary(conn, t, cols))
    if rows:
        pl.DataFrame(rows).write_csv(td / "numeric_summary.csv")

    # temporal coverage
    tc = _temporal_coverage(conn)
    if tc:
        pl.DataFrame(tc).write_csv(td / "temporal_coverage.csv")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip heavy plots (keeps row_counts / null_ratios)")
    ap.add_argument("--open", action="store_true",
                    help="open HTML in default browser when done")
    args = ap.parse_args()

    t0 = time.time()
    eda_dir = _eda_root()
    print(f"EDA output: {eda_dir}")
    from simulation.database import safe_connect  # ENGINEERING_PRINCIPLES.md §원칙 #3 — single writer
    conn = safe_connect()
    conn.row_factory = None

    # sections
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "db_path": str(DB_PATH),
        "overview": section_overview(conn),
        "collection": section_collection_status(conn),
        "disease_coverage": section_disease_coverage(conn),
        "temporal_coverage": _temporal_coverage(conn),
    }
    (eda_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  ✓ summary.json")

    dump_csvs(conn, eda_dir)
    print(f"  ✓ CSV dumps → {eda_dir}/tables/")

    fig_dir = eda_dir / "figures"
    fig_rows_per_table(conn, fig_dir)
    fig_null_ratio_top20(conn, fig_dir)
    fig_collection_status_pie(conn, fig_dir)
    print(f"  ✓ basic figures")

    if not args.quick:
        fig_ili_time_series(conn, fig_dir)
        fig_commuter_matrix_heatmap(conn, fig_dir)
        fig_disease_coverage_heatmap(conn, fig_dir)
        print(f"  ✓ heavy figures (--quick to skip)")

    html_path = build_html(summary, eda_dir)
    print(f"  ✓ {html_path}")

    conn.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Open:\n  file://{html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
