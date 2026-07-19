"""Export ~500 MB lite DB for Railway MCP deploy.

Why: epi_real_seoul.db is 12 GB (79 tables). Railway Hobby ($5/mo) gives
5 GB ephemeral storage; Pro ($20+/mo) gives 100 GB. To stay on Hobby,
copy only the tables ARIA actually queries via MCP — the same set the
ARIA digest already uses, plus a few HIRA tables for severity coverage.

Run:
    .venv/bin/python scripts/export_lite_db.py
Output:
    simulation/data/db_lite/epi_lite.db   (~500 MB target)

Then commit (it'll bypass the 12 GB DB, which is gitignored):
    git add -f simulation/data/db_lite/epi_lite.db
    git commit -m "data: epi_lite for Railway deploy"
    git push

The Dockerfile.mcp picks it up at the same relative path. Vercel does
not need this — it never sees the DB; it reaches Railway's MCP via HTTP.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.database import safe_connect

SRC = PROJECT_ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
DST_DIR = PROJECT_ROOT / "simulation" / "data" / "db_lite"
DST = DST_DIR / "epi_lite.db"

# Tables ARIA actually queries through MCP. Adjust if /api/chat surfaces
# new tool calls during the 30-user load test.
KEEP_TABLES = [
    # KDCA surveillance — primary citation source
    "sentinel_influenza",
    "sentinel_ari",
    "sentinel_sari",
    "seoul_disease_district",
    "who_flunet",
    "who_flunet_metadata",
    "overseas_ili",
    "disease_master",
    "disease_catalog",
    "disease_age",
    "disease_gender",
    # Population / mobility — sim grounding
    "daily_population_district",
    "daily_population_gu_hourly",
    "kosis_age_district",
    "commuter_matrix",
    # Weather / air quality
    "weather_historical",
    "weather_forecast",
    "rt_air_quality",
    # HIRA — severity / hospital burden coverage
    "hira_region",
    "hira_facility",
    "hira_gender_age",
    "hira_inpat_opat",
    # Subway / bus — transit indicators
    "daily_subway",
    "daily_bus",
    # Vaccination
    "childhood_vaccination_rates",
]


def main():
    if not SRC.exists():
        print(f"❌ source DB not found: {SRC}")
        sys.exit(1)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    if DST.exists():
        DST.unlink()

    src = safe_connect(str(SRC), verify=False)
    src.execute(f"ATTACH DATABASE '{DST}' AS lite")

    # Tables that get rolled up to daily granularity to fit under GitHub's
    # 100 MB single-file limit. The hourly population table alone is ~1.79 M
    # rows (~71% of total disk); aggregating to daily slashes it 24x.
    AGGREGATE_DAILY = {
        "daily_population_gu_hourly": """
            SELECT stdr_de, gu_code, gu_nm,
                CAST(AVG(tot_pop) AS BIGINT) AS tot_pop_avg,
                CAST(MAX(tot_pop) AS BIGINT) AS tot_pop_peak,
                CAST(AVG(male_pop) AS BIGINT) AS male_pop_avg,
                CAST(AVG(female_pop) AS BIGINT) AS female_pop_avg,
                CAST(AVG(pop_0_9) AS BIGINT) AS pop_0_9_avg,
                CAST(AVG(pop_10_19) AS BIGINT) AS pop_10_19_avg,
                CAST(AVG(pop_20_29) AS BIGINT) AS pop_20_29_avg,
                CAST(AVG(pop_30_39) AS BIGINT) AS pop_30_39_avg,
                CAST(AVG(pop_40_49) AS BIGINT) AS pop_40_49_avg,
                CAST(AVG(pop_50_59) AS BIGINT) AS pop_50_59_avg,
                CAST(AVG(pop_60_69) AS BIGINT) AS pop_60_69_avg,
                CAST(AVG(pop_70plus) AS BIGINT) AS pop_70plus_avg
            FROM daily_population_gu_hourly
            GROUP BY stdr_de, gu_code, gu_nm
        """,
    }

    kept = 0
    skipped = 0
    for tbl in KEEP_TABLES:
        try:
            n = src.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        except Exception as e:
            print(f"  ⚠ {tbl} — not found in source: {e}")
            skipped += 1
            continue
        if tbl in AGGREGATE_DAILY:
            src.execute(f"CREATE TABLE lite.{tbl} AS {AGGREGATE_DAILY[tbl]}")
            agg_n = src.execute(
                f"SELECT COUNT(*) FROM lite.{tbl}"
            ).fetchone()[0]
            print(f"  ⤓ {tbl:38s} {n:>10,} → {agg_n:,} rows (daily aggregate)")
        else:
            src.execute(f"CREATE TABLE lite.{tbl} AS SELECT * FROM {tbl}")
            print(f"  ✓ {tbl:38s} {n:>10,} rows")
        kept += 1

    src.execute("DETACH DATABASE lite")
    src.close()

    size_mb = DST.stat().st_size / 1024 / 1024
    print()
    print(f"→ {DST}")
    print(f"  {kept} tables copied, {skipped} skipped, {size_mb:.1f} MB")
    if size_mb > 4500:
        print(
            f"  ⚠ exceeds Railway Hobby 5 GB limit — drop a table or"
            f" upgrade to Pro plan",
        )


if __name__ == "__main__":
    main()
