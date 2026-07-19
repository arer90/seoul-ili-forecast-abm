"""Generate the 2 seoul_gu S1 setup CSVs from the DB (사용자 2026-06-07).

The original ``/tmp/seoul_step1_proxy_gt.py`` was lost, blocking
``seoul_gu --smoke``. This rebuilds both required CSVs FROM the real DB so the
"1-model full-pipeline smoke" can run:

  * ``seoul_proxy_weekly_option_a.csv`` — KR national sentinel ILI × Seoul share
    (0.1833), keyed by (season_start, week_seq).
  * ``seoul_25gu_weekly_features.csv``  — per-gu weekly mobility (real
    daily_population_district) + age shares (real daily_population_gu_hourly),
    keyed by (gu_nm, year=season_start, week_no=week_seq).

Week alignment: sentinel week_seq is sequential within the season (week_seq=1 ⇒
epi week 36). A calendar date maps to (season_start, week_seq) via ISO week:
ISO≥36 → (year, ISO−35); else → (year−1, ISO+18). Both CSVs therefore share the
same (season_start, week_seq) join key build_s1_dataset expects.

Run:  .venv/bin/python -m simulation.scripts.gen_seoul_smoke_csvs
"""
from __future__ import annotations

import csv
import datetime
from collections import defaultdict
from pathlib import Path

from simulation.database.storage import read_only_connect

SEOUL_DIR = Path("simulation/results/seoul_only")
DB = "simulation/data/db/epi_real_seoul.db"
SEOUL_SHARE = 0.1833          # Seoul population / national (Option A)


def _season_week(yyyymmdd: str) -> tuple[int, int] | None:
    try:
        d = datetime.date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    except (ValueError, IndexError):
        return None
    iso_y, iso_w, _ = d.isocalendar()
    if iso_w >= 36:
        return iso_y, iso_w - 35          # epi 36→week_seq 1
    return iso_y - 1, iso_w + 18          # epi 1→week_seq 19 … epi 35→week_seq 53


def main() -> int:
    SEOUL_DIR.mkdir(parents=True, exist_ok=True)
    c = read_only_connect(DB)
    try:
        # ── 1) proxy CSV (national sentinel × Seoul share) ──────────────────
        proxy = c.execute(
            "SELECT season_start, week_seq, AVG(ili_rate) FROM sentinel_influenza "
            "WHERE ili_rate IS NOT NULL GROUP BY season_start, week_seq "
            "ORDER BY season_start, week_seq").fetchall()
        ppath = SEOUL_DIR / "seoul_proxy_weekly_option_a.csv"
        with ppath.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["season_start", "week_seq", "seoul_proxy_ili_optA"])
            for s, wk, ili in proxy:
                w.writerow([int(s), int(wk), round(float(ili) * SEOUL_SHARE, 6)])
        print(f"  proxy: {len(proxy)} rows → {ppath}")

        # ── 2) per-gu mobility (district) → weekly mean ─────────────────────
        mob: dict = defaultdict(lambda: {"day": 0.0, "night": 0.0, "tot": 0.0, "n": 0})
        for stdr, gu, day, night, tot in c.execute(
                "SELECT stdr_de, signgu_nm, day_livpop, night_livpop, tot_livpop "
                "FROM daily_population_district WHERE signgu_nm IS NOT NULL"):
            sw = _season_week(str(stdr))
            if sw is None or not str(gu).endswith("구"):   # 25 districts only
                continue
            m = mob[(gu, sw[0], sw[1])]
            m["day"] += float(day or 0); m["night"] += float(night or 0)
            m["tot"] += float(tot or 0); m["n"] += 1

        # ── age shares (hourly) → weekly ────────────────────────────────────
        age: dict = defaultdict(lambda: {"child": 0.0, "senior": 0.0, "tot": 0.0})
        for stdr, gu, tot, p09, p1019, p6069, p70 in c.execute(
                "SELECT stdr_de, gu_nm, tot_pop, pop_0_9, pop_10_19, pop_60_69, "
                "pop_70plus FROM daily_population_gu_hourly WHERE gu_nm IS NOT NULL"):
            sw = _season_week(str(stdr))
            if sw is None:
                continue
            a = age[(gu, sw[0], sw[1])]
            a["child"] += float((p09 or 0) + (p1019 or 0))
            a["senior"] += float((p6069 or 0) + (p70 or 0))
            a["tot"] += float(tot or 0)

        fpath = SEOUL_DIR / "seoul_25gu_weekly_features.csv"
        n_rows = 0
        with fpath.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gu_nm", "year", "week_no", "daytime_pop_mean",
                        "nighttime_pop_mean", "avg_pop_mean", "child_share_mean",
                        "senior_share_mean", "day_night_ratio"])
            for (gu, s, wk), m in sorted(mob.items()):
                if m["n"] == 0:
                    continue
                day, night, tot = m["day"] / m["n"], m["night"] / m["n"], m["tot"] / m["n"]
                a = age.get((gu, s, wk))
                child = a["child"] / a["tot"] if a and a["tot"] > 0 else 0.12
                senior = a["senior"] / a["tot"] if a and a["tot"] > 0 else 0.17
                dnr = day / night if night > 0 else 1.0
                w.writerow([gu, int(s), int(wk), round(day, 2), round(night, 2),
                            round(tot, 2), round(child, 4), round(senior, 4), round(dnr, 4)])
                n_rows += 1
        print(f"  25gu features: {n_rows} rows ({len({k[0] for k in mob})} gu) → {fpath}")
    finally:
        c.close()
    print("  ✓ seoul_gu smoke unblocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
