#!/usr/bin/env python3
"""Build multi-disease weekly seasonality aggregate for the web dashboard.

Source tables (epi_real_seoul.db, read-only):
  sentinel_influenza  — age-group-averaged ILI rate/1,000명 (전국/도시 표본감시)
  sentinel_hfmd       — 수족구병 rate/1,000명 (외래환자 표본감시)
  sentinel_enterovirus— 엔테로바이러스 count  (실험실 표본감시)
  sentinel_ari        — 급성호흡기감염증 count (pathogen_group='계', pathogen_nm='계')
  sentinel_intestinal — 장관감염증 count      (pathogen_group='계', pathogen_nm='계')
  sentinel_ophlgc     — 눈병 rate/1,000명    (disease_nm='전체')

Output: web/public/aggregates/multi-disease-seasonality.json
Schema:
  {
    "built_at": "ISO8601",
    "source": "KDCA 표본감시 sentinel (epi_real_seoul.db)",
    "note": "전국/도시레벨 감시(관측, 예측 아님). 자치구 분리 불가.",
    "diseases": [
      {
        "id": str,           # machine key
        "name": str,         # English label
        "name_ko": str,      # Korean label
        "unit": str,         # "rate" or "count"
        "unit_label": str,   # human-readable unit string
        "color": str,        # hex colour for chart
        "year_range": str,   # "2017–2026"
        "n_years": int,
        "peak_week": int,    # week-of-year with highest mean value
        "source": str,       # table name
        "weekly": [          # list length 53 (week 1–53)
          {
            "week": int,        # 1–53
            "value": float,     # mean across years (null if no data)
            "value_norm": float # 0–1 min-max normalised (disease-specific max)
          }
        ]
      }
    ]
  }

Seasonality computation:
  1. For each disease, group by week_of_year → mean value across all available years.
  2. value_norm = value / max(value) within that disease (0–1 scale).
  3. peak_week = argmax of weekly mean.

Honest-label guarantees:
  - sentinel_influenza has no '전체' age_group → average across all 7 age groups.
  - sentinel_ari / sentinel_intestinal use pathogen_group='계' AND pathogen_nm='계' totals.
  - sentinel_ophlgc uses disease_nm='전체'.
  - year_range is computed from actual DB data, not hardcoded.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT = ROOT / "web" / "public" / "aggregates" / "multi-disease-seasonality.json"

# Disease colour palette — distinct, readable on dark background
_COLORS = {
    "influenza":   "#3b82f6",  # blue
    "hfmd":        "#f59e0b",  # amber  (summer peak → warm tone)
    "enterovirus": "#10b981",  # emerald
    "ari":         "#a855f7",  # purple
    "intestinal":  "#ef4444",  # red
    "ophlgc":      "#06b6d4",  # cyan
}


def _week_means(conn, sql: str, params: tuple = ()) -> dict[int, list[float]]:
    """Return {week_of_year: [values…]} mapping from a query.

    The query must return rows of (week_of_year, value) where
    week_of_year is an integer 1–53 and value is numeric.
    """
    wmap: dict[int, list[float]] = {}
    for row in conn.execute(sql, params):
        wk = int(row[0])
        val = row[1]
        if wk < 1 or wk > 53:
            continue
        if val is None:
            continue
        wmap.setdefault(wk, []).append(float(val))
    return wmap


def _build_weekly(wmap: dict[int, list[float]]) -> tuple[list[dict], int]:
    """Convert week→values map to sorted weekly list with normalisation.

    Returns:
        weekly  — list of {week, value, value_norm} for weeks 1-53
        peak_week — int week number with highest mean
    """
    means: dict[int, float] = {
        wk: sum(vals) / len(vals)
        for wk, vals in wmap.items()
        if vals
    }
    if not means:
        weekly = [{"week": w, "value": None, "value_norm": None} for w in range(1, 54)]
        return weekly, 1

    max_val = max(means.values()) or 1.0
    weekly = []
    for w in range(1, 54):
        v = means.get(w)
        weekly.append({
            "week": w,
            "value": round(v, 4) if v is not None else None,
            "value_norm": round(v / max_val, 4) if v is not None else None,
        })
    peak_week = max(means, key=lambda w: means[w])
    return weekly, int(peak_week)


def _year_range(conn, table: str, year_col: str = "year") -> tuple[int | None, int | None]:
    row = conn.execute(
        f"SELECT MIN({year_col}), MAX({year_col}) FROM {table}"
    ).fetchone()
    return row[0], row[1]


def build() -> dict[str, Any]:
    from simulation.database import read_only_connect  # project-standard read-only helper

    con = read_only_connect(str(DB))
    try:
        diseases: list[dict] = []

        # ── 1. sentinel_influenza ──────────────────────────────────────────────
        # No '전체' age_group — average across all 7 age groups per week-of-year.
        # week_label is like '01주' (calendar week), directly usable as week-of-year.
        yr_min, yr_max = _year_range(con, "sentinel_influenza", "season_start")
        wmap = _week_means(
            con,
            """
            SELECT CAST(REPLACE(week_label, '주', '') AS INTEGER) AS woy,
                   AVG(ili_rate)
            FROM sentinel_influenza
            GROUP BY woy
            """,
        )
        weekly, peak = _build_weekly(wmap)
        diseases.append({
            "id": "influenza",
            "name": "Influenza (ILI)",
            "name_ko": "인플루엔자 (ILI)",
            "unit": "rate",
            "unit_label": "ILI / 1,000명 (전 연령 평균)",
            "color": _COLORS["influenza"],
            "year_range": f"{yr_min}–{yr_max}" if yr_min else "—",
            "n_years": (yr_max - yr_min + 1) if yr_min and yr_max else 0,
            "peak_week": peak,
            "source": "sentinel_influenza (KDCA 인플루엔자 표본감시)",
            "weekly": weekly,
        })

        # ── 2. sentinel_hfmd ──────────────────────────────────────────────────
        yr_min, yr_max = _year_range(con, "sentinel_hfmd")
        wmap = _week_means(
            con,
            "SELECT week_no, AVG(rate) FROM sentinel_hfmd GROUP BY week_no",
        )
        weekly, peak = _build_weekly(wmap)
        diseases.append({
            "id": "hfmd",
            "name": "Hand-Foot-Mouth (HFMD)",
            "name_ko": "수족구병 (HFMD)",
            "unit": "rate",
            "unit_label": "환자수 / 외래환자 1,000명",
            "color": _COLORS["hfmd"],
            "year_range": f"{yr_min}–{yr_max}" if yr_min else "—",
            "n_years": (yr_max - yr_min + 1) if yr_min and yr_max else 0,
            "peak_week": peak,
            "source": "sentinel_hfmd (KDCA 수족구병 표본감시)",
            "weekly": weekly,
        })

        # ── 3. sentinel_enterovirus ────────────────────────────────────────────
        yr_min, yr_max = _year_range(con, "sentinel_enterovirus")
        wmap = _week_means(
            con,
            "SELECT week_no, AVG(count) FROM sentinel_enterovirus GROUP BY week_no",
        )
        weekly, peak = _build_weekly(wmap)
        diseases.append({
            "id": "enterovirus",
            "name": "Enterovirus",
            "name_ko": "엔테로바이러스",
            "unit": "count",
            "unit_label": "검출 건수 (주간)",
            "color": _COLORS["enterovirus"],
            "year_range": f"{yr_min}–{yr_max}" if yr_min else "—",
            "n_years": (yr_max - yr_min + 1) if yr_min and yr_max else 0,
            "peak_week": peak,
            "source": "sentinel_enterovirus (KDCA 엔테로바이러스 실험실 감시)",
            "weekly": weekly,
        })

        # ── 4. sentinel_ari ────────────────────────────────────────────────────
        # pathogen_group='계', pathogen_nm='계' = total ARI across all pathogens
        yr_min, yr_max = _year_range(con, "sentinel_ari")
        wmap = _week_means(
            con,
            """
            SELECT week_no, AVG(count) FROM sentinel_ari
            WHERE pathogen_group = '계' AND pathogen_nm = '계'
            GROUP BY week_no
            """,
        )
        weekly, peak = _build_weekly(wmap)
        diseases.append({
            "id": "ari",
            "name": "Acute Respiratory Infection (ARI)",
            "name_ko": "급성호흡기감염증 (ARI)",
            "unit": "count",
            "unit_label": "병원체 검출 건수 (주간, 전체)",
            "color": _COLORS["ari"],
            "year_range": f"{yr_min}–{yr_max}" if yr_min else "—",
            "n_years": (yr_max - yr_min + 1) if yr_min and yr_max else 0,
            "peak_week": peak,
            "source": "sentinel_ari (KDCA 급성호흡기감염증 표본감시)",
            "weekly": weekly,
        })

        # ── 5. sentinel_intestinal ─────────────────────────────────────────────
        yr_min, yr_max = _year_range(con, "sentinel_intestinal")
        wmap = _week_means(
            con,
            """
            SELECT week_no, AVG(count) FROM sentinel_intestinal
            WHERE pathogen_group = '계' AND pathogen_nm = '계'
            GROUP BY week_no
            """,
        )
        weekly, peak = _build_weekly(wmap)
        diseases.append({
            "id": "intestinal",
            "name": "Intestinal Infection",
            "name_ko": "장관감염증",
            "unit": "count",
            "unit_label": "병원체 검출 건수 (주간, 전체)",
            "color": _COLORS["intestinal"],
            "year_range": f"{yr_min}–{yr_max}" if yr_min else "—",
            "n_years": (yr_max - yr_min + 1) if yr_min and yr_max else 0,
            "peak_week": peak,
            "source": "sentinel_intestinal (KDCA 장관감염증 표본감시)",
            "weekly": weekly,
        })

        # ── 6. sentinel_ophlgc ────────────────────────────────────────────────
        # disease_nm='전체' = combined (급성출혈결막염 + 유행성각결막염)
        yr_min, yr_max = _year_range(con, "sentinel_ophlgc")
        wmap = _week_means(
            con,
            """
            SELECT week_no, AVG(rate) FROM sentinel_ophlgc
            WHERE disease_nm = '전체'
            GROUP BY week_no
            """,
        )
        weekly, peak = _build_weekly(wmap)
        diseases.append({
            "id": "ophlgc",
            "name": "Eye Infection (Ophthalmic)",
            "name_ko": "눈병 (안과감염)",
            "unit": "rate",
            "unit_label": "환자수 / 외래환자 1,000명 (전체)",
            "color": _COLORS["ophlgc"],
            "year_range": f"{yr_min}–{yr_max}" if yr_min else "—",
            "n_years": (yr_max - yr_min + 1) if yr_min and yr_max else 0,
            "peak_week": peak,
            "source": "sentinel_ophlgc (KDCA 안과감염증 표본감시)",
            "weekly": weekly,
        })

    finally:
        con.close()

    return {
        "built_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "KDCA 표본감시 sentinel (epi_real_seoul.db)",
        "note": "전국/도시레벨 감시(관측, 예측 아님). 자치구 분리 불가. 주간 연평균 계절성 곡선.",
        "diseases": diseases,
    }


def main() -> int:
    print(f"[build_multi_disease] DB = {DB}")
    if not DB.exists():
        print(f"ERROR: DB not found at {DB}", file=sys.stderr)
        return 1

    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}")
    for d in data["diseases"]:
        defined = sum(1 for w in d["weekly"] if w["value"] is not None)
        print(
            f"  {d['id']:15s}  peak_week={d['peak_week']:2d}  "
            f"weeks_with_data={defined}/53  "
            f"year_range={d['year_range']}  unit={d['unit']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
