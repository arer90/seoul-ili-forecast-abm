"""
build_live_overlays.py — emit a small JSON bundle driving the Map's
"live" overlay switcher (ILI / PM2.5 / Temperature).

Philosophy:
  The demo doesn't need a live feed for every metric. We pull the last
  known ILI rate per gu from the project DB (real observation), and
  synthesise plausible PM2.5 / temperature values keyed deterministically
  off the gu name so the map looks populated during the demo even
  offline. A banner on the overlay picker (``overlayHint`` i18n) flags
  non-ILI metrics as demo-sample in the UI.

Output:
  web/public/aggregates/live-overlays.json
  — {
      "generated_at": "...",
      "metrics": {
          "ili":  { "unit": "per 1k", "rows": [{gu_nm, value}, ...] },
          "air":  { "unit": "µg/m³ PM2.5", "rows": [...] },
          "temp": { "unit": "°C",  "rows": [...] },
      }
    }
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path


SEOUL_GU = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
    "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
    "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
    "서초구", "강남구", "송파구", "강동구",
]


def _determ(gu: str, salt: str, lo: float, hi: float) -> float:
    """
    Deterministic pseudo-random in [lo, hi] keyed by (gu, salt). Keeps
    values stable across reruns for the demo — avoids the choropleth
    jittering each page refresh.
    """
    h = hashlib.md5(f"{gu}|{salt}".encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFF_FFFF
    return round(lo + (hi - lo) * v, 2)


def _load_ili_from_db(db_path: Path) -> dict[str, float]:
    """
    Pull the latest observed flu/ILI rate per gu from the project SQLite.
    Graceful: if the table shape has changed or the DB isn't reachable
    we return {} and the caller falls back to synthesised values.
    """
    if not db_path.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            # Prefer the weekly_disease aggregate if it carries gu + rate.
            cur = con.execute(
                "SELECT gu_nm, rate FROM weekly_disease "
                "WHERE disease_cd='flu' AND gu_nm != '서울시' "
                "ORDER BY date DESC LIMIT 25"
            )
            rows = cur.fetchall()
            if rows:
                return {str(r[0]): float(r[1]) for r in rows}
        except sqlite3.Error:
            pass
        finally:
            con.close()
    except sqlite3.Error:
        pass
    return {}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    db_path = repo_root / "simulation" / "data" / "db" / "epi_real_seoul.db"
    dst = repo_root / "web" / "public" / "aggregates" / "live-overlays.json"

    # ILI: real if DB reachable, else synthesised around a seasonal baseline.
    ili_db = _load_ili_from_db(db_path)
    ili_rows = [
        {
            "gu_nm": gu,
            "value": ili_db.get(gu) if gu in ili_db else _determ(gu, "ili", 2.0, 22.0),
        }
        for gu in SEOUL_GU
    ]

    # PM2.5: Seoul typical spring range ~15-80 µg/m³. Bias northwest gus
    # slightly higher to reflect prevailing wind + Incheon imports.
    northwest = {"은평구", "서대문구", "마포구", "강서구", "양천구", "구로구"}
    air_rows = [
        {
            "gu_nm": gu,
            "value": _determ(gu, "air", 18.0, 48.0)
            + (10.0 if gu in northwest else 0.0),
        }
        for gu in SEOUL_GU
    ]

    # Temperature: April Seoul average ~13°C; tight band 10-18.
    temp_rows = [
        {"gu_nm": gu, "value": _determ(gu, "temp", 10.5, 17.5)} for gu in SEOUL_GU
    ]

    bundle = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": {
            "ili": {
                "label_ko": "ILI 예측",
                "label_en": "ILI forecast",
                "unit": "per 1k",
                "source": "ili_db" if ili_db else "synthetic",
                "rows": ili_rows,
            },
            "air": {
                "label_ko": "PM2.5 (μg/m³)",
                "label_en": "PM2.5 (μg/m³)",
                "unit": "μg/m³",
                "source": "synthetic (demo)",
                "rows": air_rows,
            },
            "temp": {
                "label_ko": "기온 (°C)",
                "label_en": "Temperature (°C)",
                "unit": "°C",
                "source": "synthetic (demo)",
                "rows": temp_rows,
            },
        },
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fp:
        json.dump(bundle, fp, ensure_ascii=False, indent=2)
    print(
        f"[build_live_overlays] wrote {dst} ({dst.stat().st_size:,} bytes), "
        f"ili={bundle['metrics']['ili']['source']}"
    )


if __name__ == "__main__":
    main()
