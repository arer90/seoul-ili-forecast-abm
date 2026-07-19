#!/usr/bin/env python3
"""Add real per-gu spatial heterogeneity to seir-forecast-360.json.

WHY
---
The base ``seir-forecast-360.json`` (seasonal SEIR forecast) is **spatially
uniform** — every one of the 25 gu carries the Seoul-wide ``city_ili`` value,
so the 3D extrusion columns are all the same height (the user's "공간만, 변화·
모양 없음" complaint applied to space). The 2D map does NOT have this problem
because ``iliFor()`` (app.jsx) modulates the city curve by **real per-gu
daytime population density** (``data.json.density`` = daily_population_gu_hourly).

This post-processor reuses the SAME density weighting so the 3D forecast
matches the 2D map's spatial pattern (dense gu — 강남/종로/중구 — peak higher),
plus a density-driven peak-time shift so the wave PROPAGATES (dense, high-mixing
gu peak slightly earlier → spreads outward) rather than every gu moving in
lockstep.

HONESTY (label baked into ``note``):
  - per-gu ILI is NOT real surveillance (KDCA sentinel ILI is CITY-LEVEL, one
    number for Seoul). This is a **model disaggregation**: city ILI × real
    per-gu daytime population density. Same assumption the 2D map already makes.

Idempotent: always derives per-gu from ``city_ili`` (never from a previous
per-gu pass), so re-running gives the same result.

Usage:
    .venv/bin/python web/scripts/spatialize_seir_forecast.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # web/scripts → web → repo root
FC = ROOT / "web" / "public" / "aggregates" / "seir-forecast-360.json"
DATA = ROOT / "web_prototype" / "data.json"

# Standard 25-gu romanization → Korean (matches data.json density keys → forecast gu keys).
ENG2KOR = {
    "jongno": "종로구", "jung": "중구", "yongsan": "용산구", "seongdong": "성동구",
    "gwangjin": "광진구", "dongdaemun": "동대문구", "jungnang": "중랑구", "seongbuk": "성북구",
    "gangbuk": "강북구", "dobong": "도봉구", "nowon": "노원구", "eunpyeong": "은평구",
    "seodaemun": "서대문구", "mapo": "마포구", "yangcheon": "양천구", "gangseo": "강서구",
    "guro": "구로구", "geumcheon": "금천구", "yeongdeungpo": "영등포구", "dongjak": "동작구",
    "gwanak": "관악구", "seocho": "서초구", "gangnam": "강남구", "songpa": "송파구",
    "gangdong": "강동구",
}

# iliFor() amplitude band (app.jsx:357) — keep identical so 2D/3D agree.
AMP_LO, AMP_HI = 0.75, 1.25
SHIFT_MAX = 7  # ± days; dense gu peak earlier (more mixing) → wave propagates


def main() -> int:
    fc = json.loads(FC.read_text(encoding="utf-8"))
    data = json.loads(DATA.read_text(encoding="utf-8"))
    dens = data.get("density", {})

    # Mean peak daytime population per gu (stable, date-averaged).
    mean_peak: dict[str, float] = {}
    for eng, by_date in dens.items():
        peaks = [
            v["peak_pop"]
            for v in by_date.values()
            if isinstance(v, dict) and v.get("peak_pop")
        ]
        if peaks:
            mean_peak[eng] = sum(peaks) / len(peaks)
    if not mean_peak:
        print("✗ density 비어있음 — 중단")
        return 1
    max_peak = max(mean_peak.values())

    # Per-gu (Korean-name keyed) amplitude + peak shift.
    amp_kor: dict[str, float] = {}
    shift_kor: dict[str, int] = {}
    unmapped = []
    for eng, mp in mean_peak.items():
        kor = ENG2KOR.get(eng)
        if not kor:
            unmapped.append(eng)
            continue
        norm = mp / max_peak  # 0..1
        amp_kor[kor] = AMP_LO + norm * (AMP_HI - AMP_LO)        # dense → higher
        shift_kor[kor] = round((1.0 - norm) * 2 * SHIFT_MAX - SHIFT_MAX)  # dense → earlier (−), sparse → later (+)
    if unmapped:
        print(f"  ⚠ 매핑 안 된 density 키: {unmapped}")

    forecast = fc["forecast"]
    city = [row["city_ili"] for row in forecast]
    n = len(city)

    for di, row in enumerate(forecast):
        for kor in list(row["gu"].keys()):
            a = amp_kor.get(kor, 1.0)
            s = shift_kor.get(kor, 0)
            src = min(max(di - s, 0), n - 1)
            row["gu"][kor] = round(city[src] * a, 4)

    # Honest provenance note.
    base_note = fc.get("note", "")
    tag = ("per-gu = city_ili × 실제 주간인구밀도 가중(0.75–1.25, iliFor와 동일) "
           "+ 밀도기반 피크시프트(±7d, 밀집구 먼저). "
           "주의: 실측 per-gu ILI 아님 — KDCA sentinel=도시레벨 1값, 모델 분해.")
    fc["note"] = (base_note + " | " + tag) if base_note and tag not in base_note else (base_note or tag)
    fc["spatialized"] = True

    FC.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")

    # Report resulting spread at the peak day.
    peak_di = max(range(n), key=lambda i: city[i])
    vals = list(forecast[peak_di]["gu"].values())
    lo, hi = min(vals), max(vals)
    top = sorted(forecast[peak_di]["gu"].items(), key=lambda kv: -kv[1])[:3]
    bot = sorted(forecast[peak_di]["gu"].items(), key=lambda kv: kv[1])[:3]
    print(f"✓ spatialized — peak day {peak_di} ({forecast[peak_di]['date']})")
    print(f"  구별 spread: {lo:.2f} ~ {hi:.2f} ({hi/max(lo,0.01):.2f}x, 이전=1.00x 균일)")
    print(f"  최고: {top}")
    print(f"  최저: {bot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
