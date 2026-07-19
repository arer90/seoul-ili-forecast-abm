#!/usr/bin/env python3
"""L0 외부 위험신호 수집 — TDD (순수 스코어링 함수, 네트워크 무관).

박제: DON 호흡기+novel 동시매칭=확증 / 계절독감≠확증, GDELT spike 임계, KDCA 기본 평시(0),
external-risk.json 스키마. 신규 기능 strict TDD.

Run:  .venv/bin/python web/scripts/test_external_risk.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_external_risk import _score_don_items, _gdelt_z, read_kdca_alert  # noqa: E402


# ── DON 스코어링: 호흡기 novel 확증 ──────────────────────────────────────────
def test_don_respiratory_novel_confirmed():
    """호흡기 키워드 AND novel 키워드 동시 항목 → respiratory_novel_confirmed=True."""
    items = [{"title": "Novel avian influenza A(H5N1) human cases — pandemic potential", "date": "x"}]
    sc = _score_don_items(items)
    assert sc["respiratory_novel_confirmed"] is True
    assert sc["respiratory_count"] >= 1 and sc["novel_count"] >= 1


def test_don_seasonal_flu_not_confirmed():
    """일반 계절 사건(호흡기지만 novel 아님, 또는 비호흡기)은 확증 아님 — 오탐 방지."""
    items = [{"title": "Seasonal influenza activity update, Europe", "date": "x"},
             {"title": "Ebola disease, Democratic Republic of the Congo", "date": "x"}]
    sc = _score_don_items(items)
    assert sc["respiratory_novel_confirmed"] is False, "계절독감/비호흡기가 novel 확증되면 오탐"


# ── GDELT spike ─────────────────────────────────────────────────────────────
def test_gdelt_spike_on_surge():
    """평탄하다 급등하는 timeline → spike=True (z≥2)."""
    pts = [{"date": str(i), "value": 0.001} for i in range(7)] + [{"date": "8", "value": 0.05}]
    z = _gdelt_z(pts)
    assert z["spike"] is True and z["z"] >= 2.0


def test_gdelt_no_spike_when_flat():
    pts = [{"date": str(i), "value": 0.001} for i in range(8)]
    z = _gdelt_z(pts)
    assert z["spike"] is False


def test_gdelt_insufficient_points():
    z = _gdelt_z([{"date": "1", "value": 0.001}, {"date": "2", "value": 0.002}])
    assert z["spike"] is False and z["z"] == 0.0


# ── KDCA 기본 평시 ──────────────────────────────────────────────────────────
def test_kdca_default_peacetime():
    """kdca-alert.json 미존재 시 평시 0 (없으면 절대 PANDEMIC 트리거 안 됨)."""
    k = read_kdca_alert()
    assert 0 <= k["level"] <= 4
    # 기본(파일 없음)이면 0
    if k["source"] == "default":
        assert k["level"] == 0


# ── external-risk.json 스키마 ───────────────────────────────────────────────
def test_external_risk_json_schema():
    f = ROOT / "web" / "public" / "aggregates" / "external-risk.json"
    assert f.is_file(), "external-risk.json 미생성 — build_external_risk.py 먼저 실행"
    d = json.loads(f.read_text(encoding="utf-8"))
    for k in ("kdca_alert_level", "don", "gdelt", "summary", "generated_at"):
        assert k in d, f"external-risk.json 에 {k} 누락"
    assert 0 <= d["kdca_alert_level"] <= 4
    for k in ("respiratory_novel_confirmed", "news_spike", "any_source_error"):
        assert k in d["summary"]


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
