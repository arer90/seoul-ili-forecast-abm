#!/usr/bin/env python3
"""다중 수평선 forecast 회귀 테스트 (TDD) — REAL data 산출물.

사용자 요구("+1M, +3M 까지 불안해도 갖추고 싶다")를 박제:
  - +1개월/+3개월 점추정이 **제공**된다(금지 아님).
  - 먼 horizon 은 ML 재귀 외삽이 아니라 climatology-anchored → **폭주 없음**(유한·합리적 상한).
  - ML 가중 w_ml 이 h 에 따라 감쇠(+1주=1.0 → +3개월=0.0).
  - 모든 수평선에 PI(lo≤point≤hi)가 있고 불확실성을 표현.

Run:  .venv/bin/python web/scripts/test_horizon_forecast.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "web" / "public" / "aggregates"


def _load() -> dict:
    return json.loads((AGG / "ili-forecast-horizons.json").read_text(encoding="utf-8"))


def test_plus_1month_to_12month_provided():
    """사용자 요구: +1개월·+3개월·+6개월·+12개월 점추정이 모두 존재한다(금지하지 않음)."""
    labels = {h["label"] for h in _load()["horizons"]}
    for L in ("+1개월", "+3개월", "+6개월", "+12개월"):
        assert L in labels, f"{L} 누락: {labels}"


def test_no_explosion_at_any_horizon():
    """어떤 수평선도 폭주하지 않는다 — 유한·합리적 상한(≤300/1k = 3×역사max 부근)."""
    for h in _load()["horizons"]:
        assert 0.0 <= h["point"] < 300.0, f"{h['label']} point {h['point']} 폭주/비정상"
        assert h["hi"] < 600.0, f"{h['label']} PI 상한 {h['hi']} 폭주"


def test_ml_weight_decays_to_climatology():
    """w_ml 이 단기 1.0 → 장기 0.0 으로 감쇠(먼 horizon = climatology 전담)."""
    hz = {h["label"]: h for h in _load()["horizons"]}
    assert hz["+1주"]["w_ml"] >= 0.99, "단기는 ML 주도여야"
    assert hz["+3개월"]["w_ml"] <= 0.01, "+3개월은 climatology 전담이어야"
    assert hz["+1주"]["w_ml"] > hz["+1개월"]["w_ml"], "w_ml 단조감쇠 위반"


def test_every_horizon_has_uncertainty_band():
    for h in _load()["horizons"]:
        assert h["lo"] <= h["point"] <= h["hi"], f"{h['label']} PI 정렬 위반"
        assert h["hi"] > h["lo"], f"{h['label']} PI 폭 0"


def test_far_horizon_labeled_climatology():
    """먼 horizon 은 method 가 climatology-anchored — 정직 라벨(비정형 시즌엔 빗나감)."""
    hz = {h["label"]: h for h in _load()["horizons"]}
    assert "climatology" in hz["+3개월"]["method"], f"+3개월 method={hz['+3개월']['method']}"


def test_per_horizon_reliability_attached():
    """사용자 '정확도 그대로 보여달라': 각 수평선에 실측 backtest MAE·coverage·reliability 부착."""
    hz = {h["label"]: h for h in _load()["horizons"]}
    for L in ("+1주", "+1개월", "+12개월"):
        h = hz[L]
        assert h.get("backtest_mae") is not None, f"{L} backtest_mae 누락"
        assert h.get("reliability"), f"{L} reliability 라벨 누락"


def test_reliability_honest_short_vs_far():
    """단기는 '신뢰', 먼 horizon 은 '계절평균' — 저하를 숨기지 않고 그대로."""
    hz = {h["label"]: h for h in _load()["horizons"]}
    assert hz["+1주"]["reliability"].startswith("신뢰"), hz["+1주"]["reliability"]
    assert "계절평균" in hz["+12개월"]["reliability"], hz["+12개월"]["reliability"]


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
