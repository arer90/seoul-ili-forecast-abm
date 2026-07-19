#!/usr/bin/env python3
"""다중 수평선 forecast (+1주 … +3개월) — 단기 ML nowcast → 장기 계절 climatology 블렌드.

사용자: "+1M, +3M 까지 어느 정도 불안해도 갖출 수 있기를 원해."

먼 horizon 을 **금지하지 않되**(점추정 제공), ML 재귀 외삽(폭주: +12개월 err +53, 합성 surge
수백만)은 쓰지 않는다. 대신:
  - 단기(≤3–4주): direct ML nowcast (in-range, 신뢰 MAE 9–12) — 가중 w_ml(h).
  - 장기(+1M, +3M): 계절 climatology (target ISO주 과거 평균 ILI) — 안정·해석가능 anchor.
  - 블렌드: point = w_ml·ml + (1−w_ml)·clim,  PI 는 h 에 따라 넓어짐(불확실성 정직 표현).
  - ML 성분엔 surge-gate(거짓폭주 가드) 적용. → far-horizon 도 '불안하지만' 폭주 없이 제공.

정직성: +1M/+3M 는 climatology-anchored 라 '이번 시즌이 평년과 다르면' 빗나감. 그래서 넓은 PI +
method 라벨('climatology-anchored')을 단다. novel-pathogen 국면은 별도 mode(기계론)로 라우팅.

Read-only on data; writes web/public/aggregates/ili-forecast-horizons.json.
Run: .venv/bin/python web/scripts/build_horizon_forecast.py
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))
logging.disable(logging.INFO)

from build_production_forecast import (  # noqa: E402
    _load_feature_matrix, _extract_basic_features, _gate_forecast, _surge_aware_bound,
    _conformal_half_width,
)
from simulation.models.epi_models import NegBinGLMForecaster  # noqa: E402

HORIZONS = [(1, "+1주"), (2, "+2주"), (3, "+3주"), (4, "+1개월"), (8, "+2개월"),
            (13, "+3개월"), (26, "+6개월"), (52, "+12개월")]


def _w_ml(h: int) -> float:
    """ML 가중: +3M(h=13)까지 ML 기여 후 0 (사용자: +3M 까지 정확도). h=1→1.0, h=4→0.75,
    h=8→0.42, h=13→0.0. 장기(+6M↑)는 climatology 전담. (검증: horizon_reliability 로 +1M~+3M
    coverage 가 4주-fade 보다 개선됨 확인.)"""
    return float(max(0.0, 1.0 - (h - 1) / 12.0))


def _d(w) -> datetime.date:
    if hasattr(w, "date"):
        return w.date()
    return w if isinstance(w, datetime.date) else datetime.date.fromisoformat(str(w)[:10])


def main() -> None:
    X_all, y_all, fcols, ws = _load_feature_matrix()
    X_all = np.asarray(X_all, float); y_all = np.asarray(y_all, float)
    Xb, _bc, _bi = _extract_basic_features(X_all, fcols)
    dates = [_d(w) for w in ws]
    n = len(y_all)
    last = n - 1
    last_date = dates[last]
    train_max = float(np.nanmax(y_all))

    # 계절 climatology: ISO주별 평균·표준편차 (전체 history)
    by_week: dict[int, list[float]] = {}
    for i in range(n):
        by_week.setdefault(dates[i].isocalendar()[1], []).append(y_all[i])
    clim_mean = {k: float(np.mean(v)) for k, v in by_week.items()}
    clim_std = {k: float(np.std(v)) for k, v in by_week.items()}
    grand = float(np.mean(y_all))

    base_q = _conformal_half_width(alpha=0.05)        # 단기 PI half-width (test OOS 잔차)

    # 수평선별 실측 신뢰도(롤링 백테스트, horizon_reliability.py) — 있으면 attach
    rel_path = AGG / "horizon-reliability.json" if (AGG := ROOT / "web" / "public" / "aggregates") else None
    reliability = {}
    if rel_path and rel_path.is_file():
        try:
            reliability = json.loads(rel_path.read_text(encoding="utf-8")).get("horizons", {})
        except Exception:
            reliability = {}

    print(f"=== 다중 수평선 forecast (origin {last_date}, ILI {y_all[last]:.1f}/1k) ===")
    print(f"  단기=ML nowcast · 장기=계절 climatology 블렌드 · PI 는 h 에 따라 확장\n")
    print(f"  {'수평선':<10}{'목표일':<13}{'예측':>7}{'95% PI':>16}{'w_ml':>6}  방법")
    print(f"  {'-'*62}")

    out_h = []
    for h, label in HORIZONS:
        tdate = last_date + datetime.timedelta(weeks=h)
        wk = tdate.isocalendar()[1]
        clim = clim_mean.get(wk, grand)
        csd = clim_std.get(wk, float(np.std(y_all)))
        w = _w_ml(h)

        ml = clim
        if w > 0:
            # direct ML: (Xb[i] → y[i+h]) 전체 학습 → Xb[last] 적용 (재귀 아님 = 폭주 없음)
            idx = [i for i in range(n - h)]
            model = NegBinGLMForecaster(topk=20)
            model.fit(Xb[idx], y_all[[i + h for i in idx]])
            raw = float(model.predict(Xb[last:last + 1])[0])
            g = _gate_forecast(np.array([raw]), y_all, fallback=float(y_all[last]), k=3.0)
            ml = float(g["pred"][0])
            ml, _surge, _r = _surge_aware_bound(ml, y_all[-4:], train_max)

        point = w * ml + (1.0 - w) * clim
        # PI: 단기 conformal(√h 확장) ↔ 장기 climatology 1.96σ 블렌드
        pi_half = w * (base_q * np.sqrt(h)) + (1.0 - w) * (1.96 * csd)
        lo = max(0.0, point - pi_half); hi = point + pi_half
        method = ("ML nowcast" if w >= 0.75 else
                  "ML+climatology" if w > 0 else "climatology-anchored")
        rel = reliability.get(label, {})
        print(f"  {label:<10}{str(tdate):<13}{point:>7.1f}{f'[{lo:.1f}, {hi:.1f}]':>16}{w:>6.2f}  {method}"
              + (f"  실측MAE {rel['mae']} · cov {int(rel['pi95_coverage']*100)}% · {rel['reliability']}" if rel else ""))
        out_h.append(dict(horizon_weeks=h, label=label, target=str(tdate),
                          point=round(point, 1), lo=round(lo, 1), hi=round(hi, 1),
                          w_ml=round(w, 2), method=method,
                          climatology=round(clim, 1),
                          backtest_mae=rel.get("mae"), pi95_coverage=rel.get("pi95_coverage"),
                          reliability=rel.get("reliability")))

    print(f"\n  정직성: +1개월 이후는 climatology-anchored — 이번 시즌이 평년과 다르면 빗나감(넓은 PI).")
    print(f"          novel-pathogen 국면은 이 경로 아님 → 기계론 엔진 mode 라우팅(별도).")

    payload = {
        "generated_from": str(last_date), "origin_ili": round(float(y_all[last]), 1),
        "method": "short-horizon ML nowcast → long-horizon seasonal climatology blend; "
                  "far-horizon은 climatology-anchored (ML 재귀 외삽 폭주 회피), PI 는 h 에 따라 확장.",
        "honesty": "+1개월 이후 점추정은 계절 평균 기반이라 비정형 시즌엔 빗나감 — 넓은 PI 로 표현. "
                   "대규모 novel-pathogen 은 mode 게이트로 기계론 엔진 라우팅.",
        "horizons": out_h,
    }
    (ROOT / "web" / "public" / "aggregates" / "ili-forecast-horizons.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → wrote web/public/aggregates/ili-forecast-horizons.json")


if __name__ == "__main__":
    main()
