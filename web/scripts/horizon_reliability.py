#!/usr/bin/env python3
"""수평선별 실측 신뢰도 — 롤링 백테스트로 "각 horizon 이 실제 얼마나 정확한가" 정직 측정.

사용자: "예측은 +12M까지 다 나오되, +3M까지는 정확하게, 그 이후 불안정/부정확은 그대로 보여달라."

build_horizon_forecast 의 블렌드(단기 ML → 장기 climatology)를 **여러 origin 에서 롤링**으로
재현해 각 horizon 의 실측 MAE·편향·PI coverage 를 계산 → horizon-reliability.json.
build_horizon_forecast 가 이를 읽어 각 수평선에 expected_error·reliability 를 붙이고, web 이
표시한다. → +3M까지 신뢰, 이후 저하가 숫자로 그대로 드러남(숨기지 않음).

Read-only. Run: .venv/bin/python web/scripts/horizon_reliability.py
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "web" / "scripts"))
logging.disable(logging.INFO)
from build_production_forecast import _load_feature_matrix, _extract_basic_features, _gate_forecast, _surge_aware_bound, _conformal_half_width  # noqa: E402
from build_horizon_forecast import HORIZONS, _w_ml, _d  # noqa: E402
from simulation.models.epi_models import NegBinGLMForecaster  # noqa: E402


def main() -> None:
    X, y, fc, ws = _load_feature_matrix()
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xb, _bc, _bi = _extract_basic_features(X, fc)
    dates = [_d(w) for w in ws]; n = len(y)
    base_q = _conformal_half_width(alpha=0.05)

    # 롤링 origin: 충분한 학습(≥150) 이후 8주 간격
    origins = [i for i in range(150, n) if (i - 150) % 8 == 0]
    print(f"=== 수평선별 실측 신뢰도 (롤링 origin {len(origins)}개, 8주 간격) ===\n")
    print(f"  {'수평선':<12}{'n':>4}{'MAE':>8}{'편향':>8}{'PI95 cov':>10}  신뢰도")
    print(f"  {'-'*52}")

    out = {}
    for h, label in HORIZONS:
        errs, covs = [], []
        w = _w_ml(h)
        for o in origins:
            tgt = o + h
            if tgt >= n:
                continue
            wk = dates[tgt].isocalendar()[1]
            # climatology from ≤o
            cv = [y[i] for i in range(o + 1) if dates[i].isocalendar()[1] == wk]
            clim = float(np.mean(cv)) if cv else float(np.mean(y[:o + 1]))
            csd = float(np.std(cv)) if len(cv) > 1 else float(np.std(y[:o + 1]))
            if w > 0:
                idx = [i for i in range(o + 1 - h) if i + h <= o]
                if len(idx) < 30:
                    continue
                m = NegBinGLMForecaster(topk=20); m.fit(Xb[idx], y[[i + h for i in idx]])
                raw = np.asarray(m.predict(Xb[o:o + 1]), float)
                ml = float(_gate_forecast(raw, y[:o + 1], fallback=float(y[o]), k=3.0)["pred"][0])
                ml, _sd, _r = _surge_aware_bound(ml, y[max(0, o - 3):o + 1], float(np.nanmax(y[:o + 1])))
            else:
                ml = clim
            point = w * ml + (1 - w) * clim
            pi = w * (base_q * np.sqrt(h)) + (1 - w) * (1.96 * csd)
            real = y[tgt]
            errs.append(point - real)
            covs.append(1 if (point - pi) <= real <= (point + pi) else 0)
        if len(errs) < 4:
            print(f"  {label:<12}{len(errs):>4}  (불충분)")
            continue
        e = np.array(errs)
        mae = float(np.mean(np.abs(e))); bias = float(np.mean(e)); cov = float(np.mean(covs))
        # 신뢰도 = PI 보정도(coverage) 기준 — 사용자 "정확도 그대로 보여달라". 먼 horizon 은
        # 계절평균이라 명시.
        if w <= 0:
            rel = "계절평균(비정형 못잡음)"
        else:
            rel = "신뢰" if cov >= 0.80 else "보통" if cov >= 0.60 else "낮음"
        print(f"  {label:<12}{len(errs):>4}{mae:>8.2f}{bias:>+8.2f}{cov*100:>9.0f}%  {rel}")
        out[label] = {"horizon_weeks": h, "n": len(errs), "mae": round(mae, 2),
                      "bias": round(bias, 2), "pi95_coverage": round(cov, 2), "reliability": rel}

    (ROOT / "web" / "public" / "aggregates" / "horizon-reliability.json").write_text(
        json.dumps({"note": "롤링 백테스트 수평선별 실측 정확도. +3M까지 신뢰·이후 저하를 그대로 표시.",
                    "horizons": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → wrote horizon-reliability.json (build_horizon_forecast 가 attach)")


if __name__ == "__main__":
    main()
