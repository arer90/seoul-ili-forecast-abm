#!/usr/bin/env python3
"""PI(예측구간) 보정법 비교 — additive(현행) vs relative(예측수준 비례) vs per-regime.

사용자: "PI coverage(84%→95%)를 TDD로 증명해봐." 누설 없는 롤링-origin 백테스트로 어느 방법이
목표 0.95 에 가까운지 실측. 비싼 부분(롤링 1-step 예측쌍)은 1회 계산→pi-pairs.json 캐시,
순수 band 함수(additive/relative/regime)는 test 에서 그 위로 증명(빠름·결정적).

누설 차단: 각 test origin 의 PI 는 **그 이전 CAL 주 잔차만**으로 보정(미래 정보 0).

Read-only(write pi-pairs.json, pi-calibration.json). Run: .venv/bin/python web/scripts/pi_calibration_compare.py
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "web" / "public" / "aggregates"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "web" / "scripts"))


def _q(a, p):
    a = sorted(a)
    if not a:
        return 0.0
    i = p * (len(a) - 1); lo = int(i); f = i - lo
    return a[lo] * (1 - f) + a[min(lo + 1, len(a) - 1)] * f


# ── 순수 band 함수 (cal = [(pred,actual)…] 과거쌍, p = 이번 예측) → (lo,hi) ──────
def band_additive(cal, pred, alpha=0.05):
    """절대 잔차 분위수 — 현행 production(_conformal_half_width) 방식."""
    res = [abs(a - p) for p, a in cal]
    q = _q(res, 1 - alpha)
    return max(0.0, pred - q), pred + q


def band_relative(cal, pred, alpha=0.05):
    """상대 잔차((actual−pred)/pred) 분위수 — 예측수준 비례(heteroscedastic)."""
    rel = [(a - p) / p for p, a in cal if p > 0.5]
    ql, qh = _q(rel, alpha / 2), _q(rel, 1 - alpha / 2)
    return max(0.0, pred * (1 + ql)), pred * (1 + qh)


def band_regime(cal, pred, alpha=0.05, thr=20.0):
    """저/고 수준 분리 절대 분위수 (peak 분산↑ 반영)."""
    side = [(p, a) for p, a in cal if (p >= thr) == (pred >= thr)]
    if len(side) < 8:
        side = cal
    res = [abs(a - p) for p, a in side]
    q = _q(res, 1 - alpha)
    return max(0.0, pred - q), pred + q


METHODS = {"additive(현행)": band_additive, "relative": band_relative, "per-regime": band_regime}


def evaluate(pairs: list[tuple[float, float]], cal_w: int = 26, alpha: float = 0.05) -> dict:
    """누설 없는 롤링 coverage: origin i 의 PI 는 [i-cal_w … i-1] 잔차로만 보정."""
    out = {}
    for name, fn in METHODS.items():
        cov, width = [], []
        for i in range(cal_w, len(pairs)):
            cal = pairs[i - cal_w:i]
            pred, actual = pairs[i]
            lo, hi = fn(cal, pred, alpha)
            cov.append(1 if lo <= actual <= hi else 0)
            width.append(hi - lo)
        out[name] = {"n": len(cov), "coverage": round(float(np.mean(cov)), 3),
                     "mean_width": round(float(np.mean(width)), 2)}
    return out


def main() -> None:
    logging.disable(logging.INFO)
    from accuracy_calibration import _onestep
    from build_production_forecast import _load_feature_matrix, _extract_basic_features

    X, y, fc, ws = _load_feature_matrix()
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xb, _bc, _bi = _extract_basic_features(X, fc)
    n = len(y)
    # 롤링 1-step 예측쌍 (최근 ~130주 = 2+겨울; 비싼 부분 1회)
    start = max(60, n - 130)
    pairs = [(_onestep(Xb, y, t), float(y[t])) for t in range(start, n)]
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "pi-pairs.json").write_text(json.dumps(
        {"note": "롤링 1-step (pred,actual) 쌍 — PI 보정 비교용", "n": len(pairs),
         "pairs": [[round(p, 3), round(a, 3)] for p, a in pairs]}, ensure_ascii=False), encoding="utf-8")

    res = evaluate(pairs)
    print(f"=== PI 보정법 비교 (누설없는 롤링 {len(pairs)}주, 목표 coverage 0.95) ===\n")
    print(f"  {'방법':<16}{'coverage':>10}{'평균폭':>10}  vs 목표")
    print(f"  {'-'*44}")
    for name, r in res.items():
        gap = abs(r["coverage"] - 0.95)
        print(f"  {name:<16}{r['coverage']:>10.3f}{r['mean_width']:>10.2f}  {'★' if gap < 0.05 else ''} (gap {gap:.3f})")
    best = min(res.items(), key=lambda kv: abs(kv[1]["coverage"] - 0.95))
    print(f"\n  → 목표 0.95 최근접 = {best[0]} (coverage {best[1]['coverage']})")
    (AGG / "pi-calibration.json").write_text(
        json.dumps({"target": 0.95, "cal_window": 26, "methods": res, "best": best[0]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → wrote pi-pairs.json + pi-calibration.json")


if __name__ == "__main__":
    main()
