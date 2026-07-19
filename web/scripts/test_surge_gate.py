#!/usr/bin/env python3
"""Surge-aware gate + log1p/expm1 구조적 한계 — TDD (REAL data + 합성 surge).

사용자 통찰("alpha 하한이 팬데믹을 죽이지 않나")로 발견한 증거를 박제:
  - alpha 튜닝은 log1p/expm1 폭주를 못 고침 (어느 alpha 든 겨울피크 3~4× 폭주) → 구조적.
  - 따라서 fix 는 alpha 가 아니라 궤적-상대 gate: 단주 거짓폭주(201)는 clamp, 지속 surge 는
    허용 + surge_detected → 기계론 엔진 deferral.
  - ML 은 학습범위 밖 대규모 surge 를 외삽 불가(폭주/포화) → 팬데믹은 기계론 엔진 몫.

Run:  .venv/bin/python web/scripts/test_surge_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_production_forecast import _surge_aware_bound, _load_feature_matrix, _extract_basic_features  # noqa: E402
from pandemic_vs_explosion_sweep import fit_v6  # noqa: E402


# ── 궤적-상대 gate: 거짓폭주 clamp ───────────────────────────────────────────
def test_false_explosion_clamped():
    """평지/하강(plateau)에서 모델이 201 로 점프 → 단주 폭주로 clamp(≪201), surge=False."""
    gated, surge, _ = _surge_aware_bound(201.4, [72, 81, 81, 81], train_max=100.0)
    assert gated < 130, f"plateau 에서 201 가 {gated} 로만 clamp (≪201 기대)"
    assert surge is False, "plateau 는 surge 아님"


# ── 궤적-상대 gate: 진짜 surge 는 허용 + 플래그 ───────────────────────────────
def test_sustained_surge_allowed_and_flagged():
    """3주 지속 2배 성장(팬데믹) → 거짓폭주 bound 보다 크게 허용 + surge_detected=True."""
    gated, surge, reason = _surge_aware_bound(480.0, [20, 40, 80, 160], train_max=100.0)
    assert gated > 130, f"지속 surge 인데 {gated} 로 과도 clamp(거짓폭주 가드에 걸림)"
    assert surge is True, "지속 surge 가 감지 안 됨"
    assert "DEFER" in reason or "mechanistic" in reason, "기계론 deferral 신호 누락"


# ── 평시(여름) forecast 는 영향 없음 ─────────────────────────────────────────
def test_summer_forecast_unchanged():
    gated, surge, _ = _surge_aware_bound(7.8, [8.2, 6.5, 4.9, 5.1], train_max=100.0)
    assert abs(gated - 7.8) < 0.01, f"여름 저ILI forecast 가 {gated} 로 변형됨"
    assert surge is False


# ── alpha 는 폭주를 못 고친다(구조적) — 사용자 통찰 박제 ──────────────────────
def test_alpha_does_not_fix_explosion():
    """log1p/expm1: 어느 alpha 든 겨울피크에서 예측이 real 대비 폭주(>2×) → 구조적, alpha 무관."""
    X_all, y_all, fc, _ = _load_feature_matrix()
    X_all = np.asarray(X_all, float); y_all = np.asarray(y_all, float)
    Xb, _bc, _ = _extract_basic_features(X_all, fc)
    peak = np.where(y_all >= 60)[0]
    for alpha in (0.001, 100.0):                 # 양 극단
        pred, _ = fit_v6(Xb, y_all, alpha)
        pk = pred(Xb[peak])                      # raw (cap 없이)
        max_over = float(np.max(pk / np.maximum(y_all[peak], 1)))
        assert max_over > 2.0, (
            f"alpha={alpha}: 피크 과대배율 {max_over:.1f}× — alpha 가 폭주를 고쳤다면 이 가정이 "
            "틀린 것이니 docs/REALTIME 및 sweep 재확인")


# ── ML 은 대규모 surge 를 외삽 못 한다(포화/폭주) → 팬데믹=기계론 ──────────────
def test_ml_cannot_extrapolate_large_surge():
    """합성 surge(lag 매우 큼)에서 안정 alpha 예측이 학습 max 대비 비현실(포화 or 폭주)."""
    X_all, y_all, fc, _ = _load_feature_matrix()
    X_all = np.asarray(X_all, float); y_all = np.asarray(y_all, float)
    Xb, bc, _ = _extract_basic_features(X_all, fc)
    i1, i2, i4 = bc.index("ili_rate_lag1"), bc.index("ili_rate_lag2"), bc.index("ili_rate_lag4")
    templ = Xb[int(np.argmax(y_all))].copy()
    templ[i1], templ[i2], templ[i4] = 320.0, 192.0, 80.0   # 대규모 surge 입력
    pred, _ = fit_v6(Xb, y_all, 3.0)             # 안정 alpha
    p = float(pred(templ.reshape(1, -1))[0])
    ymax = float(np.max(y_all))
    assert p > 5 * ymax or p < 2 * ymax, (
        f"surge 예측 {p:.0f} 이 학습 max {ymax:.0f} 의 2~5× 사이 = 신뢰가능 외삽이면 가정 재검토. "
        "실제로는 폭주(≫5×) 또는 포화(<2×) 라 ML 은 팬데믹 외삽 불가")


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn()
            print(f"  ✓ PASS  {fn.__name__}")
            p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}")
            f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
