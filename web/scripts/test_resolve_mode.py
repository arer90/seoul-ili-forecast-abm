#!/usr/bin/env python3
"""L1 모드 게이트 — TDD (순수 결정 + hysteresis, 네트워크 무관).

박제(적대검증 규칙): KDCA경계+만 즉시 PANDEMIC, 소프트신호=WATCH 2주진입/4주해제, surge=2차,
평시=SEASONAL. PANDEMIC 비가역 라우팅을 KDCA-only 로 격리(깜빡임 방지).

Run:  .venv/bin/python web/scripts/test_resolve_mode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from resolve_mode import resolve_mode_raw, apply_hysteresis, _detect_surge, routing_manifest  # noqa: E402


import datetime

D0 = datetime.date(2026, 1, 5)          # 가상 기준주
def _wk(n):                              # n주 뒤 날짜
    return D0 + datetime.timedelta(weeks=n)


def _er(kdca=0, novel=False, spike=False, stale=False):
    return {"kdca_alert_level": kdca,
            "summary": {"respiratory_novel_confirmed": novel, "news_spike": spike,
                        "kdca_stale": stale}}


# ── raw 결정 ────────────────────────────────────────────────────────────────
def test_kdca_alert_3_is_pandemic_hard():
    r = resolve_mode_raw(_er(kdca=3), [5, 5, 5, 5])
    assert r["raw_ord"] == 2 and r["kdca_hard"] is True


def test_kdca_below_3_not_pandemic():
    r = resolve_mode_raw(_er(kdca=2, novel=False), [5, 5, 5, 5])
    assert r["raw_ord"] < 2, "주의(2)는 단독으로 PANDEMIC 아님"


def test_soft_signal_is_watch():
    for er in (_er(novel=True), _er(spike=True)):
        assert resolve_mode_raw(er, [5, 5, 5, 5])["raw_ord"] == 1


def test_peacetime_is_seasonal():
    assert resolve_mode_raw(_er(), [8.2, 6.5, 4.9, 5.1])["raw_ord"] == 0


def test_trajectory_surge_is_watch_not_pandemic():
    r = resolve_mode_raw(_er(), [20, 40, 80, 160])     # 지속 급증
    assert r["raw_ord"] == 1, "trajectory surge 는 PANDEMIC 아니라 WATCH(2차)여야"
    assert r["signals"]["surge"] is True


# ── surge 감지 ──────────────────────────────────────────────────────────────
def test_detect_surge_sustained_only():
    assert _detect_surge([20, 40, 80, 160]) is True
    assert _detect_surge([60, 55, 50, 45]) is False          # 하강
    assert _detect_surge([50, 52, 51, 53]) is False          # 미미


# ── stale KDCA → WATCH fail-safe (안전 수정) ─────────────────────────────────
def test_stale_kdca_is_watch_failsafe():
    """권위 신호(KDCA)가 stale 이면 평시로 강등하지 말고 WATCH 로 fail-safe(미탐 방지)."""
    r = resolve_mode_raw(_er(kdca=0, stale=True), [5, 5, 5, 5])
    assert r["raw_ord"] == 1 and r["signals"]["kdca_stale"] is True


# ── hysteresis: 깜빡임 격리 (date-based) ─────────────────────────────────────
def test_watch_needs_two_weeks():
    """SEASONAL→WATCH: 같은 날(0주)엔 전환 안 됨, 1주 경과(2번째 주간관측)에 전환."""
    s0 = {"mode_ord": 0, "pending_ord": 0, "pending_since": _wk(0).isoformat()}
    s1 = apply_hysteresis(s0, 1, False, _wk(0))              # 진입 주(0주 경과)
    assert s1["mode_ord"] == 0, "WATCH 즉시 전환되면 깜빡임"
    s2 = apply_hysteresis(s1, 1, False, _wk(1))             # 1주 경과
    assert s2["mode_ord"] == 1 and s2["switched"] is True


def test_pandemic_via_kdca_is_immediate():
    s = apply_hysteresis({"mode_ord": 0, "pending_ord": 0, "pending_since": _wk(0).isoformat()},
                         2, True, _wk(0))
    assert s["mode_ord"] == 2 and s["switched"] is True


def test_deescalation_needs_four_weeks():
    """WATCH→SEASONAL 하향은 ≥3주 경과(4번째 주간관측)라야."""
    st = {"mode_ord": 1, "pending_ord": 1, "pending_since": _wk(0).isoformat()}
    for wk in (0, 1, 2):                                     # 0·1·2주 경과 = 아직 유지
        st = apply_hysteresis(st, 0, False, _wk(wk))
        assert st["mode_ord"] == 1, f"{wk}주 경과에 조기 해제"
    st = apply_hysteresis(st, 0, False, _wk(3))             # 3주 경과
    assert st["mode_ord"] == 0 and st["switched"] is True


def test_hysteresis_date_based_not_invocation():
    """안전 핵심: daily 로 7번 호출해도 같은 주면 전환 안 됨(호출당 증가 아님)."""
    st = {"mode_ord": 0, "pending_ord": 0, "pending_since": D0.isoformat()}
    for day in range(7):                                     # 같은 주 안에서 매일 호출
        st = apply_hysteresis(st, 1, False, D0 + datetime.timedelta(days=day))
    assert st["mode_ord"] == 0, "daily 호출이 '2주'를 며칠로 붕괴시킴(버그 재발)"
    st = apply_hysteresis(st, 1, False, D0 + datetime.timedelta(days=7))   # 1주 경과
    assert st["mode_ord"] == 1, "1주 경과 후엔 정상 전환되어야"


# ── 라우팅 매니페스트 (L2 dead-end 해소) ────────────────────────────────────
def test_routing_seasonal_is_ml_primary():
    m = routing_manifest(0)
    assert m["primary"] == "ili-forecast.json" and m["alert_level"] == "none"


def test_routing_pandemic_is_mechanistic_primary():
    """PANDEMIC 은 기계론(SEIR)이 primary — ML nowcast 외삽 불가."""
    m = routing_manifest(2)
    assert m["primary"] == "seir-forecast-360.json" and m["alert_level"] == "pandemic"
    assert m["fallback"] == "ili-forecast.json"


def test_routing_watch_keeps_ml_adds_mechanistic():
    m = routing_manifest(1)
    assert m["primary"] == "ili-forecast.json" and m["secondary"] == "seir-forecast-360.json"
    assert m["alert_level"] == "watch"


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
