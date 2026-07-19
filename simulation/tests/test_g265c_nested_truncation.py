"""G-265c (2026-06-13): nested blocked-CV enabler — MPH_DATA_END_WEEK 시간순 앞 N주 절단.

data.py run_data 수렴점의 절단 로직(argsort(dates)[:N])을 격리 재현해 회귀 가드:
  ① 정확히 N행  ② 시간 오름차순  ③ 잘린 부분 전부 미래(forward hold-out)  ④ 미설정/N이상 = no-op.
run_data 전체는 config+DB 필요라 통합경로 대신 절단 불변식을 직접 검증(bug-fix smoke, D-3).

Run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest simulation/tests/test_g265c_nested_truncation.py -x -q
"""
from __future__ import annotations

import numpy as np


def _truncate(X, y, dates, de: str):
    """data.py:210 수렴점 절단과 동일 — env 문자열 de 로 시간순 앞 N주 슬라이스."""
    de = (de or "").strip()
    if de.isdigit() and 0 < int(de) < len(y):
        n = int(de)
        if dates is not None and len(dates) == len(y):
            keep = np.argsort(dates)[:n]
            return X[keep], y[keep], dates[keep]
        return X[:n], y[:n], dates
    return X, y, dates


def _toy(n=351):
    dates = np.arange("2019-01", "2019-01", dtype="datetime64[W]")  # placeholder
    dates = np.array([np.datetime64("2019-01-06") + np.timedelta64(7 * i, "D") for i in range(n)])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)                      # 일부러 셔플 → argsort 가 시간순 복원하는지 검증
    return rng.standard_normal((n, 5))[perm], rng.standard_normal(n)[perm], dates[perm]


def test_truncate_exact_rows_and_time_order():
    X, y, d = _toy(351)
    X2, y2, d2 = _truncate(X, y, d, "231")
    assert len(y2) == 231
    assert bool((d2[1:] >= d2[:-1]).all()), "절단 결과가 시간 오름차순 아님"


def test_dropped_all_future():
    X, y, d = _toy(351)
    _, _, d2 = _truncate(X, y, d, "231")
    dropped = np.setdiff1d(d, d2)
    assert dropped.min() > d2.max(), "잘린 주가 유지된 주보다 과거 — forward hold-out 깨짐"


def test_noop_when_unset_or_full():
    X, y, d = _toy(100)
    for de in ("", "  ", "0", "100", "200", "abc"):
        X2, y2, d2 = _truncate(X, y, d, de)
        assert len(y2) == 100, f"de={de!r} 가 no-op 이어야 하는데 절단됨"


def test_idempotent_on_sorted():
    """이미 시간정렬된 입력이면 argsort 절단 == head[:n]."""
    n = 200
    d = np.array([np.datetime64("2019-01-06") + np.timedelta64(7 * i, "D") for i in range(n)])
    X = np.arange(n * 3).reshape(n, 3).astype(float)
    y = np.arange(n).astype(float)
    _, y2, d2 = _truncate(X, y, d, "120")
    assert np.array_equal(y2, y[:120]) and np.array_equal(d2, d[:120])


if __name__ == "__main__":
    test_truncate_exact_rows_and_time_order(); print("PASS rows+order")
    test_dropped_all_future(); print("PASS dropped future")
    test_noop_when_unset_or_full(); print("PASS no-op")
    test_idempotent_on_sorted(); print("PASS idempotent")
    print("=== ALL PASS ===")
