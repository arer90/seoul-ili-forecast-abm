"""test_overseas_forward.py — calendar-locked forward 헬퍼 회귀 가드 (D-3 TDD).

대상: ``simulation.scripts._overseas_forward`` 의 순수 split/window/metric 헬퍼.
FusedEpi rolling(무거움)은 제외 — 여기선 경계 규칙·공통창·feature 인과만 빠르게 검증.

회귀 case: matched / boundary(02-09=in-sample) / empty / NaN / datetime-coercion / edge.
"""
import datetime as dt

import numpy as np

from simulation.scripts._overseas_forward import (
    build_basic_features,
    common_forward_len,
    get_in_sample_end,
    isoweek_monday,
    r2,
    split_forward_by_dates,
    split_forward_by_isoweek,
    wis,
)

B1 = dt.date(2026, 2, 9)


def test_isoweek_monday_boundary():
    assert isoweek_monday(2026, 7) == dt.date(2026, 2, 9)   # in-sample 마지막
    assert isoweek_monday(2026, 8) == dt.date(2026, 2, 16)  # forward 시작
    assert isoweek_monday(2026, 99) is None                 # 부적합 week


def test_split_isoweek_boundary_in_sample():
    # W07(02-09)은 in-sample, forward 는 W08(02-16) 부터 = contiguous tail.
    yw = [(2026, 5), (2026, 6), (2026, 7), (2026, 8), (2026, 9)]
    assert split_forward_by_isoweek(yw, B1) == 3
    assert split_forward_by_isoweek([(2025, 1), (2025, 2)], B1) == 2   # all in-sample
    assert split_forward_by_isoweek([], B1) == 0                       # empty


def test_split_dates_sunday_anchored():
    # Seoul week_start = 일요일 앵커. 02-08≤02-09=in-sample, 02-15>02-09=forward.
    ds = [dt.date(2026, 2, 1), dt.date(2026, 2, 8), dt.date(2026, 2, 15), dt.date(2026, 2, 22)]
    assert split_forward_by_dates(ds, B1) == 2
    # datetime 도 coercion.
    assert split_forward_by_dates([dt.datetime(2026, 2, 8), dt.datetime(2026, 2, 15)], B1) == 1


def test_common_forward_len():
    assert common_forward_len([18, 14, 17]) == 14          # min
    assert common_forward_len([18, 20, 25], cap=18) == 18  # cap
    assert common_forward_len([0, -3, 17, 14]) == 14       # <=0 제외
    assert common_forward_len([]) == 0                     # empty


def test_in_sample_end_default():
    assert get_in_sample_end() == B1


def test_r2_wis_edge():
    assert abs(r2(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) - 1.0) < 1e-9
    assert np.isnan(r2(np.array([5.0, 5, 5]), np.array([5.0, 5, 5])))  # zero-var
    qd = {0.5: np.array([2.0, 2]), 0.025: np.array([1.0, 1]),
          0.975: np.array([3.0, 3]), 0.25: np.array([1.5, 1.5]),
          0.75: np.array([2.5, 2.5])}
    assert np.isfinite(wis(np.array([2.0, 2]), qd))


def test_basic_features_shape_and_causality():
    y = np.arange(60).astype(float)
    X = build_basic_features(y)
    assert X.shape == (60, 13)
    assert X[5, 0] == y[4]   # lag1[t] = y[t-1] (leakage-free)
