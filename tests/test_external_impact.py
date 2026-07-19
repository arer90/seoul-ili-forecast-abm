"""Smoke test — simulation.analytics.external_impact (외부충격·팬데믹 onset layer).

불변식 (TDD red→green→refactor; 사용자 지정 6-8 case):
    1. 합성 평균-점프 changepoint 적중 (CUSUM onset 이 점프 위치 부근).
    2. alert 심각도↑ → 단계 단조 비감소 (monotone, 심각 도달).
    3. 휴일/개학 calendar flag 정확 (양력 고정 공휴일·개학 윈도).
    4. leak-free — 미래 관측 변경이 과거 출력 불변 (causal 보장).
    5. shape 일치 (모든 반환 배열 == len 입력).
    6. edge — 짧은/NaN/상수 시계열 raise 없이 안전.
    7. 입력 검증 fail-fast (잘못된 method/thresholds/길이 → ValueError).
    8. 실 서울 ILI 337주 데모 1회 (placeholder 금지, 실측 사용).

Refs: Page (1954) CUSUM; Hutwagner (2003) EARS; Kang/Son/Kim (2024) KDCA threshold.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from simulation.analytics.external_impact import (
    detect_regime_shifts,
    exogenous_shock_features,
    pandemic_alert_level,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. 합성 평균-점프 changepoint 적중
# ---------------------------------------------------------------------------
def test_cusum_detects_synthetic_mean_jump():
    rng = np.random.default_rng(7)
    # 60주 baseline(평균 3) → 60주 점프(평균 12). onset = index 60 부근.
    pre = 3.0 + rng.normal(0, 0.4, size=60)
    post = 12.0 + rng.normal(0, 0.4, size=60)
    y = np.concatenate([pre, post])

    # drift=1.0 (Page slack): 저잡음 합성에서 small-window 표준화 잡음을 흡수.
    out = detect_regime_shifts(
        y, method="cusum", threshold=5.0, window=12, shift=1, drift=1.0
    )
    cps = out["changepoints"]
    assert len(cps) >= 1, "평균-점프를 하나도 못 잡음"
    first = cps[0]
    # onset 은 점프 직후 짧은 지연(누적 통계가 threshold 넘는 데 몇 주) 내.
    assert 60 <= first <= 70, f"onset({first})이 점프 위치(60) 부근이 아님"
    assert out["shift_flags"][first] == 1
    assert out["severity"][first] >= 1.0  # alarm severity = stat/threshold >= 1
    # 점프 전(레짐 안정 구간)엔 거짓 onset 없어야 (적정 slack 하에서).
    assert all(c >= 55 for c in cps), f"점프 전 가짜 onset: {cps}"
    # 최고 severity onset = 진짜 점프 위치 (default drift 라도 dominant onset 정확).
    out_def = detect_regime_shifts(y, method="cusum", threshold=5.0, window=8, shift=1)
    assert 60 <= int(np.argmax(out_def["severity"])) <= 62, \
        "dominant onset 이 점프 위치가 아님"


def test_zscore_method_flags_spike():
    rng = np.random.default_rng(11)
    y = 5.0 + rng.normal(0, 0.3, size=80)
    y[50] = 20.0  # 단발 spike
    out = detect_regime_shifts(y, method="zscore", threshold=3.0, window=10, shift=1)
    assert 50 in out["changepoints"], "명백한 spike 미탐지"
    assert out["severity"][50] == abs(out["zscore"][50])


# ---------------------------------------------------------------------------
# 2. alert 심각도↑ → 단계 단조 상승 (심각 도달)
# ---------------------------------------------------------------------------
def test_alert_level_monotone_with_severity():
    # 핵심 불변식: *동일 고정 baseline* 대비 anomaly 가 커지면 단계 비감소.
    # (rolling-std z 는 ramp 가 baseline 을 오염시키면 σ↑ 로 비단조 — 정상 통계.
    #  따라서 마지막 한 점만 변화시킨 anomaly 시리즈 집합으로 단조성을 검증.)
    rng = np.random.default_rng(5)
    base = 4.0 + rng.normal(0, 0.5, size=60)  # σ>0 (상수 baseline 은 z 정의불가)
    levels = []
    for anomaly in (4.0, 5.0, 6.0, 7.5, 10.0):  # 점점 큰 마지막 점 (σ≈0.5 기준)
        y = np.concatenate([base, [anomaly]])
        lv = pandemic_alert_level(y, baseline_window=52, shift=1, min_periods=8)
        levels.append(int(lv[-1]))
    assert levels == sorted(levels), f"심각도↑인데 단계 비단조: {levels}"
    assert levels[-1] == 3, f"최대 anomaly 가 심각(3) 미도달: {levels}"
    assert levels[0] == 0, f"무-anomaly 가 관심(0) 아님: {levels}"

    # 라벨 매핑 sanity: 단계 3 = '심각'
    res = pandemic_alert_level(
        np.concatenate([base, [40.0]]),  # σ≈0.5 baseline 대비 거대 anomaly
        baseline_window=52, shift=1, min_periods=8, return_labels=True,
    )
    assert res["level"][-1] == 3 and res["label_kr"][-1] == "심각"
    assert res["label_en"][-1] == "serious"


def test_alert_level_flat_series_stays_attention():
    y = np.full(80, 6.0)  # 변동 0 → anomaly 0 → 항상 관심(0)
    level = pandemic_alert_level(y, baseline_window=52, min_periods=8)
    assert set(np.unique(level)) <= {0}, "상수 시계열이 경보 발생시킴"


# ---------------------------------------------------------------------------
# 3. 휴일/개학 calendar flag 정확
# ---------------------------------------------------------------------------
def test_calendar_holiday_and_school_flags():
    dates = [
        "2021-01-01",  # 신정 → holiday
        "2021-03-03",  # 개학 윈도(3/2-3/9) + 삼일절(3/1 같은 주) → both
        "2021-06-06",  # 현충일 → holiday
        "2021-07-14",  # 평범한 여름 주 → neither
        "2021-08-25",  # 2학기 개학 윈도(8/16-9/1) → school
        "2021-10-09",  # 한글날 → holiday
    ]
    out = exogenous_shock_features(dates)
    #               신정 삼일절 현충일 평일 개학 한글날
    assert out["holiday_flag"].tolist() == [1, 1, 1, 0, 0, 1]
    assert out["school_start_flag"].tolist() == [0, 1, 0, 0, 1, 0]
    # mobility/subtype 미제공 → 전부 0
    assert out["mobility_drop_flag"].sum() == 0
    assert out["variant_shift_flag"].sum() == 0


def test_mobility_drop_and_variant_shift_flags():
    n = 40
    dates = [f"2021-{(i % 12) + 1:02d}-05" for i in range(n)]
    mobility = np.full(n, 100.0)
    mobility[20:25] = 50.0  # 50% drop → NPI 신호
    share = np.full(n, 0.3)
    share[30:] = 0.7  # +40%p 우위전환
    out = exogenous_shock_features(
        dates, mobility=mobility, subtype_share=share, min_periods=4
    )
    assert out["mobility_drop_flag"][20:25].sum() >= 1, "mobility drop 미탐지"
    assert out["variant_shift_flag"][30:].sum() >= 1, "변종 우위전환 미탐지"
    # drop 전 안정 구간은 flag 0
    assert out["mobility_drop_flag"][:18].sum() == 0


# ---------------------------------------------------------------------------
# 4. leak-free — 미래값 변경이 과거 출력 불변
# ---------------------------------------------------------------------------
def test_leak_free_future_does_not_affect_past():
    rng = np.random.default_rng(3)
    y = 5.0 + rng.normal(0, 0.5, size=100)

    full_cp = detect_regime_shifts(y, method="cusum", threshold=5.0)
    full_alert = pandemic_alert_level(y, baseline_window=40, min_periods=8)

    # 미래(>=t0) 를 극단 변경 → 과거(<t0) 출력 불변이어야 (causal).
    t0 = 70
    y2 = y.copy()
    y2[t0:] = 999.0
    cp2 = detect_regime_shifts(y2, method="cusum", threshold=5.0)
    alert2 = pandemic_alert_level(y2, baseline_window=40, min_periods=8)

    np.testing.assert_array_equal(
        full_cp["severity"][:t0], cp2["severity"][:t0],
    )
    np.testing.assert_array_equal(full_alert[:t0], alert2[:t0])


# ---------------------------------------------------------------------------
# 5. shape 일치
# ---------------------------------------------------------------------------
def test_shape_consistency():
    y = RNG.normal(5, 1, size=120)
    cp = detect_regime_shifts(y, method="cusum")
    assert cp["shift_flags"].shape == (120,)
    assert cp["severity"].shape == (120,)
    assert cp["zscore"].shape == (120,)
    assert pandemic_alert_level(y).shape == (120,)

    dates = [f"2020-01-{(i % 28) + 1:02d}" for i in range(120)]
    feats = exogenous_shock_features(
        dates, mobility=RNG.normal(100, 5, 120), subtype_share=RNG.uniform(0, 1, 120)
    )
    for k, v in feats.items():
        assert v.shape == (120,), f"{k} shape mismatch: {v.shape}"


# ---------------------------------------------------------------------------
# 6. edge — 짧은/NaN/상수 안전
# ---------------------------------------------------------------------------
def test_edge_short_nan_constant_no_raise():
    # 짧음 (warm-up 만)
    short = np.array([1.0, 2.0, 3.0])
    cp = detect_regime_shifts(short, method="cusum", window=8, min_periods=4)
    assert cp["changepoints"] == []  # 탐지 불가하나 raise X
    assert pandemic_alert_level(short, min_periods=8).tolist() == [0, 0, 0]

    # NaN 포함
    y = np.array([5.0, np.nan, 5.0, 6.0, np.nan, 30.0, 5.0] * 10)
    cp2 = detect_regime_shifts(y, method="zscore", threshold=3.0)
    assert np.all(np.isfinite(cp2["severity"]))  # NaN 누출 없이 finite
    lv = pandemic_alert_level(y, baseline_window=20, min_periods=5)
    assert lv.dtype == np.int8 and np.all((lv >= 0) & (lv <= 3))


# ---------------------------------------------------------------------------
# 7. 입력 검증 fail-fast
# ---------------------------------------------------------------------------
def test_input_validation_raises():
    raises = pytest.raises if pytest is not None else _manual_raises
    with raises(ValueError):
        detect_regime_shifts([], method="cusum")
    with raises(ValueError):
        detect_regime_shifts([1, 2, 3], method="bogus")
    with raises(ValueError):
        pandemic_alert_level([1, 2, 3], thresholds=(1.0, 2.0))  # 길이≠3
    with raises(ValueError):
        pandemic_alert_level([1, 2, 3], thresholds=(3.0, 2.0, 1.0))  # 비오름차순
    with raises(ValueError):
        exogenous_shock_features([])  # 빈 dates
    with raises(ValueError):
        exogenous_shock_features(["2020-01-01", "2020-01-08"], mobility=[1.0])  # 길이 불일치


class _manual_raises:  # pragma: no cover — pytest 없을 때 fallback
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), \
            f"기대 {self.exc.__name__} 미발생 (got {et})"
        return True


# ---------------------------------------------------------------------------
# 8. 실 서울 ILI 337주 데모 (실측, placeholder 금지)
# ---------------------------------------------------------------------------
def test_real_seoul_ili_demo():
    import sqlite3
    from pathlib import Path

    db = Path(__file__).resolve().parents[1] / "simulation/data/db/epi_real_seoul.db"
    if not db.exists():  # 환경에 DB 없으면 skip (CI portable)
        if pytest is not None:
            pytest.skip("epi_real_seoul.db 미존재")
        return
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(
            "SELECT season_start, week_seq, AVG(ili_rate) "
            "FROM sentinel_influenza GROUP BY season_start, week_seq "
            "ORDER BY season_start, week_seq"
        ).fetchall()
    finally:
        con.close()
    y = np.array([r[2] for r in rows], dtype=float)
    assert y.size >= 300, f"ILI 주수 부족: {y.size}"

    cp = detect_regime_shifts(y, method="cusum", threshold=5.0, window=8, shift=1)
    alert = pandemic_alert_level(y, baseline_window=52, shift=1, min_periods=8,
                                 return_labels=True)
    # 실 ILI 엔 COVID 충격·계절 정점이 있어 onset 과 심각 단계가 존재해야 정상.
    assert len(cp["changepoints"]) >= 1, "실 ILI 에서 onset 0건 (비정상)"
    assert alert["level"].max() >= 2, "실 ILI 에서 경계+ 단계 미발생 (비정상)"
    # shape 일치 + leak-free 구조 (전 시점 finite)
    assert cp["severity"].shape == y.shape
    assert np.all(np.isfinite(alert["zscore"]))
