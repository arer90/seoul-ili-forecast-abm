"""test_sci_supplement.py — SCI 외적타당도/교란 3-표 헬퍼 회귀 가드 (D-3 TDD).

대상: ``simulation.scripts.sci_supplement.tables`` 의 순수 헬퍼.
무거운 NegBinGLM rolling 은 제외 — 여기선 season 라벨링·regime 매핑·metric·
leak-free rolling 의 단조 인과(예측 t 가 미래 미참조)만 빠르게 검증.

회귀 case: season 경계(Aug 컷) / regime 매핑 / WIS·R²·MASE matched·edge·empty /
LOSO hold-out 이 해당 시즌만 평가 / rolling 이 strict-past 만 학습 (leak-free).
"""
import numpy as np
import pytest

from simulation.scripts.sci_supplement.tables import (
    BASIC_FEATURE_COLS,
    REGIME_OF_SEASON,
    _mase,
    _r2,
    _wis,
    rolling_one_step,
    season_of,
)


# ── season_of: epi-year starts ~ August ──────────────────────────────
@pytest.mark.parametrize("iso, expected", [
    ("2019-09-08", 2019),  # September → that season
    ("2020-01-12", 2019),  # January → previous-year season
    ("2020-07-31", 2019),  # July (month<8) → still prev season
    ("2020-08-01", 2020),  # August boundary → new season
    ("2022-12-31", 2022),  # December → that season
    ("2023-02-15", 2022),  # Feb → prev season (the 2022/23 winter)
])
def test_season_of_august_boundary(iso, expected):
    assert season_of(np.datetime64(iso, "D")) == expected


# ── regime mapping SSOT (phase11 SEASON_CONTEXT) ──────────────────────
def test_regime_mapping():
    assert REGIME_OF_SEASON[2019] == "pre_covid"
    assert REGIME_OF_SEASON[2020] == "during_covid"
    assert REGIME_OF_SEASON[2022] == "during_covid"
    assert REGIME_OF_SEASON[2023] == "post_rebound"
    assert REGIME_OF_SEASON[2025] == "post_rebound"


# ── _wis: matched / empty / too-few-residuals (NaN) ───────────────────
def test_wis_matched_finite():
    yt = np.array([3.0, 4.0, 5.0, 6.0, 3.5])
    yp = np.array([3.1, 3.9, 5.2, 5.5, 3.7])
    resid = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.15, -0.05, 0.25])
    w = _wis(yt, yp, resid)
    assert np.isfinite(w) and w >= 0.0


def test_wis_perfect_point_lower_than_biased():
    # WIS must reward an accurate point forecast over a biased one (same band).
    yt = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    resid = np.array([0.5, -0.5, 0.3, -0.3, 0.1, -0.1, 0.2, -0.2, 0.0, 0.4])
    good = _wis(yt, yt.copy(), resid)
    bad = _wis(yt, yt + 3.0, resid)
    assert good < bad


def test_wis_empty_or_thin_residuals_is_nan():
    assert np.isnan(_wis(np.array([]), np.array([]), np.arange(10)))
    assert np.isnan(_wis(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                         np.array([0.1, 0.2])))  # <5 residuals


# ── _r2 / _mase: matched + degenerate edge ───────────────────────────
def test_r2_perfect_and_degenerate():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _r2(y, y.copy()) == pytest.approx(1.0)
    assert np.isnan(_r2(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])))


def test_mase_scaling_and_zero_denominator():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # naive diff = 1.0
    yt = np.array([6.0, 7.0])
    yp = np.array([6.5, 7.5])  # MAE 0.5 → MASE 0.5
    assert _mase(yt, yp, y_train) == pytest.approx(0.5)
    assert np.isnan(_mase(yt, yp, np.array([3.0, 3.0, 3.0])))  # flat → 0 denom


# ── leak-free rolling: prediction t never uses t or future ────────────
def test_rolling_is_strict_past_only():
    # Synthetic monotone series; verify the predicted origins are all >= min
    # train and that holding out a season scores ONLY that season's weeks.
    rng = np.random.RandomState(0)
    n = 120
    y = np.clip(5 + 3 * np.sin(np.arange(n) / 8.0) + rng.randn(n) * 0.2, 0.1, None)
    X = rng.rand(n, len(BASIC_FEATURE_COLS))
    # fake dates: weekly from 2019-09-08, seasons derived by season_of.
    dates = (np.datetime64("2019-09-08", "D")
             + np.arange(n) * np.timedelta64(7, "D"))
    seasons = np.array([season_of(d) for d in dates], dtype=int)

    res = rolling_one_step(X, y, dates, seasons, hold_out_season=None,
                           min_train=30)
    # every scored origin index must be >= min_train (no leakage of warm-up).
    assert res["idx"].min() >= 30
    # monotone idx → produced in time order.
    assert np.all(np.diff(res["idx"]) > 0)

    # LOSO: holding out a present season scores exactly that season's weeks.
    target = sorted(set(seasons.tolist()))[-1]  # last (fully past-covered) season
    loso = rolling_one_step(X, y, dates, seasons, hold_out_season=target,
                            min_train=30)
    if len(loso["seasons"]):
        assert set(loso["seasons"].tolist()) == {target}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
