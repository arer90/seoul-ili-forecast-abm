"""C2 (M7 SCI-grade): COVID-era sensitivity guard.

Both robustness modes the flu-surveillance reviewer literature expects already
exist — leave-the-pandemic-out (``exclude``) and the NPI-era covariate
(``indicator``); the epi-survey's "no NPI covariate" finding missed them. This
guards the extracted ``apply_covid_sensitivity_mode`` deep helper.
"""
import numpy as np

from simulation.pipeline.data import apply_covid_sensitivity_mode


def _fixture():
    # 6 weeks: 2 pre-COVID, 2 COVID-era (2020-03..2022-12), 2 post-COVID
    dates = np.array(["2019-06-01", "2019-12-01", "2020-06-01", "2021-06-01",
                      "2023-06-01", "2024-06-01"], dtype="datetime64[D]")
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    return X, y, dates, ["f0", "f1"], np.ones((3, 2), dtype=float)


def test_include_is_noop():
    X, y, d, c, r = _fixture()
    X2, _y2, _d2, c2, r2 = apply_covid_sensitivity_mode(X, y, d, c, r, "include")
    assert X2.shape == (6, 2) and c2 == ["f0", "f1"] and r2.shape == (3, 2)


def test_exclude_drops_covid_weeks():
    X, y, d, c, r = _fixture()
    X2, y2, d2, c2, _r2 = apply_covid_sensitivity_mode(X, y, d, c, r, "exclude")
    assert X2.shape == (4, 2) and len(y2) == 4 and len(d2) == 4  # 2 COVID weeks gone
    assert not ((d2 >= np.datetime64("2020-03-01"))
                & (d2 <= np.datetime64("2022-12-31"))).any()
    assert c2 == ["f0", "f1"]  # no covariate under exclude


def test_indicator_adds_covariate_and_real_zeros():
    X, y, d, c, r = _fixture()
    X2, _y2, _d2, c2, r2 = apply_covid_sensitivity_mode(X, y, d, c, r, "indicator")
    assert X2.shape == (6, 3) and c2 == ["f0", "f1", "covid_era_indicator"]
    assert list(X2[:, -1]) == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]  # 1 for COVID weeks
    assert r2.shape == (3, 3) and list(r2[:, -1]) == [0.0, 0.0, 0.0]  # real slab 0s


def test_dates_none_is_noop():
    X, y, _d, c, r = _fixture()
    X2, _y2, d2, _c2, _r2 = apply_covid_sensitivity_mode(X, y, None, c, r, "exclude")
    assert X2.shape == (6, 2) and d2 is None
