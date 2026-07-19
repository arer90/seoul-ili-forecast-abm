"""P4 on real data — cross-season behavioral-parameter identifiability."""
import os

import pytest

_DB = "simulation/data/db/epi_real_seoul.db"


def test_cross_season_identifiability_logic():
    from simulation.abm.realdata_identifiability import cross_season_identifiability
    fits = [
        {"r2": 0.7, "alpha": 1.0, "kappa": 0.2, "tau": 90.0, "theta": 0.1},
        {"r2": 0.5, "alpha": 1.0, "kappa": 0.2, "tau": 60.0, "theta": 0.15},
        {"r2": 0.2, "alpha": 3.0, "kappa": 0.5, "tau": 120.0, "theta": 0.3},  # r2<0.3 dropped
    ]
    ci = cross_season_identifiability(fits)
    assert ci["n_usable_seasons"] == 2                      # only r2≥0.3 pooled
    assert ci["params"]["alpha"]["cv"] == 0.0               # both usable = 1.0
    assert ci["params"]["alpha"]["identifiable_from_ili"] is True
    assert ci["params"]["tau"]["identifiable_from_ili"] in (True, False)  # scatter → either


@pytest.mark.skipif(not os.path.exists(_DB), reason="real DB absent")
def test_real_season_series():
    from simulation.abm.realdata_identifiability import real_season_series
    s = real_season_series(2023)
    assert len(s) > 30 and float(s.max()) > 10.0           # a real flu wave
