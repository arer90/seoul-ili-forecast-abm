"""Multi-proxy behavioral validation — real Seoul mobility vs ABM-predicted loop."""
import os

import pytest

_DB = "simulation/data/db/epi_real_seoul.db"


def test_sig_handles_zero_p():
    from simulation.abm.multiproxy_behavioral_validation import _sig
    assert _sig({"null_p": 0.0}) is True      # p=0.0 IS significant (the bug we fixed)
    assert _sig({"null_p": 0.04}) is True
    assert _sig({"null_p": 0.5}) is False
    assert _sig({}) is False


def test_climatology_and_anomaly():
    from simulation.abm.multiproxy_behavioral_validation import _anomaly, _climatology
    d = {(2020, 1): 10.0, (2021, 1): 20.0, (2020, 2): 5.0}
    assert _climatology(d)[1] == 15.0                 # week-1 mean across years
    an = _anomaly(d, exclude_years=(2021,))
    assert (2021, 1) not in an                        # COVID year dropped
    assert an[(2020, 1)] == 10.0 - 15.0               # anomaly = value - climatology


@pytest.mark.skipif(not os.path.exists(_DB), reason="real DB absent")
def test_real_loaders_and_proxy_loops():
    from simulation.abm.multiproxy_behavioral_validation import (
        load_weekly_ili, load_weekly_proxies, proxy_loops)
    ili = load_weekly_ili()
    assert len(ili) > 100 and all(isinstance(k, tuple) and len(k) == 2 for k in ili)
    px = load_weekly_proxies()
    assert len(px) > 100 and {"day", "night", "inflow"} <= set(next(iter(px.values())))
    # deseasonalized + COVID-excluded loops compute without error
    r = proxy_loops(n_null=200, deseasonalize=True, exclude_covid=True)
    assert "excl_covid" in r["mode"]
    for key in ("day", "inflow"):
        lp = r["proxies"][key]
        assert "circulation" in lp and "null_p" in lp
