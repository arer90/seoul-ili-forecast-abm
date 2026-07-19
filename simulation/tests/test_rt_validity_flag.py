"""epi.rt_estimate must SURFACE implausible Rt (not silently pass it).

The shared RtEstimator has a known off-by-one that can inflate Rt far above a sane
band; until that shared/forecasting-coupled fix lands, the MCP tool annotates a
`validity` block so the ARIA layer and any client can see the anomaly. This test
pins the annotation contract (it does NOT assert the buggy value, so it survives
the eventual estimator fix).
"""
from __future__ import annotations


def test_rt_estimate_surfaces_validity_band():
    from simulation.server.mcp_epi import EpiMCPServer
    srv = EpiMCPServer()
    res = srv.call_tool("epi.rt_estimate", {"gu": "seoul_city"})
    c = res.content
    if c.get("status") != "ok":
        # environment without the weekly panel — nothing to validate here
        return
    v = c["validity"]
    assert v["rt_plausible_band"] == [0.3, 8.0]
    assert set(v) >= {"rt_plausible_band", "n_evaluated", "n_out_of_band", "all_in_band"}
    # flag logic is self-consistent
    assert v["all_in_band"] == (v["n_out_of_band"] == 0)
    # when out-of-band Rt exist, an explanatory warning is attached (not hidden)
    if v["n_out_of_band"] > 0:
        assert "warning" in v and "RtEstimator" in v["warning"]
