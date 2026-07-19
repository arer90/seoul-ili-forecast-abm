"""Stage 5 — Metapop SEIR-V-D simulator unit tests.

Covers:
  - ``MetapopParams.validate`` — rejects malformed inputs
  - Compartment conservation at epi-gate tolerance (1e-6)
  - FoI vector produces a finite, non-negative λ vector
  - Intervention resolver (scale / set / add; vaccination_rate branch)
  - Scenario registry end-to-end (baseline, NPI, vax, antiviral, combined,
    strain_mismatch)
  - Epi-validity gate attaches correctly to SimResult

These tests do **not** touch the DB — they build ``MetapopParams`` from
synthetic populations / mobility so CI can run them in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.sim.foi import compute_foi, effective_daytime_population
from simulation.sim.interventions import apply_interventions
from simulation.sim.metapop_seirvd import MetapopSEIRVD, SimResult, run_scenario
from simulation.sim.parameters import (
    COMPARTMENTS,
    DEFAULT_FLU_PARAMS,
    DiseaseParams,
    InterventionSpec,
    MetapopParams,
    N_COMPARTMENTS,
)
from simulation.sim.scenarios import SCENARIO_REGISTRY, register_scenario
from simulation.sim.stepper import rk4_step, seirvd_derivative


# ── Fixtures ───────────────────────────────────────────────────────────
def _toy_params(G=5, days=60, dt=0.25, infected0_first=10.0):
    pops = np.full(G, 100_000.0)
    M = np.full((G, G), 0.05 / (G - 1))
    np.fill_diagonal(M, 0.95)
    I0 = np.zeros(G)
    I0[0] = infected0_first
    return MetapopParams(
        disease=DEFAULT_FLU_PARAMS, populations=pops, mobility=M,
        district_names=[f"d{i}" for i in range(G)],
        initial_infected=I0, days=days, dt=dt, seed=0,
    )


# ── Parameter validation ───────────────────────────────────────────────
class TestMetapopParamsValidate:
    def test_empty_populations_rejected(self):
        p = MetapopParams(populations=np.array([]), mobility=np.zeros((0, 0)))
        with pytest.raises(ValueError, match="empty"):
            p.validate()

    def test_non_square_mobility_rejected(self):
        p = MetapopParams(populations=np.array([100., 200.]),
                          mobility=np.eye(3))
        with pytest.raises(ValueError, match="incompatible"):
            p.validate()

    def test_non_row_stochastic_rejected(self):
        M = np.full((3, 3), 0.2)  # rows sum to 0.6, not 1
        p = MetapopParams(populations=np.array([100., 200., 300.]),
                          mobility=M)
        with pytest.raises(ValueError, match="row-stochastic"):
            p.validate()

    def test_initial_infected_exceeds_pop_rejected(self):
        p = _toy_params()
        p.initial_infected = p.populations + 1.0  # more infected than people
        with pytest.raises(ValueError, match="exceeds populations"):
            p.validate()

    def test_valid_params_pass(self):
        _toy_params().validate()  # must not raise


# ── FoI ────────────────────────────────────────────────────────────────
class TestFoI:
    def test_daytime_population_sums_to_total(self):
        G = 10
        pops = np.arange(100, 100 + G, dtype=float) * 1000
        M = np.full((G, G), 0.1 / (G - 1))
        np.fill_diagonal(M, 0.9)
        daytime = effective_daytime_population(M, pops)
        assert np.isclose(daytime.sum(), pops.sum(), atol=1e-6), \
            "daytime pop total must equal night pop total (people conserved)"

    def test_foi_nonneg_and_zero_when_no_infected(self):
        G = 5
        pops = np.full(G, 1000.0)
        M = np.eye(G)
        lam = compute_foi(
            I=np.zeros(G), S=pops, populations=pops, mobility=M, beta=0.4,
        )
        assert lam.shape == (G,)
        assert np.all(lam == 0.0)

    def test_foi_scales_with_beta(self):
        G = 3
        pops = np.full(G, 1000.0); I = np.array([10., 0., 0.])
        M = np.eye(G)
        lam1 = compute_foi(I, pops, pops, M, beta=0.4)
        lam2 = compute_foi(I, pops, pops, M, beta=0.8)
        assert np.allclose(lam2, 2 * lam1)


# ── Stepper + conservation ─────────────────────────────────────────────
class TestConservation:
    def test_total_mass_preserved_across_full_run(self):
        p = _toy_params(G=5, days=120)
        sim = MetapopSEIRVD(p)
        res = sim.run(run_validator=False)
        totals = res.state.sum(axis=(1, 2))
        N_total = p.populations.sum()
        rel = np.max(np.abs(totals - N_total) / N_total)
        assert rel < 1e-6, f"mass drift {rel:.2e} exceeds COMPARTMENT_TOL"

    def test_conservation_holds_with_interventions(self):
        p = _toy_params()
        iv = [
            InterventionSpec("beta", 0.5, 10, 30, op="scale"),
            InterventionSpec("vaccination_rate", 0.01, 15, 45, op="set"),
        ]
        res = MetapopSEIRVD(p).run(interventions=iv, run_validator=False)
        totals = res.state.sum(axis=(1, 2))
        N_total = p.populations.sum()
        rel = np.max(np.abs(totals - N_total) / N_total)
        assert rel < 1e-6

    def test_compartments_remain_nonneg(self):
        p = _toy_params()
        res = MetapopSEIRVD(p).run(run_validator=False)
        assert np.all(res.state >= 0.0)

    def test_zero_beta_produces_zero_new_infections(self):
        p = _toy_params(infected0_first=0.0)
        iv = [InterventionSpec("beta", 0.0, 0, 10_000, op="set")]
        res = MetapopSEIRVD(p).run(interventions=iv, run_validator=False)
        # With β=0 and no initial I, the compartments should stay put
        # (tiny numerical drift allowed)
        assert res.city_total("I").max() < 1.0
        assert res.city_total("E").max() < 1.0


# ── Interventions ──────────────────────────────────────────────────────
class TestInterventions:
    def test_scale_op(self):
        d = DiseaseParams(R0=2.0)
        iv = [InterventionSpec("R0", 0.5, 0, 10, op="scale")]
        out_d, _ = apply_interventions(5, d, 0.0, iv, G=1)
        assert out_d.R0 == 1.0

    def test_set_op(self):
        d = DiseaseParams(VE=0.50)
        iv = [InterventionSpec("VE", 0.65, 0, 10, op="set")]
        out_d, _ = apply_interventions(3, d, 0.0, iv, G=1)
        assert out_d.VE == 0.65

    def test_add_op(self):
        d = DiseaseParams(VE=0.50)
        iv = [InterventionSpec("VE", 0.10, 0, 10, op="add")]
        out_d, _ = apply_interventions(3, d, 0.0, iv, G=1)
        assert abs(out_d.VE - 0.60) < 1e-12

    def test_outside_window_is_noop(self):
        d = DiseaseParams(R0=2.0)
        iv = [InterventionSpec("R0", 0.5, 0, 10, op="scale")]
        out_d, _ = apply_interventions(50, d, 0.0, iv, G=1)
        assert out_d.R0 == 2.0

    def test_vaccination_rate_targets(self):
        d = DiseaseParams()
        iv = [InterventionSpec("vaccination_rate", 0.02, 0, 10,
                                op="set", targets=(1, 3))]
        _, vax = apply_interventions(5, d, 0.0, iv, G=5)
        assert vax[0] == 0.0
        assert vax[1] == 0.02
        assert vax[3] == 0.02
        assert vax[4] == 0.0

    def test_unknown_parameter_raises(self):
        d = DiseaseParams()
        iv = [InterventionSpec("banana", 1.0, 0, 10)]
        with pytest.raises(ValueError, match="unknown parameter"):
            apply_interventions(1, d, 0.0, iv, G=1)

    def test_later_intervention_wins(self):
        d = DiseaseParams(R0=1.0)
        iv = [
            InterventionSpec("R0", 0.5, 0, 10, op="scale"),  # 1.0 -> 0.5
            InterventionSpec("R0", 3.0, 0, 10, op="set"),    # 0.5 -> 3.0
        ]
        out_d, _ = apply_interventions(5, d, 0.0, iv, G=1)
        assert out_d.R0 == 3.0


# ── Scenarios ──────────────────────────────────────────────────────────
class TestScenarios:
    def test_all_registered_scenarios_complete(self):
        p = _toy_params(days=60)
        for name in SCENARIO_REGISTRY:
            r = run_scenario(name, p)
            assert isinstance(r, SimResult)
            assert r.state.shape[0] == p.days + 1

    def test_npi_reduces_peak_vs_baseline(self):
        p = _toy_params(days=120)
        base = run_scenario("baseline", p)
        npi = run_scenario("npi_lockdown", p)
        assert npi.city_total("I").max() < base.city_total("I").max()

    def test_vaccination_moves_mass_to_V(self):
        p = _toy_params(days=120)
        res = run_scenario("vaccination_campaign", p)
        assert res.city_total("V")[-1] > 0.0

    def test_combined_response_lowest_attack_rate(self):
        p = _toy_params(days=180)
        results = {n: run_scenario(n, p) for n in
                   ["baseline", "npi_lockdown", "vaccination_campaign",
                    "combined_response"]}
        attack = {n: r.city_total("D")[-1] for n, r in results.items()}
        best = min(attack, key=attack.get)
        assert best == "combined_response", (
            f"combined_response should have lowest final D; got {attack}")

    def test_unknown_scenario_raises(self):
        with pytest.raises(KeyError, match="unknown scenario"):
            run_scenario("not_a_real_scenario", _toy_params())

    def test_register_scenario_accepts_new(self):
        def _custom(params):
            p = params if params else _toy_params()
            return p, [InterventionSpec("R0", 0.3, 0, 200, op="scale")]
        register_scenario("_test_custom", _custom)
        r = run_scenario("_test_custom", _toy_params(days=60))
        assert isinstance(r, SimResult)


# ── Gate integration ───────────────────────────────────────────────────
class TestEpiGateIntegration:
    def test_gate_runs_and_attaches(self):
        res = MetapopSEIRVD(_toy_params()).run(run_validator=True)
        assert "metapop_seirvd" in res.epi_validity
        assert res.epi_validity["metapop_seirvd"]["status"] in ("ok", "fail")

    def test_gate_skipped_when_disabled(self):
        res = MetapopSEIRVD(_toy_params()).run(run_validator=False)
        assert res.epi_validity == {}

    def test_conservation_check_passes_for_baseline(self):
        res = MetapopSEIRVD(_toy_params()).run(run_validator=True)
        report = res.epi_validity.get("metapop_seirvd", {})
        checks = report.get("checks", {})
        cc = checks.get("compartment_conservation", {})
        assert cc.get("max_rel_err", 1.0) < 1e-6


# ── SimResult helpers ──────────────────────────────────────────────────
class TestSimResult:
    def test_city_total_sums_across_gu(self):
        res = MetapopSEIRVD(_toy_params()).run(run_validator=False)
        I_city = res.city_total("I")
        manual = res.state[:, :, COMPARTMENTS.index("I")].sum(axis=1)
        assert np.allclose(I_city, manual)

    def test_reported_cases_scales_with_fraction(self):
        res = MetapopSEIRVD(_toy_params()).run(run_validator=False)
        a = res.reported_cases(0.10)
        b = res.reported_cases(0.20)
        assert np.allclose(b, 2 * a)

    def test_compartment_properties(self):
        res = MetapopSEIRVD(_toy_params()).run(run_validator=False)
        assert res.S.shape == res.state.shape[:2]
        assert np.allclose(
            res.S + res.E + res.I + res.R + res.V + res.D,
            res.state.sum(axis=-1),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
