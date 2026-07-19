"""#2 — kernel contact_matrix param: inject alternative age-mixing (robustness)."""
import numpy as np


def test_age_contact_factor_param_overrides_default():
    from simulation.abm.agent_kernel import _age_contact_factor_by_agent
    age = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2], dtype=np.int8)
    default = _age_contact_factor_by_agent(age)                      # POLYMOD
    homog = _age_contact_factor_by_agent(age, np.full((7, 7), 1.0))  # homogeneous
    assert not np.allclose(default, homog)        # the assumption is now overridable
    assert np.allclose(homog, 1.0)                # homogeneous mixing → all factors 1
    assert default.std() > 0                      # POLYMOD is assortative (varied)


def test_prepare_population_arrays_threads_contact_matrix():
    from simulation.abm.agent_kernel import _prepare_population_arrays
    pop = {"home_gu": np.zeros(5, dtype=np.int8), "work_gu": np.zeros(5, dtype=np.int8),
           "age_band": np.array([0, 2, 4, 6, 1], dtype=np.int8),
           "occupation": np.array(["other"] * 5), "severity": np.ones(5)}
    d = _prepare_population_arrays(pop, 5)
    dh = _prepare_population_arrays(pop, 5, contact_matrix=np.full((7, 7), 1.0))
    assert not np.allclose(d["age_contact_factor"], dh["age_contact_factor"])
