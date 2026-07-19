"""⑬: per-agent comorbidity layer (KNHANES-grounded) — age gradient + severity."""
import numpy as np

from simulation.abm.comorbidity import (
    KNHANES_PREVALENCE,
    assign_comorbidities,
    comorbidity_severity_multiplier,
    validate_comorbidity_age_gradient,
)


def test_sampled_prevalence_matches_knhanes_band():
    # a large 60+ (band 6) cohort should reproduce the encoded prevalence
    com = assign_comorbidities(np.full(20000, 6), seed=1)
    for cond in KNHANES_PREVALENCE:
        target = KNHANES_PREVALENCE[cond][6]
        assert abs(com[cond].mean() - target) < 0.02, cond


def test_severity_multiplier_no_condition_is_one_and_capped():
    n = 5
    none = {c: np.zeros(n, dtype=bool) for c in KNHANES_PREVALENCE}
    assert np.allclose(comorbidity_severity_multiplier(none), 1.0)
    allc = {c: np.ones(n, dtype=bool) for c in KNHANES_PREVALENCE}
    m = comorbidity_severity_multiplier(allc, cap=3.0)
    assert np.all(m <= 3.0) and np.all(m > 1.0)


def test_more_comorbidities_higher_multiplier():
    one = {"obesity": np.array([True]), "diabetes": np.array([False]),
           "hypertension": np.array([False]), "hypercholesterolemia": np.array([False])}
    two = {"obesity": np.array([True]), "diabetes": np.array([True]),
           "hypertension": np.array([False]), "hypercholesterolemia": np.array([False])}
    assert comorbidity_severity_multiplier(two)[0] > comorbidity_severity_multiplier(one)[0]


def test_age_gradient_reproduced():
    r = validate_comorbidity_age_gradient(n_per_band=4000)
    assert r["monotone_chronic"] is True
    assert r["elderly_gt_young"] is True
    assert r["match"] is True, r["verdict"]
    # elderly burden clearly exceeds young
    assert r["per_band_chronic_burden"][6] > r["per_band_chronic_burden"][2]


def test_enrich_population_severity_raises_for_elderly():
    from simulation.abm.comorbidity import enrich_population_severity
    sev = np.ones(6000); bands = np.repeat([2, 6], 3000)  # young vs old
    adj, com = enrich_population_severity(sev, bands)
    assert adj[bands == 6].mean() > adj[bands == 2].mean()  # elderly higher
    assert adj.min() >= 1.0
