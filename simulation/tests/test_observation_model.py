"""관측모형 TDD — 잠재 SEIR-V-D 감염 → ILI(NegBin) + 양성률(Binom).

논문 §관측모형 (WHO: ILI≠확진 → latent→observed 사상 필수). macOS: run PER-FILE.
"""
import math

import numpy as np
import pytest

from simulation.abm.observation_model import (
    ObservationParams, symptomatic_incidence, ili_mean, sample_ili,
    sample_positivity, negbin_loglik, fit_report_rate,
)


def test_params_validate():
    ObservationParams().validate()  # defaults OK
    with pytest.raises(ValueError):
        ObservationParams(symptomatic_frac=1.5).validate()
    with pytest.raises(ValueError):
        ObservationParams(care_seeking=0.0).validate()
    with pytest.raises(ValueError):
        ObservationParams(nb_dispersion=0.0).validate()


def test_symptomatic_and_mean():
    p = ObservationParams(symptomatic_frac=0.5, care_seeking=0.4, reporting_rate=1.0)
    inf = np.array([0.0, 100.0, 200.0])
    sym = symptomatic_incidence(inf, p)
    assert np.allclose(sym, [0.0, 50.0, 100.0])
    mu = ili_mean(sym, p)                 # ρ = 0.4
    assert np.allclose(mu, [0.0, 20.0, 40.0])


def test_input_validation():
    p = ObservationParams()
    with pytest.raises(ValueError):
        ili_mean(np.array([-1.0, 2.0]), p)          # negative
    with pytest.raises(ValueError):
        ili_mean(np.array([np.nan, 2.0]), p)        # non-finite


def test_sample_ili_reproducible_and_unbiased():
    p = ObservationParams(symptomatic_frac=1.0, care_seeking=1.0, reporting_rate=1.0,
                          nb_dispersion=10.0)
    sym = np.full(5000, 30.0)             # μ = 30 each
    a = sample_ili(sym, p, np.random.default_rng(0))
    b = sample_ili(sym, p, np.random.default_rng(0))
    assert np.array_equal(a, b)           # 재현성 (same seed)
    assert abs(a.mean() - 30.0) < 1.0     # 불편(mean ≈ μ)
    # 과분산: Var ≈ μ + μ²/φ = 30 + 900/10 = 120 > Poisson(30)
    assert a.var() > 60.0


def test_sample_ili_zero_mean_is_zero():
    p = ObservationParams()
    out = sample_ili(np.array([0.0, 0.0]), p, np.random.default_rng(1))
    assert np.array_equal(out, [0.0, 0.0])   # degenerate-safe


def test_sample_ili_requires_rng():
    with pytest.raises(ValueError):
        sample_ili(np.array([1.0]), ObservationParams(), None)


def test_negbin_loglik_poisson_limit():
    """φ→큰 값이면 NegBin → Poisson. NB loglik ≈ Poisson loglik."""
    y = np.array([3.0, 5.0, 2.0])
    mu = np.array([4.0, 4.0, 4.0])
    nb = negbin_loglik(y, mu, phi=1e6)
    pois = sum(yi * math.log(mi) - mi - math.lgamma(yi + 1) for yi, mi in zip(y, mu))
    assert abs(nb - pois) < 1e-2


def test_negbin_loglik_degenerate():
    assert negbin_loglik(np.array([0.0]), np.array([0.0]), phi=5.0) == 0.0   # μ=0,y=0
    assert negbin_loglik(np.array([3.0]), np.array([0.0]), phi=5.0) == float("-inf")  # μ=0,y>0


def test_negbin_loglik_peaks_at_truth():
    """우도가 올바른 μ 근처에서 최대 (보정 타당성)."""
    rng = np.random.default_rng(7)
    p = ObservationParams(symptomatic_frac=1.0, care_seeking=1.0, reporting_rate=1.0, nb_dispersion=20.0)
    sym = np.full(400, 25.0)
    y = sample_ili(sym, p, rng)
    ll_true = negbin_loglik(y, np.full(400, 25.0), 20.0)
    ll_low  = negbin_loglik(y, np.full(400, 10.0), 20.0)
    ll_high = negbin_loglik(y, np.full(400, 50.0), 20.0)
    assert ll_true > ll_low and ll_true > ll_high


def test_fit_report_rate():
    p = ObservationParams(symptomatic_frac=0.5)
    sym = np.array([100.0, 100.0])        # frac·sym = 50 each → denom 100
    y = np.array([20.0, 20.0])            # Σy=40 → ρ̂ = 40/100 = 0.4
    assert abs(fit_report_rate(y, sym, p) - 0.4) < 1e-9


def test_positivity_bounds():
    p = ObservationParams(background_rate=50.0)
    sym = np.array([0.0, 50.0, 1e6])
    n = np.array([100, 100, 100])
    pos, tot = sample_positivity(sym, n, p, np.random.default_rng(3))
    assert np.all(pos >= 0) and np.all(pos <= 100)
    assert pos[0] == 0           # I^sym=0 → π≈0
    assert pos[2] >= 90          # I^sym≫B → π≈1
