"""Age-stratified force-of-infection (FOI) lock — agent-kernel rich-population path.

박제 (what this locks):
  - DIMENSION #4. agent_kernel.py 의 rich-population 경로가 CONTACT_MATRIX_7x7 의 age
    contact-row factor 를 susceptible infection rate(FOI/lambda_gu)에 실제로 곱한다는 것
    (코드: _step_population_day susceptibility = occupation * age_contact_factor *
    contact_multiplier → infection_rate = beta * susceptibility * phase_prev). 즉 연령
    혼합(age mixing)이 전파를 정말 **변조**한다 — silent하게 무시되지 않는다.

  메커니즘 정밀 진술 (이 lock 이 인코딩하는 사실):
    age_contact_factor = CONTACT_MATRIX_7x7 의 row-mean[age_band] 을 **agent 집단 평균 1로
    정규화**한 per-agent scalar. 따라서
      (i) 동질 집단(모두 같은 band) → 모든 factor=1 → age-중립 (정규화 때문에 단일 band 를
          shift 해도 FOI 불변).
      (ii) age 효과는 집단의 **이질성(composition × row-mean 분산)** 에서만 나온다.
      (iii) row-mean 이 모두 같은 행렬(예: 대각 가중 동일합) → 정규화 후 전부 1 → age 무효.
    이 test 들은 (i)-(iii) 을 직접 확인하여, "age 가 FOI 를 변조한다"는 주장을 정확한
    조건과 함께 박제한다 (과대주장 회피 = 프로젝트 규칙).

  SCOPE (정직): 이것은 **agent-level age weighting on a single per-gu FOI** 이다 — 완전한
  age × age WAIFW 행렬 전개(연령별 별도 compartment + 연령쌍 force-of-infection)가 아니다.
  metapop ODE 경로(behavioural.MetapopSEIRVD / run_coupled_abm)는 여전히 age-agnostic
  (branch B deeper stratification = future scope, stratified_validation.py docstring).

Synthetic params only (no DB). alpha_mean=0 으로 행동 feedback 차단 → age 효과 고립.
Run:  .venv/bin/python simulation/tests/test_age_foi_modulation.py
"""
from __future__ import annotations

import sys

import numpy as np

from simulation.abm.agent_kernel import (
    run_agent_world,
    _age_contact_factor_by_agent,
)
from simulation.abm.contact_structure import CONTACT_MATRIX_7x7

# --- toy epidemic config (small, fast; behavioural feedback OFF to isolate age) ---
_N = 3000
_T = 60
_COMMON = dict(
    N=_N, T_days=_T, beta=0.9, sigma=0.5, gamma=0.25, delta=0.001, nu=0.0,
    global_seed=7, theta_sd=0.0, theta_mean=0.5, alpha_mean=0.0,
)


def _make_pop(age_bands, seed=1):
    """Synthetic SoA population with the given per-agent age bands.

    home/work gu randomised; occupation neutral ('office' multiplier 1.0);
    severity all low — so the ONLY structured lever is age_band.
    """
    rng = np.random.default_rng(seed)
    N = len(age_bands)
    return {
        "home_gu": rng.integers(0, 25, size=N).astype(np.int64),
        "work_gu": rng.integers(0, 25, size=N).astype(np.int64),
        "age_band": np.asarray(age_bands, dtype=np.int64),
        "occupation": np.array(["office"] * N, dtype=object),
        "severity": np.zeros(N, dtype=np.int64),
    }


def _peak(out):
    return int(np.max(out["I"]))


def _cum_infected(out):
    """Cumulative infections ever ≈ everyone who left S to a non-V state."""
    return int(out["R"][-1] + out["D"][-1] + out["I"][-1] + out["E"][-1])


# ────────────────────────────────────────────────────────────────────────────
# (a) CONTACT_MATRIX_7x7 is a valid 7×7 normalized contact structure
# ────────────────────────────────────────────────────────────────────────────
def test_contact_matrix_is_valid_7x7_structure():
    M = CONTACT_MATRIX_7x7
    assert M.shape == (7, 7), f"contact matrix must be 7×7 (7 age bands), got {M.shape}"
    assert np.all(np.isfinite(M)), "contact matrix must be all-finite"
    assert np.all(M > 0.0), "contact rates must be strictly positive"
    # POLYMOD-like symmetric mixing assumption.
    assert np.allclose(M, M.T), "contact matrix must be symmetric (POLYMOD-like)"
    # Anchored to the Korean 2023-24 ILI survey: overall mean ≈ 4.81 contacts/day.
    assert abs(float(M.mean()) - 4.81) < 1e-6, (
        f"overall mean contacts/day must be the 4.81 anchor, got {float(M.mean())}"
    )
    # Row-means (the quantity that becomes the age factor) genuinely vary across
    # bands — i.e. there IS age structure to modulate FOI (not a flat matrix).
    rm = M.mean(axis=1)
    assert rm.max() / rm.min() > 1.5, (
        "row-means must differ across age bands for age to have any FOI effect; "
        f"got ratio {rm.max() / rm.min():.3f}"
    )
    # Documented shape of the assumption: school-age band (10-19, idx 1) is the
    # highest-contact band; eldest (60+, idx 6) the lowest.
    assert int(np.argmax(rm)) == 1, "expected band 1 (10-19) to be highest-contact"
    assert int(np.argmin(rm)) == 6, "expected band 6 (60+) to be lowest-contact"


# ────────────────────────────────────────────────────────────────────────────
# (a') The per-agent age factor used inside FOI is normalized to mean 1
#       and ORDERS bands by their contact row-mean.
# ────────────────────────────────────────────────────────────────────────────
def test_age_contact_factor_normalized_and_ordered():
    # Equal-mix of all 7 bands → factors normalize to mean 1.
    bands = np.repeat(np.arange(7), 100)
    f = _age_contact_factor_by_agent(bands)
    assert abs(float(f.mean()) - 1.0) < 1e-9, "age factors must be mean-normalized to 1"
    # The per-band factor must track the contact row-mean ordering: the highest-
    # contact band (1) gets the largest factor, the lowest (6) the smallest.
    fac_by_band = {b: float(f[bands == b][0]) for b in range(7)}
    assert fac_by_band[1] == max(fac_by_band.values()), "band 1 must get the highest factor"
    assert fac_by_band[6] == min(fac_by_band.values()), "band 6 must get the lowest factor"

    # Homogeneous population (single band) → ALL factors = 1 (age-neutral), for ANY
    # band. This is the documented normalization behaviour, and it is WHY a clean
    # age effect needs a heterogeneous composition (see tests below).
    for b in (0, 3, 6):
        fb = _age_contact_factor_by_agent(np.full(500, b, dtype=np.int64))
        assert np.allclose(fb, 1.0), f"homogeneous band-{b} pop must be age-neutral (all 1)"


# ────────────────────────────────────────────────────────────────────────────
# (b) Shifting age composition toward / away from high-contact bands changes the
#     epidemic — heterogeneous mix ≠ flat (age-removed) on the SAME population.
# ────────────────────────────────────────────────────────────────────────────
def test_age_composition_changes_epidemic_vs_flat():
    """Same population & seed; only the age contact matrix differs.

    A heterogeneous (band-1 + band-6) population run under the REAL POLYMOD-like
    matrix must produce a *different* epidemic than the same population run under a
    FLAT matrix (which removes all age structure → every factor = 1). If age were
    silently ignored, these two would be identical.
    """
    mix = np.array([1] * (_N // 2) + [6] * (_N - _N // 2))
    flat = np.ones((7, 7))
    out_real = run_agent_world(population=_make_pop(mix), contact_matrix=None, **_COMMON)
    out_flat = run_agent_world(population=_make_pop(mix), contact_matrix=flat, **_COMMON)
    # Real age mixing must measurably move BOTH peak and cumulative attack size.
    assert _peak(out_real) != _peak(out_flat), (
        f"age matrix had NO effect on peak (real={_peak(out_real)} == "
        f"flat={_peak(out_flat)}) — age silently ignored"
    )
    assert _cum_infected(out_real) != _cum_infected(out_flat), (
        "age matrix had NO effect on cumulative attack size — age silently ignored"
    )


def test_high_age_dispersion_shifts_peak_in_expected_direction():
    """Amplifying the across-band contact CONTRAST changes the epidemic strongly,
    and in the expected direction.

    Population = half super-high-contact (band 1) + half super-low-contact
    (band 6). A high-dispersion matrix (band-1 row-mean ≫ band-6 row-mean)
    concentrates transmission in the band-1 half (which saturates) and leaves the
    band-6 half as a low-hazard near-refractory pool. Result: a LOWER, structured
    peak than a flat (homogeneous-mixing) matrix on the identical population.

    This is the directional age-modulation lock: increasing age contact contrast
    lowers the peak of a half-and-half high/low population.
    """
    mix = np.array([1] * (_N // 2) + [6] * (_N - _N // 2))
    flat = np.ones((7, 7))
    # Row-means differ a lot → after mean-1 normalization, band-1 agents carry a
    # much larger contact factor than band-6 agents.
    row_scale = np.array([1.0, 8.0, 4.0, 3.0, 2.0, 1.0, 0.3])
    disp = np.outer(row_scale, np.ones(7))  # row_mean[i] = row_scale[i]

    out_flat = run_agent_world(population=_make_pop(mix), contact_matrix=flat, **_COMMON)
    out_disp = run_agent_world(population=_make_pop(mix), contact_matrix=disp, **_COMMON)

    assert _peak(out_disp) < _peak(out_flat), (
        f"high age-contact dispersion should lower the peak of a half-high/half-low "
        f"population; got disp peak={_peak(out_disp)} >= flat peak={_peak(out_flat)}"
    )
    # The effect must be substantial (not numerical noise): > 20% peak reduction.
    rel = (_peak(out_flat) - _peak(out_disp)) / _peak(out_flat)
    assert rel > 0.20, f"age dispersion effect too small to be real: {rel:.1%} peak change"


# ────────────────────────────────────────────────────────────────────────────
# (c) Age has a REAL effect — only row-mean DISPERSION across bands matters
#     (a matrix with equal row-means is, after normalization, age-neutral).
# ────────────────────────────────────────────────────────────────────────────
def test_equal_rowmean_matrix_is_age_neutral():
    """A matrix whose row-means are all EQUAL normalizes to all-1 factors → age
    has no effect (epidemic identical to flat). This pins the *exact* mechanism:
    FOI age modulation comes from across-band row-mean DISPERSION, nothing else.
    Guards against a false-positive reading of (b)/(b')."""
    mix = np.array([1] * (_N // 2) + [6] * (_N - _N // 2))
    flat = np.ones((7, 7))
    # Diagonal-heavy but every row sums to the same total → equal row-means.
    equal_rowmean = np.eye(7) * 10.0 + 0.1
    assert np.allclose(equal_rowmean.mean(axis=1), equal_rowmean.mean(axis=1)[0]), \
        "construction error: rows must have equal means"

    out_flat = run_agent_world(population=_make_pop(mix), contact_matrix=flat, **_COMMON)
    out_eq = run_agent_world(population=_make_pop(mix), contact_matrix=equal_rowmean, **_COMMON)
    assert _peak(out_eq) == _peak(out_flat), (
        "equal-row-mean matrix must be age-neutral after normalization "
        f"(eq peak={_peak(out_eq)} != flat peak={_peak(out_flat)})"
    )


def test_age_effect_is_deterministic_and_reproducible():
    """Same population + matrix + seed → identical epidemic (age effect is a
    deterministic function of inputs, not a stochastic artifact)."""
    mix = np.array([1] * (_N // 2) + [6] * (_N - _N // 2))
    a = run_agent_world(population=_make_pop(mix), contact_matrix=None, **_COMMON)
    b = run_agent_world(population=_make_pop(mix), contact_matrix=None, **_COMMON)
    assert _peak(a) == _peak(b) and _cum_infected(a) == _cum_infected(b), \
        "age-modulated epidemic must be reproducible for fixed inputs"


# ────────────────────────────────────────────────────────────────────────────
# SCOPE GUARD: the metapop ODE path (behavioural / MetapopSEIRVD) is age-AGNOSTIC
#   — branch B deeper stratification is future scope, NOT done. This test
#   documents/locks the honest boundary so the agent-level lock above is not
#   mis-read as "full age-stratified compartmental FOI".
# ────────────────────────────────────────────────────────────────────────────
def test_metapop_ode_path_is_age_agnostic():
    """The ODE behavioural path has no age dimension (no 7-band compartments, no
    CONTACT_MATRIX use). This is the documented future-scope boundary
    (stratified_validation.py: 'branch B deeper stratification')."""
    import simulation.abm.behavioural as bh
    src = bh.__file__
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()
    # The ODE state class must not import/use the 7×7 age contact matrix.
    assert "CONTACT_MATRIX_7x7" not in text, (
        "behavioural ODE unexpectedly references the age contact matrix — if age "
        "stratification was added to the ODE path, update this scope lock"
    )
    assert "age_band" not in text, (
        "behavioural ODE unexpectedly references age_band — update this scope lock "
        "if branch B (age-stratified ODE) landed"
    )


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn()
            print(f"  ✓ PASS  {fn.__name__}")
            p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}")
            f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
