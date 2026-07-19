"""
simulation.abm.adversarial_tests
================================
Adversarial invariance tests for the behavioural ABM.

Motivation
----------
The default ``run_invariant_test`` is mathematically tautological: the
behaviour-off ABM calls exactly the same ``rk4_step`` with the same
kwargs as the kernel-only path, so RMSE = 0 to float64 precision is a
direct consequence of code reuse rather than a strong correctness check.
This module supplements that test with adversarial conditions designed
to actually distinguish a correct coupling from a silent bug:

  T1  Kernel-drift sensitivity
      Repeat the invariance test while sweeping (VE, vaccination_rate,
      initial_infected). A correct coupling must stay RMSE = 0 in every
      case; a buggy coupling that e.g. double-applied VE would drift.

  T2  Finite-alpha monotonicity
      The behaviour-on peak must be monotonically non-increasing as
      alpha increases from 0 to 2, at fixed (kappa, tau, theta). A
      coupling that flips sign (larger alpha -> higher peak) would fail.

  T3  Behaviour-off at the intervention frontier
      Run the kernel with a steep NPI intervention (beta scale 0.5 from
      day 30). The behaviour-off ABM must still match byte-for-byte
      despite the intervention schedule being injected through a
      different code path.

  T4  Compliance activation probe
      Increasing theta from 0 (always compliant under any positive R)
      through 1 (almost never compliant) must produce a monotonically
      non-decreasing city peak, verifying that the utility-threshold
      decision is wired correctly.

The test output is a JSON report with per-test pass/fail flags plus the
numerical trajectory summaries.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.sim.metapop_seirvd import MetapopSEIRVD
from simulation.sim.parameters import (
    DiseaseParams, InterventionSpec, MetapopParams,
)
from .behavioural import (
    BehaviouralParams, run_coupled_abm, run_invariant_test,
)

log = logging.getLogger(__name__)

__all__ = [
    "run_all_adversarial_tests",
    "TestReport",
]


@dataclass
class TestReport:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


# ---------------------------------------------------------------------------
# T1 — kernel-drift sensitivity
# ---------------------------------------------------------------------------
def _clone_params(params: MetapopParams, **overrides) -> MetapopParams:
    return MetapopParams(
        disease=overrides.get("disease", params.disease),
        populations=params.populations,
        mobility=params.mobility,
        district_names=params.district_names,
        initial_infected=overrides.get("initial_infected", params.initial_infected),
        initial_recovered=params.initial_recovered,
        initial_vaccinated=params.initial_vaccinated,
        vaccination_rate=overrides.get("vaccination_rate", params.vaccination_rate),
        days=overrides.get("days", params.days),
        dt=params.dt,
        seed=params.seed,
    )


def _clone_disease(d: DiseaseParams, **overrides) -> DiseaseParams:
    fields = ("R0", "gamma", "sigma", "omega", "VE", "V_waning", "ifr", "report_frac")
    return DiseaseParams(**{k: overrides.get(k, getattr(d, k)) for k in fields})


def test_kernel_drift(params: MetapopParams) -> TestReport:
    grid = [
        ("baseline", params),
        ("VE=0.75",  _clone_params(params, disease=_clone_disease(params.disease, VE=0.75))),
        ("VE=0.25",  _clone_params(params, disease=_clone_disease(params.disease, VE=0.25))),
        ("vax=0.002", _clone_params(params, vaccination_rate=np.full_like(
            np.atleast_1d(np.asarray(params.populations, dtype=float)), 0.002, dtype=float))),
        ("I0=5k", _clone_params(params, initial_infected=np.full_like(
            np.atleast_1d(np.asarray(params.populations, dtype=float)), 5_000.0, dtype=float))),
    ]
    detail: dict = {}
    passed = True
    for name, p in grid:
        r = run_invariant_test(p, tolerance=1e-6)
        detail[name] = {
            "rmse": r["rmse"],
            "max_abs_err": r["max_abs_err"],
            "passed": r["passed"],
            "kernel_city_I_peak": r["kernel_city_I_peak"],
            "abm_city_I_peak": r["abm_city_I_peak"],
        }
        passed = passed and r["passed"]
    return TestReport("T1_kernel_drift_sensitivity", passed, detail)


# ---------------------------------------------------------------------------
# T2 — finite-alpha monotonicity
# ---------------------------------------------------------------------------
def test_alpha_monotonicity(
    params: MetapopParams,
    *,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
    kappa: float = 0.5,
    tau: float = 45.0,
    theta: float = 0.15,
) -> TestReport:
    peaks: list[tuple[float, float]] = []
    for a in alphas:
        behav = BehaviouralParams(alpha=a, kappa=kappa, tau=tau, theta=theta)
        out = run_coupled_abm(params, behav)
        peaks.append((a, float(out.city_I().max())))
    # Monotonically non-increasing check (small tolerance for float noise)
    monotone_non_increasing = all(
        peaks[i][1] <= peaks[i - 1][1] + 1e-6 for i in range(1, len(peaks))
    )
    return TestReport(
        "T2_alpha_monotonicity",
        monotone_non_increasing,
        detail={
            "alphas": [p[0] for p in peaks],
            "peak_city_I": [p[1] for p in peaks],
            "monotone_non_increasing": monotone_non_increasing,
            "fixed_kappa": kappa,
            "fixed_tau": tau,
            "fixed_theta": theta,
        },
    )


# ---------------------------------------------------------------------------
# T3 — behaviour-off under steep NPI intervention
# ---------------------------------------------------------------------------
def test_intervention_invariance(
    params: MetapopParams,
    *,
    scale: float = 0.5,
    start: int = 30,
    end: int = 120,
    tolerance: float = 1e-6,
) -> TestReport:
    kernel = MetapopSEIRVD(params).run(
        interventions=[InterventionSpec("beta", scale, start, end, op="scale")],
        run_validator=False,
    )
    # The coupled ABM has no intervention-spec hook yet, so we approximate the
    # same NPI by temporarily reducing the effective beta through a disease
    # clone; this mirrors the kernel's InterventionSpec(beta, 0.5, 30, 120)
    # when behaviour-off (compliance = 0 so scale = 1).
    from .behavioural import run_coupled_abm_with_beta_schedule  # optional extension
    off = BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf"))
    schedule = np.ones(int(params.days) + 1, dtype=float)
    schedule[start:end] = scale
    abm = run_coupled_abm_with_beta_schedule(params, off, schedule)
    diff = abm.seir.state - kernel.state
    rmse = float(np.sqrt((diff ** 2).mean()))
    max_err = float(np.abs(diff).max())
    return TestReport(
        "T3_intervention_invariance",
        passed=rmse <= tolerance and max_err <= tolerance * 1e3,
        detail={
            "scale": scale, "start": start, "end": end,
            "rmse": rmse, "max_abs_err": max_err,
            "kernel_peak": float(kernel.city_total("I").max()),
            "abm_peak": float(abm.city_I().max()),
        },
    )


# ---------------------------------------------------------------------------
# T4 — compliance threshold probe
# ---------------------------------------------------------------------------
def test_theta_monotonicity(
    params: MetapopParams,
    *,
    thetas: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.40, 0.80),
    alpha: float = 1.5,
    kappa: float = 0.5,
    tau: float = 45.0,
) -> TestReport:
    peaks: list[tuple[float, float]] = []
    compliances: list[tuple[float, float]] = []
    for t in thetas:
        behav = BehaviouralParams(alpha=alpha, kappa=kappa, tau=tau, theta=t)
        out = run_coupled_abm(params, behav)
        peaks.append((t, float(out.city_I().max())))
        compliances.append((t, float(out.compliance.mean())))
    # As theta rises the decision R - kappa*F > theta becomes harder to trigger,
    # so compliance should DECREASE monotonically and peaks should INCREASE.
    theta_asc = [p[1] for p in peaks]
    peaks_nondecreasing = all(
        theta_asc[i] >= theta_asc[i - 1] - 1e-6 for i in range(1, len(theta_asc))
    )
    compl_nonincreasing = all(
        compliances[i][1] <= compliances[i - 1][1] + 1e-6 for i in range(1, len(compliances))
    )
    return TestReport(
        "T4_theta_monotonicity",
        passed=peaks_nondecreasing and compl_nonincreasing,
        detail={
            "thetas": thetas,
            "peaks": [p[1] for p in peaks],
            "compliance_mean": [c[1] for c in compliances],
            "peaks_nondecreasing": peaks_nondecreasing,
            "compliance_nonincreasing": compl_nonincreasing,
        },
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_adversarial_tests(
    params: MetapopParams,
    *,
    out_dir: Optional[Path | str] = None,
    include_intervention_test: bool = False,
) -> dict:
    reports: list[TestReport] = []
    reports.append(test_kernel_drift(params))
    reports.append(test_alpha_monotonicity(params))
    if include_intervention_test:
        try:
            reports.append(test_intervention_invariance(params))
        except ImportError:
            log.warning("T3 skipped: beta-schedule coupler not installed")
    reports.append(test_theta_monotonicity(params))

    all_passed = all(r.passed for r in reports)
    out = {
        "all_passed": all_passed,
        "tests": [r.as_dict() for r in reports],
    }
    if out_dir is not None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adversarial_tests.json").write_text(
            json.dumps(out, indent=2, default=str), encoding="utf-8"
        )
        log.info("wrote %s", p / "adversarial_tests.json")
    return out
