"""
simulation.sim.metapop_seirvd
=============================
Metapopulation SEIR-V-D simulator — the Stage 5 centerpiece.

Responsibilities
----------------
1. Build the initial state from ``MetapopParams``.
2. Loop through time, calling the intervention resolver once per sub-step
   and stepping with ``rk4_step``.
3. Record compartment trajectories at daily resolution.
4. Feed the result into the Stage-4 epi-validity gate and attach the
   report to the returned ``SimResult``.

The simulator does **not** read from the DB — its contract is a
populated ``MetapopParams``. The DB → params path lives in
``simulation.sim.io`` (separate module, kept small and safe_connect-only).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .foi import effective_daytime_population
from .interventions import (
    apply_interventions,
    expand_to_district_vector,
    resolve_active_interventions,
)
from .parameters import (
    COMPARTMENTS,
    IDX_D,
    IDX_E,
    IDX_I,
    IDX_R,
    IDX_S,
    IDX_V,
    InterventionSpec,
    MetapopParams,
    N_COMPARTMENTS,
)
from .stepper import expeuler_step, rk4_step


log = logging.getLogger(__name__)

__all__ = ["MetapopSEIRVD", "SimResult", "run_scenario"]


@dataclass
class SimResult:
    """Outcome of a metapop run.

    Attributes
    ----------
    state : (T, G, 6) ndarray, daily snapshots in (S, E, I, R, V, D) order
    days : (T,) ndarray of day indices
    district_names : list[str]
    incidence : (T, G) ndarray of new-infection counts per day
        (derived from σ·E — the flow into I)
    params : the ``MetapopParams`` used for the run (copy, not reference)
    interventions : the interventions that were applied
    epi_validity : dict from ``run_epi_validity_gate`` — populated only
        when ``run_validator=True``
    """
    state: np.ndarray
    days: np.ndarray
    district_names: list[str]
    incidence: np.ndarray
    params: MetapopParams
    interventions: list[InterventionSpec] = field(default_factory=list)
    epi_validity: dict = field(default_factory=dict)
    #: spec-index → day a reactive InterventionSpec activated (B-P5). Empty
    #: unless the run used reactive (state-triggered) interventions.
    reactive_activations: dict = field(default_factory=dict)

    # ── Convenience slicers ───────────────────────────────────────────
    @property
    def S(self) -> np.ndarray: return self.state[:, :, IDX_S]
    @property
    def E(self) -> np.ndarray: return self.state[:, :, IDX_E]
    @property
    def I(self) -> np.ndarray: return self.state[:, :, IDX_I]
    @property
    def R(self) -> np.ndarray: return self.state[:, :, IDX_R]
    @property
    def V(self) -> np.ndarray: return self.state[:, :, IDX_V]
    @property
    def D(self) -> np.ndarray: return self.state[:, :, IDX_D]

    def city_total(self, compartment: str) -> np.ndarray:
        """Sum a compartment across all districts → (T,) trajectory."""
        if compartment not in COMPARTMENTS:
            raise ValueError(
                f"unknown compartment {compartment!r}; "
                f"choose from {COMPARTMENTS}"
            )
        idx = COMPARTMENTS.index(compartment)
        return self.state[:, :, idx].sum(axis=1)

    def reported_cases(self, report_frac: Optional[float] = None) -> np.ndarray:
        """Observable weekly ILI cases ≈ σ·E scaled by report_frac.

        Applies the disease's ``report_frac`` by default so the output
        lines up with KDCA's sentinel surveillance magnitude.
        """
        frac = report_frac if report_frac is not None else self.params.disease.report_frac
        return self.incidence * float(frac)


class MetapopSEIRVD:
    """Deterministic metapop SEIR-V-D integrator.

    Instantiate once, call ``.run(interventions=...)`` to execute.
    Re-runs with a different intervention list reuse the same
    parameter object and initial conditions.

    G-184 syndromic-β framing (sprint 2026-05-06; paper §결론/§4.8/§6.4)
    -------------------------------------------------------------------
    The β parameter consumed here (via ``DiseaseParams.beta = R0·γ``)
    represents the **aggregate syndromic respiratory-pathogen
    transmission rate** — NOT influenza-specific R0. KDCA ILI =
    "fever ≥38°C + respiratory symptoms" syndromic indicator can
    include RSV / SARS-CoV-2 / hMPV / parainfluenza cocirculation.

    Paper reporting (paper §결론 정합):
    - Prefer Rt time-series (Cori 2013 EpiEstim) over single R0
    - Flag results with "syndromic" qualifier
    - Influenza-only proxy: I_t* = positivity × ILI (KDCA → WHO FluNet
      via Phase D.3 features `flu_positivity_lag1`)
    """

    def __init__(self, params: MetapopParams):
        params.validate()
        self.params = params
        self.G = int(np.asarray(params.populations).size)
        self.pops = np.asarray(params.populations, dtype=float)
        self.M = np.asarray(params.mobility, dtype=float)
        self.names = list(params.district_names) if params.district_names else [
            f"gu_{i}" for i in range(self.G)
        ]
        # Daytime populations are constant given M and N.
        self._daytime_pop = effective_daytime_population(self.M, self.pops)

    # ── Initial state ─────────────────────────────────────────────────
    def _build_initial_state(self) -> np.ndarray:
        """Shape (G, 6): distribute N across S/E/I/R/V/D from params."""
        state = np.zeros((self.G, N_COMPARTMENTS), dtype=float)

        I0 = self._as_vec(self.params.initial_infected, default=0.0)
        R0 = self._as_vec(self.params.initial_recovered, default=0.0)
        V0 = self._as_vec(self.params.initial_vaccinated, default=0.0)

        # Tiny E seed — otherwise a pure-S init with I=0 stays at zero.
        # Use σ-scaled E = I0/σ·γ approximation so the latent pool matches
        # the infectious pool at endemic equilibrium. Scales with I0, so
        # zero initial infected → zero seed.
        disease = self.params.disease
        E0 = I0 * (disease.gamma / max(disease.sigma, 1e-9))

        assigned = I0 + R0 + V0 + E0
        if np.any(assigned > self.pops):
            raise ValueError(
                "initial I + R + V + E exceeds population in some district; "
                "reduce initial_infected / initial_recovered / initial_vaccinated"
            )
        S0 = np.maximum(self.pops - assigned, 0.0)

        state[:, IDX_S] = S0
        state[:, IDX_E] = E0
        state[:, IDX_I] = I0
        state[:, IDX_R] = R0
        state[:, IDX_V] = V0
        # D starts at 0
        return state

    def _as_vec(self, x, *, default: float) -> np.ndarray:
        if x is None:
            return np.full(self.G, default)
        return expand_to_district_vector(x, self.G)

    # ── Main loop ─────────────────────────────────────────────────────
    def run(
        self,
        interventions: Optional[Iterable[InterventionSpec]] = None,
        *,
        run_validator: bool = True,
        backend: str = "numba",
    ) -> SimResult:
        """Step the model forward for ``params.days`` days.

        Parameters
        ----------
        interventions
            Iterable of ``InterventionSpec``. May be empty; None is
            treated as empty.
        run_validator
            When True (default) the resulting trajectory is passed
            through ``simulation.verifier.epi_validity.run_epi_validity_gate``
            and the report is attached to ``SimResult.epi_validity``.
        backend
            "numba" (default) | "c" | "rust". Controls which implementation
            runs the inner sub-step loop. "c" uses the batched C kernel
            (rk4_step_batch_c) for ~2.5-3× speedup on sim-heavy workloads
            (Monte Carlo, scenario sweeps). Incidence is computed via a
            trapezoid approximation (initial + final E averaged over the
            day) — typical deviation < 1% vs the Numba path, acceptable
            for Monte Carlo but prefer "numba" for paper-grade single runs.
        """
        interventions = list(interventions or [])
        # B-P5: reactive (state-triggered) interventions activate when the live
        # epidemic crosses a threshold, not on a fixed calendar day. We only pay
        # the per-day metric cost when a reactive spec is actually present.
        _has_reactive = any(s.trigger is not None for s in interventions)
        _reactive_fired: dict[int, int] = {}
        days = int(self.params.days)
        dt = float(self.params.dt)
        if dt <= 0:
            raise ValueError(f"dt must be positive; got {dt}")
        if days <= 0:
            raise ValueError(f"days must be positive; got {days}")
        sub_steps_per_day = int(round(1.0 / dt))
        if abs(sub_steps_per_day * dt - 1.0) > 1e-9:
            raise ValueError(
                f"dt={dt} must divide a day evenly (got {sub_steps_per_day} sub-steps)"
            )

        state = self._build_initial_state()
        out_state = np.zeros((days + 1, self.G, N_COMPARTMENTS), dtype=float)
        incidence = np.zeros((days + 1, self.G), dtype=float)
        out_state[0] = state
        # Day 0 incidence is seeded E × σ per day
        incidence[0] = state[:, IDX_E] * self.params.disease.sigma

        # Resolve backend options. 'c' requires the native extension; fall
        # back to 'numba' if unavailable (logged, not raised, for portability).
        use_c_batched = False
        if backend == "c":
            try:
                from simulation.sim.stepper import rk4_step_c, HAS_C_BACKEND
                if HAS_C_BACKEND:
                    use_c_batched = True
                else:
                    log.warning("backend='c' requested but HAS_C_BACKEND=False; "
                                "falling back to 'numba'. Build via simulation/c/build.sh.")
            except ImportError:
                log.warning("rk4_step_c unavailable; falling back to 'numba'.")

        # B-P1 (M7): opt-in flux-conservative exp-Euler integrator (unconditionally
        # positive + mass-conserving — no box-clamp / nan_to_num, no divergence).
        # RK4 stays default until validated on the §4.16 grid; the stable path
        # forces the numpy/numba loop (the C batched kernel is RK4-only).
        import os as _os
        _step_fn = rk4_step
        if _os.environ.get("MPH_STABLE_INTEGRATOR") == "1":
            _step_fn = expeuler_step
            use_c_batched = False
            log.info("  [sim] MPH_STABLE_INTEGRATOR=1 → flux-conservative exp-Euler")

        # Native batched kernel (lazy import so non-C installs still work).
        _batch_fn = None
        if use_c_batched:
            import ctypes
            import sys as _sys
            from pathlib import Path as _Path
            _ext = {"darwin": "dylib", "linux": "so"}.get(_sys.platform, "so")
            _lib_path = _Path(__file__).resolve().parent.parent / "c" / f"seir_core.{_ext}"
            _lib = ctypes.CDLL(str(_lib_path))
            _DP = ctypes.POINTER(ctypes.c_double)
            _batch_fn = _lib.rk4_step_batch_c
            _batch_fn.argtypes = [
                _DP, _DP, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                _DP, _DP, _DP, _DP,
            ]
            _batch_fn.restype = None

        for d in range(days):
            day_specs = interventions
            if _has_reactive:
                # City-wide trigger metrics from the current (start-of-day) state.
                _Ntot = self.pops.sum()
                metric_values = {
                    "prevalence": float(state[:, IDX_I].sum() / _Ntot),
                    "incidence": float(state[:, IDX_E].sum() / _Ntot),
                }
                day_specs = resolve_active_interventions(
                    interventions, d, metric_values, _reactive_fired,
                )
            disease, vax_vec = apply_interventions(
                d, self.params.disease, self.params.vaccination_rate,
                day_specs, self.G,
            )
            kwargs = {
                "beta": disease.beta,
                "sigma": disease.sigma,
                "gamma": disease.gamma,
                "omega": disease.omega,
                "VE": disease.VE,
                "V_waning": disease.V_waning,
                "ifr": disease.ifr,
                "vax_rate": vax_vec,
                "populations": self.pops,
                "mobility": self.M,
                "daytime_pop": self._daytime_pop,
            }

            if use_c_batched:
                # Single batched C call for all sub_steps of this day.
                # Note: ctypes / _DP / _batch_fn are set up ONCE above the day
                # loop (see `if use_c_batched` block before the loop) — do NOT
                # re-import or re-declare them here (pref audit, 2026-04-24).
                # Trapezoid incidence: avg(E_initial, E_final) * sigma * 1 day.
                E_initial = state[:, IDX_E].copy()
                s_in = np.ascontiguousarray(state, dtype=np.float64)
                s_out = np.empty_like(s_in)
                vax = np.ascontiguousarray(vax_vec, dtype=np.float64)
                pop = np.ascontiguousarray(self.pops, dtype=np.float64)
                mob = np.ascontiguousarray(self.M, dtype=np.float64)
                day_arr = np.ascontiguousarray(self._daytime_pop, dtype=np.float64)
                _batch_fn(
                    s_in.ctypes.data_as(_DP),
                    s_out.ctypes.data_as(_DP),
                    ctypes.c_int(self.G),
                    ctypes.c_int(sub_steps_per_day),
                    ctypes.c_double(dt),
                    ctypes.c_double(disease.beta),
                    ctypes.c_double(disease.sigma),
                    ctypes.c_double(disease.gamma),
                    ctypes.c_double(disease.omega),
                    ctypes.c_double(disease.VE),
                    ctypes.c_double(disease.V_waning),
                    ctypes.c_double(disease.ifr),
                    vax.ctypes.data_as(_DP),
                    pop.ctypes.data_as(_DP),
                    mob.ctypes.data_as(_DP),
                    day_arr.ctypes.data_as(_DP),
                )
                state = s_out
                E_final = state[:, IDX_E]
                # Trapezoid-rule incidence: avg(initial, final) * sigma * 1.0 day
                daily_incidence = 0.5 * (E_initial + E_final) * disease.sigma
            else:
                # Numba / pure-numpy path: exact sub-step incidence integration.
                daily_incidence = np.zeros(self.G, dtype=float)
                for _ in range(sub_steps_per_day):
                    # σ·E integrated over the sub-step as a rectangle rule.
                    daily_incidence += disease.sigma * state[:, IDX_E] * dt
                    state = _step_fn(state, dt, params_kwargs=kwargs)

            out_state[d + 1] = state
            incidence[d + 1] = daily_incidence

        result = SimResult(
            state=out_state,
            days=np.arange(days + 1),
            district_names=self.names,
            incidence=incidence,
            params=self.params,
            interventions=interventions,
            reactive_activations=_reactive_fired,
        )

        if run_validator:
            result.epi_validity = self._run_gate(result)
        return result

    # ── Validator glue ────────────────────────────────────────────────
    def _run_gate(self, result: "SimResult") -> dict:
        """Delegate to the Stage-4 epi-validity gate.

        We feed the city-wide infectious trajectory as the prediction
        series, along with compartment dict + N_total for the
        conservation check. Errors are logged, never raised.
        """
        try:
            from simulation.verifier.epi_validity import run_epi_validity_gate

            # City-wide totals (T,)
            city = {c: result.city_total(c) for c in COMPARTMENTS}
            N_total = float(self.pops.sum())
            outputs = {
                "metapop_seirvd": {
                    "predictions": city["I"],
                    "compartments": city,
                    "N_total": N_total,
                    "params": {
                        "R0": self.params.disease.R0,
                        "gamma": self.params.disease.gamma,
                        "sigma": self.params.disease.sigma,
                        "VE": self.params.disease.VE,
                        "ifr": self.params.disease.ifr,
                    },
                }
            }
            # Rt(t) via Cori 2013 EpiEstim on city-wide daily incidence (B-P2):
            # wire the *existing* RtEstimator into the gate so check_rt_sequence
            # fires — it was dead code because _run_gate never supplied `rt`.
            # Validate where Rt is estimable (drop the pre-growth NaN windows,
            # standard EpiEstim practice; default single-wave runs have only
            # leading NaN so the |ΔRt| continuity check stays meaningful).
            try:
                from simulation.models.rt_estimator import RtEstimator

                city_incidence = result.incidence.sum(axis=1)
                rt_series = (
                    RtEstimator(window_size=7)
                    .estimate(city_incidence)["Rt_mean"]
                    .to_numpy()
                )
                rt_finite = rt_series[np.isfinite(rt_series)]
                if rt_finite.size:
                    outputs["metapop_seirvd"]["rt"] = rt_finite
            except Exception as e:   # Rt is advisory — never block the gate over it
                log.warning("Rt estimation for gate skipped: %s", e)
            # B-P6 (M7): derive ISO weeks from a season-start anchor so the
            # seasonal-peak checker fires (was dead — _run_gate never supplied
            # iso_weeks). Day 0 = season start (default ISO W36, NH influenza
            # onset); a synthetic flu wave should then peak inside the W48-W8
            # window. Real-anchored runs override via MPH_SIM_SEASON_START_WEEK.
            try:
                import os as _os
                _w0 = int(_os.environ.get("MPH_SIM_SEASON_START_WEEK", "36"))
                _T = len(city["I"])
                outputs["metapop_seirvd"]["iso_weeks"] = np.array(
                    [((_w0 - 1 + d // 7) % 53) + 1 for d in range(_T)], dtype=np.int64)
            except Exception as e:
                log.debug("iso_weeks derivation for gate skipped: %s", e)
            return run_epi_validity_gate(outputs)
        except Exception as e:   # never kill a run over a validator hiccup
            log.warning("epi-validity gate skipped: %s", e)
            return {"_error": str(e)}


def run_scenario(
    name: str,
    params: Optional[MetapopParams] = None,
    *,
    overrides: Optional[dict] = None,
) -> SimResult:
    """Lookup ``name`` in the scenario registry, merge overrides, and run.

    ``overrides`` is a dict applied on top of the scenario's own
    intervention/param adjustments, e.g. ``{"days": 120}`` to shorten a
    run without editing the scenario. Keys matching ``MetapopParams``
    fields get set on the params object; everything else is ignored
    with a debug log.
    """
    # Import inside the function to avoid circular imports at package load
    from .scenarios import SCENARIO_REGISTRY

    if name not in SCENARIO_REGISTRY:
        raise KeyError(
            f"unknown scenario {name!r}; "
            f"registered: {sorted(SCENARIO_REGISTRY)}"
        )
    builder = SCENARIO_REGISTRY[name]
    resolved_params, interventions = builder(params)

    if overrides:
        for k, v in overrides.items():
            if hasattr(resolved_params, k):
                setattr(resolved_params, k, v)
            else:
                log.debug("ignoring unknown override key %r", k)

    sim = MetapopSEIRVD(resolved_params)
    return sim.run(interventions=interventions)
