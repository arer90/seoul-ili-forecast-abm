"""
simulation.sim.parameters
=========================
Typed containers for the Stage 5 metapop simulator.

Design notes
------------
- The compartment order ``(S, E, I, R, V, D)`` is **frozen**. Downstream
 consumers import ``COMPARTMENTS`` instead of hard-coding the order.
- All rates are per day. Weekly I/O is a view concern (see ``metapop_seirvd.py``).
- ``MetapopParams`` is a pure data object — no DB access, no I/O. The
 companion loader in ``simulation.sim.io`` fills these from ``epi_real_seoul.db``
 via ``safe_connect``.
- ``InterventionSpec`` is a *declarative* intervention. The simulator
 resolves it into per-day multipliers at step time, so ``start_day`` and
 ``end_day`` are zero-indexed absolute day offsets within the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ── Frozen compartment layout ──────────────────────────────────────────
COMPARTMENTS: tuple[str, str, str, str, str, str] = (
    "S", "E", "I", "R", "V", "D",
)
N_COMPARTMENTS: int = len(COMPARTMENTS)  # 6

# Pre-computed axis indices so downstream modules avoid ``.index()`` churn.
IDX_S, IDX_E, IDX_I, IDX_R, IDX_V, IDX_D = range(N_COMPARTMENTS)


@dataclass(frozen=True)
class DiseaseParams:
    """Per-disease biology. Defaults tuned for seasonal influenza.

    Ranges here must stay inside ``simulation.verifier.epi_validity.EPI_RANGE``.
    Callers who need to explore outside those ranges should pass
    ``strict=False`` to the validator — the module won't silently clamp.
    """
    # Transmission
    R0: float = 1.4              # basic reproduction number (Biggerstaff 2014)
    gamma: float = 1.0 / 3.5     # I→R rate (3.5-day infectious period, Tuite 2010)
    sigma: float = 1.0 / 2.0     # E→I rate (2-day latent period, Carrat 2008)
    # Immunity / vaccination
    omega: float = 1.0 / 180.0   # R→S waning (half-life ~6 months for flu)
    VE: float = 0.50             # vaccine efficacy, leaky (Stage-4 floor; CDC TND 2010-2023 match-year median)
    V_waning: float = 1.0 / 180.0   # V→S waning (mirrors natural immunity)
    # Mortality
    ifr: float = 0.001           # I→D rate expressed as fractional flow (cf. γ·ifr)
    # Case ascertainment (used only when producing observable "reported cases")
    report_frac: float = 0.10    # fraction of I detected via KDCA sentinel

    @property
    def beta(self) -> float:
        """β = R0 · γ — classical SIR / mobility-coupled relation.

        G-184 syndromic-β caveat (sprint 2026-05-06; paper §결론 / §4.8 / §6.4):
        - In the MPH-Seoul ILI framework this β represents the **aggregate
          syndromic respiratory-pathogen transmission rate**, NOT a
          influenza-specific R0. KDCA ILI = "fever ≥38°C + respiratory
          symptoms" syndromic indicator can include RSV / SARS-CoV-2 /
          hMPV / parainfluenza co-circulation effects.
        - Paper reporting: prefer Rt time-series (Cori 2013 EpiEstim) over
          single R0; flag with "syndromic" qualifier.
        - Influenza-only proxy (paper §sensitivity): I_t* = positivity_t ×
          ILI_rate(t), positivity from KDCA → WHO FluNet (Phase D.3
          features `flu_positivity_lag1`).
        """
        return self.R0 * self.gamma


@dataclass
class MetapopParams:
    """All inputs required to instantiate and run ``MetapopSEIRVD``.

    Attributes
    ----------
    disease
        Biology (R0, γ, σ, ω, VE, …). See ``DiseaseParams``.
    populations
        Length-G array of each district's resident population.
    mobility
        (G, G) row-stochastic mobility matrix. ``mobility[i, j]`` is the
        fraction of district i's residents present in j during the
        contact phase of a given day. Self-loops ``mobility[i, i]`` must
        be positive (nobody who never leaves home).
    district_names
        Length-G list of names (used for labeled output).
    initial_infected
        Length-G array of I(t=0) counts. Remaining population goes to S
        unless ``initial_recovered`` / ``initial_vaccinated`` are set.
    initial_recovered, initial_vaccinated
        Optional length-G arrays.
    vaccination_rate
        Length-G or scalar daily per-capita vaccination rate applied to S.
        Defaults to 0 (no baseline vax). Scenarios inject time-windowed
        campaigns via ``InterventionSpec``.
    days
        Simulation horizon in days.
    dt
        Integration step. Default 0.25 = 4 sub-steps/day (good enough for
        daily resolution at flu-scale β).
    seed
        RNG seed for deterministic initial perturbations. Leave ``None``
        for stochastic exploratory runs.
    """
    disease: DiseaseParams = field(default_factory=DiseaseParams)
    populations: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    mobility: np.ndarray = field(
        default_factory=lambda: np.asarray([[]], dtype=float)
    )
    district_names: list[str] = field(default_factory=list)
    initial_infected: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    initial_recovered: Optional[np.ndarray] = None
    initial_vaccinated: Optional[np.ndarray] = None
    vaccination_rate: float | np.ndarray = 0.0
    days: int = 365
    dt: float = 0.25
    seed: Optional[int] = 42

    # ── Validation helpers ─────────────────────────────────────────────
    def validate(self) -> None:
        """Raise ``ValueError`` if the inputs are structurally inconsistent.

        Checks covered:
          - mobility is square and matches len(populations)
          - mobility rows sum to ~1 (row-stochastic)
          - all compartment-eligible arrays have compatible length
          - initial_infected is non-negative and ≤ populations
        """
        pops = np.asarray(self.populations, dtype=float)
        G = pops.size
        if G == 0:
            raise ValueError("MetapopParams.populations is empty")

        M = np.asarray(self.mobility, dtype=float)
        if M.shape != (G, G):
            raise ValueError(
                f"mobility shape {M.shape} incompatible with G={G}; "
                f"expected ({G}, {G})"
            )
        row_sums = M.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            bad = int((np.abs(row_sums - 1.0) > 1e-6).sum())
            raise ValueError(
                f"mobility is not row-stochastic: {bad} rows have |sum−1| > 1e-6 "
                f"(min={row_sums.min():.6f}, max={row_sums.max():.6f})"
            )

        if self.district_names and len(self.district_names) != G:
            raise ValueError(
                f"district_names length {len(self.district_names)} != populations {G}"
            )

        inf = np.asarray(self.initial_infected, dtype=float)
        if inf.size not in (0, G):
            raise ValueError(
                f"initial_infected length {inf.size} != populations {G}"
            )
        if inf.size == G:
            if np.any(inf < 0):
                raise ValueError("initial_infected contains negative values")
            if np.any(inf > pops):
                raise ValueError("initial_infected exceeds populations in some gu")
            if not np.all(np.isfinite(inf)):
                raise ValueError("initial_infected contains non-finite values")

        # Finite / positivity guards: malformed inputs otherwise propagate
        # NaN/Inf through the force-of-infection matmul and the RK4 step, which
        # can surface downstream as an overflow / native abort instead of a
        # clear error. Fail fast at the boundary instead.
        if not np.all(np.isfinite(pops)):
            raise ValueError("populations contains non-finite values")
        if np.any(pops <= 0):
            raise ValueError("populations must be strictly positive")
        if not np.all(np.isfinite(M)):
            raise ValueError("mobility contains non-finite values")
        if np.any(M < 0):
            raise ValueError("mobility contains negative entries")
        d = self.disease
        for _nm in ("beta", "sigma", "gamma"):
            _v = getattr(d, _nm, None)
            if _v is not None and (not np.isfinite(_v) or _v < 0):
                raise ValueError(
                    f"disease.{_nm} must be finite and non-negative; got {_v!r}"
                )


@dataclass(frozen=True)
class ReactiveTrigger:
    """State-dependent activation rule for a *reactive* InterventionSpec.

    Unlike a fixed ``start_day``/``end_day`` window, a reactive intervention
    fires on the first simulation day its ``metric`` crosses ``threshold``
    (per ``op``) and then stays active for ``duration_days``. This makes the
    NPI self-adjust to epidemic *timing* — the WHO MEM / Vega 2013 reactive-
    threshold approach — instead of firing on a hardcoded calendar day.

    Parameters
    ----------
    metric
        City-wide state metric evaluated each simulation day:
          - ``"prevalence"`` → Σ_gu I / Σ_gu N  (fraction infectious)
          - ``"incidence"``  → Σ_gu E / Σ_gu N  (new-infection pressure proxy)
    threshold
        Crossing value in the same units as ``metric`` (a population fraction).
    duration_days
        How long the intervention stays active once triggered.
    op
        ``">"`` (default) or ``">="``.

    Notes
    -----
    A spec carrying a trigger must be resolved through
    ``interventions.resolve_active_interventions`` before reaching
    ``apply_interventions`` — the resolver converts the trigger into a concrete
    ``[fired_day, fired_day + duration_days)`` window. ``covers()`` on an
    un-resolved triggered spec is meaningless (its start/end are placeholders).
    """
    metric: str
    threshold: float
    duration_days: int
    op: Literal[">", ">="] = ">"

    def fires(self, value: float) -> bool:
        """True when ``value`` crosses ``threshold`` per ``op``."""
        return value > self.threshold if self.op == ">" else value >= self.threshold


@dataclass(frozen=True)
class InterventionSpec:
    """A piecewise-constant multiplicative modifier to simulator parameters.

    ``parameter`` is the name of a field on ``DiseaseParams`` (``"beta"``,
    ``"R0"``, ``"gamma"``, ``"sigma"``, ``"VE"``, ``"ifr"``, ``"omega"``)
    *or* the special string ``"vaccination_rate"`` which is applied to
    ``MetapopParams.vaccination_rate`` instead.

    ``op`` controls how ``value`` is applied:
      - ``"scale"``  → parameter ← parameter × value        (default)
      - ``"set"``    → parameter ← value                    (absolute override)
      - ``"add"``    → parameter ← parameter + value        (e.g. +0.1 to VE)

    ``targets`` is an optional list of district indices; when None the
    intervention applies city-wide. Used to model localised school
    closures or vaccination campaigns.

    Examples
    --------
    >>> InterventionSpec("beta", 0.6, 20, 40)       # NPI lockdown weeks 3-6
    >>> InterventionSpec("vaccination_rate", 0.005,
    ...                  40, 80, op="set")          # 0.5%/day campaign
    >>> InterventionSpec("VE", 0.65, 0, 365,
    ...                  op="set", note="matched strain season")
    """
    parameter: str
    value: float
    start_day: int
    end_day: int
    op: Literal["scale", "set", "add"] = "scale"
    targets: Optional[tuple[int, ...]] = None
    note: str = ""
    #: When set, ``start_day``/``end_day`` are placeholders and the activation
    #: window is computed at run time from epidemic state (see ReactiveTrigger).
    trigger: Optional["ReactiveTrigger"] = None

    def covers(self, day: int) -> bool:
        """Inclusive of start_day, exclusive of end_day (standard convention).

        For a *triggered* spec this is only meaningful after
        ``interventions.resolve_active_interventions`` has rewritten start/end
        to the concrete fired window.
        """
        return self.start_day <= day < self.end_day


# ── Default presets ────────────────────────────────────────────────────
DEFAULT_FLU_PARAMS = DiseaseParams()
"""Module-level default. Callers can pass this directly or mutate a copy."""


def select_disease_params(name: Optional[str] = None) -> DiseaseParams:
    """Resolve a frozen ``DiseaseParams`` for ANY catalog disease (parameterizable base).

    The metapop SEIR-V-D simulator is structurally parameterizable: its free
    parameters are exactly the fields of ``DiseaseParams`` —

        R0     basic reproduction number              (dimensionless)
        sigma  E→I rate = 1 / latent_period           (per day)
        gamma  I→R rate = 1 / infectious_period       (per day)
        omega  R→S / V→S waning rate                  (per day)
        VE     vaccine efficacy (leaky)               (fraction in [0, 1])
        ifr    I→D fractional flow                    (fraction in [0, 1])

    This helper threads a disease *selector* onto that interface so the same
    contact/mobility-coupled FOI machinery can run any entry from the
    ``simulation.disease_params`` catalog (67 catalog / 42 active) WITHOUT
    editing code — driven by the ``MPH_DISEASE`` env var or an explicit
    ``name`` argument.

    Args:
        name: Catalog disease name (Korean, e.g. ``"홍역"`` / ``"백일해"``).
            ``None`` (default) consults the ``MPH_DISEASE`` env var; if that is
            also unset/blank or equals ``"인플루엔자"``, the byte-identical
            ``DEFAULT_FLU_PARAMS`` object is returned (default path unchanged).

    Returns:
        A frozen ``DiseaseParams``. For the default flu path this is the
        *same object* as ``DEFAULT_FLU_PARAMS`` (identity, so influenza
        behaviour and the calibrated FOI are bit-for-bit preserved). For any
        other catalog disease, a fresh instance with fields mapped from the
        catalog's ``DiseaseParams`` (R0_mean→R0, 1/latent→sigma,
        1/infectious→gamma, cfr→ifr, vaccine_efficacy→VE).

    Raises:
        ValueError: ``name`` (or ``MPH_DISEASE``) is not a registered catalog
            disease — surfaced by ``get_disease_params``.

    Performance: O(1) after the catalog registry is built once (lazy DB read).
    Side effects: reads the ``MPH_DISEASE`` env var; first non-default call
        triggers the catalog's lazy DB discovery (read-only).
    Caller responsibility: scope is *directly-transmitted, contact/mobility-
        driven (respiratory)* infections. Vector / blood / sexual / fecal-oral
        routes lie outside the contact-FOI structure (R0_mean=0 catalog
        entries yield beta=0 → no epidemic) and are NOT supported by this base.
    """
    import os

    if name is None:
        name = os.environ.get("MPH_DISEASE", "").strip() or "인플루엔자"

    # Default path: return the canonical object unchanged (identity) so the
    # influenza scenario, calibrated β, and every existing test stay byte-
    # identical. No catalog lookup, no new allocation.
    if name == "인플루엔자":
        return DEFAULT_FLU_PARAMS

    # Non-default disease: map the catalog's DiseaseParams onto the frozen
    # Stage-5 interface. Latent/infectious periods are inverted into rates;
    # cfr is taken as the fractional I→D flow; omega/report_frac inherit the
    # flu-calibrated structural defaults (catalog has no per-disease waning).
    from simulation.disease_params import get_disease_params as _cat_params

    cat = _cat_params(name)  # raises ValueError for unknown disease
    latent = float(cat.latent_period) if cat.latent_period > 0 else float("inf")
    infectious = float(cat.infectious_period) if cat.infectious_period > 0 else float("inf")
    return DiseaseParams(
        R0=float(cat.R0_mean),
        sigma=(1.0 / latent) if latent != float("inf") else 0.0,
        gamma=(1.0 / infectious) if infectious != float("inf") else 0.0,
        omega=DEFAULT_FLU_PARAMS.omega,
        VE=float(cat.vaccine_efficacy),
        V_waning=DEFAULT_FLU_PARAMS.V_waning,
        ifr=float(cat.cfr),
        report_frac=DEFAULT_FLU_PARAMS.report_frac,
    )
