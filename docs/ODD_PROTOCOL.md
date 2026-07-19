# ODD+D Protocol — Seoul Influenza Behavioral Agent Model

> Standardized model description per the **ODD protocol** (Grimm et al. 2006;
> 2020 second update) extended with **ODD+D** human decision-making (Müller et
> al. 2013). Addresses the external-review recommendation to document the ABM in
> a reproducible, reviewer-recognized format. Code: `simulation/abm/`,
> `simulation/sim/`. See [ABM.md](ABM.md), [PIPO_CALIBRATION.md](PIPO_CALIBRATION.md).

---

## 1. Overview

### 1.1 Purpose and patterns
Forecast district-resolution influenza-like illness (ILI) for Seoul and run
counterfactual interventions; **test when adaptive behavior adds explanatory
value** over seasonal forcing and mobility. Patterns the model is expected to
reproduce: a single seasonal ILI wave per district, commuter-driven spatial
spread, and a behavior-dependent peak that the behavior-off variant over-shoots.

### 1.2 Entities, state variables, and scales
| Entity | State variables |
|--------|-----------------|
| **Agent** (agent-world, `agent_kernel.py`) | epidemiological state ∈ {S,E,I,R,V,D}; age band; comorbidity load; school/work affiliation; home + daytime district; risk-sensitivity α; fatigue; compliance threshold θ; vaccination & infection status |
| **District** (metapop, `metapop_seirvd.py`) | compartment counts S,E,I,R,V,D per gu (25); resident population; commuter out-flows |
| **Global/environment** | day index; seasonal forcing phase/amplitude; weather covariate; commuter matrix M (row-stochastic) |

**Scales**: 25 districts (gu); daily timestep (dt = 1 day); horizon ≈ one season
(120–365 days). Agent count adaptive (`adaptive_agent_count.py`).

### 1.3 Process overview and scheduling
Each day, in order: (1) **force of infection** computed per district from local
prevalence + commuter coupling (`foi.py`); (2) **behavioral modulation** of
contact/transmission from each agent's risk perception, fatigue, and compliance
(`behavioural.py`); (3) **state transitions** by binomial tau-leap (agent-world)
or RK4 ODE (metapop, unconditionally-positive integrator, `stepper.py`);
(4) **interventions** applied if active (`interventions.py`); (5) **observation**
(ILI proxy) recorded. Scheduling is synchronous daily updates.

---

## 2. Design concepts

- **Basic principles**: SEIR-V-D transmission with commuter-coupled metapopulation
  force of infection + a behavior–disease feedback loop.
- **Emergence**: the epidemic curve, peak timing, and spatial hotspots emerge from
  local transitions + mobility; they are not imposed.
- **Adaptation / Objectives**: agents reduce contact when perceived local risk
  exceeds a compliance threshold (objective: avoid infection), subject to fatigue.
- **Sensing**: agents sense district-level prevalence/risk (configurable lag),
  not individual infection status of others.
- **Interaction**: indirect, via the shared district force of infection and the
  commuter mixing matrix.
- **Stochasticity**: the agent-world is a **binomial tau-leap** process (random
  streams keyed by day and gu); the metapop ODE is deterministic. Output is a
  distribution — reported as a Monte-Carlo **ensemble** with percentile CIs and a
  variance-stabilization curve (`sensitivity.py`, Lee et al. 2015).
- **Collectives**: districts and age/affiliation groups.
- **Observation**: weekly ILI per 1,000 (the KDCA sentinel target), per gu and city.

### ODD+D — human decision-making
- **Theoretical/empirical background**: risk-perception + protective-behavior
  literature (behavior-in-epidemics); fatigue and compliance from NPI experience.
- **Individual decision-making**: a four-parameter rule — **α** risk-sensitivity,
  **κ** (gain), **τ** perception lag, **θ** compliance threshold — maps perceived
  risk → contact reduction → modulated transmission (`behavioural.py`).
- **Learning / prediction**: bounded; agents react to current/lagged risk rather
  than forecasting (a deliberate simplicity choice, stated as a limitation).
- **Heterogeneity**: per-agent age (KNHANES bands), comorbidity severity multiplier
  (`comorbidity.py`), school affiliation (`affiliation.py`), time-resolved daytime
  mobility (`agent_mobility.py`).
- **Stochasticity**: behavioral thresholds θ drawn per agent (θ_mean, θ_sd).

---

## 3. Details

### 3.1 Initialization
District populations + commuter matrix from `seir-metapop-init.json`
(`export_seir_metapop_init.py`); seeded infections; agent attributes sampled from
KNHANES comorbidity prevalence and real `school_info` affiliation. Seed fixed
(reproducible); ensemble varies the seed.

### 3.2 Input data
`epi_real_seoul.db`: `daily_population_*` (residents), `commuter_matrix` (mobility),
`school_info` (affiliation), KNHANES (comorbidity), `weather_historical` (covariate),
`sentinel_influenza` (KDCA ILI calibration target). Provenance per
[PIPO_CALIBRATION.md](PIPO_CALIBRATION.md).

### 3.3 Submodels
- **Force of infection** (`foi.py`): λ_g(t) = β·s(t)·[local prevalence + Σ commuter-coupled prevalence].
- **Behavioral modulation** (`behavioural.py`): contact/β scaled by the α/κ/τ/θ rule.
- **Transitions**: binomial tau-leap (agent) / RK4 (metapop), 6 compartments.
- **Interventions** (`interventions.py`): distancing, school closure, vaccination,
  antivirals, reactive state-triggers — each tied to a statutory basis (감염병예방법, the Infectious
  Disease Control and Prevention Act;
  see [ABM.md](ABM.md) §4).
- **Vaccination/waning**: S→V at rate ν; R/V waning to S.

**Validity gate**: every run checks Rt ∈ [0.3, 8], seasonal phase, and
S+E+I+R+V+D = N conservation (tol 1e-9); failures are surfaced, not hidden.
