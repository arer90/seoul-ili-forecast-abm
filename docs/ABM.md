# ABM — Metapopulation SEIR-V-D × Behavioral Agent Layer

> Agent-based infectious disease simulation over Seoul's 25 gu (districts). **Two engines** + behavioral coupling + proof modules + web dashboard.
> Code: `simulation/sim/` (metapopulation engine) · `simulation/abm/` (behavior, heterogeneity, proofs).
> Related: [ARIA.md](ARIA.md) (the LLM layer that advises the simulation) · [WEB_DASHBOARD_GUIDE.md](WEB_DASHBOARD_GUIDE.md).
> Reporting standards: [ODD_PROTOCOL.md](ODD_PROTOCOL.md) (ODD+D ABM description) · [PIPO_CALIBRATION.md](PIPO_CALIBRATION.md) (PIPO calibration reporting) · the external-review response (not shipped) (response to external review).

---

## 1. Two engines

| Engine | Entry point | Unit | Purpose |
|------|------|------|------|
| **Metapop SEIR-V-D** | `run_metapop_scenario` (`server/epimas_adapter.py`) → `sim/metapop_seirvd.run_scenario` | Deterministic ODE over 25 gu | Scenario and intervention policy, web dashboard |
| **Agent-world** | `abm/agent_world_fit.py` (`run_agent_world`) | Individual agents | Heterogeneity (comorbidity, age, affiliation, environment) and identifiability proofs |

**Key finding** (memory): agent-world fits influenza at R²≈0.95, but it does so **even with behavior turned OFF** → influenza is **driven by seasonal forcing**, and behavioral effects operate at **pandemic scale**. The behavioral layer therefore separates dynamically only in the strong-intervention regime (lockdown) — see the proofs in §5 below.

---

## 2. Metapop SEIR-V-D engine (`simulation/sim/`)

| File | Role |
|------|------|
| `metapop_seirvd.py` | Core — `MetapopSEIRVD`, `run_scenario(name, params, *, overrides)`, `SimResult` (state shape **(T, 25, 6)** = S·E·I·R·V·D) |
| `foi.py` | **Commuting-coupled force of infection** — cross-district infection pressure via the mobility matrix M (commuter_matrix, row-stochastic) |
| `stepper.py` | Integrator — **unconditional positivity preservation** (W3·B-P1; negative compartments are blocked) |
| `interventions.py` | Interventions (social distancing, school closure, vaccination, antivirals, reactive state-triggered actions) |
| `parameters.py` | `MetapopParams` — β·σ·γ·ν (vaccination)·waning·seasonal forcing (β_amp/β_phase)·`days`·`seed` |
| `scenarios.py` + `scenarios_extended.py` | Builders for the 14 scenarios |

**Epidemiological validity gate** (`SimResult.epi_validity`): Rt ∈ [0.3, 8], seasonal phase, **conservation of S+E+I+R+V+D = N** (tol 1e-9). A gate failure is flagged on the result.

**Performance**: baseline ~0.6–3s, with behavioral coupling ~3–8s (RK4, dt=1d, G=25, T=120).

---

## 3. Behavior and heterogeneity layer (`simulation/abm/`)

### Behavioral coupling
- `behavioural.py` — 4-parameter behavioral ABM: **α** (risk sensitivity)·**κ**·**τ** (delay)·**θ**. Risk perception/fatigue → contact rate → modulation of β.
- `forecast_anchor.py`, `counterfactual.py` — behavior on/off counterfactuals.

### Heterogeneity enrichment (real data + literature)
| Module | Contents |
|------|------|
| `comorbidity.py` | **KNHANES** prevalence (7 age bands × obesity/diabetes/hypertension/high cholesterol) → severity multipliers (obesity RR 1.45, diabetes 1.75) |
| `affiliation.py` | School affiliation from the real `school_info` table (25 gu, 1,324 schools) |
| `environment.py` | Environmental variables, e.g. temperature vs influenza search-query corr −0.54 |
| `agent_mobility.py` | Time-resolved mobility, per-agent daytime location (daytime_location) routing |
| `synthetic_population.py`, `contact_structure.py`, `age_validation.py` | Synthetic population, contacts, and age validation |

### Fitting and validation
- `agent_world_fit.py` — `calibrate_agent_world`, `agent_count_effect`, `evaluate_agent_world_full` (**129 metrics**: R²·RMSE·WIS·c-index·AUC, etc.), `agent_world_behavioral_sensitivity`.
- `validate_real.py`, `within_season_validation.py`, `stratified_validation.py`, `behavior_disease_validation.py` — validation against real waves.

---

## 4. Scenarios (14 registered)

`list_scenarios()` (`server/epimas_adapter.py`):
`baseline · npi_lockdown · school_closure · vaccination_campaign · antiviral_prophylaxis · combined_response · delayed_response · partial_compliance · reactive_intervention · hospital_surge · subtype_a_h1n1_pdm09 · subtype_a_h3n2 · vaccine_uptake_low · sensitivity_strain_mismatch`

**The 4 exposed on the web dashboard, with their legal basis under the Infectious Disease Control and Prevention Act** (`export_abm_scenarios.py`):

| Scenario | Label | Legal basis |
|----------|------|-----------|
| baseline | 기준 (개입 없음) (baseline, no intervention) | 제16조 (Article 16) sentinel surveillance only |
| npi_lockdown | 봉쇄·거리두기 (lockdown and distancing) | 제49조 (Article 49) restriction/prohibition of gatherings + mask mandate |
| school_closure | 학교 폐쇄 (school closure) | 제49조 (Article 49) school and kindergarten closure (linked to the School Health Act) |
| vaccination_campaign | 백신 캠페인 (vaccination campaign) | 제24·25조 (Articles 24 and 25) (mandatory and temporary) vaccination under the NIP |

---

## 5. Proof modules (P1–P4 — SCI-grade validation)

> Details: `docs/PROOF_VALIDATION_PROTOCOL_20260606.md`. **Principle**: a weakness is not resolved by downgrading the claim but by running code to either establish it as fact or retract it.

| Proof | Module | Result |
|------|------|------|
| **P1** Confounding check | `behavioral_proof.py` | Fatigue ∥ vaccination ρ=0.91 → the observational regression is confounded → **the overclaim is retracted**; the proof comes from the model instead |
| **P2** Dynamical signature | `dynamical_signatures.py` | Hysteresis loop area (shoelace + phase-randomized surrogate null) + periodogram — the mechanism generates a hysteresis loop (ON p→0.000, OFF flat) |
| **P3** sim-vs-observed dual validation | `sim_vs_observed.py` | Run the ABM → apply the identical analysis → β agrees |
| **P4** Identifiability (headline) | `identifiability.py` | Full 4-D profile likelihood — **mobility breaks the (α,θ) degeneracy** (θ identifiability 83%→17%, a 5× improvement). With forcing alone, only τ is identified |
| P5 (ARIA) | `llm_compare/` | [ARIA.md](ARIA.md) §7 |

---

## 6. Web dashboard (`/map3d/abm`)

```
simulation/scripts/export_abm_scenarios.py   # precompute I(t)/N trajectories: 4 scenarios × 121 days × 25 districts
   → web/public/aggregates/abm-scenarios.json
   → web/components/AbmMap.tsx                # district color = infection rate, day-slider animation
   → web/app/map3d/abm/page.tsx               # scenario selector + statistics + legal-basis banner + ARIA panel
```
- **ARIA advisory** on the left (bidirectional, [ARIA.md](ARIA.md) §6), SEIR animation on the right.
- Statistics: peak day, attack rate, deaths, epidemiological validity gate.

---

## 7. Running it

```bash
# Single scenario (Python)
python -c "from simulation.server.epimas_adapter import run_metapop_scenario as r; \
           x=r('npi_lockdown', horizon_days=120, seed=42); print(x.state.shape, x.epi_validity)"

# Regenerate the dashboard trajectories
python -m simulation.scripts.export_abm_scenarios

# Via the MCP tool (ARIA / web)
#   epi.scenario_run  → server/mcp_epi.py
```

**Reproducibility**: fixed seed (42), deterministic ODE, only gate-passing outputs are emitted. `export_abm_scenarios.py` is deterministic (same input → same JSON).
