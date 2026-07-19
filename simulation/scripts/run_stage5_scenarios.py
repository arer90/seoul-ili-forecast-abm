"""Stage 5 — Run all 6 metapop SEIR-V-D scenarios on Seoul 25-gu.

Outputs to ``simulation/results/sim_runs/``:
  - ``<scenario>.npz``                state + incidence + days
  - ``<scenario>_city.csv``           city-wide (S, E, I, R, V, D) per day
  - ``<scenario>_gu_weekly.csv``      per-gu weekly incidence (long form)
  - ``<scenario>_validity.json``      epi-validity gate report
  - ``_manifest.json``                run-level metadata (peak, Rt range, validity)

Note: prior versions of this docstring referenced ``<scenario>_gu.csv`` (per-day
long form). Sprint 2026-05-06 (Codex non-bio review #4) — corrected to match
the actual code emit at line ~200, which writes ``_gu_weekly.csv`` (weekly
aggregate, not daily).

Params come from ``load_metapop_params()`` — real commuter_matrix (25x25) +
night_population. Each scenario runs for 365 days (1 full flu season).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation.sim import (
    COMPARTMENTS,
    SCENARIO_REGISTRY,
    run_scenario,
    SimResult,
)
from simulation.sim.io import load_metapop_params
from simulation.database.config import SEOUL_GU_ORDERED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("stage5.scenarios")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "simulation" / "results" / "sim_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────
def _ser(x: Any) -> Any:
    """JSON-safe coercion for numpy scalars / arrays."""
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_ser(v) for v in x]
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    return str(x)


def _city_trajectory(res: SimResult) -> pd.DataFrame:
    """T x 6 wide city-totals DataFrame."""
    rows = {c: res.city_total(c) for c in COMPARTMENTS}
    rows["day"] = res.days
    rows["incidence_city"] = res.incidence.sum(axis=1)
    df = pd.DataFrame(rows)[["day", "incidence_city", *COMPARTMENTS]]
    return df


def _gu_incidence_long(res: SimResult) -> pd.DataFrame:
    """Long-form (day, gu, incidence, I, S, V) for choropleth figures."""
    days, G = res.incidence.shape
    recs = []
    names = res.district_names
    for d in range(days):
        for g in range(G):
            recs.append({
                "day": int(d),
                "gu": names[g],
                "incidence": float(res.incidence[d, g]),
                "I": float(res.I[d, g]),
                "S": float(res.S[d, g]),
                "V": float(res.V[d, g]),
            })
    return pd.DataFrame.from_records(recs)


def _scenario_summary(name: str, res: SimResult) -> dict:
    """Compact per-scenario summary for the manifest."""
    city_I = res.city_total("I")
    city_inc = res.incidence.sum(axis=1)
    peak_day = int(np.argmax(city_I))
    peak_I = float(city_I.max())
    peak_inc = float(city_inc.max())
    peak_week = peak_day // 7
    total_infections = float(res.incidence.sum())
    total_deaths = float(res.city_total("D")[-1])
    peak_gu_idx = int(np.argmax(res.I[peak_day]))
    peak_gu = res.district_names[peak_gu_idx]

    # Validity gate: status="ok" + no violations = pass. The ILI-cap warning
    # is benign (I is raw people count in SEIR-V-D, not ILI per 1000).
    gate = res.epi_validity.get("metapop_seirvd", {}) if isinstance(res.epi_validity, dict) else {}
    gate_status = gate.get("status")
    gate_violations = gate.get("violations", [])
    gate_ok = gate_status == "ok" and len(gate_violations) == 0
    cons_err = (
        gate.get("checks", {}).get("compartment_conservation", {}).get("max_rel_err", None)
    )

    return {
        "scenario": name,
        "days_run": int(res.days[-1]),
        "peak_day": peak_day,
        "peak_week": peak_week,
        "peak_I_city": peak_I,
        "peak_incidence_city": peak_inc,
        "peak_gu": peak_gu,
        "peak_gu_I": float(res.I[peak_day, peak_gu_idx]),
        "total_incidence": total_infections,
        "cumulative_deaths": total_deaths,
        "cumulative_recovered": float(res.city_total("R")[-1]),
        "cumulative_vaccinated": float(res.city_total("V")[-1]),
        "attack_rate_pct": 100.0 * total_infections / float(res.params.populations.sum()),
        "n_interventions": len(res.interventions),
        "interventions": [
            {
                "parameter": i.parameter,
                "op": i.op,
                "value": float(i.value),
                "start_day": int(i.start_day),
                "end_day": int(i.end_day),
                "note": i.note,
            }
            for i in res.interventions
        ],
        "epi_validity_ok": gate_ok,
        "epi_validity_status": gate_status,
        "epi_validity_n_violations": len(gate_violations),
        "compartment_conservation_err": cons_err,
    }


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    log.info("Stage 5 scenarios start — output dir: %s", OUT_DIR)

    # Shared DB-loaded params — same initial condition across scenarios
    base_params = load_metapop_params(
        seed_infected=10.0,
        seed_district="강남구",
        days=365,
        dt=0.25,
    )
    log.info(
        "loaded params: G=%d districts, pop=%.2fM, M row-stoch=%s",
        len(base_params.populations),
        base_params.populations.sum() / 1e6,
        np.allclose(base_params.mobility.sum(axis=1), 1.0, atol=1e-6),
    )

    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "params": {
            "G": len(base_params.populations),
            "population_total": float(base_params.populations.sum()),
            "days": int(base_params.days),
            "dt": float(base_params.dt),
            "seed": int(base_params.seed) if base_params.seed else None,
            "disease_R0": base_params.disease.R0,
            "disease_gamma": base_params.disease.gamma,
            "disease_VE": base_params.disease.VE,
            "districts": list(base_params.district_names),
        },
        "scenarios": {},
    }

    scenario_names = list(SCENARIO_REGISTRY.keys())
    log.info("running %d scenarios: %s", len(scenario_names), scenario_names)

    for name in scenario_names:
        t1 = time.time()
        log.info("▶ scenario %s", name)
        res = run_scenario(name, params=base_params)

        # Save npz (compact binary)
        np.savez_compressed(
            OUT_DIR / f"{name}.npz",
            state=res.state,
            incidence=res.incidence,
            days=res.days,
        )

        # Save city CSV
        _city_trajectory(res).to_csv(OUT_DIR / f"{name}_city.csv", index=False)

        # Save gu long CSV (weekly-downsampled to keep file small: every 7 days)
        gu_df = _gu_incidence_long(res)
        gu_df = gu_df[gu_df["day"] % 7 == 0].reset_index(drop=True)
        gu_df.to_csv(OUT_DIR / f"{name}_gu_weekly.csv", index=False)

        # Save validity JSON
        (OUT_DIR / f"{name}_validity.json").write_text(
            json.dumps(_ser(res.epi_validity), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        manifest["scenarios"][name] = _scenario_summary(name, res)
        log.info(
            "✅ %s done in %.1fs — peak_day=%d, peak_I=%.0f, attack=%.1f%%, validity_ok=%s",
            name, time.time() - t1,
            manifest["scenarios"][name]["peak_day"],
            manifest["scenarios"][name]["peak_I_city"],
            manifest["scenarios"][name]["attack_rate_pct"],
            manifest["scenarios"][name]["epi_validity_ok"],
        )

    # Write manifest
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(_ser(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Stage 5 all scenarios done in %.1fs — manifest: %s",
             time.time() - t0, OUT_DIR / "_manifest.json")


if __name__ == "__main__":
    main()
