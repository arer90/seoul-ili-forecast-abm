"""
simulation.verify_all
=====================
Single-command verification that every local simulation / ABM / LLM
component is runnable end-to-end. Used as the reviewer-facing smoke
test for the thesis §4.18 claim that the infrastructure is actually
operational rather than a pipeline mockup.

Usage::

    python -m simulation.verify_all [--fast]

Sections
--------
  1. Module import check        — everything imports cleanly
  2. DB connectivity            — safe_connect + commuter matrix loadable
  3. Kernel SEIR-V-D run        — mass conservation ≤ 1e-9
  4. ABM invariant (α = 0)      — RMSE = 0 vs kernel
  5. ABM adversarial battery    — T1 + T2 + T4 all pass
  6. ABM S1–S6 scenarios        — all 6 produce finite peak
  7. LLM backend discovery      — report tiers available
  8. LLM mock-only smoke        — 3 profiles × 2 items, deterministic
  9. LLM Ollama smoke           — skipped if daemon is down
 10. E2E pipeline dry-run       — DB → forecast → SEIR → ABM → advisor
                                    (LLM step uses --no-ollama if no daemon)

Each section prints a PASS / FAIL / SKIP line and the aggregated JSON
report is written to ``simulation/results/verify_all_report.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("verify_all")


@dataclass
class Check:
    name: str
    passed: bool
    elapsed_ms: float
    detail: dict = field(default_factory=dict)
    error: str = ""


def _timed(fn: Callable, *args, **kwargs) -> tuple[bool, float, dict, str]:
    t0 = time.time()
    try:
        ok, info = fn(*args, **kwargs)
        return bool(ok), (time.time() - t0) * 1000.0, dict(info), ""
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        return False, (time.time() - t0) * 1000.0, {}, f"{type(e).__name__}: {e}\n{tb}"


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------
def check_imports() -> tuple[bool, dict]:
    failures: list[str] = []
    probes = [
        "simulation.database",
        "simulation.sim.io",
        "simulation.sim.parameters",
        "simulation.sim.metapop_seirvd",
        "simulation.sim.stepper",
        "simulation.sim.foi",
        "simulation.abm",
        "simulation.abm.behavioural",
        "simulation.abm.scenarios",
        "simulation.abm.adversarial_tests",
        "simulation.llm_compare",
        "simulation.llm_compare.backends",
        "simulation.llm_compare.golden_set",
        "simulation.llm_compare.judge",
        "simulation.llm_compare.runner",
        "simulation.tests.test_e2e_smoke",
    ]
    for mod in probes:
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{mod}: {type(e).__name__}: {e}")
    return (not failures), {"n_probes": len(probes), "failures": failures}


def check_db() -> tuple[bool, dict]:
    from simulation.database import safe_connect
    from simulation.sim.io import load_metapop_params
    with safe_connect() as con:
        n_tables = con.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
    mp = load_metapop_params()
    G = int(mp.populations.size)
    row_sums = np.asarray(mp.mobility, dtype=float).sum(axis=1)
    row_stochastic = bool(np.allclose(row_sums, 1.0, atol=1e-6))
    return (G == 25 and row_stochastic), {
        "n_tables": int(n_tables),
        "G": G,
        "mobility_row_stochastic": row_stochastic,
        "population_sum": float(np.asarray(mp.populations).sum()),
    }


def check_kernel() -> tuple[bool, dict]:
    from simulation.sim.io import load_metapop_params
    from simulation.sim.metapop_seirvd import MetapopSEIRVD
    from simulation.sim.parameters import MetapopParams
    mp = load_metapop_params()
    G = int(mp.populations.size)
    params = MetapopParams(
        disease=mp.disease, populations=mp.populations, mobility=mp.mobility,
        district_names=mp.district_names,
        initial_infected=np.full(G, 1000.0),
        days=60, dt=mp.dt, seed=mp.seed,
    )
    r = MetapopSEIRVD(params).run(run_validator=False)
    totals = r.state.sum(axis=(1, 2))
    mass_err = float(np.abs(totals - totals[0]).max() / max(totals[0], 1.0))
    peak_I = float(r.city_total("I").max())
    return (mass_err < 1e-9 and peak_I > 0), {
        "mass_conservation_rel_err": mass_err,
        "city_I_peak": peak_I,
        "horizon_days": 60,
    }


def check_abm_invariant() -> tuple[bool, dict]:
    from simulation.abm.behavioural import run_invariant_test
    from simulation.sim.io import load_metapop_params
    from simulation.sim.parameters import MetapopParams
    mp = load_metapop_params()
    G = int(mp.populations.size)
    params = MetapopParams(
        disease=mp.disease, populations=mp.populations, mobility=mp.mobility,
        district_names=mp.district_names,
        initial_infected=np.full(G, 1000.0),
        days=90, dt=mp.dt, seed=mp.seed,
    )
    r = run_invariant_test(params, tolerance=1e-6)
    return bool(r["passed"]), r


def check_adversarial() -> tuple[bool, dict]:
    from simulation.abm.adversarial_tests import run_all_adversarial_tests
    from simulation.sim.io import load_metapop_params
    from simulation.sim.parameters import MetapopParams
    mp = load_metapop_params()
    G = int(mp.populations.size)
    params = MetapopParams(
        disease=mp.disease, populations=mp.populations, mobility=mp.mobility,
        district_names=mp.district_names,
        initial_infected=np.full(G, 1000.0),
        days=120, dt=mp.dt, seed=mp.seed,
    )
    out = run_all_adversarial_tests(params)
    return bool(out["all_passed"]), {
        "all_passed": out["all_passed"],
        "tests": [{"name": t["name"], "passed": t["passed"]} for t in out["tests"]],
    }


def check_scenarios() -> tuple[bool, dict]:
    from simulation.abm.scenarios import run_scenario_suite
    from simulation.sim.io import load_metapop_params
    from simulation.sim.parameters import MetapopParams
    mp = load_metapop_params()
    G = int(mp.populations.size)
    params = MetapopParams(
        disease=mp.disease, populations=mp.populations, mobility=mp.mobility,
        district_names=mp.district_names,
        initial_infected=np.full(G, 1000.0),
        days=90, dt=mp.dt, seed=mp.seed,
    )
    rep = run_scenario_suite(params)
    peaks = {sid: s["peak_city_I"] for sid, s in rep["per_scenario"].items()}
    all_positive = all(np.isfinite(v) and v > 0 for v in peaks.values())
    # baseline must be largest peak (every behaviour-on scenario dampens it)
    s1 = peaks.get("S1", 0.0)
    dampening = all(peaks.get(sid, 0.0) < s1 + 1e-6
                    for sid in ("S2", "S3", "S4", "S5", "S6"))
    return (all_positive and dampening), {
        "peaks": {k: round(float(v), 1) for k, v in peaks.items()},
        "all_behaviour_on_dampens_S1": dampening,
    }


def check_llm_discovery() -> tuple[bool, dict]:
    from simulation.llm_compare.backends import discover_backends, env_status
    backs = discover_backends()
    env = env_status()
    info = {
        "n_backends": len(backs),
        "backend_ids": [b.backend_id for b in backs],
        "tiers": sorted({b.tier for b in backs}),
        "api_keys_present": env["api_keys_present"],
        "ollama_installed_models": env["ollama_installed_models"],
    }
    return (len(backs) >= 3), info  # mock-only is always 3


def check_llm_mock() -> tuple[bool, dict]:
    from simulation.llm_compare.backends import discover_backends
    from simulation.llm_compare.golden_set import load_golden_set
    from simulation.llm_compare.runner import run_comparison
    backs = discover_backends(include_api=False, include_ollama=False,
                              include_mock=True, max_ollama=0)
    backs = [b for b in backs if b.tier == "mock"]  # mock-only: exclude auto-found CLI backends (claude/codex/gemini)
    items = list(load_golden_set())[:2]
    rep = run_comparison(backs, items, verbose=False)
    ranking = rep.ranking
    spread = (ranking[0]["total"] - ranking[-1]["total"]) if ranking else 0.0
    return (len(ranking) == 3 and spread > 0), {
        "n_backends": len(ranking), "winner": ranking[0]["backend_id"],
        "score_spread": round(float(spread), 4),
    }


def check_llm_ollama() -> tuple[bool, dict]:
    from simulation.llm_compare.backends import (
        discover_backends, list_ollama_models,
    )
    from simulation.llm_compare.golden_set import load_golden_set
    from simulation.llm_compare.runner import run_comparison
    installed = list_ollama_models()
    if not installed:
        return False, {"skipped": "no Ollama daemon / models detected",
                        "ollama_installed_models": []}
    backs = discover_backends(include_api=False, include_ollama=True,
                              include_mock=False, max_ollama=1)
    if not backs:
        return False, {"skipped": "no Ollama backend enabled",
                        "ollama_installed_models": installed}
    items = list(load_golden_set())[:1]
    rep = run_comparison(backs, items, verbose=False)
    text_len = len(rep.items[0].get("response_text", "")) if rep.items else 0
    return (text_len > 0), {
        "backend": backs[0].backend_id,
        "response_chars": text_len,
        "latency_ms": rep.ranking[0]["mean_latency_ms"] if rep.ranking else 0.0,
    }


def check_e2e(fast: bool = False) -> tuple[bool, dict]:
    from simulation.tests.test_e2e_smoke import run_e2e
    from simulation.utils.paths import PROJECT_ROOT  # valid_test/ test-output isolation
    res = run_e2e(
        gu="강남구", lookback_weeks=26,
        forecast_weeks=2, horizon_days=60,
        out_dir=str(PROJECT_ROOT / "valid_test" / "verify" / "verify_e2e"),
        include_llm=not fast,
        llm_no_api=True, llm_no_ollama=fast,
        llm_max_ollama=0 if fast else 1,
    )
    peak = res.seir_baseline.get("city_I_peak", 0.0)
    abm = res.abm_counterfactual
    ok = (
        bool(res.db_snapshot.get("n_weeks", 0) > 0)
        and bool(np.isfinite(peak) and peak > 0)
        and bool(abm.get("invariant_passed"))
        and bool(res.audit_chain)
    )
    return ok, {
        "db_source": res.db_snapshot.get("source"),
        "n_weeks": res.db_snapshot.get("n_weeks"),
        "forecast_point": [round(float(p), 2) for p in res.forecast.get("point", [])],
        "seir_peak": round(float(peak), 1),
        "abm_peak_shift_pct": round(float(abm.get("peak_shift_pct", 0.0)), 2),
        "abm_invariant_passed": bool(abm.get("invariant_passed")),
        "audit_entries": len(res.audit_chain),
        "llm_ran": res.llm_comparison is not None,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all(fast: bool = False) -> dict:
    logging.basicConfig(level=logging.WARNING,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    checks: list[Check] = []

    def _add(name: str, fn, *args, **kwargs):
        ok, ms, info, err = _timed(fn, *args, **kwargs)
        c = Check(name=name, passed=ok, elapsed_ms=ms, detail=info, error=err)
        checks.append(c)
        flag = "PASS" if ok else ("SKIP" if info.get("skipped") else "FAIL")
        print(f"  [{flag:4s}] {name:28s}  {ms:7.1f} ms  {info}")
        if err:
            print(f"         err = {err.splitlines()[0][:160]}")

    print("\n=== simulation.verify_all (fast=%s) ===" % fast)
    _add("imports",           check_imports)
    _add("db_connectivity",   check_db)
    _add("kernel_seir",       check_kernel)
    _add("abm_invariant",     check_abm_invariant)
    _add("abm_adversarial",   check_adversarial)
    _add("abm_scenarios",     check_scenarios)
    _add("llm_discovery",     check_llm_discovery)
    _add("llm_mock_smoke",    check_llm_mock)
    if not fast:
        _add("llm_ollama_smoke", check_llm_ollama)
        _add("e2e_pipeline",     check_e2e, False)
    else:
        _add("e2e_pipeline_fast", check_e2e, True)

    total_pass = sum(1 for c in checks if c.passed)
    total_fail = sum(1 for c in checks if not c.passed and not c.detail.get("skipped"))
    total_skip = sum(1 for c in checks if c.detail.get("skipped"))
    print(f"\n=== verify_all SUMMARY: {total_pass} PASS / {total_fail} FAIL / {total_skip} SKIP ===\n")

    report = {
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "fast_mode": fast,
        "summary": {"pass": total_pass, "fail": total_fail, "skip": total_skip,
                    "total_checks": len(checks)},
        "checks": [asdict(c) for c in checks],
    }
    from simulation.utils.paths import PROJECT_ROOT  # valid_test/ test-output isolation
    out = PROJECT_ROOT / "valid_test" / "verify" / "verify_all_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False),
                   encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip Ollama and LLM-in-E2E (pure CPU-only CI check)")
    args = ap.parse_args(argv)
    report = run_all(fast=args.fast)
    return 0 if report["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
