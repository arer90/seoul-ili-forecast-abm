"""Simulation + MCP server CLI commands — extracted from __main__.py.

Phase C2 partial (2026-05-12 cont.): Stage 5 (Metapop SEIR-V-D run) +
Stage 6a (ARIA MCP server stdio) handlers moved here.
"""
from __future__ import annotations

import logging
import sys


log = logging.getLogger(__name__)


def cmd_sim(args) -> None:
    """`python -m simulation sim` — Stage 5 Metapop SEIR-V-D scenario run."""
    # M1 forecast→ABM (사용자 2026-06-07: champion = DEFAULT basis, 변경가능). The
    # ABM/ARIA basis is the operational CHAMPION (real_eval best_model) BY DEFAULT —
    # forecast-anchoring runs unless an explicit --scenario is requested.
    # --anchor-forecast <model> CHANGES the basis to a specific model; a bare
    # --anchor-forecast (or no flag at all) → champion. --scenario opts out (fixed).
    _anchor = getattr(args, "anchor_forecast", None)
    if not getattr(args, "list_scenarios", False) and (
        _anchor is not None or not getattr(args, "scenario", None)
    ):
        from simulation.abm.forecast_anchor import run_forecast_anchored, DEFAULT_MODEL
        from simulation.utils.paths import get_results_dir
        model = _anchor or DEFAULT_MODEL  # ""/None → champion → real_eval best_model
        basis = model if (_anchor and model != DEFAULT_MODEL) else "champion (real_eval best_model)"
        out = get_results_dir() / "forecast_anchored.json"
        print(f"[sim] forecast-anchored ABM — basis={basis} (DEFAULT), "
              f"n_agents={getattr(args, 'n_agents', 37_500)} "
              f"(real_eval forecast → ABM forcing; --scenario 로 opt-out, "
              f"--anchor-forecast <model> 로 변경)")
        result = run_forecast_anchored(
            model_name=model,
            n_agents=int(getattr(args, "n_agents", 37_500)),
            output_path=str(out),
        )
        anchor = result.get("anchor", {}) if isinstance(result, dict) else {}
        print(f"[sim] anchor: corr={anchor.get('correlation')}, "
              f"degenerate={anchor.get('degenerate')} → {out}")
        return

    from simulation.sim import SCENARIO_REGISTRY, run_scenario

    if getattr(args, "list_scenarios", False):
        print("[sim] registered scenarios:")
        for name in sorted(SCENARIO_REGISTRY):
            print(f"  - {name}")
        return

    if not args.scenario:
        print("ERROR: --scenario is required (use --list-scenarios to see options)")
        sys.exit(2)

    if args.scenario not in SCENARIO_REGISTRY:
        print(f"ERROR: unknown scenario {args.scenario!r}")
        print(f"       registered: {sorted(SCENARIO_REGISTRY)}")
        sys.exit(2)

    # Base params: DB-backed when --use-db, else None (scenario provides default)
    base = None
    if getattr(args, "use_db", False):
        from simulation.sim.io import load_metapop_params
        base = load_metapop_params(
            seed_infected=args.seed_infected,
            seed_district=args.seed_district,
            days=args.days or 200,
        )
    overrides = {}
    if args.days is not None:
        overrides["days"] = int(args.days)
    # Codex non-bio review #5 + #8 (sprint 2026-05-06): thread --seed +
    # --allow-gate-bypass through scenario overrides.
    if hasattr(args, "seed") and args.seed is not None:
        overrides["seed"] = int(args.seed)
    if getattr(args, "allow_gate_bypass", False):
        overrides["run_validator"] = False
        print("[sim] WARNING: --allow-gate-bypass set, epi-validity gate disabled")

    print(f"[sim] scenario={args.scenario}  use_db={args.use_db}  "
          f"days={overrides.get('days', 'scenario-default')}  "
          f"seed={overrides.get('seed', 42)}  "
          f"gate={'BYPASSED' if not overrides.get('run_validator', True) else 'enforced'}")
    result = run_scenario(args.scenario, base, overrides=overrides or None)

    peak_I = float(result.city_total("I").max())
    final_D = float(result.city_total("D")[-1])
    peak_day = int(result.city_total("I").argmax())
    print(f"[sim] peak I = {peak_I:,.0f} (day {peak_day})   final D = {final_D:,.1f}")

    gate = result.epi_validity.get("metapop_seirvd", {})
    if gate:
        print(f"[sim] epi-validity gate: {gate.get('status', 'n/a')}")
        for v in gate.get("violations", [])[:5]:
            print(f"   · {v}")

    if args.out:
        import numpy as np

        np.savez(
            args.out,
            state=result.state,
            incidence=result.incidence,
            days=result.days,
            district_names=np.array(result.district_names),
        )
        print(f"[sim] wrote trajectory: {args.out}")


def cmd_mcp_server(args) -> None:
    """`python -m simulation mcp-server` — Stage 6a ARIA epi MCP stdio.

    Defaults to ndjson over stdin/stdout (one JSON-RPC message per line).
    With ``--list-tools`` dumps the declarative schema and exits without
    blocking, so CI / docs / UI scaffolding can consume the contract
    without spawning the server.
    """
    from simulation.server import EpiMCPServer, run_stdio_server

    artifacts_dir = None
    if getattr(args, "artifacts_dir", None):
        from pathlib import Path as _Path

        artifacts_dir = _Path(args.artifacts_dir)

    if getattr(args, "list_tools", False):
        import json as _json

        srv = EpiMCPServer(artifacts_dir=artifacts_dir)
        print(_json.dumps(
            {"tools": srv.list_tools()},
            ensure_ascii=False, indent=2,
        ))
        return

    rc = run_stdio_server(artifacts_dir=artifacts_dir)
    if rc != 0:
        sys.exit(rc)


__all__ = [
    "cmd_sim",
    "cmd_mcp_server",
]
