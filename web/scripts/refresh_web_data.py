#!/usr/bin/env python3
"""M4: refresh ALL web data artifacts from the LIVE pipeline outputs (db→web sync).

Single entry that regenerates the web's data so the dashboard reflects the latest
trained run + DB — eliminating the manual-builder / frozen-aggregate drift the user
flagged ("db에 있는 데이터를 계속하니까 동기화나 내용이 안 맞아"). Wire this into the
post-train / post-collect orchestration so the web is always current (real-time).

Runs (best-effort, each isolated; a missing/failing builder doesn't abort the rest):
  1. trained-models.json     ← per_model_eval/per_model_metrics.csv  (model list/rank)
  2. seir-metapop-init.json  ← DB commuter + population               (Map3D init)
  3. static aggregates       ← DB                                     (choropleth, edges)

Forecast / SHAP / scenario data flow LIVE via the MCP epi tools (see mcp_epi.py),
so they need no static rebuild — this script handles the file-backed aggregates only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

#: (logical name, command). Order = data dependency.
STEPS: list[tuple[str, list[str]]] = [
    ("trained-models", [PY, str(ROOT / "web" / "scripts" / "build_trained_models.py")]),
    ("seir-metapop-init", [PY, str(ROOT / "simulation" / "scripts" / "export_seir_metapop_init.py")]),
    ("abm-scenarios", [PY, str(ROOT / "simulation" / "scripts" / "export_abm_scenarios.py")]),
    ("static-aggregates", [PY, str(ROOT / "web" / "scripts" / "build-static-aggregates.py")]),
    ("subway-lines", [PY, str(ROOT / "web" / "scripts" / "build_subway_lines.py")]),
    ("agent-trips", [PY, str(ROOT / "web" / "scripts" / "build_agent_trips.py")]),
    ("bus-stops", [PY, str(ROOT / "web" / "scripts" / "build_bus_stops.py")]),
    ("bus-routes", [PY, str(ROOT / "web" / "scripts" / "build_bus_routes.py")]),
    ("subway-stations", [PY, str(ROOT / "web" / "scripts" / "build_subway_stations.py")]),
    ("air-env", [PY, str(ROOT / "web" / "scripts" / "build_air_env.py")]),
    ("weather", [PY, str(ROOT / "web" / "scripts" / "build_weather.py")]),
    ("realtime-poi", [PY, str(ROOT / "web" / "scripts" / "build_realtime_poi.py")]),
    ("disease-vax", [PY, str(ROOT / "web" / "scripts" / "build_disease_vax.py")]),
    ("schools", [PY, str(ROOT / "web" / "scripts" / "build_schools.py")]),
    ("aria-wiki", [PY, str(ROOT / "web" / "scripts" / "build_aria_wiki.py")]),
    # NOTE (D3 correction 2026-06-06): the Turso-seed export is NOT wired here —
    # the user runs everything LOCALLY (no Turso/Vercel). Turso is the deployed-
    # build path only; locally the web reads LIVE data via /api/mcp/[tool] →
    # MCP server (epi.forecast / rt_estimate / scenario_run compute dynamically
    # off the local DB). export-turso.py stays available (+ a vintage row) for
    # anyone who DOES deploy to Turso, but it is a manual deploy step, not a
    # per-refresh local step (it would dump the 12 GB DB on every refresh).
]


def refresh(steps: list[tuple[str, list[str]]] | None = None,
            timeout: int = 600) -> dict[str, str]:
    """Run each web-data builder; return {name: status}.

    status ∈ {"ok", "fail(rc=N)", "error(Type)", "missing-script"}. Never raises
    (degrade-and-continue) so one broken builder can't block the others.
    """
    steps = STEPS if steps is None else steps
    results: dict[str, str] = {}
    for name, cmd in steps:
        script = Path(cmd[1])
        if not script.exists():
            results[name] = "missing-script"
            continue
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                               text=True, timeout=timeout)
            results[name] = "ok" if r.returncode == 0 else f"fail(rc={r.returncode})"
        except Exception as e:  # noqa: BLE001 — isolate every builder
            results[name] = f"error({type(e).__name__})"
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh web data from live pipeline outputs")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these named steps (default: all).")
    args = ap.parse_args()
    steps = STEPS if not args.only else [s for s in STEPS if s[0] in set(args.only)]
    res = refresh(steps)
    for n, s in res.items():
        print(f"[refresh-web] {n}: {s}")
    n_ok = sum(1 for s in res.values() if s == "ok")
    print(f"[refresh-web] {n_ok}/{len(res)} ok")
    return 0 if n_ok == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
