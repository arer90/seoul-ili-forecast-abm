"""simulation.llm_compare.specialists
=======================================
The six read-only **research specialists** that sit between the ABM and ARIA.

Each specialist is bounded to a fixed allowlist of read-only ``epi.*`` MCP tools
(:mod:`simulation.server.mcp_epi`) and does exactly two things:

1. Call ONLY its allowlisted tools (in-process, through ``EpiMCPServer.call_tool``,
   which already enforces read-only via ``validate_read_only`` and exposes no
   mutating tool).
2. Write the tool's returned numbers onto the shared evidence blackboard,
   provenance-tagged to the tool call — so every number an agent contributes is,
   by construction, a tool receipt. A specialist **never generates a number**; it
   surfaces what a deterministic engine returned.

This is what keeps the layer inside the read-only / no-transmission-driver
constraint: the LLM narration (added by the orchestrator) can only cite blackboard
facts, and the delivery gate rejects any number without a receipt.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from simulation.llm_compare.blackboard import EvidenceBlackboard

__all__ = ["Specialist", "SPECIALISTS", "extract_leaf_facts", "run_specialist",
           "route_specialists"]

# provenance/bookkeeping keys carry versions and timestamps, not epi facts
_SKIP_KEYS = {"provenance", "server_version", "db_vintage_ts", "checker",
              "elapsed_ms", "tool"}

# Sensible default arguments for tools that require them, so a specialist surfaces
# real facts (city-level; gu-panel real data is unavailable per the thesis) rather
# than an arg error. The simulation uses a short, DB-free config for responsiveness.
_DEFAULT_ARGS = {
    "epi.rt_estimate":     {"gu": "seoul_city"},
    "epi.outbreak_detect": {"gu": "seoul_city"},
    "epi.forecast":        {"gu": "seoul_city"},
    "epi.scenario_run":    {"scenario": "baseline", "days": 60, "use_db": False},
    "epi.coupled_forward": {"n_agents": 3000, "n_seeds": 3},   # small config for responsiveness
}


@dataclass(frozen=True)
class Specialist:
    """One read-only research specialist.

    Attributes:
        name: agent id (blackboard author), e.g. ``"forecast_intelligence"``.
        role: human-readable role.
        tools: the ONLY ``epi.*`` tool names this agent may call (allowlist).
        keywords: query terms that route to this specialist.
    """

    name: str
    role: str
    tools: tuple[str, ...]
    keywords: tuple[str, ...]


SPECIALISTS: tuple[Specialist, ...] = (
    Specialist("forecast_intelligence", "Forecast Intelligence",
               ("epi.forecast", "epi.model_compare"),
               ("forecast", "예측", "champion", "챔피언", "wis", "model", "모델")),
    Specialist("spatial_transmission", "Spatial Transmission",
               ("epi.rt_estimate", "epi.outbreak_detect", "epi.coupled_forward"),
               ("spatial", "공간", "district", "자치구", "rt", "outbreak", "유행", "전파")),
    Specialist("simulation_experiment", "Simulation Experiment",
               ("epi.scenario_run", "epi.coupled_forward"),
               ("scenario", "시나리오", "counterfactual", "반사실", "simulate")),
    Specialist("intervention_optimization", "Intervention Optimization",
               ("epi.scenario_run", "epi.lead_time_analysis"),
               ("vaccine", "백신", "allocat", "배분", "intervention", "개입",
                "lead", "선행")),
    Specialist("deep_evidence_research", "Deep Evidence Research",
               ("epi.literature_rag",),
               ("literature", "문헌", "evidence", "근거", "citation", "인용")),
    Specialist("statistical_verification", "Statistical Verification",
               ("epi.validity_check", "epi.shap_features"),
               ("valid", "검증", "shap", "driver", "요인", "significance", "유의")),
)


def extract_leaf_facts(content) -> list[tuple[str, object]]:
    """Flatten a tool-return payload into ``(dotted_path, scalar)`` leaves.

    Skips bookkeeping keys (``provenance`` etc.). Used to surface every numeric
    leaf a tool returned so it can be receipted onto the blackboard.

    Args:
        content: a tool ``CallResult.content`` (dict / list / scalar).

    Returns:
        List of ``(path, value)`` for scalar leaves (numbers, bools, non-empty
        strings). Deterministic; never raises.
    """
    def _walk(node, prefix: str) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _SKIP_KEYS:
                    continue
                out += _walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                out += _walk(v, f"{prefix}[{i}]")
        elif isinstance(node, bool):
            out.append((prefix, node))
        elif isinstance(node, (int, float)):
            out.append((prefix, node))
        elif isinstance(node, str) and node:
            out.append((prefix, node))
        return out

    return _walk(content, "")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def call_allowlisted(server, spec: Specialist, name: str, arguments: dict):
    """Call a tool only if it is on the specialist's allowlist.

    Returns the tool ``CallResult``, or an error ``CallResult`` if the tool is not
    allowlisted for this agent (the bounded-tool boundary — an agent cannot reach
    outside its remit). Read-only is enforced downstream by ``call_tool``.
    """
    from simulation.server.mcp_epi import CallResult
    if name not in spec.tools:
        return CallResult(
            content={"error": "tool_not_allowed", "tool": name,
                     "agent": spec.name, "allowed": list(spec.tools)},
            is_error=True)
    return server.call_tool(name, arguments or {})


async def run_specialist(spec: Specialist, server, blackboard: EvidenceBlackboard,
                         *, args_by_tool: dict | None = None) -> dict:
    """Run one specialist: call its allowlisted tools, receipt facts to the board.

    Args:
        spec: the specialist definition (allowlist + role).
        server: an ``EpiMCPServer`` (read-only in-process tool surface).
        blackboard: the shared evidence board to write receipted facts to.
        args_by_tool: optional ``{tool_name: arguments}`` overrides.

    Returns:
        ``{agent, role, facts_written, receipts:[{tool, is_error, status}]}``.

    Side effects: calls the specialist's read-only tools (may read DB/artifacts
    off the event loop via ``asyncio.to_thread``); appends receipted facts to the
    blackboard. Writes nothing to disk.
    """
    args_by_tool = args_by_tool or {}
    facts_written = 0
    receipts = []
    for tool in spec.tools:
        args = args_by_tool.get(tool) or _DEFAULT_ARGS.get(tool, {})
        res = await asyncio.to_thread(call_allowlisted, server, spec, tool, args)
        content = res.content if isinstance(res.content, (dict, list)) else {"value": res.content}
        prov = {"tool": tool, "args": args, "return_payload": content}
        status = content.get("status") if isinstance(content, dict) else None
        for path, val in extract_leaf_facts(content):
            # receipt numbers (grounded facts) and the top-level status only —
            # skip free-text messages to keep the board a facts surface.
            if _is_number(val) or path == "status":
                try:
                    blackboard.append(spec.name, f"{spec.name}.{path}",
                                      value=val, provenance=prov)
                    if _is_number(val):
                        facts_written += 1
                except ValueError:
                    pass  # unreachable: val came from return_payload
        receipts.append({"tool": tool, "is_error": bool(res.is_error),
                         "status": status})
    return {"agent": spec.name, "role": spec.role,
            "facts_written": facts_written, "receipts": receipts}


def route_specialists(query: str) -> tuple[Specialist, ...]:
    """Which specialists a query needs (keyword routing; all if none match)."""
    q = (query or "").lower()
    picked = tuple(s for s in SPECIALISTS if any(k in q for k in s.keywords))
    return picked or SPECIALISTS
