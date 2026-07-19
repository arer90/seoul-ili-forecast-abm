"""simulation.cli.aria_commands
================================
`python -m simulation aria` — the on-path P4 entry to the multi-agent ARIA
advisory layer.

This wires the existing 3-role AutoGen crew (:mod:`simulation.llm_compare.
aria_multiagent`) on-path behind a single CLI, with three guarantees the shipped
single-pass path never had:

1. **FAIL-LOUD backend probe** — a real advisory run requires a live model. With
   no live Ollama and no explicit ``--mock`` the command exits nonzero and leaks
   no answer (G-237: never silently emit mock prose that looks like reasoning).
2. **Evidence blackboard** — the crew surfaces every validated fact onto a
   provenance-tagged store; a number without a tool receipt cannot enter it.
3. **Mandatory delivery gate** — ``verify_grounding`` (deterministic CoVe
   arbiter) + :func:`simulation.llm_compare.agentic_rag.guardrail`; an answer with
   an ungrounded number is routed for review and the command exits nonzero, never
   delivering it as advisory.

``--mock`` runs the offline structural check (no Ollama); its output is clearly
labelled and is NOT deliverable advisory.
"""
from __future__ import annotations

import sys

from simulation.llm_compare.aria_multiagent import ollama_available as _ollama_available

__all__ = ["cmd_aria"]


def _render(res: dict, gate: dict, *, mock: bool) -> str:
    """Human-readable render of a consult: trace, blackboard receipts, gate."""
    lines = []
    if mock:
        lines.append("═══ ARIA (MOCK — offline structural check, NOT advisory) ═══")
    else:
        lines.append("═══ ARIA multi-agent advisory ═══")
    lines.append(f"질의: {res['query']}")
    lines.append("")
    lines.append("─ agent trace ─")
    for s in res["trace"]:
        head = (s["output"] or "").replace("\n", " ")[:88]
        lines.append(f"  [{s['role']:18s} {s.get('model',''):12s}] {head}")
    lines.append("")
    lines.append("─ evidence blackboard (every number ← a tool receipt) ─")
    for e in res.get("blackboard", []):
        if e.get("value") is None:
            continue
        tool = (e.get("provenance") or {}).get("tool", "?")
        lines.append(f"  {e['claim']}={e['value']}  ←  {tool}")
    lines.append("")
    lines.append(f"─ final answer ─\n  {res['final_answer']}")
    v = res["verification"]
    lines.append("")
    lines.append(f"─ delivery gate ─  grounded={v['grounded']} "
                 f"(recall={v['fact_recall']}, spurious={v['n_spurious']}, "
                 f"revised={res['revised']})  gate={gate['action']}")
    if not gate["safe"]:
        lines.append(f"  ⚠ ROUTED FOR REVIEW — {gate['reason']} (NOT delivered as advisory)")
    return "\n".join(lines)


def _render_deep(res: dict, *, mock: bool) -> str:
    """Render an orchestrated (6-specialist) run: specialists, receipts, gate."""
    lines = []
    lines.append("═══ ARIA deep (MOCK — structural check, NOT advisory) ═══" if mock
                 else "═══ ARIA deep advisory — 6 read-only research specialists ═══")
    lines.append(f"질의: {res['query']}")
    lines.append("")
    lines.append("─ specialists (each bounded to read-only epi.* tools) ─")
    for s in res["specialists"]:
        tools = ",".join(r["tool"] for r in s["receipts"])
        lines.append(f"  [{s['role']:26s}] {s['facts_written']} facts  ←  {tools}")
    lines.append("")
    lines.append("─ evidence blackboard (every number ← a tool receipt) ─")
    shown = 0
    for e in res["blackboard"]:
        v = e.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        tool = (e.get("provenance") or {}).get("tool", "?")
        lines.append(f"  {e['claim']}={v}  ←  {tool}")
        shown += 1
        if shown >= 24:
            lines.append("  … (truncated)")
            break
    if shown == 0:
        lines.append("  (no numeric facts available from tools — reported honestly)")
    lines.append("")
    lines.append(f"─ synthesis ─\n  {res['final_answer']}")
    v, g = res["verification"], res["gate"]
    lines.append("")
    lines.append(f"─ delivery gate ─  grounded={v['grounded']} "
                 f"(spurious={v['n_spurious']})  gate={g['action']}")
    if not g["safe"]:
        lines.append(f"  ⚠ ROUTED FOR REVIEW — {g['reason']} (NOT delivered as advisory)")
    return "\n".join(lines)


def cmd_aria(args) -> int | None:
    """`python -m simulation aria` — run one advisory query through the crew + gate.

    Args:
        args: argparse namespace with ``query`` (required), ``root`` (results root
            for grounding facts; default active results), ``mock`` (offline
            structural check), ``host`` (Ollama daemon URL). ``deep``/``stream``
            are accepted for forward compatibility (S2 orchestrator).

    Returns:
        None (exit 0) on a grounded advisory answer or a labelled mock check.

    Raises:
        SystemExit: code 2 if ``--query`` is missing OR the delivery gate rejects
            the answer (ungrounded → routed for review); code 1 if no live backend
            and ``--mock`` was not requested (fail-loud, no answer emitted).

    Side effects: live runs call the local Ollama HTTP API (no API key, no DB, no
    writes). Prints the trace, blackboard receipts, answer, and gate decision.
    """
    # S3 — refresh few-shot exemplars from verified history (no query needed).
    if getattr(args, "refresh", False):
        from simulation.llm_compare.dspy_refresh import refresh_exemplars
        summ = refresh_exemplars()
        print(f"aria: refreshed {summ['n_exemplars']} exemplar(s) from "
              f"{summ['n_verified']} verified answer(s) [{summ['method']}] → "
              f"{summ['path']}")
        return None

    query = getattr(args, "query", None)
    if not query:
        print("aria: --query is required (or use --refresh)", file=sys.stderr)
        raise SystemExit(2)

    root = getattr(args, "root", None)
    mock = bool(getattr(args, "mock", False))
    host = getattr(args, "host", None) or "http://127.0.0.1:11434"

    # 1) FAIL-LOUD backend probe — no live model + no explicit --mock ⇒ abort loud,
    #    emit no answer (never a silent mock that looks like real reasoning).
    if not mock and not _ollama_available(host):
        print(f"aria: FAIL-LOUD — no live Ollama at {host}. Start Ollama and pull "
              "the crew models, or pass --mock for an offline structural check "
              "(whose output is NOT deliverable advisory).", file=sys.stderr)
        raise SystemExit(1)

    deep = bool(getattr(args, "deep", False))
    stream = bool(getattr(args, "stream", False))
    from simulation.llm_compare.agentic_rag import guardrail
    from simulation.llm_compare.memory import VerifiedMemory
    memory = VerifiedMemory()   # cross-run learning; only real gate-passed answers persist

    if deep:
        # 2d) S2 — fan the 6 read-only specialists out concurrently, each writing
        #     tool-receipted facts to the blackboard, then synthesise + gate. S3 —
        #     inject prior verified answers as context; persist a real grounded one.
        from simulation.llm_compare.orchestrator import run_orchestrated
        on_spec = ((lambda s: print(f"  ✓ {s['role']} — {s['facts_written']} facts"))
                   if stream else None)
        res = run_orchestrated(query, mock=mock, host=host, limit=2,
                               on_specialist=on_spec, memory=memory,
                               remember=not mock)
        gate, grounded = res["gate"], res["grounded"]
        print(_render_deep(res, mock=mock))
    else:
        # 2) Run the blackboard-backed 3-role crew (Retriever → Analyst → Verifier).
        from simulation.llm_compare.aria_multiagent import MultiAgentARIA
        crew = MultiAgentARIA(mock=mock, host=host)
        res = crew.consult(query, root=root)
        # 3) MANDATORY delivery gate: verify_grounding (in res) AND the runtime
        #    guardrail. Ungrounded numeric content is routed for review.
        gate = guardrail(res["final_answer"], res["retrieved_facts"], question=query)
        grounded = res["verification"]["grounded"] and gate["safe"]
        print(_render(res, gate, mock=mock))
        if not mock and grounded:   # S3 — remember a real, gate-passed crew answer
            receipts = [{"claim": e["claim"], "tool": (e["provenance"] or {}).get("tool")}
                        for e in res.get("blackboard", [])
                        if isinstance(e.get("value"), (int, float))
                        and not isinstance(e.get("value"), bool)]
            memory.remember(query, res["final_answer"], tool_receipts=receipts,
                            verification={"grounded": grounded,
                                          "n_spurious": res["verification"]["n_spurious"]})

    # Exit code carries the gate verdict; mock is a structural pass (exit 0).
    if not mock and not grounded:
        raise SystemExit(2)  # gate rejected — routed for review, not delivered
    return None
