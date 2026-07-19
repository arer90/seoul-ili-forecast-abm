"""simulation.llm_compare.orchestrator
========================================
The async control unit for the multi-agent ARIA research layer (S2).

Fans the routed :mod:`~simulation.llm_compare.specialists` out concurrently under
a semaphore (a single local Ollama serialises on one device, so unbounded
"parallelism" just queues — the bound is honest), lets each write receipted facts
to the shared :class:`~simulation.llm_compare.blackboard.EvidenceBlackboard`, then
synthesises one Analyst answer over the board and routes it through the mandatory
grounding gate. Control flow is code-driven asyncio, NOT LLM-driven routing, so the
verifier gate is non-bypassable and the LLM can never emit an unreceipted number.
"""
from __future__ import annotations

import asyncio

from simulation.llm_compare.blackboard import EvidenceBlackboard
from simulation.llm_compare.specialists import route_specialists, run_specialist

__all__ = ["orchestrate", "run_orchestrated"]


async def orchestrate(query: str, *, server=None, blackboard: EvidenceBlackboard | None = None,
                      limit: int = 2, mock: bool = False,
                      host: str = "http://127.0.0.1:11434",
                      args_by_tool: dict | None = None, runner=None,
                      on_specialist=None, memory=None, remember: bool = True) -> dict:
    """Run the routed specialists concurrently, then synthesise + gate one answer.

    Args:
        query: the advisory question.
        server: an ``EpiMCPServer`` (constructed if None) — the read-only tool surface.
        blackboard: shared evidence board (constructed if None).
        limit: max concurrent specialists (semaphore bound; default 2 — a single
            local Ollama serialises, so a higher bound only queues).
        mock: offline synthesis (mock Analyst) — no Ollama.
        host: Ollama daemon URL for the live Analyst.
        args_by_tool: optional per-tool argument overrides passed to specialists.
        runner: specialist runner (default :func:`run_specialist`; injectable for tests).
        on_specialist: optional callback(summary) fired as each specialist finishes
            (fact-level streaming).

    Returns:
        ``{query, specialists:[...], blackboard:[...], final_answer, verification,
        gate, grounded}``. ``grounded`` is True iff the deterministic verifier AND
        the runtime guardrail both pass.

    Side effects: calls the specialists' read-only tools; a live synthesis calls
    Ollama. No disk writes.
    """
    from simulation.server.mcp_epi import EpiMCPServer
    server = server if server is not None else EpiMCPServer()
    bb = blackboard if blackboard is not None else EvidenceBlackboard()
    run = runner or run_specialist
    specs = route_specialists(query)
    sem = asyncio.Semaphore(max(1, int(limit)))

    async def _one(spec):
        async with sem:
            summary = await run(spec, server, bb, args_by_tool=args_by_tool)
            if on_specialist is not None:
                on_specialist(summary)
            return summary

    summaries = await asyncio.gather(*[_one(s) for s in specs])

    # ── synthesis over the blackboard (Analyst) + mandatory gate ──────────────
    from simulation.llm_compare.aria_multiagent import MultiAgentARIA, verify_grounding
    from simulation.llm_compare.agentic_rag import guardrail
    gold = bb.facts_for_verifier()
    fact_block = "\n".join(gold) if gold else "(no receipted facts available)"
    # S3 — inject prior VERIFIED answers as CONTEXT only. They add no gold numbers;
    # the current answer is still gated against the current tool receipts, so a
    # remembered number the current tools did not return is flagged spurious.
    prior = memory.retrieve(query, k=3) if memory is not None else []
    prior_block = ""
    if prior:
        prior_block = ("\n\n참고 — 과거 검증된 답변(문맥용, 여기서 새 숫자를 인용하지 말 것):\n"
                       + "\n".join(f"- {p.get('final_answer', '')}" for p in prior))
    crew = MultiAgentARIA(mock=mock, host=host)
    a_prompt = (f"질의: {query}\n\n전문 에이전트들이 도구로 확정한 fact(이 숫자만 인용):\n"
                f"{fact_block}{prior_block}\n\n위 fact만 근거로 2~3문장 역학 자문 답을 "
                "작성하라. fact에 없는 숫자는 절대 쓰지 마라.")
    draft = await asyncio.to_thread(crew._ask, "analyst", a_prompt)
    vr = verify_grounding(draft, gold)
    gate = guardrail(draft, gold, question=query)
    grounded = bool(vr["grounded"] and gate["safe"])
    board = [{"agent": e.agent, "claim": e.claim, "value": e.value,
              "provenance": e.provenance} for e in bb.snapshot()]
    # persist a gate-passed answer for future runs to learn from. The CALLER sets
    # remember=False for mock/structural runs (mock output is not real advisory).
    if memory is not None and remember and grounded:
        receipts = [{"claim": b["claim"], "tool": (b["provenance"] or {}).get("tool")}
                    for b in board
                    if isinstance(b["value"], (int, float)) and not isinstance(b["value"], bool)]
        memory.remember(query, draft, tool_receipts=receipts,
                        verification={"grounded": grounded, "n_spurious": vr["n_spurious"]})
    return {
        "query": query,
        "specialists": summaries,
        "blackboard": board,
        "prior_context_used": len(prior),
        "final_answer": draft,
        "verification": vr,
        "gate": gate,
        "grounded": grounded,
    }


def run_orchestrated(query: str, **kw) -> dict:
    """Sync wrapper around :func:`orchestrate` for the CLI (no running loop)."""
    return asyncio.run(orchestrate(query, **kw))
