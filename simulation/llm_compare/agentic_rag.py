"""Agentic-RAG loop + critic agent + runtime guardrail for the ARIA layer.

Three additive, offline, dependency-injected pieces (default behaviour unchanged
unless they are explicitly invoked / MPH_AGENTIC_RAG=1):

1. ``agentic_retrieve`` — a reasoning loop around any retriever: retrieve, judge
   whether the context is sufficient, and if not REFORMULATE the query and
   re-retrieve (multi-round), accumulating de-duplicated context. Generalises the
   Self-Ask decomposition (subq.py) to the retrieval stage.
2. ``critic_review`` — a critic SEPARATE from the generator (Zheng et al. 2023:
   judge != generator avoids self-bias). It checks numeric grounding (every number
   in the answer must appear in the retrieved context) plus an optional LLM verdict.
3. ``guardrail`` — a runtime gate that inspects the final answer BEFORE delivery
   and returns an action (pass / rewrite / route_for_review) when high-risk
   (ungrounded numeric) content is detected, extending ARIA's numeric-grounding gate.

Everything is dependency-injected (retriever/generator/judge are callables or
LLMBackend-like objects), so the module is fully testable offline with stubs and
never requires a live model.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Optional

__all__ = [
    "agentic_rag_enabled",
    "agentic_retrieve",
    "critic_review",
    "guardrail",
    "agentic_answer",
]

_NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?%?")  # standalone numbers; skips R0/H1N1


def agentic_rag_enabled() -> bool:
    """True iff MPH_AGENTIC_RAG is truthy (default OFF)."""
    return os.environ.get("MPH_AGENTIC_RAG", "0") not in ("", "0", "false", "False")


def _hit_text(h: dict) -> str:
    return " ".join(str(h.get(k, "")) for k in ("title", "abstract", "snippet"))


def _ask(judge, prompt: str) -> Optional[str]:
    """One judge call -> reply text, or None on failure."""
    if judge is None:
        return None
    try:
        r = judge.generate(prompt, temperature=0.0, max_tokens=128)
        if getattr(r, "error", ""):
            return None
        return getattr(r, "text", "") or None
    except Exception:
        return None


def _sufficient(question: str, hits: list[dict], judge, min_hits: int) -> bool:
    """Is the accumulated context enough to answer? LLM judge, heuristic fallback."""
    if len(hits) < min_hits:
        return False
    reply = _ask(judge, (
        "Can the QUESTION be answered using ONLY the CONTEXT below? Answer YES or "
        "NO.\nQUESTION: " + question + "\nCONTEXT:\n"
        + "\n".join(_hit_text(h)[:300] for h in hits)
    ))
    if reply is None:
        return len(hits) >= min_hits          # heuristic when no judge
    return reply.strip().upper().startswith("Y")


def _reformulate(question: str, hits: list[dict], judge, prev: list[str]) -> Optional[str]:
    """Propose a fresh retrieval query (Self-Ask style); None if nothing new."""
    reply = _ask(judge, (
        "The current search did not retrieve enough relevant context for the "
        "QUESTION. Write ONE alternative, broader or more specific search query "
        "(different from the previous ones) that would surface better evidence. "
        "Output only the query.\nQUESTION: " + question
        + "\nPREVIOUS QUERIES: " + " | ".join(prev)
    ))
    if reply:
        q = reply.strip().splitlines()[0].strip().strip('"')
        if q and q not in prev:
            return q
    return None


def agentic_retrieve(question: str, retriever: Callable[[str, int], Optional[list]],
                     *, k: int = 5, max_rounds: int = 3, judge=None,
                     min_hits: int = 2) -> dict:
    """Multi-round agentic retrieval: retrieve -> judge sufficiency -> reformulate.

    Args:
        question: the user question.
        retriever: callable(query, k) -> list[hit dict] | None.
        k: hits per retrieval round.
        max_rounds: maximum retrieval rounds (>=1).
        judge: optional LLMBackend-like judge for sufficiency + reformulation
            (heuristics used when None — offline-safe).
        min_hits: minimum de-duplicated hits to consider context sufficient.

    Returns:
        ``{hits, rounds, queries, sufficient}`` — accumulated de-duplicated hits,
        the number of rounds run, the query sequence, and whether the stopping
        criterion judged the context sufficient.

    Side effects: up to ``max_rounds`` retriever calls + up to ``max_rounds`` judge
    calls. No global state.
    """
    queries = [question]
    cur = question
    seen, acc = set(), []
    rounds = 0
    sufficient = False
    for rounds in range(1, max(1, max_rounds) + 1):
        hits = retriever(cur, k) or []
        for h in hits:
            key = h.get("id") or h.get("pmid") or h.get("title") or _hit_text(h)[:60]
            if key not in seen:
                seen.add(key)
                acc.append(h)
        sufficient = _sufficient(question, acc, judge, min_hits)
        if sufficient:
            break
        nxt = _reformulate(question, acc, judge, queries)
        if not nxt:
            break
        queries.append(nxt)
        cur = nxt
    return {"hits": acc, "rounds": rounds, "queries": queries, "sufficient": sufficient}


def ungrounded_numbers(answer: str, contexts) -> list[str]:
    """Numbers in the answer that do NOT appear verbatim in any context (the core
    numeric-grounding check). Empty list == every number is grounded."""
    ctx = " ".join(contexts) if not isinstance(contexts, str) else contexts
    ctx_nums = set(_NUM_RE.findall(ctx))
    return [n for n in _NUM_RE.findall(answer or "") if n not in ctx_nums]


def critic_review(question: str, answer: str, contexts, *, judge=None) -> dict:
    """Critic (separate from the generator): numeric grounding + optional LLM verdict.

    Returns ``{grounded: bool, ungrounded_numbers: [...], llm_verdict: str|None,
    issues: [...]}``. ``grounded`` is False if any numeric claim is unsupported by
    the context (the load-bearing, deterministic check); the LLM verdict is an
    optional secondary signal.
    """
    raw = list(contexts) if not isinstance(contexts, str) else [contexts]
    ctx_texts = [_hit_text(h) if isinstance(h, dict) else str(h) for h in raw]
    bad_nums = ungrounded_numbers(answer, ctx_texts)
    issues = []
    if bad_nums:
        issues.append(f"ungrounded numbers: {bad_nums}")
    verdict = _ask(judge, (
        "As a strict reviewer, does the ANSWER make any claim not supported by the "
        "CONTEXT? Reply SUPPORTED or UNSUPPORTED and a one-line reason.\nQUESTION: "
        + question + "\nANSWER: " + (answer or "") + "\nCONTEXT:\n"
        + "\n".join(ctx_texts)
    ))
    if verdict and verdict.strip().upper().startswith("UNSUPPORTED"):
        issues.append("llm_critic: unsupported claim flagged")
    return {
        "grounded": not bad_nums,
        "ungrounded_numbers": bad_nums,
        "llm_verdict": verdict.strip() if verdict else None,
        "issues": issues,
    }


def guardrail(answer: str, contexts, *, critic: Optional[dict] = None,
              question: str = "", judge=None) -> dict:
    """Runtime gate: inspect the answer before delivery, decide an action.

    Args:
        answer: the generated answer.
        contexts: retrieved context (list or str).
        critic: optional pre-computed :func:`critic_review` result (else computed).
        question / judge: passed through to the critic when computing it.

    Returns:
        ``{action: "pass"|"route_for_review", safe: bool, reason, critic}``.
        Ungrounded numeric content -> ``route_for_review`` (never silently
        delivered); otherwise ``pass``. The answer is NEVER auto-rewritten away —
        high-risk output is routed for human ratification (ARIA's standing rule
        that the LLM layer requires human sign-off).

    Side effects: one critic computation (which may make one judge call).
    """
    c = critic if critic is not None else critic_review(question, answer, contexts, judge=judge)
    if not c["grounded"] or c["issues"]:
        return {"action": "route_for_review", "safe": False,
                "reason": "; ".join(c["issues"]) or "ungrounded content",
                "critic": c}
    return {"action": "pass", "safe": True, "reason": "grounded", "critic": c}


def agentic_answer(question: str, *, retriever: Callable[[str, int], Optional[list]],
                   generator: Callable[[str, list], str], judge=None,
                   k: int = 5, max_rounds: int = 3) -> dict:
    """End-to-end agentic-RAG answer: retrieve-loop -> generate -> critic -> guardrail.

    Args:
        question: the user question.
        retriever: callable(query, k) -> list[hit dict] | None.
        generator: callable(question, hits) -> answer str.
        judge: optional judge backend (sufficiency, reformulation, critic).
        k / max_rounds: retrieval breadth / loop bound.

    Returns:
        ``{answer, retrieval: {...}, critic: {...}, guardrail: {...}}`` — the answer
        plus full provenance of the agentic loop, the critic review, and the
        delivery decision. Fully offline + deterministic given deterministic
        injected callables.
    """
    retrieval = agentic_retrieve(question, retriever, k=k, max_rounds=max_rounds,
                                 judge=judge)
    hits = retrieval["hits"]
    answer = generator(question, hits)
    critic = critic_review(question, answer, hits, judge=judge)
    gate = guardrail(answer, hits, critic=critic)
    return {"answer": answer, "retrieval": retrieval, "critic": critic, "guardrail": gate}
