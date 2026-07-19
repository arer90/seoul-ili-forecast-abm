"""
simulation.llm_compare.subq
===========================
Self-Ask sub-question (SubQ) decomposition as a **backend-agnostic, reusable**
deep module — the general primitive extracted from ``aria_grounding``.

Self-Ask (Press et al. 2022, *Measuring and Narrowing the Compositionality Gap
in Language Models*, arXiv:2210.03350) answers a **compositional** question by
explicitly eliciting atomic *follow-up* sub-questions, answering each one, and
then **recomposing** a final answer from the intermediate answers. The original
method triggers decomposition with the literal phrase ``"Are follow up questions
needed here:"`` and terminates with ``"So the final answer is:"``; the follow-up
answers may be supplied by the model itself or by an external resource (search).

Why a *separate* module
-----------------------
``aria_grounding`` already contains a Self-Ask path, but it is **welded to the
numeric-grounding context**: ``_SUBQ_TEMPLATES`` (15 hard-coded epi keys),
``self_ask_decompose`` (assumes ``"key=value"`` facts), and ``self_ask_grounding``
(scores against numeric gold). That makes Self-Ask un-reusable for any other
fact pool. This module factors the *general* Self-Ask trajectory out:

    decompose(question, fact_pool) → [SubQ]            # atomic follow-ups
        → answer_subq(subq, fact_pool)  per sub-Q      # answer from the pool
        → recompose(question, answered) → final        # synthesize
        → verify(final, fact_pool)                      # CoVe-style check

It is **backend-agnostic**: every step works deterministically over a plain
``fact_pool`` (a list of ``"key=value"`` strings *or* free-text snippets), so it
runs with **no LLM** at all (the deterministic reference trajectory). When a
``backend`` with ``.generate(prompt, …) -> resp`` (``.text`` / ``.error``; the
``simulation.llm_compare.backends.LLMBackend`` contract) is supplied, the same
trajectory drives that backend (claude / Ollama / OpenAI-compat / Mock — any).

Improvements folded in from the agent-building survey (2023-2026)
----------------------------------------------------------------
  • **Explicit follow-up gating** (Self-Ask's "Are follow up questions needed
    here:"): ``needs_followup`` decides decomposition vs. a single direct answer,
    so atomic questions are not over-split (cost/latency control — AdaPlanner).
  • **Factored intermediate answering** (CoVe, Dhuliawala et al. 2023,
    arXiv:2309.11495, *Factored* variant): each sub-question is answered
    **independently** from the pool so a wrong draft cannot bias later answers.
  • **Verification pass** (CoVe / Chain-of-Verification): ``verify`` re-checks the
    recomposed answer against the pool and flags unsupported numeric claims.

This module is **read-only and side-effect-free** (no DB, no disk, no network of
its own — any network is the caller's ``backend``). It is the SSOT for the
*general* Self-Ask primitive; ``aria_grounding`` keeps its numeric-specialised
templates and can delegate the mechanics here without duplication.

CLI:  python -m simulation.llm_compare.subq --demo
      python -m simulation.llm_compare.subq --demo --mock   # drive a Mock backend
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol

__all__ = [
    "SubQ",
    "SelfAskResult",
    "needs_followup",
    "decompose",
    "answer_subq",
    "recompose",
    "verify",
    "self_ask",
    "FOLLOWUP_TRIGGER",
    "FINAL_TRIGGER",
    "SELF_ASK_INSTRUCTION",
]

# ── Self-Ask literal scaffold (Press et al. 2022) ─────────────────────────────
FOLLOWUP_TRIGGER = "Are follow up questions needed here:"
FINAL_TRIGGER = "So the final answer is:"
SELF_ASK_INSTRUCTION = (
    "복합 질문을 원자적 하위질문(Self-Ask)으로 나눠 하나씩 답한 뒤, 마지막에 "
    "종합하세요. 제공된 사실(fact pool)에 없는 수치는 지어내지 마세요.\n"
    f"{FOLLOWUP_TRIGGER}"
)

_NUM = re.compile(r"-?\d+\.?\d*")
# split a free-text/compound question into clause-level units (atomicity proxy);
# decimals (0.84) must NOT be treated as a clause boundary (epi outputs are
# decimal-heavy, see citation_trace.py's same guard).
_CLAUSE_SPLIT = re.compile(r"(?:(?<!\d)[?.;。](?!\d)|,|\band\b|\bvs\b|그리고|및|와|과)\s*")


class _Backend(Protocol):
    """Structural type for any ``simulation.llm_compare.backends.LLMBackend``."""

    def generate(self, prompt: str, *, max_tokens: int = ...): ...


# ── data containers ───────────────────────────────────────────────────────────
@dataclass
class SubQ:
    """One atomic Self-Ask follow-up question and (optionally) its answer.

    Attributes:
        question: the atomic follow-up question text (one fact / one hop).
        key: the fact-pool key this sub-question targets (``""`` if free-text).
        gold: the reference answer extracted from the pool (``""`` if unknown).
        answer: the produced intermediate answer (filled by ``answer_subq``).
        supported: whether ``answer`` is grounded in the pool (set by ``verify``).
    """

    question: str
    key: str = ""
    gold: str = ""
    answer: str = ""
    supported: Optional[bool] = None


@dataclass
class SelfAskResult:
    """Full Self-Ask trajectory: decomposition → intermediate answers → final.

    Attributes:
        question: the original compositional question.
        sub_questions: ordered list of answered ``SubQ``.
        final_answer: the recomposed final answer.
        decomposed: whether decomposition fired (``needs_followup`` True).
        verification: ``verify`` report over ``final_answer`` (recall / spurious).
        n_subq: convenience count of ``sub_questions``.
    """

    question: str
    sub_questions: list[SubQ] = field(default_factory=list)
    final_answer: str = ""
    decomposed: bool = True
    verification: dict = field(default_factory=dict)

    @property
    def n_subq(self) -> int:
        return len(self.sub_questions)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "decomposed": self.decomposed,
            "n_subq": self.n_subq,
            "sub_questions": [
                {"question": s.question, "key": s.key, "gold": s.gold,
                 "answer": s.answer, "supported": s.supported}
                for s in self.sub_questions
            ],
            "final_answer": self.final_answer,
            "verification": self.verification,
        }


# ── fact-pool helpers ─────────────────────────────────────────────────────────
def _normalize_pool(fact_pool: Iterable) -> list[tuple[str, str]]:
    """Normalize a heterogeneous fact pool into ``(key, value)`` pairs.

    Accepts ``"key=value"`` strings (key + value), bare snippets (key=snippet,
    value=""), or ``(key, value)`` / ``{key: value}`` mappings. Never raises.
    """
    pairs: list[tuple[str, str]] = []
    if isinstance(fact_pool, dict):
        return [(str(k).strip(), str(v).strip()) for k, v in fact_pool.items()]
    for f in fact_pool or []:
        if isinstance(f, (tuple, list)) and len(f) == 2:
            pairs.append((str(f[0]).strip(), str(f[1]).strip()))
        else:
            s = str(f)
            if "=" in s:
                k, _, v = s.partition("=")
                pairs.append((k.strip(), v.strip()))
            else:
                pairs.append((s.strip(), ""))
    return pairs


def _gold_numbers(fact_pool: Iterable) -> set[str]:
    gold: set[str] = set()
    for _k, v in _normalize_pool(fact_pool):
        m = _NUM.search(v)
        if m:
            gold.add(m.group())
    return gold


# ── (1) follow-up gating — Self-Ask "Are follow up questions needed here:" ─────
def needs_followup(question: str, fact_pool: Iterable) -> bool:
    """Decide whether the question warrants decomposition (Self-Ask gate).

    A question is *compositional* (worth decomposing) when the pool offers more
    than one independently-answerable fact OR the question itself spans multiple
    clauses. A single-fact, single-clause question is answered directly — this
    prevents pointless over-splitting (cost/latency, AdaPlanner-style gating).

    Args:
        question: the candidate question.
        fact_pool: the available facts (``"key=value"`` strings / snippets / map).

    Returns:
        ``True`` if decomposition should fire, else ``False``. Never raises.
    """
    pairs = _normalize_pool(fact_pool)
    if len(pairs) > 1:
        return True
    clauses = [c for c in _CLAUSE_SPLIT.split(question or "") if c.strip()]
    return len(clauses) > 1


# ── (2) decompose — one atomic follow-up per pool fact ────────────────────────
def decompose(
    question: str,
    fact_pool: Iterable,
    *,
    templates: Optional[dict] = None,
    phrase: Callable[[str], str] | None = None,
) -> list[SubQ]:
    """Decompose a compositional question into atomic Self-Ask sub-questions.

    One atomic follow-up is emitted per fact in the pool so the decomposition
    **never drops a fact** (Press et al. 2022). A caller may inject a
    ``templates`` map (key → ready question) for domain-specific phrasings —
    e.g. ``aria_grounding`` passes its 15 epi templates — without this module
    knowing any domain; absent a template, a generic ``"<key> 값은 얼마인가?"``
    is used. ``phrase`` overrides the generic phrasing for un-templated keys.

    Args:
        question: the original compositional question (kept for context).
        fact_pool: facts as ``"key=value"`` strings / snippets / mapping.
        templates: optional ``{key: question}`` overrides (domain phrasings).
        phrase: optional ``key -> question`` callable for un-templated keys.

    Returns:
        Ordered ``list[SubQ]`` (one per fact), each carrying ``key`` and ``gold``.
        Empty list if the pool is empty. Never raises.
    """
    templates = templates or {}
    pairs = _normalize_pool(fact_pool)
    subs: list[SubQ] = []
    for key, val in pairs:
        if key in templates:
            q = templates[key]
        elif phrase is not None:
            q = phrase(key)
        elif val:
            q = f"{key} 값은 얼마인가?"
        else:
            q = f"{key} 에 대해 무엇이 알려져 있는가?"
        subs.append(SubQ(question=q, key=key, gold=val))
    return subs


# ── (3) answer each sub-question — factored, from the pool ────────────────────
def answer_subq(sub: SubQ, fact_pool: Iterable, *, backend: Optional[_Backend] = None,
                max_tokens: int = 64) -> str:
    """Answer a single follow-up question **independently** from the pool.

    *Factored* answering (CoVe, arXiv:2309.11495): each sub-question is answered
    on its own so a wrong draft cannot bias later answers. With no ``backend``
    the answer is the deterministic pool value (the reference trajectory). With a
    ``backend`` the model is asked the lone sub-question plus only its own fact,
    keeping the context minimal and factored.

    Args:
        sub: the ``SubQ`` to answer (uses ``sub.key`` / ``sub.gold``).
        fact_pool: the full fact pool (for the backend's local context).
        backend: optional ``LLMBackend``; if ``None``, returns the pool value.
        max_tokens: generation cap when a backend is used.

    Returns:
        The intermediate answer string (also written to ``sub.answer``).

    Side effects: calls ``backend.generate`` when a backend is given (network for
        live backends; none for Mock / no backend). Never raises on backend error
        (falls back to the deterministic pool value).
    """
    pool_value = f"{sub.key} = {sub.gold}".strip(" =") if sub.gold else sub.key
    if backend is None:
        sub.answer = pool_value
        return sub.answer
    # factored: give the backend only this sub-question + its own fact
    own_fact = next((f"{k}={v}" for k, v in _normalize_pool(fact_pool)
                     if k == sub.key), pool_value)
    prompt = (f"{SELF_ASK_INSTRUCTION} no\n사실: {own_fact}\n"
              f"질문: {sub.question}\n간결히(한 줄) 답하세요.")
    try:
        resp = backend.generate(prompt, max_tokens=max_tokens)
        text = "" if getattr(resp, "error", "") else (getattr(resp, "text", "") or "")
        sub.answer = text.strip() or pool_value
    except Exception:                                   # backend never fatal here
        sub.answer = pool_value
    return sub.answer


# ── (4) recompose — synthesize the final answer ───────────────────────────────
def recompose(question: str, subs: list[SubQ], *, backend: Optional[_Backend] = None,
              max_tokens: int = 320, context_id: str = "") -> str:
    """Recompose a final answer from the answered sub-questions (Self-Ask join).

    Deterministically joins the intermediate answers into a grounded paragraph
    (the reference recomposition). With a ``backend`` the model is shown the
    original question, the answered sub-questions, and the ``FINAL_TRIGGER``
    ("So the final answer is:") and asked to synthesize.

    Args:
        question: the original compositional question.
        subs: answered ``SubQ`` list (``answer`` populated).
        backend: optional ``LLMBackend``; ``None`` → deterministic join.
        max_tokens: generation cap when a backend is used.
        context_id: optional tag echoed into the deterministic synthesis.

    Returns:
        The recomposed final answer string. Never raises (backend errors fall
        back to the deterministic join).
    """
    parts = ", ".join(s.answer for s in subs if s.answer)
    deterministic = (
        f"종합({context_id}): {parts}. 모든 해석은 위 제공 사실에만 근거합니다."
        if context_id else
        f"종합: {parts}. 모든 해석은 위 제공 사실에만 근거합니다.")
    if backend is None:
        return deterministic
    block = "\n".join(f"- 하위질문: {s.question}\n  중간답: {s.answer}" for s in subs)
    prompt = (f"원 질문: {question}\n\n다음 하위질문/중간답을 종합하세요(제공된 "
              f"수치만 사용):\n{block}\n\n{FINAL_TRIGGER}")
    try:
        resp = backend.generate(prompt, max_tokens=max_tokens)
        text = "" if getattr(resp, "error", "") else (getattr(resp, "text", "") or "")
        return text.strip() or deterministic
    except Exception:
        return deterministic


# ── (5) verify — CoVe-style grounding check over the final answer ─────────────
def verify(final_answer: str, fact_pool: Iterable) -> dict:
    """Verify the recomposed answer against the pool (Chain-of-Verification).

    Checks how many of the pool's gold numbers the final answer recalls and how
    many *extra* multi-digit numbers it invented (likely hallucinations). This is
    the CoVe verification pass distilled to a deterministic, network-free check —
    it mirrors ``aria_grounding.numeric_grounding`` semantics so scores stay
    comparable across the two modules.

    Args:
        final_answer: the recomposed answer to verify.
        fact_pool: the gold facts (``"key=value"`` strings / snippets / mapping).

    Returns:
        ``{fact_recall, n_gold_cited, n_gold, n_spurious, grounded}`` —
        ``grounded`` is ``True`` iff every gold number is cited and nothing
        multi-digit is invented. Never raises.
    """
    gold = _gold_numbers(fact_pool)
    ans = set(_NUM.findall(final_answer or ""))
    cited = gold & ans
    spurious = [v for v in ans if v not in gold and len(v.lstrip("-")) > 1]
    recall = round(len(cited) / len(gold), 3) if gold else 0.0
    return {
        "fact_recall": recall,
        "n_gold_cited": len(cited),
        "n_gold": len(gold),
        "n_spurious": len(spurious),
        "grounded": bool(gold) and len(cited) == len(gold) and not spurious,
    }


# ── orchestration — the full Self-Ask trajectory ─────────────────────────────
def self_ask(
    question: str,
    fact_pool: Iterable,
    *,
    backend: Optional[_Backend] = None,
    templates: Optional[dict] = None,
    phrase: Callable[[str], str] | None = None,
    context_id: str = "",
    max_tokens: int = 320,
) -> SelfAskResult:
    """Run the full Self-Ask trajectory: gate → decompose → answer → recompose
    → verify (Press et al. 2022 + CoVe verification).

    Backend-agnostic: with ``backend=None`` it produces the deterministic
    reference trajectory (no LLM, no network); with any ``LLMBackend`` it drives
    that model through the same steps. The follow-up gate (``needs_followup``)
    answers single-fact questions directly without decomposition.

    Args:
        question: the compositional question to answer.
        fact_pool: the facts to ground in (``"key=value"`` / snippets / mapping).
        backend: optional ``LLMBackend`` (claude / Ollama / OpenAI-compat / Mock).
        templates: optional ``{key: question}`` domain phrasings for decomposition.
        phrase: optional ``key -> question`` callable for un-templated keys.
        context_id: optional tag echoed into the recomposed answer.
        max_tokens: generation cap per backend call.

    Returns:
        A populated ``SelfAskResult`` (sub-questions, final answer, verification).

    Performance: O(n_facts) deterministic; O(n_facts) backend calls + 1 with a
        backend. Side effects: only the caller's ``backend`` may touch the
        network. This module performs no disk/DB/network I/O. Never raises.
    """
    if not needs_followup(question, fact_pool):
        # single-fact / single-clause → answer directly (Self-Ask "no")
        pairs = _normalize_pool(fact_pool)
        direct = SubQ(question=question,
                      key=pairs[0][0] if pairs else "",
                      gold=pairs[0][1] if pairs else "")
        answer_subq(direct, fact_pool, backend=backend, max_tokens=max_tokens)
        final = direct.answer or recompose(question, [direct], backend=None,
                                            context_id=context_id)
        res = SelfAskResult(question=question, sub_questions=[direct],
                            final_answer=final, decomposed=False)
        res.verification = verify(res.final_answer, fact_pool)
        direct.supported = res.verification.get("grounded")
        return res

    subs = decompose(question, fact_pool, templates=templates, phrase=phrase)
    for s in subs:
        answer_subq(s, fact_pool, backend=backend, max_tokens=max_tokens)
    final = recompose(question, subs, backend=backend, context_id=context_id,
                      max_tokens=max_tokens)
    res = SelfAskResult(question=question, sub_questions=subs, final_answer=final,
                        decomposed=True)
    res.verification = verify(final, fact_pool)
    # per-sub support flag: gold number present in the final answer
    final_nums = set(_NUM.findall(final or ""))
    for s in subs:
        m = _NUM.search(s.gold)
        s.supported = (m.group() in final_nums) if m else None
    return res


# ── CLI demo ──────────────────────────────────────────────────────────────────
def _demo_pool() -> tuple[str, list[str]]:
    question = ("ABM 전향 검증에서 행동 ON이 OFF보다 나은가, 그리고 보정된 행동 "
                "파라미터와 전향 R²는 얼마인가?")
    pool = ["forward_r2=0.722", "r2_behavior_on=0.557", "r2_behavior_off=0.041",
            "alpha=0.83", "tau=4"]
    return question, pool


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backend-agnostic Self-Ask SubQ demo")
    ap.add_argument("--demo", action="store_true", help="run the built-in demo")
    ap.add_argument("--mock", action="store_true",
                    help="drive a deterministic Mock backend (else: no backend)")
    args = ap.parse_args(argv)
    if not args.demo:
        ap.print_help()
        return 0
    backend = None
    if args.mock:
        from .backends import MockLLMBackend
        backend = MockLLMBackend("balanced")
    question, pool = _demo_pool()
    res = self_ask(question, pool, backend=backend, context_id="ABM_forward")
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    print(f"\ndecomposed={res.decomposed} n_subq={res.n_subq} "
          f"grounded={res.verification.get('grounded')} "
          f"fact_recall={res.verification.get('fact_recall')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
