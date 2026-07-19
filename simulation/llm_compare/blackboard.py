"""simulation.llm_compare.blackboard
====================================
The **evidence blackboard**: the single source of truth for the multi-agent
ARIA layer.

Why this module exists
----------------------
The shipped ARIA crew (:mod:`simulation.llm_compare.aria_multiagent`) passed a
static ``fact_block`` between stages, so the Retriever's *selection* was never
actually consumed by the Analyst — the "communication" was cosmetic. This module
replaces that with a real, append-only, provenance-tagged store that every agent
writes to and cites from.

The load-bearing invariant (mechanically enforced, not promised): **a numeric
claim can only enter the blackboard if its number appears in a tool-return
payload.** An agent therefore *cannot* inject a self-generated epidemic number —
which is exactly what the ARIA read-only / no-transmission-driver constraint
requires. :func:`verify_grounding` (in ``aria_multiagent``) then checks the final
answer against :meth:`EvidenceBlackboard.facts_for_verifier`, so the gate reads
from the same receipted pool.

Concurrency note: writes are atomic under CPython's cooperative asyncio
scheduling (``list.append`` / ``Queue.put_nowait`` contain no ``await``), so the
single-event-loop orchestrator needs no explicit lock. The delta bus
(:meth:`subscribe`) lets an async orchestrator stream appends live.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["Entry", "EvidenceBlackboard", "require_provenance"]

_NUM = re.compile(r"-?\d+\.?\d*")


def _claim_numbers(text: str) -> list[str]:
    """Numbers in ``text`` that constitute a real metric claim.

    Single-digit tokens (e.g. "3 strategies", a stray year digit) are prose noise
    and ignored — mirrors the convention in
    :func:`simulation.llm_compare.aria_multiagent.verify_grounding` so the gate
    and the store agree on what counts as a grounded number.
    """
    return [v for v in _NUM.findall(text or "")
            if len(v.lstrip("-").replace(".", "")) > 1]


def require_provenance(claim: str, value: Any, provenance: Optional[dict]) -> bool:
    """Whether an entry may enter the blackboard (the core grounding gate).

    A non-numeric claim (prose, no metric) is always allowed. A numeric claim is
    allowed **only if** ``provenance`` names a tool and every claimed number
    appears verbatim in the tool's ``return_payload``.

    Args:
        claim: the fact key or free-text claim.
        value: the fact value (number/str/None); folded into the numeric check.
        provenance: ``{"tool": str, "args": ..., "return_payload": Any}`` — the
            tool call this fact came from. ``None`` for prose claims.

    Returns:
        True iff the entry is admissible. Deterministic; never raises.

    Side effects: none.
    """
    text = f"{claim}={value}" if value is not None else str(claim)
    nums = _claim_numbers(text)
    if not nums:
        return True
    if not provenance or not provenance.get("tool"):
        return False
    payload_nums = set(_NUM.findall(str(provenance.get("return_payload", ""))))
    return all(n in payload_nums for n in nums)


@dataclass(frozen=True)
class Entry:
    """One immutable, provenance-tagged blackboard record.

    Attributes:
        agent: the agent that wrote the fact (e.g. ``"aria_retriever"``).
        claim: the fact key or free-text claim.
        value: the fact value (number/str) or ``None`` for a prose claim.
        provenance: ``{"tool", "args", "return_payload"}`` for numeric facts;
            ``None`` for prose.
        ts: wall-clock write time (epoch seconds).
    """

    agent: str
    claim: str
    value: Optional[Any] = None
    provenance: Optional[dict] = None
    ts: float = field(default=0.0)


class EvidenceBlackboard:
    """Append-only, provenance-gated shared store for the ARIA agent crew.

    Small interface (append / snapshot / facts_for_verifier / subscribe), rich
    invariant: every numeric entry is receipted to a tool return
    (:func:`require_provenance`). Agents read the running state via
    :meth:`snapshot`; the verifier reads the gold number pool via
    :meth:`facts_for_verifier`; an async orchestrator streams appends via
    :meth:`subscribe`.

    Side effects: in-memory only. No DB, no disk, no network.
    """

    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._subscribers: list = []  # list[asyncio.Queue]

    # -- write --------------------------------------------------------------
    def append(self, agent: str, claim: str, *, value: Any = None,
               provenance: Optional[dict] = None) -> Entry:
        """Append a fact, enforcing tool provenance for any numeric claim.

        Args:
            agent: writing agent id.
            claim: fact key or prose claim.
            value: fact value, or None for a prose claim.
            provenance: ``{"tool", "args", "return_payload"}``; required when the
                claim/value contains a metric number.

        Returns:
            The recorded (frozen) :class:`Entry`.

        Raises:
            ValueError: if a numeric claim lacks tool provenance or cites a number
                absent from the tool's ``return_payload`` (fail-loud: an
                ungrounded number must never enter the store).

        Side effects: appends to the in-memory log and notifies subscribers.
        """
        if not require_provenance(claim, value, provenance):
            raise ValueError(
                f"ungrounded numeric claim rejected: agent={agent!r} "
                f"claim={claim!r} value={value!r} — every number must trace to a "
                f"tool return_payload (provenance={provenance!r})")
        entry = Entry(agent=agent, claim=claim, value=value,
                      provenance=provenance, ts=time.time())
        self._entries.append(entry)
        for q in self._subscribers:
            try:
                q.put_nowait(entry)
            except Exception:
                pass  # a full/closed subscriber queue must not break a write
        return entry

    # -- read ---------------------------------------------------------------
    def snapshot(self) -> list[Entry]:
        """A shallow copy of all entries in write order (caller-mutation safe)."""
        return list(self._entries)

    def facts_for_verifier(self) -> list[str]:
        """The gold ``"claim=value"`` pool for :func:`verify_grounding`.

        Returns only entries that carry a metric number (all of which are, by
        construction, tool-receipted). Prose entries contribute nothing.
        """
        out = []
        for e in self._entries:
            text = f"{e.claim}={e.value}" if e.value is not None else str(e.claim)
            if _claim_numbers(text) and e.value is not None:
                out.append(f"{e.claim}={e.value}")
        return out

    # -- streaming ----------------------------------------------------------
    def subscribe(self):
        """Register and return an ``asyncio.Queue`` fed one :class:`Entry` per
        append — the delta bus an async orchestrator streams from."""
        import asyncio
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def __len__(self) -> int:
        return len(self._entries)
