"""TDD (Red→Green) for the ARIA evidence blackboard.

The blackboard is the single source of truth for the multi-agent ARIA layer:
every agent writes provenance-tagged facts and every number must trace to a
tool-return payload. These tests pin the load-bearing contract — an ungrounded
number can NEVER enter the blackboard — before the orchestrator relies on it.
"""
from __future__ import annotations

import asyncio

import pytest

from simulation.llm_compare.blackboard import (
    EvidenceBlackboard,
    Entry,
    require_provenance,
)


# ── provenance gate ───────────────────────────────────────────────────────────
def test_numeric_entry_requires_tool_provenance():
    bb = EvidenceBlackboard()
    with pytest.raises(ValueError):
        bb.append("analyst", "forward_r2", value="0.557")  # no provenance → reject


def test_numeric_entry_number_must_appear_in_payload():
    bb = EvidenceBlackboard()
    with pytest.raises(ValueError):
        # 0.557 is NOT in the tool return payload → fabricated → reject
        bb.append("analyst", "forward_r2", value="0.999",
                  provenance={"tool": "epi.forecast",
                              "return_payload": {"forward_r2": 0.557}})


def test_grounded_numeric_entry_is_accepted():
    bb = EvidenceBlackboard()
    e = bb.append("retriever", "forward_r2", value="0.557",
                  provenance={"tool": "read_artifact",
                              "return_payload": {"forward_r2": 0.557}})
    assert isinstance(e, Entry)
    assert e.claim == "forward_r2"
    assert e.provenance["tool"] == "read_artifact"


def test_nonnumeric_entry_needs_no_provenance():
    bb = EvidenceBlackboard()
    e = bb.append("analyst", "behavior-on configuration is favoured for policy")
    assert e.value is None
    assert bb.snapshot()[-1].claim.startswith("behavior-on")


def test_single_digit_tokens_are_not_treated_as_claims():
    # "3" alone is prose noise (e.g. "3 strategies"), not a grounded metric claim
    bb = EvidenceBlackboard()
    e = bb.append("analyst", "compared 3 strategies")
    assert e is not None


# ── append-only immutability ──────────────────────────────────────────────────
def test_entries_are_immutable():
    bb = EvidenceBlackboard()
    e = bb.append("analyst", "note only")
    with pytest.raises(Exception):
        e.claim = "tampered"  # frozen dataclass → cannot mutate a recorded fact


def test_snapshot_is_a_copy():
    bb = EvidenceBlackboard()
    bb.append("analyst", "note")
    snap = bb.snapshot()
    snap.clear()
    assert len(bb.snapshot()) == 1  # external clear must not affect the store


# ── verifier view ─────────────────────────────────────────────────────────────
def test_facts_for_verifier_returns_only_receipted_numbers():
    bb = EvidenceBlackboard()
    bb.append("retriever", "forward_r2", value="0.557",
              provenance={"tool": "read_artifact",
                          "return_payload": {"forward_r2": 0.557}})
    bb.append("analyst", "behavior-on is favoured")  # non-numeric, no receipt
    facts = bb.facts_for_verifier()
    assert any("0.557" in f for f in facts)
    assert all("=" in f for f in facts)  # key=value gold format for verify_grounding
    assert len(facts) == 1  # the prose entry contributes no gold number


# ── module-level helper ───────────────────────────────────────────────────────
def test_require_provenance_helper():
    ok = require_provenance("forward_r2", "0.557",
                            {"tool": "t", "return_payload": {"x": 0.557}})
    bad_missing = require_provenance("forward_r2", "0.557", None)
    bad_payload = require_provenance("forward_r2", "0.557",
                                     {"tool": "t", "return_payload": {"x": 1.0}})
    text_ok = require_provenance("just prose", None, None)
    assert ok and text_ok
    assert not bad_missing and not bad_payload


# ── streaming delta bus (async) ───────────────────────────────────────────────
def test_subscribe_receives_append_delta():
    async def _run():
        bb = EvidenceBlackboard()
        q = bb.subscribe()
        bb.append("retriever", "forward_r2", value="0.557",
                  provenance={"tool": "read_artifact",
                              "return_payload": {"forward_r2": 0.557}})
        e = await asyncio.wait_for(q.get(), timeout=1.0)
        assert e.claim == "forward_r2"

    asyncio.run(_run())
