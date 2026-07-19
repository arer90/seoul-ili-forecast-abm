"""Tests for the Hermes tamper-evident audit chain (verify side).

The chain's value as paper-grade audit evidence depends on detecting tampering;
these tests exercise the failure modes a reviewer would worry about.
"""
from __future__ import annotations

from simulation.llm_compare.runner import _append_audit, verify_audit_chain


def _build_chain(n: int = 5) -> list[dict]:
    chain: list[dict] = []
    for i in range(n):
        _append_audit(chain, {"event": f"step{i}", "value": i})
    return chain


def test_intact_chain_verifies() -> None:
    v = verify_audit_chain(_build_chain())
    assert v["intact"] is True
    assert v["n_entries"] == 5
    assert v["first_bad_index"] is None


def test_tampered_field_detected() -> None:
    chain = _build_chain()
    chain[2]["value"] = 999
    v = verify_audit_chain(chain)
    assert v["intact"] is False
    assert v["first_bad_index"] == 2


def test_deleted_entry_detected() -> None:
    chain = _build_chain()
    del chain[2]
    v = verify_audit_chain(chain)
    assert v["intact"] is False
    assert v["first_bad_index"] == 2


def test_reordered_entries_detected() -> None:
    chain = _build_chain()
    chain[1], chain[3] = chain[3], chain[1]
    assert verify_audit_chain(chain)["intact"] is False


def test_empty_chain_is_intact() -> None:
    assert verify_audit_chain([])["intact"] is True
