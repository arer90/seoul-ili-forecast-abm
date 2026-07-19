"""TDD for the exemplar refresh (S3) — curated core + optional DSPy.

Pins: refresh selects exemplars from verified history, dedups by query, no-ops on
empty history, and — critically — NEVER mutates the verified.jsonl tool receipts
(it only reads memory and writes a separate exemplar file).
"""
from __future__ import annotations


def _mem(tmp_path):
    from simulation.llm_compare.memory import VerifiedMemory
    return VerifiedMemory(path=tmp_path / "verified.jsonl")


def test_refresh_curated_from_history(tmp_path):
    from simulation.llm_compare.dspy_refresh import refresh_exemplars
    m = _mem(tmp_path)
    m.remember("백신 배분 전략?", "표적 접종이 우세.", tool_receipts=[{"tool": "epi.scenario_run"}],
               verification={"grounded": True})
    m.remember("챔피언 모델?", "FusedEpi 최고.", tool_receipts=[{"tool": "epi.model_compare"}],
               verification={"grounded": True})
    summ = refresh_exemplars(memory=m, out_path=tmp_path / "exemplars.json")
    assert summ["n_verified"] == 2 and summ["n_exemplars"] == 2
    assert summ["method"] == "curated"
    assert (tmp_path / "exemplars.json").exists()


def test_refresh_empty_history_writes_nothing(tmp_path):
    from simulation.llm_compare.dspy_refresh import refresh_exemplars
    m = _mem(tmp_path)
    summ = refresh_exemplars(memory=m, out_path=tmp_path / "exemplars.json")
    assert summ["n_exemplars"] == 0
    assert not (tmp_path / "exemplars.json").exists()


def test_refresh_dedups_by_query(tmp_path):
    from simulation.llm_compare.dspy_refresh import refresh_exemplars
    m = _mem(tmp_path)
    m.remember("동일 질의 텍스트입니다", "답 1.", tool_receipts=[], verification={"grounded": True})
    m.remember("동일 질의 텍스트입니다", "답 2.", tool_receipts=[], verification={"grounded": True})
    summ = refresh_exemplars(memory=m, out_path=tmp_path / "exemplars.json")
    assert summ["n_exemplars"] == 1          # same query prefix → one exemplar


def test_refresh_does_not_mutate_receipts(tmp_path):
    from simulation.llm_compare.dspy_refresh import refresh_exemplars
    m = _mem(tmp_path)
    m.remember("q", "상대 WIS 0.443.", tool_receipts=[{"tool": "epi.model_compare"}],
               verification={"grounded": True})
    before = m.path.read_bytes()
    refresh_exemplars(memory=m, out_path=tmp_path / "exemplars.json")
    assert m.path.read_bytes() == before     # verified.jsonl byte-identical (receipts intact)
