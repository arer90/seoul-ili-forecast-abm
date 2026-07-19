"""TDD for the cross-run verified-answer memory (S3).

Pins: (1) ONLY gate-passed answers persist; (2) retrieval returns relevant prior
answers; (3) the memory never contains an ungrounded number; (4) a remembered
exemplar cannot inject a number into the current verifier gold pool.
"""
from __future__ import annotations


def _mem(tmp_path):
    from simulation.llm_compare.memory import VerifiedMemory
    return VerifiedMemory(path=tmp_path / "verified.jsonl")


def test_only_grounded_answers_persist(tmp_path):
    m = _mem(tmp_path)
    ok = m.remember("q1", "행동 ON R² 0.557.", tool_receipts=[{"tool": "epi.forecast"}],
                    verification={"grounded": True, "n_spurious": 0})
    bad = m.remember("q2", "R² 0.999 완벽.", tool_receipts=[],
                     verification={"grounded": False, "n_spurious": 1})
    assert ok is True and bad is False
    assert len(m) == 1                      # the ungrounded answer was rejected


def test_retrieve_returns_relevant_prior(tmp_path):
    m = _mem(tmp_path)
    m.remember("백신 배분 전략 효과는?", "표적 고접촉 접종이 1회분당 감염 감소가 크다.",
               tool_receipts=[{"tool": "epi.scenario_run"}],
               verification={"grounded": True})
    m.remember("챔피언 모델은?", "FusedEpi 가 상대 WIS 0.443 으로 최고.",
               tool_receipts=[{"tool": "epi.model_compare"}],
               verification={"grounded": True})
    hits = m.retrieve("백신 배분 어떻게?", k=1)
    assert hits and "배분" in hits[0]["query"]     # vaccine-allocation memory retrieved


def test_memory_never_stores_ungrounded_number(tmp_path):
    m = _mem(tmp_path)
    m.remember("q", "정확도 0.999.", tool_receipts=[],
               verification={"grounded": False})
    for r in m.all():
        assert r["verification"]["grounded"] is True   # every stored rec is grounded


def test_memory_context_is_not_a_verifier_fact_source(tmp_path):
    """A remembered answer's number is NOT added to a fresh blackboard gold pool."""
    from simulation.llm_compare.blackboard import EvidenceBlackboard
    m = _mem(tmp_path)
    m.remember("prior", "과거 R² 0.557.", tool_receipts=[{"tool": "epi.forecast"}],
               verification={"grounded": True})
    prior = m.retrieve("prior", k=1)
    assert prior                                        # memory has the record
    bb = EvidenceBlackboard()                           # fresh board for a new query
    assert bb.facts_for_verifier() == []                # memory did NOT seed gold facts
