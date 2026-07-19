"""Smoke tests for the backend-agnostic Self-Ask SubQ module
(``simulation.llm_compare.subq``).

Self-Ask (Press et al. 2022, arXiv:2210.03350) + CoVe verification
(arXiv:2309.11495). The module is the *general* primitive extracted from
``aria_grounding`` — these tests cover matched / mismatch / empty / NaN-ish /
single-fact / multi-fact / backend / no-backend edges (6-8 case TDD smoke).
"""


# ── decompose: one atomic follow-up per fact, never drops a fact ──────────────
def test_decompose_one_subq_per_fact():
    from simulation.llm_compare.subq import decompose
    pool = ["forward_r2=0.722", "alpha=0.83", "tau=4"]
    subs = decompose("복합 질문?", pool)
    assert len(subs) == len(pool)                       # never drops a fact
    assert all(s.question and s.key for s in subs)
    assert subs[0].key == "forward_r2" and subs[0].gold == "0.722"


def test_decompose_empty_pool():
    from simulation.llm_compare.subq import decompose
    assert decompose("질문?", []) == []                 # empty edge


def test_decompose_accepts_dict_and_tuples():
    from simulation.llm_compare.subq import decompose
    # mapping form
    subs_map = decompose("q", {"r2": "0.9", "rmse": "1.2"})
    assert {s.key for s in subs_map} == {"r2", "rmse"}
    # (key, value) tuple form
    subs_tup = decompose("q", [("r2", "0.9")])
    assert subs_tup[0].key == "r2" and subs_tup[0].gold == "0.9"


def test_decompose_templates_override_phrasing():
    from simulation.llm_compare.subq import decompose
    subs = decompose("q", ["forward_r2=0.722"],
                     templates={"forward_r2": "전향 R²는?"})
    assert subs[0].question == "전향 R²는?"             # domain phrasing injected


# ── follow-up gate (Self-Ask "Are follow up questions needed here:") ──────────
def test_needs_followup_multi_fact_true_single_clause_false():
    from simulation.llm_compare.subq import needs_followup
    assert needs_followup("R²는?", ["r2=0.9", "rmse=1.2"]) is True   # >1 fact
    assert needs_followup("R²는?", ["r2=0.9"]) is False              # 1 fact, 1 clause
    # single fact but multi-clause question still decomposes
    assert needs_followup("R²는 얼마이고, 그리고 그 의미는?", ["r2=0.9"]) is True


# ── verify (CoVe): grounded vs hallucinated, matched/mismatch/empty/NaN ───────
def test_verify_discriminates_grounded_vs_hallucinated():
    from simulation.llm_compare.subq import verify
    pool = ["forward_r2=0.722", "alpha=0.83"]
    grounded = "전향 R²=0.722, alpha=0.83 입니다."
    hallucinated = "전향 R²=0.99로 완벽하고 alpha=0.50 입니다."
    g = verify(grounded, pool)
    h = verify(hallucinated, pool)
    assert g["fact_recall"] > h["fact_recall"]
    assert g["grounded"] is True and h["grounded"] is False
    assert h["n_spurious"] >= 1


def test_verify_matched_mismatch_empty():
    from simulation.llm_compare.subq import verify
    assert verify("", [])["fact_recall"] == 0.0                       # empty
    assert verify("숫자 없음", ["r2=0.5"])["fact_recall"] == 0.0      # mismatch
    assert verify("r2=0.5", ["r2=0.5"])["fact_recall"] == 1.0         # matched


def test_verify_handles_non_numeric_pool():
    from simulation.llm_compare.subq import verify
    # free-text snippet with no numbers -> no gold numbers, recall defined as 0
    rep = verify("문장 답변", ["감염병예방법 제11조 신고 의무"])
    assert rep["n_gold"] == 0 and rep["fact_recall"] == 0.0           # NaN-safe


# ── self_ask end-to-end: no-backend deterministic reference trajectory ────────
def test_self_ask_no_backend_recomposes_grounded():
    from simulation.llm_compare.subq import self_ask
    pool = ["forward_r2=0.722", "r2_behavior_on=0.557", "alpha=0.83"]
    res = self_ask("행동 ON이 더 나은가 그리고 R²는?", pool, context_id="ABM")
    assert res.decomposed is True and res.n_subq == len(pool)
    # the deterministic recomposition is strongly grounded in every gold number
    assert res.verification["fact_recall"] == 1.0
    assert res.verification["grounded"] is True
    assert all(s.answer for s in res.sub_questions)      # every sub-Q answered


def test_self_ask_single_fact_answers_directly_no_decompose():
    from simulation.llm_compare.subq import self_ask
    res = self_ask("전향 R²는?", ["forward_r2=0.722"])
    assert res.decomposed is False and res.n_subq == 1   # gate: direct answer
    assert "0.722" in res.final_answer


def test_self_ask_with_mock_backend_runs_and_grounds():
    from simulation.llm_compare.subq import self_ask
    from simulation.llm_compare.backends import MockLLMBackend
    pool = ["forward_r2=0.722", "alpha=0.83"]
    res = self_ask("R²와 alpha는?", pool, backend=MockLLMBackend("balanced"),
                   context_id="ABM")
    # backend-driven trajectory still returns a valid, structured result
    assert res.n_subq == 2 and isinstance(res.final_answer, str)
    assert "fact_recall" in res.verification


def test_self_ask_backend_error_falls_back_deterministic():
    from simulation.llm_compare.subq import self_ask

    class _BrokenBackend:
        def generate(self, prompt, *, max_tokens=64):
            raise RuntimeError("network down")

    pool = ["forward_r2=0.722", "alpha=0.83"]
    res = self_ask("R²와 alpha는?", pool, backend=_BrokenBackend())
    # broken backend must never crash the trajectory; falls back to pool values
    assert res.n_subq == 2
    assert res.verification["fact_recall"] == 1.0        # deterministic fallback


# ── reusability proof: drives aria_grounding's epi context without that module
#    knowing this module exists (general primitive, not a duplicate) ───────────
def test_self_ask_reuses_aria_grounding_context_generically():
    from simulation.llm_compare.subq import self_ask
    from simulation.llm_compare.aria_grounding import (
        load_real_context, _SUBQ_TEMPLATES,
    )
    ctx = load_real_context("identifiability")
    # feed aria_grounding's real facts + its epi templates into the GENERAL module
    res = self_ask(ctx["context"], ctx["facts"], templates=_SUBQ_TEMPLATES,
                   context_id=ctx["id"])
    assert res.n_subq == len(ctx["facts"])               # one sub-Q per real fact
    # epi template phrasing was applied (not the generic fallback)
    qs = [s.question for s in res.sub_questions]
    assert any("R²" in q for q in qs)
    assert res.verification["fact_recall"] >= 0.8        # grounded on real numbers
