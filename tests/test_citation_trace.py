"""
tests.test_citation_trace
==========================
Smoke tests for ARIA 문장-근거 매핑 (citation tracing).

불변식 검증 (8 case):
  1. 명확 매칭 claim → 정답 evidence 에 매핑 + supported=True
  2. 무근거 claim → supported=False (어느 문서도 충분 지지 X)
  3. snippet 추출 (매핑 문서 내 최다 겹침 문장)
  4. 빈 evidence 안전 (모든 claim evidence_id=None, supported=False)
  5. 빈/공백 답변 안전 (빈 trace + 빈 summary)
  6. 결정성 (동일 입력 → byte-identical 출력, 2회)
  7. attribution_summary 집계 정확성 (지지/미지지 비율·평균 support)
  8. shape/계약 (반환 키 + threshold ValueError + id fallback)

순수 텍스트 매칭 — 외부 LLM/네트워크/모델 없음.
"""
from __future__ import annotations

import pytest

from simulation.llm_compare.citation_trace import (
    attribution_summary,
    trace_claims,
)

# 서울 25구 ILI 도메인 톤의 evidence (가짜 placeholder 아님 — 실제 산출 형태).
EVIDENCE = [
    {"id": "P4", "text": "P4 식별성 분석에서 mobility는 theta 파라미터를 식별한다. "
                          "형태 적합 R2는 0.84 이다."},
    {"id": "ABM", "text": "에이전트 기반 모형 ABM 적합 메트릭은 r2 0.91, rmse 1.2 이다."},
    {"id": "champ", "content": "챔피언 모델은 TabPFN 이며 WIS 기준 최우수이다."},
]


def test_clear_match_routes_to_correct_evidence():
    """1. mobility/theta 문장 → P4 문서, ABM r2 문장 → ABM 문서."""
    answer = "mobility는 theta 파라미터를 식별한다. ABM 적합 r2는 0.91 이다."
    traces = trace_claims(answer, EVIDENCE, threshold=0.3)
    assert len(traces) == 2
    assert traces[0]["evidence_id"] == "P4"
    assert traces[0]["supported"] is True
    assert traces[1]["evidence_id"] == "ABM"
    assert traces[1]["supported"] is True


def test_unsupported_claim_flagged_false():
    """2. 근거에 전혀 없는 문장 → supported=False (무근거)."""
    answer = "오늘 점심으로 김치찌개를 먹었고 날씨가 매우 흐렸다."
    traces = trace_claims(answer, EVIDENCE, threshold=0.3)
    assert len(traces) == 1
    assert traces[0]["supported"] is False
    assert traces[0]["support_score"] < 0.3


def test_snippet_extracted_from_mapped_doc():
    """3. snippet 은 매핑 문서 내 최다 겹침 문장에서 추출."""
    answer = "형태 적합 R2는 0.84 이다."
    traces = trace_claims(answer, EVIDENCE, threshold=0.2)
    assert traces[0]["evidence_id"] == "P4"
    assert "0.84" in traces[0]["snippet"]
    assert traces[0]["snippet"]  # 비어있지 않음


def test_empty_evidence_safe():
    """4. evidence 비어있어도 안전 — 모든 claim evidence_id=None."""
    answer = "mobility는 theta를 식별한다. r2는 0.91 이다."
    traces = trace_claims(answer, [], threshold=0.3)
    assert len(traces) == 2
    for t in traces:
        assert t["evidence_id"] is None
        assert t["supported"] is False
        assert t["support_score"] == 0.0
        assert t["snippet"] == ""


def test_empty_answer_safe():
    """5. 빈/공백 답변 → 빈 trace + 빈 summary (예외 없음)."""
    for ans in ("", "   ", "\n\n", "..."):
        traces = trace_claims(ans, EVIDENCE)
        assert traces == []
        summ = attribution_summary(traces)
        assert summ["n_claims"] == 0
        assert summ["supported_ratio"] == 0.0
        assert summ["mean_support"] == 0.0


def test_determinism():
    """6. 동일 입력 → byte-identical 출력 (학습/난수 없음, leak-free)."""
    answer = "mobility는 theta를 식별한다. ABM r2는 0.91 이다. 김치찌개를 먹었다."
    a = trace_claims(answer, EVIDENCE, threshold=0.3)
    b = trace_claims(answer, EVIDENCE, threshold=0.3)
    assert a == b
    assert attribution_summary(a) == attribution_summary(b)


def test_attribution_summary_counts():
    """7. summary 집계 — 지지 2 + 미지지 1, 비율·평균 정확."""
    answer = ("mobility는 theta 파라미터를 식별한다. "
              "ABM 적합 r2는 0.91 이다. "
              "오늘 점심으로 김치찌개를 먹었다.")
    traces = trace_claims(answer, EVIDENCE, threshold=0.3)
    summ = attribution_summary(traces)
    assert summ["n_claims"] == 3
    assert summ["n_supported"] == 2
    assert summ["n_unsupported"] == 1
    assert summ["supported_ratio"] == pytest.approx(round(2 / 3, 4))
    # mean_support 는 산술평균과 일치
    expected_mean = round(sum(t["support_score"] for t in traces) / 3, 4)
    assert summ["mean_support"] == pytest.approx(expected_mean)


def test_shape_contract_and_guards():
    """8. 반환 키 계약 + threshold ValueError + id fallback(ev{idx})."""
    answer = "ABM r2는 0.91 이다."
    traces = trace_claims(answer, EVIDENCE)
    assert set(traces[0].keys()) == {
        "claim", "evidence_id", "support_score", "snippet", "supported"}
    assert isinstance(traces[0]["support_score"], float)
    assert isinstance(traces[0]["supported"], bool)

    # threshold 범위 밖 → ValueError
    with pytest.raises(ValueError):
        trace_claims(answer, EVIDENCE, threshold=1.5)
    with pytest.raises(ValueError):
        trace_claims(answer, EVIDENCE, threshold=-0.1)

    # id 없는 evidence → 위치 기반 ev{idx}
    no_id = [{"text": "ABM r2는 0.91 이다."}]
    t = trace_claims("ABM r2는 0.91 이다.", no_id, threshold=0.2)
    assert t[0]["evidence_id"] == "ev0"
