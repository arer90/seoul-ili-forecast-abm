"""
simulation.llm_compare.citation_trace
=======================================
ARIA 문장-근거 매핑 (Citation Tracing / claim attribution).

ARIA가 역학자에게 시뮬레이션 결과를 해석해 줄 때, 답변의 *각 문장이 어느
근거 문서(evidence)에 뿌리내리는지*를 투명하게 추적한다. 이는
``aria_grounding.numeric_grounding`` (수치 일치)·``comparison.faithfulness``
(전체 groundedness 점수) 와 보완 관계다 — 둘은 "답변이 얼마나 근거에
충실한가"를 *집계*하지만, 이 모듈은 **문장 ↔ 근거 문서의 명시적 매핑**을
산출해 *어느* 문장이 *어느* 근거에 지지되는지(또는 무근거인지) 드러낸다.

투명성(어떤 evidence가 어떤 claim을 받쳤는가)·과적합 0(순수 텍스트 매칭,
학습/튜닝 없음)이 survey top-pick 인 이유다. 외부 LLM·네트워크 호출 없이
결정적(deterministic) token-overlap (Jaccard) 으로만 동작한다.

설계 (deep module):
  - ``trace_claims``  : 답변 → 문장단위 claim → 최대지지 evidence 매핑.
  - ``attribution_summary`` : trace 리스트 → 지지/미지지 비율 + 평균 support.

토큰 규약은 코드베이스 표준(``comparison.faithfulness``)을 따른다:
한글/ASCII 콘텐츠 토큰 ``[가-힣a-z0-9]+``, 문장 분해 ``[.!?。\n]+``.
"""
from __future__ import annotations

import re

__all__ = ["trace_claims", "attribution_summary"]

# 코드베이스 표준 토큰/문장 규약 (comparison.faithfulness 와 동일):
#   콘텐츠 토큰 = 한글 음절 + 소문자 ASCII + 숫자 (구두점·공백 무시).
#   문장 경계   = 마침표/물음표/느낌표/句點/개행.
_TOKEN_RE = re.compile(r"[가-힣a-z0-9]+")
# 문장 경계 = 마침표/물음표/느낌표/句點/개행 (comparison.faithfulness 규약).
# 단, 숫자 사이 마침표(0.84 같은 소수)는 경계가 아니다 — 역학 산출은 R²·WIS
# 등 소수가 흔해, 경계로 처리하면 "0"·"84" 로 쪼개져 claim·토큰이 망가진다.
# (?<!\d)\.(?!\d) = 앞뒤가 숫자가 아닐 때만 마침표를 경계로 인정.
_SENT_SPLIT_RE = re.compile(r"(?:(?<!\d)\.(?!\d))|[!?。\n]+")
_MIN_CLAIM_CHARS = 4  # faithfulness 와 동일: 4자 이하 조각은 claim 아님


def _tokens(text: str) -> set[str]:
    """텍스트를 소문자 콘텐츠 토큰 집합으로 분해 (구두점·공백 제거).

    Args:
        text: 임의 문자열 (None 안전 — 빈 집합 반환).

    Returns:
        ``set[str]`` — ``[가-힣a-z0-9]+`` 매칭 토큰을 소문자화한 집합.
    """
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _split_claims(answer_text: str) -> list[str]:
    """답변을 문장(claim) 단위로 분해.

    Args:
        answer_text: ARIA 답변 전문.

    Returns:
        4자 초과 문장 문자열의 *원문 순서* 리스트. 빈 입력 → ``[]``.
    """
    if not answer_text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(answer_text)
            if len(s.strip()) > _MIN_CLAIM_CHARS]


def _evidence_id(doc: dict, idx: int) -> str:
    """근거 문서의 안정적 id 결정 (명시 id 우선, 없으면 위치 기반).

    Args:
        doc: ``{"id"?, "text"|"content"|"snippet"?}`` 형태의 evidence 문서.
        idx: ``evidence_docs`` 내 0-based 위치 (id 부재 시 fallback).

    Returns:
        문서 식별 문자열. ``id`` 키가 있으면 그 값(str), 없으면 ``"ev{idx}"``.
    """
    raw = doc.get("id")
    return str(raw) if raw is not None else f"ev{idx}"


def _evidence_text(doc: dict) -> str:
    """evidence 문서에서 본문 텍스트 추출 (여러 키 호환).

    Args:
        doc: evidence 문서. 본문은 ``text``/``content``/``snippet`` 중 하나.

    Returns:
        본문 문자열. 어느 키도 없으면 ``""``.
    """
    for key in ("text", "content", "snippet"):
        v = doc.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _best_snippet(claim_tokens: set[str], evidence_text: str, *,
                  max_chars: int = 160) -> str:
    """claim과 가장 많이 겹치는 evidence 문장(또는 앞부분)을 snippet 으로 추출.

    근거 문서를 문장 단위로 쪼개, claim 토큰과 overlap 이 최대인 문장을 고른다.
    어느 문장도 겹치지 않으면(또는 문서가 단문장) 본문 앞부분을 잘라 반환한다.

    Args:
        claim_tokens: claim 의 콘텐츠 토큰 집합.
        evidence_text: 매핑된 근거 문서 본문.
        max_chars: snippet 최대 길이 (초과 시 잘라 "…" 부착).

    Returns:
        근거 문장 또는 본문 앞부분 (≤ ``max_chars`` + "…"). 본문 없으면 ``""``.

    Performance: O(S·T) — S=문장 수, T=claim 토큰 수. 길어야 수십 문장.
    """
    if not evidence_text:
        return ""
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(evidence_text)
                 if s.strip()]
    if not sentences:
        sentences = [evidence_text.strip()]

    best_sent = sentences[0]
    best_hits = -1
    for sent in sentences:
        hits = len(claim_tokens & _tokens(sent))
        if hits > best_hits:
            best_hits = hits
            best_sent = sent

    if len(best_sent) > max_chars:
        return best_sent[:max_chars].rstrip() + "…"
    return best_sent


def trace_claims(answer_text: str, evidence_docs, *,
                 threshold: float = 0.3) -> list[dict]:
    """답변의 각 문장(claim)을 최대지지 근거 문서에 매핑한다 (citation tracing).

    답변을 문장 단위로 분해하고, 각 claim 을 모든 evidence 문서와 비교해
    콘텐츠 토큰 Jaccard 유사도가 가장 높은 문서에 매핑한다. 그 유사도가
    ``threshold`` 이상이면 ``supported=True`` (그 evidence 가 claim 을 받침),
    미만이면 ``supported=False`` (무근거 — 어느 문서도 충분히 지지 못함).

    순수 텍스트 매칭(외부 LLM/네트워크/학습 없음)이라 결정적이며 과적합이
    없다. 같은 입력은 항상 같은 출력을 낸다.

    Args:
        answer_text: ARIA 답변 전문. 문장은 ``.?!。`` 또는 개행으로 분리.
            None/빈 문자열이면 빈 리스트 반환.
        evidence_docs: 근거 문서 리스트 ``list[dict]``. 각 dict 는
            ``{"id"?: str, "text"|"content"|"snippet": str}``.
            ``id`` 부재 시 위치 기반 ``"ev{idx}"`` 부여. 빈 리스트면 모든
            claim 이 ``evidence_id=None, support_score=0.0, supported=False``.
        threshold: 지지 판정 Jaccard 컷오프 ∈ [0, 1]. claim 토큰 대비
            evidence 토큰의 교집합 / 합집합 비율. 기본 0.3.

    Returns:
        ``list[dict]`` — claim 당 한 항목, *답변 원문 순서*:
        ``{"claim": str,            # 문장 원문
           "evidence_id": str|None, # 최대지지 문서 id (evidence 없으면 None)
           "support_score": float,  # 최대 Jaccard 유사도 (round 4)
           "snippet": str,          # 매핑 문서 내 최다 겹침 문장(≤160자), 없으면 ""
           "supported": bool}``     # support_score ≥ threshold

    Raises:
        ValueError: ``threshold`` 가 [0, 1] 밖일 때.

    Performance: O(C · D · T) — C=claim 수, D=evidence 문서 수, T=평균 토큰 수.
        모두 set 연산이라 실측 수백 claim·수십 문서까지 ms 단위.
    Side effects: 없음 (순수 함수, I/O·전역 상태 변경 없음).
    Caller responsibility: evidence_docs 각 항목은 dict — 본문 키
        (text/content/snippet) 중 하나를 권장 (없으면 빈 문서로 안전 처리).
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")

    docs = list(evidence_docs) if evidence_docs else []
    # evidence 토큰을 한 번만 미리 계산 (claim 마다 재계산 회피).
    prepared = []
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        ev_text = _evidence_text(doc)
        prepared.append((
            _evidence_id(doc, idx),
            _tokens(ev_text),
            ev_text,
        ))

    claims = _split_claims(answer_text)
    traces: list[dict] = []
    for claim in claims:
        c_tokens = _tokens(claim)
        best_id: str | None = None
        best_score = 0.0
        best_ev_text = ""
        for ev_id, ev_tokens, ev_text in prepared:
            if not ev_tokens:
                continue
            union = c_tokens | ev_tokens
            jac = len(c_tokens & ev_tokens) / len(union) if union else 0.0
            # 동률은 첫(가장 앞선) 문서를 유지 → 결정성.
            if jac > best_score:
                best_score = jac
                best_id = ev_id
                best_ev_text = ev_text

        supported = best_id is not None and best_score >= threshold
        snippet = _best_snippet(c_tokens, best_ev_text) if best_id is not None else ""
        traces.append({
            "claim": claim,
            "evidence_id": best_id,
            "support_score": round(best_score, 4),
            "snippet": snippet,
            "supported": bool(supported),
        })
    return traces


def attribution_summary(traces) -> dict:
    """trace 리스트를 집계해 지지/미지지 비율과 평균 support 를 산출한다.

    ``trace_claims`` 의 출력을 받아 답변 전체의 근거 충실도를 요약한다.
    높은 ``supported_ratio`` / ``mean_support`` ⇒ 답변이 제공된 근거에 잘
    뿌리내림; 낮음 ⇒ 무근거(hallucination) claim 다수.

    Args:
        traces: ``trace_claims`` 반환 리스트. 각 항목은 최소
            ``support_score``(float)·``supported``(bool) 키 보유. 빈 리스트
            안전 처리.

    Returns:
        ``dict``:
        ``{"n_claims": int,            # 전체 claim 수
           "n_supported": int,         # supported=True claim 수
           "n_unsupported": int,       # supported=False claim 수
           "supported_ratio": float,   # n_supported / n_claims (claim 0 → 0.0)
           "mean_support": float}``    # support_score 평균 (claim 0 → 0.0)

    Raises:
        없음 (빈 입력·키 부재 모두 안전 기본값).

    Performance: O(C) — C=claim 수. Side effects: 없음 (순수 함수).
    """
    items = list(traces) if traces else []
    n = len(items)
    if n == 0:
        return {"n_claims": 0, "n_supported": 0, "n_unsupported": 0,
                "supported_ratio": 0.0, "mean_support": 0.0}
    n_supported = sum(1 for t in items if t.get("supported"))
    total_support = sum(float(t.get("support_score", 0.0)) for t in items)
    return {
        "n_claims": n,
        "n_supported": n_supported,
        "n_unsupported": n - n_supported,
        "supported_ratio": round(n_supported / n, 4),
        "mean_support": round(total_support / n, 4),
    }
