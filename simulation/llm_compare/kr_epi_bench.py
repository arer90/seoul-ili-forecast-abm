"""Korean epidemiology + infectious-disease-law benchmark (P5 priority 1).

Two external, recognized evaluation axes the 25-item bespoke golden set lacks
(both reviews + the 3-LLM synthesis flagged self-grading + contamination):

  • KR_EPI_LAW_QA — domain-primary open-ended QA anchored to OFFICIAL sources
    (law.go.kr 감염병예방법 articles, 질병관리청 sentinel system, KOSIS), NOT thesis
    sections, so a model cannot pass by surface-matching the manuscript. Scored
    by the existing judge (must_contain / must_avoid) + comparison.py metrics.
  • load_kormedmcqa() — the recognized external Korean medical benchmark
    (sean0042/KorMedMCQA, 7,469 licensing-exam items); KorMedMCQA showed US MedQA
    is NOT a valid Korean proxy. Loaded from HuggingFace if available, else a
    graceful empty (no hard dependency — portability principle #1).

ARIA is an epidemiological advisor, so KR_EPI_LAW_QA is the primary domain test
and KorMedMCQA is the external recognized anchor (per the Codex/Gemini split).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class KrEpiItem:
    id: str
    category: str          # 분류 / 신고 / 방역 / 예방접종 / 역학 / 데이터
    question: str
    answer_key: str        # correct-answer key facts (human reference)
    must_contain: tuple[str, ...]
    must_avoid: tuple[str, ...]
    official_source: str   # law.go.kr / KDCA / KOSIS / WHO — external authority


#: Seed set (expand to n≥80 for the paper). Facts verified against the cited
#: official sources; NOT anchored to thesis section numbers.
KR_EPI_LAW_QA: tuple[KrEpiItem, ...] = (
    KrEpiItem("KL01", "분류",
        "인플루엔자는 「감염병의 예방 및 관리에 관한 법률」상 몇 급 감염병이며 어떤 감시체계 대상인가?",
        "제4급감염병 — 표본감시(sentinel) 대상.",
        ("4급", "표본감시"), ("1급", "전수감시", "즉시신고"),
        "law.go.kr 감염병예방법 제2조 · KDCA 법정감염병"),
    KrEpiItem("KL02", "신고",
        "제4급감염병의 표본감시기관 신고 기한과 미신고 시 벌칙은?",
        "7일 이내 신고 / 제3·4급 미신고·거짓신고 시 300만원 이하 벌금.",
        ("7일", "300만원"), ("즉시", "24시간"),
        "law.go.kr 감염병예방법 제11조 · easylaw.go.kr"),
    KrEpiItem("KL03", "신고",
        "감염병 신고 의무자는 누구인가?",
        "의사·치과의사·한의사 및 의료기관의 장(군의관·감염병병원체 확인기관의 장 등 포함).",
        ("의사", "의료기관"), ("일반인 의무", "환자 본인만"),
        "law.go.kr 감염병예방법 제11조"),
    KrEpiItem("KL04", "방역",
        "감염병 유행 시 자치단체장이 집합 제한·금지, 휴교·휴원을 명할 수 있는 법적 근거 조항은?",
        "제49조(감염병의 예방 조치) — 집합 제한·금지, 마스크 착용, 휴업·휴교·휴원 등.",
        ("제49조", "집합"), ("근거 없음", "임의로"),
        "law.go.kr 감염병예방법 제49조"),
    KrEpiItem("KL05", "예방접종",
        "인플루엔자 국가예방접종(NIP) 우선 대상과 근거 조항을 설명하라.",
        "65세 이상 어르신·생후 6개월~13세 어린이 등; 제24조(필수예방접종)·제25조(임시예방접종).",
        ("65세", "어린이"), ("전 국민 의무", "강제"),
        "law.go.kr 감염병예방법 제24·25조 · KDCA NIP"),
    KrEpiItem("KL06", "역학",
        "KDCA 인플루엔자 표본감시의 ILI(인플루엔자 의사환자) 정의와 지표 단위는?",
        "갑작스러운 38℃ 이상 발열 + 기침 또는 인후통; 외래환자 1,000명당 ILI 환자수(/1,000).",
        ("38", "1000"), ("입원환자만", "확진자수"),
        "KDCA 감염병포털 표본감시"),
    KrEpiItem("KL07", "역학",
        "계절 인플루엔자의 기초감염재생산수(R0) 중앙값 추정치는 대략 얼마인가?",
        "약 1.28 (IQR 1.19–1.37) — 계절 독감(Biggerstaff 2014). 팬데믹은 더 높음(1.5~1.8).",
        ("1.2", "1.3"), ("5", "10", "20"),
        "Biggerstaff 2014 BMC Infect Dis · WHO"),
    KrEpiItem("KL08", "역학",
        "백신효과(VE)를 test-negative design으로 추정할 때 'VE 42%'의 올바른 해석은?",
        "접종군이 미접종군 대비 인플루엔자 발병 위험이 약 42% 감소 — 완전예방이 아님, 중증 예방 효과는 별도.",
        ("42", "감소"), ("100%", "완벽", "예방 보장"),
        "KDCA · WHO test-negative design"),
    KrEpiItem("KL09", "데이터",
        "KOSIS와 KDCA 감염병포털 데이터의 성격 차이를 설명하라.",
        "KOSIS는 국가통계포털(공식 집계 통계·OpenAPI); KDCA 포털은 법정감염병 신고·표본감시 원자료.",
        ("국가통계", "표본감시"), ("동일", "민간"),
        "kosis.kr · dportal.kdca.go.kr"),
    KrEpiItem("KL10", "방역",
        "감염병의심자에 대한 격리·입원 조치의 한계(인권 보호)는 무엇인가?",
        "필요 최소한·비례원칙, 적법절차; 강제처분은 법정 요건(제42조) 하에서만, 무기한·임의 격리 불가.",
        ("최소", "절차"), ("무기한", "임의"),
        "law.go.kr 감염병예방법 제42조"),
    KrEpiItem("KL11", "역학",
        "표본감시(sentinel)와 전수감시(전수신고)의 차이는?",
        "표본감시는 지정 표본기관이 유행 동향 파악용으로 신고(예: 인플루엔자 4급); 전수감시는 모든 발생을 의무 신고(1~3급).",
        ("표본", "전수"), ("동일", "차이 없음"),
        "KDCA 법정감염병 감시체계"),
    KrEpiItem("KL12", "분류",
        "2020년 개정 감염병예방법의 1~4급 분류 기준(개념)을 요약하라.",
        "제1급=생물테러·치명·음압격리·즉시신고 / 제2급=전파력 높음·격리·24시간 / 제3급=발생 모니터링·24시간 / 제4급=유행 조사용 표본감시.",
        ("1급", "4급", "표본감시"), ("군 분류", "법정군"),
        "law.go.kr 감염병예방법 제2조"),
    # ── 확장 (KL13~KL40): 분류·신고·방역·예방접종·역학·데이터 ──────────────────
    KrEpiItem("KL13", "분류",
        "코로나바이러스감염증-19(COVID-19)의 현재 법정감염병 등급은? (2023년 변경)",
        "2023년 8월 제4급감염병(표본감시)으로 하향 조정됨(이전 제2급).",
        ("4급",), ("1급", "여전히 2급"),
        "law.go.kr 감염병예방법 · KDCA 2023 등급조정"),
    KrEpiItem("KL14", "분류",
        "제1급감염병의 대표 예시 3가지를 들어라.",
        "에볼라바이러스병·두창·페스트·탄저·중증급성호흡기증후군(SARS)·중동호흡기증후군(MERS)·신종인플루엔자 등.",
        ("에볼라", "페스트"), ("결핵", "인플루엔자"),
        "law.go.kr 감염병예방법 제2조 제1급"),
    KrEpiItem("KL15", "분류",
        "결핵·수두·홍역·콜레라·장티푸스는 몇 급 감염병인가?",
        "제2급감염병(전파력 높아 격리 필요, 24시간 이내 신고).",
        ("2급",), ("4급", "표본감시"),
        "law.go.kr 감염병예방법 제2조 제2급"),
    KrEpiItem("KL16", "신고",
        "제2급·제3급감염병의 신고 기한은?",
        "둘 다 24시간 이내 신고(제1급=즉시, 제4급=7일).",
        ("24시간",), ("7일", "즉시"),
        "law.go.kr 감염병예방법 제11조"),
    KrEpiItem("KL17", "신고",
        "전국 인플루엔자 표본감시기관은 대략 어떤 규모·종류인가?",
        "전국 약 200개 의원급 표본감시기관(인구 약 10만명당 1개소 수준).",
        ("의원", "표본감시"), ("종합병원만", "전수"),
        "KDCA 인플루엔자 표본감시 운영지침"),
    KrEpiItem("KL18", "방역",
        "감염병 환자 등에 대한 강제 입원·격리 처분의 법적 근거 조항은?",
        "제42조(감염병에 관한 강제처분) — 적법절차·비례원칙 하에서만.",
        ("제42조", "강제처분"), ("근거 없음", "무제한"),
        "law.go.kr 감염병예방법 제42조"),
    KrEpiItem("KL19", "방역",
        "감염병 의심자의 업무 종사를 일시 제한할 수 있는 근거와 대표 직종은?",
        "제45조(업무 종사의 일시 제한) — 식품접객업·집단급식소 등 전파 위험 직종.",
        ("제45조", "식품"), ("모든 직종 영구", "근거 없음"),
        "law.go.kr 감염병예방법 제45조"),
    KrEpiItem("KL20", "예방접종",
        "인플루엔자 백신의 종류(가수·제법)를 설명하라.",
        "3가 또는 4가 백신(A/H1N1·A/H3N2·B 1~2계통), 대부분 불활화(사백신) 주사제.",
        ("불활화", "4가"), ("생백신 필수", "항생제"),
        "KDCA 예방접종 지침"),
    KrEpiItem("KL21", "예방접종",
        "임신부에게 인플루엔자 백신을 권고할 수 있는가?",
        "예 — 불활화 백신은 임신 전 시기 안전, 임신부·태아 보호 위해 권고(생백신 비강 분무는 금기).",
        ("불활화", "권고"), ("금기", "위험해서 금지"),
        "KDCA·WHO 임신부 인플루엔자 접종 권고"),
    KrEpiItem("KL22", "예방접종",
        "국내 인플루엔자 예방접종 권장 시기는?",
        "유행 전 매년 10~12월(늦어도 유행 시작 전), 항체 형성 약 2주 소요.",
        ("10", "2주"), ("여름", "유행 후"),
        "KDCA 인플루엔자 국가예방접종 안내"),
    KrEpiItem("KL23", "역학",
        "A형과 B형 인플루엔자의 주요 차이는?",
        "A형은 아형(H·N) 다양·동물 숙주·대유행 원인 / B형은 사람 중심·계통(Victoria·Yamagata)·대유행 안 일으킴.",
        ("A형", "B형", "Victoria"), ("동일", "차이 없음"),
        "WHO·KDCA 인플루엔자 바이러스 분류"),
    KrEpiItem("KL24", "역학",
        "항원소변이(antigenic drift)와 항원대변이(antigenic shift)의 차이는?",
        "drift=점진적 점돌연변이(계절 유행·매년 백신 갱신) / shift=유전자 재조합 대변이(신종·대유행, A형만).",
        ("drift", "shift", "대유행"), ("동일", "B형 shift"),
        "WHO 인플루엔자 항원변이"),
    KrEpiItem("KL25", "역학",
        "A/H3N2와 A/H1N1pdm09의 임상·역학 차이를 한 줄로.",
        "H3N2는 고령층 중증·초과사망 큼, H1N1pdm09는 상대적으로 젊은층 영향(2009 대유행 기원).",
        ("H3N2", "고령"), ("동일", "차이 없음"),
        "KDCA·문헌 subtype 임상영향"),
    KrEpiItem("KL26", "역학",
        "인플루엔자의 잠복기와 전염 가능 기간은?",
        "잠복기 약 1~4일(평균 2일); 전염은 증상 1일 전~발병 후 약 5~7일(소아·면역저하자 더 김).",
        ("1", "5"), ("2주 잠복", "전염 안 됨"),
        "KDCA·CDC 인플루엔자 임상"),
    KrEpiItem("KL27", "역학",
        "오셀타미비르의 항바이러스 효과를 위한 투여 시점 기준은?",
        "증상 발현 후 가능한 한 빨리, 48시간 이내 투여 시 효과 최대(고위험군은 이후에도 고려).",
        ("48",), ("1주 후", "효과 없음"),
        "KDCA·WHO 항바이러스제 지침"),
    KrEpiItem("KL28", "역학",
        "바로사비르 마르복실(baloxavir)의 투여 방식 특징은?",
        "단회 경구 투여(single dose), cap-dependent endonuclease 억제 기전.",
        ("단회", "1회"), ("5일 2회", "주사만"),
        "KDCA·문헌 baloxavir"),
    KrEpiItem("KL29", "역학",
        "인플루엔자 고위험군과 대표 합병증은?",
        "고위험군=65세 이상·영유아·임신부·만성질환·면역저하; 합병증=폐렴(이차세균감염), 기저질환 악화.",
        ("65", "폐렴"), ("합병증 없음", "건강한 성인만"),
        "KDCA·WHO 고위험군"),
    KrEpiItem("KL30", "역학",
        "집단면역(herd immunity)의 개념을 한 줄로.",
        "충분한 비율이 면역(접종·감염)되면 미접종자도 간접 보호되어 전파가 억제되는 현상.",
        ("간접", "전파"), ("개인만 보호", "100% 필요"),
        "WHO 집단면역 개념"),
    KrEpiItem("KL31", "역학",
        "기초감염재생산수 R0와 실효재생산수 Rt의 차이는?",
        "R0=완전감수성 집단 가정 평균 2차감염수, Rt=특정 시점 면역·개입 반영한 실제 전파; Rt<1이면 유행 감소.",
        ("Rt", "1"), ("동일", "항상 같음"),
        "역학 표준 정의 · Cori 2013"),
    KrEpiItem("KL32", "데이터",
        "WHO FluNet은 무엇이며 무엇을 제공하는가?",
        "GISRS 기반 전 세계 인플루엔자 바이러스 감시 도구(1997~), 주간 검출 건수·아형·양성률 공개.",
        ("FluNet", "주간"), ("국내만", "확진 진단 도구"),
        "WHO FluNet/GISRS"),
    KrEpiItem("KL33", "데이터",
        "미국 CDC FluView/ILINet은 한국 KDCA 표본감시와 어떻게 대응되는가?",
        "둘 다 의원급 표본감시 ILI 지표(주간); ILINet은 미국, KDCA는 국내 — 정의·denominator 유사하나 국가별 보정 필요.",
        ("ILINet", "ILI", "주간"), ("동일 데이터", "확진수"),
        "CDC FluView · KDCA 표본감시"),
    KrEpiItem("KL34", "데이터",
        "감염병 감시자료의 '보고지연(reporting lag)'이 실시간 해석에 주는 함의는?",
        "최근 주차는 추후 상향 보정될 수 있어 과소평가 위험 → nowcasting·vintage 관리 필요.",
        ("보고지연", "보정"), ("즉시 확정", "지연 없음"),
        "감시역학 표준 · nowcasting"),
    KrEpiItem("KL35", "데이터",
        "ILI '신고율(notification rate)'과 '실제 발생률(incidence)'을 동일시하면 안 되는 이유는?",
        "과소보고·표본기관 편향·denominator(외래총수) 변동 때문 — 신고지표는 추세 대리이지 절대발생률 아님.",
        ("과소", "추세"), ("동일", "절대 발생률"),
        "감시역학 표준 caveat"),
    KrEpiItem("KL36", "방역",
        "감염병 유행 시 '집합 제한·금지'와 '휴교'의 권한 주체는?",
        "질병관리청장·시·도지사·시장군수구청장(자치단체장) — 제49조에 근거.",
        ("자치단체", "제49조"), ("개인", "근거 없음"),
        "law.go.kr 감염병예방법 제49조"),
    KrEpiItem("KL37", "예방접종",
        "백신효과(VE)를 음성대조 설계(test-negative design)로 추정하는 이유는?",
        "의료이용 행태 등 교란을 줄여 접종력-검사양성 연관으로 VE를 추정(관찰연구의 표준 접근).",
        ("음성대조", "교란"), ("무작위", "RCT만 가능"),
        "문헌 test-negative design"),
    KrEpiItem("KL38", "역학",
        "B형 인플루엔자의 두 계통(lineage)은?",
        "Victoria 계통과 Yamagata 계통(4가 백신은 두 계통 모두 포함).",
        ("Victoria", "Yamagata"), ("H1N1", "H3N2"),
        "WHO B형 계통 분류"),
    KrEpiItem("KL39", "신고",
        "감염병 역학조사를 거부·방해하면 어떻게 되는가?",
        "역학조사는 의무 협조 대상이며 거부·방해·기피 시 처벌(제18조).",
        ("제18조", "처벌"), ("임의", "처벌 없음"),
        "law.go.kr 감염병예방법 제18조"),
    KrEpiItem("KL40", "데이터",
        "KDCA 인플루엔자 주의보·경보는 무엇을 근거로 발령하는가?",
        "표본감시 ILI(/1,000)가 유행기준(절기별 산출)을 초과하면 주의보, 일정 배수 초과 시 경보.",
        ("ILI", "유행기준"), ("확진수만", "임의 발령"),
        "KDCA 인플루엔자 유행주의보 기준"),
)


def load_kr_epi_law() -> tuple[KrEpiItem, ...]:
    """Return the Korean epi/law QA set (official-source-anchored)."""
    return KR_EPI_LAW_QA


def load_kormedmcqa(subset: str = "doctor", split: str = "test",
                    n: int | None = 200) -> list[dict]:
    """Load + normalize KorMedMCQA (external Korean medical benchmark, 3,009 test).

    KorMedMCQA answers are 1-indexed ints (1→A … 5→E); this maps them to a clean
    schema so the existing harness can score by exact option match.

    Args:
        subset: one of doctor / nurse / pharm / dentist.
        split: dataset split (test recommended).
        n: cap (None = all).
    Returns:
        list of ``{subset, question, options{A..E}, answer_letter, answer_text,
        cot}`` dicts, or ``[]`` with a printed note if `datasets`/network is
        unavailable (no hard dependency — portability principle #1).
    """
    try:
        from datasets import load_dataset  # optional
        ds = load_dataset("sean0042/KorMedMCQA", subset, split=split)
        rows = ds if n is None else ds.select(range(min(n, len(ds))))
        out = []
        for r in rows:
            letter = "ABCDE"[int(r["answer"]) - 1]
            out.append({
                "subset": subset, "question": r["question"],
                "options": {k: r[k] for k in "ABCDE" if r.get(k)},
                "answer_letter": letter, "answer_text": r.get(letter, ""),
                "cot": r.get("cot", ""),
            })
        return out
    except Exception as e:  # noqa: BLE001 — optional external dataset
        print(f"[kr_epi_bench] KorMedMCQA unavailable ({type(e).__name__}); "
              f"`{__import__('sys').executable} -m pip install datasets` + network to "
              f"enable. Using KR_EPI_LAW_QA only.")
        return []


def format_mcqa(item: dict) -> str:
    """Render an MCQA item as a prompt (question + lettered options)."""
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return f"{item['question']}\n{opts}\n정답 보기(A-E)만 답하세요."


def score_mcqa(item: dict, model_letter: str) -> bool:
    """Exact-match the model's chosen option letter against the gold answer."""
    return (model_letter or "").strip().upper()[:1] == item["answer_letter"]


def categories() -> dict[str, int]:
    """Item count per category in the seed set."""
    from collections import Counter
    return dict(Counter(i.category for i in KR_EPI_LAW_QA))
