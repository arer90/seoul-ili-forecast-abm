"""
simulation.llm_compare.golden_set
=================================
Twenty-item bilingual (Korean / English) advisory prompt set used for the
ARIA consultation layer comparison (thesis §4.17, §5.2c).

Structure (20 = 5 scenarios × 4 persona archetypes, mirrored in KO/EN)
---------------------------------------------------------------------
Scenarios:
    S1  Symptom triage
    S2  Vaccination counsel
    S3  Antiviral prescription decision
    S4  District-level alert decision  (couples to the forecasting registry)
    S5  Mechanistic scenario interpretation (couples to the SEIR-V-D sim)

Persona archetypes (applied to the end-user perspective):
    P1  District public-health officer  (KO default)
    P2  Primary-care physician           (EN default)
    P3  Patient-facing epidemiologist    (bilingual)
    P4  Policy advisor                   (mirrors across languages)

Each item is annotated with:
    id, scenario, persona, lang, difficulty, prompt, must_contain,
    must_avoid, style_tags, source

The ``must_contain`` / ``must_avoid`` lists drive the rule-based judge
in :mod:`simulation.llm_compare.judge`. The ``source`` field anchors
each item to the thesis section that motivated it, so the numbers
ledger (paper/numbers_ledger.csv) can audit every scored claim back to
its thesis section.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class GoldenItem:
    id: str
    scenario: str        # S1..S5
    persona: str         # P1..P4
    lang: str            # "ko" | "en"
    difficulty: str      # "textbook" | "edge" | "ambiguous"
    prompt: str
    must_contain: tuple[str, ...] = field(default_factory=tuple)
    must_avoid: tuple[str, ...] = field(default_factory=tuple)
    style_tags: tuple[str, ...] = field(default_factory=tuple)
    source: str = ""


# ---------------------------------------------------------------------------
# Twenty-item catalogue
# ---------------------------------------------------------------------------
_GOLDEN: tuple[GoldenItem, ...] = (
    # S1 Symptom triage ------------------------------------------------------
    GoldenItem(
        id="S1P1ko", scenario="S1", persona="P1", lang="ko", difficulty="textbook",
        prompt=(
            "보건소 담당자입니다. 65세 여성, 발열 38.2℃, 기침 · 근육통, 발병 2일 차. "
            "인플루엔자 신속검사 미시행. 현재 주간 ILI 12.3 / 1 000 으로 KDCA 주의 기준 초과. "
            "오셀타미비르 경험적 처방을 추천할지 근거와 함께 답하라."
        ),
        must_contain=("오셀타미비르", "48", "경험"),
        must_avoid=("절대", "100%", "반드시 효과"),
        style_tags=("triage", "antiviral", "empirical"),
        source="§3.4a, §4.13 threshold sensitivity",
    ),
    GoldenItem(
        id="S1P2en", scenario="S1", persona="P2", lang="en", difficulty="edge",
        prompt=(
            "Primary-care context. A 32-year-old healthy adult presents with 24 hours "
            "of fever, cough, and myalgia. Weekly ILI rate in the district is 8.4 / 1 000, "
            "just below the KDCA 8.6 advisory threshold. Rapid antigen test is negative. "
            "Should antiviral treatment be offered, and how do you weigh the threshold-sensitive "
            "operational recommendation from §4.13?"
        ),
        must_contain=("8.6", "threshold", "48"),
        must_avoid=("certain", "always", "guaranteed"),
        style_tags=("triage", "bordercase"),
        source="§4.13 KDCA threshold",
    ),
    GoldenItem(
        id="S1P3ko", scenario="S1", persona="P3", lang="ko", difficulty="ambiguous",
        prompt=(
            "역학자 관점에서 설명해주세요. 최근 4주간 ILI 증가세가 완만하지만 "
            "인접 구의 감시자료에서 B형 우세 신호가 감지됩니다. 환자-대면 가이드로 "
            "가족 내 전파를 최소화하기 위한 핵심 메시지 3가지를 정리해주세요."
        ),
        must_contain=("마스크", "환기", "손 씻기"),
        must_avoid=("격리", "구금", "강제"),
        style_tags=("patient-facing", "B-lineage"),
        source="§2.4 post-COVID rebounds",
    ),
    GoldenItem(
        id="S1P4en", scenario="S1", persona="P4", lang="en", difficulty="textbook",
        prompt=(
            "Policy context: cost-of-illness estimate for Seoul is requested by the "
            "Ministry of Health. Using the per-gu WIS = 3.794 and median PICP@95 = 0.865 "
            "reported in §4.9, describe the conservative uncertainty envelope you would "
            "attach to a four-week ILI forecast, and whether this supports a policy-grade "
            "recommendation."
        ),
        must_contain=("0.865", "prediction interval", "cover"),
        must_avoid=("100%", "perfect"),
        style_tags=("policy", "uncertainty"),
        source="§4.9 PI coverage",
    ),

    # S2 Vaccination counsel -------------------------------------------------
    GoldenItem(
        id="S2P1ko", scenario="S2", persona="P1", lang="ko", difficulty="textbook",
        prompt=(
            "구 단위 예방접종 담당자입니다. 현재까지 65세 이상 접종률 58 %. "
            "남은 4주간 70 % 목표 달성을 위한 우선순위 3가지를 논문의 "
            "서울 25-구 통근행렬 기반 시뮬레이션 근거와 함께 제시하세요."
        ),
        must_contain=("통근", "우선", "보건소"),
        must_avoid=("무조건", "강제접종"),
        style_tags=("vaccination", "operational"),
        source="§3.4 metapop, §4.4 Metapopulation SEIR-V-D",
    ),
    GoldenItem(
        id="S2P2en", scenario="S2", persona="P2", lang="en", difficulty="edge",
        prompt=(
            "Primary-care context. A pregnant patient at 18 weeks asks whether the "
            "inactivated influenza vaccine is safe given the current ILI activity. "
            "Summarise the benefit/risk trade-off and anchor it to the KDCA 2024-25 "
            "advisory threshold of 8.6 / 1 000. Avoid absolute safety claims."
        ),
        must_contain=("inactivated", "pregnan", "benefit"),
        must_avoid=("live", "attenuated", "guaranteed"),
        style_tags=("vaccination", "pregnancy"),
        source="§4.13",
    ),
    GoldenItem(
        id="S2P3ko", scenario="S2", persona="P3", lang="ko", difficulty="ambiguous",
        prompt=(
            "역학자-환자 면담입니다. 환자가 '백신을 맞아도 걸리는데 왜 맞아야 하냐' "
            "고 묻습니다. VE = 0.42 (TND, 2023-24 시즌 관측값) 해석과 "
            "접종 의사결정 프레임을 2-3문단으로 정리하세요."
        ),
        must_contain=("0.42", "VE", "중증"),
        must_avoid=("100%", "완벽", "예방보장"),
        style_tags=("patient-facing", "VE-interpretation"),
        source="§3.4 VE parameter",
    ),
    GoldenItem(
        id="S2P4en", scenario="S2", persona="P4", lang="en", difficulty="textbook",
        prompt=(
            "Policy context. Modelled under a +20 percentage-point baseline vaccination "
            "scenario in §4.4.3a, the counterfactual peak height drops by an estimated "
            "-12 %. Write a one-paragraph policy brief summarising the operational "
            "implication and the caveats from §6.3 (PI under-coverage)."
        ),
        must_contain=("-12", "+20", "counterfactual"),
        must_avoid=("guaranteed", "will prevent all"),
        style_tags=("policy", "counterfactual"),
        source="§4.4.3a vaccination scenario",
    ),

    # S3 Antiviral prescription decision -------------------------------------
    GoldenItem(
        id="S3P1ko", scenario="S3", persona="P1", lang="ko", difficulty="textbook",
        prompt=(
            "보건소 운영회의 자료입니다. 오셀타미비르 75 mg 2회/일 5일 요법의 "
            "노출 후 예방 (post-exposure prophylaxis) 적응증과 구내 비축 수준을 "
            "평가하기 위한 체크리스트 5항목을 작성하세요."
        ),
        must_contain=("5일", "75", "노출 후"),
        must_avoid=("무제한", "100 %"),
        style_tags=("antiviral", "PEP", "stockpile"),
        source="§3.4 antiviral intervention parameter",
    ),
    GoldenItem(
        id="S3P2en", scenario="S3", persona="P2", lang="en", difficulty="edge",
        prompt=(
            "Primary-care context. Baloxavir marboxil versus oseltamivir in an otherwise "
            "healthy adult with symptom onset 60 hours ago. Give a structured "
            "recommendation with rationale, and a hard stop on antiviral resistance "
            "concerns per recent (2024-25) Korean surveillance reports."
        ),
        must_contain=("baloxavir", "oseltamivir", "resistance"),
        must_avoid=("superior in all", "always better"),
        style_tags=("antiviral", "resistance"),
        source="§4.15 F4/F8 structural limits (analog)",
    ),

    # S4 District-level alert decision (couples to forecasting) --------------
    GoldenItem(
        id="S4P1ko", scenario="S4", persona="P1", lang="ko", difficulty="ambiguous",
        prompt=(
            "구 단위 주간 알람 회의 전 15분 브리프입니다. 예측 스택 상위 5개 모델의 "
            "다음 주 ILI 예측이 [11.8, 12.5, 14.1, 15.0, 15.7] / 1 000 (중앙값 14.1). "
            "KDCA 경보 기준 8.6, q70 11.45, 감사-보정 27.28. 브리프 헤드라인 3줄과 "
            "3-단계 운영 조치를 논문 §4.13 위너-by-threshold 표와 연결해 작성하세요."
        ),
        must_contain=("14.1", "8.6", "11.45"),
        must_avoid=("100 %", "확실하게"),
        style_tags=("alert", "operational", "ensemble"),
        source="§4.13 threshold sensitivity Table",
    ),
    GoldenItem(
        id="S4P2en", scenario="S4", persona="P2", lang="en", difficulty="textbook",
        prompt=(
            "District clinical lead, English briefing. Next-week ensemble median forecast "
            "is 11.9 / 1 000. Q70 threshold is 11.45. KDCA advisory is 8.6. "
            "Structure a 1-2 paragraph brief that names the winning model at each "
            "threshold (per §4.13 Table 4.13) and the downstream operational implication."
        ),
        must_contain=("iTransformer", "TimesFM-2.5", "SVR-RBF"),
        must_avoid=("one model fits", "universal"),
        style_tags=("alert", "ensemble", "threshold-contingent"),
        source="§4.13",
    ),
    GoldenItem(
        id="S4P3ko", scenario="S4", persona="P3", lang="ko", difficulty="edge",
        prompt=(
            "역학자 시각. 현재 예측 PI 경험적 coverage 가 목표 0.95 대비 0.865 (§4.9). "
            "알람을 낼지 보류할지 의사결정에 이 mis-calibration 을 어떻게 반영할지 "
            "한 단락으로 서술하세요. CQR / Mondrian / ACI 재보정 3가지 옵션을 언급하세요."
        ),
        must_contain=("0.865", "CQR", "Mondrian"),
        must_avoid=("무시", "상관없음"),
        style_tags=("alert", "PI-undercoverage", "recalibration"),
        source="§4.13 + CQR/Mondrian",
    ),
    GoldenItem(
        id="S4P4en", scenario="S4", persona="P4", lang="en", difficulty="ambiguous",
        prompt=(
            "Policy-grade briefing for the Seoul Metropolitan Government. The 66-model "
            "registry produces a median peak-week error of |Δ| = 1 week with 0 error for "
            "NegBinGLM, MP-PINN, PINN-Lite, Rt-Augmented, SEIR-V2-Forced. Given that "
            "Rt-Augmented and SEIR-V2-Forced are documented structural negatives in §4.15 "
            "F11, how do you report the 0-week result to the policy layer without overclaiming?"
        ),
        must_contain=("F11", "structural", "1 week"),
        must_avoid=("no uncertainty", "perfect model"),
        style_tags=("policy", "F11"),
        source="§4.15 F11",
    ),

    # S5 Mechanistic scenario interpretation --------------------------------
    GoldenItem(
        id="S5P1ko", scenario="S5", persona="P1", lang="ko", difficulty="textbook",
        prompt=(
            "SEIR-V-D 시뮬레이터가 25-구 통근 결합으로 65 % 학령기 학교 폐쇄를 "
            "28일간 시행한 반사실 결과를 반환했습니다. peak 높이 -9 %, peak 지연 +1 주. "
            "이 결과를 §4.4.3a 틀에서 운영 메시지로 변환하세요. 가계 전파 간접 효과와 "
            "§3.4a 의 4-parameter ABM 보정 권고를 포함하세요."
        ),
        must_contain=("-9", "+1", "3.4a"),
        must_avoid=("학교 폐쇄 만능", "무조건 효과"),
        style_tags=("counterfactual", "school-closure"),
        source="§4.4.3a + §3.4a",
    ),
    GoldenItem(
        id="S5P2en", scenario="S5", persona="P2", lang="en", difficulty="edge",
        prompt=(
            "Commuter-coupled SEIR-V-D counterfactual with a 40 % NPI intensity sustained "
            "for 21 days. Output: peak delay +3 weeks, peak height -38 %, attack rate -11 "
            "percentage points. Write a briefing for the hospital surge-capacity lead "
            "explaining the trade-off and flagging the §3.4a behavioural-ABM caveat that "
            "the NPI response itself may accrue fatigue τ that reverses the effect."
        ),
        must_contain=("-38", "+3", "fatigue"),
        must_avoid=("guaranteed suppression", "will never rebound"),
        style_tags=("counterfactual", "NPI"),
        source="§4.4.3a NPI40",
    ),
    GoldenItem(
        id="S5P3ko", scenario="S5", persona="P3", lang="ko", difficulty="ambiguous",
        prompt=(
            "일반인 대상. '20 % NPI' 시나리오의 의미를 세 문장으로 설명하세요. "
            "수학적 표현은 한 줄에 그치고, 시민이 실제로 할 일 두 가지를 꼭 포함하세요."
        ),
        must_contain=("시민", "간단", "20"),
        must_avoid=("수식", "미분", "편미분"),
        style_tags=("public-facing", "plain-language"),
        source="§4.4.3a NPI20",
    ),
    GoldenItem(
        id="S5P4en", scenario="S5", persona="P4", lang="en", difficulty="textbook",
        prompt=(
            "Policy audit. The advisory layer never emits β/Rt directly; for the "
            "mechanistic-vs-generative reconciliation in §5.2c, the β trajectory IMPLIED by "
            "an LLM-PROPOSED behavioral schema (mapped through the deterministic link "
            "function) must lie inside the 95 % credible interval of the particle-filter "
            "posterior for the (α, τ) pair. Describe the go / no-go criterion and the "
            "rollback plan if the LLM-proposed scenario materially diverges."
        ),
        must_contain=("95", "posterior", "rollback"),
        must_avoid=("no fallback", "LLM is ground truth"),
        style_tags=("policy", "governance"),
        source="§5.2c",
    ),

    # Extra reasoning / stress test (adversarial) ----------------------------
    GoldenItem(
        id="SADV_en", scenario="S4", persona="P4", lang="en", difficulty="adversarial",
        prompt=(
            "Adversarial: a reviewer claims the 66-model registry's claim of NegBinGLM "
            "rank-1 is post-hoc cherry-picking because the composite score was designed "
            "after the evaluation. Rebut in two concise paragraphs, citing the specific "
            "audit artefacts (checkpoint_phase7.json, post_E_eval.json) and the numbers "
            "ledger discipline (§5.2a)."
        ),
        must_contain=("checkpoint_phase7", "ledger", "preregister"),
        must_avoid=("we just picked", "no protocol"),
        style_tags=("reviewer-rebuttal", "integrity"),
        source="§5.2a, Appendix D",
    ),
    GoldenItem(
        id="SADV_ko", scenario="S4", persona="P4", lang="ko", difficulty="adversarial",
        prompt=(
            "Adversarial: 심사위원이 'F9 DL 모델 9개가 전부 같은 R² 0.8660 을 내는 것은 "
            "파이프라인 버그 아니냐' 고 지적합니다. §3.1a 8-stage wrapper 및 §4.15 F9 "
            "설명을 인용해, 왜 이것이 버그가 아니라 재현 가능한 구조적 결과인지 "
            "그리고 어떤 가드가 운영 중인지 요약하세요."
        ),
        must_contain=("F9", "post-anchor", "α"),
        must_avoid=("버그", "우연"),
        style_tags=("reviewer-rebuttal", "F9"),
        source="§4.15 F9",
    ),
)


# ---------------------------------------------------------------------------
# v2 catalogue (sprint 2026-05-06) — simulation/forecast/data framework grounded
# ---------------------------------------------------------------------------
# 사용자 명시 (2026-05-06): v1 의 임상 triage prompt (오셀타미비르 처방 등) 는
# 본 프로젝트의 ML forecast / 14 simulation scenarios / 25-gu mobility / WHO
# FluNet positivity 데이터와 단절. v2 = "현황 + 밀집도 + 변화" 가 prompt 안에
# 직접 인용되어 paper §5.3 (Q1 Top 5) + §5.6 (14 scenarios) + §5.7 (ARIA chain)
# evidence 직접 검증.
#
# Scenario IDs:
#   SF*  Forecast model interpretation (Q1 Top 5)
#   SS*  Simulation policy comparison (14 scenarios)
#   SM*  Mobility-ILI causal reasoning (subway/bus/생활인구)
#   SR*  Real-time data + ARIA chain (FluNet positivity, subtype, integration)
_GOLDEN_V2: tuple[GoldenItem, ...] = (
    # SF1 — District-level forecast alert with Q1 Top 5 PI95 ─────────────────
    GoldenItem(
        id="SF1P1ko", scenario="SF1", persona="P1", lang="ko",
        difficulty="textbook",
        prompt=(
            "보건소 직원입니다. sprint 2026-05-06 cache 데이터로 강남구 다음 주 "
            "ILI 위험도 평가 + 권고 대응 방안을 분석해주세요.\n\n"
            "[현황]\n"
            "- 강남구 ILI rate: 11.4/1000 (전주 대비 -2.1, KDCA 주의 기준 8.6/1000 초과)\n"
            "- 25-gu 전체 ILI 평균: 14.5/1000 (sentinel last 2026-04-30)\n\n"
            "[밀집도]\n"
            "- 강남구 daytime population: 1.2M (resident 540k + 통근 inflow 660k)\n"
            "- rt_subway_crowd_lag1: 3.8/4 (붐빔)\n"
            "- bus traffic: 일평균 88k (전주 -5%)\n\n"
            "[Q1 Top 5 ML forecast — service zone ACI Gibbs 2021 PI95]\n"
            "- NegBinGLM (test R²=0.924, real MAE=4.74): next-week point=10.2, "
            "PI95 [7.1, 13.5]\n"
            "- GAM-Spline (test R²=0.929): 9.8, PI95 [6.8, 13.2]\n\n"
            "[변화 — last 8 weeks monotonic decline 2026-02-22 → 2026-04-12]\n"
            "- 28.87 → 11.40 (-60%)\n"
            "- WHO FluNet positivity_lag1: 0.133 (B Victoria 86% dominant)\n\n"
            "다음 주 ILI 위험도 + 권고 대응 방안 (학교 / 병원 capacity / "
            "vaccination push) — 학술 정직성 (uncertainty 인정) 포함."
        ),
        must_contain=("PI95", "Victoria", "권고"),
        must_avoid=("절대", "100%", "확실"),
        style_tags=("forecast", "data-grounded", "district-alert"),
        source="paper §5.3 Q1 Top 5 + §5.7 ARIA chain + §6.2 통합 framework",
    ),
    GoldenItem(
        id="SF1P4en", scenario="SF1", persona="P4", lang="en",
        difficulty="edge",
        prompt=(
            "Policy advisor. The MPH-Seoul sprint 2026-05-06 ML forecasting "
            "tier reports five Q1 models (R²≥0.9 on test n=68): GAM-Spline "
            "0.929, NegBinGLM 0.924, DNN-Conformal 0.911, KRR 0.909, "
            "SVR-Linear 0.902. Family diversity: additive / Bayesian GLM / "
            "deep / kernel / linear. Service zone (real n=8, 2026-02-22 to "
            "2026-04-12) MAE 4.74-9.16, PICP95 0.88-1.00.\n\n"
            "Question: Which of the five would you recommend as the headline "
            "operational forecaster for KDCA, and how do you weigh the "
            "n=8 R²<0 caveat (paper §6.4 10번째 항)? Cite the family-diversity "
            "argument and Hyndman 2021 §3.4 metric guidance."
        ),
        must_contain=("Hyndman", "n=8", "family"),
        must_avoid=("certainly", "always", "guaranteed"),
        style_tags=("forecast", "model-selection", "Q1-Top-5"),
        source="paper §5.3 + §6.4 n=8 R² caveat",
    ),
    # SS1 — Simulation 14 scenario policy comparison ────────────────────────
    GoldenItem(
        id="SS1P1ko", scenario="SS1", persona="P1", lang="ko",
        difficulty="ambiguous",
        prompt=(
            "서울시 보건소 직원입니다. sprint 2026-05-06 의 Metapop SEIR-V-D "
            "14 scenarios sim 결과를 정책 권고로 번역해주세요.\n\n"
            "[Sim 결과 — 25-gu deterministic ODE, RK4 dt=0.25, 365 days]\n"
            "- baseline: peak I=311,725 (day 178), final D=4,490\n"
            "- npi_lockdown: peak I=88,161 (-72%), final D=409 (-91%)\n"
            "- vaccination_campaign: peak I=113,065 (-64%), final D=1,011 (-77%)\n"
            "- antiviral_prophylaxis: peak I=17,031 (-95%), final D=74 (-98%)\n"
            "- combined_response: peak I=13,220 (-96%), final D=90 (-98%)\n"
            "- sensitivity_strain_mismatch (VE=0.20): identical to baseline\n\n"
            "[β framing — paper §결론 G-184 caveat]\n"
            "본 sim β = 'aggregate syndromic respiratory-pathogen transmission '\n"
            "rate' (NOT influenza-specific R0). KDCA ILI = RSV/SARS-CoV-2/hMPV "
            "co-circulation 가능.\n\n"
            "권고 우선순위 (정책 trade-off): cost / 사회적 수용성 / 효과 "
            "(combined_response 가 best BUT 가장 큰 비용). 학술 정직 보고."
        ),
        must_contain=("syndromic", "trade-off", "combined"),
        must_avoid=("절대", "100% 확실", "유일한 답"),
        style_tags=("simulation", "policy-trade-off", "14-scenarios"),
        source="paper §5.6 sim 14 scenarios + §결론 G-184 syndromic β",
    ),
    # SM1 — Mobility-ILI causal reasoning (paper §4.8 mechanism-aware) ──────
    GoldenItem(
        id="SM1P3en", scenario="SM1", persona="P3", lang="en",
        difficulty="ambiguous",
        prompt=(
            "Patient-facing epidemiologist. Sprint 2026-05-06 ARIA simulation "
            "advisor receives the following multi-source signal for Seoul "
            "during the 2020-W10 to 2022-W30 NPI period:\n\n"
            "- subway ridership: -50 to -70% from 2019 baseline\n"
            "- school_closure_lag1 events: 8x baseline\n"
            "- KDCA sentinel ILI: 0.5/1000 (vs 5-10/1000 typical winter peak)\n"
            "- WHO FluNet positivity (KR): 0.02 (vs ~0.30 typical)\n\n"
            "Question: Did the Korean NPI bundle 'cause' the ILI suppression? "
            "Use Wagner 2002 ITS framework and paper §4.8 mechanism-aware "
            "discussion (binary dummy 회피, mobility/closure as causal "
            "proxies). Address the no-causal-claim caveat (paper §6.4 9번째 항: "
            "observational study, no DAG / IV / DiD)."
        ),
        must_contain=("Wagner", "observational", "associated"),
        must_avoid=("caused", "definitively", "proves"),
        style_tags=("causal-reasoning", "NPI", "mechanism-aware"),
        source="paper §4.8 NPI mixc 3-layer + §6.4 9번째 항 No causal claim",
    ),
    # SR1 — Real-time + ARIA chain integration ──────────────────────────────
    GoldenItem(
        id="SR1P4en", scenario="SR1", persona="P4", lang="en",
        difficulty="textbook",
        prompt=(
            "Policy advisor. ARIA chat AI (paper §4.12 Stage 6 prototype) "
            "is queried at 2026-05-06 14:00 KST with the following live "
            "data fed via MCP epi.* tools (10 tools, 6 fully wired):\n\n"
            "- epi.forecast (NegBinGLM): Seoul aggregate next-week ILI = 9.5, "
            "PI95 [6.7, 12.8] (Edge ISR cache 60s)\n"
            "- epi.rt_estimate (Cori 2013 EpiEstim): Rt = 0.84 (95% CrI 0.71-0.99)\n"
            "- epi.scenario_run (combined_response): peak shift +14 days vs "
            "baseline, peak I -96%\n"
            "- epi.outbreak_detect (EARS-C1): no flag (signal < 2σ)\n"
            "- WHO FluNet positivity_lag1: 0.133 (B Victoria 86%)\n\n"
            "Translate this multi-tool output into a single 3-paragraph public "
            "health communication for the 보건복지부 weekly briefing. Cite "
            "Hermes parallel mode (Anthropic + OpenAI + Google) consensus path "
            "if any model disagrees with the seasonal-naive baseline."
        ),
        must_contain=("Cori", "Hermes", "paragraph"),
        must_avoid=("certain", "definitely", "no risk"),
        style_tags=("ARIA-chain", "MCP-tools", "policy-comms"),
        source="paper §4.12 ARIA Stage 6 + §5.7 chain integration",
    ),
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_golden_set() -> tuple[GoldenItem, ...]:
    """Return the full catalogue (v1 임상 triage 20 + v2 sim/forecast 5).

    Sprint 2026-05-06 (사용자 명시): v1 (S1-S5 임상 + SADV) 보존 + v2 (SF1/SS1/
    SM1/SR1 — paper §5.3/§5.6/§5.7 framework grounded) 추가. 합 25 items.
    """
    return _GOLDEN + _GOLDEN_V2


def iter_scenarios() -> Iterable[str]:
    """Return the unique scenarios in definition order (full set incl. V2)."""
    seen: set[str] = set()
    for item in load_golden_set():
        if item.scenario not in seen:
            seen.add(item.scenario)
            yield item.scenario


def count_items() -> int:
    return len(load_golden_set())  # _GOLDEN + _GOLDEN_V2 (was len(_GOLDEN)=20, stale)
