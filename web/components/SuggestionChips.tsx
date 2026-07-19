/**
 * SuggestionChips — 7 epidemiology-oriented recommendation sections,
 * per `docs/internal/stage_plan.md` §6c.
 *
 * Each chip is a canned prompt that gets injected into the composer
 * (the user can still edit it before sending). The intent is to keep
 * the demo flowing even if the presenter blanks — they just tap a
 * chip and the right follow-up question appears pre-filled.
 */
"use client";

import { useState } from "react";

export interface SuggestionSection {
  title: string;
  prompts: string[];
}

/**
 * Section taxonomy mirrors the stage plan; wording is the prompt we
 * drop into the composer, not the chip label.
 */
export const SUGGESTION_SECTIONS: SuggestionSection[] = [
  {
    title: "예측 타이밍",
    prompts: [
      "다음 8주 ILI 피크 주차와 95% PI를 구해줘. 근거 feature 상위 5개도 같이.",
      "이번 주 nowcast 가 지난 주 대비 얼마나 업데이트됐어? 보고지연 영향은?",
      "지난 3년 피크 주차 분포와 이번 시즌 예측 피크를 비교해줘.",
    ],
  },
  {
    title: "Rt 해석",
    prompts: [
      "현재 Rt 와 지난 4주 추세. 1.0 돌파 여부와 자치구별 분포를 보여줘.",
      "Rt 추정치의 Cori vs Wallinga-Teunis 차이를 짧게 비교해줘.",
    ],
  },
  {
    title: "SHAP 델타",
    prompts: [
      "이번 주 vs 지난 주 SHAP top10 변화. 어떤 feature 영향력이 커졌나?",
      "지역별 SHAP 프로파일 차이 중에서 통근 네트워크로 설명 가능한 부분은?",
    ],
  },
  {
    title: "모델 성능",
    prompts: [
      "WF-CV 상위 3 모델의 MAPE/R²/coverage 비교. 어느 걸로 발표해야 할까?",
      "TabularDNN-Lite 와 TabularDNN 풀모델 성능 차이가 유의한가? DM test 결과는?",
    ],
  },
  {
    title: "서울 현황",
    prompts: [
      "현재 자치구별 ILI rate 히트맵. 상위 5구와 하위 5구는?",
      "통근 네트워크 상 감염 유입 risk 가 가장 큰 자치구를 골라줘.",
    ],
  },
  {
    title: "예측 조회",
    prompts: [
      "종로구의 다음 4주 예측 ILI rate 와 95% PI 를 테이블로 보여줘.",
      "강남 3구(강남/서초/송파) 의 예측을 비교해줘.",
    ],
  },
  {
    title: "시나리오 What-if",
    prompts: [
      "NPI 락다운을 8주 시행하면 피크 I 와 최종 D 가 얼마나 줄어?",
      "백신 커버리지 50% → 70% 로 올리면 Rt 가 언제쯤 1 미만으로 내려가?",
      "항바이러스 프로필락시스를 고위험군 20% 에 적용하면 최종 D 는?",
    ],
  },
];

export interface SuggestionChipsProps {
  onPick: (prompt: string) => void;
  /** Optional filter by section title. */
  only?: string[];
  /**
   * When true (default), the whole widget is collapsed behind a single
   * "💡 Suggestions" toggle — user explicitly opts in before seeing the
   * 7 section pills. This matches the user's request:
   *   "추천 코멘트를 옵션으로 만들어서 클릭하면 리스트가 나오고 없는 것으로"
   *
   * When false, the section pills render immediately (used on the
   * landing empty-state card where we *want* to invite exploration).
   */
  collapsible?: boolean;
}

export function SuggestionChips({
  onPick,
  only,
  collapsible = true,
}: SuggestionChipsProps) {
  // Master toggle for the whole widget (only used when collapsible).
  const [expanded, setExpanded] = useState(!collapsible);
  // Which section's prompts are open. null = section pills shown but
  // no prompts expanded — matches the user's "click shows list, else
  // hidden" mental model.
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  const sections = only
    ? SUGGESTION_SECTIONS.filter((s) => only.includes(s.title))
    : SUGGESTION_SECTIONS;

  if (collapsible && !expanded) {
    return (
      <div className="flex items-center justify-between text-[11px] text-slate-500">
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-slate-300 hover:bg-slate-800"
          aria-expanded="false"
          aria-label="Show suggested prompts"
          title="Show suggested prompts / 추천 프롬프트 열기"
        >
          <span aria-hidden>💡</span>
          <span>Suggestions</span>
          <span aria-hidden className="text-slate-500">▸</span>
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5 text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        {sections.map((s, i) => {
          const active = openIdx === i;
          return (
            <button
              key={s.title}
              onClick={() => setOpenIdx(active ? null : i)}
              className={[
                "rounded-full border px-2 py-0.5 transition-colors",
                active
                  ? "border-sky-400 bg-sky-500/10 text-sky-200"
                  : "border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800",
              ].join(" ")}
            >
              {s.title}
            </button>
          );
        })}
        {collapsible ? (
          <button
            type="button"
            onClick={() => {
              setExpanded(false);
              setOpenIdx(null);
            }}
            className="ml-auto rounded-md border border-slate-800 px-1.5 py-0.5 text-[10px] text-slate-500 hover:bg-slate-800"
            aria-expanded="true"
            aria-label="Hide suggested prompts"
            title="Hide / 접기"
          >
            ✕
          </button>
        ) : null}
      </div>
      {openIdx != null && sections[openIdx] ? (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {sections[openIdx].prompts.map((p, j) => (
            <button
              key={j}
              onClick={() => onPick(p)}
              className="max-w-[28rem] truncate rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-left text-[11px] text-slate-200 hover:bg-slate-800"
              title={p}
            >
              {p}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
