/**
 * ScenarioPicker14 — Sprint 2026-05-06 Phase D.1.
 *
 * 14 scenarios (6 base + 8 extended) selector — paper §5.6 통합. 기존
 * ScenarioPicker.tsx (6 only) 와 별도 component (호환 유지).
 *
 * 사용자 critique: "What-if 시뮬레이터는 기존의 인플루엔자와 다음의 채팅과
 * 구별이 안 되고 있어" — scenario picker 가 시각적으로 명확히 분리,
 * category badge (base / intervention / sensitivity / outcome / subtype) 으로
 * 구분.
 *
 * Header 에서 사용. 클릭 시 dropdown panel — 14 scenarios category 별 list,
 * paper § reference + description hover.
 */
"use client";
import { useEffect, useState } from "react";

type Scenario = {
  id: string;
  name_ko: string;
  name_en: string;
  category: "base" | "intervention" | "sensitivity" | "outcome" | "subtype";
  paper_section: string;
  description: string;
};

const CATEGORY_COLORS: Record<Scenario["category"], string> = {
  base: "bg-slate-700 text-slate-100",
  intervention: "bg-blue-700 text-blue-50",
  sensitivity: "bg-amber-700 text-amber-50",
  outcome: "bg-rose-700 text-rose-50",
  subtype: "bg-purple-700 text-purple-50",
};

export function ScenarioPicker14({
  selected,
  onSelect,
  className = "",
}: {
  selected?: string;
  onSelect?: (scenarioId: string) => void;
  className?: string;
}) {
  const [scenarios, setScenarios] = useState<Scenario[] | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    fetch("/api/sim/scenarios")
      .then((r) => r.json())
      .then((d) => setScenarios(d.scenarios ?? []))
      .catch(() => setScenarios([]));
  }, []);

  if (!scenarios) {
    return (
      <span className={`text-[11px] text-slate-500 ${className}`}>...</span>
    );
  }

  const sel = scenarios.find((s) => s.id === selected);

  return (
    <div className={`relative inline-flex ${className}`}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1.5 rounded border border-slate-700 bg-slate-800 px-2 py-1 text-[11px] font-medium text-slate-200 hover:bg-slate-700"
        title="14 시나리오 (paper §5.6)"
      >
        <span>📊</span>
        <span>{sel ? sel.name_ko : `시나리오 (${scenarios.length})`}</span>
        <span className="text-slate-400">▾</span>
      </button>

      {open && (
        <>
          {/* Backdrop click closes */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
            aria-hidden="true"
          />
          <div className="absolute left-0 top-full z-50 mt-1 max-h-[70vh] w-96 overflow-y-auto rounded border border-slate-700 bg-slate-900 p-2 shadow-xl">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">
                14 시나리오 (paper §5.6 — 6 base + 8 extended)
              </span>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded px-1 text-xs text-slate-400 hover:bg-slate-700"
                aria-label="close"
              >
                ✕
              </button>
            </div>
            <ul className="space-y-1">
              {scenarios.map((s) => (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => {
                      onSelect?.(s.id);
                      setOpen(false);
                    }}
                    className={`w-full rounded px-2 py-1.5 text-left text-[11px] hover:bg-slate-800 ${
                      selected === s.id ? "bg-slate-800" : ""
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <span
                        className={`rounded px-1 text-[10px] font-mono ${
                          CATEGORY_COLORS[s.category]
                        }`}
                      >
                        {s.category}
                      </span>
                      <span className="font-medium text-slate-100">
                        {s.name_ko}
                      </span>
                    </div>
                    <div className="mt-0.5 text-[10px] text-slate-400">
                      {s.paper_section} · {s.description}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}
