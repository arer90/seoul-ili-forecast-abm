"use client";
/**
 * /map3d/abm — ABM-dedicated dashboard (사용자 채택 2026-06-06).
 *
 * Left: ARIA advisory bound bidirectionally to the sim + map. Right: the 25-gu
 * SEIR-V-D infection animation over a precomputed scenario trajectory
 * (abm-scenarios.json) with scenario selector, day slider, and summary stats.
 */
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { AbmScenarios } from "@/components/AbmMap";
import { AriaSimPanel } from "@/components/AriaSimPanel";

const AbmMap = dynamic(() => import("@/components/AbmMap").then((m) => m.AbmMap), {
  ssr: false,
  loading: () => <div className="flex h-full items-center justify-center text-slate-500">지도 로딩…</div>,
});

export default function AbmDashboardPage() {
  const [data, setData] = useState<AbmScenarios | null>(null);
  const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [scenario, setScenario] = useState("baseline");
  const [day, setDay] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [highlightedGu, setHighlightedGu] = useState<string | null>(null);

  useEffect(() => {
    fetch("/aggregates/abm-scenarios.json").then((r) => r.json()).then(setData).catch(() => setData(null));
    fetch("/seoul-gu.geojson").then((r) => r.json()).then(setGeojson).catch(() => setGeojson(null));
  }, []);

  const days = data?.days ?? 1;
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setDay((d) => (d + 1) % days), 120);
    return () => clearInterval(id);
  }, [playing, days]);

  const sc = data?.scenarios[scenario] ?? null;
  const cityInc = sc?.city_incidence[day] ?? 0;
  const scenarioList = useMemo(
    () => (data ? Object.entries(data.scenarios).map(([k, v]) => ({ key: k, label: v.label })) : []),
    [data],
  );

  return (
    <main className="mx-auto max-w-7xl space-y-3 p-4">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">ABM 대시보드 — Metapop SEIR-V-D × ARIA</h1>
        <p className="text-sm text-slate-600">
          서울 25-gu 메타population 행위자 시뮬레이션. 시나리오를 고르고 재생하면 구별 감염 확산이
          애니메이션되며, ARIA가 현재 상태를 읽어 개입을 조언합니다.{" "}
          <a href="/map3d" className="underline">← 전체 지도</a>
        </p>
      </header>

      {/* scenario selector + stats */}
      <div className="flex flex-wrap items-center gap-2">
        {scenarioList.map((s) => (
          <button
            key={s.key}
            type="button"
            onClick={() => { setScenario(s.key); setDay(0); }}
            className={`rounded px-3 py-1 text-sm ${
              scenario === s.key ? "bg-sky-600 text-white" : "bg-slate-200 hover:bg-slate-300"
            }`}
          >
            {s.label}
          </button>
        ))}
        {sc && (
          <div className="ml-auto flex gap-3 text-xs text-slate-600">
            <span>피크 <b>{sc.peak_day}일</b></span>
            <span>발병률 <b>{sc.city_attack_pct}%</b></span>
            <span>사망 <b>{sc.deaths.toLocaleString()}</b></span>
            <span className={sc.epi_validity_ok ? "text-emerald-600" : "text-rose-600"}>
              게이트 {sc.epi_validity_ok ? "✓" : "✗"}
            </span>
          </div>
        )}
      </div>

      {sc?.legal_basis && (
        <div className="rounded bg-slate-100 px-3 py-1.5 text-xs text-slate-700">
          ⚖ 개입 법적 근거 (감염병예방법): <b>{sc.legal_basis}</b>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[360px_1fr]">
        {/* ARIA */}
        <div className="h-[70vh]">
          <AriaSimPanel data={data} scenario={scenario} day={day} onHighlightGu={setHighlightedGu} />
        </div>

        {/* map + slider */}
        <div className="relative h-[70vh] overflow-hidden rounded-lg bg-slate-950">
          <AbmMap
            data={data}
            geojson={geojson}
            scenario={scenario}
            day={day}
            highlightedGu={highlightedGu}
            onGuClick={setHighlightedGu}
          />
          <div className="absolute bottom-3 left-3 right-3 flex items-center gap-3 rounded bg-slate-900/85 px-3 py-2 text-xs text-slate-200 backdrop-blur">
            <button
              type="button"
              onClick={() => setPlaying(!playing)}
              className="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600"
            >
              {playing ? "⏸" : "▶"}
            </button>
            <span className="font-mono tabular-nums">{day}일 / {days - 1}</span>
            <input
              type="range"
              min={0}
              max={days - 1}
              value={day}
              onChange={(e) => { setDay(Number(e.target.value)); setPlaying(false); }}
              className="flex-1"
            />
            <span className="text-slate-400">감염 {Math.round(cityInc).toLocaleString()}명</span>
          </div>
          <div className="absolute right-3 top-3 rounded bg-slate-900/85 px-2 py-1 text-[10px] text-slate-300 backdrop-blur">
            구 색 = 감염률 I(t)/N (어두움 낮음 → 빨강 높음){highlightedGu ? ` · 강조: ${highlightedGu}` : ""}
          </div>
        </div>
      </div>
    </main>
  );
}
