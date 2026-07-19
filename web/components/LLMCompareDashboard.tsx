"use client";
/**
 * LLM Compare Dashboard (sprint 2026-05-06 — paper PART B Stage 7 viz).
 *
 * Multi-LLM benchmark for the ARIA consultation layer:
 *   - 20 golden items × 7 pillar rubric (golden_set.py + judge.py)
 *   - radar chart (per-pillar scores per backend)
 *   - bar chart (total ranking + latency)
 *   - per-item disagreement table (clickable → prompt + scenario detail)
 *
 * Data source: GET /api/llm-compare/report (filesystem read of
 * simulation/results/llm_compare/report.json).
 */
import { useEffect, useMemo, useState } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

type Pillar =
  | "correctness"
  | "hallucination"
  | "safety"
  | "calibration"
  | "specificity"
  | "structure"
  | "latency_cost";

const PILLARS: Pillar[] = [
  "correctness",
  "hallucination",
  "safety",
  "calibration",
  "specificity",
  "structure",
  "latency_cost",
];

const COLORS = [
  "#3b82f6", // blue (Anthropic baseline)
  "#ef4444", // red (OpenAI)
  "#10b981", // green (EXAONE)
  "#f59e0b", // amber (Qwen)
  "#8b5cf6", // violet (DeepSeek)
  "#ec4899", // pink (MedGemma)
  "#14b8a6", // teal (Phi-4)
];

type RankingRow = {
  backend_id: string;
  total: number;
  tier: string;
  mean_latency_ms: number;
};

type DisagreementRow = {
  item_id: string;
  n_backends: number;
  std_total: number;
  mean_total: number;
  best_backend: string;
  worst_backend: string;
};

type BackendInfo = {
  backend_id: string;
  model: string;
  tier: string;
  provider: string;
};

// Sprint 2026-05-06: report.json 의 actual schema (runner.py 로부터 생성).
// `items` is a flat list of (item × backend) results — we group/index it
// client-side for the per-item modal.
type ItemResult = {
  item_id: string;
  backend_id: string;
  model: string;
  scores: Record<Pillar, number>;
  total: number;
  missing_must_contain?: string[];
  hit_must_avoid?: string[];
  hedge_tokens_found?: string[];
  latency_ms: number;
  response_text?: string;
  error?: string;
};

type GoldenItem = {
  id: string;
  scenario: string;
  persona: string;
  lang: string;
  difficulty: string;
  prompt: string;
  must_contain: string[];
  must_avoid: string[];
  style_tags: string[];
  source: string;
};

type Report = {
  generated_at?: string;
  env?: { api_keys_present?: Record<string, boolean>;
          api_keys_missing?: string[];
          ollama_installed_models?: string[] };
  backends?: BackendInfo[];
  ranking?: RankingRow[];
  per_backend_mean?: Record<string, number>;
  per_pillar_mean?: Record<string, Record<Pillar, number>>;
  inter_backend_disagreement?: DisagreementRow[];
  items?: ItemResult[];
  golden_set?: GoldenItem[];
};

const SCENARIO_LABELS: Record<string, string> = {
  S1: "S1 Symptom triage",
  S2: "S2 Vaccination counsel",
  S3: "S3 Antiviral decision",
  S4: "S4 District alert (forecast)",
  S5: "S5 SEIR-V-D 해석 (sim)",
};

const PERSONA_LABELS: Record<string, string> = {
  P1: "P1 District officer",
  P2: "P2 Primary-care physician",
  P3: "P3 Patient epidemiologist",
  P4: "P4 Policy advisor",
};

/**
 * Sprint 2026-05-06 (#post-S3P2en): keyword highlight in response text.
 * 사용자 첨부 이미지 #2 (GPT-4 fail figure) 형식과 동일 — 정답 keyword 는
 * 초록 배경, 감점 keyword 는 빨강 배경. case-insensitive substring matching.
 *
 * Returns sanitized HTML — `dangerouslySetInnerHTML` 으로 사용. 입력 text /
 * keywords 는 모두 escape 후 매칭.
 */
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function highlightKeywordsHtml(
  text: string,
  mustContain: string[],
  mustAvoid: string[],
): string {
  let html = escapeHtml(text);
  // must_contain → green highlight
  mustContain.forEach((kw) => {
    if (!kw) return;
    const re = new RegExp(`(${escapeRegex(escapeHtml(kw))})`, "gi");
    html = html.replace(
      re,
      '<mark style="background:#bbf7d0;color:#064e3b;padding:0 2px;border-radius:2px">$1</mark>',
    );
  });
  // must_avoid → red highlight
  mustAvoid.forEach((kw) => {
    if (!kw) return;
    const re = new RegExp(`(${escapeRegex(escapeHtml(kw))})`, "gi");
    html = html.replace(
      re,
      '<mark style="background:#fecaca;color:#7f1d1d;padding:0 2px;border-radius:2px">$1</mark>',
    );
  });
  return html;
}

export function LLMCompareDashboard() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/llm-compare/report")
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(
            `${r.status} ${r.statusText}: ${body.message || body.error || ""}`,
          );
        }
        return r.json();
      })
      .then((data) => setReport(data as Report))
      .catch((e) => setError(String(e)));
  }, []);

  const radarData = useMemo(() => {
    if (!report?.per_pillar_mean) return [];
    return PILLARS.map((p) => ({
      pillar: p,
      ...Object.fromEntries(
        Object.entries(report.per_pillar_mean!).map(([b, scores]) => [
          b,
          scores[p] ?? 0,
        ]),
      ),
    }));
  }, [report]);

  const rankingData = useMemo(() => {
    return (report?.ranking ?? []).map((r) => ({
      name: r.backend_id,
      total: r.total,
      latency: r.mean_latency_ms,
    }));
  }, [report]);

  // Group items by item_id → {backend_id: ItemResult}
  const itemsByItem = useMemo(() => {
    const m = new Map<string, Map<string, ItemResult>>();
    (report?.items ?? []).forEach((it) => {
      if (!m.has(it.item_id)) m.set(it.item_id, new Map());
      m.get(it.item_id)!.set(it.backend_id, it);
    });
    return m;
  }, [report]);

  const selectedAnswers: [string, ItemResult][] = selectedItem
    ? Array.from(itemsByItem.get(selectedItem)?.entries() ?? [])
    : [];
  // First non-empty prompt-equivalent — show response_text excerpt of best
  const selectedFirstResult = selectedAnswers[0]?.[1] ?? null;
  // Golden-set lookup for the selected item (정답 기준 메타데이터)
  const selectedGolden: GoldenItem | null = selectedItem
    ? (report?.golden_set ?? []).find((g) => g.id === selectedItem) ?? null
    : null;

  if (error) {
    return (
      <div className="rounded border border-red-300 bg-red-50 p-4">
        <h3 className="font-semibold text-red-800">Failed to load report</h3>
        <p className="mt-1 text-sm text-red-700">{error}</p>
        <p className="mt-2 text-xs text-red-700">
          Hint: Run{" "}
          <code className="bg-red-100 px-1 py-0.5 rounded">
            python -m simulation.llm_compare.runner --out-dir
            simulation/results/llm_compare
          </code>{" "}
          to regenerate.
        </p>
      </div>
    );
  }
  if (!report) {
    return <div className="text-slate-500">Loading report…</div>;
  }

  const backends = (report.backends ?? []).map((b) =>
    typeof b === "string" ? b : b.backend_id,
  );

  return (
    <div className="space-y-8">
      {/* Header summary */}
      <section className="rounded border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
          <div>
            <span className="text-slate-500">Backends:</span>{" "}
            <strong>{backends.length}</strong> ({backends.join(", ")})
          </div>
          <div>
            <span className="text-slate-500">Items:</span>{" "}
            <strong>{(report.inter_backend_disagreement ?? []).length}</strong> distinct items × <strong>{(report.items ?? []).length}</strong> calls
          </div>
          <div>
            <span className="text-slate-500">Generated:</span>{" "}
            <code className="text-xs">{report.generated_at ?? "n/a"}</code>
          </div>
        </div>
      </section>

      {/* Radar — per-pillar */}
      <section className="rounded border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="mb-2 text-lg font-semibold">
          Per-pillar scores (radar) — 7 pillars × backends
        </h2>
        <p className="mb-3 text-xs text-slate-500">
          correctness · hallucination · safety · calibration · specificity ·
          structure · latency_cost (all 0 — 1 normalised)
        </p>
        <ResponsiveContainer width="100%" height={420}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="pillar" />
            <PolarRadiusAxis domain={[0, 1]} />
            {backends.map((b, i) => (
              <Radar
                key={b}
                name={b}
                dataKey={b}
                stroke={COLORS[i % COLORS.length]}
                fill={COLORS[i % COLORS.length]}
                fillOpacity={0.18}
              />
            ))}
            <Legend />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </section>

      {/* Total ranking + latency */}
      <section className="rounded border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="mb-2 text-lg font-semibold">
          Total ranking + mean latency
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={rankingData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis yAxisId="left" label={{ value: "total", angle: -90, position: "insideLeft" }} />
            <YAxis
              yAxisId="right"
              orientation="right"
              label={{ value: "latency (ms)", angle: 90, position: "insideRight" }}
            />
            <Tooltip />
            <Legend />
            <Bar yAxisId="left" dataKey="total" name="total score" fill="#3b82f6" />
            <Bar yAxisId="right" dataKey="latency" name="latency ms" fill="#f59e0b" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      {/* Per-item disagreement table */}
      <section className="rounded border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="mb-2 text-lg font-semibold">
          Inter-backend disagreement (per item) — clickable
        </h2>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-100">
              <tr>
                <th className="px-3 py-2 text-left">item</th>
                <th className="px-3 py-2 text-left">scenario</th>
                <th className="px-3 py-2 text-left">persona</th>
                <th className="px-3 py-2 text-right">std</th>
                <th className="px-3 py-2 text-right">mean</th>
                <th className="px-3 py-2 text-left">best</th>
                <th className="px-3 py-2 text-left">worst</th>
              </tr>
            </thead>
            <tbody>
              {(report.inter_backend_disagreement ?? []).map((d) => {
                // Item id pattern: S<scn>P<persona><lang> e.g. "S1P1ko"
                const m = /^([SADV]+\d*)P?(\d+)?([a-z]+)?$/.exec(d.item_id);
                const scn = m?.[1] ?? "";
                const persona = m?.[2] ?? "";
                const lang = m?.[3] ?? "";
                return (
                  <tr
                    key={d.item_id}
                    className={`cursor-pointer hover:bg-blue-50 ${
                      selectedItem === d.item_id ? "bg-blue-100" : ""
                    }`}
                    onClick={() =>
                      setSelectedItem(selectedItem === d.item_id ? null : d.item_id)
                    }
                  >
                    <td className="px-3 py-2 font-mono text-xs">{d.item_id}</td>
                    <td className="px-3 py-2">
                      {SCENARIO_LABELS[scn] ?? scn}
                    </td>
                    <td className="px-3 py-2">
                      {PERSONA_LABELS[`P${persona}`] ?? `P${persona}`}
                      {lang ? ` (${lang})` : ""}
                    </td>
                    <td className="px-3 py-2 text-right">{d.std_total.toFixed(3)}</td>
                    <td className="px-3 py-2 text-right">{d.mean_total.toFixed(3)}</td>
                    <td className="px-3 py-2 text-emerald-700">{d.best_backend}</td>
                    <td className="px-3 py-2 text-rose-700">{d.worst_backend}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Item × Backend score grid (이미지 표 style — best green highlighted) */}
      <section className="rounded border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="mb-2 text-lg font-semibold">
          Item × Backend score grid
        </h2>
        <p className="mb-3 text-xs text-slate-500">
          각 cell = total score ([0, 1] normalised). 행별 best score = 녹색 굵은체.
          파란 배경 강도 = score (Anthropic Claude 3.5 Sonnet release 표 style).
        </p>
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-slate-100">
              <tr>
                <th className="px-2 py-1 text-left">item</th>
                <th className="px-2 py-1 text-left">scn</th>
                <th className="px-2 py-1 text-left">lang</th>
                {backends.map((b) => (
                  <th
                    key={b}
                    className="px-2 py-1 text-center font-mono"
                    style={{ fontSize: "10px" }}
                  >
                    {b.replace(/^(api|cli|ollama):/, "")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(report.inter_backend_disagreement ?? []).map((d) => {
                const itMap = itemsByItem.get(d.item_id);
                const itemScores = backends.map(
                  (b) => itMap?.get(b)?.total ?? null,
                );
                const valid = itemScores.filter((s): s is number => s !== null);
                const max = valid.length ? Math.max(...valid) : 0;
                return (
                  <tr
                    key={d.item_id}
                    className="hover:bg-blue-50 cursor-pointer"
                    onClick={() =>
                      setSelectedItem(selectedItem === d.item_id ? null : d.item_id)
                    }
                  >
                    <td className="px-2 py-1 font-mono text-xs">{d.item_id}</td>
                    <td className="px-2 py-1 text-xs">{(/^([SADV]+\d*)/.exec(d.item_id))?.[1] ?? "—"}</td>
                    <td className="px-2 py-1 text-xs">{(/(ko|en|mix)$/.exec(d.item_id))?.[1] ?? "—"}</td>
                    {itemScores.map((s, i) => {
                      const isMax = s !== null && Math.abs(s - max) < 1e-6 && valid.length > 1;
                      return (
                        <td
                          key={i}
                          className={`px-2 py-1 text-center font-mono ${
                            isMax
                              ? "bg-emerald-100 font-bold text-emerald-800"
                              : ""
                          }`}
                          style={
                            s !== null && !isMax
                              ? {
                                  backgroundColor: `rgba(59, 130, 246, ${Math.min(1, Math.max(0, s)) * 0.30})`,
                                }
                              : undefined
                          }
                        >
                          {s !== null ? s.toFixed(3) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Selected item: 진짜 overlay modal (사용자 critique sprint 2026-05-06,
          third pass: "질문도 모르고 정답도 모르고 어떤점이 다른지 없다고 나오니까"
          — 이전 선택 시 modal 이 page 하단 section 이라 사용자가 scroll 안 해서
          못 봤음. fixed overlay + backdrop + Esc close 으로 즉시 표시). */}
      {selectedItem && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) setSelectedItem(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") setSelectedItem(null);
          }}
        >
          <section className="max-h-[92vh] w-full max-w-5xl overflow-y-auto rounded-lg border-2 border-blue-300 bg-blue-50 p-5 shadow-2xl">
          <div className="mb-3 flex items-center justify-between sticky top-0 -mx-5 -mt-5 mb-3 border-b border-blue-200 bg-blue-50 px-5 py-3">
            <h2 className="text-lg font-semibold">
              Item {selectedItem}
              {selectedFirstResult ? (
                <span className="ml-2 text-xs text-slate-500">
                  ({selectedAnswers.length} backends 비교)
                </span>
              ) : null}
            </h2>
            <button
              type="button"
              onClick={() => setSelectedItem(null)}
              className="rounded border border-slate-300 bg-white px-3 py-1 text-xs hover:bg-slate-100"
              aria-label="close (Esc)"
            >
              ✕ close (Esc)
            </button>
          </div>
          {/* Sprint 2026-05-06 (#post-S3P2en, second pass): 첨부 이미지 #2
              (Mahowald 2023 Counting/Article swapping/Shift cipher/Linear function)
              형식 — task box 안에 Input → Correct → per-LLM ✓/✗ 응답 vertical stack.
              명시 라벨 (Input / Correct / Forbid) + 응답 안 keyword highlight. */}
          {selectedGolden ? (
            <>
              {/* Task header — 첨부 이미지의 "Counting" / "Article swapping" 박스 헤더 */}
              <div className="mb-3 rounded-t border-b-2 border-slate-700 bg-slate-800 px-4 py-2 font-semibold text-white">
                {SCENARIO_LABELS[selectedGolden.scenario] ?? selectedGolden.scenario}
                <span className="ml-3 text-xs font-normal text-slate-300">
                  · {PERSONA_LABELS[selectedGolden.persona] ?? selectedGolden.persona}
                  · lang={selectedGolden.lang}
                  · difficulty={selectedGolden.difficulty}
                  {selectedGolden.source ? ` · src=${selectedGolden.source}` : ""}
                </span>
              </div>

              {/* Input — 첨부 이미지의 "Input 1: ..." */}
              <div className="mb-2 grid grid-cols-[80px_1fr] gap-2 text-sm">
                <strong className="text-right text-slate-700">Input:</strong>
                <p className="whitespace-pre-wrap text-slate-900">
                  {selectedGolden.prompt}
                </p>
              </div>

              {/* Correct — 첨부 이미지의 "Correct: 30" */}
              <div className="mb-2 grid grid-cols-[80px_1fr] gap-2 text-sm">
                <strong className="text-right text-emerald-700">Correct:</strong>
                <p className="text-slate-900">
                  <span className="text-[11px] text-slate-500">must_contain →</span>{" "}
                  {(selectedGolden.must_contain ?? []).length === 0 ? (
                    <span className="text-slate-500">(none)</span>
                  ) : (
                    (selectedGolden.must_contain ?? []).map((k, i) => (
                      <span key={k}>
                        {i > 0 ? ", " : ""}
                        <code className="rounded bg-emerald-100 px-1 text-emerald-800">{k}</code>
                      </span>
                    ))
                  )}
                </p>
              </div>

              {/* Forbid — must_avoid (없으면 hide) */}
              {(selectedGolden.must_avoid ?? []).length > 0 && (
                <div className="mb-3 grid grid-cols-[80px_1fr] gap-2 text-sm">
                  <strong className="text-right text-rose-700">Forbid:</strong>
                  <p className="text-slate-900">
                    <span className="text-[11px] text-slate-500">must_avoid →</span>{" "}
                    {(selectedGolden.must_avoid ?? []).map((k, i) => (
                      <span key={k}>
                        {i > 0 ? ", " : ""}
                        <code className="rounded bg-rose-100 px-1 text-rose-800">{k}</code>
                      </span>
                    ))}
                  </p>
                </div>
              )}

              {/* Divider */}
              <hr className="my-3 border-slate-300" />

              {/* Per-LLM verdict + response (vertical stack — 첨부 이미지 #2 형식) */}
              <p className="mb-2 text-[11px] text-slate-500">
                각 LLM 의 응답 + verdict (✓ pass / ✗ fail). 응답 안{" "}
                <span className="rounded bg-emerald-200 px-1">초록 highlight</span>{" "}
                = must_contain 매칭,{" "}
                <span className="rounded bg-rose-200 px-1">빨강 highlight</span>{" "}
                = must_avoid 위반.
              </p>
              <div className="space-y-2">
                {selectedAnswers.map(([backendId, ans]) => {
                  const missing = ans.missing_must_contain ?? [];
                  const hitAvoid = ans.hit_must_avoid ?? [];
                  const hedge = ans.hedge_tokens_found ?? [];
                  const isPass =
                    missing.length === 0 &&
                    hitAvoid.length === 0 &&
                    !!ans.response_text &&
                    !ans.error;
                  const shortBackend = backendId.replace(/^(api|cli|ollama):/, "");
                  return (
                    <div
                      key={backendId}
                      className={`grid grid-cols-[24px_1fr] gap-2 rounded border-l-4 p-2 text-sm ${
                        isPass
                          ? "border-emerald-500 bg-emerald-50/30"
                          : "border-rose-500 bg-rose-50/30"
                      }`}
                    >
                      <div
                        className={`flex items-start justify-center pt-1 text-base font-bold ${
                          isPass ? "text-emerald-600" : "text-rose-600"
                        }`}
                        aria-label={isPass ? "pass" : "fail"}
                      >
                        {isPass ? "✓" : "✗"}
                      </div>
                      <div>
                        <div className="mb-1 flex items-center justify-between">
                          <strong className="font-mono text-xs text-slate-700">
                            {shortBackend}
                          </strong>
                          <span className="text-[11px] text-slate-500">
                            total: {ans.total.toFixed(3)} · {ans.latency_ms.toFixed(0)}ms
                          </span>
                        </div>
                        {ans.response_text ? (
                          <p
                            className="whitespace-pre-wrap text-xs text-slate-800 max-h-48 overflow-y-auto"
                            dangerouslySetInnerHTML={{
                              __html: highlightKeywordsHtml(
                                ans.response_text,
                                selectedGolden?.must_contain ?? [],
                                selectedGolden?.must_avoid ?? [],
                              ),
                            }}
                          />
                        ) : (
                          <p className="text-xs text-rose-700">
                            ⚠ 응답 없음 — error: {ans.error || "(empty)"}
                          </p>
                        )}
                        {(missing.length > 0 || hitAvoid.length > 0 || hedge.length > 0) && (
                          <div className="mt-1 space-y-0.5 border-t border-slate-200 pt-1 text-[11px]">
                            {missing.length > 0 && (
                              <div>
                                <span className="font-semibold text-rose-700">missing:</span>{" "}
                                {missing.map((k) => (
                                  <code key={k} className="mx-0.5 bg-rose-100 px-1 text-rose-800">
                                    {k}
                                  </code>
                                ))}
                              </div>
                            )}
                            {hitAvoid.length > 0 && (
                              <div>
                                <span className="font-semibold text-rose-700">violated:</span>{" "}
                                {hitAvoid.map((k) => (
                                  <code key={k} className="mx-0.5 bg-rose-200 px-1 text-rose-900">
                                    {k}
                                  </code>
                                ))}
                              </div>
                            )}
                            {hedge.length > 0 && (
                              <div>
                                <span className="font-semibold text-amber-700">hedge:</span>{" "}
                                {hedge.map((k) => (
                                  <code key={k} className="mx-0.5 bg-amber-100 px-1 text-amber-800">
                                    {k}
                                  </code>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <p className="mb-3 text-xs text-slate-500">
              Golden-set 메타데이터 미발견 — report.json 의 <code>golden_set</code> field
              누락. <code>python -m simulation.llm_compare.runner</code> 재실행으로 schema 갱신.
            </p>
          )}
        </section>
        </div>
      )}
    </div>
  );
}
