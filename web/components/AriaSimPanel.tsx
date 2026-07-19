"use client";
/**
 * AriaSimPanel — ARIA advisory bound to the ABM simulation + map.
 *
 * Bidirectional (사용자 채택 2026-06-06):
 *   map → ARIA: a rule-based briefing is derived from the current scenario/day
 *               trajectory (top-infected gu, attack rate, intervention effect).
 *   ARIA → map: gu names in the briefing/answer are clickable chips that
 *               highlight the gu on the map; free-form questions go to the LLM
 *               (/api/chat-cli, best-effort — degrades to the rule-based briefing if
 *               the local CLI is unavailable) and any gu it names is surfaced as a chip.
 */
import { useMemo, useRef, useState } from "react";
import type { AbmScenarios } from "./AbmMap";

function topInfected(data: AbmScenarios, scenario: string, day: number, k = 3) {
  const sc = data.scenarios[scenario];
  if (!sc?.I_frac[day]) return [];
  return sc.I_frac[day]
    .map((frac, i) => ({ gu: data.gu_names[i], frac }))
    .sort((a, b) => b.frac - a.frac)
    .slice(0, k);
}

/** Rule-based ARIA briefing from the trajectory (no LLM needed). */
function briefing(data: AbmScenarios, scenario: string, day: number): {
  text: string;
  gus: string[];
} {
  const sc = data.scenarios[scenario];
  if (!sc) return { text: "시나리오 데이터 없음.", gus: [] };
  const top = topInfected(data, scenario, day);
  const base = data.scenarios["baseline"];
  const reduction =
    base && scenario !== "baseline"
      ? Math.round((1 - sc.city_attack_pct / Math.max(base.city_attack_pct, 1e-6)) * 100)
      : null;
  const lines = [
    `시나리오 「${sc.label}」 · ${day}일차 / 피크 ${sc.peak_day}일.`,
    `도시 누적 발병률 ${sc.city_attack_pct}% · 누적 사망 ${sc.deaths.toLocaleString()}명.`,
    top.length
      ? `현재 감염 상위: ${top.map((t) => `${t.gu}(${(t.frac * 100).toFixed(2)}%)`).join(", ")}.`
      : "",
    reduction != null
      ? `기준 대비 발병 ${reduction >= 0 ? reduction : 0}% ${reduction >= 0 ? "감소" : "증가"} — 개입 효과${reduction > 30 ? " 큼" : reduction > 5 ? " 보통" : " 미미"}.`
      : "개입 없는 기준 시나리오.",
    sc.legal_basis ? `법적 근거: ${sc.legal_basis} (감염병예방법).` : "",
    sc.epi_validity_ok ? "역학 타당성 게이트 통과." : "⚠ 역학 타당성 게이트 실패 — 결과 주의.",
  ].filter(Boolean);
  return { text: lines.join(" "), gus: top.map((t) => t.gu) };
}

type Msg = { role: "user" | "aria"; text: string; gus: string[] };

export function AriaSimPanel({
  data,
  scenario,
  day,
  onHighlightGu,
}: {
  data: AbmScenarios | null;
  scenario: string;
  day: number;
  onHighlightGu: (gu: string | null) => void;
}) {
  const sessionIdRef = useRef<string>(crypto.randomUUID());
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);

  const brief = useMemo(
    () => (data ? briefing(data, scenario, day) : { text: "로딩 중…", gus: [] }),
    [data, scenario, day],
  );

  const scanGus = (text: string): string[] =>
    (data?.gu_names ?? []).filter((g) => text.includes(g));

  async function ask(question: string) {
    if (!question.trim() || !data) return;
    setInput("");
    setHistory((h) => [...h, { role: "user", text: question, gus: scanGus(question) }]);
    setBusy(true);
    // Inject the current sim state so ARIA "reads" the map (map → ARIA).
    const context = briefing(data, scenario, day).text;
    const sys =
      "당신은 서울 인플루엔자 ABM 시뮬레이션 자문 ARIA입니다. " +
      "현재 시뮬레이션 상태를 근거로, 25개 자치구 단위 개입(거리두기·학교폐쇄·백신)을 간결히 조언하세요.";
    try {
      const res = await fetch("/api/chat-cli", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages: [
            { role: "system", content: sys },
            { role: "user", content: question },
          ],
          sessionId: sessionIdRef.current,
          context: {
            simDay: day,
            scenario,
            scenarioLabel: data.scenarios[scenario]?.label ?? scenario,
            topInfectedGus: brief.gus,
            briefing: context,
          },
        }),
      });
      let answer = "";
      if (res.ok && res.body) {
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          for (const line of buf.split("\n")) {
            const m = line.match(/^data:\s*(.+)$/);
            if (!m) continue;
            try {
              const ev = JSON.parse(m[1]);
              const piece =
                ev.delta ?? ev.content ?? ev.text ?? ev.token ?? ev.message?.content;
              if (typeof piece === "string") answer += piece;
            } catch {
              /* keepalive / non-JSON event */
            }
          }
          buf = buf.slice(buf.lastIndexOf("\n") + 1);
        }
      }
      const text = answer.trim() || `(LLM 응답 없음 — 규칙기반 브리핑) ${context}`;
      const gus = scanGus(text);
      setHistory((h) => [...h, { role: "aria", text, gus }]);
      if (gus[0]) onHighlightGu(gus[0]);
    } catch {
      const text = `(LLM 미연결 — 규칙기반 브리핑) ${context}`;
      setHistory((h) => [...h, { role: "aria", text, gus: brief.gus }]);
      if (brief.gus[0]) onHighlightGu(brief.gus[0]);
    } finally {
      setBusy(false);
    }
  }

  const Chip = ({ gu }: { gu: string }) => (
    <button
      type="button"
      onClick={() => onHighlightGu(gu)}
      className="rounded bg-sky-700/60 px-1.5 py-0.5 text-[11px] text-sky-100 hover:bg-sky-600"
    >
      {gu}
    </button>
  );

  return (
    <div className="flex h-full flex-col gap-2 rounded-lg bg-slate-900/90 p-3 text-sm text-slate-200">
      <div className="flex items-center justify-between">
        <span className="font-semibold">ARIA · 시뮬레이션 자문</span>
        <span className="text-[10px] text-slate-400">지도↔ARIA 양방향</span>
      </div>

      {/* map → ARIA: live rule-based briefing of the current sim state */}
      <div className="rounded bg-slate-800/70 p-2 text-[12px] leading-relaxed">
        <div className="mb-1 text-[10px] font-semibold text-slate-400">현재 상황 브리핑</div>
        {brief.text}
        {brief.gus.length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {brief.gus.map((g) => (
              <Chip key={g} gu={g} />
            ))}
          </div>
        )}
      </div>

      {/* conversation */}
      <div className="flex-1 space-y-2 overflow-y-auto">
        {history.map((m, i) => (
          <div
            key={i}
            className={`rounded p-2 text-[12px] ${
              m.role === "user" ? "bg-sky-900/40" : "bg-slate-800/70"
            }`}
          >
            <div className="mb-0.5 text-[10px] font-semibold text-slate-400">
              {m.role === "user" ? "나" : "ARIA"}
            </div>
            <div className="whitespace-pre-wrap leading-relaxed">{m.text}</div>
            {m.gus.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {m.gus.map((g) => (
                  <Chip key={g} gu={g} />
                ))}
              </div>
            )}
          </div>
        ))}
        {busy && <div className="text-[11px] text-slate-400">ARIA 분석 중…</div>}
      </div>

      {/* quick + free-form ask */}
      <div className="flex gap-1">
        <button
          type="button"
          disabled={busy}
          onClick={() => ask("현재 상황에서 어떤 개입을 우선해야 할까요?")}
          className="rounded bg-slate-700 px-2 py-1 text-[11px] hover:bg-slate-600 disabled:opacity-50"
        >
          개입 추천
        </button>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask(input)}
          placeholder="ARIA에게 질문…"
          className="flex-1 rounded bg-slate-800 px-2 py-1 text-[12px] outline-none"
        />
        <button
          type="button"
          disabled={busy}
          onClick={() => ask(input)}
          className="rounded bg-sky-600 px-2 py-1 text-[11px] text-white hover:bg-sky-500 disabled:opacity-50"
        >
          전송
        </button>
      </div>
    </div>
  );
}
