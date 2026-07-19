/**
 * RefreshDataButton — Sprint 2026-05-06 Phase C.1.
 *
 * 사용자 critique: "14일 업데이트 lag, agent 가 바로 체크/업데이트 안 해주는
 * 거야?" — 단순 trigger 버튼. POST /api/collect/run 호출 → Python collector
 * subprocess → KDCA / WHO FluNet 새 데이터 fetch.
 *
 * KDCA reporting 의 본질적 lag (~14d) 는 fix 못 함 — 단 우리 시스템이 KDCA
 * 새 발표 즉시 반영 가능.
 */
"use client";
import { useState } from "react";

export function RefreshDataButton({
  groups = "weekly_disease,who_flunet",
  className = "",
}: {
  groups?: string;
  className?: string;
}) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const onClick = async () => {
    setBusy(true);
    setMsg(null);
    setErr(null);
    try {
      const r = await fetch("/api/collect/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ groups }),
      });
      const body = await r.json();
      if (!r.ok || !body.ok) {
        setErr(`fail (${body.code ?? r.status}): ${(body.stderr || body.error || "").slice(-200)}`);
      } else {
        setMsg(`✓ ${(body.elapsed_ms / 1000).toFixed(1)}s · ${body.groups}`);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      // Auto-clear status after 8s
      setTimeout(() => {
        setMsg(null);
        setErr(null);
      }, 8000);
    }
  };

  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] ${className}`}>
      <button
        type="button"
        onClick={onClick}
        disabled={busy}
        title={`Trigger collector: ${groups}`}
        className="inline-flex items-center gap-1 rounded border border-slate-700 bg-slate-800 px-2 py-1 font-medium text-slate-200 hover:bg-slate-700 disabled:opacity-50"
      >
        <span className={busy ? "animate-spin" : ""}>🔄</span>
        <span>{busy ? "수집 중…" : "데이터 새로고침"}</span>
      </button>
      {msg && <span className="text-emerald-400">{msg}</span>}
      {err && <span className="text-rose-400" title={err}>✗ {err.slice(0, 60)}</span>}
    </span>
  );
}
