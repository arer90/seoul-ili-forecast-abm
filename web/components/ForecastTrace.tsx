/**
 * ForecastTrace — compact timeline of MCP tool calls made during the
 * current assistant turn. The validity badge tells the audience
 * whether the reply held up; the trace tells them *how* it was built.
 *
 * Each entry shows tool name, args preview, elapsed ms, and a tiny
 * traffic-light for ok / error.
 */
"use client";

import type { ToolCall, ToolResult } from "@/lib/providers/types";

export interface ForecastTraceEntry {
  call: ToolCall;
  result?: ToolResult;
  /** ms since turn start when the call *started*. */
  startedAt?: number;
  /** ms elapsed for the round-trip. */
  elapsedMs?: number;
}

export interface ForecastTraceProps {
  entries: ForecastTraceEntry[];
}

function argsPreview(args: Record<string, unknown>): string {
  const keys = Object.keys(args ?? {});
  if (!keys.length) return "{}";
  const picks = keys.slice(0, 3).map((k) => {
    const v = (args as Record<string, unknown>)[k];
    let s: string;
    if (typeof v === "string") s = v.length > 16 ? v.slice(0, 16) + "…" : v;
    else if (Array.isArray(v)) s = `[${v.length}]`;
    else if (v && typeof v === "object") s = "{…}";
    else s = String(v);
    return `${k}=${s}`;
  });
  const extra = keys.length > 3 ? ` +${keys.length - 3}` : "";
  return `{ ${picks.join(", ")}${extra} }`;
}

export function ForecastTrace({ entries }: ForecastTraceProps) {
  if (!entries.length) return null;
  return (
    <ol className="flex flex-col gap-1 rounded-md border border-slate-800 bg-slate-950/60 p-2 text-[11px] font-mono">
      {entries.map((e, i) => {
        const isErr = e.result?.isError === true;
        const dotCx = !e.result
          ? "bg-slate-500 animate-pulse"
          : isErr
            ? "bg-red-500"
            : "bg-green-500";
        return (
          <li
            key={e.call.id ?? i}
            className="flex items-start gap-2 leading-snug"
          >
            <span
              aria-hidden
              className={[
                "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full",
                dotCx,
              ].join(" ")}
            />
            <span className="min-w-0 flex-1">
              <span className="text-sky-300">{e.call.name}</span>
              <span className="text-slate-400">
                {" "}
                {argsPreview(e.call.arguments)}
              </span>
              {e.elapsedMs != null ? (
                <span className="ml-1 text-slate-500">
                  ({e.elapsedMs.toFixed(0)} ms)
                </span>
              ) : null}
              {isErr ? (
                <span className="ml-1 text-red-300">[error]</span>
              ) : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
