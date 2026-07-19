/**
 * McpStatusBadge — "is the MCP bridge up?" indicator.
 *
 * Why this matters for the demo
 *   Even with a healthy bridge, the chat can feel "dead" if the selected
 *   LLM doesn't support function-calling. Showing the bridge status
 *   separately makes the mental model explicit:
 *
 *     bridge OK + tool-capable model  → chat grounds itself in the DB
 *     bridge OK + tool-limited model  → chat answers from prior only
 *                                        (ChatPanel renders a banner)
 *     bridge DOWN                     → no tool would work regardless
 *
 * Implementation
 *   · Polls GET /api/mcp/_list every 45 s (soft TTL matching Next.js
 *     dev recompile cadence).
 *   · ``status = "ready"`` when the payload contains a non-empty tools
 *     array; ``"down"`` on network/JSON error; ``"checking"`` during the
 *     initial in-flight fetch.
 *   · Click reveals the same tooltip content as the help icon, so demos
 *     can get a one-sentence explanation without digging into docs.
 */
"use client";

import { useEffect, useState } from "react";

import { useT } from "@/lib/i18n";
import { HelpIcon } from "./ui/HelpIcon";

type Status = "checking" | "ready" | "down";

interface ToolListResp {
  tools?: Array<{ name?: string }>;
}

export function McpStatusBadge({ className = "" }: { className?: string }) {
  const { t } = useT();
  const [status, setStatus] = useState<Status>("checking");
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const probe = async () => {
      try {
        const r = await fetch("/api/mcp/_list", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const body = (await r.json()) as ToolListResp;
        if (cancelled) return;
        const n = Array.isArray(body.tools) ? body.tools.length : 0;
        setCount(n);
        setStatus(n > 0 ? "ready" : "down");
      } catch {
        if (!cancelled) {
          setStatus("down");
          setCount(null);
        }
      }
    };
    void probe();
    const id = window.setInterval(probe, 45_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const dot =
    status === "ready"
      ? "bg-emerald-400"
      : status === "down"
        ? "bg-red-400"
        : "bg-amber-400";
  const text =
    status === "ready"
      ? `${t("mcpStatusReady")}${count != null ? ` · ${count}` : ""}`
      : status === "down"
        ? t("mcpStatusDown")
        : t("mcpStatusChecking");

  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900/70 px-2 py-0.5",
        "text-[11px] font-medium leading-none text-slate-200",
        className,
      ].join(" ")}
      title={t("mcpStatusTooltip")}
    >
      <span
        aria-hidden="true"
        className={["inline-block h-2 w-2 rounded-full", dot].join(" ")}
      />
      <span>{text}</span>
      <HelpIcon
        label={t("mcpStatusTooltip")}
        content={t("mcpStatusTooltip")}
        side="bottom"
      />
    </span>
  );
}
