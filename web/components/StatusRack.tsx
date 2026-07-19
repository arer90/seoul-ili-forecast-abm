/**
 * StatusRack — three compact status chips for the three moving parts
 * of the consultation stack:
 *
 *   [Agent]   the selected LLM provider + tool-call capability
 *   [Hermes]  the client-side orchestrator (solo / parallel / synthesis / relay)
 *   [MCP]     the Python → SQLite/DuckDB bridge at :8787
 *
 * Each chip has three colour states:
 *   · emerald  = fully ready
 *   · amber    = partially ready (e.g. provider up but model lacks tools)
 *   · red      = down / unreachable
 *
 * Clicking a chip's `?` icon explains what "green" means for that layer,
 * so the user understands WHY the chat might or might not call a tool.
 *
 * Why three chips instead of one status pill
 *   ARIA's health has three orthogonal failure modes and users
 *   reported the single "MCP" badge made it hard to diagnose whether
 *   the chat was "down" or just "using a tool-less model". Splitting
 *   the indicator tracks each layer independently.
 */
"use client";

import { useEffect, useState } from "react";

import { useT } from "@/lib/i18n";
import type { ProviderId } from "@/lib/providers/types";
import { HelpIcon } from "./ui/HelpIcon";

type Status = "ready" | "partial" | "down";

interface ToolListResp {
  tools?: Array<{ name?: string }>;
}

interface ProvidersResp {
  providers?: Array<{
    id: ProviderId;
    available: boolean;
    models?: string[];
  }>;
  environment?: {
    recommended?: ProviderId;
  };
}

/**
 * Best-effort heuristic for whether a given Ollama / API model supports
 * function calling. exaone3.5 does NOT; the Qwen-2.5 tool variants DO.
 * Cloud-provider defaults (claude / gemini / gpt) all support tools at
 * the latest generations we target.
 */
function modelSupportsTools(provider: ProviderId, modelId: string | null): boolean {
  if (!modelId) return provider !== "ollama"; // optimistic for cloud
  if (provider === "ollama") {
    const m = modelId.toLowerCase();
    // Known-bad: exaone family rejects tools/ parameter (HTTP 400).
    if (m.startsWith("exaone")) return false;
    // Known-ok: qwen2.5 (>=7b) supports tools in Ollama's chat API.
    if (m.startsWith("qwen2.5") || m.startsWith("qwen3")) return true;
    if (m.startsWith("mistral") || m.startsWith("llama3.1") || m.startsWith("llama3.2")) {
      return true;
    }
    return false;
  }
  return true;
}

export interface StatusRackProps {
  /** Current provider selected in the chat header — drives Agent chip. */
  selectedProvider?: ProviderId;
  /** Current model id for that provider. */
  selectedModel?: string | null;
  className?: string;
}

function ChipDot({ status }: { status: Status }) {
  const cls =
    status === "ready"
      ? "bg-emerald-400"
      : status === "partial"
        ? "bg-amber-400"
        : "bg-red-400";
  return (
    <span
      aria-hidden="true"
      className={["inline-block h-2 w-2 rounded-full", cls].join(" ")}
    />
  );
}

function Chip({
  dot,
  label,
  title,
  help,
}: {
  dot: Status;
  label: string;
  title: string;
  help: string;
}) {
  // Sprint 2026-05-06 (#5): HTML native title= 제거 — HelpIcon (click popover)
  // 와 충돌하여 사용자 critique "?아이콘 일부만 작동". 모든 chip 의 ? 아이콘
  // 동작이 HelpIcon 으로 통일됨 (hover desktop / click mobile).
  // 단 title prop 은 hint extra context 으로 HelpIcon content 안에 통합.
  const richHelp = title && title !== help ? `${help} (${title})` : help;
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900/70 px-1.5 py-0.5 text-[10.5px] font-medium leading-none text-slate-200">
      <ChipDot status={dot} />
      <span>{label}</span>
      <HelpIcon label={help} content={richHelp} side="bottom" />
    </span>
  );
}

export function StatusRack({
  selectedProvider,
  selectedModel,
  className = "",
}: StatusRackProps) {
  const { t } = useT();

  // MCP bridge state
  const [mcpStatus, setMcpStatus] = useState<Status>("partial");
  const [mcpCount, setMcpCount] = useState<number | null>(null);

  // Agent state — fetched from /api/providers, then narrowed by the
  // parent-supplied (provider, model) pair.
  const [availability, setAvailability] = useState<Partial<Record<ProviderId, boolean>>>({});

  // Hermes = client-side JS. If this component renders it's effectively
  // "up". We just expose the chip for symmetry + so users know to look
  // here when the bundle fails to load. The "down" path is reserved for
  // future server-side orchestration.
  const hermesStatus: Status = "ready";

  useEffect(() => {
    let cancelled = false;
    const probeMcp = async () => {
      try {
        const r = await fetch("/api/mcp/_list", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const body = (await r.json()) as ToolListResp;
        if (cancelled) return;
        const n = Array.isArray(body.tools) ? body.tools.length : 0;
        setMcpCount(n);
        setMcpStatus(n > 0 ? "ready" : "down");
      } catch {
        if (!cancelled) {
          setMcpStatus("down");
          setMcpCount(null);
        }
      }
    };
    const probeProviders = async () => {
      try {
        const r = await fetch("/api/providers", { cache: "no-store" });
        if (!r.ok) return;
        const body = (await r.json()) as ProvidersResp;
        if (cancelled) return;
        const next: Partial<Record<ProviderId, boolean>> = {};
        for (const p of body.providers ?? []) next[p.id] = p.available;
        setAvailability(next);
      } catch {
        /* availability is optional — leave empty */
      }
    };
    void probeMcp();
    void probeProviders();
    const id = window.setInterval(() => {
      void probeMcp();
      void probeProviders();
    }, 45_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  // Derive Agent status: provider available? selected model tool-capable?
  const providerUp =
    selectedProvider != null ? availability[selectedProvider] === true : false;
  const toolOk = providerUp
    ? modelSupportsTools(selectedProvider as ProviderId, selectedModel ?? null)
    : false;
  const agentStatus: Status = !providerUp ? "down" : toolOk ? "ready" : "partial";

  const agentLabel =
    agentStatus === "down"
      ? t("statusAgentDown")
      : agentStatus === "partial"
        ? `Agent · ${t("toolsDisabledNotice").split(".")[0]}`
        : t("statusAgentReady");
  const hermesLabel = t("statusHermesReady");
  const mcpLabel =
    mcpStatus === "ready"
      ? `${t("mcpStatusReady")}${mcpCount != null ? ` · ${mcpCount}` : ""}`
      : mcpStatus === "down"
        ? t("mcpStatusDown")
        : t("mcpStatusChecking");

  return (
    <div className={["flex items-center gap-1.5", className].join(" ")}>
      <Chip
        dot={agentStatus}
        label={agentLabel}
        title={`${selectedProvider ?? "?"} / ${selectedModel ?? "?"}`}
        help={t("helpAgentStatus")}
      />
      <Chip
        dot={hermesStatus}
        label={hermesLabel}
        title="Hermes orchestrator (client-side)"
        help={t("helpHermesStatus")}
      />
      <Chip
        dot={mcpStatus}
        label={mcpLabel}
        title={t("mcpStatusTooltip")}
        help={t("mcpStatusTooltip")}
      />
    </div>
  );
}
