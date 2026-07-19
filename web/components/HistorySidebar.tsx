/**
 * HistorySidebar — left rail with chat session list.
 *
 * Layout (top → bottom):
 *   · NewChatButton (always visible at top — the user complained the
 *     3-dot gating made it feel hidden)
 *   · Search input + archived toggle
 *   · Session list — each row shows title, (provider), mtime
 *     · Hover → pin / archive / delete icons
 *   · Footer: UserBadge + "Export" button
 *
 * Visibility
 *   The sidebar itself is toggled open/closed by AppShell (Claude-style
 *   persistent drawer) — this component doesn't know about that; it
 *   just renders whenever it's mounted.
 *
 * Category chips were removed 2026-04-22 at user request — the seed
 * list (baseline/NPI/antiviral/...) was noise for the demo audience.
 */
"use client";

import { useMemo, useState } from "react";

import { PROVIDER_LABELS } from "@/lib/constants";
import { useT } from "@/lib/i18n";
import type { ProviderId } from "@/lib/providers/types";
import { useSessionStore } from "@/lib/use-session-store";
import type { FdSession } from "@/lib/history-db";

import { NewChatButton } from "./NewChatButton";
import { UserBadge } from "./UserBadge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

function relTime(updatedAt: number): string {
  const now = Date.now() / 1000;
  const delta = Math.max(0, now - updatedAt);
  if (delta < 60) return "just now";
  if (delta < 3_600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86_400) return `${Math.floor(delta / 3_600)} h ago`;
  if (delta < 604_800) return `${Math.floor(delta / 86_400)} d ago`;
  const d = new Date(updatedAt * 1_000);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function providerShort(id?: string | null): string {
  if (!id) return "";
  const known = PROVIDER_LABELS as Record<string, string>;
  return known[id] ?? id;
}

interface SidebarRowProps {
  session: FdSession;
  active: boolean;
  onOpen: () => void;
  onPin: () => void;
  onArchive: () => void;
  onDelete: () => void;
}

function SidebarRow({
  session,
  active,
  onOpen,
  onPin,
  onArchive,
  onDelete,
}: SidebarRowProps) {
  const { t } = useT();
  const provider = providerShort(session.provider_id);
  return (
    <li
      className={[
        "group relative rounded-md border border-transparent px-2 py-1.5 text-xs",
        "hover:border-slate-700 hover:bg-slate-900/60",
        active ? "border-sky-500/60 bg-slate-900/80" : "",
      ].join(" ")}
    >
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full flex-col items-start gap-0.5 text-left"
        title={session.title}
      >
        <div className="flex w-full items-center gap-1">
          {session.pinned ? (
            <span className="text-[10px] text-amber-400" aria-label="pinned">
              ◆
            </span>
          ) : null}
          <span className="min-w-0 flex-1 truncate text-slate-100">
            {session.title || "(untitled)"}
          </span>
          <span className="shrink-0 text-[10px] text-slate-500">
            {relTime(session.updated_at)}
          </span>
        </div>
        {provider ? (
          <div className="flex items-center gap-1 text-[10px] text-slate-500">
            <span className="truncate">{provider}</span>
          </div>
        ) : null}
      </button>
      <div
        className="pointer-events-none absolute right-1 top-1 hidden gap-0.5 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:flex group-hover:opacity-100"
        role="toolbar"
      >
        <button
          type="button"
          onClick={onPin}
          title={session.pinned ? t("unpinTooltip") : t("pinTooltip")}
          className="rounded px-1 text-[11px] text-slate-400 hover:bg-slate-800 hover:text-amber-300"
        >
          ◆
        </button>
        <button
          type="button"
          onClick={onArchive}
          title={t("archiveTooltip")}
          className="rounded px-1 text-[11px] text-slate-400 hover:bg-slate-800 hover:text-slate-200"
        >
          ▣
        </button>
        <button
          type="button"
          onClick={onDelete}
          title={t("deleteTooltip")}
          className="rounded px-1 text-[11px] text-slate-400 hover:bg-slate-800 hover:text-red-400"
        >
          ✕
        </button>
      </div>
    </li>
  );
}

export function HistorySidebar() {
  const { t } = useT();
  const store = useSessionStore();
  const [query, setQuery] = useState("");
  const [showArchived, setShowArchived] = useState(false);

  const filteredSessions = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return store.sessions;
    return store.sessions.filter((s) =>
      s.title.toLowerCase().includes(q),
    );
  }, [store.sessions, query]);

  const toggleArchived = () => {
    const next = !showArchived;
    setShowArchived(next);
    void store.setFilter({ archived: next });
  };

  if (store.disabled) {
    // Even when Turso is down, New Chat should still work locally.
    // The empty composer is useful on its own; only the persistence
    // (history list, export) is gated.
    return (
      <aside className="flex h-full w-full flex-col border-r border-slate-800 bg-slate-950/60">
        <div className="border-b border-slate-800 p-2">
          <NewChatButton />
        </div>
        <div className="flex flex-1 flex-col items-center justify-center gap-2 p-4 text-center text-xs text-slate-500">
          <span>{t("historyUnavailable")}</span>
          <span className="text-slate-600">{t("historyUnavailableHint")}</span>
        </div>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-full flex-col border-r border-slate-800 bg-slate-950/60">
      <div className="flex flex-col gap-2 border-b border-slate-800 p-2">
        <span className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
          {t("historyLabel")}
        </span>
        <NewChatButton />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={t("searchPlaceholder")}
          className="h-8 text-xs"
          aria-label={t("searchPlaceholder")}
        />
        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              className="accent-sky-500"
              checked={showArchived}
              onChange={toggleArchived}
            />
            {t("showArchived")}
          </label>
          <button
            type="button"
            onClick={() => void store.refresh()}
            className="text-slate-500 hover:text-slate-200"
            title={t("refresh")}
          >
            ↻
          </button>
        </div>
      </div>

      <ul className="flex-1 space-y-0.5 overflow-y-auto p-2">
        {store.booting ? (
          <li className="p-2 text-xs text-slate-500">{t("loadingEllipsis")}</li>
        ) : filteredSessions.length === 0 ? (
          <li className="p-2 text-xs text-slate-500">
            {query ? t("noMatches") : t("noSessions")}
          </li>
        ) : (
          filteredSessions.map((s) => (
            <SidebarRow
              key={s.id}
              session={s}
              active={s.id === store.activeSessionId}
              onOpen={() => void store.openSession(s.id)}
              onPin={() =>
                void store.updateSession(s.id, { pinned: !s.pinned })
              }
              onArchive={() => void store.updateSession(s.id, { archived: true })}
              onDelete={() => {
                if (
                  typeof window !== "undefined" &&
                  !window.confirm(t("confirmDelete"))
                ) {
                  return;
                }
                void store.removeSession(s.id);
              }}
            />
          ))
        )}
      </ul>

      <div className="flex items-center justify-between border-t border-slate-800 p-2">
        <UserBadge />
        <Button
          size="sm"
          variant="ghost"
          onClick={() => void store.exportAll()}
          title={t("exportHistory")}
        >
          {t("exportHistory")}
        </Button>
      </div>
      {store.error ? (
        <div className="border-t border-red-900/50 bg-red-950/40 p-2 text-[11px] text-red-300">
          {store.error}
        </div>
      ) : null}
    </aside>
  );
}

export function _providerLabelStatic(id: ProviderId): string {
  return providerShort(id);
}
