/**
 * SessionHeader — sits above ChatPanel's message list.
 *
 * Shows the active session's title (inline-editable), category chip,
 * pin / archive toggles, and a meatball for delete + duplicate. Also
 * renders the "New chat" pseudo-state when ``activeSessionId`` is
 * null — clicking the title in that state creates the session.
 *
 * State handling lives in ``use-session-store``; this component is
 * purely presentational and dispatches via the context.
 */
"use client";

import { useEffect, useState } from "react";

import { useSessionStore } from "@/lib/use-session-store";

import { Button } from "./ui/button";

const CATEGORY_SEEDS = [
  "baseline",
  "NPI",
  "vaccination",
  "antiviral",
  "validation",
  "ad hoc",
] as const;

export function SessionHeader() {
  const store = useSessionStore();
  const activeSession = store.sessions.find(
    (s) => s.id === store.activeSessionId,
  );

  // Local buffer so the user can edit the title without every keystroke
  // firing a PATCH. Commits on blur or Enter.
  const [titleDraft, setTitleDraft] = useState<string>(
    activeSession?.title ?? "",
  );
  const [editingTitle, setEditingTitle] = useState(false);
  useEffect(() => {
    setTitleDraft(activeSession?.title ?? "");
  }, [activeSession?.id, activeSession?.title]);

  const commitTitle = async () => {
    setEditingTitle(false);
    if (!activeSession) return;
    const next = titleDraft.trim();
    if (!next || next === activeSession.title) {
      setTitleDraft(activeSession.title);
      return;
    }
    await store.updateSession(activeSession.id, { title: next });
  };

  if (!activeSession) {
    return (
      <div className="flex items-center justify-between border-b border-slate-800 bg-slate-950/60 px-3 py-1.5 text-xs text-slate-400">
        <span>
          <span className="text-slate-300">New chat</span>
          <span className="ml-2 text-slate-600">
            (your next message will create a session)
          </span>
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 bg-slate-950/60 px-3 py-1.5 text-xs">
      {editingTitle ? (
        <input
          autoFocus
          value={titleDraft}
          onChange={(e) => setTitleDraft(e.target.value)}
          onBlur={() => void commitTitle()}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void commitTitle();
            } else if (e.key === "Escape") {
              setTitleDraft(activeSession.title);
              setEditingTitle(false);
            }
          }}
          className="min-w-0 flex-1 rounded border border-slate-700 bg-slate-900 px-2 py-0.5 text-sm text-slate-100 outline-none focus:border-sky-400"
          aria-label="Session title"
        />
      ) : (
        <button
          type="button"
          onClick={() => setEditingTitle(true)}
          className="min-w-0 flex-1 truncate text-left text-sm font-medium text-slate-100 hover:text-sky-300"
          title="Click to rename"
        >
          {activeSession.title}
        </button>
      )}

      <select
        value={activeSession.category ?? ""}
        onChange={(e) => {
          const v = e.target.value;
          void store.updateSession(activeSession.id, {
            category: v === "" ? null : v,
          });
        }}
        className="rounded border border-slate-700 bg-slate-900 px-1.5 py-0.5 text-[11px] text-slate-200 outline-none focus:border-sky-400"
        aria-label="Category"
      >
        <option value="">No category</option>
        {CATEGORY_SEEDS.map((cat) => (
          <option key={cat} value={cat}>
            {cat}
          </option>
        ))}
        {/* preserve any custom category the user set via export/reimport */}
        {activeSession.category &&
        !CATEGORY_SEEDS.includes(
          activeSession.category as (typeof CATEGORY_SEEDS)[number],
        ) ? (
          <option value={activeSession.category}>
            {activeSession.category}
          </option>
        ) : null}
      </select>

      <button
        type="button"
        onClick={() =>
          void store.updateSession(activeSession.id, {
            pinned: !activeSession.pinned,
          })
        }
        className={[
          "rounded border px-1.5 py-0.5 text-[11px]",
          activeSession.pinned
            ? "border-amber-500/70 bg-amber-500/20 text-amber-200"
            : "border-slate-700 text-slate-400 hover:bg-slate-800",
        ].join(" ")}
        title={activeSession.pinned ? "Unpin" : "Pin"}
      >
        ◆ {activeSession.pinned ? "Pinned" : "Pin"}
      </button>

      <Button
        size="sm"
        variant="outline"
        onClick={() =>
          void store.updateSession(activeSession.id, { archived: true })
        }
        title="Archive this session"
      >
        Archive
      </Button>

      <Button
        size="sm"
        variant="danger"
        onClick={() => {
          if (
            typeof window !== "undefined" &&
            !window.confirm(
              `Delete "${activeSession.title}"? This is permanent and cannot be undone.`,
            )
          ) {
            return;
          }
          void store.removeSession(activeSession.id);
        }}
        title="Delete permanently"
      >
        Delete
      </Button>
    </div>
  );
}
