/**
 * UserBadge — shown in the sidebar footer and (on mobile) in the
 * drawer header. Two states:
 *
 *   - anonymous: circle + first 8 chars of ``fd_uid`` ("a4fc0b71…")
 *                clicking opens a label modal
 *   - labelled : circle + ``display_name`` (falls back to email local-part)
 *                clicking opens the same modal for editing
 *
 * This is also where we surface the privacy consent moment — the
 * modal explains that adding a name/email attaches to the history
 * stored in our DB. Closing the modal without saving keeps the user
 * anonymous.
 */
"use client";

import { useEffect, useState } from "react";

import { useSessionStore } from "@/lib/use-session-store";

import { Button } from "./ui/button";
import { Input } from "./ui/input";

function hashColor(uid: string): string {
  // Pleasant-ish distinct colours derived from the uid prefix.
  // Not crypto — just avoids everyone having the same grey badge.
  let h = 0;
  for (let i = 0; i < Math.min(uid.length, 12); i++) {
    h = (h * 31 + uid.charCodeAt(i)) >>> 0;
  }
  const hue = h % 360;
  return `hsl(${hue} 55% 45%)`;
}

function shortLabel(uid: string, displayName: string | null, email: string | null): string {
  if (displayName && displayName.trim().length > 0) return displayName.trim();
  if (email && email.includes("@")) return email.split("@")[0];
  return uid.slice(0, 8) + "…";
}

export function UserBadge() {
  const store = useSessionStore();
  const user = store.user;
  const [open, setOpen] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [emailDraft, setEmailDraft] = useState("");

  useEffect(() => {
    setNameDraft(user?.display_name ?? "");
    setEmailDraft(user?.label_email ?? "");
  }, [user?.id, user?.display_name, user?.label_email]);

  if (!user) {
    return (
      <span className="text-[11px] text-slate-600">Loading…</span>
    );
  }

  const label = shortLabel(user.id, user.display_name, user.label_email);
  const color = hashColor(user.id);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 rounded-md px-1 py-0.5 text-[11px] text-slate-300 hover:bg-slate-800"
        title={`fd_uid: ${user.id}`}
      >
        <span
          aria-hidden
          className="inline-block h-5 w-5 rounded-full text-center text-[10px] leading-5 text-white"
          style={{ backgroundColor: color }}
        >
          {label[0]?.toUpperCase() ?? "?"}
        </span>
        <span className="max-w-[120px] truncate">{label}</span>
      </button>

      {open ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="user-badge-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) setOpen(false);
          }}
        >
          <div className="w-full max-w-md rounded-md border border-slate-700 bg-slate-900 p-4 text-sm text-slate-100 shadow-xl">
            <h2 id="user-badge-title" className="mb-1 text-base font-semibold">
              Your chat identity
            </h2>
            <p className="mb-3 text-xs text-slate-400">
              We keep your history under an opaque id (
              <code className="rounded bg-slate-800 px-1 text-[11px]">
                {user.id.slice(0, 8)}…
              </code>
              ) so two tabs on this browser share the same chats. Adding a
              name or email is optional and only stored to help you spot your
              own sessions on shared devices.
            </p>
            <label className="mb-2 block">
              <span className="mb-1 block text-xs text-slate-400">
                Display name
              </span>
              <Input
                value={nameDraft}
                onChange={(e) => setNameDraft(e.target.value)}
                placeholder="(optional)"
                maxLength={60}
                aria-label="Display name"
              />
            </label>
            <label className="mb-3 block">
              <span className="mb-1 block text-xs text-slate-400">
                Contact email (optional, never emailed)
              </span>
              <Input
                type="email"
                value={emailDraft}
                onChange={(e) => setEmailDraft(e.target.value)}
                placeholder="(optional)"
                maxLength={120}
                aria-label="Contact email"
              />
            </label>
            <div className="flex items-center justify-between">
              <Button
                size="sm"
                variant="ghost"
                onClick={async () => {
                  await store.updateMyLabel({
                    display_name: null,
                    label_email: null,
                  });
                  setOpen(false);
                }}
                title="Remove display name and email"
              >
                Go anonymous
              </Button>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setOpen(false)}>
                  Cancel
                </Button>
                <Button
                  size="sm"
                  onClick={async () => {
                    await store.updateMyLabel({
                      display_name: nameDraft.trim() || null,
                      label_email: emailDraft.trim() || null,
                    });
                    setOpen(false);
                  }}
                >
                  Save
                </Button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
