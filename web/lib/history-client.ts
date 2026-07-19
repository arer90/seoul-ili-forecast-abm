/**
 * Client-side HTTP wrappers for ``/api/history/*``.
 *
 * Purely functional — no React, no state. Parses JSON, throws on
 * non-2xx, returns typed shapes. The React context in
 * ``use-session-store.tsx`` is the only caller today, but keeping the
 * fetch layer separate makes it easy to:
 *   - unit-test without a React tree
 *   - reuse from the /api/history/export download button, which needs
 *     a Blob instead of JSON
 *   - swap for a mocked client in Storybook
 *
 * All routes require the ``fd_uid`` cookie; the browser sends it
 * automatically because it was set ``SameSite=Lax`` by middleware.
 */
import type {
  CategoryCount,
  FdMessage,
  FdSession,
  FdUser,
} from "./history-db";

// ── Shared fetch error ───────────────────────────────────────────────

export class HistoryApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly path: string,
  ) {
    super(message);
    this.name = "HistoryApiError";
  }
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
    // history endpoints are all same-origin; the cookie is auto-sent
    credentials: "same-origin",
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body?.error) msg = body.error;
    } catch {
      // body wasn't JSON — keep the statusText fallback
    }
    throw new HistoryApiError(msg, res.status, path);
  }
  if (res.status === 204) return undefined as unknown as T;
  return (await res.json()) as T;
}

// ── Me ───────────────────────────────────────────────────────────────

export async function fetchMe(): Promise<FdUser> {
  const { user } = await fetchJson<{ user: FdUser }>("/api/history/me");
  return user;
}

export async function patchMe(patch: {
  display_name?: string | null;
  label_email?: string | null;
}): Promise<FdUser> {
  const { user } = await fetchJson<{ user: FdUser }>("/api/history/me", {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
  return user;
}

// ── Sessions ─────────────────────────────────────────────────────────

export interface ListSessionsQuery {
  /** ``null`` → "No category" bucket; ``string`` → exact match; ``undefined`` → all */
  category?: string | null;
  archived?: boolean;
  limit?: number;
}

export async function fetchSessions(
  q: ListSessionsQuery = {},
): Promise<FdSession[]> {
  const params = new URLSearchParams();
  if (q.category !== undefined) {
    params.set("category", q.category === null ? "none" : q.category);
  }
  if (q.archived === true) params.set("archived", "1");
  if (q.limit && Number.isFinite(q.limit)) params.set("limit", String(q.limit));
  const qs = params.toString();
  const path = qs ? `/api/history/sessions?${qs}` : `/api/history/sessions`;
  const { sessions } = await fetchJson<{ sessions: FdSession[] }>(path);
  return sessions;
}

export async function createSession(init: {
  title?: string;
  category?: string | null;
  provider_id?: string | null;
  mode?: string | null;
}): Promise<FdSession> {
  const { session } = await fetchJson<{ session: FdSession }>(
    "/api/history/sessions",
    {
      method: "POST",
      body: JSON.stringify(init ?? {}),
    },
  );
  return session;
}

export async function fetchSession(id: string): Promise<{
  session: FdSession;
  messages: FdMessage[];
}> {
  return fetchJson<{ session: FdSession; messages: FdMessage[] }>(
    `/api/history/sessions/${encodeURIComponent(id)}`,
  );
}

export async function patchSession(
  id: string,
  patch: {
    title?: string;
    category?: string | null;
    provider_id?: string | null;
    mode?: string | null;
    pinned?: boolean;
    archived?: boolean;
  },
): Promise<FdSession> {
  const { session } = await fetchJson<{ session: FdSession }>(
    `/api/history/sessions/${encodeURIComponent(id)}`,
    {
      method: "PATCH",
      body: JSON.stringify(patch),
    },
  );
  return session;
}

export async function deleteSession(id: string): Promise<void> {
  await fetchJson<void>(`/api/history/sessions/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

// ── Messages ─────────────────────────────────────────────────────────

export interface AppendMessageArgs {
  role: "user" | "assistant" | "tool";
  content: string;
  provider_id?: string | null;
  tool_calls?: unknown;
  validity?: unknown;
  tokens_in?: number | null;
  tokens_out?: number | null;
  turn_idx?: number;
}

export async function appendMessage(
  sessionId: string,
  msg: AppendMessageArgs,
): Promise<{ message: FdMessage; session?: FdSession }> {
  return fetchJson<{ message: FdMessage; session?: FdSession }>(
    `/api/history/sessions/${encodeURIComponent(sessionId)}/messages`,
    {
      method: "POST",
      body: JSON.stringify(msg),
    },
  );
}

// ── Categories ───────────────────────────────────────────────────────

export async function fetchCategories(): Promise<CategoryCount[]> {
  const { categories } = await fetchJson<{ categories: CategoryCount[] }>(
    "/api/history/categories",
  );
  return categories;
}

// ── Export ───────────────────────────────────────────────────────────

/** Returns a Blob so the caller can trigger a download without re-fetching. */
export async function exportHistory(): Promise<Blob> {
  const res = await fetch("/api/history/export", {
    method: "POST",
    credentials: "same-origin",
  });
  if (!res.ok) {
    throw new HistoryApiError(
      `${res.status} ${res.statusText}`,
      res.status,
      "/api/history/export",
    );
  }
  return res.blob();
}

/** Fire a browser download of the user's history without touching the DOM tree. */
export async function downloadHistory(filename?: string): Promise<void> {
  const blob = await exportHistory();
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download =
      filename ??
      `frame-d-history-${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    // Free the blob URL on the next tick so the click has time to take.
    setTimeout(() => URL.revokeObjectURL(url), 2_000);
  }
}
