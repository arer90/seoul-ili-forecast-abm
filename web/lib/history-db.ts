/**
 * ARIA history-db — CRUD over ``fd_users`` / ``fd_sessions`` /
 * ``fd_messages`` via the Turso libSQL edge client.
 *
 * Why live next to ``turso.ts`` and not inside ``app/api/history/``
 * -----------------------------------------------------------------
 *  - Multiple route handlers (list, create, patch, messages, export,
 *    categories, me) all need the same primitives. Centralising keeps
 *    the SQL in one auditable spot.
 *  - Edge runtime → no ``fs``; the canonical DDL in
 *    ``scripts/migrations/001_history_schema.sql`` is duplicated here
 *    so ``ensureSchema()`` can self-heal a fresh Turso DB without
 *    reading a file off disk. The migration file remains the manual
 *    ``turso db shell <  … .sql`` entry point for ops.
 *
 * Design rules
 * ------------
 *  - ``userId`` is the ``fd_uid`` cookie value, forwarded via the
 *    ``x-fd-uid`` header by ``middleware.ts``. No OAuth, no passwords —
 *    a ``display_name`` / ``label_email`` upgrade is opt-in.
 *  - Soft caps protect the DB without destroying history:
 *      · per-user sessions > 200 → oldest non-pinned non-archived
 *        sessions are auto-archived in batches of 20. They remain
 *        queryable with ``archived: true``.
 *      · per-session messages > 1000 → oldest 100 hard-deleted.
 *        Chat history that deep is almost never re-read, and leaving
 *        it bloats every sidebar payload.
 *  - Every mutation bumps ``updated_at`` on the owning session so the
 *    sidebar ordering never lags.
 *  - Ownership is enforced on every read/write by joining ``user_id``
 *    into the WHERE clause. There is no admin bypass path.
 */
import type { InStatement } from "@libsql/client/web";

import { turso } from "./turso";

// ── Domain types ─────────────────────────────────────────────────────

export interface FdUser {
  id: string;
  display_name: string | null;
  label_email: string | null;
  created_at: number;
  last_seen_at: number;
}

export interface FdSession {
  id: string;
  user_id: string;
  title: string;
  category: string | null;
  provider_id: string | null;
  mode: string | null;
  created_at: number;
  updated_at: number;
  /** 0 | 1 — kept as numeric to match SQLite storage. */
  pinned: number;
  /** 0 | 1 — soft-delete flag. */
  archived: number;
}

export type FdMessageRole = "user" | "assistant" | "tool";

export interface FdMessage {
  id: string;
  session_id: string;
  turn_idx: number;
  role: FdMessageRole;
  provider_id: string | null;
  content: string;
  /** JSON array string or NULL. Parse on read if needed. */
  tool_calls: string | null;
  /** JSON object string or NULL. Parse on read if needed. */
  validity: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  created_at: number;
}

// ── Schema self-heal ─────────────────────────────────────────────────
// Mirrors ``scripts/migrations/001_history_schema.sql`` — keep them in
// sync. Runtime uses this; ops uses the SQL file via ``turso db shell``.
const DDL: InStatement[] = [
  `CREATE TABLE IF NOT EXISTS fd_users (
    id            TEXT PRIMARY KEY,
    display_name  TEXT,
    label_email   TEXT,
    created_at    INTEGER NOT NULL,
    last_seen_at  INTEGER NOT NULL
  )`,
  `CREATE TABLE IF NOT EXISTS fd_sessions (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    title         TEXT NOT NULL,
    category      TEXT,
    provider_id   TEXT,
    mode          TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    pinned        INTEGER NOT NULL DEFAULT 0,
    archived      INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES fd_users(id) ON DELETE CASCADE
  )`,
  `CREATE INDEX IF NOT EXISTS idx_fd_sessions_user_updated
    ON fd_sessions(user_id, archived, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_fd_sessions_user_category
    ON fd_sessions(user_id, category, updated_at DESC)`,
  `CREATE TABLE IF NOT EXISTS fd_messages (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    turn_idx      INTEGER NOT NULL,
    role          TEXT NOT NULL,
    provider_id   TEXT,
    content       TEXT NOT NULL,
    tool_calls    TEXT,
    validity      TEXT,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    created_at    INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES fd_sessions(id) ON DELETE CASCADE
  )`,
  `CREATE INDEX IF NOT EXISTS idx_fd_messages_session_turn
    ON fd_messages(session_id, turn_idx, created_at)`,
];

// Promise is shared across the isolate — first caller pays the cost,
// every subsequent call awaits the same resolved promise. On failure we
// null it out so the next call can retry (e.g. transient network).
let _schemaPromise: Promise<void> | null = null;

export function ensureSchema(): Promise<void> {
  if (_schemaPromise) return _schemaPromise;
  const p = (async () => {
    const c = turso();
    await c.batch(DDL, "write");
  })();
  _schemaPromise = p.catch((e) => {
    _schemaPromise = null;
    throw e;
  });
  return _schemaPromise;
}

/** Test-only — wipe the cached schema promise so the next call re-runs DDL. */
export function _resetSchemaCacheForTest(): void {
  _schemaPromise = null;
}

// ── Local helpers ────────────────────────────────────────────────────

function nowSec(): number {
  return Math.floor(Date.now() / 1000);
}

function newId(): string {
  return crypto.randomUUID();
}

function strOrNull(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const t = v.trim();
  return t.length > 0 ? t : null;
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * Collapse whitespace + trim the first user turn down to a sidebar-
 * friendly title. Falls back to "(untitled)" when the content is all
 * whitespace. Caller can still rename via ``patchSession({title})``.
 */
export function titleFromContent(content: string): string {
  const cleaned = content.replace(/\s+/g, " ").trim();
  if (cleaned.length === 0) return "(untitled)";
  return cleaned.length <= 60 ? cleaned : `${cleaned.slice(0, 57)}…`;
}

function rowToUser(r: unknown): FdUser {
  const row = r as Record<string, unknown>;
  return {
    id: String(row.id),
    display_name: (row.display_name ?? null) as string | null,
    label_email: (row.label_email ?? null) as string | null,
    created_at: Number(row.created_at),
    last_seen_at: Number(row.last_seen_at),
  };
}

function rowToSession(r: unknown): FdSession {
  const row = r as Record<string, unknown>;
  return {
    id: String(row.id),
    user_id: String(row.user_id),
    title: String(row.title),
    category: (row.category ?? null) as string | null,
    provider_id: (row.provider_id ?? null) as string | null,
    mode: (row.mode ?? null) as string | null,
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
    pinned: Number(row.pinned ?? 0),
    archived: Number(row.archived ?? 0),
  };
}

function rowToMessage(r: unknown): FdMessage {
  const row = r as Record<string, unknown>;
  const rawRole = String(row.role ?? "user");
  const role: FdMessageRole =
    rawRole === "assistant" || rawRole === "tool" ? rawRole : "user";
  return {
    id: String(row.id),
    session_id: String(row.session_id),
    turn_idx: Number(row.turn_idx ?? 0),
    role,
    provider_id: (row.provider_id ?? null) as string | null,
    content: String(row.content ?? ""),
    tool_calls: (row.tool_calls ?? null) as string | null,
    validity: (row.validity ?? null) as string | null,
    tokens_in: row.tokens_in == null ? null : Number(row.tokens_in),
    tokens_out: row.tokens_out == null ? null : Number(row.tokens_out),
    created_at: Number(row.created_at ?? 0),
  };
}

// ── Users ────────────────────────────────────────────────────────────

/**
 * Idempotent upsert. Called on every authenticated request to refresh
 * ``last_seen_at``; on the very first call also creates the row with
 * NULL label fields — the user is anonymous until they hit UserBadge.
 */
export async function upsertUser(uid: string): Promise<FdUser> {
  if (!uid) throw new Error("upsertUser: uid is required");
  await ensureSchema();
  const c = turso();
  const now = nowSec();
  await c.execute({
    sql: `INSERT INTO fd_users (id, display_name, label_email, created_at, last_seen_at)
          VALUES (?, NULL, NULL, ?, ?)
          ON CONFLICT(id) DO UPDATE SET last_seen_at = excluded.last_seen_at`,
    args: [uid, now, now],
  });
  const user = await getUser(uid);
  if (!user) throw new Error(`upsertUser: failed to read back uid=${uid}`);
  return user;
}

export async function getUser(uid: string): Promise<FdUser | null> {
  if (!uid) return null;
  await ensureSchema();
  const c = turso();
  const rs = await c.execute({
    sql: `SELECT id, display_name, label_email, created_at, last_seen_at
          FROM fd_users WHERE id = ? LIMIT 1`,
    args: [uid],
  });
  if (rs.rows.length === 0) return null;
  return rowToUser(rs.rows[0]);
}

export interface UserLabelPatch {
  display_name?: string | null;
  label_email?: string | null;
}

export async function updateUserLabel(
  uid: string,
  patch: UserLabelPatch,
): Promise<FdUser> {
  await ensureSchema();
  await upsertUser(uid); // make sure the row exists before UPDATE
  const sets: string[] = [];
  const args: Array<string | number | null> = [];
  if ("display_name" in patch) {
    sets.push("display_name = ?");
    args.push(strOrNull(patch.display_name));
  }
  if ("label_email" in patch) {
    sets.push("label_email = ?");
    args.push(strOrNull(patch.label_email));
  }
  if (sets.length > 0) {
    args.push(uid);
    const c = turso();
    await c.execute({
      sql: `UPDATE fd_users SET ${sets.join(", ")} WHERE id = ?`,
      args,
    });
  }
  const out = await getUser(uid);
  if (!out) throw new Error(`updateUserLabel: user disappeared mid-update`);
  return out;
}

// ── Sessions ─────────────────────────────────────────────────────────

export interface ListSessionsOpts {
  /**
   * ``undefined`` → any category (no WHERE clause on category).
   * ``null``      → only sessions where category IS NULL ("No category").
   * string        → exact match.
   */
  category?: string | null;
  /** Default false — sidebar normally hides archived sessions. */
  archived?: boolean;
  /** Default 50, clamped to [1, 200]. */
  limit?: number;
}

export async function listSessions(
  userId: string,
  opts: ListSessionsOpts = {},
): Promise<FdSession[]> {
  if (!userId) return [];
  await ensureSchema();
  const c = turso();
  const archived = opts.archived === true ? 1 : 0;
  const limit = Math.max(1, Math.min(opts.limit ?? 50, 200));
  const clauses: string[] = ["user_id = ?", "archived = ?"];
  const args: Array<string | number | null> = [userId, archived];
  if (opts.category !== undefined) {
    if (opts.category === null) {
      clauses.push("category IS NULL");
    } else {
      clauses.push("category = ?");
      args.push(opts.category);
    }
  }
  args.push(limit);
  const rs = await c.execute({
    sql: `SELECT id, user_id, title, category, provider_id, mode,
                 created_at, updated_at, pinned, archived
          FROM fd_sessions
          WHERE ${clauses.join(" AND ")}
          ORDER BY pinned DESC, updated_at DESC
          LIMIT ?`,
    args,
  });
  return rs.rows.map(rowToSession);
}

export interface CreateSessionInput {
  title?: string;
  category?: string | null;
  provider_id?: string | null;
  mode?: string | null;
}

const MAX_ACTIVE_SESSIONS_PER_USER = 200;
const AUTO_ARCHIVE_BATCH = 20;

export async function createSession(
  userId: string,
  init: CreateSessionInput = {},
): Promise<FdSession> {
  if (!userId) throw new Error("createSession: userId is required");
  await ensureSchema();
  await upsertUser(userId);
  const c = turso();
  const id = newId();
  const now = nowSec();
  const rawTitle = (init.title ?? "").trim();
  const title = rawTitle.length > 0 ? rawTitle.slice(0, 200) : "New chat";
  await c.execute({
    sql: `INSERT INTO fd_sessions
          (id, user_id, title, category, provider_id, mode,
           created_at, updated_at, pinned, archived)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0)`,
    args: [
      id,
      userId,
      title,
      strOrNull(init.category),
      strOrNull(init.provider_id),
      strOrNull(init.mode),
      now,
      now,
    ],
  });
  // Keep the active set bounded so the sidebar stays snappy. The cost
  // is cheap and amortised — at most one UPDATE of ~20 rows per
  // creation beyond the cap.
  await maybeAutoArchive(userId);
  const session = await getSessionRaw(id, userId);
  if (!session) throw new Error(`createSession: insert succeeded but read-back failed`);
  return session;
}

async function maybeAutoArchive(userId: string): Promise<void> {
  const c = turso();
  const countRs = await c.execute({
    sql: `SELECT COUNT(*) AS n FROM fd_sessions
          WHERE user_id = ? AND archived = 0`,
    args: [userId],
  });
  const n = Number(
    (countRs.rows[0] as unknown as Record<string, unknown>).n ?? 0,
  );
  if (n <= MAX_ACTIVE_SESSIONS_PER_USER) return;
  const excess = n - MAX_ACTIVE_SESSIONS_PER_USER;
  const toArchive = Math.min(AUTO_ARCHIVE_BATCH, excess);
  await c.execute({
    sql: `UPDATE fd_sessions SET archived = 1
          WHERE id IN (
            SELECT id FROM fd_sessions
            WHERE user_id = ? AND archived = 0 AND pinned = 0
            ORDER BY updated_at ASC
            LIMIT ?
          )`,
    args: [userId, toArchive],
  });
}

/**
 * Load a session + all its messages in a single call. Returns ``null``
 * when the session doesn't exist or is owned by someone else — callers
 * should treat both cases as 404.
 */
export async function getSession(
  id: string,
  userId: string,
): Promise<{ session: FdSession; messages: FdMessage[] } | null> {
  const session = await getSessionRaw(id, userId);
  if (!session) return null;
  const messages = await listMessages(id);
  return { session, messages };
}

async function getSessionRaw(
  id: string,
  userId: string,
): Promise<FdSession | null> {
  if (!id || !userId) return null;
  await ensureSchema();
  const c = turso();
  const rs = await c.execute({
    sql: `SELECT id, user_id, title, category, provider_id, mode,
                 created_at, updated_at, pinned, archived
          FROM fd_sessions WHERE id = ? AND user_id = ? LIMIT 1`,
    args: [id, userId],
  });
  if (rs.rows.length === 0) return null;
  return rowToSession(rs.rows[0]);
}

export interface SessionPatch {
  title?: string;
  category?: string | null;
  provider_id?: string | null;
  mode?: string | null;
  pinned?: boolean;
  archived?: boolean;
}

export async function patchSession(
  id: string,
  userId: string,
  patch: SessionPatch,
): Promise<FdSession> {
  await ensureSchema();
  const sets: string[] = [];
  const args: Array<string | number | null> = [];
  if (patch.title !== undefined) {
    const t = patch.title.trim();
    if (t.length === 0) throw new Error("patchSession: title cannot be empty");
    sets.push("title = ?");
    args.push(t.slice(0, 200));
  }
  if ("category" in patch) {
    sets.push("category = ?");
    args.push(strOrNull(patch.category));
  }
  if ("provider_id" in patch) {
    sets.push("provider_id = ?");
    args.push(strOrNull(patch.provider_id));
  }
  if ("mode" in patch) {
    sets.push("mode = ?");
    args.push(strOrNull(patch.mode));
  }
  if (patch.pinned !== undefined) {
    sets.push("pinned = ?");
    args.push(patch.pinned ? 1 : 0);
  }
  if (patch.archived !== undefined) {
    sets.push("archived = ?");
    args.push(patch.archived ? 1 : 0);
  }
  if (sets.length > 0) {
    sets.push("updated_at = ?");
    args.push(nowSec());
    args.push(id, userId);
    const c = turso();
    await c.execute({
      sql: `UPDATE fd_sessions SET ${sets.join(", ")}
            WHERE id = ? AND user_id = ?`,
      args,
    });
  }
  const out = await getSessionRaw(id, userId);
  if (!out) throw new Error(`patchSession: session ${id} not found or not owned`);
  return out;
}

export async function deleteSession(
  id: string,
  userId: string,
): Promise<void> {
  if (!id || !userId) return;
  await ensureSchema();
  const c = turso();
  // Child messages are removed via ``ON DELETE CASCADE``.
  await c.execute({
    sql: `DELETE FROM fd_sessions WHERE id = ? AND user_id = ?`,
    args: [id, userId],
  });
}

// ── Messages ─────────────────────────────────────────────────────────

export interface AppendMessageInput {
  role: FdMessageRole;
  content: string;
  provider_id?: string | null;
  /** Arbitrary JSON-serialisable payload — stringified before insert. */
  tool_calls?: unknown;
  /** Arbitrary JSON-serialisable payload — stringified before insert. */
  validity?: unknown;
  tokens_in?: number | null;
  tokens_out?: number | null;
  /**
   * Override the auto-assigned turn index. Use when appending a second
   * assistant reply that should share the same turn as an existing one
   * (parallel-mode siblings). Ignored when < 0 or not a number.
   */
  turn_idx?: number;
}

const MAX_MESSAGES_PER_SESSION = 1000;
const MESSAGE_TRIM_BATCH = 100;

/**
 * Insert one message under ``sessionId``. The session is validated
 * against ``userId`` first, so this also doubles as an ownership check.
 * Returns the committed row with the server-assigned ``id`` /
 * ``turn_idx`` / ``created_at``.
 *
 * Turn-index rule:
 *   - role === "assistant" → shares MAX(turn_idx) (parallel replies
 *     cluster under the same user turn)
 *   - role === "user" | "tool" → MAX(turn_idx) + 1 (advances the turn)
 *   - explicit ``turn_idx`` override wins over both
 */
export async function appendMessage(
  sessionId: string,
  userId: string,
  msg: AppendMessageInput,
): Promise<FdMessage> {
  if (!sessionId || !userId) throw new Error("appendMessage: sessionId & userId required");
  if (!msg.content && msg.role !== "tool") {
    throw new Error("appendMessage: content required for user/assistant messages");
  }
  await ensureSchema();
  const session = await getSessionRaw(sessionId, userId);
  if (!session) throw new Error("appendMessage: session not found or not owned");

  const c = turso();
  const id = newId();
  const now = nowSec();

  let turnIdx: number;
  if (typeof msg.turn_idx === "number" && Number.isFinite(msg.turn_idx) && msg.turn_idx >= 0) {
    turnIdx = Math.floor(msg.turn_idx);
  } else {
    const rs = await c.execute({
      sql: `SELECT COALESCE(MAX(turn_idx), -1) AS mx FROM fd_messages
            WHERE session_id = ?`,
      args: [sessionId],
    });
    const mx = Number(
      (rs.rows[0] as unknown as Record<string, unknown>).mx ?? -1,
    );
    turnIdx = msg.role === "assistant" ? Math.max(0, mx) : mx + 1;
  }

  const toolCalls =
    msg.tool_calls !== undefined && msg.tool_calls !== null
      ? JSON.stringify(msg.tool_calls)
      : null;
  const validity =
    msg.validity !== undefined && msg.validity !== null
      ? JSON.stringify(msg.validity)
      : null;
  const providerId = strOrNull(msg.provider_id);
  const tokensIn = numOrNull(msg.tokens_in);
  const tokensOut = numOrNull(msg.tokens_out);

  // Insert + bump parent session atomically so the sidebar order never
  // lags behind the message tail. ``provider_id`` on the session
  // remembers the last adapter used (used by the client to restore the
  // radio selection on session open).
  await c.batch(
    [
      {
        sql: `INSERT INTO fd_messages
              (id, session_id, turn_idx, role, provider_id, content,
               tool_calls, validity, tokens_in, tokens_out, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          id,
          sessionId,
          turnIdx,
          msg.role,
          providerId,
          msg.content,
          toolCalls,
          validity,
          tokensIn,
          tokensOut,
          now,
        ],
      },
      {
        sql: `UPDATE fd_sessions SET updated_at = ?,
              provider_id = COALESCE(?, provider_id)
              WHERE id = ?`,
        args: [now, providerId, sessionId],
      },
    ],
    "write",
  );

  await maybeTrimMessages(sessionId);

  // Construct the return from the values we just wrote — saves a
  // round-trip vs. ``getMessage(id)``.
  return {
    id,
    session_id: sessionId,
    turn_idx: turnIdx,
    role: msg.role,
    provider_id: providerId,
    content: msg.content,
    tool_calls: toolCalls,
    validity,
    tokens_in: tokensIn,
    tokens_out: tokensOut,
    created_at: now,
  };
}

async function maybeTrimMessages(sessionId: string): Promise<void> {
  const c = turso();
  const rs = await c.execute({
    sql: `SELECT COUNT(*) AS n FROM fd_messages WHERE session_id = ?`,
    args: [sessionId],
  });
  const n = Number(
    (rs.rows[0] as unknown as Record<string, unknown>).n ?? 0,
  );
  if (n <= MAX_MESSAGES_PER_SESSION) return;
  const excess = n - MAX_MESSAGES_PER_SESSION;
  const toTrim = Math.min(MESSAGE_TRIM_BATCH, excess);
  await c.execute({
    sql: `DELETE FROM fd_messages
          WHERE id IN (
            SELECT id FROM fd_messages
            WHERE session_id = ?
            ORDER BY turn_idx ASC, created_at ASC
            LIMIT ?
          )`,
    args: [sessionId, toTrim],
  });
}

export async function listMessages(sessionId: string): Promise<FdMessage[]> {
  if (!sessionId) return [];
  await ensureSchema();
  const c = turso();
  const rs = await c.execute({
    sql: `SELECT id, session_id, turn_idx, role, provider_id, content,
                 tool_calls, validity, tokens_in, tokens_out, created_at
          FROM fd_messages
          WHERE session_id = ?
          ORDER BY turn_idx ASC, created_at ASC`,
    args: [sessionId],
  });
  return rs.rows.map(rowToMessage);
}

export async function getMessage(id: string): Promise<FdMessage | null> {
  if (!id) return null;
  await ensureSchema();
  const c = turso();
  const rs = await c.execute({
    sql: `SELECT id, session_id, turn_idx, role, provider_id, content,
                 tool_calls, validity, tokens_in, tokens_out, created_at
          FROM fd_messages WHERE id = ? LIMIT 1`,
    args: [id],
  });
  if (rs.rows.length === 0) return null;
  return rowToMessage(rs.rows[0]);
}

// ── Categories summary ───────────────────────────────────────────────

export interface CategoryCount {
  /** ``null`` represents the "No category" sidebar bucket. */
  category: string | null;
  count: number;
}

export async function listCategoriesWithCounts(
  userId: string,
): Promise<CategoryCount[]> {
  if (!userId) return [];
  await ensureSchema();
  const c = turso();
  const rs = await c.execute({
    sql: `SELECT category, COUNT(*) AS count
          FROM fd_sessions
          WHERE user_id = ? AND archived = 0
          GROUP BY category
          ORDER BY count DESC, category ASC`,
    args: [userId],
  });
  return rs.rows.map((r) => {
    const row = r as unknown as Record<string, unknown>;
    return {
      category: (row.category ?? null) as string | null,
      count: Number(row.count ?? 0),
    };
  });
}

// ── Title auto-gen ──────────────────────────────────────────────────

/**
 * Rename the session based on its first user message if the current
 * title is still the "New chat" placeholder. No-op for already-renamed
 * sessions, so callers can fire this on every user turn safely.
 */
export async function maybeAutoTitle(
  sessionId: string,
  userId: string,
  firstUserMessage: string,
): Promise<FdSession | null> {
  const s = await getSessionRaw(sessionId, userId);
  if (!s) return null;
  if (s.title && s.title !== "New chat") return s;
  const newTitle = titleFromContent(firstUserMessage);
  if (newTitle === s.title) return s;
  const c = turso();
  await c.execute({
    sql: `UPDATE fd_sessions SET title = ?, updated_at = ?
          WHERE id = ? AND user_id = ?`,
    args: [newTitle, nowSec(), sessionId, userId],
  });
  return getSessionRaw(sessionId, userId);
}

// ── Export (JSONL) ──────────────────────────────────────────────────

/**
 * Stream the user's full history out as newline-delimited JSON. Active
 * + archived, messages interleaved under their sessions. Meant for the
 * ``POST /api/history/export`` route so a user can download their own
 * data before deleting the cookie.
 */
export async function exportJsonl(userId: string): Promise<string> {
  if (!userId) return "";
  await ensureSchema();
  const active = await listSessions(userId, { archived: false, limit: 200 });
  const archived = await listSessions(userId, { archived: true, limit: 200 });
  const all = [...active, ...archived];
  const lines: string[] = [];
  for (const s of all) {
    const messages = await listMessages(s.id);
    lines.push(JSON.stringify({ type: "session", ...s }));
    for (const m of messages) {
      lines.push(JSON.stringify({ type: "message", ...m }));
    }
  }
  return lines.join("\n");
}
