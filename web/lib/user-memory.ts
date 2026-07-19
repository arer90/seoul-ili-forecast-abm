/**
 * user-memory — long-term per-user memory layer.
 *
 * Architecture (memory tiers)
 * ---------------------------
 * Short-term (within a session):
 *   `fd_messages` in Turso — every turn stored via appendMessage().
 *   The chat-cli route loads recent messages for the session and includes
 *   them in the Claude prompt as conversation context.
 *
 * Long-term (cross-session):
 *   `aria_user_memory` table in Turso — one row per (uid, key) fact.
 *   Keys are free-form strings (e.g. "preferred_district", "concern",
 *   "role").  Values are short strings (≤500 chars).  Cap: 50 facts per
 *   user (oldest displaced first).  Retrieved facts are injected into
 *   the system prompt so ARIA remembers across sessions.
 *
 * Design rules
 * ------------
 *  - TURSO_URL absent → every function is a silent no-op (local-only dev).
 *  - Edge-compatible: only @libsql/client/web, no `fs` / `crypto` APIs.
 *  - Single DDL string: ensureMemorySchema() idempotently creates the
 *    table + index.  Self-heals on first call.
 *  - One public write path: upsertMemory(uid, key, value).
 *    One public read path: getMemoryBlock(uid) → formatted prompt string.
 *
 * Caller responsibility (chat-cli route):
 *   1. After each completed assistant turn, call upsertMemory() for any
 *      key facts the user mentioned (name, district preference, concern).
 *      The route uses a simple heuristic extraction — a full NER pass
 *      can be added later.
 *   2. Before building the prompt, call getMemoryBlock(uid) and append
 *      the result to the system grounding string.
 */
import type { InStatement } from "@libsql/client/web";

import { turso } from "./turso";

// ── Types ──────────────────────────────────────────────────────────────

export interface UserMemoryFact {
  uid: string;
  key: string;
  value: string;
  updated_at: number;
}

// ── DDL ───────────────────────────────────────────────────────────────

const MEMORY_DDL: InStatement[] = [
  `CREATE TABLE IF NOT EXISTS aria_user_memory (
    uid        TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (uid, key)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_aria_memory_uid_updated
    ON aria_user_memory(uid, updated_at DESC)`,
];

const MAX_FACTS_PER_USER = 50;

let _memSchemaPromise: Promise<void> | null = null;

/**
 * Idempotently create `aria_user_memory`.  Cached per isolate — first
 * call pays the DDL cost, subsequent calls are free.
 *
 * Returns: void (no-op when TURSO_URL is absent).
 */
export function ensureMemorySchema(): Promise<void> {
  if (!process.env.TURSO_URL) return Promise.resolve();
  if (_memSchemaPromise) return _memSchemaPromise;
  const p = (async () => {
    const c = turso();
    await c.batch(MEMORY_DDL, "write");
  })();
  _memSchemaPromise = p.catch((e) => {
    _memSchemaPromise = null;
    throw e;
  });
  return _memSchemaPromise;
}

// ── Write ──────────────────────────────────────────────────────────────

/**
 * Upsert one long-term memory fact for `uid`.
 *
 * Args:
 *   uid:   user identifier (fd_uid cookie value).
 *   key:   short camelCase label, e.g. "preferredDistrict", "role".
 *   value: string fact (truncated to 500 chars).
 *
 * Returns: void.  Silent no-op when TURSO_URL absent or on any DB error.
 *
 * Side effects: inserts or replaces the (uid, key) row; when the user
 * exceeds MAX_FACTS_PER_USER, the oldest fact is deleted.
 */
export async function upsertMemory(
  uid: string,
  key: string,
  value: string,
): Promise<void> {
  if (!process.env.TURSO_URL) return;
  if (!uid || !key || !value.trim()) return;
  try {
    await ensureMemorySchema();
    const c = turso();
    const now = Math.floor(Date.now() / 1000);
    const safeValue = value.slice(0, 500);
    await c.execute({
      sql: `INSERT INTO aria_user_memory (uid, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(uid, key) DO UPDATE
              SET value = excluded.value, updated_at = excluded.updated_at`,
      args: [uid, key, safeValue, now],
    });
    // Enforce per-user cap: delete oldest fact(s) beyond MAX_FACTS_PER_USER.
    await c.execute({
      sql: `DELETE FROM aria_user_memory
            WHERE uid = ? AND key NOT IN (
              SELECT key FROM aria_user_memory
              WHERE uid = ?
              ORDER BY updated_at DESC
              LIMIT ?
            )`,
      args: [uid, uid, MAX_FACTS_PER_USER],
    });
  } catch {
    // Never crash the caller — memory is best-effort.
  }
}

// ── Read ───────────────────────────────────────────────────────────────

/**
 * Load all memory facts for `uid` and format as a prompt injection block.
 *
 * Args:
 *   uid: user identifier.
 *
 * Returns:
 *   A multi-line string block ready to append to the system prompt, or ""
 *   when TURSO_URL absent / no facts exist / DB error.
 *
 * Performance: one SELECT per call; result not cached here (TTL cache
 * could be layered via Upstash KV if needed).
 */
export async function getMemoryBlock(uid: string): Promise<string> {
  if (!process.env.TURSO_URL) return "";
  if (!uid) return "";
  try {
    await ensureMemorySchema();
    const c = turso();
    const rs = await c.execute({
      sql: `SELECT key, value FROM aria_user_memory
            WHERE uid = ?
            ORDER BY updated_at DESC
            LIMIT ?`,
      args: [uid, MAX_FACTS_PER_USER],
    });
    if (rs.rows.length === 0) return "";
    const facts = rs.rows.map((r) => {
      const row = r as unknown as Record<string, unknown>;
      return `  · ${row.key}: ${row.value}`;
    });
    return (
      "\n\n[장기기억 — 이 사용자에 대한 이전 세션 기록]:\n" +
      facts.join("\n")
    );
  } catch {
    return "";
  }
}

/**
 * Read all raw facts for a user (for API exposure / export).
 *
 * Returns: array of UserMemoryFact, or [] on any error / unavailability.
 */
export async function listMemory(uid: string): Promise<UserMemoryFact[]> {
  if (!process.env.TURSO_URL) return [];
  if (!uid) return [];
  try {
    await ensureMemorySchema();
    const c = turso();
    const rs = await c.execute({
      sql: `SELECT uid, key, value, updated_at
            FROM aria_user_memory WHERE uid = ?
            ORDER BY updated_at DESC`,
      args: [uid],
    });
    return rs.rows.map((r) => {
      const row = r as unknown as Record<string, unknown>;
      return {
        uid: String(row.uid),
        key: String(row.key),
        value: String(row.value),
        updated_at: Number(row.updated_at),
      };
    });
  } catch {
    return [];
  }
}

/**
 * Delete one fact (e.g. user-initiated "forget this").
 *
 * Returns: void.  Silent no-op on any error.
 */
export async function deleteMemory(uid: string, key: string): Promise<void> {
  if (!process.env.TURSO_URL) return;
  if (!uid || !key) return;
  try {
    await ensureMemorySchema();
    const c = turso();
    await c.execute({
      sql: `DELETE FROM aria_user_memory WHERE uid = ? AND key = ?`,
      args: [uid, key],
    });
  } catch {
    // silent no-op
  }
}

// ── Heuristic extractor (lightweight, no LLM) ─────────────────────────

/**
 * Extract candidate (key, value) pairs from a user message using simple
 * pattern matching.  This is intentionally conservative — only fires on
 * clear self-disclosure phrases.  An LLM-based extraction pass can be
 * added later; this gives day-one coverage without an extra round-trip.
 *
 * Args:
 *   content: the user's message text.
 *
 * Returns:
 *   Array of {key, value} pairs to upsert into user memory.  May be empty.
 */
export function extractMemoryFacts(
  content: string,
): Array<{ key: string; value: string }> {
  const facts: Array<{ key: string; value: string }> = [];
  const lower = content.toLowerCase();

  // Preferred district / gu
  const guMatch = content.match(
    /(?:저는?|우리|내가?|제가?)?\s*([가-힣]+구)\s*(?:에\s*살|거주|관심|담당|관할)/,
  );
  if (guMatch) {
    facts.push({ key: "preferredDistrict", value: guMatch[1] });
  }

  // Role / affiliation keywords
  if (/역학자|epidemiolog/i.test(content)) {
    facts.push({ key: "role", value: "역학자" });
  } else if (/공중보건|보건소|public health/i.test(content)) {
    facts.push({ key: "role", value: "공중보건 종사자" });
  } else if (/연구자|researcher|researcher/i.test(content)) {
    facts.push({ key: "role", value: "연구자" });
  } else if (/학생|student/i.test(content)) {
    facts.push({ key: "role", value: "학생" });
  }

  // Primary concern
  if (/독감|인플루엔자|influenza|ili/i.test(lower)) {
    facts.push({ key: "primaryConcern", value: "인플루엔자/ILI" });
  } else if (/코로나|covid|sars/i.test(lower)) {
    facts.push({ key: "primaryConcern", value: "COVID-19" });
  }

  // Season / year reference (let ARIA recall what season user cares about)
  const seasonMatch = content.match(/(\d{4}[-\/]\d{2,4})\s*(?:시즌|절기)/);
  if (seasonMatch) {
    facts.push({ key: "referenceSeason", value: seasonMatch[1] });
  }

  return facts;
}
