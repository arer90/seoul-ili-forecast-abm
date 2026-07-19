-- Frame D chat history schema (Phase 3 of the 2026-04-21 roll-out).
--
-- Idempotent — safe to rerun against the live Turso DB. All three
-- tables coexist with the static-aggregate tables already loaded by
-- ``scripts/turso_seed.sql``; prefix ``fd_`` on every table name keeps
-- them grouped and avoids any chance of collision with the SQLite-side
-- epi schema.
--
-- Apply manually with the Turso CLI:
--     turso db shell <your-db> < scripts/migrations/001_history_schema.sql
--
-- At runtime the same DDL is issued via ``lib/history-db.ts#ensureSchema``
-- so a freshly-provisioned DB self-heals on the first history request.

PRAGMA foreign_keys = ON;

-- ── Users ─────────────────────────────────────────────────────────────
-- ``id`` is the same UUID we set in the ``fd_uid`` cookie. Anonymous
-- users have NULL ``display_name`` / ``label_email``. The UI shows only
-- the first 8 chars of the uid as a badge until a label is attached.
CREATE TABLE IF NOT EXISTS fd_users (
  id            TEXT PRIMARY KEY,
  display_name  TEXT,
  label_email   TEXT,
  created_at    INTEGER NOT NULL,
  last_seen_at  INTEGER NOT NULL
);

-- ── Sessions ──────────────────────────────────────────────────────────
-- ``title`` is auto-generated from the first user message (first 60
-- chars, collapsed whitespace). User can rename inline in
-- SessionHeader.
--
-- ``category`` is free-text but the UI suggests 6 seed values
-- (baseline/NPI/vaccination/antiviral/validation/ad hoc). NULL = "No
-- category" in the sidebar.
--
-- ``provider_id`` + ``mode`` preserve the last-used LLM configuration
-- so reopening a session restores the right Claude/GPT/etc. selection.
--
-- Soft delete via ``archived = 1`` — the sidebar hides these by
-- default but they're still query-able for export/undelete.
CREATE TABLE IF NOT EXISTS fd_sessions (
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
);

-- Sidebar query: user's non-archived sessions sorted by recency.
CREATE INDEX IF NOT EXISTS idx_fd_sessions_user_updated
  ON fd_sessions(user_id, archived, updated_at DESC);

-- Category filter: same dimensions + category.
CREATE INDEX IF NOT EXISTS idx_fd_sessions_user_category
  ON fd_sessions(user_id, category, updated_at DESC);

-- ── Messages ──────────────────────────────────────────────────────────
-- ``turn_idx`` is a 0-based sequence within the session. User message
-- gets one index; each provider's assistant reply gets its own index
-- (parallel mode → N sibling assistant rows sharing the same turn, so
-- we allow ``turn_idx`` duplicates and disambiguate with ``provider_id``).
--
-- ``tool_calls`` + ``validity`` are stored as stringified JSON for
-- query simplicity. libSQL has no native JSON type but ``json_extract``
-- works fine on TEXT columns when we need to filter/aggregate.
--
-- ``tokens_in`` / ``tokens_out`` are usage telemetry — NULL when the
-- provider doesn't report usage (Ollama, some OpenAI streaming modes).
CREATE TABLE IF NOT EXISTS fd_messages (
  id            TEXT PRIMARY KEY,
  session_id    TEXT NOT NULL,
  turn_idx      INTEGER NOT NULL,
  role          TEXT NOT NULL,           -- user | assistant | tool
  provider_id   TEXT,
  content       TEXT NOT NULL,
  tool_calls    TEXT,                    -- JSON array string or NULL
  validity      TEXT,                    -- JSON object string or NULL
  tokens_in     INTEGER,
  tokens_out    INTEGER,
  created_at    INTEGER NOT NULL,
  FOREIGN KEY (session_id) REFERENCES fd_sessions(id) ON DELETE CASCADE
);

-- Session replay: ordered fetch of all messages.
CREATE INDEX IF NOT EXISTS idx_fd_messages_session_turn
  ON fd_messages(session_id, turn_idx, created_at);
