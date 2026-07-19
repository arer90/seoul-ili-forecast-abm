-- 002_aria_user_memory.sql
-- Long-term per-user memory for ARIA chat (cross-session).
-- Apply via: turso db shell <db-name> < 002_aria_user_memory.sql
-- Runtime also self-heals via ensureMemorySchema() in lib/user-memory.ts.

CREATE TABLE IF NOT EXISTS aria_user_memory (
  uid        TEXT    NOT NULL,
  key        TEXT    NOT NULL,
  value      TEXT    NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (uid, key)
);

CREATE INDEX IF NOT EXISTS idx_aria_memory_uid_updated
  ON aria_user_memory(uid, updated_at DESC);
