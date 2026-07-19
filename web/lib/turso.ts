/**
 * Turso libSQL edge client — used only when TURSO_URL is set (chat
 * history persistence). When TURSO_URL is absent (local-only mode),
 * ``turso()`` throws "TURSO_URL not set" and callers fall back
 * gracefully via their own try/catch blocks.
 *
 * ILI overlay data no longer goes through this module (removed
 * 2026-06-08 — see ``web/lib/live-overlays/turso-ili.ts`` which now
 * fetches from ``/aggregates/ili-local.json`` instead).
 */
import { createClient, type Client } from "@libsql/client/web";

let _client: Client | null = null;

export function turso(): Client {
  if (_client) return _client;
  const url = process.env.TURSO_URL;
  const authToken = process.env.TURSO_TOKEN;
  if (!url) throw new Error("TURSO_URL not set");
  _client = createClient({ url, authToken });
  return _client;
}

/** Convenience row-dict query. Returns an empty array when TURSO_URL is
 *  not set so callers that don't check the env var stay non-crashing. */
export async function query<T = Record<string, unknown>>(
  sql: string,
  args: (string | number | null)[] = [],
): Promise<T[]> {
  if (!process.env.TURSO_URL) return [];
  const rs = await turso().execute({ sql, args });
  return rs.rows.map((row) => {
    const o: Record<string, unknown> = {};
    for (const col of rs.columns) {
      o[col] = (row as unknown as Record<string, unknown>)[col] ?? null;
    }
    return o as T;
  });
}
