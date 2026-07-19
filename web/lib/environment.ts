/**
 * Environment detector — single source of truth for "which provider
 * should the UI default to, and why".
 *
 * Rationale
 * ---------
 * The demo runs in two very different contexts:
 *
 *   local dev   — Ollama is up on 11434, cloud keys often blank, the
 *                 operator wants zero-cost iteration
 *   Vercel prod — cloud keys present, Ollama unreachable by design
 *                 (we don't self-host the Ollama daemon on Vercel)
 *
 * Hardcoding ``providers = ["anthropic"]`` in the client broke both
 * cases — locally the user saw "ANTHROPIC_API_KEY not set" on every
 * turn, and on prod we were locked to Claude even when GPT/Gemini
 * would have been fine fallbacks.
 *
 * This module computes a single ``detectEnvironment()`` answer and
 * every caller (``/api/providers``, tests, future Hermes fallback
 * chain) routes through it so the rules stay consistent.
 *
 * Caching
 * -------
 * Detection touches process.env (synchronous, free) and one network
 * probe to Ollama (500 ms timeout). We cache the whole result for 5
 * seconds per isolate — long enough that a page load causing 3–4
 * /api/providers hits doesn't fire 3–4 Ollama probes, short enough
 * that flipping OLLAMA_BASE_URL or toggling the daemon during dev
 * shows up on the next reload.
 */
import type { ProviderId } from "./providers/types";

export type DeployMode = "local" | "cloud";

export interface ProviderAvailability {
  anthropic: boolean;
  openai: boolean;
  google: boolean;
  ollama: boolean;
}

export interface EnvironmentInfo {
  /** "cloud" when VERCEL or NODE_ENV=production, else "local". */
  mode: DeployMode;
  /** Per-provider availability at detection time. */
  available: ProviderAvailability;
  /** The provider the UI should pre-select on first load. */
  recommended: ProviderId;
  /** Fallback chain to try when ``recommended`` fails with 429/5xx. */
  fallbackOrder: ProviderId[];
  /** When the detection was computed (unix sec) — exposed so the UI
   *  can decide to re-fetch after long idle periods. */
  computedAt: number;
  /** Non-authoritative diagnostics — e.g. "ollama unreachable: ETIMEDOUT". */
  notes: Partial<Record<ProviderId, string>>;
}

// ── Module-scoped cache ────────────────────────────────────────────────
// Edge runtime creates fresh isolates frequently, so this cache is
// naturally short-lived across deploys. The 5-second TTL handles the
// common "user reloads the page" burst on a single isolate.
const CACHE_TTL_MS = 5_000;
let _cache: { info: EnvironmentInfo; expiresAt: number } | null = null;

// ── Helpers ────────────────────────────────────────────────────────────

function detectMode(): DeployMode {
  if (process.env.VERCEL === "1") return "cloud";
  if (process.env.VERCEL_ENV === "production") return "cloud";
  if (process.env.NODE_ENV === "production") return "cloud";
  return "local";
}

function hasKey(name: string): boolean {
  const v = process.env[name];
  return typeof v === "string" && v.length > 0;
}

async function probeOllama(): Promise<{ ok: boolean; note?: string }> {
  const base = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 500);
    const r = await fetch(`${base}/api/tags`, { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) return { ok: false, note: `HTTP ${r.status}` };
    return { ok: true };
  } catch (e) {
    // AbortError, DNS failure, connection refused — all treated as
    // "daemon not there". The note is diagnostic only; the boolean
    // decides availability.
    return {
      ok: false,
      note: e instanceof Error ? e.message : String(e),
    };
  }
}

function pickRecommended(
  mode: DeployMode,
  a: ProviderAvailability,
): ProviderId {
  // 2026-05-07: User decision — Claude is the ONLY API provider.
  // Both Gemini (google) and GPT (openai) provider adapters stay in
  // lib/providers/* (callable when explicitly selected via URL/state) but
  // never recommended or auto-fallback. Ollama remains for local-mode
  // fallback because it is not a paid API (loopback to localhost daemon).
  if (mode === "local") {
    if (a.ollama) return "ollama";
    if (a.anthropic) return "anthropic";
    return "anthropic"; // show the most useful error if nothing is up
  }
  // cloud — Claude only
  if (a.anthropic) return "anthropic";
  if (a.ollama) return "ollama";
  return "anthropic";
}

function pickFallbackOrder(
  mode: DeployMode,
  recommended: ProviderId,
  a: ProviderAvailability,
): ProviderId[] {
  // Gemini + GPT omitted from fallback chains — see pickRecommended note.
  const cloudBase: ProviderId[] = ["anthropic"];
  const localBase: ProviderId[] = ["ollama", "anthropic"];
  const chain = mode === "cloud" ? cloudBase : localBase;
  return chain
    .filter((id) => id !== recommended)
    .filter((id) => a[id]);
}

// ── Public API ─────────────────────────────────────────────────────────

export async function detectEnvironment(): Promise<EnvironmentInfo> {
  const now = Date.now();
  if (_cache && _cache.expiresAt > now) return _cache.info;

  const mode = detectMode();
  const notes: Partial<Record<ProviderId, string>> = {};

  // Cloud-key checks are sync and free.
  const available: ProviderAvailability = {
    anthropic: hasKey("ANTHROPIC_API_KEY"),
    openai: hasKey("OPENAI_API_KEY"),
    google: hasKey("GOOGLE_API_KEY") || hasKey("GEMINI_API_KEY"),
    ollama: false,
  };
  if (!available.anthropic) notes.anthropic = "ANTHROPIC_API_KEY not set";
  if (!available.openai) notes.openai = "OPENAI_API_KEY not set";
  if (!available.google) notes.google = "GOOGLE_API_KEY / GEMINI_API_KEY not set";

  // Ollama probe — 500 ms timeout so we never block a request meaningfully.
  // Skip entirely in cloud mode since there's nowhere for Ollama to live.
  if (mode === "local") {
    const probe = await probeOllama();
    available.ollama = probe.ok;
    if (!probe.ok) notes.ollama = `unreachable: ${probe.note ?? "timeout"}`;
  } else {
    notes.ollama = "hidden in cloud mode";
  }

  const recommended = pickRecommended(mode, available);
  const fallbackOrder = pickFallbackOrder(mode, recommended, available);

  const info: EnvironmentInfo = {
    mode,
    available,
    recommended,
    fallbackOrder,
    computedAt: Math.floor(now / 1000),
    notes,
  };
  _cache = { info, expiresAt: now + CACHE_TTL_MS };
  return info;
}

/** Test-only — wipe the cache so the next call re-probes. */
export function _resetEnvironmentCacheForTest(): void {
  _cache = null;
}
