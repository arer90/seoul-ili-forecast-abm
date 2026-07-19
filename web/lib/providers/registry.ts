/**
 * Provider registry — singleton-per-process, lazy-initialised.
 * Edge runtime safe (module-level cache across requests in the same
 * isolate).
 */
import { createAnthropic } from "./anthropic";
import { createGoogle } from "./google";
import { createOllama } from "./ollama";
import { createOpenAI } from "./openai";
import type { ProviderAdapter, ProviderId } from "./types";

let _cache: Map<ProviderId, ProviderAdapter> | null = null;

export function getProviders(): Map<ProviderId, ProviderAdapter> {
  if (_cache) return _cache;
  const m = new Map<ProviderId, ProviderAdapter>();
  for (const make of [createAnthropic, createGoogle, createOpenAI, createOllama]) {
    const p = make();
    m.set(p.id, p);
  }
  _cache = m;
  return m;
}

export function getProvider(id: ProviderId): ProviderAdapter {
  const p = getProviders().get(id);
  if (!p) throw new Error(`unknown provider: ${id}`);
  return p;
}

export function listAvailableProviders(): ProviderAdapter[] {
  return [...getProviders().values()].filter((p) => p.available());
}
