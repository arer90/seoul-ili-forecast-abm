/**
 * ChatPanel — the conversation surface.
 *
 * Responsibilities:
 *   - sticky mode/provider bar at the top (solo/parallel/synthesis/relay)
 *   - scrolling turn list; assistant turns show streaming text + tool
 *     trace + validity badge
 *   - composer at the bottom with suggestion chips, Enter-to-send,
 *     Esc-to-abort
 *
 * Streaming uses the Edge SSE from /api/chat — we parse ndjson events
 * out of each `data:` frame. Each provider's text goes into its own
 * column when mode !== "solo".
 */
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { MODE_LABELS, PROVIDER_LABELS, RESPONSE_MODES } from "@/lib/constants";
import type { FdMessage } from "@/lib/history-db";
import { useT } from "@/lib/i18n";
import type { ProviderId, ResponseMode, ToolCall, ToolResult } from "@/lib/providers/types";
import { useSessionStore } from "@/lib/use-session-store";
import { readSSE } from "@/lib/util/sse";
import { checkReply, type ValidityBadge as ValidityResult } from "@/lib/validity";

import { ForecastTrace, type ForecastTraceEntry } from "./ForecastTrace";
import {
  ForecastModelPicker,
  FORECAST_MODEL_AUTO,
} from "./ForecastModelPicker";
import { PersonaPicker } from "./PersonaPicker";
import { ProviderPicker } from "./ProviderPicker";
import { SuggestionChips } from "./SuggestionChips";
// Sprint 2026-05-07: TrainedModelsCard 제거 (사용자 명시 '관심 없음').
// import { TrainedModelsCard } from "./TrainedModelsCard";
import { Button } from "./ui/button";
import { Select } from "./ui/select";
import { Textarea } from "./ui/input";
import { HelpIcon } from "./ui/HelpIcon";
import { ValidityBadge } from "./ValidityBadge";
import { PERSONA_GENERAL, getPersonaById } from "@/lib/personas";

/**
 * Sprint 2026-05-06 (#14): persona command router. 사용자 명시 —
 * "'상담관점' 아이콘을 없애. 다 할 수 있어야해. 특화는 대화에서 `!역학` 같이
 * 조치." 즉 default = general (모든 영역 답변), 특화는 chat 명령어로.
 *
 * Boundary 매칭: `!역학` (혼자) 또는 `!역학 ...` (뒤에 space) 만 매칭.
 * `!역학데이터` 같은 false-positive 회피.
 */
const PERSONA_COMMANDS: Record<string, string> = {
  "!역학": "epi-advisor",
  "!임상": "clinical-advisor",
  "!모델": "model-advisor",
  "!시뮬레이션": "simulation-advisor",
  "!일반": PERSONA_GENERAL,
  "!epi": "epi-advisor",
  "!clinical": "clinical-advisor",
  "!model": "model-advisor",
  "!simulation": "simulation-advisor",
  "!general": PERSONA_GENERAL,
};

function parsePersonaCommand(
  text: string,
): { persona: string | null; cleanText: string } {
  for (const [cmd, personaId] of Object.entries(PERSONA_COMMANDS)) {
    if (text === cmd || text.startsWith(cmd + " ")) {
      return { persona: personaId, cleanText: text.slice(cmd.length).trimStart() };
    }
  }
  return { persona: null, cleanText: text };
}

const HIDE_OLLAMA =
  (process.env.NEXT_PUBLIC_HIDE_OLLAMA ?? "").toLowerCase() === "1";

// Initial model picked before /api/providers resolves. The
// ProviderPicker then overwrites this with the first model from the
// live /api/tags list for Ollama (see app/api/providers/route.ts), so
// this constant only matters for the first paint and for cloud
// providers. exaone3.5:7.8b is the Korean-capable default the user
// pulled on the dev laptop.
// 2026-05-07: User decision — Claude is the ONLY API provider.
// GPT + Gemini default models kept here so the adapters still have a sane
// model when explicitly selected via URL state, but environment.ts no longer
// recommends them. Ollama stays for local-mode (no paid API).
const DEFAULT_MODELS: Record<ProviderId, string> = {
  anthropic: "claude-sonnet-4-6",
  ollama: "exaone3.5:7.8b",
  openai: "gpt-5-mini",
  google: "gemini-2.5-flash",
};

interface TurnNotice {
  level: "info" | "warn" | "error";
  message: string;
}

interface Turn {
  id: string;
  role: "user" | "assistant";
  /** For assistant turns, text accumulated per provider. */
  text: Partial<Record<ProviderId, string>>;
  /** For assistant turns, tool calls seen per provider. */
  trace: Partial<Record<ProviderId, ForecastTraceEntry[]>>;
  /** Validity badges computed after stream done. */
  validity?: Partial<Record<ProviderId, ValidityResult>>;
  /**
   * Per-provider status notices surfaced by the provider stream —
   * e.g. "this model does not support tools", "max tool hops reached".
   * Rendered as a subtle advisory above the reply so the user knows
   * why the answer lacks DB-backed numbers when that's the cause.
   */
  notices?: Partial<Record<ProviderId, TurnNotice[]>>;
  /** For user turns. */
  content?: string;
  mode?: ResponseMode;
  providers?: ProviderId[];
  status?: "streaming" | "done" | "aborted" | "error";
  error?: string;
}

function freshTurn(partial: Partial<Turn> & Pick<Turn, "role">): Turn {
  return {
    id: crypto.randomUUID(),
    text: {},
    trace: {},
    validity: {},
    status: "streaming",
    ...partial,
  };
}

/**
 * Convert a persisted message list (sorted by turn_idx, then created_at)
 * into the in-memory ``Turn[]`` shape the UI renders. Parallel-mode
 * assistant replies share a single turn_idx, so we group them into one
 * assistant Turn with per-provider text slots — mirroring what the
 * live stream produces for the same case.
 */
function messagesToTurns(messages: FdMessage[]): Turn[] {
  const turns: Turn[] = [];
  const byTurn = new Map<number, FdMessage[]>();
  for (const m of messages) {
    const arr = byTurn.get(m.turn_idx) ?? [];
    arr.push(m);
    byTurn.set(m.turn_idx, arr);
  }
  const sortedTurnIndices = Array.from(byTurn.keys()).sort((a, b) => a - b);
  for (const idx of sortedTurnIndices) {
    const group = byTurn.get(idx) ?? [];
    const userMsg = group.find((m) => m.role === "user");
    if (userMsg) {
      turns.push({
        id: userMsg.id,
        role: "user",
        text: {},
        trace: {},
        content: userMsg.content,
        status: "done",
      });
    }
    const asstMsgs = group.filter((m) => m.role === "assistant");
    if (asstMsgs.length > 0) {
      const text: Partial<Record<ProviderId, string>> = {};
      const providers: ProviderId[] = [];
      let validity: Partial<Record<ProviderId, ValidityResult>> | undefined;
      for (const m of asstMsgs) {
        const pid = (m.provider_id ?? "anthropic") as ProviderId;
        text[pid] = m.content;
        if (!providers.includes(pid)) providers.push(pid);
        if (m.validity) {
          try {
            const parsed = JSON.parse(m.validity) as ValidityResult;
            validity = { ...(validity ?? {}), [pid]: parsed };
          } catch {
            /* stale/malformed validity blob — let the UI recompute */
          }
        }
      }
      turns.push({
        id: asstMsgs[0].id,
        role: "assistant",
        text,
        trace: {},
        providers,
        mode: providers.length > 1 ? "parallel" : "solo",
        validity,
        status: "done",
      });
    }
  }
  return turns;
}

export interface ChatPanelProps {
  /**
   * Fires whenever the "primary" provider (providers[0]) or its model
   * changes. AppShell uses this to drive the ``<StatusRack>`` Agent
   * chip so the user can see at a glance whether the picked pair can
   * call MCP tools.
   */
  onSelectionChange?: (provider: ProviderId, model: string | null) => void;
}

export function ChatPanel({ onSelectionChange }: ChatPanelProps = {}) {
  const { t } = useT();
  const [mode, setMode] = useState<ResponseMode>("solo");
  const [providers, setProviders] = useState<ProviderId[]>(["anthropic"]);
  const [modelByProvider, setModelByProvider] =
    useState<Record<ProviderId, string>>(DEFAULT_MODELS);

  // Lift (primary provider, its model) to the parent whenever it
  // changes — the header's StatusRack uses it to derive tool-call
  // capability. We debounce via effect so the parent sees coherent
  // pairs instead of intermediate "provider changed but model hasn't
  // been rehydrated" frames.
  useEffect(() => {
    const primary = providers[0];
    if (!primary) return;
    const model = modelByProvider[primary] ?? DEFAULT_MODELS[primary] ?? null;
    onSelectionChange?.(primary, model);
  }, [providers, modelByProvider, onSelectionChange]);
  const [availability, setAvailability] = useState<Record<ProviderId, boolean>>({
    anthropic: true,
    google: true,
    openai: true,
    ollama: !HIDE_OLLAMA,
  });
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // Which of the 53 trained models the user wants the chat to refer
  // to. ``__auto__`` = defer to the ensemble; otherwise we prepend a
  // system hint so the LLM biases its epi.forecast calls / references
  // toward the picked model. See ForecastModelPicker.tsx for rationale.
  const [forecastModel, setForecastModel] = useState<string>(
    FORECAST_MODEL_AUTO,
  );
  // Consultation persona (Q2/Q5). ``__general__`` = no extra framing;
  // anything else prepends a short system prompt from lib/personas.ts
  // to anchor the LLM's tone (epi / model / simulation / clinical).
  // NOT a model-routing control — the underlying provider and 53-model
  // tournament are unaffected; only the interpretive frame changes.
  const [persona, setPersona] = useState<string>(PERSONA_GENERAL);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // ── Per-turn actions (2026-04-22) ────────────────────────────────
  //
  // Every turn bubble carries a hover-action menu: Retry / Edit /
  // Delete / Branch / Copy. Semantics:
  //
  //   · **Retry on a user turn**  — remove that user + every turn
  //     after, prefill the composer with the text, auto-send.
  //   · **Retry on an assistant turn** — same, but using the paired
  //     user turn one slot above. This is how "regenerate the reply"
  //     works in Claude/ChatGPT.
  //   · **Edit on a user turn**   — remove turn + everything after,
  //     prefill the composer, do NOT auto-send (lets the user tweak).
  //   · **Edit on an assistant turn** — inline prose-edit the reply
  //     locally (no re-send). Stored back into the turn's text map so
  //     the user can annotate / correct a draft without regenerating.
  //   · **Delete** — remove just that turn + its counterpart (user+
  //     assistant pair). Confirms before removing.
  //   · **Branch** — copy every turn up to AND including the clicked
  //     one into a new session and switch to it. Lets the user fork
  //     the conversation at a specific point.
  //   · **Copy** — clipboard-copy the message text. Cheap, universally
  //     useful for prompts and replies alike.
  const [editingTurnId, setEditingTurnId] = useState<string | null>(null);
  const [editingDraft, setEditingDraft] = useState<string>("");

  // `send` is defined later but actions need a reference. Stash it
  // through a ref so callbacks always see the latest closure.
  const sendRef = useRef<(() => Promise<void>) | null>(null);

  const userTextForAsst = useCallback(
    (asstId: string): string | null => {
      const idx = turnsFindIndex(turns, asstId);
      if (idx <= 0) return null;
      const u = turns[idx - 1];
      return u.role === "user" ? u.content ?? null : null;
    },
    [turns],
  );

  /**
   * Drop this turn *and every turn after it*, then optionally prefill
   * the composer. Centralises the retry/edit-on-user branches so we
   * don't duplicate the slicing logic.
   */
  const truncateFrom = useCallback(
    (turnId: string, opts: { prefill?: string; send?: boolean } = {}) => {
      setTurns((ts) => {
        const idx = ts.findIndex((x) => x.id === turnId);
        if (idx < 0) return ts;
        return ts.slice(0, idx);
      });
      if (opts.prefill != null) setInput(opts.prefill);
      if (opts.send) queueMicrotask(() => void sendRef.current?.());
    },
    [],
  );

  const retryTurn = useCallback(
    (turnId: string) => {
      const idx = turnsFindIndex(turns, turnId);
      if (idx < 0) return;
      const t = turns[idx];
      const userText =
        t.role === "user"
          ? t.content ?? ""
          : userTextForAsst(turnId) ?? "";
      if (!userText) return;
      const firstToRemove =
        t.role === "user" ? t.id : turns[idx - 1]?.id ?? t.id;
      truncateFrom(firstToRemove, { prefill: userText, send: true });
    },
    [turns, truncateFrom, userTextForAsst],
  );

  const editTurn = useCallback(
    (turnId: string) => {
      const idx = turnsFindIndex(turns, turnId);
      if (idx < 0) return;
      const t = turns[idx];
      if (t.role === "user") {
        truncateFrom(t.id, { prefill: t.content ?? "" });
      } else {
        // Inline-edit an assistant reply. We snapshot the current text
        // into the composer-less editor; save writes it back via
        // saveInlineEdit() below.
        const firstText = Object.values(t.text ?? {})[0] ?? "";
        setEditingTurnId(t.id);
        setEditingDraft(firstText);
      }
    },
    [turns, truncateFrom],
  );

  const saveInlineEdit = useCallback(() => {
    const id = editingTurnId;
    if (!id) return;
    const draft = editingDraft;
    setTurns((ts) =>
      ts.map((t) => {
        if (t.id !== id) return t;
        const providers = Object.keys(t.text ?? {}) as ProviderId[];
        const nextText: Partial<Record<ProviderId, string>> = { ...(t.text ?? {}) };
        if (providers.length === 0) {
          nextText.anthropic = draft;
        } else {
          nextText[providers[0]] = draft;
        }
        return { ...t, text: nextText };
      }),
    );
    setEditingTurnId(null);
    setEditingDraft("");
  }, [editingTurnId, editingDraft]);

  const cancelInlineEdit = useCallback(() => {
    setEditingTurnId(null);
    setEditingDraft("");
  }, []);

  const deleteTurn = useCallback(
    (turnId: string) => {
      const idx = turnsFindIndex(turns, turnId);
      if (idx < 0) return;
      const ok =
        typeof window !== "undefined"
          ? window.confirm(
              t("confirmDeleteMsg") ||
                "Delete this message?",
            )
          : true;
      if (!ok) return;
      const t0 = turns[idx];
      // Remove both the clicked turn and its pair (the adjacent
      // user/assistant turn from the same request). For the common
      // case the pair is at idx±1.
      const pairIdx =
        t0.role === "user"
          ? idx + 1 < turns.length && turns[idx + 1].role === "assistant"
            ? idx + 1
            : -1
          : idx - 1 >= 0 && turns[idx - 1].role === "user"
            ? idx - 1
            : -1;
      const dropIds = new Set([t0.id]);
      if (pairIdx >= 0) dropIds.add(turns[pairIdx].id);
      setTurns((ts) => ts.filter((x) => !dropIds.has(x.id)));
    },
    [turns, t],
  );

  const copyTurn = useCallback(
    (turnId: string) => {
      const idx = turnsFindIndex(turns, turnId);
      if (idx < 0) return;
      const t = turns[idx];
      const text =
        t.role === "user"
          ? t.content ?? ""
          : Object.values(t.text ?? {}).join("\n\n");
      if (!text) return;
      void navigator.clipboard?.writeText(text);
    },
    [turns],
  );

  // `branchTurn` is declared lower (after `sanitisedProviders` /
  // `mode` exist as stable references) — we hold a ref here so the UI
  // can wire up the callback via the same shape as the other actions
  // without hoisting half the file.
  const branchTurnRef = useRef<(turnId: string) => void>(() => undefined);

  // ── History wiring ───────────────────────────────────────────────
  //
  // The store is the source of truth for which chat session is open.
  // Two events drive turns-list updates from the store:
  //   1. User opens an existing session in the sidebar → active
  //      session id changes, we hydrate ``turns`` from the persisted
  //      messages.
  //   2. User clicks "New chat" → active id becomes null, we clear
  //      turns and let the next Send create a fresh session.
  //
  // ``syncedSessionIdRef`` is our "I've already synced this id" flag.
  // Without it, the append dispatched by ``persistMessage`` (which
  // lazy-creates a session during Send) would also trigger this effect
  // and wipe the in-flight streaming turn.
  const store = useSessionStore();
  const syncedSessionIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (syncedSessionIdRef.current === store.activeSessionId) return;
    syncedSessionIdRef.current = store.activeSessionId;
    // Whatever was streaming is no longer relevant — cancel it.
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setErr(null);
    if (store.activeSessionId == null) {
      setTurns([]);
    } else {
      setTurns(messagesToTurns(store.activeMessages));
      // Restore mode + provider selection the user last used on this
      // session, so parallel chats don't quietly flip to solo-Claude.
      const sess = store.sessions.find((s) => s.id === store.activeSessionId);
      if (sess?.mode && RESPONSE_MODES.includes(sess.mode as ResponseMode)) {
        setMode(sess.mode as ResponseMode);
      }
      if (sess?.provider_id) {
        setProviders([sess.provider_id as ProviderId]);
      }
    }
  }, [store.activeSessionId, store.activeMessages, store.sessions]);

  // Ask the server which providers actually have keys **and** which
  // one it recommends for the current environment. The server-side
  // ``detectEnvironment()`` (see ``lib/environment.ts``) already
  // decided: local dev → Ollama first, cloud prod → Claude → GPT →
  // Gemini. We honour that decision unless the user has manually
  // picked something else that is still available.
  const [deployMode, setDeployMode] = useState<"local" | "cloud" | null>(null);
  const [initialProviderApplied, setInitialProviderApplied] = useState(false);
  useEffect(() => {
    const ctl = new AbortController();
    void (async () => {
      try {
        const r = await fetch("/api/providers", { signal: ctl.signal });
        if (!r.ok) return;
        const body = (await r.json()) as {
          providers: Array<{
            id: ProviderId;
            available: boolean;
            models: string[];
          }>;
          environment?: {
            mode: "local" | "cloud";
            recommended: ProviderId;
            fallbackOrder: ProviderId[];
          };
        };
        const next: Record<ProviderId, boolean> = {
          anthropic: false,
          google: false,
          openai: false,
          ollama: false,
        };
        for (const p of body.providers) next[p.id] = p.available;
        setAvailability(next);
        if (body.environment) setDeployMode(body.environment.mode);

        // First load: apply the server's recommendation verbatim if
        // nothing has been manually chosen yet. After that, only auto-
        // switch when the currently-selected provider becomes
        // unavailable — respect the user's intent.
        setProviders((cur) => {
          if (!initialProviderApplied && body.environment) {
            return [body.environment.recommended];
          }
          const keep = cur.filter((id) => next[id]);
          if (keep.length) return keep;
          const fallback =
            body.environment?.fallbackOrder.find((id) => next[id]) ??
            (Object.keys(next) as ProviderId[]).find((id) => next[id]);
          return fallback ? [fallback] : cur;
        });
        setInitialProviderApplied(true);
      } catch {
        // leave defaults as-is if the probe fails
      }
    })();
    return () => ctl.abort();
  }, [initialProviderApplied]);

  useEffect(() => {
    // Auto-scroll to the bottom when new text arrives.
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns]);

  // Allow MapPanel and SuggestionChips to prefill the composer by
  // dispatching a ``frame-d:prefill`` event on window.
  useEffect(() => {
    function onPrefill(ev: Event) {
      const detail = (ev as CustomEvent<{ prompt?: string }>).detail;
      if (!detail?.prompt) return;
      setInput((v) => (v ? v : detail.prompt ?? ""));
    }
    window.addEventListener("frame-d:prefill", onPrefill as EventListener);
    return () =>
      window.removeEventListener("frame-d:prefill", onPrefill as EventListener);
  }, []);

  // ── Playback auto-context (B2, 2026-04-22) ───────────────────────
  //
  // TransmissionPlayer broadcasts ``frame-d:playback-state`` every tick
  // (via AppShell's handleTransmissionFrame). We stash the latest frame
  // in a ref so send() can inject a short system note like:
  //
  //     The current simulation frame is Day 63 / 180 (Week 9 / 26,
  //     peak expected in 4 weeks). Cumulative infections 18,240,
  //     active districts 11/25, hottest: 강남구 · 서초구 · 송파구.
  //     When the user asks "이번 주는 어때?" they mean THIS week.
  //
  // A ref (rather than state) keeps the ChatPanel from re-rendering 26
  // times during playback — nothing on screen depends on the playback
  // state until the user actually presses Send. Each Send reads the ref
  // once and prepends the note to the message array.
  const playbackStateRef = useRef<{
    weekIdx: number;
    weekLabel?: string;
    dayIdx: number;
    totalDays: number;
    peakWeek: number;
    cumulative: number;
    activeGuCount: number;
    topGus: string[];
    playing: boolean;
    stampedAt: number;
  } | null>(null);
  useEffect(() => {
    function onPlaybackState(ev: Event) {
      const detail = (ev as CustomEvent<{
        weekIdx: number;
        weekLabel?: string;
        dayIdx: number;
        totalDays: number;
        peakWeek: number;
        cumulative: number;
        activeGuCount: number;
        topGus: string[];
        playing: boolean;
      }>).detail;
      if (!detail) return;
      playbackStateRef.current = {
        ...detail,
        stampedAt: performance.now(),
      };
    }
    window.addEventListener(
      "frame-d:playback-state",
      onPlaybackState as EventListener,
    );
    return () =>
      window.removeEventListener(
        "frame-d:playback-state",
        onPlaybackState as EventListener,
      );
  }, []);

  const visibleProviders = useMemo<ProviderId[]>(() => {
    // Hide Ollama when the server has declared cloud mode — there's
    // no daemon on Vercel and showing a disabled row only confuses
    // the audience. The env-var override is kept for manual QA.
    const hideOllama = HIDE_OLLAMA || deployMode === "cloud";
    return (["anthropic", "google", "openai", "ollama"] as ProviderId[]).filter(
      (p) => !(hideOllama && p === "ollama"),
    );
  }, [deployMode]);

  const sanitisedProviders = useMemo<ProviderId[]>(() => {
    if (mode === "solo") return providers.slice(0, 1);
    return providers;
  }, [mode, providers]);

  /**
   * Branch = fork the conversation at ``turnId``. Copies every turn
   * up to and including the clicked one into a brand-new session,
   * switches the sidebar to it, and leaves the composer empty so the
   * user can continue in the fork. Persistence is best-effort — if
   * the store is in disabled mode we keep the fork in memory.
   */
  const branchTurn = useCallback(
    async (turnId: string) => {
      const idx = turnsFindIndex(turns, turnId);
      if (idx < 0) return;
      const keep = turns.slice(0, idx + 1);
      const messages: Array<{
        role: "user" | "assistant";
        content: string;
        provider_id: string | null;
      }> = [];
      for (const x of keep) {
        if (x.role === "user" && x.content) {
          messages.push({ role: "user", content: x.content, provider_id: null });
        } else if (x.role === "assistant") {
          for (const [pid, content] of Object.entries(x.text ?? {})) {
            if (content) {
              messages.push({ role: "assistant", content, provider_id: pid });
            }
          }
        }
      }
      try {
        if (store.disabled) {
          setTurns(keep);
          return;
        }
        const created = await store.newSession({
          title: (keep.find((x) => x.role === "user")?.content ?? "Branch").slice(0, 60),
          mode,
          provider_id: sanitisedProviders[0] ?? null,
        });
        for (const m of messages) {
          await store.persistMessage(m);
        }
        await store.openSession(created.id);
      } catch (e) {
        console.warn("[ChatPanel] branch persistence failed:", e);
        setTurns(keep);
      }
    },
    [turns, store, mode, sanitisedProviders],
  );
  useEffect(() => {
    branchTurnRef.current = (id) => void branchTurn(id);
  }, [branchTurn]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
  }, []);

  const send = useCallback(async () => {
    const rawText = input.trim();
    if (!rawText || busy) return;
    setErr(null);

    // Sprint 2026-05-06 (#14): persona command router. `!역학 상담 ...` 같은
    // prefix 면 그 turn 부터 해당 persona 로 specialize. cleanText 가 비면
    // 사용자 의도 모호 — 그대로 placeholder 으로 보내고 다음 turn 에 적용.
    const parsed = parsePersonaCommand(rawText);
    if (parsed.persona) {
      setPersona(parsed.persona);
    }
    const text = parsed.cleanText || rawText;

    // Ensure a session exists BEFORE we touch the turns list. Doing
    // the create synchronously (and claiming the sync ref) prevents
    // the store sync effect from racing with the streaming state.
    let sessionId: string | null = store.activeSessionId;
    if (sessionId == null && !store.disabled) {
      try {
        const created = await store.newSession({
          provider_id: sanitisedProviders[0] ?? null,
          mode,
        });
        sessionId = created.id;
        syncedSessionIdRef.current = created.id;
      } catch {
        // Persistence is best-effort — fall through without a session.
        sessionId = null;
      }
    }

    const userTurn = freshTurn({
      role: "user",
      content: text,
      status: "done",
    });
    const asstTurn = freshTurn({
      role: "assistant",
      mode,
      providers: sanitisedProviders,
    });

    setTurns((ts) => [...ts, userTurn, asstTurn]);
    setInput("");
    setBusy(true);

    // Persist the user turn in the background — we don't need the row
    // back before the SSE starts, and a failed persist shouldn't block
    // the chat from answering.
    if (sessionId != null && !store.disabled) {
      void store.persistMessage(
        {
          role: "user",
          content: text,
          provider_id: null,
        },
        {
          provider_id: sanitisedProviders[0] ?? null,
          mode,
        },
      );
    }

    const controller = new AbortController();
    abortRef.current = controller;

    // Inject a one-shot system hint if the user has picked a specific
    // forecasting model (rather than "Auto"). LLMs with tool access can
    // forward the hint through ``epi.forecast`` ``model_name``; LLMs
    // without tools just mention the model in their explanation.
    const systemHint =
      forecastModel !== FORECAST_MODEL_AUTO
        ? [
            {
              role: "system" as const,
              content:
                `The user has selected "${forecastModel}" as the preferred forecasting model ` +
                `from the post_E v22.6 registry of 53 trained models. When calling epi.forecast, ` +
                `pass this name via the model_name argument if supported. In prose, reference ` +
                `this model's point estimate and PI, and note when the ensemble would differ.`,
            },
          ]
        : [];

    // Persona framing (Q2/Q5). Prepended BEFORE the model hint so the
    // persona's role definition frames the whole reply, then the model
    // hint narrows the specific citations. ``general`` yields an empty
    // prompt so we skip the push.
    const personaDef = getPersonaById(persona);
    const personaHint =
      personaDef && personaDef.system_prompt
        ? [
            {
              role: "system" as const,
              content: personaDef.system_prompt,
            },
          ]
        : [];

    // Playback-state context (B2). If the user is mid-simulation, fold
    // the current playhead position into the system preamble so the LLM
    // can ground answers like "이번 주 강남은 어때?" in the actual sim
    // state instead of hallucinating. We only inject when the last
    // broadcast is fresh (< 30s old) so stale context from an earlier
    // run doesn't bias a new question.
    const pb = playbackStateRef.current;
    const pbFresh =
      pb != null && performance.now() - pb.stampedAt < 30_000;
    const playbackHint =
      pbFresh && pb
        ? [
            {
              role: "system" as const,
              content:
                `Simulation playback state (for grounding "이번 주 / this week" style ` +
                `questions): Day ${pb.dayIdx}/${pb.totalDays}, Week ${pb.weekIdx + 1}` +
                (pb.weekLabel ? ` (${pb.weekLabel})` : "") +
                `, peak week = ${pb.peakWeek + 1} (${pb.peakWeek - pb.weekIdx >= 0 ? `${pb.peakWeek - pb.weekIdx}w away` : `${pb.weekIdx - pb.peakWeek}w past`}). ` +
                `Cumulative new infections = ${pb.cumulative.toLocaleString()}, ` +
                `active districts = ${pb.activeGuCount}/25, ` +
                `hottest districts this week = ${pb.topGus.join(", ") || "(none)"}. ` +
                `Playing = ${pb.playing}. Treat "현재 / 이번 주 / this week" as referring ` +
                `to Week ${pb.weekIdx + 1}. Do NOT claim a different week unless the user ` +
                `explicitly names one.`,
            },
          ]
        : [];

    // Sprint 2026-05-06 (#15): today-grounded reasoning. Always inject the
    // current date so the LLM grounds influenza season phase / antiviral
    // resistance / vaccine effectiveness reasoning as-of-now.
    const todayHint = [
      {
        role: "system" as const,
        content:
          `Today's date is ${new Date().toISOString().slice(0, 10)}. ` +
          `Ground epidemiological reasoning (influenza season phase, ` +
          `antiviral resistance reports, vaccine effectiveness) as of this date.`,
      },
    ];

    const body = {
      mode,
      messages: [
        ...todayHint,
        ...personaHint,
        ...playbackHint,
        ...systemHint,
        ...turns
          .filter((t) => t.role === "user" && t.content)
          .map((t) => ({ role: "user" as const, content: t.content ?? "" })),
        { role: "user" as const, content: text },
      ],
      providers: sanitisedProviders.map((id) => ({
        id,
        model: modelByProvider[id] ?? DEFAULT_MODELS[id],
      })),
      synthesiser:
        mode === "synthesis" ? sanitisedProviders[0] : undefined,
    };

    const path = mode === "solo" ? "/api/chat" : "/api/chat/parallel";
    let resp: Response;
    try {
      resp = await fetch(path, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (e) {
      // Network failure before any bytes returned — mark the assistant
      // turn as errored so the user gets an actionable retry/edit
      // button instead of an orphan "…" streaming bubble.
      const msg = e instanceof Error ? e.message : String(e);
      setBusy(false);
      setErr(msg);
      setTurns((ts) =>
        ts.map((x) =>
          x.id === asstTurn.id ? { ...x, status: "error", error: msg } : x,
        ),
      );
      return;
    }
    if (!resp.ok) {
      setBusy(false);
      const txt = await resp.text().catch(() => "");
      const msg = `HTTP ${resp.status}: ${txt.slice(0, 200)}`;
      setErr(msg);
      setTurns((ts) =>
        ts.map((x) =>
          x.id === asstTurn.id ? { ...x, status: "error", error: msg } : x,
        ),
      );
      return;
    }

    const startTs = performance.now();
    // Track per-provider in-flight tool calls by id so we can attach
    // the elapsed time when the result arrives.
    const inFlight: Record<string, number> = {};

    try {
      for await (const evt of readSSE(resp)) {
        const providerId = (evt.providerId as ProviderId) ?? sanitisedProviders[0];
        setTurns((ts) =>
          ts.map((t) => {
            if (t.id !== asstTurn.id) return t;
            const text = { ...(t.text ?? {}) };
            const trace = { ...(t.trace ?? {}) };
            const perProv = [...(trace[providerId] ?? [])];
            const notices = { ...(t.notices ?? {}) };
            if (evt.type === "text") {
              text[providerId] =
                (text[providerId] ?? "") + String(evt.delta ?? "");
            } else if (evt.type === "tool_call") {
              const call = evt.call as ToolCall;
              inFlight[call.id] = performance.now();
              perProv.push({
                call,
                startedAt: performance.now() - startTs,
              });
              trace[providerId] = perProv;
            } else if (evt.type === "tool_result") {
              const result = evt.result as ToolResult;
              const idx = perProv.findIndex(
                (x) => x.call.id === result.toolCallId,
              );
              if (idx >= 0) {
                const started = inFlight[result.toolCallId];
                const elapsed =
                  started != null ? performance.now() - started : undefined;
                perProv[idx] = { ...perProv[idx], result, elapsedMs: elapsed };
              }
              trace[providerId] = perProv;
            } else if (evt.type === "status") {
              // Provider emitted a status advisory (tools-disabled,
              // max-hops, MCP bridge unavailable). Accumulate per
              // provider so multi-mode replies don't overwrite each
              // other's notices.
              const msg = String((evt as { message?: string }).message ?? "").trim();
              if (msg) {
                const level =
                  ((evt as { level?: "info" | "warn" | "error" }).level) ?? "info";
                const existing = notices[providerId] ?? [];
                notices[providerId] = [...existing, { level, message: msg }];
              }
            }
            return { ...t, text, trace, notices };
          }),
        );
      }

      setTurns((ts) =>
        ts.map((t) => (t.id === asstTurn.id ? { ...t, status: "done" } : t)),
      );

      // Persist each provider's final reply. In parallel mode this
      // writes N sibling rows under the same turn_idx (the server
      // auto-shares the index for role="assistant"). We fire them in
      // parallel and ignore failures — chat UI already shows the text;
      // a missing DB row just means it won't reappear on reload.
      if (sessionId != null && !store.disabled) {
        for (const id of sanitisedProviders) {
          const reply = (turnsRefLast(asstTurn.id, id) ?? "").replace(/\s+$/, "");
          if (!reply) continue;
          void store.persistMessage({
            role: "assistant",
            content: reply,
            provider_id: id,
          });
        }
      }

      // Validity check per provider reply — fire-and-forget.
      for (const id of sanitisedProviders) {
        const reply =
          (turnsRefLast(asstTurn.id, id) ?? "")
            .replace(/\s+$/, "");
        if (!reply) continue;
        checkReply(reply, controller.signal)
          .then((badge) => {
            setTurns((ts) =>
              ts.map((t) => {
                if (t.id !== asstTurn.id) return t;
                return {
                  ...t,
                  validity: { ...(t.validity ?? {}), [id]: badge },
                };
              }),
            );
          })
          .catch(() => void 0);
      }
    } catch (e) {
      const aborted = controller.signal.aborted;
      setTurns((ts) =>
        ts.map((t) =>
          t.id === asstTurn.id
            ? {
                ...t,
                status: aborted ? "aborted" : "error",
                error: aborted ? undefined : e instanceof Error ? e.message : String(e),
              }
            : t,
        ),
      );
    } finally {
      setBusy(false);
      abortRef.current = null;
    }

    // Helper closes over the freshest turn state via the state updater
    // pattern; we cache last-seen text here.
    function turnsRefLast(turnId: string, providerId: ProviderId): string | null {
      // We can't read React state directly in this closure — instead,
      // resolve via DOM later. Simpler: reassign inside the setter.
      let snapshot: string | null = null;
      setTurns((ts) => {
        const found = ts.find((t) => t.id === turnId);
        snapshot = found?.text?.[providerId] ?? null;
        return ts;
      });
      return snapshot;
    }
  }, [busy, input, mode, modelByProvider, sanitisedProviders, store, turns, forecastModel, persona]);

  // Keep the ref in sync with the latest `send` closure so
  // retryFailedTurn fires the current version.
  useEffect(() => {
    sendRef.current = send;
  }, [send]);

  return (
    <section className="flex h-full flex-col gap-2 bg-slate-950 p-2">
      <header className="sticky top-0 z-10 flex flex-wrap items-center gap-2 rounded-md border border-slate-800 bg-slate-900/80 p-2 backdrop-blur">
        <div className="flex items-center gap-1">
          <Select
            label={t("mode")}
            value={mode}
            onChange={(e) => setMode(e.target.value as ResponseMode)}
            className="w-32"
          >
            {RESPONSE_MODES.map((m) => (
              <option key={m} value={m}>
                {MODE_LABELS[m] ?? m}
              </option>
            ))}
          </Select>
          <HelpIcon label={t("helpMode")} content={t("helpMode")} side="bottom" />
        </div>
        <div className="flex items-center gap-1">
          <ProviderPicker
            selected={sanitisedProviders}
            onChange={setProviders}
            hidden={HIDE_OLLAMA ? ["ollama"] : []}
            single={mode === "solo"}
            availability={availability}
          />
          <HelpIcon label={t("helpProvider")} content={t("helpProvider")} side="bottom" />
        </div>
        <div className="flex items-center gap-1">
          <ForecastModelPicker
            value={forecastModel}
            onChange={setForecastModel}
          />
          <HelpIcon
            label={t("helpForecastModel")}
            content={t("helpForecastModel")}
            side="bottom"
          />
        </div>
        {/* Sprint 2026-05-06 (#14): PersonaPicker 제거 — 사용자 명시
            "다 할 수 있어야해". 특화는 chat 명령어 (!역학 / !임상 /
            !모델 / !시뮬레이션 / !일반) 으로. 현재 persona 는 hidden
            state 으로만 유지. */}
        {busy ? (
          <Button size="sm" variant="danger" onClick={stop} className="ml-auto">
            Stop
          </Button>
        ) : null}
      </header>

      <div
        ref={scrollRef}
        role="log"
        aria-live="polite"
        className="flex-1 overflow-y-auto rounded-md border border-slate-800 bg-slate-950/50 p-3"
      >
        {turns.length === 0 ? (
          <EmptyState onPick={(p) => setInput((v) => (v ? v : p))} />
        ) : (
          <ul className="flex flex-col gap-3">
            {turns.map((x) => {
              const actions: TurnActionHandlers = {
                onRetry: () => retryTurn(x.id),
                onEdit: () => editTurn(x.id),
                onDelete: () => deleteTurn(x.id),
                onBranch: () => branchTurnRef.current(x.id),
                onCopy: () => copyTurn(x.id),
              };
              return (
                <li key={x.id}>
                  {x.role === "user" ? (
                    <UserBubble
                      text={x.content ?? ""}
                      actions={actions}
                      editing={editingTurnId === x.id}
                      draft={editingDraft}
                      onDraftChange={setEditingDraft}
                      onSaveEdit={saveInlineEdit}
                      onCancelEdit={cancelInlineEdit}
                    />
                  ) : (
                    <AssistantBubble
                      turn={x}
                      visibleProviders={visibleProviders}
                      actions={actions}
                      editing={editingTurnId === x.id}
                      draft={editingDraft}
                      onDraftChange={setEditingDraft}
                      onSaveEdit={saveInlineEdit}
                      onCancelEdit={cancelInlineEdit}
                    />
                  )}
                </li>
              );
            })}
          </ul>
        )}
        {err ? (
          <div className="mt-2 rounded-md border border-red-700 bg-red-950/60 p-2 text-xs text-red-300">
            {err}
          </div>
        ) : null}
      </div>

      <div className="rounded-md border border-slate-800 bg-slate-900/70 p-2">
        <PlaybackBadge />
        <SuggestionChips onPick={(p) => setInput((v) => (v ? v : p))} />
        <div className="mt-2 flex items-end gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              } else if (e.key === "Escape") {
                stop();
              }
            }}
            placeholder={t("askPlaceholder")}
            rows={2}
            className="flex-1"
          />
          <Button
            onClick={() => void send()}
            disabled={busy || !input.trim()}
            size="md"
          >
            {busy ? t("sending") : t("send")}
          </Button>
        </div>
      </div>
    </section>
  );
}

/**
 * New-chat empty state — replaces the hardcoded Korean placeholder.
 * Sprint 2026-05-07: TrainedModelsCard 제거 (사용자 명시 '관심 없음').
 * 단순 hero text + SuggestionChips 만 표시.
 */
function EmptyState({ onPick }: { onPick: (prompt: string) => void }) {
  const { t } = useT();
  return (
    <div className="flex flex-col gap-3 py-4">
      <div className="text-center">
        <div className="text-sm font-medium text-slate-200">
          {t("startNewChat")}
        </div>
        <div className="mt-1 text-[11px] text-slate-500">
          {t("startNewChatHint")}
        </div>
      </div>
      <div className="rounded-md border border-slate-800 bg-slate-900/30 p-2">
        <SuggestionChips onPick={onPick} collapsible={false} />
      </div>
    </div>
  );
}

function turnsFindIndex(turns: Turn[], id: string): number {
  for (let i = 0; i < turns.length; i++) if (turns[i].id === id) return i;
  return -1;
}

/**
 * Small pill shown above the composer while the TransmissionPlayer is
 * running. Makes the "current Day / Week / top districts" visible so the
 * user knows their question will be auto-grounded to THIS frame, even
 * though no prose has been streamed into the chat yet.
 *
 * Subscribes to the ``frame-d:playback-state`` event directly (the same
 * source the send() hint reads) rather than lifting the state up through
 * ChatPanel — keeps the 4 Hz tick from re-rendering the whole chat tree
 * during a long playback.
 */
export function PlaybackBadge() {
  const { t } = useT();
  const [state, setState] = useState<{
    weekIdx: number;
    weekLabel?: string;
    dayIdx: number;
    totalDays: number;
    peakWeek: number;
    topGus: string[];
    playing: boolean;
  } | null>(null);
  useEffect(() => {
    function onPlaybackState(ev: Event) {
      const d = (ev as CustomEvent<typeof state>).detail;
      setState(d);
    }
    window.addEventListener(
      "frame-d:playback-state",
      onPlaybackState as EventListener,
    );
    return () =>
      window.removeEventListener(
        "frame-d:playback-state",
        onPlaybackState as EventListener,
      );
  }, []);
  if (!state) return null;
  const weekNum = state.weekIdx + 1;
  const peakDelta = state.peakWeek - state.weekIdx;
  return (
    <div
      className="mb-1 flex flex-wrap items-center gap-2 rounded-md border border-sky-800/60 bg-sky-950/40 px-2 py-1 text-[11px] text-sky-100"
      role="status"
      aria-live="polite"
      title={t("playbackBadgeHint")}
    >
      <span
        aria-hidden="true"
        className={[
          "inline-block h-1.5 w-1.5 rounded-full",
          state.playing ? "bg-rose-400 animate-pulse" : "bg-sky-400",
        ].join(" ")}
      />
      <span className="font-medium tabular-nums">
        Day {state.dayIdx}/{state.totalDays}
      </span>
      <span className="text-slate-400">·</span>
      <span className="tabular-nums">
        W{weekNum}
        {state.weekLabel ? ` (${state.weekLabel})` : ""}
      </span>
      {peakDelta !== 0 ? (
        <>
          <span className="text-slate-400">·</span>
          <span className="text-amber-200">
            {peakDelta > 0
              ? t("hudPeakInWeeks", { n: peakDelta })
              : t("hudPastPeakWeeks", { n: -peakDelta })}
          </span>
        </>
      ) : (
        <span className="rounded bg-rose-600/50 px-1 text-rose-50">
          {t("hudAtPeak")}
        </span>
      )}
      {state.topGus.length > 0 ? (
        <>
          <span className="text-slate-400">·</span>
          <span className="text-slate-300">{state.topGus.join(" · ")}</span>
        </>
      ) : null}
      <span className="ml-auto text-[10px] italic text-slate-400">
        {t("playbackBadgeAuto")}
      </span>
    </div>
  );
}

/**
 * Shape shared by every message bubble — makes the hover-action menu
 * a single implementation instead of branching twice in JSX.
 */
export interface TurnActionHandlers {
  onRetry: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onBranch: () => void;
  onCopy: () => void;
}

/**
 * Floating action menu rendered on every bubble. Shows on hover/focus
 * via Tailwind's ``group-hover:opacity-100`` + ``group-focus-within``
 * so keyboard users can tab to actions too.
 *
 *   ↻ retry    ✎ edit    🗑 delete    ⎇ branch    ⧉ copy
 */
function TurnActions({
  actions,
  tone = "light",
}: {
  actions: TurnActionHandlers;
  /** ``dark`` = bubble has a dark fill (assistant); ``light`` = user (sky). */
  tone?: "dark" | "light";
}) {
  const { t } = useT();
  const base =
    tone === "dark"
      ? "border-slate-700 bg-slate-900/90 text-slate-200 hover:border-sky-500/60 hover:bg-slate-800 hover:text-sky-100"
      : "border-sky-800 bg-slate-900/90 text-sky-100 hover:border-sky-500/70 hover:bg-slate-800";
  const btn = [
    "inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[12px] leading-none transition-colors",
    base,
  ].join(" ");
  return (
    <div
      className="mt-1 flex flex-wrap items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
      role="toolbar"
      aria-label="Message actions"
    >
      <button type="button" onClick={actions.onRetry} className={btn} title={t("retry")}>
        <span aria-hidden="true">↻</span>
        <span>{t("retry")}</span>
      </button>
      <button type="button" onClick={actions.onEdit} className={btn} title={t("edit")}>
        <span aria-hidden="true">✎</span>
        <span>{t("edit")}</span>
      </button>
      <button type="button" onClick={actions.onDelete} className={btn} title={t("deleteMsg")}>
        <span aria-hidden="true">🗑</span>
        <span>{t("deleteMsg")}</span>
      </button>
      <button type="button" onClick={actions.onBranch} className={btn} title={t("branchMsg")}>
        <span aria-hidden="true">⎇</span>
        <span>{t("branchMsg")}</span>
      </button>
      <button type="button" onClick={actions.onCopy} className={btn} title={t("copyMsg")}>
        <span aria-hidden="true">⧉</span>
        <span>{t("copyMsg")}</span>
      </button>
    </div>
  );
}

/**
 * Minimal inline editor for the "✎ Edit on an assistant reply" case.
 * Not a Markdown editor — just a plain textarea with Save/Cancel —
 * because the underlying provider output is freeform prose and the
 * user's goal is local annotation, not re-prompting. (For user-turn
 * edits we reuse the main composer, not this helper.)
 */
function InlineEditor({
  draft,
  onChange,
  onSave,
  onCancel,
}: {
  draft: string;
  onChange: (v: string) => void;
  onSave: () => void;
  onCancel: () => void;
}) {
  const { t } = useT();
  return (
    <div className="flex flex-col gap-1.5">
      <textarea
        value={draft}
        onChange={(e) => onChange(e.target.value)}
        rows={Math.min(12, Math.max(3, draft.split("\n").length + 1))}
        className="w-full resize-y rounded-md border border-slate-700 bg-slate-950/70 p-2 text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
      />
      <div className="flex justify-end gap-1.5 text-[12px]">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-slate-300 hover:bg-slate-800"
        >
          {t("cancelMsg")}
        </button>
        <button
          type="button"
          onClick={onSave}
          className="rounded-md border border-sky-700 bg-sky-900/60 px-2 py-1 text-sky-100 hover:bg-sky-800"
        >
          {t("saveMsg")}
        </button>
      </div>
    </div>
  );
}

function UserBubble({
  text,
  actions,
  editing,
  draft,
  onDraftChange,
  onSaveEdit,
  onCancelEdit,
}: {
  text: string;
  actions: TurnActionHandlers;
  editing: boolean;
  draft: string;
  onDraftChange: (v: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
}) {
  return (
    <div className="group ml-auto max-w-[85%]">
      {editing ? (
        <div className="rounded-lg border border-sky-700 bg-sky-950/50 p-2">
          <InlineEditor
            draft={draft}
            onChange={onDraftChange}
            onSave={onSaveEdit}
            onCancel={onCancelEdit}
          />
        </div>
      ) : (
        <div className="rounded-lg border border-sky-800 bg-sky-950/40 px-3 py-2 text-sm text-sky-100 whitespace-pre-wrap">
          {text}
        </div>
      )}
      <div className="flex justify-end">
        <TurnActions actions={actions} tone="light" />
      </div>
    </div>
  );
}

function AssistantBubble({
  turn,
  visibleProviders,
  actions,
  editing,
  draft,
  onDraftChange,
  onSaveEdit,
  onCancelEdit,
}: {
  turn: Turn;
  visibleProviders: ProviderId[];
  actions: TurnActionHandlers;
  editing: boolean;
  draft: string;
  onDraftChange: (v: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
}) {
  const { t } = useT();
  const providers = turn.providers ?? visibleProviders;
  const multi = providers.length > 1;
  const errored = turn.status === "error" || turn.status === "aborted";
  return (
    <div className="group flex flex-col gap-2">
      {turn.mode ? (
        <div className="text-[10px] uppercase tracking-wide text-slate-500">
          {MODE_LABELS[turn.mode] ?? turn.mode} · {providers.join(" / ")}
        </div>
      ) : null}
      <div
        className={[
          "grid gap-2",
          multi ? "md:grid-cols-2" : "",
        ].join(" ")}
      >
        {providers.map((id, pIdx) => {
          const text = turn.text?.[id] ?? "";
          const trace = turn.trace?.[id] ?? [];
          const badge = turn.validity?.[id] ?? null;
          const notices = turn.notices?.[id] ?? [];
          const canInlineEdit = pIdx === 0 && editing; // only first slot edits
          // Detect the common "this model doesn't support MCP tools"
          // advisory so we can show it with a clearer, translated
          // explanation rather than the raw provider string.
          const toolsDisabled = notices.some((n) =>
            /does not support tools/i.test(n.message),
          );
          return (
            <article
              key={id}
              className={[
                "rounded-lg border p-3",
                errored
                  ? "border-red-900/70 bg-red-950/30"
                  : "border-slate-800 bg-slate-900/60",
              ].join(" ")}
            >
              <header className="mb-1.5 flex items-center justify-between text-[11px] text-slate-400">
                <span>{PROVIDER_LABELS[id] ?? id}</span>
                <ValidityBadge result={badge} />
              </header>
              {toolsDisabled ? (
                <div
                  className="mb-2 rounded-md border border-amber-700/50 bg-amber-900/20 px-2 py-1 text-[11px] text-amber-200"
                  role="note"
                >
                  ⚠ {t("toolsDisabledNotice")}
                </div>
              ) : null}
              {notices
                .filter((n) => !/does not support tools/i.test(n.message))
                .map((n, nIdx) => (
                  <div
                    key={nIdx}
                    className={[
                      "mb-2 rounded-md border px-2 py-1 text-[11px]",
                      n.level === "error"
                        ? "border-red-800/60 bg-red-950/30 text-red-200"
                        : n.level === "warn"
                          ? "border-amber-700/50 bg-amber-900/20 text-amber-200"
                          : "border-slate-700/60 bg-slate-900/40 text-slate-300",
                    ].join(" ")}
                    role="note"
                  >
                    {n.message}
                  </div>
                ))}
              {canInlineEdit ? (
                <InlineEditor
                  draft={draft}
                  onChange={onDraftChange}
                  onSave={onSaveEdit}
                  onCancel={onCancelEdit}
                />
              ) : (
                <div className="whitespace-pre-wrap text-sm text-slate-100">
                  {text || (turn.status === "streaming" ? "…" : t("noOutput"))}
                </div>
              )}
              {trace.length ? (
                <div className="mt-2">
                  <ForecastTrace entries={trace} />
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
      {errored ? (
        <div className="text-[11px] text-red-300">
          ⚠ {t("errorOccurred")}
          {turn.error ? `: ${turn.error.slice(0, 200)}` : ""}
        </div>
      ) : null}
      <TurnActions actions={actions} tone="dark" />
    </div>
  );
}
