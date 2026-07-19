/**
 * ``use-session-store`` — React context holding the caller's chat
 * history (sessions + active-session messages + user profile) and the
 * mutations that talk to ``/api/history/*``.
 *
 * Why a context instead of prop-drilling?
 *   - HistorySidebar, SessionHeader, ChatPanel, UserBadge and the
 *     mobile drawer all need some slice of the same state. Passing
 *     ten props through AppShell got unwieldy fast.
 *   - The ChatPanel also writes into the store (on every user/assistant
 *     turn complete) — a context with dispatch is the simplest way to
 *     let a deeply-nested writer push into a sibling's reader.
 *
 * Why a reducer instead of multiple ``useState``?
 *   - We need a handful of derived operations to be atomic
 *     (e.g. "open a session" = set activeId + replace messages + clear
 *     error). Reducer makes that one action instead of three setters.
 *
 * Degradation
 *   - If Turso isn't configured (``TURSO_URL`` missing in the deploy),
 *     every /api/history/* call returns 500. The store catches the
 *     first error and puts the app in ``disabled`` mode — the sidebar
 *     renders an empty state, the ChatPanel still works but doesn't
 *     persist, and we don't pound the network with retries.
 */
"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";

import {
  appendMessage,
  createSession,
  deleteSession,
  downloadHistory,
  fetchCategories,
  fetchMe,
  fetchSession,
  fetchSessions,
  HistoryApiError,
  patchMe,
  patchSession,
  type AppendMessageArgs,
  type ListSessionsQuery,
} from "./history-client";
import type {
  CategoryCount,
  FdMessage,
  FdSession,
  FdUser,
} from "./history-db";

// ── State shape ──────────────────────────────────────────────────────

export interface SessionStoreState {
  /** Current browser identity. ``null`` until /api/history/me settles. */
  user: FdUser | null;
  /** Sessions in the sidebar's current filter view. */
  sessions: FdSession[];
  /** Category chips + counts (only non-archived). */
  categories: CategoryCount[];
  /** Sidebar filter state — drives ``refresh`` queries. */
  filter: ListSessionsQuery;
  /** Active session id, or ``null`` when the user is composing a brand-new chat. */
  activeSessionId: string | null;
  /** Cached messages for the active session (loaded on open). */
  activeMessages: FdMessage[];
  /** True while the first bootstrap (me + sessions) is in flight. */
  booting: boolean;
  /** Last error surfaced to the UI; cleared by the next successful op. */
  error: string | null;
  /**
   * ``true`` when /api/history/* is unreachable (500/503) — UI hides
   * sidebar, ChatPanel stops persisting. User can still chat; nothing
   * gets saved. Clears on the next successful call.
   */
  disabled: boolean;
}

const INITIAL_STATE: SessionStoreState = {
  user: null,
  sessions: [],
  categories: [],
  filter: { archived: false },
  activeSessionId: null,
  activeMessages: [],
  booting: true,
  error: null,
  disabled: false,
};

// ── Reducer ──────────────────────────────────────────────────────────

type Action =
  | { type: "BOOT_DONE"; user: FdUser | null; sessions: FdSession[]; categories: CategoryCount[] }
  | { type: "SET_USER"; user: FdUser }
  | { type: "SET_SESSIONS"; sessions: FdSession[] }
  | { type: "SET_CATEGORIES"; categories: CategoryCount[] }
  | { type: "SET_FILTER"; filter: ListSessionsQuery }
  | { type: "SET_ACTIVE"; id: string | null; messages?: FdMessage[] }
  | { type: "UPSERT_SESSION"; session: FdSession }
  | { type: "REMOVE_SESSION"; id: string }
  | { type: "APPEND_MESSAGE"; message: FdMessage }
  | { type: "REPLACE_MESSAGES"; messages: FdMessage[] }
  | { type: "ERROR"; message: string | null }
  | { type: "DISABLE"; disabled: boolean }
  | { type: "RESET" };

function reducer(state: SessionStoreState, action: Action): SessionStoreState {
  switch (action.type) {
    case "BOOT_DONE":
      return {
        ...state,
        user: action.user,
        sessions: action.sessions,
        categories: action.categories,
        booting: false,
        disabled: false,
        error: null,
      };
    case "SET_USER":
      return { ...state, user: action.user, disabled: false };
    case "SET_SESSIONS":
      return { ...state, sessions: action.sessions, disabled: false };
    case "SET_CATEGORIES":
      return { ...state, categories: action.categories, disabled: false };
    case "SET_FILTER":
      return { ...state, filter: { ...state.filter, ...action.filter } };
    case "SET_ACTIVE":
      return {
        ...state,
        activeSessionId: action.id,
        activeMessages: action.messages ?? [],
        error: null,
      };
    case "UPSERT_SESSION": {
      const idx = state.sessions.findIndex((s) => s.id === action.session.id);
      let sessions: FdSession[];
      if (idx === -1) {
        sessions = [action.session, ...state.sessions];
      } else {
        sessions = state.sessions.slice();
        sessions[idx] = action.session;
      }
      // Re-sort: pinned first, then updated_at desc.
      sessions.sort((a, b) => {
        if (a.pinned !== b.pinned) return b.pinned - a.pinned;
        return b.updated_at - a.updated_at;
      });
      return { ...state, sessions };
    }
    case "REMOVE_SESSION": {
      const sessions = state.sessions.filter((s) => s.id !== action.id);
      const clearActive = state.activeSessionId === action.id;
      return {
        ...state,
        sessions,
        activeSessionId: clearActive ? null : state.activeSessionId,
        activeMessages: clearActive ? [] : state.activeMessages,
      };
    }
    case "APPEND_MESSAGE":
      return {
        ...state,
        activeMessages: [...state.activeMessages, action.message],
      };
    case "REPLACE_MESSAGES":
      return { ...state, activeMessages: action.messages };
    case "ERROR":
      return { ...state, error: action.message };
    case "DISABLE":
      return { ...state, disabled: action.disabled, booting: false };
    case "RESET":
      return { ...INITIAL_STATE, booting: false };
    default:
      return state;
  }
}

// ── Context + provider ───────────────────────────────────────────────

export interface SessionStoreActions {
  /** Re-fetch sessions + categories under the current filter. */
  refresh: () => Promise<void>;
  /** Change the sidebar filter and re-query. */
  setFilter: (f: ListSessionsQuery) => Promise<void>;
  /** Open an existing session — loads its messages. */
  openSession: (id: string) => Promise<void>;
  /** Leave any open session — ChatPanel shows an empty composer. */
  closeSession: () => void;
  /** Create a fresh chat session and make it active. */
  newSession: (init?: {
    title?: string;
    category?: string | null;
    provider_id?: string | null;
    mode?: string | null;
  }) => Promise<FdSession>;
  /** Rename / recategorise / pin / archive an existing session. */
  updateSession: (
    id: string,
    patch: {
      title?: string;
      category?: string | null;
      provider_id?: string | null;
      mode?: string | null;
      pinned?: boolean;
      archived?: boolean;
    },
  ) => Promise<void>;
  /** Hard-delete a session. Cascades to messages. */
  removeSession: (id: string) => Promise<void>;
  /**
   * Persist a message against the active session. If no session is
   * active yet, creates one first using ``initIfMissing`` for the
   * default title/category/provider_id/mode. Returns the persisted row.
   */
  persistMessage: (
    msg: AppendMessageArgs,
    initIfMissing?: {
      title?: string;
      category?: string | null;
      provider_id?: string | null;
      mode?: string | null;
    },
  ) => Promise<FdMessage | null>;
  /** Update the user's display_name / label_email. */
  updateMyLabel: (patch: {
    display_name?: string | null;
    label_email?: string | null;
  }) => Promise<void>;
  /** Trigger a JSONL browser download of the full history. */
  exportAll: () => Promise<void>;
}

interface ContextShape extends SessionStoreState, SessionStoreActions {}

const SessionStoreContext = createContext<ContextShape | null>(null);

export function SessionStoreProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  // Keep a ref to the current filter so callbacks don't need to be
  // re-created every time it changes.
  const filterRef = useRef<ListSessionsQuery>(INITIAL_STATE.filter);
  filterRef.current = state.filter;

  // Centralised error handler — decides between "surface the message"
  // and "mark the whole feature as disabled".
  const handleApiError = useCallback((e: unknown, ctx: string) => {
    const status = e instanceof HistoryApiError ? e.status : 0;
    const msg = e instanceof Error ? e.message : String(e);
    if (status >= 500 || status === 0) {
      // Network down, TURSO_URL unset, etc. Degrade quietly.
      console.warn(`[session-store] ${ctx}: ${msg} — disabling history`);
      dispatch({ type: "DISABLE", disabled: true });
    } else {
      dispatch({ type: "ERROR", message: `${ctx}: ${msg}` });
    }
  }, []);

  // ── Boot: /api/history/me + initial sessions + categories ────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [user, sessions, categories] = await Promise.all([
          fetchMe(),
          fetchSessions(filterRef.current),
          fetchCategories(),
        ]);
        if (cancelled) return;
        dispatch({ type: "BOOT_DONE", user, sessions, categories });
      } catch (e) {
        if (cancelled) return;
        handleApiError(e, "boot");
        dispatch({ type: "BOOT_DONE", user: null, sessions: [], categories: [] });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [handleApiError]);

  // ── Actions ──────────────────────────────────────────────────────

  const refresh = useCallback(async () => {
    try {
      const [sessions, categories] = await Promise.all([
        fetchSessions(filterRef.current),
        fetchCategories(),
      ]);
      dispatch({ type: "SET_SESSIONS", sessions });
      dispatch({ type: "SET_CATEGORIES", categories });
    } catch (e) {
      handleApiError(e, "refresh");
    }
  }, [handleApiError]);

  const setFilter = useCallback(
    async (f: ListSessionsQuery) => {
      dispatch({ type: "SET_FILTER", filter: f });
      filterRef.current = { ...filterRef.current, ...f };
      try {
        const sessions = await fetchSessions(filterRef.current);
        dispatch({ type: "SET_SESSIONS", sessions });
      } catch (e) {
        handleApiError(e, "setFilter");
      }
    },
    [handleApiError],
  );

  const openSession = useCallback(
    async (id: string) => {
      try {
        const { session, messages } = await fetchSession(id);
        dispatch({ type: "SET_ACTIVE", id: session.id, messages });
        dispatch({ type: "UPSERT_SESSION", session });
      } catch (e) {
        handleApiError(e, "openSession");
      }
    },
    [handleApiError],
  );

  const closeSession = useCallback(() => {
    dispatch({ type: "SET_ACTIVE", id: null, messages: [] });
  }, []);

  const newSession = useCallback(
    async (init: {
      title?: string;
      category?: string | null;
      provider_id?: string | null;
      mode?: string | null;
    } = {}) => {
      try {
        const session = await createSession(init);
        dispatch({ type: "UPSERT_SESSION", session });
        dispatch({ type: "SET_ACTIVE", id: session.id, messages: [] });
        // Categories count may have changed — refresh async, don't block.
        void (async () => {
          try {
            const categories = await fetchCategories();
            dispatch({ type: "SET_CATEGORIES", categories });
          } catch {
            /* silent — not critical */
          }
        })();
        return session;
      } catch (e) {
        handleApiError(e, "newSession");
        throw e;
      }
    },
    [handleApiError],
  );

  const updateSession = useCallback(
    async (
      id: string,
      patch: {
        title?: string;
        category?: string | null;
        provider_id?: string | null;
        mode?: string | null;
        pinned?: boolean;
        archived?: boolean;
      },
    ) => {
      try {
        const session = await patchSession(id, patch);
        if (patch.archived === true) {
          dispatch({ type: "REMOVE_SESSION", id });
        } else {
          dispatch({ type: "UPSERT_SESSION", session });
        }
        // Refresh category counts if the category or archived state
        // changed. Cheap enough to always fire.
        void (async () => {
          try {
            const categories = await fetchCategories();
            dispatch({ type: "SET_CATEGORIES", categories });
          } catch {
            /* silent */
          }
        })();
      } catch (e) {
        handleApiError(e, "updateSession");
      }
    },
    [handleApiError],
  );

  const removeSession = useCallback(
    async (id: string) => {
      try {
        await deleteSession(id);
        dispatch({ type: "REMOVE_SESSION", id });
        void (async () => {
          try {
            const categories = await fetchCategories();
            dispatch({ type: "SET_CATEGORIES", categories });
          } catch {
            /* silent */
          }
        })();
      } catch (e) {
        handleApiError(e, "removeSession");
      }
    },
    [handleApiError],
  );

  const persistMessage = useCallback(
    async (
      msg: AppendMessageArgs,
      initIfMissing?: {
        title?: string;
        category?: string | null;
        provider_id?: string | null;
        mode?: string | null;
      },
    ): Promise<FdMessage | null> => {
      if (state.disabled) return null;
      try {
        let sessionId = state.activeSessionId;
        if (!sessionId) {
          const created = await createSession(initIfMissing ?? {});
          sessionId = created.id;
          dispatch({ type: "UPSERT_SESSION", session: created });
          dispatch({ type: "SET_ACTIVE", id: created.id, messages: [] });
        }
        const { message, session } = await appendMessage(sessionId, msg);
        dispatch({ type: "APPEND_MESSAGE", message });
        if (session) dispatch({ type: "UPSERT_SESSION", session });
        return message;
      } catch (e) {
        handleApiError(e, "persistMessage");
        return null;
      }
    },
    [handleApiError, state.activeSessionId, state.disabled],
  );

  const updateMyLabel = useCallback(
    async (patch: { display_name?: string | null; label_email?: string | null }) => {
      try {
        const user = await patchMe(patch);
        dispatch({ type: "SET_USER", user });
      } catch (e) {
        handleApiError(e, "updateMyLabel");
      }
    },
    [handleApiError],
  );

  const exportAll = useCallback(async () => {
    try {
      await downloadHistory();
    } catch (e) {
      handleApiError(e, "exportAll");
    }
  }, [handleApiError]);

  const value = useMemo<ContextShape>(
    () => ({
      ...state,
      refresh,
      setFilter,
      openSession,
      closeSession,
      newSession,
      updateSession,
      removeSession,
      persistMessage,
      updateMyLabel,
      exportAll,
    }),
    [
      state,
      refresh,
      setFilter,
      openSession,
      closeSession,
      newSession,
      updateSession,
      removeSession,
      persistMessage,
      updateMyLabel,
      exportAll,
    ],
  );

  return (
    <SessionStoreContext.Provider value={value}>
      {children}
    </SessionStoreContext.Provider>
  );
}

/**
 * Read the session store from any client component beneath
 * ``<SessionStoreProvider>``. Throws loudly if the tree is missing the
 * provider — that's always a bug, not a graceful-degrade case.
 */
export function useSessionStore(): ContextShape {
  const ctx = useContext(SessionStoreContext);
  if (!ctx) {
    throw new Error(
      "useSessionStore must be used inside <SessionStoreProvider>",
    );
  }
  return ctx;
}
