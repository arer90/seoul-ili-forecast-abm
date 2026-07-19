/**
 * AppShell — top-level layout.
 *
 * Desktop (≥ md):
 *   ┌──┬────────┬────────────────────────────────────┐
 *   │☰ │        │            TimeTrack              │
 *   │  │History ├──────────┬─────────────────────────┤
 *   │  │        │   Map  ⛶ ║  Chat  ⛶  (+ header)    │
 *   │  │        │          ║                         │
 *   └──┴────────┴──────────┴─────────────────────────┘
 *
 * Sidebar behaviour (2026-04-22 rework):
 *   · Claude-style persistent toggle — a tiny rail with a ☰ button
 *     sits on the far left regardless of state.
 *   · Clicking ☰ flips a ``sidebarOpen`` state that persists in
 *     localStorage. When closed, the History column collapses to
 *     width 0 and the map/chat widen naturally because the grid uses
 *     ``grid-cols-[auto_1fr]``.
 *   · Works identically on desktop and mobile — no separate drawer
 *     overlay to diverge from the desktop.
 *
 * Panel expand icons (new 2026-04-22):
 *   · Above the map and the chat, small ⛶ buttons call
 *     ``panelGroupRef.setLayout([100, 0])`` (map full) or
 *     ``[0, 100]`` (chat full). A third button resets to the default
 *     [45, 55] split.
 *
 * The resize layout still persists per ``autoSaveId`` via
 * ``react-resizable-panels``.
 */
"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
  type ImperativePanelGroupHandle,
} from "react-resizable-panels";

import { I18nProvider, LocaleToggle, useT } from "@/lib/i18n";
import { SessionStoreProvider } from "@/lib/use-session-store";

import { ChatPanel } from "./ChatPanel";
import { HistorySidebar } from "./HistorySidebar";
import { SessionHeader } from "./SessionHeader";
import { TimeTrack, type TimeTrackEvent, type TimeTrackPoint } from "./TimeTrack";
import type { ContextAction, GuChoroplethRow, PlaybackHud } from "./MapPanel";
import { HelpIcon } from "./ui/HelpIcon";
import { StatusRack } from "./StatusRack";
import { RefreshDataButton } from "./RefreshDataButton";
import { ScenarioPicker14 } from "./ScenarioPicker14";
import { TransmissionPlayer, type PlayerFrame } from "./TransmissionPlayer";
import type { ProviderId } from "@/lib/providers/types";

// Leaflet touches `window` at import time; load the panel client-only.
const MapPanel = dynamic(() => import("./MapPanel"), {
  ssr: false,
  loading: () => <MapLoadingFallback />,
});

function MapLoadingFallback() {
  const { t } = useT();
  return (
    <div className="flex h-full items-center justify-center text-xs text-slate-500">
      {t("mapLoading")}
    </div>
  );
}

// WASM glue uses ``WebAssembly.instantiateStreaming`` and a runtime
// ``new URL('seir_wasm_bg.wasm', import.meta.url)`` fallback; neither
// belongs on the edge server. Keeping this client-only via dynamic
// import also lets the rest of the shell paint before the wasm
// payload arrives.
const WhatIfSimCard = dynamic(
  () => import("./WhatIfSimCard").then((m) => m.WhatIfSimCard),
  {
    ssr: false,
    loading: () => <WhatIfLoadingFallback />,
  },
);

function WhatIfLoadingFallback() {
  const { t } = useT();
  return (
    <div className="rounded-md border border-slate-800 bg-slate-950/60 p-2 text-xs text-slate-500">
      {t("whatIfLoading")}
    </div>
  );
}

function useIsDesktop() {
  const [desktop, setDesktop] = useState(true);
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 768px)");
    const update = () => setDesktop(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return desktop;
}

const SIDEBAR_KEY = "frame_d_sidebar_open";
function useSidebarOpen(initialDesktop: boolean) {
  // Hidden by default on mobile, visible on desktop. Both persist once
  // the user explicitly toggles.
  const [open, setOpen] = useState<boolean>(initialDesktop);
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(SIDEBAR_KEY);
      if (saved === "1") setOpen(true);
      else if (saved === "0") setOpen(false);
    } catch {
      /* ignore */
    }
  }, []);
  const setOpenPersisted = useCallback((v: boolean) => {
    setOpen(v);
    try {
      window.localStorage.setItem(SIDEBAR_KEY, v ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, []);
  return [open, setOpenPersisted] as const;
}

/**
 * Seed the time-track with a few rows even before the MCP bridge is
 * up. Real data comes in via `epi.forecast` calls the chat makes. The
 * seed prevents the SVG from collapsing to zero and also serves as a
 * smoke-test in offline mode.
 */
function seedTimeTrack(): { points: TimeTrackPoint[]; events: TimeTrackEvent[] } {
  const points: TimeTrackPoint[] = [];
  const weeks = 52;
  for (let i = 0; i < weeks; i++) {
    const year = 2025 + Math.floor(i / 52);
    const wk = (i % 52) + 1;
    const label = `${year}-W${String(wk).padStart(2, "0")}`;
    const seasonal = 40 + 30 * Math.sin((i / 52) * 2 * Math.PI);
    const noise = (i * 13) % 11 - 5;
    const observed = Math.max(0, seasonal + noise);
    const pi_lo = Math.max(0, observed * 0.7);
    const pi_hi = observed * 1.3;
    points.push({
      week: label,
      observed: i < weeks - 8 ? observed : null,
      forecast: i >= weeks - 12 ? observed : null,
      pi_lo: i >= weeks - 12 ? pi_lo : null,
      pi_hi: i >= weeks - 12 ? pi_hi : null,
      rt: 0.8 + 0.4 * Math.cos((i / 52) * 2 * Math.PI),
    });
  }
  const events: TimeTrackEvent[] = [
    { week: "2025-W05", label: "Winter break", kind: "vacation" },
    { week: "2025-W30", label: "Summer break", kind: "vacation" },
    { week: "2025-W40", label: "Vaccine rollout", kind: "vaccination" },
    { week: "2026-W02", label: "Setup guidance", kind: "npi" },
  ];
  return { points, events };
}

/**
 * Slim left rail showing the ☰ toggle at all times. Always 1.75 rem
 * wide, so the sidebar can collapse entirely without leaving a layout
 * hole. When the sidebar is open this rail still shows ``◂`` as a
 * close affordance, like Claude's sidebar header.
 */
function SidebarRail({
  open,
  onToggle,
}: {
  open: boolean;
  onToggle: () => void;
}) {
  const { t } = useT();
  return (
    <div className="flex h-full w-7 flex-col items-center border-r border-slate-800 bg-slate-950/80 py-2">
      <button
        type="button"
        onClick={onToggle}
        className="rounded border border-slate-700 bg-slate-900/60 p-1 text-[13px] leading-none text-slate-300 hover:bg-slate-800"
        aria-label={open ? t("sidebarClose") : t("sidebarOpen")}
        aria-expanded={open}
        title={open ? t("sidebarClose") : t("sidebarOpen")}
      >
        {open ? "◂" : "☰"}
      </button>
    </div>
  );
}

/**
 * Panel-split mode controls. Lives above the PanelGroup and drives
 * ``ImperativePanelGroupHandle.setLayout``.
 *
 *   🗺 Map   → [100, 0]   (chat collapsed)
 *   💬 Chat  → [0, 100]   (map collapsed)
 *   ▥ Split → [45, 55]    (default split, snaps back)
 *
 * Sized large enough to hit reliably with touch — the old `text-[11px]`
 * + `px-1.5 py-0.5` glyphs were too small on 4K displays and near the
 * right edge of the resize handle. Now ~32 px tall (md: 36 px) with a
 * leading icon + short label.
 */
function SplitControls({
  panelGroupRef,
}: {
  panelGroupRef: React.RefObject<ImperativePanelGroupHandle | null>;
}) {
  const { t } = useT();
  const setLayout = (layout: number[]) => {
    const api = panelGroupRef.current;
    if (api) api.setLayout(layout);
  };
  const btn =
    "inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900/70 px-2.5 py-1 text-[13px] font-medium leading-none text-slate-200 hover:border-sky-500/60 hover:bg-slate-800 hover:text-sky-100 transition-colors";
  return (
    <div className="flex items-center gap-1.5 pr-1">
      <button
        type="button"
        onClick={() => setLayout([100, 0])}
        className={btn}
        title={t("expandMap")}
        aria-label={t("expandMap")}
      >
        <span aria-hidden="true" className="text-[15px]">🗺</span>
        <span>Map</span>
      </button>
      <button
        type="button"
        onClick={() => setLayout([0, 100])}
        className={btn}
        title={t("expandChat")}
        aria-label={t("expandChat")}
      >
        <span aria-hidden="true" className="text-[15px]">💬</span>
        <span>Chat</span>
      </button>
      <button
        type="button"
        onClick={() => setLayout([45, 55])}
        className={btn}
        title={t("restoreSplit")}
        aria-label={t("restoreSplit")}
      >
        <span aria-hidden="true" className="text-[15px]">▥</span>
        <span>Split</span>
      </button>
    </div>
  );
}

function AppShellInner() {
  const { t } = useT();
  const desktop = useIsDesktop();
  const [sidebarOpen, setSidebarOpen] = useSidebarOpen(desktop);
  const { points, events } = useMemo(() => seedTimeTrack(), []);
  const [selectedGu, setSelectedGu] = useState<string | null>(null);
  const [cursorWeek, setCursorWeek] = useState<string | undefined>(
    points[points.length - 1]?.week,
  );
  // ``guRows`` was previously const+empty — the static choropleth fell
  // back to MapPanel's own fetch. We now hand it a setter so the
  // TransmissionPlayer can repaint the map frame-by-frame during
  // playback of the SEIR-V-D wave.
  const [guRows, setGuRows] = useState<GuChoroplethRow[]>([]);
  // Top-center HUD on the map (Day X / Y, W N, peak delta, cumulative,
  // active-gu count, top-3 districts). ``null`` hides the HUD — that is
  // the default before the WASM sim finishes its first tick. When the
  // TransmissionPlayer emits a PlayerFrame we hydrate this so the HUD
  // appears synchronised with the choropleth recolour.
  const [playbackHud, setPlaybackHud] = useState<PlaybackHud | null>(null);
  const [whatIfOpen, setWhatIfOpen] = useState(true);
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);

  // Lifted from ChatPanel so the header's StatusRack can derive the
  // Agent chip state (provider available? tool-capable model?).
  const [chatProvider, setChatProvider] = useState<ProviderId | undefined>();
  const [chatModel, setChatModel] = useState<string | null>(null);
  const handleChatSelectionChange = useCallback(
    (provider: ProviderId, model: string | null) => {
      setChatProvider(provider);
      setChatModel(model);
    },
    [],
  );

  // Seed gu index for the transmission animation. We resolve the user's
  // clicked gu on the map to an index into ``init.district_names`` via
  // the static aggregate — but since this component doesn't hold the
  // init JSON, we pass the selected *name* down and let the player look
  // it up. For now default to 강남구 (idx 22) which matches the paper's
  // benchmark seed so the demo is reproducible.
  const seedGuIdx = 22;

  // Freshness — read the static overlay bundle's generated_at once so
  // the TimeTrack badge has something concrete to display. The MapPanel
  // fetches the same file independently for the choropleth; this second
  // fetch is memoised by the browser cache, so it's effectively free.
  const [dataGeneratedAt, setDataGeneratedAt] = useState<string | null>(null);
  useEffect(() => {
    const ctl = new AbortController();
    void (async () => {
      try {
        const r = await fetch("/aggregates/live-overlays.json", {
          signal: ctl.signal,
          cache: "force-cache",
        });
        if (!r.ok) return;
        const body = (await r.json()) as { generated_at?: string };
        if (body?.generated_at) setDataGeneratedAt(body.generated_at);
      } catch {
        /* freshness is a nice-to-have; stay silent on failure */
      }
    })();
    return () => ctl.abort();
  }, []);

  const handleContextAction = useCallback<
    (action: ContextAction, gu: string) => void
  >((action, gu) => {
    if (action === "copy_name") {
      void navigator.clipboard?.writeText(gu);
      return;
    }
    const prompt =
      action === "ask_forecast"
        ? `${gu} 의 다음 4주 ILI 예측과 95% PI 를 알려줘.`
        : action === "ask_rt"
          ? `${gu} 의 현재 Rt 와 지난 8주 추이를 알려줘.`
          : `${gu} 의 이번 주 SHAP top 10 을 알려줘.`;
    window.dispatchEvent(
      new CustomEvent("frame-d:prefill", { detail: { prompt } }),
    );
  }, []);

  const weekLabels = useMemo(() => points.map((p) => p.week), [points]);

  // Called on every TransmissionPlayer tick. We re-pin the week cursor,
  // repaint the choropleth, AND hydrate the map HUD so the Map, the
  // TimeTrack cursor, the big Day-counter, and the Play/Pause controls
  // all stay in lock-step. Using ``useCallback`` keeps TransmissionPlayer's
  // own effects from re-firing because the ``onFrame`` reference is
  // stable across renders.
  //
  // We also broadcast the frame over a window CustomEvent. The ChatPanel
  // listens for ``frame-d:playback-state`` and folds the current week /
  // peak / top-gus into the system prompt so the LLM's answer "이번 주
  // 강남은 어때?" can use the actual simulation state instead of hallucinating.
  const handleTransmissionFrame = useCallback(
    (frame: PlayerFrame) => {
      const wk = weekLabels[frame.weekIdx];
      if (wk) setCursorWeek(wk);
      setGuRows(frame.guRows);
      setPlaybackHud({
        weekIdx: frame.weekIdx,
        dayIdx: frame.dayIdx,
        totalDays: frame.totalDays,
        nFrames: frame.nFrames,
        peakWeek: frame.peakWeek,
        cumulative: frame.cumulative,
        activeGuCount: frame.activeGuCount,
        topGus: frame.topGus,
        weekLabel: wk,
        playing: frame.playing,
      });
      if (typeof window !== "undefined") {
        window.dispatchEvent(
          new CustomEvent("frame-d:playback-state", {
            detail: {
              weekIdx: frame.weekIdx,
              weekLabel: wk,
              dayIdx: frame.dayIdx,
              totalDays: frame.totalDays,
              peakWeek: frame.peakWeek,
              cumulative: frame.cumulative,
              activeGuCount: frame.activeGuCount,
              topGus: frame.topGus,
              playing: frame.playing,
            },
          }),
        );
      }
    },
    [weekLabels],
  );

  const centerColumn = (
    <div className="flex h-full min-h-0 flex-col">
      {/* Sprint 2026-05-07 (사용자 critique): TimeTrack / TransmissionPlayer /
          What-if 시뮬레이터 모두 mobile 에서 hide (md:block). 사용자: "What-if
          와 그 위 내용 왜 필요?" — desktop 에서만 의미 있음. */}
      <div className="hidden px-2 md:block">
        <TimeTrack
          data={points}
          events={events}
          cursorWeek={cursorWeek}
          onCursorChange={setCursorWeek}
          dataGeneratedAt={dataGeneratedAt}
        />
      </div>

      {/* Transmission player — animates the WASM SEIR-V-D run onto the
          choropleth so a viewer *sees* "전파" instead of reading a static
          snapshot. Sits under TimeTrack because the Play/Pause buttons
          drive the same week cursor the TimeTrack exposes. */}
      <div className="hidden px-2 pt-1 md:block">
        <TransmissionPlayer
          weekLabels={weekLabels}
          cursorWeek={cursorWeek}
          seedGuIdx={seedGuIdx}
          onFrame={handleTransmissionFrame}
        />
      </div>

      <div className="hidden px-2 pt-1 md:block">
        <details
          open={whatIfOpen}
          onToggle={(e) => setWhatIfOpen((e.target as HTMLDetailsElement).open)}
          className="group"
        >
          <summary className="flex cursor-pointer select-none items-center justify-between rounded-md border border-slate-800 bg-slate-900/40 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800/40">
            <span className="flex items-center gap-1.5 font-medium">
              <span>{t("whatIfTitle")}</span>
              <HelpIcon label={t("helpWhatIf")} content={t("helpWhatIf")} side="bottom" />
              <span className="ml-1 text-[10px] text-slate-500">
                ({t("whatIfSubtitle")})
              </span>
              {/* Sprint 2026-05-07: drag/scroll fix — open 상태에서도 아래
                  Map+Chat 으로 scroll 가능하게 max-height 명시 */}
            </span>
            <span className="text-slate-500 group-open:rotate-180 transition-transform">
              ▾
            </span>
          </summary>
          {/* Sprint 2026-05-07 (#7 What-if drag fix): max-h + overflow-y-auto
              — open 상태에서도 What-if 안에서 자체 scroll, 외부 page 의 Map+Chat
              영역 안 막힘. 사용자 critique '클릭해서 숨기지 않으면 아래로 못 내려감'. */}
          <div className="mt-1 max-h-[40vh] overflow-y-auto rounded border border-slate-800/50">
            <WhatIfSimCard />
          </div>
        </details>
      </div>

      <main className="flex min-h-0 flex-1 flex-col p-2 pt-1">
        <div className="flex items-center justify-end pb-1">
          <SplitControls panelGroupRef={panelGroupRef} />
        </div>
        {desktop ? (
          <PanelGroup
            ref={panelGroupRef}
            direction="horizontal"
            autoSaveId="frame-d-main"
            className="h-full min-h-0 overflow-hidden rounded-md border border-slate-800"
          >
            <Panel defaultSize={45} minSize={0} collapsible>
              <MapPanel
                rows={guRows}
                selectedGu={selectedGu}
                onSelect={setSelectedGu}
                onContextAction={handleContextAction}
                playbackHud={playbackHud}
              />
            </Panel>
            <PanelResizeHandle className="w-1 bg-slate-800 hover:bg-sky-500/40" data-resize-handle />
            <Panel defaultSize={55} minSize={0} collapsible>
              <div className="flex h-full flex-col">
                <SessionHeader />
                <div className="flex-1 min-h-0">
                  <ChatPanel onSelectionChange={handleChatSelectionChange} />
                </div>
              </div>
            </Panel>
          </PanelGroup>
        ) : (
          <PanelGroup
            ref={panelGroupRef}
            direction="vertical"
            autoSaveId="frame-d-mobile"
            className="h-full min-h-0 overflow-hidden rounded-md border border-slate-800"
          >
            <Panel defaultSize={45} minSize={0} collapsible>
              <MapPanel
                rows={guRows}
                selectedGu={selectedGu}
                onSelect={setSelectedGu}
                onContextAction={handleContextAction}
                playbackHud={playbackHud}
              />
            </Panel>
            <PanelResizeHandle className="h-1 bg-slate-800 hover:bg-sky-500/40" data-resize-handle />
            <Panel defaultSize={55} minSize={0} collapsible>
              <div className="flex h-full flex-col">
                <SessionHeader />
                <div className="flex-1 min-h-0">
                  <ChatPanel onSelectionChange={handleChatSelectionChange} />
                </div>
              </div>
            </Panel>
          </PanelGroup>
        )}
      </main>
    </div>
  );

  const header = (
    <AppHeader
      desktop={desktop}
      chatProvider={chatProvider}
      chatModel={chatModel}
    />
  );

  // Root height discipline (2026-04-21 fix):
  //   · Was ``min-h-svh`` — a LOWER bound — so when the chat log grew
  //     past the viewport the whole page stretched and the map/time-
  //     track grew with it. User report: "대화창에서는 길어지는데 화면이
  //     지나가면서 지도가 길어지는데?"
  //   · Now ``h-svh`` (exact) + inner ``overflow-hidden``. The chat
  //     log's own ``overflow-y-auto`` div (inside ChatPanel) handles
  //     the scroll instead of pushing siblings around.
  //   · ``overflow-hidden`` on the middle row also clips HistorySidebar
  //     when the rail shows and the gu context menu briefly overflows.
  return (
    <div className="flex h-svh max-h-svh flex-col overflow-hidden bg-[var(--bg)]">
      {header}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* Sprint 2026-05-07 (사용자 critique '스마트폰 sidebar 왜 나와?'):
            SidebarRail + HistorySidebar 모두 mobile 에서 hide. desktop md:
            (≥768px) 에서만 표시. */}
        <div className="hidden md:contents">
          <SidebarRail
            open={sidebarOpen}
            onToggle={() => setSidebarOpen(!sidebarOpen)}
          />
          {sidebarOpen ? (
            <div className="w-[min(18rem,80vw)] shrink-0 overflow-hidden border-r border-slate-800">
              <HistorySidebar />
            </div>
          ) : null}
        </div>
        <div className="min-h-0 min-w-0 flex-1 overflow-hidden">{centerColumn}</div>
      </div>
    </div>
  );
}

/** Header split into its own component so it can use `useT()` — the
 *  hook only works inside a descendant of `<I18nProvider>`.
 *
 *  2026-04-21: replaced the single ``<McpStatusBadge />`` with a
 *  three-chip ``<StatusRack />`` so the three orthogonal failure modes
 *  (LLM provider / Hermes orchestrator / MCP bridge) are each
 *  individually legible. The Agent chip takes the live provider+model
 *  selection from ChatPanel so it can flag "provider up but model
 *  doesn't support tools" (e.g. exaone3.5) as ``partial`` — that
 *  silently-tool-less state was exactly what users were conflating
 *  with a down MCP bridge.
 */
function AppHeader({
  desktop,
  chatProvider,
  chatModel,
}: {
  desktop: boolean;
  chatProvider?: ProviderId;
  chatModel: string | null;
}) {
  const { t } = useT();
  return (
    <div className="flex items-center justify-between px-3 py-1.5 text-xs text-slate-300">
      <div className="flex items-baseline gap-2">
        <span className="font-semibold text-slate-100">{t("appTitle")}</span>
        {/* Subtitle — hidden on very narrow screens to keep the brand
            legible; reappears at sm: so the header is descriptive without
            wrapping on mobile. */}
        <span className="hidden text-[11px] text-slate-500 sm:inline">
          · {t("appSubtitle")}
        </span>
      </div>
      <div className="flex items-center gap-2">
        {/* Sprint 2026-05-07 (사용자 critique '글자 깨짐'): mobile 에서
            StatusRack + ScenarioPicker + RefreshDataButton 다 hide.
            desktop md: (≥768px) 에서만. mobile 은 LocaleToggle 만. */}
        <div className="hidden md:flex md:items-center md:gap-2">
          <StatusRack
            selectedProvider={chatProvider}
            selectedModel={chatModel}
          />
          <ScenarioPicker14 />
          <RefreshDataButton groups="weekly_disease,who_flunet" />
        </div>
        <LocaleToggle />
      </div>
    </div>
  );
}

export default function AppShell() {
  return (
    <I18nProvider>
      <SessionStoreProvider>
        <AppShellInner />
      </SessionStoreProvider>
    </I18nProvider>
  );
}
