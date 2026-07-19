/**
 * i18n — Korean / English translation layer for ARIA.
 *
 * Design choices (deliberately minimal):
 *
 *   · A flat `Record<Key, {ko,en}>` bag — no ICU MessageFormat, no
 *     pluralisation. The demo has ~100 strings total; a 20-line
 *     lookup beats a 200KB library for that.
 *   · No lazy loading. Both locales ship in the client bundle; the
 *     cost is trivial (< 5 KB gzipped) and avoids flash-of-wrong-text.
 *   · Locale persisted in `localStorage.frame_d_locale`. First render
 *     uses `NEXT_PUBLIC_DEFAULT_LOCALE` (set to `ko` in .env.local for
 *     the Korean demo audience).
 *   · `useT()` is the only entry point. `t.myKey` returns the string;
 *     unknown keys return the key itself so missing translations are
 *     visible during development.
 *
 * Adding a new string:
 *   1. Add a `{ko, en}` entry to MESSAGES below.
 *   2. In a client component: `const t = useT();` then `t.myKey`.
 *
 * Not using a fancier solution because the surface area stays
 * small — v22.x is a demo shell, not a multi-tenant SaaS.
 */
"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type Locale = "ko" | "en";

export interface Messages {
  // Header
  appTitle: string;
  appSubtitle: string;
  layoutDesktop: string;
  layoutMobile: string;
  // History sidebar
  historyLabel: string;
  openActions: string;
  closeActions: string;
  newChat: string;
  searchPlaceholder: string;
  showArchived: string;
  refresh: string;
  all: string;
  noCategory: string;
  loadingEllipsis: string;
  noMatches: string;
  noSessions: string;
  pinTooltip: string;
  unpinTooltip: string;
  archiveTooltip: string;
  deleteTooltip: string;
  confirmDelete: string;
  exportHistory: string;
  historyUnavailable: string;
  historyUnavailableHint: string;
  // Chat panel
  mode: string;
  provider: string;
  send: string;
  sending: string;
  stop: string;
  askPlaceholder: string;
  startNewChat: string;
  startNewChatHint: string;
  trainedModels: string;
  trainedModelsHint: string;
  viewAllModels: string;
  hideAllModels: string;
  retry: string;
  edit: string;
  deleteMsg: string;
  branchMsg: string;
  copyMsg: string;
  saveMsg: string;
  cancelMsg: string;
  confirmDeleteMsg: string;
  errorOccurred: string;
  toolsDisabledNotice: string;
  noOutput: string;
  // Map overlays
  overlayLabel: string;
  overlayIli: string;
  overlayAir: string;
  overlayTemp: string;
  overlayNone: string;
  overlayLegend: string;
  overlayHint: string;
  // Live-overlay additions (8 layers total: 3 ILI tiers + 5 environmental)
  overlayIliForecast: string;
  overlayIliLive: string;
  overlayIliAlert: string;
  overlayHumidity: string;
  overlayER: string;
  overlayMetro: string;
  overlayGroupIli: string;
  overlayGroupEnv: string;
  overlayGroupOps: string;
  // Freshness badge (🔴 LIVE / 🟡 STALE / ⚫ OFFLINE)
  freshnessLive: string;
  freshnessStale: string;
  freshnessOffline: string;
  freshnessTooltipLive: string;
  freshnessTooltipStale: string;
  freshnessTooltipOffline: string;
  overlaySourceLabel: string;
  // Suggestions
  suggestionsShow: string;
  suggestionsHide: string;
  // Map
  mapLoading: string;
  askForecast: string;
  askRt: string;
  askShap: string;
  copyName: string;
  rightClickHint: string;
  // Forecast model picker (next to ProviderPicker)
  forecastModelLabel: string;
  forecastModelAuto: string;
  forecastModelHint: string;
  // Panel expand toggles
  expandMap: string;
  expandChat: string;
  restoreSplit: string;
  // Sidebar collapse
  sidebarOpen: string;
  sidebarClose: string;
  // What-if
  whatIfTitle: string;
  whatIfSubtitle: string;
  whatIfLoading: string;
  // Freshness + MCP status
  lastUpdated: string;
  mcpStatusReady: string;
  mcpStatusDown: string;
  mcpStatusChecking: string;
  mcpStatusTooltip: string;
  // Help tooltips (per-row / per-widget short copy)
  helpWeek: string;
  helpIli: string;
  helpEvents: string;
  helpRt: string;
  helpOverlay: string;
  helpForecastModel: string;
  helpProvider: string;
  helpMode: string;
  helpWhatIf: string;
  helpTrainedModels: string;
  // Transmission player (Q3 — animated SEIR spread)
  transmissionLabel: string;
  transmissionPlay: string;
  transmissionPause: string;
  transmissionReset: string;
  transmissionJumpPeak: string;
  transmissionSpeed: string;
  transmissionSeed: string;
  transmissionLoading: string;
  helpTransmission: string;
  // Status rack (Q4)
  statusAgentReady: string;
  statusAgentDown: string;
  statusHermesReady: string;
  statusHermesDown: string;
  helpAgentStatus: string;
  helpHermesStatus: string;
  // Persona picker (Q2/Q5 — advisor framing layer)
  personaLabel: string;
  personaHint: string;
  helpPersona: string;
  // Map HUD (B1 — big Day/Week overlay during playback)
  hudCumulative: string;
  hudActiveGu: string;
  hudAtPeak: string;
  hudPeakInWeeks: string;
  hudPastPeakWeeks: string;
  // Commuter-edge layer (A1)
  edgeLayerLabel: string;
  edgeLayerHint: string;
  helpEdgeLayer: string;
  // Playback badge in chat composer (B2)
  playbackBadgeHint: string;
  playbackBadgeAuto: string;
  // GIF export (hybrid Option 1 — static deliverable for paper/README)
  exportGifLabel: string;
  exportGifHint: string;
  exportGifCancel: string;
  // Misc
  langToggle: string;
}

type Bag = { [K in keyof Messages]: { ko: string; en: string } };

const MESSAGES: Bag = {
  // Paper title (shortened). The thesis's full title is:
  //   KO: 감염병 전파 양상에 따른 적응적 행동 반응 기반 다중 에이전트
  //       시뮬레이션 연구
  //   EN: Multi-Agent Simulation of Adaptive Behavioral Responses to
  //       Infectious Disease Transmission Patterns
  // For the header we use the short brand "적응행동 시뮬레이터 / Adaptive
  // Behavior Simulators (ABS)". The plural "Simulators" is deliberate —
  // the app runs five distinct simulators side-by-side (Metapop SEIR-V-D
  // in WASM, the 53-model forecaster, the What-if scenario player, the
  // commuter-flow animator, and the LLM advisor) — and "ABS" is a
  // clean three-letter acronym that avoids clashing with ABM (which
  // only covers the SEIR layer). The KUIRB proposal file
  // (paper/[서식02...감염병_전파양상.hwp) carries the full thesis title.
  appTitle: {
    ko: "적응행동 시뮬레이터",
    en: "Adaptive Behavior Simulators",
  },
  appSubtitle: {
    ko: "ABS · 감염병 전파 양상 기반 다중 에이전트 시뮬레이션",
    en: "ABS · multi-agent simulation of disease transmission",
  },
  layoutDesktop: { ko: "데스크톱", en: "desktop" },
  layoutMobile: { ko: "모바일", en: "mobile" },

  historyLabel: { ko: "히스토리", en: "History" },
  openActions: { ko: "동작 열기", en: "Open actions" },
  closeActions: { ko: "동작 닫기", en: "Close actions" },
  newChat: { ko: "＋ 새 대화", en: "+ New chat" },
  searchPlaceholder: { ko: "제목 검색…", en: "Search titles…" },
  showArchived: { ko: "보관된 대화", en: "Show archived" },
  refresh: { ko: "새로고침", en: "Refresh" },
  all: { ko: "전체", en: "All" },
  noCategory: { ko: "카테고리 없음", en: "No category" },
  loadingEllipsis: { ko: "불러오는 중…", en: "Loading…" },
  noMatches: { ko: "일치 항목 없음", en: "No matches." },
  noSessions: { ko: "대화가 없습니다.", en: "No sessions yet." },
  pinTooltip: { ko: "상단 고정", en: "Pin to top" },
  unpinTooltip: { ko: "고정 해제", en: "Unpin" },
  archiveTooltip: { ko: "보관", en: "Archive" },
  deleteTooltip: { ko: "삭제", en: "Delete" },
  confirmDelete: {
    ko: "대화를 삭제하시겠습니까? 복구할 수 없습니다.",
    en: "Delete this chat? This is permanent.",
  },
  exportHistory: { ko: "내보내기", en: "Export" },
  historyUnavailable: {
    ko: "이 빌드에서는 히스토리 기능이 꺼져 있습니다.",
    en: "History is unavailable in this build.",
  },
  historyUnavailableHint: {
    ko: "TURSO_URL 을 설정하면 활성화됩니다.",
    en: "Set TURSO_URL to enable.",
  },

  mode: { ko: "모드", en: "Mode" },
  provider: { ko: "프로바이더", en: "Provider" },
  send: { ko: "보내기", en: "Send" },
  sending: { ko: "전송 중…", en: "…" },
  stop: { ko: "정지", en: "Stop" },
  askPlaceholder: {
    ko: "질문을 입력하세요. Enter=전송, Shift+Enter=줄바꿈, Esc=중단",
    en: "Ask about ILI forecasts, Rt, SHAP, or scenarios. Enter to send, Shift+Enter for newline, Esc to stop.",
  },
  startNewChat: {
    ko: "새 대화를 시작합니다",
    en: "Start a new conversation",
  },
  startNewChatHint: {
    ko: "우측 지도에서 자치구를 우클릭하거나, 아래에서 추천 프롬프트를 열어보세요.",
    en: "Right-click a district on the map, or open the suggestion chips below.",
  },
  trainedModels: { ko: "학습된 모델", en: "Trained models" },
  trainedModelsHint: {
    ko: "총 {n}개 (post_E v22.6) — WIS 오름차순 상위 10",
    en: "{n} total (post_E v22.6) — top 10 by WIS (lower is better)",
  },
  viewAllModels: { ko: "전체 {n}개 보기", en: "View all {n} models" },
  hideAllModels: { ko: "접기", en: "Hide" },
  retry: { ko: "재시도", en: "Retry" },
  edit: { ko: "수정", en: "Edit" },
  deleteMsg: { ko: "삭제", en: "Delete" },
  branchMsg: { ko: "분기", en: "Branch" },
  copyMsg: { ko: "복사", en: "Copy" },
  saveMsg: { ko: "저장", en: "Save" },
  cancelMsg: { ko: "취소", en: "Cancel" },
  confirmDeleteMsg: {
    ko: "이 메시지를 삭제하시겠습니까?",
    en: "Delete this message?",
  },
  errorOccurred: { ko: "오류가 발생했습니다", en: "An error occurred" },
  toolsDisabledNotice: {
    ko: "이 모델은 MCP 툴 호출을 지원하지 않아 직접 DB 접근 없이 답합니다.",
    en: "This model does not support MCP tool calls — answering from prior context only.",
  },
  noOutput: { ko: "(출력 없음)", en: "(no output)" },

  overlayLabel: { ko: "지도 오버레이", en: "Map overlay" },
  overlayIli: { ko: "ILI 예측", en: "ILI forecast" },
  overlayAir: { ko: "대기질(PM2.5)", en: "Air (PM2.5)" },
  overlayTemp: { ko: "기온(°C)", en: "Temp (°C)" },
  overlayNone: { ko: "없음", en: "None" },
  overlayLegend: { ko: "범례", en: "Legend" },
  overlayHint: {
    ko: "실시간 데이터 — 데모용 샘플값이 포함될 수 있습니다",
    en: "Live-ish data — may include sample values in demo mode",
  },

  // ── Tier A+B+C overlay labels (8 layers) ──────────────────────────
  // A. ILI tiers (3): post_E forecast · KDCA observed · q70 alert flag
  // B. Environmental (4): PM2.5 / temp / humidity / subway crowding
  // C. Operational (1): ER bed occupancy
  // The group headings come from ``overlayGroupIli`` / ``…Env`` / `…Ops`
  // so the <optgroup>s stay translatable.
  overlayIliForecast: {
    ko: "ILI 예측 (post_E v22.6)",
    en: "ILI forecast (post_E v22.6)",
  },
  overlayIliLive: {
    ko: "ILI 실측 (KDCA)",
    en: "ILI observed (KDCA)",
  },
  overlayIliAlert: {
    ko: "유행 경보 (q70 돌파)",
    en: "Outbreak alert (q70 breach)",
  },
  overlayHumidity: { ko: "습도 (%)", en: "Humidity (%)" },
  overlayER: { ko: "응급실 과밀도", en: "ER crowding" },
  overlayMetro: { ko: "지하철 혼잡도", en: "Subway crowding" },
  overlayGroupIli: { ko: "인플루엔자(ILI)", en: "Influenza (ILI)" },
  overlayGroupEnv: { ko: "환경 · 기상", en: "Environment" },
  overlayGroupOps: { ko: "의료 · 교통", en: "Health & transit" },

  // Freshness badge states. ``live`` = observation within 15 min,
  // ``stale`` = 15–60 min old, ``offline`` = older or no upstream.
  freshnessLive: { ko: "🔴 LIVE", en: "🔴 LIVE" },
  freshnessStale: { ko: "🟡 지연", en: "🟡 STALE" },
  freshnessOffline: { ko: "⚫ 오프라인", en: "⚫ OFFLINE" },
  freshnessTooltipLive: {
    ko: "최근 15분 이내의 실시간 데이터입니다. 업데이트 시각과 출처는 아래에 표시됩니다.",
    en: "Observed within the last 15 minutes. Source and timestamp shown below.",
  },
  freshnessTooltipStale: {
    ko: "15–60분 전에 관측된 데이터입니다. 상류 API 가 잠시 멈췄거나 ISR 캐시가 갱신 대기 중일 수 있습니다.",
    en: "Observed 15–60 minutes ago. Upstream API may be paused or ISR cache is awaiting revalidation.",
  },
  freshnessTooltipOffline: {
    ko: "실시간 소스에 접근할 수 없어 정적 fallback(public/aggregates/) 또는 결정론적 합성값이 표시됩니다. 데모 모드에서는 정상입니다.",
    en: "Live source unreachable — showing the static fallback (public/aggregates/) or a deterministic synthetic filler. Expected in demo mode.",
  },
  overlaySourceLabel: { ko: "출처", en: "Source" },

  suggestionsShow: { ko: "추천 프롬프트 열기", en: "Show suggestions" },
  suggestionsHide: { ko: "접기", en: "Hide" },

  mapLoading: { ko: "지도 불러오는 중…", en: "Loading map…" },
  askForecast: { ko: "예측 질문 — {gu}", en: "Ask forecast — {gu}" },
  askRt: { ko: "Rt 질문 — {gu}", en: "Ask Rt — {gu}" },
  askShap: { ko: "SHAP 질문 — {gu}", en: "Ask SHAP — {gu}" },
  copyName: { ko: "이름 복사 — {gu}", en: "Copy {gu}" },
  rightClickHint: {
    ko: "우클릭으로 질문",
    en: "Right-click for actions",
  },
  forecastModelLabel: { ko: "예측 모델", en: "Forecast model" },
  forecastModelAuto: {
    ko: "자동 (앙상블)",
    en: "Auto (ensemble)",
  },
  forecastModelHint: {
    ko: "LLM에게 힌트로 전달됩니다 — 실제 계산은 MCP epi.forecast 가 수행",
    en: "Passed to the LLM as a hint — actual forecasts run through MCP epi.forecast",
  },
  expandMap: { ko: "지도 크게", en: "Expand map" },
  expandChat: { ko: "대화 크게", en: "Expand chat" },
  restoreSplit: { ko: "원래대로", en: "Reset split" },
  sidebarOpen: { ko: "사이드바 열기", en: "Open sidebar" },
  sidebarClose: { ko: "사이드바 닫기", en: "Close sidebar" },

  whatIfTitle: { ko: "What-if 시뮬레이터", en: "What-if simulator" },
  whatIfSubtitle: {
    ko: "브라우저 WASM — 실행당 약 27ms",
    en: "browser WASM — ~27 ms/run",
  },
  whatIfLoading: {
    ko: "what-if 시뮬레이터 불러오는 중…",
    en: "Loading what-if simulator…",
  },

  lastUpdated: { ko: "업데이트", en: "Updated" },
  mcpStatusReady: { ko: "MCP 연결됨", en: "MCP ready" },
  mcpStatusDown: { ko: "MCP 오프라인", en: "MCP offline" },
  mcpStatusChecking: { ko: "MCP 확인 중…", en: "MCP checking…" },
  mcpStatusTooltip: {
    ko: "Hermes(오케스트레이터) → MCP 브릿지 → SQLite/DuckDB 파이프라인 상태. 녹색이면 LLM 이 epi.query_db / epi.forecast 등으로 실제 DB 를 조회할 수 있습니다. (툴을 지원하지 않는 LLM 은 이 상태와 무관하게 도구 호출 없이 답합니다.)",
    en: "Hermes (orchestrator) → MCP bridge → SQLite/DuckDB pipeline health. Green means an LLM can actually call epi.query_db / epi.forecast and hit the real DB. (LLMs that don't support tools answer without calling MCP regardless of this status.)",
  },

  helpWeek: {
    ko: "지도 · 차트가 참조하는 현재 주 (ISO-week). 슬라이더를 움직이면 같은 주의 시·구별 예측값이 지도에 반영됩니다.",
    en: "The ISO-week cursor shared by the map and chart. Drag it to snapshot the choropleth + forecast at that week.",
  },
  helpIli: {
    ko: "ILI(인플루엔자 유사증상) 주별 관측치와 예측치. 실선=관측, 점선=예측 중앙값, 파란 밴드=95% 예측구간. 단위는 1k명당 사례수.",
    en: "Weekly observed vs forecast ILI rate. Solid=observed, dashed=forecast median, blue band=95% PI. Unit: cases per 1k people.",
  },
  helpEvents: {
    ko: "NPI(주황) · 휴일(보라) · 방학(청) · 백신 캠페인(녹) 타임라인. hover 하면 주와 이름이 뜹니다.",
    en: "Timeline pips for NPI (orange), holidays (violet), school breaks (cyan), vaccine campaigns (green). Hover for the week + label.",
  },
  helpRt: {
    ko: "실효감염재생산지수 Rt. 파선=1.0 기준. Rt>1 이면 증가국면, <1 이면 감소국면으로 해석합니다.",
    en: "Effective reproduction number Rt. Dashed=1.0 threshold. Rt>1 signals growing epidemic, <1 declining.",
  },
  helpOverlay: {
    ko: "지도에 덮을 레이어 선택 (총 8종): ILI 예측/실측/경보 (Turso+post_E v22.6) · PM2.5 (서울 열린데이터) · 기온·습도 (기상청 ASOS) · 지하철 혼잡도 (서울교통공사 t-data) · 응급실 과밀도 (NEDIS). 상단의 🔴 LIVE / 🟡 STALE / ⚫ OFFLINE 배지가 각 레이어의 실시간성을 알려줍니다. 키가 없으면 정적 fallback 이나 결정론적 합성값이 표시됩니다 — 지도가 비지 않습니다.",
    en: "Choose one of 8 choropleth layers: ILI forecast / observed / alert (Turso + post_E v22.6), PM2.5 (Seoul open data), temperature & humidity (KMA ASOS), subway crowding (Seoul Metro t-data), ER crowding (NEDIS). The 🔴 LIVE / 🟡 STALE / ⚫ OFFLINE badge at the top indicates each layer's freshness. When an API key is missing, the layer falls back to the static aggregate or a deterministic synthetic filler — the map never renders blank.",
  },
  helpForecastModel: {
    ko: "post_E v22.6 평가에서 상위 53개 모델 중 어느 것을 LLM 답변의 '기준 모델' 로 쓸지 지정합니다. 실제 수치는 MCP epi.forecast 가 계산하며, 이 옵션은 LLM 에 힌트로만 전달됩니다.",
    en: "Pick which of the 53 post_E v22.6 models the LLM should anchor on. Actual numbers still come from MCP epi.forecast; this is a hint to the LLM only.",
  },
  helpProvider: {
    ko: "사용할 LLM 공급자(Google Gemini / Anthropic / OpenAI / Ollama 로컬). API 키가 없는 공급자는 비활성으로 표시됩니다. 툴 호출 지원 여부도 공급자별로 다릅니다.",
    en: "Which LLM backend to use (Google Gemini / Anthropic / OpenAI / local Ollama). Providers without an API key show as disabled. Tool-call support varies per model.",
  },
  helpMode: {
    ko: "Hermes 오케스트레이션 모드: solo(단일) · parallel(동시 스트리밍) · synthesis(메타 합성) · relay(순차 이어쓰기).",
    en: "Hermes orchestration mode: solo (single provider), parallel (concurrent streams), synthesis (meta-aggregator), relay (hand-off chain).",
  },
  helpWhatIf: {
    ko: "브라우저 WASM 으로 도는 Metapop SEIR-V-D 시나리오 엔진. 백신 커버리지 · NPI · Rt 를 흔들어 '예방 불가피'의 가상실험을 30ms 안에 실행합니다.",
    en: "In-browser WASM Metapop SEIR-V-D scenario engine. Tweak vaccine coverage / NPI / Rt and re-run what-ifs in ~30 ms.",
  },
  helpTrainedModels: {
    ko: "v22.6 post_E 평가 기준 상위 10개 예측 모델 (WIS 낮을수록 우수). NegBinGLM=1위, Ensemble-NNLS=5위, TabularDNN-Lite=11위 (원본 TabularDNN=57위를 대체).",
    en: "Top 10 forecasters from v22.6 post_E (lower WIS = better). NegBinGLM #1, Ensemble-NNLS #5, TabularDNN-Lite #11 (replaces the original TabularDNN at #57).",
  },

  transmissionLabel: { ko: "감염 전파", en: "Spread" },
  transmissionPlay: { ko: "재생", en: "Play" },
  transmissionPause: { ko: "일시정지", en: "Pause" },
  transmissionReset: { ko: "처음으로", en: "Reset" },
  transmissionJumpPeak: { ko: "피크로", en: "Jump to peak" },
  transmissionSpeed: { ko: "속도", en: "Speed" },
  transmissionSeed: { ko: "발원지", en: "Seed" },
  transmissionLoading: { ko: "WASM 불러오는 중…", en: "Loading WASM…" },
  helpTransmission: {
    ko: "Metapop SEIR-V-D 시뮬레이션을 브라우저에서 27ms로 돌려 25개 자치구의 주별 신규감염을 지도 choropleth 에 애니메이션으로 보여줍니다. 발원지(seed) 에서 통근 매트릭스를 따라 전파되는 파형을 관찰할 수 있습니다. Python 엔진과 peak_I 오차 0.007%.",
    en: "Runs the Metapop SEIR-V-D simulation in-browser (~27 ms) and animates weekly new infections across all 25 districts on the map. Watch the wave spread from the seed gu along the commuter matrix. Matches the Python engine on peak_I to within 0.007%.",
  },

  statusAgentReady: { ko: "Agent 준비", en: "Agent ready" },
  statusAgentDown: { ko: "Agent 비활성", en: "Agent offline" },
  statusHermesReady: { ko: "Hermes 정상", en: "Hermes ok" },
  statusHermesDown: { ko: "Hermes 오류", en: "Hermes err" },
  helpAgentStatus: {
    ko: "선택한 LLM 공급자의 연결 상태 + 해당 모델이 function-calling (툴 호출) 을 지원하는지 여부. 녹색=툴 호출 가능, 호박색=공급자는 살아있지만 선택 모델이 툴 미지원, 적색=공급자 자체 불가.",
    en: "Health of the selected LLM provider AND whether its current model supports function calling. Green = can call MCP tools, amber = provider alive but selected model lacks tool support, red = provider itself unavailable.",
  },
  helpHermesStatus: {
    ko: "Hermes 오케스트레이터 (solo/parallel/synthesis/relay) 의 로컬 실행 상태. 항상 브라우저 내부에서 돕니다 — 외부 의존이 없어 'down' 으로 뜨면 번들 로드 자체 실패를 뜻합니다.",
    en: "Local Hermes orchestrator (solo/parallel/synthesis/relay). Runs in-browser with no external dependencies — a 'down' state means the bundle itself failed to load.",
  },

  personaLabel: { ko: "상담 관점", en: "Persona" },
  personaHint: {
    ko: "답변을 어떤 관점으로 프레이밍할지 선택합니다. 숫자는 그대로, 해석 틀만 바뀝니다.",
    en: "Pick a framing lens for the reply. Numbers don't change — only the interpretive frame does.",
  },
  helpPersona: {
    ko: "epi/model/simulation/clinical 어드바이저 역할을 채팅 안에서도 간단히 적용할 수 있는 프레이밍 옵션입니다. 예측 숫자는 영향을 받지 않고, LLM의 답변 어조와 인용 범위만 바뀝니다. 53-모델 레지스트리 전체가 아니라 PAPER_PRIMARY_11 + Ensemble-NNLS 안에서만 모델을 언급하도록 제한되어 있습니다.",
    en: "A framing lens that applies an epi/model/simulation/clinical advisor role inside the chat. Forecast numbers are unaffected — only the reply's tone and citation scope change. Personas are constrained to cite within PAPER_PRIMARY_11 + Ensemble-NNLS, not the full 53-model registry, to prevent post-hoc cherry-picking.",
  },

  hudCumulative: { ko: "누적", en: "Cumulative" },
  hudActiveGu: { ko: "활성 구", en: "Active gu" },
  hudAtPeak: { ko: "피크 주", en: "At peak" },
  hudPeakInWeeks: { ko: "피크까지 {n}주", en: "{n}w to peak" },
  hudPastPeakWeeks: { ko: "피크 지난 {n}주", en: "{n}w past peak" },

  edgeLayerLabel: { ko: "통근 전파", en: "Commuter flow" },
  edgeLayerHint: {
    ko: "상위 30개 통근 경로를 매 주 감염 전파량으로 두께가 변하는 선으로 표시합니다. 0 = 끄기.",
    en: "Top-30 commuter edges drawn with thickness proportional to this week's transmitted infections. 0 = off.",
  },
  helpEdgeLayer: {
    ko: "통근 매트릭스 × 각 구의 감염자 수로 계산한 주간 전파 흐름. CFD 스타일에 가까운 '파동이 도시 사이로 흘러가는' 느낌을 주며, 메타팝 모델의 commuter-coupled FoI 가 실제로 지도에서 돌고 있음을 시각화합니다. 체크를 끄면 choropleth만 남아 기존 버전과 동일해집니다.",
    en: "Weekly transmission flux = commuter matrix × per-gu infected. Visualises the metapop commuter-coupled FoI on the map in a CFD-like 'wave flowing between cities' style. Turn off to revert to choropleth-only.",
  },

  playbackBadgeHint: {
    ko: "시뮬레이션 재생 중일 때 현재 Day / Week / 피크까지 거리를 보여줍니다. 이 상태는 다음 질문에 자동 반영됩니다 — '이번 주 강남은?' 같은 질문이 바로 이 주(week)로 해석됩니다.",
    en: "Shows the current Day / Week / distance-to-peak while the simulation is running. This state is auto-injected into your next question — asking '이번 주 강남은?' resolves to THIS week.",
  },
  playbackBadgeAuto: { ko: "질문에 자동 반영", en: "auto-context" },

  exportGifLabel: { ko: "GIF 내보내기", en: "Export GIF" },
  exportGifHint: {
    ko: "현재 seed·R0 시뮬을 한 번 주간별로 돌려 26프레임 GIF로 저장합니다. OSM 타일은 제외되어 어두운 배경 위에 구 choropleth + 통근 엣지 + HUD만 남아 논문 figure/README/슬라이드용으로 바로 쓸 수 있습니다.",
    en: "Walks the current seed·R0 simulation once, snapshots each week (26 frames), encodes as an animated GIF. OSM tiles are excluded — dark background + gu choropleth + commuter edges + HUD — ready for the paper figure, README, or slides.",
  },
  exportGifCancel: { ko: "내보내기 취소", en: "Cancel export" },

  langToggle: { ko: "EN", en: "KO" },
};

// ── Interpolation helper — replace {foo} / {bar} from a params map ──
function format(template: string, params?: Record<string, string | number>): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (_, k) =>
    params[k] != null ? String(params[k]) : `{${k}}`,
  );
}

// ── Context ───────────────────────────────────────────────────────
interface I18nCtx {
  locale: Locale;
  setLocale: (l: Locale) => void;
  toggle: () => void;
  /** Resolved string for a key, optionally interpolated. */
  t: (key: keyof Messages, params?: Record<string, string | number>) => string;
  /** Raw messages bag — handy for tests. */
  messages: Bag;
}

const Ctx = createContext<I18nCtx | null>(null);
const STORAGE_KEY = "frame_d_locale";

export function I18nProvider({ children }: { children: ReactNode }) {
  // SSR: honour the env default. On the client, useEffect will upgrade
  // from localStorage if present so hydration matches first paint.
  const envDefault =
    (process.env.NEXT_PUBLIC_DEFAULT_LOCALE as Locale | undefined) === "en"
      ? "en"
      : "ko";
  const [locale, setLocaleState] = useState<Locale>(envDefault);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY) as Locale | null;
      if (saved === "ko" || saved === "en") setLocaleState(saved);
    } catch {
      /* ignore */
    }
  }, []);

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    try {
      window.localStorage.setItem(STORAGE_KEY, l);
      document.documentElement.lang = l;
    } catch {
      /* ignore */
    }
  }, []);
  const toggle = useCallback(() => {
    setLocale(locale === "ko" ? "en" : "ko");
  }, [locale, setLocale]);

  const t = useCallback(
    (key: keyof Messages, params?: Record<string, string | number>) => {
      const entry = MESSAGES[key];
      if (!entry) return String(key);
      return format(entry[locale] ?? entry.en ?? String(key), params);
    },
    [locale],
  );

  const value = useMemo<I18nCtx>(
    () => ({ locale, setLocale, toggle, t, messages: MESSAGES }),
    [locale, setLocale, toggle, t],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useT(): I18nCtx {
  const ctx = useContext(Ctx);
  if (!ctx) {
    // Permissive fallback for tests / Storybook — yields English.
    return {
      locale: "en",
      setLocale: () => void 0,
      toggle: () => void 0,
      t: (key, params) => format(MESSAGES[key]?.en ?? String(key), params),
      messages: MESSAGES,
    };
  }
  return ctx;
}

/** Small toggle button — pairs with the `langToggle` message. */
export function LocaleToggle(props: { className?: string }) {
  const { locale, toggle, t } = useT();
  return (
    <button
      type="button"
      onClick={toggle}
      className={
        props.className ??
        "rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-300 hover:bg-slate-800"
      }
      aria-label={`Switch to ${locale === "ko" ? "English" : "한국어"}`}
      title={`Current: ${locale.toUpperCase()} — click to switch`}
    >
      🌐 {t("langToggle")}
    </button>
  );
}
