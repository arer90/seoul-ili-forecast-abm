/**
 * MapPanel — Leaflet choropleth over Seoul's 25 자치구.
 *
 * UX:
 *   - PC: right-click on a gu → context menu with "ask forecast / Rt /
 *     SHAP" and "copy name". Binding is done via ``bindContextMenu`` —
 *     setting ``layer.options.contextmenu`` *after* the layer is built
 *     does NOT work with leaflet-contextmenu, it has to be at bind time.
 *   - Mobile: long-press (leaflet-contextmenu translates the touchhold
 *     to the same menu handler).
 *   - Any-button click selects a gu and emits ``onSelect``.
 *   - A Seoul outer boundary layer sits on top of the gu fills as a
 *     non-interactive bold stroke so the city silhouette is readable
 *     even when the choropleth colours are subtle.
 *   - The top-left corner hosts an overlay picker with 8 layers grouped
 *     into 3 domains, backed by ``/api/overlays/live`` (Edge, ISR 300s):
 *       · Influenza (ILI):  forecast · observed · q70 alert
 *       · Environment:      PM2.5 · temperature · humidity
 *       · Health & transit: ER crowding · subway crowding
 *     A LIVE / STALE / OFFLINE freshness badge sits next to the picker
 *     so the viewer knows when a layer's upstream is missing (and is
 *     rendering a deterministic synthetic fill instead).
 *
 * Ref-stabilisation
 *   ``onSelect`` / ``onContextAction`` come from the parent as inline
 *   arrows (they close over AppShell state) and would be a new ref
 *   every render. If we put them in the layer effect deps array, the
 *   GeoJSON layer gets rebuilt on every parent render, which wipes any
 *   in-flight tooltip and rebuilds the context menu mid-right-click.
 *   We stash them in refs so the effect depends on stable values only.
 *
 * This component is dynamically imported with ``{ ssr: false }`` from
 * AppShell because Leaflet touches ``window`` at module load.
 */
"use client";

import "leaflet/dist/leaflet.css";
import "leaflet-contextmenu";
import "leaflet-contextmenu/dist/leaflet.contextmenu.css";

import { useEffect, useMemo, useRef, useState } from "react";

import { useT } from "@/lib/i18n";
import type {
  Freshness,
  LiveMetricId,
  LiveOverlaysResponse,
  MetricPayload,
} from "@/lib/live-overlays/types";
import { HelpIcon } from "./ui/HelpIcon";
import { LastUpdated } from "./ui/LastUpdated";

type LeafletModule = typeof import("leaflet");

export interface GuChoroplethRow {
  gu_nm: string;
  value: number;   // e.g. forecast ILI rate for the selected week
}

/**
 * Overlay picker value. ``"none"`` hides the choropleth (so the user can
 * see a bare gu outline), otherwise we drive the colouring from one of
 * the 8 ``LiveMetricId`` layers emitted by ``/api/overlays/live``.
 *
 * Legacy note: the prior build had `"none" | "ili" | "air" | "temp"` and
 * persisted any of those strings to ``localStorage``. The upgrade logic
 * in the init effect migrates old values (``"ili" → "ili_forecast"``)
 * so returning users don't lose their selection on first load.
 */
export type OverlayMetric = LiveMetricId | "none";

/** Set of valid picker values — also used as a localStorage guard. */
const VALID_METRICS = new Set<OverlayMetric>([
  "none",
  "ili_forecast",
  "ili_live",
  "ili_alert",
  "air",
  "temp",
  "humidity",
  "er",
  "metro",
]);

/** Upgrade the pre-8-layer localStorage values. "ili" / "air" / "temp"
 *  used to be the union; map them to the closest new ID so a returning
 *  user doesn't silently get "none" on their first post-upgrade load. */
const LEGACY_METRIC_UPGRADE: Record<string, OverlayMetric> = {
  ili: "ili_forecast",
  // "air" and "temp" survive the rename unchanged, but include them for
  // explicit documentation.
  air: "air",
  temp: "temp",
};

/**
 * Playback summary used by the large Day/Week HUD in the top-left
 * corner. The parent (AppShell) fills this from
 * ``TransmissionPlayer.onFrame`` — passing ``null`` hides the HUD,
 * which is how the map looks when no simulation is running.
 */
export interface PlaybackHud {
  weekIdx: number;
  dayIdx: number;
  totalDays: number;
  nFrames: number;
  peakWeek: number;
  cumulative: number;
  activeGuCount: number;
  topGus: string[];
  /** Optional human-readable label for the current week, e.g. "2025-W12". */
  weekLabel?: string;
  /** Is the animation currently playing? Used to colour the HUD dot. */
  playing?: boolean;
}

export interface MapPanelProps {
  /** GeoJSON with feature.properties.gu_nm for each district. */
  geojsonUrl?: string;
  /** GeoJSON for the outer Seoul silhouette — single polygon, no gu seams. */
  boundaryUrl?: string;
  /** Live-data bundle (ILI / PM2.5 / Temp) for the overlay picker. */
  overlaysUrl?: string;
  rows?: GuChoroplethRow[];
  selectedGu?: string | null;
  onSelect?: (gu: string) => void;
  /** Fires when the user picks a context-menu action on a gu. */
  onContextAction?: (action: ContextAction, gu: string) => void;
  /** Per-frame transmission summary driving the top-left HUD. */
  playbackHud?: PlaybackHud | null;
}

export type ContextAction =
  | "ask_forecast"
  | "ask_rt"
  | "ask_shap"
  | "copy_name";

const FALLBACK_URL = "/seoul-gu.geojson";
const BOUNDARY_URL = "/aggregates/seoul-boundary.geojson";
/**
 * v22.7 Stage 6 live-overlay route. The edge function at
 * ``/api/overlays/live`` fans out to 5 upstream providers
 * (Turso / Seoul open-data PM2.5 / KMA ASOS / NEDIS ER / Metro t-data)
 * with a shared ISR window of 5 min and backfills any missing layer
 * from the static ``/aggregates/live-overlays.json`` aggregate so the
 * map is never blank. Passing a bare static URL (legacy) also works —
 * the shape check in the fetch effect tolerates both.
 */
const OVERLAYS_URL = "/api/overlays/live";
const COMMUTER_EDGES_URL = "/aggregates/commuter-edges.json";
const OVERLAY_KEY = "frame_d_map_overlay";
const EDGE_KEY = "frame_d_map_edges";
/** Fallback if the server forgets to send ttl_seconds. */
const DEFAULT_TTL_SECONDS = 300;
/** How many commuter polylines to keep alive per frame — the bundled
 *  JSON has 50 edges (25 pairs × 2 directions). Rendering all 50 makes
 *  the map look like a spider-web; picking the top 30 by per-frame flux
 *  keeps the dominant commute corridors legible. */
const EDGE_TOP_N = 30;

/**
 * Simple sequential colormap — dark blue → bright yellow (viridis-ish).
 * Keeping it inline avoids d3-scale for a 5-stop scale.
 */
function colorFor(v: number, vmax: number): string {
  if (!Number.isFinite(v) || vmax <= 0) return "#1e293b";
  const t = Math.max(0, Math.min(1, v / vmax));
  const stops = [
    [11, 15, 20],      // #0b0f14
    [59, 48, 126],     // #3b307e
    [86, 123, 181],    // #567bb5
    [146, 201, 171],   // #92c9ab
    [240, 236, 90],    // #f0ec5a
  ];
  const idx = t * (stops.length - 1);
  const i = Math.floor(idx);
  const frac = idx - i;
  const a = stops[i];
  const b = stops[Math.min(stops.length - 1, i + 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * frac);
  const g = Math.round(a[1] + (b[1] - a[1]) * frac);
  const bl = Math.round(a[2] + (b[2] - a[2]) * frac);
  return `rgb(${r}, ${g}, ${bl})`;
}

export default function MapPanel({
  geojsonUrl = FALLBACK_URL,
  boundaryUrl = BOUNDARY_URL,
  overlaysUrl = OVERLAYS_URL,
  rows = [],
  selectedGu,
  onSelect,
  onContextAction,
  playbackHud = null,
}: MapPanelProps) {
  const { t, locale } = useT();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<import("leaflet").Map | null>(null);
  const layerRef = useRef<import("leaflet").GeoJSON | null>(null);
  const boundaryLayerRef = useRef<import("leaflet").GeoJSON | null>(null);
  const [L, setL] = useState<LeafletModule | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Live overlay state — fetches ``/api/overlays/live`` and keeps all 8
  // metric payloads (ili_forecast / ili_live / ili_alert / air / temp /
  // humidity / er / metro) resident. Switching ``overlayMetric`` swaps
  // which metric drives the choropleth without a round-trip.
  //
  // The Edge route is wrapped in ISR with ``revalidate = 300`` so even if
  // the user refreshes their tab every second, Vercel only talks to the
  // upstream providers every 5 min. We additionally schedule a
  // client-side refetch after ``ttl_seconds`` so the freshness badge
  // flips LIVE → STALE → LIVE again without a manual reload.
  //
  // Persisted to localStorage so a reload doesn't lose the selection.
  // The init effect migrates the legacy ``"ili"`` value to
  // ``"ili_forecast"`` so returning users keep their picker intact.
  const [overlayMetric, setOverlayMetric] = useState<OverlayMetric>("none");
  // Sprint 2026-05-07 (#9): collapse overlay panel — Leaflet zoom 버튼 가림 회피
  const [overlayPanelHidden, setOverlayPanelHidden] = useState(false);
  const [overlays, setOverlays] = useState<LiveOverlaysResponse | null>(null);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(OVERLAY_KEY);
      if (!saved) return;
      if (VALID_METRICS.has(saved as OverlayMetric)) {
        setOverlayMetric(saved as OverlayMetric);
        return;
      }
      const upgraded = LEGACY_METRIC_UPGRADE[saved];
      if (upgraded) setOverlayMetric(upgraded);
    } catch {
      /* ignore */
    }
  }, []);
  useEffect(() => {
    try {
      window.localStorage.setItem(OVERLAY_KEY, overlayMetric);
    } catch {
      /* ignore */
    }
  }, [overlayMetric]);

  useEffect(() => {
    const ctl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;

    const load = async () => {
      try {
        // ``cache: "default"`` lets the browser respect the Edge
        // route's ``cache-control: s-maxage=300, stale-while-revalidate``
        // headers — we still get a blazing-fast warm repaint from the
        // disk cache when the metric picker changes mid-session.
        const r = await fetch(overlaysUrl, {
          signal: ctl.signal,
          cache: "default",
        });
        if (!r.ok) {
          setOverlayError(`overlays HTTP ${r.status}`);
          return;
        }
        const body = (await r.json()) as LiveOverlaysResponse;
        setOverlays(body);
        setOverlayError(null);

        // Re-fetch at TTL boundary so the freshness chip auto-updates.
        const ttl =
          typeof body.ttl_seconds === "number" && body.ttl_seconds > 0
            ? body.ttl_seconds
            : DEFAULT_TTL_SECONDS;
        timer = setTimeout(() => void load(), ttl * 1000);
      } catch (e) {
        if ((e as DOMException)?.name === "AbortError") return;
        setOverlayError(e instanceof Error ? e.message : String(e));
      }
    };
    void load();

    return () => {
      ctl.abort();
      if (timer) clearTimeout(timer);
    };
  }, [overlaysUrl]);

  // ── Commuter edge layer ────────────────────────────────────────────
  //
  // The 25-gu choropleth alone shows *where* the flu is, but not *how*
  // it's spreading between districts. The commuter matrix already drives
  // the Metapop simulation's force-of-infection coupling, so overlaying
  // the top corridors turns the invisible coupling into a visible "wave-
  // front flowing from Gangnam → Eunpyeong" story.
  //
  // Data shape: ``{edges: [{src, dst, weight}, ...]}`` with 50 directed
  // edges (25 pairs × 2 directions) pre-computed from KOSIS commuter data.
  //
  // Rendering strategy: we keep all 50 polyline refs resident, pre-built
  // from the GeoJSON centroids, and every frame we (a) compute the flux
  // proxy ``weight * value_at_src`` for each edge, (b) pick the top 30 by
  // flux, (c) toggle visibility + restyle on each. No remove/add churn.
  const [showEdges, setShowEdges] = useState(true);
  const [commuterEdges, setCommuterEdges] = useState<
    Array<{ src: string; dst: string; weight: number }>
  >([]);
  /** gu_nm → [lat, lng] centroid, derived from the GeoJSON feature bbox. */
  const centroidsRef = useRef<Map<string, [number, number]>>(new Map());
  /** Parallel array of Leaflet polylines, one per commuter edge. */
  const edgePolylinesRef = useRef<
    Array<import("leaflet").Polyline | null>
  >([]);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(EDGE_KEY);
      if (saved === "1") setShowEdges(true);
      else if (saved === "0") setShowEdges(false);
    } catch {
      /* ignore */
    }
  }, []);
  useEffect(() => {
    try {
      window.localStorage.setItem(EDGE_KEY, showEdges ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [showEdges]);

  // One-shot fetch of the commuter aggregate. Same ``force-cache`` hint
  // as the other static bundles — this never changes at runtime.
  useEffect(() => {
    const ctl = new AbortController();
    void (async () => {
      try {
        const r = await fetch(COMMUTER_EDGES_URL, {
          signal: ctl.signal,
          cache: "force-cache",
        });
        if (!r.ok) return;
        const body = (await r.json()) as {
          edges?: Array<{ src: string; dst: string; weight: number }>;
        };
        setCommuterEdges(body.edges ?? []);
      } catch {
        /* edge overlay is opt-in; silent on failure */
      }
    })();
    return () => ctl.abort();
  }, []);

  // Refs over props that the GeoJSON effect would otherwise re-depend
  // on — keeping them stable means the layer is only rebuilt when the
  // underlying *data* (geojsonUrl, rows, selectedGu) changes, not when
  // the parent happens to re-render with new inline arrow closures.
  const onSelectRef = useRef(onSelect);
  const onContextActionRef = useRef(onContextAction);
  const tRef = useRef(t);
  const unitRef = useRef<string | null>(null);
  useEffect(() => {
    onSelectRef.current = onSelect;
    onContextActionRef.current = onContextAction;
    tRef.current = t;
  }, [onSelect, onContextAction, t]);

  /** Resolve the picker selection to the live payload when possible. */
  const activePayload = useMemo<MetricPayload | null>(() => {
    if (overlayMetric === "none") return null;
    return overlays?.metrics?.[overlayMetric] ?? null;
  }, [overlayMetric, overlays]);

  // Active data source — the overlay picker overrides the ``rows`` prop
  // when something other than "none" is selected. Otherwise we defer to
  // whatever the parent passed in (usually empty on the demo).
  const activeRows = useMemo<GuChoroplethRow[]>(() => {
    return activePayload?.rows ?? rows;
  }, [activePayload, rows]);

  const activeUnit = useMemo<string | null>(() => {
    return activePayload?.unit ?? null;
  }, [activePayload]);
  // Keep the tooltip callback's unit string in sync without adding
  // activeUnit to the layer effect's deps (which would rebuild the
  // whole GeoJSON layer on every locale / overlay change).
  useEffect(() => {
    unitRef.current = activeUnit;
  }, [activeUnit]);

  const valueByGu = useMemo(() => {
    const m = new Map<string, number>();
    for (const r of activeRows) m.set(r.gu_nm, r.value);
    return m;
  }, [activeRows]);
  const vmax = useMemo(
    () => activeRows.reduce((m, r) => (r.value > m ? r.value : m), 0),
    [activeRows],
  );
  const vmin = useMemo(
    () =>
      activeRows.length
        ? activeRows.reduce((m, r) => (r.value < m ? r.value : m), Infinity)
        : 0,
    [activeRows],
  );

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const mod = await import("leaflet");
        await import("leaflet-contextmenu");
        if (!cancelled) setL(mod);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!L || !containerRef.current) return;
    if (mapRef.current) return;
    const container = containerRef.current;
    const map = L.map(container, {
      center: [37.5665, 126.978],
      zoom: 11,
      minZoom: 9,
      maxZoom: 18,
      zoomControl: true,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      touchZoom: true,
      boxZoom: true,
      // @ts-expect-error — leaflet-contextmenu augments L.Map options.
      contextmenu: true,
      contextmenuWidth: 200,
      contextmenuItems: [],
      preferCanvas: true,
    });
    // 2026-04-28: Multi-tile-layer 토글 — 무료 + API-key 옵션
    // 8 base layers + 5 overlays + VWorld placeholder
    const baseLayers: Record<string, L.TileLayer> = {
      "🗺️ Streets (OSM)": L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        { maxZoom: 18, attribution: "© OSM" },
      ),
      "🛰️ Satellite (Esri)": L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
          maxZoom: 19,
          attribution: "Tiles © Esri — Source: Esri, USDA, USGS, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP",
        },
      ),
      "🌑 Dark (CartoDB)": L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        {
          maxZoom: 19,
          attribution: '© <a href="https://carto.com/attributions">CARTO</a> © OSM',
          subdomains: "abcd",
        },
      ),
      "🌕 Light (CartoDB)": L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        {
          maxZoom: 19,
          attribution: '© CARTO © OSM',
          subdomains: "abcd",
        },
      ),
      "📡 Topography (Esri)": L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, attribution: "Tiles © Esri" },
      ),
      "🛣️ Streets (Esri)": L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, attribution: "Tiles © Esri" },
      ),
      "📜 Watercolor (Stamen)": L.tileLayer(
        "https://stamen-tiles-{s}.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg",
        {
          maxZoom: 14,
          attribution: 'Tiles © Stamen Design — © OSM',
          subdomains: "abcd",
        },
      ),
    };

    // ── VWorld 한국 위성 (API key 필요) ──────────────────────────
    // .env.local 에 NEXT_PUBLIC_VWORLD_KEY 설정 시 자동 활성화
    // 발급: https://www.vworld.kr/dev/v4dv_apikey_s001.do
    const VWORLD_KEY = process.env.NEXT_PUBLIC_VWORLD_KEY;
    if (VWORLD_KEY) {
      baseLayers["🇰🇷 VWorld 위성"] = L.tileLayer(
        `https://api.vworld.kr/req/wmts/1.0.0/${VWORLD_KEY}/Satellite/{z}/{y}/{x}.jpeg`,
        {
          maxZoom: 19,
          attribution: "© <a href='https://www.vworld.kr'>VWorld</a> 국토교통부",
        },
      );
      baseLayers["🇰🇷 VWorld Hybrid"] = L.tileLayer(
        `https://api.vworld.kr/req/wmts/1.0.0/${VWORLD_KEY}/Hybrid/{z}/{y}/{x}.png`,
        {
          maxZoom: 19,
          attribution: "© VWorld",
        },
      );
    }

    // OSM default
    baseLayers["🗺️ Streets (OSM)"].addTo(map);

    // ── NASA GIBS Overlays (실시간, key 불필요) ──────────────────
    // GIBS 는 보통 1일 lag → 어제 날짜 사용
    const yest = new Date(Date.now() - 86400000)
      .toISOString().split("T")[0];
    const last3d = new Date(Date.now() - 3 * 86400000)
      .toISOString().split("T")[0];

    const overlays: Record<string, L.TileLayer> = {
      // 구름 (MODIS True Color, Aqua) — 발표 데모 fav
      "☁️ NASA Clouds (어제)": L.tileLayer(
        `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Aqua_CorrectedReflectance_TrueColor/default/${yest}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
        {
          maxZoom: 9,
          opacity: 0.5,
          attribution: "Imagery © NASA EOSDIS GIBS",
          tileSize: 256,
        },
      ),
      // 대기 오염 (Aerosol Optical Depth) — ILI 약한 상관 가능
      "🌫️ NASA AOD (대기오염)": L.tileLayer(
        `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Combined_Value_Added_AOD/default/${last3d}/GoogleMapsCompatible_Level6/{z}/{y}/{x}.png`,
        {
          maxZoom: 6,
          opacity: 0.6,
          attribution: "© NASA EOSDIS GIBS / MODIS Aerosol",
          tileSize: 256,
        },
      ),
      // 지표 온도 (Land Surface Temperature, Aqua Day)
      "🌡️ NASA LST (지표온도)": L.tileLayer(
        `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Aqua_Land_Surface_Temp_Day/default/${last3d}/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png`,
        {
          maxZoom: 7,
          opacity: 0.5,
          attribution: "© NASA EOSDIS GIBS / MODIS LST",
          tileSize: 256,
        },
      ),
      // 적설 (Snow Cover) — 인플루엔자 시즌
      "❄️ NASA Snow (적설)": L.tileLayer(
        `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_Snow_Cover/default/${yest}/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png`,
        {
          maxZoom: 8,
          opacity: 0.5,
          attribution: "© NASA EOSDIS GIBS / MODIS Snow Cover",
          tileSize: 256,
        },
      ),
      // 야간 빛 (VIIRS, 인구 밀도 proxy)
      "🌃 NASA Night Lights": L.tileLayer(
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/2016-01-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png",
        {
          maxZoom: 8,
          opacity: 0.7,
          attribution: "© NASA EOSDIS GIBS / VIIRS Black Marble",
          tileSize: 256,
        },
      ),
    };

    L.control.layers(baseLayers, overlays, {
      position: "topright",
      collapsed: true,
    }).addTo(map);

    mapRef.current = map;

    // ── Size fixes ───────────────────────────────────────────────────
    //
    // Leaflet computes its canvas size from the container's offsetWidth
    // only at init time, which is *before* the PanelGroup has finished
    // layout (height=0, tiles look like a single horizontal band) and
    // before `md:`-breakpoint flex sizes are applied. Worse, when the
    // user drags the PanelResizeHandle, Leaflet still thinks the pane
    // is the original width — tiles don't extend and zoom feels stuck.
    //
    // Fix = two things:
    //   1. requestAnimationFrame(invalidateSize) twice after mount so
    //      the first paint lines up with the real height, and
    //   2. a ResizeObserver on the container so every drag / window
    //      resize pushes Leaflet a fresh offsetWidth/Height.
    const kickSize = () => {
      if (mapRef.current) mapRef.current.invalidateSize({ pan: false });
    };
    const raf1 = requestAnimationFrame(() => {
      kickSize();
      requestAnimationFrame(kickSize);
    });
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(kickSize);
      ro.observe(container);
    }
    window.addEventListener("resize", kickSize);

    return () => {
      cancelAnimationFrame(raf1);
      window.removeEventListener("resize", kickSize);
      ro?.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, [L]);

  // Seoul outer boundary layer — non-interactive, drawn ABOVE the gu
  // choropleth so the silhouette stays visible against dark fills.
  // Panes make sure z-order is deterministic across Leaflet versions
  // regardless of which layer was added last.
  useEffect(() => {
    if (!L || !mapRef.current) return;
    const map = mapRef.current;
    if (!map.getPane("boundaryPane")) {
      const pane = map.createPane("boundaryPane");
      // SVG/canvas pane z-index default is 400; push boundary to 450.
      pane.style.zIndex = "450";
      pane.style.pointerEvents = "none";
    }
    let cancelled = false;
    void (async () => {
      try {
        const resp = await fetch(boundaryUrl, { cache: "force-cache" });
        if (!resp.ok) return; // boundary is optional — skip quietly
        const gj = (await resp.json()) as GeoJSON.FeatureCollection;
        if (cancelled || !mapRef.current) return;
        if (boundaryLayerRef.current) {
          mapRef.current.removeLayer(boundaryLayerRef.current);
        }
        const layer = L.geoJSON(gj, {
          pane: "boundaryPane",
          interactive: false,
          style: {
            color: "#fcd34d",
            weight: 3,
            opacity: 0.95,
            fillOpacity: 0,
            lineCap: "round",
            lineJoin: "round",
          },
        }).addTo(mapRef.current);
        boundaryLayerRef.current = layer;
      } catch {
        /* boundary is purely cosmetic — don't surface fetch errors */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [L, boundaryUrl]);

  useEffect(() => {
    if (!L || !mapRef.current) return;
    let cancelled = false;
    void (async () => {
      try {
        const resp = await fetch(geojsonUrl, { cache: "force-cache" });
        if (!resp.ok) throw new Error(`geojson HTTP ${resp.status}`);
        const gj = (await resp.json()) as GeoJSON.FeatureCollection;
        if (cancelled) return;

        if (layerRef.current && mapRef.current) {
          mapRef.current.removeLayer(layerRef.current);
        }

        const layer = L.geoJSON(gj, {
          style: (feat) => {
            const gu = (feat?.properties as { gu_nm?: string })?.gu_nm ?? "";
            const v = valueByGu.get(gu);
            // Make the selected district pop visually — brighter
            // border + thicker stroke. Previously the 2-px border on
            // #fde68a was too subtle on the mid-range choropleth
            // colours and users couldn't tell clicks had registered.
            return {
              color: gu === selectedGu ? "#fcd34d" : "#94a3b8",
              weight: gu === selectedGu ? 3.5 : 0.6,
              fillColor: colorFor(v ?? 0, vmax),
              fillOpacity: v != null ? 0.75 : 0.35,
            };
          },
          onEachFeature: (feat, lyr) => {
            const gu = (feat?.properties as { gu_nm?: string })?.gu_nm ?? "";
            // Capture the gu centroid once while the GeoJSON layer is
            // being built — the commuter-edge layer needs lat/lng pairs
            // to render polylines, and computing them from the layer's
            // own bounds avoids a separate polygon-centroid pass.
            if (gu && "getBounds" in lyr) {
              try {
                // @ts-expect-error — GeoJSON path layers have getBounds().
                const c = lyr.getBounds().getCenter();
                centroidsRef.current.set(gu, [c.lat, c.lng]);
              } catch {
                /* non-polygon features (labels, etc.) — skip. */
              }
            }
            lyr.bindTooltip(
              () => {
                const v = valueByGu.get(gu);
                const unit = unitRef.current;
                if (v == null) return gu;
                return unit
                  ? `${gu} · ${v.toFixed(2)} ${unit}`
                  : `${gu} · ${v.toFixed(2)}`;
              },
              { sticky: true },
            );
            lyr.on("click", () => {
              onSelectRef.current?.(gu);
            });

            // leaflet-contextmenu: bind at construction via the
            // plugin's ``bindContextMenu`` API. Mutating
            // ``lyr.options.contextmenu`` post-hoc doesn't take — the
            // plugin snapshots options when the layer is added, so
            // late mutation is a no-op and right-clicks fall through
            // to the browser menu. Building the items from ``tRef``
            // so Korean/English stays fresh without re-binding.
            const items = [
              {
                text: tRef.current("askForecast", { gu }),
                callback: () =>
                  onContextActionRef.current?.("ask_forecast", gu),
              },
              {
                text: tRef.current("askRt", { gu }),
                callback: () => onContextActionRef.current?.("ask_rt", gu),
              },
              {
                text: tRef.current("askShap", { gu }),
                callback: () => onContextActionRef.current?.("ask_shap", gu),
              },
              { separator: true },
              {
                text: tRef.current("copyName", { gu }),
                callback: () =>
                  onContextActionRef.current?.("copy_name", gu),
              },
            ];
            // @ts-expect-error — method added by leaflet-contextmenu at runtime.
            if (typeof lyr.bindContextMenu === "function") {
              // @ts-expect-error — plugin-augmented method.
              lyr.bindContextMenu({
                contextmenu: true,
                contextmenuInheritItems: false,
                contextmenuItems: items,
              });
            } else {
              // Fallback: pre-plugin versions exposed the same data via
              // options. Keeps the menu working if an older build of
              // leaflet-contextmenu is ever pinned.
              // @ts-expect-error — plugin extends options at runtime.
              lyr.options.contextmenu = true;
              // @ts-expect-error — plugin extends options at runtime.
              lyr.options.contextmenuItems = items;
            }
          },
        }).addTo(mapRef.current!);
        layerRef.current = layer;
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [L, geojsonUrl, valueByGu, vmax, selectedGu]);

  // ── Commuter edge polylines ────────────────────────────────────────
  //
  // Build once when (L && commuterEdges && centroids) are all ready, then
  // restyle per frame in the next effect. We keep one polyline per edge
  // in ``edgePolylinesRef`` (same order as ``commuterEdges``) so the
  // style effect can drive them without lookups.
  useEffect(() => {
    if (!L || !mapRef.current) return;
    if (!commuterEdges.length) return;
    // Defer one rAF so the GeoJSON effect has populated centroidsRef.
    let disposed = false;
    const raf = requestAnimationFrame(() => {
      if (disposed || !mapRef.current || !L) return;
      const map = mapRef.current;
      if (!map.getPane("edgePane")) {
        const pane = map.createPane("edgePane");
        // Between the gu fills (400) and the boundary (450) — keeps the
        // yellow silhouette on top of the orange commuter lines.
        pane.style.zIndex = "430";
        pane.style.pointerEvents = "none";
      }
      // Clear any previous polylines — e.g. when commuterEdges reloads.
      for (const p of edgePolylinesRef.current) {
        if (p) map.removeLayer(p);
      }
      const out: Array<import("leaflet").Polyline | null> = [];
      for (const e of commuterEdges) {
        const a = centroidsRef.current.get(e.src);
        const b = centroidsRef.current.get(e.dst);
        if (!a || !b) {
          out.push(null);
          continue;
        }
        const pl = L.polyline([a, b], {
          pane: "edgePane",
          // Default hidden style — the restyle effect will paint it. We
          // still mount the layer so Leaflet caches the SVG path elt and
          // subsequent setStyle() calls are cheap.
          color: "#fb923c",
          weight: 0,
          opacity: 0,
          lineCap: "round",
          className: "frame-d-edge-pulse",
          interactive: false,
        });
        pl.addTo(map);
        out.push(pl);
      }
      edgePolylinesRef.current = out;
    });
    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      // Don't remove layers here — keep them mounted so the toggle can
      // flip on/off by visibility, not by rebuild churn. The map.remove()
      // in the outer mount effect handles full cleanup.
    };
  }, [L, commuterEdges]);

  // Per-frame edge restyle. Fires when ``activeRows`` (source intensity)
  // or ``showEdges`` toggles. Computes ``flux = weight * value_at_src``
  // for each edge, picks top-N, paints those; hides the rest.
  useEffect(() => {
    if (!L) return;
    const lines = edgePolylinesRef.current;
    if (!lines.length || !commuterEdges.length) return;
    if (!showEdges) {
      // Hidden: zero every weight/opacity so the layer disappears cleanly.
      for (const p of lines) {
        if (p) p.setStyle({ weight: 0, opacity: 0 });
      }
      return;
    }
    // Flux for each edge. ``valueByGu`` can be all zero (demo, no sim) —
    // fall back to the weight alone so the corridors are still visible
    // on an idle map.
    const fluxes: Array<{ idx: number; flux: number }> = commuterEdges.map(
      (e, idx) => {
        const srcV = valueByGu.get(e.src) ?? 0;
        const base = srcV > 0 ? srcV : 1;
        return { idx, flux: e.weight * base };
      },
    );
    const sorted = [...fluxes].sort((a, b) => b.flux - a.flux);
    const keep = new Set(sorted.slice(0, EDGE_TOP_N).map((f) => f.idx));
    const fluxMax = sorted[0]?.flux ?? 1;
    for (let i = 0; i < commuterEdges.length; i++) {
      const p = lines[i];
      if (!p) continue;
      if (!keep.has(i)) {
        p.setStyle({ weight: 0, opacity: 0 });
        continue;
      }
      const f = fluxes[i].flux;
      // Quadratic scaling ensures the dominant Gangnam–Eunpyeong pair
      // reads as "clearly dominant" rather than just 2× the smallest.
      const norm = fluxMax > 0 ? f / fluxMax : 0;
      const weight = 0.8 + 4.6 * Math.sqrt(norm);
      const opacity = 0.28 + 0.52 * norm;
      p.setStyle({
        color: "#fb923c",
        weight,
        opacity,
      });
    }
  }, [L, commuterEdges, valueByGu, showEdges]);

  /**
   * Prefer the live payload's own label (which may include a seasonal
   * tag like "ILI 실측 (KDCA)") — it's more specific than our static i18n
   * bag. Fall back to the i18n label for pre-fetch renders.
   */
  const metricLabel = (m: LiveMetricId): string => {
    const live = overlays?.metrics?.[m];
    if (live) return locale === "ko" ? live.label_ko : live.label_en;
    // Static fallback by metric id. Keep in lockstep with
    // ``lib/live-overlays/types.ts :: LiveMetricId``.
    switch (m) {
      case "ili_forecast": return t("overlayIliForecast");
      case "ili_live":     return t("overlayIliLive");
      case "ili_alert":    return t("overlayIliAlert");
      case "air":          return t("overlayAir");
      case "temp":         return t("overlayTemp");
      case "humidity":     return t("overlayHumidity");
      case "er":           return t("overlayER");
      case "metro":        return t("overlayMetro");
      default:             return m;
    }
  };

  /** Current freshness, falling back to OFFLINE when no payload. */
  const activeFreshness: Freshness = activePayload?.freshness ?? "offline";
  const freshnessChip = (() => {
    switch (activeFreshness) {
      case "live":
        return {
          label: t("freshnessLive"),
          tooltip: t("freshnessTooltipLive"),
          cls: "border-emerald-500/60 bg-emerald-600/20 text-emerald-100",
        };
      case "stale":
        return {
          label: t("freshnessStale"),
          tooltip: t("freshnessTooltipStale"),
          cls: "border-amber-500/60 bg-amber-600/20 text-amber-100",
        };
      default:
        return {
          label: t("freshnessOffline"),
          tooltip: t("freshnessTooltipOffline"),
          cls: "border-slate-600 bg-slate-800/70 text-slate-300",
        };
    }
  })();

  return (
    <div className="relative h-full w-full" data-map-root="true">
      {/* Commuter-edge pulse animation. Leaflet renders polylines into an
          SVG path element; attaching ``className: 'frame-d-edge-pulse'``
          to the polyline options makes the path pick up this dashed-dash
          offset keyframe, so the orange corridors appear to *flow* from
          the source gu toward the destination. Kept inline (rather than
          in globals.css) because it only applies here and sitting next
          to the code that emits the class makes the coupling obvious. */}
      <style>{`
        @keyframes frame-d-edge-dash {
          from { stroke-dashoffset: 0; }
          to   { stroke-dashoffset: -24; }
        }
        .frame-d-edge-pulse {
          stroke-dasharray: 6 8;
          animation: frame-d-edge-dash 1.6s linear infinite;
        }
      `}</style>
      <div ref={containerRef} className="h-full w-full" />

      {/* Overlay picker — Sprint 2026-05-07 (#9 user critique): zoom 버튼 가림
          fix. left-12 (zoom 버튼 옆) 으로 이동 + collapse toggle 추가. */}
      <button
        type="button"
        onClick={() => setOverlayPanelHidden((v) => !v)}
        className="pointer-events-auto absolute left-2 top-[88px] z-[600] rounded-md border border-slate-700 bg-slate-900/90 px-1.5 py-0.5 text-[10px] text-slate-300 shadow-lg hover:bg-slate-800"
        aria-label={overlayPanelHidden ? "show overlay" : "hide overlay"}
        title={overlayPanelHidden ? "오버레이 보이기" : "오버레이 숨기기"}
      >
        {overlayPanelHidden ? "▸ overlay" : "▾ overlay"}
      </button>
      <div
        className={`pointer-events-auto absolute left-12 top-2 z-[400] flex flex-col gap-1 ${
          overlayPanelHidden ? "hidden" : ""
        }`}
      >
        <label className="flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900/90 px-2 py-1 text-[11px] text-slate-200 shadow-lg">
          <span className="flex items-center gap-1 font-medium text-slate-300">
            <span>{t("overlayLabel")}</span>
            <HelpIcon label={t("helpOverlay")} content={t("helpOverlay")} side="bottom" />
          </span>
          <select
            value={overlayMetric}
            onChange={(e) => setOverlayMetric(e.target.value as OverlayMetric)}
            className="rounded border border-slate-700 bg-slate-950 px-1 py-0.5 text-[11px] text-slate-200 focus:border-sky-500 focus:outline-none"
          >
            <option value="none">{t("overlayNone")}</option>
            {/* Grouped so the 8 layers don't read as a flat bag — the
                user scans the group labels first ("which physical
                domain?") before picking the exact metric. */}
            <optgroup label={t("overlayGroupIli")}>
              <option value="ili_forecast">{metricLabel("ili_forecast")}</option>
              <option value="ili_live">{metricLabel("ili_live")}</option>
              <option value="ili_alert">{metricLabel("ili_alert")}</option>
            </optgroup>
            <optgroup label={t("overlayGroupEnv")}>
              <option value="air">{metricLabel("air")}</option>
              <option value="temp">{metricLabel("temp")}</option>
              <option value="humidity">{metricLabel("humidity")}</option>
            </optgroup>
            <optgroup label={t("overlayGroupOps")}>
              <option value="er">{metricLabel("er")}</option>
              <option value="metro">{metricLabel("metro")}</option>
            </optgroup>
          </select>
        </label>
        {/* Freshness chip — always visible when a layer is active so the
            user can tell at a glance whether they're staring at real-time
            data, a stale snapshot, or the deterministic synthetic
            fallback (OFFLINE). The tooltip explains what the badge means. */}
        {overlayMetric !== "none" ? (
          <div
            className={[
              "flex items-center justify-between gap-2 rounded-md border px-2 py-0.5 text-[10px] font-medium shadow-lg",
              freshnessChip.cls,
            ].join(" ")}
            title={freshnessChip.tooltip}
          >
            <span className="flex items-center gap-1">
              <span>{freshnessChip.label}</span>
              <HelpIcon
                label={freshnessChip.tooltip}
                content={freshnessChip.tooltip}
                side="bottom"
              />
            </span>
            {activePayload?.source ? (
              <span className="truncate font-mono text-[9px] opacity-80">
                {t("overlaySourceLabel")}: {activePayload.source}
              </span>
            ) : null}
          </div>
        ) : null}
        {/* Commuter edge overlay toggle. Kept next to the metric picker
            so the user has all map-layer controls in one place. Defaults
            ON because the whole point of the 전파 animation is to *see*
            the corridors. */}
        <label className="flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900/90 px-2 py-1 text-[11px] text-slate-200 shadow-lg">
          <span className="flex items-center gap-1 font-medium text-slate-300">
            <span>{t("edgeLayerLabel")}</span>
            <HelpIcon
              label={t("helpEdgeLayer")}
              content={t("helpEdgeLayer")}
              side="bottom"
            />
          </span>
          <button
            type="button"
            onClick={() => setShowEdges((v) => !v)}
            aria-pressed={showEdges}
            className={[
              "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px]",
              showEdges
                ? "border-orange-500/60 bg-orange-500/20 text-orange-100"
                : "border-slate-700 bg-slate-950 text-slate-300",
            ].join(" ")}
          >
            <span aria-hidden="true">{showEdges ? "●" : "○"}</span>
            <span>{showEdges ? "on" : "off"}</span>
          </button>
        </label>
        {/* Per-layer observed_at (what matters for freshness) —
            distinct from the orchestrator's ``generated_at`` which only
            tells you when the Edge function ran. For the static
            fallback case (observed_at = 1970-…) we suppress the badge
            because "updated 1970-01-01" is worse than no stamp at all. */}
        {overlayMetric !== "none" &&
        activePayload?.observed_at &&
        !activePayload.observed_at.startsWith("1970") ? (
          <LastUpdated at={activePayload.observed_at} />
        ) : null}
        {/* Fetch-level error chip. Distinct from the freshness badge —
            this fires only when the HTTP request itself failed (net
            down, 5xx from the Edge route). Providers that returned 0
            rows surface via the OFFLINE freshness badge instead. */}
        {overlayError ? (
          <div className="rounded-md border border-red-700 bg-red-950/80 px-2 py-0.5 text-[10px] text-red-200 shadow-lg">
            overlays: {overlayError}
          </div>
        ) : null}
        {overlayMetric !== "none" && activeRows.length > 0 ? (
          <div className="rounded-md border border-slate-700 bg-slate-900/90 px-2 py-1 text-[10px] text-slate-300 shadow-lg">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="font-medium">{t("overlayLegend")}</span>
              {activeUnit ? (
                <span className="text-slate-400">{activeUnit}</span>
              ) : null}
            </div>
            <div className="flex items-center gap-1.5">
              <span className="tabular-nums">
                {Number.isFinite(vmin) ? vmin.toFixed(1) : "0"}
              </span>
              <span
                aria-hidden="true"
                className="h-1.5 w-20 rounded-full"
                style={{
                  background:
                    "linear-gradient(to right, rgb(11,15,20), rgb(59,48,126), rgb(86,123,181), rgb(146,201,171), rgb(240,236,90))",
                }}
              />
              <span className="tabular-nums">
                {vmax ? vmax.toFixed(1) : "0"}
              </span>
            </div>
            {/* Per-layer provenance hint (e.g. "week of 2025-W12",
                "no live source configured — deterministic demo"). */}
            {activePayload?.note ? (
              <div
                className="mt-1 text-[10px] italic text-slate-400"
                title={activePayload.note}
              >
                {activePayload.note.length > 60
                  ? `${activePayload.note.slice(0, 60)}…`
                  : activePayload.note}
              </div>
            ) : (
              <div className="mt-1 text-[10px] italic text-slate-500">
                {t("overlayHint")}
              </div>
            )}
          </div>
        ) : null}
      </div>

      {/* Top-center Day / Week / Peak HUD — driven by the
          TransmissionPlayer. Deliberately big typography so the viewer
          IMMEDIATELY sees where in the simulated season we are. When
          ``playbackHud`` is null (no simulation running) the HUD stays
          hidden so the map isn't cluttered on idle. */}
      {playbackHud ? <PlaybackHudPanel hud={playbackHud} /> : null}

      {error ? (
        <div className="pointer-events-none absolute left-2 bottom-2 z-[500] rounded-md border border-red-700 bg-red-950/80 px-2 py-1 text-[11px] text-red-200">
          map: {error}
        </div>
      ) : null}
      {selectedGu ? (
        <div className="pointer-events-none absolute right-2 top-2 z-[500] rounded-md border border-amber-500/60 bg-slate-900/90 px-2 py-1 text-xs text-amber-200 shadow-lg">
          <span className="font-medium">{selectedGu}</span>
          <span className="ml-2 text-[10px] text-slate-400">
            {t("rightClickHint")}
          </span>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Big-typography HUD drawn over the map while the TransmissionPlayer is
 * active. Shows ``Day X / Y``, ``Week W (2025-W12)``, distance-to-peak,
 * cumulative infections, active-gu count, and the 3 hottest districts.
 *
 * Lives inside MapPanel.tsx rather than a separate file because it's a
 * tight coupling to the map's coordinate system (absolute positioning)
 * and carries no reusable state — inlining keeps the data flow local.
 */
function PlaybackHudPanel({ hud }: { hud: PlaybackHud }) {
  const { t } = useT();
  const weekNum = hud.weekIdx + 1;
  const peakDelta = hud.peakWeek - hud.weekIdx;
  const atPeak = peakDelta === 0;
  const beforePeak = peakDelta > 0;
  const peakText = atPeak
    ? t("hudAtPeak")
    : beforePeak
      ? t("hudPeakInWeeks", { n: peakDelta })
      : t("hudPastPeakWeeks", { n: -peakDelta });
  const dotColor = hud.playing ? "bg-rose-400" : "bg-sky-400";
  return (
    <div
      className="pointer-events-none absolute left-1/2 top-2 z-[500] -translate-x-1/2 rounded-lg border border-slate-700 bg-slate-900/90 px-3 py-1.5 text-slate-100 shadow-xl backdrop-blur"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center gap-3">
        <span
          className={["inline-block h-2 w-2 rounded-full", dotColor].join(" ")}
          aria-hidden="true"
        />
        <div className="flex items-baseline gap-2">
          <span className="text-[20px] font-semibold tabular-nums leading-none">
            Day {hud.dayIdx}
          </span>
          <span className="text-[11px] text-slate-400">/ {hud.totalDays}</span>
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-[16px] font-medium tabular-nums leading-none">
            W{weekNum}
          </span>
          {hud.weekLabel ? (
            <span className="text-[11px] text-slate-400">({hud.weekLabel})</span>
          ) : null}
          <span className="text-[10px] text-slate-500">/ {hud.nFrames}</span>
        </div>
        <span
          className={[
            "rounded-md px-1.5 py-0.5 text-[10px] font-medium",
            atPeak
              ? "bg-rose-600/70 text-rose-50"
              : beforePeak
                ? "bg-amber-700/50 text-amber-100"
                : "bg-emerald-700/40 text-emerald-100",
          ].join(" ")}
        >
          {peakText}
        </span>
      </div>
      <div className="mt-0.5 flex items-center gap-3 text-[11px] text-slate-300">
        <span>
          {t("hudCumulative")}:{" "}
          <span className="tabular-nums text-slate-100">
            {hud.cumulative.toLocaleString()}
          </span>
        </span>
        <span>
          {t("hudActiveGu")}:{" "}
          <span className="tabular-nums text-slate-100">
            {hud.activeGuCount}
          </span>
          <span className="text-slate-500">/25</span>
        </span>
        {hud.topGus.length > 0 ? (
          <span className="truncate text-slate-400">
            · {hud.topGus.join(" · ")}
          </span>
        ) : null}
      </div>
    </div>
  );
}
