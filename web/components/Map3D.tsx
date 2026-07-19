"use client";
/**
 * Map3D — 3D 지도 시각화 mini-prototype (G1 path).
 *
 * Sprint 2026-05-06 (#16): 사용자 영감 이미지 (Phoenix LST heatmap / drone flock
 * trail / voronoi tessellation / Detroit categorical) 를 DeckGL stack 으로
 * cover. dark MapLibre base + HeatmapLayer + ArcLayer + TripsLayer + time slider.
 *
 * 후속 (별 sprint): real sim_runs/*.npz 통합, voronoi tessellation, 3D column
 * extrusion, paper §5.7 ARIA Stage 6 evidence 강화.
 *
 * 참조: docs/3D_MAP_OPTIONS.md
 */
import { useState, useMemo, useEffect } from "react";
import DeckGL from "@deck.gl/react";
import { LinearInterpolator } from "@deck.gl/core";
import { Map } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import { ArcLayer, GeoJsonLayer, ScatterplotLayer, LineLayer } from "@deck.gl/layers";
import { TripsLayer } from "@deck.gl/geo-layers";

/** Seoul Metro line hex color → RGB triple for deck.gl. */
function hexToRgb(h: string): [number, number, number] {
  const n = parseInt((h || "#888888").slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Seoul 25-gu center coordinates (approximate centroid). Real: load from
// simulation/data/external/seoul_gu_centroid.csv (next sprint).
const SEOUL_GU = [
  { name: "강남구", lng: 127.0473, lat: 37.5172 },
  { name: "강동구", lng: 127.1238, lat: 37.5301 },
  { name: "강북구", lng: 127.0257, lat: 37.6396 },
  { name: "강서구", lng: 126.8495, lat: 37.5509 },
  { name: "관악구", lng: 126.9514, lat: 37.4783 },
  { name: "광진구", lng: 127.0823, lat: 37.5384 },
  { name: "구로구", lng: 126.8874, lat: 37.4954 },
  { name: "금천구", lng: 126.8954, lat: 37.4519 },
  { name: "노원구", lng: 127.0566, lat: 37.6543 },
  { name: "도봉구", lng: 127.0471, lat: 37.6688 },
  { name: "동대문구", lng: 127.0398, lat: 37.5744 },
  { name: "동작구", lng: 126.9393, lat: 37.5124 },
  { name: "마포구", lng: 126.9015, lat: 37.5664 },
  { name: "서대문구", lng: 126.9367, lat: 37.5793 },
  { name: "서초구", lng: 127.0327, lat: 37.4837 },
  { name: "성동구", lng: 127.0367, lat: 37.5634 },
  { name: "성북구", lng: 127.0167, lat: 37.5894 },
  { name: "송파구", lng: 127.1058, lat: 37.5145 },
  { name: "양천구", lng: 126.8666, lat: 37.5169 },
  { name: "영등포구", lng: 126.8956, lat: 37.5263 },
  { name: "용산구", lng: 126.9788, lat: 37.5326 },
  { name: "은평구", lng: 126.9292, lat: 37.6027 },
  { name: "종로구", lng: 126.9794, lat: 37.5735 },
  { name: "중구", lng: 126.9974, lat: 37.5636 },
  { name: "중랑구", lng: 127.0925, lat: 37.6063 },
];

const INITIAL_VIEW = {
  longitude: 126.978,
  latitude: 37.5665,
  zoom: 10.5,
  pitch: 50,
  bearing: -15,
};

// Viridis colormap (purple → green → yellow). Public-domain reference.
const VIRIDIS_RGBA: [number, number, number, number][] = [
  [68, 1, 84, 0],
  [59, 82, 139, 200],
  [33, 144, 141, 220],
  [93, 201, 99, 240],
  [253, 231, 37, 255],
];

type FlowEdge = {
  fromIdx: number;
  toIdx: number;
  weight: number;
};

/** Sprint 2026-05-06 Phase B.2 — real commuter data API response. */
type CommuterEdge = {
  origin: string;
  dest: string;
  coupling: number;
  night_population: number;
  source_lng: number;
  source_lat: number;
  target_lng: number;
  target_lat: number;
};

/** Sprint 2026-05-06 Phase B.3 — hospital marker (404 + ER 250). */
type Hospital = {
  name: string;
  addr: string;
  gu: string;
  clcd: string;
  bed_cnt: number;
  dr_cnt: number;
  lat: number;
  lng: number;
  has_er: boolean;
  er_beds: number | null;
  er_icu: number | null;
};

// Demo commuter flows (random 8 hubs → 25-gu) — fallback when /api fails.
function makeDemoFlows(): FlowEdge[] {
  const HUBS = [0, 14, 17, 12, 13, 22, 23, 20]; // Gangnam / Seocho / Songpa / Mapo / Seodaemun / Eunpyeong / Jongno / Yongsan
  const flows: FlowEdge[] = [];
  HUBS.forEach((from) => {
    SEOUL_GU.forEach((_, to) => {
      if (from === to) return;
      flows.push({
        fromIdx: from,
        toIdx: to,
        weight: 0.1 + Math.random() * 0.9,
      });
    });
  });
  return flows.filter((f) => f.weight > 0.4); // top 60%
}

// Demo time-evolved trip paths. Real: from sim ABM trail (next sprint).
function makeDemoTrips(t: number) {
  return SEOUL_GU.slice(0, 12).map((g, i) => {
    const angle = (i / 12) * Math.PI * 2 + t * 0.01;
    return {
      path: [
        [g.lng, g.lat],
        [g.lng + Math.cos(angle) * 0.04, g.lat + Math.sin(angle) * 0.04],
        [g.lng + Math.cos(angle + 0.5) * 0.06, g.lat + Math.sin(angle + 0.5) * 0.06],
      ],
      timestamps: [t, t + 30, t + 60],
      severity: i % 3,
    };
  });
}

// ── Complexity control: layer groups × presets ──────────────────────────────
const PRESET_LABELS = { simple: "간단", detail: "상세", full: "전체" } as const;
type Preset = keyof typeof PRESET_LABELS;

/** Toggleable layer groups, in panel order: preset membership + legend (size/색). */
const LAYER_GROUPS: { id: string; label: string; presets: Preset[]; legend: string }[] = [
  { id: "boundary", label: "구 경계", presets: ["simple", "detail", "full"], legend: "25개 자치구 경계선" },
  { id: "ili", label: "ILI 히트맵", presets: ["simple", "detail", "full"], legend: "색 = ILI rate (보라 낮음 → 노랑 높음)" },
  { id: "iliExtrusion", label: "ILI 3D 높이", presets: ["simple", "detail", "full"], legend: "높이 = 구별 ILI rate · 색 = alert(빨강 ≥ 피크 70%) · 360일 SEIR 파동 애니메이션" },
  { id: "airQuality", label: "대기질·미세먼지", presets: ["detail", "full"], legend: "구 색 = PM10 (녹색 좋음 → 빨강 매우나쁨)" },
  { id: "vaccination", label: "인플루엔자 백신", presets: ["detail", "full"], legend: "구 색 = 접종률% (낮을수록 빨강)" },
  { id: "disease", label: "법정감염병 발생", presets: ["detail", "full"], legend: "구 색 = 발생 건수 (많을수록 진한 자주)" },
  { id: "subwayLines", label: "지하철 노선", presets: ["detail", "full"], legend: "8개 노선 공식색" },
  { id: "subwayStations", label: "지하철역·승하차", presets: ["detail", "full"], legend: "크기 = 승하차량 · 색 = 구 연령(청록 청년 ↔ 자홍 고령)" },
  { id: "transferHubs", label: "환승역(교합점)", presets: ["detail", "full"], legend: "노란 링 · 크기 = 환승 노선 수" },
  { id: "agentTrips", label: "agent 이동", presets: ["detail", "full"], legend: "인구집단별 통근 이동 (애니메이션)" },
  { id: "population", label: "실시간 인구·연령", presets: ["full"], legend: "크기 = 인구 · 색 = 혼잡도(여유 녹색 → 붐빔 빨강)" },
  { id: "forecast", label: "인구 예보(혼잡)", presets: ["full"], legend: "크기 = 예보 인구 · 색 = 예보 혼잡도" },
  { id: "subwayCrowd", label: "지하철 혼잡(POI)", presets: ["full"], legend: "크기 = 누적 승차량 (POI 79곳 중 68)" },
  { id: "category", label: "관광지역 분류", presets: ["full"], legend: "색 = 장소유형(관광특구 빨강·공원 녹색·고궁 보라·역세권 파랑·상권 주황)" },
  { id: "traffic", label: "도로교통", presets: ["full"], legend: "링 색 = 소통(원활 녹색 / 서행 노랑 / 정체 빨강)" },
  { id: "busRoutes", label: "버스 노선", presets: ["full"], legend: "간선 호박색 / 지선 회청색" },
  { id: "busStops", label: "버스 정류소", presets: ["full"], legend: "색 = 정류소 유형(중앙차로 빨강 등)" },
  { id: "busRidership", label: "버스 승하차", presets: ["full"], legend: "크기 = 일 승하차량 (709개 표본)" },
  { id: "wind", label: "바람(S-DoT)", presets: ["full"], legend: "화살표 = 풍향 · 길이 = 풍속 (6개 구)" },
  { id: "commuter", label: "통근 흐름", presets: ["full"], legend: "아크 = 구간 통근량" },
  { id: "hospitals", label: "병원·응급실", presets: ["full"], legend: "크기 = 병상수 · 색 = ER 가용(녹색 가용 / 빨강 만실·없음)" },
  { id: "schools", label: "학교", presets: ["full"], legend: "색 = 학교종류(초 녹색·중 파랑·고 주황·유치원 분홍·특수 보라) · gu중심 근사" },
];

/** Visibility map for a preset: layer id → on/off. */
function presetVis(p: Preset): Record<string, boolean> {
  return Object.fromEntries(LAYER_GROUPS.map((g) => [g.id, g.presets.includes(p)]));
}

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** PM10 air-quality fill (환경부 등급: 좋음≤30 / 보통≤80 / 나쁨≤150 / 매우나쁨). */
function pmColor(pm: number | undefined): [number, number, number, number] {
  if (pm == null) return [40, 40, 50, 50];
  if (pm <= 30) return [80, 190, 120, 150];
  if (pm <= 80) return [240, 210, 90, 160];
  if (pm <= 150) return [240, 150, 70, 185];
  return [230, 70, 90, 205];
}

/** gu age composition → diverging color (youth teal ↔ elderly magenta). */
function ageColor(
  age: { youth: number; adult: number; elderly: number } | null | undefined,
): [number, number, number] {
  if (!age) return [150, 150, 160];
  const t = clamp01((age.elderly - age.youth + 0.15) / 0.3);
  return [Math.round(60 + t * 170), Math.round(200 - t * 120), Math.round(200 - t * 40)];
}

/** Seoul 실시간 혼잡도(여유→붐빔) color. */
const CONGEST_COLOR: Record<string, [number, number, number]> = {
  여유: [80, 190, 120],
  보통: [240, 210, 90],
  "약간 붐빔": [240, 150, 70],
  붐빔: [230, 70, 90],
};
/** Road traffic index(원활/서행/정체) color. */
const TRAFFIC_COLOR: Record<string, [number, number, number]> = {
  원활: [80, 200, 130],
  서행: [240, 200, 80],
  정체: [230, 70, 90],
};

/** Influenza vaccination % → fill (low coverage = red risk, high = green). */
function vaxColor(pct: number | undefined): [number, number, number, number] {
  if (pct == null) return [40, 40, 50, 40];
  const t = clamp01((pct - 36) / 14); // observed ≈ 36–50%
  return [Math.round(230 - t * 150), Math.round(80 + t * 110), 90, 165];
}
/** Notifiable-disease cases → fill (more = deeper magenta). */
function diseaseColor(cases: number | undefined, max: number): [number, number, number, number] {
  if (cases == null || max <= 0) return [40, 40, 50, 40];
  const t = clamp01(cases / max);
  return [Math.round(120 + t * 110), Math.round(60 - t * 30), Math.round(120 + t * 40), Math.round(70 + t * 120)];
}
/** Seoul place category → color. */
const CATEGORY_COLOR: Record<string, [number, number, number]> = {
  관광특구: [230, 70, 90],
  공원: [80, 190, 120],
  "고궁·문화유산": [170, 110, 220],
  역세권: [90, 140, 230],
  발달상권: [240, 160, 70],
  기타: [150, 150, 160],
};

type SubwayStation = {
  name: string;
  position: [number, number];
  lines: string[];
  n_lines: number;
  transfer: boolean;
  ridership: number;
  gu: string | null;
  age: { youth: number; adult: number; elderly: number } | null;
  hourly: number[] | null;
};
type AirEnv = {
  air: Record<string, { pm10: number; pm25: number; khai_grade: number; o3: number; no2: number }>;
  env: Record<string, { temperature?: number; humidity?: number; wind_speed?: number; wind_dir?: number }>;
  collected_at: string;
};
type WeatherPoint = { valid_at: string; TMP?: number; POP?: number; REH?: number; WSD?: number; SKY?: number; PTY?: number };
type Weather = { issued_at: string; forecast: WeatherPoint[]; historical: unknown[] };
type RealtimePoi = {
  area_nm: string;
  position: [number, number];
  category: string;
  congestion: string;
  ppltn_min: number;
  ppltn_max: number;
  ages: Record<string, number>;
  traffic_idx?: string;
  traffic_spd?: number;
  fcst_peak_ppltn?: number;
  fcst_peak_congest?: string;
  subway_on?: number;
  subway_off?: number;
};
type DiseaseVax = {
  vax: Record<string, number>;
  disease: Record<string, number>;
  vax_year: number;
  disease_year: number;
};
/** per-gu ILI rate from /aggregates/ili-local.json */
type IliLocal = {
  generated_at: string;
  observed_at: string;
  gu: Record<string, { ili: number; q70: number; alert: boolean }>;
};
/** 360-day SEIR forecast from /aggregates/seir-forecast-360.json */
type Forecast360Row = {
  day: number;
  date: string;
  city_ili: number;
  gu: Record<string, number>;
};
type Forecast360 = {
  forecast: Forecast360Row[];
};
type RealtimeData = {
  pois: RealtimePoi[];
  bike: { stations: number; bikes: number; shared_pct: number } | null;
  collected_at: string;
};
type ViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
  transitionDuration?: number;
  transitionInterpolator?: LinearInterpolator;
};

export function Map3D() {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [commuterEdges, setCommuterEdges] = useState<CommuterEdge[] | null>(null);
  const [hospitals, setHospitals] = useState<Hospital[] | null>(null);
  // Real Seoul Metro line geometry (8 lines, 254 stations) — static aggregate.
  const [subwayLines, setSubwayLines] = useState<GeoJSON.FeatureCollection | null>(
    null,
  );

  // Real subway route LineStrings from web/public/aggregates/subway-lines.geojson
  useEffect(() => {
    fetch("/aggregates/subway-lines.geojson")
      .then((r) => r.json())
      .then((d) => setSubwayLines(d))
      .catch(() => setSubwayLines(null));
  }, []);

  // Real per-agent commute trips (1,500), real route geometry.
  // New format: path = [[lon,lat],...] (avg 13.9 waypoints), timestamps = [t0,t1,...],
  //             group='adult'|'child'|'elderly', mode='subway'|'bus'|'walk',
  //             period='am'|'pm'.
  // AM trips: timestamps ~51-121 (home→work). PM trips: ~180-353 (work→home).
  const [agentTrips, setAgentTrips] = useState<
    { path: [number, number][]; timestamps: number[]; color: number[]; mode?: string; period?: string; group?: string }[] | null
  >(null);
  useEffect(() => {
    fetch("/aggregates/agent-trips.json")
      .then((r) => r.json())
      .then((d) => {
        const raw = d.trips ?? null;
        if (!raw) { setAgentTrips(null); return; }
        // Override color by mode for clear visual differentiation:
        // subway=#22c55e(green=[34,197,94]), bus=#3b82f6(blue=[59,130,246]), walk=#9ca3af(gray=[156,163,175])
        const colored = raw.map((tr: { path: [number,number][]; timestamps: number[]; color?: number[]; mode?: string; period?: string; group?: string }) => {
          let c: [number,number,number];
          if (tr.mode === 'subway') c = [34, 197, 94];
          else if (tr.mode === 'bus') c = [59, 130, 246];
          else c = [156, 163, 175];
          return { ...tr, color: c };
        });
        setAgentTrips(colored);
      })
      .catch(() => setAgentTrips(null));
  }, []);

  // Real Seoul bus stops (11,253) — ① trunk corridor (중앙차로, red) / ② density
  // (dense point cloud) / ③ full network, colored by stop type.
  const [busStops, setBusStops] = useState<
    { position: [number, number]; color: number[]; trunk: boolean; ridership?: number }[] | null
  >(null);
  useEffect(() => {
    fetch("/aggregates/bus-stops.json")
      .then((r) => r.json())
      .then((d) => setBusStops(d.stops ?? null))
      .catch(() => setBusStops(null));
  }, []);

  // Real Seoul bus ROUTE polylines (1,453 lines from masterRouteNode, joined to
  // stop coords). trunk_frac = fraction of nodes on 중앙차로 corridors → ① 간선
  // routes light up; all routes drawn = ③ full network.
  const [busRoutes, setBusRoutes] = useState<GeoJSON.FeatureCollection | null>(
    null,
  );
  useEffect(() => {
    fetch("/aggregates/bus-routes.geojson")
      .then((r) => r.json())
      .then((d) => setBusRoutes(d))
      .catch(() => setBusRoutes(null));
  }, []);

  // gu boundary polygons — outline (간단 base) + air-quality choropleth fill.
  const [guBoundary, setGuBoundary] = useState<GeoJSON.FeatureCollection | null>(null);
  useEffect(() => {
    fetch("/seoul-gu.geojson")
      .then((r) => r.json())
      .then((d) => setGuBoundary(d))
      .catch(() => setGuBoundary(null));
  }, []);

  // Enriched subway stations: ridership / transfer hubs / hourly pulse / gu age.
  const [subwayStations, setSubwayStations] = useState<SubwayStation[] | null>(null);
  useEffect(() => {
    fetch("/aggregates/subway-stations.json")
      .then((r) => r.json())
      .then((d) => setSubwayStations(d.stations ?? null))
      .catch(() => setSubwayStations(null));
  }, []);

  // Per-gu real-time air quality (미세먼지) + S-DoT environment (wind/temp).
  const [airEnv, setAirEnv] = useState<AirEnv | null>(null);
  useEffect(() => {
    fetch("/aggregates/air-env.json")
      .then((r) => r.json())
      .then((d) => setAirEnv(d))
      .catch(() => setAirEnv(null));
  }, []);

  // Weather forecast (72h) + historical trend, for the time-synced panel.
  const [weather, setWeather] = useState<Weather | null>(null);
  useEffect(() => {
    fetch("/aggregates/weather.json")
      .then((r) => r.json())
      .then((d) => setWeather(d))
      .catch(() => setWeather(null));
  }, []);

  // Real-time POI (실시간도시데이터 79 places): population·age + road traffic.
  const [realtime, setRealtime] = useState<RealtimeData | null>(null);
  useEffect(() => {
    fetch("/aggregates/realtime-poi.json")
      .then((r) => r.json())
      .then((d) => setRealtime(d))
      .catch(() => setRealtime(null));
  }, []);

  // Per-gu epidemiology: influenza vaccination % + notifiable-disease cases.
  const [diseaseVax, setDiseaseVax] = useState<DiseaseVax | null>(null);
  useEffect(() => {
    fetch("/aggregates/disease-vax.json")
      .then((r) => r.json())
      .then((d) => setDiseaseVax(d))
      .catch(() => setDiseaseVax(null));
  }, []);
  const diseaseMax = useMemo(
    () => (diseaseVax ? Math.max(1, ...Object.values(diseaseVax.disease)) : 1),
    [diseaseVax],
  );

  // Per-gu ILI rate (ili-local.json) — static snapshot fallback for extrusion.
  const [iliLocal, setIliLocal] = useState<IliLocal | null>(null);
  useEffect(() => {
    fetch("/aggregates/ili-local.json")
      .then((r) => r.json())
      .then((d) => setIliLocal(d))
      .catch(() => setIliLocal(null));
  }, []);
  // Max ILI rate across gu — fallback normalization when forecast360 is absent.
  const iliMax = useMemo(
    () => (iliLocal ? Math.max(1, ...Object.values(iliLocal.gu).map((v) => v.ili)) : 1),
    [iliLocal],
  );

  // 360-day SEIR forecast — drives animated 3D ILI extrusion.
  const [forecast360, setForecast360] = useState<Forecast360Row[] | null>(null);
  useEffect(() => {
    fetch("/aggregates/seir-forecast-360.json")
      .then((r) => r.json())
      .then((d: Forecast360) => setForecast360(d.forecast ?? null))
      .catch(() => setForecast360(null));
  }, []);
  // Global peak ILI across all gu × all days — computed once for stable height normalization.
  // Pre-computing here avoids per-frame O(days × gu) scans in getElevation callbacks.
  const forecast360GlobalPeak = useMemo(() => {
    if (!forecast360) return 1;
    let peak = 1;
    for (const row of forecast360) {
      for (const val of Object.values(row.gu)) {
        if (val > peak) peak = val;
      }
    }
    return peak;
  }, [forecast360]);

  // Schools (1,422) — gu-centroid spiral approximation, colored by kind.
  const [schools, setSchools] = useState<
    { position: [number, number]; kind: string; color: number[]; name: string }[] | null
  >(null);
  useEffect(() => {
    fetch("/aggregates/schools.json")
      .then((r) => r.json())
      .then((d) => setSchools(d.schools ?? null))
      .catch(() => setSchools(null));
  }, []);

  // 2D / 3D view toggle — drives the controlled viewState (pitch/bearing).
  const [is3D, setIs3D] = useState(true);
  const [viewState, setViewState] = useState<ViewState>(INITIAL_VIEW);
  const toggleDim = () => {
    setIs3D((d) => {
      const next = !d;
      setViewState((v) => ({
        ...v,
        pitch: next ? 50 : 0,
        bearing: next ? -15 : 0,
        transitionDuration: 450,
        transitionInterpolator: new LinearInterpolator(["pitch", "bearing"]),
      }));
      return next;
    });
  };

  // Complexity control: active preset + per-layer visibility overrides.
  const [preset, setPreset] = useState<Preset>("simple");
  const [vis, setVis] = useState<Record<string, boolean>>(() => presetVis("simple"));
  const applyPreset = (p: Preset) => {
    setPreset(p);
    setVis(presetVis(p));
  };
  const toggleLayer = (id: string) =>
    setVis((v) => ({ ...v, [id]: !v[id] }));

  // Sprint 2026-05-06 Phase B.2: real commuter_matrix fetch
  useEffect(() => {
    fetch("/api/overlays/commuter")
      .then((r) => r.json())
      .then((d) => setCommuterEdges(d.edges ?? []))
      .catch(() => setCommuterEdges(null));
  }, []);

  // Sprint 2026-05-06 Phase B.3: hospitals + ER availability
  useEffect(() => {
    fetch("/api/overlays/hospitals")
      .then((r) => r.json())
      .then((d) => setHospitals(d.hospitals ?? []))
      .catch(() => setHospitals(null));
  }, []);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setTime((t) => (t + 1) % 360), 80);
    return () => clearInterval(id);
  }, [playing]);

  // Demo data: 25-gu × random ILI rate (3-25 per 1000)
  const heatmapData = useMemo(
    () =>
      SEOUL_GU.map((g, i) => ({
        position: [g.lng, g.lat] as [number, number],
        weight: 3 + (Math.sin((time + i * 30) * 0.05) + 1) * 11,
      })),
    [time],
  );

  // ArcLayer: real commuter_matrix (KOSIS) if available, fallback demo.
  const arcData = useMemo(() => makeDemoFlows(), []);
  const tripData = useMemo(() => makeDemoTrips(time), [time]);

  // Slider time → hour-of-day (subway rush pulse) + forecast hour (weather panel).
  const hourOfDay = Math.floor((time / 360) * 24) % 24;
  const weatherNow =
    weather?.forecast && weather.forecast.length > 0
      ? weather.forecast[
          Math.min(
            weather.forecast.length - 1,
            Math.floor((time / 360) * weather.forecast.length),
          )
        ]
      : null;

  const layers = [
    // gu boundary outline — base layer present in every preset.
    guBoundary
      ? new GeoJsonLayer({
          id: "gu-boundary",
          data: guBoundary,
          visible: vis.boundary,
          stroked: true,
          filled: false,
          getLineColor: [120, 140, 180, 150],
          getLineWidth: 25,
          lineWidthUnits: "meters",
          lineWidthMinPixels: 1,
          lineWidthMaxPixels: 2,
        })
      : null,
    // ILI 3D extrusion choropleth — animated by 360-day SEIR forecast.
    // Height and color track the current time slider day.
    // Primary source: forecast360[time].gu[name] (float ILI per 1k).
    // Fallback: iliLocal static snapshot when forecast360 is not yet loaded.
    // Global peak pre-computed once → height scale is stable across all frames.
    // Alert threshold: ILI > 80th percentile of day-0 snapshot OR > 0.7×peak.
    guBoundary && (forecast360 || iliLocal)
      ? new GeoJsonLayer({
          id: "ili-extrusion",
          data: guBoundary,
          visible: vis.iliExtrusion,
          stroked: true,
          filled: true,
          extruded: true,
          wireframe: false,
          getElevation: (f: GeoJSON.Feature) => {
            const name = (f.properties?.name as string) ?? "";
            if (forecast360) {
              // clamp dayIdx to valid range [0, forecast360.length-1]
              const dayIdx = Math.min(time, forecast360.length - 1);
              const rate = forecast360[dayIdx]?.gu[name] ?? 0;
              return (rate / forecast360GlobalPeak) * 4000;
            }
            // fallback: static snapshot
            const rate = iliLocal!.gu[name]?.ili ?? 0;
            return (rate / iliMax) * 4000;
          },
          getFillColor: (f: GeoJSON.Feature) => {
            const name = (f.properties?.name as string) ?? "";
            if (forecast360) {
              const dayIdx = Math.min(time, forecast360.length - 1);
              const rate = forecast360[dayIdx]?.gu[name] ?? 0;
              const t = Math.min(1, rate / forecast360GlobalPeak);
              const alertThreshold = forecast360GlobalPeak * 0.7;
              return rate > alertThreshold
                ? [220 + Math.round(t * 35), 60, 70, 200] as [number, number, number, number]
                : [Math.round(60 + t * 80), Math.round(90 + t * 50), Math.round(180 + t * 60), Math.round(130 + t * 80)] as [number, number, number, number];
            }
            // fallback: static snapshot
            const rec = iliLocal!.gu[name];
            if (!rec) return [60, 80, 120, 100] as [number, number, number, number];
            const t = Math.min(1, rec.ili / iliMax);
            return rec.alert
              ? [220 + Math.round(t * 35), 60, 70, 200] as [number, number, number, number]
              : [Math.round(60 + t * 80), Math.round(90 + t * 50), Math.round(180 + t * 60), Math.round(130 + t * 80)] as [number, number, number, number];
          },
          getLineColor: [220, 230, 255, 60] as [number, number, number, number],
          getLineWidth: 20,
          lineWidthUnits: "meters" as const,
          opacity: 0.75,
          pickable: true,
          updateTriggers: {
            getElevation: [time, forecast360, forecast360GlobalPeak, iliLocal, iliMax],
            getFillColor: [time, forecast360, forecast360GlobalPeak, iliLocal, iliMax],
          },
        })
      : null,
    // Real-time air-quality choropleth (미세먼지 pm10, 환경부 등급 색). Fills each
    // gu by rt_air_quality; tooltip via pickable.
    guBoundary && airEnv
      ? new GeoJsonLayer({
          id: "air-quality",
          data: guBoundary,
          visible: vis.airQuality,
          stroked: false,
          filled: true,
          getFillColor: (f: GeoJSON.Feature) =>
            pmColor(airEnv.air[(f.properties?.name as string) ?? ""]?.pm10),
          opacity: 0.5,
          pickable: true,
          updateTriggers: { getFillColor: [airEnv] },
        })
      : null,
    // 인플루엔자 백신 접종률 choropleth (낮을수록 빨강 = 미접종 위험).
    guBoundary && diseaseVax
      ? new GeoJsonLayer({
          id: "vaccination",
          data: guBoundary,
          visible: vis.vaccination,
          stroked: false,
          filled: true,
          getFillColor: (f: GeoJSON.Feature) =>
            vaxColor(diseaseVax.vax[(f.properties?.name as string) ?? ""]),
          opacity: 0.5,
          pickable: true,
          updateTriggers: { getFillColor: [diseaseVax] },
        })
      : null,
    // 법정감염병 발생 건수 choropleth (많을수록 진한 자주).
    guBoundary && diseaseVax
      ? new GeoJsonLayer({
          id: "disease",
          data: guBoundary,
          visible: vis.disease,
          stroked: false,
          filled: true,
          getFillColor: (f: GeoJSON.Feature) =>
            diseaseColor(diseaseVax.disease[(f.properties?.name as string) ?? ""], diseaseMax),
          opacity: 0.5,
          pickable: true,
          updateTriggers: { getFillColor: [diseaseVax, diseaseMax] },
        })
      : null,
    new HeatmapLayer({
      id: "ili-heatmap",
      data: heatmapData,
      visible: vis.ili,
      getPosition: (d: { position: [number, number] }) => d.position,
      getWeight: (d: { weight: number }) => d.weight,
      radiusPixels: 80,
      colorRange: VIRIDIS_RGBA,
      intensity: 1,
      threshold: 0.05,
    }),
    // Real Seoul bus ROUTE polylines (1,453 lines, masterRouteNode→stop coords).
    // ① 간선: trunk routes (trunk_frac ≥ 0.3, on 중앙차로 corridors) drawn bright
    // amber + thicker; 지선/마을 routes dim grey-blue. ③ full = every route.
    busRoutes && busRoutes.features?.length > 0
      ? new GeoJsonLayer({
          id: "bus-routes",
          data: busRoutes,
          visible: vis.busRoutes,
          stroked: true,
          filled: false,
          getLineColor: (f: GeoJSON.Feature) =>
            (f.properties?.trunk
              ? [255, 170, 60, 200]
              : [90, 120, 170, 90]) as [number, number, number, number],
          getLineWidth: (f: GeoJSON.Feature) =>
            f.properties?.trunk ? 28 : 14,
          lineWidthUnits: "meters",
          lineWidthMinPixels: 0.5,
          lineWidthMaxPixels: 4,
          pickable: true,
        })
      : null,
    // Real Seoul bus stops (11,253) — ScatterplotLayer covers all 3 requested
    // options: trunk corridor (중앙차로, red, larger) = ①, dense point cloud =
    // ② density, every stop colored by type = ③ full network.
    busStops && busStops.length > 0
      ? new ScatterplotLayer({
          id: "bus-stops",
          data: busStops,
          visible: vis.busStops,
          getPosition: (d: { position: [number, number] }) => d.position,
          getFillColor: (d: { color: number[] }) =>
            [...d.color, 200] as [number, number, number, number],
          getRadius: (d: { trunk: boolean }) => (d.trunk ? 110 : 45),
          radiusUnits: "meters",
          radiusMinPixels: 1,
          radiusMaxPixels: 5,
          opacity: 0.7,
          pickable: true,
        })
      : null,
    // Real Seoul Metro route lines (LineString geometry, 8 lines × official
    // colors). 사용자 critique: "지하철 선로가 제대로 안 됨". Renders the actual
    // subway network, not the commuter-flow arcs.
    subwayLines && subwayLines.features?.length > 0
      ? new GeoJsonLayer({
          id: "subway-lines",
          data: subwayLines,
          visible: vis.subwayLines,
          stroked: true,
          filled: false,
          getLineColor: (f: GeoJSON.Feature) =>
            [...hexToRgb((f.properties?.color as string) ?? "#888888"), 230] as [
              number,
              number,
              number,
              number,
            ],
          getLineWidth: 40,
          lineWidthUnits: "meters",
          lineWidthMinPixels: 2,
          lineWidthMaxPixels: 6,
          pickable: true,
        })
      : null,
    // Subway stations: ① radius = √ridership × ③ hourly rush pulse (slider hour),
    // ④ fill = gu age composition (youth teal ↔ elderly magenta). pickable.
    subwayStations && subwayStations.length > 0
      ? new ScatterplotLayer({
          id: "subway-stations",
          data: subwayStations,
          visible: vis.subwayStations,
          getPosition: (d: SubwayStation) => d.position,
          getRadius: (d: SubwayStation) =>
            Math.sqrt(d.ridership) *
            0.4 *
            (0.5 + 0.5 * (d.hourly ? d.hourly[hourOfDay] : 1)),
          radiusUnits: "meters",
          radiusMinPixels: 1.5,
          radiusMaxPixels: 26,
          getFillColor: (d: SubwayStation) =>
            [...ageColor(d.age), 200] as [number, number, number, number],
          opacity: 0.75,
          pickable: true,
          updateTriggers: { getRadius: [hourOfDay] },
        })
      : null,
    // ② Transfer hubs (교합점): 29 환승역 highlighted, size by interchange degree.
    subwayStations && subwayStations.length > 0
      ? new ScatterplotLayer({
          id: "transfer-hubs",
          data: subwayStations.filter((s) => s.transfer),
          visible: vis.transferHubs,
          getPosition: (d: SubwayStation) => d.position,
          getRadius: (d: SubwayStation) => 240 + d.n_lines * 200,
          radiusUnits: "meters",
          radiusMinPixels: 4,
          radiusMaxPixels: 28,
          getFillColor: [255, 224, 90, 70],
          stroked: true,
          getLineColor: [255, 230, 120, 230],
          lineWidthMinPixels: 1.5,
          pickable: true,
        })
      : null,
    // Sprint 2026-05-06 Phase B.2: real commuter_matrix (KOSIS) ArcLayer.
    // 사용자 critique: "전파 시작점 / 끝이 인구밀집도 높은 데서". paper §4.4
    // metapop coupling 의 dashboard evidence. 102 edges, coupling > 0.005.
    commuterEdges && commuterEdges.length > 0
      ? new ArcLayer({
          id: "commuter-arcs-real",
          visible: vis.commuter,
          data: commuterEdges,
          getSourcePosition: (d: CommuterEdge) => [d.source_lng, d.source_lat],
          getTargetPosition: (d: CommuterEdge) => [d.target_lng, d.target_lat],
          getSourceColor: [255, 180, 80, 220],
          getTargetColor: [80, 180, 255, 220],
          // width = log(coupling × 1000) — coupling 0.005 - 0.8 →
          // width 1-7 px (visual 충분)
          getWidth: (d: CommuterEdge) => 1 + Math.log(d.coupling * 1000) * 0.8,
          getHeight: 0.5,
          greatCircle: false,
          pickable: true,
        })
      : new ArcLayer({
          id: "commuter-arcs-demo",
          visible: vis.commuter,
          data: arcData,
          getSourcePosition: (d: FlowEdge) => [
            SEOUL_GU[d.fromIdx].lng,
            SEOUL_GU[d.fromIdx].lat,
          ],
          getTargetPosition: (d: FlowEdge) => [
            SEOUL_GU[d.toIdx].lng,
            SEOUL_GU[d.toIdx].lat,
          ],
          getSourceColor: [255, 180, 80, 200],
          getTargetColor: [80, 180, 255, 200],
          getWidth: (d: FlowEdge) => 1 + d.weight * 4,
          getHeight: 0.6,
          greatCircle: false,
        }),
    // Per-agent commute movement — real route geometry (avg 13.9 waypoints).
    // Period filter: hourOfDay 5-12 → AM trips; 14-23 → PM trips; else all.
    // Time mapping: AM hour 6→ts~51, hour 9→ts~121; PM hour 16→ts~180, hour 20→ts~300.
    new TripsLayer({
      id: "abm-trips",
      visible: vis.agentTrips,
      data: (() => {
        if (!agentTrips || agentTrips.length === 0) return tripData;
        const isAM = hourOfDay >= 5 && hourOfDay < 13;
        const isPM = hourOfDay >= 14 && hourOfDay <= 23;
        if (isAM) return agentTrips.filter((t) => t.period === 'am');
        if (isPM) return agentTrips.filter((t) => t.period === 'pm');
        return agentTrips;
      })(),
      getPath: (d) => d.path,
      getTimestamps: (d) => d.timestamps,
      getColor: (d) =>
        d.color
          ? (d.color as [number, number, number])
          : d.severity === 0
            ? [255, 100, 100]
            : d.severity === 1
              ? [255, 200, 80]
              : [120, 200, 255],
      opacity: 0.9,
      widthMinPixels: 2.5,
      rounded: true,
      trailLength: 60,
      // Map hourOfDay to real timestamp space:
      // AM (6-9h) → ts 51-121; PM (16-20h) → ts 180-300; else full cycle.
      currentTime: (() => {
        if (!agentTrips || agentTrips.length === 0) return time;
        const isAM = hourOfDay >= 5 && hourOfDay < 13;
        const isPM = hourOfDay >= 14 && hourOfDay <= 23;
        if (isAM) return Math.max(0, Math.min(121, 51 + ((hourOfDay - 6) / 3) * 70));
        if (isPM) return Math.max(180, Math.min(300, 180 + ((hourOfDay - 16) / 4) * 120));
        return time % 360;
      })(),
      updateTriggers: {
        data: [hourOfDay],
        currentTime: [hourOfDay, time],
      },
    }),
    // Sprint 2026-05-06 Phase B.3: hospital markers (사용자 critique
    // "병원 위치도 없고"). 404 hospitals × bed_cnt size + ER 보유 색상 강조.
    hospitals && hospitals.length > 0
      ? new ScatterplotLayer({
          id: "hospital-markers",
          visible: vis.hospitals,
          data: hospitals,
          getPosition: (d: Hospital) => [d.lng, d.lat],
          getRadius: (d: Hospital) =>
            // bed_cnt 0-1000+ → radius 80-400m
            80 + Math.min(d.bed_cnt, 1000) * 0.32,
          radiusUnits: "meters",
          radiusMinPixels: 3,
          radiusMaxPixels: 22,
          getFillColor: (d: Hospital) =>
            d.has_er
              ? // 응급실: 실시간 가용 병상 있으면 녹색, 만실/없음 빨강
                ((d.er_beds ?? 0) > 0
                  ? [80, 200, 120, 230]
                  : [240, 70, 70, 230])
              : d.clcd === "상급종합" || d.clcd === "종합병원"
                ? // 종합병원 = 주황
                  [255, 150, 60, 200]
                : // 일반 = 파랑 light
                  [120, 180, 255, 175],
          getLineColor: [255, 255, 255, 220],
          lineWidthMinPixels: 1,
          stroked: true,
          pickable: true,
        })
      : null,
    // Wind (S-DoT): per-gu arrow from centroid. Meteorological wind_dir is the
    // FROM bearing → draw toward dir+180; length ∝ wind_speed. Only sensor gu.
    airEnv
      ? new LineLayer({
          id: "wind",
          visible: vis.wind,
          data: SEOUL_GU.filter((g) => airEnv.env[g.name]?.wind_dir != null).map((g) => {
            const e = airEnv.env[g.name];
            const to = (((e.wind_dir as number) + 180) * Math.PI) / 180;
            const len = 0.004 + (e.wind_speed ?? 0) * 0.006;
            return {
              source: [g.lng, g.lat] as [number, number],
              target: [g.lng + Math.sin(to) * len, g.lat + Math.cos(to) * len] as [
                number,
                number,
              ],
            };
          }),
          getSourcePosition: (d: { source: [number, number] }) => d.source,
          getTargetPosition: (d: { target: [number, number] }) => d.target,
          getColor: [120, 220, 255, 220],
          getWidth: 3,
        })
      : null,
    // 실시간 인구·연령 (79 POIs): radius ∝ √population, fill by 혼잡도. Crowds ×
    // elderly share = flu-relevant; age mix + counts in the tooltip.
    realtime && realtime.pois.length > 0
      ? new ScatterplotLayer({
          id: "population",
          data: realtime.pois,
          visible: vis.population,
          getPosition: (d: RealtimePoi) => d.position,
          getRadius: (d: RealtimePoi) => Math.sqrt(d.ppltn_max ?? 0) * 2.2,
          radiusUnits: "meters",
          radiusMinPixels: 3,
          radiusMaxPixels: 38,
          getFillColor: (d: RealtimePoi) =>
            [...(CONGEST_COLOR[d.congestion] ?? [150, 150, 160]), 150] as [
              number,
              number,
              number,
              number,
            ],
          stroked: true,
          getLineColor: [255, 255, 255, 120],
          lineWidthMinPixels: 0.5,
          pickable: true,
        })
      : null,
    // 도로교통 (원활/서행/정체) ring at the same POIs — stroked so it frames the
    // population dot instead of hiding it.
    realtime && realtime.pois.length > 0
      ? new ScatterplotLayer({
          id: "traffic",
          data: realtime.pois.filter((p) => p.traffic_idx),
          visible: vis.traffic,
          getPosition: (d: RealtimePoi) => d.position,
          getRadius: 650,
          radiusUnits: "meters",
          radiusMinPixels: 6,
          radiusMaxPixels: 20,
          filled: false,
          stroked: true,
          getLineColor: (d: RealtimePoi) =>
            [...(TRAFFIC_COLOR[d.traffic_idx ?? ""] ?? [150, 150, 160]), 230] as [
              number,
              number,
              number,
              number,
            ],
          getLineWidth: 80,
          lineWidthMinPixels: 1.5,
          pickable: true,
        })
      : null,
    // 인구 예보: peak forecast crowd per POI (size ∝ √예보인구, color by 예보혼잡도).
    realtime && realtime.pois.length > 0
      ? new ScatterplotLayer({
          id: "forecast",
          data: realtime.pois.filter((p) => p.fcst_peak_ppltn),
          visible: vis.forecast,
          getPosition: (d: RealtimePoi) => d.position,
          getRadius: (d: RealtimePoi) => Math.sqrt(d.fcst_peak_ppltn ?? 0) * 2.2,
          radiusUnits: "meters",
          radiusMinPixels: 3,
          radiusMaxPixels: 38,
          getFillColor: (d: RealtimePoi) =>
            [...(CONGEST_COLOR[d.fcst_peak_congest ?? ""] ?? [150, 150, 160]), 140] as [
              number,
              number,
              number,
              number,
            ],
          stroked: true,
          getLineColor: [255, 255, 255, 110],
          lineWidthMinPixels: 0.5,
          pickable: true,
        })
      : null,
    // 지하철 혼잡(POI): accumulated boarding at the 68 subway POIs (size ∝ √승차).
    realtime && realtime.pois.length > 0
      ? new ScatterplotLayer({
          id: "subwayCrowd",
          data: realtime.pois.filter((p) => p.subway_on),
          visible: vis.subwayCrowd,
          getPosition: (d: RealtimePoi) => d.position,
          getRadius: (d: RealtimePoi) => Math.sqrt(d.subway_on ?? 0) * 1.6,
          radiusUnits: "meters",
          radiusMinPixels: 3,
          radiusMaxPixels: 42,
          getFillColor: [90, 160, 240, 150],
          stroked: true,
          getLineColor: [180, 220, 255, 200],
          lineWidthMinPixels: 0.5,
          pickable: true,
        })
      : null,
    // 관광지역 분류: POI colored by place category (관광특구/공원/고궁/역세권/상권).
    realtime && realtime.pois.length > 0
      ? new ScatterplotLayer({
          id: "category",
          data: realtime.pois,
          visible: vis.category,
          getPosition: (d: RealtimePoi) => d.position,
          getRadius: 360,
          radiusUnits: "meters",
          radiusMinPixels: 4,
          radiusMaxPixels: 15,
          getFillColor: (d: RealtimePoi) =>
            [...(CATEGORY_COLOR[d.category] ?? [150, 150, 160]), 205] as [
              number,
              number,
              number,
              number,
            ],
          stroked: true,
          getLineColor: [255, 255, 255, 150],
          lineWidthMinPixels: 0.5,
          pickable: true,
        })
      : null,
    // 버스 승하차: 709 stops with daily_bus ridership, sized by 승하차량.
    busStops && busStops.length > 0
      ? new ScatterplotLayer({
          id: "busRidership",
          data: busStops.filter((s) => (s.ridership ?? 0) > 0),
          visible: vis.busRidership,
          getPosition: (d: { position: [number, number] }) => d.position,
          getRadius: (d: { ridership?: number }) => Math.sqrt(d.ridership ?? 0) * 0.7,
          radiusUnits: "meters",
          radiusMinPixels: 2,
          radiusMaxPixels: 30,
          getFillColor: [255, 180, 80, 180],
          stroked: true,
          getLineColor: [255, 230, 180, 200],
          lineWidthMinPixels: 0.5,
          pickable: true,
        })
      : null,
    // 학교 (1,422, gu중심 근사): kind별 색. ABM 학교폐쇄 개입의 공간 맥락.
    schools && schools.length > 0
      ? new ScatterplotLayer({
          id: "schools",
          data: schools,
          visible: vis.schools,
          getPosition: (d: { position: [number, number] }) => d.position,
          getFillColor: (d: { color: number[] }) =>
            [...d.color, 200] as [number, number, number, number],
          getRadius: 130,
          radiusUnits: "meters",
          radiusMinPixels: 1.5,
          radiusMaxPixels: 6,
          stroked: true,
          getLineColor: [255, 255, 255, 90],
          lineWidthMinPixels: 0.3,
          pickable: true,
        })
      : null,
  ].filter(Boolean);

  return (
    <div className="relative h-[80vh] w-full overflow-hidden rounded-lg bg-slate-950">
      <DeckGL
        viewState={viewState}
        onViewStateChange={(e) => setViewState(e.viewState as unknown as ViewState)}
        controller={true}
        layers={layers}
      >
        {/* MapLibre demo style — 무료, 토큰 없음 */}
        <Map
          mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
          reuseMaps
        />
      </DeckGL>

      {/* Time slider + playback */}
      <div className="absolute bottom-4 left-4 right-4 flex items-center gap-3 rounded bg-slate-900/85 px-4 py-2 text-xs text-slate-200 backdrop-blur">
        <button
          type="button"
          onClick={() => setPlaying(!playing)}
          className="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600"
        >
          {playing ? "⏸ pause" : "▶ play"}
        </button>
        <span className="font-mono tabular-nums">
          Day {time.toString().padStart(3, "0")}
          {forecast360 && forecast360[Math.min(time, forecast360.length - 1)]
            ? ` · ${forecast360[Math.min(time, forecast360.length - 1)].date}`
            : ""}
        </span>
        {/* Clock + Rush-hour badge */}
        <span
          className="font-mono tabular-nums"
          style={{
            color: hourOfDay >= 7 && hourOfDay < 9
              ? "#f59e0b"
              : hourOfDay >= 17 && hourOfDay < 20
                ? "#f97316"
                : hourOfDay >= 5 && hourOfDay < 13
                  ? "#34d399"
                  : hourOfDay >= 13 && hourOfDay <= 23
                    ? "#818cf8"
                    : "#475569",
            fontWeight: 600,
          }}
        >
          {String(hourOfDay).padStart(2, "0")}:00
          {" · "}
          {hourOfDay >= 7 && hourOfDay < 9
            ? "아침 러시"
            : hourOfDay >= 17 && hourOfDay < 20
              ? "저녁 러시"
              : hourOfDay >= 5 && hourOfDay < 13
                ? "오전 AM"
                : hourOfDay >= 13 && hourOfDay <= 23
                  ? "오후 PM"
                  : "야간"}
        </span>
        <input
          type="range"
          min={0}
          max={360}
          value={time}
          onChange={(e) => {
            setTime(Number(e.target.value));
            setPlaying(false);
          }}
          className="flex-1"
        />
        <span className="text-slate-400">
          {forecast360
            ? "360일 SEIR 예측 파동(계절강제 ODE)"
            : "DeckGL G1 prototype · maplibre dark"}
        </span>
      </div>

      {/* Complexity control — 2D/3D view + presets (간단/상세/전체) + toggles */}
      <div className="absolute right-4 top-4 w-52 rounded bg-slate-900/85 p-3 text-[11px] text-slate-200 backdrop-blur">
        <button
          type="button"
          onClick={toggleDim}
          className="mb-2 w-full rounded bg-slate-700 px-2 py-1 font-semibold hover:bg-slate-600"
        >
          {is3D ? "🗺 2D 평면 보기" : "⛰ 3D 입체 보기"}
        </button>
        <div className="mb-2 flex gap-1">
          {(Object.keys(PRESET_LABELS) as Preset[]).map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => applyPreset(p)}
              className={`flex-1 rounded px-2 py-1 ${
                preset === p
                  ? "bg-sky-600 text-white"
                  : "bg-slate-700 hover:bg-slate-600"
              }`}
            >
              {PRESET_LABELS[p]}
            </button>
          ))}
        </div>
        <ul className="max-h-64 space-y-0.5 overflow-y-auto">
          {LAYER_GROUPS.map((g) => (
            <li key={g.id}>
              <label className="flex cursor-pointer items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={!!vis[g.id]}
                  onChange={() => toggleLayer(g.id)}
                  className="accent-sky-500"
                />
                <span className={vis[g.id] ? "" : "text-slate-500"}>{g.label}</span>
              </label>
            </li>
          ))}
        </ul>
      </div>

      {/* Weather readout — slider-synced KMA forecast + air-quality timestamp */}
      {weatherNow && (
        <div className="absolute left-4 top-4 rounded bg-slate-900/85 px-3 py-2 text-[11px] text-slate-200 backdrop-blur">
          <div className="font-semibold">
            날씨 · KMA 예보 {weatherNow.valid_at?.slice(8, 10)}시
          </div>
          <div className="mt-1 flex gap-3 font-mono tabular-nums">
            <span>🌡 {weatherNow.TMP ?? "–"}°C</span>
            <span>💧 {weatherNow.REH ?? "–"}%</span>
            <span>🌬 {weatherNow.WSD ?? "–"}m/s</span>
            <span>☔ {weatherNow.POP ?? "–"}%</span>
          </div>
          {airEnv && (
            <div className="mt-1 text-slate-400">
              대기질 PM10 25구 실시간 · {airEnv.collected_at?.slice(0, 16).replace("T", " ")}
            </div>
          )}
          {realtime?.bike && (
            <div className="mt-0.5 text-slate-400">
              따릉이 {realtime.bike.bikes.toLocaleString()}대 · 인구/교통 POI {realtime.collected_at?.slice(0, 10)}
            </div>
          )}
        </div>
      )}

      {/* Legend — size/color meaning for the currently-visible layers */}
      <div className="absolute bottom-16 left-4 max-w-[19rem] rounded bg-slate-900/85 p-2 text-[10px] text-slate-300 backdrop-blur">
        <div className="mb-1 font-semibold text-slate-100">범례 · 켜진 레이어</div>
        <ul className="max-h-44 space-y-0.5 overflow-y-auto">
          {LAYER_GROUPS.filter((g) => vis[g.id]).map((g) => (
            <li key={g.id}>
              <span className="text-slate-100">{g.label}</span>
              <span className="text-slate-400"> — {g.legend}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
