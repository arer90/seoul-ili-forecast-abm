/**
 * /map3d — 3D 지도 mini-prototype 페이지 (G1 path).
 *
 * Sprint 2026-05-06 (#16): DeckGL HeatmapLayer + ArcLayer + TripsLayer + dark
 * MapLibre. 사용자 영감 이미지 (Phoenix LST / drone flock / voronoi /
 * Detroit categorical) 의 G1 cover 검증.
 *
 * 후속 (별 sprint): real sim_runs/*.npz 통합, voronoi tessellation, 3D column
 * extrusion, paper §5.7 ARIA Stage 6 evidence 강화.
 */
import dynamic from "next/dynamic";

const Map3D = dynamic(
  () => import("@/components/Map3D").then((m) => m.Map3D),
  { ssr: false },
);

export const metadata = {
  title: "ABS · 3D 지도 prototype (DeckGL)",
};

export default function Map3DPage() {
  return (
    <main className="mx-auto max-w-7xl space-y-4 p-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">3D 지도 — DeckGL prototype</h1>
        <p className="text-sm text-slate-600">
          G1 path 검증 (DeckGL + MapLibre dark). HeatmapLayer + ArcLayer + TripsLayer
          + time slider. 사용자 영감 이미지 (Phoenix LST / drone flock trail / voronoi)
          를 단일 stack 으로 cover 가능한지 evidence.
        </p>
        <p className="text-xs text-slate-500">
          docs:{" "}
          <a
            href="https://github.com/arer90/MPH_infection_simulation/blob/main/docs/3D_MAP_OPTIONS.md"
            className="underline"
          >
            3D_MAP_OPTIONS.md
          </a>{" "}
          · G2/G3/G4 평가 + 권장 path + Claude artifact fallback.
        </p>
        <p className="pt-1 text-sm">
          <a
            href="/map3d/abm"
            className="inline-block rounded bg-sky-600 px-3 py-1 font-semibold text-white hover:bg-sky-500"
          >
            → ABM 대시보드 (Metapop SEIR-V-D × ARIA)
          </a>
        </p>
      </header>
      <Map3D />
      <section className="rounded border border-slate-200 bg-slate-50 p-4 text-xs text-slate-700">
        <h2 className="mb-2 text-sm font-semibold">현재 prototype 상태 (mini)</h2>
        <ul className="ml-4 list-disc space-y-1">
          <li>
            HeatmapLayer · 25-gu demo (실제 데이터: <code>simulation/cache/feature_cache.parquet</code>{" "}
            의 <code>ili_rate</code> + <code>gu_centroid</code> next sprint 통합)
          </li>
          <li>
            ArcLayer · 8 hub → 25 gu 곡선 (실제 데이터:{" "}
            <code>epi_real_seoul.db.commuter_flow</code> next sprint)
          </li>
          <li>
            TripsLayer · 12 gu 의 random circular path animated trail (실제: ABM agent
            trail next sprint)
          </li>
          <li>아직 미구현: voronoi tessellation, 3D column extrusion, sim playback 통합</li>
        </ul>
      </section>
    </main>
  );
}
