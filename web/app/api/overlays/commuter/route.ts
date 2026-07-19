/**
 * /api/overlays/commuter — Sprint 2026-05-06 Phase B.2.
 *
 * commuter_matrix 의 origin × dest × coupling × night_population 응답.
 * Map3D 의 DeckGL ArcLayer 가 source/target/width 매핑 — 사용자 critique
 * "전파 시작점 / 끝이 인구밀집도 / 지하철 hub 에서" 의 직접 evidence.
 *
 * paper §4.4 metapop coupling 의 dashboard 매핑 — 현재 화면의 주황색
 * synthetic flow 가 사실 이 데이터 (KOSIS 통근 행렬, 625 rows = 25 × 25 - 일부).
 */
import { NextResponse } from "next/server";
import path from "path";

export const runtime = "nodejs";
export const revalidate = 3600; // 1h cache

// Seoul 25-gu centroid (approximate). Real: load from
// simulation/data/external/seoul_gu_centroid.csv (별 sprint).
const GU_CENTROID: Record<string, [number, number]> = {
  강남구: [127.0473, 37.5172], 강동구: [127.1238, 37.5301], 강북구: [127.0257, 37.6396],
  강서구: [126.8495, 37.5509], 관악구: [126.9514, 37.4783], 광진구: [127.0823, 37.5384],
  구로구: [126.8874, 37.4954], 금천구: [126.8954, 37.4519], 노원구: [127.0566, 37.6543],
  도봉구: [127.0471, 37.6688], 동대문구: [127.0398, 37.5744], 동작구: [126.9393, 37.5124],
  마포구: [126.9015, 37.5664], 서대문구: [126.9367, 37.5793], 서초구: [127.0327, 37.4837],
  성동구: [127.0367, 37.5634], 성북구: [127.0167, 37.5894], 송파구: [127.1058, 37.5145],
  양천구: [126.8666, 37.5169], 영등포구: [126.8956, 37.5263], 용산구: [126.9788, 37.5326],
  은평구: [126.9292, 37.6027], 종로구: [126.9794, 37.5735], 중구: [126.9974, 37.5636],
  중랑구: [127.0925, 37.6063],
};

export async function GET() {
  // Dynamic import — better-sqlite3 not available in Edge runtime
  let Database: typeof import("better-sqlite3");
  try {
    Database = (await import("better-sqlite3")).default;
  } catch {
    // Fallback: 정적 mock for prod when better-sqlite3 not installed.
    return NextResponse.json({
      edges: [],
      gu_centroid: GU_CENTROID,
      source: "mock_no_db",
      message: "better-sqlite3 not installed",
    });
  }

  // DB path: web/ 에서 한 단계 위로
  const dbPath = path.join(
    process.cwd(),
    "..",
    "simulation",
    "data",
    "db",
    "epi_real_seoul.db",
  );

  try {
    const db = new Database(dbPath, { readonly: true, fileMustExist: true });
    const rows = db
      .prepare(
        `SELECT origin_gu, dest_gu, coupling, night_population
         FROM commuter_matrix
         WHERE origin_gu != dest_gu
           AND coupling > 0.005
         ORDER BY coupling DESC
         LIMIT 200`,
      )
      .all() as Array<{
      origin_gu: string;
      dest_gu: string;
      coupling: number;
      night_population: number;
    }>;
    db.close();

    // Filter: only known gu in centroid map (정적 25)
    const edges = rows
      .filter((r) => r.origin_gu in GU_CENTROID && r.dest_gu in GU_CENTROID)
      .map((r) => ({
        origin: r.origin_gu,
        dest: r.dest_gu,
        coupling: r.coupling,
        night_population: r.night_population,
        source_lng: GU_CENTROID[r.origin_gu][0],
        source_lat: GU_CENTROID[r.origin_gu][1],
        target_lng: GU_CENTROID[r.dest_gu][0],
        target_lat: GU_CENTROID[r.dest_gu][1],
      }));

    return NextResponse.json({
      edges,
      gu_centroid: GU_CENTROID,
      n_edges: edges.length,
      source: "commuter_matrix (KOSIS, off-diagonal, coupling > 0.005)",
    });
  } catch (e) {
    return NextResponse.json(
      {
        edges: [],
        gu_centroid: GU_CENTROID,
        error: e instanceof Error ? e.message : String(e),
        source: "error",
      },
      { status: 200 }, // 200 with empty edges so client renders fallback
    );
  }
}
