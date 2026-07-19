/**
 * /api/overlays/hospitals — Sprint 2026-05-06 Phase B.3.
 *
 * 병원 marker layer (hospitals 404 + emergency_room_availability 250 merge).
 * 사용자 critique: "병원 위치도 없고" — 직접 fix.
 *
 * 응답 schema (Map3D ScatterplotLayer 호환):
 *   [{name, addr, gu, clcd, bed_cnt, dr_cnt, lat, lng, has_er, er_beds}]
 *
 * Real data source: epi_real_seoul.db `hospitals` (lat/lng / bed_cnt / clcd_nm) +
 * `emergency_room_availability` (실시간 ER bed 가용성) — outer join by name + gu.
 */
import { NextResponse } from "next/server";
import path from "path";

export const runtime = "nodejs";
export const revalidate = 1800; // 30 min cache (ER 가용성 30분 단위 변동)

export async function GET() {
  let Database: typeof import("better-sqlite3");
  try {
    Database = (await import("better-sqlite3")).default;
  } catch {
    return NextResponse.json({
      hospitals: [],
      source: "no_db",
      message: "better-sqlite3 not installed",
    });
  }
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
    // Hospitals 본 테이블 (lat/lng 보유)
    const hospitals = db
      .prepare(
        `SELECT inst_nm AS name, addr, gu_nm AS gu, clcd_nm AS clcd,
                bed_cnt, dr_cnt, lat, lng
         FROM hospitals
         WHERE lat IS NOT NULL AND lng IS NOT NULL
         ORDER BY bed_cnt DESC`,
      )
      .all() as Array<{
      name: string;
      addr: string;
      gu: string;
      clcd: string;
      bed_cnt: number;
      dr_cnt: number;
      lat: number;
      lng: number;
    }>;

    // ER 실시간 가용성 (latest by hp_nm)
    const erRows = db
      .prepare(
        `SELECT hp_nm AS name, gu_nm AS gu, hvec, hvoc, hvicc, hv2, hv3, latitude, longitude
         FROM emergency_room_availability
         WHERE collected_at = (SELECT MAX(collected_at) FROM emergency_room_availability)`,
      )
      .all() as Array<{
      name: string;
      gu: string;
      hvec: number; // 응급실 가용 bed
      hvoc: number; // 수술실 가용
      hvicc: number; // ICU 가용
      hv2: number; // 신생아 ICU
      hv3: number; // 일반 입원
      latitude: number;
      longitude: number;
    }>;
    db.close();

    // Merge: hospital list + ER availability (matched by name)
    const erByName = new Map(
      erRows.map((r) => [r.name, r]),
    );

    const merged = hospitals.map((h) => {
      const er = erByName.get(h.name);
      return {
        ...h,
        has_er: !!er,
        er_beds: er?.hvec ?? null,
        er_icu: er?.hvicc ?? null,
        er_surgery: er?.hvoc ?? null,
      };
    });

    return NextResponse.json({
      hospitals: merged,
      n_hospitals: merged.length,
      n_with_er: merged.filter((h) => h.has_er).length,
      source: "hospitals (lat/lng) ⨯ emergency_room_availability (latest)",
    });
  } catch (e) {
    return NextResponse.json(
      {
        hospitals: [],
        error: e instanceof Error ? e.message : String(e),
      },
      { status: 200 },
    );
  }
}
