/**
 * /api/collect/run — Sprint 2026-05-06 Phase C.1.
 *
 * Dashboard refresh 버튼 trigger. POST → spawn Python `simulation collect`
 * → KDCA / WHO FluNet 등 fresh data fetch → DB update.
 *
 * 사용자 critique: "14일 업데이트가 늦다는 데이터가 나오는데 바로 업데이트가
 * 되어야 하는거 아니야?" — KDCA reporting 의 본질적 lag (~14d) 외, 우리
 * collector 는 manual trigger 필요했음. 이제 dashboard 클릭 → 즉시 fetch.
 *
 * Security: 본 route 는 dev / local 환경 만 (production 시 auth + rate limit
 * 필요, 다음 sprint).
 */
import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

export const runtime = "nodejs";
export const maxDuration = 60;

const ALLOWED_GROUPS = new Set([
  "weekly_disease",
  "who_flunet",
  "school_closure_seoul",
  "school_info_seoul",
  "vaccination_coverage",
  "rt_population",
  "rt_subway_crowd",
  "rt_air_quality",
  "rt_sdot_env",
  "weather_forecast",
  "all",
]);

export async function POST(req: NextRequest) {
  let groups = "weekly_disease,who_flunet";
  try {
    const body = await req.json();
    if (typeof body.groups === "string") groups = body.groups;
  } catch {
    // empty body OK, default
  }

  // Allowlist filter — block arbitrary collector group names
  const requested = groups.split(",").map((s) => s.trim()).filter(Boolean);
  const validGroups = requested.filter((g) => ALLOWED_GROUPS.has(g));
  if (validGroups.length === 0) {
    return NextResponse.json(
      {
        ok: false,
        error: `no valid groups. allowed: ${Array.from(ALLOWED_GROUPS).join(", ")}`,
      },
      { status: 400 },
    );
  }

  const safeGroups = validGroups.join(",");
  const cwd = path.join(process.cwd(), "..");
  const pythonBin = path.join(cwd, ".venv", "bin", "python");

  return new Promise<Response>((resolve) => {
    const t0 = Date.now();
    const proc = spawn(
      pythonBin,
      ["-m", "simulation", "collect", "--groups", safeGroups],
      { cwd, env: { ...process.env, PYTHONUNBUFFERED: "1" } },
    );
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      proc.kill("SIGTERM");
    }, 55_000);

    proc.stdout.on("data", (d) => {
      stdout += d.toString();
    });
    proc.stderr.on("data", (d) => {
      stderr += d.toString();
    });
    proc.on("exit", (code) => {
      clearTimeout(timeout);
      const elapsed_ms = Date.now() - t0;
      resolve(
        NextResponse.json({
          ok: code === 0,
          code,
          groups: safeGroups,
          elapsed_ms,
          stdout: stdout.slice(-2000),
          stderr: stderr.slice(-2000),
        }),
      );
    });
    proc.on("error", (e) => {
      clearTimeout(timeout);
      resolve(
        NextResponse.json(
          { ok: false, error: e.message, groups: safeGroups },
          { status: 500 },
        ),
      );
    });
  });
}

export async function GET() {
  return NextResponse.json({
    method: "POST",
    body: { groups: "comma-separated, default 'weekly_disease,who_flunet'" },
    allowed_groups: Array.from(ALLOWED_GROUPS),
    note: "POST only. timeout 55s.",
  });
}
