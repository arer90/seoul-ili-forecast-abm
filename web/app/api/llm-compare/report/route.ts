/**
 * GET /api/llm-compare/report
 *
 * Loads the LLM comparison report produced by
 * `python -m simulation.llm_compare.runner` from the project's
 * `simulation/results/llm_compare/report.json`. Read-only filesystem
 * access — for production deploy, swap the source to Turso / S3 /
 * Vercel Blob.
 *
 * sprint 2026-05-06 — paper PART B (TRIPOD-LLM Stage 7) dashboard.
 */
import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // Project root = parent of `web/` (Next.js cwd is web/ at runtime)
    const reportPath = path.resolve(
      process.cwd(),
      "..",
      "simulation",
      "results",
      "llm_compare",
      "report.json",
    );
    const raw = await readFile(reportPath, "utf-8");
    const data = JSON.parse(raw);
    return NextResponse.json(data, {
      headers: {
        // 60s edge cache; report regenerates on each runner invocation
        "Cache-Control": "public, s-maxage=60, stale-while-revalidate=300",
      },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json(
      {
        error: "report_not_found",
        message: msg,
        hint:
          "Run `python -m simulation.llm_compare.runner --out-dir simulation/results/llm_compare` first.",
      },
      { status: 404 },
    );
  }
}
