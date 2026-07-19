/**
 * POST /api/mcp/:tool — thin proxy to the Python MCP bridge.
 *
 * Used by the UI to call tools directly (e.g. ``epi.scenario_run`` from
 * the ScenarioPicker) without going through an LLM turn. Bypassing the
 * chat path is faster and avoids provider token costs for pure data
 * pulls.
 */
import type { NextRequest } from "next/server";

import { requireAuth } from "@/lib/auth";
import { callTool, listTools } from "@/lib/mcp-client";

export const runtime = "edge";

export async function GET(
  req: NextRequest,
  ctx: { params: { tool: string } },
): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  if (ctx.params.tool === "_list") {
    const tools = await listTools({ signal: req.signal });
    return Response.json({ tools });
  }
  return new Response("method not allowed", { status: 405 });
}

export async function POST(
  req: NextRequest,
  ctx: { params: { tool: string } },
): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const { tool } = ctx.params;
  let args: Record<string, unknown> = {};
  try {
    args = (await req.json()) as Record<string, unknown>;
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }
  try {
    const result = await callTool(tool, args, { signal: req.signal });
    return Response.json({
      toolCallId: result.toolCallId,
      output: result.output,
      isError: result.isError,
    });
  } catch (e) {
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : String(e) }),
      {
        status: 502,
        headers: { "content-type": "application/json" },
      },
    );
  }
}
