/**
 * POST /api/report — parse a ``/report`` command and dispatch to the
 * Python backend which owns the docx / xlsx / pptx / pdf skills.
 *
 * The Python side has the Anthropic skills mounted (docx / pdf / pptx /
 * xlsx) and knows how to turn a pinned turn + chart payload into a
 * deliverable. This route is a thin dispatcher that:
 *
 *   1. parses `/report <format> [--pinned turn_id]`
 *   2. serialises the selected conversation + MCP artifact paths
 *   3. POSTs to the Python bridge at `${MCP_BRIDGE_URL}/report`
 *   4. returns the {url, format, bytes} hand-off for the browser
 *
 * We deliberately do not render the docx inside the edge runtime —
 * python-docx / python-pptx are not edge-compatible, and the skills
 * already know how to do it properly.
 */
import type { NextRequest } from "next/server";

import { requireAuth } from "@/lib/auth";

export const runtime = "edge";

type Format = "docx" | "pdf" | "pptx" | "xlsx";
const ALL_FORMATS: Format[] = ["docx", "pdf", "pptx", "xlsx"];

interface ReportRequest {
  command: string;   // raw "/report ..." string
  pinned: {          // the conversation turn the user pinned
    turnId: string;
    markdown: string;
    attachments?: Array<{ kind: string; url?: string; payload?: unknown }>;
  };
  /** Extra context (scenario run, leaderboard) attached by the UI. */
  context?: Record<string, unknown>;
}

export async function POST(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;

  let body: ReportRequest;
  try {
    body = (await req.json()) as ReportRequest;
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }

  const parsed = parseCommand(body.command);
  if (!parsed.ok) return Response.json({ error: parsed.reason }, { status: 400 });

  const bridge = process.env.MCP_BRIDGE_URL ?? "http://localhost:8787";
  const res = await fetch(`${bridge}/report`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      format: parsed.format,
      pinned: body.pinned,
      context: body.context ?? {},
    }),
    signal: req.signal,
  });
  if (!res.ok) {
    return Response.json(
      { error: `bridge ${res.status}: ${await res.text()}` },
      { status: 502 },
    );
  }
  // Bridge returns { url, format, bytes } — just pass through.
  const payload = await res.json();
  return Response.json(payload);
}

function parseCommand(
  command: string,
):
  | { ok: true; format: Format }
  | { ok: false; reason: string } {
  const trimmed = command.trim();
  if (!trimmed.startsWith("/report")) {
    return { ok: false, reason: "command must start with /report" };
  }
  const parts = trimmed.split(/\s+/).slice(1);
  const fmt = (parts[0] ?? "docx").toLowerCase() as Format;
  if (!ALL_FORMATS.includes(fmt)) {
    return {
      ok: false,
      reason: `unknown format: ${fmt}; use ${ALL_FORMATS.join(" / ")}`,
    };
  }
  return { ok: true, format: fmt };
}
