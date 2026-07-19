/**
 * Post-hoc validity check — runs ``epi.validity_check`` on a set of
 * numeric claims plucked out of the assistant reply.
 *
 * The strategy is deliberately conservative: we only extract claims we
 * can cheaply verify (Rt, peak I, final D, VE). Prose statements get a
 * grey "unverified" badge rather than a false-positive pass.
 */

import { callTool } from "./mcp-client";

export type BadgeStatus = "ok" | "warn" | "fail" | "unverified";

export interface ValidityBadge {
  status: BadgeStatus;
  details: Record<string, unknown>;
  extractedClaims: ExtractedClaim[];
}

export interface ExtractedClaim {
  kind: "rt" | "peak_i" | "final_d" | "ve";
  value: number;
  span: [number, number]; // character offsets in the reply
}

const RT_RX = /\bR(?:t|_t|\s*\(t\))\s*[=:]\s*([-0-9.]+)/gi;
const PEAK_RX = /peak\s*(?:I|infected|감염)[^0-9]*([0-9,\.]+)/gi;
const FINAL_RX = /(?:final|최종)\s*D[^0-9]*([0-9,\.]+)/gi;
const VE_RX = /\bVE\s*[=:]\s*([0-9.]+)\s*%?/gi;

export function extractClaims(reply: string): ExtractedClaim[] {
  const out: ExtractedClaim[] = [];
  for (const [rx, kind] of [
    [RT_RX, "rt"],
    [PEAK_RX, "peak_i"],
    [FINAL_RX, "final_d"],
    [VE_RX, "ve"],
  ] as const) {
    rx.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = rx.exec(reply))) {
      const v = parseFloat(m[1]?.replace(/,/g, "") ?? "");
      if (!Number.isFinite(v)) continue;
      out.push({
        kind,
        value: v,
        span: [m.index, m.index + m[0].length],
      });
    }
  }
  return out;
}

export async function checkReply(
  reply: string,
  signal?: AbortSignal,
): Promise<ValidityBadge> {
  const claims = extractClaims(reply);
  if (claims.length === 0) {
    return { status: "unverified", details: {}, extractedClaims: claims };
  }
  const params: Record<string, number> = {};
  for (const c of claims) {
    if (c.kind === "rt" && params.R0 === undefined) params.R0 = c.value;
    if (c.kind === "ve" && params.VE === undefined) {
      params.VE = c.value > 1 ? c.value / 100 : c.value;
    }
  }
  try {
    const res = await callTool(
      "epi.validity_check",
      { params: Object.keys(params).length ? params : undefined },
      { signal },
    );
    const payload = res.output as {
      status?: BadgeStatus;
      details?: Record<string, unknown>;
    };
    return {
      status: (payload.status as BadgeStatus) ?? "unverified",
      details: payload.details ?? {},
      extractedClaims: claims,
    };
  } catch {
    return { status: "unverified", details: {}, extractedClaims: claims };
  }
}
