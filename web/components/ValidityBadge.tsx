/**
 * Validity badge — ok / warn / fail / unverified. Renders at the bottom
 * of each assistant turn. The hover tooltip shows which claims were
 * extracted + which check failed.
 */
import { Tooltip } from "./ui/tooltip";
import type { BadgeStatus, ValidityBadge as ValidityResult } from "@/lib/validity";

const STATUS_CX: Record<BadgeStatus, string> = {
  ok: "bg-green-950/60 border-green-700 text-green-300",
  warn: "bg-amber-950/60 border-amber-700 text-amber-300",
  fail: "bg-red-950/60 border-red-700 text-red-300",
  unverified:
    "bg-slate-900/60 border-slate-700 text-slate-400",
};

const STATUS_LABEL: Record<BadgeStatus, string> = {
  ok: "✓ Validity OK",
  warn: "△ Validity warn",
  fail: "✕ Validity fail",
  unverified: "? Unverified",
};

export interface ValidityBadgeProps {
  result?: ValidityResult | null;
}

export function ValidityBadge({ result }: ValidityBadgeProps) {
  const status: BadgeStatus = result?.status ?? "unverified";
  const claims = result?.extractedClaims ?? [];
  const tooltipLines: string[] = [];
  if (claims.length) {
    for (const c of claims) {
      tooltipLines.push(`${c.kind}: ${c.value}`);
    }
  }
  if (result?.details && Object.keys(result.details).length) {
    for (const [k, v] of Object.entries(result.details)) {
      tooltipLines.push(`${k}=${typeof v === "number" ? v : JSON.stringify(v)}`);
    }
  }
  const tooltip =
    tooltipLines.join("\n") ||
    (status === "unverified"
      ? "No quantitative claim was extracted from this reply."
      : "No detail payload.");

  return (
    <Tooltip content={tooltip} side="top">
      <span
        className={[
          "inline-flex items-center gap-1 rounded-full border px-2 py-0.5",
          "text-[11px] font-medium tabular-nums",
          STATUS_CX[status],
        ].join(" ")}
      >
        {STATUS_LABEL[status]}
      </span>
    </Tooltip>
  );
}
