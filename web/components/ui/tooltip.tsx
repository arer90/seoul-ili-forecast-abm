/**
 * CSS-only tooltip. Radix's tooltip pulls a sizeable runtime dependency
 * that we don't need for a handful of badges — a lightweight wrapper is
 * fine. The tooltip only appears on :hover and :focus-within.
 */
import type { ReactNode } from "react";

export interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom";
  className?: string;
}

export function Tooltip({
  content,
  children,
  side = "top",
  className = "",
}: TooltipProps) {
  const position =
    side === "top"
      ? "bottom-full mb-1.5"
      : "top-full mt-1.5";
  return (
    <span className={["relative inline-flex group", className].join(" ")}>
      {children}
      <span
        role="tooltip"
        className={[
          "pointer-events-none absolute left-1/2 z-30 -translate-x-1/2",
          "whitespace-pre rounded-md border border-slate-700 bg-slate-900",
          "px-2 py-1 text-xs text-slate-100 shadow-lg",
          "opacity-0 transition-opacity duration-100",
          "group-hover:opacity-100 group-focus-within:opacity-100",
          position,
        ].join(" ")}
      >
        {content}
      </span>
    </span>
  );
}
