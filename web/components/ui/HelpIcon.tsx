/**
 * HelpIcon — a small circled `?` button that reveals a short
 * explanation on hover (desktop) or click/tap (touch).
 *
 * Why a bespoke component instead of reusing <Tooltip>
 *   · Tooltip only shows on :hover / :focus-within, which doesn't work
 *     on touch devices — finger-taps never sustain a hover.
 *   · HelpIcon layers two affordances:
 *       - group-hover:opacity for the hover reveal (desktop)
 *       - a React `pinned` state for click/tap-to-pin (mobile + a11y)
 *   · Clicking toggles the popover and locks it open; clicking outside
 *     or pressing Esc closes it. Keyboard focus alone still shows the
 *     popover (group-focus-within).
 *
 * Usage:
 *   <HelpIcon content="Rt > 1 means outbreak is growing" />
 *   <HelpIcon label="What is WIS?" content={<>…JSX…</>} side="bottom" />
 *
 * Size: 14 px circle, deliberately smaller than surrounding labels so
 * the `?` reads as a secondary affordance. Users who want details
 * can still focus it with Tab.
 */
"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

export interface HelpIconProps {
  /** Body of the popover. Plain string or JSX are both fine. */
  content: ReactNode;
  /** aria-label — defaults to "Help" / localised caller-supplied value. */
  label?: string;
  /** Popover placement relative to the icon. Default "top". */
  side?: "top" | "bottom" | "right" | "left";
  /** Extra classes for the container (rarely needed). */
  className?: string;
}

export function HelpIcon({
  content,
  label = "Help",
  side = "top",
  className = "",
}: HelpIconProps) {
  const [pinned, setPinned] = useState(false);
  const rootRef = useRef<HTMLSpanElement | null>(null);

  // Click-outside closes the pinned popover. Touching the button again
  // toggles, so we explicitly skip clicks that originate inside rootRef.
  useEffect(() => {
    if (!pinned) return;
    const onDoc = (e: MouseEvent | TouchEvent) => {
      if (!rootRef.current) return;
      const target = e.target as Node | null;
      if (target && rootRef.current.contains(target)) return;
      setPinned(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPinned(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("touchstart", onDoc, { passive: true });
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("touchstart", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [pinned]);

  const toggle = useCallback(() => setPinned((p) => !p), []);

  const pos =
    side === "top"
      ? "bottom-full left-1/2 -translate-x-1/2 mb-1.5"
      : side === "bottom"
        ? "top-full left-1/2 -translate-x-1/2 mt-1.5"
        : side === "right"
          ? "left-full top-1/2 -translate-y-1/2 ml-1.5"
          : "right-full top-1/2 -translate-y-1/2 mr-1.5";

  return (
    <span
      ref={rootRef}
      className={["relative inline-flex group align-middle", className].join(" ")}
    >
      <button
        type="button"
        onClick={toggle}
        aria-label={label}
        aria-expanded={pinned}
        title={label}
        className={[
          "flex h-[14px] w-[14px] items-center justify-center rounded-full border",
          "border-slate-500/70 text-[9px] font-semibold leading-none",
          "text-slate-400 hover:border-sky-400 hover:text-sky-300",
          "focus:outline-none focus:ring-1 focus:ring-sky-500",
          pinned ? "border-sky-400 text-sky-200" : "",
        ].join(" ")}
      >
        ?
      </button>
      <span
        role="tooltip"
        className={[
          "absolute z-40 w-56 whitespace-normal rounded-md border border-slate-700",
          "bg-slate-900 px-2.5 py-1.5 text-[11px] leading-snug text-slate-100 shadow-xl",
          "transition-opacity duration-100",
          // Desktop hover reveal, persistent when focused within.
          "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
          // Pinned by click/tap — force visible above other interactions.
          pinned ? "opacity-100" : "pointer-events-none",
          pos,
        ].join(" ")}
      >
        {content}
      </span>
    </span>
  );
}
