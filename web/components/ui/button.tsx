/**
 * Minimal shadcn-style button. We don't pull the real dependency
 * because we only need three variants and want to keep the bundle
 * small for the Edge runtime.
 */
import { forwardRef } from "react";
import type { ButtonHTMLAttributes } from "react";

type Variant = "default" | "ghost" | "outline" | "danger";
type Size = "sm" | "md" | "icon";

const variantCx: Record<Variant, string> = {
  default:
    "bg-sky-500 text-slate-950 hover:bg-sky-400 disabled:opacity-50",
  ghost:
    "bg-transparent text-slate-200 hover:bg-slate-800 disabled:opacity-50",
  outline:
    "border border-slate-700 bg-transparent text-slate-100 " +
    "hover:bg-slate-800 disabled:opacity-50",
  danger:
    "bg-red-600 text-white hover:bg-red-500 disabled:opacity-50",
};

const sizeCx: Record<Size, string> = {
  sm: "h-7 px-2 text-xs",
  md: "h-9 px-3 text-sm",
  icon: "h-8 w-8 p-0 text-sm",
};

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    {
      className = "",
      variant = "default",
      size = "md",
      type = "button",
      ...rest
    },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        className={[
          "inline-flex select-none items-center justify-center gap-1.5 rounded-md",
          "font-medium transition-colors outline-none",
          "focus-visible:ring-2 focus-visible:ring-sky-400/70",
          variantCx[variant],
          sizeCx[size],
          className,
        ].join(" ")}
        {...rest}
      />
    );
  },
);
