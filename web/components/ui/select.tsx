/**
 * Native <select> styled to match our dark theme. We don't need the
 * full Radix listbox here — native select is accessible and works fine
 * on mobile, which is important for the tablet/phone demo drill.
 */
import { forwardRef } from "react";
import type { SelectHTMLAttributes } from "react";

export interface SelectProps
  extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  function Select({ className = "", label, id, children, ...rest }, ref) {
    return (
      <label className="flex flex-col gap-1 text-xs text-slate-300">
        {label ? <span className="font-medium">{label}</span> : null}
        <select
          ref={ref}
          id={id}
          className={[
            "h-8 rounded-md border border-slate-700 bg-slate-900 px-2",
            "text-sm text-slate-100 outline-none",
            "focus:border-sky-400 focus:ring-1 focus:ring-sky-400/70",
            "disabled:opacity-50",
            className,
          ].join(" ")}
          {...rest}
        >
          {children}
        </select>
      </label>
    );
  },
);
