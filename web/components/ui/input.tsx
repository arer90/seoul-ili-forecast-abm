/**
 * Minimal text input + textarea. Used by ChatPanel; kept stateless.
 */
import { forwardRef } from "react";
import type { InputHTMLAttributes, TextareaHTMLAttributes } from "react";

const base =
  "rounded-md border border-slate-700 bg-slate-900 px-3 py-2 " +
  "text-sm text-slate-100 placeholder:text-slate-500 outline-none " +
  "focus:border-sky-400 focus:ring-1 focus:ring-sky-400/70 " +
  "disabled:opacity-50";

export const Input = forwardRef<
  HTMLInputElement,
  InputHTMLAttributes<HTMLInputElement>
>(function Input({ className = "", type = "text", ...rest }, ref) {
  return (
    <input
      ref={ref}
      type={type}
      className={[base, className].join(" ")}
      {...rest}
    />
  );
});

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  TextareaHTMLAttributes<HTMLTextAreaElement>
>(function Textarea({ className = "", rows = 3, ...rest }, ref) {
  return (
    <textarea
      ref={ref}
      rows={rows}
      className={[base, "resize-none", className].join(" ")}
      {...rest}
    />
  );
});
