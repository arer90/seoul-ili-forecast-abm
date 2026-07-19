import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // Korean subset font (Pretendard) loaded via CSS; fall back to
        // system ui.
        sans: [
          "Pretendard",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Apple SD Gothic Neo",
          "Malgun Gothic",
          "sans-serif",
        ],
      },
      colors: {
        // Validity badges
        ok: "#16a34a",
        warn: "#d97706",
        fail: "#dc2626",
      },
    },
  },
  plugins: [],
};

export default config;
