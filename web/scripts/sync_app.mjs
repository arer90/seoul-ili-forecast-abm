#!/usr/bin/env node
// D5 (Codex+Gemini): single source of truth for the Babel-in-browser dashboard.
//
//   web_prototype/app.jsx   = tracked SOURCE (committed)
//   web/public/abs/app.jsx  = gitignored SERVED copy (what Next.js serves at /abs)
//
// Two failure modes this guards:
//   1. SYNTAX — Babel compiles app.jsx in the browser at runtime, so a single parse
//      error white-screens the whole dashboard with no build-time signal. We esbuild-
//      parse the source first to catch it before it ships.
//   2. DRIFT — editing the source without copying to the served path (or vice-versa)
//      silently serves stale code. We regenerate the copy from source.
//
// Usage:
//   node web/scripts/sync_app.mjs           # syntax-check + regenerate served copies
//   node web/scripts/sync_app.mjs --check    # check only; exit 1 on syntax error OR drift
//                                            #   (use in pre-commit / CI)
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createRequire } from "node:module";

const __dir = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dir, "..", "..");
const require = createRequire(import.meta.url);
const esbuild = require(resolve(ROOT, "web/node_modules/esbuild"));

// [sourceRelPath, servedRelPath, esbuildLoader|null]
const PAIRS = [
  ["web_prototype/app.jsx", "web/public/abs/app.jsx", "jsx"],
  ["web_prototype/abm-stochastic.js", "web/public/abs/abm-stochastic.js", "js"],
  ["web_prototype/styles.css", "web/public/abs/styles.css", null],
];

const checkOnly = process.argv.includes("--check");
let failed = false;

for (const [srcRel, dstRel, loader] of PAIRS) {
  const src = resolve(ROOT, srcRel);
  const dst = resolve(ROOT, dstRel);
  const code = readFileSync(src, "utf8");

  // 1. Syntax gate (source only).
  if (loader) {
    try {
      esbuild.transformSync(code, { loader });
    } catch (e) {
      console.error(`✗ SYNTAX  ${srcRel}: ${String(e.message || e).split("\n")[0]}`);
      failed = true;
      continue; // don't sync a broken file
    }
  }

  // 2. Drift gate / regenerate.
  let served = null;
  try {
    served = readFileSync(dst, "utf8");
  } catch {
    served = null; // served copy missing
  }
  const drift = served !== code;

  if (checkOnly) {
    if (drift) {
      console.error(`✗ DRIFT   ${dstRel} != ${srcRel}  (run: node web/scripts/sync_app.mjs)`);
      failed = true;
    } else {
      console.log(`✓ ${srcRel} — synced + valid`);
    }
  } else {
    if (drift) {
      writeFileSync(dst, code);
      console.log(`↻ synced  ${dstRel}  (from ${srcRel})`);
    } else {
      console.log(`✓ ${srcRel} — already in sync`);
    }
  }
}

process.exit(failed ? 1 : 0);
