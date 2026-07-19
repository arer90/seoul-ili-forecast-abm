/**
 * dev-with-env.mjs — `next dev` wrapper that explicitly loads .env.local
 *
 * Why this exists
 * ---------------
 * Next.js 14 dev server logs "Environments: .env.local" but the values do
 * NOT propagate into Edge runtime `process.env`. Routes that run on Edge
 * (e.g. /api/chat → lib/providers/anthropic.ts → `process.env.ANTHROPIC_API_KEY`)
 * see the var as empty even though .env.local has it. The Anthropic SDK
 * then refuses to initialize and the chat falls back to mock.
 *
 * Workaround: load .env.local into the parent Node process via the
 * built-in `--env-file=` flag (Node 20.6+). Variables become real shell
 * env vars, which Edge runtime reads correctly.
 *
 * Vercel production is unaffected — env vars come from the dashboard
 * Settings → Environment Variables tab, never from .env.local.
 */
import { existsSync, readFileSync } from "node:fs";
import { spawn } from "node:child_process";
import { resolve } from "node:path";

const envFile = resolve(process.cwd(), ".env.local");
// Use the actual JS entry, not the .bin/ shell wrapper (Node can't exec sh).
const nextEntry = resolve(process.cwd(), "node_modules/next/dist/bin/next");

// Parse .env.local manually and merge into process.env so the spawned child
// inherits a fully-resolved environment. Just `--env-file=` was not enough —
// Next.js Edge runtime workers spawned by `next dev` did not see vars loaded
// via that flag (process.env was empty inside Edge isolates), but they DO
// see vars set on the spawn's env option. Same mechanism that `set -a;
// source .env.local; set +a` provides at the shell level.
const env = { ...process.env };
let loaded = 0;
if (existsSync(envFile)) {
  const raw = readFileSync(envFile, "utf8");
  for (const line of raw.split(/\r?\n/)) {
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1);
    // Strip matched surrounding quotes (rare but valid in dotenv files)
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    env[key] = val;
    loaded += 1;
  }
  console.log(`[dev-with-env] loaded ${loaded} vars from ${envFile}`);
} else {
  console.warn(`[dev-with-env] no .env.local — running with shell env only`);
}

const proc = spawn(process.execPath, [nextEntry, "dev"], {
  stdio: "inherit",
  env,
});
proc.on("exit", (code) => process.exit(code ?? 0));
proc.on("error", (e) => {
  console.error("[dev-with-env] spawn error:", e);
  process.exit(1);
});
