#!/usr/bin/env node
// Minimal faithful Gemini bridge (recreated; original was missing).
// Owns: arg parsing, file/dir ingestion, structured prompt assembly, gemini CLI invocation.
const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const argv = process.argv.slice(2);
const dd = argv.indexOf("--");
const task = dd >= 0 ? argv.slice(dd + 1).join(" ") : "";
const opts = dd >= 0 ? argv.slice(0, dd) : argv;
function val(flag) { const i = opts.indexOf(flag); return i >= 0 ? opts[i + 1] : null; }
const dirs = (val("--dirs") || "").split(",").map(s => s.trim()).filter(Boolean);
const filesArg = (val("--files") || "").split(",").map(s => s.trim()).filter(Boolean);
const model = val("--model");
const fmt = val("--format");

if (!task) { console.error("usage: gemini-bridge.js [--dirs a,b] [--files glob] [--model M] [--format json] -- \"TASK\""); process.exit(2); }

function walk(d, acc) {
  let ents = [];
  try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
  for (const e of ents) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) { if (!/node_modules|__pycache__|\.git/.test(p)) walk(p, acc); }
    else if (/\.(py|md|json|csv|txt|js|ts|sh)$/.test(e.name)) acc.push(p);
  }
}
const files = [];
for (const d of dirs) walk(d, files);
for (const f of filesArg) { try { if (fs.statSync(f).isFile()) files.push(f); } catch {} }

let ctx = "";
let budget = 600000; // char budget for inlined context
for (const f of files) {
  try {
    const c = fs.readFileSync(f, "utf-8");
    const chunk = `\n\n===== FILE: ${f} =====\n${c}`;
    if (chunk.length > budget) continue;
    ctx += chunk; budget -= chunk.length;
  } catch {}
}

const prompt = (ctx ? `You are given repository context.\n${ctx}\n\n----- TASK -----\n` : "") + task +
  (fmt === "json" ? "\n\nReturn machine-readable JSON." : "");

const gArgs = [];
if (model) gArgs.push("-m", model);
try {
  const out = execFileSync("gemini", gArgs, { input: prompt, encoding: "utf-8", maxBuffer: 64 * 1024 * 1024 });
  process.stdout.write(out);
} catch (e) {
  console.error("GEMINI_BRIDGE_ERROR:", e.message);
  if (e.stdout) process.stdout.write(e.stdout);
  if (e.stderr) process.stderr.write(e.stderr);
  process.exit(1);
}
