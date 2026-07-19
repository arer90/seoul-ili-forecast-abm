#!/usr/bin/env node
// D5+ render smoke: catch RUNTIME white-screen / crash that esbuild (syntax-only) misses.
//
// The Babel-in-browser dashboard has no error boundary, so a single render-time throw
// (e.g. `model.metrics` when model is null) blanks the whole app — and esbuild's syntax
// gate happily passes it.  This mounts the real App in jsdom (browser globals that the app
// degrades-gracefully on — Leaflet/WASM/KaTeX — are left undefined; the app's own
// `if(!window.L) return` guards bail), then CLICKS the heavy panels (예측검증/ABM/다질병)
// to render the components that only mount behind a drawer — which is where the
// model-null class of crash lives.
//
// Exit 1 = a real mount/render crash.  Exit 0 = rendered + panels opened clean.
// Tooling-missing (jsdom/esbuild) → warn + exit 0 (never block a commit on smoke setup).
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __dir = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dir, "..", "..");
const require = createRequire(import.meta.url);

function req(p) { return require(resolve(ROOT, "web/node_modules", p)); }

let esbuild, JSDOM, React, ReactDOMClient;
try {
  esbuild = req("esbuild");
  ({ JSDOM } = req("jsdom"));
  React = req("react");
  ReactDOMClient = req("react-dom/client");
} catch (e) {
  console.warn("⚠ smoke_render: tooling missing (jsdom/esbuild/react) → skip:", String(e.message || e).split("\n")[0]);
  process.exit(0);
}

// Default target = the tracked source; pass a path arg to smoke a specific file (CI / self-test).
const SRC = process.argv[2] ? resolve(process.cwd(), process.argv[2]) : resolve(ROOT, "web_prototype/app.jsx");

async function main() {
  // 1. JSX → JS (global React.createElement, matching the UMD/global the app uses).
  let code;
  try {
    code = esbuild.transformSync(readFileSync(SRC, "utf8"), {
      loader: "jsx", jsx: "transform",
      jsxFactory: "React.createElement", jsxFragment: "React.Fragment",
    }).code;
  } catch (e) {
    console.error("✗ smoke_render: esbuild transform failed:", String(e).split("\n")[0]);
    process.exit(1);
  }

  // 2. jsdom DOM + #root.
  const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
    url: "http://localhost/abs/index.html", pretendToBeVisual: true,
  });
  const { window } = dom;
  const doc = window.document;

  // My own output bypasses console (overridden below) so it always shows clean.
  const out = (m) => process.stdout.write(m + "\n");
  const fail = (m) => process.stderr.write(m + "\n");

  // 3. Crash capture.  react-dom/app log render errors via the GLOBAL console.error;
  //    override it (+ window's) to record crash-pattern messages and swallow jsdom-effect
  //    noise (headless useEffect failures that aren't real render crashes).
  const crashes = [];
  const CRASH_RE = /Cannot read|is not a function|is not defined|undefined is not|null is not|reading '/;
  window.addEventListener("error", (e) => crashes.push("uncaught: " + (e.error?.message || e.message)));
  const captureErr = (...a) => {
    const s = a.map((x) => (x && x.stack) ? x.stack : String(x)).join(" ");
    if (CRASH_RE.test(s)) crashes.push("render: " + s.slice(0, 180));
  };
  console.error = captureErr;
  window.console.error = captureErr;
  // Swallow the app's own warn/info/log (offline-fetch warnings + status logs are expected
  // in the headless run, not failures).  My output uses out()/fail() → unaffected.
  const noop = () => {};
  for (const m of ["warn", "info", "log", "debug"]) {
    console[m] = noop;
    try { window.console[m] = noop; } catch { /* ignore */ }
  }

  // 4. Graceful browser stubs.  window.L / __seirWasm* / renderMathInElement are LEFT
  //    undefined on purpose — the app guards on them and degrades to no-op.
  const ReactDOM = { ...require(resolve(ROOT, "web/node_modules/react-dom")), createRoot: ReactDOMClient.createRoot };
  window.fetch = () => Promise.reject(new Error("smoke: offline"));
  window.matchMedia = window.matchMedia || (() => ({ matches: false, media: "", onchange: null, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent() { return false; } }));
  window.requestAnimationFrame = window.requestAnimationFrame || ((cb) => setTimeout(() => cb(Date.now()), 0));
  window.cancelAnimationFrame = window.cancelAnimationFrame || ((id) => clearTimeout(id));
  if (!window.ResizeObserver) window.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  if (!window.IntersectionObserver) window.IntersectionObserver = class { observe() {} unobserve() {} disconnect() {} takeRecords() { return []; } };

  // 5. Expose jsdom + libs as Node globals so BOTH the app code AND react-dom
  //    (which reads the global `window`) resolve them.
  const setG = (k, v) => {
    if (v === undefined) return;
    try { global[k] = v; }
    catch { try { Object.defineProperty(global, k, { value: v, configurable: true, writable: true }); } catch { /* read-only built-in — leave Node's */ } }
  };
  setG("window", window);
  setG("document", doc);
  setG("navigator", window.navigator);
  setG("React", React);
  setG("ReactDOM", ReactDOM);
  setG("fetch", window.fetch);
  setG("localStorage", window.localStorage);
  setG("sessionStorage", window.sessionStorage);
  setG("location", window.location);
  setG("requestAnimationFrame", window.requestAnimationFrame);
  setG("cancelAnimationFrame", window.cancelAnimationFrame);
  // react-dom 19 profiler calls performance.measure.bind(...) — jsdom's performance lacks
  // measure/mark.  Patch window.performance; keep Node's global.performance (has them).
  try {
    const wp = window.performance || {};
    if (typeof wp.now !== "function") wp.now = () => Date.now();
    if (typeof wp.measure !== "function") wp.measure = () => undefined;
    if (typeof wp.mark !== "function") wp.mark = () => undefined;
    if (typeof wp.clearMeasures !== "function") wp.clearMeasures = () => {};
    if (typeof wp.clearMarks !== "function") wp.clearMarks = () => {};
  } catch { /* ignore */ }
  for (const k of ["HTMLElement", "Node", "Element", "Event", "CustomEvent", "MouseEvent", "getComputedStyle", "SVGElement", "DOMParser", "XMLHttpRequest"]) {
    setG(k, window[k]);
  }

  // 6. Execute the app (calls ReactDOM.createRoot(#root).render(<App/>) at the end).
  try {
    // eslint-disable-next-line no-new-func
    new Function(code)();
  } catch (e) {
    fail("✗ smoke_render: app threw on mount: " + String(e.stack || e.message || e).slice(0, 240));
    process.exit(1);
  }

  // 7. Let effects + initial render settle.
  await new Promise((r) => setTimeout(r, 250));

  if ((doc.getElementById("root")?.children.length ?? 0) === 0) {
    fail("✗ smoke_render: #root is empty after mount (white-screen on load).");
    if (crashes.length) fail("    " + crashes[0]);
    process.exit(1);
  }

  // 7. Click the drawer panels — these only mount on demand (where the model-null crash lived).
  const wantText = ["예측 검증", "ABM 시뮬", "다질병", "다질병 감시"];
  const opened = [];
  for (const want of wantText) {
    const btn = [...doc.querySelectorAll("button")].find((b) => (b.textContent || "").includes(want));
    if (!btn) continue;
    try {
      btn.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
      await new Promise((r) => setTimeout(r, 120));
      opened.push(want);
    } catch (e) {
      crashes.push(`click '${want}': ` + String(e.message || e).slice(0, 160));
    }
  }

  if (crashes.length) {
    fail("✗ smoke_render: render crash detected (" + crashes.length + "):");
    crashes.slice(0, 4).forEach((c) => fail("   • " + c));
    process.exit(1);
  }

  out(`✓ smoke_render: App mounted (#root ok) + panels opened clean [${opened.join(", ") || "no panel buttons found"}]`);
  process.exit(0);
}

main().catch((e) => {
  process.stderr.write("✗ smoke_render: unexpected: " + String(e.stack || e.message || e).slice(0, 240) + "\n");
  process.exit(1);
});
