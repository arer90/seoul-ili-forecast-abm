/**
 * POST /api/abm/run — Python ABM 정확 시뮬레이션 엔드포인트.
 *
 * Body: { nAgents: number, r0: number, npi: number, vaccCov: number,
 *         days: number, seedGu: string }
 * Response: { attackRate: number, peakDay: number, peakIli: number,
 *             cv: number, nAgents: number, source: 'python-abm' }
 *
 * Architecture
 * ------------
 * Node.js runtime (not Edge) — needs spawn for Python subprocess.
 * Spawns .venv/bin/python -c <inline script> which runs the validated
 * agent_kernel.run_agent_world with the supplied params.
 *
 * Python ABM: simulation/abm/agent_kernel.run_agent_world — daily binomial
 * tau-leap SEIR-V-D, Structure-of-Arrays agent layout, calibrated from
 * simulation/abm/epi_proof.py defaults (beta=0.18, sigma=0.45, gamma=0.18,
 * delta=0.002, nu=0.0002).
 *
 * n_agents cap: 100 000 (API enforced). Timeout: 60 s hard kill.
 * vaccCov [0,1] → initial_vaccinated mask. npi [0,1] → beta multiplier.
 * 3 replicate seeds → mean attackRate/peakDay/peakIli + seed CV.
 *
 * DB: not accessed (no read_only_connect needed — pure simulation).
 *
 * Error handling:
 *   - invalid body → 400
 *   - timeout → 504 with {source:'python-abm', error:'timeout'}
 *   - Python exit non-zero → 500 with stderr snippet
 *   - success → 200 JSON matching the interface contract
 *
 * Performance: ~2-8s for n=10k, ~20-60s for n=100k. UI should show spinner.
 * Side effects: none (no DB write, no file write).
 * Caller responsibility: nAgents in [1, 100000], r0 in [0.5, 5.0].
 */
import { NextRequest, NextResponse } from "next/server";
import { spawn } from "node:child_process";
import path from "node:path";

export const runtime = "nodejs";
export const maxDuration = 65;

const N_AGENTS_MAX = 100_000;
const TIMEOUT_MS = 60_000;

// Default calibrated disease params from epi_proof.DEFAULT_DISEASE
const DEFAULT_BETA = 0.18;
const DEFAULT_SIGMA = 0.45;
const DEFAULT_GAMMA = 0.18;
const DEFAULT_DELTA = 0.002;
const DEFAULT_NU = 0.0002;

/** Inline Python script: runs run_agent_world 3 times (seeds 1,2,3),
 *  derives attackRate/peakDay/peakIli/cv, prints single JSON line. */
function buildPyScript(
  nAgents: number,
  beta: number,
  sigma: number,
  gamma: number,
  delta: number,
  nu: number,
  days: number,
  vaccCovFrac: number,
): string {
  return `
import json, sys, numpy as np
from simulation.abm.agent_kernel import run_agent_world

N = ${nAgents}
T = ${days}
beta = ${beta}
sigma = ${sigma}
gamma = ${gamma}
delta = ${delta}
nu = ${nu}
vacc_cov = ${vaccCovFrac}

SEEDS = [1, 2, 3]
attack_rates = []
peak_days = []
peak_ilis = []

for seed in SEEDS:
    # vaccCov -> initial_vaccinated boolean mask
    rng = np.random.default_rng(seed)
    if vacc_cov > 0:
        init_vacc = rng.random(N) < vacc_cov
    else:
        init_vacc = None

    result = run_agent_world(
        N=N,
        T_days=T,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        delta=delta,
        nu=nu,
        global_seed=seed,
        initial_vaccinated=init_vacc,
    )

    S = np.asarray(result['S'], dtype=np.float64)
    I = np.asarray(result['I'], dtype=np.float64)
    D = np.asarray(result['D'], dtype=np.float64)

    # Attack rate: fraction of population that left S
    initial_s = float(S[0])
    final_s = float(S[-1])
    dead = float(D[-1])
    attacked = max(0.0, initial_s - final_s + dead)
    ar = attacked / N if N > 0 else 0.0
    attack_rates.append(ar)

    # Peak day: day of max I
    I_arr = np.asarray(I)
    pk_day = int(np.argmax(I_arr))
    peak_days.append(pk_day)

    # Peak ILI rate per 1000 (ILI ~ I / N * 1000)
    peak_ili = float(I_arr[pk_day]) / N * 1000.0
    peak_ilis.append(peak_ili)

ar_mean = float(np.mean(attack_rates))
ar_std = float(np.std(attack_rates, ddof=0))
cv = float(ar_std / ar_mean) if ar_mean > 0 else 0.0

print(json.dumps({
    "attackRate": round(ar_mean, 4),
    "peakDay": int(round(float(np.mean(peak_days)))),
    "peakIli": round(float(np.mean(peak_ilis)), 3),
    "cv": round(cv, 4),
    "nAgents": N,
    "source": "python-abm",
    "_replicates": {
        "attackRates": [round(a, 4) for a in attack_rates],
        "peakDays": peak_days,
        "peakIlis": [round(p, 3) for p in peak_ilis],
    },
}))
`.trim();
}

export async function POST(req: NextRequest) {
  // 1. Parse + validate body
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { ok: false, error: "invalid JSON body" },
      { status: 400 },
    );
  }

  const nAgentsRaw = Number(body.nAgents ?? 1000);
  const r0Raw = Number(body.r0 ?? 1.4);
  const npiRaw = Number(body.npi ?? 0.0);
  const vaccCovRaw = Number(body.vaccCov ?? 0.0);
  const daysRaw = Number(body.days ?? 180);

  if (!Number.isFinite(nAgentsRaw) || nAgentsRaw < 1 || nAgentsRaw > N_AGENTS_MAX) {
    return NextResponse.json(
      { ok: false, error: `nAgents must be in [1, ${N_AGENTS_MAX}]` },
      { status: 400 },
    );
  }
  if (!Number.isFinite(r0Raw) || r0Raw < 0.5 || r0Raw > 5.0) {
    return NextResponse.json(
      { ok: false, error: "r0 must be in [0.5, 5.0]" },
      { status: 400 },
    );
  }
  if (!Number.isFinite(npiRaw) || npiRaw < 0 || npiRaw > 1) {
    return NextResponse.json(
      { ok: false, error: "npi must be in [0, 1]" },
      { status: 400 },
    );
  }
  if (!Number.isFinite(vaccCovRaw) || vaccCovRaw < 0 || vaccCovRaw > 1) {
    return NextResponse.json(
      { ok: false, error: "vaccCov must be in [0, 1]" },
      { status: 400 },
    );
  }
  if (!Number.isFinite(daysRaw) || daysRaw < 1 || daysRaw > 730) {
    return NextResponse.json(
      { ok: false, error: "days must be in [1, 730]" },
      { status: 400 },
    );
  }

  const nAgents = Math.round(nAgentsRaw);
  const days = Math.round(daysRaw);

  // 2. Derive disease params from r0 + npi
  //    beta = (r0 * gamma) * (1 - npi)
  //    Keep sigma, gamma, delta, nu at calibrated defaults.
  const gammaSEIR = DEFAULT_GAMMA; // 1/infectious_days
  const betaRaw = r0Raw * gammaSEIR; // R0 = beta/gamma
  const beta = betaRaw * (1.0 - Math.max(0, Math.min(1, npiRaw)));

  // 3. Build Python inline script
  const pyCode = buildPyScript(
    nAgents,
    beta,
    DEFAULT_SIGMA,
    DEFAULT_GAMMA,
    DEFAULT_DELTA,
    DEFAULT_NU,
    days,
    vaccCovRaw,
  );

  // 4. Spawn Python subprocess (mirrors rag-bridge.ts pattern)
  const projectRoot =
    process.env.MCP_CWD ?? path.join(process.cwd(), "..");
  const pythonBin =
    process.env.MCP_PYTHON ??
    (process.platform === "win32"
      ? path.join(projectRoot, ".venv", "Scripts", "python.exe")
      : path.join(projectRoot, ".venv", "bin", "python"));

  return new Promise<Response>((resolve) => {
    const t0 = Date.now();
    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const child = spawn(pythonBin, ["-c", pyCode], {
      cwd: projectRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, TIMEOUT_MS);

    child.stdout.on("data", (d: Buffer) => {
      stdout += d.toString();
    });
    child.stderr.on("data", (d: Buffer) => {
      stderr += d.toString();
    });

    child.on("exit", (code) => {
      clearTimeout(timer);
      const elapsed_ms = Date.now() - t0;

      if (timedOut) {
        resolve(
          NextResponse.json(
            {
              ok: false,
              error: "timeout",
              source: "python-abm",
              nAgents,
              elapsed_ms,
              note: `n_agents=${nAgents} exceeded ${TIMEOUT_MS / 1000}s limit`,
            },
            { status: 504 },
          ),
        );
        return;
      }

      if (code !== 0) {
        resolve(
          NextResponse.json(
            {
              ok: false,
              error: `python exit ${code}`,
              source: "python-abm",
              nAgents,
              elapsed_ms,
              stderr: stderr.slice(-1000),
            },
            { status: 500 },
          ),
        );
        return;
      }

      // Parse last JSON line from stdout
      const lines = stdout.trim().split("\n");
      let lastLine: string | undefined;
      for (let i = lines.length - 1; i >= 0; i--) {
        if (lines[i].trim().startsWith("{")) { lastLine = lines[i]; break; }
      }
      if (!lastLine) {
        resolve(
          NextResponse.json(
            {
              ok: false,
              error: "no JSON output from python",
              source: "python-abm",
              nAgents,
              elapsed_ms,
              stdout: stdout.slice(-500),
            },
            { status: 500 },
          ),
        );
        return;
      }

      let result: Record<string, unknown>;
      try {
        result = JSON.parse(lastLine);
      } catch (e) {
        resolve(
          NextResponse.json(
            {
              ok: false,
              error: `JSON parse failed: ${e}`,
              source: "python-abm",
              nAgents,
              elapsed_ms,
            },
            { status: 500 },
          ),
        );
        return;
      }

      resolve(
        NextResponse.json({
          ok: true,
          elapsed_ms,
          ...result,
        }),
      );
    });

    child.on("error", (e: Error) => {
      clearTimeout(timer);
      resolve(
        NextResponse.json(
          {
            ok: false,
            error: e.message,
            source: "python-abm",
            nAgents,
            elapsed_ms: Date.now() - t0,
          },
          { status: 500 },
        ),
      );
    });
  });
}

export async function GET() {
  return NextResponse.json({
    method: "POST",
    body: {
      nAgents: `integer [1, ${N_AGENTS_MAX}] — number of agents`,
      r0: "float [0.5, 5.0] — basic reproduction number",
      npi: "float [0, 1] — NPI effect on beta (0=off, 1=full suppression)",
      vaccCov: "float [0, 1] — initial vaccination coverage fraction",
      days: "integer [1, 730] — simulation horizon in days",
      seedGu: "string — seed district name (informational only)",
    },
    response: {
      attackRate: "float [0,1] — mean fraction of population infected",
      peakDay: "integer — mean day of peak infectious",
      peakIli: "float — mean peak ILI rate per 1000",
      cv: "float — coefficient of variation across 3 replicate seeds",
      nAgents: "integer — actual agents used",
      source: "'python-abm'",
    },
    note: `timeout ${TIMEOUT_MS / 1000}s. large n_agents (>50k) may be slow.`,
  });
}
