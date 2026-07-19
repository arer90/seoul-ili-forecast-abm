"""Acceptance tests for the Rust SEIR-V-D agent kernel dispatch."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

seir_core = pytest.importorskip("seir_core")
if not hasattr(seir_core, "run_agent_world_rs"):
    pytest.skip("Rust backend not built", allow_module_level=True)

from simulation.abm import agent_kernel


ROOT = Path(__file__).resolve().parents[2]


def _run_numpy_oracle(**kwargs):
    available = agent_kernel.RUST_BACKEND_AVAILABLE
    try:
        agent_kernel.RUST_BACKEND_AVAILABLE = False
        return agent_kernel.run_agent_world(**kwargs)
    finally:
        agent_kernel.RUST_BACKEND_AVAILABLE = available


def _iter_output_arrays(result):
    for name in "SEIRVD":
        yield name, np.asarray(result[name])
    for name, value in result["agents"].items():
        yield f"agents.{name}", np.asarray(value)


def _max_delta(a, b) -> float:
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def test_rust_dispatch_matches_numpy_oracle_three_seeds():
    mixing = np.eye(25) * 0.8 + np.ones((25, 25), dtype=np.float64) * (0.2 / 25.0)
    age_dist = np.tile(np.arange(1.0, 8.0), (25, 1))
    kwargs = dict(
        N=777,
        T_days=20,
        beta=0.50,
        sigma=0.30,
        gamma=0.16,
        delta=0.002,
        nu=np.linspace(0.0, 0.002, 25),
        mixing_matrix=mixing,
        age_dist=age_dist,
    )

    for seed in [0, 1, 42]:
        oracle = _run_numpy_oracle(global_seed=seed, **kwargs)
        rust = agent_kernel.run_agent_world(global_seed=seed, **kwargs)
        for name, oracle_arr in _iter_output_arrays(oracle):
            rust_arr = dict(_iter_output_arrays(rust))[name]
            delta = _max_delta(oracle_arr, rust_arr)
            if np.issubdtype(oracle_arr.dtype, np.floating):
                # Rust libm and NumPy can differ by a few ulps for exp/logistic
                # while preserving every stochastic transition and count exactly.
                #
                # The tolerance has to be expressed in ulps of the array's own
                # dtype, not as a fixed 1e-9. These arrays are float32
                # (agent_kernel param_dtype), where one ulp at alpha_mean=0.3 is
                # 2.98e-08 — so a flat 1e-9 demanded agreement 30x finer than the
                # smallest representable step, and the suite failed on a delta of
                # 1.19e-08, i.e. well under a single ulp.
                # Precision follows the KERNEL's working dtype, not the array's
                # storage dtype: agent_kernel computes in float32 (param_dtype)
                # and some outputs (agents.fatigue) are widened to float64 on the
                # way out. Keying the tolerance off the storage dtype gave those
                # a float64 budget of 1.8e-15 for a value carrying float32 error.
                eps = max(float(np.finfo(np.float32).eps),
                          float(np.finfo(oracle_arr.dtype).eps))
                scale = max(1.0, float(np.max(np.abs(oracle_arr))))
                tol = 8.0 * eps * scale
                assert delta <= tol, (seed, name, delta, tol)
            else:
                assert delta == 0.0, (seed, name, delta)


def test_rust_dispatch_is_rayon_thread_count_invariant():
    script = r"""
import hashlib
import json
import numpy as np
from simulation.abm.agent_kernel import run_agent_world

mixing = np.eye(25) * 0.8 + np.ones((25, 25), dtype=np.float64) * (0.2 / 25.0)
age_dist = np.tile(np.arange(1.0, 8.0), (25, 1))
r = run_agent_world(
    777,
    20,
    beta=0.50,
    sigma=0.30,
    gamma=0.16,
    delta=0.002,
    nu=np.linspace(0.0, 0.002, 25),
    mixing_matrix=mixing,
    age_dist=age_dist,
    global_seed=42,
)
h = hashlib.sha256()
for key in list("SEIRVD"):
    arr = np.asarray(r[key])
    h.update(key.encode())
    h.update(str(arr.dtype).encode())
    h.update(json.dumps(arr.shape).encode())
    h.update(arr.tobytes())
for key, value in sorted(r["agents"].items()):
    arr = np.asarray(value)
    h.update(("agents." + key).encode())
    h.update(str(arr.dtype).encode())
    h.update(json.dumps(arr.shape).encode())
    h.update(arr.tobytes())
print(h.hexdigest())
"""

    digests = []
    for threads in ["1", "4"]:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = threads
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        digests.append(proc.stdout.strip())

    assert digests[0] == digests[1]
