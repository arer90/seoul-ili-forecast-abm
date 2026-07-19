# `seir_core` — Rust-accelerated SEIR stepper (optional)

Native extension for the tau-leap SEIR stepper used by
`simulation/sim/stepper.py`. Provides 30–100× speedup over the pure
Python/Numba path on the 25-gu Seoul metapopulation.

**This build is OPTIONAL.** If `seir_core` is not installed, the Python
code falls back to a Numba `@njit` implementation transparently.

## Build

```bash
# From repo root, with a Mac/Linux/Windows uv venv active:
uv pip install -e .[rust]   # installs maturin into the venv
cd simulation/rust
maturin develop --release   # builds seir_core and installs into venv
```

Verify:
```python
>>> import seir_core
>>> seir_core.tau_leap_step_batch
<built-in function tau_leap_step_batch>
```

## Requirements

- Rust toolchain (stable ≥1.75). Install via [rustup](https://rustup.rs/).
- `maturin` (installed via `uv pip install -e .[rust]`).

## API

- `tau_leap_step_batch(s, e, i, r, v, d, foi, sigma, gamma, nu, mu, dt, seed)`
  — one tau-leap step for all 25 gu in parallel (rayon).
- `commuter_foi(commuter_matrix, i_infectious, population, beta)`
  — commuter-coupled force of infection, vectorized.

Both functions accept/return NumPy arrays (`float64`, shape `(25,)` or `(25, 25)`).

## Fallback path

If you skip the Rust build, `simulation/sim/stepper.py` uses `@numba.njit`
decorators automatically. Benchmark both with
`simulation/scripts/bench_seir_python.py` and
`simulation/scripts/bench_seir_python_ext.py`.
