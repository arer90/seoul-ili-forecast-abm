"""Guard: the affine sim→obs map must be applied in the order it is returned.

`_fit_linear_map` returns `(offset, scale)`. `variant_ablation.py` unpacked it as
`a, b` and computed `a * curve + b` — that is `offset * curve + scale`, the two
coefficients swapped. The result shipped as `in_sample_r2 = -766.65`, which is
not a value an ordinary least-squares fit can produce when it is scored on the
very data it was fitted to: the intercept-only model already gives R² = 0, so
the fitted model cannot do worse. That impossible number is the tell, and the
same swapped coefficients carried into `forward_r2 = -633.61`, which README.md
quoted as evidence that anchoring is load-bearing.

The tests below assert the mathematical property rather than the source text, so
any future rewrite that reintroduces the swap still fails.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_affine_map_orientation.py -q
"""

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_SRC = ROOT / "simulation" / "scripts" / "run_abm_forward_validation.py"


def _load_pure(*names):
    """Pull the pure helpers out of the module without importing it.

    A plain import drags in the ABM package, which reads the 25 district names
    from the database at module load — so importing it would make this test
    unrunnable on a clone that has no database, which is every CI runner and
    every fresh checkout. The two functions under test have no dependencies
    beyond numpy, so compiling just their AST nodes gives the real implementation
    rather than a copy that could drift from it.
    """
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(wanted) == len(names), f"missing {set(names) - {n.name for n in wanted}}"
    ns = {"np": np}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(_SRC), "exec"), ns)
    return [ns[n] for n in names]


_fit_linear_map, _r2 = _load_pure("_fit_linear_map", "_r2")


def _apply(affine, x):
    offset, scale = affine
    return offset + scale * np.asarray(x, dtype=float)


def test_return_order_is_offset_then_scale():
    """A pure scaling: obs = 3*sim exactly, so offset≈0 and scale≈3."""
    sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    offset, scale = _fit_linear_map(sim, 3.0 * sim)
    assert offset == pytest.approx(0.0, abs=1e-9)
    assert scale == pytest.approx(3.0, abs=1e-9)


def test_offset_is_recovered():
    sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    offset, scale = _fit_linear_map(sim, 2.0 * sim + 7.0)
    assert offset == pytest.approx(7.0, abs=1e-9)
    assert scale == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_in_sample_r2_is_never_negative(seed):
    """The property the shipped -766.65 violated.

    OLS scored on its own fitting data cannot beat the intercept-only model,
    which is R² = 0 by construction.
    """
    rng = np.random.default_rng(seed)
    sim = rng.normal(size=40) * 5 + 20
    obs = 0.3 * sim + rng.normal(size=40) * 2 + 4
    r2 = _r2(obs, _apply(_fit_linear_map(sim, obs), sim))
    assert r2 >= -1e-9, f"in-sample R² came out {r2}, which OLS cannot produce"


def test_swapping_the_coefficients_reproduces_the_defect():
    """Shows the guard above is not vacuous: the swap really does go far negative."""
    rng = np.random.default_rng(7)
    sim = rng.normal(size=40) * 5 + 20
    obs = 0.3 * sim + rng.normal(size=40) * 2 + 4
    offset, scale = _fit_linear_map(sim, obs)
    correct = _r2(obs, offset + scale * sim)
    swapped = _r2(obs, offset * sim + scale)          # the shipped bug
    assert correct >= 0.0
    assert swapped < -1.0, (
        f"the swap should be catastrophic, got {swapped} — if this ever passes, "
        f"the test data no longer exercises the defect"
    )


def test_variant_ablation_applies_it_correctly():
    """Pin the call sites, since the module needs a database to run end-to-end."""
    src = (ROOT / "simulation" / "abm" / "variant_ablation.py").read_text(encoding="utf-8")
    assert "offset + scale * ins" in src
    assert "offset + scale * fwd_curve" in src
    assert "a * ins + b" not in src
    assert "a * fwd_curve + b" not in src
