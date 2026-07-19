"""A6 (M7): Caruana forward-stepwise ensemble wired as a real-slab candidate.

Caruana already existed (ensembles/caruana.py) but was never a candidate in the
operational eval — only median/stacking/NNLS were. This guards both the wiring
and the apply pattern (OOF weights → real-slab combination).
"""
from pathlib import Path

import numpy as np

from simulation.ensembles import caruana_forward_stepwise
from simulation.ensembles.stacking_crps import predict_with_stacking


def test_caruana_wired_into_real_eval():
    src = Path("simulation/pipeline/real_eval.py").read_text(encoding="utf-8")
    assert "caruana_forward_stepwise" in src, "Caruana not wired into real_eval"
    assert 'candidates["caruana_ensemble"]' in src


def test_caruana_to_ensemble_forecast_pattern():
    rng = np.random.default_rng(0)
    y = np.arange(40, dtype=float) + rng.normal(0, 1, 40)
    oof = {
        "good": y + rng.normal(0, 0.5, 40),
        "bad": rng.normal(20.0, 10.0, 40),
        "ok": y + rng.normal(0, 2.0, 40),
    }
    car = caruana_forward_stepwise(oof, y, n_steps=25, random_state=42)
    assert car.model_weights
    assert abs(sum(car.model_weights.values()) - 1.0) < 1e-6   # normalized
    assert car.model_weights.get("good", 0.0) >= car.model_weights.get("bad", 0.0)

    # apply the learned weights to (real-slab-shaped) candidate forecasts
    cand = {k: v[:5] for k, v in oof.items() if k in car.model_weights}
    ens = predict_with_stacking(car.model_weights, cand)
    assert ens.shape == (5,) and np.all(np.isfinite(ens))
