"""G-305: baseline applies NO pipeline-level y-transform (user principle: x/y transforms ONLY in
phase-13 preproc/HP Optuna — nothing else). Baseline = pure raw y + BASIC features. Model-internal
preproc (scaler/log-link) is kept (that's model definition, not a 'transform'). macOS: run PER-FILE.
"""
import inspect
import numpy as np


def test_g305_baseline_forces_none_transform_and_empty_per_model():
    """run_baseline overrides the preset transform to none and empties the per-model map."""
    from simulation.pipeline import baseline as B
    src = inspect.getsource(B.run_baseline)
    assert 'TargetTransformer(method="none")' in src, "baseline must force tt=none (no pipeline transform)"
    assert "per_model_map = {}" in src, "baseline must drop per-model fixed transforms"
    # the old log1p path (get_preset's tt used directly + get_per_model_strategy) must be gone
    assert "per_model_map = get_per_model_strategy" not in src, "per-model 'optimal' transform map must be removed"


def test_g305_none_transform_is_raw_roundtrip():
    """The 'none' transform the baseline now uses is a true identity (raw in, raw out)."""
    from simulation.models.target_transform import TargetTransformer
    tt = TargetTransformer(method="none")
    y = np.array([0.0, 1.0, 5.0, 12.0, 30.0, 100.7])
    yt = tt.transform(y)
    assert np.allclose(yt, y), "none transform must keep y raw (no log1p)"
    assert np.allclose(tt.inverse_transform(yt), y), "none inverse must return raw y"


def test_g305_external_also_forces_none_transform():
    """external (phase 5) — same fixed-preset transform violation → also forced to none/raw."""
    import inspect
    from simulation.pipeline import external as E
    # the module's run function applies the transform; check the source contains the override
    src = inspect.getsource(E)
    assert 'TargetTransformer(method="none")' in src, "external must force tt=none (no pipeline transform)"
    assert "per_model_map = {}" in src, "external must drop per-model fixed transforms"
