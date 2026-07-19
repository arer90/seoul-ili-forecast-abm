"""Guard: every Optuna-sampled norm choice must accept 2-D tabular input (G-237).

Incident (2026-05-30): the feature-optuna DNN/TCN folds failed with
``ValueError: expected input's size at dim=0 to match num_features (16), but got: 32``.
Root cause: ``_get_norm("instance", dim)`` built ``nn.InstanceNorm1d(dim, affine=True)``,
which on a 2-D ``(batch, features)`` tensor treats dim-0 as the channel axis and
compares ``batch_size`` (32) against ``num_features`` (hidden dim 16) → raises. The
"16 vs 32" were hidden-dim vs batch-size, NOT feature counts. Fix: map "instance"
to ``Identity`` (tabular has no spatial axis).

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
import pytest


# The 7 norm choices sampled by simulation/models/_optuna_samplers.py (td_norm).
_NORM_CHOICES = ("none", "batch", "layer", "group", "instance", "weight", "spectral")


def test_get_norm_safe_on_2d_tabular():
    """Each norm choice applied to (batch, features) input must not raise."""
    torch = pytest.importorskip("torch")
    from simulation.tools.run_optuna_feature_selection import _get_norm

    B, H = 32, 16  # batch_size, hidden_dim — the exact (32 vs 16) that broke
    x = torch.randn(B, H)
    for name in _NORM_CHOICES:
        layer = _get_norm(name, H)
        out = layer(x)  # pre-fix: name="instance" raised ValueError here
        assert out.shape == x.shape, f"norm '{name}' changed shape {x.shape}→{out.shape}"
