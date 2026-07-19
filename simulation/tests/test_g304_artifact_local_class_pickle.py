"""G-304: ChampionArtifact must serialize models whose nn.Module is a function-LOCAL class.

Smoke (2026-06-17) found OverseasTransfer.pt loaded as a "legacy bare-model pickle" → inference
used identity transform + no scaler (predictions ≠ training-time pipeline). Root cause: GAT
(graph_models._build_gat_model.<locals>.GraphAttentionDNN) and OverseasTransfer
(overseas_transfer._build_finetuning_model.<locals>.TransferModel) define their nn.Module as a
function-local class, so ChampionArtifact.to_pickle_bytes()'s standard pickle.dumps raised
"Can't get local object" → the ChampionArtifact (with transform_state + scaler) was never saved →
the preproc-less base.py torch-dict .pt remained. Fix: cloudpickle (by-value) in to_pickle_bytes.

macOS: run PER-FILE.
"""
import numpy as np
import pytest


def _local_class_model():
    """A torch nn.Module defined as a function-LOCAL class — exactly the GAT/OverseasTransfer
    pattern that standard pickle cannot serialize."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _LocalNet(nn.Module):  # <locals> — unpicklable by standard pickle
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, x):
            return self.lin(x).squeeze(-1)

    return _LocalNet()


def test_g304_standard_pickle_fails_on_local_class_model():
    """Documents the bug: a ChampionArtifact wrapping a local-class model can't standard-pickle."""
    import pickle
    from sklearn.preprocessing import StandardScaler
    from simulation.utils.model_artifact import make_artifact

    art = make_artifact(
        model=_local_class_model(), transform_name="log1p", transform_inv_obj=None,
        fitted_scaler=StandardScaler().fit(np.random.RandomState(0).normal(size=(20, 4))),
        feature_indices=[0, 1, 2, 3], model_name="GAT",
    )
    with pytest.raises(Exception, match="local object|pickle|Can't"):
        pickle.dumps(art)   # the standard path that failed for GAT/OverseasTransfer


def test_g304_to_pickle_bytes_roundtrips_local_class_as_champion_artifact(tmp_path):
    """FIX: to_pickle_bytes (cloudpickle) serializes the local-class model, and load_artifact
    returns a real ChampionArtifact (NOT the legacy bare-model wrapper) with transform + scaler."""
    from sklearn.preprocessing import StandardScaler
    from simulation.utils.model_artifact import make_artifact, load_artifact, ChampionArtifact

    scaler = StandardScaler().fit(np.random.RandomState(1).normal(size=(20, 4)))
    art = make_artifact(
        model=_local_class_model(), transform_name="log1p", transform_inv_obj=None,
        fitted_scaler=scaler, feature_indices=[0, 1, 2, 3], model_name="OverseasTransfer",
    )

    # 1. to_pickle_bytes must now SUCCEED (cloudpickle) where standard pickle failed
    b = art.to_pickle_bytes()
    assert isinstance(b, bytes) and len(b) > 0

    # 2. write the .pt exactly as ChampionLog does, then load_artifact it
    pt = tmp_path / "OverseasTransfer.pt"
    pt.write_bytes(b)
    loaded = load_artifact(pt)

    # 3. it must come back as a REAL ChampionArtifact — NOT the legacy bare wrapper
    assert isinstance(loaded, ChampionArtifact)
    assert loaded.config.get("legacy") is not True, "must not be the legacy bare-model fallback"
    # 4. and the preproc that the bare path would have dropped is preserved
    assert loaded.transform_name == "log1p", "transform must survive (was identity in bare path)"
    assert loaded.scaler is not None, "scaler must survive (was None in bare path)"
    assert loaded.feature_indices == [0, 1, 2, 3]


def test_g304_normal_models_still_roundtrip(tmp_path):
    """No regression: a normal (module-level) sklearn model still serializes + loads."""
    from sklearn.linear_model import Ridge
    from simulation.utils.model_artifact import make_artifact, load_artifact, ChampionArtifact

    X = np.random.RandomState(2).normal(size=(30, 4))
    y = X[:, 0] * 2 + 1
    art = make_artifact(model=Ridge().fit(X, y), transform_name="identity",
                        fitted_scaler=None, feature_indices=None, model_name="Ridge")
    pt = tmp_path / "Ridge.pt"
    pt.write_bytes(art.to_pickle_bytes())
    loaded = load_artifact(pt)
    assert isinstance(loaded, ChampionArtifact)
    assert loaded.config.get("legacy") is not True
