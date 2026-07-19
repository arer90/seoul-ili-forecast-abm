"""G-279 fix: OverseasTransfer encoder cache — pretrain once, reuse (frozen ⇒ no leakage).

The LSTM encoder was re-pretrained (100 epochs) on EVERY fit() — and fit() is called hundreds
of times inside the per-model 3-stage Optuna (preproc × OOF folds × HP) → the single biggest
OverseasTransfer bottleneck. The encoder is FROZEN during fine-tuning (only the decoder head
trains), so a (countries, hyperparams)-keyed cache reuses the same frozen encoder across fits
with zero leakage. Gated behind MPH_OVERSEAS_ENCODER_CACHE (default OFF = live-run safe).

macOS: run PER-FILE.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from simulation.models import overseas_transfer as OT


def _tiny():
    rng = np.random.default_rng(0)
    n, p = 40, 6
    X = rng.normal(size=(n, p))
    y = np.abs(rng.normal(5, 2, size=n))
    names = ["ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag3", "ili_rate_lag4", "temp", "humidity"]
    return X, y, names


def _make():
    m = OT.OverseasTransferForecaster()
    m.epochs_pretrain = 1      # keep the test fast
    m.epochs_finetune = 1
    return m


def test_encoder_reused_when_cache_enabled(monkeypatch):
    monkeypatch.setenv("MPH_OVERSEAS_ENCODER_CACHE", "1")
    OT._ENCODER_CACHE.clear()
    X, y, names = _tiny()
    m1 = _make(); m1.fit(X, y, feature_names=names)
    assert len(OT._ENCODER_CACHE) == 1, "first fit must populate the cache"
    enc1 = m1._encoder
    m2 = _make(); m2.fit(X, y, feature_names=names)
    assert m2._encoder is enc1, "second fit must REUSE the cached encoder (no re-pretrain)"
    assert len(OT._ENCODER_CACHE) == 1


def test_no_cache_when_disabled_live_run_safe(monkeypatch):
    monkeypatch.setenv("MPH_OVERSEAS_ENCODER_CACHE", "0")
    OT._ENCODER_CACHE.clear()
    X, y, names = _tiny()
    m1 = _make(); m1.fit(X, y, feature_names=names)
    assert len(OT._ENCODER_CACHE) == 0, "gate OFF → no caching (live run unaffected)"
    m2 = _make(); m2.fit(X, y, feature_names=names)
    assert m2._encoder is not m1._encoder, "gate OFF → fresh encoder each fit (old behavior)"
