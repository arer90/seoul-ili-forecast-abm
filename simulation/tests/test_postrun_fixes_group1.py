"""Post-run fix group 1 — 모델 캡슐 smoke (2026-06-15, 패치 wsr99u0vb).

bug-fix test-after smoke(D-3). heavy-dep(lightgbm/torch/statsmodels/pygam) 혼재 →
macOS OpenMP segfault 회피 위해 per-test 실행 권장:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
        simulation/tests/test_postrun_fixes_group1.py -k <name> -x -q
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Fix 1: CQR-LightGBM n_jobs=1 (OMP #179 SIGSEGV → 52/53) ──────────────────
def test_cqr_lightgbm_njobs_is_1(monkeypatch):
    captured = []

    class _StubLGBM:
        def __init__(self, **kw):
            captured.append(kw)

        def fit(self, X, y):
            return self

    import lightgbm
    monkeypatch.setattr(lightgbm, "LGBMRegressor", _StubLGBM)
    from simulation.models.cqr_models import CQRLightGBMForecaster
    m = CQRLightGBMForecaster()
    m.fit(np.random.rand(20, 3), np.random.rand(20))
    assert len(captured) == 2, "q_lo·q_hi 두 head 생성"
    assert all(c.get("n_jobs") == 1 for c in captured), \
        f"n_jobs 는 1 이어야(OMP #179 회피), got {[c.get('n_jobs') for c in captured]}"


# ── Fix 2: torch.save round-trip 게이트 (0-byte .pt loud-fail) ────────────────
def _tiny_torch_forecaster():
    import torch.nn as nn
    from simulation.models.base import BaseForecaster, ModelMeta

    class _TinyTorch(BaseForecaster):
        meta = ModelMeta(name="TinyTorch", category="dl", level=1, min_data=1,
                         description="t", dependencies=[])

        def fit(self, X, y):
            self._fitted = True
            return self

        def predict(self, X):
            return np.zeros(len(X))

    m = _TinyTorch()
    m._model = nn.Linear(3, 1)
    m._fitted = True
    return m


def test_torch_save_rejects_zero_byte(monkeypatch, tmp_path):
    import torch
    m = _tiny_torch_forecaster()
    p = tmp_path / "z.pt"
    monkeypatch.setattr(torch, "save", lambda *a, **k: p.write_bytes(b""))
    with pytest.raises(RuntimeError, match="무결성"):
        m.save(str(p))
    assert not p.exists(), "0-byte 손상 .pt 는 unlink 되어야"


def test_torch_save_rejects_corrupt(monkeypatch, tmp_path):
    import torch
    m = _tiny_torch_forecaster()
    p = tmp_path / "b.pt"
    monkeypatch.setattr(torch, "save", lambda *a, **k: p.write_bytes(b"broken"))
    with pytest.raises(RuntimeError, match="무결성"):
        m.save(str(p))
    assert not p.exists()


def test_torch_save_normal_roundtrip(tmp_path):
    import torch
    m = _tiny_torch_forecaster()
    p = tmp_path / "ok.pt"
    m.save(str(p))                       # 진짜 torch.save → round-trip 통과해야
    assert p.exists() and p.stat().st_size > 0
    d = torch.load(str(p), weights_only=False)
    assert isinstance(d, dict) and "model_state_dict" in d


# ── Fix 3: GAM-Spline cap (log-space 외삽 폭발) ───────────────────────────────
def test_gam_spline_cap():
    from simulation.models.epi_models import GAMForecaster
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(float)
    y = np.clip(5 + 2 * X[:, 0] + rng.normal(size=60), 0, None)
    m = GAMForecaster()
    m.fit(X, y)
    assert hasattr(m, "_y_train_max")
    X_ext = rng.normal(size=(20, 4)).astype(float) * 50.0   # 극단 외삽
    pred = m.predict(X_ext)
    assert np.all(pred <= 2.0 * np.max(y) + 1e-6), "외삽 cap = 2*y_max"
    assert np.all(pred >= 0.0)


def test_gam_spline_allzero_y():
    from simulation.models.epi_models import GAMForecaster
    X = np.random.default_rng(1).normal(size=(40, 3)).astype(float)
    m = GAMForecaster()
    m.fit(X, np.zeros(40))
    assert np.all(m.predict(X * 10) == 0.0), "y_max=0 → cap 0"


# ── Fix 4: hhh4 per-step mu cap (AR(1) exp 폭발) ─────────────────────────────
def test_hhh4_perstep_cap():
    from simulation.models.hhh4_models import Hhh4EquivalentForecaster
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3)).astype(float)
    y = np.clip(5 + np.sin(np.arange(80) / 5.0) * 3 + rng.normal(size=80), 0, None)
    m = Hhh4EquivalentForecaster()
    m.fit(X, y)
    ymax = float(np.max(y))

    # _glm_res 가 폭발값을 내도 per-step cap 이 y_prev 발산을 차단해야
    class _Boom:
        def predict(self, design):
            return np.array([1e6])
    m._glm_res = _Boom()
    pred = m.predict(rng.normal(size=(30, 3)).astype(float))
    assert np.all(pred <= 3.0 * ymax + 1e-6), f"per-step cap 3*y_max, got max={pred.max()}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q"]))
