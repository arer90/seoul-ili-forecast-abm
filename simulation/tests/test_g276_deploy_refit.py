"""Q5 / G-276: 배포용 artifact(전체-데이터 재학습) 회귀 가드.

eval .pt = train+val fit (hold-out metric 동결). _deploy.pt = train+val+test+real fit
(운영 forecast 최신 관측 반영). 선택/HP/transform 은 frozen, fitting 데이터만 확장.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from simulation.pipeline.per_model_optimize import _build_deploy_artifact
from simulation.models.epi_models import BayesianRidgeForecaster
from simulation.utils.model_artifact import load_artifact


def _data():
    rng = np.random.RandomState(0)
    def blk(n):
        X = rng.randn(n, 5)
        y = np.abs(X[:, 0] * 5 + 20 + rng.randn(n))
        return X, y
    return blk(100), blk(30), blk(8)   # pool, test, real


_BEST = {"transform": "identity", "scaler": "none",
         "preproc_optuna_params": None, "n_features": 5}
_FCOLS = [f"f{i}" for i in range(5)]


def test_deploy_artifact_created_with_full_data(tmp_path):
    (Xp, yp), (Xt, yt), (Xr, yr) = _data()
    path = _build_deploy_artifact(
        BayesianRidgeForecaster, _BEST, None, _FCOLS, "none", None,
        "TestBR", tmp_path, Xp, yp, Xt, yt, Xr, yr,
    )
    assert path is not None, "deploy artifact 미생성"
    p = Path(path)
    assert p.exists() and p.name == "TestBR_deploy.pt"
    art = load_artifact(p)
    assert art is not None
    # 전체 = 100+30+8 = 138 (eval pool 100 보다 큼)
    assert art.config.get("n_train_full") == 138, art.config
    assert art.config.get("deploy") is True


def test_deploy_predicts_roundtrip(tmp_path):
    (Xp, yp), (Xt, yt), (Xr, yr) = _data()
    path = _build_deploy_artifact(
        BayesianRidgeForecaster, _BEST, None, _FCOLS, "none", None,
        "TestBR2", tmp_path, Xp, yp, Xt, yt, Xr, yr,
    )
    art = load_artifact(path)
    pred = art.predict(np.random.RandomState(1).randn(10, 5))
    assert pred is not None and len(pred) == 10
    assert np.all(np.isfinite(pred))


def test_deploy_disabled_by_env(tmp_path):
    os.environ["MPH_DEPLOY_REFIT"] = "0"
    try:
        (Xp, yp), (Xt, yt), (Xr, yr) = _data()
        path = _build_deploy_artifact(
            BayesianRidgeForecaster, _BEST, None, _FCOLS, "none", None,
            "TestBR3", tmp_path, Xp, yp, Xt, yt, Xr, yr,
        )
        assert path is None, "MPH_DEPLOY_REFIT=0 인데 생성됨"
    finally:
        os.environ.pop("MPH_DEPLOY_REFIT", None)


def test_deploy_without_real(tmp_path):
    """real 슬랩 없어도(서비스존 0주) train+val+test 로 동작."""
    (Xp, yp), (Xt, yt), _ = _data()
    path = _build_deploy_artifact(
        BayesianRidgeForecaster, _BEST, None, _FCOLS, "none", None,
        "TestBR4", tmp_path, Xp, yp, Xt, yt, None, None,
    )
    assert path is not None
    art = load_artifact(path)
    assert art.config.get("n_train_full") == 130   # 100+30
