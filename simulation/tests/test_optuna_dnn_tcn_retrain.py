"""Regression guard for the DNN/TCN Optuna retrain KeyError (G-237, 2026-05-30).

Incident: the Optuna objective fixes ``aug_factor = 0`` and never calls
``trial.suggest_int("augment_factor", ...)``, so ``study.best_params`` has no
``augment_factor`` key. The retrain step then did ``bp["augment_factor"]`` →
``KeyError: 'augment_factor'`` → DNN-Optuna / TCN-Optuna failed 100% of the time,
and runner.py:1330 mislabeled it as "subprocess 실패 (OOM 또는 timeout)".

Fix: ``bp.get("augment_factor", 0)`` at dl_models.py:899 (DNN) and :1339 (TCN).

macOS: run PER-FILE (memory ``test-suite-execution``)::

    .venv/bin/python -m pytest simulation/tests/test_optuna_dnn_tcn_retrain.py -q
"""
from pathlib import Path

import numpy as np
import pytest


def _xy(n: int = 80, n_feat: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = 10.0 + 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.5, n)  # learnable signal
    return X, y


def test_dl_models_no_bare_augment_factor_subscript():
    """Static guard (instant): augment_factor must use bp.get(...), never bp['augment_factor']."""
    src = Path("simulation/models/dl_models.py").read_text(encoding="utf-8")
    assert 'bp["augment_factor"]' not in src and "bp['augment_factor']" not in src, (
        "bp['augment_factor'] subscript present — use bp.get('augment_factor', 0) "
        "(G-237: objective fixes aug=0 so the key is absent in best_params)"
    )


def test_optuna_dnn_retrain_no_keyerror():
    """OptunaDNNForecaster.fit→predict must not raise KeyError: 'augment_factor'."""
    from simulation.models.dl_models import OptunaDNNForecaster

    X, y = _xy(n=80)
    m = OptunaDNNForecaster()
    m.N_TRIALS = 2  # __init__ already ran _get_trials; override before fit
    m.fit(X, y)  # pre-fix: KeyError 'augment_factor' on retrain
    pred = np.atleast_1d(m.predict(X[-12:]))
    assert pred.shape[0] >= 1 and np.all(np.isfinite(pred))


def test_optuna_tcn_retrain_no_keyerror():
    """OptunaTCNForecaster.fit→predict must not raise KeyError: 'augment_factor'."""
    from simulation.models.dl_models import OptunaTCNForecaster

    X, y = _xy(n=100)
    m = OptunaTCNForecaster()
    m.N_TRIALS = 2
    m.fit(X, y)
    pred = np.atleast_1d(m.predict(X[-12:]))
    assert pred.shape[0] >= 1 and np.all(np.isfinite(pred))
