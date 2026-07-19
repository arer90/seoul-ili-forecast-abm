import numpy as np
import pytest


def _interval_score(y_true, y_pred, lower, upper, alpha=0.05):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    width = upper - lower
    over = np.maximum(lower - y_true, 0.0)
    under = np.maximum(y_true - upper, 0.0)
    return float(np.mean(np.abs(y_true - y_pred) + (alpha / 2.0) * width + over + under))


def test_tide_caps_point_predictions_and_intervals(monkeypatch):
    pytest.importorskip("torch")

    from simulation.models import _optuna_torch
    from simulation.models import dl_models
    from simulation.models.modern_ts.tide import TiDEForecaster

    monkeypatch.setattr(_optuna_torch, "run_optuna_loop", lambda *args, **kwargs: ({}, 0.0))
    monkeypatch.setattr(dl_models, "_train_loop", lambda *args, **kwargs: 0.0)

    def _huge_standardized_prediction(model, X_test):
        return np.full(len(X_test), 1000.0, dtype=np.float32)

    monkeypatch.setattr(dl_models, "_predict_torch", _huge_standardized_prediction)

    n = 96
    t = np.arange(n, dtype=np.float64)
    X_train = np.column_stack([
        t,
        np.sin(t / 6.0),
        np.cos(t / 6.0),
        np.linspace(-1.0, 1.0, n),
    ])
    y_train = np.linspace(0.0, 80.0, n)
    X_test = np.full((16, X_train.shape[1]), 1_000_000.0, dtype=np.float64)
    y_test = np.full(len(X_test), 80.0, dtype=np.float64)

    model = TiDEForecaster().fit(X_train, y_train)
    pred = model.predict(X_test)
    lower, upper = model.predict_interval(X_test, alpha=0.05)

    point_cap = float(y_train.max() * 1.5)
    assert np.all(pred <= point_cap + 1e-6)
    assert np.all(lower >= 0.0)
    assert np.all(lower <= upper)
    assert np.all(lower <= y_train.max() * 5.0)
    assert np.all(upper <= y_train.max() * 5.0)

    mae = float(np.mean(np.abs(y_test - pred)))
    wis = _interval_score(y_test, pred, lower, upper, alpha=0.05)
    assert mae > 0.0
    assert wis < 10.0 * mae
