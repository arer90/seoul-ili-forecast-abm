"""Phase A — STABILITY selection must force-include target AR-lags (anti-collapse).

Root cause of negative hold-out R² on many models: the pure-|corr| STABILITY screen
(`select_features_stability`) could drop the target autoregressive lags (ili_rate_lag1-4) when
weather/ARI features have higher marginal correlation — leaving a forecaster with NO recent-ILI
signal → collapse. The `mandatory` force-include guarantees the AR backbone survives selection.
Verified-by-construction here; end-to-end recovery is verified by the clean re-run (Phase C).

macOS: run PER-FILE.
"""
import numpy as np

from simulation.pipeline.feature_select_corr1se import select_features_stability


def _toy():
    """y strongly driven by 3 'weather' cols; 'lag1' is weakly correlated; 'lag_const' is constant.

    p=30 ≫ inner_k(=n//20=12) so the |corr| screen actually drops the weak AR-lag (without it,
    inner_k≥p would select everything and the test would be vacuous).
    """
    rng = np.random.default_rng(0)
    n = 240
    weather = rng.normal(size=(n, 3))
    y = weather @ np.array([3.0, 2.0, 1.5]) + rng.normal(scale=0.3, size=n)
    lag1 = 0.02 * y + rng.normal(scale=5.0, size=n)      # weak marginal corr (gets screened out)
    lag_const = np.ones(n)                               # constant → must NOT be forced
    noise = rng.normal(size=(n, 25))
    X = np.column_stack([weather, lag1, lag_const, noise])
    names = (["temp", "humidity", "ari", "ili_rate_lag1", "lag_const"]
             + [f"n{i}" for i in range(25)])
    return X, y, names


def test_ar_lag_dropped_without_mandatory():
    X, y, names = _toy()
    sel = select_features_stability(X, y, B=40, seed=42)
    # the weak AR-lag (index 3) is screened out by pure |corr|
    assert 3 not in sel["selected_indices"], "precondition: weak AR-lag should be dropped w/o force"


def test_ar_lag_force_included_with_mandatory():
    X, y, names = _toy()
    sel = select_features_stability(X, y, B=40, seed=42,
                                    feature_names=names, mandatory={"ili_rate_lag1"})
    assert 3 in sel["selected_indices"], "AR-lag must be force-included"
    assert sel["n_forced_mandatory"] >= 1


def test_constant_mandatory_not_forced():
    X, y, names = _toy()
    sel = select_features_stability(X, y, B=40, seed=42,
                                    feature_names=names, mandatory={"lag_const"})
    # constant column (index 4) carries no signal → must NOT be forced in
    assert 4 not in sel["selected_indices"]


def test_mandatory_already_selected_no_double_count():
    X, y, names = _toy()
    # 'temp' (index 0) is strongly selected anyway; forcing it adds nothing new
    sel = select_features_stability(X, y, B=40, seed=42,
                                    feature_names=names, mandatory={"temp"})
    assert 0 in sel["selected_indices"]
    assert sel["n_forced_mandatory"] == 0


def test_no_names_is_noop():
    X, y, _ = _toy()
    a = select_features_stability(X, y, B=40, seed=42)["selected_indices"]
    b = select_features_stability(X, y, B=40, seed=42, mandatory={"ili_rate_lag1"})["selected_indices"]
    assert a == b, "mandatory without feature_names must be a no-op"
