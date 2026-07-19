"""B (selection metric): `_evaluate_config` WIS must be CALIBRATION-AWARE, not the
degenerate fixed-sigma point-MAE that the user (rightly) suspected.

Background: selection minimized OOF WIS, but `sigma_for_wis=std(y_train)` is
model-INDEPENDENT → the Gaussian PI is identical across configs → WIS collapsed to
point-MAE ranking (calibration never rewarded). codex+gemini converged: switch to
`weighted_interval_score_empirical` (split-conformal, the model's OWN in-sample train
residuals, original target space) — avoids the transform-space sigma dead-end AND rewards
calibration. The gate's PICP95 (also fixed-sigma) gets the same empirical treatment.

RED on the old Gaussian-fixed-sigma path; GREEN after the swap.

macOS: run PER-FILE.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
# (2026-06-24 fix) 합성 _ConstResidStub.fit 에 **kwargs 추가 — _evaluate_config 가 feature_names=
#   를 넘기는데 옛 stub 이 안 받아 TypeError→sentinel inf 였음. 이제 calibration-aware WIS 속성 검증됨.


class _ConstResidStub:
    """fit: store train y. predict: on train-length X return ``y_train - s`` (so every train
    residual == s, a controllable spread); on val-length X return a FIXED val forecast (same
    point prediction / same MAE across stubs). Two stubs then differ ONLY in train-residual
    spread → a calibration-aware WIS scores them differently; degenerate fixed-sigma does not.
    """

    def __init__(self, resid_scale, n_train, val_pred):
        self.s = float(resid_scale)
        self.n_train = int(n_train)
        self.val_pred = np.asarray(val_pred, float)

    def fit(self, X, y, **kwargs):          # **kwargs: _evaluate_config 가 feature_names= 등 전달
        self._ytr = np.asarray(y, float).ravel()
        return self

    def predict(self, X):
        if len(X) == self.n_train:
            return self._ytr - self.s          # train residual = y - (y - s) = s  (constant)
        return self.val_pred                    # fixed val forecast (same across stubs)


def _data(n_train=80, n_val=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_val, 4))
    y = np.abs(X[:, 0]) + rng.gamma(2.0, 1.0, n_train + n_val)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _run(resid_scale, sigma):
    from simulation.pipeline.per_model_optimize import _evaluate_config
    Xtr, ytr, Xva, yva = _data()
    val_pred = yva + 0.5    # identical point forecast (identical MAE) for every stub
    res = _evaluate_config(
        lambda: _ConstResidStub(resid_scale, len(Xtr), val_pred),
        Xtr, ytr, Xva, yva,
        transform_name="identity", scaler_name="none",
        sigma_for_wis=sigma, feature_cols=["f0", "f1", "f2", "f3"],
    )
    return res


def test_wis_rewards_calibration_same_point_forecast():
    """Same val prediction (same MAE), different train-residual spread → WIS MUST differ.
    Degenerate fixed-sigma WIS gives them the SAME score (RED)."""
    wis_narrow = _run(0.3, 1.0)["wis"]
    wis_wide = _run(8.0, 1.0)["wis"]
    assert abs(wis_narrow - wis_wide) > 1e-6, (
        f"WIS ignores model calibration (degenerate point-MAE): "
        f"narrow_resid={wis_narrow} wide_resid={wis_wide}")


def test_wis_no_longer_driven_by_fixed_sigma():
    """The passed sigma_for_wis must NOT drive WIS anymore (empirical residuals do).
    Old Gaussian path: WIS swings wildly with sigma (RED)."""
    wis_s_lo = _run(0.3, 0.01)["wis"]
    wis_s_hi = _run(0.3, 100.0)["wis"]
    assert abs(wis_s_lo - wis_s_hi) < 1e-6, (
        f"WIS still depends on fixed sigma_for_wis (degenerate): lo={wis_s_lo} hi={wis_s_hi}")


def test_gate_picp_is_empirical_not_fixed_sigma():
    """Gate PICP95 must also use the empirical residual band, not Z95·fixed-sigma."""
    pi_narrow = _run(0.3, 1.0)["pi95_coverage"]
    pi_wide = _run(8.0, 1.0)["pi95_coverage"]
    # narrow train residuals (q95≈0.3) vs val error 0.5 → under-covers (low PICP);
    # wide residuals (q95≈8) → over-covers (PICP≈1). Fixed-sigma gives them the SAME PICP.
    assert abs(pi_narrow - pi_wide) > 1e-6, (
        f"gate PICP ignores model residuals (fixed-sigma): "
        f"narrow={pi_narrow} wide={pi_wide}")
