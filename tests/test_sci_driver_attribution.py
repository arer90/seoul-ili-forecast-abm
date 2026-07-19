"""Regression guard for simulation.scripts.sci_driver_attribution.

These tests pin the SCIENTIFIC INVARIANTS of the driver-attribution analysis so a
future cache/champion change cannot silently break the discovery claim:

  * the reconstructed champion test split aligns to the stored predictions
    (y_true max-abs-diff == 0) — the whole analysis is invalid otherwise;
  * the AR+seasonal control captures the bulk of ILI variance (R^2 high) —
    the premise that autocorrelation+seasonality is the dominant signal;
  * the partial-correlation confounding control runs and at least one external
    driver survives WHILE at least one is confounded-away (the discovery vs
    confirmation split is real, not all-or-nothing);
  * the zero-variance real-time rt_* drivers are flagged NON-IDENTIFIABLE, not
    silently scored as zero-importance "drivers";
  * permutation importance carries finite bootstrap CIs for every feature.

Run (per macOS per-file policy):
    .venv/bin/python -m pytest tests/test_sci_driver_attribution.py -q
"""
from __future__ import annotations

import numpy as np

from simulation.scripts import sci_driver_attribution as M


def _design():
    return M.load_design()


def test_split_aligns_to_stored_champion_predictions():
    d = _design()
    diff = M.verify_alignment(d)
    assert diff == 0.0, f"test y_true must match stored champion preds, got {diff}"
    assert d["n_test"] == 68 and d["n"] == 337


def test_ar_seasonal_is_dominant_signal():
    d = _design()
    _, r2_te = M.ar_seasonal_fit(d)
    assert r2_te > 0.7, f"AR+seasonal should dominate (R^2>0.7), got {r2_te:.3f}"


def test_partial_control_splits_genuine_vs_confounded():
    d = _design()
    rows, dof = M.partial_correlation(d)
    assert dof > 0 and rows, "partial correlation must produce rows with positive dof"
    survivors = [r for r in rows if r["survives"]]
    confounded = [r for r in rows
                  if not r["survives"] and "NON-IDENTIFIABLE" not in r["verdict"]]
    # the discovery story requires BOTH outcomes to be present
    assert survivors, "expected >=1 external driver to survive confounding control"
    assert confounded, "expected >=1 external driver to be confounded-away"
    # temperature is the canonical survivor in this dataset
    assert any(r["driver"].startswith("temp_") and r["survives"] for r in rows)
    # humidity is the canonical confounded-away driver (seasonal proxy)
    hum = next(r for r in rows if r["driver"] == "humidity")
    assert not hum["survives"]


def test_block_F_external_is_significant_but_small():
    d = _design()
    blk = M.incremental_block_test(d)
    assert blk["external_block_significant"] is True
    assert 0.0 < blk["delta_r2_external_block"] < 0.05, (
        "external block adds genuine but small incremental R^2")


def test_rt_drivers_flagged_non_identifiable():
    d = _design()
    # the champion's real-time mobility/density/air features are constant in-sample
    assert d["degenerate"], "rt_* drivers must be detected as zero-variance"
    rows, _ = M.partial_correlation(d)
    biases, vif_rows, dens = M.bias_and_vif(d, rows)
    non_ident = [r for r in vif_rows if r["vif"] is None]
    assert non_ident, "non-identifiable rt_* drivers must appear in the vif table"
    assert all("NON-IDENTIFIABLE" in r["vif_flag"] for r in non_ident)
    # a named bias must document the by-construction density identity
    assert any("by construction" in b["bias"] for b in biases)


def test_permutation_importance_has_finite_cis():
    d = _design()
    resid, _ = M.ar_seasonal_fit(d)
    # keep it cheap for CI: monkeypatch down the bootstrap budget
    old_b, old_r = M.N_BOOT, M.N_PERM_REPEAT
    M.N_BOOT, M.N_PERM_REPEAT = 200, 5
    try:
        rows, base_mae = M.permutation_importance_with_ci(d, resid)
    finally:
        M.N_BOOT, M.N_PERM_REPEAT = old_b, old_r
    assert len(rows) == 32 and np.isfinite(base_mae)
    for r in rows:
        assert np.isfinite(r["ci95_lo"]) and np.isfinite(r["ci95_hi"])
        assert r["ci95_lo"] <= r["importance_mae_increase"] <= r["ci95_hi"] + 1e-9
    # at least the Fourier seasonal terms reach significance
    assert any(r["significant_gt0"] and r["category"] == "SEAS" for r in rows)
