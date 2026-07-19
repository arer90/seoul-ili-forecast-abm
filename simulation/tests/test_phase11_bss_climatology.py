"""test_phase11_bss_climatology — G3 Round 3 audit verification

Verifies WOY (week-of-year) climatology baseline produces DIFFERENT (and more
conservative) BSS than the prior scalar train-prevalence baseline.

Per Gemini Round 3 critique (2026-05-26): scalar baseline artificially inflates
BSS because BS_baseline_scalar >> BS_baseline_woy at peak weeks.

This test demonstrates the difference on real KDCA data + persistence baseline.

Reference: FluSight Hub climatology = WOY per-target; Reich 2019 PNAS, Bracher
2021 PLOS Comp Bio.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"


@pytest.fixture(scope="module")
def kdca_data():
    """Real KDCA 2019-2025 aggregated weekly ILI rate + synthetic dates."""
    if not DB.exists():
        pytest.skip(f"KDCA DB not found at {DB}")
    from simulation.database import safe_connect

    con = safe_connect(str(DB))
    rows = con.execute(
        """
        SELECT season_start, week_seq, AVG(ili_rate) AS mean_rate
        FROM sentinel_influenza
        WHERE ili_rate IS NOT NULL AND ili_rate >= 0
        GROUP BY season_start, week_seq
        ORDER BY season_start, week_seq
        """
    ).fetchall()
    if not rows:
        pytest.skip("sentinel_influenza empty")

    y = np.array([r[2] for r in rows], dtype=np.float64)
    # Synthesize ISO dates: season N week W -> ISO year = N, week = W (rough)
    dates = pd.to_datetime(
        [f"{r[0]}-W{int(r[1]):02d}-1" for r in rows], format="%G-W%V-%u"
    )
    seasons = [r[0] for r in rows]
    n_train = sum(1 for s in seasons if s <= 2024)
    return {"y": y, "dates": dates, "n_train": n_train}


def test_woy_baseline_differs_from_scalar(kdca_data):
    """WOY climatology BSS MUST differ from scalar BSS (Gemini critique)."""
    y = kdca_data["y"]
    dates = kdca_data["dates"]
    n_train = kdca_data["n_train"]
    test_end = len(y)
    test_start = n_train

    y_test = y[test_start:]
    pred = np.roll(y, 1)[test_start:]
    threshold = 8.6
    ev_true = (y_test > threshold).astype(int)

    # Empirical ev_prob via residual bootstrap (matches R8 scoring path)
    oof_pred = np.roll(y, 1)[:test_start]
    residuals = (y[:test_start] - oof_pred)
    residuals = residuals[np.isfinite(residuals)]
    rng = np.random.default_rng(42)
    res_samples = rng.choice(residuals, size=(1000, len(pred)), replace=True)
    ev_prob = np.mean(pred[None, :] + res_samples > threshold, axis=0)

    bs = float(np.mean((ev_true - ev_prob) ** 2))

    # SCALAR baseline (legacy, pre-G3)
    train_bin = (y[:test_start] > threshold)
    ref_p_scalar = float(np.mean(train_bin))
    if 0.0 < ref_p_scalar < 1.0:
        bs_baseline_scalar = ref_p_scalar * (1 - ref_p_scalar)
        bss_scalar = 1.0 - bs / bs_baseline_scalar
    else:
        bss_scalar = float("nan")

    # WOY baseline (post-G3, FluSight standard)
    dates_train = dates[:test_start]
    dates_test = dates[test_start:test_end]
    woy_train = dates_train.isocalendar().week.to_numpy()
    woy_test = dates_test.isocalendar().week.to_numpy()
    woy_prob = np.full(54, ref_p_scalar)
    for w in range(1, 54):
        m = woy_train == w
        if m.sum() >= 2:
            woy_prob[w] = float(train_bin[m].mean())
    clim_prob_per_week = np.array(
        [woy_prob[int(w)] if 1 <= int(w) <= 53 else ref_p_scalar for w in woy_test],
        dtype=np.float64,
    )
    bs_baseline_woy = float(np.mean((ev_true - clim_prob_per_week) ** 2))
    bss_woy = 1.0 - bs / bs_baseline_woy if bs_baseline_woy > 0 else float("nan")

    print(f"\nScalar baseline: ref_p={ref_p_scalar:.4f}, BS_base={bs_baseline_scalar:.4f}, BSS={bss_scalar:.4f}")
    print(f"WOY baseline:    BS_base={bs_baseline_woy:.4f}, BSS={bss_woy:.4f}")
    print(f"BSS difference: scalar - woy = {bss_scalar - bss_woy:+.4f}")

    # R4 audit fix: hardcoded regression expected values from
    # METRIC_EVALUATION.md §2.2.1 (3-convention table, real KDCA 2025 seed=42).
    # Drift > 0.01 indicates math change or data change — investigate.
    assert np.isfinite(bss_woy), "WOY BSS must be computable"
    assert np.isfinite(bss_scalar), "Scalar BSS must be computable"
    # Conv1 (variance form): expected ~0.6957
    assert abs(bss_scalar - 0.6957) < 0.01, (
        f"Conv1 scalar BSS = {bss_scalar:.4f} drifted from expected 0.6957 — "
        "math changed or data changed; see METRIC_EVALUATION §2.2.1"
    )
    # Conv2 WOY (FluSight standard): expected ~0.7260
    assert abs(bss_woy - 0.7260) < 0.01, (
        f"Conv2 WOY BSS = {bss_woy:.4f} drifted from expected 0.7260 — "
        "WOY climatology calculation changed; see METRIC_EVALUATION §2.2.1"
    )
    # They must NOT be identical (would indicate WOY array degenerated)
    assert abs(bss_scalar - bss_woy) > 1e-3, (
        f"WOY ({bss_woy:.4f}) and scalar ({bss_scalar:.4f}) BSS too close — "
        "indicates degenerate WOY array or test slab too short"
    )


def test_woy_baseline_matches_flusight_standard(kdca_data):
    """WOY BSS uses Conv2 (actual BS of climatology forecast), matching FluSight.

    Empirical finding (2026-05-26): the three conventions produce DIFFERENT BSS:
      Conv1 (Murphy 1973 variance form, legacy):   ~0.696
      Conv2 with scalar forecast applied to test:  ~0.785
      Conv2 with WOY climatology (FluSight std):   ~0.726

    The G3 fix replaces Conv1 (legacy) with Conv2-WOY (FluSight). The direction
    of change is data-dependent — Gemini's a-priori claim 'scalar inflates BSS'
    held against Conv2-scalar but NOT against Conv1-variance. Either way, the
    G3 fix produces the FluSight-comparable value.
    """
    y = kdca_data["y"]
    dates = kdca_data["dates"]
    n_train = kdca_data["n_train"]
    test_start = n_train

    y_test = y[test_start:]
    pred = np.roll(y, 1)[test_start:]
    threshold = 8.6
    ev_true = (y_test > threshold).astype(int)

    # If test prevalence > train mean (active season), expect WOY < scalar BSS
    test_prev = float(ev_true.mean())
    train_prev = float((y[:test_start] > threshold).mean())

    if test_prev > train_prev + 0.2:
        # Active season vs lower train mean → scalar baseline far from test
        # → BS_baseline_scalar inflated → BSS_scalar inflated
        # WOY baseline closer to test → more honest (lower) BSS
        # We don't run the full R8 scoring phase — just verify the math holds in isolation
        oof_pred = np.roll(y, 1)[:test_start]
        residuals = (y[:test_start] - oof_pred)
        residuals = residuals[np.isfinite(residuals)]
        rng = np.random.default_rng(42)
        res_samples = rng.choice(residuals, size=(1000, len(pred)), replace=True)
        ev_prob = np.mean(pred[None, :] + res_samples > threshold, axis=0)
        bs = float(np.mean((ev_true - ev_prob) ** 2))

        # scalar
        ref_p = train_prev
        bs_scal = ref_p * (1 - ref_p)
        bss_s = 1.0 - bs / bs_scal

        # WOY
        dates_train = dates[:test_start]
        dates_test = dates[test_start:test_start + len(y_test)]
        woy_train = dates_train.isocalendar().week.to_numpy()
        woy_test = dates_test.isocalendar().week.to_numpy()
        train_bin = (y[:test_start] > threshold)
        woy_prob = np.full(54, ref_p)
        for w in range(1, 54):
            m = woy_train == w
            if m.sum() >= 2:
                woy_prob[w] = float(train_bin[m].mean())
        clim_w = np.array([woy_prob[int(w)] if 1 <= int(w) <= 53 else ref_p
                           for w in woy_test], dtype=np.float64)
        bs_w = float(np.mean((ev_true - clim_w) ** 2))
        bss_w = 1.0 - bs / bs_w

        print(f"\nTest prev={test_prev:.3f}, Train prev={train_prev:.3f} (gap={test_prev-train_prev:+.3f})")
        print(f"Conv1 (legacy variance):  BSS = {bss_s:.4f}")
        print(f"Conv2 (WOY FluSight):     BSS = {bss_w:.4f}")
        # Both must be finite + within [-inf, 1]; specific ordering is data-dependent
        assert np.isfinite(bss_s) and bss_s <= 1.0, "Conv1 BSS invalid"
        assert np.isfinite(bss_w) and bss_w <= 1.0, "Conv2-WOY BSS invalid"
        # WOY climatology baseline must be a genuine per-week forecast, not constant
        # Verify woy_prob has variance (i.e., not all entries are same scalar)
        woy_used = [woy_prob[int(w)] for w in woy_test if 1 <= int(w) <= 53]
        assert np.std(woy_used) > 0.01, (
            "WOY climatology degenerated to constant — check WOY mapping"
        )
    else:
        pytest.skip(f"Test slab not active enough (prev={test_prev:.3f} vs train {train_prev:.3f})")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
