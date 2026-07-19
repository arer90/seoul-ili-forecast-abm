"""A3 (M7 SCI-grade): per-horizon prediction-interval coverage.

The pipeline reported PI coverage only globally / by COVID regime — always at
h=1 (WF-CV step_size=1). Reviewers expect a per-horizon coverage table because
coverage almost always *decays* with the forecast horizon. This module computes
empirical split-conformal coverage per horizon from the multi-horizon
rolling-origin forecasts (``_rolling_origin_multihorizon``).

Pairs with the existing CQR / AdaptiveConformalTracker / EnbPI machinery in
``simulation.models.conformal`` for the heteroscedastic / non-exchangeable cases.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def per_horizon_coverage(
    multihorizon_preds: dict,
    y_test: np.ndarray,
    alpha: float = 0.05,
    oof_residuals_by_h: Optional[dict] = None,
) -> dict:
    """Empirical split-conformal PI coverage PER forecast horizon.

    For each horizon ``h`` the PI half-width is the ``1-α`` quantile of
    ``|residuals|`` — the leakage-free OOF per-horizon residuals when provided,
    else the test-slab residuals (the split-conformal baseline, same leak caveat
    as ``intervals._conformal_pi``). ``multihorizon_preds[h][i]`` is the forecast
    of ``y_test[i + h - 1]``.

    Args:
        multihorizon_preds: ``{h: preds_array}`` from
            ``real_eval._rolling_origin_multihorizon``.
        y_test: (n_test,) observed test values.
        alpha: miscoverage level (0.05 → nominal 95% PI).
        oof_residuals_by_h: optional ``{h: residual_array}`` for leak-free
            per-horizon calibration.

    Returns:
        ``{"per_horizon": {h: {coverage, width, quantile, n}}, "min_coverage",
        "nominal", "decays_with_h"}``. Never raises.
    """
    y_test = np.asarray(y_test, dtype=np.float64)
    out: dict = {}
    for h, preds in sorted(multihorizon_preds.items()):
        try:
            preds = np.asarray(preds, dtype=np.float64)
            actual = y_test[h - 1: h - 1 + len(preds)]
            n = min(len(preds), len(actual))
            preds, actual = preds[:n], actual[:n]
            m = np.isfinite(preds) & np.isfinite(actual)
            if int(m.sum()) < 2:
                out[h] = {"coverage": float("nan"), "width": float("nan"),
                          "quantile": float("nan"), "n": int(m.sum())}
                continue
            if oof_residuals_by_h and h in oof_residuals_by_h:
                resid = np.asarray(oof_residuals_by_h[h], dtype=np.float64)
                resid = resid[np.isfinite(resid)]
            else:
                resid = actual[m] - preds[m]  # test-slab (leak caveat)
            q = float(np.quantile(np.abs(resid), 1.0 - alpha)) if resid.size else float("nan")
            lo, hi = preds[m] - q, preds[m] + q
            cov = float(np.mean((actual[m] >= lo) & (actual[m] <= hi)))
            out[h] = {"coverage": round(cov, 4), "width": round(2.0 * q, 2),
                      "quantile": round(q, 4), "n": int(m.sum())}
        except Exception as e:   # never break the report over one horizon
            out[h] = {"coverage": float("nan"), "error": str(e)}
    covs = [v["coverage"] for v in out.values()
            if isinstance(v.get("coverage"), float) and np.isfinite(v["coverage"])]
    hs = sorted(out)
    decays = bool(len(covs) >= 2 and all(
        (out[hs[i]]["coverage"] >= out[hs[i + 1]]["coverage"] - 1e-9)
        for i in range(len(hs) - 1)
        if np.isfinite(out[hs[i]].get("coverage", np.nan))
        and np.isfinite(out[hs[i + 1]].get("coverage", np.nan))
    ))
    return {"per_horizon": out, "min_coverage": (min(covs) if covs else float("nan")),
            "nominal": round(1.0 - alpha, 4), "decays_with_h": decays}
