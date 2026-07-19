"""
r3_5_npi_ablation.py
====================
SEIR-V2-Forced NPI ablation: kappa=0 (no NPI covariate) vs kappa fit (full model).

Quantifies how much the explicit NPI term contributes to fit quality in the
COVID window (2020-03-02 ~ 2022-12-26). Only retrains the SEIR-V2-Forced
model — does NOT touch the other 62 ML/DL models (cheap, ~10 min).

Outputs:
  simulation/results/post_E/r3_5_npi_ablation.json
    {
      "full":     {"kappa": 0.xx, "rmse": ..., "mape": ..., "r2": ...},
      "ablated":  {"kappa": 0.00, "rmse": ..., "mape": ..., "r2": ...},
      "delta":    {"rmse_pct": ..., "r2_abs": ..., "p_dm": ...}
    }

Interpretation:
  - delta_r2_abs > 0.05    => NPI covariate materially improves fit
  - p_dm < 0.05            => DM-test rejects "equal loss" null
  - otherwise              => NPI covariate is absorbed by seasonal/intercept
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "simulation" / "results" / "post_E" / "r3_5_npi_ablation.json"


def _load_ili() -> np.ndarray:
    """주간 ILI rate (KDCA sentinel_influenza, 전국 전연령 평균)."""
    from simulation.models.feature_engine.loaders import _load_sentinel_ili
    from simulation.database.config import DB_PATH
    df = _load_sentinel_ili(str(DB_PATH))
    return df["ili_rate"].to_numpy()


def _fit_seir_v2(y: np.ndarray, kappa_bounds: tuple[float, float]) -> dict:
    """SEIR-V2-Forced 를 주어진 κ bounds 로 L-BFGS-B fit. 테스트 15% holdout.

    κ bounds = (0, 0) 이면 NPI covariate 무효화 (ablation).
    """
    from simulation.models.seir_forced import SEIRForcedForecaster
    from simulation.models.base import ModelMeta

    n = len(y)
    n_train = int(n * 0.85)
    y_tr, y_te = y[:n_train], y[n_train:]
    X_tr, X_te = np.zeros((n_train, 1)), np.zeros((len(y_te), 1))

    mdl = SEIRForcedForecaster()
    # inject bound override via attribute (monkey-patch safe since local instance)
    _orig_fit = mdl.fit

    def _fit_with_kappa_bound(X_train, y_train, **kw):
        """kappa bounds 를 override 하기 위해 fit 를 interception 한 뒤
        내부 scipy L-BFGS-B bounds 교체. 실제로는 전체 fit 를 복제한 뒤
        κ=0 고정 (bounds=(0,0)) 으로 재실행."""
        import scipy.optimize as spo
        from simulation.models.seir_forced import SEIRForcedParams
        mdl._train_len = len(y_train)
        y_scaled = np.clip(y_train, 0.0, None).astype(float)

        def _obj(theta):
            p = SEIRForcedParams(
                beta0=float(theta[0]), epsilon=float(theta[1]), phi=float(theta[2]),
                sigma=float(theta[3]), gamma=float(theta[4]),
                kappa=float(theta[5]), I0_frac=float(theta[6]),
            )
            sim = mdl._simulate(p, 0.0, float(mdl._train_len - 1))
            if np.any(np.isnan(sim)):
                return 1e12
            I_t = sim[2]
            pred = I_t / mdl._population * mdl._rate_scale
            sse_log = np.sum((np.log1p(pred) - np.log1p(y_scaled)) ** 2)
            sse_lin = np.sum((pred - y_scaled) ** 2)
            y_var = max(float(np.var(y_scaled)), 1e-6)
            return float(0.7 * sse_log + 0.3 * sse_lin / y_var)

        x0 = np.array([0.45, 0.20, 2.0, 0.5, 0.2,
                       (kappa_bounds[0] + kappa_bounds[1]) / 2.0, 0.001])
        bounds = [
            (0.1, 1.5),     # beta0
            (0.0, 0.35),    # epsilon
            (0.0, 52.0),    # phi
            (0.25, 1.0),    # sigma
            (0.10, 0.5),    # gamma
            kappa_bounds,   # <-- ablation control
            (1e-5, 0.01),   # I0_frac
        ]
        res = spo.minimize(_obj, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-6})
        mdl._params = SEIRForcedParams(
            beta0=res.x[0], epsilon=res.x[1], phi=res.x[2],
            sigma=res.x[3], gamma=res.x[4], kappa=res.x[5], I0_frac=res.x[6],
        )
        sim = mdl._simulate(mdl._params, 0.0, float(mdl._train_len - 1))
        if not np.any(np.isnan(sim)):
            mdl._last_state = sim[:, -1]
        mdl._fitted = True
        return mdl

    _fit_with_kappa_bound(X_tr, y_tr)
    y_pred = mdl.predict(X_te)
    resid = y_te - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mape = float(np.mean(np.abs(resid / np.clip(y_te, 1e-3, None))) * 100)
    return {
        "kappa": float(mdl._params.kappa),
        "beta0": float(mdl._params.beta0),
        "epsilon": float(mdl._params.epsilon),
        "rmse": rmse, "mape": mape, "r2": r2,
        "y_pred": y_pred.tolist(),
        "y_true": y_te.tolist(),
    }


def _diebold_mariano(y_true: np.ndarray, f1: np.ndarray, f2: np.ndarray, h: int = 1) -> dict:
    """Squared-error DM test. f1 = ablated, f2 = full."""
    d = (y_true - f1) ** 2 - (y_true - f2) ** 2
    n = len(d)
    mean_d = float(np.mean(d))
    # Newey-West variance with lag = h-1 (Harvey et al. 1997 small-sample adj)
    var_d = float(np.var(d, ddof=1)) / max(n, 1)
    if var_d <= 0:
        return {"stat": float("nan"), "p_value": float("nan")}
    stat = mean_d / np.sqrt(var_d)
    from scipy import stats as sstats
    p = 2.0 * (1.0 - sstats.norm.cdf(abs(stat)))
    return {"stat": float(stat), "p_value": float(p)}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    y = _load_ili()
    print(f"[r3_5] ILI series loaded: n={len(y)}")

    print("[r3_5] fitting FULL model (kappa bounds = [0, 0.6]) ...")
    full = _fit_seir_v2(y, kappa_bounds=(0.0, 0.6))
    print(f"  kappa={full['kappa']:.3f}  R2={full['r2']:.4f}  RMSE={full['rmse']:.3f}")

    print("[r3_5] fitting ABLATED model (kappa bounds = [0, 0]) ...")
    ablated = _fit_seir_v2(y, kappa_bounds=(0.0, 0.0))
    print(f"  kappa={ablated['kappa']:.3f}  R2={ablated['r2']:.4f}  RMSE={ablated['rmse']:.3f}")

    y_te = np.asarray(full["y_true"])
    yp_full = np.asarray(full["y_pred"])
    yp_abl = np.asarray(ablated["y_pred"])
    dm = _diebold_mariano(y_te, yp_abl, yp_full)

    delta = {
        "rmse_pct": (ablated["rmse"] - full["rmse"]) / full["rmse"] * 100,
        "r2_abs": full["r2"] - ablated["r2"],
        "dm_stat": dm["stat"],
        "p_dm": dm["p_value"],
    }
    print(f"\n[r3_5] Ablation delta: dRMSE={delta['rmse_pct']:+.2f}% "
          f"dR2={delta['r2_abs']:+.4f}  DM_stat={delta['dm_stat']:.3f} "
          f"p={delta['p_dm']:.4f}")

    if delta["r2_abs"] > 0.05 and delta["p_dm"] < 0.05:
        verdict = "NPI covariate MATERIALLY improves fit"
    elif delta["r2_abs"] > 0.01:
        verdict = "NPI covariate MARGINALLY improves fit"
    else:
        verdict = "NPI covariate ABSORBED by seasonal/intercept (no unique signal)"
    print(f"[r3_5] Verdict: {verdict}")

    # Drop the heavy arrays before JSON dump
    full_light = {k: v for k, v in full.items() if k not in ("y_pred", "y_true")}
    abl_light = {k: v for k, v in ablated.items() if k not in ("y_pred", "y_true")}

    OUT.write_text(json.dumps({
        "full": full_light,
        "ablated": abl_light,
        "delta": delta,
        "verdict": verdict,
        "n_train": int(len(y) * 0.85),
        "n_test": len(y_te),
    }, indent=2), encoding="utf-8")
    print(f"\n[r3_5] OK -> {OUT}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
