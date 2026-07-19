"""
simulation/tests/smoke_epi_l1_fix.py
========================================
fix: NegBinGLM + PoissonAutoreg L1 regularization + tighter PCA.

baseline : NegBinGLM R2=-2.18, PoissonAutoreg (baseline 비보고)
목표: R2 > -0.5 (개선) 또는 R2 > 0.0 (합격).

현실 shape: n_train=234, n_test=12, p=30.
ILI-like seasonal series (mean~7, peak~20).

직접 실행:
 .venv\\Scripts\\python.exe -m simulation.tests.smoke_epi_l1_fix
"""
from __future__ import annotations

import time
import warnings
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")


def make_ili_like(n_train=234, n_test=12, p=30, seed=42):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    t = np.arange(n)
    trend = 6.0 + 0.01 * t
    seasonal = 8.0 * np.sin(2 * np.pi * t / 52 - 1.0)
    noise = rng.normal(0, 1.5, n)
    y = np.maximum(trend + seasonal + noise, 0.1).astype(np.float32)
    # X 에 y 와 약한 상관 일부 (계절성 lag 반영) + 나머지는 noise
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, 0] = np.roll(y, 1) + 0.3 * rng.standard_normal(n)
    X[:, 1] = np.roll(y, 2) + 0.3 * rng.standard_normal(n)
    X[:, 2] = np.sin(2 * np.pi * t / 52) + 0.1 * rng.standard_normal(n)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _metric(name, y_true, y_pred, elapsed):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "name": name, "r2": float(r2), "rmse": float(rmse),
        "mean_pred": float(np.mean(y_pred)), "mean_true": float(np.mean(y_true)),
        "time": elapsed,
    }


def test_negbin(X_tr, y_tr, X_te, y_te):
    from simulation.models.epi_models import NegBinGLMForecaster
    t0 = time.time()
    m = NegBinGLMForecaster()
    m.fit(X_tr, y_tr)
    yhat = m.predict(X_te)
    return _metric("NegBinGLM", y_te, yhat, time.time() - t0)


def test_poisson_ar(X_tr, y_tr, X_te, y_te):
    from simulation.models.epi_models import PoissonAutoregForecaster
    t0 = time.time()
    m = PoissonAutoregForecaster()
    m.fit(X_tr, y_tr)
    yhat = m.predict(X_te)
    return _metric("PoissonAutoreg", y_te, yhat, time.time() - t0)


def test_gp_rbf(X_tr, y_tr, X_te, y_te):
    from simulation.models.epi_models import GaussianProcessForecaster
    t0 = time.time()
    m = GaussianProcessForecaster()
    m.fit(X_tr, y_tr)
    yhat = m.predict(X_te)
    return _metric("GP-RBF-Periodic", y_te, yhat, time.time() - t0)


def test_gam(X_tr, y_tr, X_te, y_te):
    from simulation.models.epi_models import GAMForecaster
    t0 = time.time()
    m = GAMForecaster()
    m.fit(X_tr, y_tr)
    yhat = m.predict(X_te)
    return _metric("GAM-Spline", y_te, yhat, time.time() - t0)


def main():
    X_tr, y_tr, X_te, y_te = make_ili_like()
    print(f"data: X_tr={X_tr.shape} y_tr=[{y_tr.min():.2f},{y_tr.max():.2f}, mean={y_tr.mean():.2f}] "
          f"y_te=[{y_te.min():.2f},{y_te.max():.2f}, mean={y_te.mean():.2f}]")
    print("=" * 78)

    results = []
    for name, fn in [
        ("NegBinGLM", test_negbin),
        ("PoissonAutoreg", test_poisson_ar),
        ("GP-RBF-Periodic", test_gp_rbf),
        ("GAM-Spline", test_gam),
    ]:
        print(f"[{name}] fitting...")
        try:
            r = fn(X_tr, y_tr, X_te, y_te)
            print(f"  R2={r['r2']:+.4f}  RMSE={r['rmse']:.2f}  "
                  f"mean_pred={r['mean_pred']:.2f} vs true={r['mean_true']:.2f}  "
                  f"time={r['time']:.1f}s")
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {type(e).__name__}: {e}")
            results.append({"name": name, "r2": float("-inf"), "rmse": float("inf"), "error": str(e)})

    print()
    print("=" * 78)
    print("VERDICT:")
    for r in results:
        r2 = r.get("r2", float("-inf"))
        verdict = "PASS" if r2 > 0.0 else ("IMPROVED" if r2 > -0.5 else "STILL_BAD")
        print(f"  [{verdict:>9}]  {r['name']:<20}  R2={r2:+.4f}")
    return results


if __name__ == "__main__":
    main()
