"""
simulation/tests/smoke_pyg_lifecycle.py
========================================
graph_models_pyg.py lifecycle smoke test:
  - 8 pyg forecaster 전부 instantiate
  - BaseForecaster fit(X,y) -> predict(X_test) 호출
  - 출력 shape/range/non-negativity 가 WF-CV 파이프라인과 호환되는지 확인

현실 shape (n_train=222, n_test=12, p=30) 로 테스트.
EPOCHS 축소 없이 기본 200 epoch 이지만, early stopping (patience=25) 가 있어
실제 소요는 훨씬 짧음. 소표본(n=222) + p=30 이라 ~30초/모델 내외.

직접 실행:
  .venv\\Scripts\\python.exe -m simulation.tests.smoke_pyg_lifecycle
"""
from __future__ import annotations

import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def make_data(n_train=222, n_test=12, p=30, seed=42):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.standard_normal((n, p)).astype(np.float32)
    base = np.linspace(0, 6, n)
    seasonal = 2.0 * np.sin(2 * np.pi * np.arange(n) / 52)
    noise = 0.5 * rng.standard_normal(n)
    y = np.maximum(base + seasonal + noise, 0.1).astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def main():
    from simulation.models.graph_models_pyg import (
        GEChebForecaster, GESAGEForecaster, GETransformerForecaster,
        GEPNAForecaster, GEGCNpygForecaster, GEGINForecaster,
        GEARMAForecaster, GEResGatedForecaster,
    )

    classes = [
        # Tier 1
        GEChebForecaster, GESAGEForecaster, GETransformerForecaster,
        GEPNAForecaster,
        # Tier 2
        GEGCNpygForecaster, GEGINForecaster,
        GEARMAForecaster, GEResGatedForecaster,
    ]

    X_tr, y_tr, X_te, y_te = make_data(n_train=222, n_test=12, p=30)
    print(f"data: X_tr={X_tr.shape}, y_tr={y_tr.shape}, "
          f"X_te={X_te.shape}, y_te={y_te.shape}")
    print(f"y_tr range=[{y_tr.min():.2f}, {y_tr.max():.2f}], "
          f"y_te range=[{y_te.min():.2f}, {y_te.max():.2f}]")
    print("=" * 72)

    results = []
    for cls in classes:
        name = cls.meta.name
        t0 = time.time()
        try:
            m = cls()
            m.fit(X_tr, y_tr)
            yhat = m.predict(X_te)
            total = time.time() - t0
            ok_shape = (yhat.shape == (len(X_te),))
            ok_nonneg = bool(np.all(yhat >= 0))
            ok_finite = bool(np.all(np.isfinite(yhat)))
            status = "OK" if (ok_shape and ok_nonneg and ok_finite) else "BAD"
            print(f"[{status:>3}] {name:<18} time={total:5.1f}s "
                  f"shape={yhat.shape} nonneg={ok_nonneg} finite={ok_finite} "
                  f"range=[{yhat.min():.2f},{yhat.max():.2f}]")
            results.append({
                "model": name, "status": status, "time": total,
                "shape": yhat.shape,
                "range": (float(yhat.min()), float(yhat.max())),
            })
        except Exception as e:
            total = time.time() - t0
            print(f"[FAIL] {name:<18} time={total:5.1f}s  "
                  f"{type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
            results.append({
                "model": name, "status": "FAIL",
                "error": f"{type(e).__name__}: {e}",
            })
        print("-" * 72)

    print()
    print("SUMMARY")
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"  OK: {n_ok}/{len(results)}")
    for r in results:
        print(f"  {r['status']:<4} {r['model']}")
    return results


if __name__ == "__main__":
    main()
