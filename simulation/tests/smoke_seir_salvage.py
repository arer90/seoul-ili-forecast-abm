"""
simulation/tests/smoke_seir_salvage.py
========================================
Metapop-SEIR / Bayesian-SEIR forecasting 재현 테스트.

[질문]
  baseline_v22.4.3 에서 Test R² = -1.32 (Metapop), -1.30 (Bayesian).
  원인이 (a) 파라미터 보정 버그, (b) MCMC 수렴 실패, (c) 구조적 한계 중 무엇인가?

[프로토콜]
  1. 같은 shape (n_tr=234, n_te=12) 의 현실적 시뮬 데이터로 baseline 성능 측정
  2. Metapop: β_scale Optuna 10 trials 로 최적화 → 회복 여부
  3. Bayesian-SEIR: burn_in 3x + proposal scale 2x → MCMC 수렴 여부
  4. 여전히 R² < 0.3 이면 forecasting 에서 격리 권고

직접 실행:
  .venv\\Scripts\\python.exe -m simulation.tests.smoke_seir_salvage
"""
from __future__ import annotations

import time
import warnings
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")


def make_ili_like(n_train=234, n_test=12, seed=42):
    """Realistic ILI-rate series (per-mille).

    mean ~5.0, peak ~20, noise σ ~1.5 (일반적 서울 influenza 계절형).
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    t = np.arange(n)
    # 52-week seasonality + slow drift
    trend = 6.0 + 0.01 * t
    seasonal = 8.0 * np.sin(2 * np.pi * t / 52 - 1.0)
    noise = rng.normal(0, 1.5, n)
    y = np.maximum(trend + seasonal + noise, 0.1).astype(np.float32)
    X = rng.standard_normal((n, 30)).astype(np.float32)  # dummy, unused
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _metric(name, y_true, y_pred, elapsed):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_pred = float(np.mean(y_pred))
    return {
        "name": name, "r2": float(r2), "rmse": float(rmse),
        "mean_pred": mean_pred, "mean_true": float(np.mean(y_true)),
        "time": elapsed,
    }


def test_metapop_default(X_tr, y_tr, X_te, y_te):
    from simulation.models.metapop_seir import MetapopSEIRForecaster
    t0 = time.time()
    m = MetapopSEIRForecaster()
    m.fit(X_tr, y_tr)
    yhat = m.predict(X_te)
    return _metric("Metapop-SEIR (default)", y_te, yhat, time.time() - t0)


def test_metapop_optuna_beta(X_tr, y_tr, X_te, y_te, n_trials=10):
    """β_scale 을 continuous search 로 최적화 → 회복 여부."""
    from simulation.models.metapop_seir import MetapopSEIRForecaster

    # 훈련을 마지막 20% (val) vs 나머지로 나눠 β 보정
    n_val = max(int(len(y_tr) * 0.15), 8)
    y_inner_tr = y_tr[:-n_val]
    y_val = y_tr[-n_val:]
    X_inner_tr = X_tr[:-n_val]
    X_val = X_tr[-n_val:]

    best_scale, best_r2 = 1.0, -np.inf
    for scale in np.linspace(0.4, 2.5, n_trials):
        m = MetapopSEIRForecaster()
        m.fit(X_inner_tr, y_inner_tr)
        m.beta_calibrated = m.disease.beta * scale
        yhat = m.predict(X_val)
        try:
            r2 = r2_score(y_val, yhat)
        except Exception:
            r2 = -np.inf
        if r2 > best_r2:
            best_r2, best_scale = r2, scale

    t0 = time.time()
    m_final = MetapopSEIRForecaster()
    m_final.fit(X_tr, y_tr)
    m_final.beta_calibrated = m_final.disease.beta * best_scale
    yhat = m_final.predict(X_te)
    result = _metric(f"Metapop-SEIR (β_scale={best_scale:.2f})", y_te, yhat, time.time() - t0)
    result["best_val_r2"] = float(best_r2)
    return result


def test_bayesian_default(X_tr, y_tr, X_te, y_te):
    from simulation.models.bayesian_seir import BayesianSEIRForecaster
    t0 = time.time()
    m = BayesianSEIRForecaster()
    m.fit(X_tr, y_tr, n_samples=500, burn_in=500)  # smoke: 절반
    yhat = m.predict(X_te)
    return _metric("Bayesian-SEIR (default, halved)", y_te, yhat, time.time() - t0)


def test_bayesian_longer_burnin(X_tr, y_tr, X_te, y_te):
    from simulation.models.bayesian_seir import BayesianSEIRForecaster
    t0 = time.time()
    m = BayesianSEIRForecaster()
    m.fit(X_tr, y_tr, n_samples=2000, burn_in=3000)
    yhat = m.predict(X_te)
    return _metric("Bayesian-SEIR (burn_in=3000, n=2000)", y_te, yhat, time.time() - t0)


def main():
    X_tr, y_tr, X_te, y_te = make_ili_like(n_train=234, n_test=12)
    print(f"data: X_tr={X_tr.shape} y_tr=[{y_tr.min():.2f},{y_tr.max():.2f}, mean={y_tr.mean():.2f}] "
          f"y_te=[{y_te.min():.2f},{y_te.max():.2f}, mean={y_te.mean():.2f}]")
    print("=" * 78)

    # Baseline + fix 비교
    results = []

    print("[1/4] Metapop-SEIR default (baseline 재현)...")
    r = test_metapop_default(X_tr, y_tr, X_te, y_te)
    results.append(r)
    print(f"   R2={r['r2']:.4f}  RMSE={r['rmse']:.2f}  mean_pred={r['mean_pred']:.2f} vs true={r['mean_true']:.2f}")

    print("[2/4] Metapop-SEIR + Optuna β_scale (10 trials)...")
    r = test_metapop_optuna_beta(X_tr, y_tr, X_te, y_te, n_trials=10)
    results.append(r)
    print(f"   R2={r['r2']:.4f}  RMSE={r['rmse']:.2f}  mean_pred={r['mean_pred']:.2f}  "
          f"best_val_r2={r['best_val_r2']:.4f}")

    print("[3/4] Bayesian-SEIR default (n=500, burn=500)...")
    r = test_bayesian_default(X_tr, y_tr, X_te, y_te)
    results.append(r)
    print(f"   R2={r['r2']:.4f}  RMSE={r['rmse']:.2f}  mean_pred={r['mean_pred']:.2f}")

    print("[4/4] Bayesian-SEIR burn_in=3000 n=2000...")
    r = test_bayesian_longer_burnin(X_tr, y_tr, X_te, y_te)
    results.append(r)
    print(f"   R2={r['r2']:.4f}  RMSE={r['rmse']:.2f}  mean_pred={r['mean_pred']:.2f}")

    print()
    print("=" * 78)
    print("VERDICT:")
    for r in results:
        verdict = "SAVE" if r['r2'] > 0.3 else ("MAYBE" if r['r2'] > 0 else "DROP")
        print(f"  [{verdict:>5}]  {r['name']:<45}  R2={r['r2']:+.4f}  time={r['time']:.1f}s")

    n_save = sum(1 for r in results if r['r2'] > 0.3)
    print(f"\n  {n_save}/{len(results)} passed threshold (R2 > 0.3)")

    return results


if __name__ == "__main__":
    main()
