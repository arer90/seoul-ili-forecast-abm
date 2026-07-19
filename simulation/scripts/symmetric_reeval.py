"""symmetric_reeval.py — §8.6 symmetric 재평가 (있는 그대로 config + 매 origin 전체 재fit).

파이프라인 ``run_data`` (burn 회피) + per_model_optimal best_config + ``_symmetric_rolling_eval``.
빠른 모델만(DL/graph/foundation 게이트 = 68× 재fit 비현실 + 챔피언 후보 아님). MPH_EVAL_FEATURES=basic.
출력: symmetric 공정 ranking (ARIMA fit-once 비대칭 제거 후 ML 이 따라잡나).
"""
import os, json, glob

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
import numpy as np

from simulation.pipeline.data import run_data
from simulation.pipeline.per_model_optimize import _symmetric_rolling_eval
from simulation.models.base import REGISTRY
import importlib, pkgutil
import simulation.models as _M
for _, _nm, _ in pkgutil.iter_modules(_M.__path__):   # 전 모델 모듈 import → registry populate
    try:
        importlib.import_module(f"simulation.models.{_nm}")
    except Exception:
        pass

from simulation.pipeline.config import PipelineConfig
config = PipelineConfig()

d = run_data(config)
X_all = np.asarray(d["X_all"], dtype=float)
y_all = np.asarray(d["y_all"], dtype=float).ravel()
feature_cols = d.get("feature_cols")
n_train = d.get("n_train")
pool_end = d.get("pool_end", (n_train or 0) + d.get("n_val", 0))
n = len(y_all)
X_pool, y_pool = X_all[:pool_end], y_all[:pool_end]
X_test, y_test = X_all[pool_end:], y_all[pool_end:]
print(f"[data] n={n} pool_end={pool_end} test={len(y_test)} p={X_all.shape[1]} (run_data, burn 회피)")

# 챔피언 후보 우선 순서 (LightGBM/CQR-LightGBM 제외 = macOS OpenMP segfault, 프로세스 죽임).
PRIORITY = ["ARIMA", "PoissonAutoreg", "NegBinGLM", "NegBinGLM-Glum", "SARIMA", "SARIMAX",
            "RandomForest", "XGBoost", "TabPFN", "ElasticNet", "KRR", "SVR-Linear", "SVR-RBF",
            "Theta", "CQR-QuantReg", "GAM-Spline", "hhh4-equivalent", "EpiEstim",
            "BayesianRidge", "Wallinga-Teunis", "GLARMA"]

rows = []
for nm in PRIORITY:
    f = f"simulation/results/per_model_optimal/{nm}.json"
    if not os.path.exists(f):
        continue
    cls = REGISTRY.get(nm)
    if cls is None:
        continue
    bc = json.load(open(f)).get("best_config", {}) or {}
    factory = (lambda c=cls: c())
    try:
        r = _symmetric_rolling_eval(
            factory, bc.get("transform", "identity"), bc.get("scaler", "none"),
            X_pool, y_pool, X_test, y_test,
            feature_indices=bc.get("feature_indices"), feature_cols=feature_cols,
            hier_frozen_params=bc.get("preproc_optuna_params"), sigma_for_wis=1.0)
        w, r2, nn = r.get("wis"), r.get("r2"), r.get("n")
        rows.append((nm, w, r2, nn))
        print(f"  {nm:18} sym_wis={w!s:>9} r2={r2!s:>8} n={nn}")
    except Exception as e:
        print(f"  {nm:18} FAIL {str(e)[:70]}")

rows = [x for x in rows if isinstance(x[1], (int, float)) and np.isfinite(x[1])]
rows.sort(key=lambda x: x[1])
print("\n=== §8.6 SYMMETRIC 공정 ranking (WIS 오름차순 = 매 origin 재fit, ARIMA/ML 동일) ===")
for i, (nm, w, r2, nn) in enumerate(rows, 1):
    print(f"  {i:2}. {nm:18} sym_wis={w:.4f}  r2={r2:.3f}")
