"""new_model_eval.py — 새 모델 baseline-급 가벼운 학습+평가 (1-step, fit-once-apply-rowwise).

env 자유 시 standalone 가벼운 평가(full 파이프라인 X). SeirCount-TabPFN(±mechanistic) vs 베이스라인.
run_data(burn 회피) + evaluate_predictions_full(129-metric SSOT) + native-NB-WIS.
"""
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
import importlib, pkgutil
import numpy as np

from simulation.pipeline.data import run_data
from simulation.pipeline.config import PipelineConfig
from simulation.pipeline.phase_evaluator import evaluate_predictions_full
from simulation.pipeline.per_model_optimize import _native_interval_wis
from simulation.models.seir_count import SeirCountForecaster
from simulation.models.feature_engine._loaders.mechanistic import mechanistic_features
from simulation.models.base import REGISTRY
import simulation.models as _M
for _, _n, _ in pkgutil.iter_modules(_M.__path__):
    try:
        importlib.import_module(f"simulation.models.{_n}")
    except Exception:
        pass

d = run_data(PipelineConfig())
X = np.asarray(d["X_all"], dtype=float)
y = np.asarray(d["y_all"], dtype=float).ravel()
pe = d.get("pool_end", d.get("n_train", 0) + d.get("n_val", 0))
print(f"[data] n={len(y)} pool_end={pe} test={len(y)-pe} p={X.shape[1]}")


def ev(name, model, Xtr, ytr, Xte, yte):
    try:
        model.fit(Xtr, ytr)
        yp = np.asarray(model.predict(Xte), dtype=float)
        m = evaluate_predictions_full(yte, yp, sigma=1.0, y_train_pool=ytr, phase_id="new_model")
        nb = ""
        if hasattr(model, "predict_quantiles"):
            nv = _native_interval_wis(model, Xte, yte)
            if nv:
                nb = f"  native_wis={nv['native_wis']:.3f} picp95={nv['native_picp95']:.2f}"
        print(f"  {name:28} wis={m.get('wis', float('nan')):.3f} r2={m.get('r2', float('nan')):.3f}"
              f" mae={m.get('mae', float('nan')):.3f}{nb}")
    except Exception as e:
        print(f"  {name:28} FAIL {str(e)[:70]}")


print("\n=== 새 모델 (baseline-급 1-step 평가) ===")
ev("SeirCount-TabPFN(lag)", SeirCountForecaster(engine="tabpfn"), X[:pe], y[:pe], X[pe:], y[pe:])
mech = mechanistic_features(y)                       # (T,3) — mech[t]는 inc[t]=y[t] 사용(현재)
# ★누수 방지: y[t] 예측에 mech[t](=y[t] 포함) 붙이면 누수 → 1-lag 해서 mech_lag[t]=mech[t-1](y[:t-1]만).
mech_lag = np.vstack([mech[:1], mech[:-1]])          # shift down 1 (mech_lag[t]=mech[t-1])
Xm = np.hstack([X, mech_lag])
ev("SeirCount-TabPFN+mech", SeirCountForecaster(engine="tabpfn"), Xm[:pe], y[:pe], Xm[pe:], y[pe:])

print("\n=== 베이스라인 (동일 1-step, 비교) ===")
for bn in ["TabPFN", "NegBinGLM", "ARIMA", "ElasticNet"]:
    cls = REGISTRY.get(bn)
    if cls is not None:
        ev(f"[base] {bn}", cls(), X[:pe], y[:pe], X[pe:], y[pe:])
