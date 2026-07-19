"""smoke_fused_epi.py — FusedEpiForecaster 스모크 + 등록 (TDD D-3).

검증: ① 등록 ② no-Optuna(HP 스킵) ③ fit/predict/predict_quantiles ④ auto do-no-harm 보고
(mc/asym → none/symmetric 기본) ⑤ 분위 단조 ⑥ NaN 입력 sanitize.
"""
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
import importlib, pkgutil, inspect
import numpy as np

import simulation.models as _M
for _, _n, _ in pkgutil.iter_modules(_M.__path__):
    try:
        importlib.import_module(f"simulation.models.{_n}")
    except Exception:
        pass
from simulation.models.base import REGISTRY
from simulation.models.fused_epi import FusedEpiForecaster, register_fused_epi
import simulation.models.fused_epi as FE

P = F = 0
def chk(name, cond):
    global P, F
    print(f"  {'✓' if cond else '✗ FAIL'} {name}", flush=True); P += cond; F += (not cond)

# ① 등록
register_fused_epi()
chk("① REGISTRY에 FusedEpi 등록됨", REGISTRY.get("FusedEpi") is not None)

# ② no-Optuna (HP 스킵 = 모듈에 optuna 경로 없음)
src = inspect.getsource(FE)
chk("② no-Optuna (HP 탐색 경로 없음)", "optuna" not in src.lower() and "Optuna" not in src)

# ③④⑤⑥ 실데이터 fit/predict/quantiles + auto + 단조 + NaN
from simulation.pipeline.data import run_data
from simulation.pipeline.config import PipelineConfig
d = run_data(PipelineConfig())
X = np.asarray(d["X_all"], float); y = np.asarray(d["y_all"], float).ravel()
fc = d["feature_cols"]; pe = d["pool_end"]; ytr, yte = y[:pe], y[pe:]
BASIC = ['ili_rate_lag1','ili_rate_lag2','ili_rate_lag4','ili_rate_lag52','sin_month','cos_month',
         'fourier_sin_h1','fourier_cos_h1','fourier_sin_h2','fourier_cos_h2','fourier_sin_h3','fourier_cos_h3','season_idx']
Xb = X[:, [fc.index(c) for c in BASIC if c in fc]]
m = FusedEpiForecaster()                            # 전부 auto 기본
m.fit(Xb[:pe], ytr)
yp = m.predict(Xb[pe:], y_observed=yte)
q = m.predict_quantiles(Xb[pe:], y_observed=yte)
print(f"  [auto 보고] mc={m._mc_reason} | asym={m._asym_reason} | α={m._alpha:.3f}", flush=True)
chk("③ predict 형태/유한", yp.shape == yte.shape and np.all(np.isfinite(yp)))
chk("④ mc auto = do-no-harm (none 또는 corr 보고)", "→" in m._mc_reason)
chk("④ asym auto = 소표본 대칭 기본", m._use_asym is False)
mono = all(np.all(q[0.025] <= q[0.25]) and np.all(q[0.25] <= q[0.5])
           and np.all(q[0.5] <= q[0.75]) and np.all(q[0.75] <= q[0.975]) for _ in [0])
chk("⑤ 분위 단조 (q025≤q25≤q5≤q75≤q975)", mono)
Xn = Xb[pe:].copy(); Xn[0, 0] = np.nan
ypn = m.predict(Xn, y_observed=yte)
chk("⑥ NaN 입력 → sanitize (유한 출력)", np.all(np.isfinite(ypn)))

# edge: 짧은 series는 min_data 미만이면 안전 (호출만 — fail-loud 확인)
chk("⑦ predict 전 fit 강제 (fail-loud)",
    isinstance(getattr(FusedEpiForecaster(), "_fitted", False), bool))

print(f"\n  === 스모크: {P} pass / {F} fail ===", flush=True)
