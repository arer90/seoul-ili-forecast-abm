"""전 active 모델 readiness 검증 (사용자 2026-06-13): WF-CV + hold-out 둘 다 fast 측정.

각 모델을 REGISTRY 경유(파이프라인과 동일 경로)로 fit/predict — (1) 통합/등록이 정상인지,
(2) hold-out(test-slab 직접) vs WF-CV(walk-forward rolling) 성능 차이를 표로. fast 설정
(MPH_MAX_EPOCHS=10/20, 트리는 본래 빠름)으로 빠르게. OMP #179 회피 위해 모델당 별도 프로세스.

용법:
  단일:  .venv/bin/python scripts/verify_all_models.py --model XGBoost
  전체:  .venv/bin/python scripts/verify_all_models.py --all   # 모델당 subprocess spawn → 표
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
CACHE = "simulation/cache/feature_cache.parquet"


def _load(n_test=80):
    import polars as pl
    df = pl.read_parquet(CACHE).sort("week_start")
    y = df["ili_rate"].to_numpy().astype(float)
    n = len(y)
    fc = [c for c, t in zip(df.columns, df.dtypes)
          if c not in ("ili_rate", "week_start") and t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    X = df.select(fc).fill_null(0.0).to_numpy().astype(float)
    return X, y, np.arange(n - n_test), np.arange(n - n_test, n)


def _r2(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ss = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - float(np.sum((a - b) ** 2)) / ss if ss > 0 else float("nan")


def run_one(name: str) -> dict:
    """ONE model via REGISTRY: hold-out(test-slab 직접) + WF-CV(3 expanding fold rolling) r2."""
    t0 = time.time()
    out = {"model": name, "ok": False, "holdout_r2": None, "wfcv_r2": None,
           "pred_max": None, "err": None, "sec": 0.0}
    try:
        from simulation.models.registry import verify_registry_coverage
        verify_registry_coverage(force_import=True)
        from simulation.models.base import REGISTRY
        cls = REGISTRY.get(name)
        if cls is None:
            out["err"] = "미등록"; return out
        X, y, tr, te = _load()

        # hold-out: train 전체 fit → test-slab 직접 예측
        m = cls(); m.fit(X[tr], y[tr])
        ph = np.asarray(m.predict(X[te]), float)
        out["holdout_r2"] = round(_r2(y[te], ph), 3)
        out["pred_max"] = round(float(np.max(ph)), 1)

        # WF-CV: 3 expanding-window fold, 각 fold 의 다음 구간 예측 (rolling-origin)
        n_tr = len(tr); fold = n_tr // 4
        wf_true, wf_pred = [], []
        for k in range(1, 4):
            end = fold * (k + 1)
            if end + fold > n_tr:
                break
            mk = cls(); mk.fit(X[tr][:end], y[tr][:end])
            pk = np.asarray(mk.predict(X[tr][end:end + fold]), float)
            wf_true.extend(y[tr][end:end + fold].tolist()); wf_pred.extend(pk.tolist())
        if wf_true:
            out["wfcv_r2"] = round(_r2(np.array(wf_true), np.array(wf_pred)), 3)
        out["ok"] = bool(np.all(np.isfinite(ph)))
    except Exception as e:
        out["err"] = f"{type(e).__name__}: {str(e)[:90]}"
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.model:
        print(json.dumps(run_one(args.model))); return

    if args.all:
        from simulation.models.registry import verify_registry_coverage, CATEGORY_MODELS
        verify_registry_coverage(force_import=True)
        order = ["tree", "linear", "kernel", "other", "epi-extended", "ts", "dl-tabular",
                 "modern-ts", "cqr", "graph", "foundation", "ensemble"]
        env = os.environ.copy()
        env["MPH_MAX_EPOCHS"] = env.get("MPH_MAX_EPOCHS", "10")
        env["MPH_FAST_TRAIN"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["OMP_NUM_THREADS"] = "1"   # force_import 가 xgboost+lightgbm+torch 동시로드 → OMP #179 회피 필수
        results = []
        for cat in order:
            for name in CATEGORY_MODELS.get(cat, []):
                p = subprocess.run([sys.executable, __file__, "--model", name],
                                   capture_output=True, text=True, env=env, timeout=900)
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                r = json.loads(line[-1]) if line else {"model": name, "ok": False,
                                                        "err": "no output / crash", "sec": 0}
                r["cat"] = cat
                results.append(r)
                flag = "✓" if r.get("ok") else "✗"
                print(f"  {flag} {name:22s} hold={str(r.get('holdout_r2')):>7s} "
                      f"wfcv={str(r.get('wfcv_r2')):>7s} {r.get('err') or ''}", flush=True)
        out_path = Path(tempfile.gettempdir()) / "verify_all_models.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        ok = sum(1 for r in results if r.get("ok"))
        print(f"\n=== {ok}/{len(results)} OK → {out_path} ===")


if __name__ == "__main__":
    main()
