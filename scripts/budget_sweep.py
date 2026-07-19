"""Budget-sensitivity smoke (user 2026-06-13): "빠르면서도 test 성능 높은" epoch/n_estimator 추천.

각 모델을 budget(n_estimator for trees, max_iter=epoch for MLP)별로 train→test 예측, **hold-out
test slab 의 WIS/r2 + 벽시계**를 측정해 곡선을 그린다. knee = budget 을 2배 늘려도 test-WIS 개선이
<2% 인 첫 지점 = "빠르면서도 충분히 좋은" 추천값. (Optuna trial 수 sweep 은 별도: 파이프라인 필요.)

STANDALONE — model_upgrade_harness 의 load_series/_wis_point 재사용(sim import 없음). 각 (model,budget)
셀은 격리 subprocess(OMP G-251) — driver 가 nice 로 병렬 spawn 후 JSON diff → CSV.

worker:  .venv/bin/python scripts/budget_sweep.py --worker <model> <knob> <val> [--n-test 68]
driver:  .venv/bin/python scripts/budget_sweep.py --out simulation/results/budget_sweep.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_upgrade_harness import _r2, _wis_point, load_series  # noqa: E402

# knob → 모델군 → budget 격자.
# ⚑ 트리(n_estimator) 전용 standalone — fit 에 preproc 불요라 빠르고 faithful(=test slab).
#   epoch(DL) 와 trial(Optuna) 은 sklearn MLP 가 (a) early-stop off 시 발산 (b) 실제 DL(TFT/NHITS)
#   epoch 와 구조가 달라 전이 불가 → 별도 파이프라인 기반 sweep(scripts/budget_sweep_pipeline.sh)에서.
TREE = ["lightgbm", "catboost", "randomforest", "xgboost", "extratrees"]
N_EST_GRID = [25, 50, 100, 200, 400]


def _fit_predict(model: str, knob: str, val: int, y, X, tr, te):
    """한 (model, budget) 셀 적합 → test 예측. 트리=n_estimators, mlp=max_iter."""
    Xtr, ytr, Xte = X[tr], y[tr].copy(), X[te]
    if model == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        ymu, ysd = float(ytr.mean()), (float(ytr.std()) or 1.0)
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=val,
                         early_stopping=False, random_state=42)
        m.fit(Xtr, (ytr - ymu) / ysd)
        return np.asarray(m.predict(Xte), float) * ysd + ymu
    if model == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=val, random_state=42, verbose=-1, n_jobs=2)
    elif model == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=val, depth=6, learning_rate=0.1,
                              random_state=42, verbose=0, thread_count=2)
    elif model == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=val, random_state=42, n_jobs=2)
    elif model == "xgboost":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=val, random_state=42, n_jobs=2, verbosity=0)
    elif model == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        m = ExtraTreesRegressor(n_estimators=val, random_state=42, n_jobs=2)
    else:
        raise SystemExit(f"unknown model: {model}")
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float)


def _worker(model: str, knob: str, val: int, n_test: int):
    t0 = time.time()
    y, X, _cols, tr, te = load_series(n_test)
    pred = _fit_predict(model, knob, val, y, X, tr, te)
    yt, ytr = y[te], y[tr]
    sigma = float(np.std(ytr))
    print(json.dumps({
        "model": model, "knob": knob, "val": val,
        "wis": round(_wis_point(yt, pred, sigma), 4),
        "r2": round(_r2(yt, pred), 4),
        "secs": round(time.time() - t0, 2),
    }))


def _cells():
    for m in TREE:
        for v in N_EST_GRID:
            yield (m, "n_estimator", v)


def _driver(out: str, n_test: int, jobs: int):
    cells = list(_cells())
    print(f"[budget] {len(cells)} 셀 (model×budget), 병렬 {jobs}, n_test={n_test}", file=sys.stderr)
    results, running = [], []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "KMP_DUPLICATE_LIB_OK": "TRUE"}

    def launch(cell):
        m, k, v = cell
        p = subprocess.Popen(
            ["nice", "-n", "10", sys.executable, os.path.abspath(__file__),
             "--worker", m, k, str(v), "--n-test", str(n_test)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True)
        return (cell, p)

    idx = 0
    while idx < len(cells) or running:
        while idx < len(cells) and len(running) < jobs:
            running.append(launch(cells[idx])); idx += 1
        cell, p = running.pop(0)
        out_s, _ = p.communicate()
        try:
            results.append(json.loads(out_s.strip().splitlines()[-1]))
            r = results[-1]
            print(f"  ✓ {r['model']:13s} {r['knob']:11s}={r['val']:<4} wis={r['wis']:.4f} "
                  f"r2={r['r2']:.4f} {r['secs']:.1f}s", file=sys.stderr)
        except Exception:
            print(f"  ✗ {cell} (실패)", file=sys.stderr)

    # CSV
    import csv
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "knob", "val", "wis", "r2", "secs"])
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["model"], x["val"])):
            w.writerow(r)
    print(f"\n[budget] → {out}", file=sys.stderr)
    _knee_report(results)


def _knee_report(results: list):
    """모델별: budget 2배에 test-WIS 개선 <2% 인 첫 지점 = knee(추천 budget)."""
    print("\n=== knee (빠르면서도 test 성능 충분) — budget 2배에 WIS 개선 <2% ===", file=sys.stderr)
    by_model: dict = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    recs = {}
    for model, rows in sorted(by_model.items()):
        rows = sorted(rows, key=lambda x: x["val"])
        knee = rows[0]
        for i in range(1, len(rows)):
            prev, cur = rows[i - 1]["wis"], rows[i]["wis"]
            improve = (prev - cur) / prev if prev > 0 else 0.0
            if improve < 0.02:           # <2% 개선 → 이전이 knee
                knee = rows[i - 1]; break
            knee = rows[i]
        recs[model] = knee
        best = min(rows, key=lambda x: x["wis"])
        print(f"  {model:13s} knee={knee['knob']}={knee['val']:<4} "
              f"(wis={knee['wis']:.4f}, {knee['secs']:.1f}s)  |  best={best['val']} wis={best['wis']:.4f}",
              file=sys.stderr)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=3, metavar=("MODEL", "KNOB", "VAL"))
    ap.add_argument("--out", default="simulation/results/budget_sweep.csv")
    ap.add_argument("--n-test", type=int, default=68)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.worker:
        _worker(args.worker[0], args.worker[1], int(args.worker[2]), args.n_test)
    else:
        _driver(args.out, args.n_test, args.jobs)


if __name__ == "__main__":
    main()
