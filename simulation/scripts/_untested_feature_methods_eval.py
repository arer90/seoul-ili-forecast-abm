"""Part 3: 아직 안 해본 feature 선택 방법 추가 비교 (사용자 "안 해본 feature 방법").

기존 측정: forward/backward/binary/threshold/STABILITY(|corr|)/mRMR/embedded + model-based + POOL/BLEND.
추가(미측정):
  - MI       : mutual-information stability (|corr| 의 비선형 marginal 버전 — 비선형 의존 포착)
  - RFE      : Recursive Feature Elimination (RandomForest wrapper, n_features=9)
실데이터 n=242, OOF-R²(선택목적 proxy) + test-R²(일반화), RF 평가, subprocess 격리.
가설: ILI 는 AR-지배(Part 2)라 비선형 MI 도 lag1 을 1순위로 → |corr| stability 와 비슷하거나 못 이김.

worker: python -m simulation.scripts._untested_feature_methods_eval --method <CORR|MI|RFE>
parent: python -m simulation.scripts._untested_feature_methods_eval
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
METHODS = ["CORR", "MI", "RFE"]
PER_TIMEOUT = 900


def _r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else -9.0


def _oof_r2(fac, X, y, idx, n_folds=3):
    idx = list(idx); n = len(y); fs = n // (n_folds + 1); r2s = []
    for k in range(1, n_folds + 1):
        etr = k * fs; eva = (k + 1) * fs if k < n_folds else n
        if eva - etr < 4 or not idx:
            continue
        try:
            m = fac(); m.fit(X[:etr][:, idx], y[:etr])
            r2s.append(_r2(y[etr:eva], m.predict(X[etr:eva][:, idx])))
        except Exception:
            pass
    return float(np.median(r2s)) if r2s else -9.0


def worker_main(method):
    from simulation.tests._real_data_prep import _prep_full
    from sklearn.ensemble import RandomForestRegressor
    Pp, Pt, yp, yt, ylog, inv, cols = _prep_full()
    rf = lambda: RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0, n_jobs=1)
    _ylog = np.log1p(np.clip(yp, 0, None))

    if method == "CORR":
        from simulation.pipeline.feature_select_corr1se import select_features_stability
        idx = select_features_stability(Pp, _ylog, pi=0.6, epv_ratio=20, seed=42)["selected_indices"]
    elif method == "MI":
        from simulation.pipeline.feature_select_corr1se import select_features_stability
        from sklearn.feature_selection import mutual_info_regression
        def mi_fn(Xs, ys):
            return mutual_info_regression(Xs, ys, random_state=0)
        idx = select_features_stability(Pp, _ylog, pi=0.6, epv_ratio=20, seed=42,
                                        importance_fn=mi_fn, model_based_min_n=1)["selected_indices"]
    elif method == "RFE":
        from sklearn.feature_selection import RFE
        sel = RFE(RandomForestRegressor(n_estimators=80, max_depth=6, random_state=0, n_jobs=1),
                  n_features_to_select=9, step=0.2).fit(Pp, _ylog)
        idx = sorted(np.where(sel.support_)[0].tolist())
    else:
        print("RESULT_JSON null", flush=True); return

    out = {"method": method, "k": len(idx),
           "oof_r2": _oof_r2(rf, Pp, yp, idx),
           "test_r2": _r2(yt, rf().fit(Pp[:, idx], yp).predict(Pt[:, idx])),
           "sel": [cols[i] for i in idx][:8]}
    print("RESULT_JSON " + json.dumps(out), flush=True)


def parent_main():
    print("=" * 88, flush=True)
    print("Part 3: 안 해본 feature 방법 (MI/RFE) vs |corr| STABILITY (실데이터 n=242)", flush=True)
    print("=" * 88, flush=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = []
    for mth in METHODS:
        try:
            cp = subprocess.run(
                [sys.executable, "-u", "-m", "simulation.scripts._untested_feature_methods_eval", "--method", mth],
                cwd=repo, env=env, capture_output=True, text=True, timeout=PER_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"  {mth} TIMEOUT", flush=True); continue
        if cp.returncode != 0:
            print(f"  {mth} CRASH rc={cp.returncode}: {(cp.stderr or '')[-120:]}", flush=True); continue
        line = next((l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        r = json.loads(line[len("RESULT_JSON "):]) if line and line != "RESULT_JSON null" else None
        if r:
            rows.append(r)
            print(f"  {r['method']:5s} k={r['k']:>2}  OOF-R²={r['oof_r2']:+.3f}  test-R²={r['test_r2']:+.3f}  sel={r['sel']}", flush=True)
    if rows:
        print("-" * 88, flush=True)
        best = max(rows, key=lambda r: r["test_r2"])
        print(f"  test-R² 우수: {best['method']} ({best['test_r2']:+.3f})", flush=True)
        corr = next((r for r in rows if r["method"] == "CORR"), None)
        if corr:
            print(f"  → |corr| STABILITY test-R²={corr['test_r2']:+.3f}; "
                  f"MI/RFE 가 {'못 이김 = |corr| 유지 정당' if best['method']=='CORR' or abs(best['test_r2']-corr['test_r2'])<0.02 else '이김 → 재검토'}", flush=True)
    print("=" * 88, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default=None); a = ap.parse_args()
    if a.method:
        worker_main(a.method)
    else:
        parent_main()


if __name__ == "__main__":
    main()
