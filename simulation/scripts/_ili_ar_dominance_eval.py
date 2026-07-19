"""Part 2: ILI 가 AR-지배적이고 이용 가능한 상호작용 구조가 없음을 실데이터로 확인.

Part 1 결론: model-based 는 비선형/상호작용 구조가 있을 때 |corr| 를 이긴다. → ILI 에 그 구조가
없다면 |corr| 가 ILI 에 맞는 선택. 이 스크립트가 그 가설을 실데이터(n=242)로 검정:

  (1) AR-지배도: R²(ili_rate_lag1 단독) vs R²(전체 feature). 비슷하면 AR-지배.
  (2) 상호작용 부재: RF(상호작용 포착) vs Ridge(선형) OOF-R². RF 가 Ridge 를 크게 못 이기면
      이용 가능한 비선형/상호작용 구조 없음 → model-based 가 찾을 게 없음 → |corr| 충분.
  (3) 선택 가치: |corr| stability subset vs 전체 feature (test-R²). subset 이 낫거나 동등하면 선택 정당.

worker: python -m simulation.scripts._ili_ar_dominance_eval --task <ar|interact|selection>
parent: python -m simulation.scripts._ili_ar_dominance_eval
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
PER_TASK_TIMEOUT = 900


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


def worker_main(task):
    from simulation.tests._real_data_prep import _prep_full
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    Pp, Pt, yp, yt, ylog, inv, cols = _prep_full()
    ridge = lambda: Ridge(alpha=1.0)
    rf = lambda: RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0, n_jobs=1)

    # lag1 feature index
    lag_idx = next((i for i, c in enumerate(cols) if c == "ili_rate_lag1"), None)
    out = {"task": task}

    if task == "ar":
        # R²(lag1 단독) vs R²(전체) — OOF + test, RF
        if lag_idx is None:
            out["error"] = "ili_rate_lag1 없음"
        else:
            allidx = list(range(Pp.shape[1]))
            out["lag1_only_oof_r2"] = _oof_r2(rf, Pp, yp, [lag_idx])
            out["full_oof_r2"] = _oof_r2(rf, Pp, yp, allidx)
            # test
            m1 = rf(); m1.fit(Pp[:, [lag_idx]], yp); out["lag1_only_test_r2"] = _r2(yt, m1.predict(Pt[:, [lag_idx]]))
            mf = rf(); mf.fit(Pp, yp); out["full_test_r2"] = _r2(yt, mf.predict(Pt))
    elif task == "interact":
        # RF(상호작용) vs Ridge(선형) — 같은 |corr| top-k subset 위에서 OOF-R²
        from simulation.scripts._ili_ar_dominance_eval import _abscorr_top
        topk = _abscorr_top(Pp, yp, 12)
        out["ridge_oof_r2"] = _oof_r2(ridge, Pp, yp, topk)
        out["rf_oof_r2"] = _oof_r2(rf, Pp, yp, topk)
    elif task == "selection":
        from simulation.pipeline.feature_select_corr1se import select_features_stability
        _ylog = np.log1p(np.clip(yp, 0, None))
        sel = select_features_stability(Pp, _ylog, pi=0.6, epv_ratio=20, seed=42)["selected_indices"]
        allidx = list(range(Pp.shape[1]))
        mfull = rf(); mfull.fit(Pp, yp); out["full_test_r2"] = _r2(yt, mfull.predict(Pt))
        msel = rf(); msel.fit(Pp[:, sel], yp); out["stability_test_r2"] = _r2(yt, msel.predict(Pt[:, sel]))
        out["stability_k"] = len(sel); out["full_k"] = len(allidx)
    print("RESULT_JSON " + json.dumps(out), flush=True)


def _abscorr_top(X, y, k):
    y = np.asarray(y, float).ravel()
    sc = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 1e-9 and np.std(y) > 1e-9 else 0.0
                   for j in range(X.shape[1])], float)
    sc[~np.isfinite(sc)] = 0.0
    return sorted(np.argsort(sc)[::-1][:k].tolist())


def parent_main():
    print("=" * 92, flush=True)
    print("Part 2: ILI AR-지배도 + 상호작용 부재 (실데이터 n=242) → |corr| 충분성 검정", flush=True)
    print("=" * 92, flush=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = {}
    for task in ["ar", "interact", "selection"]:
        try:
            cp = subprocess.run(
                [sys.executable, "-u", "-m", "simulation.scripts._ili_ar_dominance_eval", "--task", task],
                cwd=repo, env=env, capture_output=True, text=True, timeout=PER_TASK_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"  {task} TIMEOUT", flush=True); continue
        if cp.returncode != 0:
            print(f"  {task} CRASH rc={cp.returncode}: {(cp.stderr or '')[-120:]}", flush=True); continue
        line = next((l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        if line:
            res[task] = json.loads(line[len("RESULT_JSON "):])
    print("", flush=True)
    if "ar" in res:
        a = res["ar"]
        print(f"  (1) AR-지배도: lag1-only OOF-R²={a.get('lag1_only_oof_r2'):+.3f} vs 전체 OOF-R²={a.get('full_oof_r2'):+.3f}", flush=True)
        print(f"       (test) lag1-only={a.get('lag1_only_test_r2'):+.3f} vs 전체={a.get('full_test_r2'):+.3f}", flush=True)
        gain = a.get('full_oof_r2', 0) - a.get('lag1_only_oof_r2', 0)
        print(f"       → 전체 feature 추가 이득(OOF) = {gain:+.3f} ({'작음=AR지배' if gain < 0.1 else '있음'})", flush=True)
    if "interact" in res:
        i = res["interact"]
        d = i.get('rf_oof_r2', 0) - i.get('ridge_oof_r2', 0)
        print(f"  (2) 상호작용: RF OOF-R²={i.get('rf_oof_r2'):+.3f} vs Ridge(선형)={i.get('ridge_oof_r2'):+.3f}  Δ={d:+.3f}", flush=True)
        print(f"       → RF가 선형 대비 {'크게 못 이김 = 이용가능 상호작용 없음 → model-based 찾을것 없음' if d < 0.05 else '이김(상호작용 존재)'}", flush=True)
    if "selection" in res:
        s = res["selection"]
        print(f"  (3) 선택 가치: |corr|stability(k={s.get('stability_k')}) test-R²={s.get('stability_test_r2'):+.3f} "
              f"vs 전체(k={s.get('full_k')}) test-R²={s.get('full_test_r2'):+.3f}", flush=True)
        print(f"       → 선택이 {'동등/우수 = 정당' if s.get('stability_test_r2',-9) >= s.get('full_test_r2',-9) - 0.02 else '손해'}", flush=True)
    print("\n  종합: ILI 가 AR-지배 + 상호작용 부재면 → |corr| 가 ILI 에 맞음 (model-based 찾을 구조 없음). "
          "사용자 아이디어는 구조 있는 데이터에서 유효(Part 1).", flush=True)
    print("=" * 92, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default=None); a = ap.parse_args()
    if a.task:
        worker_main(a.task)
    else:
        parent_main()


if __name__ == "__main__":
    main()
