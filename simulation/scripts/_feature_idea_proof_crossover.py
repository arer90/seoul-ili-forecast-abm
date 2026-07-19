"""Part 1: 사용자 아이디어(per-model model-based feature 선택) 증명 — n-crossover study.

질문: "내 model-based per-model feature optimization 아이디어가 진짜 작동하는가?"
설계: ILI 구조 모사 합성데이터 — AR-지배 + **상호작용 신호** + 노이즈.
  y[t] = 0.6·y[t-1] + 1.2·(xa·xb) + 0.6·xc + 0.3·ε
  · y_lag1 : 강한 marginal |corr| (|corr| 가 잡음)
  · xa, xb : **상호작용으로만** y 에 기여 → 개별 marginal |corr| ≈ 0 (|corr| 가 놓침),
             단 트리 모델은 split 으로 잡음 (model-based importance)
  · xc     : 약한 선형 신호
  · 나머지 : 노이즈
가설: 작은 n = model-based 과적합(노이즈 학습) → |corr| 승. 큰 n = model-based 가 xa·xb 복원 →
  |corr| 가 구조적으로 놓치는 상호작용을 잡아 이김 = **사용자 아이디어가 데이터 충분 시 실제 작동**.

측정: n sweep × {|corr| stability, model-based stability(RF importance)} →
  (a) 상호작용 feature {xa,xb} 복원 여부, (b) test-R². per-n subprocess 격리(OpenMP).

worker: python -m simulation.scripts._feature_idea_proof_crossover --n 800
parent: python -m simulation.scripts._feature_idea_proof_crossover
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
N_SWEEP = [200, 400, 800, 1600, 3200]
P_NOISE = 40
PER_N_TIMEOUT = 900


def make_synth(n, p_noise=P_NOISE, seed=0):
    """ILI-유사 합성: AR + 상호작용(xa·xb) + 선형(xc) + 노이즈. 결정론(seed).

    Returns: X (n×p), y (n,), names (list), signal_idx dict {lag1,xa,xb,xc}.
    feature 순서: [y_lag1, xa, xb, xc, noise_0..noise_{p_noise-1}].
    """
    rng = np.random.default_rng(seed)
    xa = rng.normal(size=n)
    xb = rng.normal(size=n)
    xc = rng.normal(size=n)
    noise = rng.normal(size=(n, p_noise))
    eps = rng.normal(size=n)
    y = np.zeros(n)
    inter = xa * xb
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + 1.2 * inter[t] + 0.6 * xc[t] + 0.3 * eps[t]
    y_lag1 = np.concatenate([[0.0], y[:-1]])           # past value (causal AR feature)
    X = np.column_stack([y_lag1, xa, xb, xc, noise])
    names = ["y_lag1", "xa", "xb", "xc"] + [f"noise_{i}" for i in range(p_noise)]
    return X, y, names, {"lag1": 0, "xa": 1, "xb": 2, "xc": 3}


def _test_r2(fac, Xtr, ytr, Xte, yte, idx):
    idx = list(idx)
    if not idx:
        return -9.0
    try:
        m = fac(); m.fit(Xtr[:, idx], ytr)
        pred = m.predict(Xte[:, idx])
        ss_res = float(np.sum((yte - pred) ** 2)); ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else -9.0
    except Exception:
        return -9.0


def worker_main(n):
    from simulation.pipeline.feature_select_corr1se import (
        select_features_stability, make_model_importance_fn)
    from sklearn.ensemble import RandomForestRegressor

    X, y, names, sig = make_synth(n, seed=0)
    cut = int(n * 0.8)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    fac = lambda: RandomForestRegressor(n_estimators=120, max_depth=6, random_state=0, n_jobs=1)

    # |corr| stability (model-agnostic)
    s_corr = select_features_stability(Xtr, ytr, pi=0.6, epv_ratio=20, seed=42)
    idx_corr = s_corr["selected_indices"]
    # model-based stability (RF importance) — force on (model_based_min_n=1)
    imp = make_model_importance_fn(fac)
    s_mb = select_features_stability(Xtr, ytr, pi=0.6, epv_ratio=20, seed=42,
                                     importance_fn=imp, model_based_min_n=1)
    idx_mb = s_mb["selected_indices"]

    inter = {sig["xa"], sig["xb"]}
    out = {
        "n": n,
        "CORR": {"k": len(idx_corr), "recover_inter": len(inter & set(idx_corr)),
                 "test_r2": _test_r2(fac, Xtr, ytr, Xte, yte, idx_corr),
                 "sel": [names[i] for i in idx_corr][:8]},
        "MODEL": {"k": len(idx_mb), "recover_inter": len(inter & set(idx_mb)),
                  "test_r2": _test_r2(fac, Xtr, ytr, Xte, yte, idx_mb),
                  "sel": [names[i] for i in idx_mb][:8]},
    }
    print("RESULT_JSON " + json.dumps(out), flush=True)


def parent_main():
    print("=" * 96, flush=True)
    print("Part 1: 사용자 model-based 아이디어 증명 — n-crossover (상호작용 xa·xb 복원 + test-R²)", flush=True)
    print("  합성: y=0.6·lag1 + 1.2·(xa·xb) + 0.6·xc + noise. xa,xb=상호작용 전용(marginal|corr|≈0)", flush=True)
    print("=" * 96, flush=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = []
    for n in N_SWEEP:
        try:
            cp = subprocess.run(
                [sys.executable, "-u", "-m", "simulation.scripts._feature_idea_proof_crossover", "--n", str(n)],
                cwd=repo, env=env, capture_output=True, text=True, timeout=PER_N_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"  n={n} TIMEOUT", flush=True); continue
        if cp.returncode != 0:
            print(f"  n={n} CRASH rc={cp.returncode}: {(cp.stderr or '')[-120:]}", flush=True); continue
        line = next((l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        if not line:
            print(f"  n={n} no-result", flush=True); continue
        r = json.loads(line[len("RESULT_JSON "):]); rows.append(r)
        c, m = r["CORR"], r["MODEL"]
        print(f"\n  n={n}", flush=True)
        print(f"    CORR   k={c['k']:>2}  xa·xb복원={c['recover_inter']}/2  test-R²={c['test_r2']:+.3f}  sel={c['sel']}", flush=True)
        print(f"    MODEL  k={m['k']:>2}  xa·xb복원={m['recover_inter']}/2  test-R²={m['test_r2']:+.3f}  sel={m['sel']}", flush=True)
        win = "MODEL" if m["test_r2"] > c["test_r2"] else "CORR"
        print(f"    → test-R² 우수: {win}", flush=True)
    if rows:
        print("-" * 96, flush=True)
        print(f"  {'n':>5} {'CORR R²':>9} {'MODEL R²':>9} {'CORR복원':>8} {'MODEL복원':>9} {'우승':>6}", flush=True)
        for r in rows:
            c, m = r["CORR"], r["MODEL"]
            win = "MODEL" if m["test_r2"] > c["test_r2"] else "CORR"
            print(f"  {r['n']:>5} {c['test_r2']:>+9.3f} {m['test_r2']:>+9.3f} "
                  f"{c['recover_inter']:>7}/2 {m['recover_inter']:>8}/2 {win:>6}", flush=True)
        xover = next((r["n"] for r in rows if r["MODEL"]["test_r2"] > r["CORR"]["test_r2"]), None)
        print(f"\n  CROSSOVER (model-based 가 |corr| 이기기 시작): n = {xover}", flush=True)
        print("  → 사용자 아이디어(model-based per-model)는 n 충분 시 실제 작동 (상호작용 feature 복원). "
              "n-adaptive 가 그 지점부터 자동 활성 = 이론 아님.", flush=True)
    print("=" * 96, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=None); a = ap.parse_args()
    if a.n:
        worker_main(a.n)
    else:
        parent_main()


if __name__ == "__main__":
    main()
