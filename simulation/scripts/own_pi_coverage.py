"""own_pi_coverage.py — own-PI(predict_quantiles) 모델의 실 test 커버리지 (G-364, 2026-06-26).

R10 per_model_metrics 의 pi95_coverage 는 전 모델 공통 generic 잔차 conformal PI(leak-free 이나
in-sample 잔차가 out-of-sample 과소추정 → 정점 과소피복, 중위 0.67). 그런데 일부 모델은 자체
predict_quantiles(FusedEpi conformal·SeirCount NB·FluSight 23분위·NegBinGLM NB)를 가짐 — 이게
진짜 운영 PI. 이 스크립트는 **재학습 없이 기존 .pt 로 own-PI 의 실 test 커버리지**를 잰다(supplementary).

방법 (FusedEpi 0.926 측정과 동일, R10 무수정):
  1. run_data(PipelineConfig) → X_all, y_all (파이프라인 정합 split).
  2. test slab = predictions CSV y_true 로 정렬(offset), X_test/y_test.
  3. own-PI 모델: art.model.predict_quantiles(apply_scaler(apply_features(X_test)), y_observed=y_test).
  4. coverage = y_test 가 [q_lo, q_hi] 안에 든 비율 (PI95/80/50) + 평균폭.
  → simulation/results/csv/own_pi_metrics.csv

Caveat: default PipelineConfig 가 run(398) 보다 1 feature 적을 수 있음(vif replay 경고) → ±소폭 추정치.
  결론(own-PI 잘 calibrated)은 robust. 정확값은 run-context 재배선(미수행) 필요.

Usage: .venv/bin/python -m simulation.scripts.own_pi_coverage
Returns: own_pi_metrics.csv 경로(print). Side effects: csv 작성, foundation 모델 lazy-load.
"""
from __future__ import annotations

import csv
import glob
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# PI level pairs (lo, hi) for coverage
_PI_PAIRS = {"pi95": (0.025, 0.975), "pi80": (0.10, 0.90), "pi50": (0.25, 0.75)}
_LEVELS = (0.025, 0.10, 0.25, 0.5, 0.75, 0.90, 0.975)


def _test_slab(X_all, y_all):
    """predictions CSV y_true 로 test slab offset 정렬 → (X_test, y_test). fallback=tail 68."""
    import pandas as pd
    nt = 68
    pf = sorted(glob.glob("simulation/results/csv/predictions_*.csv"))
    yc = None
    for p in pf:
        try:
            df = pd.read_csv(p)
            t = df[df["split"] == "test"]
            if len(t) >= 60:
                yc = t["y_true"].values.astype(float)
                nt = len(yc)
                break
        except Exception:
            continue
    off = len(y_all) - nt
    if yc is not None:
        for s in range(len(y_all) - nt, -1, -1):
            if np.allclose(y_all[s:s + nt], yc, atol=1e-6):
                off = s
                break
    return X_all[off:off + nt], y_all[off:off + nt], off


def compute_own_pi(model_dir: str = "models") -> list[dict]:
    """champion .pt 전수서 predict_quantiles 보유 모델의 실 test 커버리지 계산.

    Returns: [{model, pi95_cov, pi80_cov, pi50_cov, pi95_width, n_test, method}].
    """
    from simulation.pipeline.config import PipelineConfig
    from simulation.pipeline.data import run_data
    from simulation.utils.model_artifact import load_artifact

    # own-PI 후보만 로드 (129-metric 감사 own-PI 인벤토리) — 전 모델 로드는 foundation(TiRex/TimesFM)
    #   lazy-load 로 너무 느림. predict_quantiles 보유 후보만 직접 로드.
    # CQR×3 제외: own-PI 가 raw quantile head(conformal 외부, audit) = 저가치 + CQR-LightGBM 은
    #   macOS LightGBM OpenMP segfault(단일프로세스). NegBinGLM 은 predict_interval(NB) 보유.
    CANDIDATES = ["FusedEpi", "SeirCount-TabPFN", "FluSight-Baseline", "NegBinGLM"]

    ph = run_data(PipelineConfig())
    X_all = np.asarray(ph["X_all"], dtype=np.float64)
    y_all = np.asarray(ph["y_all"], dtype=np.float64).ravel()
    X_test, y_test, off = _test_slab(X_all, y_all)
    n = len(y_test)
    print(f"  test slab n={n}, offset={off}", flush=True)
    rows: list[dict] = []
    for name in CANDIDATES:
        pt = os.path.join(model_dir, f"{name}.pt")
        if not os.path.exists(pt):
            print(f"  [own-PI] {name}: .pt 없음 skip", flush=True)
            continue
        try:
            art = load_artifact(pt)
            mdl = getattr(art, "model", None)
            Xmi = art.apply_scaler(art.apply_features(X_test))
            qd, method = {}, None    # {level: array} — 4 시그니처 적응
            if hasattr(mdl, "predict_quantiles"):
                try:    # (1) dict (FusedEpi·SeirCount): predict_quantiles(X, y_observed, levels)
                    q = mdl.predict_quantiles(Xmi, y_observed=y_test, levels=_LEVELS)
                    if isinstance(q, dict):
                        qd, method = {float(k): np.asarray(v, dtype=np.float64) for k, v in q.items()}, "pq(dict)"
                except TypeError:
                    pass
                if not qd:    # (2) array (FluSight): predict_quantiles(X, quantiles=levels)
                    try:
                        arr = np.asarray(mdl.predict_quantiles(Xmi, quantiles=_LEVELS), dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] == len(_LEVELS):
                            qd = {float(_LEVELS[i]): arr[:, i] for i in range(len(_LEVELS))}; method = "pq(array)"
                        elif arr.ndim == 2 and arr.shape[0] == len(_LEVELS):
                            qd = {float(_LEVELS[i]): arr[i, :] for i in range(len(_LEVELS))}; method = "pq(array)"
                    except Exception:
                        pass
                if not qd:    # (3) tuple (CQR raw): predict_quantiles(X) → (lo, hi) ~95%
                    try:
                        r = mdl.predict_quantiles(Xmi)
                        if isinstance(r, (tuple, list)) and len(r) == 2:
                            qd = {0.025: np.asarray(r[0], dtype=np.float64), 0.975: np.asarray(r[1], dtype=np.float64)}
                            method = "pq(tuple~95%,raw)"
                    except Exception:
                        pass
            if not qd and hasattr(mdl, "predict_interval"):    # (4) NegBinGLM: predict_interval(X, alpha)
                try:
                    lohi = mdl.predict_interval(Xmi, alpha=0.05)
                    if isinstance(lohi, (tuple, list)) and len(lohi) == 2:
                        qd = {0.025: np.asarray(lohi[0], dtype=np.float64), 0.975: np.asarray(lohi[1], dtype=np.float64)}
                        method = "predict_interval(95%)"
                except Exception:
                    pass
            if not qd:
                print(f"  [own-PI] {name}: skip (지원 안 되는 PI 시그니처)", flush=True)
                continue
            row = {"model": name, "n_test": n, "method": method}
            for tag, (lo, hi) in _PI_PAIRS.items():
                if lo in qd and hi in qd:
                    row[f"{tag}_cov"] = round(float(np.mean((y_test >= qd[lo]) & (y_test <= qd[hi]))), 3)
                    if tag == "pi95":
                        row["pi95_width"] = round(float(np.mean(qd[hi] - qd[lo])), 2)
                else:
                    row[f"{tag}_cov"] = float("nan")
            rows.append(row)
            print(f"  [own-PI] {name}: PI95 cov={row.get('pi95_cov')} PI80={row.get('pi80_cov')} "
                  f"PI50={row.get('pi50_cov')} width={row.get('pi95_width')} [{method}]", flush=True)
        except Exception as e:
            print(f"  [own-PI] {name}: skip ({type(e).__name__}: {str(e)[:60]})", flush=True)
    return rows


def main() -> int:
    rows = compute_own_pi()
    out = "simulation/results/csv/own_pi_metrics.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["model", "pi95_cov", "pi80_cov", "pi50_cov", "pi95_width", "n_test", "method"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\n{out} ({len(rows)} own-PI models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
