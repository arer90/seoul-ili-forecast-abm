"""
Seoul-only forecast pipeline (standalone module, 서울 한정).

⚠️ CIRCULAR REASONING REJECTED (2026-05-27, post-§6 audit):
==========================================================
본 모듈의 S1 + S3 시나리오는 mathematical circular reasoning 으로 REJECT 됨.
사용된 ground truth 합성이 KR national 의 constant scaling 으로 환원되어
새 정보 0 — paper main result 부적합.

세부 reject 이유:
- S1: Seoul_ILI[t] = KR_national_ILI[t] × 0.1833 (constant) → R²(cY, cŶ) = R²(Y, Ŷ)
  즉 §5.1.1 의 KR national 평가표를 그대로 가져와서 MAE/RMSE 만 0.1833 곱하면 동일 결과
- S3: gu_ILI[g,t] = KR_seasonal[t] × const[g] → 모든 gu 시계열이 같은 shape 의 scaled copy
  → "spatial heterogeneity" 가 mathematically forced artifact, 아닌 evidence

본 모듈의 smoke 결과 (BayesianRidge R²=0.816, 25 자치구 평균 R²=0.806) 는
sentinel mobility feature 가 fit 잘 되는 정도 만 측정 — Seoul-specific ILI dynamics 아님.

CURRENT STATUS:
- §6 of 2_REPORT_RESULTS.html 에 limitations 로 명시 disclosure
- Paper main 에서 제외 (Seoul direct gu-level ILI 확보 시 재시도 가능)
- Future work: KDCA NEDSS 시·군·구 정식 공개 대기 (현 2024 시범공개분만)

DO NOT USE for paper main result. Smoke output 만 educational reference.
==========================================================

Scenarios (REJECTED):
- S1: Seoul as single forecast unit (KR national sentinel × 인구비례 = ground truth)
- S3: 25 자치구 spatial heterogeneity (HIRA Seoul annual × gu 인구비례 + KR seasonal pattern)

USAGE
-----
.venv/bin/python -m simulation.pipeline.seoul_gu --scenario S1 --models ElasticNet,SVR-Linear,RandomForest
.venv/bin/python -m simulation.pipeline.seoul_gu --scenario S3 --models all

OUTPUT
------
simulation/results/phase15_seoul/
├── S1_seoul_metrics.csv        # 1 cell × 53 model × 118 metric
├── S1_seoul_predictions/*.csv  # per-model observed-vs-predicted
├── S3_25gu_metrics.csv         # 25 gu × 53 model × 118 metric
└── S3_25gu_predictions/*.csv

NOTE
----
This is the SKELETON / scenario-defining module. Full training requires:
1. Wire to existing simulation.models registry (53 active CATEGORY_MODELS)
2. Reuse simulation.pipeline.per_model_eval for metric computation
3. Use simulation.pipeline.preproc_optuna_hierarchical for Y transforms
4. Optuna HP search via simulation.pipeline.per_model_optimize

For SMOKE test (3 baseline models, no Optuna), use --smoke flag.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEOUL_DIR = REPO / "simulation" / "results" / "seoul_only"
OUT_DIR = REPO / "simulation" / "results" / "phase15_seoul"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 25 자치구 인구 (천명, KOSIS 2023)
GU_POPULATION = {
    "강남구": 540, "강동구": 460, "강북구": 290, "강서구": 580, "관악구": 500,
    "광진구": 350, "구로구": 410, "금천구": 230, "노원구": 510, "도봉구": 310,
    "동대문구": 350, "동작구": 380, "마포구": 370, "서대문구": 310, "서초구": 410,
    "성동구": 290, "성북구": 430, "송파구": 670, "양천구": 440, "영등포구": 390,
    "용산구": 230, "은평구": 470, "종로구": 150, "중구": 130, "중랑구": 390,
}
SEOUL_TOTAL = sum(GU_POPULATION.values())  # 9,400 (thousand)


# ── Data loading ──
def load_proxy_weekly(option: str = "A") -> dict:
    """Load Seoul proxy ground truth (S1).

    Args:
        option: "A" = KR national × Seoul share, "B" = HIRA-based.
    Returns:
        {(season_start, week_seq): ili_value}
    """
    fname = f"seoul_proxy_weekly_option_{option.lower()}.csv"
    path = SEOUL_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate from the DB first:\n"
            f"  .venv/bin/python -m simulation.scripts.gen_seoul_smoke_csvs")
    out = {}
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            key = (int(r["season_start"]), int(r["week_seq"]))
            if option == "A":
                out[key] = float(r["seoul_proxy_ili_optA"])
            else:
                out[key] = float(r["seoul_proxy_ili_optB_patients"])
    return out


def load_25gu_features() -> dict:
    """Load 25 자치구 × weekly mobility features.

    Returns:
        {(gu, year, week_no): {daytime_pop, nighttime_pop, ...}}
    """
    path = SEOUL_DIR / "seoul_25gu_weekly_features.csv"
    out = {}
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            key = (r["gu_nm"], int(r["year"]), int(r["week_no"]))
            out[key] = {
                "daytime_pop": float(r["daytime_pop_mean"]),
                "nighttime_pop": float(r["nighttime_pop_mean"]),
                "avg_pop": float(r["avg_pop_mean"]),
                "child_share": float(r["child_share_mean"]),
                "senior_share": float(r["senior_share_mean"]),
                "day_night_ratio": float(r["day_night_ratio"]),
            }
    return out


def build_s1_dataset() -> tuple[np.ndarray, np.ndarray, list]:
    """S1: Seoul single-unit dataset.

    Features: Seoul aggregate mobility (sum across 25 gu) + lag-1 ILI + week
    Target:   Seoul proxy ILI (Option A, KR national × 0.1833)
    """
    proxy = load_proxy_weekly("A")
    feats_25gu = load_25gu_features()

    # Aggregate features across 25 gu
    seoul_agg = defaultdict(lambda: {"daytime_pop": 0, "nighttime_pop": 0, "child_share": 0, "senior_share": 0, "n_gu": 0})
    for (gu, y, w), v in feats_25gu.items():
        key = (y, w)
        seoul_agg[key]["daytime_pop"] += v["daytime_pop"]
        seoul_agg[key]["nighttime_pop"] += v["nighttime_pop"]
        seoul_agg[key]["child_share"] += v["child_share"]
        seoul_agg[key]["senior_share"] += v["senior_share"]
        seoul_agg[key]["n_gu"] += 1
    # Normalize age shares (avg across 25 gu)
    for k, v in seoul_agg.items():
        if v["n_gu"] > 0:
            v["child_share"] /= v["n_gu"]
            v["senior_share"] /= v["n_gu"]

    # Match proxy with features by (year, iso_week)
    # Note: sentinel_influenza uses season_start + week_seq (epi week)
    # For simple smoke, use ordered tuples
    sorted_proxy = sorted(proxy.items())
    X_rows, y_vals, idx_labels = [], [], []
    prev_ili = None
    for (season, week), ili in sorted_proxy:
        # Calendar year = season (Korea ILI uses calendar year-week)
        feat_key = (season, week)
        if feat_key not in seoul_agg:
            prev_ili = ili
            continue
        v = seoul_agg[feat_key]
        if v["n_gu"] < 20:  # incomplete week
            prev_ili = ili
            continue
        # Features: lag-1 ILI, week_of_year, sin/cos, daytime/nighttime, age shares
        if prev_ili is None:
            prev_ili = ili
            continue
        X_rows.append([
            prev_ili,
            week,
            math.sin(2 * math.pi * week / 52),
            math.cos(2 * math.pi * week / 52),
            v["daytime_pop"] / 1e6,
            v["nighttime_pop"] / 1e6,
            v["child_share"],
            v["senior_share"],
        ])
        y_vals.append(ili)
        idx_labels.append(f"{season}W{week:02d}")
        prev_ili = ili

    X = np.array(X_rows)
    y = np.array(y_vals)
    return X, y, idx_labels


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Core metrics (eLife 4 + extended)."""
    n = len(y_true)
    eps = 1e-9
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    sq = diff ** 2
    nonzero = np.abs(y_true) > eps
    return {
        "n": n,
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(sq))),
        "mape": float(np.mean(abs_diff[nonzero] / np.abs(y_true[nonzero])) * 100) if nonzero.any() else float("nan"),
        "smape": float(np.mean(2 * abs_diff / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100),
        "r2": float(1 - np.sum(sq) / max(np.sum((y_true - y_true.mean())**2), eps)),
        "bias": float(np.mean(diff)),
        "max_abs_err": float(np.max(abs_diff)),
    }


# ── Models (smoke: 3 baseline, no Optuna) ──
def fit_predict_models(X_tr, y_tr, X_te, model_names):
    """Fit and predict for given model names. Returns {name: (y_pred, metrics)}."""
    from sklearn.linear_model import ElasticNet, BayesianRidge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    available = {
        "ElasticNet":    ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        "BayesianRidge": BayesianRidge(),
        "RandomForest":  RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=2, random_state=42),
    }
    out = {}
    for name in model_names:
        if name in available:
            model = available[name]
            model.fit(X_tr_s, y_tr)
            out[name] = model.predict(X_te_s)
        elif name in ("NegBinGLM", "NegBinGLM-V7"):
            # epi count-GLM via the project's BaseForecaster (raw X — own scaling)
            from simulation.models.epi_models import NegBinGLMForecaster
            pred = NegBinGLMForecaster(topk=20).fit_predict(X_tr, y_tr, X_te, name=name)
            out[name] = np.asarray(pred, dtype=float)
        else:
            print(f"  ⚠️ Model not available in smoke mode: {name}")
    return out


def run_s1_smoke(model_names: list[str], hold_out_weeks: int = 8) -> None:
    """S1 smoke test: 3 baseline models × Seoul single unit."""
    print(f"\n{'='*60}\nS1 smoke — Seoul single unit ({len(model_names)} models)\n{'='*60}")
    X, y, idx = build_s1_dataset()
    print(f"Dataset: X={X.shape}, y={y.shape}")
    print(f"  Period: {idx[0]} → {idx[-1]}")

    n = len(y)
    n_train = n - hold_out_weeks
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]
    print(f"  Train: {n_train} weeks, Test (hold-out): {hold_out_weeks} weeks")

    preds = fit_predict_models(X_tr, y_tr, X_te, model_names)

    # Save metrics
    rows = []
    for name, y_pred in preds.items():
        m = metrics(y_te, y_pred)
        rows.append({"scenario": "S1", "unit": "Seoul", "model": name, **m})
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")

    with open(OUT_DIR / "S1_seoul_metrics.csv", "w", encoding="utf-8") as f:
        if rows:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"\n✓ S1 metrics → {OUT_DIR}/S1_seoul_metrics.csv")

    # Save predictions
    pred_dir = OUT_DIR / "S1_seoul_predictions"
    pred_dir.mkdir(exist_ok=True)
    test_idx = idx[n_train:]
    for name, y_pred in preds.items():
        with open(pred_dir / f"pred_{name}.csv", "w", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["week_label", "y_obs", "y_pred"])
            for i, lbl in enumerate(test_idx):
                wr.writerow([lbl, f"{y_te[i]:.6f}", f"{y_pred[i]:.6f}"])
    print(f"✓ S1 predictions → {pred_dir}")


# ── Main ──
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["S1", "S3"], default="S1")
    p.add_argument("--models", default="ElasticNet,BayesianRidge,RandomForest")
    p.add_argument("--smoke", action="store_true", help="Smoke test (3 baseline only, no Optuna)")
    p.add_argument("--hold-out", type=int, default=8)
    args = p.parse_args()

    if args.scenario == "S1":
        run_s1_smoke(args.models.split(","), hold_out_weeks=args.hold_out)
    elif args.scenario == "S3":
        print("S3 (25 자치구) requires full pipeline integration with per_model_optimize (R9) + per_model_eval (R10) — defer to background launch")
        print("See docs/SEOUL_ONLY_ANALYSIS_PLAN_20260527.md §5.3 for command.")
        sys.exit(2)


if __name__ == "__main__":
    main()
