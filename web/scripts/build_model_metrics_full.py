#!/usr/bin/env python3
"""web 모델 전체 평가지표(129) 빌더 — 재학습 없이 기존 예측 CSV → model-metrics-full.json.

사용자 요구: web 까지 129개 평가지표 다 반영 + 기본 표시 = {r2, wis, rmse, mae, auc-roc, c-index}.
방법: ili-forecast-models.json 의 표시 모델별 predictions_<name>.csv(split,idx,y_true,y_pred, 기존
산출 — champion frozen 무관) 의 test split 을 SSOT evaluate_predictions_full(129키)에 통과시켜
roc_auc·c_index 포함 129지표를 계산해 산출. 재학습 0(예측은 이미 저장됨).

WIS = evaluator 내부(Gaussian σ) 값 — 운영 conformal WIS(ili-forecast.json)와 약간 다를 수 있음(둘 다 표기).
Run: .venv/bin/python web/scripts/build_model_metrics_full.py
"""
from __future__ import annotations

import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
AGG = ROOT / "web" / "public" / "aggregates"
CSV_DIR = ROOT / "simulation" / "results" / "csv"

from simulation.pipeline.phase_evaluator import evaluate_predictions_full  # noqa: E402

# 기본 표시 6 (사용자 명시). c_index/roc_auc 는 §4.2b paper 지표.
DEFAULT_VISIBLE = ["r2", "wis", "rmse", "mae", "roc_auc", "c_index"]
LABELS = {"r2": "R²", "wis": "WIS", "rmse": "RMSE", "mae": "MAE",
          "roc_auc": "AUC-ROC", "c_index": "C-index", "mape": "MAPE", "alert_f1": "Alert-F1"}
# 표시 시 lower=better 인 지표(색상/정렬 방향)
LOWER_BETTER = {"wis", "rmse", "mae", "mape", "mse", "smape", "crps_gaussian", "log_wis"}


def load_test_split(name: str):
    """predictions_<name>.csv 의 test split → (y_true, y_pred) np arrays. 없으면 None."""
    path = CSV_DIR / f"predictions_{name}.csv"
    if not path.is_file():
        return None
    yt, yp = [], []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test":
                try:
                    yt.append(float(row["y_true"])); yp.append(float(row["y_pred"]))
                except (ValueError, KeyError, TypeError):
                    continue
    if len(yt) < 4:
        return None
    return np.asarray(yt), np.asarray(yp)


def metrics_for(name: str) -> dict | None:
    """모델 1개 → 129지표 dict (유한값만, NaN/inf → None)."""
    sp = load_test_split(name)
    if sp is None:
        return None
    res = evaluate_predictions_full(sp[0], sp[1])
    out = {}
    for k, v in res.items():
        try:
            fv = float(v)
            out[k] = round(fv, 5) if np.isfinite(fv) else None
        except (TypeError, ValueError):
            out[k] = v if isinstance(v, (str, bool, int)) else None
    return out


def main() -> int:
    fm_path = AGG / "ili-forecast-models.json"
    if not fm_path.is_file():
        print("  ✗ ili-forecast-models.json 없음"); return 1
    names = [m["name"] for m in json.loads(fm_path.read_text(encoding="utf-8")).get("models", [])]

    models, n_test = {}, 0
    keyset = None
    for nm in names:
        m = metrics_for(nm)
        if m is None:
            print(f"  ⏸ {nm}: 예측 CSV 없음 — skip")
            continue
        models[nm] = m
        keyset = keyset or list(m.keys())
        sp = load_test_split(nm); n_test = len(sp[0]) if sp else n_test

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "evaluate_predictions_full(129키) on predictions_<name>.csv test split — 재학습 없음",
        "n_test": n_test, "n_metrics": len(keyset or []),
        "metric_keys": keyset or [],
        "default_visible": DEFAULT_VISIBLE,
        "labels": LABELS,
        "lower_better": sorted(LOWER_BETTER),
        "note": "전체 129 평가지표(roc_auc=AUC-ROC, c_index=C-index 포함). WIS=evaluator 내부 Gaussian "
                "(운영 conformal WIS 는 ili-forecast.json). champion frozen 무관 — 기존 예측 재평가.",
        "models": models,
    }
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "model-metrics-full.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 전체 평가지표(129) → model-metrics-full.json ===")
    print(f"  {len(models)} 모델 × {len(keyset or [])} 지표 (test n={n_test})")
    print(f"  기본 표시 6: {DEFAULT_VISIBLE}")
    for nm, m in list(models.items())[:3]:
        print(f"  {nm:16} R²={m.get('r2')} WIS={m.get('wis')} AUC={m.get('roc_auc')} C={m.get('c_index')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
