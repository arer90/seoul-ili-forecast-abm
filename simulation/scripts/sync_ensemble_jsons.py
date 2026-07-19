"""Ensemble JSON sync — sidecar.ensemble_results → per_model_optimal/Ensemble-*.json.

문제 (G-176 후속):
  Ensemble 11 모델의 per_model_optimal JSON 이 placeholder
  (refit_test_predictions/test_metrics 부재).
  실제 ensemble 결과는 phase4_baseline_sidecar.pkl 의
  ensemble_results dict 에 saved (val_pred, test_pred, weights, AR2).

이 script 가:
  ① sidecar.ensemble_results read
  ② per_model_optimal/Ensemble-*.json 갱신 (refit_test_predictions + test_metrics)
  ③ AR(2) 보정 결과 별도 키로 saved

학술 정직성: sidecar 의 데이터를 그대로 reflect. 새 fit 없음 (재현성 보장).

사용:
    .venv/bin/python -m simulation.scripts.sync_ensemble_jsons

영향: per_model_optimal/Ensemble-*.json 만 갱신. 다른 모델 / 학습 영향 0.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)


def main() -> None:
    PMO = get_results_dir() / "per_model_optimal"
    sidecar = get_results_dir() / "phase4_baseline_sidecar.pkl"

    if not sidecar.exists():
        print(f"sidecar not found: {sidecar}")
        return

    with open(sidecar, "rb") as f:
        sc = pickle.load(f)
    ens = sc.get("ensemble_results", {})

    cache = pl.read_parquet("simulation/cache/feature_cache.parquet")
    y = cache.select("ili_rate").to_numpy().flatten()
    y_test = y[269: 269 + 68]
    ss = float(((y_test - y_test.mean()) ** 2).sum())

    print(f"=== Ensemble JSON sync (sidecar → per_model_optimal) ===")
    synced = []
    skipped = []

    for nm, sub in ens.items():
        if not isinstance(sub, dict):
            skipped.append((nm, "sidecar entry not dict"))
            continue
        j = PMO / f"{nm}.json"
        if not j.exists():
            skipped.append((nm, "per_model_optimal JSON not exist"))
            continue
        test_pred = sub.get("test_pred")
        if test_pred is None:
            skipped.append((nm, "test_pred is None"))
            continue
        try:
            pred = np.asarray(test_pred, dtype=float).flatten()
        except Exception as e:
            skipped.append((nm, f"pred convert fail: {e}"))
            continue
        if len(pred) != 68:
            skipped.append((nm, f"pred len={len(pred)}"))
            continue

        r2 = 1.0 - float(((y_test - pred) ** 2).sum()) / ss
        mape = float(np.mean(np.abs((y_test - pred) / np.maximum(y_test, 0.01))) * 100)
        wis = float(np.mean(np.abs(y_test - pred)))

        # 사용자 정책 (sprint 2026-05-06): R²<0 절대 차단, R²<0.7 baseline 영역
        if r2 < 0:
            skipped.append((nm, f"R²={r2:+.4f} NEGATIVE — 사용자 정책 차단"))
            continue
        if r2 < 0.7:
            skipped.append((nm, f"R²={r2:+.4f} <0.7 baseline 영역 차단"))
            continue

        # Update saved JSON
        d = json.loads(j.read_text())
        d["refit_test_predictions"] = pred.tolist()

        # test_metrics merge
        prior_tm = d.get("test_metrics", {}) if isinstance(d.get("test_metrics"), dict) else {}
        sidecar_tm = sub.get("test_metrics", {}) if isinstance(sub.get("test_metrics"), dict) else {}
        new_tm = {**sidecar_tm, **prior_tm}  # prior takes precedence if conflict
        new_tm.setdefault("r2", r2)
        new_tm.setdefault("mape", mape)
        new_tm.setdefault("wis", wis)
        new_tm.setdefault("mae", float(np.mean(np.abs(y_test - pred))))
        d["test_metrics"] = new_tm

        # best_config — add ensemble-specific
        bc = d.get("best_config", {}) or {}
        if "weights" not in bc and "weights" in sub:
            try:
                bc["weights"] = sub["weights"] if isinstance(sub["weights"], (list, dict)) else None
            except Exception:
                pass
        bc.setdefault("method", nm.replace("Ensemble-", ""))
        d["best_config"] = bc

        # AR(2) post-correction (sidecar saved)
        if "test_pred_ar" in sub:
            try:
                ar2 = np.asarray(sub["test_pred_ar"], dtype=float).flatten()
                if len(ar2) == 68:
                    d["refit_test_ar2_predictions"] = ar2.tolist()
                    r2_ar2 = 1.0 - float(((y_test - ar2) ** 2).sum()) / ss
                    d["test_metrics_ar2"] = sub.get("test_metrics_ar", {}) or {"r2": r2_ar2}
            except Exception:
                pass

        # Provenance
        d["synced_from_sidecar"] = {
            "date": "2026-05-06",
            "source": "simulation/results/phase4_baseline_sidecar.pkl::ensemble_results",
            "script": "simulation/scripts/sync_ensemble_jsons.py",
        }

        j.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        synced.append((nm, r2, mape, wis))

    # Summary
    print(f"\n## Synced ({len(synced)} / {len(ens)})")
    for nm, r2, mape, wis in sorted(synced, key=lambda x: -x[1]):
        ar2_str = ""
        d = json.loads((PMO / f"{nm}.json").read_text())
        if "refit_test_ar2_predictions" in d:
            arr = np.asarray(d["refit_test_ar2_predictions"], dtype=float)
            r2_ar = 1.0 - float(((y_test - arr) ** 2).sum()) / ss
            ar2_str = f"  AR2_R²={r2_ar:+.4f}"
        print(f"  {nm:25s}: R²={r2:+.4f} MAPE={mape:>5.1f}% WIS={wis:.3f}{ar2_str}")

    if skipped:
        print(f"\n## Skipped ({len(skipped)})")
        for nm, reason in skipped:
            print(f"  {nm}: {reason}")


if __name__ == "__main__":
    main()
