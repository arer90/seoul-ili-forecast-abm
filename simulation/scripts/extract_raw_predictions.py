"""Post-hoc α-blend decomposition — raw prediction extraction.

Cluster 1 (17 modern DL/TS 모델) 의 prediction 이 corr ≥ 0.999 — blend 수렴.
이 script 가 saved `refit_test_predictions = α·raw + (1−α)·ref` 가정 하에
raw prediction 을 별도 저장. reference estimator = ARIMA prediction.

학술 정직성: paper 에 raw 와 blended 둘 다 보고 (Bühlmann 2018 anchor regression).

사용:
    .venv/bin/python -m simulation.scripts.extract_raw_predictions

출력:
    simulation/results/raw_predictions_blend_decomp/{name}.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)


# Cluster 1 — pair correlation ≥ 0.99 인 17 모델 (sprint 2026-05-06 진단)
CLUSTER1 = [
    "ARIMA", "SARIMA", "SARIMAX",
    "TimesNet", "Mamba", "TFT", "PatchTST",
    "N-BEATS", "TinyMLP", "iTransformer",
    "GE-DNN", "TCN-Optuna", "TabularDNN",
    "TiDE", "DNN-Optuna", "TCN", "DNN",
]


def main() -> None:
    PMO = get_results_dir() / "per_model_optimal"
    RAW_DIR = get_results_dir() / "raw_predictions_blend_decomp"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # reference estimator = ARIMA refit_test_predictions
    arima_j = PMO / "ARIMA.json"
    arima = json.loads(arima_j.read_text())
    arima_pred = np.asarray(arima["refit_test_predictions"], dtype=float)
    n_test = len(arima_pred)

    # y_test for diagnostics
    cache = pl.read_parquet("simulation/cache/feature_cache.parquet")
    y = cache.select("ili_rate").to_numpy().flatten()
    y_test = y[269: 269 + n_test]
    ss = float(((y_test - y_test.mean()) ** 2).sum())

    summary: list[dict] = []
    for nm in CLUSTER1:
        j = PMO / f"{nm}.json"
        if not j.exists():
            continue
        d = json.loads(j.read_text())
        pred = np.asarray(d.get("refit_test_predictions", []), dtype=float)
        if len(pred) != n_test:
            continue
        bc = d.get("best_config", {})
        # persisted JSON key = "alpha_anchor" (옛 α-blend 학습 산출) — 변경 불가
        alpha = bc.get("alpha_anchor")

        if alpha is None or alpha < 0.001:
            raw = pred.copy()
            decomp_note = "α-blend None or near-zero — raw = pred"
        else:
            raw = (pred - (1.0 - alpha) * arima_pred) / alpha
            decomp_note = f"raw = (pred - (1-{alpha:.4f}) * ARIMA_pred) / {alpha:.4f}"

        # Test metric for both
        r2_orig = 1.0 - float(((y_test - pred) ** 2).sum()) / ss
        r2_raw = 1.0 - float(((y_test - raw) ** 2).sum()) / ss
        mae_raw = float(np.mean(np.abs(y_test - raw)))

        out = {
            "name": nm,
            "raw_test_predictions": raw.tolist(),
            "orig_test_predictions": pred.tolist(),
            "best_config": bc,
            "decomp": {
                "ref_estimator": "ARIMA refit_test_predictions",
                "alpha": alpha,
                "formula": decomp_note,
                "date": "2026-05-06",
                "purpose": "Cluster 1 blend 수렴 fix — paper 정직성 (Bühlmann 2018)",
            },
            "metrics": {
                "r2_orig": r2_orig,
                "r2_raw": r2_raw,
                "mae_raw": mae_raw,
                "delta_r2": r2_raw - r2_orig,
            },
        }
        out_path = RAW_DIR / f"{nm}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        summary.append({
            "name": nm,
            "alpha": alpha,
            "r2_orig": r2_orig,
            "r2_raw": r2_raw,
            "delta": r2_raw - r2_orig,
        })

    # Print summary
    print(f"=== α-blend decomposition — {len(summary)} models → {RAW_DIR} ===")
    print(f"{'name':18s} {'α':>7s}  {'R²orig':>8s}  {'R²raw':>9s}  {'delta':>8s}")
    print("-" * 60)
    for s in sorted(summary, key=lambda x: x["delta"]):
        a_str = f"{s['alpha']:.4f}" if isinstance(s['alpha'], (int, float)) else "  None"
        print(f"{s['name']:18s} {a_str:>7s}  "
              f"{s['r2_orig']:>+.4f}  {s['r2_raw']:>+.4f}   {s['delta']:>+.4f}")

    # Pair correlation re-check
    raws = {}
    for nm in CLUSTER1:
        p = RAW_DIR / f"{nm}.json"
        if p.exists():
            raws[nm] = np.asarray(
                json.loads(p.read_text())["raw_test_predictions"], dtype=float)

    n_pairs = 0
    n_high_orig = 0
    n_high_raw = 0
    pmo_orig = {nm: np.asarray(
        json.loads((PMO / f"{nm}.json").read_text())["refit_test_predictions"],
        dtype=float) for nm in raws}
    names = list(raws.keys())
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j <= i:
                continue
            n_pairs += 1
            c_o = float(np.corrcoef(pmo_orig[ni], pmo_orig[nj])[0, 1])
            c_r = float(np.corrcoef(raws[ni], raws[nj])[0, 1])
            if c_o > 0.99:
                n_high_orig += 1
            if c_r > 0.99:
                n_high_raw += 1
    print()
    print(f"Pair correlation (cluster 1):")
    print(f"  orig: corr > 0.99 = {n_high_orig}/{n_pairs} ({n_high_orig*100//n_pairs}%)")
    print(f"  raw:  corr > 0.99 = {n_high_raw}/{n_pairs} ({n_high_raw*100//n_pairs}%)")
    print(f"  → Recovered diversity: {n_high_orig - n_high_raw} pair separated")


if __name__ == "__main__":
    main()
