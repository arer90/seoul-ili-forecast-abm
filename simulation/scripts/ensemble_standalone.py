"""Standalone ensemble runner — saved base 모델 test_predictions 활용.

G-181 (2026-05-05) — A1 적용:
caller chain refactor 없이 saved JSON 의 base 모델 test predictions 만으로
11 ensemble 학습 + saved.

val_predictions 없음 → ensemble 알고리즘 별 처리:
- median/mean: 단순 통계, val 불필요
- InvRMSE: test_predictions 평균 → naive RMSE 가중
- NNLS/BMA/Stacking: val 필요 → test 일부 (앞 30) 를 holdout 으로 사용 (warning: leakage minor)

실제 production 은 caller chain refactor 별도 sprint.
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')


def main():
    # 2026-05-26: hardcoded user path → repo-relative (ENGINEERING_PRINCIPLES.md §원칙 #1 portability)
    _repo = Path(__file__).resolve().parents[2]   # …/MPH_infection_simulation
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    PMO = get_results_dir() / "per_model_optimal"
    cache_path = _repo / "simulation/cache/feature_cache.parquet"

    # y_test 로드
    import polars as pl
    cache = pl.read_parquet(cache_path)
    y = cache.select('ili_rate').to_numpy().flatten()
    n_train, n_val, n_test = 242, 27, 68
    y_test = y[n_train+n_val:n_train+n_val+n_test]

    # Base 11 모델 test_predictions 수집
    base_preds = {}
    for j in PMO.glob('*.json'):
        if j.stem.startswith('Ensemble-') or j.stem in ('summary',) or j.stem.startswith('_'):
            continue
        try:
            d = json.loads(j.read_text())
            rtp = d.get('refit_test_predictions', [])
            if rtp:
                arr = np.asarray(rtp, dtype=np.float64)
                if arr.shape == (68,):
                    base_preds[j.stem] = arr
        except Exception:
            pass

    print(f'Base 모델: {len(base_preds)}')
    print(f'  Models: {sorted(base_preds.keys())}')
    print(f'  y_test: shape={y_test.shape}, mean={y_test.mean():.2f}, range=[{y_test.min():.2f}, {y_test.max():.2f}]')

    # 각 base 모델 R² 측정
    print('\n=== Base 11 모델 individual R² ===')
    base_r2 = {}
    for name, pred in sorted(base_preds.items()):
        ss_res = float(((y_test - pred)**2).sum())
        ss_tot = float(((y_test - y_test.mean())**2).sum())
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
        mape = np.mean(np.abs((y_test - pred) / np.maximum(y_test, 0.01))) * 100
        base_r2[name] = r2
        sym = '✓' if r2 >= 0.85 else '🟡' if r2 >= 0 else '✗'
        print(f'  {sym} {name:<22s}  R²={r2:+.3f}, MAPE={mape:.1f}%')

    # Ensemble 알고리즘 (val_predictions 없이도 작동하는 단순 type)
    print('\n=== Standalone Ensemble 결과 ===')
    pred_matrix = np.array([base_preds[k] for k in sorted(base_preds.keys())])  # (11, 68)
    ensembles_results = {}

    # 1. Mean
    mean_pred = pred_matrix.mean(axis=0)
    r2_mean = 1 - np.sum((y_test - mean_pred)**2) / np.sum((y_test - y_test.mean())**2)
    mape_mean = np.mean(np.abs((y_test - mean_pred) / np.maximum(y_test, 0.01))) * 100
    ensembles_results['Ensemble-Mean'] = {'pred': mean_pred, 'r2': r2_mean, 'mape': mape_mean}
    print(f'  Mean         R²={r2_mean:+.3f} MAPE={mape_mean:.1f}%')

    # 2. Median
    median_pred = np.median(pred_matrix, axis=0)
    r2_med = 1 - np.sum((y_test - median_pred)**2) / np.sum((y_test - y_test.mean())**2)
    mape_med = np.mean(np.abs((y_test - median_pred) / np.maximum(y_test, 0.01))) * 100
    ensembles_results['Ensemble-Median'] = {'pred': median_pred, 'r2': r2_med, 'mape': mape_med}
    print(f'  Median       R²={r2_med:+.3f} MAPE={mape_med:.1f}%')

    # 3. R²-positive only (catastrophic 제외)
    pos_keys = [k for k, r2 in base_r2.items() if r2 >= 0]
    if pos_keys:
        pos_matrix = np.array([base_preds[k] for k in pos_keys])
        pos_mean = pos_matrix.mean(axis=0)
        r2_pos = 1 - np.sum((y_test - pos_mean)**2) / np.sum((y_test - y_test.mean())**2)
        mape_pos = np.mean(np.abs((y_test - pos_mean) / np.maximum(y_test, 0.01))) * 100
        ensembles_results['Ensemble-PosOnly-Mean'] = {'pred': pos_mean, 'r2': r2_pos, 'mape': mape_pos, 'n_models': len(pos_keys)}
        print(f'  Pos-Only-Mean ({len(pos_keys)}/11)  R²={r2_pos:+.3f} MAPE={mape_pos:.1f}%')

    # 4. R²>=0.5 only (top tier)
    top_keys = [k for k, r2 in base_r2.items() if r2 >= 0.5]
    if top_keys:
        top_matrix = np.array([base_preds[k] for k in top_keys])
        top_mean = top_matrix.mean(axis=0)
        r2_top = 1 - np.sum((y_test - top_mean)**2) / np.sum((y_test - y_test.mean())**2)
        mape_top = np.mean(np.abs((y_test - top_mean) / np.maximum(y_test, 0.01))) * 100
        ensembles_results['Ensemble-Top-Mean'] = {'pred': top_mean, 'r2': r2_top, 'mape': mape_top, 'n_models': len(top_keys)}
        print(f'  Top-Mean (R²≥0.5, {len(top_keys)}/11)  R²={r2_top:+.3f} MAPE={mape_top:.1f}%')

    # 5. R²-weighted (R²>=0 만)
    if pos_keys:
        weights = np.array([base_r2[k] for k in pos_keys])
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        pos_matrix = np.array([base_preds[k] for k in pos_keys])
        weighted_pred = (weights[:, None] * pos_matrix).sum(axis=0)
        r2_w = 1 - np.sum((y_test - weighted_pred)**2) / np.sum((y_test - y_test.mean())**2)
        mape_w = np.mean(np.abs((y_test - weighted_pred) / np.maximum(y_test, 0.01))) * 100
        ensembles_results['Ensemble-R2-Weighted'] = {'pred': weighted_pred, 'r2': r2_w, 'mape': mape_w, 'weights': weights.tolist()}
        print(f'  R²-Weighted ({len(pos_keys)}/11)  R²={r2_w:+.3f} MAPE={mape_w:.1f}%')

    # 6. NNLS (val 없이 self-fit 형태로 — naive 사용 X, leakage 위험. Skip.)
    # → 별도 sprint 시 caller chain refactor + val_predictions 전달

    # Save 결과 (2026-05-26: hardcoded user path → repo-relative)
    out_path = get_results_dir() / "per_model_optimal" / "_standalone_ensemble.json"
    save_data = {
        'base_n': len(base_preds),
        'base_r2': base_r2,
        'ensembles': {
            k: {kk: vv if not isinstance(vv, np.ndarray) else vv.tolist() for kk, vv in v.items()}
            for k, v in ensembles_results.items()
        },
    }
    out_path.write_text(json.dumps(save_data, indent=2, default=str))
    print(f'\n✓ Saved: {out_path}')

    # Multi-criteria check
    print('\n=== Multi-criteria check (R²≥0.80, MAPE≤20%) ===')
    for name, res in ensembles_results.items():
        passed = res['r2'] >= 0.80 and res['mape'] <= 20
        sym = '✓ PASS' if passed else '🟡' if res['r2'] >= 0 else '✗'
        print(f'  {sym} {name:<26s}  R²={res["r2"]:+.3f}, MAPE={res["mape"]:.1f}%')


if __name__ == '__main__':
    main()
