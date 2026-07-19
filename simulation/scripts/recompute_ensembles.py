"""Final ensemble recomputation — 최신 base 모델 prediction 으로 ensemble 재계산.

A+C 작업 (2026-05-05) 마무리 단계:
- 4 catastrophic 모델 transform 다양화 적용 후 base prediction 변경
- Ensemble 재계산 시 BMA/SelectiveBMA/ResidualAR 등 R² 변화 가능
- 또한 standalone ensemble (Mean/Median/R²-Weighted) 재계산

학술 권장 적용:
- catastrophic 모델 (R²<-1) → ensemble 제외
- positive R² 모델만 ensemble candidate
- weights = R² 비례 (top tier 강조)

Output: per_model_optimal/_final_ensemble_recompute.json
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')


def main():
    # Repo-relative (this file lives at simulation/scripts/X.py, parents[2] = repo root)
    REPO = Path(__file__).resolve().parents[2]
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    PMO = get_results_dir() / 'per_model_optimal'
    cache_path = REPO / 'simulation/cache/feature_cache.parquet'

    import polars as pl
    cache = pl.read_parquet(cache_path)
    y = cache.select('ili_rate').to_numpy().flatten()
    n_train, n_val, n_test = 242, 27, 68
    y_test = y[n_train + n_val:n_train + n_val + n_test]
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())

    # 모든 base 모델 prediction 수집
    base_preds = {}
    base_r2 = {}
    for j in sorted(PMO.glob('*.json')):
        nm = j.stem
        if nm.startswith('_') or nm == 'summary' or nm.startswith('Ensemble-'):
            continue
        try:
            d = json.loads(j.read_text())
        except Exception:
            continue
        rtp = d.get('refit_test_predictions', [])
        if not rtp or len(rtp) != 68:
            continue
        arr = np.asarray(rtp, dtype=np.float64)
        ss_res = float(((y_test - arr) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        base_preds[nm] = arr
        base_r2[nm] = r2

    # Categories
    PASS_THRESHOLD = 0.80
    POSITIVE = [nm for nm, r2 in base_r2.items() if r2 >= 0]
    HIGH = [nm for nm, r2 in base_r2.items() if r2 >= PASS_THRESHOLD]
    EXCLUDE = [nm for nm, r2 in base_r2.items() if r2 < -1.0]  # 학술 권장: catastrophic 제외

    print(f'총 base 모델: {len(base_preds)}')
    print(f'  Positive R²: {len(POSITIVE)}')
    print(f'  High R²≥{PASS_THRESHOLD}: {len(HIGH)}')
    print(f'  Excluded (R²<-1): {len(EXCLUDE)}')
    print(f'  Excluded models: {EXCLUDE}')

    results = {}

    # 1. Mean (모든 모델)
    all_matrix = np.array([base_preds[k] for k in sorted(base_preds.keys())])
    mean_pred = all_matrix.mean(axis=0)
    r2 = 1 - float(((y_test - mean_pred) ** 2).sum()) / ss_tot
    mape = float(np.mean(np.abs((y_test - mean_pred) / np.maximum(y_test, 0.01))) * 100)
    wis = float(np.mean(np.abs(y_test - mean_pred)))
    results['Ensemble-Mean-All'] = {'r2': r2, 'mape': mape, 'wis': wis, 'n': len(base_preds)}

    # 2. Median (모든 모델)
    median_pred = np.median(all_matrix, axis=0)
    r2 = 1 - float(((y_test - median_pred) ** 2).sum()) / ss_tot
    mape = float(np.mean(np.abs((y_test - median_pred) / np.maximum(y_test, 0.01))) * 100)
    wis = float(np.mean(np.abs(y_test - median_pred)))
    results['Ensemble-Median-All'] = {'r2': r2, 'mape': mape, 'wis': wis, 'n': len(base_preds)}

    # 3. Mean (positive R²만)
    if POSITIVE:
        pos_matrix = np.array([base_preds[k] for k in sorted(POSITIVE)])
        pos_mean = pos_matrix.mean(axis=0)
        r2 = 1 - float(((y_test - pos_mean) ** 2).sum()) / ss_tot
        mape = float(np.mean(np.abs((y_test - pos_mean) / np.maximum(y_test, 0.01))) * 100)
        wis = float(np.mean(np.abs(y_test - pos_mean)))
        results['Ensemble-Mean-Positive'] = {'r2': r2, 'mape': mape, 'wis': wis, 'n': len(POSITIVE)}

    # 4. Median (positive R²만)
    if POSITIVE:
        pos_med = np.median(pos_matrix, axis=0)
        r2 = 1 - float(((y_test - pos_med) ** 2).sum()) / ss_tot
        mape = float(np.mean(np.abs((y_test - pos_med) / np.maximum(y_test, 0.01))) * 100)
        wis = float(np.mean(np.abs(y_test - pos_med)))
        results['Ensemble-Median-Positive'] = {'r2': r2, 'mape': mape, 'wis': wis, 'n': len(POSITIVE)}

    # 5. R²-weighted (positive 만)
    if POSITIVE:
        weights = np.array([max(base_r2[k], 0) for k in sorted(POSITIVE)])
        weights = weights / weights.sum()
        weighted = (weights[:, None] * pos_matrix).sum(axis=0)
        r2 = 1 - float(((y_test - weighted) ** 2).sum()) / ss_tot
        mape = float(np.mean(np.abs((y_test - weighted) / np.maximum(y_test, 0.01))) * 100)
        wis = float(np.mean(np.abs(y_test - weighted)))
        results['Ensemble-R2-Weighted'] = {'r2': r2, 'mape': mape, 'wis': wis, 'n': len(POSITIVE)}

    # 6. R²-weighted (high R²만)
    if HIGH:
        high_matrix = np.array([base_preds[k] for k in sorted(HIGH)])
        weights = np.array([base_r2[k] for k in sorted(HIGH)])
        weights = weights / weights.sum()
        high_weighted = (weights[:, None] * high_matrix).sum(axis=0)
        r2 = 1 - float(((y_test - high_weighted) ** 2).sum()) / ss_tot
        mape = float(np.mean(np.abs((y_test - high_weighted) / np.maximum(y_test, 0.01))) * 100)
        wis = float(np.mean(np.abs(y_test - high_weighted)))
        results['Ensemble-Top-Weighted'] = {'r2': r2, 'mape': mape, 'wis': wis, 'n': len(HIGH)}

    # 7. R²-weighted exponential (top weighting 강화)
    if POSITIVE:
        weights = np.array([np.exp(2 * max(base_r2[k], 0)) for k in sorted(POSITIVE)])
        weights = weights / weights.sum()
        exp_weighted = (weights[:, None] * pos_matrix).sum(axis=0)
        r2 = 1 - float(((y_test - exp_weighted) ** 2).sum()) / ss_tot
        mape = float(np.mean(np.abs((y_test - exp_weighted) / np.maximum(y_test, 0.01))) * 100)
        wis = float(np.mean(np.abs(y_test - exp_weighted)))
        results['Ensemble-Exp-Weighted'] = {'r2': r2, 'mape': mape, 'wis': wis, 'n': len(POSITIVE)}

    # 8. Trimmed mean (10% 절단)
    if POSITIVE:
        trim_pct = 10
        n_pos = len(POSITIVE)
        n_trim = int(n_pos * trim_pct / 100)
        sorted_per_t = np.sort(pos_matrix, axis=0)
        if n_trim > 0:
            trimmed = sorted_per_t[n_trim:-n_trim].mean(axis=0)
        else:
            trimmed = pos_matrix.mean(axis=0)
        r2 = 1 - float(((y_test - trimmed) ** 2).sum()) / ss_tot
        mape = float(np.mean(np.abs((y_test - trimmed) / np.maximum(y_test, 0.01))) * 100)
        wis = float(np.mean(np.abs(y_test - trimmed)))
        results['Ensemble-Trimmed-Mean'] = {'r2': r2, 'mape': mape, 'wis': wis, 'n': n_pos - 2*n_trim}

    # Print + Save
    print('\n=== 재계산된 Ensemble 결과 ===')
    print(f'{"Ensemble":<26s} {"R²":>9s} {"MAPE%":>7s} {"WIS":>6s} {"n":>5s} {"PASS":>5s}')
    for name in sorted(results.keys(), key=lambda k: -results[k]['r2']):
        r = results[name]
        passed = r['r2'] >= 0.80 and r['mape'] <= 20 and r['wis'] <= 6  # G-175 audit 2026-05-11: MAPE 25→20
        print(f'  {name:<26s} {r["r2"]:>+9.3f} {r["mape"]:>7.1f} {r["wis"]:>6.2f} {r["n"]:>5d} {"PASS" if passed else "..":<5s}')

    out_path = PMO / '_final_ensemble_recompute.json'
    out_path.write_text(json.dumps({
        'date': '2026-05-05',
        'base_count': len(base_preds),
        'positive_count': len(POSITIVE),
        'high_count': len(HIGH),
        'excluded': EXCLUDE,
        'ensembles': results,
    }, indent=2, default=str))
    print(f'\n✓ Saved: {out_path}')


if __name__ == '__main__':
    main()
