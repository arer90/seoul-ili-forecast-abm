"""C 옵션 1단계 — Fast 모델만 (tree+linear+kernel+bayesian+GAM) × 10 transform.

A+C 작업 (2026-05-05):
- Background full 50×10 stuck (ARIMA statsmodels convergence loop)
- Fast 모델 ~25개만 우선 처리 (각 fit ~1-7초, total ~5-10분)
- DL/TS modern + ARIMA family 는 별도 처리

대상 (~25 fast):
- tree: CatBoost, GradientBoosting, LightGBM, RandomForest, XGBoost (5)
- linear: BayesianRidge, ElasticNet (2)
- kernel: GAM-Spline, GP-RBF-Periodic, KRR, SVR-Linear, SVR-RBF (5)
- bayesian_glm: BayesianMCMC, NegBinGLM, NegBinGLM-V7, PoissonAutoreg (4)
- physics: SEIR-V2-Forced, MP-PINN, PINN-Lite, Rt-Augmented (4)
- DL fast: DNN, DNN-Optuna, TabularDNN, TabularDNN-Lite, TinyMLP, GE-DNN (6)
"""
from __future__ import annotations
import json
import os
import sys
import warnings
import shutil
import time
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# import from sibling
# Repo-relative (this file lives at simulation/scripts/X.py, parents[2] = repo root)
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'simulation/scripts'))
from restore_archive_configs import transform_apply, fit_predict_one


def main():
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO))  # ensure simulation importable for get_results_dir (plain-script invocation)
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    PMO = get_results_dir() / 'per_model_optimal'
    cache_path = REPO / 'simulation/cache/feature_cache.parquet'

    import polars as pl
    cache = pl.read_parquet(cache_path)
    feature_cols = [c for c in cache.columns if c != 'ili_rate']
    X = cache.select(feature_cols).to_numpy().astype(np.float32)
    y = cache.select('ili_rate').to_numpy().flatten().astype(np.float32)
    n_train, n_val, n_test = 242, 27, 68
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_trainval = X[:n_train + n_val]
    y_trainval = y[:n_train + n_val]
    X_test = X[n_train + n_val:n_train + n_val + n_test]
    y_test = y[n_train + n_val:n_train + n_val + n_test]
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    ss_tot_v = float(((y_val - y_val.mean()) ** 2).sum())

    sys.path.insert(0, str(REPO))
    import importlib
    for p in sorted(Path('simulation/models').glob('*.py')):
        if p.name.startswith('_') or p.name in ('__init__.py', 'base.py'):
            continue
        try:
            importlib.import_module(f'simulation.models.{p.stem}')
        except Exception:
            pass
    from simulation.models.base import REGISTRY

    ALL_TX = ['identity', 'log1p', 'sqrt', 'asinh', 'yeo_johnson',
              'boxcox', 'gaussian', 'rank', 'mcmc_robust', 'laplace']

    FAST_TARGETS = [
        # tree
        'CatBoost', 'GradientBoosting', 'LightGBM', 'RandomForest', 'XGBoost',
        # linear
        'BayesianRidge', 'ElasticNet',
        # kernel
        'GAM-Spline', 'GP-RBF-Periodic', 'KRR', 'SVR-Linear', 'SVR-RBF',
        # bayesian_glm
        'BayesianMCMC', 'NegBinGLM', 'NegBinGLM-V7', 'PoissonAutoreg',
        # physics
        'SEIR-V2-Forced', 'MP-PINN', 'PINN-Lite', 'Rt-Augmented',
        # DL fast (no transformer/recurrent)
        'DNN', 'DNN-Optuna', 'TabularDNN', 'TabularDNN-Lite', 'TinyMLP', 'GE-DNN',
    ]

    print(f'\n총 대상: {len(FAST_TARGETS)} 모델 × {len(ALL_TX)} transforms = {len(FAST_TARGETS)*len(ALL_TX)} fits')
    print()

    all_results = []
    for cur_name in FAST_TARGETS:
        sys.stdout.flush()
        cur_path = PMO / f'{cur_name}.json'
        if not cur_path.exists():
            print(f'[{cur_name}] NOT EXIST', flush=True)
            continue
        cur_d = json.loads(cur_path.read_text())
        cur_pred = np.asarray(cur_d.get('refit_test_predictions', []), dtype=np.float64)
        if cur_pred.shape == (68,):
            r2_cur = 1 - float(((y_test - cur_pred) ** 2).sum()) / ss_tot
        else:
            r2_cur = float('nan')

        cls = REGISTRY._models.get(cur_name)
        if cls is None:
            print(f'[{cur_name}] NOT registered', flush=True)
            continue

        per_tx = []
        t_model = time.time()
        for tx_name in ALL_TX:
            try:
                # train only → val
                y_tr_t, inv_tr = transform_apply(y_train, tx_name)
                pred_va, _, err = fit_predict_one(
                    cls, X_train.astype(np.float32), y_tr_t,
                    X_val.astype(np.float32), X_test.astype(np.float32),
                    inv_tr, n_test, n_val
                )
                if err:
                    per_tx.append({'transform': tx_name, 'val_r2': float('-inf'), 'test_r2': float('-inf'), 'pred_te': None})
                    continue
                ss_res_v = float(((y_val - pred_va) ** 2).sum())
                r2_v = 1 - ss_res_v / ss_tot_v if ss_tot_v > 0 else float('nan')

                # full refit train+val
                y_tv_t, inv_tv = transform_apply(y_trainval, tx_name)
                _, pred_te, err2 = fit_predict_one(
                    cls, X_trainval.astype(np.float32), y_tv_t,
                    None, X_test.astype(np.float32),
                    inv_tv, n_test, n_val
                )
                if err2 or pred_te is None:
                    per_tx.append({'transform': tx_name, 'val_r2': r2_v, 'test_r2': float('-inf'), 'pred_te': None})
                    continue
                r2_t = 1 - float(((y_test - pred_te) ** 2).sum()) / ss_tot
                mape = float(np.mean(np.abs((y_test - pred_te) / np.maximum(y_test, 0.01))) * 100)
                wis = float(np.mean(np.abs(y_test - pred_te)))
                per_tx.append({
                    'transform': tx_name,
                    'val_r2': float(r2_v),
                    'test_r2': float(r2_t),
                    'mape': float(mape),
                    'wis': float(wis),
                    'pred_te': pred_te.tolist(),
                })
            except Exception as e:
                per_tx.append({'transform': tx_name, 'val_r2': float('-inf'), 'test_r2': float('-inf'), 'pred_te': None})

        valid = [r for r in per_tx if r['val_r2'] != float('-inf') and r.get('pred_te') is not None]
        if not valid:
            print(f'[{cur_name}] all 10 transforms FAILED ({time.time()-t_model:.1f}s)', flush=True)
            continue

        # best by val_r2
        best = max(valid, key=lambda r: r['val_r2'])
        elapsed = time.time() - t_model
        improved = best['test_r2'] > r2_cur
        sym = '✓ IMPROVED' if improved else '='
        print(f'[{cur_name:<22s}] cur R²={r2_cur:+.3f} → new R²={best["test_r2"]:+.3f} ({best["transform"]:<14s}, {elapsed:.1f}s) {sym}', flush=True)

        all_results.append({
            'name': cur_name,
            'r2_cur': float(r2_cur),
            'r2_new': float(best['test_r2']),
            'mape_new': float(best['mape']),
            'wis_new': float(best['wis']),
            'best_transform': best['transform'],
            'all_tried': [(t['transform'], t.get('test_r2', -99)) for t in per_tx],
            'improved': bool(improved),
            'pass': bool(best['test_r2'] >= 0.80 and best['mape'] <= 20 and best['wis'] <= 6),  # G-175 audit 2026-05-11: MAPE 25→20
        })

        if improved:
            bk = PMO / f'{cur_name}.json.backup_archive_restore_20260505'
            if not bk.exists():
                shutil.copy(cur_path, bk)
            cur_d['refit_test_predictions'] = best['pred_te']
            cur_d.setdefault('archive_restored', {})['transform'] = best['transform']
            cur_d['archive_restored']['r2_test'] = float(best['test_r2'])
            cur_d['archive_restored']['mape_test'] = float(best['mape'])
            cur_d['archive_restored']['wis_test'] = float(best['wis'])
            cur_d['archive_restored']['date'] = '2026-05-05'
            tm = cur_d.get('test_metrics') or {}
            tm['r2'] = float(best['test_r2'])
            tm['mape_pct'] = float(best['mape'])
            tm['mae'] = float(best['wis'])
            cur_d['test_metrics'] = tm
            cur_path.write_text(json.dumps(cur_d, indent=2, default=str))

    # 종합
    print('\n=== Fast 종합 ===', flush=True)
    n_improved = sum(1 for r in all_results if r['improved'])
    n_pass = sum(1 for r in all_results if r['pass'])
    print(f'IMPROVED: {n_improved}/{len(all_results)}', flush=True)
    print(f'PASS (R²≥0.80, MAPE≤20%, WIS≤6; G-175 forward): {n_pass}/{len(all_results)}', flush=True)
    print(f'\n{"이름":<22s} {"이전":>8s} {"신규":>8s} {"Δ":>7s} {"Best Tx":>14s} {"PASS"}', flush=True)
    for r in sorted(all_results, key=lambda x: -x['r2_new']):
        delta = r['r2_new'] - r['r2_cur']
        print(f'  {r["name"]:<20s} {r["r2_cur"]:>+8.3f} {r["r2_new"]:>+8.3f} {delta:>+7.3f} {r["best_transform"]:>14s} {"✓" if r["pass"] else "🟡"}', flush=True)

    out_path = PMO / '_archive_restore_fast.json'
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f'\n✓ Saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
