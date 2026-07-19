"""Full ensemble runner — base 모델 val 재학습 → val_predictions dict → 11 ensemble.

A 진행 (caller chain refactor 우회):
1. saved base 11 모델의 best_config 추출
2. 각 모델 train only refit → predict on val (val_predictions dict)
3. saved test_predictions 와 합쳐 11 ensemble 호출
4. 모든 결과 saved JSON

용 폴라스 cache 활용 (실 데이터).
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def main():
    # 2026-05-26: hardcoded user path → repo-relative (ENGINEERING_PRINCIPLES.md §원칙 #1 portability)
    _repo = Path(__file__).resolve().parents[2]   # …/MPH_infection_simulation
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    PMO = get_results_dir() / "per_model_optimal"
    cache_path = _repo / "simulation/cache/feature_cache.parquet"

    # 1. 데이터 로드 + split
    import polars as pl
    cache = pl.read_parquet(cache_path)
    feature_cols = [c for c in cache.columns if c != 'ili_rate']
    X = cache.select(feature_cols).to_numpy().astype(np.float32)
    y = cache.select('ili_rate').to_numpy().flatten().astype(np.float32)
    n_train, n_val, n_test = 242, 27, 68
    X_train = X[:n_train]; y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]; y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:n_train+n_val+n_test]; y_test = y[n_train+n_val:n_train+n_val+n_test]
    print(f'Data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')

    # 2. base 11 모델 saved best config 로드 + train only fit + predict val/test
    import simulation.models
    import importlib
    for p in sorted(Path('simulation/models').glob('*.py')):
        if p.name.startswith('_') or p.name in ('__init__.py', 'base.py'): continue
        try: importlib.import_module(f'simulation.models.{p.stem}')
        except Exception: pass   # 2026-05-26: bare except → Exception (preserve KeyboardInterrupt)
    for p in sorted(Path('simulation/models/modern_ts').glob('*.py')):
        if p.name.startswith('_') or p.name == '__init__.py': continue
        try: importlib.import_module(f'simulation.models.modern_ts.{p.stem}')
        except Exception: pass   # 2026-05-26: bare except → Exception (preserve KeyboardInterrupt)
    from simulation.models.base import REGISTRY

    BASE_MODELS = ['DNN-Conformal', 'GE-DNN-GAT', 'KRR', 'LightGBM', 'Mamba',
                   'NegBinGLM', 'SVR-Linear', 'SVR-RBF', 'TiDE', 'TimesNet', 'iTransformer']

    val_predictions = {}
    test_predictions = {}
    base_test_r2 = {}

    print('\n=== Base 11 모델 train+predict (val/test) ===')
    for name in BASE_MODELS:
        cls = REGISTRY._models.get(name)
        if cls is None:
            print(f'  {name}: NOT_REGISTERED')
            continue
        # saved best_config 로드
        p = PMO / f'{name}.json'
        if not p.exists():
            print(f'  {name}: NO_SAVED_CONFIG')
            continue
        try:
            d = json.loads(p.read_text())
            bc = d.get('best_config', {})
            feat_idx = bc.get('feature_indices')
            transform = bc.get('transform', 'identity')

            # subset features (saved best config)
            X_tr = X_train[:, feat_idx] if feat_idx else X_train
            X_va = X_val[:, feat_idx] if feat_idx else X_val
            X_te = X_test[:, feat_idx] if feat_idx else X_test

            # transform y_train
            if transform == 'log1p':
                y_tr_t = np.log1p(np.maximum(y_train, 0))
                inv = lambda x: np.maximum(np.expm1(np.clip(x, -2, 8)), 0)
            elif transform == 'sqrt':
                y_tr_t = np.sqrt(np.maximum(y_train, 0))
                inv = lambda x: np.maximum(x, 0) ** 2
            elif transform == 'asinh':
                y_tr_t = np.arcsinh(y_train)
                inv = lambda x: np.sinh(np.clip(x, -10, 10))
            else:
                y_tr_t = y_train.astype(np.float32)
                inv = lambda x: x

            # fit + predict
            m = cls()
            try:
                pred_val_t = m.fit_predict(X_tr, y_tr_t, X_va)
                pred_test_t = m.predict(X_te) if hasattr(m, 'predict') and m._fitted else m.fit_predict(X_tr, y_tr_t, X_te)
            except Exception as e:
                # fallback: 직접 fit + predict
                m.fit(X_tr, y_tr_t)
                pred_val_t = m.predict(X_va)
                pred_test_t = m.predict(X_te)

            # inverse transform
            pred_val = np.maximum(inv(np.asarray(pred_val_t).flatten())[:n_val], 0)
            pred_test = np.maximum(inv(np.asarray(pred_test_t).flatten())[:n_test], 0)

            val_predictions[name] = pred_val
            test_predictions[name] = pred_test

            # test R²
            ss_res = float(((y_test - pred_test)**2).sum())
            ss_tot = float(((y_test - y_test.mean())**2).sum())
            r2_test = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
            ss_res_v = float(((y_val - pred_val)**2).sum())
            ss_tot_v = float(((y_val - y_val.mean())**2).sum())
            r2_val = 1 - ss_res_v/ss_tot_v if ss_tot_v > 0 else float('nan')
            base_test_r2[name] = r2_test
            print(f'  ✓ {name:<22s}  val R²={r2_val:+.3f}, test R²={r2_test:+.3f}, transform={transform}')

        except Exception as e:
            print(f'  ✗ {name}: {type(e).__name__}: {str(e)[:80]}')

    print(f'\n수집됨: {len(val_predictions)} val, {len(test_predictions)} test')

    # 3. 11 Ensemble fit + predict (val_predictions 활용)
    ENSEMBLES = ['Ensemble-Adaptive', 'Ensemble-BMA', 'Ensemble-Blending', 'Ensemble-Diversity',
                 'Ensemble-InvRMSE', 'Ensemble-NNLS', 'Ensemble-NNLS-Filtered',
                 'Ensemble-ResidualAR', 'Ensemble-SelectiveBMA', 'Ensemble-Stacking',
                 'Ensemble-Temporal']
    print(f'\n=== 11 Ensemble fit + predict ===')
    ensemble_results = {}
    for name in ENSEMBLES:
        cls = REGISTRY._models.get(name)
        if cls is None:
            continue
        try:
            m = cls()
            m.fit(np.zeros((len(y_train), 1), dtype=np.float32), y_train,
                  val_predictions=val_predictions, val_actual=y_val)
            pred = m.predict(np.zeros((n_test, 1), dtype=np.float32),
                             model_predictions=test_predictions)
            pred = np.asarray(pred, dtype=np.float64).flatten()[:n_test]
            ss_res = float(((y_test - pred)**2).sum())
            ss_tot = float(((y_test - y_test.mean())**2).sum())
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
            mape = np.mean(np.abs((y_test - pred) / np.maximum(y_test, 0.01))) * 100
            wis = float(np.mean(np.abs(y_test - pred)))  # MAE proxy
            ensemble_results[name] = {'pred': pred.tolist(), 'r2': r2, 'mape': mape, 'wis': wis}
            sym = '✓' if r2 >= 0.85 else '🟡' if r2 >= 0 else '✗'
            print(f'  {sym} {name:<24s}  R²={r2:+.3f}  MAPE={mape:.1f}%  WIS={wis:.2f}')
        except Exception as e:
            print(f'  ✗ {name:<24s}  {type(e).__name__}: {str(e)[:80]}')

    # 4. Save
    out_path = PMO / '_ensemble_full.json'
    save_data = {
        'base_n': len(val_predictions),
        'base_test_r2': base_test_r2,
        'ensembles': ensemble_results,
    }
    out_path.write_text(json.dumps(save_data, indent=2, default=str))
    print(f'\n✓ Saved: {out_path}')

    # Multi-criteria PASS
    print(f'\n=== Multi-criteria PASS (R²≥0.80, MAPE≤20%) ===')
    n_pass = 0
    for n, r in ensemble_results.items():
        passed = r['r2'] >= 0.80 and r['mape'] <= 20
        if passed:
            print(f'  ✓ {n}: R²={r["r2"]:+.3f}, MAPE={r["mape"]:.1f}%')
            n_pass += 1
    print(f'\nPASS: {n_pass}/11 ensemble')


if __name__ == '__main__':
    main()
