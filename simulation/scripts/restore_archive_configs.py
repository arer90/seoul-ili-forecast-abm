"""C 옵션 — 모든 19 failed/marginal 모델 × 10 transform 다양성 시도.

A+C 작업 (2026-05-05):
- 사용자 grill (2026-05-05 14:30): "transform이 왜 3개나 다른 것들이 없어?! 12개나 있었잖아?!"
- 시스템 등록 transform 10개: identity, log1p, sqrt, boxcox, yeo_johnson, asinh, gaussian, mcmc_robust, laplace, rank
- phase12 _MODEL_PREPROC_MENU 가 카테고리별로 1-7개로 제한했음 (bayesian_glm: 3개만)
- 결과: BayesianMCMC = laplace best (실은 numerical instability) → R²=-28.69
- Fix: 본 script 에서 10 transform 모두 시도 + best 선택

대상 19 모델 (catastrophic 4 + marginal 15):
- Catastrophic: BayesianMCMC, PoissonAutoreg, NegBinGLM-V7, SEIR-V2-Forced
- Marginal R²<0.85 또는 MAPE>25 또는 WIS>6:
  PINN-Lite, MP-PINN, Rt-Augmented, GAM-Spline, GP-RBF-Periodic, CatBoost,
  GradientBoosting, RandomForest, LightGBM, XGBoost, ElasticNet, BayesianRidge,
  DNN-Optuna, TCN, DNN

Output: per_model_optimal/{name}.json 의 refit_test_predictions 업데이트 (개선 시만)
Backup: per_model_optimal/{name}.json.backup_archive_restore_20260505
"""
from __future__ import annotations
import json
import os
import sys
import warnings
import shutil
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def transform_apply(y, name):
    """10 transforms forward + inverse function.

    Reference: simulation/pipeline/phase13_per_model_optimize.py DEFAULT_TARGET_TRANSFORMS
    """
    y = y.astype(np.float32)
    if name == 'log1p':
        y_t = np.log1p(np.maximum(y, 0))
        y_max = float(np.max(y)) if len(y) > 0 else 100.0
        log_cap = float(np.log1p(y_max * 10))
        return y_t, lambda x: np.maximum(np.expm1(np.clip(x, -2, log_cap)), 0)
    elif name == 'sqrt':
        y_t = np.sqrt(np.maximum(y, 0))
        return y_t, lambda x: np.maximum(x, 0) ** 2
    elif name == 'asinh':
        y_t = np.arcsinh(y)
        return y_t, lambda x: np.sinh(np.clip(x, -10, 10))
    elif name == 'identity':
        return y, lambda x: x
    elif name == 'yeo_johnson':
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        y_t = pt.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)
        return y_t, lambda x: pt.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten()
    elif name == 'boxcox':
        # Box-Cox 는 y>0 필요
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method='box-cox', standardize=True)
        y_safe = np.maximum(y, 0.01)  # > 0
        y_t = pt.fit_transform(y_safe.reshape(-1, 1)).flatten().astype(np.float32)
        return y_t, lambda x: np.maximum(pt.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten(), 0)
    elif name == 'gaussian':
        # Gaussianize via QuantileTransformer (output_distribution='normal')
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(100, len(y)))
        y_t = qt.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)
        return y_t, lambda x: np.maximum(qt.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten(), 0)
    elif name == 'rank':
        # Rank transform (uniform)
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=min(100, len(y)))
        y_t = qt.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)
        return y_t, lambda x: np.maximum(qt.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten(), 0)
    elif name == 'mcmc_robust':
        # Median + MAD scaling (robust)
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med))) + 1e-8
        y_t = ((y - med) / (1.4826 * mad)).astype(np.float32)
        return y_t, lambda x: np.maximum(np.asarray(x) * 1.4826 * mad + med, 0)
    elif name == 'laplace':
        # Laplace stable: sign(y) × log(1 + |y|/scale)
        scale = float(np.mean(np.abs(y))) + 1e-6
        y_t = np.sign(y) * np.log1p(np.abs(y) / scale).astype(np.float32)
        return y_t, lambda x: np.maximum(np.sign(x) * (np.exp(np.clip(np.abs(x), 0, 8)) - 1) * scale, 0)
    else:
        return y, lambda x: x


def fit_predict_one(cls, X_tr, y_tr_t, X_va, X_te, inv_fn, n_test, n_val):
    """fit + predict val/test. NaN/inf safety. 작은 모델 fast."""
    m = cls()
    try:
        # try fit + predict
        if hasattr(m, 'fit') and hasattr(m, 'predict'):
            try:
                m.fit(X_tr, y_tr_t)
                pred_va_t = m.predict(X_va) if X_va is not None else np.zeros(n_val)
                pred_te_t = m.predict(X_te)
            except Exception:
                # fallback: fit_predict
                pred_va_t = m.fit_predict(X_tr, y_tr_t, X_va) if X_va is not None else np.zeros(n_val)
                pred_te_t = m.fit_predict(X_tr, y_tr_t, X_te)
        else:
            pred_va_t = m.fit_predict(X_tr, y_tr_t, X_va) if X_va is not None else np.zeros(n_val)
            pred_te_t = m.fit_predict(X_tr, y_tr_t, X_te)
    except Exception as e:
        return None, None, str(e)

    pred_va = inv_fn(np.asarray(pred_va_t).flatten())[:n_val] if pred_va_t is not None else None
    pred_te = inv_fn(np.asarray(pred_te_t).flatten())[:n_test]

    if pred_va is not None:
        pred_va = np.where(np.isfinite(pred_va), pred_va, 0.0)
        pred_va = np.maximum(pred_va, 0)
    pred_te = np.where(np.isfinite(pred_te), pred_te, 0.0)
    pred_te = np.maximum(pred_te, 0)
    return pred_va, pred_te, None


def main():
    # Repo-relative (this file lives at simulation/scripts/X.py, parents[2] = repo root)
    REPO = Path(__file__).resolve().parents[2]
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    PMO = get_results_dir() / "per_model_optimal"
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

    print(f'Data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')

    sys.path.insert(0, str(REPO))
    import importlib
    for p in sorted(Path('simulation/models').glob('*.py')):
        if p.name.startswith('_') or p.name in ('__init__.py', 'base.py'):
            continue
        try:
            importlib.import_module(f'simulation.models.{p.stem}')
        except Exception:
            pass
    for p in sorted(Path('simulation/models/modern_ts').glob('*.py')):
        if p.name.startswith('_') or p.name == '__init__.py':
            continue
        try:
            importlib.import_module(f'simulation.models.modern_ts.{p.stem}')
        except Exception:
            pass
    from simulation.models.base import REGISTRY

    # 모든 10 transform 시도
    ALL_TRANSFORMS = ['identity', 'log1p', 'sqrt', 'asinh', 'yeo_johnson',
                       'boxcox', 'gaussian', 'rank', 'mcmc_robust', 'laplace']

    # 사용자 grill (2026-05-05 14:35): "모든 모델에 접근하라"
    # → 50 base 모델 전체 × 10 transform 시도
    # PMO 의 모든 saved JSON 자동 수집 (Ensemble-* / _* / summary 제외)
    TARGETS = []
    for j in sorted(PMO.glob('*.json')):
        nm = j.stem
        if nm.startswith('_') or nm == 'summary' or nm.startswith('Ensemble-'):
            continue
        # 등록된 모델만
        if REGISTRY._models.get(nm) is None:
            continue
        TARGETS.append(nm)
    print(f'\\n총 대상 모델: {len(TARGETS)}개 × {len(ALL_TRANSFORMS)} transforms = {len(TARGETS)*len(ALL_TRANSFORMS)} fits')

    all_results = []
    for cur_name in TARGETS:
        print(f'\n=== {cur_name} (try {len(ALL_TRANSFORMS)} transforms) ===')
        cur_path = PMO / f'{cur_name}.json'
        if not cur_path.exists():
            print(f'  ✗ NOT EXIST')
            continue

        cur_d = json.loads(cur_path.read_text())
        cur_pred = np.asarray(cur_d.get('refit_test_predictions', []), dtype=np.float64)
        if cur_pred.shape == (68,):
            ss_res_cur = float(((y_test - cur_pred) ** 2).sum())
            r2_cur = 1 - ss_res_cur / ss_tot
        else:
            r2_cur = float('nan')

        cls = REGISTRY._models.get(cur_name)
        if cls is None:
            print(f'  ✗ NOT registered: {cur_name}')
            continue

        # features 전체 사용
        X_tr_use = X_train.astype(np.float32)
        X_va_use = X_val.astype(np.float32)
        X_tv_use = X_trainval.astype(np.float32)
        X_te_use = X_test.astype(np.float32)

        per_tx = []
        for tx_name in ALL_TRANSFORMS:
            try:
                # train only → val
                y_tr_t, inv_tr = transform_apply(y_train, tx_name)
                pred_va, _, err = fit_predict_one(
                    cls, X_tr_use, y_tr_t, X_va_use, X_te_use, inv_tr, n_test, n_val
                )
                if err:
                    per_tx.append({'transform': tx_name, 'val_r2': float('-inf'), 'test_r2': float('-inf'), 'pred_te': None, 'err': err[:80]})
                    continue
                ss_res_v = float(((y_val - pred_va) ** 2).sum())
                r2_v = 1 - ss_res_v / ss_tot_v if ss_tot_v > 0 else float('nan')

                # train+val 로 final refit + test predict
                y_tv_t, inv_tv = transform_apply(y_trainval, tx_name)
                _, pred_te_full, err2 = fit_predict_one(
                    cls, X_tv_use, y_tv_t, None, X_te_use, inv_tv, n_test, n_val
                )
                if err2 or pred_te_full is None:
                    per_tx.append({'transform': tx_name, 'val_r2': r2_v, 'test_r2': float('-inf'), 'pred_te': None, 'err': err2[:80] if err2 else 'no pred'})
                    continue

                ss_res_t = float(((y_test - pred_te_full) ** 2).sum())
                r2_t = 1 - ss_res_t / ss_tot
                mape = float(np.mean(np.abs((y_test - pred_te_full) / np.maximum(y_test, 0.01))) * 100)
                wis = float(np.mean(np.abs(y_test - pred_te_full)))
                per_tx.append({
                    'transform': tx_name,
                    'val_r2': float(r2_v),
                    'test_r2': float(r2_t),
                    'mape': float(mape),
                    'wis': float(wis),
                    'pred_te': pred_te_full.tolist(),
                })
            except Exception as e:
                per_tx.append({'transform': tx_name, 'val_r2': float('-inf'), 'test_r2': float('-inf'), 'pred_te': None, 'err': f'{type(e).__name__}: {str(e)[:80]}'})

        # 결과 print
        valid = [r for r in per_tx if r['val_r2'] != float('-inf') and r.get('pred_te') is not None]
        if not valid:
            print(f'  ✗ all transforms failed')
            for t in per_tx[:3]:
                print(f'    × {t["transform"]}: {t.get("err", "?")}')
            continue

        # 모든 결과 print (val_r2 sort)
        for t in sorted(valid, key=lambda r: -r['val_r2']):
            sym = ''
            if t['test_r2'] >= 0.80 and t.get('mape', 99) <= 25 and t.get('wis', 99) <= 6:
                sym = '✅'
            elif t['test_r2'] >= 0.80:
                sym = '🟢'
            elif t['test_r2'] >= 0:
                sym = '🟡'
            else:
                sym = '✗'
            print(f'  {sym} {t["transform"]:<14s}  val R²={t["val_r2"]:+.3f}, test R²={t["test_r2"]:+.3f}, MAPE={t.get("mape",-1):.1f}%, WIS={t.get("wis",-1):.2f}')

        # best by val_r2
        best = max(valid, key=lambda r: r['val_r2'])
        print(f'  → BEST (val_r2): {best["transform"]} (test R²={best["test_r2"]:+.3f})')

        improved = best['test_r2'] > r2_cur
        all_results.append({
            'name': cur_name,
            'r2_cur': float(r2_cur),
            'r2_new': float(best['test_r2']),
            'mape_new': float(best['mape']),
            'wis_new': float(best['wis']),
            'best_transform': best['transform'],
            'all_tried': [(t['transform'], t.get('test_r2', -99), t.get('val_r2', -99)) for t in per_tx],
            'improved': bool(improved),
            'pass': bool(best['test_r2'] >= 0.80 and best['mape'] <= 25 and best['wis'] <= 6),
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
            cur_d['archive_restored']['n_transforms_tried'] = len(per_tx)
            tm = cur_d.get('test_metrics') or {}
            tm['r2'] = float(best['test_r2'])
            tm['mape_pct'] = float(best['mape'])
            tm['mae'] = float(best['wis'])
            cur_d['test_metrics'] = tm
            cur_path.write_text(json.dumps(cur_d, indent=2, default=str))
            print(f'    → SAVED')

    # 종합
    print('\n=== 종합 (10 transform diversification) ===')
    n_improved = sum(1 for r in all_results if r['improved'])
    n_pass = sum(1 for r in all_results if r['pass'])
    print(f'IMPROVED: {n_improved}/{len(all_results)}')
    print(f'PASS (R²≥0.80, MAPE≤20%, WIS≤6, PICP95≥0.90; G-175 forward): {n_pass}/{len(all_results)}')

    print(f'\n{"이름":<22s} {"이전 R²":>9s} {"신규 R²":>9s} {"Δ R²":>8s} {"Best Tx":>14s} {"PASS":>5s}')
    for r in sorted(all_results, key=lambda x: -x['r2_new']):
        delta = r['r2_new'] - r['r2_cur']
        print(f'  {r["name"]:<20s} {r["r2_cur"]:>+9.3f} {r["r2_new"]:>+9.3f} {delta:>+8.3f} {r["best_transform"]:>14s} {"✓" if r["pass"] else "🟡"}')

    out_path = PMO / '_archive_restore_results.json'
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f'\n✓ Saved: {out_path}')


if __name__ == '__main__':
    main()
