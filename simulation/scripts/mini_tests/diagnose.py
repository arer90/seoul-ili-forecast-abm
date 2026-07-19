"""Per-model diagnostic — 1 model fit/predict + 6 health checks.

Returns dict with: status (PASS/WARN/FAIL), metrics, issues.
"""
from __future__ import annotations
import time
import warnings
from typing import Optional
import numpy as np

warnings.filterwarnings('ignore')


def diagnose_model(model_cls, data: dict, *, model_name: str = '',
                   timeout_sec: int = 60) -> dict:
    """Single model diagnose with timeout.

    Args:
        model_cls: model class (callable, no args needed)
        data: synthetic dataset dict (see synthetic.py)
        model_name: identifier
        timeout_sec: per-model time budget

    Returns:
        {
          'name', 'status', 'category', 'level', 'time_sec',
          'pred', 'metrics', 'issues': [str], 'fix_hints': [str],
        }
    """
    result = {
        'name': model_name or model_cls.__name__,
        'status': 'UNKNOWN',
        'category': getattr(model_cls.meta, 'category', '?'),
        'level': getattr(model_cls.meta, 'level', None),
        'time_sec': 0.0,
        'metrics': {},
        'issues': [],
        'fix_hints': [],
    }

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    feat_names = data['feature_names']

    # G-181 (2026-05-05): Foundation 류는 train+val context 사용 (paper 표준).
    # R9(per_model_optimize) 의 X_train_pool = train+val 합쳐 보내는 것과 align.
    # 이전 mini test = train only 240 → foundation R²=-3.08 catastrophic (false negative).
    # G-261 (2026-06-13): 'chronos' 제거 — Chronos retire. 'timesfm'/'tirex' 추가 (현 foundation).
    _name_l = (model_name or '').lower()
    if any(k in _name_l for k in ('timesfm', 'tirex', 'foundation', 'overseas', '-pf')):
        X_train = np.vstack([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])

    t0 = time.time()
    try:
        # ── 1. Instantiation ──
        try:
            model = model_cls()
        except Exception as e:
            result['status'] = 'FAIL'
            result['issues'].append(f'instantiation: {type(e).__name__}: {str(e)[:200]}')
            return result

        # ── 2. Min data check ──
        min_n = getattr(model.meta, 'min_data', 30) if hasattr(model, 'meta') else 30
        if len(y_train) < min_n:
            result['status'] = 'SKIP'
            result['issues'].append(f'meta.min_data={min_n} > train n={len(y_train)}')
            return result

        # ── 3. Fit ──
        # Try fit signatures: (X, y), (X, y, X_test), fit_series(y) for some
        fit_method = None
        try:
            if hasattr(model, 'fit_predict'):
                pred = model.fit_predict(X_train, y_train, X_test)
                fit_method = 'fit_predict'
            elif hasattr(model, 'fit') and hasattr(model, 'predict'):
                # try (X, y) fit
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                fit_method = 'fit+predict'
            elif hasattr(model, 'fit_series') and hasattr(model, 'forecast'):
                model.fit_series(y_train)
                pred = model.forecast(len(y_test))
                fit_method = 'fit_series+forecast'
            else:
                result['status'] = 'FAIL'
                result['issues'].append('no fit/predict method found')
                return result
        except Exception as e:
            result['status'] = 'FAIL'
            result['issues'].append(f'{fit_method or "fit"}: {type(e).__name__}: {str(e)[:300]}')
            # Common patterns → fix hints
            err_str = str(e).lower()
            if 'shape' in err_str or 'broadcast' in err_str:
                result['fix_hints'].append('shape mismatch — _validate_shapes 적용 필요')
            elif 'cuda' in err_str or 'mps' in err_str or 'device' in err_str:
                result['fix_hints'].append('device mismatch — pick_device() 검증')
            elif 'nan' in err_str or 'inf' in err_str:
                result['fix_hints'].append('numerical instability — sanitize_predictions 또는 transform clip')
            elif 'memory' in err_str or 'oom' in err_str:
                result['fix_hints'].append('OOM — batch_size 축소 또는 subprocess isolation')
            return result

        result['time_sec'] = time.time() - t0
        if result['time_sec'] > timeout_sec:
            result['status'] = 'WARN'
            result['issues'].append(f'time {result["time_sec"]:.1f}s > {timeout_sec}s budget')

        # ── 4. Pred 기본 검증 ──
        pred = np.asarray(pred, dtype=float).flatten()

        # 4-1. shape
        if len(pred) != len(y_test):
            result['issues'].append(f'pred shape mismatch: {len(pred)} vs y_test {len(y_test)}')
            result['fix_hints'].append('predict 반환 shape 검증 필요')
            # try to align
            pred = pred[:len(y_test)] if len(pred) > len(y_test) else \
                   np.concatenate([pred, np.full(len(y_test) - len(pred), pred[-1] if len(pred) > 0 else 0)])

        # 4-2. NaN/Inf
        n_nan = int(np.sum(np.isnan(pred)))
        n_inf = int(np.sum(np.isinf(pred)))
        if n_nan > 0:
            result['issues'].append(f'pred NaN: {n_nan}/{len(pred)}')
            result['fix_hints'].append('sanitize_predictions 미적용 — base.py:fit_predict 검증')
        if n_inf > 0:
            result['issues'].append(f'pred Inf: {n_inf}/{len(pred)}')
            result['fix_hints'].append('transform inverse 발산 — log1p cap 또는 clip')

        # 4-3. Negative pred (ILI ≥ 0 도메인 제약)
        n_neg = int(np.sum(pred < 0))
        if n_neg > 0:
            result['issues'].append(f'negative pred: {n_neg}/{len(pred)}')
            result['fix_hints'].append('clip_nonneg=True flag 필요 (G-180 P2)')

        # 4-4. Pred 발산 (vs y_test scale)
        pred_clean = pred[~(np.isnan(pred) | np.isinf(pred))]
        if len(pred_clean) > 0:
            y_max = max(y_test.max(), y_train.max())
            pred_max = pred_clean.max()
            if pred_max > 5 * y_max:
                result['issues'].append(f'pred 발산: max={pred_max:.1f} >> y_max={y_max:.1f} (5×)')
                result['fix_hints'].append('학습 발산 — transform/scaler/α-blend 검증 (G-180/G-181)')

        # ── 5. Metrics ──
        valid_mask = np.isfinite(pred) & np.isfinite(y_test)
        if valid_mask.sum() >= 3:
            yp = pred[valid_mask]
            yt = y_test[valid_mask]
            ss_res = float(((yt - yp) ** 2).sum())
            ss_tot = float(((yt - yt.mean()) ** 2).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
            mape = float(np.mean(np.abs((yt - yp) / np.maximum(yt, 0.01))) * 100)
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            mae = float(np.mean(np.abs(yt - yp)))
            result['metrics'] = {
                'r2': r2, 'mape': mape, 'rmse': rmse, 'mae': mae,
                'pred_mean': float(pred_clean.mean()) if len(pred_clean) else None,
                'pred_max': float(pred_clean.max()) if len(pred_clean) else None,
                'pred_min': float(pred_clean.min()) if len(pred_clean) else None,
            }
            # Catastrophic check
            if r2 < 0:
                result['issues'].append(f'catastrophic R²={r2:.2f} on synthetic')
                result['fix_hints'].append('synthetic 에서도 발산 = 모델 자체 문제 (α-blend / transform / scaler 재검토)')
        else:
            result['issues'].append(f'insufficient valid pred: {valid_mask.sum()}/{len(pred)}')

        # ── 6. Status 결정 ──
        if not result['issues']:
            result['status'] = 'PASS'
        elif n_nan == 0 and n_inf == 0 and n_neg == 0 and result['metrics'].get('r2', -999) >= 0:
            result['status'] = 'WARN'
        else:
            result['status'] = 'FAIL'

    except Exception as e:
        result['status'] = 'FAIL'
        result['issues'].append(f'unexpected: {type(e).__name__}: {str(e)[:200]}')
    finally:
        # cleanup
        try:
            del model
        except Exception:
            pass
        import gc; gc.collect()
        result['time_sec'] = time.time() - t0

    return result
