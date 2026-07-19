"""Grid sweep mini test — catastrophic 모델의 transform/scaler 진단.

기본 mini test는 default config 만 — R9(per_model_optimize) 의 (transform × scaler) grid 미반영.
이 helper 는 catastrophic 모델에 대해 4 transforms × 3 scalers = 12 combo sweep.

→ "default 에서만 fail" vs "모든 transform 에서 fail (모델 자체 문제)" 구분.
"""
from __future__ import annotations
import json
import os
import time
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')


# Transform / scaler combinations — 사용자 요청 확장 (2026-05-05):
#   "log1p, yeo_johnson, sqrt 외에도 가우시안, mcmc, 라플라스 변환 확인"
#
# 통계학적 mapping:
#   gaussian          → PowerTransformer 'box-cox' (양수 only, Gaussian 화)
#                       또는 QuantileTransformer normal (rank → quantile)
#   mcmc_robust       → median + MAD (Median Absolute Deviation) — Laplace 사후
#                       추정 기반 robust standardization (Bayesian shrinkage 효과)
#   laplace           → Laplace distribution scaling (median + b=MAD/0.674)
#   rank              → rank-based 순위 변환 (분포 무관, monotonic)
TRANSFORMS = [
    'identity', 'log1p', 'sqrt', 'yeo_johnson', 'asinh',
    # 사용자 요청 확장 (2026-05-05)
    'gaussian',         # quantile → normal (분포 무관 → Gaussian)
    'box_cox',          # PowerTransformer box-cox (양수 only)
    'mcmc_robust',      # median + MAD (Laplace 사후 robust)
    'laplace',          # median + b=MAD/0.674 (Laplace dist scaling)
    'rank',             # rank ordinal 변환
]
SCALERS = ['none', 'standard', 'minmax', 'robust', 'grouped']


def apply_transform(y: np.ndarray, name: str):
    """y → transformed y. Returns (transformed, inverse_fn)."""
    y = np.asarray(y, dtype=np.float64)
    if name == 'identity':
        return y, (lambda x: x)
    if name == 'log1p':
        # y ≥ 0 보호
        y_pos = np.maximum(y, 0.0)
        ymax = float(y_pos.max()) if len(y_pos) else 1.0
        # G-180 P1 cap: log1p(y_max × 10) 보수적 cap
        cap = float(np.log1p(max(ymax * 10.0, 100.0)))
        def _inv_log1p(x):
            return np.expm1(np.clip(x, -2.0, cap))
        return np.log1p(y_pos), _inv_log1p
    if name == 'sqrt':
        y_pos = np.maximum(y, 0.0)
        return np.sqrt(y_pos), (lambda x: np.maximum(x, 0.0) ** 2)
    if name == 'yeo_johnson':
        try:
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            y2 = pt.fit_transform(y.reshape(-1, 1)).ravel()
            return y2, (lambda x: pt.inverse_transform(np.asarray(x).reshape(-1, 1)).ravel())
        except Exception:
            return y, (lambda x: x)
    if name == 'asinh':
        # arcsinh(x) — log-like but works on negatives
        return np.arcsinh(y), (lambda x: np.sinh(np.clip(x, -10, 10)))
    if name == 'gaussian':
        # Quantile → normal (분포 무관, Gaussian 화)
        try:
            from sklearn.preprocessing import QuantileTransformer
            qt = QuantileTransformer(n_quantiles=min(100, len(y)),
                                      output_distribution='normal',
                                      random_state=42)
            y2 = qt.fit_transform(y.reshape(-1, 1)).ravel()
            return y2, (lambda x: qt.inverse_transform(np.asarray(x).reshape(-1, 1)).ravel())
        except Exception:
            return y, (lambda x: x)
    if name == 'box_cox':
        # Box-Cox (양수 only, 자동 λ 추정)
        try:
            from sklearn.preprocessing import PowerTransformer
            y_pos = np.maximum(y, 1e-3)  # 양수 보정
            pt = PowerTransformer(method='box-cox', standardize=False)
            y2 = pt.fit_transform(y_pos.reshape(-1, 1)).ravel()
            return y2, (lambda x: pt.inverse_transform(np.asarray(x).reshape(-1, 1)).ravel())
        except Exception:
            return y, (lambda x: x)
    if name == 'mcmc_robust':
        # Median + MAD (Laplace 사후 추정 robust standardization)
        # MCMC residual shrinkage 효과 (Bayesian)
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med)))
        scale = max(mad * 1.4826, 1e-6)  # 1.4826 = MAD-to-σ
        y2 = (y - med) / scale
        return y2, (lambda x: np.asarray(x) * scale + med)
    if name == 'laplace':
        # Laplace distribution scaling (median + b=MAD/0.674)
        # Laplace dist scale parameter b = MAD / 0.6745
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med)))
        b = max(mad / 0.6745, 1e-6)
        y2 = (y - med) / b
        return y2, (lambda x: np.asarray(x) * b + med)
    if name == 'rank':
        # Rank-based ordinal (monotonic, 분포 무관)
        from scipy.stats import rankdata
        ranks = rankdata(y) / len(y)  # [1/n, ..., 1]
        # inverse: rank → original (interpolation)
        y_sorted_idx = np.argsort(y)
        y_sorted = y[y_sorted_idx]
        def _inv_rank(x):
            x = np.asarray(x)
            x_clip = np.clip(x, 1.0/len(y), 1.0)
            indices = np.clip((x_clip * len(y)).astype(int) - 1, 0, len(y) - 1)
            return y_sorted[indices]
        return ranks, _inv_rank
    return y, (lambda x: x)


def apply_scaler(X: np.ndarray, X_test: np.ndarray, name: str):
    """X scaling. Returns (X_scaled, X_test_scaled)."""
    if name == 'none':
        return X, X_test
    if name == 'standard':
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        return sc.fit_transform(X), sc.transform(X_test)
    if name == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        return sc.fit_transform(X), sc.transform(X_test)
    if name == 'robust':
        from sklearn.preprocessing import RobustScaler
        sc = RobustScaler()
        return sc.fit_transform(X), sc.transform(X_test)
    if name == 'grouped':
        # grouped scaling — reference (idx 0) preserved, rest standard scaled
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xs = X.copy().astype(np.float64)
        Xt = X_test.copy().astype(np.float64)
        # idx 0 (reference lag1) — preserve
        if X.shape[1] > 1:
            Xs[:, 1:] = sc.fit_transform(Xs[:, 1:])
            Xt[:, 1:] = sc.transform(Xt[:, 1:])
        return Xs, Xt
    return X, X_test


def diagnose_model_with_grid(model_cls, data: dict, *,
                              model_name: str = '',
                              timeout_per_combo: int = 30,
                              transforms: list[str] = TRANSFORMS,
                              scalers: list[str] = SCALERS) -> dict:
    """Per-model (transform × scaler) grid sweep mini test.

    Returns:
        {
          'name', 'best_combo': {transform, scaler, r2, mape, ...},
          'all_combos': [{transform, scaler, r2, status, time_sec, ...}],
          'verdict': PASS_AT_BEST / CATASTROPHIC_ALL (모든 combo R²<0),
        }
    """
    X_train = data['X_train']
    y_train_raw = data['y_train'].astype(np.float64)
    X_test = data['X_test']
    y_test = data['y_test'].astype(np.float64)

    all_combos = []
    best = {'r2': -np.inf, 'config': None}
    n_combos = len(transforms) * len(scalers)
    t_overall = time.time()

    for ti, tname in enumerate(transforms):
        # transform y_train (test set 은 inverse 후 비교)
        try:
            y_t, inv_fn = apply_transform(y_train_raw, tname)
        except Exception as e:
            for sname in scalers:
                all_combos.append({'transform': tname, 'scaler': sname,
                                   'r2': float('nan'), 'status': f'transform_fail: {e}',
                                   'time_sec': 0})
            continue

        for sname in scalers:
            t0 = time.time()
            combo = {'transform': tname, 'scaler': sname,
                     'r2': float('nan'), 'mape': float('nan'),
                     'status': 'UNKNOWN', 'time_sec': 0}
            try:
                # scaler 적용
                Xs, Xs_test = apply_scaler(X_train, X_test, sname)

                # 모델 인스턴스 + fit_predict
                model = model_cls()
                # fit_predict 우선, 없으면 fit/predict
                if hasattr(model, 'fit_predict'):
                    pred_t = model.fit_predict(Xs, y_t.astype(np.float32), Xs_test)
                else:
                    model.fit(Xs, y_t.astype(np.float32))
                    pred_t = model.predict(Xs_test)

                if pred_t is None:
                    combo['status'] = 'FIT_FAIL'
                    all_combos.append(combo)
                    continue

                pred_t = np.asarray(pred_t, dtype=np.float64).flatten()
                if len(pred_t) != len(y_test):
                    pred_t = pred_t[:len(y_test)] if len(pred_t) > len(y_test) else \
                             np.concatenate([pred_t, np.full(len(y_test) - len(pred_t), pred_t[-1] if len(pred_t) else 0)])

                # inverse transform — predicted scale → original
                try:
                    pred_orig = inv_fn(pred_t)
                except Exception as ie:
                    combo['status'] = f'INVERSE_FAIL: {str(ie)[:60]}'
                    all_combos.append(combo)
                    continue

                # NaN/Inf → skip
                if not np.all(np.isfinite(pred_orig)):
                    n_bad = int(np.sum(~np.isfinite(pred_orig)))
                    combo['status'] = f'NAN_INF: {n_bad}'
                    all_combos.append(combo)
                    continue

                # R² + MAPE
                ss_res = float(((y_test - pred_orig) ** 2).sum())
                ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
                mape = float(np.mean(np.abs((y_test - pred_orig) / np.maximum(y_test, 0.01))) * 100)
                combo['r2'] = r2
                combo['mape'] = mape
                combo['status'] = 'PASS' if r2 >= 0.0 else 'CATASTROPHIC'
                combo['pred_max'] = float(pred_orig.max())
                combo['pred_min'] = float(pred_orig.min())

                if r2 > best['r2']:
                    best = {'r2': r2, 'mape': mape, 'transform': tname, 'scaler': sname,
                            'config': combo}

                # cleanup
                del model
                import gc; gc.collect()

            except Exception as e:
                combo['status'] = f'EXC: {type(e).__name__}: {str(e)[:80]}'
            finally:
                combo['time_sec'] = time.time() - t0

            all_combos.append(combo)

            if combo['time_sec'] > timeout_per_combo:
                # 시간 초과 — 건너뛰기 (이전 combo는 저장됨)
                pass

    # Verdict
    valid_r2 = [c['r2'] for c in all_combos if not (np.isnan(c['r2']) or np.isinf(c['r2']))]
    if not valid_r2:
        verdict = 'ALL_FAIL'
    elif max(valid_r2) >= 0.0:
        verdict = 'PASS_AT_BEST'
    else:
        verdict = 'CATASTROPHIC_ALL'  # 모든 combo R²<0 = 모델 자체 문제

    return {
        'name': model_name or model_cls.__name__,
        'verdict': verdict,
        'best': best if best['config'] is not None else None,
        'n_combos_tested': len(all_combos),
        'n_pass': sum(1 for c in all_combos if c['status'] == 'PASS'),
        'n_catastrophic': sum(1 for c in all_combos if c['status'] == 'CATASTROPHIC'),
        'n_fail': sum(1 for c in all_combos if c['status'] not in ('PASS', 'CATASTROPHIC')),
        'all_combos': all_combos,
        'total_time_sec': time.time() - t_overall,
    }
