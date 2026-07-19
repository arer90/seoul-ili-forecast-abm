"""Synthetic time-series + features for mini test.

ILI rate 처럼 trend + seasonal + noise + lag1 reference 가 있는 작은 dataset.
"""
from __future__ import annotations
import numpy as np


def make_synthetic(n_train: int = 240, n_val: int = 30, n_test: int = 50,
                   n_features: int = 20, seed: int = 42) -> dict:
    """240/30/50 split 의 synthetic ILI-like data (실제 242/27/68 근사).

    Returns:
        {
          'X_train', 'X_val', 'X_test': (n_*, n_features) — feature[0]=lag1 (reference)
          'y_train', 'y_val', 'y_test': (n_*,) — ILI rate ≥ 0
          'feature_names': [str]
        }
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test

    # ILI rate: trend + seasonal + noise (always ≥ 0)
    t = np.arange(n_total)
    trend = 5.0 + 0.05 * t  # slow rise
    season = 8.0 * np.sin(2 * np.pi * t / 52.0)  # 52-week season
    noise = rng.normal(0, 1.5, n_total)
    y = np.maximum(0.5, trend + season + noise)

    # Features: lag1 (reference) + 19 random features (some correlated with y)
    X = np.zeros((n_total, n_features), dtype=np.float32)
    # reference at idx 0 = lag1
    X[1:, 0] = y[:-1]
    X[0, 0] = y[0]  # initial fill

    # 5 features correlated with y (lag-2, week_of_year, etc.)
    X[2:, 1] = y[:-2]  # lag2
    X[:, 2] = np.sin(2 * np.pi * t / 52.0)  # sin
    X[:, 3] = np.cos(2 * np.pi * t / 52.0)  # cos
    X[:, 4] = trend / 10.0  # normalized trend
    X[:, 5] = (t % 52) / 52.0  # week_of_year norm

    # 14 random features
    X[:, 6:] = rng.normal(0, 1, (n_total, n_features - 6)).astype(np.float32)

    feature_names = (
        ['ili_rate_lag1', 'ili_rate_lag2', 'season_sin', 'season_cos',
         'trend_norm', 'woy_norm']
        + [f'noise_{i}' for i in range(n_features - 6)]
    )

    return {
        'X_train': X[:n_train].astype(np.float32),
        'X_val':   X[n_train:n_train+n_val].astype(np.float32),
        'X_test':  X[n_train+n_val:].astype(np.float32),
        'y_train': y[:n_train].astype(np.float32),
        'y_val':   y[n_train:n_train+n_val].astype(np.float32),
        'y_test':  y[n_train+n_val:].astype(np.float32),
        'feature_names': feature_names,
        'meta': {'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
                 'n_features': n_features, 'seed': seed,
                 'y_range': [float(y.min()), float(y.max())]},
    }
