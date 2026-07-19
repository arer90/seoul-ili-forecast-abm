"""
simulation/models/modern_ts/nbeats.py
=====================================
N-BEATS (Neural Basis Expansion) -- Level 11

Interpretable configuration: Trend stack (polynomial) + Seasonal stack (Fourier).
Each stack has 2 blocks, hidden=64. Stable even with small samples (~341 weeks).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

__all__ = ["NBEATSForecaster"]


class _NBEATSNet:
    """N-BEATS: interpretable configuration (Trend + Seasonal stacks)."""

    @staticmethod
    def build(n_features: int, lookback: int = 12, hidden_size: int = 64,
              n_blocks: int = 2, dropout: float = 0.2,
              activation: str = "gelu", init: str = "default"):
        """(P0-D): hidden/dropout/n_blocks/activation/init 을 hp 로 수신."""
        import torch
        import torch.nn as nn
        from simulation.models.dl_models import (
            _get_activation_fn, _apply_weight_init,
        )

        class _BasisBlock(nn.Module):
            """Single N-BEATS block with basis expansion."""

            def __init__(self, input_dim: int, hidden: int,
                         basis_type: str, n_theta: int):
                super().__init__()
                self.basis_type = basis_type
                self.n_theta = n_theta
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    _get_activation_fn(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    _get_activation_fn(activation),
                    nn.Dropout(dropout),
                )
                # theta for backcast and forecast
                self.theta_b = nn.Linear(hidden, n_theta)
                self.theta_f = nn.Linear(hidden, n_theta)

            def forward(self, x, T_back: int, T_fore: int):
                h = self.fc(x)
                tb = self.theta_b(h)  # (batch, n_theta)
                tf = self.theta_f(h)  # (batch, n_theta)
                backcast = self._basis_expand(tb, T_back)
                forecast = self._basis_expand(tf, T_fore)
                return backcast, forecast

            def _basis_expand(self, theta, T: int):
                """Expand theta coefficients using basis functions."""
                device = theta.device
                batch = theta.size(0)
                t = torch.linspace(0, 1, T, device=device).unsqueeze(0)  # (1, T)

                if self.basis_type == "trend":
                    # Polynomial basis: [1, t, t^2, ...]
                    degree = self.n_theta
                    basis = torch.stack(
                        [t.squeeze(0) ** i for i in range(degree)], dim=0
                    )  # (n_theta, T)
                    return torch.matmul(theta, basis)  # (batch, T)

                elif self.basis_type == "seasonal":
                    # Fourier basis: sin/cos pairs, period=52 weeks
                    K = self.n_theta // 2
                    basis_list = []
                    for k in range(1, K + 1):
                        basis_list.append(
                            torch.cos(2 * math.pi * k * t.squeeze(0) / 1.0)
                        )
                        basis_list.append(
                            torch.sin(2 * math.pi * k * t.squeeze(0) / 1.0)
                        )
                    basis = torch.stack(basis_list[:self.n_theta], dim=0)
                    return torch.matmul(theta, basis)

                else:
                    # Generic linear
                    proj = nn.Linear(self.n_theta, T).to(device)
                    return proj(theta)

        class NBEATSModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Feature projection: (batch, lookback, n_features) → (batch, lookback)
                self.feat_proj = nn.Linear(n_features, 1)

                input_dim = lookback

                # Trend stack: n_blocks, polynomial degree=3 → 3 theta coeffs
                self.trend_blocks = nn.ModuleList([
                    _BasisBlock(input_dim, hidden_size, "trend", n_theta=3)
                    for _ in range(n_blocks)
                ])

                # Seasonal stack: n_blocks, K=5 harmonics → 10 theta coeffs
                self.seasonal_blocks = nn.ModuleList([
                    _BasisBlock(input_dim, hidden_size, "seasonal", n_theta=10)
                    for _ in range(n_blocks)
                ])

                self.forecast_len = 1  # single-step forecast

            def forward(self, x):
                # x: (batch, lookback, n_features)
                # Project features to univariate
                x_proj = self.feat_proj(x).squeeze(-1)  # (batch, lookback)

                residual = x_proj
                forecast_total = torch.zeros(
                    x_proj.size(0), self.forecast_len, device=x.device
                )

                # Trend stack
                for block in self.trend_blocks:
                    backcast, forecast = block(
                        residual, T_back=x_proj.size(1), T_fore=self.forecast_len
                    )
                    residual = residual - backcast
                    forecast_total = forecast_total + forecast

                # Seasonal stack
                for block in self.seasonal_blocks:
                    backcast, forecast = block(
                        residual, T_back=x_proj.size(1), T_fore=self.forecast_len
                    )
                    residual = residual - backcast
                    forecast_total = forecast_total + forecast

                return forecast_total  # (batch, 1)

        m = NBEATSModel()
        _apply_weight_init(m, init)
        return m


class NBEATSForecaster(BaseForecaster):
    """N-BEATS -- Neural Basis Expansion Analysis for Time Series.

    Interpretable 구성: Trend stack (polynomial) + Seasonal stack (Fourier).
    각 stack 2 blocks, hidden=64. 소표본(~341주)에서도 안정적.
    """

    meta = ModelMeta(
        name="N-BEATS",
        category="dl",
        level=11,
        min_data=100,
        description="N-BEATS. interpretable Trend+Seasonal basis expansion. 소표본 안정적.",
        dependencies=["torch"],
    )

    SEQ_LEN = 12

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> NBEATSForecaster:
        """(P0-D): Optuna HPO 내장."""
        from sklearn.preprocessing import StandardScaler
        from simulation.models.dl_models import _make_sequences, _train_loop
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop,
            UNIT_MIN_DEFAULT, UNIT_MAX_TS_DL, INITS, NORMS,
        )

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap

        X_seq, y_seq = _make_sequences(X_s, y_s, self.SEQ_LEN)
        n_features = X_s.shape[1]

        n_trials = get_trials("N-BEATS", default=0)

        def _static_defaults():
            self._model = _NBEATSNet.build(
                n_features=n_features,
                lookback=self.SEQ_LEN,
                hidden_size=64,
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=400, lr=5e-4, patience=35,
                weight_decay=1e-4,
                augment=True, augment_factor=3,
            )

        def _objective(trial):
            import gc
            # FIX: N-BEATS stacks hidden × n_blocks × lookback → VRAM-safe cap 512.
            hidden = trial.suggest_int("hidden", UNIT_MIN_DEFAULT, UNIT_MAX_TS_DL, log=True)
            n_blocks = trial.suggest_int("n_blocks", 1, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            init_t = trial.suggest_categorical("init", INITS)
            tr = suggest_training_hp(trial)
            model = _NBEATSNet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                hidden_size=hidden, n_blocks=n_blocks,
                dropout=dropout, activation="gelu", init=init_t,
            )
            neg_r2 = _train_loop(
                model, X_seq, y_seq,
                epochs=150, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=25, weight_decay=tr["weight_decay"],
                augment=True, augment_factor=tr["augment_factor"],
                trial=trial,
                optimizer_type=tr["optimizer"],
                loss_type=tr["loss"],
                return_r2=True,
            )
            del model
            gc.collect()
            return neg_r2

        best, _ = run_optuna_loop("N-BEATS", _objective, n_trials, _static_defaults)
        if best:
            self._model = _NBEATSNet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                hidden_size=best.get("hidden", 64),
                n_blocks=best.get("n_blocks", 2),
                dropout=best.get("dropout", 0.2),
                activation="gelu", init=best.get("init", "default"),
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=400, lr=best.get("lr", 5e-4),
                batch_size=best.get("batch_size", 32),
                patience=35, weight_decay=best.get("weight_decay", 1e-4),
                augment=True, augment_factor=best.get("augment_factor", 3),
                optimizer_type=best.get("optimizer", "adamw"),
                loss_type=best.get("loss", "mse"),
            )
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.dl_models import _predict_torch

        X_s = self._scaler_X.transform(X_test)
        preds = []
        for i in range(len(X_s)):
            start = max(0, i + 1 - self.SEQ_LEN)
            seq = X_s[start:i + 1]
            if len(seq) < self.SEQ_LEN:
                pad = np.zeros((self.SEQ_LEN - len(seq), X_s.shape[1]))
                seq = np.vstack([pad, seq])
            preds.append(seq)
        X_seq = np.array(preds)
        pred_s = _predict_torch(self._model, X_seq)
        _pred = np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )
        from simulation.models.safety import apply_extrapolation_cap  # G-289
        return apply_extrapolation_cap(_pred, getattr(self, "_y_train_max", None))
