"""
simulation/models/modern_ts/nhits.py
====================================
N-HiTS (Neural Hierarchical Interpolation for Time Series) -- Level 12

Multi-scale hierarchical interpolation with 3 stacks (pool_sizes=[1, 4, 12]).
More memory efficient than N-BEATS, better for longer horizons.
"""

from __future__ import annotations

import math

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

__all__ = ["NHiTSForecaster"]


class _NHiTSNet:
    """N-HiTS: multi-scale hierarchical interpolation."""

    @staticmethod
    def build(n_features: int, lookback: int = 12, hidden_size: int = 64,
              dropout: float = 0.2, activation: str = "gelu",
              init: str = "default"):
        """(P0-D): hidden/dropout/activation/init 도 hp 로 수신."""
        import torch
        import torch.nn as nn
        from simulation.models.dl_models import (
            _get_activation_fn, _apply_weight_init,
        )

        pool_sizes = [1, 4, 12]

        class _HiTSStack(nn.Module):
            """Single N-HiTS stack with pooling + interpolation."""

            def __init__(self, input_dim: int, pool_k: int, hidden: int,
                         n_theta: int, basis_type: str):
                super().__init__()
                self.pool_k = pool_k
                self.basis_type = basis_type
                self.n_theta = n_theta

                pooled_len = max(1, input_dim // pool_k)
                self.pool = nn.AdaptiveMaxPool1d(pooled_len) if pool_k > 1 else nn.Identity()

                fc_input = pooled_len
                self.fc = nn.Sequential(
                    nn.Linear(fc_input, hidden),
                    _get_activation_fn(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    _get_activation_fn(activation),
                    nn.Dropout(dropout),
                )
                self.theta_b = nn.Linear(hidden, n_theta)
                self.theta_f = nn.Linear(hidden, n_theta)

            def forward(self, x, T_back: int, T_fore: int):
                # x: (batch, lookback)
                if self.pool_k > 1:
                    # Pool: (batch, 1, lookback) → (batch, 1, pooled)
                    xp = self.pool(x.unsqueeze(1)).squeeze(1)
                else:
                    xp = x

                h = self.fc(xp)
                tb = self.theta_b(h)
                tf = self.theta_f(h)

                backcast = self._expand(tb, T_back)
                forecast = self._expand(tf, T_fore)
                return backcast, forecast

            def _expand(self, theta, T: int):
                device = theta.device
                t = torch.linspace(0, 1, T, device=device).unsqueeze(0)
                if self.basis_type == "trend":
                    basis = torch.stack(
                        [t.squeeze(0) ** i for i in range(self.n_theta)], dim=0
                    )
                    return torch.matmul(theta, basis)
                else:
                    K = self.n_theta // 2
                    basis_list = []
                    for k in range(1, K + 1):
                        basis_list.append(torch.cos(2 * math.pi * k * t.squeeze(0)))
                        basis_list.append(torch.sin(2 * math.pi * k * t.squeeze(0)))
                    basis = torch.stack(basis_list[:self.n_theta], dim=0)
                    return torch.matmul(theta, basis)

        class NHiTSModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.feat_proj = nn.Linear(n_features, 1)
                self.forecast_len = 1

                # 3 stacks with different pooling scales
                self.stacks = nn.ModuleList([
                    _HiTSStack(lookback, pool_sizes[0], hidden_size,
                               n_theta=3, basis_type="trend"),
                    _HiTSStack(lookback, pool_sizes[1], hidden_size,
                               n_theta=10, basis_type="seasonal"),
                    _HiTSStack(lookback, pool_sizes[2], hidden_size,
                               n_theta=3, basis_type="trend"),
                ])

            def forward(self, x):
                x_proj = self.feat_proj(x).squeeze(-1)
                residual = x_proj
                forecast_total = torch.zeros(
                    x_proj.size(0), self.forecast_len, device=x.device
                )

                for stack in self.stacks:
                    backcast, forecast = stack(
                        residual, T_back=x_proj.size(1), T_fore=self.forecast_len
                    )
                    residual = residual - backcast
                    forecast_total = forecast_total + forecast

                return forecast_total

        m = NHiTSModel()
        _apply_weight_init(m, init)
        return m


class NHiTSForecaster(BaseForecaster):
    """N-HiTS -- Neural Hierarchical Interpolation for Time Series.

    3 stacks (pool_sizes=[1, 4, 12]) -- 다중 스케일로 계절성/추세 분리.
    N-BEATS 대비 메모리 효율적, 긴 horizon 예측에 유리.
    """

    meta = ModelMeta(
        name="N-HiTS",
        category="dl",
        level=12,
        min_data=100,
        description="N-HiTS. 다중 스케일 pooling으로 계층적 시계열 분해. N-BEATS 대비 효율적.",
        dependencies=["torch"],
    )

    SEQ_LEN = 12

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> NHiTSForecaster:
        """(P0-D): Optuna HPO 내장."""
        from sklearn.preprocessing import StandardScaler
        from simulation.models.dl_models import _make_sequences, _train_loop
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop,
            UNIT_MIN_DEFAULT, UNIT_MAX_TS_DL, INITS,
        )

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap

        X_seq, y_seq = _make_sequences(X_s, y_s, self.SEQ_LEN)
        n_features = X_s.shape[1]
        n_trials = get_trials("N-HiTS", default=0)

        def _static_defaults():
            self._model = _NHiTSNet.build(n_features=n_features,
                                          lookback=self.SEQ_LEN, hidden_size=64)
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=400, lr=5e-4, patience=35,
                weight_decay=1e-4, augment=True, augment_factor=3,
            )

        def _objective(trial):
            import gc
            # FIX: N-HiTS hierarchical blocks → hidden × stack × lookback OOM. cap 512.
            hidden = trial.suggest_int("hidden", UNIT_MIN_DEFAULT, UNIT_MAX_TS_DL, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            init_t = trial.suggest_categorical("init", INITS)
            tr = suggest_training_hp(trial)
            model = _NHiTSNet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                hidden_size=hidden, dropout=dropout,
                activation="gelu", init=init_t,
            )
            neg_r2 = _train_loop(
                model, X_seq, y_seq,
                epochs=150, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=25, weight_decay=tr["weight_decay"],
                augment=True, augment_factor=tr["augment_factor"],
                trial=trial,
                optimizer_type=tr["optimizer"], loss_type=tr["loss"],
                return_r2=True,
            )
            del model
            gc.collect()
            return neg_r2

        best, _ = run_optuna_loop("N-HiTS", _objective, n_trials, _static_defaults)
        if best:
            self._model = _NHiTSNet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                hidden_size=best.get("hidden", 64),
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
