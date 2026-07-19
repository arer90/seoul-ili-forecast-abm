"""
simulation/models/modern_ts/mamba.py
====================================
Mamba (Selective State Space Model) -- Level 17 [실험적]

Pure PyTorch S4/Mamba implementation without external dependencies.
Input-dependent B, C matrices + SiLU gating.
d_model=32, state_dim=16, 2 layers.
"""

from __future__ import annotations

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

__all__ = ["MambaForecaster"]


class _SimpleMambaNet:
    """Simplified selective state space model. Pure PyTorch, no mamba-ssm."""

    @staticmethod
    def build(n_features: int, seq_len: int = 12,
              d_model: int = 32, state_dim: int = 16,
              n_layers: int = 2, dropout: float = 0.3,
              init: str = "default"):
        """(P0-D): d_model / state_dim / n_layers / dropout / init 을 hp 로 수신."""
        import torch
        import torch.nn as nn
        from simulation.models.dl_models import _apply_weight_init

        class _MambaLayer(nn.Module):
            """Single selective state space layer."""

            def __init__(self, d_in: int, d_state: int, drop: float):
                super().__init__()
                self.d_state = d_state

                # Input projection
                self.in_proj = nn.Linear(d_in, d_in * 2)

                # State space parameters
                # A: diagonal, log-parameterized for stability
                self.A_log = nn.Parameter(
                    torch.log(torch.randn(d_in, d_state).abs() + 1e-4)
                )
                self.D = nn.Parameter(torch.ones(d_in))

                # Selective mechanism: input-dependent B, C, dt
                self.B_proj = nn.Linear(d_in, d_state)
                self.C_proj = nn.Linear(d_in, d_state)
                self.dt_proj = nn.Linear(d_in, d_in)

                self.out_proj = nn.Linear(d_in, d_in)
                self.dropout = nn.Dropout(drop)
                self.norm = nn.LayerNorm(d_in)

            def forward(self, x):
                # x: (batch, seq_len, d_in)
                residual = x
                x = self.norm(x)
                batch, L, d = x.shape

                # Gate + input
                xz = self.in_proj(x)  # (batch, L, 2*d)
                x_in, z = xz.chunk(2, dim=-1)

                # Selective parameters
                A = -torch.exp(self.A_log)  # (d, d_state)
                B = self.B_proj(x_in)  # (batch, L, d_state)
                C = self.C_proj(x_in)  # (batch, L, d_state)
                dt = torch.nn.functional.softplus(self.dt_proj(x_in))  # (batch, L, d)

                # Discretize A, B using zero-order hold
                # dA = exp(dt * A), dB = dt * B
                # Scan through sequence
                y_list = []
                h = torch.zeros(batch, d, self.d_state, device=x.device)

                for t in range(L):
                    dt_t = dt[:, t, :].unsqueeze(-1)  # (batch, d, 1)
                    A_disc = torch.exp(dt_t * A.unsqueeze(0))  # (batch, d, d_state)
                    B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
                    x_t = x_in[:, t, :].unsqueeze(-1)  # (batch, d, 1)
                    dB = dt_t * B_t * x_t.expand_as(B_t.expand(-1, d, -1))

                    h = A_disc * h + dB
                    C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
                    y_t = (h * C_t).sum(dim=-1)  # (batch, d)
                    y_list.append(y_t)

                y = torch.stack(y_list, dim=1)  # (batch, L, d)

                # Gating with SiLU
                y = y * torch.nn.functional.silu(z)

                # Skip connection + D
                y = y + x_in * self.D.unsqueeze(0).unsqueeze(0)

                out = self.out_proj(y)
                out = self.dropout(out)
                return out + residual

        class MambaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                self.layers = nn.ModuleList([
                    _MambaLayer(d_model, state_dim, dropout)
                    for _ in range(n_layers)
                ])
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 1),
                )

            def forward(self, x):
                # x: (batch, seq_len, n_features)
                h = self.input_proj(x)  # (batch, seq_len, d_model)
                for layer in self.layers:
                    h = layer(h)
                # Take last timestep
                return self.head(h[:, -1, :])

        m = MambaModel()
        _apply_weight_init(m, init)
        return m


class MambaForecaster(BaseForecaster):
    """Mamba -- Selective State Space Model.

    [실험적] Pure PyTorch S4/Mamba 구현 (외부 의존성 없음).
    입력 의존적 B, C 행렬 + SiLU gating.
    d_model=32, state_dim=16, 2 layers.
    """

    meta = ModelMeta(
        name="Mamba",
        category="dl",
        level=17,
        min_data=120,
        description="[실험적] Mamba. 선택적 상태공간 모델. 순수 PyTorch 구현, 외부 의존성 없음.",
        dependencies=["torch"],
    )

    SEQ_LEN = 12

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> MambaForecaster:
        """(P0-D): Optuna HPO 내장.

 Python-loop scan 으로 sequence 를 도는 순수 PyTorch 구현이라
 seq_len·d_model·state_dim 에 대해 시간이 선형으로 증가한다.
 → d_model ≤ 64, state_dim ≤ 32 로 search 범위 제한.
 """
        from sklearn.preprocessing import StandardScaler
        from simulation.models.dl_models import _make_sequences, _train_loop, lag_backbone_seq
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop, INITS,
        )

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-281: 외삽 cap

        # G-319d: lag backbone(과거 y) 입력으로 입력버그 회복(n_features=1 closure 전파, lookback 불변).
        _lag_seq, self._lag_idx = lag_backbone_seq(X_s, kwargs.get("feature_names"), self.SEQ_LEN)
        if self._lag_idx is not None:
            X_seq, y_seq, n_features = _lag_seq, y_s, 1
        else:
            X_seq, y_seq = _make_sequences(X_s, y_s, self.SEQ_LEN)
            n_features = X_s.shape[1]
        n_trials = get_trials("Mamba", default=0)

        def _static_defaults():
            self._model = _SimpleMambaNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=32, state_dim=16,
                n_layers=2, dropout=0.3, init="default",
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=300, lr=5e-4, patience=30,
                weight_decay=5e-4,
                augment=True, augment_factor=3,
            )

        def _objective(trial):
            import gc
            d_model = trial.suggest_int("d_model", 8, 64, log=True)
            state_dim = trial.suggest_int("state_dim", 4, 32, log=True)
            n_layers = trial.suggest_int("n_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            init_t = trial.suggest_categorical("init", INITS)
            tr = suggest_training_hp(trial)

            model = _SimpleMambaNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=d_model, state_dim=state_dim,
                n_layers=n_layers, dropout=dropout, init=init_t,
            )
            neg_r2 = _train_loop(
                model, X_seq, y_seq,
                epochs=120, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=25, weight_decay=tr["weight_decay"],
                augment=True, augment_factor=tr["augment_factor"],
                trial=trial,
                optimizer_type=tr["optimizer"], loss_type=tr["loss"],
                return_r2=True,
            )
            del model
            gc.collect()
            return neg_r2

        best, _ = run_optuna_loop("Mamba", _objective, n_trials, _static_defaults)
        if best:
            self._model = _SimpleMambaNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=best.get("d_model", 32),
                state_dim=best.get("state_dim", 16),
                n_layers=best.get("n_layers", 2),
                dropout=best.get("dropout", 0.3),
                init=best.get("init", "default"),
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=300, lr=best.get("lr", 5e-4),
                batch_size=best.get("batch_size", 32),
                patience=30, weight_decay=best.get("weight_decay", 5e-4),
                augment=True, augment_factor=best.get("augment_factor", 3),
                optimizer_type=best.get("optimizer", "adamw"),
                loss_type=best.get("loss", "mse"),
            )
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.dl_models import _predict_torch, lag_backbone_from_idx

        X_s = self._scaler_X.transform(X_test)
        if getattr(self, "_lag_idx", None) is not None:
            X_seq = lag_backbone_from_idx(X_s, self._lag_idx, self.SEQ_LEN)  # G-319d
        else:
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
        pred = np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )
        # G-281 (3자 감사): outbreak 외삽 cap — DNN/TCN/GCN 은 모두 y_train_max×1.5 cap 보유,
        #   Mamba 만 누락이었음(under-identified SSM 외삽 폭주 가드). y_train 파생=누수 0.
        _ymax = getattr(self, "_y_train_max", None)
        if _ymax is not None and _ymax > 0:
            pred = np.minimum(pred, max(_ymax * 1.5, 100.0))
        return pred
