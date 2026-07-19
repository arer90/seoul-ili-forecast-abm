"""
simulation/models/modern_ts/timesnet.py
========================================
TimesNet (Temporal 2D Variation Modeling) -- Level 18 [실험적]

Discovers dominant periods via FFT, reshapes 1D→2D, applies 2D Inception conv.
Top-K=3 periods, d_model=32, 2 layers.
"""

from __future__ import annotations

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

__all__ = ["TimesNetForecaster"]


class _TimesNetNet:
    """TimesNet: convert 1D→2D via FFT-discovered periods, apply 2D conv."""

    @staticmethod
    def build(n_features: int, seq_len: int = 12,
              d_model: int = 32, n_layers: int = 2,
              top_k: int = 3, dropout: float = 0.3,
              init: str = "default"):
        """(P0-D): d_model / n_layers / top_k / dropout / init 을 hp 로 수신."""
        import torch
        import torch.nn as nn
        from simulation.models.dl_models import _apply_weight_init

        class _InceptionBlock(nn.Module):
            """Simplified Inception-like 2D convolution block."""

            def __init__(self, in_ch: int, out_ch: int):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
                self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
                self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
                self.pool_conv = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1),
                )
                self.bn = nn.BatchNorm2d(out_ch * 4)
                self.relu = nn.ReLU()

            def forward(self, x):
                # x: (batch, in_ch, h, w)
                out = torch.cat([
                    self.conv1(x),
                    self.conv3(x),
                    self.conv5(x),
                    self.pool_conv(x),
                ], dim=1)
                return self.relu(self.bn(out))

        class _TimesBlock(nn.Module):
            """Single TimesNet block: FFT → 2D reshape → Inception → back."""

            def __init__(self, d: int, k: int):
                super().__init__()
                self.top_k = k
                self.inception = _InceptionBlock(1, d // 4)
                # Output channels from inception: 4 * (d//4) = d
                self.proj = nn.Linear(d * (d // 4) * 4, d)  # will be reshaped
                self.dropout = nn.Dropout(dropout)
                self.norm = nn.LayerNorm(d)

                # Adaptive projection for variable 2D sizes
                self.adapt_proj = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(d // 4 * 4, d)

            def forward(self, x):
                # x: (batch, seq_len, d_model)
                batch, L, d = x.shape
                residual = x

                # FFT to find dominant periods
                x_freq = torch.fft.rfft(x.mean(dim=-1), dim=1)
                amp = x_freq.abs().mean(dim=0)  # (L//2+1,)
                # Exclude DC component (index 0)
                amp[0] = 0
                _, top_indices = torch.topk(amp, min(self.top_k, len(amp) - 1))

                # For each dominant period, reshape to 2D and apply conv
                agg = torch.zeros_like(x)
                for idx in top_indices:
                    period = max(2, L // max(1, idx.item()))
                    if period > L:
                        period = L

                    # Pad sequence to be divisible by period
                    n_pad = (period - L % period) % period
                    x_pad = torch.nn.functional.pad(x, (0, 0, 0, n_pad))
                    padded_len = L + n_pad

                    # Reshape: (batch, d, period, padded_len//period)
                    n_cols = padded_len // period
                    x_2d = x_pad.permute(0, 2, 1)  # (batch, d, padded_len)
                    x_2d = x_2d.reshape(batch * d, 1, period, n_cols)

                    # Apply 2D conv (Inception)
                    out_2d = self.inception(x_2d)  # (batch*d, 4*(d//4), period, n_cols)

                    # Pool back to 1D
                    out_pool = self.adapt_proj(out_2d)  # (batch*d, ch, 1, 1)
                    out_pool = out_pool.squeeze(-1).squeeze(-1)  # (batch*d, ch)
                    out_1d = self.fc(out_pool)  # (batch*d, d)
                    out_1d = out_1d.reshape(batch, d, d)

                    # Average over d dimension → add to sequence
                    out_avg = out_1d.mean(dim=-1, keepdim=True)  # (batch, d, 1)
                    agg = agg + out_avg.permute(0, 2, 1).expand_as(x)

                agg = agg / max(1, len(top_indices))
                out = self.dropout(agg)
                return self.norm(out + residual)

        class TimesNetModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                self.layers = nn.ModuleList([
                    _TimesBlock(d_model, top_k) for _ in range(n_layers)
                ])
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )

            def forward(self, x):
                # x: (batch, seq_len, n_features)
                h = self.input_proj(x)
                for layer in self.layers:
                    h = layer(h)
                # Take last timestep
                return self.head(h[:, -1, :])

        m = TimesNetModel()
        _apply_weight_init(m, init)
        return m


class TimesNetForecaster(BaseForecaster):
    """TimesNet -- Temporal 2D Variation Modeling.

    [실험적] FFT로 주기 발견 → 1D→2D 변환 → Inception 2D conv.
    Top-K=3 주기, d_model=32, 2 layers.
    """

    meta = ModelMeta(
        name="TimesNet",
        category="dl",
        level=18,
        min_data=120,
        description="[실험적] TimesNet. FFT 기반 1D→2D 변환 + Inception 2D conv. 주기 패턴 학습.",
        dependencies=["torch"],
    )

    SEQ_LEN = 12

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> TimesNetForecaster:
        """(P0-D): Optuna HPO 내장.

 FFT+Inception2D 계열은 top_k 주기 각각에 대해 2D conv 를 돌리므로
 top_k·n_layers·d_model 모두 비용 선형. d_model ≤ 64, layer ≤ 3 로 제한.
 """
        from sklearn.preprocessing import StandardScaler
        from simulation.models.dl_models import _make_sequences, _train_loop
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop, INITS,
        )

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap

        X_seq, y_seq = _make_sequences(X_s, y_s, self.SEQ_LEN)
        n_features = X_s.shape[1]
        n_trials = get_trials("TimesNet", default=0)

        def _static_defaults():
            self._model = _TimesNetNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=32, n_layers=2, top_k=3,
                dropout=0.3, init="default",
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=300, lr=5e-4, patience=30,
                weight_decay=5e-4,
                augment=True, augment_factor=3,
            )

        def _objective(trial):
            import gc
            # d_model 은 4 의 배수여야 Inception (d//4) 분기가 성립.
            d_raw = trial.suggest_int("d_model_raw", 8, 64, log=True)
            d_model = ((d_raw + 3) // 4) * 4
            n_layers = trial.suggest_int("n_layers", 1, 3)
            top_k = trial.suggest_int("top_k", 1, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            init_t = trial.suggest_categorical("init", INITS)
            tr = suggest_training_hp(trial)

            model = _TimesNetNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=d_model, n_layers=n_layers,
                top_k=top_k, dropout=dropout, init=init_t,
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

        best, _ = run_optuna_loop("TimesNet", _objective, n_trials, _static_defaults)
        if best:
            d_model = ((best.get("d_model_raw", 32) + 3) // 4) * 4
            self._model = _TimesNetNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=d_model,
                n_layers=best.get("n_layers", 2),
                top_k=best.get("top_k", 3),
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
