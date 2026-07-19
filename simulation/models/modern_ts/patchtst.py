"""
simulation/models/modern_ts/patchtst.py
========================================
PatchTST (Patch Time Series Transformer) -- Level 15 [실험적]

Channel-independent patching + lightweight Transformer.
patch_len=4, stride=2 → 5 patches from lookback=12.
Heavy regularization: dropout=0.3, weight_decay=5e-4.
"""

from __future__ import annotations

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

__all__ = ["PatchTSTForecaster"]


class _PatchTSTNet:
    """PatchTST: channel-independent patching + Transformer."""

    @staticmethod
    def build(n_features: int, lookback: int = 12,
              patch_len: int = 4, stride: int = 2,
              d_model: int = 32, n_heads: int = 4,
              n_layers: int = 2, dim_ff: int = 64,
              dropout: float = 0.3):
        import torch
        import torch.nn as nn

        n_patches = max(1, (lookback - patch_len) // stride + 1)

        class PatchTSTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_len = patch_len
                self.stride = stride
                self.n_patches = n_patches
                self.n_features = n_features

                # Feature projection → univariate
                self.feat_proj = nn.Linear(n_features, 1)

                # Patch embedding
                self.patch_embed = nn.Linear(patch_len, d_model)

                # Learnable positional encoding
                self.pos_embed = nn.Parameter(
                    torch.randn(1, n_patches, d_model) * 0.02
                )

                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_ff,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers
                )

                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * n_patches, 1),
                )

            def forward(self, x):
                # x: (batch, lookback, n_features)
                batch_size = x.size(0)

                # Project to univariate: (batch, lookback)
                x_uni = self.feat_proj(x).squeeze(-1)

                # Extract patches: (batch, n_patches, patch_len)
                patches = []
                for i in range(self.n_patches):
                    start = i * self.stride
                    end = start + self.patch_len
                    if end > x_uni.size(1):
                        # Pad if needed
                        p = torch.zeros(batch_size, self.patch_len, device=x.device)
                        valid = x_uni.size(1) - start
                        p[:, :valid] = x_uni[:, start:start + valid]
                        patches.append(p)
                    else:
                        patches.append(x_uni[:, start:end])
                patches = torch.stack(patches, dim=1)  # (batch, n_patches, patch_len)

                # Embed patches
                patch_emb = self.patch_embed(patches) + self.pos_embed

                # Transformer
                out = self.transformer(patch_emb)

                # Prediction head
                return self.head(out)

        return PatchTSTModel()


class PatchTSTForecaster(BaseForecaster):
    """PatchTST -- Patch Time Series Transformer.

    [실험적] Channel-independent patching + lightweight Transformer.
    patch_len=4, stride=2 → 5 patches from lookback=12.
    Heavy regularization: dropout=0.3, weight_decay=5e-4.
    """

    meta = ModelMeta(
        name="PatchTST",
        category="dl",
        level=15,
        min_data=120,
        description="[실험적] PatchTST. Patch embedding + Transformer. 소표본 강한 정규화.",
        dependencies=["torch"],
    )

    SEQ_LEN = 12

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> PatchTSTForecaster:
        """(P0-D): Optuna HPO 내장."""
        from sklearn.preprocessing import StandardScaler
        from simulation.models.dl_models import _make_sequences, _train_loop, lag_backbone_seq
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop,
        )

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap

        # G-319d: lag backbone(과거 y) 입력으로 feat_proj(398→1) 입력버그 회복. lag 있으면 AR
        #   단일채널 시퀀스(n_features=1 — nested build 가 closure 로 전파), 없으면 _make_sequences fallback.
        _lag_seq, self._lag_idx = lag_backbone_seq(X_s, kwargs.get("feature_names"), self.SEQ_LEN)
        if self._lag_idx is not None:
            X_seq, y_seq, n_features = _lag_seq, y_s, 1
        else:
            X_seq, y_seq = _make_sequences(X_s, y_s, self.SEQ_LEN)
            n_features = X_s.shape[1]

        n_trials = get_trials("PatchTST", default=0)

        def _static_defaults():
            self._model = _PatchTSTNet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                patch_len=4, stride=2,
                d_model=32, n_heads=4, n_layers=2, dim_ff=64, dropout=0.3,
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=300, lr=5e-4, patience=30,
                weight_decay=5e-4, augment=True, augment_factor=3,
            )

        def _objective(trial):
            import gc
            # d_model 은 n_heads 의 배수여야 함
            n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
            d_raw = trial.suggest_int("d_model_raw", 8, 512, log=True)
            d_model = ((d_raw + n_heads - 1) // n_heads) * n_heads
            n_layers_tf = trial.suggest_int("n_layers_tf", 1, 3)
            dim_ff = trial.suggest_int("dim_ff", 16, 1024, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            patch_len = trial.suggest_categorical("patch_len", [3, 4, 6])
            stride = trial.suggest_categorical("stride", [1, 2, 3])
            tr = suggest_training_hp(trial)

            model = _PatchTSTNet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                patch_len=patch_len, stride=stride,
                d_model=d_model, n_heads=n_heads,
                n_layers=n_layers_tf, dim_ff=dim_ff, dropout=dropout,
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

        best, _ = run_optuna_loop("PatchTST", _objective, n_trials, _static_defaults)
        if best:
            n_heads = best.get("n_heads", 4)
            d_model = ((best.get("d_model_raw", 32) + n_heads - 1) // n_heads) * n_heads
            self._model = _PatchTSTNet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                patch_len=best.get("patch_len", 4),
                stride=best.get("stride", 2),
                d_model=d_model, n_heads=n_heads,
                n_layers=best.get("n_layers_tf", 2),
                dim_ff=best.get("dim_ff", 64),
                dropout=best.get("dropout", 0.3),
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
            # G-319d: fit 과 동일 lag backbone 구성(leak-free, X_test 의 lag 컬럼).
            X_seq = lag_backbone_from_idx(X_s, self._lag_idx, self.SEQ_LEN)
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
        _pred = np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )
        from simulation.models.safety import apply_extrapolation_cap  # G-289
        return apply_extrapolation_cap(_pred, getattr(self, "_y_train_max", None))
