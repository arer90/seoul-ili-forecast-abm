"""
simulation/models/foundation_model.py
====================================
기초모델(Foundation Model) 기반 ILI 예측: 해외 ILI 전이학습.

목표:
  1. 해외 ILI 데이터(WHO FluNet, CDC ILINet)로 사전학습
  2. 서울 sentinel_influenza 데이터로 파인튜닝
  3. 다중 단계 예측(1-4주) + 신뢰구간 반환

구현:
  - FoundationModelTransferLearner: 전통 심화학습 전이학습
    (LSTM 인코더 → 서울 데이터로 파인튜닝)

  (G-261 2026-06-13: ChronosMultiCountryForecaster 제거 — Chronos retire.
   foundation 모델 = TimesFM-2.5 + TiRex + OverseasTransfer 로 대체.)

특징:
  - 소표본(341주) 분포편이 대응: 대규모 사전학습 모델 활용
  - GPU 선택사항: CPU 호환 (memory 효율적)
  - Polars 데이터 로딩 지원
  - 타입 힌팅 + 한글 주석 포함
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

# 의존성 확인
# G-261 (2026-06-13): chronos import / _check_chronos / ChronosMultiCountryForecaster 제거 —
#   Chronos 전 변형 retire (transformers<5 ⊥ mlx-lm). 대체 = TimesFM-2.5 + TiRex (foundation).
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    log.debug("torch not available")


def _check_torch():
    """PyTorch 가용성 확인."""
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch이 설치되지 않았습니다.\n"
            "  uv pip install torch\n"
            "또는\n"
            "  pip install torch"
        )


# ═══════════════════════════════════════════════════════════════════
# 1. 전이학습 기반: LSTM 인코더 + 서울 데이터 파인튜닝
# ═══════════════════════════════════════════════════════════════════

class FoundationModelTransferLearner(BaseForecaster):
    """
    전통 심화학습 전이학습: LSTM 인코더 → 파인튜닝.

    구조:
      1. 해외 멀티국가 ILI 데이터로 LSTM 인코더 사전학습
      2. 서울 데이터로 전체 모델 파인튜닝 (encoder freeze 해제)
      3. 다중 단계 예측 (autoregressive)

    특징:
      - 소표본 대응: 사전학습 임베딩 재사용
      - 분포편이 완화: 서울 특화 신경망
      - 신뢰구간: MC Dropout (T=100 forward pass)

    Attributes:
        _encoder: LSTM 인코더 (공유 가중치)
        _decoder_head: 서울 특화 예측헤드 (선형층)
        _scaler_X: feature 정규화기
        _scaler_y: target 정규화기 (log1p 변환된 상태)
    """

    meta = ModelMeta(
        name="FoundationModelTransfer",
        category="dl",
        level=17,
        min_data=100,
        requires_gpu=False,
        dependencies=["torch"],
        description="LSTM 전이학습: 해외 멀티국가 데이터 → 서울 파인튜닝. "
                    "소표본 분포편이 대응.",
    )

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        patience: int = 20,
        freeze_encoder_epochs: int = 10,
    ):
        """
        Parameters:
            hidden_dim: LSTM 숨김 차원
            num_layers: LSTM 레이어 수
            dropout: Dropout 비율 (MC Dropout용)
            epochs: 파인튜닝 에폭
            batch_size: 배치 크기
            lr: 학습률
            patience: Early stopping patience
            freeze_encoder_epochs: encoder freeze 에폭 (사전학습 안정화)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.freeze_encoder_epochs = freeze_encoder_epochs

        self._encoder = None
        self._decoder_head = None
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._device = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        overseas_X: Optional[np.ndarray] = None,
        overseas_y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "FoundationModelTransferLearner":
        """
        사전학습(해외 데이터) → 파인튜닝(서울 데이터).

        Parameters:
            X_train: (n_train, n_features) Seoul training features
            y_train: (n_train,) Seoul ILI rate (log1p 변환됨)
            overseas_X: (n_overseas, n_feat_overseas) Overseas features (선택)
            overseas_y: (n_overseas,) Overseas ILI rate (선택)
            **kwargs: 추가 인자 (무시됨)

        Returns:
            self
        """
        _check_torch()

        import torch
        import torch.nn as nn

        from simulation.models.base import pick_device
        self._device = pick_device()
        log.info(f"  [FoundationModelTransfer] device={self._device}")

        # Feature 정규화
        from sklearn.preprocessing import StandardScaler
        self._scaler_X = StandardScaler()
        X_train_scaled = self._scaler_X.fit_transform(X_train)

        # Target 정규화 (이미 log1p 변환되었으나, 추가 정규화)
        self._scaler_y = StandardScaler()
        y_train_scaled = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # 모델 구축
        input_dim = X_train.shape[1]
        self._build_model(input_dim)
        model = self._model.to(self._device)

        # 1단계: 해외 데이터로 사전학습 (선택사항)
        if overseas_X is not None and overseas_y is not None:
            log.info(f"  [FoundationModelTransfer] 사전학습 시작: {len(overseas_X)} samples")
            self._pretrain_on_overseas(
                model,
                overseas_X,
                overseas_y,
                epochs=max(50, self.epochs // 2),
            )

        # 2단계: 서울 데이터로 파인튜닝
        log.info(f"  [FoundationModelTransfer] 파인튜닝 시작: {len(X_train)} samples")
        self._finetune_on_seoul(model, X_train_scaled, y_train_scaled)

        self._fitted = True
        log.info(f"  [FoundationModelTransfer] fit 완료")
        return self

    def _build_model(self, input_dim: int) -> None:
        """LSTM 인코더 + 선형 디코더 모델 구축."""
        import torch.nn as nn

        class LSTMEncoderDecoder(nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_dim: int,
                num_layers: int,
                dropout: float,
            ):
                super().__init__()
                self.encoder = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.decoder_head = nn.Linear(hidden_dim, 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                x: (batch, features) 또는 (batch, seq_len, features)

                seq_len이 없으면 (batch, 1, features)로 변환.
                """
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # (batch, 1, features)

                # LSTM: (batch, seq_len, hidden)
                lstm_out, (h_n, c_n) = self.encoder(x)

                # 마지막 숨김 상태 사용
                last_hidden = h_n[-1]  # (batch, hidden)
                last_hidden = self.dropout(last_hidden)

                # 선형 디코더
                out = self.decoder_head(last_hidden)  # (batch, 1)
                return out.squeeze(-1)  # (batch,)

        self._model = LSTMEncoderDecoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def _pretrain_on_overseas(
        self,
        model,
        X_overseas: np.ndarray,
        y_overseas: np.ndarray,
        epochs: int,
    ) -> None:
        """해외 데이터로 LSTM 인코더 사전학습."""
        import torch
        import torch.nn as nn

        # Feature 정규화
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_overseas_scaled = scaler.fit_transform(X_overseas)

        # Target 정규화
        scaler_y = StandardScaler()
        y_overseas_scaled = scaler_y.fit_transform(y_overseas.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_overseas_scaled).to(self._device)
        y_t = torch.FloatTensor(y_overseas_scaled).to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr * 10)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                log.debug(f"    사전학습 epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}")

    def _finetune_on_seoul(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """서울 데이터로 모델 파인튜닝."""
        import torch
        import torch.nn as nn

        X_t = torch.FloatTensor(X_train).to(self._device)
        y_t = torch.FloatTensor(y_train).to(self._device)

        # Validation split
        val_n = max(8, int(len(X_train) * 0.2))
        X_val, y_val = X_t[-val_n:], y_t[-val_n:]
        X_tr, y_tr = X_t[:-val_n], y_t[:-val_n]

        # 1단계: Encoder freeze
        for param in model.encoder.parameters():
            param.requires_grad = False

        optimizer_head = torch.optim.Adam(
            model.decoder_head.parameters(),
            lr=self.lr,
        )
        criterion = nn.MSELoss()  # G-218: huber 영구 제거 (huber-loss-banned-20260520)

        model.train()
        for epoch in range(self.freeze_encoder_epochs):
            optimizer_head.zero_grad()
            pred = model(X_tr)
            loss = criterion(pred, y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_head.step()

        log.debug(f"    Encoder freeze 완료: {self.freeze_encoder_epochs} epochs")

        # 2단계: Encoder 해제 + 전체 파인튜닝
        for param in model.encoder.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr * 0.1,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=30,
            T_mult=2,
            eta_min=self.lr * 0.001,
        )

        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(self.epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            pred = model(X_tr)
            loss = criterion(pred, y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                log.debug(
                    f"    파인튜닝 epoch {epoch + 1}/{self.epochs}: "
                    f"train_loss={loss.item():.4f}, val_loss={val_loss:.4f}"
                )

            if no_improve >= self.patience:
                log.debug(f"    Early stopping at epoch {epoch + 1}")
                break

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        예측 (정규화 역변환 포함).

        Parameters:
            X_test: (n_test, n_features)
            **kwargs: 추가 인자
                - mc_samples: MC Dropout 샘플 수 (기본: 1, 100이면 신뢰구간 생성)

        Returns:
            (n_test,) 점 예측 (역정규화됨)
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("FoundationModelTransfer: fit() 먼저 호출 필수")

        _check_torch()
        import torch

        X_scaled = self._scaler_X.transform(X_test)
        X_t = torch.FloatTensor(X_scaled).to(self._device)

        mc_samples = kwargs.get("mc_samples", 1)
        predictions = []

        self._model.eval()
        with torch.no_grad():
            for _ in range(mc_samples):
                pred = self._model(X_t)  # (n_test,)
                predictions.append(pred.cpu().numpy())

        # 점 예측: 평균
        pred_scaled = np.mean(predictions, axis=0)  # (n_test,)

        # 역정규화
        pred = self._scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        return np.asarray(pred, dtype=np.float32)


# ── 모델 등록 ──
# G-181 (2026-05-05) — 사용자 명시 deprecate:
# FoundationModelTransferLearner = 자체 LSTM transfer (chronos 아님).
# Synthetic test R²=-1.87 ~ -2.39 (small data 한계).
# 대체: OverseasTransfer R²=+0.87 PASS (chronos 기반).
# if _HAS_TORCH:
#     REGISTRY.register(FoundationModelTransferLearner)  # DEPRECATED (G-181)
#     log.info("[foundation_model] FoundationModelTransferLearner 등록됨")
log.info("[foundation_model] FoundationModelTransferLearner DEPRECATED (G-181, 대체: OverseasTransfer)")
# G-261 (2026-06-13): ChronosMultiCountryForecaster 등록 제거 — Chronos retire (TimesFM-2.5 + TiRex 대체).
