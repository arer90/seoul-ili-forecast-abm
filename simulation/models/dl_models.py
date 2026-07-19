"""
simulation/models/dl_models.py
==============================
딥러닝(Deep Learning) 범주 모델: DNN, TCN + Optuna HPO 변형

모두 PyTorch 기반.
- 공통 학습 루프: AdamW + CosineAnnealingWarmRestarts + EarlyStopping
- DNN: (batch, features) 형태의 flat input
- TCN: (batch, seq_len, features) 형태의 sliding window
- 출력: (batch, 1) -- 다음 1주 ILI rate 예측

ILI rate(‰) ~340주 소표본 → 과적합 방지 중요:
 - Dropout, weight_decay, early stopping 필수

변경 이력:
 - : MLP, LSTM, GRU, Bi-LSTM, TCN
 - : 학습 루프 개선 (Cosine Annealing, Augmentation) — HuberLoss 는 G-218 로 영구 제거
 - (2026-03-25): MLP→DNN 명칭 변경, LSTM/GRU/BiLSTM 제거
 - LSTM/GRU/BiLSTM: 204주 소표본 + distribution shift(3.4x)에서
 hidden state 전파 기반 recurrent 모델은 R²<0.5로 학습 불가 확인
 (walk-forward 2-fold CV 검증 완료). 코드는 _archive로 이동.
 - TFT는 tft_wrapper.py에서 별도 관리
 - (2026-03-30): 개발 마일스톤 1 -- Optuna HPO 추가
 - OptunaDNNForecaster: 50-trial MedianPruner HPO (level=10)
 - OptunaTCNForecaster: 50-trial MedianPruner HPO (level=14)
 - _train_loop: trial/sample_weights/curriculum_mode 파라미터 추가 (curriculum learning 준비)
 - _build_dnn_model 헬퍼 함수 추출
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 공통 PyTorch 유틸리티
# ═══════════════════════════════════════════════════════════════

def _check_torch():
    """PyTorch 가용성 확인."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _train_loop(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 30,
    weight_decay: float = 1e-4,
    augment: bool = True,
    # G-231 + 2026-05-26 archive: augment_factor permanently 0 (PI augmentation
    # was tied to the α-blend logic; both removed). Previous G-144 env-var override
    # `MPH_PI_AUGMENT_FACTOR=3` no longer applies — flag kept for API compat only.
    augment_factor: int = 0,
    trial=None,  # Optuna trial for pruning
    sample_weights: Optional[np.ndarray] = None,  # curriculum learning prep
    curriculum_mode: Optional[str] = None,  # curriculum learning: 'linear'|'exponential'|None
    optimizer_type: str = "adamw",  # : 'adamw'(default)|'adam'|'radam'|'rmsprop'|'sgd_momentum'
    loss_type: str = "mse",         # G-218: huber 영구 제거 (memory huber-loss-banned-20260520) — 'mse'(default)|'mae'
    return_r2: bool = False,        # P0-3b NEW: True → return -val_r2 (Optuna R² 최대화)
    history_sink: Optional[list] = None,  # : if given, per-epoch {epoch,train_loss,val_loss,lr} 기록
) -> float:
    """공통 학습 루프 (AdamW + Cosine Annealing + EarlyStopping + 증강).

 - 데이터 증강: jittering + scaling으로 학습 데이터 증가 (2D/3D 호환)
 - Cosine Annealing LR: 더 안정적 수렴
 - Loss: 기본 MSE (평가 metric 과 일치). G-218: huber 영구 금지, Optuna 는 mse/mae 만 탐색.
 - Warmup: 초기 5 에폭 warmup
 - trial: Optuna trial 객체 (pruning 지원)
 - sample_weights: 샘플별 가중치 (curriculum learning 준비)
 - curriculum_mode: 'linear'|'exponential' (curriculum learning 준비)
 - return_r2: P0-3b — True 이면 best_state 복원 후 val R² 계산,
 `-val_r2` 반환 (Optuna direction="minimize" 와 결합해 R² 최대화).
 False (기본) 는 기존과 동일하게 best_val_loss 반환. 호출처 영향 없음.

 Returns:
 return_r2=False: best_val_loss (float)
 return_r2=True : -best_val_r2 (float, Optuna minimize objective)
 """
    import torch
    import torch.nn as nn

    # r3 / Mac: cuda > mps > cpu 우선순위. MPH_DEVICE / MPH_FORCE_CPU 로 override.
    from simulation.models.base import pick_device
    device = pick_device()
    model = model.to(device)
    # P1 (R8 2026-05-28): torch.compile 배선 — GLOBAL.package_c.compile_models gate
    #   (default OFF, env MPH_PC_COMPILE=1 opt-in). self-gated + fallback 라 OFF 시 no-op.
    #   주의: autocast(package_c_autocast_ctx) 는 fp16 GradScaler 통합 + A/B 검증 필요 →
    #   학습 종료 후 별도 적용 (현재 미배선).
    model = package_c_compile_helper(model, device.type)

    # ════════════════════════════════════════════════════════════════
    # Package M (G-150 진짜 root cause): augment 와 train/val split 순서 fix
    # ════════════════════════════════════════════════════════════════
    # 이전 버그: X_train 전체를 augment → X_train[-val_n:] (val 부분) 의 augmented
    #          copy 가 train 에 포함 → 모델이 val 정답 학습 → val R²=+1.0 leakage.
    # Fix: train/val split FIRST, augment ONLY train portion (val 은 원본 그대로).
    # ════════════════════════════════════════════════════════════════
    val_n = max(8, int(len(X_train) * 0.2))
    X_train_only = X_train[:-val_n]
    y_train_only = y_train[:-val_n]
    X_val_orig = X_train[-val_n:]
    y_val_orig = y_train[-val_n:]

    # D-3: augment_factor 동적 cap (소표본 과적합 방지, 총 샘플 ≤ 1000)
    # Package M: augment 는 train portion 만 (val 제외)
    if augment and len(X_train_only) < 500:
        aug_f = min(augment_factor, max(1, 1000 // max(len(X_train_only), 1)))
        from simulation.models.feature_engine import TimeSeriesAugmentor
        aug = TimeSeriesAugmentor(seed=42)
        X_aug, y_aug = aug.augment_dataset(X_train_only, y_train_only, n_augments=aug_f)
    else:
        X_aug, y_aug = X_train_only, y_train_only

    X_t = torch.FloatTensor(X_aug).to(device)
    y_t = torch.FloatTensor(y_aug).to(device)

    # Validation: 원본 train 의 마지막 20% (augment 안 함, leakage 방지)
    X_val = torch.FloatTensor(X_val_orig).to(device)
    y_val = torch.FloatTensor(y_val_orig).to(device)
    X_tr, y_tr = X_t, y_t

    # Sample weights for curriculum learning (curriculum learning 준비)
    use_curriculum = sample_weights is not None or curriculum_mode is not None
    if use_curriculum:
        if sample_weights is not None:
            sw_t = torch.FloatTensor(sample_weights).to(device)
            # Augmented data가 있으면 weights도 확장
            if len(sw_t) < len(X_tr):
                sw_t = sw_t.repeat(len(X_tr) // len(sw_t) + 1)[:len(X_tr)]
        else:
            sw_t = torch.ones(len(X_tr), device=device)

    # : optimizer 선택 (rmsprop 추가)
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":  # D-5 NEW
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9,
        )
    elif optimizer_type == "sgd_momentum":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            momentum=0.9, nesterov=True,
        )
    else:  # adamw (default)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=lr * 0.01,
    )

    # G-218: huber 영구 금지 (memory huber-loss-banned-20260520) — MSE/MAE 만
    def _make_loss(reduction="mean"):
        if loss_type == "mae":
            return nn.L1Loss(reduction=reduction)
        else:  # mse (default)
            return nn.MSELoss(reduction=reduction)

    if use_curriculum:
        criterion = _make_loss(reduction='none')
    else:
        criterion = _make_loss(reduction='mean')

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    warmup_epochs = 5

    model.train()
    for epoch in range(epochs):
        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Curriculum learning: epoch-dependent weight adjustment
        if use_curriculum and curriculum_mode is not None:
            progress = epoch / max(epochs - 1, 1)
            if curriculum_mode == 'linear':
                # 초기: uniform → 후기: sample_weights 반영
                epoch_weights = (1 - progress) * torch.ones_like(sw_t) + progress * sw_t
            elif curriculum_mode == 'exponential':
                epoch_weights = sw_t ** progress
            else:
                epoch_weights = sw_t
        elif use_curriculum:
            epoch_weights = sw_t
        else:
            epoch_weights = None

        # Mini-batch (셔플)
        indices = torch.randperm(len(X_tr))
        epoch_loss = 0.0
        n_batches = 0
        _cur_lr = float(optimizer.param_groups[0]["lr"])  # history snapshot

        for i in range(0, len(X_tr), batch_size):
            idx = indices[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]

            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)

            if epoch_weights is not None:
                # Per-sample weighted loss
                loss_unreduced = criterion(pred, yb)
                wb = epoch_weights[idx]
                loss = (loss_unreduced * wb).mean()
            else:
                loss = criterion(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        # D-2: val loss 를 학습 loss 와 동일 함수로 통일 (early-stop metric 일관성)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze(-1)
            val_criterion = _make_loss(reduction="mean")
            val_loss = val_criterion(val_pred, y_val).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # : per-epoch history 로그 (plot/csv 에서 학습곡선 재구성용)
        if history_sink is not None:
            history_sink.append({
                "epoch": int(epoch),
                "train_loss": float(epoch_loss / max(n_batches, 1)),
                "val_loss": float(val_loss),
                "lr": _cur_lr,
            })

        # Optuna pruning
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        # G-123: subprocess stall 방지용 heartbeat — 50 epoch마다 stdout flush
        if epoch % 50 == 0 or epoch == epochs - 1:
            import sys as _sys
            print(f"[TRAIN] epoch={epoch}/{epochs} val_loss={val_loss:.6f} "
                  f"best={best_val_loss:.6f} patience={no_improve}/{patience}",
                  flush=True, file=_sys.stdout)

        if no_improve >= patience:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # P0-3b: R² objective 옵션 — best_state 복원 후 val 예측으로 R² 계산
    if return_r2:
        from sklearn.metrics import r2_score
        with torch.no_grad():
            final_val_pred = model(X_val).squeeze(-1).cpu().numpy()
        y_val_np = y_val.cpu().numpy()
        try:
            val_r2 = float(r2_score(y_val_np, final_val_pred))
        except Exception:
            val_r2 = -1e6  # degenerate case → worst score
        # Optuna direction="minimize" 와 결합: -R² 를 min 하면 R² 최대화
        return -val_r2

    return best_val_loss


def _predict_torch(model, X_test: np.ndarray) -> np.ndarray:
    """PyTorch 모델 예측.

 r3: train 때 GPU 에 올라간 모델이 inference 때 device 불일치로
 터지는 것을 방지 — 모델이 있는 디바이스로 입력을 맞춘다.
 """
    import torch
    model.eval()
    with torch.no_grad():
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        X_t = torch.FloatTensor(X_test).to(model_device)
        pred = model(X_t).squeeze(-1).cpu().numpy()
    # G-278 (2026-06-16, 3자 감사): 스케일공간(StandardScaler y, 0=train평균) raw 출력 그대로 반환.
    #   옛 np.maximum(pred,0) 는 평균이하 주를 train평균까지 끌어올리는 양의 bias → 제거.
    #   모든 호출자가 inverse_transform 後 원공간 nonneg(maximum)를 적용하므로 ILI≥0 은 보존됨.
    return pred


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 8):
    """
    (n_samples, n_features) → (n_samples - seq_len, seq_len, n_features)
    시계열 lookback window 생성. TCN, TFT에서 사용.
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def lag_backbone_from_idx(X: np.ndarray, lag_idx, seq_len: int):
    """lag_idx 로 (n, seq_len, 1) padded AR-backbone 시퀀스 구성 (fit/predict 공통).

    lag 수 < seq_len 이면 oldest 값 반복으로 front-pad, ≥ 이면 most-recent seq_len 개. lookback
    을 seq_len 으로 고정 → 모델 build(lookback) 변경 0 (n_features 만 1).
    """
    seq = np.asarray(X)[:, lag_idx][:, :, None]  # (n, n_lags, 1), oldest→newest
    nl = seq.shape[1]
    if nl >= seq_len:
        return seq[:, -seq_len:, :]
    pad = np.repeat(seq[:, :1, :], seq_len - nl, axis=1)
    return np.concatenate([pad, seq], axis=1)


def lag_backbone_seq(X: np.ndarray, feature_names, seq_len: int, min_lags: int = 4):
    """AR-lag 컬럼을 (n, seq_len, 1) 과거-y lookback 시퀀스로 추출 (oldest→newest).

    G-319d (2026-06-19, 전체 라인업 감사): modern-ts deep(PatchTST/iTransformer/Mamba)이
    feat_proj=Linear(398→1) 로 과거 y(자기회귀)를 합성채널로 뭉개는 입력버그 회복용. X 안의
    ``ili_rate_lag{N}`` 컬럼들 = predict X_test 에도 존재(leak-free, A1 이 feature 선택서 보존)
    → 이를 AR backbone 단일채널 시퀀스로 사용. A/B(ab_canonical_input_all6) 입증: tabular(398)
    음수 → lag backbone +0.46~0.73. lag 없거나 min_lags 미만이면 None(현 _make_sequences fallback).

    Args:
        X: (n, p) 스케일된 feature 행렬.
        feature_names: 컬럼 이름 list. None/빈 list 면 (None, None).
        seq_len: 출력 시퀀스 길이(모델 lookback). pad/truncate 기준.
        min_lags: 최소 lag 컬럼 수. 미만이면 backbone 미구성(None).

    Returns:
        (X_seq, lag_idx): X_seq=(n, seq_len, 1) padded 시퀀스, lag_idx=원본 컬럼 인덱스 list.
        또는 (None, None) — lag 부족 시.

    Side effects: 없음 (순수 함수).
    Caller responsibility: predict 시 lag_backbone_from_idx(X_test, lag_idx, seq_len) 동일 적용.
    """
    if feature_names is None or len(feature_names) == 0:
        return None, None
    pairs = []
    _pfx = "ili_rate_lag"
    for j, nm in enumerate(feature_names):
        if isinstance(nm, str) and nm.startswith(_pfx):
            suf = nm[len(_pfx):]
            if suf.isdigit():
                pairs.append((int(suf), j))
    if len(pairs) < min_lags:
        return None, None
    pairs.sort(key=lambda t: -t[0])  # 큰 lag(oldest) → 작은 lag(newest)
    idx = [j for _, j in pairs]
    return lag_backbone_from_idx(X, idx, seq_len), idx


# ═══════════════════════════════════════════════════════════════
# DNN 모델 빌더 헬퍼
# ═══════════════════════════════════════════════════════════════

def _get_activation_fn(name: str = "relu"):
    """이름 → nn.Module. DNN/TabularDNN 공통."""
    import torch.nn as nn
    _MAP = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
        "mish": nn.Mish,
        "elu": nn.ELU,
        "swish": nn.SiLU,
    }
    cls = _MAP.get(name, nn.ReLU)
    return cls()


def _get_norm_layer(name: str, dim: int):
    """이름 → normalization layer."""
    import torch.nn as nn
    if name == "layer":
        return nn.LayerNorm(dim)
    elif name == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


def _apply_weight_init(model, init_type: str = "default"):
    """가중치 초기화 전략."""
    import torch.nn as nn
    if init_type == "default":
        return
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "lecun":
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _build_dnn_model(n_features: int, hidden_dims: list = None,
                     dropouts: list = None, activation: str = "relu",
                     norm: str = "none", init: str = "default",
                     # ── backward compat (호출 지원) ──
                     hidden1: int = None, hidden2: int = None,
                     dropout1: float = 0.25, dropout2: float = 0.15,
                     n_layers: int = 2):
    """Build DNN Sequential model (: 동적 N-layer, activation/norm/init 탐색).

 GPU 환경: hidden_dims=[512, 256, 128], dropouts=[0.3, 0.2, 0.1] 등
 소표본 CPU: hidden_dims=[64, 32], dropouts=[0.3, 0.2] 등

 Args:
 n_features: 입력 피처 수
 hidden_dims: 각 hidden layer 크기 리스트 (예: [256, 128, 64])
 None이면 hidden1/hidden2/n_layers로 backward-compat 구성
 dropouts: 각 layer dropout rate 리스트 (len == len(hidden_dims))
 None이면 dropout1/dropout2로 자동 구성
 activation: 'relu'|'gelu'|'selu'|'leaky_relu'|'mish'|'elu'|'swish'
 norm: 'none'|'layer'|'batch'
 init: 'default'|'kaiming'|'xavier'|'lecun'
 """
    import torch.nn as nn

    # ── backward compat: 스타일 호출 → hidden_dims 변환 ──
    if hidden_dims is None:
        h1 = hidden1 if hidden1 is not None else max(64, n_features * 3)
        h2 = hidden2 if hidden2 is not None else max(32, n_features)
        if n_layers == 1:
            hidden_dims = [h1]
        elif n_layers == 2:
            hidden_dims = [h1, h2]
        elif n_layers >= 3:
            h3 = max(16, h2 // 2)
            hidden_dims = [h1, h2, h3]
            if n_layers >= 4:
                # 4+ layers: 점진적 축소
                for i in range(3, n_layers):
                    hidden_dims.append(max(16, hidden_dims[-1] // 2))
        else:
            hidden_dims = [h1, h2]

    if dropouts is None:
        # 첫 layer는 dropout1, 나머지는 dropout2
        dropouts = [dropout1] + [dropout2] * (len(hidden_dims) - 1)
    # dropouts 길이 맞춤
    while len(dropouts) < len(hidden_dims):
        dropouts.append(dropouts[-1])

    # ── 동적 N-layer 구성 ──
    layers = []
    in_dim = n_features
    for i, (h_dim, drop) in enumerate(zip(hidden_dims, dropouts)):
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(_get_norm_layer(norm, h_dim))
        layers.append(_get_activation_fn(activation))
        if drop > 0:
            layers.append(nn.Dropout(drop))
        in_dim = h_dim

    # Output layer
    layers.append(nn.Linear(in_dim, 1))

    model = nn.Sequential(*layers)
    _apply_weight_init(model, init)
    return model


# ═══════════════════════════════════════════════════════════════
# 1. DNN (Deep Neural Network) -- Level 9
#    구 명칭: MLP (Multi-Layer Perceptron)
# ═══════════════════════════════════════════════════════════════

class DNNForecaster(BaseForecaster):
    """DNN (Deep Neural Network) -- 다층 완전연결 신경망.

 2-hidden-layer FC + Dropout 정규화.
 소표본 시계열(~340주)에서 가장 안정적인 DL 모델.
 Walk-forward 2-fold CV R²=0.85 (3-seed ensemble).

 최적 설정 ( 2026-03-25):
 - hidden: [n_feat×3, n_feat] (wider first layer)
 - dropout: 0.25/0.15
 - epochs: 500, lr: 5e-4, patience: 40
 - augment_factor: 4
 - log-target: log(1+y) 변환 필수
 """

    meta = ModelMeta(
        name="DNN",
        category="dl",
        level=9,
        min_data=80,
        description="DNN. 2-hidden-layer FC, Dropout 정규화, 비선형 피처 학습. 소표본 최적.",
        dependencies=["torch"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._y_train_max = None  # D-1: fold-local prediction cap

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> DNNForecaster:
        from sklearn.preprocessing import StandardScaler

        # D-1: fold-local 상한 저장 (predict 에서 cap)
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 1.0

        # Sprint 1.5 R4 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_train)

        n_features = X_s.shape[1]
        hidden1 = max(64, n_features * 3)
        hidden2 = max(32, n_features)

        # : COVID-era 가중치 ( 수준 유지, 과도한 강화 방지)
        n = len(y_train)
        sample_weights = np.ones(n)
        recent_start = int(n * 0.6)
        sample_weights[recent_start:] = 2.0

        # : 3-seed ensemble (v6과 동일, v7의 5-seed는 과적합 경향)
        self._models = []
        self._history = []  # : all seeds concat, epoch 재라벨링 하지 않음
        for seed in [42, 2024, 31415]:
            import torch
            torch.manual_seed(seed)
            model = _build_dnn_model(
                n_features, hidden1=hidden1, hidden2=hidden2,
                dropout1=0.30, dropout2=0.20,
            )
            _seed_hist: list = []
            _train_loop(
                model, X_s, y_s,
                epochs=600, lr=3e-4, patience=50,
                augment=True, augment_factor=5,
                sample_weights=sample_weights,
                curriculum_mode='linear',
                loss_type='mse',          # D-2
                optimizer_type='adamw',   # default
                history_sink=_seed_hist,
            )
            for _r in _seed_hist:
                _r["seed"] = seed
            self._history.extend(_seed_hist)
            self._models.append(model)

        self._model = self._models[0]  # backward compat for save
        self._fitted = True
        log.info(f"  [DNN] 3-seed ensemble + curriculum learning 완료 "
                 f"(y_train_max={self._y_train_max:.2f}, cap={max(self._y_train_max*3.0, 200.0):.2f})")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        # 3-seed ensemble 평균
        preds = [_predict_torch(m, X_s) for m in self._models]
        pred_s = np.mean(preds, axis=0)
        y_raw = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        # D-1: fold-local cap (y_train_max * 1.5) — 외삽 폭주 방지
        # Package O (G-153 fix): cap 강화 (3.0×→1.5×, 200→100)
        # 이전 cap=200 이 TCN-Optuna 의 |pred|=200.8 폭주 허용 → R²=−4.98
        cap = max(self._y_train_max * 1.5, 100.0) if self._y_train_max else 100.0
        # Package L (G-146): NaN/Inf finite guard — log1p inverse 발산 시 reference (mean) 로 대체
        y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
        return np.clip(y_raw, 0.0, cap)


# 2026-05-26 (Cleanup quick-win): MLPForecaster = DNNForecaster alias 제거.
# 외부 사용 0건 확인 (grep "MLPForecaster" → 본 파일 외 정의 X). 2026-03-25
# MLP→DNN 명칭 변경 후 1년+ alias 유지했으나 callers 모두 새 이름 사용 중.

# ═══════════════════════════════════════════════════════════════
# 1a. TinyMLP -- Level 8 (S2-3 sanity floor baseline)
# ═══════════════════════════════════════════════════════════════

class TinyMLPForecaster(BaseForecaster):
    """TinyMLP — small 2-layer MLP baseline (S2-3, backlog).

 Motivation (ENGINEERING_PRINCIPLES.md S2-3):
 n=343 주에 DNN+Attention+FM (TabularDNNForecaster) 은 over-
 parameterized 될 수 있다. 소표본 sanity floor 역할을 맡는 plain
 baseline 이 없었다. TinyMLP 는 고의적으로 미니멀하게 유지:
 - 고정 (32 → 16 → 1) 아키텍처 (feature 수에 무관)
 - Dropout 0.2
 - Single seed (no 3-seed ensemble)
 - No data augmentation / curriculum / sample weighting
 - 200 epochs, lr=1e-3, patience=30
 DNN / OptunaDNN / TabularDNN 이 이 베이스라인을 의미 있게
 이기지 못하면 복잡한 구조가 n=343 에 과도하다는 신호다.
 """

    meta = ModelMeta(
        name="TinyMLP",
        category="dl",
        level=8,
        min_data=60,
        description=(
            "TinyMLP. 고정 (32,16) hidden + Dropout 0.2. 소표본 sanity "
            "floor 베이스라인 — 복잡 DL 모델의 over-param 진단용."
        ),
        dependencies=["torch"],
    )

    # Fixed architecture — intentionally does NOT scale with n_features.
    # 2026-04-29: dropout 0.2 → 0.5, patience 30 → 15 (n=242, p=382 underdetermined)
    # 사건: TinyMLP val=3.4 → test R²=-2.37 (val/OOF=3.4/5.07 = noise 아닌 진짜 overfit)
    _HIDDEN = (32, 16)
    _DROPOUT = 0.5         # ← 0.2 → 0.5 (n=242 + 382 features p>n 대응)
    _EPOCHS = 200
    _LR = 1e-3
    _PATIENCE = 15         # ← 30 → 15 (early stop 강화)
    _WEIGHT_DECAY = 1e-3   # ← 신규 L2 regularization
    _SEED = 42

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._y_train_max = None  # D-1: fold-local prediction cap

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(self._SEED)
        h1, h2 = self._HIDDEN
        return nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Dropout(self._DROPOUT),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(self._DROPOUT),
            nn.Linear(h2, 1),
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> TinyMLPForecaster:
        from sklearn.preprocessing import StandardScaler

        if not _check_torch():
            raise RuntimeError("TinyMLP requires torch")

        # D-1: fold-local 상한 저장
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 1.0

        # : log1p 내부 변환 — TinyMLP 가 train 평균을 외워 test (2.54×
        # 분포 이동) 에서 R²=-0.93 을 받던 패턴 대응. log 공간 학습이면 외삽이
        # multiplicative 방향으로 작동 → sanity-floor 역할을 유지하면서 고장을
        # 완화. train y ≥ 0 전제.
        self._y_log_used = bool(np.all(y_train >= 0))
        y_fit = np.log1p(y_train) if self._y_log_used else y_train.astype(float)

        # Sprint 1.5 R4 (2026-05-26): use shared setup_xy_scalers helper
        # (y_fit pre-transformed by log1p above when self._y_log_used is True)
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_fit)

        self._model = self._build(X_s.shape[1])
        self._history = []  # 
        _train_loop(
            self._model, X_s, y_s,
            epochs=self._EPOCHS, lr=self._LR, patience=self._PATIENCE,
            augment=False,
            sample_weights=None,
            curriculum_mode=None,
            history_sink=self._history,
        )
        self._fitted = True
        log.info(
            "  [TinyMLP] sanity-floor baseline (%d→%d→%d→1) 학습 완료 "
            "(y_train_max=%.2f, cap=%.2f)",
            X_s.shape[1], self._HIDDEN[0], self._HIDDEN[1],
            self._y_train_max, max(self._y_train_max * 3.0, 200.0),
        )
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        pred_s = _predict_torch(self._model, X_s)
        y_raw = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        # : log1p 역변환
        # Bug-B fix (2026-04-25): np.expm1 입력이 매우 크면 inf 로 overflow.
        # 학습 시 y_log 가 ~6+ 인 경우 expm1(50) ≈ 5.18e21 → cap 으로 잘려 200,
        # 그러나 expm1(100) → inf 자체가 발생해 inverse_transform 결과를 망침.
        # log 공간에서 [-50, 50] 으로 clip 후 expm1 호출.
        if getattr(self, "_y_log_used", False):
            y_raw = np.expm1(np.clip(y_raw, -50.0, 50.0))
        # D-1: fold-local cap (y_train_max * 1.5)
        # Package O (G-153 fix): cap 강화 (3.0×→1.5×, 200→100)
        # 이전 cap=200 이 TCN-Optuna 의 |pred|=200.8 폭주 허용 → R²=−4.98
        cap = max(self._y_train_max * 1.5, 100.0) if self._y_train_max else 100.0
        # Package L (G-146): NaN/Inf finite guard — log1p inverse 발산 시 reference (mean) 로 대체
        y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
        return np.clip(y_raw, 0.0, cap)


# ═══════════════════════════════════════════════════════════════
# 1b. DNN-Optuna -- Level 10 (Optuna HPO)
# ═══════════════════════════════════════════════════════════════

class OptunaDNNForecaster(BaseForecaster):
    """DNN with Optuna HPO. 50 trials, MedianPruner.

    검색 공간:
    - hidden layer 크기 (n_features 배수), dropout, lr, weight_decay
    - augment_factor, batch_size, n_layers (2-3)
    최적 파라미터는 results/optuna_dnn_best.json에 캐시.
    """

    meta = ModelMeta(
        name="DNN-Optuna",
        category="dl",
        level=10,
        min_data=80,
        description="DNN + Optuna HPO. 자동 하이퍼파라미터 최적화.",
        dependencies=["torch", "optuna"],
    )
    N_TRIALS = 20  # : 50→20 (G-038: OOM 방지). : per_model_trials 로 재정의 가능

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._best_params = None
        self._y_train_max = None  # D-1: fold-local prediction cap
        # : per_model_trials 예산 조회 (env-var 경유)
        from simulation.models._optuna_budget import get_trials as _get_trials
        self.N_TRIALS = _get_trials("DNN-Optuna", default=self.N_TRIALS)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> OptunaDNNForecaster:
        import optuna
        from sklearn.preprocessing import StandardScaler

        # D-1: fold-local 상한 저장
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 1.0

        # Sprint 1.5 R4 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_train)
        n_features = X_s.shape[1]

        def objective(trial):
            import gc

            # : 동적 N-layer 아키텍처 탐색
            n_layers = trial.suggest_int("n_layers", 1, 5)

            # : unit 탐색 범위 2-9999 log-scale (사용자 요청).
            # TPE + MedianPruner 로 과도한 큰 net 자동 가지치기.
            hidden_dims = []
            dropouts = []
            for i in range(n_layers):
                h = trial.suggest_int(f"hidden_{i}", 2, 9999, log=True)
                d = trial.suggest_float(f"dropout_{i}", 0.0, 0.5)
                hidden_dims.append(h)
                dropouts.append(d)

            # r2 (사용자 지시): lr 탐색 범위 1e-5..1e-2 log-scale
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            # G-231 + 2026-05-26 archive: PI augmentation permanently disabled.
            # augment_factor fixed at 0 (was env-var suggest_int with MPH_PI_AUGMENT_LO/HI).
            aug_factor = 0
            bs = trial.suggest_categorical("batch_size", [16, 32, 64])

            # r2: activation 은 GELU 로 고정 (Optuna 탐색 제외, 예산 절약).
            # - GELU 는 LayerNorm+MLP 조합의 사실상 표준 (BERT/GPT/ViT 계열).
            # - ReLU: dead-neuron, Mish: GELU 유사하나 느림, SELU: 특수 init 조건.
            act = "gelu"
            norm_type = trial.suggest_categorical(
                "norm", ["none", "layer", "batch"]
            )
            init_type = trial.suggest_categorical(
                "init", ["kaiming", "xavier", "default"]
            )

            # D-5: optimizer / loss 를 Optuna 탐색에 추가
            opt_type = trial.suggest_categorical(
                "optimizer", ["adamw", "adam", "radam", "rmsprop", "sgd_momentum"]
            )
            loss_choice = trial.suggest_categorical("loss", ["mse", "mae"])  # G-218: huber 영구 제외

            model = _build_dnn_model(
                n_features, hidden_dims=hidden_dims, dropouts=dropouts,
                activation=act, norm=norm_type, init=init_type,
            )

            # P0-3b: R² objective — `-val_r2` 를 minimize 하면 val R² 최대화
            neg_val_r2 = _train_loop(
                model, X_s, y_s,
                epochs=300, lr=lr, batch_size=bs, patience=30,
                weight_decay=wd, augment=True, augment_factor=aug_factor,
                trial=trial,
                optimizer_type=opt_type,
                loss_type=loss_choice,
                return_r2=True,
            )
            # : trial 모델 즉시 삭제 → OOM 방지
            # FIX: model 참조 해제 후 반드시 CUDA allocator 까지 회수
            try:
                del model
            except Exception:
                pass
            try:
                from simulation.models._optuna_torch import _trial_gpu_cleanup
                _trial_gpu_cleanup()
            except Exception:
                import torch as _t
                gc.collect(); gc.collect()  # PEP-442 cycle (ENGINEERING_PRINCIPLES.md #2)
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
                elif hasattr(_t.backends, "mps") and _t.backends.mps.is_available():
                    if hasattr(_t, "mps") and hasattr(_t.mps, "empty_cache"):
                        _t.mps.empty_cache()
            return neg_val_r2

        # P0-3a: Optuna 진행 상황을 로그로 노출
        optuna.logging.set_verbosity(optuna.logging.INFO)

        def _trial_logger(study, trial):
            """각 trial 결과를 [DNN-Optuna] 프리픽스로 로깅."""
            import sys as _sys
            if trial.value is not None:
                try:
                    best_so_far = study.best_value
                except Exception:
                    best_so_far = float("inf")
                val_r2 = -float(trial.value)
                best_r2 = -float(best_so_far) if best_so_far != float("inf") else float("-inf")
                msg = (f"  [DNN-Optuna] Trial {trial.number:>3d}/{self.N_TRIALS}: "
                       f"val_R²={val_r2:+.4f} (best={best_r2:+.4f}) "
                       f"| n_layers={trial.params.get('n_layers')} "
                       f"lr={trial.params.get('lr', 0):.4f} "
                       f"opt={trial.params.get('optimizer')} "
                       f"loss={trial.params.get('loss')}")
                log.info(msg)
                print(msg, flush=True, file=_sys.stdout)

        # ── Storage 통합 (2026-04-27): warm-start 위해 SQLite ──
        # MPH_OPTUNA_FORCE=1 → 기존 study 삭제 후 새로
        # MPH_OPTUNA_FORCE=0 (default) → 있으면 resume, 없으면 새로
        import optuna as _opt_p
        from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
        _store, _name = None, None
        if _GCFG.optuna.use_storage:
            try:
                from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
                _db = get_results_dir() / "optuna_study.db"
                _db.parent.mkdir(parents=True, exist_ok=True)
                _store = f"sqlite:///{_db}"
                from simulation.models._study_ctx import study_ctx_hash as _sctx_dnn
                _name = f"DNN-Optuna_v2_{_sctx_dnn()}"   # G-14H: stale warm-start 무효화
                if _GCFG.optuna.force:
                    try:
                        _opt_p.delete_study(study_name=_name, storage=_store)
                        log.info(f"  [DNN-Optuna] 🗑 FORCE=1 → 기존 study 삭제")
                    except Exception:
                        pass
            except Exception:
                pass

        study = optuna.create_study(
            direction="minimize",  # -R² minimize == R² maximize
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=50,
            ),
            storage=_store,
            study_name=_name if _store else None,
            load_if_exists=bool(_store),
        )

        # 2026-04-28 v2: cap=200, 재학습 자유
        _existing_n = len(study.trials) if _store else 0
        _MAX = _GCFG.optuna.remaining_cap
        _remaining = min(_MAX, self.N_TRIALS)
        if _existing_n > 0:
            log.info(f"  [DNN-Optuna] 🔁 existing {_existing_n} + {_remaining} 추가 (cap={_MAX})")

        # FIX: trial 간 VRAM fragmentation 방지 콜백 추가
        from simulation.models._optuna_torch import make_trial_cleanup_callback as _mk_cb
        if _remaining > 0:
            study.optimize(
                objective, n_trials=_remaining,
                show_progress_bar=True,
                callbacks=[_trial_logger, _mk_cb("DNN-Optuna")],
                gc_after_trial=True,  # G-161 (Codex audit 2026-05-27 fix)
            )

        # : study에서 best_params만 추출 후 study 삭제 → 메모리 해제
        bp = study.best_params
        best_val = study.best_value
        del study
        import gc; gc.collect(); gc.collect()  # PEP-442 cycle (ENGINEERING_PRINCIPLES.md #2)

        # : Retrain with best params -- 3-seed ensemble
        self._best_params = bp
        n_layers_best = bp["n_layers"]
        hidden_dims = [bp[f"hidden_{i}"] for i in range(n_layers_best)]
        dropouts = [bp[f"dropout_{i}"] for i in range(n_layers_best)]

        self._models = []
        self._history = []  # : Optuna 후 3-seed 학습 곡선 수집
        for seed in [42, 2024, 31415]:
            import torch
            torch.manual_seed(seed)
            model = _build_dnn_model(
                n_features, hidden_dims=hidden_dims, dropouts=dropouts,
                activation=bp.get("activation", "relu"),
                norm=bp.get("norm", "none"),
                init=bp.get("init", "default"),
            )
            _seed_hist: list = []
            _train_loop(
                model, X_s, y_s,
                epochs=500, lr=bp["lr"], batch_size=bp["batch_size"],
                patience=40, weight_decay=bp["weight_decay"],
                augment=True, augment_factor=bp.get("augment_factor", 0),  # G-237: objective fixes aug=0 → key absent in best_params
                # D-5: best trial 의 optimizer/loss 를 재학습에도 적용
                optimizer_type=bp.get("optimizer", "adamw"),
                loss_type=bp.get("loss", "mse"),
                history_sink=_seed_hist,
            )
            for _r in _seed_hist:
                _r["seed"] = seed
            self._history.extend(_seed_hist)
            self._models.append(model)
        self._model = self._models[0]
        self._fitted = True
        # P0-3b: best_val 은 -val_r2 이므로 부호 반전해 R² 표기
        best_r2 = -float(best_val) if best_val != float("inf") else float("nan")
        log.info(
            f"  [DNN-Optuna] Best val_R²={best_r2:+.4f}, 3-seed ensemble, "
            f"y_train_max={self._y_train_max:.2f}, cap={max(self._y_train_max*3.0, 200.0):.2f}, "
            f"params={bp}"
        )

        # Cache best params to JSON
        try:
            import json
            from simulation.utils.paths import get_results_dir  # SSOT (MPH_OUTPUT_ROOT)
            cache_path = get_results_dir() / "optuna_dnn_best.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(bp, indent=2, ensure_ascii=False))
        except Exception:
            pass

        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        # : 3-seed ensemble 평균
        if hasattr(self, '_models') and len(self._models) > 1:
            preds = [_predict_torch(m, X_s) for m in self._models]
            pred_s = np.mean(preds, axis=0)
        else:
            pred_s = _predict_torch(self._model, X_s)
        y_raw = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        # D-1: fold-local cap
        # Package O (G-153 fix): cap 강화 (3.0×→1.5×, 200→100)
        # 이전 cap=200 이 TCN-Optuna 의 |pred|=200.8 폭주 허용 → R²=−4.98
        cap = max(self._y_train_max * 1.5, 100.0) if self._y_train_max else 100.0
        # Package L (G-146): NaN/Inf finite guard — log1p inverse 발산 시 reference (mean) 로 대체
        y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
        return np.clip(y_raw, 0.0, cap)


# ═══════════════════════════════════════════════════════════════
# 2. TCN -- Level 13 (Temporal Convolutional Network)
# ═══════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# TCN module-level classes (2026-04-29 fix: champion-log pickle 가능)
# ────────────────────────────────────────────────────────────────
# 이전 closure 정의 (`class TCN` inside `_TCNNet.build()`) 가 pickle 실패:
#   "Can't get local object '_TCNNet.build.<locals>.TCN'"
# → module-level 로 이동. torch 가 없으면 import 시 자체 실패하므로
#   try/except 로 보호 (다른 DL 모델과 동일 패턴).
# ════════════════════════════════════════════════════════════════
try:
    import torch.nn as _tcn_nn

    class _TCNCausalBlock(_tcn_nn.Module):
        """Dilated causal convolution + residual (module-level for pickle)."""
        def __init__(self, in_ch, out_ch, k, dilation, drop):
            super().__init__()
            padding = (k - 1) * dilation
            self.conv1 = _tcn_nn.Conv1d(in_ch, out_ch, k, dilation=dilation, padding=padding)
            self.conv2 = _tcn_nn.Conv1d(out_ch, out_ch, k, dilation=dilation, padding=padding)
            self.relu = _tcn_nn.ReLU()
            self.drop = _tcn_nn.Dropout(drop)
            self.downsample = (_tcn_nn.Conv1d(in_ch, out_ch, 1)
                                if in_ch != out_ch else _tcn_nn.Identity())
            self.padding = padding

        def forward(self, x):
            out = self.drop(self.relu(self.conv1(x)))
            if self.padding > 0:
                out = out[:, :, :-self.padding]
            out = self.drop(self.relu(self.conv2(out)))
            if self.padding > 0:
                out = out[:, :, :-self.padding]
            res = self.downsample(x)
            return self.relu(out + res)

    class _TCNNetwork(_tcn_nn.Module):
        """TCN network — module-level, picklable (champion-log 호환)."""
        def __init__(self, n_features, n_channels, kernel_size, dropout):
            super().__init__()
            layers = []
            in_ch = n_features
            for i, out_ch in enumerate(n_channels):
                dilation = 2 ** i
                layers.append(_TCNCausalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
                in_ch = out_ch
            self.network = _tcn_nn.Sequential(*layers)
            self.fc = _tcn_nn.Linear(n_channels[-1], 1)

        def forward(self, x):
            # x: (batch, seq_len, features) → (batch, features, seq_len)
            out = self.network(x.transpose(1, 2))
            return self.fc(out[:, :, -1])

except ImportError:
    _TCNCausalBlock = None
    _TCNNetwork = None


class _TCNNet:
    @staticmethod
    def build(n_features: int, n_channels: list[int] = None,
              kernel_size: int = 3, dropout: float = 0.3):
        if _TCNNetwork is None:
            raise RuntimeError("TCN requires torch")
        if n_channels is None:
            n_channels = [32, 32, 16]
        return _TCNNetwork(n_features, n_channels, kernel_size, dropout)


class TCNForecaster(BaseForecaster):
    """TCN -- Temporal Convolutional Network, dilated causal convolution.

 최적 설정 ( 2026-03-25):
 - SEQ_LEN: 12, channels: [48, 32, 16]
 - dropout: 0.25, lr: 5e-4, patience: 35
 - Walk-forward 2-fold CV R²=0.78
 """

    meta = ModelMeta(
        name="TCN",
        category="dl",
        level=13,
        min_data=100,
        description="TCN. dilated causal conv로 긴 시퀀스를 효율적으로 처리. 병렬 학습 가능.",
        dependencies=["torch"],
    )

    SEQ_LEN = 12  # 8→12 복원: 과도한 축소는 시간 패턴 손실 (G-029)

    def __init__(self):
        super().__init__()
        self._models = []  # 3-seed ensemble
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._y_train_max = None  # D-1: fold-local prediction cap

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> TCNForecaster:
        import torch
        from sklearn.preprocessing import StandardScaler

        # D-1: fold-local 상한 저장
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 1.0

        # Sprint 1.5 R4 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_train)

        X_seq, y_seq = _make_sequences(X_s, y_s, self.SEQ_LEN)

        # COVID-era 가중치
        n = len(y_train)
        sample_weights = np.ones(n)
        sample_weights[int(n * 0.6):] = 2.0
        sw_seq = sample_weights[self.SEQ_LEN:]
        if len(sw_seq) < len(y_seq):
            sw_seq = np.pad(sw_seq, (0, len(y_seq) - len(sw_seq)), constant_values=1.0)
        sw_seq = sw_seq[:len(y_seq)]

        # 2-seed ensemble (3→2: 메모리 절약, G-044)
        self._models = []
        self._history = []  # : 학습 곡선 수집
        for seed in [42, 2024]:
            torch.manual_seed(seed)
            model = _TCNNet.build(
                n_features=X_s.shape[1],
                n_channels=[32, 16, 8],  # [48,32,16]→[32,16,8] 메모리 절약 (G-044)
                kernel_size=3,
                dropout=0.25,
            )
            _seed_hist: list = []
            _train_loop(
                model, X_seq, y_seq,
                epochs=400, lr=3e-4, patience=40,
                augment=True, augment_factor=2,  # 5→2: 메모리 절약 (G-044)
                sample_weights=sw_seq,
                curriculum_mode='linear',
                batch_size=16,  # 32→16: 메모리 절약 (G-044)
                history_sink=_seed_hist,
            )
            for _r in _seed_hist:
                _r["seed"] = seed
            self._history.extend(_seed_hist)
            self._models.append(model)
            # seed 간 메모리 해제
            import gc; gc.collect(); gc.collect()  # PEP-442 cycle (ENGINEERING_PRINCIPLES.md #2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

        self._model = self._models[0]
        self._fitted = True
        log.info(
            "  [TCN] 3-seed ensemble + curriculum learning 완료 "
            "(y_train_max=%.2f, cap=%.2f)",
            self._y_train_max, max(self._y_train_max * 3.0, 200.0),
        )
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        seqs = []
        for i in range(len(X_s)):
            start = max(0, i + 1 - self.SEQ_LEN)
            seq = X_s[start:i + 1]
            if len(seq) < self.SEQ_LEN:
                pad = np.zeros((self.SEQ_LEN - len(seq), X_s.shape[1]))
                seq = np.vstack([pad, seq])
            seqs.append(seq)
        X_seq = np.array(seqs)

        # 3-seed ensemble 평균
        if self._models:
            all_preds = [_predict_torch(m, X_seq) for m in self._models]
            pred_s = np.mean(all_preds, axis=0)
        else:
            pred_s = _predict_torch(self._model, X_seq)

        y_raw = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        # D-1: fold-local cap
        # Package O (G-153 fix): cap 강화 (3.0×→1.5×, 200→100)
        # 이전 cap=200 이 TCN-Optuna 의 |pred|=200.8 폭주 허용 → R²=−4.98
        cap = max(self._y_train_max * 1.5, 100.0) if self._y_train_max else 100.0
        # Package L (G-146): NaN/Inf finite guard — log1p inverse 발산 시 reference (mean) 로 대체
        y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
        return np.clip(y_raw, 0.0, cap)


# ═══════════════════════════════════════════════════════════════
# 2b. TCN-Optuna -- Level 14 (Optuna HPO)
# ═══════════════════════════════════════════════════════════════

class OptunaTCNForecaster(BaseForecaster):
    """TCN with Optuna HPO. 50 trials, MedianPruner.

    검색 공간:
    - n_channels: 2-4개 채널 레이어, 각 16-64
    - kernel_size: [2, 3, 5]
    - seq_len: [8, 12, 16]
    - dropout, lr, weight_decay, augment_factor, batch_size
    최적 파라미터는 results/optuna_tcn_best.json에 캐시.
    """

    meta = ModelMeta(
        name="TCN-Optuna",
        category="dl",
        level=14,
        min_data=100,
        description="TCN + Optuna HPO. 자동 하이퍼파라미터 최적화.",
        dependencies=["torch", "optuna"],
    )
    N_TRIALS = 20  # : 50→20 (G-038: OOM 방지). : per_model_trials 로 재정의 가능

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._best_params = None
        self._seq_len = 12  # default, updated by HPO
        self._y_train_max = None  # D-1: fold-local prediction cap
        # : per_model_trials 예산 조회
        from simulation.models._optuna_budget import get_trials as _get_trials
        self.N_TRIALS = _get_trials("TCN-Optuna", default=self.N_TRIALS)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> OptunaTCNForecaster:
        import optuna
        from sklearn.preprocessing import StandardScaler

        # D-1: fold-local 상한 저장
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 1.0

        # Sprint 1.5 R4 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_train)
        n_features = X_s.shape[1]

        def objective(trial):
            import gc
            seq_len = trial.suggest_categorical("seq_len", [8, 12, 16])
            n_ch_layers = trial.suggest_int("n_channel_layers", 2, 4)
            channels = []
            for i in range(n_ch_layers):
                ch = trial.suggest_int(f"channel_{i}", 16, 48, step=8)
                channels.append(ch)
            kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 5])
            dropout = trial.suggest_float("dropout", 0.15, 0.40)
            # r2 (사용자 지시): lr 탐색 범위 1e-5..1e-2 log-scale
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
            # G-231 + 2026-05-26 archive: PI augmentation permanently disabled.
            # augment_factor fixed at 0 (was env-var suggest_int with MPH_PI_AUGMENT_LO/HI).
            aug_factor = 0
            bs = trial.suggest_categorical("batch_size", [16, 32])

            X_seq, y_seq = _make_sequences(X_s, y_s, seq_len)
            if len(X_seq) < 30:
                raise optuna.TrialPruned()

            model = _TCNNet.build(
                n_features=n_features,
                n_channels=channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            # P0-3b: R² objective — `-val_r2` 를 minimize 하면 val R² 최대화
            neg_val_r2 = _train_loop(
                model, X_seq, y_seq,
                epochs=300, lr=lr, batch_size=bs, patience=30,
                weight_decay=wd, augment=True, augment_factor=aug_factor,
                trial=trial,
                return_r2=True,
            )
            # : trial 모델 즉시 삭제
            # FIX: CUDA allocator cached block 까지 회수 (VRAM fragmentation 방지)
            try:
                del model, X_seq, y_seq
            except Exception:
                pass
            try:
                from simulation.models._optuna_torch import _trial_gpu_cleanup
                _trial_gpu_cleanup()
            except Exception:
                import torch as _t
                gc.collect(); gc.collect()  # PEP-442 cycle (ENGINEERING_PRINCIPLES.md #2)
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
                elif hasattr(_t.backends, "mps") and _t.backends.mps.is_available():
                    if hasattr(_t, "mps") and hasattr(_t.mps, "empty_cache"):
                        _t.mps.empty_cache()
            return neg_val_r2

        # P0-3a: Optuna 진행 로그 노출
        optuna.logging.set_verbosity(optuna.logging.INFO)

        def _trial_logger(study, trial):
            import sys as _sys
            if trial.value is not None:
                try:
                    best_so_far = study.best_value
                except Exception:
                    best_so_far = float("inf")
                val_r2 = -float(trial.value)
                best_r2 = -float(best_so_far) if best_so_far != float("inf") else float("-inf")
                msg = (f"  [TCN-Optuna] Trial {trial.number:>3d}/{self.N_TRIALS}: "
                       f"val_R²={val_r2:+.4f} (best={best_r2:+.4f}) "
                       f"| seq_len={trial.params.get('seq_len')} "
                       f"n_ch={trial.params.get('n_channel_layers')} "
                       f"kernel={trial.params.get('kernel_size')} "
                       f"lr={trial.params.get('lr', 0):.4f}")
                log.info(msg)
                print(msg, flush=True, file=_sys.stdout)

        # ── Storage 통합 (2026-04-27): TCN-Optuna 도 warm-start ──
        # MPH_OPTUNA_FORCE=1 → 기존 삭제 후 새로
        import optuna as _opt_p2
        from simulation.config_global import GLOBAL as _GCFG2  # SSOT (2026-05-28)
        _store2, _name2 = None, None
        if _GCFG2.optuna.use_storage:
            try:
                from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
                _db2 = get_results_dir() / "optuna_study.db"
                _db2.parent.mkdir(parents=True, exist_ok=True)
                _store2 = f"sqlite:///{_db2}"
                from simulation.models._study_ctx import study_ctx_hash as _sctx_tcn
                _name2 = f"TCN-Optuna_v2_{_sctx_tcn()}"   # G-14H: stale warm-start 무효화
                if _GCFG2.optuna.force:
                    try:
                        _opt_p2.delete_study(study_name=_name2, storage=_store2)
                        log.info(f"  [TCN-Optuna] 🗑 FORCE=1 → 기존 study 삭제")
                    except Exception:
                        pass
            except Exception:
                pass

        study = optuna.create_study(
            direction="minimize",  # -R² minimize == R² maximize
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=50,
            ),
            storage=_store2,
            study_name=_name2 if _store2 else None,
            load_if_exists=bool(_store2),
        )
        # 2026-04-28 v2: cap=200, 재학습 자유
        _existing_n2 = len(study.trials) if _store2 else 0
        _MAX2 = _GCFG2.optuna.remaining_cap
        _remaining2 = min(_MAX2, self.N_TRIALS)
        if _existing_n2 > 0:
            log.info(f"  [TCN-Optuna] 🔁 existing {_existing_n2} + {_remaining2} 추가 (cap={_MAX2})")

        # FIX: trial 간 VRAM fragmentation 방지 콜백 추가
        from simulation.models._optuna_torch import make_trial_cleanup_callback as _mk_cb
        if _remaining2 > 0:
            study.optimize(
                objective, n_trials=_remaining2,
                show_progress_bar=True,
                callbacks=[_trial_logger, _mk_cb("TCN-Optuna")],
                gc_after_trial=True,  # G-161 (Codex audit 2026-05-27 fix)
            )

        # : study에서 필요한 것만 추출 후 삭제
        bp = study.best_params
        best_val = study.best_value
        del study
        import gc; gc.collect(); gc.collect()  # PEP-442 cycle (ENGINEERING_PRINCIPLES.md #2)

        # Retrain with best params
        self._best_params = bp
        self._seq_len = bp["seq_len"]

        # Reconstruct channel list from best params
        channels = [bp[f"channel_{i}"] for i in range(bp["n_channel_layers"])]

        X_seq, y_seq = _make_sequences(X_s, y_s, self._seq_len)
        self._model = _TCNNet.build(
            n_features=n_features,
            n_channels=channels,
            kernel_size=bp["kernel_size"],
            dropout=bp["dropout"],
        )
        self._history = []  # 
        _train_loop(
            self._model, X_seq, y_seq,
            epochs=500, lr=bp["lr"], batch_size=bp["batch_size"],
            patience=40, weight_decay=bp["weight_decay"],
            augment=True, augment_factor=bp.get("augment_factor", 0),  # G-237: objective fixes aug=0 → key absent in best_params
            history_sink=self._history,
        )
        self._fitted = True
        # P0-3b: best_val 은 -val_r2 이므로 부호 반전
        best_r2 = -float(best_val) if best_val != float("inf") else float("nan")
        log.info(
            f"  [TCN-Optuna] Best trial val_R²={best_r2:+.4f}, params={bp}"
        )

        # Cache best params to JSON
        try:
            import json
            from simulation.utils.paths import get_results_dir  # SSOT (MPH_OUTPUT_ROOT)
            cache_path = get_results_dir() / "optuna_tcn_best.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(bp, indent=2, ensure_ascii=False))
        except Exception:
            pass

        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        preds = []
        for i in range(len(X_s)):
            start = max(0, i + 1 - self._seq_len)
            seq = X_s[start:i + 1]
            if len(seq) < self._seq_len:
                pad = np.zeros((self._seq_len - len(seq), X_s.shape[1]))
                seq = np.vstack([pad, seq])
            preds.append(seq)
        X_seq = np.array(preds)
        pred_s = _predict_torch(self._model, X_seq)
        y_raw = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        # D-1: fold-local cap
        # Package O (G-153 fix): cap 강화 (3.0×→1.5×, 200→100)
        # 이전 cap=200 이 TCN-Optuna 의 |pred|=200.8 폭주 허용 → R²=−4.98
        cap = max(self._y_train_max * 1.5, 100.0) if self._y_train_max else 100.0
        # Package L (G-146): NaN/Inf finite guard — log1p inverse 발산 시 reference (mean) 로 대체
        y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
        return np.clip(y_raw, 0.0, cap)


# ═══════════════════════════════════════════════════════════════
# 3. TabularDNN -- Level 11
#    소표본 테이블 데이터 전용 아키텍처
#    핵심: Feature-wise Attention + Bottleneck + Residual
# ═══════════════════════════════════════════════════════════════

def _build_tabular_dnn(n_features: int, hp: dict = None):
    """소표본 최적화 Tabular DNN.

    기존 DNN 문제:
      - nn.Linear(67, 201) → 파라미터 13,000+ >> 샘플 223개
      - 피처 간 interaction을 weight matrix로만 학습 (비효율적)

    개선 (TabNet/FT-Transformer 아이디어 차용):
      1. Feature-wise Attention: 피처별 중요도를 학습 (soft feature selection)
      2. Bottleneck: n_feat → 32 → 64 → 32 → 1 (파라미터 대폭 감소)
      3. Residual connection: skip connection으로 gradient 안정화
      4. Feature interaction layer: 피처 간 곱 자동 학습
    """
    import torch
    import torch.nn as nn

    hp = hp or {}
    bottleneck = hp.get("bottleneck", 32)
    hidden = hp.get("hidden", 64)
    dropout = hp.get("dropout", 0.3)
    use_interaction = hp.get("use_interaction", True)
    n_heads = hp.get("n_heads", 4)
    act_name = hp.get("activation", "gelu")
    norm_name = hp.get("norm", "layer")
    init_name = hp.get("init", "default")
    fm_k = hp.get("fm_k", 8)
    # P1-4: FeatureAttention 게이트 bottleneck 크기 (over-parameterization 방지)
    # 기본 16 → Linear(n,n) 의 O(n²) 파라미터를 O(n·bn) 으로 축소
    # 예: n_feat=313 일 때 97,969 → 10,016 (약 10배 감소)
    attn_bn = hp.get("attn_bottleneck", 16)

    class FeatureAttention(nn.Module):
        """피처별 attention weight 학습 (soft gating) — bottleneck 구조.

 P1-4: Linear(n_feat, n_feat) → Linear(n_feat, bn) + Linear(bn, n_feat).
 n≈343, n_feat=313 의 p≈n 환경에서 attention 게이트 단독으로
 98K 파라미터를 쓰던 구조를 ~10K 로 감소시켜 over-parameterization 완화.
 표현력은 bn=16 (or hp.attn_bottleneck) 선에서 보존.
 """
        def __init__(self, n_feat, bn=attn_bn):
            super().__init__()
            bn_eff = max(4, min(bn, n_feat))  # 극단적 값 방어
            self.gate = nn.Sequential(
                nn.Linear(n_feat, bn_eff),
                nn.ReLU(inplace=True),
                nn.Linear(bn_eff, n_feat),
                nn.Sigmoid(),
            )

        def forward(self, x):
            attn = self.gate(x)
            return x * attn

    class FeatureInteractionLayer(nn.Module):
        """FM 스타일 2차 interaction (n_feat × k 파라미터)."""
        def __init__(self, n_feat, k=8):
            super().__init__()
            self.V = nn.Parameter(torch.randn(n_feat, k) * 0.01)

        def forward(self, x):
            xv = x.unsqueeze(-1) * self.V.unsqueeze(0)
            sum_sq = xv.sum(dim=1).pow(2)
            sq_sum = (xv.pow(2)).sum(dim=1)
            interaction = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)
            return interaction

    class ResidualBlock(nn.Module):
        """Residual FC block (: activation/norm 동적)."""
        def __init__(self, dim, drop=0.2):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                _get_activation_fn(act_name),
                nn.Dropout(drop),
                nn.Linear(dim, dim),
            )
            self.norm = _get_norm_layer(norm_name, dim)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            return self.norm(x + self.drop(self.block(x)))

    class TabularNet(nn.Module):
        """Feature Attention + FM Interaction + Bottleneck Residual."""
        def __init__(self, n_feat):
            super().__init__()
            self.n_feat = n_feat

            # 1. Feature attention (soft feature selection)
            self.feat_attn = FeatureAttention(n_feat)

            # 2. Bottleneck encoder: n_feat → bottleneck
            self.encoder = nn.Sequential(
                nn.Linear(n_feat, bottleneck),
                _get_norm_layer(norm_name, bottleneck),
                _get_activation_fn(act_name),
                nn.Dropout(dropout),
            )

            # 3. Residual blocks (: 동적 개수)
            n_res = hp.get("n_res_blocks", 2)
            self.res_blocks = nn.ModuleList([
                ResidualBlock(bottleneck, dropout) for _ in range(n_res)
            ])

            # 4. Feature interaction (FM)
            self.use_fm = use_interaction
            if self.use_fm:
                self.fm = FeatureInteractionLayer(n_feat, k=fm_k)

            # 5. Head: bottleneck(+1) → hidden → 1
            head_in = bottleneck + (1 if self.use_fm else 0)
            self.head = nn.Sequential(
                nn.Linear(head_in, hidden),
                _get_activation_fn(act_name),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            # Feature attention
            x_attn = self.feat_attn(x)

            # Bottleneck encoding + residual blocks
            h = self.encoder(x_attn)
            for res_block in self.res_blocks:
                h = res_block(h)

            # FM interaction
            if self.use_fm:
                fm_out = self.fm(x)  # (batch, 1)
                h = torch.cat([h, fm_out], dim=-1)

            return self.head(h).squeeze(-1)

    model = TabularNet(n_features)
    _apply_weight_init(model, init_name)
    return model


class TabularDNNForecaster(BaseForecaster):
    """Tabular DNN -- 소표본 테이블 데이터 전용 아키텍처.

    기존 DNN 대비 핵심 차이:
      1. Feature-wise Attention: 267개 중 중요 피처에 자동 집중
      2. FM Interaction Layer: 피처 간 2차 상호작용 자동 학습
         → Tree 모델이 분기 조합으로 하는 것을 명시적으로 학습
      3. Bottleneck (32차원): 파라미터 수 대폭 감소 (13,000 → ~3,000)
      4. Residual + LayerNorm: gradient 안정화

    파라미터 비교:
      기존 DNN:   Linear(67→201) + Linear(201→67) + Linear(67→1) ≈ 27,000
      TabularDNN: Attn(67) + Linear(67→32) + 2×Res(32) + FM(67×8) + Head(33→64→1) ≈ 5,400
    """

    meta = ModelMeta(
        name="TabularDNN",
        category="dl",
        level=11,
        min_data=80,
        description="Tabular DNN with Feature Attention + FM Interaction. 소표본 최적화.",
        dependencies=["torch"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._y_train_max = None  # D-1: fold-local prediction cap

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TabularDNNForecaster":
        import torch
        from sklearn.preprocessing import StandardScaler

        # D-1: fold-local 상한 저장
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 1.0

        # Sprint 1.5 R4 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_train)
        n_features = X_s.shape[1]

        # : kwargs에서 HP 오버라이드 지원 (Optuna 연동)
        hp = {
            "bottleneck": kwargs.get("bottleneck", min(64, max(16, n_features // 2))),
            "hidden": kwargs.get("hidden", 128),
            "dropout": kwargs.get("dropout", 0.3),
            "use_interaction": kwargs.get("use_interaction", True),
            "activation": kwargs.get("activation", "gelu"),
            "norm": kwargs.get("norm", "layer"),
            "init": kwargs.get("init", "kaiming"),
            "fm_k": kwargs.get("fm_k", 8),
            "n_res_blocks": kwargs.get("n_res_blocks", 2),
        }

        # 3-seed ensemble — torch.manual_seed 가 DNN 결정성의 전부.
        # numpy 쪽은 TimeSeriesAugmentor(seed=42) 가 per-instance rng 를 쓰므로
        # np.random global 을 굳이 seed 할 필요 없음 .
        self._models = []
        self._history = []  # : 3-seed 학습 곡선 concat
        for seed in [42, 2024, 31415]:
            torch.manual_seed(seed)
            model = _build_tabular_dnn(n_features, hp)

            _seed_hist: list = []
            _train_loop(
                model, X_s, y_s,
                epochs=500, lr=kwargs.get("lr", 3e-4),
                patience=kwargs.get("patience", 40),
                weight_decay=kwargs.get("weight_decay", 5e-4),
                augment=True, augment_factor=kwargs.get("augment_factor", 4),
                optimizer_type=kwargs.get("optimizer", "adamw"),
                history_sink=_seed_hist,
            )
            for _r in _seed_hist:
                _r["seed"] = seed
            self._history.extend(_seed_hist)
            self._models.append(model)

        self._model = self._models[0]
        self._fitted = True
        log.info(f"  [TabularDNN] 3-seed ensemble 완료 (features={n_features}, "
                 f"bottleneck={hp['bottleneck']}, hidden={hp['hidden']}, "
                 f"act={hp['activation']}, params≈{sum(p.numel() for p in model.parameters())})")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        preds = [_predict_torch(m, X_s) for m in self._models]
        pred_s = np.mean(preds, axis=0)
        y_raw = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        # D-1: fold-local cap
        # Package O (G-153 fix): cap 강화 (3.0×→1.5×, 200→100)
        # 이전 cap=200 이 TCN-Optuna 의 |pred|=200.8 폭주 허용 → R²=−4.98
        cap = max(self._y_train_max * 1.5, 100.0) if self._y_train_max else 100.0
        # Package L (G-146): NaN/Inf finite guard — log1p inverse 발산 시 reference (mean) 로 대체
        y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
        return np.clip(y_raw, 0.0, cap)


# ═══════════════════════════════════════════════════════════════
# TabularDNN-Lite -- (parsimony-first replacement for TabularDNN)
# ═══════════════════════════════════════════════════════════════
class TabularDNNLiteForecaster(BaseForecaster):
    """TabularDNN-Lite -- 신규. n=343 에 맞는 경량 DL.

 Motivation
 ----------
 bench_full.csv 실측:
 TinyMLP R²=0.8166 4 m ~10k params
 DNN(plain) R²=0.8364 3h47m ~30k params
 TabularDNN R²=0.7185 5h55m 58,126 params (169/week)
 TabularDNN 은 n=343 주 데이터에 과도하게 파라미터화되어 TinyMLP 보다
 **더 나쁜** 결과를 보였다. 는 TabularDNN 을 negative_control 로
 유지하면서, attention/FM 없는 경량 변형 **TabularDNN-Lite** 를 신규
 등록한다. TinyMLP 와 TabularDNN 사이 중간 복잡도.

 Architecture (intentionally simple)
 -----------------------------------
 - fixed hidden (64, 64) — does NOT scale with n_features
 - LayerNorm + GELU (modern defaults from TabularDNN idiom)
 - Dropout 0.3
 - NO attention, NO FM, NO bottleneck, NO residual
 - 3-seed ensemble (fair comparison vs TabularDNN's 3-seed ensemble)
 - 200 epochs, lr=3e-4, patience=30

 Parameter count @ n_features=309:
 Linear(309→64) + LayerNorm(64) + Linear(64→64) + LayerNorm(64)
 + Linear(64→1) ≈ 24,000 params (3 seeds averaged, not summed)
 vs TabularDNN 58,126 / TinyMLP 10,000.
 """

    meta = ModelMeta(
        name="TabularDNN-Lite",
        category="dl",
        level=9,  # between TinyMLP(8) and DNN(10)
        min_data=80,
        description=(
            "TabularDNN-Lite. 고정 (64, 64) + LayerNorm + GELU + Dropout 0.3. "
            "Attention/FM 없는 parsimony-first 변형 — bench 에서 "
            "TabularDNN 이 TinyMLP 에 졌던 것을 교정."
        ),
        dependencies=["torch"],
    )

    _HIDDEN = (64, 64)
    _DROPOUT = 0.3
    _EPOCHS = 200
    _LR = 3e-4
    _PATIENCE = 30
    _SEEDS = (42, 2024, 31415)

    def __init__(self):
        super().__init__()
        self._models = None
        self._scaler_X = None
        self._scaler_y = None
        self._y_train_max = None  # D-1: fold-local prediction cap

    def _build(self, n_features: int, seed: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        h1, h2 = self._HIDDEN
        model = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(self._DROPOUT),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(self._DROPOUT),
            nn.Linear(h2, 1),
        )
        _apply_weight_init(model, "kaiming")
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TabularDNNLiteForecaster":
        from sklearn.preprocessing import StandardScaler

        if not _check_torch():
            raise RuntimeError("TabularDNN-Lite requires torch")

        # D-1: fold-local 상한 저장
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 1.0

        # Sprint 1.5 R4 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_train)
        n_features = X_s.shape[1]

        self._models = []
        self._history = []  # : 3-seed 학습 곡선 concat
        for seed in self._SEEDS:
            model = self._build(n_features, seed)
            _seed_hist: list = []
            _train_loop(
                model, X_s, y_s,
                epochs=kwargs.get("epochs", self._EPOCHS),
                lr=kwargs.get("lr", self._LR),
                patience=kwargs.get("patience", self._PATIENCE),
                weight_decay=kwargs.get("weight_decay", 1e-4),
                augment=True,
                augment_factor=kwargs.get("augment_factor", 3),
                optimizer_type="adamw",
                history_sink=_seed_hist,
            )
            for _r in _seed_hist:
                _r["seed"] = seed
            self._history.extend(_seed_hist)
            self._models.append(model)

        self._fitted = True
        total_params = sum(p.numel() for p in self._models[0].parameters())
        log.info(
            "  [TabularDNN-Lite] 3-seed ensemble (features=%d → %d → %d → 1, "
            "params≈%d) 학습 완료",
            n_features, self._HIDDEN[0], self._HIDDEN[1], total_params,
        )
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        preds = [_predict_torch(m, X_s) for m in self._models]
        pred_s = np.mean(preds, axis=0)
        y_raw = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        # D-1: fold-local cap
        # Package O (G-153 fix): cap 강화 (3.0×→1.5×, 200→100)
        # 이전 cap=200 이 TCN-Optuna 의 |pred|=200.8 폭주 허용 → R²=−4.98
        cap = max(self._y_train_max * 1.5, 100.0) if self._y_train_max else 100.0
        # Package L (G-146): NaN/Inf finite guard — log1p inverse 발산 시 reference (mean) 로 대체
        y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
        return np.clip(y_raw, 0.0, cap)


# ═══════════════════════════════════════════════════════════════
# 등록 -- DNN + TCN + Optuna 변형 + TabularDNN (+ Lite)
# LSTM/GRU/BiLSTM: 2026-03-25 제거 (WF R²<0.5, 소표본 부적합)
# TFT: tft_wrapper.py에서 별도 등록
# TabularDNN: 에서 negative_control 로 강등 (see simulation.models.registry.NEGATIVE_CONTROL).
#   registry 에는 남아있지만 ensemble 가중치에서 제외되고, bench/논문용
#   negative result 증거로만 실행된다. TabularDNN-Lite 가 DL Tier A 대표.
# ═══════════════════════════════════════════════════════════════
# 2026-05-26 prune (user explicit): TinyMLP REMOVED — DNN-Optuna + TabularDNN
# cover the parsimony-first slot. Class kept for ad-hoc smoke tests.
# REGISTRY.register(TinyMLPForecaster)  # S2-3: sanity-floor baseline
REGISTRY.register(DNNForecaster)
REGISTRY.register(TCNForecaster)
REGISTRY.register(OptunaDNNForecaster)
REGISTRY.register(OptunaTCNForecaster)
REGISTRY.register(TabularDNNForecaster)  # : negative_control (see registry.py)
REGISTRY.register(TabularDNNLiteForecaster)  # 신규: parsimony-first DL Tier A


# Package C A-3 + A-4: Mixed precision (autocast) + torch.compile
# 안전 fallback — 실패 시 일반 경로 (MPS/CPU 일부 op 미지원 시 자동 fallback).
def package_c_compile_helper(model, device_type: str = "mps"):
    """torch.compile wrap with safe fallback. Returns compiled model or original."""
    from simulation.config_global import GLOBAL as _GCFG_pc  # SSOT (2026-05-28)
    if not _GCFG_pc.package_c.compile_models:
        return model
    try:
        import torch as _torch_pc
        if hasattr(_torch_pc, "compile"):
            mode = "reduce-overhead" if device_type == "mps" else "max-autotune"
            return _torch_pc.compile(model, mode=mode, dynamic=True)
    except Exception as _e_pc:
        print(f"  [package_c] torch.compile skipped: {type(_e_pc).__name__}")
    return model


def package_c_autocast_ctx(device_type: str = "mps"):
    """Returns autocast context manager or no-op."""
    import contextlib
    from simulation.config_global import GLOBAL as _GCFG_pc  # SSOT (2026-05-28)
    if not _GCFG_pc.package_c.autocast:
        return contextlib.nullcontext()
    try:
        from torch import autocast
        import torch as _torch_pc
        if device_type == "cuda":
            return autocast(device_type="cuda", dtype=_torch_pc.float16)
        elif device_type == "mps":
            # MPS autocast 부분 지원 — bf16 안정적
            return autocast(device_type="cpu", dtype=_torch_pc.bfloat16, enabled=False)
        else:
            return contextlib.nullcontext()
    except Exception:
        return contextlib.nullcontext()


# Package C B-C: Pinball loss menu (additive, opt-in via MPH_PC_LOSS)
# G-218: huber 영구 제거 (memory huber-loss-banned-20260520).
def package_c_loss_menu(name: str = "mse", quantile: float = 0.5):
    """Returns loss callable. Default mse. Options: mse | mae | pinball.

    Pinball (quantile regression) — directly learn quantile τ for PI.
    """
    import torch as _torch_pc
    import torch.nn.functional as _F_pc

    def _mse(pred, y, **kw):
        return _F_pc.mse_loss(pred, y)

    def _mae(pred, y, **kw):
        return _F_pc.l1_loss(pred, y)

    def _pinball(pred, y, q: float = quantile, **kw):
        diff = y - pred
        return _torch_pc.mean(_torch_pc.maximum(q * diff, (q - 1) * diff))

    return {
        "mse": _mse, "mae": _mae,
        "pinball": _pinball,
    }.get(name.lower(), _mse)
