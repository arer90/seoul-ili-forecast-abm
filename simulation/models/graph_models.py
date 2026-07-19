"""
simulation/models/graph_models.py
=================================
그래프 신경망(GNN) 기반 ILI rate 예측 모델.

[설계 배경]
 서울시 25개 구의 통근 행렬(CommutingMatrix)을 그래프 인접 행렬로 활용.
 단, 구별 주간 ILI rate가 없으므로 전통적 ST-GCN/DCRNN은 적용 불가.
 → 대안: Graph-Enhanced DNN (GE-DNN)
 1. 입력 피처를 25개 노드에 브로드캐스트
 2. GCN 레이어로 공간적 메시지 패싱 (통근 구조 반영)
 3. 노드 임베딩 풀링 → 최종 서울 ILI rate 예측
 4. 그래프 라플라시안 정규화로 공간 평활성 유도

[구현]
 - PyTorch 기반 (torch_geometric 선택적 의존)
 - torch_geometric 없으면 수동 GCN 구현 (fallback)
 - BaseForecaster ABC 상속
 - dl_models.py의 _train_loop 패턴 재사용

[제한사항]
 - 구별 주간 ILI 없음 → 노드별 타겟 학습 불가 (aggregate-level만)
 - 343주 소표본 → 과적합 위험 높음
 - 학술 기여: 통근 네트워크가 집계 ILI 예측에 기여하는지 평가

변경 이력:
 - (2026-04-13): 초기 구현 (GE-DNN)
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# GCN 유틸리티 (torch_geometric fallback)
# ═══════════════════════════════════════════════════════════════

def _get_adjacency_matrix() -> np.ndarray:
    """CommutingMatrix에서 25×25 인접 행렬 로드.

    실패 시 균등 연결 행렬 반환 (graceful degradation).
    """
    try:
        from simulation.models.metapop_seir import CommutingMatrix
        cm = CommutingMatrix()
        adj = cm.get_matrix()
        if adj is not None and adj.shape == (25, 25):
            return adj
    except Exception as e:
        log.warning(f"[GraphModel] CommutingMatrix 로드 실패: {e}")

    # Fallback: 균등 연결 (self-loop 없음)
    log.info("[GraphModel] 균등 연결 인접 행렬 사용 (fallback)")
    adj = np.ones((25, 25)) / 24.0
    np.fill_diagonal(adj, 0)
    return adj


def _normalize_adjacency(adj: np.ndarray, add_self_loop: bool = True) -> np.ndarray:
    """대칭 정규화: D^{-1/2} A D^{-1/2}.

    GCN (Kipf & Welling, 2017) 표준 정규화.
    """
    if add_self_loop:
        adj = adj + np.eye(adj.shape[0])

    # 대칭화 (통근 행렬은 비대칭 → 대칭화)
    adj = (adj + adj.T) / 2.0

    # D^{-1/2}
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)

    # D^{-1/2} A D^{-1/2}
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    return adj_norm


# ═══════════════════════════════════════════════════════════════
# Graph-safe normalization (R-1 fix)
# ═══════════════════════════════════════════════════════════════
#
# Problem observed in GAT diagnosis experiment iterations 2.1 + 2.2 (2026-04-21):
#
#   Both GE-DNN and GE-GAT crashed with
#     RuntimeError: running_mean should contain 25 elements not 41
#
#   when Optuna picked norm_name="batch". The offending code was
#     _get_norm_layer("batch", node_hidden)  →  nn.BatchNorm1d(node_hidden)
#
#   applied to a 3-D tensor of shape (batch, N=25, node_hidden). PyTorch
#   BatchNorm1d on 3-D input interprets dim 1 as the channel axis, so it
#   expects num_features == N (=25), but the layer was initialized with
#   num_features == node_hidden (e.g. 41 from Optuna). Shape mismatch → crash.
#
# Fix: a tiny adapter that accepts (batch, N, dim) input and normalizes the
# last dim. For "batch" we reshape to 2-D, run BN1d(dim), and restore the
# shape. For "layer" we forward to nn.LayerNorm(dim) which already does the
# right thing on the last dim. This preserves Optuna's ability to search
# over {"layer","batch","none"} without the BN crash.
def _graph_norm_layer(name: str, dim: int):
    """Graph-safe normalization for (batch, N, dim) tensors.

    Args:
        name: one of {"layer", "batch", "none"} (from Optuna HPO).
        dim:  last-axis feature count (node_hidden / gcn_hidden / gat_hidden).

    Returns:
        nn.Module that accepts (batch, N, dim) and returns same shape.
    """
    import torch.nn as nn

    if name == "layer":
        return nn.LayerNorm(dim)

    if name == "batch":
        class _BN1dOnGraph(nn.Module):
            """BatchNorm1d(dim) applied after flattening (batch, N, dim) →
            (batch*N, dim) so PyTorch interprets the correct channel axis."""

            def __init__(self, d: int):
                super().__init__()
                self.bn = nn.BatchNorm1d(d)

            def forward(self, x):
                # x: (batch, N, dim). Flatten node axis into batch,
                # run BN1d(dim), restore.
                b, n, d = x.shape
                return self.bn(x.reshape(b * n, d)).reshape(b, n, d)

        return _BN1dOnGraph(dim)

    return nn.Identity()


# ═══════════════════════════════════════════════════════════════
# GCN Layer (수동 구현 — torch_geometric 불요)
# ═══════════════════════════════════════════════════════════════

def _build_graph_model(n_features: int, adj_norm_tensor, n_nodes: int = 25,
                       hp: dict = None):
    """Graph-Enhanced DNN 모델 빌드 (: activation/norm HP 지원).

 Architecture:
 1. Feature → Node Embedding: Linear(n_features → node_hidden)
 2. GCN Layer ×2: message passing on commuting graph
 3. Readout: mean pooling over nodes
 4. MLP Head: pooled_dim → hidden → 1

 (P0-D): node_hidden / gcn_hidden / mlp_hidden / dropout / init 도
 hp dict 로 수신 → Optuna 탐색 가능.

 Returns:
 (model, adj_tensor) tuple
 """
    import torch
    import torch.nn as nn
    from simulation.models.dl_models import (
        _get_activation_fn, _get_norm_layer, _apply_weight_init,
    )

    hp = hp or {}
    act_name = hp.get("activation", "relu")
    norm_name = hp.get("norm", "layer")
    init_name = hp.get("init", "default")
    node_hidden = int(hp.get("node_hidden", 32))
    gcn_hidden = int(hp.get("gcn_hidden", 32))
    mlp_hidden = int(hp.get("mlp_hidden", 64))
    drop_enc = float(hp.get("dropout_enc", 0.2))
    drop_gcn = float(hp.get("dropout_gcn", 0.3))
    drop_head = float(hp.get("dropout_head", 0.3))

    class GCNLayer(nn.Module):
        """수동 GCN 레이어: H' = σ(Â @ H @ W + b)."""

        def __init__(self, in_dim, out_dim, adj):
            super().__init__()
            # r3: register_buffer 로 등록해야 model.to(device) 따라 이동.
            self.register_buffer("adj", adj, persistent=False)
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            # x: (batch, N, in_dim)
            # adj: (N, N)
            # Â @ H: spatial aggregation
            agg = torch.matmul(self.adj.unsqueeze(0), x)  # (batch, N, in_dim)
            out = self.linear(agg)  # (batch, N, out_dim)
            return out

    # ════════════════════════════════════════════════════════════════════════
    # ⚠ ARCHIVED (2026-05-28) — 사용자 명시 "GE-DNN 없는데 왜 코드 있어?"
    # GraphEnhancedDNN class + Forecaster 모두 registry register X (L981 주석).
    # CATEGORY_MODELS["graph"] = ["GAT", "GCN"] 만 active.
    # class 정의 보존 이유: GAT (GraphAttentionDNNForecaster) 와 일부 helper 공유.
    # 완전 archive 는 다음 sprint (caller dependency 분리 후).
    # ════════════════════════════════════════════════════════════════════════
    class GraphEnhancedDNN(nn.Module):
        """GE-DNN: 통근 그래프를 활용한 ILI 예측 모델. ⚠ ARCHIVED (register X, 2026-05-28)."""

        def __init__(self, n_feat, adj_t, n_nodes=25):
            super().__init__()
            self.n_nodes = n_nodes
            # r3: register_buffer → model.to(device) 시 자동 이동.
            self.register_buffer("adj", adj_t, persistent=False)

            # node_hidden / gcn_hidden / mlp_hidden 는 closure 스코프에서 읽음
            # (Optuna hp dict 주입으로 바뀔 수 있음, P0-D)

            # 1. Feature → Node embedding
            # R-1: _graph_norm_layer handles (batch, N=25, dim) inputs
            # correctly — BN1d(dim) on 3-D tensor previously crashed because
            # PyTorch interpreted N as the channel axis.
            self.node_encoder = nn.Sequential(
                nn.Linear(n_feat, node_hidden),
                _graph_norm_layer(norm_name, node_hidden),
                _get_activation_fn(act_name),
                nn.Dropout(drop_enc),
            )

            # 2. GCN layers (2-layer) -- in == out 으로 residual 차원 맞춤
            gcn_dim = node_hidden
            self.gcn1 = GCNLayer(gcn_dim, gcn_dim, adj_t)
            self.gcn2 = GCNLayer(gcn_dim, gcn_dim, adj_t)
            self.gcn_norm1 = _graph_norm_layer(norm_name, gcn_dim)
            self.gcn_norm2 = _graph_norm_layer(norm_name, gcn_dim)
            self.gcn_act = _get_activation_fn(act_name)
            self.gcn_drop = nn.Dropout(drop_gcn)

            # 3. Readout + MLP head
            self.head = nn.Sequential(
                nn.Linear(gcn_dim * 2, mlp_hidden),
                _get_activation_fn(act_name),
                nn.Dropout(drop_head),
                nn.Linear(mlp_hidden, max(8, mlp_hidden // 2)),
                _get_activation_fn(act_name),
                nn.Linear(max(8, mlp_hidden // 2), 1),
            )

        def forward(self, x):
            """
            x: (batch, n_features)
            → node broadcast → GCN → readout → prediction
            """
            batch_size = x.size(0)

            # Broadcast features to all nodes: (batch, n_feat) → (batch, N, n_feat)
            x_nodes = x.unsqueeze(1).expand(-1, self.n_nodes, -1)

            # Node encoding
            h = self.node_encoder(x_nodes)  # (batch, N, node_hidden)

            # GCN layer 1 (residual)
            h1 = self.gcn_act(self.gcn_norm1(self.gcn1(h)))
            h1 = self.gcn_drop(h1) + h  # residual (dims must match)

            # GCN layer 2
            h2 = self.gcn_act(self.gcn_norm2(self.gcn2(h1)))
            h2 = self.gcn_drop(h2) + h1  # residual

            # Readout: mean + max pooling
            h_mean = h2.mean(dim=1)  # (batch, gcn_hidden)
            h_max = h2.max(dim=1).values  # (batch, gcn_hidden)
            h_pool = torch.cat([h_mean, h_max], dim=-1)  # (batch, 2*gcn_hidden)

            # MLP head
            out = self.head(h_pool)  # (batch, 1)
            return out.squeeze(-1)  # (batch,)

        def graph_laplacian_loss(self, x):
            """그래프 라플라시안 정규화 (공간 평활성).

            L_graph = Σ_i Σ_j A_ij ||h_i - h_j||^2
            node embedding이 이웃과 유사하도록 유도.
            """
            batch_size = x.size(0)
            x_nodes = x.unsqueeze(1).expand(-1, self.n_nodes, -1)
            h = self.node_encoder(x_nodes)  # (batch, N, hidden)

            # A_ij * ||h_i - h_j||^2
            # h_diff: (batch, N, N, hidden)
            h_i = h.unsqueeze(2).expand(-1, -1, self.n_nodes, -1)
            h_j = h.unsqueeze(1).expand(-1, self.n_nodes, -1, -1)
            diff_sq = ((h_i - h_j) ** 2).sum(dim=-1)  # (batch, N, N)

            adj = self.adj.unsqueeze(0)  # (1, N, N)
            loss = (adj * diff_sq).sum(dim=(1, 2)).mean()
            return loss

    model = GraphEnhancedDNN(n_features, adj_norm_tensor, n_nodes)
    _apply_weight_init(model, init_name)
    return model


# ═══════════════════════════════════════════════════════════════
# 학습 루프 (dl_models.py 패턴 참고, graph 전용)
# ═══════════════════════════════════════════════════════════════

def _train_graph_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    lr: float = 5e-4,
    batch_size: int = 32,
    patience: int = 25,
    weight_decay: float = 1e-3,
    graph_reg_weight: float = 0.01,
    augment: bool = True,
    augment_factor: int = 2,
    history_sink: "Optional[list]" = None,    # 2026-05-28 V2: DL/modern-ts loss curve
) -> float:
    """GAT 학습 루프 helper (historically named "Graph-Enhanced DNN").

    2026-05-28 사용자 명시 정정 — GE-DNN 자체는 archived (registry X).
    본 함수는 GAT (GraphAttentionDNNForecaster) 의 train loop helper.

    Returns:
        best_val_loss (float)

    Args (added 2026-05-28 V2 사용자 명시 "DL/modern-ts 학습 그래프"):
        history_sink: optional list — if given, per-epoch
            {epoch, train_loss, val_loss, lr} 기록 (plotting.py 의 learning_curve_<model>.png 위한 input).
    """
    from typing import Optional  # noqa: F401
    import torch
    import torch.nn as nn

    # 2026-05-20 사용자 영구 명시: "모든 epoch는 100이야"
    # 2026-05-21 Gemini fix: override → ceiling — GE-GAT prior stability fix
    # (epochs=60 line 846, B FIX 2026-04-22) 보존 필요. min() 으로 cap.
    from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
    _moe = _GCFG.training.max_epochs_override
    if _moe > 0:
        epochs = min(epochs, _moe)

    # Mac/Win/Linux: cuda > mps > cpu. MPH_DEVICE / MPH_FORCE_CPU 로 override.
    from simulation.models.base import pick_device
    device = pick_device()
    model = model.to(device)

    # ════════════════════════════════════════════════════════════════
    # Package M v2 (G-150): split FIRST + augment ONLY train portion
    # ════════════════════════════════════════════════════════════════
    # 이전 버그: augment 후 split 하면 augmented val copy 가 train 에 포함
    #          → val R²=+1.0 leakage (TCN-Optuna 와 동일 mechanism).
    # Fix: 원본 train/val split 먼저, augment 는 train portion 만.
    # ════════════════════════════════════════════════════════════════
    n_val_orig = max(int(len(X_train) * 0.2), 5)
    X_train_only = X_train[:-n_val_orig]
    y_train_only = y_train[:-n_val_orig]
    X_val_only = X_train[-n_val_orig:]
    y_val_only = y_train[-n_val_orig:]

    X_tr_t = torch.FloatTensor(X_train_only).to(device)
    y_tr_t = torch.FloatTensor(y_train_only).to(device)

    # 증강 (jittering) — train portion 만, val 은 원본 그대로
    if augment and len(X_train_only) < 500:
        aug_X, aug_y = [], []
        for _ in range(augment_factor):
            noise = torch.randn_like(X_tr_t) * 0.02
            aug_X.append(X_tr_t + noise)
            aug_y.append(y_tr_t + torch.randn_like(y_tr_t) * y_tr_t.std() * 0.01)
        X_tr_t = torch.cat([X_tr_t] + aug_X, dim=0)
        y_tr_t = torch.cat([y_tr_t] + aug_y, dim=0)

    # Validation: 원본 (augment 안 함, leakage 방지)
    X_val = torch.FloatTensor(X_val_only).to(device)
    y_val = torch.FloatTensor(y_val_only).to(device)
    X_tr = X_tr_t
    y_tr = y_tr_t

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    criterion = nn.MSELoss()  # G-218: huber 영구 제거 (huber-loss-banned-20260520)

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        # Mini-batch SGD
        indices = torch.randperm(len(X_tr))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_tr), batch_size):
            idx = indices[i:i + batch_size]
            x_batch = X_tr[idx]
            y_batch = y_tr[idx]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            # Graph Laplacian regularization
            if graph_reg_weight > 0:
                g_loss = model.graph_laplacian_loss(x_batch)
                loss = loss + graph_reg_weight * g_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        model.train()

        # 2026-05-28 V2: history sink (plotting.py learning_curve_<model>.png)
        if history_sink is not None:
            history_sink.append({
                "epoch": int(epoch),
                "train_loss": float(total_loss / max(n_batches, 1)),
                "val_loss": float(val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            # 가중치 저장
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  [GE-DNN] 조기종료 epoch={epoch}, best_val_loss={best_val_loss:.6f}")
                break

    # 최적 가중치 복원
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    return best_val_loss


# ═══════════════════════════════════════════════════════════════
# GraphEnhancedDNNForecaster (BaseForecaster 상속)
# ═══════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# ⚠ ARCHIVED (2026-05-28) — class 정의 보존 but registry.register X (L981).
# 활성화: graph_models.py L981 의 register comment 해제 + CATEGORY_MODELS graph 에 "GE-DNN" 추가.
# ════════════════════════════════════════════════════════════════════════════
class GraphEnhancedDNNForecaster(BaseForecaster):
    """
    Graph-Enhanced DNN (GE-DNN): 서울시 통근 그래프 기반 ILI 예측.

    Architecture:
      - Input: (batch, n_features) — 기존 267 피처 동일
      - Node broadcast → 25개 구로 복제
      - 2-layer GCN on 통근 인접 행렬
      - Mean+Max pooling → MLP head → ILI rate 예측

    학술적 의의:
      - 메타개체군(metapopulation) 통근 구조가 집계 ILI 예측에 기여하는지 정량 평가
      - Ablation: graph regularization weight=0 vs >0 비교

    제한:
      - 구별 주간 ILI 부재 → 노드별 지도학습 불가
      - 343주 소표본 → 그래프 구조의 정보 이득이 제한적일 수 있음
    """

    meta = ModelMeta(
        name="GE-DNN",
        category="dl",
        level=15,
        min_data=80,
        description="Graph-Enhanced DNN with Seoul commuting network (25 districts)",
        requires_gpu=False,
        dependencies=["torch"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._adj_norm = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "GraphEnhancedDNNForecaster":
        """학습. (P0-D): Optuna HPO 내장."""
        try:
            import torch
        except ImportError:
            log.error("[GE-DNN] PyTorch가 설치되지 않았습니다.")
            return self

        from sklearn.preprocessing import StandardScaler
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop,
            UNIT_MIN_DEFAULT, UNIT_MAX_DEFAULT,
            ACTIVATIONS, NORMS, INITS,
        )

        # 스케일링
        self._scaler_X = StandardScaler()
        X_scaled = self._scaler_X.fit_transform(X_train)

        self._scaler_y = StandardScaler()
        y_scaled = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # 인접 행렬 정규화
        adj_raw = _get_adjacency_matrix()
        self._adj_norm = _normalize_adjacency(adj_raw, add_self_loop=True)
        adj_tensor = torch.FloatTensor(self._adj_norm)
        n_features = X_scaled.shape[1]

        n_trials = get_trials("GE-DNN", default=0)

        def _static_defaults():
            self._model = _build_graph_model(n_features, adj_tensor, n_nodes=25)
            _train_graph_model(
                self._model, X_scaled, y_scaled,
                epochs=200, lr=5e-4, batch_size=32,
                patience=25, weight_decay=1e-3,
                graph_reg_weight=0.01,
            )

        def _objective(trial):
            import gc
            # node/gcn/mlp hidden 은 Optuna 로 탐색 (log 2..9999)
            hp = {
                "node_hidden": trial.suggest_int("node_hidden", UNIT_MIN_DEFAULT, UNIT_MAX_DEFAULT, log=True),
                "mlp_hidden": trial.suggest_int("mlp_hidden", UNIT_MIN_DEFAULT, UNIT_MAX_DEFAULT, log=True),
                "dropout_enc": trial.suggest_float("dropout_enc", 0.0, 0.5),
                "dropout_gcn": trial.suggest_float("dropout_gcn", 0.0, 0.5),
                "dropout_head": trial.suggest_float("dropout_head", 0.0, 0.5),
                "activation": trial.suggest_categorical("activation", ACTIVATIONS),
                "norm": trial.suggest_categorical("norm", NORMS),
                "init": trial.suggest_categorical("init", INITS),
            }
            tr = suggest_training_hp(trial)
            graph_reg = trial.suggest_float("graph_reg_weight", 1e-4, 1e-1, log=True)

            model = _build_graph_model(n_features, adj_tensor, n_nodes=25, hp=hp)
            best_val = _train_graph_model(
                model, X_scaled, y_scaled,
                epochs=120, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=20, weight_decay=tr["weight_decay"],
                graph_reg_weight=graph_reg,
                augment=True, augment_factor=tr["augment_factor"],
            )
            del model
            gc.collect()
            return float(best_val)

        best, _ = run_optuna_loop("GE-DNN", _objective, n_trials, _static_defaults)
        if best:
            hp = {
                "node_hidden": best.get("node_hidden", 32),
                "mlp_hidden": best.get("mlp_hidden", 64),
                "dropout_enc": best.get("dropout_enc", 0.2),
                "dropout_gcn": best.get("dropout_gcn", 0.3),
                "dropout_head": best.get("dropout_head", 0.3),
                "activation": best.get("activation", "relu"),
                "norm": best.get("norm", "layer"),
                "init": best.get("init", "default"),
            }
            self._model = _build_graph_model(n_features, adj_tensor, n_nodes=25, hp=hp)
            _train_graph_model(
                self._model, X_scaled, y_scaled,
                epochs=300, lr=best.get("lr", 5e-4),
                batch_size=best.get("batch_size", 32),
                patience=30,
                weight_decay=best.get("weight_decay", 1e-3),
                graph_reg_weight=best.get("graph_reg_weight", 0.01),
                augment=True, augment_factor=best.get("augment_factor", 2),
            )
        self._fitted = True
        log.info(f"[GE-DNN] 학습 완료 (features={n_features}, nodes=25, trials={n_trials})")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """예측."""
        if not self._fitted or self._model is None:
            log.warning("[GE-DNN] 학습되지 않은 모델로 예측 시도")
            return np.zeros(len(X_test))

        import torch

        X_scaled = self._scaler_X.transform(X_test)
        # device-fix: model trained on cuda — send input to same device.
        model_device = next(self._model.parameters()).device
        X_t = torch.FloatTensor(X_scaled).to(model_device)

        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(X_t).cpu().numpy()

        # 역변환
        pred = self._scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        return np.maximum(pred, 0)  # ILI rate ≥ 0


# ═══════════════════════════════════════════════════════════════
# GE-GAT: Graph Attention Network (GATv2Conv)
# ═══════════════════════════════════════════════════════════════
#
# Veličković et al. (2018) "Graph Attention Networks"
# Brody et al. (2022) "How Attentive are Graph Attention Networks?" (GATv2)
#
# 동기:
#   - GCN 은 대칭 정규화 인접 행렬로 고정 가중. 통근 비대칭 무시.
#   - GAT 은 (h_i, h_j) 쌍별 attention weight 를 학습 → 25×25 이질적
#     연결을 데이터-의존적으로 가중.
#   - torch_geometric 미설치 환경에서는 수동 MHA-graph attention 폴백.
# ═══════════════════════════════════════════════════════════════

def _build_gat_model(n_features: int, edge_index_tensor, edge_weight_tensor,
                     n_nodes: int = 25, hp: dict = None):
    """GATv2 기반 Graph-Attention DNN 모델 빌드.

 torch_geometric 가용 → GATv2Conv 사용.
 미가용 → MHA 수동 구현 (fallback).

 (P0-D): node_hidden/gat_hidden/mlp_hidden/dropout/init 도 hp 로 수신.
 """
    import torch
    import torch.nn as nn
    from simulation.models.dl_models import (
        _get_activation_fn, _get_norm_layer, _apply_weight_init,
    )

    hp = hp or {}
    act_name = hp.get("activation", "gelu")
    norm_name = hp.get("norm", "layer")
    init_name = hp.get("init", "default")
    heads = int(hp.get("gat_heads", 4))
    node_hidden = int(hp.get("node_hidden", 32))
    # R-1b: GATv2Conv 는 out_dim // heads 를 per-head dim 으로 쓰고
    # concat=True 면 실제 출력 = heads × (out_dim // heads). out_dim % heads
    # != 0 이면 dim 이 잘려 gat_norm*/residual 와 mismatch 가 난다.
    # (예: node_hidden=41, heads=2 → 실제 출력 40). 여기서 미리 정합을
    # 맞춰 두면 gat_hidden = node_hidden 을 그대로 쓸 수 있다.
    if heads > 0 and node_hidden % heads != 0:
        node_hidden = max(heads, (node_hidden // heads) * heads)
    mlp_hidden = int(hp.get("mlp_hidden", 64))
    drop_enc = float(hp.get("dropout_enc", 0.2))
    drop_gat = float(hp.get("dropout_gat", 0.3))
    drop_head = float(hp.get("dropout_head", 0.3))

    try:
        from torch_geometric.nn import GATv2Conv
        _PYG = True
    except ImportError:
        _PYG = False
        log.info("[GE-GAT] torch_geometric 미가용 → MHA fallback")

    class _PYGGAT(nn.Module):
        """torch_geometric.GATv2Conv 를 쓰는 정식 GAT 블록."""

        def __init__(self, in_dim, out_dim, n_heads, edge_idx, edge_w):
            super().__init__()
            # r3: register_buffer → model.to(device) 시 자동 이동.
            self.register_buffer("edge_index", edge_idx, persistent=False)
            if edge_w is not None:
                self.register_buffer("edge_weight", edge_w, persistent=False)
            else:
                self.edge_weight = None
            self.conv = GATv2Conv(
                in_dim, out_dim // n_heads,
                heads=n_heads, concat=True,
                edge_dim=1, dropout=0.1, add_self_loops=True,
            )

        def forward(self, x):
            # x: (batch, N, in_dim). G-FIX (2026-06-03): per-sample Python 루프 제거 — 배치 b개를
            # (b*N, in_dim) 으로 펴고 offset edge_index 로 단일 GATv2Conv 호출(벡터화). 이전 루프가
            # b 배 순차 conv = GAT 느림 주범(attention 아님; GCN(pyg) 과 동일 배치 방식). 결과 동치
            # (offset 으로 샘플 간 edge 없음 → 각 subgraph 내 attention).
            from simulation.models.graph_models_pyg import _batched_edge_index
            b, n, d = x.shape
            x_flat = x.reshape(b * n, d)
            ei_b, ew_b = _batched_edge_index(self.edge_index, self.edge_weight, b, n)
            ew = ew_b.unsqueeze(-1) if ew_b is not None else None
            return self.conv(x_flat, ei_b, edge_attr=ew).reshape(b, n, -1)

    class _ManualGAT(nn.Module):
        """torch_geometric 미가용 시 MHA 기반 대체 GAT."""

        def __init__(self, in_dim, out_dim, n_heads, edge_w_dense):
            super().__init__()
            # r3: register_buffer → model.to(device) 시 자동 이동.
            if edge_w_dense is not None:
                self.register_buffer("edge_w", edge_w_dense, persistent=False)
            else:
                self.edge_w = None
            self.mha = nn.MultiheadAttention(
                embed_dim=in_dim, num_heads=n_heads,
                dropout=0.1, batch_first=True,
            )
            self.proj = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            # x: (batch, N, in_dim). MHA attn-mask 로 edge 가중 반영.
            attn_bias = None
            if self.edge_w is not None:
                # 작은 edge weight → 큰 마스크 페널티 (음수 bias)
                attn_bias = torch.log(self.edge_w.clamp_min(1e-6))
            h, _ = self.mha(x, x, x, attn_mask=attn_bias, need_weights=False)
            return self.proj(h)

    class GraphAttentionDNN(nn.Module):
        """GE-GAT: 2-layer GAT + pooling + MLP."""

        def __init__(self, n_feat, n_nodes=25):
            super().__init__()
            self.n_nodes = n_nodes
            gat_hidden = node_hidden  # keep residual dim aligned

            # R-1: _graph_norm_layer fix (same bug as GE-DNN above).
            self.node_encoder = nn.Sequential(
                nn.Linear(n_feat, node_hidden),
                _graph_norm_layer(norm_name, node_hidden),
                _get_activation_fn(act_name),
                nn.Dropout(drop_enc),
            )

            if _PYG:
                self.gat1 = _PYGGAT(node_hidden, gat_hidden, heads,
                                    edge_index_tensor, edge_weight_tensor)
                self.gat2 = _PYGGAT(gat_hidden, gat_hidden, heads,
                                    edge_index_tensor, edge_weight_tensor)
            else:
                # edge_weight_tensor 를 (N,N) dense 로 복원 시도
                dense = None
                if edge_weight_tensor is not None and edge_index_tensor is not None:
                    dense = torch.zeros(n_nodes, n_nodes)
                    ei = edge_index_tensor
                    for k in range(ei.shape[1]):
                        i, j = int(ei[0, k].item()), int(ei[1, k].item())
                        dense[i, j] = float(edge_weight_tensor[k].item())
                self.gat1 = _ManualGAT(node_hidden, gat_hidden, heads, dense)
                self.gat2 = _ManualGAT(gat_hidden, gat_hidden, heads, dense)

            self.gat_norm1 = _graph_norm_layer(norm_name, gat_hidden)
            self.gat_norm2 = _graph_norm_layer(norm_name, gat_hidden)
            self.gat_act = _get_activation_fn(act_name)
            self.gat_drop = nn.Dropout(drop_gat)

            self.head = nn.Sequential(
                nn.Linear(gat_hidden * 2, mlp_hidden),
                _get_activation_fn(act_name),
                nn.Dropout(drop_head),
                nn.Linear(mlp_hidden, max(8, mlp_hidden // 2)),
                _get_activation_fn(act_name),
                nn.Linear(max(8, mlp_hidden // 2), 1),
            )

        def forward(self, x):
            b = x.size(0)
            x_nodes = x.unsqueeze(1).expand(-1, self.n_nodes, -1)
            h = self.node_encoder(x_nodes)

            h1 = self.gat_act(self.gat_norm1(self.gat1(h)))
            h1 = self.gat_drop(h1) + h

            h2 = self.gat_act(self.gat_norm2(self.gat2(h1)))
            h2 = self.gat_drop(h2) + h1

            h_mean = h2.mean(dim=1)
            h_max = h2.max(dim=1).values
            h_pool = torch.cat([h_mean, h_max], dim=-1)

            return self.head(h_pool).squeeze(-1)

        def graph_laplacian_loss(self, x):
            """GAT 에서도 동일한 smoothness prior 유지 (dense adj 사용)."""
            if not hasattr(self, "_dense_adj_buf"):
                return torch.tensor(0.0, device=x.device)
            b = x.size(0)
            x_nodes = x.unsqueeze(1).expand(-1, self.n_nodes, -1)
            h = self.node_encoder(x_nodes)
            h_i = h.unsqueeze(2).expand(-1, -1, self.n_nodes, -1)
            h_j = h.unsqueeze(1).expand(-1, self.n_nodes, -1, -1)
            diff_sq = ((h_i - h_j) ** 2).sum(dim=-1)
            adj = self._dense_adj_buf.unsqueeze(0).to(x.device)
            return (adj * diff_sq).sum(dim=(1, 2)).mean()

    m = GraphAttentionDNN(n_features, n_nodes)
    _apply_weight_init(m, init_name)
    return m


def _adj_to_edge_index(adj: np.ndarray, threshold: float = 0.0):
    """Dense adjacency → (edge_index, edge_weight) COO representation.

    Returns:
        edge_index: (2, E) LongTensor
        edge_weight: (E,) FloatTensor
    """
    import torch
    src, dst = np.where(adj > threshold)
    w = adj[src, dst]
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    return edge_index, edge_weight


class GraphAttentionDNNForecaster(BaseForecaster):
    """GE-GAT: GATv2 기반 attention graph 모델.

    - 통근 비대칭을 MultiHeadAttention 으로 학습.
    - torch_geometric 가용 시 GATv2Conv, 미가용 시 MHA fallback.
    - 동일한 학습 파이프라인(_train_graph_model) 재사용: forward 서명 동일.

    학술적 차별점:
      - GE-DNN(GCN): 고정 대칭 정규화 가중 → 구조적 사전확률 강함.
      - GE-GAT: 데이터-의존 attention → 시기별 통근 중요도 변이 포착.
    """

    meta = ModelMeta(
        name="GAT",
        category="dl",
        level=16,
        min_data=80,
        description="Graph-Attention DNN (GATv2) on Seoul commuting network.",
        requires_gpu=False,
        dependencies=["torch"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._edge_index = None
        self._edge_weight = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "GraphAttentionDNNForecaster":
        """(P0-D): Optuna HPO 내장."""
        try:
            import torch
        except ImportError:
            log.error("[GE-GAT] PyTorch 미설치 → 학습 스킵")
            return self

        from sklearn.preprocessing import StandardScaler
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop,
            UNIT_MIN_DEFAULT, UNIT_MAX_GAT,
            ACTIVATIONS, NORMS, INITS,
        )

        self._scaler_X = StandardScaler()
        X_scaled = self._scaler_X.fit_transform(X_train)

        self._scaler_y = StandardScaler()
        y_scaled = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap (GCN parity)

        adj_raw = _get_adjacency_matrix()
        # GAT 은 정규화 없이 raw 엣지 가중치 전달 (attention 이 내부적으로 softmax)
        self._edge_index, self._edge_weight = _adj_to_edge_index(adj_raw, threshold=1e-6)
        n_features = X_scaled.shape[1]

        n_trials = get_trials("GE-GAT", default=0)

        def _static_defaults():
            self._model = _build_gat_model(
                n_features, self._edge_index, self._edge_weight, n_nodes=25,
            )
            _train_graph_model(
                self._model, X_scaled, y_scaled,
                epochs=200, lr=5e-4, batch_size=32,
                patience=25, weight_decay=1e-3,
                graph_reg_weight=0.0,
            )

        def _objective(trial):
            # FIX (2026-04-22): vanilla-GAT 진단 기반 탐색공간 축소
            #   1. gat_heads ∈ [2, 4] — 8-head × 큰 hidden = VRAM 폭발 + kernel stall
            #   2. node_hidden/mlp_hidden ≤ UNIT_MAX_GAT(256) — vanilla 진단 결과
            #      32-unit 으로도 loss 수렴했음, 9999 상한은 TPE 예산 낭비 + stall 유발
            #   3. epochs 120 → 80 — 진단 상 val loss 는 ep11 에 plateau, 80 은 안전 margin
            #   4. patience 20 → 15 — pruner 가 더 빠르게 dead-end trial 을 잘라냄
            # -B FIX (2026-04-22, Iter-11 GAT diagnosis experiment 2.1 [2/3] 재실패 후 추가 축소):
            #   heads ∈ {2} 고정 (4 제거), epochs 80→60, patience 15→10, UNIT_MAX_GAT 256→128.
            #   TPE 가 상한 근처 trial 을 피하도록 탐색공간 자체를 반감.
            import gc
            gat_heads = trial.suggest_categorical("gat_heads", [2])
            hp = {
                "gat_heads": gat_heads,
                # node_hidden 은 heads 배수로 반올림 → GATv2 concat 안전
                "node_hidden": (trial.suggest_int("node_hidden_raw", UNIT_MIN_DEFAULT, UNIT_MAX_GAT, log=True)
                                + gat_heads - 1) // gat_heads * gat_heads,
                "mlp_hidden": trial.suggest_int("mlp_hidden", UNIT_MIN_DEFAULT, UNIT_MAX_GAT, log=True),
                "dropout_enc": trial.suggest_float("dropout_enc", 0.0, 0.5),
                "dropout_gat": trial.suggest_float("dropout_gat", 0.0, 0.5),
                "dropout_head": trial.suggest_float("dropout_head", 0.0, 0.5),
                "activation": trial.suggest_categorical("activation", ACTIVATIONS),
                "norm": trial.suggest_categorical("norm", NORMS),
                "init": trial.suggest_categorical("init", INITS),
            }
            tr = suggest_training_hp(trial)

            model = _build_gat_model(
                n_features, self._edge_index, self._edge_weight, n_nodes=25, hp=hp,
            )
            best_val = _train_graph_model(
                model, X_scaled, y_scaled,
                epochs=60, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=10, weight_decay=tr["weight_decay"],
                graph_reg_weight=0.0,
                augment=True, augment_factor=tr["augment_factor"],
            )
            del model
            gc.collect()
            return float(best_val)

        best, _ = run_optuna_loop("GE-GAT", _objective, n_trials, _static_defaults)
        if best:
            # : clip best HP to the tightened search space — 구 버전 study 재사용 방어.
            gat_heads = best.get("gat_heads", 4)
            if gat_heads not in (2, 4):
                gat_heads = 4
            raw_hid = min(int(best.get("node_hidden_raw", 32)), UNIT_MAX_GAT)
            node_hidden = ((raw_hid + gat_heads - 1) // gat_heads) * gat_heads
            mlp_hidden = min(int(best.get("mlp_hidden", 64)), UNIT_MAX_GAT)
            hp = {
                "gat_heads": gat_heads,
                "node_hidden": node_hidden,
                "mlp_hidden": mlp_hidden,
                "dropout_enc": best.get("dropout_enc", 0.2),
                "dropout_gat": best.get("dropout_gat", 0.3),
                "dropout_head": best.get("dropout_head", 0.3),
                "activation": best.get("activation", "gelu"),
                "norm": best.get("norm", "layer"),
                "init": best.get("init", "default"),
            }
            self._model = _build_gat_model(
                n_features, self._edge_index, self._edge_weight, n_nodes=25, hp=hp,
            )
            _train_graph_model(
                self._model, X_scaled, y_scaled,
                epochs=300, lr=best.get("lr", 5e-4),
                batch_size=best.get("batch_size", 32),
                patience=30,
                weight_decay=best.get("weight_decay", 1e-3),
                graph_reg_weight=0.0,
                augment=True, augment_factor=best.get("augment_factor", 2),
            )
        self._fitted = True
        log.info(f"[GE-GAT] 학습 완료 (features={n_features}, edges={self._edge_index.shape[1]}, trials={n_trials})")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if not self._fitted or self._model is None:
            log.warning("[GE-GAT] 학습되지 않은 모델로 예측 시도")
            return np.zeros(len(X_test))

        import torch

        X_scaled = self._scaler_X.transform(X_test)
        # device-fix: model trained on cuda — send input to same device.
        model_device = next(self._model.parameters()).device
        X_t = torch.FloatTensor(X_scaled).to(model_device)

        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(X_t).cpu().numpy()

        pred = self._scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        from simulation.models.safety import apply_extrapolation_cap  # G-289 (GCN parity)
        return apply_extrapolation_cap(np.maximum(pred, 0), getattr(self, "_y_train_max", None))


# ═══════════════════════════════════════════════════════════════
# 레지스트리 등록
# ═══════════════════════════════════════════════════════════════

# 2026-05-26 prune (user explicit): GE-DNN REMOVED — not in paper proper
# (sec_graph_models_pyg_failure.md "history of removal" only). Codex agreed
# REMOVE (graph training path uses hardcoded Huber loss). GAT covers modern
# graph-attention slot.
# 2026-05-28 사용자 명시 "GE-DNN 없는데 왜 코드 있어?" — orphan dead code 명시:
#   class GraphEnhancedDNN (L200) + GraphEnhancedDNNForecaster (L451+) 는 archived
#   (registry register X, learning path 무관). _train_graph_model docstring 의
#   "Graph-Enhanced DNN 학습 루프" 는 historical 명명 — 실제로는 GAT helper.
#   graph_models_variants.py (V1/V2/V3) = simulation/models/_archive/ 이동 (2026-05-28).
# REGISTRY.register(GraphEnhancedDNNForecaster)   # ARCHIVED — dead code (사용자 명시)
REGISTRY.register(GraphAttentionDNNForecaster)
