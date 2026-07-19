"""
simulation/models/graph_models_pyg.py
=====================================
torch_geometric 2.7.0 기반 Graph-Enhanced DNN 바리에이션 (Tier 1 + Tier 2).

[배경]
 기존 graph_models.py 는 manual GCN (GE-DNN) + GATv2Conv (GE-GAT) 2개만 사용.
 pyg 2.7.0 에 69 conv + 20 pre-built model 이 있음을 smoke_pyg_layers.py 로 검증
 (20/21 OK). 그 중 시계열-동형-비동형 특성이 서로 다른 8 가지를 추가 등록해
 forecaster Tier 1+2 를 확장한다.

[편입된 모델]
 Tier 1 (핵심 대안, PAPER_PRIMARY_11 후보):
 - GE-Cheb (L15) : ChebConv K=3, 다항식 스펙트럴 필터
 - GE-SAGE (L15) : SAGEConv, inductive mean-aggregator
 - GE-Transformer (L16) : TransformerConv, multi-head attention (linear proj)
 - GE-PNA (L17) : PNAConv, degree-aware multi-aggregator

 Tier 2 (부가 대안, bench 용):
 - GE-GCN (L15) : GCNConv, pyg 표준 (manual GE-DNN 과 A/B 비교)
 - GE-GIN (L15) : GINConv, Weisfeiler-Lehman 동형성 최대
 - GE-ARMA (L16) : ARMAConv, 다항식 유리함수 필터
 - GE-ResGated (L16) : ResGatedGraphConv, residual+gating

[공통 구조]
 (batch, n_features) → node_encoder → (batch, 25, hidden)
 → Conv1 (+norm+act+drop+residual) → Conv2 (+ 동일)
 → mean+max readout → MLP head → (batch)

 Batch 처리: pyg conv 는 flat (N, F) 만 받으므로 batched edge_index 를
 (2, B*E) 로 오프셋 복제하여 B*N 개 노드로 동시 처리.

 그래프 구조: _get_adjacency_matrix 에서 25×25 통근 행렬 → edge_index + edge_weight.
 (graph_models.py 의 기존 함수 재사용)

변경 이력:
 - (2026-04-19): Tier 1+2 8개 forecaster 초기 등록 (+pyg 의존성 선택적 처리)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY
from simulation.models.graph_models import (
    _get_adjacency_matrix,
    _adj_to_edge_index,
    _train_graph_model,
)

log = logging.getLogger(__name__)

# ─── torch_geometric 가용성 ────────────────────────────────────────
try:
    import torch_geometric  # noqa: F401
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    log.warning("[graph_models_pyg] torch_geometric 미설치 → 등록 스킵")


# ═══════════════════════════════════════════════════════════════
# Batched edge_index 헬퍼
# ═══════════════════════════════════════════════════════════════

def _batched_edge_index(edge_index, edge_weight, batch_size: int, n_nodes: int):
    """단일 edge_index (2, E) 를 batch_size 만큼 오프셋 복제.

    pyg conv 는 flat (B*N, F) 입력에 대해 (2, B*E) edge_index 를 요구.
    각 배치의 노드 id 에 i*N 오프셋을 더한다.
    """
    import torch
    E = edge_index.size(1)
    offsets = (torch.arange(batch_size, device=edge_index.device) * n_nodes).view(batch_size, 1, 1)
    ei = edge_index.unsqueeze(0).expand(batch_size, -1, -1) + offsets  # (B, 2, E)
    ei_b = ei.permute(1, 0, 2).reshape(2, batch_size * E)
    ew_b = None
    if edge_weight is not None:
        ew_b = edge_weight.repeat(batch_size)
    return ei_b, ew_b


# ═══════════════════════════════════════════════════════════════
# Conv wrapper (batched-flat 변환)
# ═══════════════════════════════════════════════════════════════

def _make_conv_block(conv_cls_name: str, in_dim: int, out_dim: int, **kwargs):
    """conv_cls_name → instantiated pyg conv layer."""
    from torch_geometric.nn import (
        GCNConv, ChebConv, SAGEConv, TransformerConv, PNAConv,
        GINConv, ARMAConv, ResGatedGraphConv,
    )
    import torch.nn as nn

    if conv_cls_name == "GCNConv":
        return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True), True, False
    if conv_cls_name == "ChebConv":
        return ChebConv(in_dim, out_dim, K=kwargs.get("K", 3)), True, False
    if conv_cls_name == "ARMAConv":
        return ARMAConv(in_dim, out_dim), True, False
    if conv_cls_name == "SAGEConv":
        return SAGEConv(in_dim, out_dim), False, False
    if conv_cls_name == "TransformerConv":
        heads = kwargs.get("heads", 2)
        return TransformerConv(in_dim, out_dim, heads=heads, concat=False), False, False
    if conv_cls_name == "PNAConv":
        deg = kwargs["deg"]
        return PNAConv(
            in_channels=in_dim, out_channels=out_dim,
            aggregators=["mean", "max", "sum"],
            scalers=["identity", "amplification"],
            deg=deg,
        ), False, False
    if conv_cls_name == "GINConv":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        return GINConv(mlp), False, False
    if conv_cls_name == "ResGatedGraphConv":
        return ResGatedGraphConv(in_dim, out_dim), False, False
    raise ValueError(f"unknown conv: {conv_cls_name}")


# ═══════════════════════════════════════════════════════════════
# GE-* 공통 모델 팩토리
# ═══════════════════════════════════════════════════════════════

def _build_pyg_model(n_features: int, edge_index, edge_weight, conv_cls_name: str,
                     n_nodes: int = 25, hp: Optional[dict] = None,
                     conv_kwargs: Optional[dict] = None):
    """공통 GE-Pyg DNN 빌드.

 (P0-D): node_hidden / mlp_hidden / dropout / init 도 hp 로 수신.
 """
    import torch
    import torch.nn as nn
    from simulation.models.dl_models import (
        _get_activation_fn, _get_norm_layer, _apply_weight_init,
    )

    hp = hp or {}
    conv_kwargs = conv_kwargs or {}
    act_name = hp.get("activation", "gelu")
    norm_name = hp.get("norm", "layer")
    init_name = hp.get("init", "default")
    node_hidden = int(hp.get("node_hidden", 32))
    mlp_hidden = int(hp.get("mlp_hidden", 64))
    drop_enc = float(hp.get("dropout_enc", 0.2))
    drop_conv = float(hp.get("dropout_conv", 0.3))
    drop_head = float(hp.get("dropout_head", 0.3))

    class _Norm3D(nn.Module):
        """(b,N,d) node-feature 텐서 norm. BatchNorm1d 는 (b·N,d) 로 펴서 적용해 feature-dim(d)
        정규화 — 3D 직접 적용 시 BatchNorm1d 가 dim 1(N_nodes)을 정규화하는 버그 회피
        ("running_mean should contain {d} not {N}", G-FIX 2026-06-03). LayerNorm/Identity 는 3D-safe."""
        def __init__(self, name, dim):
            super().__init__()
            self.norm = _get_norm_layer(name, dim)
            self._bn = (name == "batch")

        def forward(self, x):
            if self._bn and x.dim() == 3:
                b, n, d = x.shape
                return self.norm(x.reshape(b * n, d)).reshape(b, n, d)
            return self.norm(x)

    class PygGraphEnhancedDNN(nn.Module):
        def __init__(self, n_feat, ei, ew):
            super().__init__()
            self.n_nodes = n_nodes
            self.register_buffer("edge_index", ei)
            if ew is not None:
                self.register_buffer("edge_weight", ew)
            else:
                self.edge_weight = None

            gcn_hidden = node_hidden  # keep residual dim aligned

            self.node_encoder = nn.Sequential(
                nn.Linear(n_feat, node_hidden),
                _Norm3D(norm_name, node_hidden),
                _get_activation_fn(act_name),
                nn.Dropout(drop_enc),
            )

            c1, nw1, _ = _make_conv_block(conv_cls_name, node_hidden, gcn_hidden, **conv_kwargs)
            c2, nw2, _ = _make_conv_block(conv_cls_name, gcn_hidden, gcn_hidden, **conv_kwargs)
            self.conv1 = c1
            self.conv2 = c2
            self.needs_edge_weight = nw1  # 같은 conv → 같은 signature

            self.norm1 = _Norm3D(norm_name, gcn_hidden)
            self.norm2 = _Norm3D(norm_name, gcn_hidden)
            self.act = _get_activation_fn(act_name)
            self.drop = nn.Dropout(drop_conv)

            self.head = nn.Sequential(
                nn.Linear(gcn_hidden * 2, mlp_hidden),
                _get_activation_fn(act_name),
                nn.Dropout(drop_head),
                nn.Linear(mlp_hidden, max(8, mlp_hidden // 2)),
                _get_activation_fn(act_name),
                nn.Linear(max(8, mlp_hidden // 2), 1),
            )

        def _run_conv(self, conv, x_flat, ei_b, ew_b):
            if self.needs_edge_weight:
                return conv(x_flat, ei_b, ew_b)
            return conv(x_flat, ei_b)

        def forward(self, x):
            # x: (batch, n_features)
            b = x.size(0)
            x_nodes = x.unsqueeze(1).expand(-1, self.n_nodes, -1)  # (b, N, f)
            h = self.node_encoder(x_nodes)  # (b, N, d)

            ei_b, ew_b = _batched_edge_index(
                self.edge_index,
                self.edge_weight if hasattr(self, "edge_weight") else None,
                b, self.n_nodes,
            )

            h_flat = h.reshape(b * self.n_nodes, -1)
            h1 = self._run_conv(self.conv1, h_flat, ei_b, ew_b)
            h1 = h1.reshape(b, self.n_nodes, -1)
            h1 = self.drop(self.act(self.norm1(h1))) + h  # residual

            h1_flat = h1.reshape(b * self.n_nodes, -1)
            h2 = self._run_conv(self.conv2, h1_flat, ei_b, ew_b)
            h2 = h2.reshape(b, self.n_nodes, -1)
            h2 = self.drop(self.act(self.norm2(h2))) + h1  # residual

            h_mean = h2.mean(dim=1)
            h_max = h2.max(dim=1).values
            h_pool = torch.cat([h_mean, h_max], dim=-1)

            return self.head(h_pool).squeeze(-1)

        def graph_laplacian_loss(self, x):
            """_train_graph_model 이 호출하는 smoothness prior (0 으로 반환: conv 가 이미 spatial)."""
            import torch
            return torch.tensor(0.0, device=x.device)

    m = PygGraphEnhancedDNN(n_features, edge_index, edge_weight)
    _apply_weight_init(m, init_name)
    return m


# ═══════════════════════════════════════════════════════════════
# BaseForecaster 베이스 클래스
# ═══════════════════════════════════════════════════════════════

class _PygGraphForecasterBase(BaseForecaster):
    """공통: 피처 스케일링 + adj 로드 + edge_index 생성 + _train_graph_model 재사용."""

    CONV_CLS_NAME: str = ""            # 서브클래스 지정
    CONV_KWARGS: dict = {}             # 서브클래스 지정 (e.g. K=3, heads=2)
    NEEDS_RAW_ADJ: bool = False        # True 면 threshold 전 raw edges 사용
    USE_EDGE_WEIGHT: bool = True       # False 면 edge_weight 전달 안 함

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._edge_index = None
        self._edge_weight = None

    def _resolve_conv_kwargs(self, adj_raw: np.ndarray):
        """PNA 처럼 그래프 통계가 필요한 conv 를 위한 훅."""
        return dict(self.CONV_KWARGS)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """(P0-D): Optuna HPO 공통 내장 (모든 GE-* subclass 에 자동 적용)."""
        if not _PYG_AVAILABLE:
            log.error(f"[{self.meta.name}] torch_geometric 미설치 → 학습 스킵")
            return self

        try:
            import torch
        except ImportError:
            log.error(f"[{self.meta.name}] PyTorch 미설치")
            return self

        from sklearn.preprocessing import StandardScaler
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop,
            UNIT_MIN_DEFAULT, UNIT_MAX_DEFAULT,
            ACTIVATIONS, NORMS, INITS,
        )

        self._scaler_X = StandardScaler()
        X_scaled = self._scaler_X.fit_transform(X_train)
        self._scaler_y = StandardScaler()
        y_scaled = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap

        adj_raw = _get_adjacency_matrix()
        self._edge_index, self._edge_weight = _adj_to_edge_index(adj_raw, threshold=1e-6)

        # conv 별 kwargs 해소 (PNA deg 등)
        conv_kw = self._resolve_conv_kwargs(adj_raw)

        n_features = X_scaled.shape[1]
        ew = self._edge_weight if self.USE_EDGE_WEIGHT else None

        n_trials = get_trials(self.meta.name, default=0)

        def _static_defaults():
            self._model = _build_pyg_model(
                n_features, self._edge_index, ew,
                conv_cls_name=self.CONV_CLS_NAME, n_nodes=25,
                conv_kwargs=conv_kw,
            )
            _train_graph_model(
                self._model, X_scaled, y_scaled,
                epochs=200, lr=5e-4, batch_size=32,
                patience=25, weight_decay=1e-3,
                graph_reg_weight=0.0,
            )

        def _objective(trial):
            import gc
            hp = {
                "node_hidden": trial.suggest_int("node_hidden", UNIT_MIN_DEFAULT, UNIT_MAX_DEFAULT, log=True),
                "mlp_hidden": trial.suggest_int("mlp_hidden", UNIT_MIN_DEFAULT, UNIT_MAX_DEFAULT, log=True),
                "dropout_enc": trial.suggest_float("dropout_enc", 0.0, 0.5),
                "dropout_conv": trial.suggest_float("dropout_conv", 0.0, 0.5),
                "dropout_head": trial.suggest_float("dropout_head", 0.0, 0.5),
                "activation": trial.suggest_categorical("activation", ACTIVATIONS),
                "norm": trial.suggest_categorical("norm", NORMS),
                "init": trial.suggest_categorical("init", INITS),
            }
            tr = suggest_training_hp(trial)
            try:
                model = _build_pyg_model(
                    n_features, self._edge_index, ew,
                    conv_cls_name=self.CONV_CLS_NAME, n_nodes=25,
                    hp=hp, conv_kwargs=conv_kw,
                )
            except Exception as e:
                # conv_kwargs constraint (e.g. PNA deg) 로 특정 hp 불가 → prune
                import optuna
                raise optuna.TrialPruned() from e
            best_val = _train_graph_model(
                model, X_scaled, y_scaled,
                epochs=120, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=20, weight_decay=tr["weight_decay"],
                graph_reg_weight=0.0,
                augment=True, augment_factor=tr["augment_factor"],
            )
            del model
            gc.collect()
            return float(best_val)

        best, _ = run_optuna_loop(self.meta.name, _objective, n_trials, _static_defaults)
        if best:
            hp = {
                "node_hidden": best.get("node_hidden", 32),
                "mlp_hidden": best.get("mlp_hidden", 64),
                "dropout_enc": best.get("dropout_enc", 0.2),
                "dropout_conv": best.get("dropout_conv", 0.3),
                "dropout_head": best.get("dropout_head", 0.3),
                "activation": best.get("activation", "gelu"),
                "norm": best.get("norm", "layer"),
                "init": best.get("init", "default"),
            }
            self._model = _build_pyg_model(
                n_features, self._edge_index, ew,
                conv_cls_name=self.CONV_CLS_NAME, n_nodes=25,
                hp=hp, conv_kwargs=conv_kw,
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
        log.info(
            f"[{self.meta.name}] 학습 완료 "
            f"(features={n_features}, edges={self._edge_index.shape[1]}, trials={n_trials})"
        )
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if not self._fitted or self._model is None:
            log.warning(f"[{self.meta.name}] 학습되지 않은 모델로 예측 시도")
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
        from simulation.models.safety import apply_extrapolation_cap  # G-289
        return apply_extrapolation_cap(np.maximum(pred, 0), getattr(self, "_y_train_max", None))


# ═══════════════════════════════════════════════════════════════
# Tier 1 — 핵심 대안
# ═══════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# ⚠ ARCHIVED (2026-05-28) — 사용자 명시 "정리해" 잔여 dead code:
# 아래 GE-Cheb / GE-SAGE / GE-Transformer / GE-PNA / GE-GIN / GE-ARMA /
# GE-ResGated 7 class 정의 보존되어 있으나 _PYG_FORECASTERS (L555) 에 미등록
# (= REGISTRY.register X). CATEGORY_MODELS["graph"] = ["GAT", "GCN"] 만 active.
# 활성화: _PYG_FORECASTERS list 에 class 추가 + CATEGORY_MODELS 변경.
# Tier 1 (옛 분류, 사용자 2026-05-12 명시 "GE-GCN 만 등록"):
# ════════════════════════════════════════════════════════════════════════════

class GEChebForecaster(_PygGraphForecasterBase):
    """GE-Cheb: ChebConv K=3 다항식 스펙트럴 필터. ⚠ ARCHIVED (register X, 2026-05-28).

    Defferrard et al. (2016) NeurIPS.
    고정 스펙트럴 기저 위에서 학습 가능한 다항식 계수.
    """

    meta = ModelMeta(
        name="GE-Cheb", category="dl", level=15, min_data=80,
        description="Graph-Enhanced DNN with ChebConv (K=3) spectral filter.",
        requires_gpu=False, dependencies=["torch", "torch_geometric"],
    )
    CONV_CLS_NAME = "ChebConv"
    CONV_KWARGS = {"K": 3}
    USE_EDGE_WEIGHT = True


class GESAGEForecaster(_PygGraphForecasterBase):
    """GE-SAGE: SAGEConv inductive mean-aggregator.

    Hamilton et al. (2017) NeurIPS.
    이웃 샘플링을 지원하는 inductive GNN (전이 학습 시 유리).
    """

    meta = ModelMeta(
        name="GE-SAGE", category="dl", level=15, min_data=80,
        description="Graph-Enhanced DNN with GraphSAGE (inductive mean aggregator).",
        requires_gpu=False, dependencies=["torch", "torch_geometric"],
    )
    CONV_CLS_NAME = "SAGEConv"
    USE_EDGE_WEIGHT = False


class GETransformerForecaster(_PygGraphForecasterBase):
    """GE-Transformer: TransformerConv multi-head attention on graph.

    Shi et al. (2021) IJCAI. GATv2 와 달리 projection matrix Q/K/V 를 분리.
    GATv2 (GE-GAT) 의 대안 비교군.
    """

    meta = ModelMeta(
        name="GE-Transformer", category="dl", level=16, min_data=80,
        description="Graph-Enhanced DNN with TransformerConv (Q/K/V attention).",
        requires_gpu=False, dependencies=["torch", "torch_geometric"],
    )
    CONV_CLS_NAME = "TransformerConv"
    CONV_KWARGS = {"heads": 2}
    USE_EDGE_WEIGHT = False


class GEPNAForecaster(_PygGraphForecasterBase):
    """GE-PNA: Principal Neighbourhood Aggregation, degree-aware multi-aggregator.

    Corso et al. (2020) NeurIPS. [mean, max, sum] × [identity, amplification].
    그래프 degree 히스토그램에 민감 → fit 시점에 계산해서 주입.
    """

    meta = ModelMeta(
        name="GE-PNA", category="dl", level=17, min_data=80,
        description="Graph-Enhanced DNN with PNAConv (degree-aware multi-aggregator).",
        requires_gpu=False, dependencies=["torch", "torch_geometric"],
    )
    CONV_CLS_NAME = "PNAConv"
    USE_EDGE_WEIGHT = False

    def _resolve_conv_kwargs(self, adj_raw: np.ndarray):
        """PNA 는 degree 히스토그램 prior 가 필수."""
        import torch
        from torch_geometric.utils import degree

        ei = self._edge_index  # (2, E)
        deg = degree(ei[1], num_nodes=25, dtype=torch.long)
        max_d = int(deg.max())
        hist = torch.zeros(max_d + 1, dtype=torch.long)
        for d in deg:
            hist[d] += 1
        return {"deg": hist}


# ═══════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# Tier 2 — ✅ ACTIVE (GEGCNpygForecaster only). 위 7 class 는 archived.
# 아래 GE-GIN / GE-ARMA / GE-ResGated 도 ARCHIVED (register X) — class 만 보존.
# ════════════════════════════════════════════════════════════════════════════

class GEGCNpygForecaster(_PygGraphForecasterBase):
    """GE-GCN: pyg 표준 GCNConv (기존 manual GE-DNN 과 A/B 비교). ✅ ACTIVE (CATEGORY_MODELS graph).

    Kipf & Welling (2017) ICLR. pyg 구현이 sparse 및 gcn_norm 을 제공해
    수치 안정성 측면에서 다른 결과를 낼 수 있다.
    """

    meta = ModelMeta(
        name="GCN", category="dl", level=15, min_data=80,
        description="Graph Convolutional Network (GCNConv, Kipf-Welling 2017) on Seoul commuting network.",
        requires_gpu=False, dependencies=["torch", "torch_geometric"],
    )
    CONV_CLS_NAME = "GCNConv"
    USE_EDGE_WEIGHT = True


class GEGINForecaster(_PygGraphForecasterBase):
    """GE-GIN: Graph Isomorphism Network, Weisfeiler-Lehman 동형성 최대.

    Xu et al. (2019) ICLR. 단순 sum aggregator + MLP.
    구 25개의 구조 구분력이 가장 필요한 경우.
    """

    meta = ModelMeta(
        name="GE-GIN", category="dl", level=15, min_data=80,
        description="Graph-Enhanced DNN with GIN (Weisfeiler-Lehman expressive).",
        requires_gpu=False, dependencies=["torch", "torch_geometric"],
    )
    CONV_CLS_NAME = "GINConv"
    USE_EDGE_WEIGHT = False


class GEARMAForecaster(_PygGraphForecasterBase):
    """GE-ARMA: ARMA filter, 다항식 유리함수 스펙트럴 필터.

    Bianchi et al. (2021) TPAMI. Cheb 보다 광대역 주파수 응답.
    """

    meta = ModelMeta(
        name="GE-ARMA", category="dl", level=16, min_data=80,
        description="Graph-Enhanced DNN with ARMAConv (rational spectral filter).",
        requires_gpu=False, dependencies=["torch", "torch_geometric"],
    )
    CONV_CLS_NAME = "ARMAConv"
    USE_EDGE_WEIGHT = True


class GEResGatedForecaster(_PygGraphForecasterBase):
    """GE-ResGated: Residual Gated Graph Conv.

    Bresson & Laurent (2017). Edge gating + residual connection.
    """

    meta = ModelMeta(
        name="GE-ResGated", category="dl", level=16, min_data=80,
        description="Graph-Enhanced DNN with Residual Gated Graph Conv.",
        requires_gpu=False, dependencies=["torch", "torch_geometric"],
    )
    CONV_CLS_NAME = "ResGatedGraphConv"
    USE_EDGE_WEIGHT = False


# ═══════════════════════════════════════════════════════════════
# 레지스트리 등록 (pyg 가용 시에만)
# ═══════════════════════════════════════════════════════════════

_PYG_FORECASTERS = [
    # 2026-05-12 (사용자 명시): GE-GCN 만 등록.
    # GE-Cheb / GE-SAGE / GE-Transformer / GE-PNA / GE-GIN / GE-ARMA /
    # GE-ResGated 는 미등록 (class 정의는 보존 — 추후 enable 가능).
    GEGCNpygForecaster,
]

if _PYG_AVAILABLE:
    try:
        for cls in _PYG_FORECASTERS:
            REGISTRY.register(cls)
    except Exception as e:
        log.warning(f"[graph_models_pyg] REGISTRY 등록 실패: {e}")
else:
    log.warning(
        f"[graph_models_pyg] torch_geometric 미설치 → "
        f"{len(_PYG_FORECASTERS)}개 forecaster 스킵"
    )


__all__ = [
    "GEChebForecaster",
    "GESAGEForecaster",
    "GETransformerForecaster",
    "GEPNAForecaster",
    "GEGCNpygForecaster",
    "GEGINForecaster",
    "GEARMAForecaster",
    "GEResGatedForecaster",
]
