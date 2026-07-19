"""
simulation/tests/smoke_pyg_layers.py
====================================
torch_geometric 2.7.0 layer/model smoke test.

목적: 기존 graph_models.py 는 GE-DNN (manual GCN) + GE-DNN-GAT (GATv2Conv)
  2개만 사용. pyg 에 69 conv + 20 pre-built model 이 있으므로 다양한
  대안을 forward-pass 수준에서 빠르게 검증한다.

Seoul 25-gu 통근 그래프 shape 으로 smoke:
  - n_nodes = 25
  - in_features = 30
  - out_features = 16 (임의)
  - edge_index: fully-connected (625 edges) -- adj 는 실제로는 가중치 있지만
    smoke 에서는 structure 만 확인.

각 layer 에 대해:
  1. 인스턴스화 가능한가?
  2. forward(x, edge_index) 통과?
  3. out shape 이 (n_nodes, out_features) 인가?
  4. backward 가 동작하는가? (SGD 1-step)

직접 실행:
  .venv\\Scripts\\python.exe -m simulation.tests.smoke_pyg_layers
"""
from __future__ import annotations

import time
import traceback
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _build_seoul_graph(n_nodes: int = 25):
    """25×25 인접 행렬 → edge_index (2, E) + edge_weight (E,).

    smoke 목적이므로 실제 통근 가중치 대신 fully-connected (self-loop 제외).
    """
    adj = np.ones((n_nodes, n_nodes), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(adj[src, dst], dtype=torch.float32)
    return edge_index, edge_weight


def _smoke_conv(name, ctor, x, edge_index, edge_weight=None,
                needs_edge_weight=False, needs_edge_attr=False,
                out_features=16):
    """단일 conv layer forward + backward smoke."""
    import torch.nn as nn

    t0 = time.time()
    try:
        layer = ctor()

        # forward
        if needs_edge_weight:
            out = layer(x, edge_index, edge_weight)
        elif needs_edge_attr:
            # edge_attr: fake 1-dim features
            edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
            out = layer(x, edge_index, edge_attr)
        else:
            out = layer(x, edge_index)

        ok_shape = (out.shape[0] == x.shape[0])
        ok_out_dim = (out.shape[1] == out_features)

        # backward
        loss = out.sum()
        loss.backward()
        ok_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in layer.parameters() if p.requires_grad)

        total = time.time() - t0
        status = "OK" if (ok_shape and ok_out_dim and ok_grad) else "BAD"
        n_params = sum(p.numel() for p in layer.parameters())
        return {
            "name": name, "status": status,
            "time": round(total, 3), "params": n_params,
            "out_shape": tuple(out.shape),
            "ok_shape": ok_shape, "ok_out_dim": ok_out_dim,
            "ok_grad": ok_grad,
        }
    except Exception as e:
        total = time.time() - t0
        return {
            "name": name, "status": "FAIL",
            "time": round(total, 3),
            "error": f"{type(e).__name__}: {str(e)[:120]}",
        }


def main():
    from torch_geometric.nn import (
        GCNConv, GATConv, GATv2Conv, SAGEConv, ChebConv, TransformerConv,
        ARMAConv, SGConv, GINConv, GINEConv, ResGatedGraphConv,
        GeneralConv, PNAConv, TAGConv, SuperGATConv, FAConv, LEConv,
    )
    import torch.nn as nn

    n_nodes = 25
    in_dim = 30
    out_dim = 16

    x = torch.randn(n_nodes, in_dim, requires_grad=False)
    edge_index, edge_weight = _build_seoul_graph(n_nodes)

    print("=" * 74)
    print("torch_geometric 2.7.0 layer smoke (Seoul 25-gu dummy)")
    print(f"  x={tuple(x.shape)}, edge_index={tuple(edge_index.shape)}, "
          f"target out_dim={out_dim}")
    print("=" * 74)

    # ── 후보 레이어 목록 ──
    # 각 layer constructor / special forward needs 정리
    specs = [
        # (name, ctor, needs_edge_weight, needs_edge_attr, category)
        ("GCNConv",         lambda: GCNConv(in_dim, out_dim),          True,  False, "spectral"),
        ("ChebConv (K=3)",  lambda: ChebConv(in_dim, out_dim, K=3),    True,  False, "spectral"),
        ("SGConv (K=2)",    lambda: SGConv(in_dim, out_dim, K=2),      True,  False, "spectral"),
        ("ARMAConv",        lambda: ARMAConv(in_dim, out_dim),         True,  False, "spectral"),
        ("TAGConv",         lambda: TAGConv(in_dim, out_dim),          True,  False, "spectral"),
        ("FAConv",          lambda: FAConv(in_dim, eps=0.1),           False, False, "spectral"),
        ("LEConv",          lambda: LEConv(in_dim, out_dim),           False, False, "spectral"),

        ("GATConv (h=2)",   lambda: GATConv(in_dim, out_dim, heads=2, concat=False),
                                                                        False, False, "attention"),
        ("GATv2Conv (h=2)", lambda: GATv2Conv(in_dim, out_dim, heads=2, concat=False),
                                                                        False, False, "attention"),
        ("TransformerConv", lambda: TransformerConv(in_dim, out_dim, heads=2, concat=False),
                                                                        False, False, "attention"),
        ("SuperGATConv",    lambda: SuperGATConv(in_dim, out_dim, heads=2, concat=False),
                                                                        False, False, "attention"),

        ("SAGEConv",        lambda: SAGEConv(in_dim, out_dim),          False, False, "inductive"),
        ("ResGatedGraphConv", lambda: ResGatedGraphConv(in_dim, out_dim), False, False, "gated"),

        ("GINConv",         lambda: GINConv(nn.Sequential(
                                nn.Linear(in_dim, out_dim),
                                nn.ReLU(),
                                nn.Linear(out_dim, out_dim))),          False, False, "msg_passing"),
        ("GINEConv",        lambda: GINEConv(nn.Sequential(
                                nn.Linear(in_dim, out_dim),
                                nn.ReLU(),
                                nn.Linear(out_dim, out_dim)),
                                edge_dim=1),                            False, True,  "msg_passing"),

        ("GeneralConv",     lambda: GeneralConv(in_dim, out_dim),       False, False, "generic"),
    ]

    # PNAConv 는 degree-prior 가 필요 -- 별도로 처리
    # PNA: get degree histogram
    from torch_geometric.utils import degree
    deg = degree(edge_index[1], num_nodes=n_nodes, dtype=torch.long)
    max_degree = int(deg.max())
    deg_hist = torch.zeros(max_degree + 1, dtype=torch.long)
    for d in deg:
        deg_hist[d] += 1
    specs.append(("PNAConv", lambda: PNAConv(
        in_channels=in_dim, out_channels=out_dim,
        aggregators=["mean", "max", "sum"],
        scalers=["identity", "amplification"],
        deg=deg_hist,
    ), False, False, "aggregation"))

    # ── run ──
    results = []
    by_cat = {}
    for tup in specs:
        name, ctor, nw, na, cat = tup
        r = _smoke_conv(name, ctor, x, edge_index, edge_weight,
                        needs_edge_weight=nw, needs_edge_attr=na,
                        out_features=out_dim)
        r["category"] = cat
        results.append(r)
        by_cat.setdefault(cat, []).append(r)

        if r["status"] == "OK":
            print(f"  [OK ]  {r['name']:<20} ({cat:<12}) "
                  f"params={r['params']:>6}  time={r['time']:.3f}s  "
                  f"out={r['out_shape']}")
        else:
            print(f"  [FAIL] {r['name']:<20} ({cat:<12}) "
                  f"time={r['time']:.3f}s  {r.get('error','')}")

    # ── pre-built GNN models (torch_geometric.nn.models) ──
    print()
    print("-- pre-built GNN models (torch_geometric.nn.models) --")
    from torch_geometric.nn import GCN, GraphSAGE, GAT, GIN
    prebuilt = [
        ("GCN",        lambda: GCN(in_dim, out_dim, num_layers=2)),
        ("GraphSAGE",  lambda: GraphSAGE(in_dim, out_dim, num_layers=2)),
        ("GAT",        lambda: GAT(in_dim, out_dim, num_layers=2, v2=True, heads=2)),
        ("GIN",        lambda: GIN(in_dim, out_dim, num_layers=2)),
    ]
    for name, ctor in prebuilt:
        t0 = time.time()
        try:
            m = ctor()
            out = m(x, edge_index)
            loss = out.sum()
            loss.backward()
            total = time.time() - t0
            n_params = sum(p.numel() for p in m.parameters())
            print(f"  [OK ]  {name:<12} params={n_params:>6}  time={total:.3f}s  "
                  f"out={tuple(out.shape)}")
            results.append({"name": name, "category": "pre-built",
                            "status": "OK", "time": round(total, 3),
                            "params": n_params, "out_shape": tuple(out.shape)})
        except Exception as e:
            total = time.time() - t0
            print(f"  [FAIL] {name:<12} {type(e).__name__}: {str(e)[:100]}")
            results.append({"name": name, "category": "pre-built",
                            "status": "FAIL", "error": str(e)})

    # ── summary ──
    print()
    print("=" * 74)
    print("SUMMARY by category")
    print("=" * 74)
    for cat in sorted(by_cat.keys()) + ["pre-built"]:
        rows = [r for r in results if r.get("category") == cat]
        if not rows:
            continue
        n_ok = sum(1 for r in rows if r["status"] == "OK")
        print(f"  {cat:<14} {n_ok}/{len(rows)} OK")
        for r in rows:
            flag = "+" if r["status"] == "OK" else "x"
            print(f"     {flag} {r['name']}")

    n_ok_all = sum(1 for r in results if r["status"] == "OK")
    print(f"\nTOTAL: {n_ok_all}/{len(results)} usable")
    return results


if __name__ == "__main__":
    main()
