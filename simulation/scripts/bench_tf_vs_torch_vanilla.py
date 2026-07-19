"""Vanilla head-to-head: PyTorch vs TensorFlow on the actual Seoul ILI task.

Setup (identical for both frameworks):
  * Dataset: sentinel_ili weekly rate from epi_real_seoul.db (345 weeks)
  * Features: 10 lag features (lag_1 through lag_10) + week-of-year sin/cos
  * Target: next-week ILI rate
  * Model: MLP [input → 32 → 16 → 1] with ReLU + dropout 0.2
  * Loss: MSE, optimizer: Adam(1e-3), batch_size=16, epochs=200
  * Split: 70/15/15 train/val/test (temporal order preserved)

Measures:
  * Cold-start (first call) latency
  * Wall-clock fit time (all epochs)
  * Wall-clock inference time (batch of 52 test samples, avg over 100 runs)
  * Test R² and RMSE
  * Peak RSS memory delta (via psutil)
  * Import time (how long to `import torch` vs `import tensorflow`)

Output: stdout + simulation/results/bench_tf_vs_torch.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Seed RNG for reproducibility
SEED = 42
np.random.seed(SEED)

# Silence TF noise before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024**2


def load_ili_data():
    """Pull sentinel_ili from DB + build a simple lag-feature matrix."""
    from simulation.database.config import DB_PATH
    from simulation.models.feature_engine.loaders import _load_sentinel_ili
    df = _load_sentinel_ili(str(DB_PATH))
    # Aggregate to weekly city-wide mean (simple baseline for this test)
    import polars as pl
    weekly = (
        df.group_by("cal_date").agg(pl.col("ili_rate").mean())
        .sort("cal_date")
    )
    y = weekly["ili_rate"].to_numpy().astype(np.float32)
    # Build lag features
    L = 10
    n = len(y)
    X = np.zeros((n - L, L + 2), dtype=np.float32)
    for i in range(L, n):
        X[i - L, :L] = y[i - L : i]
        # seasonal features
        week = i % 52
        X[i - L, L] = np.sin(2 * np.pi * week / 52).astype(np.float32)
        X[i - L, L + 1] = np.cos(2 * np.pi * week / 52).astype(np.float32)
    y_trim = y[L:]
    return X, y_trim


def torch_bench(X_tr, y_tr, X_va, y_va, X_te, y_te, epochs=200, batch=16):
    """Train a vanilla MLP in PyTorch, time every step."""
    t_import = time.perf_counter()
    import torch
    import torch.nn as nn
    torch_import_s = time.perf_counter() - t_import
    torch.manual_seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  [torch] version={torch.__version__}, device={device}, import={torch_import_s:.2f}s")

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device).unsqueeze(1)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device).unsqueeze(1)
    X_te_t = torch.from_numpy(X_te).to(device)

    mem0 = _rss_mb()
    class MLP(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.fc1 = nn.Linear(d_in, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.drop = nn.Dropout(0.2)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.drop(x)
            x = torch.relu(self.fc2(x))
            x = self.drop(x)
            return self.fc3(x)

    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Cold-start = first forward pass
    t_cold = time.perf_counter()
    with torch.no_grad():
        _ = model(X_tr_t[:batch])
    cold_s = time.perf_counter() - t_cold

    # Train
    t_fit = time.perf_counter()
    model.train()
    n = X_tr_t.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            j = idx[i : i + batch]
            opt.zero_grad()
            pred = model(X_tr_t[j])
            loss = loss_fn(pred, y_tr_t[j])
            loss.backward()
            opt.step()
    # Sync MPS before timing stop
    if device == "mps":
        torch.mps.synchronize()
    fit_s = time.perf_counter() - t_fit

    # Inference timing (avg of 100 calls on test set)
    model.eval()
    t_inf = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            out = model(X_te_t).cpu().numpy().ravel()
    if device == "mps":
        torch.mps.synchronize()
    inf_s = (time.perf_counter() - t_inf) / 100

    # Metrics
    pred_te = out  # last pass
    rmse = float(np.sqrt(np.mean((pred_te - y_te) ** 2)))
    ss_res = float(np.sum((y_te - pred_te) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mem_peak = _rss_mb() - mem0

    return {
        "framework": "PyTorch",
        "version": torch.__version__,
        "device": device,
        "import_s": round(torch_import_s, 3),
        "cold_start_ms": round(cold_s * 1000, 3),
        "fit_s": round(fit_s, 3),
        "inference_ms_per_call": round(inf_s * 1000, 4),
        "test_rmse": round(rmse, 4),
        "test_r2": round(r2, 4),
        "peak_mem_mb": round(mem_peak, 1),
    }


def tf_bench(X_tr, y_tr, X_va, y_va, X_te, y_te, epochs=200, batch=16):
    """Train the same vanilla MLP in TensorFlow/Keras."""
    t_import = time.perf_counter()
    import tensorflow as tf
    tf_import_s = time.perf_counter() - t_import
    tf.random.set_seed(SEED)

    # TF auto-detects Metal if tensorflow-metal is installed
    gpus = tf.config.list_physical_devices("GPU")
    device = "GPU (Metal)" if gpus else "CPU"
    print(f"  [tf]    version={tf.__version__}, device={device}, import={tf_import_s:.2f}s")

    mem0 = _rss_mb()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_tr.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    # Cold start
    t_cold = time.perf_counter()
    _ = model(X_tr[:batch], training=False)
    cold_s = time.perf_counter() - t_cold

    # Fit
    t_fit = time.perf_counter()
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch, verbose=0, shuffle=True)
    fit_s = time.perf_counter() - t_fit

    # Inference (avg of 100 calls)
    t_inf = time.perf_counter()
    for _ in range(100):
        pred = model.predict(X_te, verbose=0).ravel()
    inf_s = (time.perf_counter() - t_inf) / 100

    # Metrics
    rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mem_peak = _rss_mb() - mem0

    return {
        "framework": "TensorFlow",
        "version": tf.__version__,
        "device": device,
        "import_s": round(tf_import_s, 3),
        "cold_start_ms": round(cold_s * 1000, 3),
        "fit_s": round(fit_s, 3),
        "inference_ms_per_call": round(inf_s * 1000, 4),
        "test_rmse": round(rmse, 4),
        "test_r2": round(r2, 4),
        "peak_mem_mb": round(mem_peak, 1),
    }


def main():
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Vanilla MLP head-to-head: PyTorch vs TensorFlow              ║")
    print("║  Same data, same arch, same seed, same optimizer/epochs       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    X, y = load_ili_data()
    n = len(y)
    i_tr = int(n * 0.70)
    i_va = int(n * 0.85)
    X_tr, X_va, X_te = X[:i_tr], X[i_tr:i_va], X[i_va:]
    y_tr, y_va, y_te = y[:i_tr], y[i_tr:i_va], y[i_va:]
    print(f"  Data: train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}  features={X.shape[1]}")
    print()

    gc.collect()
    print("─── PyTorch ────────────────────────────────────────────────────")
    torch_res = torch_bench(X_tr, y_tr, X_va, y_va, X_te, y_te)
    print(f"    fit={torch_res['fit_s']}s  inf={torch_res['inference_ms_per_call']}ms  "
          f"R²={torch_res['test_r2']}  RMSE={torch_res['test_rmse']}  "
          f"mem+={torch_res['peak_mem_mb']}MB")
    print()

    gc.collect()
    print("─── TensorFlow ─────────────────────────────────────────────────")
    tf_res = tf_bench(X_tr, y_tr, X_va, y_va, X_te, y_te)
    print(f"    fit={tf_res['fit_s']}s  inf={tf_res['inference_ms_per_call']}ms  "
          f"R²={tf_res['test_r2']}  RMSE={tf_res['test_rmse']}  "
          f"mem+={tf_res['peak_mem_mb']}MB")
    print()

    # Summary
    print("━" * 65)
    print(f"{'metric':<26} {'PyTorch':>15} {'TF':>15}  {'ratio':>6}")
    print("━" * 65)
    for k, unit in [
        ("import_s", "s"),
        ("cold_start_ms", "ms"),
        ("fit_s", "s"),
        ("inference_ms_per_call", "ms"),
        ("test_r2", ""),
        ("test_rmse", ""),
        ("peak_mem_mb", "MB"),
    ]:
        a = torch_res[k]; b = tf_res[k]
        try:
            ratio = a / b if b else float("inf")
            ratio_s = f"{ratio:.2f}"
        except Exception:
            ratio_s = "—"
        print(f"{k:<26} {a:>12}{unit:>3} {b:>12}{unit:>3}  {ratio_s:>6}")
    print()

    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    _bench_out = get_results_dir() / "bench_tf_vs_torch.json"
    _bench_out.write_text(
        json.dumps({"torch": torch_res, "tf": tf_res}, indent=2)
    )
    print(f"Wrote: {_bench_out}")


if __name__ == "__main__":
    main()
