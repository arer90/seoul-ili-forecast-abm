"""Standalone Chronos-Bolt forecast — runs in the ISOLATED .venv_chronos (transformers 4.x).

WHY standalone: chronos-forecasting requires transformers<5, but the main env requires
transformers>=5 (mlx-lm / ARIA LLM router). They cannot coexist (uv: No solution). So Chronos
runs SEPARATELY here — no `simulation` package import (which would pull mlx-lm) — reading the
feature cache directly and writing a predictions CSV that the main pipeline can merge into the
ensemble / eval afterward. (User 2026-06-13: "별개로 접근하면 되지.")

Run:  .venv_chronos/bin/python scripts/chronos_standalone.py [--model amazon/chronos-bolt-small] [--n-test 68]
Out:  simulation/results/csv/predictions_Chronos-2.csv  (split,idx,y_true,y_pred,pi10,pi90)
"""
from __future__ import annotations
import argparse, os
import numpy as np
import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--n-test", type=int, default=68)
    ap.add_argument("--cache", default="simulation/cache/feature_cache.parquet")
    ap.add_argument("--out", default="simulation/results/csv/predictions_Chronos-2.csv")
    ap.add_argument("--name", default="Chronos-2")
    args = ap.parse_args()

    # ── data: ILI series in time order (no simulation package → no mlx-lm) ──
    df = pl.read_parquet(args.cache).sort("week_start")        # polars-native (no pandas in venv)
    y = df["ili_rate"].to_numpy().astype(np.float32)
    n = len(y)
    test_idx = list(range(n - args.n_test, n))

    import torch
    from chronos import ChronosBoltPipeline
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = ChronosBoltPipeline.from_pretrained(args.model, device_map=dev, torch_dtype=torch.float32)
    print(f"[chronos] {args.model} on {dev} | series n={n}, test={args.n_test} (rolling 1-step)")

    # ── rolling-origin 1-step: context = history up to t, predict t (operational, leakage-free) ──
    QIDX_MED, QIDX_LO, QIDX_HI = 4, 0, 8   # bolt 9 quantiles: [0.1..0.9] → median=0.5, lo=0.1, hi=0.9
    preds, lo, hi = [], [], []
    for t in test_idx:
        ctx = torch.tensor(y[:t], dtype=torch.float32)        # all history before t (no leakage)
        q = pipe.predict(ctx, prediction_length=1)            # (1, 9, 1)
        q = np.asarray(q[0, :, 0], dtype=float)
        preds.append(float(q[QIDX_MED])); lo.append(float(q[QIDX_LO])); hi.append(float(q[QIDX_HI]))
    preds, lo, hi = np.array(preds), np.array(lo), np.array(hi)
    yt = y[test_idx]

    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - float(np.sum((yt - preds) ** 2)) / ss if ss > 0 else float("nan")
    mae = float(np.mean(np.abs(yt - preds)))
    cov = float(np.mean((yt >= lo) & (yt <= hi)))
    print(f"[chronos] test r2={r2:+.3f}  mae={mae:.2f}  pi80_coverage={cov:.2f}  pred범위[{preds.min():.0f},{preds.max():.0f}] (실제 max={yt.max():.0f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = pl.DataFrame({
        "split": ["test"] * len(test_idx),
        "idx": test_idx,
        "y_true": yt.tolist(),
        "y_pred": preds.tolist(),
        "pi10": lo.tolist(),
        "pi90": hi.tolist(),
    })
    out.write_csv(args.out)
    print(f"[chronos] → {args.out} ({len(test_idx)} rows). 메인 env에서 ensemble/eval에 merge 가능.")


if __name__ == "__main__":
    main()
