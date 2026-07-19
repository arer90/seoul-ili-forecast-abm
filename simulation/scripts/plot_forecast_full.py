"""모델별 4-way split (train/val/test/real) 시계열 forecast 시각화.

사용자 요청 (2026-05-02):
  - train/val/test/real 4 split 모두 시계열 그래프
  - test/real 신뢰구간 (PI) — 시간 흐름에 따른 불확실성 증가 시각화
  - 모델별 별도 PNG + 종합 grid

사용:
    .venv/bin/python -m simulation.scripts.plot_forecast_full
    .venv/bin/python -m simulation.scripts.plot_forecast_full --models DNN,XGBoost
    .venv/bin/python -m simulation.scripts.plot_forecast_full --top 5  # top 5 만

Output:
    simulation/results/plots_forecast_full/
        forecast_DNN.png
        forecast_XGBoost.png
        ...
        _summary_grid.png  # 모든 모델 한 페이지
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from simulation.config_global import Z95  # SSOT (2026-05-28)
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def load_split_data():
    """HWP §3 4-way split 의 y 값 + 시간축.

    Phase H.4 (사용자 2026-05-06): cache parquet 의 week_start 로
    year-month xticks 생성용 dates array 추가.
    """
    from simulation.database import safe_connect
    conn = safe_connect()
    rows = conn.execute("""
        SELECT season_start, week_seq, AVG(ili_rate) AS ili
        FROM sentinel_influenza
        WHERE ili_rate IS NOT NULL
        GROUP BY season_start, week_seq
        ORDER BY season_start, week_seq
    """).fetchall()
    conn.close()

    y_all = np.array([r[2] for r in rows], dtype=float)
    n = len(y_all)

    # HWP §3: train 242 / val 27 / test 68 / real 8
    n_train = 242
    n_val = 27
    n_test = 68
    n_real = 8
    if n < n_train + n_val + n_test + n_real:
        # fallback split (현재 데이터)
        n_test = max(8, int(n * 0.20))
        n_real = 8
        n_val = max(8, int((n - n_test - n_real) * 0.10))
        n_train = n - n_val - n_test - n_real

    # Phase H.4: dates from cache parquet (week_start) — for year-month xticks
    dates = None
    cache_path = ROOT / "simulation/cache/feature_cache.parquet"
    if cache_path.exists():
        try:
            import polars as pl  # noqa: local import
            df = pl.read_parquet(cache_path)
            if "week_start" in df.columns:
                dates = df["week_start"].to_numpy()
        except Exception:
            pass

    return {
        "train": y_all[:n_train],
        "val": y_all[n_train:n_train + n_val],
        "test": y_all[n_train + n_val:n_train + n_val + n_test],
        "real": y_all[n_train + n_val + n_test:n_train + n_val + n_test + n_real],
        "n_train": n_train, "n_val": n_val, "n_test": n_test, "n_real": n_real,
        "dates": dates,
    }


def load_model_predictions(model_name: str) -> dict:
    """per_model_optimal/{model}.json + P1 real_forecaster fallback.

    Phase C.6 + C.7 (sprint 2026-05-06): saved JSON 의 새 fields 우선 사용:
        - refit_real_predictions (rolling-origin Real-Slab, methodology §4.1)
        - refit_real_pi95_lo / refit_real_pi95_hi (ACI Gibbs & Candès 2021)
        - real_metrics (Section B descriptive: mae/picp95/peak_hit/aci_coverage)

    G-164 (2026-05-02): fallback path 보존 — saved JSON 에 새 fields 없으면
    P1 real_forecaster (real_eval) 결과 사용; 둘 다 없으면 None (plot 이 명시 처리).
    """
    p = get_results_dir() / "per_model_optimal" / f"{model_name}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))

    # Phase C.6 우선 — saved JSON 의 refit_real_predictions
    real_pred = d.get("refit_real_predictions")
    real_pi95_lo = d.get("refit_real_pi95_lo")
    real_pi95_hi = d.get("refit_real_pi95_hi")
    real_metrics = d.get("real_metrics") or {}
    aci_alpha_history = d.get("aci_alpha_history") or []

    # Fallback (G-164): P1 real_forecaster (real_eval) 의 진짜 source
    if real_pred is None:
        p10_per = get_results_dir() / "real_eval" / "per_model" / f"{model_name}.json"
        if p10_per.exists():
            try:
                r = json.loads(p10_per.read_text(encoding="utf-8"))
                real_pred = r.get("predictions") or r.get("real_pred") or None
                if not real_metrics:
                    real_metrics = {k: r[k] for k in
                                     ("r2", "mae", "rmse", "wis", "mape")
                                     if k in r}
            except Exception:
                pass
    if real_pred is None:
        p10_full = get_results_dir() / "real_eval" / "metrics_full.json"
        if p10_full.exists():
            try:
                full = json.loads(p10_full.read_text(encoding="utf-8"))
                if isinstance(full, dict) and model_name in full:
                    entry = full[model_name]
                    if isinstance(entry, dict):
                        real_pred = entry.get("predictions") or None
                        if not real_metrics:
                            real_metrics = {k: entry[k] for k in
                                            ("r2", "mae", "rmse", "wis", "mape")
                                            if k in entry}
            except Exception:
                pass

    return {
        "test_pred": d.get("refit_test_predictions", []),
        "val_metrics": d.get("val_metrics", {}),
        "test_metrics": d.get("test_metrics", {}),
        "best_metrics": d.get("best_metrics", {}),
        "best_config": d.get("best_config", {}),
        # G-164: real prediction (진짜 source) — 없으면 None (plot 이 명시 처리)
        "real_pred": real_pred,
        "real_metrics": real_metrics,
        # Phase C.6 + C.7 (2026-05-06): saved ACI PI + α history
        "real_pi95_lo": real_pi95_lo,
        "real_pi95_hi": real_pi95_hi,
        "aci_alpha_history": aci_alpha_history,
    }


def estimate_pi(test_pred: np.ndarray, residual_std: float,
                 expand_factor: float = 1.0) -> tuple:
    """Fixed-width PI (95%) — 시간 갈수록 너비 expand_factor 로 확장.

    Args:
        test_pred: predictions
        residual_std: train residual std (PI 너비 기준)
        expand_factor: 시간 흐름 시 PI 너비 증가율 (1.0=일정, 1.5=50% 증가)

    Returns:
        (low, high) 95% PI
    """
    n = len(test_pred)
    # 시간 흐름에 따른 너비 증가 (linear)
    widths = np.linspace(1.0, expand_factor, n) * Z95 * residual_std
    low = test_pred - widths
    high = test_pred + widths
    return low, high


def plot_model_forecast(model_name: str, data: dict, pred_data: dict,
                          out_dir: Path):
    """단일 모델의 4-way split + PI fan chart.

    Phase H.4 (sprint 2026-05-06): research/service zone shading + saved
    ACI PI (Phase C.7 Gibbs2021) + Section B real_metrics annotation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not pred_data.get("test_pred"):
        print(f"  ⊘ {model_name}: no predictions")
        return None

    test_pred = np.asarray(pred_data["test_pred"][:data["n_test"]], dtype=float)
    if not np.isfinite(test_pred).all():
        print(f"  ⊘ {model_name}: non-finite predictions")
        return None

    # Time axis (week index)
    n_train, n_val = data["n_train"], data["n_val"]
    n_test, n_real = data["n_test"], data["n_real"]

    t_train = np.arange(0, n_train)
    t_val = np.arange(n_train, n_train + n_val)
    t_test = np.arange(n_train + n_val, n_train + n_val + n_test)
    t_real = np.arange(n_train + n_val + n_test, n_train + n_val + n_test + n_real)

    # Residual std (train residual 기반 — actual 사용)
    test_metrics = pred_data.get("test_metrics", {})
    rmse = test_metrics.get("rmse", np.std(test_pred - data["test"][:len(test_pred)]))
    residual_std = float(rmse)

    # Test PI: split-conformal (research zone)
    test_low, test_high = estimate_pi(test_pred, residual_std, expand_factor=1.0)

    # G-164 + Phase C.6: real prediction
    real_pred_raw = pred_data.get("real_pred")
    real_unavailable = False
    if real_pred_raw is not None and len(real_pred_raw) >= 1:
        real_pred = np.asarray(real_pred_raw[:n_real], dtype=float)
        if len(real_pred) < n_real:
            real_pred = np.concatenate([
                real_pred,
                np.full(n_real - len(real_pred), real_pred[-1])
            ])
    else:
        real_pred = np.full(n_real, np.nan)
        real_unavailable = True

    # Phase C.7: saved ACI PI 우선, fallback estimate_pi
    saved_real_lo = pred_data.get("real_pi95_lo")
    saved_real_hi = pred_data.get("real_pi95_hi")
    pi_source = "fallback (estimate_pi)"
    if (saved_real_lo is not None and saved_real_hi is not None
            and len(saved_real_lo) >= n_real and len(saved_real_hi) >= n_real):
        real_low = np.asarray(saved_real_lo[:n_real], dtype=float)
        real_high = np.asarray(saved_real_hi[:n_real], dtype=float)
        pi_source = "ACI Gibbs & Candès 2021"
    else:
        real_low, real_high = estimate_pi(real_pred, residual_std, expand_factor=2.5)

    # Plot setup — figsize 확장하여 service annotation 을 plot 외부 panel 로
    fig, ax = plt.subplots(figsize=(17, 5.8))
    fig.subplots_adjust(left=0.06, right=0.76, top=0.86, bottom=0.16)

    # Phase H.4 (사용자 명시 2026-05-06): research/service zone shading
    # — methodology §4.1 + paper §4.8 framing
    research_end = n_train + n_val + n_test
    service_end = research_end + n_real
    ax.axvspan(0, research_end, alpha=0.06, color="lightgray", zorder=0)
    ax.axvspan(research_end, service_end, alpha=0.12, color="gold", zorder=0)

    # Train/val/test/real (actual)
    ax.plot(t_train, data["train"], "k-", alpha=0.4, lw=0.8,
            label="actual (train)")
    ax.plot(t_val, data["val"], "b-", alpha=0.7, lw=1.0, label="actual (val)")
    ax.plot(t_test, data["test"], "g-", alpha=0.9, lw=1.2, label="actual (test)")
    ax.plot(t_real, data["real"], "r-", alpha=0.9, lw=1.8, label="actual (real)")

    # Test prediction + PI (research zone)
    ax.plot(t_test, test_pred, "g--", lw=1.5, label="pred (test)")
    ax.fill_between(t_test, test_low, test_high, color="green", alpha=0.15,
                     label="95% PI (test, split-CP)")

    # Real prediction + ACI PI (service zone)
    if not real_unavailable:
        ax.plot(t_real, real_pred, "r--", lw=2.0, label="pred (real)")
        ax.fill_between(t_real, real_low, real_high, color="red", alpha=0.22,
                         label=f"95% PI (real, {pi_source})")
    else:
        ax.text(t_real[len(t_real) // 2], data["real"].mean(),
                "real prediction\nunavailable\n(Phase E retrain 후 자동 갱신)",
                ha="center", va="center", fontsize=9, color="red",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"))

    # Split boundaries (vertical lines)
    for boundary, label, color in [
        (n_train, "train|val", "blue"),
        (n_train + n_val, "val|test", "green"),
        (research_end, "test|real", "red"),
    ]:
        ax.axvline(boundary, color=color, alpha=0.3, ls=":", lw=1)

    # Set ylim BEFORE positioning text (matplotlib quirk)
    ax.relim(); ax.autoscale_view()
    y_max = ax.get_ylim()[1]

    # Phase H.4: zone labels (top, large)
    ax.text((research_end) / 2, y_max * 1.02,
            "Research zone (inferential — paper §결과 headline)",
            ha="center", fontsize=10, color="gray", weight="bold")
    ax.text(research_end + n_real / 2, y_max * 1.02,
            "Service zone\n(operational, descriptive)",
            ha="center", fontsize=10, color="darkorange", weight="bold")

    # Split boundary labels (rotated, small)
    for boundary, label, color in [
        (n_train, "train|val", "blue"),
        (n_train + n_val, "val|test", "green"),
        (research_end, "test|real", "red"),
    ]:
        ax.text(boundary, y_max * 0.92, label,
                rotation=90, fontsize=8, color=color, alpha=0.6, ha="right")

    # Title — research zone forecasting metrics (R² + MAPE + WIS, methodology §5)
    test_r2 = test_metrics.get("r2", float("nan"))
    test_wis = test_metrics.get("wis", float("nan"))
    test_mape = test_metrics.get("mape", float("nan"))
    test_smape = test_metrics.get("smape", float("nan"))
    title = (f"{model_name} — Research (n={n_test}): "
             f"R²={test_r2:+.4f}, MAPE={test_mape:.1f}%, "
             f"SMAPE={test_smape:.1f}%, WIS={test_wis:.2f}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("ILI rate (%)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Phase H.4 (사용자 2026-05-06): xlabel — Year-Month + Week index
    # cache parquet 의 week_start 활용; fallback "Week index"
    dates = data.get("dates")
    n_full = n_train + n_val + n_test + n_real
    if dates is not None and len(dates) >= n_full:
        try:
            import pandas as pd  # noqa: local import
            dt_arr = pd.to_datetime(dates[:n_full])
            # 약 1년(52주) 간격 + 0/끝 표시
            tick_step = max(26, n_full // 7)
            tick_indices = list(range(0, n_full, tick_step))
            if (n_full - 1) not in tick_indices:
                tick_indices.append(n_full - 1)
            tick_labels = []
            for i in tick_indices:
                if i < len(dt_arr):
                    d = dt_arr[i]
                    iso_w = int(d.isocalendar().week)
                    tick_labels.append(f"{d.year}-{d.month:02d}\n(W{iso_w})\nidx={i}")
                else:
                    tick_labels.append(f"idx={i}")
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels, fontsize=7.5)
            ax.set_xlabel("Date (Year-Month, ISO week) + week index",
                          fontsize=10)
        except Exception:
            ax.set_xlabel("Week index")
    else:
        ax.set_xlabel("Week index")

    # Service zone annotation (figure-level right panel — plot 데이터 가리지 X).
    # 사용자 명시 priority: MAPE/SMAPE headline, R² n=8 underpowered caveat.
    rm = pred_data.get("real_metrics") or {}
    if rm and not real_unavailable:
        real_mape = rm.get("mape", float("nan"))
        real_smape = rm.get("smape", float("nan"))
        real_mae = rm.get("mae", float("nan"))
        real_picp = rm.get("picp95", float("nan"))
        real_peak = rm.get("peak_hit_week_diff", 0)
        real_aci = rm.get("aci_realized_coverage", float("nan"))
        real_r2 = rm.get("r2", float("nan"))
        annot_lines = [
            f"Service zone (n={n_real}, descriptive)",
            "─ Headline (Hyndman 2021) ─",
            f"MAPE  = {real_mape:.1f}%",
            f"SMAPE = {real_smape:.1f}%",
            "─ Supplementary ─",
            f"MAE      = {real_mae:.2f}",
            f"PICP95   = {real_picp:.2f}/{n_real}",
            f"peak Δ   = {real_peak:+d}w",
            f"ACI cov  = {real_aci:.2f}",
            f"PI: {pi_source}",
            "─ Caveat (paper §6.4) ─",
            f"R²={real_r2:+.3f}  (n=8 underpowered)",
        ]
        # figure-level (plot 외부 right panel) — overlap 차단
        fig.text(0.78, 0.86,
                 "\n".join(annot_lines),
                 fontsize=8, color="darkorange",
                 ha="left", va="top",
                 family="monospace",
                 bbox=dict(facecolor="white", alpha=0.95,
                           edgecolor="darkorange", lw=1.2,
                           boxstyle="round,pad=0.5"))

    out_path = out_dir / f"forecast_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary_grid(model_results: list, out_dir: Path):
    """모든 모델 grid (3 cols × N rows)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(model_results)
    if n == 0:
        return None
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for ax, item in zip(axes, model_results):
        # 미니 forecast (test 부분만)
        name = item["name"]
        test_pred = np.asarray(item.get("test_pred", []), dtype=float)
        test_actual = np.asarray(item.get("test_actual", []), dtype=float)
        if len(test_pred) == 0:
            ax.text(0.5, 0.5, f"{name}\n(no data)", ha="center", va="center",
                     transform=ax.transAxes)
            continue
        n_t = min(len(test_pred), len(test_actual))
        ax.plot(test_actual[:n_t], "g-", lw=1.0, label="actual")
        ax.plot(test_pred[:n_t], "r--", lw=1.0, label="pred")
        r2 = item.get("test_r2", float("nan"))
        ax.set_title(f"{name} (R²={r2:+.3f})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide remaining
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Model forecast summary (test)", fontsize=14, fontweight="bold")
    out_path = out_dir / "_summary_grid.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(get_results_dir() / "plots_forecast_full"))
    ap.add_argument("--models", default=None,
                       help="Comma-separated model names (default: all in per_model_optimal)")
    ap.add_argument("--top", type=int, default=0,
                       help="Top N models (by test R²)")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data = load_split_data()
    print(f"Split: train={data['n_train']}, val={data['n_val']}, "
          f"test={data['n_test']}, real={data['n_real']}")

    # Models
    pmo_dir = get_results_dir() / "per_model_optimal"
    if args.models:
        model_list = args.models.split(",")
    else:
        model_list = sorted(f.stem for f in pmo_dir.glob("*.json"))

    print(f"Models: {len(model_list)}")
    print()

    # Filter top by R²
    if args.top > 0:
        scored = []
        for m in model_list:
            d = load_model_predictions(m)
            r2 = d.get("test_metrics", {}).get("r2", float("-inf"))
            if r2 == r2:  # not nan
                scored.append((m, r2))
        scored.sort(key=lambda x: -x[1])
        model_list = [m for m, _ in scored[:args.top]]
        print(f"Top {args.top}: {model_list}")

    # Plot each
    grid_data = []
    for m in model_list:
        pd = load_model_predictions(m)
        if not pd.get("test_pred"):
            print(f"  ⊘ {m}: skip (no predictions)")
            continue
        path = plot_model_forecast(m, data, pd, out_dir)
        if path:
            print(f"  ✓ {m} → {path.name}")
            grid_data.append({
                "name": m,
                "test_pred": pd["test_pred"][:data["n_test"]],
                "test_actual": data["test"].tolist(),
                "test_r2": pd.get("test_metrics", {}).get("r2", float("nan")),
            })

    # Summary grid
    grid_path = plot_summary_grid(grid_data, out_dir)
    if grid_path:
        print(f"\n  ✓ Summary grid: {grid_path}")

    print(f"\nOutput: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
