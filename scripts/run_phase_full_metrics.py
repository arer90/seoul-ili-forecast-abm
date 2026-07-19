"""
Phase-별 full metric wire-in + eLife 4 metric 표시.

OUTPUTS:
    simulation/results/phase_full_metrics/
    ├── phase11_full_metrics.csv          # 22 model × 53 metric
    ├── phase12_full_metrics.csv          # 54 model × 27 metric
    ├── phase13_comprehensive_full.csv    # 36 model × 77 col (copy)
    ├── phase11_heatmap.png               # model × metric heatmap (z-score normalized)
    ├── phase12_heatmap.png               # 동일
    ├── per_phase_metric_coverage.csv     # phase × metric matrix (1=available, 0=missing)
    └── elife_metric_inclusion_table.csv  # 각 phase 의 4 eLife metric 포함 확인
"""
from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).parent.parent
OUT = REPO / "simulation" / "results" / "phase_full_metrics"
OUT.mkdir(parents=True, exist_ok=True)

# eLife 4 standard metrics
ELIFE_4 = {"MAE", "RMSE", "MAPE", "SMAPE"}
ELIFE_4_LOWER = {"mae", "rmse", "mape", "smape"}


def load_phase11():
    """Phase 11 SSOT — 22 model × 53 metric."""
    src = REPO / "simulation" / "results" / "phase11_per_model_eval" / "per_model_metrics.csv"
    if not src.exists():
        return None, None
    rows, header = [], None
    with open(src) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        for r in rdr:
            rows.append(r)
    return rows, header


def load_phase12():
    """Phase 12 — extract test_metrics from per_model_optimal/*.json (54 model × 27 metric)."""
    src_dir = REPO / "simulation" / "results" / "per_model_optimal"
    all_metric_keys = set()
    model_data = {}
    for json_path in sorted(src_dir.glob("*.json")):
        if json_path.name.startswith("_") or json_path.name == "summary.json":
            continue
        try:
            d = json.load(open(json_path))
        except Exception:
            continue
        model_name = d.get("model", json_path.stem)
        tm = d.get("test_metrics", {})
        if not tm:
            continue
        model_data[model_name] = tm
        all_metric_keys.update(tm.keys())
    return model_data, sorted(all_metric_keys)


def write_phase11_csv(rows, header):
    out_path = OUT / "phase11_full_metrics.csv"
    with open(out_path, "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    log.info(f"✓ {out_path.name} — {len(rows)} models × {len(header)-1} metrics")
    return out_path


def write_phase12_csv(model_data, metric_keys):
    """Phase 12 wide CSV — model × metric."""
    out_path = OUT / "phase12_full_metrics.csv"
    with open(out_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["model"] + metric_keys)
        for model in sorted(model_data.keys()):
            row = [model]
            for k in metric_keys:
                v = model_data[model].get(k, "")
                row.append(f"{v:.6f}" if isinstance(v, (int, float)) else str(v))
            w.writerow(row)
    log.info(f"✓ {out_path.name} — {len(model_data)} models × {len(metric_keys)} metrics")
    return out_path


def plot_heatmap(rows, header, title, outpath, model_col="model", invert_metrics=None):
    """Generate metric heatmap (model × metric, z-score normalized per metric)."""
    invert_metrics = invert_metrics or set()
    # Parse numeric metrics
    model_names = []
    metric_names = [c for c in header[1:] if c not in {"n_test", "n", "sigma_in_sample"}]
    matrix = []
    for row in rows:
        if not row:
            continue
        model_names.append(row[0])
        values = []
        for i, c in enumerate(header[1:], 1):
            if c in {"n_test", "n", "sigma_in_sample"}:
                continue
            try:
                v = float(row[i])
                if np.isnan(v) or np.isinf(v):
                    v = np.nan
            except (ValueError, IndexError):
                v = np.nan
            values.append(v)
        matrix.append(values)
    M = np.array(matrix, dtype=float)

    # z-score per column (metric)
    M_z = np.zeros_like(M)
    for j, mn in enumerate(metric_names):
        col = M[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) > 1 and valid.std() > 1e-9:
            M_z[:, j] = (col - valid.mean()) / valid.std()
        else:
            M_z[:, j] = 0
        # Invert if "lower is better"
        if mn.lower() in ("mae", "rmse", "mse", "mape", "smape", "mdape", "wis", "log_wis",
                            "crps_gaussian", "peak_week_err", "peak_int_relerr", "bias_mean_error",
                            "pi95_width", "pi80_width", "pi50_width", "pinball_q05", "pinball_q50",
                            "pinball_q95", "log_score_gauss", "pit_std", "rank_wis", "rank_log_wis",
                            "relative_wis_pairwise", "msle", "theils_u") or mn in invert_metrics:
            M_z[:, j] = -M_z[:, j]

    fig, ax = plt.subplots(figsize=(max(16, len(metric_names) * 0.35), max(10, len(model_names) * 0.32)))
    im = ax.imshow(M_z, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    # Highlight eLife 4 metric columns
    for j, mn in enumerate(metric_names):
        if mn.lower() in ELIFE_4_LOWER:
            ax.axvline(j - 0.5, color="#dc2626", linewidth=1.5, alpha=0.7)
            ax.axvline(j + 0.5, color="#dc2626", linewidth=1.5, alpha=0.7)
    ax.set_title(f"{title}\n(Z-score per metric, green=better, red=worse. eLife 4 metric (MAE/RMSE/MAPE/SMAPE) outlined red.)",
                  fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"✓ {outpath.name}")
    return outpath


def coverage_matrix(phase_metrics: dict[str, list[str]], outpath: Path):
    """Build phase × metric coverage matrix."""
    all_metrics = set()
    for ms in phase_metrics.values():
        all_metrics.update(m.lower() for m in ms)
    all_metrics = sorted(all_metrics)

    with open(outpath, "w") as f:
        w = csv.writer(f)
        w.writerow(["metric", "elife_4"] + list(phase_metrics.keys()))
        for m in all_metrics:
            is_elife = "✅" if m in ELIFE_4_LOWER else ""
            row = [m, is_elife]
            for phase, ms in phase_metrics.items():
                row.append("✅" if m in [x.lower() for x in ms] else "")
            w.writerow(row)
    log.info(f"✓ {outpath.name} — {len(all_metrics)} unique metrics across {len(phase_metrics)} phases")


def elife_inclusion_summary(phase_metrics: dict[str, list[str]], outpath: Path):
    """Check each phase 의 eLife 4 metric 포함 여부."""
    with open(outpath, "w") as f:
        w = csv.writer(f)
        w.writerow(["phase", "n_metrics_total", "MAE", "RMSE", "MAPE", "SMAPE", "elife_4_complete"])
        for phase, ms in phase_metrics.items():
            ms_lower = {m.lower() for m in ms}
            present = {k: ("✅" if k.lower() in ms_lower else "❌") for k in ["MAE", "RMSE", "MAPE", "SMAPE"]}
            complete = "✅" if all(v == "✅" for v in present.values()) else "❌"
            w.writerow([phase, len(ms), present["MAE"], present["RMSE"], present["MAPE"], present["SMAPE"], complete])
    log.info(f"✓ {outpath.name}")


def main():
    # Phase 11
    p11_rows, p11_header = load_phase11()
    if p11_rows:
        write_phase11_csv(p11_rows, p11_header)
        plot_heatmap(p11_rows, p11_header,
                      f"Phase 11 SSOT — {len(p11_rows)} models × {len(p11_header)-1} metrics (z-score per metric)",
                      OUT / "phase11_heatmap.png")
        phase11_metric_list = p11_header[1:]
    else:
        log.warning("Phase 11 CSV not found")
        phase11_metric_list = []

    # Phase 12
    p12_data, p12_metrics = load_phase12()
    if p12_data:
        write_phase12_csv(p12_data, p12_metrics)
        # Convert to rows for heatmap
        p12_header = ["model"] + p12_metrics
        p12_rows = []
        for m in sorted(p12_data.keys()):
            row = [m]
            for k in p12_metrics:
                v = p12_data[m].get(k, "")
                row.append(str(v) if v != "" else "")
            p12_rows.append(row)
        plot_heatmap(p12_rows, p12_header,
                      f"Phase 12 HP Optuna — {len(p12_data)} models × {len(p12_metrics)} metrics",
                      OUT / "phase12_heatmap.png")

    # Coverage matrix + eLife inclusion
    phase_metrics = {
        "Phase 11 (SSOT)": phase11_metric_list,
        "Phase 12 (HP Optuna)": p12_metrics if p12_data else [],
        "Phase 15 (Cross-country)": ["MAE", "RMSE", "MAPE", "SMAPE"],
        "Phase 10 (Real eval slab)": ["mae", "rmse", "mape", "r2", "bias", "smape", "regime_high", "regime_low"],
    }
    coverage_matrix(phase_metrics, OUT / "per_phase_metric_coverage.csv")
    elife_inclusion_summary(phase_metrics, OUT / "elife_metric_inclusion_table.csv")

    # Copy Phase 13 comprehensive (optional — set MPH_PHASE13_COMPREHENSIVE_CSV to enable;
    # 하드코딩 개인 절대경로 제거, R8 2026-05-28 공개 repo 이식성 #1).
    _src13_env = os.environ.get("MPH_PHASE13_COMPREHENSIVE_CSV")
    src13 = Path(_src13_env) if _src13_env else None
    if src13 and src13.exists():
        import shutil
        shutil.copy(src13, OUT / "phase13_comprehensive_full.csv")
        log.info(f"✓ phase13_comprehensive_full.csv (36 model × 77 metric)")

    log.info(f"\n=== Summary ===")
    for phase, ms in phase_metrics.items():
        ms_lower = {m.lower() for m in ms}
        elife_count = sum(1 for k in ELIFE_4_LOWER if k in ms_lower)
        log.info(f"  {phase}: {len(ms)} metrics, eLife 4/{elife_count} present")
    log.info(f"\n✓ Output: {OUT}")


if __name__ == "__main__":
    main()
