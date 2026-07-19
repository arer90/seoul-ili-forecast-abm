"""Phase A/B 4-method multicollinearity 비교 보고서 생성기.

사용법:
    .venv/bin/python simulation/scripts/generate_4method_report.py            # Phase A
    .venv/bin/python simulation/scripts/generate_4method_report.py --phase b  # Phase B

출력:
    simulation/results/phase_{a|b}_4method_report.html
    simulation/results/phase_{a|b}_4method_metrics.csv
    simulation/results/phase_{a|b}_4method_summary.csv
"""
from __future__ import annotations
import argparse
import json
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _ROOT / "simulation" / "results"

METHOD_LABELS = {
    "none": "D — None (baseline)",
    "vif":  "A — VIF iterative drop (threshold=10)",
    "corr": "B — |corr|>0.9 + MI tie-break",
    "pca":  "C — PCA (95% variance)",
}

# 60 metric 표시 그룹 (compute_full_metrics 키 기준)
METRIC_GROUPS = [
    ("Point",        ["rmse", "mae", "mape", "smape", "r2", "pearson_r", "spearman_r", "mse"]),
    ("MASE-multi",   ["mase_h4", "mase_h13", "mase_h52"]),
    ("Bias-Scale",   ["bias_mean_error", "msle", "theils_u"]),
    ("Probabilistic",["wis", "crps_naive", "log_score_gauss", "interval_score_50",
                      "interval_score_95", "skill_score_wis"]),
    ("PIT",          ["pit_ks_stat", "pit_ks_p_value", "pit_coverage_68"]),
    ("PI-coverage",  ["pi50_coverage", "pi80_coverage", "pi95_coverage",
                      "pi50_coverage_2", "pi80_coverage_2", "pi95_coverage_2"]),
    ("PI-width",     ["pi50_width", "pi80_width", "pi95_width",
                      "pi50_width_2", "pi80_width_2", "pi95_width_2"]),
    ("Epi",          ["epi_peak_week_err", "epi_peak_mae", "epi_season_total_mae"]),
    ("Alert",        ["alert_f1"]),
    ("Calibration",  ["brier_score", "brier_skill_score",
                      "calibration_slope", "calibration_intercept",
                      "hl_chi2", "hl_p_value"]),
    ("Discrimination",["c_index"]),
    ("Cost",         ["s_index"]),
    ("Meta",         ["pi_method", "n_test", "n_train_pool", "sigma_used"]),
]

ALL_METRICS = [m for _, ms in METRIC_GROUPS for m in ms]


# G-233 이전 flat-grid 결과 — 4-method 비교에 사용 불가 (명시적 제외)
_LEGACY_DIR_NAMES = {
    "per_model_optimal_none_20260522_222334",
    "per_model_optimal_none_20260522_222503",
    "per_model_optimal_backup_v16_20260522_173039",
}


def _find_method_dir(method: str, phase: str = "a") -> Path | None:
    """Latest G-233 이후 hierarchical per_model_optimal directory.
    Legacy flat-grid 결과 (G-233 이전) 는 자동 제외.
    """
    prefix = "b_" if phase == "b" else ""
    dirs = [
        d for d in sorted(RESULTS_DIR.glob(f"per_model_optimal_{prefix}{method}_*"))
        if d.name not in _LEGACY_DIR_NAMES
    ]
    return dirs[-1] if dirs else None


def _load_method_results(method: str, phase: str = "a") -> dict[str, dict]:
    """method → {model: test_metrics dict}."""
    d = _find_method_dir(method, phase)
    if d is None:
        return {}
    results = {}
    for jf in d.glob("*.json"):
        if jf.stem == "summary" or jf.stem == "multicollinearity_meta":
            continue
        try:
            data = json.loads(jf.read_text())
            # test_metrics 우선, 없으면 val_metrics
            metrics = data.get("test_metrics") or data.get("val_metrics") or {}
            if metrics:
                results[jf.stem] = metrics
        except Exception:
            pass
    return results


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (v != v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _color_r2(v) -> str:
    try:
        fv = float(v)
        if fv >= 0.9:  return "#d4edda"
        if fv >= 0.80: return "#fff3cd"
        return "#f8d7da"
    except Exception:
        return ""


def _color_mape(v) -> str:
    try:
        fv = float(v)
        if fv <= 15:   return "#d4edda"
        if fv <= 20:   return "#fff3cd"
        return "#f8d7da"
    except Exception:
        return ""


def build_csv(all_data: dict[str, dict[str, dict]], out_dir: Path, phase: str = "a") -> list[Path]:
    paths = []

    # 1) full metrics CSV (method × model × metric)
    rows = []
    for method, model_map in all_data.items():
        for model, metrics in model_map.items():
            row = {"method": method, "model": model}
            for k in ALL_METRICS:
                row[k] = metrics.get(k, "")
            rows.append(row)

    cols = ["method", "model"] + ALL_METRICS
    p1 = out_dir / f"phase_{phase}_4method_metrics.csv"
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    paths.append(p1)

    # 2) summary: best per method (r2 / rmse / mape / wis)
    summary_rows = []
    for method, model_map in all_data.items():
        for model, metrics in model_map.items():
            summary_rows.append({
                "method": method,
                "model": model,
                "r2":    metrics.get("r2", float("nan")),
                "rmse":  metrics.get("rmse", float("nan")),
                "mape":  metrics.get("mape", float("nan")),
                "wis":   metrics.get("wis", float("nan")),
                "pi95":  metrics.get("pi95_coverage", float("nan")),
                "n_features_kept": metrics.get("n_features_kept", ""),
            })

    p2 = out_dir / f"phase_{phase}_4method_summary.csv"
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method","model","r2","rmse","mape","wis","pi95","n_features_kept"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    paths.append(p2)

    return paths


def build_html(all_data: dict[str, dict[str, dict]], csv_paths: list[Path], out_path: Path,
               phase: str = "a"):
    methods = list(all_data.keys())
    all_models = sorted({m for mm in all_data.values() for m in mm})
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    phase_label = phase.upper()

    lines = [
        "<!DOCTYPE html><html lang='ko'><head>",
        "<meta charset='UTF-8'>",
        f"<title>Phase {phase_label} — 4-Method Multicollinearity Report</title>",
        "<style>",
        "body{font-family:'Noto Sans KR',sans-serif;margin:20px;font-size:13px}",
        "h1{color:#2c3e50}h2{color:#34495e;margin-top:30px}",
        "table{border-collapse:collapse;width:100%;margin-bottom:20px}",
        "th,td{border:1px solid #ddd;padding:5px 8px;text-align:center}",
        "th{background:#2c3e50;color:white}",
        "tr:hover{background:#f0f4f8}",
        ".method-D{background:#e8f4f8}.method-A{background:#fef9e7}",
        ".method-B{background:#eafaf1}.method-C{background:#fdf2f8}",
        ".pass{background:#d4edda}.warn{background:#fff3cd}.fail{background:#f8d7da}",
        ".toc{background:#f8f9fa;padding:10px;border-radius:6px;margin-bottom:20px}",
        ".toc a{color:#2980b9;text-decoration:none;margin-right:15px}",
        "</style></head><body>",
        f"<h1>Phase {phase_label} — 4-Method Multicollinearity 비교 보고서</h1>",
        f"<p>생성: {now} &nbsp;|&nbsp; 모델: {len(all_models)}개 &nbsp;|&nbsp; "
        f"Methods: {len(methods)}개</p>",
        f"<div class='toc'>",
        "<b>목차</b><br>",
        "<a href='#overview'>1. 개요</a>",
        "<a href='#r2'>2. R² 비교</a>",
        "<a href='#mape'>3. MAPE 비교</a>",
        "<a href='#wis'>4. WIS 비교</a>",
        "<a href='#pi95'>5. PI95 Coverage</a>",
        "<a href='#full'>6. 전체 60 metric</a>",
        "<a href='#download'>7. CSV 다운로드</a>",
        "</div>",
    ]

    # 1. Overview
    lines.append("<h2 id='overview'>1. Method 개요</h2>")
    lines.append("<table><tr><th>코드</th><th>Method</th><th>적용 기준</th><th>모델 수</th></tr>")
    for method in ["none", "vif", "corr", "pca"]:
        n = len(all_data.get(method, {}))
        desc = {
            "none": "passthrough — Stage 2 Optuna parsimony 만",
            "vif": "VIF > 10 iterative drop (Belsley/Kuh/Welsch 1980)",
            "corr": "|corr| > 0.9 + MI tie-break (Dormann 2013, Guyon 2003)",
            "pca": "PCA 95% variance retained (Jolliffe 2002)",
        }[method]
        lines.append(f"<tr class='method-{method.upper() if method!='none' else 'D'}'>"
                     f"<td>{method}</td><td>{METHOD_LABELS.get(method,method)}</td>"
                     f"<td>{desc}</td><td>{n}</td></tr>")
    lines.append("</table>")

    def metric_table(metric_key: str, section_id: str, title: str, color_fn=None, higher_better=True):
        lines.append(f"<h2 id='{section_id}'>{title}</h2>")
        lines.append("<table><tr><th>Model</th>")
        for m in methods:
            lines.append(f"<th class='method-{m.upper() if m!='none' else 'D'}'>{m}</th>")
        lines.append("<th>Best Method</th></tr>")
        for model in all_models:
            vals = {}
            for m in methods:
                v = all_data.get(m, {}).get(model, {}).get(metric_key)
                try:
                    vals[m] = float(v)
                except Exception:
                    vals[m] = None
            valid = {m: v for m, v in vals.items() if v is not None and not np.isnan(v)}
            best_m = (max if higher_better else min)(valid, key=lambda m: valid[m]) if valid else None
            lines.append(f"<tr><td><b>{model}</b></td>")
            for m in methods:
                v = vals.get(m)
                cell_style = ""
                if color_fn and v is not None:
                    cell_style = f" style='background:{color_fn(v)}'"
                if m == best_m:
                    cell_style = " style='background:#b8e0d2;font-weight:bold'"
                lines.append(f"<td{cell_style}>{_fmt(v)}</td>")
            lines.append(f"<td>{best_m or '—'}</td></tr>")
        lines.append("</table>")

    metric_table("r2",   "r2",   "2. R² (높을수록 좋음)", _color_r2, higher_better=True)
    metric_table("mape", "mape", "3. MAPE (낮을수록 좋음)", _color_mape, higher_better=False)
    metric_table("wis",  "wis",  "4. WIS (낮을수록 좋음)", higher_better=False)
    metric_table("pi95_coverage", "pi95", "5. PI95 Coverage (≥0.90 목표)", higher_better=True)

    # 6. Full 60-metric table (by group)
    lines.append("<h2 id='full'>6. 전체 Metric (그룹별)</h2>")
    for group_name, group_metrics in METRIC_GROUPS:
        lines.append(f"<h3>{group_name}</h3>")
        lines.append("<table><tr><th>Model</th><th>Metric</th>")
        for m in methods:
            lines.append(f"<th>{m}</th>")
        lines.append("</tr>")
        for model in all_models:
            for gi, metric in enumerate(group_metrics):
                row_label = f"<b>{model}</b>" if gi == 0 else ""
                lines.append(f"<tr><td>{row_label}</td><td>{metric}</td>")
                for m in methods:
                    v = all_data.get(m, {}).get(model, {}).get(metric)
                    lines.append(f"<td>{_fmt(v)}</td>")
                lines.append("</tr>")
        lines.append("</table>")

    # 7. CSV download
    lines.append("<h2 id='download'>7. CSV 다운로드</h2><ul>")
    for p in csv_paths:
        lines.append(f"<li><a href='{p.name}'>{p.name}</a></li>")
    lines.append("</ul>")

    lines.append("</body></html>")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  HTML → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="4-method multicollinearity 비교 보고서 생성")
    parser.add_argument("--phase", choices=["a", "b"], default="a",
                        help="Phase A (ML+STAT) or B (DL+EPI+ENSEMBLE)")
    args = parser.parse_args()
    phase = args.phase

    all_data: dict[str, dict[str, dict]] = {}
    for method in ["none", "vif", "corr", "pca"]:
        d = _load_method_results(method, phase)
        if d:
            all_data[method] = d
            print(f"  {method}: {len(d)} models")
        else:
            print(f"  {method}: no results (skip)")

    if not all_data:
        print(f"결과 없음. Phase {phase.upper()} 학습 완료 후 다시 실행하세요.")
        sys.exit(1)

    out_dir = RESULTS_DIR
    csv_paths = build_csv(all_data, out_dir, phase)
    html_path = out_dir / f"phase_{phase}_4method_report.html"
    build_html(all_data, csv_paths, html_path, phase)

    print(f"\n보고서 완료:")
    print(f"  HTML: {html_path}")
    for p in csv_paths:
        print(f"  CSV:  {p}")


if __name__ == "__main__":
    main()
