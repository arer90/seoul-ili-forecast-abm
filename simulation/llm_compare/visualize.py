"""
simulation.llm_compare.visualize
================================
Publication figures for the LLM comparison (`factual_bench` reports).

Reads a ``factual_report.json`` and renders four thesis-grade panels:

  1. accuracy_bar      — ranked accuracy, cloud (CLI) vs local (MLX) colored
  2. forest_ci         — accuracy ± 95% bootstrap CI (overlap ⇒ not distinguishable)
  3. category_heatmap  — model × epi/law category accuracy (where each is strong/weak)
  4. accuracy_latency  — efficiency frontier (accuracy vs response time, log-x)

Plus a 2×2 combined panel. Korean labels render via the first available CJK font
(AppleGothic / NanumGothic / Malgun Gothic), else fall back to default.

CLI:  python -m simulation.llm_compare.visualize \
          --report simulation/results/llm_compare_combined/factual_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# CJK font (portable: try the common ones, fall back silently)
for _f in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"):
    if any(_f == ft.name for ft in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

_TIER_COLOR = {"cli": "#2c7fb8", "openai_compat": "#e6550d", "ollama": "#31a354",
               "local": "#e6550d", "mock": "#999999", "api": "#756bb1"}
_CATS = ["분류", "신고", "방역", "예방접종", "역학", "데이터", "KorMedMCQA"]


def _short(bid: str) -> str:
    """Compact display label for a backend_id."""
    if bid.startswith("cli:"):
        return bid.split(":")[1]
    if bid.startswith(("oai:", "local:", "ollama:")):
        name = bid.split("@")[0].split("/")[-1].split(":", 1)[-1]
        for s in ("-Instruct-4bit", "-Instruct", "-it-4bit", "-it", "-4bit", "-bf16"):
            name = name.replace(s, "")
        return name
    return bid


def _category_map() -> dict:
    from .kr_epi_bench import load_kr_epi_law
    return {it.id: it.category for it in load_kr_epi_law()}


def _cat_of(item_id: str, catmap: dict) -> str:
    if item_id in catmap:
        return catmap[item_id]
    return "KorMedMCQA" if item_id.startswith("KMQ") else "기타"


# ── individual panels (each takes a Matplotlib Axes) ─────────────────────────
def plot_accuracy_bar(report: dict, ax) -> None:
    rk = sorted(report["ranking"], key=lambda r: r["accuracy"])
    labels = [_short(r["backend_id"]) for r in rk]
    vals = [r["accuracy"] for r in rk]
    colors = [_TIER_COLOR.get(r["tier"], "#999") for r in rk]
    ax.barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.01, 0.92), i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("정확도 (accuracy)")
    ax.set_title("① 모델별 정확도 (cloud=파랑 · local=주황)")


def plot_forest_ci(report: dict, ax) -> None:
    rows = sorted(report.get("statistical_comparison", {}).get("ranking", []),
                  key=lambda r: r["mean"])
    if not rows:
        ax.text(0.5, 0.5, "통계 비교 없음 (백엔드 <2)", ha="center"); return
    y = list(range(len(rows)))
    means = [r["mean"] for r in rows]
    xerr = [[r["mean"] - r["lo"] for r in rows], [r["hi"] - r["mean"] for r in rows]]
    ax.errorbar(means, y, xerr=xerr, fmt="o", color="#333", ecolor="#999", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([_short(r["backend"]) for r in rows])
    ax.set_xlim(0, 1)
    ax.axvline(rows[-1]["lo"], ls="--", color="red", alpha=0.4)  # top model's CI lower edge
    ax.set_xlabel("정확도 ± 95% bootstrap CI")
    ax.set_title("② CI 겹치면 통계적으로 구분 불가 (빨강선=1위 하한)")


def plot_category_heatmap(report: dict, ax) -> None:
    catmap = _category_map()
    acc: dict = defaultdict(lambda: defaultdict(list))
    for pi in report["per_item"]:
        if pi.get("error"):
            continue
        acc[_short(pi["backend_id"])][_cat_of(pi["item_id"], catmap)].append(pi["total"])
    models = [_short(r["backend_id"]) for r in sorted(report["ranking"],
              key=lambda r: -r["accuracy"])]
    M = np.full((len(models), len(_CATS)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(_CATS):
            v = acc[m].get(c)
            if v:
                M[i, j] = sum(v) / len(v)
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(_CATS))); ax.set_xticklabels(_CATS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(_CATS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("③ 도메인 카테고리별 정확도 (녹색=강함)")


def plot_accuracy_vs_latency(report: dict, ax) -> None:
    lat: dict = defaultdict(list)
    for pi in report["per_item"]:
        if pi.get("error") or not pi.get("latency_ms"):
            continue
        lat[pi["backend_id"]].append(pi["latency_ms"])
    rk = {r["backend_id"]: r for r in report["ranking"]}
    for bid, ls in lat.items():
        if bid not in rk:
            continue
        x = (sum(ls) / len(ls)) / 1000.0
        y = rk[bid]["accuracy"]
        ax.scatter(x, y, color=_TIER_COLOR.get(rk[bid]["tier"], "#999"), s=60, zorder=3)
        ax.annotate(_short(bid), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel("평균 응답시간 (초, log)"); ax.set_ylabel("정확도")
    ax.set_title("④ 효율 프런티어 (좌상단=빠르고 정확)")


# ── orchestration ────────────────────────────────────────────────────────────
def make_figures(report_path, out_dir) -> list:
    """Render the four panels + a 2×2 combined figure. Returns written paths."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    n = report.get("n_items", "?")
    panels = [("accuracy_bar", plot_accuracy_bar), ("forest_ci", plot_forest_ci),
              ("category_heatmap", plot_category_heatmap),
              ("accuracy_latency", plot_accuracy_vs_latency)]
    written = []
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fn(report, ax)
        fig.tight_layout(); p = out / f"{name}.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(p)
    # combined 2×2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for (_, fn), ax in zip(panels, axes.flat):
        fn(report, ax)
    fig.suptitle(f"LLM 비교 (kr_epi+KorMedMCQA, n={n}) — config {report.get('repro_manifest',{}).get('config_sha256','')}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); p = out / "panel.png"
    fig.savefig(p, dpi=150); plt.close(fig); written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LLM comparison figures")
    ap.add_argument("--report", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out = args.out_dir or str(Path(args.report).parent / "figures")
    paths = make_figures(args.report, out)
    print("wrote:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
