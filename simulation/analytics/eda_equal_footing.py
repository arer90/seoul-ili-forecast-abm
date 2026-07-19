"""
EDA Equal-Footing Helper — cross-country fair visualization (§2 of 0_REPORT_MASTER).

PURPOSE: Generate 5 equal-footing EDA figures for cross-country cohort analysis.
Replaces /tmp/eda_28country_equal.py ad-hoc script.

CITATIONS:
- z-score normalization for unit incompatibility — Reis et al. (2018) PMC6185890
- Small multiples for cross-country comparison — Tufte (1983)

USAGE:
    from simulation.analytics.eda_equal_footing import (
        generate_all_eda_figures, CLUSTER_COLORS,
    )

    generate_all_eda_figures(
        cohort_data={"KR": z_series, "BE": z_series, ...},
        sources={"KR": "KDCA_sentinel", "BE": "ecdc_erviss", ...},
        output_dir=Path("figures/"),
        cohort_name="I-B",
        period_label="2021-2025",
    )

OUTPUT: 5 PNG files
    - eda_timeseries_grid.png  (small multiples)
    - eda_sample_size.png      (bar chart)
    - eda_boxplot.png          (z-score distribution)
    - eda_seasonal_phase.png   (week-of-year average)
    - eda_source_table.png     (source/unit comparison table)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

log = logging.getLogger(__name__)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ── Geographic / Climate Clusters ──
CLUSTER_COLORS = {
    # NW EU temperate
    "DE": ("NW EU", "#3b82f6"), "FR": ("NW EU", "#3b82f6"), "NL": ("NW EU", "#3b82f6"),
    "BE": ("NW EU", "#3b82f6"), "LU": ("NW EU", "#3b82f6"), "IE": ("NW EU", "#3b82f6"),
    "GB": ("NW EU", "#3b82f6"), "AT": ("NW EU", "#3b82f6"),
    # Nordic
    "SE": ("Nordic", "#0ea5e9"), "FI": ("Nordic", "#0ea5e9"), "NO": ("Nordic", "#0ea5e9"),
    "DK": ("Nordic", "#0ea5e9"), "IS": ("Nordic", "#0ea5e9"),
    # Baltic
    "EE": ("Baltic", "#06b6d4"), "LV": ("Baltic", "#06b6d4"), "LT": ("Baltic", "#06b6d4"),
    # CEE (Central/Eastern Europe)
    "PL": ("CEE", "#8b5cf6"), "CZ": ("CEE", "#8b5cf6"), "SK": ("CEE", "#8b5cf6"),
    "HU": ("CEE", "#8b5cf6"), "RO": ("CEE", "#8b5cf6"), "SI": ("CEE", "#8b5cf6"),
    "HR": ("CEE", "#8b5cf6"),
    # Mediterranean
    "IT": ("Mediterranean", "#f59e0b"), "ES": ("Mediterranean", "#f59e0b"),
    "PT": ("Mediterranean", "#f59e0b"), "GR": ("Mediterranean", "#f59e0b"),
    "MT": ("Mediterranean", "#f59e0b"),
    # East Asia
    "KR": ("E Asia", "#dc2626"), "JP": ("E Asia", "#ec4899"),
    "CN": ("E Asia", "#a855f7"), "HK": ("E Asia", "#a855f7"),
    "SG": ("SE Asia", "#10b981"),
    # N America / Oceania
    "US": ("N America", "#7c3aed"), "AU": ("Oceania", "#10b981"),
}


def cluster_of(country: str) -> tuple[str, str]:
    """Get (cluster_name, hex_color) for country."""
    return CLUSTER_COLORS.get(country, ("Other", "#94a3b8"))


# ── Figure Generators ──

def fig_timeseries_grid(
    data: dict[str, np.ndarray],
    sources: dict[str, str],
    period_label: str,
    outpath: Path,
    ncol: int = 4,
):
    """Small multiples — z-score time series per country, KR highlight."""
    sorted_cs = sorted(data.keys())
    N = len(sorted_cs)
    nrow = (N + ncol - 1) // ncol
    fig = plt.figure(figsize=(5 * ncol, 2.6 * nrow))
    gs = gridspec.GridSpec(nrow, ncol, hspace=0.45, wspace=0.18)
    n_weeks = len(data[sorted_cs[0]])
    weeks = np.arange(n_weeks)

    for i, c in enumerate(sorted_cs):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        _, color = cluster_of(c)
        ax.plot(weeks, data[c], color=color, linewidth=1.0, alpha=0.85)
        if c == "KR":
            ax.set_facecolor("#fef3c7")
            for sp in ax.spines.values():
                sp.set_linewidth(2.5); sp.set_edgecolor("#dc2626")
        ax.set_title(f"{c} ({sources[c][:15]})", fontsize=10, fontweight="bold", color=color)
        ax.set_ylim(-2, 6)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.2)
        # Year ticks for 5yr cohort (53 weeks/yr)
        if n_weeks >= 200:
            ax.set_xticks(list(range(0, n_weeks, 53)))
            ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"EDA Time Series — TRUE ILI Cohort ({N} countries, z-score, {period_label})",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"[eda] {outpath.name} written")


def fig_sample_size(
    raw_arrs: dict[str, np.ndarray],
    period_full_weeks: int,
    period_label: str,
    outpath: Path,
):
    """Bar chart of n_valid weeks per country, KR highlight."""
    sorted_cs = sorted(raw_arrs.keys())
    ns = [int(np.isfinite(raw_arrs[c]).sum()) for c in sorted_cs]
    order = sorted(range(len(sorted_cs)), key=lambda i: -ns[i])
    cs_ord = [sorted_cs[i] for i in order]
    ns_ord = [ns[i] for i in order]
    colors = [cluster_of(c)[1] for c in cs_ord]

    fig, ax = plt.subplots(figsize=(max(14, len(cs_ord) * 0.5), 7))
    ax.bar(range(len(cs_ord)), ns_ord, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(cs_ord)))
    ax.set_xticklabels(cs_ord, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(f"Valid weeks ({period_label})", fontsize=12, fontweight="bold")
    ax.set_title(f"EDA Sample Size per Country ({len(cs_ord)} countries, full = {period_full_weeks} weeks)",
                 fontsize=13, fontweight="bold")
    ax.axhline(period_full_weeks, color="green", linestyle="--", linewidth=2, label=f"{period_full_weeks} = full")
    ax.axhline(100, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="100 = min threshold")
    ax.legend(fontsize=10, loc="upper right")
    for i, (c, n) in enumerate(zip(cs_ord, ns_ord)):
        col_t = "red" if c == "KR" else "black"
        wt = "bold" if c == "KR" else "normal"
        ax.text(i, n + 3, str(n), ha="center", fontsize=9, color=col_t, fontweight=wt)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"[eda] {outpath.name} written")


def fig_boxplot(data: dict[str, np.ndarray], period_label: str, outpath: Path):
    """Z-score distribution boxplot per country."""
    sorted_cs = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(max(15, len(sorted_cs) * 0.6), 8))
    box_data = [data[c] for c in sorted_cs]
    bp = ax.boxplot(box_data, tick_labels=sorted_cs, patch_artist=True, showfliers=True,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
    for patch, c in zip(bp["boxes"], sorted_cs):
        _, col = cluster_of(c)
        patch.set_facecolor(col); patch.set_alpha(0.7)
        if c == "KR":
            patch.set_edgecolor("#dc2626"); patch.set_linewidth(2.5)
    ax.set_xticklabels(sorted_cs, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Z-score of ILI rate", fontsize=12, fontweight="bold")
    ax.set_title(f"EDA Distribution per Country ({len(sorted_cs)} countries, z-score, {period_label})",
                 fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"[eda] {outpath.name} written")


def fig_source_table(
    sources: dict[str, str],
    raw_arrs: dict[str, np.ndarray],
    period_label: str,
    outpath: Path,
):
    """Source / unit comparison table figure."""
    sorted_cs = sorted(sources.keys())
    unit_map = {
        "KDCA_sentinel": "per 1000 외래",
        "ecdc_erviss": "per 100k",
        "cdc_ilinet": "weighted %",
        "delphi_national": "weighted %",
        "japan_jihs": "per clinic",
        "japan_jihs_hist": "per clinic",
        "influnet_it": "per 100k",
        "sentiweb_fr": "per 100k",
    }
    fig, ax = plt.subplots(figsize=(14, max(10, len(sorted_cs) * 0.35)))
    ax.axis("off")
    rows = []
    for c in sorted_cs:
        cl, _ = cluster_of(c)
        n = int(np.isfinite(raw_arrs[c]).sum())
        rows.append([c, cl, sources[c], unit_map.get(sources[c], "?"), n])
    table = ax.table(cellText=rows, colLabels=["Country", "Cluster", "Source", "Unit", "n_valid"],
                     loc="center", cellLoc="center",
                     colWidths=[0.10, 0.18, 0.22, 0.20, 0.12])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.6)
    for j in range(5):
        cell = table[(0, j)]; cell.set_facecolor("#1e3a8a"); cell.set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows):
        c = row[0]; _, col = cluster_of(c)
        for j in range(5):
            cell = table[(i+1, j)]
            if c == "KR":
                cell.set_facecolor("#fecaca"); cell.set_text_props(fontweight="bold", color="#7f1d1d")
            else:
                cell.set_facecolor(col); cell.set_alpha(0.3)
    ax.set_title(f"EDA Source & Unit Comparison ({len(sorted_cs)} countries, {period_label})",
                 fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"[eda] {outpath.name} written")


def generate_all_eda_figures(
    data: dict[str, np.ndarray],
    raw_arrs: dict[str, np.ndarray],
    sources: dict[str, str],
    output_dir: Path,
    cohort_name: str = "I-B",
    period_label: str = "2021-2025",
    period_full_weeks: int = 265,
    prefix: str = "eda",
) -> list[Path]:
    """Generate all 4 standard EDA figures for a cohort.

    Args:
        data: z-scored series dict.
        raw_arrs: raw (pre-zscore) series dict for sample size.
        sources: country → source name dict.
        output_dir: where to write PNGs.
        cohort_name: label for output filenames (e.g. "I-B").
        period_label: label for title (e.g. "2021-2025").
        period_full_weeks: full theoretical weeks (for sample size reference line).
        prefix: filename prefix.

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    p = output_dir / f"{prefix}_{cohort_name}_timeseries.png"
    fig_timeseries_grid(data, sources, period_label, p); files.append(p)
    p = output_dir / f"{prefix}_{cohort_name}_sample_size.png"
    fig_sample_size(raw_arrs, period_full_weeks, period_label, p); files.append(p)
    p = output_dir / f"{prefix}_{cohort_name}_boxplot.png"
    fig_boxplot(data, period_label, p); files.append(p)
    p = output_dir / f"{prefix}_{cohort_name}_source_table.png"
    fig_source_table(sources, raw_arrs, period_label, p); files.append(p)
    return files
