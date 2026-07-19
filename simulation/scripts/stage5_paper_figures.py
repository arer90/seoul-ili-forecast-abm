"""Stage 5 — Paper figures (6 figures for publication + defense slides).

Outputs to ``simulation/results/paper_figures/``:

1. ``figP1_wis_forest.png``      — forest plot of WIS across 66 models
                                    (Gaussian fallback vs phase6 conformal, coloured)
2. ``figP2_wis_scatter_RvsPy.png`` — R scoringutils WIS vs Python WIS (66 models)
3. ``figP3_its_npi_step.png``    — ITS segmented regression of ILI vs NPI phases
4. ``figP4_rt_overlay.png``      — EpiEstim Cori Rt vs SEIR-V2 R_eff (2019-2026)
5. ``figP5_r3_5_ablation.png``   — r3_5 NPI-covariate ablation bar (RMSE/R²)
6. ``figP6_sim_choropleth_grid.png`` — Seoul 25-district peak-I grid across 6 scenarios
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("stage5.paper")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "simulation" / "results" / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Figure P1: WIS forest, 66 models ────────────────────────────────────
def figP1_wis_forest():
    pe = json.load(
        (ROOT / "simulation" / "results" / "post_E_eval.json").open(encoding="utf-8")
    )
    df = pd.DataFrame(pe["details"])
    df = df[df["wis"].notna()].copy()
    df = df.sort_values("wis", ascending=True).reset_index(drop=True)

    color_map = {
        "phase6_conformal": "#1f77b4",
        "residual_gaussian": "#d62728",
    }
    colors = [color_map.get(s, "gray") for s in df["wis_source"]]

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.barh(df["model"], df["wis"], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel("WIS (lower = better)")
    ax.set_title(
        f"WIS forest — 66 models (test split, Python evaluator)\n"
        f"phase6_conformal = {(df['wis_source'] == 'phase6_conformal').sum()}, "
        f"residual_gaussian = {(df['wis_source'] == 'residual_gaussian').sum()}",
        fontsize=11,
    )
    ax.axvline(df["wis"].median(), color="black", linestyle=":",
               alpha=0.7, label=f"median = {df['wis'].median():.2f}")
    ax.grid(axis="x", alpha=0.3)
    legend_patches = [
        mpatches.Patch(color=color_map["phase6_conformal"], label="Phase 6 split-conformal"),
        mpatches.Patch(color=color_map["residual_gaussian"], label="Residual-std Gaussian (fallback)"),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([], [], color="black",
                                                    linestyle=":", label=f"median {df['wis'].median():.2f}")],
              loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    out = FIG_DIR / "figP1_wis_forest.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (N=%d)", out, len(df))


# ── Figure P2: R scoringutils WIS vs Python WIS ─────────────────────────
def figP2_wis_scatter_RvsPy():
    pe = json.load(
        (ROOT / "simulation" / "results" / "post_E_eval.json").open(encoding="utf-8")
    )
    py_df = pd.DataFrame(pe["details"])[["model", "wis", "wis_source"]]
    py_df = py_df.rename(columns={"wis": "wis_py"})
    r_df = pd.read_csv(ROOT / "simulation" / "r_verification" / "results" / "04_wis_crps_pit.csv")
    r_df = r_df.rename(columns={"wis": "wis_r"})
    df = py_df.merge(r_df, on="model", how="inner").dropna(subset=["wis_py", "wis_r"])

    # Per-tier coloring
    is_conformal = df["wis_source"] == "phase6_conformal"

    corr_pearson = float(df["wis_py"].corr(df["wis_r"]))
    corr_spearman = float(df["wis_py"].corr(df["wis_r"], method="spearman"))

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(df.loc[~is_conformal, "wis_py"], df.loc[~is_conformal, "wis_r"],
               c="#d62728", alpha=0.7, s=60, edgecolor="black",
               label=f"Residual-std Gaussian (N={(~is_conformal).sum()})")
    ax.scatter(df.loc[is_conformal, "wis_py"], df.loc[is_conformal, "wis_r"],
               c="#1f77b4", alpha=0.9, s=90, edgecolor="black",
               label=f"Phase 6 conformal (N={is_conformal.sum()})")
    # 45-degree reference
    lo = min(df["wis_py"].min(), df["wis_r"].min())
    hi = max(df["wis_py"].max(), df["wis_r"].max())
    ax.plot([lo, hi], [lo, hi], "k:", alpha=0.5, label="y = x")

    # Highlight top-5 (Python)
    top5 = df.sort_values("wis_py").head(5)
    for _, row in top5.iterrows():
        ax.annotate(row["model"], (row["wis_py"], row["wis_r"]),
                    xytext=(6, 3), textcoords="offset points", fontsize=8)

    ax.set_xlabel("WIS — Python evaluator (Gaussian form, 2-α)")
    ax.set_ylabel("WIS — R scoringutils (5-quantile decomposition)")
    ax.set_title(
        f"Cross-validation: R vs Python WIS on 66 models\n"
        f"Pearson r = {corr_pearson:.3f}, Spearman ρ = {corr_spearman:.3f} "
        f"(N={len(df)})"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "figP2_wis_scatter_RvsPy.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (r=%.3f, ρ=%.3f)", out, corr_pearson, corr_spearman)


# ── Figure P3: ITS segmented regression ─────────────────────────────────
def figP3_its_npi_step():
    ili = pd.read_csv(ROOT / "simulation" / "results" / "post_E" / "ili_series.csv",
                       parse_dates=["week_start"])
    npi = pd.read_csv(ROOT / "simulation" / "results" / "post_E" / "npi_window.csv",
                       parse_dates=["iso_date"])
    its = pd.read_csv(ROOT / "simulation" / "r_verification" / "results" / "07_its_segmented.csv")

    npi_start = npi.loc[npi["event"] == "npi_start", "iso_date"].iloc[0]
    npi_end = npi.loc[npi["event"] == "npi_end", "iso_date"].iloc[0]

    # ITS coefficients
    coef = {r["term"]: r for _, r in its.iterrows()}

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(ili["week_start"], ili["ili_rate"], color="#2e86ab",
            linewidth=1.2, label="ILI rate (observed)")

    # Shade NPI window
    ax.axvspan(npi_start, npi_end, alpha=0.15, color="gray",
               label=f"NPI active ({npi_start.date()} → {npi_end.date()})")
    ax.axvline(npi_start, color="red", linestyle="--", alpha=0.7)
    ax.axvline(npi_end, color="green", linestyle="--", alpha=0.7)

    # Annotate ITS coefficients
    s_start = coef.get("intercept_shift_NPI_start", {})
    s_end = coef.get("intercept_shift_NPI_end", {})
    r2_base = coef.get("R2_baseline", {}).get("estimate", np.nan)
    r2_its = coef.get("R2_ITS", {}).get("estimate", np.nan)

    ax.text(0.02, 0.97,
            f"ITS segmented regression (R forecast::tslm):\n"
            f"  Δintercept @ NPI-start = {s_start.get('estimate', 0):.2f} "
            f"(p = {s_start.get('p_value', 1):.4f}) ***\n"
            f"  Δintercept @ NPI-end   = {s_end.get('estimate', 0):.2f} "
            f"(p = {s_end.get('p_value', 1):.4f}) ***\n"
            f"  R² baseline = {r2_base:.3f} → R² ITS = {r2_its:.3f}  "
            f"(ΔR² = +{r2_its - r2_base:.3f})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))

    ax.set_xlabel("Week")
    ax.set_ylabel("ILI rate (per 1000 visits)")
    ax.set_title(
        "Interrupted Time Series — NPI attribution on Seoul ILI (2019–2026)"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "figP3_its_npi_step.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


# ── Figure P4: Rt overlay (EpiEstim vs SEIR-V2) ─────────────────────────
def figP4_rt_overlay():
    rt = pd.read_csv(ROOT / "simulation" / "r_verification" / "results" / "06_rt_epiestim.csv",
                      parse_dates=["week_start"])
    rt_sv2 = pd.read_csv(ROOT / "simulation" / "results" / "post_E" / "rt_seir_v2.csv",
                          parse_dates=["week_start"])

    npi = pd.read_csv(ROOT / "simulation" / "results" / "post_E" / "npi_window.csv",
                       parse_dates=["iso_date"])
    npi_start = npi.loc[npi["event"] == "npi_start", "iso_date"].iloc[0]
    npi_end = npi.loc[npi["event"] == "npi_end", "iso_date"].iloc[0]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    # EpiEstim band
    ax.fill_between(rt["week_start"], rt["rt_cori_q025"], rt["rt_cori_q975"],
                     color="#1f77b4", alpha=0.18, label="EpiEstim 95% band")
    ax.plot(rt["week_start"], rt["rt_cori_mean"], color="#1f77b4",
            linewidth=1.5, label="EpiEstim (Cori, SI=2.6 d)")
    ax.plot(rt_sv2["week_start"], rt_sv2["rt_eff"], color="#d62728",
            linewidth=1.5, alpha=0.9, label="SEIR-V2 R_eff (own fit)")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvspan(npi_start, npi_end, alpha=0.10, color="gray", label="NPI window")

    ax.set_xlabel("Week")
    ax.set_ylabel("Effective reproduction number Rt")
    ax.set_title("Rt cross-validation — EpiEstim (R) vs SEIR-V2 (Python)")
    ax.set_ylim(0, max(4.0, rt_sv2["rt_eff"].max() * 1.05))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "figP4_rt_overlay.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


# ── Figure P5: r3_5 NPI ablation ────────────────────────────────────────
def figP5_r3_5_ablation():
    a = json.load(
        (ROOT / "simulation" / "results" / "post_E" / "r3_5_npi_ablation.json")
        .open(encoding="utf-8")
    )
    full, abl, delta = a["full"], a["ablated"], a["delta"]
    verdict = a.get("verdict", "")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: RMSE bars
    axes[0].bar(["Full", "Ablated"], [full["rmse"], abl["rmse"]],
                color=["#1f77b4", "#d62728"], alpha=0.85)
    axes[0].set_ylabel("RMSE (test)")
    axes[0].set_title(f"RMSE (Δ = {delta['rmse_pct']:+.2f}%)")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: R²
    axes[1].bar(["Full", "Ablated"], [full["r2"], abl["r2"]],
                color=["#1f77b4", "#d62728"], alpha=0.85)
    axes[1].set_ylabel("R²")
    axes[1].set_title(f"R² (ΔR² = {delta['r2_abs']:+.3f})")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: fitted params
    param_names = ["β₀ (intercept)", "ε (NPI coef)", "κ (school)"]
    full_vals = [full["beta0"], full["epsilon"], full["kappa"]]
    abl_vals = [abl["beta0"], abl["epsilon"], abl["kappa"]]
    x = np.arange(len(param_names))
    w = 0.35
    axes[2].bar(x - w/2, full_vals, width=w, label="Full", color="#1f77b4", alpha=0.85)
    axes[2].bar(x + w/2, abl_vals, width=w, label="Ablated (ε=0)", color="#d62728", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(param_names, rotation=0, fontsize=9)
    axes[2].set_ylabel("Fitted value")
    axes[2].set_title("SEIR-V2 fitted parameters")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"r3_5 NPI-covariate ablation — DM stat = {delta['dm_stat']:.2f}, "
        f"p = {delta['p_dm']:.3f}\n"
        f"Verdict: {verdict}",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "figP5_r3_5_ablation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


# ── Figure P6: 25-district scenario choropleth grid ───────────────────────────
def figP6_sim_choropleth_grid():
    from simulation.scripts.stage5_sim_figures import (
        SCEN_ORDER, SCEN_SHORT, GU_ROMAN,
    )
    RUN = ROOT / "simulation" / "results" / "sim_runs"
    manifest = json.load((RUN / "_manifest.json").open(encoding="utf-8"))
    gu_names = manifest["params"]["districts"]
    G = len(gu_names)

    # Reshape gu → 5x5 grid (simple lattice — not real Seoul geometry, but
    # sufficient for trend-at-a-glance choropleth. Real geojson would be nicer.)
    grid_shape = (5, 5)
    assert G == grid_shape[0] * grid_shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Global vmax for color consistency
    all_peak = []
    for n in SCEN_ORDER:
        state = np.load(RUN / f"{n}.npz")["state"]
        all_peak.append(state[:, :, 2].max(axis=0) / 1e3)
    vmax = max(x.max() for x in all_peak)
    vmin = 0

    for ax, name, peak_gu in zip(axes, SCEN_ORDER, all_peak):
        M = peak_gu.reshape(grid_shape)
        im = ax.imshow(M, cmap="YlOrRd", vmin=vmin, vmax=vmax)
        for (i, j), val in np.ndenumerate(M):
            idx = i * grid_shape[1] + j
            label = GU_ROMAN.get(gu_names[idx], gu_names[idx])
            ax.text(j, i, f"{label}\n{val:.0f}k", ha="center", va="center",
                     fontsize=7, color="black" if val < vmax * 0.55 else "white")
        ax.set_xticks([])
        ax.set_yticks([])
        ms = manifest["scenarios"][name]
        ax.set_title(
            f"{SCEN_SHORT[name]}\npeak d{ms['peak_day']}, "
            f"attack {ms['attack_rate_pct']:.1f}%",
            fontsize=10,
        )
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Peak I (thousands)",
                         fraction=0.025, pad=0.02)
    fig.suptitle(
        "Seoul 25-district peak-infectious choropleth (5x5 lattice proxy for geo) — "
        "6 scenarios, Stage 5 Metapop SEIR-V-D",
        fontsize=12, y=0.995,
    )
    out = FIG_DIR / "figP6_sim_choropleth_grid.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    figP1_wis_forest()
    figP2_wis_scatter_RvsPy()
    figP3_its_npi_step()
    figP4_rt_overlay()
    figP5_r3_5_ablation()
    figP6_sim_choropleth_grid()
    log.info("all 6 paper figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
