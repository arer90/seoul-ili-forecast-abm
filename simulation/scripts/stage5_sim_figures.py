"""Stage 5 — Simulation figures from 6 scenario runs.

Consumes ``simulation/results/sim_runs/*.npz`` + ``_manifest.json`` and
produces matplotlib + plotly figures under
``simulation/results/sim_runs/figures/``:

1. ``fig1_trajectories_grid.png``     — 2x3 (S/E/I/R/V/D) per scenario
2. ``fig2_I_overlay.png``             — city-wide I, all scenarios overlaid
3. ``fig3_gu_heatmap.png``            — 25-district peak-I heatmap (scenarios x gu)
4. ``fig4_scenario_comparison.png``   — peak day / attack rate / deaths bars
5. ``fig5_intervention_impact.png``   — Δ attack-rate / Δ peak vs baseline
6. ``fig_interactive_I.html``         — plotly overlay (hover-readable)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("stage5.figures")

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "simulation" / "results" / "sim_runs"
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCEN_ORDER = [
    "baseline",
    "npi_lockdown",
    "vaccination_campaign",
    "antiviral_prophylaxis",
    "combined_response",
    "sensitivity_strain_mismatch",
]
SCEN_SHORT = {
    "baseline": "Baseline",
    "npi_lockdown": "NPI lockdown",
    "vaccination_campaign": "Vaccination",
    "antiviral_prophylaxis": "Antiviral",
    "combined_response": "Combined",
    "sensitivity_strain_mismatch": "Strain mismatch",
}
COLORS = {
    "baseline": "#888888",
    "npi_lockdown": "#1f77b4",
    "vaccination_campaign": "#2ca02c",
    "antiviral_prophylaxis": "#d62728",
    "combined_response": "#9467bd",
    "sensitivity_strain_mismatch": "#ff7f0e",
}
COMPARTMENTS = ("S", "E", "I", "R", "V", "D")

# Korean gu → romanization (Revised Romanization, in SEOUL_GU_ORDERED insertion order)
GU_ROMAN = {
    "종로구": "Jongno", "중구": "Jung", "용산구": "Yongsan", "성동구": "Seongdong",
    "광진구": "Gwangjin", "동대문구": "Dongdaemun", "중랑구": "Jungnang",
    "성북구": "Seongbuk", "강북구": "Gangbuk", "도봉구": "Dobong",
    "노원구": "Nowon", "은평구": "Eunpyeong", "서대문구": "Seodaemun",
    "마포구": "Mapo", "양천구": "Yangcheon", "강서구": "Gangseo",
    "구로구": "Guro", "금천구": "Geumcheon", "영등포구": "Yeongdeungpo",
    "동작구": "Dongjak", "관악구": "Gwanak", "서초구": "Seocho",
    "강남구": "Gangnam", "송파구": "Songpa", "강동구": "Gangdong",
}


def _rom(name: str) -> str:
    """Romanize a Korean gu name; returns original if unmapped."""
    return GU_ROMAN.get(name, name)


def _load_all() -> tuple[dict, dict]:
    """Returns (runs, manifest). runs[name] = {state, incidence, days}."""
    manifest = json.load((RUN_DIR / "_manifest.json").open(encoding="utf-8"))
    runs = {}
    for n in SCEN_ORDER:
        npz = np.load(RUN_DIR / f"{n}.npz")
        runs[n] = {
            "state": npz["state"],       # (T, G, 6)
            "incidence": npz["incidence"],  # (T, G)
            "days": npz["days"],         # (T,)
        }
    return runs, manifest


def _city_totals(state: np.ndarray) -> dict:
    """Sum state across gu → per-compartment (T,) arrays."""
    return {c: state[:, :, i].sum(axis=1) for i, c in enumerate(COMPARTMENTS)}


# ── Figure 1: 2x3 trajectories grid ─────────────────────────────────────
def fig1_trajectories_grid(runs, manifest):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    for ax, name in zip(axes, SCEN_ORDER):
        r = runs[name]
        totals = _city_totals(r["state"])
        days = r["days"]
        ax.stackplot(
            days,
            totals["S"] / 1e6, totals["E"] / 1e6, totals["I"] / 1e6,
            totals["R"] / 1e6, totals["V"] / 1e6, totals["D"] / 1e6,
            labels=list(COMPARTMENTS),
            colors=["#c6dbef", "#fdd0a2", "#fdae6b", "#a1d99b", "#bcbddc", "#252525"],
            alpha=0.85,
        )
        ms = manifest["scenarios"][name]
        ax.axvline(ms["peak_day"], color="red", linestyle="--", alpha=0.5)
        ax.set_title(
            f"{SCEN_SHORT[name]} (peak d{ms['peak_day']}, "
            f"attack={ms['attack_rate_pct']:.1f}%)",
            fontsize=11,
        )
        ax.set_ylabel("Millions")
        if name == SCEN_ORDER[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    for ax in axes[3:]:
        ax.set_xlabel("Day")
    fig.suptitle(
        "Stage 5: Metapop SEIR-V-D Seoul 25-district — Compartment Trajectories "
        "(city-wide, 6 scenarios)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig1_trajectories_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


# ── Figure 2: city-wide I overlay ───────────────────────────────────────
def fig2_I_overlay(runs, manifest):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in SCEN_ORDER:
        r = runs[name]
        totals = _city_totals(r["state"])
        ax.plot(
            r["days"], totals["I"] / 1e3,
            label=SCEN_SHORT[name], color=COLORS[name], linewidth=2.0,
        )
    # Intervention windows annotated from baseline's siblings
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious I (thousands, city-wide)")
    ax.set_title(
        "Scenario comparison — infectious compartment I(t) "
        "(Seoul 9.88M, R0=1.4, 365-d horizon)"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig2_I_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


# ── Figure 3: 25-district heatmap ─────────────────────────────────────────────
def fig3_gu_heatmap(runs, manifest):
    """Rows = scenarios, cols = gu, value = peak I per gu."""
    gu_names = manifest["params"]["districts"]
    G = len(gu_names)
    M = np.zeros((len(SCEN_ORDER), G))
    for i, name in enumerate(SCEN_ORDER):
        state = runs[name]["state"]         # (T, G, 6)
        peak_I_gu = state[:, :, 2].max(axis=0)  # I is index 2
        M[i] = peak_I_gu / 1e3  # thousands

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        M,
        xticklabels=[_rom(g) for g in gu_names],
        yticklabels=[SCEN_SHORT[n] for n in SCEN_ORDER],
        annot=True, fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Peak I (thousands)"},
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title("Peak infectious population per gu × scenario (thousands)")
    ax.set_xlabel("Gu")
    ax.set_ylabel("Scenario")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out = FIG_DIR / "fig3_gu_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


# ── Figure 4: scenario comparison bars ──────────────────────────────────
def fig4_scenario_comparison(runs, manifest):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scen = SCEN_ORDER
    peaks = [manifest["scenarios"][n]["peak_day"] for n in scen]
    attacks = [manifest["scenarios"][n]["attack_rate_pct"] for n in scen]
    deaths = [manifest["scenarios"][n]["cumulative_deaths"] for n in scen]
    colors = [COLORS[n] for n in scen]
    labels = [SCEN_SHORT[n] for n in scen]

    axes[0].barh(labels, peaks, color=colors)
    axes[0].set_xlabel("Peak day")
    axes[0].set_title("Peak timing")
    axes[0].axvline(manifest["scenarios"]["baseline"]["peak_day"],
                    color="k", linestyle=":", alpha=0.5, label="baseline")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].barh(labels, attacks, color=colors)
    axes[1].set_xlabel("Attack rate (% of pop)")
    axes[1].set_title("Total attack rate")
    axes[1].axvline(manifest["scenarios"]["baseline"]["attack_rate_pct"],
                    color="k", linestyle=":", alpha=0.5)

    axes[2].barh(labels, deaths, color=colors)
    axes[2].set_xlabel("Cumulative deaths")
    axes[2].set_title("Mortality (IFR=0.001)")
    axes[2].axvline(manifest["scenarios"]["baseline"]["cumulative_deaths"],
                    color="k", linestyle=":", alpha=0.5)

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    fig.suptitle("Scenario KPIs — Stage 5 Seoul metapop SEIR-V-D", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig4_scenario_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


# ── Figure 5: Δ-from-baseline effect sizes ──────────────────────────────
def fig5_intervention_impact(runs, manifest):
    base = manifest["scenarios"]["baseline"]
    fig, ax = plt.subplots(figsize=(11, 5))
    non_base = [n for n in SCEN_ORDER if n != "baseline"]
    delay = [manifest["scenarios"][n]["peak_day"] - base["peak_day"] for n in non_base]
    d_attack = [
        base["attack_rate_pct"] - manifest["scenarios"][n]["attack_rate_pct"]
        for n in non_base
    ]
    peak_ratio = [
        manifest["scenarios"][n]["peak_I_city"] / base["peak_I_city"] * 100
        for n in non_base
    ]

    x = np.arange(len(non_base))
    w = 0.25
    ax.bar(x - w, delay, width=w, color="#1f77b4", label="Peak delay (days)")
    ax.bar(x, d_attack, width=w, color="#2ca02c", label="Attack-rate reduction (pp)")
    ax.bar(x + w, [p - 100 for p in peak_ratio], width=w, color="#d62728",
           label="Peak-I change vs baseline (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_SHORT[n] for n in non_base], rotation=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Intervention impact relative to baseline (Seoul metapop SEIR-V-D)")
    ax.set_ylabel("Δ vs baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig5_intervention_impact.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


# ── Optional: interactive plotly overlay ────────────────────────────────
def fig_interactive_I(runs, manifest):
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.info("plotly not installed — skipping interactive figure")
        return
    fig = go.Figure()
    for name in SCEN_ORDER:
        r = runs[name]
        totals = _city_totals(r["state"])
        fig.add_trace(go.Scatter(
            x=r["days"], y=totals["I"] / 1e3,
            mode="lines", name=SCEN_SHORT[name],
            line=dict(color=COLORS[name], width=2.5),
            hovertemplate="%{fullData.name}<br>day %{x}<br>I = %{y:.0f}k<extra></extra>",
        ))
    fig.update_layout(
        title="Infectious I(t) — 6 scenarios, Seoul 25-district metapop",
        xaxis_title="Day",
        yaxis_title="Infectious I (thousands)",
        template="plotly_white",
        hovermode="x unified",
        height=550,
    )
    out = FIG_DIR / "fig_interactive_I.html"
    fig.write_html(str(out))
    log.info("wrote %s", out)


def main() -> None:
    log.info("loading %d scenarios from %s", len(SCEN_ORDER), RUN_DIR)
    runs, manifest = _load_all()
    fig1_trajectories_grid(runs, manifest)
    fig2_I_overlay(runs, manifest)
    fig3_gu_heatmap(runs, manifest)
    fig4_scenario_comparison(runs, manifest)
    fig5_intervention_impact(runs, manifest)
    fig_interactive_I(runs, manifest)
    log.info("all figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
