"""Thesis §4.16 / §4.18 figure generator.

Generates three PNG panels from simulation/results/abm_scenarios_v1/trajectories.npz:

  fig_abm_s1s6.png        — city-wide I trajectory per scenario
  fig_abm_per_gu_heatmap.png — per-gu peak-week heatmap, scenarios x districts
  fig_abm_spatial_wave.png — spatial wave propagation per scenario

Everything is deterministic and depends only on numpy + matplotlib.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("abm_figures")


def main() -> int:
    if plt is None:
        log.error("matplotlib not installed")
        return 2

    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    _abm = str(get_results_dir() / "abm_scenarios_v1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefact-dir", default=_abm)
    ap.add_argument("--out-dir", default=_abm)
    args = ap.parse_args()
    art = Path(args.artefact_dir) / "trajectories.npz"
    if not art.exists():
        log.error("missing %s — run _run_scenarios.py first", art)
        return 3

    d = np.load(art, allow_pickle=True)
    scens = [str(s) for s in d["scenarios"]]
    districts = [str(x) for x in d["districts"]]
    I_tensor = d["per_gu_I"]   # (S, T+1, G)
    S, Tp1, G = I_tensor.shape
    log.info("loaded tensor shape=%s  scens=%d  G=%d", I_tensor.shape, S, G)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fig 1 — city-wide trajectory per scenario
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    x = np.arange(Tp1)
    for i, sid in enumerate(scens):
        city_I = I_tensor[i].sum(axis=1)
        ax.plot(x, city_I, label=sid, lw=1.5)
    ax.set_xlabel("Day")
    ax.set_ylabel("City-wide infectious count (I)")
    ax.set_title("S1-S6 city-wide infectious trajectory")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_abm_s1s6.png")
    plt.close(fig)
    log.info("wrote %s", out / "fig_abm_s1s6.png")

    # ------------------------------------------------------------------
    # Fig 2 — per-gu peak-week heatmap (scenarios x districts)
    # ------------------------------------------------------------------
    peak_week = np.argmax(I_tensor, axis=1) // 7    # (S, G)
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=130)
    im = ax.imshow(peak_week, aspect="auto", cmap="viridis")
    ax.set_yticks(range(S), scens)
    # show every 2nd district tick to avoid crowding for G=25
    step = max(1, G // 20)
    tick_idx = list(range(0, G, step))
    ax.set_xticks(tick_idx, [districts[i] for i in tick_idx], rotation=75, fontsize=7)
    ax.set_title("Peak week per district (darker = earlier peak)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Peak week (0 = day 0-6)")
    fig.tight_layout()
    fig.savefig(out / "fig_abm_per_gu_heatmap.png")
    plt.close(fig)
    log.info("wrote %s", out / "fig_abm_per_gu_heatmap.png")

    # ------------------------------------------------------------------
    # Fig 3 — spatial wave propagation: districts ordered by S1 peak day
    # ------------------------------------------------------------------
    order = np.argsort(peak_week[0])    # S1 peak-day ascending
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=130, sharex=True, sharey=True)
    for i, (sid, ax) in enumerate(zip(scens, axes.ravel())):
        for rank, gu_idx in enumerate(order):
            traj = I_tensor[i, :, gu_idx]
            ax.plot(traj, color=plt.cm.plasma(rank / max(G - 1, 1)),
                    lw=0.7, alpha=0.8)
        ax.set_title(sid, fontsize=10)
        ax.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("I per district")
    axes[1, 0].set_ylabel("I per district")
    for ax in axes[-1, :]:
        ax.set_xlabel("Day")
    fig.suptitle("Spatial wave — districts ordered by S1 peak arrival (dark = early)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_abm_spatial_wave.png")
    plt.close(fig)
    log.info("wrote %s", out / "fig_abm_spatial_wave.png")

    # ------------------------------------------------------------------
    # Fig 4 — attack rate per district, scenario overlay
    # ------------------------------------------------------------------
    # attack rate approx = (peak I / N) scaled; simpler proxy: integral / N
    I_tensor[0, 0].sum(axis=-1) if False else None
    # Use total infections proxy = integral of I over horizon / horizon
    attack_proxy = I_tensor.sum(axis=1) / Tp1       # (S, G)
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=130)
    width = 0.13
    x = np.arange(G)
    for i, sid in enumerate(scens):
        ax.bar(x + i * width, attack_proxy[i], width=width, label=sid)
    ax.set_xticks(x + width * (S - 1) / 2, districts, rotation=75, fontsize=7)
    ax.set_ylabel("Mean I per day (integral / horizon)")
    ax.set_title("Per-district infection burden by scenario")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_abm_attack_rate.png")
    plt.close(fig)
    log.info("wrote %s", out / "fig_abm_attack_rate.png")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
