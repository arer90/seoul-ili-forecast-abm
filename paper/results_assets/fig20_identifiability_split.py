"""fig20_identifiability_split.py — Figure 20 split + enlarged (4-param identifiability).

PURPOSE
    The thesis Figure 20 ("Practical identifiability of the four behavioural
    parameters") packed four profile-likelihood sub-plots PLUS a posterior
    correlation heatmap into one small image, so the profile curves were tiny
    and the heatmap labels hard to read. This regenerator SPLITS the same
    real-data SSOT into two enlarged figures:

      Figure 28.1  Profile likelihoods (Raue 2009) for alpha, kappa, tau, theta -
                   2x2 enlarged; the min+threshold line and per-parameter
                   identifiable/sloppy verdict are kept.
      Figure 28.2  ABC-SMC posterior pairwise correlation heatmap (Gutenkunst
                   2007 sloppiness) - enlarged, annotated cells.

DATA SSOT (read-only, measured — NO re-running of the expensive ABC-SMC):
    simulation/results/abm_identifiability/profile_likelihood.csv  (profile curves)
    simulation/results/abm_identifiability/posterior_corr.csv      (posterior r)
    simulation/results/abm_identifiability/verdict.json            (thresholds, status)

This mirrors the layout/labels of the original source figure
``simulation/scripts/abm_identifiability_4param.py:make_figure`` exactly; only
the split + enlarge differs. No values are altered.

Output (PNG, white bg, dpi=160):
    paper/results_assets/fig20_1_profile_likelihoods.png
    paper/results_assets/fig20_2_posterior_correlation.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
DATA_DIR = PROJECT_ROOT / "simulation" / "results" / "abm_identifiability"
PROFILE_CSV = DATA_DIR / "profile_likelihood.csv"
CORR_CSV = DATA_DIR / "posterior_corr.csv"
VERDICT_JSON = DATA_DIR / "verdict.json"

OUT_DIR = _THIS.parent
OUT_1 = OUT_DIR / "fig20_1_profile_likelihoods.png"
OUT_2 = OUT_DIR / "fig20_2_posterior_correlation.png"

PARAM_NAMES = ("alpha", "kappa", "tau", "theta")
PARAM_LABEL = {
    "alpha": "alpha (risk perception)",
    "kappa": "kappa (fatigue weight)",
    "tau": "tau (fatigue time-constant, d)",
    "theta": "theta (compliance threshold)",
}
STATUS_COLOR = {"identifiable": "#2ca02c", "sloppy": "#d62728",
                "weakly_identifiable": "#ff7f0e"}


def _font() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _load_profile() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """param -> (values, profile_loss) arrays from profile_likelihood.csv."""
    by: dict[str, list[tuple[float, float]]] = {p: [] for p in PARAM_NAMES}
    with PROFILE_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = r["param"]
            if p in by:
                by[p].append((float(r["value"]), float(r["profile_loss"])))
    out = {}
    for p, rows in by.items():
        rows.sort()
        out[p] = (np.array([v for v, _ in rows]), np.array([l for _, l in rows]))
    return out


def _load_corr() -> tuple[list[str], np.ndarray]:
    """names + correlation matrix from posterior_corr.csv (first block only)."""
    names: list[str] = []
    mat: list[list[float]] = []
    with CORR_CSV.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                break  # blank line = end of correlation block (CSV has a 2nd table)
            parts = line.split(",")
            if parts[0] == "":  # header row
                continue
            names.append(parts[0])
            mat.append([float(x) for x in parts[1:1 + len(PARAM_NAMES)]])
    return names, np.asarray(mat)


def make_profiles(profiles: dict, verdict: dict) -> None:
    """Figure 28.1 — 2x2 enlarged profile-likelihood panels."""
    thr_abs = verdict["thresholds"]["profile_loss_threshold_abs"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    axes = axes.ravel()
    for ax, p in zip(axes, PARAM_NAMES):
        vals, losses = profiles[p]
        fin = np.isfinite(losses)
        ax.plot(vals[fin], losses[fin], "o-", color="#1f77b4", markersize=7, lw=2.2)
        if fin.any():
            lo = losses[fin].min()
            ax.axhline(lo + thr_abs, color="#d62728", ls="--", lw=1.6,
                       label=f"min + {thr_abs:.2f} (identifiability threshold)")
            ax.axhline(lo, color="#2ca02c", ls=":", lw=1.2, label="profile minimum")
        st = verdict["per_param"][p]["status"]
        frac = verdict["per_param"][p]["profile_interval_frac"]
        col = STATUS_COLOR.get(st, "#333333")
        ax.set_title(f"{PARAM_LABEL[p]}\nprofile interval = {frac:.0%}  [{st}]",
                     fontsize=14, color=col, fontweight="bold")
        ax.set_xlabel(p, fontsize=13)
        ax.set_ylabel("profile loss (min over 3 nuisance params)", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10.5, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Figure 28.1  ABM behavioural 4-parameter profile likelihoods (real Seoul ILI shape, 2023-2024)\n"
        "Raue (2009): a sharp valley => practically identifiable; a flat profile => sloppy / non-identifiable",
        fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_1, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_1}")


def make_corr(names: list[str], corr: np.ndarray, verdict: dict) -> None:
    """Figure 28.2 — enlarged posterior correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=13, rotation=20)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=13)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=15,
                    color="white" if abs(corr[i, j]) > 0.55 else "black")
    corr_strong = verdict["thresholds"].get("corr_strong", 0.6)
    ax.set_title(
        "Figure 28.2  ABC-SMC posterior pairwise correlation\n"
        f"(|r| -> 1 = sloppy / degenerate direction; |r| >= {corr_strong} flagged as strong)",
        fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(OUT_2, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_2}")


def main() -> int:
    _font()
    verdict = json.loads(VERDICT_JSON.read_text(encoding="utf-8"))
    profiles = _load_profile()
    names, corr = _load_corr()
    make_profiles(profiles, verdict)
    make_corr(names, corr, verdict)
    # honest verdict echo
    for p in PARAM_NAMES:
        d = verdict["per_param"][p]
        print(f"  {p:6s} status={d['status']:20s} profile_frac={d['profile_interval_frac']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
