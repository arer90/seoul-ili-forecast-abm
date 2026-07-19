#!/usr/bin/env python3
"""Figure Q.1 — critical-difference (Nemenyi) diagram over the 68 rolling origins.

The figure was in the thesis with no generator anywhere in the repo: a grep for
Nemenyi / Friedman / critical-difference across every ``.py`` returns nothing but textbook
citations. It had been produced once, by hand, and its numbers went stale the moment any
model's predictions changed — which is exactly what happened when ``NegBinGLM`` and
``PoissonAutoreg`` were reimplemented as the log-link GLMs their names claim.

Everything the figure shows is recoverable, so this rebuilds it from the committed
predictions rather than leaving an orphan image in the manuscript:

  * per origin (each of the 68 sealed-test weeks) rank all models by absolute error;
  * mean rank per model over the origins is the plotted statistic;
  * Friedman tests whether the rank distributions differ at all;
  * Nemenyi's critical difference CD = q_alpha * sqrt(k(k+1) / (6N)) marks the band inside
    which no model is separable from the best (Demsar 2006, JMLR 7:1-30).

Only models with a full 68-week prediction vector take part — a model ranked on a subset
would be ranked against a different field, which is not a comparison.

Run:
    .venv/bin/python -m simulation.scripts.fig_nemenyi_cd
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_PRED = _ROOT / "simulation" / "results" / "csv"
_METRICS = _ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"
_OUT = _ROOT / "simulation" / "results" / "figures" / "fig_nemenyi_cd.png"

TOP_N = 15
ALPHA = 0.05
# The original figure blued two models as joint title-holders. That framing is gone — the thesis
# names a single champion — so only FusedEpi is emphasised.
HIGHLIGHT = ("FusedEpi",)

# Studentised-range q_0.05 / sqrt(2) for k models (Demsar 2006, Table 5), extended by the
# normal approximation past the tabulated range.
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164}


def _q_alpha(k: int) -> float:
    if k in _Q05:
        return _Q05[k]
    # Nemenyi's q is the studentised range at infinite df, divided by sqrt(2).
    return float(stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2.0))


def _load() -> tuple[list[str], np.ndarray]:
    """Return (models, |error| matrix of shape (n_models, n_origins)) for the full-length set."""
    # The benchmark is the committed 48-model lineup. Prediction CSVs also exist for models
    # that are not in it (retired ensembles); ranking against a different field would not be
    # the comparison the thesis reports.
    lineup = {r["model"] for r in csv.DictReader(_METRICS.open(encoding="utf-8"))}

    errors: dict[str, np.ndarray] = {}
    for path in sorted(_PRED.glob("predictions_*.csv")):
        model = path.stem.removeprefix("predictions_")
        if model not in lineup:
            continue
        rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
                if r.get("split", "test") == "test"]
        if not rows:
            continue
        try:
            y = np.array([float(r["y_true"]) for r in rows])
            p = np.array([float(r["y_pred"]) for r in rows])
        except (KeyError, ValueError):
            continue
        e = np.abs(y - p)
        if np.isfinite(e).all():
            errors[model] = e

    if not errors:
        print("no usable prediction CSVs", file=sys.stderr)
        return [], np.empty((0, 0))

    n = max(len(e) for e in errors.values())
    full = {m: e for m, e in errors.items() if len(e) == n}
    dropped = sorted(set(errors) - set(full))
    if dropped:
        print(f"  excluded (prediction vector != {n} origins): {dropped}")

    models = sorted(full)
    return models, np.vstack([full[m] for m in models])


def main() -> int:
    models, err = _load()
    if not models:
        return 1
    k, n_origins = err.shape

    # rank per origin: 1 = smallest absolute error; ties share the average rank
    ranks = np.apply_along_axis(stats.rankdata, 0, err)
    mean_rank = ranks.mean(axis=1)

    chi2, p_friedman = stats.friedmanchisquare(*[err[i] for i in range(k)])
    cd = _q_alpha(k) * np.sqrt(k * (k + 1) / (6.0 * n_origins))

    order = np.argsort(mean_rank)[:TOP_N]
    names = [models[i] for i in order]
    vals = mean_rank[order]
    best = float(mean_rank.min())

    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    ax.axvspan(best, best + cd, color="#dbeafe", zorder=0)
    ax.axvline(best, color="#3b82f6", ls="--", lw=1.2, zorder=1)

    ys = np.arange(len(names))[::-1]
    for y, name, v in zip(ys, names, vals):
        hi = name in HIGHLIGHT
        ax.plot(v, y, "o", ms=11, color="#2563eb" if hi else "#4b5563", zorder=3)
        ax.text(v + 0.10, y, f"{v:.1f}", va="center", fontsize=10, color="#6b7280")

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=11)
    ax.tick_params(axis="y", length=0)            # the original has no y tick marks
    for tick, name in zip(ax.get_yticklabels(), names):
        if name in HIGHLIGHT:
            tick.set_color("#2563eb")
            tick.set_fontweight("bold")

    hi_x = best + cd
    ax.set_xlim(best - 0.5, hi_x + 2.0)
    ax.set_ylim(-0.8, len(names) - 0.2)

    exp = int(np.floor(np.log10(p_friedman))) if p_friedman > 0 else -100
    p_txt = f"$p < 10^{{{exp}}}$" if p_friedman > 0 else "$p < 10^{-100}$"
    ax.set_title(
        f"Critical-difference (Nemenyi) diagram — {k}-model benchmark, Friedman {p_txt}\n"
        f"Top {len(names)} shown; the top point-accuracy cluster is statistically tied",
        fontsize=13, pad=14,
    )
    # sit the caption inside the band at the right, clear of the two title lines
    ax.text(hi_x - 0.25, len(names) - 1.1,
            f"within CD of the best (CD={cd:.1f}, α={ALPHA}):\nnot significantly different",
            fontsize=10, color="#3b82f6", ha="right", va="top")
    ax.set_xlabel(f"Mean rank across {n_origins} rolling origins "
                  f"(per-origin absolute error; lower = better)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"  models={k}  origins={n_origins}  Friedman chi2={chi2:.1f} p={p_friedman:.3g}  "
          f"CD={cd:.2f}")
    for name, v in zip(names, vals):
        mark = " ←" if name in HIGHLIGHT else ""
        print(f"    {name:<22} {v:5.1f}{mark}")
    print(f"[ok] {_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
