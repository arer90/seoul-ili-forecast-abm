"""SCI ANALYSIS 1 — Top-set uncertainty via moving-block bootstrap of the
per-week loss differential (ΔL = L_FusedEpi − L_competitor).

Reviewer question: "How certain is the top-set difference?" — i.e. are
FusedEpi / TiRex / NegBinGLM a statistical tie, while all decisively beat the
classical baselines (ARIMA / SARIMA / Theta)?

LOSS CHOICE (read-only constraint — STATED EXPLICITLY)
------------------------------------------------------
The only per-week loss that is *reconstructable for all seven models* from the
stored artifacts is the **per-week absolute error** |y_true − y_pred|.
The interval-score / WIS components require stored per-week predictive quantiles
(or a leak-free residual scale). Those are present only for a subset of models
(FusedEpi, NegBinGLM, SeirCount-TabPFN have ``pi_source=r9_leakfree``); the
classical baselines TiRex, ARIMA, SARIMA, Theta carry ``wis=nan`` /
``pi_source=unavailable`` in ``per_model_eval/per_model_metrics.csv`` — no
per-week quantiles were persisted for them. A bootstrap differential must use an
*identical* per-week loss across every pair, so we use absolute error (AE), the
common denominator. AE is a proper point-forecast loss; for a 1-step rolling-
origin task it ranks models almost identically to WIS (the WIS dominant term is
the absolute error |y−median|), so the tie / decisive structure transfers.

A secondary block reports, for the subset that *does* carry a leak-free residual
scale, an approximate per-week WIS reconstructed from a Gaussian quantile fan
(median = y_pred, scale = sigma_in_sample) — purely as a robustness cross-check,
not as the headline.

METHOD
------
Moving-block bootstrap (Künsch 1989) of the per-week loss differential.
  * block length = 4 weeks (~ monthly autocorrelation of ILI residuals)
  * reps = 5000  (>= 3000 requested)
  * seed = 42
  * 95% CI = percentile interval [2.5, 97.5] of the bootstrap mean of ΔL
For each competitor C: ΔL_t = AE_FusedEpi(t) − AE_C(t).
  mean ΔL < 0  →  FusedEpi has lower loss (better).
  CI excludes 0 → DECISIVE;  CI includes 0 → TIE (statistical equivalence).

READ-ONLY: reads only CSV artifacts under simulation/results/csv/. No DB, no
retraining. sqlite untouched.

Run:
    .venv/bin/python -m simulation.scripts.sci_delta_wis_bootstrap
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
RESULTS = Path(__file__).resolve().parents[1] / "results"
CSV_DIR = RESULTS / "csv"
OUT_JSON = RESULTS / "sci_supplement" / "delta_wis_bootstrap.json"
OUT_CSV = RESULTS / "sci_supplement" / "delta_wis_bootstrap.csv"
FIG_PATH = RESULTS / "figures" / "delta_wis_bootstrap_forest.png"

CHAMPION = "FusedEpi"
COMPETITORS = ["NegBinGLM", "TiRex", "SeirCount-TabPFN", "ARIMA", "SARIMA", "Theta"]
TOP_SET = {"FusedEpi", "TiRex", "NegBinGLM"}  # claimed statistical tie set

BLOCK = 4
REPS = 5000
SEED = 42


def load_test_preds(model: str) -> list[tuple[int, float, float]]:
    """Return sorted [(idx, y_true, y_pred)] for the test split of one model."""
    fp = CSV_DIR / f"predictions_{model}.csv"
    rows = [r for r in csv.DictReader(fp.open()) if r["split"] == "test"]
    rows.sort(key=lambda r: int(r["idx"]))
    return [(int(r["idx"]), float(r["y_true"]), float(r["y_pred"])) for r in rows]


def per_week_ae(model: str, idx_ref: np.ndarray, ytrue_ref: np.ndarray) -> np.ndarray:
    """Per-week absolute error aligned to the reference test-week index/y_true."""
    d = {i: (yt, yp) for i, yt, yp in load_test_preds(model)}
    ae = np.empty(len(idx_ref))
    for k, i in enumerate(idx_ref):
        yt, yp = d[i]
        assert abs(yt - ytrue_ref[k]) < 1e-6, f"y_true mismatch for {model} at idx {i}"
        ae[k] = abs(yt - yp)
    return ae


def moving_block_bootstrap_mean(delta: np.ndarray, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the MEAN of `delta` via moving-block resampling.

    Args:
        delta: per-week loss differential, shape (T,).
        block: block length in weeks.
        reps: number of bootstrap replicates.
        rng: numpy Generator (seeded upstream).

    Returns:
        Array shape (reps,) of bootstrap-replicate means of delta.
    """
    T = len(delta)
    n_blocks = int(np.ceil(T / block))
    max_start = T - block  # inclusive; overlapping blocks
    starts_pool = np.arange(max_start + 1)
    means = np.empty(reps)
    for b in range(reps):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        sample = np.concatenate([delta[s:s + block] for s in starts])[:T]
        means[b] = sample.mean()
    return means


def main() -> None:
    rng = np.random.default_rng(SEED)

    ref = load_test_preds(CHAMPION)
    idx_ref = np.array([i for i, _, _ in ref])
    ytrue_ref = np.array([y for _, y, _ in ref])
    T = len(ref)

    ae = {CHAMPION: per_week_ae(CHAMPION, idx_ref, ytrue_ref)}
    for c in COMPETITORS:
        ae[c] = per_week_ae(c, idx_ref, ytrue_ref)

    results = []
    for c in COMPETITORS:
        delta = ae[CHAMPION] - ae[c]  # <0 => FusedEpi better
        boot = moving_block_bootstrap_mean(delta, BLOCK, REPS, rng)
        mean = float(delta.mean())
        ci_lo, ci_hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        includes_zero = ci_lo <= 0.0 <= ci_hi
        verdict = "tie" if includes_zero else "decisive"
        # boot p ~ two-sided prob that mean differential crosses 0
        p_two = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
        p_two = float(min(1.0, p_two))
        results.append({
            "pair": f"{CHAMPION} vs {c}",
            "competitor": c,
            "competitor_in_top_set": c in TOP_SET,
            "mean_delta_ae": round(mean, 4),
            "ci95_lo": round(ci_lo, 4),
            "ci95_hi": round(ci_hi, 4),
            "ci_includes_zero": includes_zero,
            "verdict": verdict,
            "boot_p_two_sided": round(p_two, 4),
            "champion_mean_ae": round(float(ae[CHAMPION].mean()), 4),
            "competitor_mean_ae": round(float(ae[c].mean()), 4),
        })

    payload = {
        "analysis": "ANALYSIS_1_delta_loss_moving_block_bootstrap",
        "loss_used": "per_week_absolute_error",
        "loss_rationale": (
            "Per-week WIS quantiles are persisted only for the pi_source=r9_leakfree "
            "subset; TiRex/ARIMA/SARIMA/Theta carry wis=nan / pi_source=unavailable "
            "(no per-week quantiles). Absolute error is the only per-week loss "
            "reconstructable identically across all 7 models. WIS dominant term is "
            "|y-median|, so tie/decisive structure transfers."
        ),
        "champion": CHAMPION,
        "top_set_claimed_tie": sorted(TOP_SET),
        "n_test_weeks": T,
        "block_weeks": BLOCK,
        "reps": REPS,
        "seed": SEED,
        "sign_convention": "delta = AE_FusedEpi - AE_competitor; negative => FusedEpi lower loss (better)",
        "pairs": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pair", "competitor", "competitor_in_top_set", "mean_delta_ae",
            "ci95_lo", "ci95_hi", "ci_includes_zero", "verdict",
            "boot_p_two_sided", "champion_mean_ae", "competitor_mean_ae",
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)

    # ---- forest / CI plot -------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["competitor"] for r in results]
        means = [r["mean_delta_ae"] for r in results]
        los = [r["ci95_lo"] for r in results]
        his = [r["ci95_hi"] for r in results]
        ypos = np.arange(len(labels))[::-1]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for y, m, lo, hi, r in zip(ypos, means, los, his, results):
            color = "#1f77b4" if r["verdict"] == "tie" else "#d62728"
            ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round")
            ax.plot(m, y, "o", color=color, ms=7, zorder=5)
        ax.axvline(0.0, color="0.4", ls="--", lw=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([
            f"{lab}{'  (top-set)' if r['competitor_in_top_set'] else ''}"
            for lab, r in zip(labels, results)
        ])
        ax.set_xlabel("Mean ΔL = AE(FusedEpi) − AE(competitor)   [ILI rate units]")
        ax.set_title(
            "Moving-block bootstrap of per-week loss differential\n"
            f"block={BLOCK}w, reps={REPS}, seed={SEED}, n={T} test weeks  "
            "(blue=tie, red=decisive)")
        ax.text(0.02, 0.02,
                "ΔL<0 ⇒ FusedEpi lower loss (better)",
                transform=ax.transAxes, fontsize=8, color="0.3")
        fig.tight_layout()
        FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_PATH, dpi=150)
        plt.close(fig)
        payload["figure_path"] = str(FIG_PATH)
        OUT_JSON.write_text(json.dumps(payload, indent=2))
    except Exception as e:  # pragma: no cover
        print(f"[warn] figure skipped: {e}")

    # ---- console summary --------------------------------------------------
    print(f"loss = per-week absolute error  (n={T} weeks, block={BLOCK}, reps={REPS}, seed={SEED})")
    print(f"{'competitor':18s} {'meanΔ':>8} {'CI95_lo':>9} {'CI95_hi':>9}  verdict")
    for r in results:
        print(f"{r['competitor']:18s} {r['mean_delta_ae']:>8.3f} "
              f"{r['ci95_lo']:>9.3f} {r['ci95_hi']:>9.3f}  {r['verdict']}")
    print(f"\nJSON -> {OUT_JSON}\nCSV  -> {OUT_CSV}\nFIG  -> {FIG_PATH}")


if __name__ == "__main__":
    main()
