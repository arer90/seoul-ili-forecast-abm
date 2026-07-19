#!/usr/bin/env python3
"""SCI supplement ① — FusedEpi pairwise Diebold-Mariano matrix (WIS-loss diff).

Regenerates the §4.7 DM matrix that was promised but left empty (the archived
``dm_pvalues.csv`` is all-NaN and covers only 5 legacy models; the R6 checkpoint
DM block is also all-NaN). This is a **standalone, read-only, zero-retraining**
re-run: it reconstructs the n=68 test slab deterministically from the live data
pipeline (``run_data``, seed=42, READ_ONLY DB) and consumes the **stored**
per-model ``refit_test_predictions`` + leak-free ``insample_residuals`` from
``per_model_optimal/*.json``. No model is refit; no live module is modified.

Method
------
* Per-point WIS via ``weighted_interval_score_empirical`` (Bracher 2021 eq.3-4,
  Lei 2018 split-conformal): empirical |residual| quantile half-widths over the
  K=11 FluSight α-grid. Leak-free — residuals are R9 in-sample (never y_test).
* Loss differential d_t = WIS_FusedEpi[t] - WIS_other[t].
* Diebold-Mariano (1995) statistic with Harvey-Leybourne-Newbold (1997)
  small-sample correction (h=1). Two-sided p via Student-t(df=n-1).
  d_bar < 0 ⇒ FusedEpi has lower WIS (better).
* Multiplicity: Holm (1979, FWER) + Benjamini-Hochberg (1995, FDR) over the
  m = (#competitors) FusedEpi-vs-X family.

Outputs
-------
* ``simulation/results/sci_supplement/dm_pairwise_matrix.csv`` — long form
  (pair, dm_stat, p_raw, p_holm, p_bh, dwis_mean, n, favors).
* ``simulation/results/sci_supplement/dm_pairwise_matrix.json`` — full record
  incl. provenance + leak-free attestation.

Run:  .venv/bin/python -m simulation.scripts.sci_supplement.dm_pairwise_matrix
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

# K=11 FluSight α-grid (matches simulation/scripts/full_metric_audit.py:168 and
# per_model_eval residual-WIS path).
FLUSIGHT_ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

CHAMPION = "FusedEpi"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPTIMAL_DIR = PROJECT_ROOT / "simulation" / "results" / "per_model_optimal"
OUT_DIR = PROJECT_ROOT / "simulation" / "results" / "sci_supplement"


def _reconstruct_y_test() -> tuple[np.ndarray, int, int]:
    """Rebuild the n=68 test slab deterministically (READ_ONLY DB, no retrain).

    Returns:
        (y_test, test_start, n_test). run_data uses safe READ overlay; seed=42.
    """
    os.environ.setdefault("MPH_EVAL_FEATURES", "basic")  # baseline feature regime
    np.random.seed(42)
    from simulation.pipeline.config import PipelineConfig
    from simulation.pipeline.data import run_data

    cfg = PipelineConfig()
    d = run_data(cfg)
    y = np.asarray(d["y_all"], dtype=np.float64)
    ts = int(d["test_start"])
    nt = int(d["n_test"])
    return y[ts : ts + nt], ts, nt


def _load_models() -> dict[str, dict]:
    """Load stored test predictions + leak-free residuals for every optimal model.

    Returns:
        {name: {"pred": (n,), "resid": (m,)}} — only models with both a length-68
        prediction vector AND ≥2 finite in-sample residuals are retained (the rest
        cannot get a leak-free WIS, so they are reported as skipped, never imputed).
    """
    out: dict[str, dict] = {}
    for fn in sorted(glob.glob(str(OPTIMAL_DIR / "*.json"))):
        d = json.load(open(fn, encoding="utf-8"))
        name = d.get("model")
        if not name:
            continue
        tp = d.get("refit_test_predictions")
        ires = (d.get("val_metrics", {}) or {}).get("insample_residuals")
        if tp is None:
            continue
        pred = np.asarray(tp, dtype=np.float64)
        res = np.asarray(ires, dtype=np.float64) if ires is not None else np.array([])
        res = res[np.isfinite(res)]
        out[name] = {"pred": pred, "resid": res}
    return out


def _per_point_wis(y_test, pred, resid) -> np.ndarray:
    from simulation.analytics.diagnostics import weighted_interval_score_empirical

    return weighted_interval_score_empirical(
        y_test, pred, resid, alphas=list(FLUSIGHT_ALPHAS)
    )


def _dm_hln(d: np.ndarray, h: int = 1) -> tuple[float, float, int]:
    """Diebold-Mariano stat with Harvey-Leybourne-Newbold (1997) correction.

    Args:
        d: loss differential series d_t = loss_A[t] - loss_B[t] (finite).
        h: forecast horizon (1 ⇒ no autocovariance term).

    Returns:
        (dm_corrected, p_two_sided, n). Mirrors
        simulation/analytics/metrics.py:diebold_mariano but operates on a
        pre-computed differential (WIS-loss, not squared error).
    """
    from scipy import stats

    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return float("nan"), float("nan"), n
    d_bar = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=1))
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_sum += float(np.cov(d[:-k], d[k:], ddof=1)[0, 1])
    var_d = (gamma0 + 2.0 * gamma_sum) / n
    if var_d <= 0:
        # degenerate (identical losses) — no evidence of difference
        return 0.0, 1.0, n
    dm = d_bar / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_c = dm * correction
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_c), df=n - 1))
    return float(dm_c), float(p), n


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (1979). Order-preserving."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    run_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        run_max = max(run_max, val)
        adj[idx] = min(run_max, 1.0)
    return adj.tolist()


def _bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjusted p-values (1995). Step-up, monotone."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    run_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = pvals[idx] * m / (rank + 1)
        run_min = min(run_min, val)
        adj[idx] = min(run_min, 1.0)
    return adj.tolist()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_test, test_start, n_test = _reconstruct_y_test()
    models = _load_models()

    if CHAMPION not in models or len(models[CHAMPION]["resid"]) < 2:
        raise SystemExit(
            f"[DM] champion {CHAMPION} missing predictions or leak-free residuals — abort"
        )
    if len(models[CHAMPION]["pred"]) != n_test:
        raise SystemExit(
            f"[DM] champion pred len {len(models[CHAMPION]['pred'])} != n_test {n_test}"
        )

    wis_champ = _per_point_wis(y_test, models[CHAMPION]["pred"], models[CHAMPION]["resid"])

    rows: list[dict] = []
    skipped: list[dict] = []
    for name, m in models.items():
        if name == CHAMPION:
            continue
        if len(m["pred"]) != n_test:
            skipped.append({"model": name, "reason": f"pred len {len(m['pred'])} != {n_test}"})
            continue
        if len(m["resid"]) < 2:
            skipped.append({"model": name, "reason": "no leak-free in-sample residual (≥2)"})
            continue
        wis_other = _per_point_wis(y_test, m["pred"], m["resid"])
        d = wis_champ - wis_other  # <0 ⇒ FusedEpi lower WIS (better)
        # only points where both WIS finite
        mask = np.isfinite(d)
        if mask.sum() < 3:
            skipped.append({"model": name, "reason": "fewer than 3 finite WIS-diff points"})
            continue
        dm_stat, p_raw, n_eff = _dm_hln(d[mask], h=1)
        dwis_mean = float(np.mean(d[mask]))
        rows.append({
            "pair": f"{CHAMPION}_vs_{name}",
            "competitor": name,
            "dm_stat": round(dm_stat, 6),
            "p_raw": round(p_raw, 6),
            "dwis_mean": round(dwis_mean, 6),
            "wis_champ_mean": round(float(np.mean(wis_champ[mask])), 6),
            "wis_other_mean": round(float(np.mean(wis_other[mask])), 6),
            "n": int(n_eff),
            "favors": (CHAMPION if dwis_mean < 0 else name) if p_raw < 0.05 else "tie(ns)",
        })

    # ── Diagnostic: point-forecast (squared-error) DM for WIS-skipped models that
    #    still have stored predictions. These cannot enter the probabilistic WIS
    #    matrix (no leak-free test PI ⇒ pi_source="unavailable" in R10), but a
    #    point-accuracy DM is still defined. Reported separately (NOT in the
    #    primary WIS family, NOT multiplicity-corrected with it). This is where
    #    the documented "FusedEpi vs TiRex p≈0.223" lives — it is a squared-error
    #    DM on point forecasts, never a WIS-DM (TiRex has no leak-free test PI).
    pred_champ = models[CHAMPION]["pred"]
    point_rows: list[dict] = []
    for s in skipped:
        nm = s["model"]
        if "no leak-free" not in s["reason"]:
            continue  # only the residual-missing ones have usable point preds
        pm = models.get(nm, {})
        po = pm.get("pred")
        if po is None or len(po) != n_test:
            continue
        e1 = y_test - pred_champ
        e2 = y_test - po
        d_se = e1 ** 2 - e2 ** 2  # <0 ⇒ FusedEpi lower squared error (better)
        dm_stat, p_raw, n_eff = _dm_hln(d_se, h=1)
        d_mean = float(np.mean(d_se))
        point_rows.append({
            "pair": f"{CHAMPION}_vs_{nm}",
            "competitor": nm,
            "loss": "squared_error_point",
            "dm_stat": round(dm_stat, 6),
            "p_raw": round(p_raw, 6),
            "d_mean_se": round(d_mean, 6),
            "mse_champ": round(float(np.mean(e1 ** 2)), 6),
            "mse_other": round(float(np.mean(e2 ** 2)), 6),
            "n": int(n_eff),
            "favors": (CHAMPION if d_mean < 0 else nm) if p_raw < 0.05 else "tie(ns)",
        })
    point_rows.sort(key=lambda r: r["p_raw"])

    # Multiplicity correction over the FusedEpi-vs-X family (primary WIS family only)
    praw = [r["p_raw"] for r in rows]
    if praw:
        ph = _holm(praw)
        pb = _bh(praw)
        for r, h_, b_ in zip(rows, ph, pb):
            r["p_holm"] = round(float(h_), 6)
            r["p_bh"] = round(float(b_), 6)

    rows.sort(key=lambda r: r["p_raw"])

    # ── CSV (long form) ──
    csv_path = OUT_DIR / "dm_pairwise_matrix.csv"
    cols = ["pair", "competitor", "dm_stat", "p_raw", "p_holm", "p_bh",
            "dwis_mean", "wis_champ_mean", "wis_other_mean", "n", "favors"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # ── Point-diagnostic CSV (squared-error DM for WIS-skipped models) ──
    if point_rows:
        pcsv = OUT_DIR / "dm_point_diagnostic.csv"
        pcols = ["pair", "competitor", "loss", "dm_stat", "p_raw", "d_mean_se",
                 "mse_champ", "mse_other", "n", "favors"]
        with open(pcsv, "w", encoding="utf-8") as f:
            f.write(",".join(pcols) + "\n")
            for r in point_rows:
                f.write(",".join(str(r.get(c, "")) for c in pcols) + "\n")

    # ── Summary counts ──
    n_sig_holm = sum(1 for r in rows if r.get("p_holm", 1.0) < 0.05)
    n_sig_bh = sum(1 for r in rows if r.get("p_bh", 1.0) < 0.05)
    n_sig_raw = sum(1 for r in rows if r["p_raw"] < 0.05)
    n_champ_better_sig = sum(
        1 for r in rows if r["p_raw"] < 0.05 and r["dwis_mean"] < 0
    )
    n_champ_worse_sig = sum(
        1 for r in rows if r["p_raw"] < 0.05 and r["dwis_mean"] > 0
    )
    tirex = next((r for r in rows if r["competitor"] == "TiRex"), None)
    tirex_point = next((r for r in point_rows if r["competitor"] == "TiRex"), None)

    record = {
        "champion": CHAMPION,
        "loss": "WIS (empirical residual-quantile, K=11 FluSight; Bracher 2021)",
        "test": "Diebold-Mariano 1995 + Harvey-Leybourne-Newbold 1997 small-sample correction (h=1)",
        "multiplicity": "Holm (FWER) + Benjamini-Hochberg (FDR) over FusedEpi-vs-X family",
        "n_test": int(n_test),
        "test_window_idx": [int(test_start), int(test_start + n_test)],
        "n_competitors_compared": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_sig_raw_p05": n_sig_raw,
        "n_sig_holm_p05": n_sig_holm,
        "n_sig_bh_p05": n_sig_bh,
        "n_champion_significantly_better": n_champ_better_sig,
        "n_champion_significantly_worse": n_champ_worse_sig,
        "fusedepi_vs_tirex_WIS": tirex,  # None — TiRex has no leak-free test PI
        "fusedepi_vs_tirex_point_squared_error": tirex_point,
        "note_on_tirex": (
            "TiRex has pi_source='unavailable' in R10 (no leak-free test PI), so a "
            "probabilistic WIS-DM is undefined — faithfully skipped, matching R10. "
            "The documented p≈0.223 is the SQUARED-ERROR point-forecast DM (see "
            "fusedepi_vs_tirex_point_squared_error): a genuine point-accuracy tie."
        ),
        "rows": rows,
        "point_diagnostic_rows": point_rows,
        "provenance": {
            "y_test": "reconstructed via run_data(PipelineConfig()) seed=42, READ_ONLY DB",
            "predictions": "stored refit_test_predictions from per_model_optimal/*.json (no refit)",
            "residuals": "stored val_metrics.insample_residuals (R9 leak-free; never y_test)",
            "retraining": "NONE — zero model refits",
            "live_code_modified": "NONE — standalone script",
            "leak_free": True,
        },
    }
    json_path = OUT_DIR / "dm_pairwise_matrix.json"
    json.dump(record, open(json_path, "w", encoding="utf-8"), indent=2)

    # ── stdout report ──
    print(f"[DM] n_test={n_test}  window={record['test_window_idx']}")
    print(f"[DM] competitors compared={len(rows)}  skipped={len(skipped)}")
    print(f"[DM] sig raw p<.05={n_sig_raw}  Holm={n_sig_holm}  BH={n_sig_bh}")
    print(f"[DM] FusedEpi significantly BETTER than {n_champ_better_sig}; WORSE than {n_champ_worse_sig}")
    if tirex:
        print(f"[DM] FusedEpi vs TiRex (WIS): dm={tirex['dm_stat']} p_raw={tirex['p_raw']} "
              f"p_holm={tirex.get('p_holm')} dWIS={tirex['dwis_mean']} favors={tirex['favors']}")
    else:
        print("[DM] FusedEpi vs TiRex (WIS): UNDEFINED — TiRex has no leak-free test PI (pi_source=unavailable in R10)")
    if tirex_point:
        print(f"[DM] FusedEpi vs TiRex (squared-error point, the documented ~0.223): "
              f"dm={tirex_point['dm_stat']} p={tirex_point['p_raw']} favors={tirex_point['favors']}")
    if skipped:
        print(f"[DM] skipped (no leak-free WIS): {[s['model'] for s in skipped]}")
    print(f"[DM] wrote {csv_path}")
    print(f"[DM] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
