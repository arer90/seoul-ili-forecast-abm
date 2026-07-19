"""Full metric audit (Phase C — sprint 2026-05-06).

학술 표준 metric (Codex review TOP 5 + δ + 4 추가) — 모든 saved 모델에 일괄 적용.

추가 metric:
1. Block-bootstrap CI for R² (Künsch 1989; Efron & Tibshirani 1993) — n=68 신뢰성
2. Multi-α PICP (50/80/90/95) + sharpness (Gneiting et al. 2007 JRSS-B)
3. Relative WIS vs seasonal-naive (Bracher 2019 BMC Infect Dis)
4. FluSight peak-week / peak-intensity / onset (Reich 2019; Bracher 2019)
5. DSS (Dawid-Sebastiani 1999) / Energy Score (Gneiting & Raftery 2007)
6. sMAPE / WAPE (Hyndman 2006)
7. MASE seasonal (Hyndman & Koehler 2006 IJF) — scale-free
8. Diebold-Mariano test (1995 JBES) — pairwise forecast comparison (vs seasonal-naive)
9. Skill score (Murphy 1988) — 1 − MSE_model / MSE_seasonal-naive
10. R² ∈ [0.7, 0.9] window classification (사용자 목표 + Codex Q5 — n=68 SE 0.05-0.10)

사용:
    .venv/bin/python -m simulation.scripts.full_metric_audit
    .venv/bin/python -m simulation.scripts.full_metric_audit --output simulation/results/FULL_METRIC_AUDIT_20260506.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)


def block_bootstrap_r2(y_true: np.ndarray, y_pred: np.ndarray,
                       B: int = 1000, block_len: int = 4,
                       seed: int = 42) -> tuple[float, float, float]:
    """Künsch 1989 stationary block bootstrap CI for R².

    Returns:
        (lo_2.5, hi_97.5, median) — block bootstrap CI of R² statistic.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_blocks = n // block_len + 1
    r2s: list[float] = []
    for _ in range(B):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        yt = y_true[idx]
        yp = y_pred[idx]
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        if ss_tot < 1e-12:
            continue
        r2s.append(1.0 - float(((yt - yp) ** 2).sum()) / ss_tot)
    if not r2s:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.array(r2s)
    return (float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
            float(np.median(arr)))


def seasonal_naive(y_train: np.ndarray, n_test: int, period: int = 52) -> np.ndarray:
    """Last full season repeats."""
    if len(y_train) < period:
        return np.repeat(y_train.mean(), n_test)
    last_season = y_train[-period:]
    reps = (n_test // period) + 1
    return np.tile(last_season, reps)[:n_test]


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(2.0 * np.abs(y_true - y_pred) /
                         np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) /
                 max(float(np.sum(np.abs(y_true))), 1e-6) * 100)


def mase_seasonal(y_true: np.ndarray, y_pred: np.ndarray,
                  y_train: np.ndarray, period: int = 52) -> float:
    """Hyndman & Koehler 2006 MASE — seasonal naive denominator."""
    if len(y_train) <= period:
        return float("nan")
    naive_mae = float(np.mean(np.abs(y_train[period:] - y_train[:-period])))
    if naive_mae < 1e-6:
        return float("inf")
    mae_model = float(np.mean(np.abs(y_true - y_pred)))
    return mae_model / naive_mae


def peak_week_error(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return int(abs(int(np.argmax(y_true)) - int(np.argmax(y_pred))))


def peak_intensity_log_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(abs(np.log1p(float(y_true.max())) -
                     np.log1p(max(float(y_pred.max()), 0.0))))


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray) -> tuple[float, float]:
    """DM test — squared-error loss differential (Diebold & Mariano 1995).

    Returns (t_stat, two_sided_p_value).
    """
    d = e_a ** 2 - e_b ** 2
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    d_mean = float(d.mean())
    var_d = float(d.var(ddof=1)) / n
    if var_d <= 0 or not np.isfinite(var_d):
        return (float("nan"), float("nan"))
    t = d_mean / np.sqrt(var_d)
    try:
        from scipy import stats
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(t))))
    except Exception:
        p = float("nan")
    return (float(t), p)


def picp_at(y_true: np.ndarray, y_pred: np.ndarray,
            sigma: np.ndarray | float, alpha: float) -> float:
    """Multi-α PICP using normal-approx PI."""
    from scipy import stats
    z = float(stats.norm.ppf(1 - (1 - alpha) / 2))
    s = np.asarray(sigma, dtype=float)
    if s.ndim == 0:
        s = np.full_like(y_true, float(s))
    lo = y_pred - z * s
    hi = y_pred + z * s
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def sharpness(sigma: np.ndarray | float, alpha: float = 0.95) -> float:
    """Average PI width at given α."""
    from scipy import stats
    z = float(stats.norm.ppf(1 - (1 - alpha) / 2))
    s = np.asarray(sigma, dtype=float)
    return float(np.mean(2.0 * z * s))


def dss_score(y_true: np.ndarray, y_pred: np.ndarray,
              sigma: np.ndarray | float) -> float:
    """Dawid-Sebastiani 1999 — log(σ²) + (y-μ)²/σ²"""
    s = np.asarray(sigma, dtype=float)
    if s.ndim == 0:
        s = np.full_like(y_true, float(s))
    s2 = np.maximum(s ** 2, 1e-6)
    return float(np.mean(np.log(s2) + (y_true - y_pred) ** 2 / s2))


def r2_window_class(r2: float) -> str:
    """사용자 목표 R² ∈ [0.7, 0.9] window classification."""
    if r2 < 0.7:
        return "below"   # 개선 필요 또는 baseline
    elif r2 > 0.9:
        return "above"   # leakage audit 필요 (A.3 실제 audit pass 시 strong)
    else:
        return "window"  # 양호


# FluSight 11-α-level standard (Bracher 2021; CDC FluSight 2022-23):
#   α/2 quantile + (1 - α/2) quantile for each α ∈ FLUSIGHT_ALPHAS, plus median.
FLUSIGHT_ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def wis_normal_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    alphas: tuple[float, ...] = FLUSIGHT_ALPHAS,
) -> float:
    """Weighted Interval Score (Bracher 2021 PLOS CB) via normal predictive dist.

    A7 (G-175 audit fix): 기존 ``wis = mae`` proxy 대체. sigma 가 있을 때만
    의미 있는 prob score. K-α 별 interval score 의 가중 평균 + 0.5×|y-μ| (median).

    Formula:
      IS_α(y, l, u) = (u − l) + (2/α)·(l − y)·𝟙(y<l) + (2/α)·(y − u)·𝟙(y>u)
      WIS = (1/(K+0.5))·[0.5·|y − μ| + Σ_α (α/2)·IS_α(y, l_α, u_α)]
      with l_α = μ + z(α/2)·σ, u_α = μ + z(1-α/2)·σ.

    Returns:
        Mean WIS across n test points (float).

    Performance: O(n × K), K = len(alphas) = 11 default.
    Caller responsibility: sigma > 0 (proxy = std(residual) acceptable for
        point forecasters; conformal/CQR/heteroscedastic σ preferred).
    Reference:
        Bracher 2021 PLOS Comput Biol — WIS formal definition + FluSight α set.
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full_like(y_true, float(sigma))
    sigma = np.maximum(sigma, 1e-6)

    n = len(y_true)
    K = len(alphas)
    wis_per_point = np.zeros(n, dtype=float)
    weight_total = 0.5 + sum(a / 2.0 for a in alphas)

    for i in range(n):
        y, mu, s = float(y_true[i]), float(y_pred[i]), float(sigma[i])
        # median term
        agg = 0.5 * abs(y - mu)
        # K interval terms
        for a in alphas:
            z_lo = float(stats.norm.ppf(a / 2.0))
            z_hi = float(stats.norm.ppf(1.0 - a / 2.0))
            lo = mu + z_lo * s
            hi = mu + z_hi * s
            width = hi - lo
            under = (lo - y) if y < lo else 0.0
            over = (y - hi) if y > hi else 0.0
            is_a = width + (2.0 / a) * (under + over)
            agg += (a / 2.0) * is_a
        wis_per_point[i] = agg / weight_total

    return float(np.mean(wis_per_point))


def pairwise_dm_matrix(
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pairwise Diebold-Mariano test matrix across all models.

    A7 (G-175 audit fix): 기존 "n × n DM 비교는 separate sprint" 코멘트 대체.

    Args:
        predictions: name → 1-D prediction vector (all same length as y_test).
        y_test: ground truth.

    Returns:
        (t_matrix, p_matrix, names):
          t_matrix[i,j] = DM t-stat for loss_i vs loss_j (squared-error).
                          positive = model_i has WORSE squared error than j.
          p_matrix[i,j] = two-sided p-value.
          names = ordered model names.

    Performance: O(n_models² × n_test). For 60 models × n=68 ≈ 0.5s.
    """
    names = sorted(predictions.keys())
    n = len(names)
    t_mat = np.full((n, n), np.nan, dtype=float)
    p_mat = np.full((n, n), np.nan, dtype=float)
    errs = {nm: y_test - predictions[nm] for nm in names}
    for i in range(n):
        for j in range(n):
            if i == j:
                t_mat[i, j] = 0.0
                p_mat[i, j] = 1.0
                continue
            t, p = diebold_mariano(errs[names[i]], errs[names[j]])
            t_mat[i, j] = t
            p_mat[i, j] = p
    return t_mat, p_mat, names


def mcs_alpha_05(
    p_matrix: np.ndarray,
    names: list[str],
    losses: dict[str, float],
    alpha: float = 0.05,
) -> tuple[list[str], list[str]]:
    """Hansen 2011 Model Confidence Set (simplified t_max elimination).

    A7 (G-175 audit fix): simplified Hansen 2011 MCS. Eliminates models whose
    worst pairwise DM stat (vs surviving set) rejects null at α.

    Args:
        p_matrix: pairwise DM p-values from pairwise_dm_matrix.
        names: ordered model names matching p_matrix.
        losses: per-model summary loss (MSE / WIS) used to direction-orient.
        alpha: significance level (default 0.05 = 95% MCS).

    Returns:
        (survivors, eliminated): names that stay in MCS vs eliminated.

    Reference:
        Hansen, Lunde & Nason 2011 Econometrica — Model Confidence Set.
        Simplified: drop the model whose loss is highest among any pair with
        p < α. Repeat until no rejection remains.

    Caveat: full MCS uses bootstrap distribution; this simplified variant uses
        asymptotic normal DM p-values. For Q1-grade reporting use proper
        Hansen MCS via the `arch` package — this is a smoke implementation.
    """
    survivors = list(names)
    eliminated: list[str] = []
    name_idx = {nm: i for i, nm in enumerate(names)}

    while True:
        # Find any rejected pair (p < alpha) among survivors
        worst_model = None
        worst_loss = -np.inf
        for nm_a in survivors:
            for nm_b in survivors:
                if nm_a == nm_b:
                    continue
                p = p_matrix[name_idx[nm_a], name_idx[nm_b]]
                if not np.isfinite(p):
                    continue
                if p < alpha:
                    # at least one of them has rejected; worse-loss model eliminated
                    la, lb = losses.get(nm_a, np.inf), losses.get(nm_b, np.inf)
                    candidate = nm_a if la > lb else nm_b
                    cand_loss = max(la, lb)
                    if cand_loss > worst_loss:
                        worst_loss = cand_loss
                        worst_model = candidate
        if worst_model is None:
            break
        survivors.remove(worst_model)
        eliminated.append(worst_model)
        if len(survivors) <= 1:
            break

    return survivors, eliminated


# ----- main -----

def audit_all_models(test_window: tuple[int, int] = (269, 269 + 68),
                     out_path: Optional[Path] = None) -> dict:
    PMO = get_results_dir() / "per_model_optimal"
    cache = pl.read_parquet("simulation/cache/feature_cache.parquet")
    y = cache.select("ili_rate").to_numpy().flatten()
    y_train = y[: test_window[0]]
    y_test = y[test_window[0]: test_window[1]]
    n_test = len(y_test)
    ss_test = float(((y_test - y_test.mean()) ** 2).sum())

    # Baseline = seasonal naive (52w lag)
    sn_pred = seasonal_naive(y_train, n_test, period=52)
    sn_mae = float(np.mean(np.abs(y_test - sn_pred)))
    sn_mse = float(np.mean((y_test - sn_pred) ** 2))
    # A7 (G-175 audit fix 2026-05-12): real WIS for baseline using
    # residual-std sigma proxy (seasonal-naive has no σ; using SD of error).
    sn_sigma = max(float(np.std(y_test - sn_pred)), 1e-6)
    sn_wis = wis_normal_quantile(y_test, sn_pred, np.full(n_test, sn_sigma))

    results: list[dict] = []
    predictions: dict[str, np.ndarray] = {}  # A7 fix: needed for MCS/DM matrix
    losses_mse: dict[str, float] = {}
    for jp in sorted(PMO.glob("*.json")):
        nm = jp.stem
        if nm.startswith("_") or ".backup" in nm:
            continue
        try:
            d = json.loads(jp.read_text())
        except Exception:
            continue
        rtp = d.get("refit_test_predictions", [])
        if not rtp or len(rtp) != n_test:
            continue
        pred = np.asarray(rtp, dtype=float)

        # standard
        mae = float(np.mean(np.abs(y_test - pred)))
        mse = float(np.mean((y_test - pred) ** 2))
        r2 = 1.0 - float(((y_test - pred) ** 2).sum()) / ss_test
        mape = float(np.mean(np.abs((y_test - pred) /
                                    np.maximum(y_test, 0.01))) * 100)

        # C.1 Block-bootstrap CI
        r2_lo, r2_hi, r2_med = block_bootstrap_r2(y_test, pred, B=500, block_len=4)

        # σ from saved test_metrics if available, else proxy = std(residual).
        # (Moved earlier in A7 fix so real WIS can use it.)
        sigma_arr = np.asarray(d.get("refit_test_sigma", []), dtype=float)
        if sigma_arr.size != n_test:
            sigma_proxy = float(np.std(y_test - pred))
            sigma_arr = np.full(n_test, max(sigma_proxy, 1e-6))

        # C.3 Real WIS (A7, G-175 audit fix 2026-05-12) — Bracher 2021
        # 11-α FluSight quantile sum under normal predictive distribution.
        wis = wis_normal_quantile(y_test, pred, sigma_arr)
        rel_wis = wis / max(sn_wis, 1e-6)

        # C.4 FluSight peak
        pw_err = peak_week_error(y_test, pred)
        pi_err = peak_intensity_log_error(y_test, pred)

        # C.6 sMAPE / WAPE
        smape_v = smape(y_test, pred)
        wape_v = wape(y_test, pred)

        # C.7 MASE seasonal
        mase_v = mase_seasonal(y_test, pred, y_train, period=52)

        # C.8 DM test (vs seasonal-naive)
        e_model = y_test - pred
        e_naive = y_test - sn_pred
        t_dm, p_dm = diebold_mariano(e_model, e_naive)

        # C.10 Skill score
        skill = 1.0 - mse / max(sn_mse, 1e-6)

        # (sigma_arr already computed above for real WIS)
        # C.2 Multi-α PICP
        picp50 = picp_at(y_test, pred, sigma_arr, 0.50)
        picp80 = picp_at(y_test, pred, sigma_arr, 0.80)
        picp90 = picp_at(y_test, pred, sigma_arr, 0.90)
        picp95 = picp_at(y_test, pred, sigma_arr, 0.95)
        sharp95 = sharpness(sigma_arr, 0.95)

        # C.5 DSS
        dss = dss_score(y_test, pred, sigma_arr)

        # window classification
        r2_class = r2_window_class(r2)

        # PASS gate (4-criteria, G-175 audit 2026-05-11: MAPE 25→20, PICP95 0.85→0.90)
        pass_main = (r2 >= 0.80 and mape <= 20 and wis <= 6 and picp95 >= 0.90)

        results.append({
            "name": nm,
            "r2": r2, "r2_ci": (r2_lo, r2_hi), "r2_median_bs": r2_med,
            "mae": mae, "mse": mse, "mape": mape, "smape": smape_v, "wape": wape_v,
            "wis": wis, "rel_wis": rel_wis,
            "mase": mase_v,
            "skill_score": skill,
            "dm_t": t_dm, "dm_p": p_dm,
            "picp50": picp50, "picp80": picp80, "picp90": picp90, "picp95": picp95,
            "sharp95": sharp95,
            "dss": dss,
            "pw_err_weeks": pw_err, "pi_log_err": pi_err,
            "r2_class": r2_class, "pass_main": pass_main,
        })
        predictions[nm] = pred
        losses_mse[nm] = mse

    # A7 (G-175 audit fix 2026-05-12): MCS / DM matrix (pairwise)
    # 이전: "separate sprint" 코멘트 → 실제 미구현. 이번 fix: 60-model
    # pairwise DM matrix O(60²) ≈ 0.5s + Hansen 2011 simplified MCS.
    mcs_survivors: list[str] = []
    mcs_eliminated: list[str] = []
    dm_pairwise_summary: dict = {}
    if len(predictions) >= 2:
        try:
            _, p_mat, names = pairwise_dm_matrix(predictions, y_test)
            mcs_survivors, mcs_eliminated = mcs_alpha_05(
                p_mat, names, losses_mse, alpha=0.05
            )
            # Per-model summary: n_rejected_against (count of p_ij < 0.05 vs others)
            for i, nm in enumerate(names):
                n_rej = int(np.sum(p_mat[i, :] < 0.05) - 1)  # exclude self
                dm_pairwise_summary[nm] = {
                    "n_rejected_against": n_rej,
                    "in_mcs_05": nm in mcs_survivors,
                }
        except (ValueError, RuntimeError, ImportError) as e:
            dm_pairwise_summary = {"error": str(e)}

    return {
        "n_test": n_test,
        "baseline_sn": {"mae": sn_mae, "mse": sn_mse, "wis": sn_wis},
        "results": results,
        "mcs_alpha_05": {
            "survivors": mcs_survivors,
            "eliminated": mcs_eliminated,
            "n_survivors": len(mcs_survivors),
            "n_eliminated": len(mcs_eliminated),
        },
        "dm_pairwise_summary": dm_pairwise_summary,
    }


def print_summary(audit: dict) -> str:
    res = audit["results"]
    n = len(res)
    n_pass = sum(1 for r in res if r["pass_main"])
    n_window = sum(1 for r in res if r["r2_class"] == "window")
    n_above = sum(1 for r in res if r["r2_class"] == "above")
    n_below = sum(1 for r in res if r["r2_class"] == "below")

    lines = []
    lines.append(f"=== Full Metric Audit (n={n}, sprint 2026-05-06) ===")
    lines.append(f"baseline (seasonal-naive): WIS={audit['baseline_sn']['wis']:.3f}")
    lines.append(f"R² window classification: below(<0.7)={n_below}  window(0.7-0.9)={n_window}  above(>0.9)={n_above}")
    lines.append(f"PASS @ Main (R²≥0.80, MAPE≤20, WIS≤6, PICP95≥0.90; G-175 forward): {n_pass}/{n}")
    lines.append("")
    header = (f"{'name':22s} {'R²':>7s} {'CI':>14s} "
              f"{'MAPE':>5s} {'sMAPE':>5s} {'WAPE':>5s} "
              f"{'WIS':>5s} {'rWIS':>5s} {'MASE':>5s} {'skill':>6s} "
              f"{'PICP':>5s} {'PWerr':>5s} {'PIerr':>5s} {'DM_p':>6s} {'class':>6s} {'PASS':>4s}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in sorted(res, key=lambda x: -x["r2"]):
        ci_str = f"[{r['r2_ci'][0]:+.2f},{r['r2_ci'][1]:+.2f}]"
        lines.append(
            f"{r['name']:22s} {r['r2']:>+.4f} {ci_str:>14s} "
            f"{r['mape']:>4.1f}% {r['smape']:>4.1f}% {r['wape']:>4.1f}% "
            f"{r['wis']:>5.2f} {r['rel_wis']:>5.2f} {r['mase']:>5.2f} {r['skill_score']:>+6.2f} "
            f"{r['picp95']:>5.2f} {r['pw_err_weeks']:>4d}w {r['pi_log_err']:>5.2f} "
            f"{r['dm_p']:>6.3f} {r['r2_class']:>6s} {'PASS' if r['pass_main'] else 'fail':>4s}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None,
                        help="output markdown file path")
    args = parser.parse_args()

    audit = audit_all_models()
    summary = print_summary(audit)
    print(summary)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        # markdown
        md_lines = [
            "# Full Metric Audit — sprint 2026-05-06",
            "",
            "학술 표준 metric (Codex Top 5 + δ + 4 추가) — 모든 saved 모델 일괄 적용.",
            "",
            "## Summary",
            "",
            f"- n_test = {audit['n_test']}",
            f"- baseline (seasonal-naive) WIS = {audit['baseline_sn']['wis']:.3f}",
            "",
            "## Per-model metrics",
            "",
            "```",
            summary,
            "```",
        ]
        out.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\n[OK] markdown written → {out}")


if __name__ == "__main__":
    main()
