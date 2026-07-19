"""simulation.abm.behavior_disease_validation

행동-질병 결합의 **SCI-급 validation** — real ILI 대비 adaptive(behaviour-on) vs
static(behaviour-off) 를 **WIS·RMSE·MAE·CRPS·PI coverage + 통계적 유의성(HLN-DM, bootstrap CI)**
로 비교한다.

"adaptive 가 곡선을 바꾼다"(내부) 가 아니라 **"adaptive 가 real 감시자료에 통계적으로 유의하게
더 맞는다"** 를 증명하는 것이 목표 (논문 RQ-A/H-A 1차 결과표).

입력: 각 arm 의 앙상블 예측 reps (n_seeds × T, real-ILI 스케일로 affine 매핑됨) + real ILI y.
  - epi_proof._evaluate_arm 의 ``mapped_replicates`` 를 그대로 넣으면 된다.

지표 (모두 per-week → 평균):
  - WIS (Bracher 2021, 앙상블 분위수 기반, proper score) — 1차.
  - CRPS (energy form, 앙상블) — proper score.
  - RMSE, MAE (median 점예측) — 보조.
  - PI coverage @ 95/50 — 보정.
유의성:
  - HLN-corrected Diebold-Mariano (per-week WIS 차이, h=1, t_{n-1}) — adaptive vs static.
  - paired bootstrap 95% CI of mean ΔWIS.
  - 판정: adaptive 유의 우수 ⇔ ΔWIS<0 ∧ DM p<0.05 ∧ bootstrap CI upper<0.

Gray-box 계약: 순수 함수(부작용 없음). reps shape (S,T); y shape (T,). NaN/shape 위반 → ValueError.
"""
from __future__ import annotations

import math

import numpy as np

# FluSight 11 중심구간 (Bracher 2021)
ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def _as_reps(reps, name: str) -> np.ndarray:
    a = np.asarray(reps, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_seeds, T), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite")
    return a


def _as_y(y, T: int) -> np.ndarray:
    a = np.asarray(y, dtype=np.float64).ravel()
    if a.shape[0] != T:
        raise ValueError(f"y length {a.shape[0]} != reps T {T}")
    if not np.all(np.isfinite(a)):
        raise ValueError("y contains non-finite")
    return a


def wis_per_week(y, reps, alphas=ALPHAS) -> np.ndarray:
    """앙상블 분위수 기반 WIS per week (Bracher 2021). reps (S,T), y (T,)."""
    reps = _as_reps(reps, "reps")
    T = reps.shape[1]
    y = _as_y(y, T)
    med = np.median(reps, axis=0)
    out = np.zeros(T)
    K = len(alphas)
    for k, al in enumerate(alphas):
        lo = np.quantile(reps, al / 2.0, axis=0)
        hi = np.quantile(reps, 1.0 - al / 2.0, axis=0)
        is_a = (hi - lo) + (2.0/al)*np.maximum(0.0, lo - y) + (2.0/al)*np.maximum(0.0, y - hi)
        out += (al / 2.0) * is_a
    out += 0.5 * np.abs(y - med)
    return out / (K + 0.5)


def crps_per_week(y, reps) -> np.ndarray:
    """Energy-form CRPS per week: E|X−y| − 0.5 E|X−X'|. reps (S,T)."""
    reps = _as_reps(reps, "reps")
    T = reps.shape[1]
    y = _as_y(y, T)
    out = np.zeros(T)
    for t in range(T):
        x = reps[:, t]
        term1 = np.mean(np.abs(x - y[t]))
        term2 = np.mean(np.abs(x[:, None] - x[None, :]))
        out[t] = term1 - 0.5 * term2
    return out


def pi_coverage(y, reps, alpha=0.05) -> float:
    """경험 PI coverage @ (1−alpha) — y 가 앙상블 중심구간에 드는 비율."""
    reps = _as_reps(reps, "reps")
    T = reps.shape[1]
    y = _as_y(y, T)
    lo = np.quantile(reps, alpha / 2.0, axis=0)
    hi = np.quantile(reps, 1.0 - alpha / 2.0, axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def point_metrics(y, reps) -> dict:
    """median 점예측 RMSE·MAE·R²."""
    reps = _as_reps(reps, "reps")
    T = reps.shape[1]
    y = _as_y(y, T)
    m = np.median(reps, axis=0)
    e = m - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": 1.0 - float(np.sum(e ** 2)) / sst if sst > 0 else float("nan"),
    }


# ── 유의성 (HLN-DM + bootstrap) ──
def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lb) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2*m - 1) * (a + 2*m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2*m) * (a + 2*m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1 - c * d) < 1e-10:
            break
    val = front * (f - 1.0)
    if x > (a + 1) / (a + b + 2):
        val = 1.0 - _betainc(b, a, 1 - x)
    return min(max(val, 0.0), 1.0)


def dm_hln(loss_a, loss_b, h: int = 1) -> tuple[float, float]:
    """HLN-corrected Diebold-Mariano (per-week loss; 음수 stat = a 가 우수). 반환 (stat, two-sided p)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    dbar = float(np.mean(d))
    var = float(np.var(d)) / n
    if var <= 0:
        return (0.0, 1.0) if abs(dbar) < 1e-12 else (math.copysign(1e6, dbar), 0.0)
    dm = dbar / math.sqrt(var)
    hln = dm * math.sqrt(max((n + 1 - 2*h + h*(h-1)/n) / n, 1e-9))
    p = _betainc((n - 1) / 2.0, 0.5, (n - 1) / ((n - 1) + hln * hln))
    return float(hln), float(p)


def bootstrap_ci_mean(diff, n_boot=2000, seed=0, level=0.95) -> tuple[float, float, float]:
    """paired bootstrap mean(diff) CI. 반환 (mean, lo, hi)."""
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(d.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def fit_nb_dispersion(y, mu, grid=(0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200)) -> float:
    """관측 ILI y 와 ABM 평균 μ 로 NegBin φ 최우도 추정 (grid). 예측구간 보정용."""
    from simulation.abm.observation_model import negbin_loglik
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    best = (float("-inf"), grid[len(grid) // 2])
    for phi in grid:
        ll = negbin_loglik(y, mu, float(phi))
        if math.isfinite(ll) and ll > best[0]:
            best = (ll, float(phi))
    return float(best[1])


def observation_ensemble(mu, phi: float, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """ABM 평균 μ 둘레로 NegBin(μ,φ) 예측 앙상블 (n_draws, T). 관측노이즈 fold —
    seed-변동과 달리 n_agents 와 무관하게 calibrated 구간을 준다."""
    mu = np.maximum(np.asarray(mu, float), 0.0)
    T = mu.shape[0]
    reps = np.zeros((n_draws, T))
    pos = mu > 0
    phi = max(float(phi), 1e-6)
    for d in range(n_draws):
        lam = np.zeros(T)
        if np.any(pos):
            lam[pos] = rng.gamma(shape=phi, scale=mu[pos] / phi)
        reps[d] = rng.poisson(lam)
    return reps


def validate_arms_calibrated(y_real, mu_adaptive, mu_static, *, n_draws=500, seed=0,
                             threshold: float = 8.6, n_boot=1000) -> dict:
    """관측노이즈 fold된 SCI 검증 — ABM 평균 μ → NegBin(φ̂) 예측 앙상블 → validate_arms.

    seed-변동 앙상블의 coverage 붕괴(n↑→구간 0) 문제를 해결: φ 를 real ILI 에 최우도
    적합해 calibrated 구간을 만든 뒤 비교. (각 arm 자기 φ̂ — FluSight식 self-submitted PI.)
    반환: validate_arms + {"dispersion": {phi_adaptive, phi_static}}.
    """
    y = np.asarray(y_real, float).ravel()
    ma = np.asarray(mu_adaptive, float).ravel()
    ms = np.asarray(mu_static, float).ravel()
    rng = np.random.default_rng(seed)
    phi_a = fit_nb_dispersion(y, ma)
    phi_s = fit_nb_dispersion(y, ms)
    reps_a = observation_ensemble(ma, phi_a, n_draws, rng)
    reps_s = observation_ensemble(ms, phi_s, n_draws, rng)
    out = validate_arms(y, reps_a, reps_s, n_boot=n_boot, seed=seed, threshold=threshold)
    out["dispersion"] = {"phi_adaptive": phi_a, "phi_static": phi_s}
    out["ensemble"] = "observation-folded NegBin (n_agents-stable, calibrated)"
    return out


def validate_arms(y_real, reps_adaptive, reps_static, *, n_boot=2000, seed=0,
                  threshold: float = 8.6) -> dict:
    """adaptive vs static 를 real ILI 로 SCI-급 비교 (전 지표 + AUC-ROC·C-index + 유의성 + 판정).

    threshold: 이상유행(outbreak) 기준 (KDCA 8.6 기본). AUC-ROC 의 binary event = (y > threshold).
    """
    ra = _as_reps(reps_adaptive, "reps_adaptive")
    rs = _as_reps(reps_static, "reps_static")
    if ra.shape[1] != rs.shape[1]:
        raise ValueError(f"T mismatch: adaptive {ra.shape[1]} vs static {rs.shape[1]}")
    y = _as_y(y_real, ra.shape[1])

    # 기존 SSOT 구현 재사용 (중복 회피, D-4): simulation.analytics.metrics
    from simulation.analytics.metrics import c_index as _c_index, roc_auc as _roc_auc
    outbreak = (y > float(threshold)).astype(np.float64)   # AUC-ROC binary event

    wis_a, wis_s = wis_per_week(y, ra), wis_per_week(y, rs)
    crps_a, crps_s = crps_per_week(y, ra), crps_per_week(y, rs)
    pm_a, pm_s = point_metrics(y, ra), point_metrics(y, rs)

    def arm(reps, wis, crps, pm):
        med = np.median(reps, axis=0)
        return {"wis": float(np.mean(wis)), "crps": float(np.mean(crps)),
                "rmse": pm["rmse"], "mae": pm["mae"], "r2": pm["r2"],
                "coverage95": pi_coverage(y, reps, 0.05),
                "coverage50": pi_coverage(y, reps, 0.50),
                "c_index": float(_c_index(y, med)),          # Harrell 순위 일치 (concordance)
                "auc_roc": float(_roc_auc(outbreak, med))}   # 이상유행(>θ) discrimination

    dwis = wis_a - wis_s                      # per-week ΔWIS (음수 = adaptive 우수)
    dm_stat, dm_p = dm_hln(wis_a, wis_s)      # adaptive vs static
    bmean, blo, bhi = bootstrap_ci_mean(dwis, n_boot=n_boot, seed=seed)
    significant = bool(bmean < 0 and dm_p < 0.05 and bhi < 0)

    return {
        "n_weeks": int(len(y)),
        "adaptive": arm(ra, wis_a, crps_a, pm_a),
        "static": arm(rs, wis_s, crps_s, pm_s),
        "significance": {
            "delta_wis_mean": float(np.mean(dwis)),
            "rel_wis_improvement": float((np.mean(wis_s) - np.mean(wis_a)) / np.mean(wis_s))
            if np.mean(wis_s) > 0 else 0.0,
            "dm_hln_stat": dm_stat, "dm_p_value": dm_p,
            "bootstrap_dwis_mean": bmean, "bootstrap_ci95_lo": blo, "bootstrap_ci95_hi": bhi,
            "adaptive_significantly_better": significant,
        },
        "verdict": ("adaptive 유의 우수 (ΔWIS<0, DM p<0.05, bootstrap CI<0)" if significant
                    else "유의차 없음 — 더 긴 season/seed 필요 또는 효과 미미"),
    }
