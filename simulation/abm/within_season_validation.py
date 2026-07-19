"""simulation.abm.within_season_validation

**Within-season** adaptive 검증 — 같은 시즌의 **early 주차로 보정 → late 주차로 held-out 평가**.

cross-season(2023 보정→2024 평가)은 행동이 transfer 안 돼 over-damping → adaptive 패배였다.
within-season 은 행동이 **그 시즌 자체 dynamics 에 적응**하므로 adaptive 가 유의하게 이길
legitimate 경로 (operational forecast setup; late 주차는 보정에 미사용 → winner's curse 없음).

절차
----
1. 한 시즌 ILI (T주) 로드. early=[:t_split](보정), late=[t_split:](held-out 평가).
2. β(R0) grid: full-season ABM(behaviour-off) → 주간 집계 → **early 주차 affine+WIS** 최소 β 선택.
3. 선택 β 에서 adaptive(behaviour-on) vs static(behaviour-off) full-season 실행 → 주간 μ.
4. 각 arm μ 를 **early 주차로 affine** 매핑 (late 는 hold-out).
5. **late 주차** 에서 SCI 검증 (validate_arms_calibrated: WIS/RMSE/MAE/CRPS/coverage/AUC/C-index
   + HLN-DM + bootstrap). adaptive 가 static 을 유의하게 이기나?

Gray-box: run_coupled_abm(결정적 metapop ODE) 사용 → grid 빠름. 부작용: DB read only.
"""
from __future__ import annotations


import numpy as np

from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
from simulation.abm.behavior_disease_eval import build_demo_metapop
from simulation.abm.behavior_disease_validation import validate_arms_calibrated, wis_per_week
from simulation.abm.observation_model import ObservationParams, ili_mean


def load_season_ili(db_path, season: int | None = None) -> tuple[np.ndarray, int]:
    """sentinel_influenza 전연령 ILI 시계열 (한 시즌, 주차순). 반환 (ili, season)."""
    from simulation.database.storage import read_only_connect
    con = read_only_connect(db_path)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(sentinel_influenza)")]
        age = "전체" if "age_group" in cols else None
        seasons = [r[0] for r in con.execute(
            "SELECT DISTINCT season_start FROM sentinel_influenza "
            "WHERE ili_rate IS NOT NULL ORDER BY season_start")]
        if not seasons:
            raise ValueError("sentinel_influenza 에 ili_rate 없음")
        season = int(season) if season is not None else int(seasons[-1])
        q = ("SELECT week_seq, ili_rate FROM sentinel_influenza "
             "WHERE season_start=? AND ili_rate IS NOT NULL")
        params = [season]
        if age is not None:
            agevals = [r[0] for r in con.execute(
                "SELECT DISTINCT age_group FROM sentinel_influenza WHERE season_start=?", [season])]
            pick = "전체" if "전체" in agevals else (agevals[0] if agevals else None)
            if pick is not None:
                q += " AND age_group=?"
                params.append(pick)
        q += " ORDER BY week_seq"
        rows = con.execute(q, params).fetchall()
        ili = np.array([float(r[1]) for r in rows], dtype=np.float64)
        return ili, season
    finally:
        con.close()


def _weekly_from_daily(city_daily: np.ndarray, n_weeks: int) -> np.ndarray:
    """일일 city 발생 → 주간 합 (앞 n_weeks*7 일)."""
    need = n_weeks * 7
    d = np.asarray(city_daily, float)[:need]
    if d.size < need:
        d = np.pad(d, (0, need - d.size))
    return d.reshape(n_weeks, 7).sum(axis=1)


def _fit_affine(sim: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    """offset+scale 최소제곱 (scale≥0). 반환 (offset, scale)."""
    s = np.asarray(sim, float); r = np.asarray(real, float)
    vs = float(np.var(s))
    if vs <= 1e-12:
        return float(np.mean(r)), 0.0
    scale = max(float(np.cov(s, r, ddof=0)[0, 1] / vs), 0.0)
    offset = float(np.mean(r) - scale * np.mean(s))
    return offset, scale


def _arm_weekly(metapop, behaviour, n_weeks, obs: ObservationParams) -> np.ndarray:
    """full-season run → 주간 기대 ILI μ (관측모형 평균)."""
    res = run_coupled_abm(metapop, behaviour)
    city_daily = res.seir.incidence.sum(axis=1)
    weekly_inf = _weekly_from_daily(city_daily, n_weeks)
    return ili_mean(obs.symptomatic_frac * weekly_inf, obs)


def run_within_season(db_path, season: int | None = None, *, cal_frac: float = 0.6,
                      r0_grid=(1.2, 1.4, 1.6, 1.8, 2.0, 2.4), G: int = 3,
                      obs: ObservationParams | None = None) -> dict:
    """within-season adaptive vs static 검증. early 보정 → late held-out SCI."""
    obs = obs or ObservationParams()
    ili, season = load_season_ili(db_path, season)
    T = len(ili)
    if T < 10:
        raise ValueError(f"season {season} 주차 부족 (T={T})")
    t_split = max(int(round(cal_frac * T)), 4)
    early, late = slice(0, t_split), slice(t_split, T)
    days = T * 7

    # 1) β(R0) 보정 — early 주차 WIS 최소 (behaviour-off baseline)
    off = BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf"))
    best = None
    for r0 in r0_grid:
        mp = build_demo_metapop(G=G, days=days, R0=float(r0), seed=0)
        mu = _arm_weekly(mp, off, T, obs)
        a_off, s_off = _fit_affine(mu[early], ili[early])
        mapped_early = a_off + s_off * mu[early]
        w = float(np.mean(wis_per_week(ili[early], mapped_early[None, :])))
        if best is None or w < best[0]:
            best = (w, float(r0))
    best_wis_early, best_r0 = best

    # 2) 선택 β 에서 adaptive vs static full-season
    mp = build_demo_metapop(G=G, days=days, R0=best_r0, seed=0)
    mu_adapt = _arm_weekly(mp, BehaviouralParams(), T, obs)          # behaviour-on
    mu_static = _arm_weekly(mp, off, T, obs)                          # behaviour-off

    # 3) early 주차로 affine 매핑 (late 는 hold-out)
    def map_arm(mu):
        o, sc = _fit_affine(mu[early], ili[early])
        return o + sc * mu
    mu_adapt_m, mu_static_m = map_arm(mu_adapt), map_arm(mu_static)

    # 4) late 주차 held-out SCI 검증
    sci = validate_arms_calibrated(ili[late], mu_adapt_m[late], mu_static_m[late],
                                   n_draws=500, threshold=float(np.median(ili)))
    return {
        "season": season, "T_weeks": T, "t_split": t_split,
        "cal_weeks": t_split, "eval_weeks": T - t_split,
        "calibration": {"best_r0": best_r0, "early_wis": best_wis_early},
        "sci_validation": sci,
        "setup": "within-season (early 보정 → late held-out)",
    }
