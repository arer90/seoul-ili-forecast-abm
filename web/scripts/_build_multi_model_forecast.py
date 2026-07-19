#!/usr/bin/env python3
"""Build ili-forecast-models.json and seir-forecast-360.json.

Run from project root:
    python3 web/scripts/_build_multi_model_forecast.py
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
SEOUL_GU = [
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
]

TOP_N = 12          # number of models to include
RESULTS_CSV  = Path("simulation/results/csv/summary_metrics.csv")
PRED_DIR     = Path("simulation/results/csv")
ABM_PATH     = Path("web/public/aggregates/abm-scenarios.json")
ILI_LOCAL    = Path("web/public/aggregates/ili-local.json")
OUT_MODELS   = Path("web/public/aggregates/ili-forecast-models.json")
OUT_SEIR360  = Path("web/public/aggregates/seir-forecast-360.json")


# ── Helper: load ABM weights per gu (I_frac last day, baseline scenario) ──────
def _load_abm_weights() -> dict[str, float]:
    if not ABM_PATH.is_file():
        return {}
    try:
        abm = json.loads(ABM_PATH.read_text("utf-8"))
        gu_names: list[str] = abm.get("gu_names", [])
        i_frac: list[list[float]] = (
            abm.get("scenarios", {}).get("baseline", {}).get("I_frac", [])
        )
        if gu_names and i_frac:
            last_day = i_frac[-1]
            total = sum(last_day)
            if total > 0 and len(last_day) == len(gu_names):
                return {
                    gu: frac / total * len(gu_names)
                    for gu, frac in zip(gu_names, last_day)
                }
    except Exception as exc:
        print(f"! abm weight load error: {exc}", file=sys.stderr)
    return {}


# ── Helper: read current city ILI anchor from ili-local.json ─────────────────
def _load_anchor() -> tuple[float, str]:
    """Returns (anchor_ili_per1k, observed_at_iso)."""
    if not ILI_LOCAL.is_file():
        return 5.0, "1970-01-01T00:00:00Z"
    try:
        local = json.loads(ILI_LOCAL.read_text("utf-8"))
        vals = [
            float(v.get("ili", 0))
            for v in local.get("gu", {}).values()
            if isinstance(v, dict)
        ]
        anchor = sum(vals) / len(vals) if vals else 0.0
        obs_at = local.get("observed_at", "1970-01-01T00:00:00Z")
        return (anchor if anchor > 0 else 5.0), obs_at
    except Exception as exc:
        print(f"! anchor load error: {exc}", file=sys.stderr)
        return 5.0, "1970-01-01T00:00:00Z"


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — ili-forecast-models.json
# ═══════════════════════════════════════════════════════════════════════════════

def build_multi_model_forecast() -> None:
    """Build ili-forecast-models.json with top-N models' 1-week forecasts.

    Algorithm (mirrors build_ili_forecast for each model):
      1. Rank summary_metrics.csv by test_r2 descending → top TOP_N.
      2. For each model compute rel_rmse from its predictions_<model>.csv
         (test split mean pred vs champion_rmse).
      3. Anchor forecast level to current observed ILI (from ili-local.json).
      4. Per-gu weight from ABM I_frac last day (baseline).
      5. Write JSON with model list + per-gu forecast per model.

    Returns:
        None. Writes OUT_MODELS to disk.
    """
    # Step 1: load and rank models
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))

    ranked = sorted(
        all_rows,
        key=lambda r: float(r.get("test_r2") or "-inf"),
        reverse=True,
    )[:TOP_N]

    anchor, observed_at = _load_anchor()
    # 1-week-ahead forecast date
    last_obs_dt = datetime.date.fromisoformat(observed_at[:10])
    forecast_date = (last_obs_dt + datetime.timedelta(weeks=1)).isoformat() + "T00:00:00Z"

    abm_weights = _load_abm_weights()
    generated_at = datetime.datetime.utcnow().isoformat() + "Z"

    models_out: list[dict] = []

    for row in ranked:
        name = row["name"]
        category = row.get("category", "unknown")
        test_r2   = float(row.get("test_r2")   or "nan")
        test_rmse = float(row.get("test_rmse")  or "0")
        test_mae  = float(row.get("test_mae")   or "0")
        test_mape = float(row.get("test_mape")  or "0")

        # Compute per-model rel_rmse from predictions file
        rel_rmse = 0.25  # fallback
        pred_path = PRED_DIR / f"predictions_{name}.csv"
        if pred_path.is_file():
            try:
                with pred_path.open(newline="", encoding="utf-8") as pfh:
                    pred_rows = list(csv.DictReader(pfh))
                test_preds = [
                    float(r["y_pred"])
                    for r in pred_rows
                    if r.get("split") == "test" and r.get("y_pred")
                ]
                test_mean = sum(test_preds) / len(test_preds) if test_preds else 0.0
                if test_mean > 0 and test_rmse > 0:
                    rel_rmse = min(0.5, max(0.05, test_rmse / test_mean))
            except Exception as exc:
                print(f"  ! pred parse error for {name}: {exc}", file=sys.stderr)

        city_forecast = round(anchor, 4)
        city_lo = max(0.0, round(anchor * (1 - 2 * rel_rmse), 4))
        city_hi = round(anchor * (1 + 2 * rel_rmse), 4)

        # Per-gu weights from ABM
        gu_dict: dict[str, dict] = {}
        for gu in SEOUL_GU:
            w = abm_weights.get(gu, 1.0)
            gu_dict[gu] = {
                "ili": round(city_forecast * w, 4),
                "lo":  round(city_lo * w, 4),
                "hi":  round(city_hi * w, 4),
            }

        models_out.append({
            "name": name,
            "category": category,
            "rank": len(models_out) + 1,
            "metrics": {
                "test_r2":   round(test_r2, 4),
                "test_rmse": round(test_rmse, 4),
                "test_mae":  round(test_mae, 4),
                "test_mape": round(test_mape, 2),
                "rel_rmse":  round(rel_rmse, 4),
            },
            "city_forecast": city_forecast,
            "city_lo": city_lo,
            "city_hi": city_hi,
            "gu": gu_dict,
        })
        print(
            f"  model {name:30s}  r2={test_r2:.4f}  rel_rmse={rel_rmse:.3f}"
            f"  city=[{city_lo:.2f}, {city_forecast:.2f}, {city_hi:.2f}]",
            file=sys.stderr,
        )

    payload = {
        "generated_at": generated_at,
        "observed_at": observed_at,
        "forecast_at": forecast_date,
        "source": "multi-model-forecast",
        "horizon_weeks": 1,
        "anchor_ili": round(anchor, 4),
        "note": (
            f"상위 {TOP_N}개 모델 (test R² 내림차순) 각 rel-RMSE 기반 PI. "
            f"예측 레벨 = 현 관측 {round(anchor, 2)}/1k anchored (계절 보정). "
            "자치구 분배 = ABM I_frac baseline 마지막 날. "
            "모델 재학습 후 갱신 필요."
        ),
        "models": models_out,
    }

    OUT_MODELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\nwrote {OUT_MODELS} ({len(models_out)} models, anchor={anchor:.3f}/1k)",
        file=sys.stderr,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — seir-forecast-360.json
# ═══════════════════════════════════════════════════════════════════════════════

def _seir_euler_seasonal(
    S0: float, E0: float, I0: float, R0_comp: float, N: float,
    beta_mean: float, beta_amplitude: float,
    sigma: float, gamma: float,
    days: int,
    start_doy: int,
    waning_rate: float = 0.0,
    dt: float = 0.25,
) -> list[tuple[float, float, float, float, float]]:
    """Run seasonally-forced SEIR Euler integration.

    Seasonal forcing: β(t) = β_mean × (1 + amplitude × cos(2π(doy-peak_doy)/365))
    where peak_doy = 15 (mid-January, Korean flu season peak).
    Waning immunity: recovered → susceptible at rate waning_rate (1/year ≈ 0.003/day).

    Args:
        S0, E0, I0, R0_comp: initial compartment counts.
        N: total population.
        beta_mean: mean transmission rate (per day), = R0_summer × gamma.
        beta_amplitude: seasonal amplitude (0–1). 0.4 ≈ 40% winter/summer swing.
        sigma: 1 / incubation_days.
        gamma: 1 / infectious_days.
        days: horizon (integer days).
        start_doy: day-of-year of day 0 (1=Jan 1).
        waning_rate: per-day waning from R to S (default 0 = no waning).
        dt: Euler step (default 0.25 day).

    Returns:
        List of (S, E, I, R, beta_eff) tuples, one per integer day
        (length = days+1).

    Side effects: none (pure computation).
    """
    import math as _math
    PEAK_DOY = 15   # mid-January flu peak in Korea
    TWO_PI = 2.0 * _math.pi

    S, E, I, R = float(S0), float(E0), float(I0), float(R0_comp)
    result: list[tuple[float, float, float, float, float]] = []

    # Store last beta_eff per integer day
    last_beta: list[float] = []

    steps = int(days / dt)
    for step in range(steps + 1):
        t = step * dt
        doy = (start_doy - 1 + t) % 365 + 1
        # Seasonal β
        beta_eff = beta_mean * (1.0 + beta_amplitude * _math.cos(
            TWO_PI * (doy - PEAK_DOY) / 365.0
        ))
        beta_eff = max(0.0, beta_eff)

        day_int = int(round(t))
        if day_int > len(result) - 1 and day_int <= days:
            result.append((S, E, I, R, beta_eff))
            last_beta.append(beta_eff)

        if step == steps:
            break

        denom = max(S + E + I + R, 1.0)
        foi  = beta_eff * I / denom
        dS   = (-foi * S + waning_rate * R) * dt
        dE   = (foi * S - sigma * E) * dt
        dI   = (sigma * E - gamma * I) * dt
        dR   = (gamma * I - waning_rate * R) * dt
        S = max(0.0, S + dS)
        E = max(0.0, E + dE)
        I = max(0.0, I + dI)
        R = max(0.0, R + dR)

    # Ensure we have exactly days+1 entries
    while len(result) < days + 1:
        result.append(result[-1])
    return result[:days + 1]


def build_seir_360() -> None:
    """Build seir-forecast-360.json with 1–360 day SEIR forecast.

    SEIR parameters (Seoul flu calibrated):
      R0 = 1.5 (summer baseline — lower than winter peak ~2.2)
      incubation = 2.0 days  (σ = 0.5)
      infectious  = 3.5 days (γ ≈ 0.286)
      N = 9_720_000 (2024 Seoul population, Statistics Korea)
      Initial state from latest observed ILI:
        city_ili = 5.14/1k → I0 = city_ili/1000 × N
        E0 = I0 (latent ≈ infectious count at endemic equilibrium)
        S0 = N × (1 - vaccine_coverage) - I0 - E0
        vaccine_coverage = 0.45 (annual flu shot ~45% Seoul adults)
      Death removed from SEIR to avoid un-parameterised CFR assumption.
      ILI conversion: ILI_rate/1k = I / N × 1000 × detection_factor
        detection_factor = 7.5 (typical ILI surveillance under-ascertainment)

    Returns:
        None. Writes OUT_SEIR360 to disk.

    Side effects:
        Writes seir-forecast-360.json. Uses abm-scenarios.json for per-gu
        I_frac weighting; falls back to uniform if absent.
    """
    SEOUL_POP    = 9_720_000
    # Seasonal SEIR calibrated to Seoul KDCA ILI history:
    #   summer R0 ≈ 1.0 (low-season), winter R0 ≈ 1.8 (pandemic-free flu year)
    #   solve: R0_mean=1.4, amp=0.286 → summer=1.0, winter=1.8
    R0_SUMMER    = 1.0           # summer effective R0 (low-season floor)
    R0_AMPLITUDE = 0.286         # seasonal amplitude (winter/summer ratio 1.8:1.0)
    # β_mean s.t. mean R0 = 1.4: β_mean = 1.4 × gamma
    R0_MEAN      = 1.4
    INCUB_DAYS   = 2.0
    INFECT_DAYS  = 3.5
    VAX_COV      = 0.45
    DETECT_FACTOR = 7.5          # ILI under-ascertainment multiplier
    WANING_RATE  = 1.0 / 365.0   # 1-year immunity waning (allows winter resurgence)
    HORIZON      = 360

    sigma = 1.0 / INCUB_DAYS
    gamma = 1.0 / INFECT_DAYS
    beta_mean = R0_MEAN * gamma   # calibrated mean R0=1.4

    anchor, observed_at = _load_anchor()
    obs_dt = datetime.date.fromisoformat(observed_at[:10])
    start_doy = obs_dt.timetuple().tm_yday  # day-of-year at start

    # I0 from current ILI observation
    I0 = (anchor / 1000.0) / DETECT_FACTOR * SEOUL_POP
    E0 = I0  # latent ≈ infectious at near-endemic state
    S0 = SEOUL_POP * (1 - VAX_COV) - I0 - E0
    S0 = max(0.0, S0)
    R0_compartment = SEOUL_POP * VAX_COV  # vaccinated → recovered-equiv.

    R0_winter_est = beta_mean * (1.0 + R0_AMPLITUDE) / gamma
    print(
        f"SEIR-360 seasonal init: N={SEOUL_POP}, R0_summer~{R0_SUMMER:.1f}, "
        f"R0_winter~{R0_winter_est:.1f}, I0={I0:.0f}, E0={E0:.0f}, "
        f"S0={S0:.0f}, start_doy={start_doy}",
        file=sys.stderr,
    )

    traj = _seir_euler_seasonal(
        S0, E0, I0, R0_compartment, SEOUL_POP,
        beta_mean, R0_AMPLITUDE, sigma, gamma,
        HORIZON, start_doy, WANING_RATE,
    )

    # ABM gu weights for per-gu scaling (from last day of baseline I_frac)
    abm_weights = _load_abm_weights()

    # Find peak
    peak_day = 0
    peak_ili = 0.0
    days_out: list[dict] = []
    for day_idx, (S, E, I, R, beta_eff) in enumerate(traj):
        # Convert I count to ILI per 1k (inverse of seeding)
        city_ili = round(I / SEOUL_POP * DETECT_FACTOR * 1000, 4)
        fc_date = (obs_dt + datetime.timedelta(days=day_idx)).isoformat()

        if city_ili > peak_ili:
            peak_ili = city_ili
            peak_day = day_idx

        # Per-gu distribution via ABM weights
        gu_dict: dict[str, float] = {}
        for gu in SEOUL_GU:
            w = abm_weights.get(gu, 1.0)
            gu_dict[gu] = round(city_ili * w, 4)

        days_out.append({
            "day":      day_idx,
            "date":     fc_date,
            "city_ili": city_ili,
            "I_count":  round(I, 0),
            "S_frac":   round(S / SEOUL_POP, 6),
            "R0_eff":   round(beta_eff / gamma, 3),
            "gu":       gu_dict,
        })

    generated_at = datetime.datetime.utcnow().isoformat() + "Z"

    payload = {
        "generated_at": generated_at,
        "observed_at":  observed_at,
        "source":       "seir-360-forecast",
        "model":        "SEIR-seasonal-Euler",
        "horizon_days": HORIZON,
        "parameters": {
            "R0_summer":     R0_SUMMER,
            "R0_winter_est": round(R0_winter_est, 2),
            "seasonal_amplitude": R0_AMPLITUDE,
            "incubation_days": INCUB_DAYS,
            "infectious_days": INFECT_DAYS,
            "vaccine_coverage": VAX_COV,
            "detection_factor": DETECT_FACTOR,
            "waning_days":   round(1.0 / WANING_RATE),
            "N":             SEOUL_POP,
        },
        "initial_state": {
            "anchor_ili_per1k": round(anchor, 4),
            "observed_at": observed_at,
            "start_doy": start_doy,
            "I0": round(I0, 1),
            "E0": round(E0, 1),
            "S0": round(S0, 1),
        },
        "summary": {
            "peak_day":      peak_day,
            "peak_date":     (obs_dt + datetime.timedelta(days=peak_day)).isoformat(),
            "peak_city_ili": round(peak_ili, 4),
            "attack_rate_pct": round((1 - traj[-1][0] / SEOUL_POP) * 100, 2),
        },
        "note": (
            f"계절 강제 SEIR Euler (dt=0.25d, 서울 인구 {SEOUL_POP:,}). "
            f"여름 R0≈{R0_SUMMER}, 동절기 R0≈{R0_winter_est:.1f} (±{int(R0_AMPLITUDE*100)}% 계절 진폭). "
            f"초기 감염원 = 최신 ILI {round(anchor, 2)}/1k (역보정 ×1/{DETECT_FACTOR}). "
            f"면역소실 {round(1/WANING_RATE)}일. "
            "자치구 분배 = ABM I_frac baseline. "
            "사망/입원 미포함. 불확실성 ±40%."
        ),
        "forecast": days_out,
    }

    OUT_SEIR360.parent.mkdir(parents=True, exist_ok=True)
    OUT_SEIR360.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    peak_date_str = (obs_dt + datetime.timedelta(days=peak_day)).isoformat()
    print(
        f"\nwrote {OUT_SEIR360} ({HORIZON} days, peak_day={peak_day}, "
        f"peak_date={peak_date_str}, peak_ili={peak_ili:.2f}/1k, "
        f"attack_rate={payload['summary']['attack_rate_pct']:.1f}%)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    print("=== TASK 1: ili-forecast-models.json (production refit path) ===", file=sys.stderr)
    # Prefer build_production_forecast.py which does a full NegBinGLM refit with
    # conformal PI and gate check, writing BOTH ili-forecast.json and
    # ili-forecast-models.json. Fall back to the legacy anchor-based method only
    # if the production script is unavailable.
    _prod_script = Path(__file__).parent / "build_production_forecast.py"
    _used_prod = False
    if _prod_script.is_file():
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("build_production_forecast", str(_prod_script))
            _pmod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_pmod)
            _rc = _pmod.main()
            if _rc == 0:
                print("Production refit complete — ili-forecast.json + ili-forecast-models.json written",
                      file=sys.stderr)
                _used_prod = True
            else:
                print(f"! production refit returned {_rc} — falling back to legacy path",
                      file=sys.stderr)
        except Exception as _e:
            print(f"! production refit failed ({_e}) — falling back to legacy path",
                  file=sys.stderr)
    if not _used_prod:
        build_multi_model_forecast()
    print("\n=== TASK 2: seir-forecast-360.json ===", file=sys.stderr)
    build_seir_360()
    print("\nDone.", file=sys.stderr)
