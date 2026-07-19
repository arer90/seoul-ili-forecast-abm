"""Pillar-1 proof infrastructure: the OBSERVED behavioral signal + dose-response.

The behavioral ABM's contact-reduction function is *latent* in the ILI fit but
*directly observed* in transit ridership. This module builds the empirical target
the simulation must predict (double-validation, PROOF_VALIDATION_PROTOCOL §1.8):

  observed behavioral response  =  deseasonalised mobility deviation  vs
                                   perceived prevalence (COVID 2020 / ILI rebound).

Regimes (epi-advisor — never pool): R0 pre-COVID (clean baseline), R1 2020-03→
2022-09 (NPI+collapse, descriptive only), R2 2022-10→ (post-relaxation, voluntary).
The COVID driver (`data/external/covid_kr_who.csv`, WHO) is the 2020 perceived-
prevalence — its Mar-2020 spike (9,027) aligns with the 43% subway drop. The match
statistic is the dose-response slope/magnitude, NOT raw time-series correlation
(both trend with the epidemic → spurious).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_R1_START, _R2_START = "2020-03", "2022-10"   # regime boundaries (YYYY-MM)


def _regime(ym: str) -> str:
    if ym < _R1_START:
        return "R0"
    if ym < _R2_START:
        return "R1"
    return "R2"


def load_covid_kr_monthly(csv_path: str) -> dict[str, int]:
    """WHO Korea weekly COVID → monthly new_cases. ``{YYYY-MM: cases}``."""
    out: dict[str, int] = {}
    p = Path(csv_path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ym = (r.get("Date_reported") or "")[:7]
            if len(ym) == 7:
                try:
                    out[ym] = out.get(ym, 0) + int(r.get("New_cases") or 0)
                except ValueError:
                    continue
    return out


def load_monthly_mobility(db_path: str) -> dict[str, float]:
    """Monthly total subway ridership → ``{YYYY-MM: ride_cnt}`` (the 86-mo asset;
    NOT daily_subway). use_ym 'YYYYMM' → 'YYYY-MM'."""
    out: dict[str, float] = {}
    try:
        from simulation.database import read_only_connect  # lock-free analytics read
        con = read_only_connect(db_path)
        try:
            for ym, tot in con.execute(
                "SELECT use_ym, SUM(ride_cnt) FROM monthly_subway_hourly GROUP BY use_ym"
            ).fetchall():
                s = str(ym)
                if len(s) == 6:
                    out[f"{s[:4]}-{s[4:]}"] = float(tot or 0)
        finally:
            con.close()
    except Exception:
        return {}
    return out


def build_behavior_panel(db_path: str, covid_csv: str) -> list[dict]:
    """Aligned monthly panel with a DESEASONALISED mobility deviation + driver.

    ``mobility_dev`` = mobility / (R0 same-calendar-month baseline) − 1, so the
    annual mobility cycle is removed and the deviation reflects *behavioral*
    reduction. Each row: ``{ym, regime, mobility, mobility_dev, covid}``. Rows
    without a usable R0 baseline for that calendar month are dropped. Never raises.
    """
    mob = load_monthly_mobility(db_path)
    covid = load_covid_kr_monthly(covid_csv)
    # R0 baseline = mean mobility per calendar month over the pre-COVID regime
    base: dict[str, list[float]] = {}
    for ym, v in mob.items():
        if _regime(ym) == "R0":
            base.setdefault(ym[5:], []).append(v)
    baseline = {mm: float(np.mean(vs)) for mm, vs in base.items() if vs}
    panel: list[dict] = []
    for ym in sorted(mob):
        mm = ym[5:]
        if mm not in baseline or baseline[mm] <= 0:
            continue
        panel.append({
            "ym": ym, "regime": _regime(ym), "mobility": mob[ym],
            "mobility_dev": mob[ym] / baseline[mm] - 1.0,
            "covid": covid.get(ym, 0),
        })
    return panel


def behavioral_decomposition(panel: list[dict], regimes: tuple[str, ...] = ("R1",)) -> dict:
    """Separate the RISK response from FATIGUE in the observed mobility — the model's
    two mechanisms, empirically.

    A naive single-variable dose-response is the WRONG statistic here: the
    mobility–prevalence relationship is **hysteretic** (path-dependent), because
    the behavioral response *wanes over time* (fatigue) even as prevalence rises —
    e.g. 2020-04 (−37% at 1.1k cases) vs 2022-04 (−22% at 4.9M cases). The naive
    slope comes out POSITIVE and misses the mechanism. The correct decomposition is
    a 2-variable OLS that separates the two model mechanisms:

        mobility_dev  ~  β_risk·log1p(cases)  +  β_fatigue·t

    - ``β_risk < 0``    : contact falls as perceived prevalence rises (risk
      perception, the model's α) — holding time fixed.
    - ``β_fatigue > 0`` : at fixed prevalence the response erodes over time
      (fatigue, the model's F/τ) — mobility recovers.

    BOTH present (β_risk<0 AND β_fatigue>0) ⇒ the behavioral mechanism is observed
    in independent data, not just in the model. Returns the coefficients, partial
    correlations, n, and a verdict. Never raises.
    """
    rows = [r for r in panel if r["regime"] in regimes and np.isfinite(r["mobility_dev"])]
    if len(rows) < 5:
        return {"error": f"only {len(rows)} rows in regimes {regimes}"}
    cases = np.log1p(np.array([r["covid"] for r in rows], dtype=np.float64))
    y = np.array([r["mobility_dev"] for r in rows], dtype=np.float64)
    t = np.arange(len(rows), dtype=np.float64)
    if cases.std() == 0 or t.std() == 0:
        return {"error": "no driver / time variation"}
    # standardise predictors so coefficients are comparable; OLS via lstsq
    Xc = (cases - cases.mean()) / cases.std()
    Xt = (t - t.mean()) / t.std()
    X = np.column_stack([Xc, Xt, np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b_risk, b_fatigue = float(beta[0]), float(beta[1])
    # naive single-variable slope (the WRONG statistic — reported to show the trap)
    naive = float(np.polyfit(cases, y, 1)[0])
    # ★ HONESTY (2026-06-06): this OBSERVATIONAL decomposition does NOT identify the
    # mechanisms. The time-trend (β_fatigue) is collinear with COVID vaccination
    # (ρ≈0.91 over R1) → "fatigue" is NOT separable from "vaccination reassurance",
    # and β_risk is sign-UNSTABLE across control specifications (time vs vax). So a
    # declining-response pattern is *present and consistent with* WHO-documented
    # pandemic fatigue, but the raw regression is NOT proof. The proof must come
    # from the MODEL: out-of-sample behavior-ON vs OFF predictive gain, the
    # dynamical signatures (hysteresis etc.), and identifiability — see
    # confounding_check() and PROOF_VALIDATION_PROTOCOL §1.5-1.8.
    verdict = ("declining-response pattern present (β_risk={:+.3f}, β_fatigue={:+.3f}) "
               "but CONFOUNDED — not a standalone proof; run confounding_check() + "
               "validate via the model (signatures / out-of-sample)").format(b_risk, b_fatigue)
    return {
        "beta_risk": round(b_risk, 4), "beta_fatigue": round(b_fatigue, 4),
        "naive_pooled_slope": round(naive, 4), "n": len(rows),
        "regimes": list(regimes), "verdict": verdict,
        "caveat": "β_fatigue collinear with vaccination (see confounding_check); "
                  "observational, not identifying",
    }


def load_vax_kr_monthly(csv_path: str) -> dict[str, float]:
    """Korea COVID vaccination → monthly mean ``people_fully_vaccinated_per_hundred``.

    Domestic-official provenance: OWID sources Korea vaccination DIRECTLY from KDCA
    (질병관리청, ``ncv.kdca.go.kr``), so these are the official 질병관리청 figures
    routed through OWID — not a foreign estimate (standing rule: 공식데이터 기준)."""
    from collections import defaultdict
    acc: dict[str, list] = defaultdict(list)
    p = Path(csv_path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = r.get("people_fully_vaccinated_per_hundred")
            ym = (r.get("date") or "")[:7]
            if v and len(ym) == 7:
                try:
                    acc[ym].append(float(v))
                except ValueError:
                    continue
    return {k: float(np.mean(vs)) for k, vs in acc.items() if vs}


def confounding_check(panel: list[dict], vax_csv: str, regime: str = "R1") -> dict:
    """Honest identification check: is the time-trend ('fatigue') separable from
    COVID vaccination, and is the risk response sign-stable?

    Returns ``{time_vax_corr, beta_risk_time_model, beta_risk_vax_model,
    risk_sign_stable, separable, verdict}``. The defensible reading: a
    declining-response pattern consistent with pandemic fatigue, but the
    observational data does NOT separate fatigue from vaccination (collinear) nor
    robustly sign the risk response — so the PROOF must come from the model, not
    this regression. Never raises.

    Note (domestic data / 2026-06-06): "fatigue" has no clean monthly official
    series in ANY country — it is a survey construct (국내 공식 측정 = 서울대 보건
    대학원 코로나19 국민인식조사 / KDCA KCHS / 통계청 사회조사 / 한국갤럽, all periodic
    SNAPSHOTS). Crucially the collinearity is STRUCTURAL: any monotone-over-pandemic
    fatigue proxy is collinear with the (also monotone) vaccination coverage, so
    swapping WHO for a domestic series cannot break ρ≈0.91. Vaccination here is
    KDCA-origin (see load_vax_kr_monthly). See PROOF_VALIDATION_PROTOCOL §국내 공식 데이터.
    """
    vax = load_vax_kr_monthly(vax_csv)
    rows = [r for r in panel if r["regime"] == regime and r["ym"] in vax
            and np.isfinite(r["mobility_dev"])]
    if len(rows) < 6:
        return {"error": f"only {len(rows)} rows with vaccination in {regime}"}
    t = np.arange(len(rows), dtype=np.float64)
    vx = np.array([vax[r["ym"]] for r in rows], dtype=np.float64)
    cases = np.log1p(np.array([r["covid"] for r in rows], dtype=np.float64))
    y = np.array([r["mobility_dev"] for r in rows], dtype=np.float64)

    def _z(a):
        return (a - a.mean()) / a.std() if a.std() else a

    def _brisk(second):
        X = np.column_stack([_z(cases), _z(second), np.ones_like(y)])
        return float(np.linalg.lstsq(X, y, rcond=None)[0][0])

    tv_corr = float(np.corrcoef(t, vx)[0, 1])
    br_time, br_vax = _brisk(t), _brisk(vx)
    sign_stable = (br_time < 0) == (br_vax < 0)
    separable = abs(tv_corr) < 0.7
    verdict = (
        f"NOT separable: time≈vaccination (ρ={tv_corr:.2f}); risk-response sign "
        f"{'stable' if sign_stable else 'UNSTABLE'} across controls "
        f"(β_risk: time-model {br_time:+.3f} vs vax-model {br_vax:+.3f}). "
        "⇒ observational decomposition does NOT identify the behavioral mechanisms; "
        "proof requires the model (signatures + out-of-sample), not this regression."
    )
    return {
        "time_vax_corr": round(tv_corr, 4),
        "beta_risk_time_model": round(br_time, 4),
        "beta_risk_vax_model": round(br_vax, 4),
        "risk_sign_stable": bool(sign_stable), "separable": bool(separable),
        "n": len(rows), "verdict": verdict,
    }


# back-compat alias (the naive dose-response is retained as a teaching foil)
def observed_dose_response(panel: list[dict], regimes: tuple[str, ...] = ("R1",)) -> dict:
    """Deprecated: use :func:`behavioral_decomposition`. The pooled single-variable
    dose-response is hysteresis-confounded (see that function's docstring)."""
    return behavioral_decomposition(panel, regimes)
