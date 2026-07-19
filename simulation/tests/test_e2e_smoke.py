"""
simulation.tests.test_e2e_smoke
===============================
End-to-end smoke test (moved from simulation/pipeline_demo/e2e.py — Phase C5,
2026-05-12). Heavy: DB + SEIR + ABM + LLM. Pytest collects it but the
auto-collected `test_e2e_smoke_skipped` marker keeps CI fast — the real entry
remains `run_e2e()` invoked by `simulation/verify_all.py::check_e2e`.

End-to-end pipeline:

    1. Pull weekly ILI series from simulation/data/db/epi_real_seoul.db (25 gu).
    2. Forecast the next 4 weeks with a persistence + AR baseline
       (lightweight; the 66-model registry is not required to demonstrate
       the pipeline wiring).
    3. Run the metapop SEIR-V-D simulator for 120 days seeded from the
       most recent weekly state (behaviour-off baseline).
    4. Run the 4-parameter behavioural ABM counterfactual
       (simulation.abm.behavioural) with a post-COVID fatigue profile.
    5. Assemble a structured advisor question packet with concrete
       numbers (next-week ILI, peak shift %, compliance fraction).
    6. Invoke every enabled LLM backend (simulation.llm_compare) on the
       packet, score the responses with the 7-pillar rubric, and record
       the full exchange into a Hermes-style hash-chained audit log.
    7. Write a thesis-grade artefact set:
         valid_test/pipeline_demo/
             pipeline_summary.json         (numeric outputs)
             advisor_packet.json           (advisor prompts)
             llm_comparison.md/json        (LLM ranking table)
             audit_chain.jsonl             (Hermes log; append-only SHA-256)

CLI::

    python -m simulation.tests.test_e2e_smoke \
        --gu gangnam-gu --weeks 4 --horizon-days 120
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from simulation.abm import (
    BehaviouralParams,
    run_coupled_abm,
    run_invariant_test,
)
from simulation.database import safe_connect
from simulation.llm_compare import (
    discover_backends,
    load_golden_set,
    run_comparison,
)
from simulation.sim.io import load_metapop_params
from simulation.sim.metapop_seirvd import MetapopSEIRVD, SimResult
from simulation.sim.parameters import MetapopParams

log = logging.getLogger(__name__)

__all__ = [
    "run_e2e",
    "E2EResult",
    "retrieve_ili_from_db",
    "forecast_next_n_weeks",
    "run_seir_projection",
    "run_abm_counterfactual",
    "build_advisor_questions",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class E2EResult:
    """Everything the pipeline produced, suitable for JSON persistence."""
    gu: str
    generated_at: str
    db_snapshot: dict
    forecast: dict
    seir_baseline: dict
    abm_counterfactual: dict
    advisor_packet: dict
    llm_comparison: Optional[dict]
    audit_chain: list[dict]


# ---------------------------------------------------------------------------
# 1) DB -> weekly ILI
# ---------------------------------------------------------------------------
def retrieve_ili_from_db(
    *,
    gu: str = "강남구",
    lookback_weeks: int = 52,
) -> dict:
    """Pull the last ``lookback_weeks`` weekly ILI rates for a single gu
    from the sentinel_influenza table. Falls back gracefully if the
    table is absent on this machine.
    """
    with safe_connect() as con:
        cur = con.cursor()
        # probe available tables
        tables = {row[0] for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        candidate_tables = [
            "sentinel_influenza", "weekly_ili", "weekly_disease",
            "seoul_weekly_ili",
        ]
        chosen = next((t for t in candidate_tables if t in tables), None)
        if chosen is None:
            log.warning("no weekly ILI table found; falling back to synthetic series")
            return _synthetic_series(gu, lookback_weeks)
        cur.execute(
            f"SELECT name FROM pragma_table_info('{chosen}')"
        )
        cols = {r[0] for r in cur.fetchall()}
        # try to infer date / gu / value columns
        date_col = next((c for c in ("epi_week", "yearweek", "week_ending",
                                     "date", "yyyymmdd") if c in cols), None)
        gu_col = next((c for c in ("gu_nm", "district", "gu", "sgg_nm") if c in cols), None)
        rate_col = next((c for c in ("ili_rate", "rate", "rate_per_1000",
                                     "incidence", "value") if c in cols), None)
        if date_col is None or rate_col is None:
            log.warning("schema mismatch on %s (cols=%s); falling back", chosen, cols)
            return _synthetic_series(gu, lookback_weeks)
        where_gu = f"WHERE {gu_col}=?" if gu_col else ""
        sql = (
            f"SELECT {date_col}, {rate_col} FROM {chosen} {where_gu} "
            f"ORDER BY {date_col} DESC LIMIT {int(lookback_weeks)}"
        )
        rows = cur.execute(sql, (gu,) if gu_col else ()).fetchall()
        rows.reverse()
        if not rows:
            return _synthetic_series(gu, lookback_weeks)
        dates = [str(r[0]) for r in rows]
        values = [float(r[1]) for r in rows]
    return {
        "gu": gu,
        "table": chosen,
        "n_weeks": len(values),
        "dates": dates,
        "ili_rate": values,
        "last_value": values[-1] if values else None,
        "source": "epi_real_seoul.db",
    }


def _synthetic_series(gu: str, lookback_weeks: int) -> dict:
    """Deterministic synthetic weekly ILI series for environments without
    an installed epi_real_seoul.db. Anchored around the KDCA threshold of 8.6."""
    rng = np.random.default_rng(seed=42)
    t = np.arange(lookback_weeks)
    season = 5.0 + 4.0 * np.maximum(0.0, np.cos(2 * np.pi * (t - 12) / 52))
    noise = rng.normal(0.0, 0.7, size=lookback_weeks)
    series = np.clip(season + noise, 1.0, None)
    return {
        "gu": gu,
        "table": "synthetic",
        "n_weeks": int(lookback_weeks),
        "dates": [f"2024W{w:02d}" for w in range(1, lookback_weeks + 1)],
        "ili_rate": [float(x) for x in series],
        "last_value": float(series[-1]),
        "source": "synthetic_fallback",
    }


# ---------------------------------------------------------------------------
# 2) Forecast: persistence + AR(1)
# ---------------------------------------------------------------------------
def forecast_next_n_weeks(series: list[float], *, n_weeks: int = 4) -> dict:
    """Lightweight persistence + AR(1) forecaster. NOT a substitute for
    the 66-model registry; used here only to demonstrate pipeline
    plumbing end-to-end. Emits point forecast plus a naive 95 % band
    computed from the residual std of the in-sample AR(1) fit."""
    x = np.asarray(series, dtype=float)
    if x.size < 4:
        return {"point": [float(x[-1])] * n_weeks, "lo95": [float(x[-1])] * n_weeks,
                "hi95": [float(x[-1])] * n_weeks, "phi": 0.0, "sigma": 0.0}
    y = x[1:]
    z = x[:-1]
    phi = float(np.dot(y - y.mean(), z - z.mean()) / max(np.dot(z - z.mean(), z - z.mean()), 1e-9))
    intercept = float(y.mean() - phi * z.mean())
    pred = []
    prev = float(x[-1])
    for _ in range(n_weeks):
        nxt = intercept + phi * prev
        pred.append(nxt)
        prev = nxt
    resid = y - (intercept + phi * z)
    sigma = float(resid.std(ddof=1))
    lo = [p - 1.96 * sigma for p in pred]
    hi = [p + 1.96 * sigma for p in pred]
    return {
        "point": [float(p) for p in pred],
        "lo95": [float(v) for v in lo],
        "hi95": [float(v) for v in hi],
        "phi": float(phi),
        "sigma": sigma,
        "method": "AR(1) + persistence baseline",
    }


# ---------------------------------------------------------------------------
# 3) SEIR projection
# ---------------------------------------------------------------------------
def run_seir_projection(
    metapop_params: MetapopParams,
    *,
    horizon_days: int = 120,
    initial_rate: float = 5.0,
) -> dict:
    """Run the kernel-only SEIR-V-D projection seeded from a current
    weekly ILI rate (per 1 000). Returns city totals plus peak timing."""
    G = int(metapop_params.populations.size)
    pops = np.asarray(metapop_params.populations, dtype=float)
    # Convert rate per 1 000 into district-level current-infectious counts
    # (assuming sentinel ratio == ILI ratio at the aggregate level)
    I0 = pops * (initial_rate / 1000.0)

    params = MetapopParams(
        disease=metapop_params.disease,
        populations=metapop_params.populations,
        mobility=metapop_params.mobility,
        district_names=metapop_params.district_names,
        initial_infected=I0,
        initial_recovered=metapop_params.initial_recovered,
        initial_vaccinated=metapop_params.initial_vaccinated,
        vaccination_rate=metapop_params.vaccination_rate,
        days=int(horizon_days),
        dt=metapop_params.dt,
        seed=metapop_params.seed,
    )
    sim = MetapopSEIRVD(params).run(run_validator=False)
    city_I = sim.city_total("I")
    peak_day = int(np.argmax(city_I))
    return {
        "horizon_days": int(horizon_days),
        "city_I_peak": float(city_I.max()),
        "peak_day": peak_day,
        "final_I": float(city_I[-1]),
        "city_I_trajectory": [float(v) for v in city_I],
        "R0": float(metapop_params.disease.R0),
        "VE": float(metapop_params.disease.VE),
    }


# ---------------------------------------------------------------------------
# 4) ABM counterfactual
# ---------------------------------------------------------------------------
def run_abm_counterfactual(
    metapop_params: MetapopParams,
    *,
    horizon_days: int = 120,
    initial_rate: float = 5.0,
) -> dict:
    """Run behaviour-off baseline + behaviour-on (post-COVID fatigue)
    counterfactual. Returns peak shift %, mean compliance fraction,
    and the invariant-test pass flag for the current params."""
    G = int(metapop_params.populations.size)
    pops = np.asarray(metapop_params.populations, dtype=float)
    I0 = pops * (initial_rate / 1000.0)
    params = MetapopParams(
        disease=metapop_params.disease,
        populations=metapop_params.populations,
        mobility=metapop_params.mobility,
        district_names=metapop_params.district_names,
        initial_infected=I0,
        initial_recovered=metapop_params.initial_recovered,
        initial_vaccinated=metapop_params.initial_vaccinated,
        vaccination_rate=metapop_params.vaccination_rate,
        days=int(horizon_days),
        dt=metapop_params.dt,
        seed=metapop_params.seed,
    )
    off = run_coupled_abm(params, BehaviouralParams(
        alpha=0.0, kappa=0.0, tau=float("inf")
    ))
    on = run_coupled_abm(params, BehaviouralParams(
        alpha=1.5, kappa=0.8, tau=60.0, theta=0.15,
    ))
    I_off = off.city_I()
    I_on = on.city_I()
    peak_off = float(I_off.max())
    peak_on = float(I_on.max())
    shift_pct = 100.0 * (peak_on - peak_off) / peak_off if peak_off > 0 else 0.0
    invariant = run_invariant_test(params, tolerance=1e-6)
    return {
        "behaviour_off": {
            "peak_city_I": peak_off,
            "peak_day": int(np.argmax(I_off)),
        },
        "behaviour_on": {
            "peak_city_I": peak_on,
            "peak_day": int(np.argmax(I_on)),
            "mean_compliance": float(on.compliance.mean()),
            "behav_params": {
                "alpha": 1.5, "kappa": 0.8, "tau": 60.0, "theta": 0.15,
            },
        },
        "peak_shift_pct": shift_pct,
        "invariant_passed": bool(invariant["passed"]),
        "invariant_rmse": float(invariant["rmse"]),
    }


# ---------------------------------------------------------------------------
# 5) Advisor question builder
# ---------------------------------------------------------------------------
def build_advisor_questions(
    *,
    gu: str,
    db_snapshot: dict,
    forecast: dict,
    seir_baseline: dict,
    abm: dict,
) -> dict:
    """Turn the numeric pipeline outputs into a structured advisor packet:
    two natural-language questions (KO + EN) that an epidemiologist might
    realistically ask the ARIA consultation layer.
    """
    last = db_snapshot.get("last_value") or 0.0
    pred4 = forecast.get("point", [])
    pred_week1 = pred4[0] if pred4 else last
    pred_week4 = pred4[-1] if pred4 else last
    peak_off = seir_baseline.get("city_I_peak", 0.0)
    peak_on = abm.get("behaviour_on", {}).get("peak_city_I", 0.0)
    shift = abm.get("peak_shift_pct", 0.0)
    compl = abm.get("behaviour_on", {}).get("mean_compliance", 0.0)

    ko_prompt = (
        f"{gu} 보건소 주간 알람 회의 브리프입니다. 최신 관측 ILI = {last:.2f} / 1 000. "
        f"다음 주 AR(1) 예측 = {pred_week1:.2f} (95 % CI ±{forecast.get('sigma', 0.0):.2f}), "
        f"4주 후 예측 = {pred_week4:.2f}. KDCA 경보 기준 8.6, q70 11.45, 감사 27.28. "
        f"kernel-only SEIR 120일 피크 = {peak_off:.0f}명 @ day {seir_baseline.get('peak_day', 0)}. "
        f"4-param 행동 ABM (α=1.5, κ=0.8, τ=60, θ=0.15) 적용 시 피크 = {peak_on:.0f}명 "
        f"({shift:+.1f}%), 평균 준수율 = {compl:.3f}. "
        f"지금 알람 발령 여부를 3-단계 운영 권고로 정리하고, §4.13 알람-기준 sensitivity 결과와 "
        f"§4.16 ABM 재바운드 검증 결과를 연결해 답변하세요."
    )
    en_prompt = (
        f"{gu} district public-health-centre weekly alert briefing. Most-recent ILI = "
        f"{last:.2f} / 1 000. Next-week AR(1) forecast = {pred_week1:.2f} "
        f"(95 % CI ±{forecast.get('sigma', 0.0):.2f}), four-week horizon = {pred_week4:.2f}. "
        f"KDCA threshold = 8.6, q70 = 11.45, audit = 27.28. Kernel-only SEIR 120-day peak = "
        f"{peak_off:.0f} infectious @ day {seir_baseline.get('peak_day', 0)}. "
        f"Four-parameter behavioural ABM (alpha=1.5, kappa=0.8, tau=60, theta=0.15) peak = "
        f"{peak_on:.0f} ({shift:+.1f}%) with mean compliance {compl:.3f}. "
        f"Give a three-step operational recommendation that ties back to §4.13 threshold "
        f"sensitivity and §4.16 ABM rebound validation; flag any §4.9 PI coverage caveats."
    )
    return {
        "gu": gu,
        "ko_prompt": ko_prompt,
        "en_prompt": en_prompt,
        "numeric_inputs": {
            "last_ili": last,
            "pred_week1": pred_week1,
            "pred_week4": pred_week4,
            "peak_off": peak_off,
            "peak_on": peak_on,
            "shift_pct": shift,
            "mean_compliance": compl,
        },
    }


# ---------------------------------------------------------------------------
# Hermes audit helper
# ---------------------------------------------------------------------------
def _append_audit(chain: list[dict], entry: dict) -> None:
    prev = chain[-1]["hash"] if chain else "genesis"
    payload = json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
    h = hashlib.sha256(prev.encode("utf-8") + payload).hexdigest()
    out = dict(entry)
    out["prev_hash"] = prev
    out["hash"] = h
    chain.append(out)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def run_e2e(
    *,
    gu: str = "강남구",
    lookback_weeks: int = 52,
    forecast_weeks: int = 4,
    horizon_days: int = 120,
    out_dir: Optional[Path | str] = "valid_test/pipeline_demo",
    include_llm: bool = True,
    llm_max_ollama: int = 3,
    llm_no_api: bool = False,
    llm_no_ollama: bool = False,
    llm_no_mock: bool = False,
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 384,
) -> E2EResult:
    audit: list[dict] = []
    _append_audit(audit, {"event": "e2e.start",
                          "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "gu": gu})

    log.info("[1/6] DB -> weekly ILI for %s", gu)
    db_snap = retrieve_ili_from_db(gu=gu, lookback_weeks=lookback_weeks)
    _append_audit(audit, {"event": "db.fetch",
                          "gu": gu, "n_weeks": db_snap["n_weeks"],
                          "source": db_snap["source"]})

    log.info("[2/6] Forecast next %d weeks", forecast_weeks)
    fc = forecast_next_n_weeks(db_snap["ili_rate"], n_weeks=forecast_weeks)
    _append_audit(audit, {"event": "forecast.run", "method": fc["method"],
                          "horizon": forecast_weeks})

    log.info("[3/6] Kernel-only SEIR projection, horizon %d days", horizon_days)
    mp = load_metapop_params()
    seir = run_seir_projection(
        mp, horizon_days=horizon_days,
        initial_rate=db_snap["last_value"] or 5.0,
    )
    _append_audit(audit, {"event": "seir.run",
                          "city_I_peak": seir["city_I_peak"],
                          "peak_day": seir["peak_day"]})

    log.info("[4/6] ABM behaviour-on counterfactual")
    abm = run_abm_counterfactual(
        mp, horizon_days=horizon_days,
        initial_rate=db_snap["last_value"] or 5.0,
    )
    _append_audit(audit, {"event": "abm.run",
                          "peak_shift_pct": abm["peak_shift_pct"],
                          "invariant_passed": abm["invariant_passed"]})

    log.info("[5/6] Advisor question packet")
    advisor = build_advisor_questions(
        gu=gu, db_snapshot=db_snap, forecast=fc,
        seir_baseline=seir, abm=abm,
    )
    _append_audit(audit, {"event": "advisor.build",
                          "prompt_ko_len": len(advisor["ko_prompt"]),
                          "prompt_en_len": len(advisor["en_prompt"])})

    llm_report: Optional[dict] = None
    if include_llm:
        log.info("[6/6] Multi-LLM consultation")
        backends = discover_backends(
            include_api=not llm_no_api,
            include_ollama=not llm_no_ollama,
            include_mock=not llm_no_mock,
            max_ollama=llm_max_ollama,
        )
        if not backends:
            log.warning("no LLM backends available; skipping step 6")
        else:
            from simulation.llm_compare.golden_set import GoldenItem
            # Build two ad-hoc GoldenItems from the advisor prompts
            adv_items = (
                GoldenItem(
                    id=f"{gu}-ko", scenario="S4", persona="P1", lang="ko",
                    difficulty="ambiguous", prompt=advisor["ko_prompt"],
                    must_contain=("§4.13", "§4.16"),
                    must_avoid=("100 %", "절대"),
                    style_tags=("advisor", "e2e"),
                    source="pipeline_demo",
                ),
                GoldenItem(
                    id=f"{gu}-en", scenario="S4", persona="P1", lang="en",
                    difficulty="ambiguous", prompt=advisor["en_prompt"],
                    must_contain=("§4.13", "§4.16", "PI"),
                    must_avoid=("guaranteed", "perfect"),
                    style_tags=("advisor", "e2e"),
                    source="pipeline_demo",
                ),
            )
            report = run_comparison(
                backends=backends, items=adv_items,
                temperature=llm_temperature, max_tokens=llm_max_tokens,
                verbose=True,
            )
            llm_report = json.loads(report.to_json())
            _append_audit(audit, {
                "event": "llm.multi_compare",
                "n_backends": len(backends),
                "winner": report.ranking[0]["backend_id"] if report.ranking else "",
                "winner_total": report.ranking[0]["total"] if report.ranking else 0.0,
            })

    _append_audit(audit, {"event": "e2e.end",
                          "time": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    # Persist artefacts
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pipeline_summary.json").write_text(
            json.dumps({
                "gu": gu,
                "db_snapshot": db_snap,
                "forecast": fc,
                "seir_baseline": seir,
                "abm_counterfactual": abm,
            }, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        (out / "advisor_packet.json").write_text(
            json.dumps(advisor, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out / "audit_chain.jsonl").write_text(
            "\n".join(json.dumps(e, default=str) for e in audit) + "\n",
            encoding="utf-8",
        )
        if llm_report is not None:
            (out / "llm_comparison.json").write_text(
                json.dumps(llm_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # also reconstruct a markdown summary from the ranking
            lines = ["# E2E LLM consultation ranking", ""]
            lines.append(f"Gu: {gu}")
            lines.append(f"Backends: {len(llm_report.get('backends', []))}")
            lines.append("")
            lines.append("| rank | backend_id | total | tier | latency ms |")
            lines.append("|---|---|---|---|---|")
            for i, row in enumerate(llm_report.get("ranking", []), start=1):
                lines.append(f"| {i} | {row['backend_id']} | {row['total']:.4f} | "
                             f"{row['tier']} | {row['mean_latency_ms']:.0f} |")
            (out / "llm_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("artefacts written to %s", out)

    return E2EResult(
        gu=gu,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        db_snapshot=db_snap,
        forecast=fc,
        seir_baseline=seir,
        abm_counterfactual=abm,
        advisor_packet=advisor,
        llm_comparison=llm_report,
        audit_chain=audit,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    _configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gu", default="강남구")
    ap.add_argument("--weeks", type=int, default=4, help="forecast horizon in weeks")
    ap.add_argument("--horizon-days", type=int, default=120)
    ap.add_argument("--out-dir", default="valid_test/pipeline_demo")
    ap.add_argument("--no-llm", action="store_true", help="skip step 6 (LLM consultation)")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--no-mock", action="store_true")
    ap.add_argument("--max-ollama", type=int, default=3)
    args = ap.parse_args(argv)

    res = run_e2e(
        gu=args.gu,
        forecast_weeks=args.weeks,
        horizon_days=args.horizon_days,
        out_dir=args.out_dir,
        include_llm=not args.no_llm,
        llm_max_ollama=args.max_ollama,
        llm_no_api=args.no_api,
        llm_no_ollama=args.no_ollama,
        llm_no_mock=args.no_mock,
    )
    # Compact one-line summary
    print("\n--- E2E summary ---")
    print(f"gu={res.gu}")
    print(f"DB source={res.db_snapshot['source']}  last ILI={res.db_snapshot['last_value']}")
    print(f"4-week forecast: {res.forecast['point']}")
    print(f"SEIR peak: {res.seir_baseline['city_I_peak']:.0f} @ day {res.seir_baseline['peak_day']}")
    print(f"ABM peak shift: {res.abm_counterfactual['peak_shift_pct']:+.1f}%  "
          f"invariant={'PASS' if res.abm_counterfactual['invariant_passed'] else 'FAIL'}")
    if res.llm_comparison:
        top = res.llm_comparison.get("ranking", [{}])[0]
        print(f"LLM winner: {top.get('backend_id')}  total={top.get('total'):.4f}")
    return 0


def test_e2e_smoke_skipped() -> None:
    """Pytest collection target — skipped by default (DB + LLM heavy).

    Run directly with ``python -m simulation.tests.test_e2e_smoke`` or call
    ``simulation.verify_all.check_e2e`` for nightly verify integration.
    """
    import pytest

    pytest.skip(
        "E2E smoke heavy (DB + SEIR + ABM + LLM) — run via CLI "
        "(`python -m simulation.tests.test_e2e_smoke`) or verify_all.check_e2e."
    )


if __name__ == "__main__":
    sys.exit(main())
