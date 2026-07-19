"""Reproducibility smoke (사용자 2026-06-07): NegBinGLM 모델 + e2e 파이프라인을
독립 2회 실행 → 결과가 비트/값 수준으로 동일한가(재현성, 엔지니어링 원칙 #5).

"모든 결과를 지우고 다시 만들어서 재현성" = 각 run을 독립 출력 디렉토리에 만들고
(이전 repro_* 는 _trash 로 이동) 두 run의 산출을 비교한다. 결정성이면 max|Δ|=0.

Run:  .venv/bin/python -m simulation.scripts.repro_smoke
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT = Path("simulation/results/repro_smoke")


def _fixed_dataset(n_lag: int = 3):
    """결정적 고정 데이터셋: 전국 sentinel ILI(주차) → lag + 계절 feature."""
    from simulation.database.storage import read_only_connect
    c = read_only_connect("simulation/data/db/epi_real_seoul.db")
    try:
        rows = c.execute(
            "SELECT season_start, week_seq, AVG(ili_rate) FROM sentinel_influenza "
            "WHERE ili_rate IS NOT NULL GROUP BY season_start, week_seq "
            "ORDER BY season_start, week_seq").fetchall()
    finally:
        c.close()
    ili = np.array([float(r[2]) for r in rows], dtype=np.float64)
    X, y = [], []
    for i in range(n_lag, len(ili)):
        X.append([ili[i - 1], ili[i - 2], ili[i - 3], float(i % 52)])
        y.append(ili[i])
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    s = int(len(y) * 0.8)
    return X[:s], y[:s], X[s:], y[s:]


def _negbin_once(Xtr, ytr, Xte) -> np.ndarray:
    from simulation.models.epi_models import NegBinGLMForecaster
    m = NegBinGLMForecaster(topk=20)
    pred = m.fit_predict(Xtr, ytr, Xte, name="NegBinGLM")
    return np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0)


def _e2e_once() -> dict:
    """전 파이프라인 e2e (db→forecast→SEIR→ABM→audit), LLM 제외(속도)."""
    from simulation.tests.test_e2e_smoke import run_e2e
    res = run_e2e(gu="강남구", lookback_weeks=26, forecast_weeks=2, horizon_days=60,
                  out_dir=str(OUT / "e2e_tmp"), include_llm=False,
                  llm_no_api=True, llm_no_ollama=True, llm_max_ollama=0)
    return {
        "db_source": res.db_snapshot.get("source"),
        "n_weeks": res.db_snapshot.get("n_weeks"),
        "forecast_point": [round(float(p), 6) for p in res.forecast.get("point", [])],
        "seir_peak": round(float(res.seir_baseline.get("city_I_peak", 0.0)), 6),
        "abm_peak_shift_pct": round(float(res.abm_counterfactual.get("peak_shift_pct", 0.0)), 6),
        "abm_invariant_passed": bool(res.abm_counterfactual.get("invariant_passed")),
        "audit_entries": len(res.audit_chain),
    }


def main() -> int:
    # "모든 결과를 지우고" — 이전 repro 산출을 _trash 로 이동(파괴적 rm 금지)
    if OUT.exists():
        trash = Path("_trash"); trash.mkdir(exist_ok=True)
        OUT.rename(trash / f"repro_smoke_{int(time.time())}")
    OUT.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xte, yte = _fixed_dataset()
    print(f"고정 데이터셋: train={len(ytr)} test={len(yte)} feat={Xtr.shape[1]}")

    # ── 1) NegBinGLM 결정성 (독립 2회) ───────────────────────────────────
    p1 = _negbin_once(Xtr, ytr, Xte)
    p2 = _negbin_once(Xtr, ytr, Xte)
    nb_max = float(np.max(np.abs(p1 - p2))) if p1.shape == p2.shape else float("inf")
    np.save(OUT / "negbin_pred_run1.npy", p1)
    np.save(OUT / "negbin_pred_run2.npy", p2)
    # 정확도(고정 test 대비) — 동일해야 함
    def _r2(yp):
        ss = float(np.sum((yte - yp) ** 2)); st = float(np.sum((yte - yte.mean()) ** 2))
        return 1.0 - ss / st if st > 0 else float("nan")
    nb = {"max_abs_diff": nb_max, "reproducible": nb_max == 0.0,
          "r2_run1": round(_r2(p1), 6), "r2_run2": round(_r2(p2), 6),
          "n_test": len(yte)}
    print(f"[1] NegBinGLM: max|Δ|={nb_max:.2e} reproducible={nb['reproducible']} "
          f"r2_run1={nb['r2_run1']} r2_run2={nb['r2_run2']}")

    # ── 2) e2e 파이프라인 결정성 (독립 2회) ──────────────────────────────
    e1 = _e2e_once()
    e2 = _e2e_once()
    e_diffs = {k: (e1[k], e2[k]) for k in e1 if e1[k] != e2[k]}
    ee = {"run1": e1, "run2": e2, "identical": len(e_diffs) == 0, "diffs": e_diffs}
    print(f"[2] e2e: identical={ee['identical']} "
          + (f"diffs={e_diffs}" if e_diffs else "(forecast·SEIR·ABM 전부 동일)"))

    report = {"negbin_glm": nb, "e2e": ee,
              "verdict": "REPRODUCIBLE" if (nb["reproducible"] and ee["identical"])
              else "NON-DETERMINISTIC (조사 필요)"}
    (OUT / "repro_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nVERDICT: {report['verdict']}  → {OUT / 'repro_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
