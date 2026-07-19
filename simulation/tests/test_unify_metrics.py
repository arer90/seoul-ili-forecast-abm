"""Tests for the unified honest metric table + extrapolation-stability flag."""
from __future__ import annotations

import csv
import pathlib

from simulation.analytics.unify_metrics import (
    build_unified_table,
    compute_stability,
)


def _write_csv(path, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_stability_flags_extrapolation(tmp_path) -> None:
    p = tmp_path / "pred.csv"
    _write_csv(p, ["model", "y_true", "y_pred"], [
        ["StableM", 10, 9], ["StableM", 100, 95],     # 95 <= 1.5*100 -> stable
        ["BlowupM", 10, 9], ["BlowupM", 100, 353],    # 353 > 1.5*100 -> unstable
    ])
    st = compute_stability(p)
    assert st["StableM"]["stable"] is True
    assert st["BlowupM"]["stable"] is False
    assert st["BlowupM"]["extrapolation_ratio"] == 3.53


def test_build_unified_table_sorts_and_flags(tmp_path) -> None:
    mp = tmp_path / "metrics.csv"
    _write_csv(mp, ["model", "wis", "r2", "mape", "pi95_coverage", "rank_wis"], [
        ["NegBin", 3.2, 0.93, 15.8, 0.971, 1],
        ["Elastic", 30.7, -8.45, 63.1, 0.706, 12],
    ])
    pp = tmp_path / "pred.csv"
    _write_csv(pp, ["model", "y_true", "y_pred"], [
        ["NegBin", 100, 92], ["Elastic", 100, 353],
    ])
    t = build_unified_table(mp, pp)
    assert t["rows"][0]["model"] == "NegBin"      # lower WIS ranks first
    assert t["n_unstable"] == 1
    elastic = next(r for r in t["rows"] if r["model"] == "Elastic")
    assert elastic["stable"] is False
    assert "conformal" in t["pi95_note"]


def test_real_artifacts_if_present() -> None:
    m = pathlib.Path("simulation/results/per_model_eval/per_model_metrics.csv")
    p = pathlib.Path("simulation/results/eda/phase11_per_model_eval/predictions_per_model.csv")
    if not (m.exists() and p.exists()):
        return
    t = build_unified_table(m, p)
    assert t["n_models"] >= 10
    el = next((r for r in t["rows"] if r["model"] == "ElasticNet"), None)
    if el is not None:
        # G-275 (2026-06-16): ElasticNet 에 2×y_max 외삽 cap 추가 → 더 이상 폭발을 단정하지 않음
        #   (cap 이 blow-up 차단). detection 이 bool 결과를 내는지만 확인 — 실 artifact 상태에
        #   따라 stable/unstable 모두 정상(폭발 박제 = stale 가정). 폭발 탐지 로직 자체는
        #   위 합성 케이스(BlowupM/Elastic 353)에서 검증됨.
        assert isinstance(el["stable"], bool)
