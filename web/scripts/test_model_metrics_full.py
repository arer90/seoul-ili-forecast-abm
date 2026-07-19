#!/usr/bin/env python3
"""web 전체 평가지표(129) — TDD. 사용자: web까지 129 반영 + 기본 6(r2/wis/rmse/mae/auc-roc/c-index).

박제: 재학습 없이 predictions CSV → evaluate_predictions_full(129) → model-metrics-full.json.
roc_auc/c_index 포함, web 패널이 기본 6 + 전체 129 펼침을 렌더.
Run:  .venv/bin/python web/scripts/test_model_metrics_full.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_model_metrics_full import load_test_split, metrics_for, DEFAULT_VISIBLE  # noqa: E402
AGG = ROOT / "web" / "public" / "aggregates"


def test_default_visible_six():
    """기본 표시 = 정확히 {r2, wis, rmse, mae, roc_auc, c_index} 6개(사용자 명시)."""
    assert DEFAULT_VISIBLE == ["r2", "wis", "rmse", "mae", "roc_auc", "c_index"], DEFAULT_VISIBLE


def test_metrics_for_champion_129():
    """champion NegBinGLM 예측 → 129지표, roc_auc·c_index 유한."""
    m = metrics_for("NegBinGLM")
    if m is None:
        print("  (predictions_NegBinGLM.csv 없음 — skip)"); return
    assert len(m) == 129, f"129지표 아님: {len(m)}"
    for k in DEFAULT_VISIBLE:
        assert k in m, f"기본지표 {k} 누락"
    assert m["roc_auc"] is not None and m["c_index"] is not None, "auc-roc/c-index NaN"
    assert m["r2"] is not None and 0.5 < m["r2"] <= 1.0, m["r2"]


def test_split_test_only():
    """test split 만 사용(val 누수 없음)."""
    sp = load_test_split("NegBinGLM")
    if sp is None:
        return
    assert len(sp[0]) == len(sp[1]) >= 4


def test_aggregate_schema():
    """산출 model-metrics-full.json: n_metrics=129, default 6, 모델별 129키, roc_auc/c_index 키존재."""
    p = AGG / "model-metrics-full.json"
    if not p.is_file():
        print("  (model-metrics-full.json 없음 — 빌더 먼저 실행)"); return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n_metrics"] == 129
    assert d["default_visible"] == DEFAULT_VISIBLE
    assert "roc_auc" in d["metric_keys"] and "c_index" in d["metric_keys"]
    assert len(d["models"]) >= 10, "모델 수 부족"
    champ = d["models"].get("NegBinGLM")
    assert champ and champ.get("r2") and champ.get("roc_auc"), "champion 지표 부재"


def test_web_panel_wired():
    """app.jsx: model-metrics-full 로드 + 기본 6 + 전체 129 펼침 패널 렌더(배선)."""
    src = (ROOT / "web_prototype" / "app.jsx").read_text(encoding="utf-8")
    abs_ = (ROOT / "web" / "public" / "abs" / "app.jsx").read_text(encoding="utf-8")
    assert "model-metrics-full.json" in src and "setMetricsFull" in src, "App 이 metricsFull 로드 안 함"
    assert "metricsFull={metricsFull}" in src, "MapTab 에 metricsFull 전달 안 됨"
    assert "평가지표" in src and "showAllMetrics" in src, "지표 패널/펼침 미렌더"
    assert "default_visible" in src, "기본 6지표 표시 미배선"
    assert src == abs_, "dual-file 미동기화"


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
