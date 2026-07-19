"""Sprint α R1 (2026-05-26): eda_writer atomicity + non-fatal tests.

Codex § 3.3: atomic write (tmp + Path.replace) + try/except never raises.
G-057 portability: utf-8 encoding 명시.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from simulation.pipeline.eda_writer import write_phase_eda, DEFAULT_ISSUE_RULES


def test_writes_3_files_per_phase(tmp_path):
    """phase{NN}_{tag}/ 디렉토리 + predictions.csv + metrics.json + issues.md."""
    y_true = np.arange(10, dtype=float)
    preds = {"M1": y_true + 0.1, "M2": y_true - 0.1}
    result = write_phase_eda(
        phase_id=2, phase_tag="smoke",
        y_true=y_true, predictions=preds,
        save_dir=tmp_path,
    )
    assert result is True
    out_dir = tmp_path / "phase02_smoke"
    assert (out_dir / "predictions_per_model.csv").exists()
    assert (out_dir / "metrics_summary.json").exists()
    assert (out_dir / "issues.md").exists()


def test_phase_id_zero_padded(tmp_path):
    """phase{NN:02d} — phase 2 → phase02, phase 11 → phase11."""
    y = np.array([1.0, 2.0, 3.0])
    p = {"M": np.array([1.0, 2.0, 3.0])}
    write_phase_eda(phase_id=2, phase_tag="t", y_true=y, predictions=p, save_dir=tmp_path)
    write_phase_eda(phase_id=11, phase_tag="t", y_true=y, predictions=p, save_dir=tmp_path)
    assert (tmp_path / "phase02_t").exists()
    assert (tmp_path / "phase11_t").exists()


def test_never_raises_on_bad_input(tmp_path):
    """Codex § 3.3: 어떤 입력 에러도 raise 안 됨 — False 반환."""
    # Mismatched shapes
    result = write_phase_eda(
        phase_id=99, phase_tag="bad",
        y_true=np.array([1.0, 2.0, 3.0]),
        predictions={"M": np.array([1.0])},  # wrong length
        save_dir=tmp_path,
    )
    # Per-model handles length mismatch internally; outer call still succeeds
    assert result is True   # phase dir exists, length-mismatch model gets "error" status

    # Empty predictions dict → False
    result = write_phase_eda(
        phase_id=99, phase_tag="empty",
        y_true=np.array([1.0]),
        predictions={},
        save_dir=tmp_path,
    )
    assert result is False


def test_disabled_via_global_flag(tmp_path, monkeypatch):
    """GLOBAL.ops.disable_eda_sidecar (= MPH_DISABLE_EDA_SIDECAR) → no-op,
    False 반환. (2026-05-29 정정: env→GLOBAL SSOT 이행 후 GLOBAL 은 import 시
    1회 freeze 되므로 setenv 로는 못 바꿈 — writer 가 보는 GLOBAL 을 직접 patch.)"""
    import dataclasses
    import simulation.pipeline.eda_writer as ew
    patched = dataclasses.replace(
        ew.GLOBAL, ops=dataclasses.replace(ew.GLOBAL.ops, disable_eda_sidecar=True)
    )
    monkeypatch.setattr(ew, "GLOBAL", patched)
    result = write_phase_eda(
        phase_id=1, phase_tag="disabled",
        y_true=np.array([1.0, 2.0]),
        predictions={"M": np.array([1.0, 2.0])},
        save_dir=tmp_path,
    )
    assert result is False
    assert not (tmp_path / "phase01_disabled").exists()


def test_utf8_encoding_for_korean_text(tmp_path):
    """G-057: encoding=utf-8 명시. Korean issue text 무손실."""
    y = np.array([1.0, 2.0, 3.0])
    p = {"한국어모델_A": np.array([np.nan, np.nan, np.nan])}  # all NaN → issue flagged in Korean
    write_phase_eda(phase_id=99, phase_tag="utf8", y_true=y, predictions=p, save_dir=tmp_path)
    txt = (tmp_path / "phase99_utf8" / "issues.md").read_text(encoding="utf-8")
    assert "한국어모델_A" in txt


def test_atomic_write_no_partial_file_on_concurrent(tmp_path):
    """Tempfile + Path.replace: tmp file 만 남아도 final file 깨끗."""
    y = np.array([1.0, 2.0, 3.0])
    p = {"M": np.array([1.0, 2.0, 3.0])}
    write_phase_eda(phase_id=99, phase_tag="atomic", y_true=y, predictions=p, save_dir=tmp_path)
    out_dir = tmp_path / "phase99_atomic"
    # Check: no .tmp leftover (tempfile cleanup)
    tmp_leftovers = list(out_dir.glob(".predictions_per_model.csv.tmp.*"))
    assert tmp_leftovers == []


def test_metrics_summary_json_structure(tmp_path):
    """metrics_summary.json 의 schema: phase_id / phase_tag / per_model."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = {"good": np.array([1.05, 1.95, 3.1, 3.95])}
    write_phase_eda(phase_id=11, phase_tag="meta", y_true=y, predictions=p,
                    save_dir=tmp_path, extra_meta={"sprint": "alpha"})
    doc = json.loads((tmp_path / "phase11_meta" / "metrics_summary.json").read_text(encoding="utf-8"))
    assert doc["phase_id"] == 11
    assert doc["phase_tag"] == "meta"
    assert doc["n_models"] == 1
    assert "good" in doc["per_model"]
    assert "r2" in doc["per_model"]["good"]
    assert doc["sprint"] == "alpha"   # extra_meta merged


def test_issue_detection_flags_nan(tmp_path):
    """NaN fraction > 5% → issues.md 에 flag."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p_bad = {"nan_model": np.array([1.0, np.nan, 3.0, np.nan, 5.0])}  # 40% NaN
    write_phase_eda(phase_id=99, phase_tag="nan", y_true=y, predictions=p_bad, save_dir=tmp_path)
    md = (tmp_path / "phase99_nan" / "issues.md").read_text(encoding="utf-8")
    assert "NaN" in md
    assert "nan_model" in md


def test_issue_detection_flags_catastrophic_r2(tmp_path):
    """R² < 0 → catastrophic flag."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = {"bad": np.array([10.0, -5.0, 12.0, -3.0, 9.0])}
    write_phase_eda(phase_id=99, phase_tag="catastrophic", y_true=y, predictions=p, save_dir=tmp_path)
    md = (tmp_path / "phase99_catastrophic" / "issues.md").read_text(encoding="utf-8")
    assert "catastrophic" in md.lower()


def test_default_issue_rules_present():
    """기본 rules 키 4종 (NaN/R²/MAPE/ACF/outlier)."""
    keys = {"nan_threshold", "r2_floor", "mape_ceiling", "residual_acf_max", "outlier_z"}
    assert keys.issubset(DEFAULT_ISSUE_RULES.keys())
