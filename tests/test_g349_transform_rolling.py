"""G-349 (2026-06-25, 감사 P1): transform-rolling test eval 의 'none'→'identity' alias 정규화.

N-BEATS/N-HiTS/TiDE 가 y_mode='none'(best_config.transform='HIER_none')인데 transform-rolling 블록
(per_model_optimize.py:1383)이 strip-only('HIER_none'→'none')로 _apply_single_y_transform 에 넘겨
ValueError→static fallback → 선정(rolling)≠평가(static) 비대칭(test 수치 static-collapse). 'none'=
'identity' alias 정규화로 해소. sibling :1308(비-hier 경로)과 동일 정규화.

macOS: per-file.
"""
import pathlib

import numpy as np
import pytest

from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_apply_y_transform_none_raises_identity_works():
    """근본: 'none'은 ValueError(인식 안 됨), 'identity'는 no-op. → strip-only 가 'none' 넘기면 죽음."""
    y = np.abs(np.random.RandomState(0).randn(20) * 5 + 20)
    with pytest.raises(Exception):
        _apply_single_y_transform(y, "none")
    out, _, _ = _apply_single_y_transform(y, "identity")
    assert np.allclose(out, y), "identity 는 no-op(입력 보존)"


def test_g349_alias_normalization_in_source():
    """transform-rolling 블록에 'none'→'identity' alias 정규화가 있어야(없으면 ValueError→static 비대칭)."""
    src = (ROOT / "simulation/pipeline/per_model_optimize.py").read_text(encoding="utf-8")
    assert '_tnr = "identity" if _tnr in ("none", "")' in src, "G-349 alias 정규화 누락(strip-only 만이면 버그)"


def test_g350_spa_benchmark_includes_flusight():
    """G-350: SPA benchmark 후보에 FluSight-Baseline(실재 persistence) 포함 — vacuous fallback 방지."""
    src = (ROOT / "simulation/pipeline/per_model_eval.py").read_text(encoding="utf-8")
    assert '"FluSight-Baseline", "persistence"' in src, "G-350: SPA benchmark 에 FluSight-Baseline 누락"


def test_g351_rerank_filters_defer():
    """G-351: rerank _load 가 DEFER_MODELS 제외 — deprecated 가 후보 pool/진단에서 빠짐."""
    src = (ROOT / "simulation/scripts/rerank_champion.py").read_text(encoding="utf-8")
    assert "DEFER_MODELS" in src and "_defer" in src, "G-351: rerank DEFER 필터 누락"
