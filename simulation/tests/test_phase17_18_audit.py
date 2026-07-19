"""audit deferred (2026-06-01): phase17/18 smoke + phase18 RENUMBER 라벨 회귀.

- phase17 run_inference: champion artifact 없을 때 crash 없이 dict 반환(graceful degrade).
- phase18: 자체 로그 prefix [phase_c] → [phase18] 정합 (RENUMBER 라벨 sweep 회귀).
(phase17 실 champion 예측 / phase18 해외 fetch 풀 e2e 는 실 아티팩트·네트워크 필요 → 별도.)

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q
"""
import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_phase17_degrades_on_no_champions(tmp_path):
    """champion artifact 없을 때 crash 없이 dict 반환 (degrade-and-continue)."""
    from simulation.pipeline.inference import run_inference
    X = np.zeros((8, 20), dtype=float)
    r = run_inference(X, models_dir=tmp_path, log_path=tmp_path / "champion_log.json", out_dir=tmp_path)
    assert isinstance(r, dict), "champion 없을 때 dict 반환해야 (graceful)"


def test_phase18_label_renumbered_no_phase_c():
    """phase18 자체 로그 prefix 가 옛 [phase_c] 아닌 [phase18] (RENUMBER 라벨 회귀 가드)."""
    import simulation.pipeline.overseas as m
    src = inspect.getsource(m)
    assert "[phase_c]" not in src, "phase18 에 옛 [phase_c] 라벨 잔존 (라벨 sweep 회귀)"
    assert "[phase18]" in src, "phase18 자체 로그 prefix 미정합"
