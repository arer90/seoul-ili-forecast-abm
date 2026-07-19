"""SSOT train/val/test 분할 경계 TDD (audit 2026-06-01 — 이 함수가 전체 파이프라인의
train/test 경계를 정하는데 직접 테스트가 없었음).

compute_split_indices(n, config) = 4-way HWP 분할의 단일 출처. phase1/5/6/13 모두 사용.
HWP: n_test=ceil(n·test_ratio), pool=n−n_test, n_val=round(pool·val_ratio), n_train=pool−n_val.

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q
"""
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_hwp_split_exact_337():
    """HWP n=337 → (242, 27, 68): test=ceil(0.20·337)=68, pool=269, val=round(26.9)=27, train=242.

    프로젝트 표준 in-sample 분할(242/27/68) 을 고정 — 경계 규칙 변경 시 FAIL.
    """
    from simulation.pipeline.data import compute_split_indices
    from simulation.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    cfg.split.in_sample_test_ratio = 0.20      # HWP 경로 강제 (formula 결정론 검증)
    cfg.split.in_sample_val_ratio = 0.10
    assert compute_split_indices(337, cfg) == (242, 27, 68)


def test_split_invariants_default_config():
    """기본 config: 합=n, 양수, train 최대 (어느 분기든 불변)."""
    from simulation.pipeline.data import compute_split_indices
    from simulation.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    for n in (100, 242, 337, 500):
        nt, nv, ntest = compute_split_indices(n, cfg)
        assert nt + nv + ntest == n, f"분할 합 != n at n={n}: {nt}+{nv}+{ntest}"
        assert nt > 0 and nv >= 0 and ntest > 0, f"음수/0 분할 at n={n}"
        assert nt > ntest, f"train 이 test 보다 작음 at n={n}"


def test_split_deterministic_and_scales():
    from simulation.pipeline.data import compute_split_indices
    from simulation.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert compute_split_indices(337, cfg) == compute_split_indices(337, cfg)
    assert compute_split_indices(500, cfg)[0] > compute_split_indices(100, cfg)[0], "n↑ train↑ 위배"
