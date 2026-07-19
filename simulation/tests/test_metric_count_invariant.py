"""Lock the metric SSOT count invariant + 4-criteria(g175) 완전 부재 가드.

History: G-238 (2026-05-30) pinned 134 keys (incl. 5 g175 4-criteria flags).
**2026-06-05 (사용자 명시): 4-criteria(g175) 완전 제거** → 5개 g175_*_pass 키 삭제 →
canonical count 134→**129**. champion = 순수 best-WIS, R²/MAPE/WIS/PICP 는 개별
metric 으로만 존재. `enable_g175_binding` 인자도 제거 (전 call-site 정리 2026-06-05).

3-way ground-truth (RUN the function, not doc-read): evaluate_predictions_full
returns EXACTLY 129 keys for MIN inputs and FULL inputs.
test_no_g175_keys 가 4-criteria 재출현을 영구 차단.

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
import numpy as np

from simulation.pipeline.phase_evaluator import evaluate_predictions_full as ev

CANON = 129  # 2026-06-05: 134 − 5 g175 keys (4-criteria 제거)


def _yp(n: int = 40):
    rng = np.random.default_rng(0)
    y = rng.uniform(5.0, 30.0, n)
    return y, y + rng.normal(0, 2.0, n)


def test_count_minimal_and_default():
    y, p = _yp()
    assert len(ev(y, p)) == CANON
    assert len(ev(y, p, phase_id="x")) == CANON


def test_no_g175_keys():
    """4-criteria(g175) 완전 제거 영구 가드 — 어떤 g175_* 키도 없어야 한다."""
    y, p = _yp()
    out = ev(y, p)
    assert not any(str(k).startswith("g175") for k in out), \
        f"g175 키 재출현: {[k for k in out if str(k).startswith('g175')]}"
    assert "g175_4criteria_pass" not in out


def test_count_with_full_inputs():
    y, p = _yp()
    pool = np.random.default_rng(1).uniform(5.0, 30.0, 200)
    out = ev(y, p, residuals=y - p, sigma=3.0, y_train_pool=pool, threshold=8.6,
             baseline_predictions={"persistence": p},
             all_model_wis={"x": 3.0, "persistence": 4.0},
             all_model_mae={"x": 2.0, "persistence": 2.5})
    assert len(out) == CANON
