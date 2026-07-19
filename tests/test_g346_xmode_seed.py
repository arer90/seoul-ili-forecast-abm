"""G-346 (2026-06-25): preproc seed builder 가 force_x_identity 모델에도 x_mode='none' 를 항상 기록.

옛 가드(if not force_x_identity)가 x_mode 를 omit → 영속 preproc_optuna_params 에 x_mode 부재 →
refit FixedTrial replay 가 'x_mode' 키 못 찾아 HIER_FAIL→oof_wis=inf (foundation TiRex/TimesFM-2.5/DLinear).
fix: 가드 제거 → x_mode='none' 항상 기록. y_mode 와 대칭. 검색공간 0 증가(단일값).

macOS: per-file.
"""
from simulation.pipeline._inline_optuna_3stage import _seed_y_transform_trials


class _CaptureStudy:
    def __init__(self):
        self.enqueued = []

    def enqueue_trial(self, params, skip_if_exists=True):
        self.enqueued.append(dict(params))


def test_force_x_identity_still_records_x_mode():
    """★ force_x_identity=True(foundation) 여도 모든 seed 에 x_mode='none' — 옛 버그(oof=inf) 회귀 가드."""
    st = _CaptureStudy()
    n = _seed_y_transform_trials(st, "TiRex", force_y_identity=False, force_x_identity=True,
                                 restrict_centered=False, n_trials=20, grid_mode=True)
    assert n > 0 and st.enqueued, "seed 가 enqueue 돼야"
    for p in st.enqueued:
        assert p.get("x_mode") == "none", f"force_x_identity 인데 x_mode 누락(옛 버그): {p}"


def test_non_force_unchanged():
    """force_x_identity=False(일반 모델 FusedEpi 등)도 x_mode='none' — fix 전후 동일(회귀 없음)."""
    st = _CaptureStudy()
    _seed_y_transform_trials(st, "FusedEpi", force_y_identity=False, force_x_identity=False,
                             restrict_centered=False, n_trials=20, grid_mode=True)
    assert st.enqueued
    for p in st.enqueued:
        assert p.get("x_mode") == "none"


def test_identity_anchor_has_x_mode():
    """identity anchor(force_y_identity=True) seed 도 x_mode='none' 포함."""
    st = _CaptureStudy()
    _seed_y_transform_trials(st, "TiRex", force_y_identity=True, force_x_identity=True,
                             restrict_centered=False, n_trials=5, grid_mode=False)
    assert any("x_mode" in p for p in st.enqueued), "anchor 에 x_mode 있어야"
    for p in st.enqueued:
        assert p.get("x_mode") == "none"
