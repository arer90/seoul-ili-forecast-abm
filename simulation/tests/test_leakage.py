"""
simulation/tests/test_leakage.py
================================
Data Leakage 단위 테스트 — P2-7.

커버 범위:
 1. LeakageChecker.run_all_checks — 6개 검사 동작 확인
 2. 인덱스 겹침: (date, gu_nm) 키 기반 train/val/test 교집합 탐지
 3. 미래 정보 피처명(`_lead`, `_forward`, `_future`, `_next`, `_lag0`) 탐지
 4. train/test split 경계 중복 (마지막 train 행 = 첫 test 행)
 5. 비현실적 상관 (corr ≥ 0.98) CRITICAL 트리거
 6. 분포 누출 (train/test 모두 mean≈0, std≈1) WARNING 트리거

의존성:
 - pytest
 - numpy
 - simulation.models.leakage_checker

실행:
 .venv/Scripts/python.exe -m pytest simulation/tests/test_leakage.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.models.leakage_checker import (
    LeakageChecker,
    LeakageReport,
    LeakageWarning,
    check_leakage_quick,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_clean_split(n_train: int = 200, n_test: int = 50,
                     n_feat: int = 10, seed: int = 42):
    """누출이 없는 깨끗한 split."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_feat))
    X_test = rng.standard_normal((n_test, n_feat))
    # target 은 피처와 약한 상관 (|r| < 0.5)
    y_train = X_train[:, 0] * 0.3 + rng.standard_normal(n_train) * 1.0
    y_test = X_test[:, 0] * 0.3 + rng.standard_normal(n_test) * 1.0
    feature_names = [f"feat_{i}" for i in range(n_feat)]
    return X_train, X_test, y_train, y_test, feature_names


# ═══════════════════════════════════════════════════════════════
# 1. Clean split — no warnings
# ═══════════════════════════════════════════════════════════════

def test_clean_split_passes():
    """누출 없는 랜덤 split 은 CRITICAL 없이 통과."""
    X_tr, X_te, y_tr, y_te, names = _make_clean_split()
    report = check_leakage_quick(X_tr, X_te, y_tr, y_te, names)
    assert isinstance(report, LeakageReport)
    assert report.passed, (
        f"clean split 에서 CRITICAL 이 떠서는 안 됨: {report.summary()}"
    )
    assert report.n_critical == 0


# ═══════════════════════════════════════════════════════════════
# 2. 비현실적 상관 — CRITICAL
# ═══════════════════════════════════════════════════════════════

def test_perfect_correlation_triggers_critical():
    """피처-타겟 corr ≥ 0.98 이면 CRITICAL."""
    X_tr, X_te, y_tr, y_te, names = _make_clean_split(n_feat=5)
    # 0번 피처를 타겟과 거의 동일하게
    X_tr[:, 0] = y_tr + np.random.default_rng(0).standard_normal(len(y_tr)) * 1e-4
    report = check_leakage_quick(X_tr, X_te, y_tr, y_te, names)
    assert not report.passed
    assert report.n_critical >= 1
    critical_cats = [w.category for w in report.warnings if w.level == "CRITICAL"]
    assert "correlation" in critical_cats


# ═══════════════════════════════════════════════════════════════
# 3. 피처명 패턴 — 미래 시점 / lag0
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("suspect_name", [
    "ili_rate_lead1", "temp_forward2", "cases_future_week",
    "humidity_next_week", "ili_rate_lag0",
])
def test_future_keyword_triggers_critical(suspect_name: str):
    """피처명에 `_lead`, `_forward`, `_future`, `_next`, `_lag0` 포함시 CRITICAL."""
    X_tr, X_te, y_tr, y_te, names = _make_clean_split(n_feat=6)
    names = names[:-1] + [suspect_name]
    report = check_leakage_quick(X_tr, X_te, y_tr, y_te, names)
    assert not report.passed, f"{suspect_name} → CRITICAL 없음"
    temporal_critical = [
        w for w in report.warnings
        if w.level == "CRITICAL" and w.category == "temporal"
        and w.feature == suspect_name
    ]
    assert len(temporal_critical) >= 1


# ═══════════════════════════════════════════════════════════════
# 4. split 경계 중복 — CRITICAL
# ═══════════════════════════════════════════════════════════════

def test_split_boundary_overlap_triggers_critical():
    """Train 마지막 행 = Test 첫 행 → CRITICAL."""
    X_tr, X_te, y_tr, y_te, names = _make_clean_split(n_train=50, n_test=20)
    X_te[0] = X_tr[-1].copy()   # 경계 누출 시뮬레이션
    report = check_leakage_quick(X_tr, X_te, y_tr, y_te, names)
    assert not report.passed
    boundary_warnings = [
        w for w in report.warnings
        if w.level == "CRITICAL" and w.feature == "split_boundary"
    ]
    assert len(boundary_warnings) == 1


# ═══════════════════════════════════════════════════════════════
# 5. (date, gu_nm) 인덱스 교집합 — 운영 단위 누출 테스트
# ═══════════════════════════════════════════════════════════════

def _assert_no_date_gu_overlap(train_keys, val_keys, test_keys):
    """train / val / test 의 (date, gu_nm) 튜플 집합이 서로소인지 확인."""
    s_tr = set(train_keys)
    s_va = set(val_keys)
    s_te = set(test_keys)
    tr_va = s_tr & s_va
    tr_te = s_tr & s_te
    va_te = s_va & s_te
    assert not tr_va, f"train∩val overlap: {len(tr_va)} keys (first 3: {list(tr_va)[:3]})"
    assert not tr_te, f"train∩test overlap: {len(tr_te)} keys (first 3: {list(tr_te)[:3]})"
    assert not va_te, f"val∩test overlap: {len(va_te)} keys (first 3: {list(va_te)[:3]})"


def test_date_gu_split_disjoint_clean():
    """시간순 + 구별 정렬된 clean split 은 교집합이 없어야 함."""
    gus = ["강남구", "종로구", "마포구"]
    dates = [f"2024-W{w:02d}" for w in range(1, 21)]  # 20 weeks
    keys = [(d, g) for d in dates for g in gus]       # 60 keys, ordered

    n = len(keys)
    tr_end = int(n * 0.7)       # 42
    va_end = int(n * 0.85)      # 51
    train_keys = keys[:tr_end]
    val_keys = keys[tr_end:va_end]
    test_keys = keys[va_end:]

    _assert_no_date_gu_overlap(train_keys, val_keys, test_keys)
    # 추가: 합집합 == 전체
    assert set(train_keys) | set(val_keys) | set(test_keys) == set(keys)


def test_date_gu_split_overlap_detected():
    """인위적으로 train∩test 에 (date, gu_nm) 중복을 넣으면 assert 가 발동."""
    train_keys = [("2024-W01", "강남구"), ("2024-W02", "강남구")]
    val_keys = [("2024-W03", "강남구")]
    test_keys = [("2024-W04", "강남구"), ("2024-W01", "강남구")]  # 겹침
    with pytest.raises(AssertionError, match="train∩test overlap"):
        _assert_no_date_gu_overlap(train_keys, val_keys, test_keys)


def test_date_gu_split_within_group_shuffling_detected():
    """구별로는 나뉘었지만 주(week)가 섞인 경우도 교집합 테스트로 잡힌다."""
    # 실수로 val 에서 train 의 (W05, 강남구) 를 사용한 경우
    train_keys = [(f"2024-W{w:02d}", "강남구") for w in range(1, 8)]   # W01~W07
    val_keys = [(f"2024-W{w:02d}", "강남구") for w in [5, 8, 9]]       # W05 중복!
    test_keys = [(f"2024-W{w:02d}", "강남구") for w in range(10, 13)]
    with pytest.raises(AssertionError, match="train∩val overlap"):
        _assert_no_date_gu_overlap(train_keys, val_keys, test_keys)


# ═══════════════════════════════════════════════════════════════
# 6. 중복 행 검사
# ═══════════════════════════════════════════════════════════════

def test_duplicate_row_detection_small():
    """500 행 이하에서 brute-force row match 로 중복 탐지."""
    X_tr, X_te, y_tr, y_te, names = _make_clean_split(n_train=100, n_test=30)
    # test[5] 를 train[50] 과 동일하게
    X_te[5] = X_tr[50].copy()
    report = check_leakage_quick(X_tr, X_te, y_tr, y_te, names)
    dup = [w for w in report.warnings if w.feature == "duplicate_rows"]
    assert len(dup) == 1
    assert dup[0].level == "CRITICAL"
    assert dup[0].value >= 1.0


def test_duplicate_row_detection_large():
    """500 행 초과에서 hash 기반 탐지."""
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((600, 5))
    X_te = rng.standard_normal((100, 5))
    y_tr = rng.standard_normal(600)
    y_te = rng.standard_normal(100)
    # test[10..12] 를 train[100..102] 로 복사
    X_te[10:13] = X_tr[100:103].copy()
    names = [f"f{i}" for i in range(5)]
    report = check_leakage_quick(X_tr, X_te, y_tr, y_te, names)
    dup = [w for w in report.warnings if w.feature == "duplicate_rows"]
    assert len(dup) == 1
    assert dup[0].value >= 3.0


# ═══════════════════════════════════════════════════════════════
# 7. LeakageWarning / LeakageReport 데이터클래스
# ═══════════════════════════════════════════════════════════════

def test_report_add_updates_counts():
    """LeakageReport.add 가 level 별 카운터를 정확히 올리는지."""
    r = LeakageReport()
    r.add(LeakageWarning(level="CRITICAL", category="correlation",
                         feature="f1", message="x", value=0.99))
    r.add(LeakageWarning(level="WARNING", category="target_proxy",
                         feature="f2", message="x", value=0.96))
    r.add(LeakageWarning(level="INFO", category="distribution",
                         feature="f3", message="x", value=0.0))
    assert r.n_critical == 1
    assert r.n_warning == 1
    assert r.n_info == 1
    assert not r.passed   # CRITICAL 이 있으면 passed=False


def test_report_summary_contains_levels():
    """LeakageReport.summary() 출력에 level 표시가 포함."""
    r = LeakageReport()
    r.add(LeakageWarning(level="CRITICAL", category="temporal",
                         feature="foo_lag0", message="lag0 직접 누출",
                         value=1.0))
    summary = r.summary()
    assert "CRITICAL" in summary
    assert "foo_lag0" in summary
    assert "FAILED" in summary


# ═══════════════════════════════════════════════════════════════
# 8. 실행 결합: checker 를 클래스로 직접 호출
# ═══════════════════════════════════════════════════════════════

def test_checker_class_interface():
    """LeakageChecker 인스턴스 인터페이스로 직접 실행."""
    X_tr, X_te, y_tr, y_te, names = _make_clean_split()
    checker = LeakageChecker(X_tr, X_te, y_tr, y_te, names)
    report = checker.run_all_checks()
    assert isinstance(report, LeakageReport)
    assert report is checker.report
