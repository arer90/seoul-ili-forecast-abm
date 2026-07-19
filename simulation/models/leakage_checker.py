"""
simulation/models/leakage_checker.py
=====================================
Data Leakage 탐지 및 경고 시스템.

3가지 레벨에서 누출을 검사:
  1) 피처-타겟 상관 이상 탐지 (비현실적으로 높은 상관)
  2) Train/Test 통계 누출 (전처리 시 test 정보 사용 여부)
  3) 시간 순서 위반 탐지 (미래 → 과거 정보 흐름)

사용법:
    from simulation.models.leakage_checker import LeakageChecker
    checker = LeakageChecker(X_train, X_test, y_train, y_test, feature_names)
    report = checker.run_all_checks()
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class LeakageWarning:
    """단일 누출 경고."""
    level: str          # "CRITICAL", "WARNING", "INFO"
    category: str       # "correlation", "temporal", "distribution", "target_proxy"
    feature: str        # 문제 피처명
    message: str        # 상세 설명
    value: float = 0.0  # 관련 수치


@dataclass
class LeakageReport:
    """전체 누출 검사 리포트."""
    warnings: list[LeakageWarning] = field(default_factory=list)
    n_critical: int = 0
    n_warning: int = 0
    n_info: int = 0
    passed: bool = True

    def add(self, w: LeakageWarning):
        self.warnings.append(w)
        if w.level == "CRITICAL":
            self.n_critical += 1
            self.passed = False
        elif w.level == "WARNING":
            self.n_warning += 1
        else:
            self.n_info += 1

    def summary(self) -> str:
        lines = [
            f"=== Data Leakage Report ===",
            f"  CRITICAL: {self.n_critical}  |  WARNING: {self.n_warning}  |  INFO: {self.n_info}",
            f"  Overall: {'✓ PASSED' if self.passed else '✗ FAILED -- 누출 의심'}",
        ]
        for w in self.warnings:
            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(w.level, "")
            # 0-D: include the offending feature name so the summary
            # is actionable. Previously w.feature was only inside w.message,
            # which meant automated parsers / pytest could not reliably
            # identify which column flagged the warning.
            _feat = f" feature={w.feature}" if getattr(w, "feature", None) else ""
            lines.append(
                f"  {icon} [{w.level}] {w.category}:{_feat} {w.message} "
                f"(val={w.value:.4f})"
            )
        return "\n".join(lines)


class LeakageChecker:
    """Data Leakage 종합 검사기."""

    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        y_val: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.y_val = y_val
        self.X_val = X_val
        self.report = LeakageReport()

    def run_all_checks(self) -> LeakageReport:
        """모든 누출 검사를 순차 실행."""
        log.info("=== Data Leakage 검사 시작 ===")
        self._check_perfect_correlation()
        self._check_target_proxy()
        self._check_train_test_distribution_leak()
        self._check_temporal_ordering()
        self._check_feature_future_leak()
        self._check_duplicate_rows()

        # 결과 로깅
        log.info(self.report.summary())
        return self.report

    # ── 1. 비현실적 상관 탐지 ──
    def _check_perfect_correlation(self, threshold: float = 0.98):
        """피처-타겟 간 비현실적으로 높은 상관 탐지."""
        for i, fname in enumerate(self.feature_names):
            col = self.X_train[:, i]
            if np.std(col) < 1e-10:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = np.corrcoef(col, self.y_train)[0, 1]
            if np.isnan(corr):
                continue
            if abs(corr) >= threshold:
                self.report.add(LeakageWarning(
                    level="CRITICAL",
                    category="correlation",
                    feature=fname,
                    message=f"피처-타겟 상관 {corr:.4f} ≥ {threshold} -- 직접 누출 의심",
                    value=abs(corr),
                ))

    # ── 2. 타겟 프록시 탐지 ──
    def _check_target_proxy(self, threshold: float = 0.95):
        """현재 시점 타겟의 직접 프록시(동시 값) 탐지.
 lag0 피처가 혼입되었는지 확인.

 E-1: lag1-derived interaction 피처 (humid_ili, subway_ili, bus_ili 등)
 는 ili_rate_lag1 × (humidity|subway|bus|school|...) / 100 꼴로 정의되어
 있고, y 의 자기상관(0.85~0.95) 때문에 이름에 "lag" 가 없어도 y_train 과
 매우 높은 상관을 보인다. causal 로 정의된 feature 이므로 타겟 프록시가
 아닌 false positive 다. → "_ili" suffix 를 whitelist 에 추가한다.
 """
        # "lag0" 또는 현재 시점 키워드가 없는데 상관이 매우 높은 피처
        for i, fname in enumerate(self.feature_names):
            # lag/shift 가 포함되거나, lag1-derived interaction (ends with "_ili") 는 스킵
            if "lag" in fname or "shift" in fname:
                continue
            # E-1: lag1-derived interactions (humid_ili, subway_ili, bus_ili,
            # school_ili, age_mixing_ili 등 _add_interaction_features 에서 ili_rate_lag1
            # 로부터 파생된 피처) 화이트리스트
            if fname.endswith("_ili"):
                continue
            col = self.X_train[:, i]
            # E-3: std=0 이면 상관 정의 불가 → 스킵
            if np.std(col) < 1e-10:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = np.corrcoef(col, self.y_train)[0, 1]
            if np.isnan(corr):
                continue
            if abs(corr) >= threshold:
                self.report.add(LeakageWarning(
                    level="WARNING",
                    category="target_proxy",
                    feature=fname,
                    message=f"lag 없는 피처인데 타겟 상관 {corr:.4f} -- 동시 변수 프록시 가능성",
                    value=abs(corr),
                ))

    # ── 3. Train/Test 분포 누출 탐지 ──
    def _check_train_test_distribution_leak(self, ks_threshold: float = 0.01):
        """Train과 Test 피처 분포가 비정상적으로 동일하면 누출 의심.
        전처리 시 전체 데이터로 스케일링했을 가능성."""
        from scipy import stats as sp_stats

        n_identical = 0
        for i, fname in enumerate(self.feature_names):
            tr_col = self.X_train[:, i]
            te_col = self.X_test[:, i]
            if np.std(tr_col) < 1e-10 and np.std(te_col) < 1e-10:
                continue

            # KS test: 두 분포가 동일한지 검정
            try:
                ks_stat, ks_p = sp_stats.ks_2samp(tr_col, te_col)
            except Exception:
                continue

            # Train/Test mean, std가 거의 동일하면 의심
            tr_mean, te_mean = np.mean(tr_col), np.mean(te_col)
            tr_std, te_std = np.std(tr_col), np.std(te_col)

            # 스케일링 누출: mean≈0, std≈1 인데 train과 test 모두 해당
            if (abs(tr_mean) < 0.05 and abs(te_mean) < 0.05 and
                abs(tr_std - 1) < 0.05 and abs(te_std - 1) < 0.05):
                n_identical += 1

        if n_identical > len(self.feature_names) * 0.5:
            self.report.add(LeakageWarning(
                level="WARNING",
                category="distribution",
                feature=f"{n_identical}개 피처",
                message=f"Train/Test 모두 mean≈0, std≈1 -- 전체 데이터로 스케일링했을 가능성 "
                        f"({n_identical}/{len(self.feature_names)})",
                value=n_identical / len(self.feature_names),
            ))

    # ── 4. 시간 순서 위반 탐지 ──
    def _check_temporal_ordering(self):
        """Train 마지막 인덱스 < Test 첫 인덱스 확인 (겹침 탐지)."""
        # 간접 검사: Train의 마지막 행과 Test의 첫 행이 동일하면 겹침
        if len(self.X_train) > 0 and len(self.X_test) > 0:
            last_train = self.X_train[-1]
            first_test = self.X_test[0]
            if np.allclose(last_train, first_test, atol=1e-8):
                self.report.add(LeakageWarning(
                    level="CRITICAL",
                    category="temporal",
                    feature="split_boundary",
                    message="Train 마지막 행 = Test 첫 행 -- 데이터 겹침 발생",
                    value=1.0,
                ))

        # y값 겹침 검사
        if len(self.y_train) > 0 and len(self.y_test) > 0:
            overlap = set(range(len(self.y_train))) & set(range(len(self.y_test)))
            # 인덱스가 아닌 값 기반 겹침은 자연스러울 수 있으므로 skip

    # ── 5. 미래 정보 피처 탐지 ──
    def _check_feature_future_leak(self):
        """피처 이름 패턴으로 미래 정보 혼입 탐지."""
        suspicious_patterns = ["_lead", "_forward", "_future", "_next"]
        for fname in self.feature_names:
            fname_lower = fname.lower()
            for pat in suspicious_patterns:
                if pat in fname_lower:
                    self.report.add(LeakageWarning(
                        level="CRITICAL",
                        category="temporal",
                        feature=fname,
                        message=f"피처명에 미래 시점 키워드 '{pat}' 포함 -- 미래 정보 누출 의심",
                        value=1.0,
                    ))

        # lag0 (현재 시점) 피처 탐지
        for fname in self.feature_names:
            if "_lag0" in fname.lower():
                self.report.add(LeakageWarning(
                    level="CRITICAL",
                    category="temporal",
                    feature=fname,
                    message="lag0 피처 -- 현재 시점 타겟 정보 직접 누출",
                    value=1.0,
                ))

    # ── 6. 중복 행 탐지 ──
    def _check_duplicate_rows(self):
        """Train과 Test 간 완전 동일 행 탐지."""
        if len(self.X_train) > 500:
            # 대용량: 해시 비교
            tr_hashes = set(hash(row.tobytes()) for row in self.X_train)
            n_dup = sum(1 for row in self.X_test if hash(row.tobytes()) in tr_hashes)
        else:
            # 소용량: 직접 비교
            n_dup = 0
            for te_row in self.X_test:
                for tr_row in self.X_train:
                    if np.allclose(te_row, tr_row, atol=1e-8):
                        n_dup += 1
                        break

        if n_dup > 0:
            self.report.add(LeakageWarning(
                level="CRITICAL",
                category="temporal",
                feature="duplicate_rows",
                message=f"Train/Test 간 동일 행 {n_dup}개 발견 -- 데이터 중복 누출",
                value=float(n_dup),
            ))


def check_leakage_quick(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> LeakageReport:
    """편의 함수: 한 줄로 누출 검사 실행."""
    checker = LeakageChecker(X_train, X_test, y_train, y_test, feature_names)
    return checker.run_all_checks()
