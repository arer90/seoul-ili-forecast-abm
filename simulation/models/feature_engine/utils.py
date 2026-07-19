"""
Utility functions and helpers for feature engineering.

- Database connection utilities
- Time mapping helpers
- TimeSeriesAugmentor class for data augmentation
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 데이터베이스 연결 헬퍼
# ═══════════════════════════════════════════════════════════════

@contextmanager
def _db_conn(db_path: str):
    """SQLite 연결 컨텍스트 매니저 — 자동 close + 안전 PRAGMA.

 : `simulation.database.safe_connect` 로 단일화. simulation 패키지
 내부에서 호출되므로 해당 import 는 **항상** 성공한다. 과거의
 `except ImportError: sqlite3.connect(...)` 폴백은 (a) verify-audit
 `sqlite3_connect_bypass` 위반을 만들고 (b) safe_connect 의 quick_check /
 튜닝 PRAGMA 를 우회하므로 제거했다.
 """
    from simulation.database import safe_connect
    conn = safe_connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _read_sql(sql: str, db_path: str) -> pl.DataFrame:
    """SQLite → polars DataFrame 읽기. `safe_connect` 단일 경로.

 2026-04-20: sparse-null columns (e.g. rt_sdot_env.noise) caused
 polars schema-inference to infer Null from leading rows and then
 ComputeError on later f64 values. Build columnwise + infer_schema_length=None
 to scan all rows. Returns empty df on query failure (preserves old behavior).
 """
    from simulation.database import safe_connect
    conn = safe_connect(db_path)
    try:
        cursor = conn.execute(sql)
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        if not rows:
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
        col_data = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
        return pl.DataFrame(col_data, infer_schema_length=None)
    finally:
        conn.close()


def _season_weekseq_to_date(season_start: int, week_seq: int) -> datetime:
    """(season_start, week_seq) → 해당 주의 월요일 날짜.

    한국 인플루엔자 시즌: season_start년 ISO W36(9월 첫째 주) 시작.
    week_seq=1 → ISO W36, week_seq=2 → ISO W37, ...
    """
    jan1 = datetime(season_start, 1, 1)
    # ISO week 1의 월요일 찾기
    if jan1.weekday() <= 3:  # Mon–Thu
        iso_w1_mon = jan1 - timedelta(days=jan1.weekday())
    else:
        iso_w1_mon = jan1 + timedelta(days=7 - jan1.weekday())

    # ISO W36 start
    w36_mon = iso_w1_mon + timedelta(weeks=35)
    return w36_mon + timedelta(weeks=week_seq - 1)


# ═══════════════════════════════════════════════════════════════
# 시계열 데이터 증강기
# ═══════════════════════════════════════════════════════════════

class TimeSeriesAugmentor:
    """시계열 데이터 증강기."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def jitter(self, X: np.ndarray, sigma: float = 0.03) -> np.ndarray:
        """가우시안 노이즈 추가."""
        noise = self.rng.normal(0, sigma, X.shape)
        return X + noise

    def scale(self, X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """랜덤 스케일링."""
        n_feat = X.shape[-1]
        scales = self.rng.normal(1, sigma, (1,) * (X.ndim - 1) + (n_feat,))
        return X * scales

    def window_warp(self, X: np.ndarray, ratio: float = 0.1) -> np.ndarray:
        """시간축 워핑 (부분 구간 확대/축소)."""
        if X.ndim != 2 or len(X) < 10:
            return X
        n = len(X)
        warp_size = max(int(n * ratio), 2)
        start = self.rng.integers(0, n - warp_size)
        factor = self.rng.uniform(0.5, 2.0)

        warp_idx = np.linspace(start, start + warp_size - 1, int(warp_size * factor))
        warp_idx = np.clip(warp_idx, 0, n - 1)
        warped_segment = np.array([
            np.interp(warp_idx, np.arange(n), X[:, j])
            for j in range(X.shape[1])
        ]).T

        result = np.copy(X)
        insert_len = min(len(warped_segment), n - start)
        result[start:start + insert_len] = warped_segment[:insert_len]
        return result

    def mixup(self, X1: np.ndarray, y1: np.ndarray,
              X2: np.ndarray, y2: np.ndarray,
              alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Mixup: 두 시퀀스 가중 평균."""
        lam = self.rng.beta(alpha, alpha)
        X_mix = lam * X1 + (1 - lam) * X2
        y_mix = lam * y1 + (1 - lam) * y2
        return X_mix, y_mix

    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_augments: int = 3,
        methods: list[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """데이터셋 증강."""
        if methods is None:
            methods = ["jitter", "scale", "jitter+scale"]

        all_X = [X]
        all_y = [y]

        for _ in range(n_augments):
            method = self.rng.choice(methods)
            X_aug = np.copy(X)

            if "jitter" in method:
                X_aug = self.jitter(X_aug)
            if "scale" in method:
                X_aug = self.scale(X_aug)

            all_X.append(X_aug)
            all_y.append(y.copy())

        n_mix = len(X) // 4
        if n_mix > 0:
            idx1 = self.rng.integers(0, len(X), n_mix)
            idx2 = self.rng.integers(0, len(X), n_mix)
            X_mix, y_mix = self.mixup(X[idx1], y[idx1], X[idx2], y[idx2])
            all_X.append(X_mix)
            all_y.append(y_mix)

        return np.vstack(all_X), np.concatenate(all_y)