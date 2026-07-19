"""G-319d — lag backbone 입력 helper (modern-ts deep 입력버그 회복).

lag_backbone_seq/from_idx 가 ili_rate_lag* 컬럼을 (n, seq_len, 1) AR backbone 시퀀스로 추출
(oldest→newest, pad/truncate). fit/predict 일관 + lag 부족 시 None(fallback).

macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import numpy as np

from simulation.models.dl_models import lag_backbone_seq, lag_backbone_from_idx


def test_extracts_orders_oldest_to_newest():
    X = np.arange(40).reshape(4, 10).astype(float)
    names = [f"ili_rate_lag{k}" for k in (1, 2, 3, 4)] + ["temp", "hum", "a", "b", "c", "d"]
    seq, idx = lag_backbone_seq(X, names, seq_len=4, min_lags=4)
    assert seq is not None
    assert seq.shape == (4, 4, 1)               # (n, seq_len, 1)
    assert idx == [3, 2, 1, 0]                  # lag4(oldest, col3) → lag1(newest, col0)
    # 마지막 step = lag1 = col0 값
    assert np.allclose(seq[:, -1, 0], X[:, 0])


def test_none_when_no_lag_columns():
    seq, idx = lag_backbone_seq(np.zeros((4, 5)), ["a", "b", "c", "d", "e"], seq_len=4)
    assert seq is None and idx is None


def test_none_when_too_few_lags():
    names = ["ili_rate_lag1", "ili_rate_lag2", "c", "d", "e"]   # 2 < min_lags=4
    seq, _ = lag_backbone_seq(np.zeros((4, 5)), names, seq_len=4, min_lags=4)
    assert seq is None


def test_none_when_names_missing():
    assert lag_backbone_seq(np.zeros((4, 5)), None, seq_len=4)[0] is None
    assert lag_backbone_seq(np.zeros((4, 5)), [], seq_len=4)[0] is None


def test_front_pad_when_fewer_than_seqlen():
    X = np.arange(24).reshape(4, 6).astype(float)
    names = [f"ili_rate_lag{k}" for k in (1, 2, 3, 4, 6, 8)]    # 6 lags
    seq, _ = lag_backbone_seq(X, names, seq_len=12, min_lags=4)
    assert seq.shape == (4, 12, 1)                              # padded to 12


def test_truncate_to_recent_when_more_than_seqlen():
    X = np.arange(40).reshape(4, 10).astype(float)
    names = [f"ili_rate_lag{k}" for k in range(1, 11)]          # 10 lags
    seq, idx = lag_backbone_seq(X, names, seq_len=4, min_lags=4)
    assert seq.shape == (4, 4, 1)                               # most-recent 4 (truncated)
    # newest 4 = lag4,lag3,lag2,lag1 = cols 3,2,1,0
    assert np.allclose(seq[:, -1, 0], X[:, 0])                  # last = lag1


def test_from_idx_matches_seq():
    X = np.arange(40).reshape(4, 10).astype(float)
    names = [f"ili_rate_lag{k}" for k in (1, 2, 3, 4)] + ["a", "b", "c", "d", "e", "f"]
    seq, idx = lag_backbone_seq(X, names, seq_len=4, min_lags=4)
    seq2 = lag_backbone_from_idx(X, idx, seq_len=4)             # predict 경로 = fit 과 동일
    assert np.allclose(seq, seq2)
