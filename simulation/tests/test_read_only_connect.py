"""Smoke test for the lock-free read-only DB helper (database.read_only_connect).

Guards the contract the 3 rewired call sites depend on: reads work, writes are
refused, and a reader opens + reads even while a writer holds the WAL write lock
(never blocks a training writer) — the reason mode=ro is used instead of
safe_connect (which runs quick_check and can block).
"""
import sqlite3

import pytest

from simulation.database import read_only_connect


def _seed(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (k INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()


def test_reads_committed_data(tmp_path):
    db = tmp_path / "ro.db"
    _seed(str(db))
    con = read_only_connect(str(db))
    try:
        assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3
    finally:
        con.close()


def test_blocks_writes(tmp_path):
    db = tmp_path / "ro.db"
    _seed(str(db))
    con = read_only_connect(str(db))
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("INSERT INTO t VALUES (99)")
    finally:
        con.close()


def test_opens_during_write_lock_without_hanging(tmp_path):
    # WAL: a held (uncommitted) writer must NOT block a read-only reader.
    db = tmp_path / "ro.db"
    _seed(str(db))
    writer = sqlite3.connect(str(db))
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO t VALUES (4)")  # uncommitted → holds the write lock
    try:
        con = read_only_connect(str(db), timeout=0.5)  # short: would surface a hang
        try:
            # sees the 3 COMMITTED rows (snapshot), not the uncommitted 4th
            assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3
        finally:
            con.close()
    finally:
        writer.rollback()
        writer.close()


def test_missing_file_raises_not_creates(tmp_path):
    # mode=ro must NOT create the file — a read of a missing DB fails loudly.
    missing = tmp_path / "nope.db"
    with pytest.raises(sqlite3.OperationalError):
        read_only_connect(str(missing))
    assert not missing.exists()
