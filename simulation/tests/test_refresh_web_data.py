"""M4: refresh_web_data orchestration (db→web sync) — degrade-and-continue."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web" / "scripts"))
import refresh_web_data as rwd  # noqa: E402


def test_runs_existing_step(monkeypatch, tmp_path):
    s = tmp_path / "ok.py"; s.write_text("", encoding="utf-8")
    monkeypatch.setattr(rwd.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    assert rwd.refresh([("s1", ["py", str(s)])]) == {"s1": "ok"}


def test_missing_script(tmp_path):
    assert rwd.refresh([("m", ["py", str(tmp_path / "nope.py")])]) == {"m": "missing-script"}


def test_isolates_failure(monkeypatch, tmp_path):
    a = tmp_path / "a.py"; a.write_text("", encoding="utf-8"); b = tmp_path / "b.py"; b.write_text("", encoding="utf-8")
    ran = []

    def _run(cmd, **k):
        ran.append(cmd[1])
        return type("R", (), {"returncode": 2 if cmd[1].endswith("a.py") else 0})()

    monkeypatch.setattr(rwd.subprocess, "run", _run)
    res = rwd.refresh([("a", ["py", str(a)]), ("b", ["py", str(b)])])
    assert res["a"].startswith("fail") and res["b"] == "ok"  # b ran despite a failing
    assert len(ran) == 2


# ── D3 (M7, corrected): local architecture = no Turso/Vercel ──
def test_local_refresh_excludes_turso_seed():
    """Local refresh must NOT dump the Turso seed — the user runs everything
    locally (web reads live via /api/mcp → MCP server); Turso is deploy-only."""
    by_name = dict(rwd.STEPS)
    assert "turso-seed" not in by_name, "turso seed wrongly wired into local refresh"
    # the live local-data builders ARE wired
    assert "trained-models" in by_name and "seir-metapop-init" in by_name


def test_vintage_sql_emits_timestamped_row():
    """export-turso emits web_data_vintage for an honest 'data as of' badge."""
    import importlib.util
    import sqlite3

    path = Path(__file__).resolve().parents[2] / "web" / "scripts" / "export-turso.py"
    spec = importlib.util.spec_from_file_location("export_turso_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE weekly_disease (week TEXT, vintage_ts TEXT)")
    con.execute("INSERT INTO weekly_disease VALUES ('2026-W22', '2026-06-01T00:00:00')")
    sql = "\n".join(mod._vintage_sql(con))
    assert "web_data_vintage" in sql and "generated_at" in sql
    assert "2026-06-01" in sql  # db_max_vintage picked up from vintage_ts
