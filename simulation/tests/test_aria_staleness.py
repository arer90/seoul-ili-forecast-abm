"""D2 (M7 SCI-grade): ARIA staleness guard — never serve a stale artifact silently.

The MCP server checks artifact *existence* but never *age* vs the DB. After D2 a
result over an artifact older than the DB's latest data is flagged
``freshness:'STALE'`` (+ ``freshness_detail``) while keeping ``status:'live'`` so
Hermes/UI can surface the caveat. Tests mock ``_db_data_time`` for determinism.
"""
from datetime import datetime, timedelta

from simulation.server.mcp_epi import CallResult, EpiMCPServer


def _srv_with_artifact(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("model,wis\nA,1.0\n", encoding="utf-8")
    return EpiMCPServer(artifacts_dir=tmp_path), f


def test_fresh_artifact_marked_live(tmp_path, monkeypatch):
    srv, f = _srv_with_artifact(tmp_path)
    art_mtime = datetime.fromtimestamp(f.stat().st_mtime)
    monkeypatch.setattr(srv, "_db_data_time", lambda: art_mtime - timedelta(hours=2))
    out = srv._attach_provenance(CallResult(content={"status": "live", "source": "m.csv"}))
    assert out.content["freshness"] == "LIVE"
    assert "freshness_detail" not in out.content


def test_stale_artifact_flagged_but_still_live(tmp_path, monkeypatch):
    srv, f = _srv_with_artifact(tmp_path)
    art_mtime = datetime.fromtimestamp(f.stat().st_mtime)
    monkeypatch.setattr(srv, "_db_data_time", lambda: art_mtime + timedelta(days=10))
    out = srv._attach_provenance(CallResult(content={"status": "live", "source": "m.csv"}))
    assert out.content["freshness"] == "STALE"
    assert out.content["status"] == "live"  # not hidden — flagged, not dropped
    assert out.content["freshness_detail"]["db_ahead_days"] >= 9
    assert "newer" in out.content["freshness_detail"]["reason"]


def test_unknown_when_db_time_missing(tmp_path, monkeypatch):
    srv, _f = _srv_with_artifact(tmp_path)
    monkeypatch.setattr(srv, "_db_data_time", lambda: None)
    out = srv._attach_provenance(CallResult(content={"status": "live", "source": "m.csv"}))
    assert out.content["freshness"] == "UNKNOWN"


def test_no_freshness_without_artifact_source(tmp_path):
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    out = srv._attach_provenance(CallResult(content={"status": "live"}))  # no source
    assert "freshness" not in out.content  # nothing to judge
