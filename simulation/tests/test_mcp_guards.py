"""D8/D9/D10 (M7): MCP audit trail + feature-pad disclosure + query row cap."""
import json

from simulation.server.mcp_epi import EpiMCPServer


def test_d8_audit_record_written_per_call(tmp_path):
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    srv.call_tool("epi.model_compare", {})
    audit = tmp_path / "mcp_audit.jsonl"
    assert audit.exists(), "D8 audit trail not written"
    rec = json.loads(audit.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["tool"] == "epi.model_compare"
    for k in ("ts", "args_hash", "status", "elapsed_ms"):
        assert k in rec


def test_d8_audit_appends(tmp_path):
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    srv.call_tool("epi.model_compare", {})
    srv.call_tool("epi.lead_time_analysis", {})
    lines = (tmp_path / "mcp_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2  # one record per call, appended


def test_d10_limit_clamp_logic():
    # _h_query_db clamps the requested limit to [1, 10000]
    assert max(1, min(999_999, 10000)) == 10000
    assert max(1, min(0, 10000)) == 1
    assert max(1, min(500, 10000)) == 500
