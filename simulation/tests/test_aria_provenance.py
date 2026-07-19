"""D1 (M7 SCI-grade): MCP provenance envelope — every advisory is traceable.

call_tool stamps content['provenance'] = {server_version, db_vintage_ts, +
artifact_path/mtime/sha256/config_sha256 when the handler declared a file
`source`}, so a reader can reconstruct which model version + data epoch produced
an advisory. Error / non-dict results are left untouched.
"""
import hashlib

from simulation.server.mcp_epi import CallResult, EpiMCPServer


def test_provenance_has_server_version_and_db_vintage(tmp_path):
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    prov = srv._attach_provenance(CallResult(content={"status": "live"})).content["provenance"]
    assert prov["server_version"] == srv.SERVER_VERSION
    assert "db_vintage_ts" in prov  # value may be None when DB is unavailable


def test_provenance_includes_artifact_path_mtime_sha(tmp_path):
    f = tmp_path / "metrics.csv"
    f.write_text("model,wis\nA,1.0\n", encoding="utf-8")
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    prov = srv._attach_provenance(
        CallResult(content={"status": "live", "source": "metrics.csv"})).content["provenance"]
    assert prov["artifact_path"] == "metrics.csv"
    assert prov["artifact_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert "artifact_mtime_iso" in prov


def test_provenance_skips_error_and_nondict(tmp_path):
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    err = srv._attach_provenance(CallResult(content={"error": "boom"}, is_error=True))
    assert "provenance" not in err.content
    txt = srv._attach_provenance(CallResult(content="plain text"))
    assert txt.content == "plain text"  # non-dict untouched


def test_provenance_surfaces_config_sha_from_manifest(tmp_path):
    (tmp_path / "run_manifest.json").write_text('{"config_sha256": "abc123"}', encoding="utf-8")
    (tmp_path / "m.csv").write_text("x\n", encoding="utf-8")
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    prov = srv._attach_provenance(
        CallResult(content={"status": "live", "source": "m.csv"})).content["provenance"]
    assert prov["config_sha256"] == "abc123"


def test_call_tool_attaches_provenance_end_to_end(tmp_path):
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    res = srv.call_tool("epi.model_compare", {})
    assert isinstance(res.content, dict)
    if not res.content.get("error"):
        assert "provenance" in res.content


# ── ARIA audit ledger SHA-256 hash chain (tamper-evidence) ────────────────────
# _append_audit builds an append-only chain; verify_audit_chain proves any edit,
# insertion, deletion, or reorder is detectable. This is the code proof behind
# the thesis "tamper-evident" claim for the LLM-comparison provenance ledger.

def _build_chain(n=4):
    from simulation.llm_compare.runner import _append_audit
    chain: list[dict] = []
    for i in range(n):
        _append_audit(chain, {"event": f"step.{i}", "i": i, "payload": f"v{i}"})
    return chain


def test_clean_chain_verifies_intact():
    from simulation.llm_compare.runner import verify_audit_chain
    chain = _build_chain(5)
    res = verify_audit_chain(chain)
    assert res["intact"] is True
    assert res["n_entries"] == 5
    assert res["first_bad_index"] is None


def test_tampered_payload_breaks_verification():
    from simulation.llm_compare.runner import verify_audit_chain
    chain = _build_chain(4)
    # tamper: edit a payload field WITHOUT recomputing the hash
    chain[2]["payload"] = "TAMPERED"
    res = verify_audit_chain(chain)
    assert res["intact"] is False
    assert res["first_bad_index"] == 2
    assert "hash mismatch" in res["reason"]


def test_deleted_entry_breaks_linkage():
    from simulation.llm_compare.runner import verify_audit_chain
    chain = _build_chain(4)
    del chain[1]  # removing an entry breaks the prev_hash linkage of the next
    res = verify_audit_chain(chain)
    assert res["intact"] is False
    assert res["first_bad_index"] == 1


def test_reordered_entries_break_linkage():
    from simulation.llm_compare.runner import verify_audit_chain
    chain = _build_chain(4)
    chain[1], chain[2] = chain[2], chain[1]
    res = verify_audit_chain(chain)
    assert res["intact"] is False


def test_empty_chain_is_trivially_intact():
    from simulation.llm_compare.runner import verify_audit_chain
    res = verify_audit_chain([])
    assert res["intact"] is True and res["n_entries"] == 0
