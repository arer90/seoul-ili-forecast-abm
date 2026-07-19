"""D7 (M7): per-run reproducibility manifest (git+env+vintage+config_sha256)."""
from simulation.pipeline import run_manifest as rm
from simulation.pipeline.run_manifest import build_run_manifest, write_run_manifest


def test_manifest_has_required_keys(tmp_path):
    m = write_run_manifest(tmp_path, seed=7)
    assert m["seed"] == 7
    for k in ("git_sha", "package_hash", "db_vintage_ts", "env", "config_sha256"):
        assert k in m
    assert len(m["config_sha256"]) == 64
    assert (tmp_path / "run_manifest.json").exists()


def test_config_sha_is_deterministic(monkeypatch):
    monkeypatch.setattr(rm, "_git_sha", lambda: "abc")
    monkeypatch.setattr(rm, "_package_hash", lambda: "def")
    monkeypatch.setattr(rm, "_db_vintage", lambda: "2026-06-06")
    a = build_run_manifest(seed=1)
    b = build_run_manifest(seed=1)
    assert a["config_sha256"] == b["config_sha256"]
    c = build_run_manifest(seed=2)
    assert c["config_sha256"] != a["config_sha256"]  # seed enters the hash


def test_d1_provenance_surfaces_manifest_config_sha(tmp_path):
    from simulation.server.mcp_epi import CallResult, EpiMCPServer
    m = write_run_manifest(tmp_path, seed=42)
    (tmp_path / "x.csv").write_text("a\n", encoding="utf-8")
    srv = EpiMCPServer(artifacts_dir=tmp_path)
    prov = srv._attach_provenance(
        CallResult(content={"status": "live", "source": "x.csv"})).content["provenance"]
    assert prov["config_sha256"] == m["config_sha256"]
