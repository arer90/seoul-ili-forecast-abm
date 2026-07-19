"""Smoke test for the LLM-comparison visualization (figure rendering)."""
import json


def test_make_figures_smoke(tmp_path):
    from simulation.llm_compare.visualize import make_figures
    report = {
        "n_items": 2, "repro_manifest": {"config_sha256": "testsha"},
        "backends": [{"backend_id": "cli:claude:x", "tier": "cli"},
                     {"backend_id": "oai:mlx-community/Qwen@u", "tier": "openai_compat"}],
        "ranking": [{"backend_id": "cli:claude:x", "accuracy": 0.8, "tier": "cli"},
                    {"backend_id": "oai:mlx-community/Qwen@u", "accuracy": 0.5,
                     "tier": "openai_compat"}],
        "statistical_comparison": {"ranking": [
            {"backend": "cli:claude:x", "mean": 0.8, "lo": 0.6, "hi": 0.95, "n": 2},
            {"backend": "oai:mlx-community/Qwen@u", "mean": 0.5, "lo": 0.3, "hi": 0.7, "n": 2}]},
        "per_item": [
            {"item_id": "KL01", "backend_id": "cli:claude:x", "total": 1.0,
             "latency_ms": 1000.0, "error": ""},
            {"item_id": "KMQ_doctor_x", "backend_id": "oai:mlx-community/Qwen@u",
             "total": 0.0, "latency_ms": 80.0, "error": ""}],
    }
    p = tmp_path / "report.json"; p.write_text(json.dumps(report), encoding="utf-8")
    paths = make_figures(p, tmp_path / "fig")
    assert len(paths) == 5                          # 4 panels + combined
    assert all(x.exists() and x.stat().st_size > 0 for x in paths)


def test_short_label():
    from simulation.llm_compare.visualize import _short
    assert _short("cli:claude:claude-default") == "claude"
    assert _short("oai:mlx-community/Qwen2.5-7B-Instruct-4bit@http://h/v1") == "Qwen2.5-7B"
