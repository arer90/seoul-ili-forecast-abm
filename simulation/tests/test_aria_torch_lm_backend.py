"""Guard: the from-scratch ARIA torch LM is a flag-gated comparison backend.

Confirms (a) it is NOT in the default backend pool (env-safe — the existing
Claude/Gemini/Ollama comparison is unchanged), (b) it joins the pool when the
flag is set and the checkpoint exists, (c) it produces a valid LLMResponse.
The model is WEAK; this checks the WIRING into ARIA's multi-model comparison,
not answer quality.

Run per-file:
    .venv/bin/python -m pytest simulation/tests/test_aria_torch_lm_backend.py -q
"""
from pathlib import Path

import pytest

from simulation.llm_compare.backends import AriaTorchLMBackend, discover_backends

CKPT = Path(__file__).resolve().parents[1] / "results" / "aria_modern_lm" / "aria_modern_lm.pt"


def test_not_in_default_pool(monkeypatch):
    """Flag off (default) -> torch LM absent; the existing comparison pool is unchanged."""
    monkeypatch.delenv("MPH_ARIA_TORCH_LM", raising=False)
    ids = {b.backend_id for b in discover_backends(
        include_api=False, include_cli=False, include_ollama=False, include_mock=True)}
    assert "local:aria-torch-lm" not in ids


def test_joins_pool_when_enabled():
    """Flag on + checkpoint present -> torch LM is in the comparison pool."""
    ids = {b.backend_id for b in discover_backends(
        include_api=False, include_cli=False, include_ollama=False,
        include_mock=False, include_aria_torch_lm=True)}
    if CKPT.exists():
        assert "local:aria-torch-lm" in ids
    else:                                   # not trained -> not available -> not added
        assert "local:aria-torch-lm" not in ids


@pytest.mark.skipif(not CKPT.exists(), reason="from-scratch modern torch LM not trained yet")
def test_backend_generates():
    b = AriaTorchLMBackend()
    assert b.is_available() is True
    r = b.generate("Influenza surveillance in Seoul is", max_tokens=24)
    assert r.error == ""
    assert isinstance(r.text, str)
