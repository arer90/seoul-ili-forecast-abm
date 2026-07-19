"""Guard: the from-scratch ARIA torch LM is usable as a SubQ (Self-Ask) backend.

Confirms the WIRING (the from-scratch torch model satisfies the backend-agnostic
SubQ contract and self_ask runs end-to-end with it). The model is WEAK, so this
checks the integration, not answer quality.

Run per-file:
    .venv/bin/python -m pytest simulation/tests/test_torch_lm_subq.py -q
"""
from pathlib import Path

import pytest

CKPT = Path(__file__).resolve().parents[1] / "results" / "aria_torch_lm" / "aria_torch_lm.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="from-scratch torch LM not trained yet")
def test_torch_lm_backend_generates():
    from simulation.scripts.train_aria_torch_lm import TorchLMBackend
    b = TorchLMBackend()
    assert b.is_available() is True
    r = b.generate("인플루엔자란", max_tokens=16)
    assert isinstance(getattr(r, "text", None), str)
    assert getattr(r, "error", "") == ""


@pytest.mark.skipif(not CKPT.exists(), reason="from-scratch torch LM not trained yet")
def test_subq_runs_with_torch_backend():
    from simulation.scripts.train_aria_torch_lm import TorchLMBackend
    from simulation.llm_compare.subq import self_ask
    b = TorchLMBackend()
    res = self_ask(
        "What is influenza vaccine effectiveness?",
        ["Influenza vaccine effectiveness is about 60% in healthy adults [data:ve]"],
        backend=b, max_tokens=40,
    )
    # The from-scratch torch model drives the Self-Ask flow end-to-end (weak output).
    assert res is not None
