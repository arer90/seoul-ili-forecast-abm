"""TDD for `python -m simulation aria` (cmd_aria) — FAIL-LOUD + no silent mock.

The on-path P4 ARIA CLI must never silently degrade to a mock that looks like
agent reasoning (G-237). Without a live backend and without an explicit --mock,
it must exit nonzero and leak no answer to stdout. With --mock it runs an offline
structural check that is clearly labelled NOT-advisory.
"""
from __future__ import annotations

from argparse import Namespace

import pytest


def _args(**kw):
    base = dict(query="행동을 켠 ABM 전향 예측이 나은가? 수치 근거로.",
                root=None, mock=False, host="http://127.0.0.1:11434",
                deep=False, stream=False)
    base.update(kw)
    return Namespace(**base)


def test_cmd_aria_failloud_when_no_live_backend(monkeypatch, capsys):
    from simulation.cli import aria_commands as A
    monkeypatch.setattr(A, "_ollama_available", lambda host: False)  # force down
    with pytest.raises(SystemExit) as ei:
        A.cmd_aria(_args(mock=False))
    assert ei.value.code == 1                      # loud nonzero
    out = capsys.readouterr().out
    assert "행동 ON" not in out and "final" not in out.lower()  # NO answer leaked


def test_cmd_aria_mock_runs_and_labels_not_advisory(capsys):
    from simulation.cli.aria_commands import cmd_aria
    # mock needs no Ollama; must run, show blackboard receipts, and label MOCK
    rc = cmd_aria(_args(mock=True))
    out = capsys.readouterr().out
    assert rc in (None, 0)
    assert "MOCK" in out.upper()                    # clearly not deliverable advisory
    assert "read_artifact" in out                   # provenance receipts shown
    assert ("forward_r2" in out) or ("0.722" in out)


def test_cmd_aria_requires_query(capsys):
    from simulation.cli import aria_commands as A
    with pytest.raises(SystemExit) as ei:
        A.cmd_aria(_args(query=None))
    assert ei.value.code == 2


def test_cmd_aria_deep_renders_specialists_and_gate(monkeypatch, capsys):
    # isolate the thin deep CLI branch from the (slow, DB-backed) real server
    canned = {
        "query": "q",
        "specialists": [{"role": "Forecast Intelligence", "facts_written": 1,
                         "receipts": [{"tool": "epi.forecast"}]}],
        "blackboard": [{"agent": "forecast_intelligence",
                        "claim": "forecast_intelligence.rel_wis", "value": 0.443,
                        "provenance": {"tool": "epi.forecast"}}],
        "final_answer": "[mock] 상대 WIS 0.443.",
        "verification": {"grounded": True, "n_spurious": 0},
        "gate": {"safe": True, "action": "pass", "reason": "grounded"},
        "grounded": True,
    }
    monkeypatch.setattr("simulation.llm_compare.orchestrator.run_orchestrated",
                        lambda *a, **k: canned)
    from simulation.cli.aria_commands import cmd_aria
    rc = cmd_aria(_args(mock=True, deep=True))
    out = capsys.readouterr().out
    assert rc in (None, 0)
    assert "Forecast Intelligence" in out          # specialist shown
    assert "epi.forecast" in out                    # tool receipt shown
    assert "0.443" in out                           # blackboard number rendered

