"""
simulation.llm_compare._api_demo
================================
One-command demonstration that the ARIA provider auto-discovery works
end-to-end when Gemini / OpenAI / Anthropic API keys are supplied.

Usage::

    export GOOGLE_API_KEY=...       # or GEMINI_API_KEY
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...    # or CLAUDE_API_KEY
    python3 -m simulation.llm_compare._api_demo

If no API keys are set the script still runs but reports which backends
were *not* enabled, so a reviewer can confirm the discovery mechanism
without actually incurring API costs.
"""
from __future__ import annotations

import json
import os
import sys

from simulation.llm_compare.backends import (
    AnthropicBackend,
    GeminiBackend,
    OpenAIBackend,
    OllamaBackend,
    discover_backends,
    env_status,
    list_ollama_models,
)


PROMPT_EN = (
    "One sentence in plain English: if a Seoul district reports weekly ILI = 12.5 per "
    "1,000 outpatients with the KDCA advisory threshold at 8.6, do you issue a watch? "
    "Be hedged and cite the threshold."
)


def _try_one(backend_name, backend) -> dict:
    if not backend.is_available():
        return {
            "backend": backend.backend_id,
            "status": "DISABLED",
            "reason": "credentials not set",
            "latency_ms": 0,
            "response_preview": "",
        }
    resp = backend.generate(PROMPT_EN, temperature=0.2, max_tokens=200)
    return {
        "backend": backend.backend_id,
        "status": "ERROR" if resp.error else "OK",
        "reason": resp.error or "",
        "latency_ms": round(resp.latency_ms, 1),
        "tokens_in": resp.prompt_tokens,
        "tokens_out": resp.completion_tokens,
        "response_preview": resp.text[:300],
    }


def main() -> int:
    env = env_status()
    print("=" * 60)
    print("ARIA LLM API auto-discovery demo")
    print("=" * 60)
    print(f"API keys detected: {sorted(env['api_keys_present'].keys()) or 'NONE'}")
    print(f"Ollama local models: {env['ollama_installed_models'] or 'NONE'}")
    print()

    attempts = [
        ("Google Gemini",  GeminiBackend(model="gemini-2.5-flash")),
        ("OpenAI GPT",     OpenAIBackend(model="gpt-4o-mini")),
        ("Anthropic Claude", AnthropicBackend(model="claude-haiku-4-5-20251001")),
    ]

    # Always include at least one Ollama backend if daemon is up
    ollama_models = list_ollama_models()
    if ollama_models:
        attempts.append(("Ollama (local)", OllamaBackend(ollama_models[0])))

    results = []
    for name, b in attempts:
        print(f"-- {name} ({b.backend_id})")
        r = _try_one(name, b)
        results.append(r)
        if r["status"] == "OK":
            print(f"   [OK] {r['latency_ms']:.0f} ms · {r['tokens_out']} tokens")
            preview = r['response_preview'].replace("\n", " ")
            print(f"   > {preview[:150]}")
        elif r["status"] == "DISABLED":
            print(f"   [SKIP] {r['reason']}")
        else:
            print(f"   [ERR] {r['reason'][:120]}")
        print()

    all_backs = discover_backends()
    print(f"discover_backends() returned {len(all_backs)} enabled backends:")
    for b in all_backs:
        print(f"   · {b.backend_id} ({b.tier})")

    # Persist the report for reviewers
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    out_dir = str(get_results_dir() / "llm_api_demo")
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/api_demo_report.json", "w", encoding="utf-8") as f:
        json.dump({"env": env, "attempts": results,
                   "enabled": [b.backend_id for b in all_backs]}, f, indent=2)
    print(f"\nReport written to {out_dir}/api_demo_report.json")

    # Exit code 0 if any real LLM responded; 1 if only mock/Ollama; 2 if nothing
    ok_api = any(r["status"] == "OK" and "api:" in r["backend"] for r in results)
    ok_local = any(r["status"] == "OK" and "ollama:" in r["backend"] for r in results)
    if ok_api:
        print("\nResult: at least one managed-API backend responded successfully.")
        return 0
    if ok_local:
        print("\nResult: no managed-API key supplied — Ollama fallback succeeded.")
        return 0
    print("\nResult: no real LLM responded. Check Ollama daemon or set API keys.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
