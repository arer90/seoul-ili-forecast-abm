"""
simulation.llm_compare
======================
Multi-backend LLM comparison harness for the ARIA consultation layer
(thesis §4.6.10, §4.17, §5.2c) and the companion journal paper.

Backend priority (auto-detected at import time):

  1. API clients, when credentials are set as environment variables
     - GOOGLE_API_KEY / GEMINI_API_KEY  → GeminiBackend
     - OPENAI_API_KEY                   → OpenAIBackend
     - ANTHROPIC_API_KEY                → AnthropicBackend
  2. Ollama HTTP daemon (localhost:11434) → OllamaBackend per installed model
  3. Local-path GGUF / safetensors weights → LocalModelBackend (llama.cpp or
     transformers)
  4. MockLLM deterministic profiles (always available) for dry-run and CI

The comparison runner (``simulation.llm_compare.runner``) invokes every
enabled backend against the 20-item bilingual golden set and produces a
``ComparisonReport`` with per-backend scores, inter-backend disagreement,
and a persistable Hermes-style audit log.
"""
from __future__ import annotations

from .backends import (
    LLMBackend,
    LLMResponse,
    GeminiBackend,
    OpenAIBackend,
    AnthropicBackend,
    OllamaBackend,
    LocalModelBackend,
    MockLLMBackend,
    discover_backends,
)
from .comparison import (
    compare_backends,
    cohen_kappa,
    fleiss_kappa_binary,
    pairwise_wilcoxon_holm,
)
from .golden_set import load_golden_set, GoldenItem
from .judge import ScoredResponse, score_response, SEVEN_PILLARS
from .runner import (
    ComparisonReport,
    run_comparison,
)

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "GeminiBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "OllamaBackend",
    "LocalModelBackend",
    "MockLLMBackend",
    "discover_backends",
    "load_golden_set",
    "GoldenItem",
    "ScoredResponse",
    "score_response",
    "SEVEN_PILLARS",
    "ComparisonReport",
    "run_comparison",
    "compare_backends",
    "pairwise_wilcoxon_holm",
    "fleiss_kappa_binary",
    "cohen_kappa",
]
