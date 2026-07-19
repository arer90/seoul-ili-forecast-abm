"""LLM Router — ARIA Stage 6 의 multi-LLM fallback.

목적
----
Claude (primary, reasoning) → Gemma (fallback, local privacy) → 다른 옵션

사용 패턴:
    from simulation.server.rag.llm_router import LLMRouter

    router = LLMRouter(
        primary="claude-opus-4.7",
        fallback="gemma3:9b",      # Ollama
        budget="$10",
    )
    answer = router.complete("ILI 정의는?", max_tokens=500)

Status
------
**EXPERIMENTAL — NOT wired into the served pipeline.** ``complete()`` is
implemented (Anthropic / OpenAI via env keys), but none of the served
``epi.*`` MCP tools route through this class — the web Hermes layer calls the
provider directly. Prototype for future provider-routing; do NOT report it as a
production component. (2026-06-06 D5 honesty relabel — was mislabeled "SKELETON".)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 1. 환경 검증
# ══════════════════════════════════════════════════════════

def check_llm_env() -> dict:
    """LLM 백엔드 + API key 검증."""
    status = {
        "claude": {"available": False, "key": False},
        "openai": {"available": False, "key": False},
        "ollama": {"available": False, "models": []},
        "mlx_lm": {"available": False},
    }

    # Anthropic Claude
    try:
        import anthropic    # noqa: F401
        status["claude"]["available"] = True
    except ImportError:
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        status["claude"]["key"] = True

    # OpenAI
    try:
        import openai    # noqa: F401
        status["openai"]["available"] = True
    except ImportError:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        status["openai"]["key"] = True

    # Ollama (local)
    if shutil.which("ollama"):
        status["ollama"]["available"] = True
        try:
            r = subprocess.run(["ollama", "list"], capture_output=True,
                                 text=True, timeout=5)
            if r.returncode == 0:
                # parse model names (skip header)
                lines = r.stdout.strip().splitlines()[1:]
                models = [l.split()[0] for l in lines if l.strip()]
                status["ollama"]["models"] = models
        except Exception:
            pass

    # MLX-LM (Apple Silicon)
    try:
        import mlx_lm    # noqa: F401
        status["mlx_lm"]["available"] = True
    except ImportError:
        pass

    return status


def install_recommended_models():
    """Ollama 가 설치된 경우 권장 모델 install 명령 출력."""
    cmds = []
    env = check_llm_env()

    if env["ollama"]["available"]:
        existing = set(env["ollama"]["models"])
        recommended = [
            ("gemma3:4b",        "Gemma 3 4B (작음, 한국어 OK, 2.5GB)"),
            ("gemma3:9b",        "Gemma 3 9B (중간, 의료 reasoning, 5.5GB) — 권장"),
            ("medgemma:27b",     "MedGemma 27B (의료 vision 포함, 16GB) — 의료 전용"),
            ("solar:10.7b",      "Solar 10.7B (Upstage, 한국어 강함, 6GB)"),
            ("qwen2.5:14b",      "Qwen 2.5 14B (다국어, 8GB)"),
        ]
        for model, desc in recommended:
            if model.split(":")[0] not in {m.split(":")[0] for m in existing}:
                cmds.append(f"ollama pull {model}    # {desc}")

    return cmds


# ══════════════════════════════════════════════════════════
# 2. 인터페이스 (구현 예정)
# ══════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    fallback_used: bool = False


class LLMUnavailable(RuntimeError):
    """No LLM backend (API key or local Ollama model) is reachable."""


class LLMRouter:
    """Multi-LLM fallback router — Anthropic primary → OpenAI → local Ollama."""

    def __init__(self, primary: str = "claude-sonnet-4.6",
                 fallback: Optional[str] = None,
                 budget_usd: float = 10.0):
        self.primary = primary
        self.fallback = fallback
        self.budget_usd = budget_usd
        self.spent_usd = 0.0

    def complete(self, prompt: str, max_tokens: int = 500,
                 system: Optional[str] = None,
                 force_fallback: bool = False) -> LLMResponse:
        """Generate a completion: Anthropic (primary) → OpenAI → local Ollama.

        Args:
            prompt: user prompt.
            max_tokens: generation cap.
            system: optional system prompt.
            force_fallback: skip the paid providers and go straight to Ollama.

        Returns:
            LLMResponse (text + model + latency_ms + fallback_used).

        Raises:
            LLMUnavailable: no backend (no API key, no local model) is reachable.

        Side effects: network call to the selected provider / localhost:11434.
        """
        import time as _t
        t0 = _t.perf_counter()
        env = check_llm_env()
        errs: list[str] = []

        def _ms() -> float:
            return (_t.perf_counter() - t0) * 1000.0

        # 1) Anthropic (primary)
        if not force_fallback and env["claude"]["available"] and env["claude"]["key"]:
            try:
                import anthropic
                msg = anthropic.Anthropic().messages.create(
                    model=self.primary, max_tokens=max_tokens,
                    system=system or "",
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(getattr(b, "text", "") for b in msg.content)
                return LLMResponse(
                    text=text, model=self.primary,
                    tokens_out=int(getattr(getattr(msg, "usage", None), "output_tokens", 0) or 0),
                    latency_ms=_ms(), fallback_used=False)
            except Exception as e:  # noqa: BLE001
                errs.append(f"anthropic:{type(e).__name__}")

        # 2) OpenAI
        if not force_fallback and env["openai"]["available"] and env["openai"]["key"]:
            try:
                import openai
                msgs = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": prompt}]
                r = openai.OpenAI().chat.completions.create(
                    model="gpt-4o-mini", max_tokens=max_tokens, messages=msgs)
                return LLMResponse(text=r.choices[0].message.content or "",
                                   model="gpt-4o-mini", latency_ms=_ms(),
                                   fallback_used=True)
            except Exception as e:  # noqa: BLE001
                errs.append(f"openai:{type(e).__name__}")

        # 3) Local Ollama fallback (no API key required)
        if env["ollama"]["available"] and env["ollama"]["models"]:
            models = env["ollama"]["models"]
            model = self.fallback if self.fallback in models else models[0]
            try:
                import json
                import urllib.request
                body = json.dumps({
                    "model": model, "stream": False,
                    "prompt": (f"{system}\n\n" if system else "") + prompt,
                    "options": {"num_predict": max_tokens},
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    out = json.loads(resp.read())
                return LLMResponse(text=out.get("response", ""), model=model,
                                   tokens_out=int(out.get("eval_count", 0) or 0),
                                   latency_ms=_ms(), fallback_used=True)
            except Exception as e:  # noqa: BLE001
                errs.append(f"ollama:{type(e).__name__}")

        raise LLMUnavailable(
            "no LLM backend reachable — set ANTHROPIC_API_KEY / OPENAI_API_KEY, "
            "or `ollama pull <model>`. tried: " + ("; ".join(errs) or "none configured"))


# ══════════════════════════════════════════════════════════
# 3. main — 환경 검증 (수동 실행)
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  LLM Router 환경 검증")
    print("=" * 60)

    env = check_llm_env()

    print("\n[1] Claude (Anthropic)")
    print(f"  SDK: {'✓' if env['claude']['available'] else '✗'}")
    print(f"  API key: {'✓' if env['claude']['key'] else '✗ (ANTHROPIC_API_KEY 미설정)'}")

    print("\n[2] OpenAI")
    print(f"  SDK: {'✓' if env['openai']['available'] else '✗'}")
    print(f"  API key: {'✓' if env['openai']['key'] else '✗'}")

    print("\n[3] Ollama (local)")
    print(f"  binary: {'✓' if env['ollama']['available'] else '✗'}")
    print(f"  models: {len(env['ollama']['models'])} 개")
    for m in env["ollama"]["models"][:10]:
        print(f"    - {m}")

    print("\n[4] MLX-LM (Apple Silicon)")
    print(f"  SDK: {'✓' if env['mlx_lm']['available'] else '✗'}")

    print("\n[5] 권장 install")
    cmds = install_recommended_models()
    if not cmds:
        print("  ✓ 권장 모델 모두 설치됨")
    else:
        for c in cmds:
            print(f"  {c}")

    print()
    print("=" * 60)
    print("  Status: EXPERIMENTAL — complete() works but NOT wired into served epi.* tools")
    print("=" * 60)


if __name__ == "__main__":
    main()
