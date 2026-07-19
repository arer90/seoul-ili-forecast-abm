"""
simulation.llm_compare.backends
===============================
Unified LLM backend interface for the comparison harness.

Contract
--------
Every concrete backend exposes ``.is_available() -> bool`` and
``.generate(prompt, *, system=None, temperature=0.2, max_tokens=512) ->
LLMResponse``. The factory ``discover_backends()`` inspects the
environment, returns the list of backends that can actually respond,
and tags each response with its ``backend_id`` for downstream scoring.

No backend raises on network / SDK failure; failures are recorded as
``LLMResponse(text="", error="<reason>")`` so the comparison runner
can include them in the report as "observed unavailable".
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "GeminiBackend",
    "OpenAIBackend",
    "OpenAICompatBackend",
    "AnthropicBackend",
    "OllamaBackend",
    "LocalModelBackend",
    "MockLLMBackend",
    "discover_backends",
]


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """Uniform response record. ``error`` is non-empty iff generation failed."""
    backend_id: str            # e.g. "ollama:qwen2.5:3b" or "api:gemini-1.5-flash"
    model: str                 # vendor-specific model identifier
    text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # Keep raw compact in persisted JSON
        d.pop("raw", None)
        return d


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class LLMBackend:
    """Abstract base. Concrete subclasses override ``generate`` and
    set ``self.backend_id`` / ``self.model``."""

    backend_id: str = "abstract"
    model: str = ""
    provider: str = ""
    tier: str = "unknown"   # "api" | "ollama" | "local" | "mock"

    def is_available(self) -> bool:  # pragma: no cover - subclasses override
        return False

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:  # pragma: no cover - subclasses override
        raise NotImplementedError


# ---------------------------------------------------------------------------
# API-tier backends (Gemini / OpenAI / Anthropic)
# ---------------------------------------------------------------------------
class GeminiBackend(LLMBackend):
    """Google Gemini API (prefers GOOGLE_API_KEY, falls back to GEMINI_API_KEY)."""

    tier = "api"
    provider = "google"

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.backend_id = f"api:gemini:{model}"
        self._key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )

    def is_available(self) -> bool:
        return bool(self._key)

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        if not self.is_available():
            return LLMResponse(self.backend_id, self.model, "", 0.0,
                               error="GOOGLE_API_KEY / GEMINI_API_KEY not set")
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            t0 = time.time()
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                f"{self._endpoint}?key={self._key}",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            dt = (time.time() - t0) * 1000.0
            cand = payload.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            usage = payload.get("usageMetadata", {})
            return LLMResponse(
                self.backend_id, self.model, text, dt,
                prompt_tokens=int(usage.get("promptTokenCount", 0)),
                completion_tokens=int(usage.get("candidatesTokenCount", 0)),
                raw=payload,
            )
        except urllib.error.HTTPError as e:
            return LLMResponse(self.backend_id, self.model, "", 0.0,
                               error=f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}")
        except Exception as e:  # noqa: BLE001
            return LLMResponse(self.backend_id, self.model, "", 0.0, error=str(e))


class OpenAIBackend(LLMBackend):
    """OpenAI Responses / Chat-Completions API."""

    tier = "api"
    provider = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.backend_id = f"api:openai:{model}"
        self._key = os.environ.get("OPENAI_API_KEY")
        self._endpoint = "https://api.openai.com/v1/chat/completions"

    def is_available(self) -> bool:
        return bool(self._key)

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        if not self.is_available():
            return LLMResponse(self.backend_id, self.model, "", 0.0,
                               error="OPENAI_API_KEY not set")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        try:
            t0 = time.time()
            req = urllib.request.Request(
                self._endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            dt = (time.time() - t0) * 1000.0
            text = payload["choices"][0]["message"]["content"].strip()
            usage = payload.get("usage", {})
            return LLMResponse(
                self.backend_id, self.model, text, dt,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                raw=payload,
            )
        except Exception as e:  # noqa: BLE001
            return LLMResponse(self.backend_id, self.model, "", 0.0, error=str(e))


class OpenAICompatBackend(LLMBackend):
    """Any OpenAI-compatible ``/v1/chat/completions`` server — the ONE adapter
    that covers every local serving ENGINE (they all speak this API):
    **vLLM · MLX-LM · SGLang · LM Studio · llama.cpp-server · LocalAI · LiteLLM**.

    Used IN PLACE OF :class:`OllamaBackend` when you want explicit quantization /
    throughput control (the precise-comparison path). The serving engine is the
    transport, NOT the comparison target — the *model* is what's compared.

    Args:
        model: served model id (as the server lists it, e.g. ``Qwen/Qwen2.5-7B-Instruct``).
        base_url: server base INCLUDING the version segment, e.g.
            ``http://localhost:8000/v1`` (vLLM) / ``http://localhost:8080/v1`` (MLX-LM).
        api_key: most local servers ignore it; defaults to env
            ``OPENAI_COMPAT_API_KEY`` or the literal ``"EMPTY"``.
        label: backend_id override (else ``oai:<model>@<host>``).

    Side effects: network only. Never raises — failures → ``LLMResponse(error=…)``.
    """

    tier = "openai_compat"

    def __init__(self, model: str, base_url: str, *, api_key: Optional[str] = None,
                 label: Optional[str] = None, provider: str = "openai_compat"):
        self.model = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self._endpoint = f"{self.base_url}/chat/completions"
        self._models_url = f"{self.base_url}/models"
        self._key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY") or "EMPTY"
        self.backend_id = label or f"oai:{model}@{self.base_url}"

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(
                self._models_url, headers={"Authorization": f"Bearer {self._key}"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                return getattr(resp, "status", 200) == 200
        except Exception:
            return False

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages,
                "temperature": float(temperature), "max_tokens": int(max_tokens)}
        try:
            t0 = time.time()
            req = urllib.request.Request(
                self._endpoint, data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {self._key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            dt = (time.time() - t0) * 1000.0
            text = payload["choices"][0]["message"]["content"].strip()
            usage = payload.get("usage", {})
            return LLMResponse(
                self.backend_id, self.model, text, dt,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                raw=payload)
        except Exception as e:  # noqa: BLE001
            return LLMResponse(self.backend_id, self.model, "", 0.0, error=str(e))


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API (Claude)."""

    tier = "api"
    provider = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self.backend_id = f"api:anthropic:{model}"
        self._key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        self._endpoint = "https://api.anthropic.com/v1/messages"

    def is_available(self) -> bool:
        return bool(self._key)

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        if not self.is_available():
            return LLMResponse(self.backend_id, self.model, "", 0.0,
                               error="ANTHROPIC_API_KEY / CLAUDE_API_KEY not set")
        body = {
            "model": self.model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        try:
            t0 = time.time()
            req = urllib.request.Request(
                self._endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "x-api-key": self._key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            dt = (time.time() - t0) * 1000.0
            blocks = payload.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            usage = payload.get("usage", {})
            return LLMResponse(
                self.backend_id, self.model, text, dt,
                prompt_tokens=int(usage.get("input_tokens", 0)),
                completion_tokens=int(usage.get("output_tokens", 0)),
                raw=payload,
            )
        except Exception as e:  # noqa: BLE001
            return LLMResponse(self.backend_id, self.model, "", 0.0, error=str(e))


# ---------------------------------------------------------------------------
# CLI-tier backends (Claude Code / Codex / Gemini CLI)
# ---------------------------------------------------------------------------
# Sprint 2026-05-06 (사용자 명시): API key 없이 사용자 OAuth login session
# 보유한 CLI 도구를 subprocess 호출로 활용. claude/codex/gemini CLI 모두
# non-interactive prompt 모드 지원.
#
# Pros:
#   - API key 미보유 환경에서도 작동 (CLI 가 OAuth/login session 보유)
#   - 사용자 quota / billing 그대로 적용 (CLI 의 plan 따라)
# Cons:
#   - subprocess overhead (~1-3s per call, network 외 추가)
#   - timeout 의존 (Python `subprocess.run(timeout=...)` 사용; macOS `timeout`
#     명령 없음 — ENGINEERING_PRINCIPLES.md macOS 한계 명시)
#   - CLI 의 system_prompt / temperature / max_tokens flag 가 CLI 별 다름;
#     본 wrapper 는 prompt 만 pass-through, 그 외는 best-effort flag
class _CliBackendBase(LLMBackend):
    """Common subprocess wrapper for CLI-tier backends."""

    tier = "cli"
    cli_cmd: tuple[str, ...] = ()
    cli_timeout: int = 180  # seconds; CLI 응답 typical 5-60s

    def is_available(self) -> bool:
        import shutil
        return shutil.which(self.cli_cmd[0]) is not None if self.cli_cmd else False

    def _build_argv(self, prompt: str) -> list[str]:  # subclass override
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:
        import subprocess
        import time as _time
        if not self.is_available():
            return LLMResponse(
                self.backend_id, self.model, "", 0.0,
                error=f"{self.cli_cmd[0]} CLI not in PATH",
            )
        full_prompt = prompt if system is None else f"{system}\n\n{prompt}"
        argv = self._build_argv(full_prompt)
        t0 = _time.perf_counter()
        try:
            # Codex fail fix (sprint 2026-05-06): some CLIs (`codex exec`)
            # also wait on stdin for additional context. Default
            # `stdin=None` (inherit) makes them hang in subprocess
            # invocation. Force `stdin=DEVNULL` so the CLI knows there is
            # no more input. Verified failure mode for codex 0.128.0:
            # rc=1 with stderr "Reading additional input from stdin...".
            # Scrub Claude-Code-injected env (2026-06-07): an integrated/nested
            # terminal inherits ANTHROPIC_BASE_URL etc. that misroute the `claude`
            # CLI to a session endpoint → 401. Removing them makes `claude` use its
            # own OAuth login + default endpoint. Harmless for codex/gemini (they
            # ignore these). Does NOT fix a logged-OUT CLI — that needs `claude login`.
            _scrub = {"ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                      "CLAUDECODE", "CLAUDE_AGENT_SDK_VERSION"}
            child_env = {k: v for k, v in os.environ.items()
                         if k not in _scrub and not k.startswith("CLAUDE_CODE_")}
            result = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=self.cli_timeout,
                stdin=subprocess.DEVNULL,
                env=child_env,
            )
        except subprocess.TimeoutExpired:
            return LLMResponse(
                self.backend_id, self.model, "",
                float(self.cli_timeout) * 1000.0,
                error=f"{self.cli_cmd[0]} CLI timeout {self.cli_timeout}s",
            )
        except Exception as e:  # noqa: BLE001
            return LLMResponse(self.backend_id, self.model, "", 0.0, error=str(e))
        latency_ms = (_time.perf_counter() - t0) * 1000.0
        # Auth-failure detection (CRITICAL — must NOT be scored as a wrong answer).
        # The CLI may return rc=0 with a "Not logged in" banner (e.g. when run
        # inside a nested agent session that can't reach the OAuth keychain), or
        # rc!=0 with a 401. Either way it's an UNAVAILABLE backend, not a model
        # result → flag as error so the runner excludes it from the ranking.
        combined = ((result.stdout or "") + " " + (result.stderr or "")).lower()
        _AUTH_FAILS = ("not logged in", "please run /login", "invalid authentication",
                       "failed to authenticate", "unauthorized", "401 invalid",
                       "please run `claude login`", "run /login")
        if any(s in combined for s in _AUTH_FAILS):
            return LLMResponse(
                self.backend_id, self.model, "", latency_ms,
                error=(f"{self.cli_cmd[0]} CLI auth failed (not logged in / 401) — "
                       f"run from a normal terminal, not a nested agent session"),
            )
        if result.returncode != 0:
            return LLMResponse(
                self.backend_id, self.model, "", latency_ms,
                error=(f"{self.cli_cmd[0]} CLI rc={result.returncode}: "
                       f"{(result.stderr or '')[:300]}"),
            )
        return LLMResponse(
            self.backend_id, self.model,
            (result.stdout or "").strip(),
            latency_ms,
            raw={"stderr_tail": (result.stderr or "")[-200:]},
        )


class ClaudeCliBackend(_CliBackendBase):
    """Claude Code CLI (`claude -p "<prompt>"`).

    Uses the user's existing `claude` OAuth login session. Model is the
    user's default per Claude Code config (see `claude --help`).
    """

    provider = "anthropic"
    cli_cmd = ("claude",)

    def __init__(self, model: str = "claude-default"):
        self.model = model
        self.backend_id = f"cli:claude:{model}"

    def _build_argv(self, prompt: str) -> list[str]:
        return ["claude", "-p", prompt]


class CodexCliBackend(_CliBackendBase):
    """OpenAI Codex CLI (`codex exec "<prompt>"`).

    Uses the user's existing `codex login` session.
    """

    provider = "openai"
    cli_cmd = ("codex",)

    def __init__(self, model: str = "codex-default"):
        self.model = model
        self.backend_id = f"cli:codex:{model}"

    def _build_argv(self, prompt: str) -> list[str]:
        # Sprint 2026-05-06 (#post-S3P2en, codex stuck fix): codex 의 default
        # reasoning_effort 가 xhigh — 단일 call 3-5분 소요. 25 items batch 평가
        # 에는 부적합 → low (sub-30s) 으로 명시. quality vs latency trade-off:
        # - low/minimal: 응답 빠름, surface-level reasoning (적절 for benchmark)
        # - xhigh: deep reasoning (적절 for production single query)
        # `-c key=value` 으로 override (TOML literal).
        return ["codex", "exec", "-c", "reasoning_effort=low", prompt]


class GeminiCliBackend(_CliBackendBase):
    """Google Gemini CLI (`gemini -p "<prompt>"`).

    Uses the user's existing Gemini CLI auth (gcloud / API key in CLI
    config — separate from Python env vars).
    """

    provider = "google"
    cli_cmd = ("gemini",)

    def __init__(self, model: str = "gemini-default"):
        self.model = model
        self.backend_id = f"cli:gemini:{model}"

    def _build_argv(self, prompt: str) -> list[str]:
        return ["gemini", "-p", prompt]


def _default_cli_models() -> list[LLMBackend]:
    """Default CLI-tier backends to probe in :func:`discover_backends`."""
    return [
        ClaudeCliBackend(),
        CodexCliBackend(),
        GeminiCliBackend(),
    ]


# ---------------------------------------------------------------------------
# Ollama HTTP backend
# ---------------------------------------------------------------------------
class OllamaBackend(LLMBackend):
    """Ollama HTTP API (defaults to localhost:11434)."""

    tier = "ollama"
    provider = "ollama"

    def __init__(self, model: str, base_url: Optional[str] = None):
        self.model = model
        self.backend_id = f"ollama:{model}"
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434"
        ).rstrip("/")

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=3
            ) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            names = {m.get("model", "") for m in tags.get("models", [])}
            return self.model in names or any(n.startswith(self.model) for n in names)
        except Exception:
            return False

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if system:
            body["system"] = system
        try:
            t0 = time.time()
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            dt = (time.time() - t0) * 1000.0
            text = (payload.get("response") or "").strip()
            return LLMResponse(
                self.backend_id, self.model, text, dt,
                prompt_tokens=int(payload.get("prompt_eval_count", 0)),
                completion_tokens=int(payload.get("eval_count", 0)),
                raw={k: v for k, v in payload.items() if k != "response"},
            )
        except Exception as e:  # noqa: BLE001
            return LLMResponse(self.backend_id, self.model, "", 0.0, error=str(e))


def list_ollama_models(base_url: Optional[str] = None) -> list[str]:
    """Return the list of locally pulled Ollama models (tag-qualified names)."""
    base = (base_url or os.environ.get("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        return sorted({m.get("model", "") for m in tags.get("models", []) if m.get("model")})
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Local-path GGUF / safetensors loader (best-effort)
# ---------------------------------------------------------------------------
class LocalModelBackend(LLMBackend):
    """Backend that loads a model from a filesystem path.

    Two concrete implementations are auto-detected:

    * ``llama_cpp`` for GGUF paths
    * ``transformers`` for safetensors / HF-format directories

    If neither SDK is installed the backend is marked unavailable and the
    comparison harness simply skips it.
    """

    tier = "local"
    provider = "local_path"

    def __init__(self, model_path: str, *, display_name: Optional[str] = None):
        self.model = display_name or os.path.basename(model_path.rstrip("/")) or model_path
        self.backend_id = f"local:{self.model}"
        self.model_path = str(model_path)
        self._llama = None       # llama_cpp.Llama
        self._tokenizer = None
        self._hf_pipeline = None

    def is_available(self) -> bool:
        p = Path(self.model_path)
        if not p.exists():
            return False
        if p.is_file() and p.suffix == ".gguf":
            try:
                import llama_cpp  # noqa: F401
                return True
            except ImportError:
                return False
        if p.is_dir():
            try:
                import transformers  # noqa: F401
                import torch  # noqa: F401
                return any((p / f).exists() for f in ("config.json",))
            except ImportError:
                return False
        return False

    def _load(self):
        p = Path(self.model_path)
        if p.is_file() and p.suffix == ".gguf":
            from llama_cpp import Llama
            self._llama = Llama(model_path=self.model_path, n_ctx=4096, verbose=False)
        else:
            from transformers import pipeline
            self._hf_pipeline = pipeline(
                "text-generation", model=self.model_path, device_map="auto"
            )

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        if not self.is_available():
            return LLMResponse(self.backend_id, self.model, "", 0.0,
                               error=f"local model not loadable: {self.model_path}")
        try:
            if self._llama is None and self._hf_pipeline is None:
                self._load()
            t0 = time.time()
            if self._llama is not None:
                sys_prefix = f"<|system|>\n{system}\n" if system else ""
                full_prompt = f"{sys_prefix}<|user|>\n{prompt}\n<|assistant|>\n"
                res = self._llama(full_prompt, max_tokens=max_tokens,
                                  temperature=float(temperature))
                text = (res.get("choices", [{}])[0].get("text", "") or "").strip()
                raw = res
            else:
                res = self._hf_pipeline(
                    prompt, max_new_tokens=max_tokens,
                    temperature=float(temperature), do_sample=temperature > 0,
                )
                text = (res[0].get("generated_text") or "").removeprefix(prompt).strip()
                raw = {"_pipeline": "transformers"}
            dt = (time.time() - t0) * 1000.0
            return LLMResponse(self.backend_id, self.model, text, dt, raw=raw)
        except Exception as e:  # noqa: BLE001
            return LLMResponse(self.backend_id, self.model, "", 0.0, error=str(e))


# ---------------------------------------------------------------------------
# MockLLM — deterministic profiles, always available
# ---------------------------------------------------------------------------
class MockLLMBackend(LLMBackend):
    """Rule-based deterministic LLM for CI, dry-run, and baseline control.

    Three shipped profiles:

    * ``cautious``   — conservative, heavy on disclaimers, long lead time
    * ``aggressive`` — action-first, short, low disclaimer density
    * ``balanced``   — middle of the road, uses 7-pillar-friendly structure

    The profiles produce differentiated rubric scores so the judge can be
    validated end-to-end without any network traffic. They are NOT a claim
    about any real LLM family.
    """

    tier = "mock"
    provider = "mock"

    _TEMPLATES = {
        "cautious": (
            "Recommendation: hold escalation pending further surveillance confirmation. "
            "Uncertainty is high; the 95 % prediction interval spans a wide range and "
            "the post-COVID regime shift raises structural concern. Suggested action: "
            "re-check the KDCA weekly advisory in seven days, consult an epidemiologist "
            "before issuing any public-health alert, and document the decision in the "
            "Hermes audit log. Reference: §4.9 conformal coverage limits, §4.15 F10 guard."
        ),
        "aggressive": (
            "Alert now. Peak likely within two weeks. Pre-position antivirals and open "
            "surge capacity in top-3 districts by predicted rate. Lead time is sufficient "
            "for a single decisive action; do not wait for a second weekly confirmation."
        ),
        "balanced": (
            "Recommendation: issue a district-level watch (not yet a public alert). "
            "Forecast confidence is moderate; empirical PICP@95 is 0.865 (§4.9). "
            "Concrete steps: (1) cross-check q70 and KDCA thresholds on the latest weekly "
            "value (§4.13), (2) run a commuter-coupled SEIR what-if at 20 % NPI to bound "
            "the downside, (3) schedule a reassessment at t + 7 days with the Hermes "
            "audit log attached. Epidemiologist-in-the-loop per §7.5 governance."
        ),
    }

    def __init__(self, profile: str = "balanced"):
        profile = profile.lower()
        if profile not in self._TEMPLATES:
            raise ValueError(f"unknown mock profile {profile!r}")
        self.model = f"mock-{profile}"
        self.backend_id = f"mock:{profile}"
        self._profile = profile

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        # Deterministic: echo a prompt-specific suffix so scoring can detect the item.
        suffix = f"\n\n[mock:{self._profile}] prompt length = {len(prompt)} chars"
        text = self._TEMPLATES[self._profile] + suffix
        return LLMResponse(
            self.backend_id, self.model, text[:max_tokens * 4],
            latency_ms=1.0, prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            raw={"profile": self._profile},
        )


# ---------------------------------------------------------------------------
# Factory / auto-discover
# ---------------------------------------------------------------------------
def _default_api_models() -> list[LLMBackend]:
    """API-tier backends with recommended per-provider models."""
    return [
        GeminiBackend(model="gemini-2.5-flash"),
        OpenAIBackend(model="gpt-4o-mini"),
        AnthropicBackend(model="claude-haiku-4-5-20251001"),
    ]


def _default_ollama_models() -> list[str]:
    """Preferred Ollama model list for the thesis comparison.

    Ordering expresses priority: the first K in the locally-installed set
    are kept for the experiment. We cap at 5 to avoid blowing up latency
    on the 20-item golden set.
    """
    return [
        "gemma3:1b",               # Google Gemini open-weight sibling
        "qwen2.5:3b",              # Alibaba, strong multilingual
        "deepseek-r1:1.5b",        # DeepSeek reasoning distillation
        "phi3.5:3.8b",             # Microsoft
        "llama3.2:3b",             # Meta
    ]


class AriaTorchLMBackend(LLMBackend):
    """The project's OWN from-scratch PyTorch language model, as a comparison backend.

    A small modern decoder-only Transformer (RoPE / grouped-query attention / SwiGLU /
    RMSNorm) trained from scratch on the epidemiology + Korean public-health corpus
    (``simulation/scripts/train_aria_modern_lm.py``). Exposed here so it can be compared
    side-by-side with the hosted (Claude / Gemini / OpenAI) and Ollama backends. It is a
    deliberately small, WEAK demonstration model - the grounding layer remains the
    arbiter of correctness - and is flag-gated (``MPH_ARIA_TORCH_LM``), so the default
    backend pool is unchanged.

    Performance: autoregressive byte-level generation on MPS/CPU (~seconds per answer).
    Side effects: lazily loads the checkpoint into memory on first ``generate``.
    """
    tier = "local"
    provider = "local"

    def __init__(self, ckpt: Optional[str] = None, device: Optional[str] = None):
        self.ckpt = Path(ckpt) if ckpt else (
            Path(__file__).resolve().parents[1] / "results" / "aria_modern_lm" / "aria_modern_lm.pt")
        self.model = "aria-torch-lm"
        self.backend_id = "local:aria-torch-lm"
        self._dev = device
        self._m = None
        self._block = 256
        self._vocab = None      # word/digit/Korean tokenizer (new checkpoints); None = byte-level (old)
        self._inv = None

    def is_available(self) -> bool:
        return self.ckpt.exists()

    def _ensure_loaded(self) -> None:
        if self._m is not None:
            return
        import torch
        from simulation.scripts.train_aria_modern_lm import ModernGPT
        self._dev = self._dev or ("mps" if torch.backends.mps.is_available() else "cpu")
        ckpt = torch.load(self.ckpt, map_location=self._dev)
        m = ModernGPT(**ckpt["config"]).to(self._dev)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        self._m = m
        self._block = ckpt["config"]["block_size"]
        self._vocab = ckpt.get("tokenizer")
        self._inv = {v: k for k, v in self._vocab.items()} if self._vocab else None

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        import torch
        t0 = time.time()
        try:
            self._ensure_loaded()
            p = ((system + "\n") if system else "") + (prompt or "")
            if self._vocab is not None:                          # word/digit/Korean tokenizer path
                import re as _re
                from simulation.scripts.train_aria_modern_lm import (
                    encode as _enc, decode as _dec, make_number_constraint as _mkc)
                ids = _enc(p, self._vocab)[-self._block:]
                idx = torch.tensor([ids], dtype=torch.long, device=self._dev)
                nl = self._vocab.get("\n")
                stop = [nl, nl] if nl is not None else None
                nums = set(_re.findall(r"-?\d+\.?\d*", p))        # numbers present in the input
                constrain = _mkc(self._vocab, self._inv, nums) if nums else None
                # low temp keeps the model in the grounding mode (raising it drifts to PubMed mode);
                # mild repetition penalty breaks literal loops; constraint bans fabricated numbers.
                out = self._m.generate(idx, int(max_tokens), temp=max(0.2, temperature),
                                       repetition_penalty=1.15, stop=stop, constrain=constrain)[0].tolist()
                gen = _dec(out[len(ids):], self._inv)
            else:                                                # byte-level fallback (old checkpoints)
                pb = list(p.encode("utf-8"))[-self._block:]
                idx = torch.tensor([pb], dtype=torch.long, device=self._dev)
                out = self._m.generate(idx, int(max_tokens), temp=max(0.1, temperature),
                                       stop=b"\n\n")[0].tolist()
                gen = bytes(out[len(pb):]).decode("utf-8", errors="replace")
            return LLMResponse(self.backend_id, self.model, gen, (time.time() - t0) * 1000.0)
        except Exception as e:  # pragma: no cover - defensive
            return LLMResponse(self.backend_id, self.model, "", (time.time() - t0) * 1000.0, error=str(e))


def discover_backends(
    *,
    include_api: bool = True,
    include_cli: bool = True,
    include_ollama: bool = True,
    include_local_paths: Iterable[str] | None = None,
    include_openai_compat: Iterable[dict] | None = None,
    include_mock: bool = True,
    include_aria_torch_lm: Optional[bool] = None,
    max_ollama: int = 5,
    ollama_base_url: Optional[str] = None,
) -> list[LLMBackend]:
    """Return the list of backends that can actually respond right now.

    Selection priority:

    1. API tier (Gemini / OpenAI / Anthropic) — any backend whose key env
       variable is set is kept unconditionally.
    2. Ollama tier — intersect ``_default_ollama_models`` with
       ``list_ollama_models``; keep the first ``max_ollama``.
    3. Local paths — one ``LocalModelBackend`` per supplied path that
       returns ``is_available() == True``.
    4. Mock profiles — three deterministic profiles added only if the
       enabled-set above is empty OR ``include_mock`` is True. Always kept
       as a control group so scores are interpretable even when every
       real backend fails.
    """
    out: list[LLMBackend] = []

    if include_api:
        for b in _default_api_models():
            if b.is_available():
                out.append(b)

    if include_cli:
        for b in _default_cli_models():
            if b.is_available():
                out.append(b)

    if include_ollama:
        installed = set(list_ollama_models(base_url=ollama_base_url))
        preferred = _default_ollama_models()
        picked = 0
        for m in preferred:
            if m in installed and picked < max_ollama:
                out.append(OllamaBackend(m, base_url=ollama_base_url))
                picked += 1
        # also pick any other installed model up to the cap
        for m in sorted(installed):
            if picked >= max_ollama:
                break
            if m not in preferred:
                out.append(OllamaBackend(m, base_url=ollama_base_url))
                picked += 1

    if include_local_paths:
        for p in include_local_paths:
            b = LocalModelBackend(p)
            if b.is_available():
                out.append(b)

    # OpenAI-compatible servers (vLLM / MLX-LM / SGLang / LM Studio / LiteLLM …)
    # — used in place of Ollama for quantization/throughput control. Each spec:
    # ``{"model": ..., "base_url": ".../v1", "label"?: ..., "api_key"?: ...}``.
    if include_openai_compat:
        for spec in include_openai_compat:
            b = OpenAICompatBackend(
                spec["model"], spec["base_url"],
                api_key=spec.get("api_key"), label=spec.get("label"))
            if b.is_available():
                out.append(b)

    # Project's own from-scratch torch LM (modern Transformer) — additive + flag-gated:
    # joins the comparison pool only when MPH_ARIA_TORCH_LM is set (default pool unchanged).
    if include_aria_torch_lm is None:
        include_aria_torch_lm = os.environ.get("MPH_ARIA_TORCH_LM", "").strip().lower() in {"1", "true", "yes", "on"}
    if include_aria_torch_lm:
        b = AriaTorchLMBackend()
        if b.is_available():
            out.append(b)
        else:
            # Fail-loud (G-237): a user who explicitly asked for the torch-LM
            # comparison backend must be told WHY it did not join the pool,
            # rather than silently getting a smaller pool.
            log.warning(
                "MPH_ARIA_TORCH_LM is set but the from-scratch LM checkpoint is "
                "missing at %s — the torch-LM comparison backend is UNAVAILABLE. "
                "Restore it from a champion-run archive or run "
                "simulation/scripts/train_aria_modern_lm.py.", b.ckpt)

    if include_mock:
        for profile in ("cautious", "balanced", "aggressive"):
            out.append(MockLLMBackend(profile))

    # dedup by backend_id
    seen: set[str] = set()
    deduped: list[LLMBackend] = []
    for b in out:
        if b.backend_id in seen:
            continue
        seen.add(b.backend_id)
        deduped.append(b)
    return deduped


def env_status() -> dict:
    """Snapshot of detected credentials, Ollama models, and local paths.

    Used by the runner to prefix its audit log with the environment it
    actually ran in.
    """
    keys = {
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "CLAUDE_API_KEY": bool(os.environ.get("CLAUDE_API_KEY")),
    }
    return {
        "api_keys_present": {k: v for k, v in keys.items() if v},
        "api_keys_missing": [k for k, v in keys.items() if not v],
        "ollama_installed_models": list_ollama_models(),
    }
