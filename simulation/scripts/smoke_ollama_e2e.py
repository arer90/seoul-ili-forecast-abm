"""simulation/scripts/smoke_ollama_e2e.py
=============================================================================
OLLAMA + MCP 10-TOOL END-TO-END SMOKE — zero API cost, offline-safe.

ARIA 의 Hermes orchestration (provider → MCP → provider → user) 을
**API 키 없이** 로컬 Ollama 로 검증한다.  Vercel 배포 전에 돈을 쓰기
전 마지막 방어선.  Qwen 2.5 / Llama 3.1 Instruct 계열의 OpenAI-compatible
tool-use 인터페이스를 사용한다.

--- 검증 체인 --------------------------------------------------------------
    ① GET  http://localhost:11434/api/tags      (모델 존재 확인)
    ② POST http://localhost:11434/api/chat      (시스템 프롬프트 + 10 tools)
    ③ tool_calls 돌려받음 → EpiMCPServer.call_tool() 로 로컬 실행
    ④ result 를 history 에 넣고 다시 /api/chat (답변 합성)
    ⑤ 최종 텍스트 sanity check (숫자 인용 / 툴 인용 존재 여부)

--- 사용법 ----------------------------------------------------------------
    # 1) Ollama 실행 중 + qwen2.5:14b-instruct-q5_K_M 설치 확인
    ollama list
    ollama pull qwen2.5:14b-instruct-q5_K_M     # 이미 있으면 skip

    # 2) 스모크
    .venv\\Scripts\\python.exe -m simulation.scripts.smoke_ollama_e2e
    .venv\\Scripts\\python.exe -m simulation.scripts.smoke_ollama_e2e \\
        --model qwen2.5:14b-instruct-q5_K_M --max-hops 4 --timeout 180

--- 출력 ------------------------------------------------------------------
    simulation/results/smoke_ollama_e2e.json
        {
          "model": "...",
          "n_tool_calls": 3,
          "tools_invoked": ["epi.forecast", "epi.rt_estimate", ...],
          "final_answer": "...",
          "answer_cites_numbers": true,
          "elapsed_sec": 42.1,
          "exit_code": 0
        }
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from simulation.server.mcp_epi import EpiMCPServer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "simulation" / "results" / "smoke_ollama_e2e.json"

# ---------------------------------------------------------------------------
# Defaults (match web/lib/providers/ollama.ts DEFAULT_MODELS[1] — the one
# we recommend for presentation laptops; 16 GB RAM friendly).
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5:14b-instruct-q5_K_M"
DEFAULT_BASE = "http://localhost:11434"
DEFAULT_MAX_HOPS = 4
DEFAULT_TIMEOUT = 180  # seconds per HTTP call

# The prompt is deliberately concrete so a good model MUST call MCP tools
# (it can't answer from parametric memory alone).
#
# Smaller Ollama models (Qwen 2.5 7b, Llama 3.2 3b) tend to emit an empty
# message after receiving the tool_result — so the instruction MUST
# explicitly demand a written synthesis, not just "be helpful".
SYSTEM_PROMPT = (
    "You are a Seoul flu epidemiology consultant with access to 10 epi.* "
    "MCP tools.\n"
    "MANDATORY workflow (violation = failure):\n"
    "  Step 1. FIRST call at least 2 tools.  Do NOT write any Korean prose "
    "until tool_result messages are in the conversation.  Answering from "
    "memory is forbidden — you have no knowledge of current Seoul data.\n"
    "  Step 2. AFTER tool_result messages arrive, you MUST write a Korean "
    "prose summary in your next assistant turn.  An empty reply is a "
    "failure.  Quote the tool name in [square brackets] next to every "
    "number you cite.\n"
    "  Step 3. Never fabricate — only cite values returned by tools.\n"
    "Answer language: Korean (한국어로 답변)."
)
USER_QUESTION = (
    "현재 서울시 전체 인플루엔자 상황을 3문장으로 요약하고, 가장 최근 주 "
    "Rt 값과 SHAP 상위 3개 feature 를 알려줘."
)


# =========================================================================
# HTTP helpers (stdlib only — no new deps)
# =========================================================================

def _http_json(url: str, body: Optional[Dict[str, Any]] = None,
                *, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"content-type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_stream_lines(url: str, body: Dict[str, Any],
                        *, timeout: int = DEFAULT_TIMEOUT):
    """Yield ndjson lines from Ollama /api/chat with stream=true."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    buf = b""
    for chunk in resp:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                try:
                    yield json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue


# =========================================================================
# MCP bridge — map Ollama tool name (dots replaced) back to epi.* schema
# =========================================================================

class McpBridge:
    """Wrap EpiMCPServer with name mangling for Ollama (dots→underscores)."""

    def __init__(self) -> None:
        self.server = EpiMCPServer()
        self.tools_raw = self.server.list_tools()
        # Ollama function name can't contain dots, so we mangle.
        self.ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"].replace(".", "_"),
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {}),
                },
            }
            for t in self.tools_raw
        ]

    def call(self, mangled_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        real_name = mangled_name.replace("_", ".", 1)  # only first underscore
        # Sanity: if the original had more dots, restore them by lookup.
        for t in self.tools_raw:
            if t["name"].replace(".", "_") == mangled_name:
                real_name = t["name"]
                break
        try:
            res = self.server.call_tool(real_name, arguments)
            return {
                "tool": real_name,
                "is_error": bool(res.is_error),
                "content": res.content,
            }
        except Exception as e:   # noqa: BLE001
            return {"tool": real_name, "is_error": True, "content": {"error": str(e)}}


# =========================================================================
# Ollama probe
# =========================================================================

def probe_ollama(base: str, model: str) -> Tuple[bool, str]:
    try:
        tags = _http_json(f"{base}/api/tags", timeout=10)
    except urllib.error.URLError as e:
        return False, f"ollama unreachable at {base}: {e}"
    names = [m.get("name", "") for m in tags.get("models", [])]
    if not any(n.startswith(model.split(":")[0]) for n in names):
        return False, f"model '{model}' not installed; try: ollama pull {model}"
    if model not in names:
        # Allow tag-prefix matches (e.g. qwen2.5:14b vs qwen2.5:14b-instruct-q5_K_M)
        nearest = next((n for n in names if n.startswith(model.split(":")[0])), None)
        return True, f"requested '{model}' not exact; nearest installed = '{nearest}'"
    return True, f"ok ({len(names)} models installed)"


# =========================================================================
# E2E orchestrator
# =========================================================================

def run_e2e(model: str, base: str, max_hops: int, timeout: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    bridge = McpBridge()
    log.info(f"MCP tools exposed = {len(bridge.ollama_tools)}")

    history: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_QUESTION},
    ]

    tools_invoked: List[str] = []
    tool_results: List[Dict[str, Any]] = []
    assistant_texts: List[str] = []

    for hop in range(1, max_hops + 1):
        log.info(f"── hop {hop} ───────────────────────────────────────")
        body = {
            "model": model,
            "messages": history,
            "stream": True,
            "options": {"temperature": 0.0, "num_predict": 1024},
            "tools": bridge.ollama_tools,
        }

        assistant_text = ""
        pending_calls: List[Dict[str, Any]] = []

        try:
            for chunk in _http_stream_lines(f"{base}/api/chat", body, timeout=timeout):
                msg = chunk.get("message", {}) or {}
                delta = msg.get("content") or ""
                if delta:
                    assistant_text += delta
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc.get("function", {}) or {}
                    pending_calls.append({
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", {}) if isinstance(fn.get("arguments"), dict)
                                     else _safe_json(fn.get("arguments", "{}")),
                    })
                if chunk.get("done"):
                    break
        except urllib.error.URLError as e:
            return {
                "model": model, "exit_code": 2,
                "error": f"ollama /api/chat failed: {e}",
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }

        if assistant_text:
            assistant_texts.append(assistant_text.strip())
            log.info(f"  assistant (len={len(assistant_text)}): "
                      f"{assistant_text.strip()[:140]}")

        if not pending_calls:
            # Some Ollama+Qwen2.5 builds return an empty message after
            # a tool_result.  If we already have tool outputs but zero
            # assistant text, nudge the model explicitly once before
            # declaring the hop terminal.
            if not assistant_text and tool_results and hop < max_hops:
                log.info("  empty reply after tool_results → nudge & retry")
                history.append({
                    "role": "user",
                    "content": (
                        "이제 위에서 얻은 tool_result 들을 바탕으로 원 질문에 "
                        "대한 한국어 답변을 3~5문장으로 써줘. 숫자마다 옆에 "
                        "[도구이름] 을 붙여 인용해줘."
                    ),
                })
                continue
            log.info("  no tool_calls → final answer")
            break

        history.append({
            "role": "assistant",
            "content": assistant_text,
            "tool_calls": [{"type": "function", "function": c} for c in pending_calls],
        })

        for call in pending_calls:
            mangled = call["name"]
            args = call["arguments"] if isinstance(call["arguments"], dict) else {}
            log.info(f"  tool_call: {mangled}({args})")
            result = bridge.call(mangled, args)
            tools_invoked.append(result["tool"])
            tool_results.append(result)
            history.append({
                "role": "tool",
                "content": _safe_json_str(result["content"])[:8000],
                "name": mangled,
            })

    final_answer = assistant_texts[-1] if assistant_texts else ""

    return {
        "model": model,
        "exit_code": 0 if final_answer else 1,
        "n_hops": hop,
        "n_tool_calls": len(tools_invoked),
        "tools_invoked": tools_invoked,
        "tool_results_preview": [
            {"tool": r["tool"], "is_error": r["is_error"],
             "content_keys": list(r["content"].keys()) if isinstance(r["content"], dict) else "non_dict"}
            for r in tool_results
        ],
        "final_answer": final_answer,
        "final_answer_length": len(final_answer),
        "answer_cites_numbers": bool(re.search(r"\d", final_answer)),
        "answer_cites_tools": bool(re.search(r"epi\.\w+|\[epi[._]\w+\]", final_answer)),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


# =========================================================================
# helpers
# =========================================================================

def _safe_json(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s) if isinstance(s, str) else {}
    except Exception:   # noqa: BLE001
        return {}


def _safe_json_str(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False, default=str)
    except Exception:   # noqa: BLE001
        return str(v)


# =========================================================================
# Entry
# =========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--base", default=DEFAULT_BASE,
                         help=f"Ollama base URL (default: {DEFAULT_BASE})")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS,
                         help=f"Max tool-call hops (default: {DEFAULT_MAX_HOPS})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                         help=f"HTTP timeout sec (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--skip-probe", action="store_true",
                         help="Don't check /api/tags first (allow offline retry).")
    args = parser.parse_args()

    # Probe Ollama reachability + model presence
    if not args.skip_probe:
        ok, msg = probe_ollama(args.base, args.model)
        log.info(f"probe: {msg}")
        if not ok:
            payload = {"model": args.model, "exit_code": 3, "error": msg,
                       "hint": "start ollama + 'ollama pull <model>' then retry"}
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            log.error(msg)
            return 3

    result = run_e2e(args.model, args.base, args.max_hops, args.timeout)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"wrote {OUT}")

    # Print summary for PS caller
    print("")
    print(f"=== Ollama E2E smoke ({result['model']}) ===")
    print(f"  hops        : {result.get('n_hops', '?')}")
    print(f"  tool calls  : {result.get('n_tool_calls', 0)}  {result.get('tools_invoked', [])}")
    print(f"  answer len  : {result.get('final_answer_length', 0)}")
    print(f"  cites num   : {result.get('answer_cites_numbers', False)}")
    print(f"  elapsed     : {result.get('elapsed_sec', 0)}s")
    print(f"  exit_code   : {result['exit_code']}")
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
