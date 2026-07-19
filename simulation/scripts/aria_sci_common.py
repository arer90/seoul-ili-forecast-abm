"""
simulation.scripts.aria_sci_common
====================================
Shared, reproducible helpers for the SCI-grade ARIA grounding-eval RIGOR upgrade
(addresses reviewer 'single non-deterministic pass / prototype' criticism).

All three sub-analyses (multi-pass CI, grounded-vs-ungrounded significance,
inter-rater kappa) reuse the EXISTING grounding infrastructure in
``simulation.llm_compare`` (``aria_grounding`` + ``comparison.faithfulness``)
and the local Ollama models. Nothing here retrains the thesis pipeline; every
call is read-only against ``simulation/results`` artifacts.

Why a thin seed-capable subclass:
  The shipped ``OllamaBackend.generate`` does not forward Ollama's ``seed``
  option, so two passes at the same temperature are NOT reproducible. For an
  SCI claim of "mean ± CI over N passes" we want each pass to be *individually*
  reproducible (seed) while still *varying* across passes (seed + temperature
  schedule). ``SeededOllamaBackend`` adds exactly that — small interface
  (one extra kwarg path), same response contract (``LLMResponse``).

⚠ Ollama calls MUST be run SEQUENTIALLY (one model loaded at a time) to avoid
thrashing the local server — every driver in this campaign iterates models
in an outer loop and never parallelizes across models.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path

from simulation.llm_compare.backends import LLMResponse, OllamaBackend

# The five local Ollama models specified for the ARIA campaign.
ARIA_OLLAMA_MODELS = [
    "qwen2.5:3b",
    "phi3.5:3.8b",
    "mistral:7b",
    "llama3.2:1b",
    "gemma3:1b",
]

# Short display labels (match the existing multi_llm report).
MODEL_LABELS = {
    "qwen2.5:3b": "Qwen2.5-3B",
    "phi3.5:3.8b": "Phi3.5-3.8B",
    "mistral:7b": "Mistral-7B",
    "llama3.2:1b": "Llama3.2-1B",
    "gemma3:1b": "Gemma3-1B",
}


class SeededOllamaBackend(OllamaBackend):
    """OllamaBackend that forwards a per-call ``seed`` to the Ollama API.

    Args:
        model: tag-qualified Ollama model name (e.g. ``"mistral:7b"``).
        base_url: optional override; defaults to localhost:11434.

    The only added capability is ``generate(..., seed=N)``: an integer ``seed``
    is placed in the Ollama ``options`` block so a (prompt, temperature, seed)
    triple is reproducible. ``seed=None`` reproduces the stock stochastic path.

    Performance: one blocking HTTP call per ``generate`` (180s timeout).
    Side effects: network call to the local Ollama server only. Never raises —
        failures return an ``LLMResponse`` with non-empty ``.error``.
    """

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512, seed=None):
        options = {"temperature": float(temperature), "num_predict": int(max_tokens)}
        if seed is not None:
            options["seed"] = int(seed)
        body = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
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


def mean_sd_ci95(values) -> dict:
    """Mean, sd, and 95% CI (t-based for small N) of a list of floats.

    Args:
        values: list of numeric scores (one per pass). NaN/None filtered.

    Returns:
        ``{mean, sd, ci95_lo, ci95_hi, half_width, n}``. For n<2 the CI is the
        point itself (half_width=0); sd uses sample (ddof=1).

    Side effects: none. Never raises (empty -> all None).
    """
    xs = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(xs)
    if n == 0:
        return {"mean": None, "sd": None, "ci95_lo": None, "ci95_hi": None, "half_width": None, "n": 0}
    mean = sum(xs) / n
    if n < 2:
        return {"mean": round(mean, 4), "sd": 0.0, "ci95_lo": round(mean, 4),
                "ci95_hi": round(mean, 4), "half_width": 0.0, "n": n}
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    # t critical (two-sided 95%) via scipy for correctness on small N.
    try:
        from scipy.stats import t as _t
        tcrit = float(_t.ppf(0.975, n - 1))
    except Exception:
        tcrit = 1.96
    hw = tcrit * sd / math.sqrt(n)
    return {"mean": round(mean, 4), "sd": round(sd, 4),
            "ci95_lo": round(mean - hw, 4), "ci95_hi": round(mean + hw, 4),
            "half_width": round(hw, 4), "n": n}


def real_contexts():
    """The two REAL thesis grounding contexts (ABM forward + ABM real-wave fit).

    Returns:
        ``[load_real_context('identifiability'), load_real_context('abm')]`` —
        prompt-ready dicts with gold numeric ``facts``. Read-only disk access.
    """
    from simulation.llm_compare.aria_grounding import load_real_context
    return [load_real_context("identifiability"), load_real_context("abm")]


def out_dir() -> Path:
    p = Path("simulation/results/aria_sci")
    p.mkdir(parents=True, exist_ok=True)
    return p
