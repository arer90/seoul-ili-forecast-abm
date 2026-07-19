"""Factual-accuracy benchmark runner — scoring + CLI-confound-controlled wiring."""
from simulation.llm_compare.backends import (
    LLMResponse,
    MockLLMBackend,
    OpenAICompatBackend,
    discover_backends,
)
from simulation.llm_compare.factual_bench import (
    NO_TOOLS_PREAMBLE,
    extract_mcqa_letter,
    factual_prompt,
    iter_factual_items,
    run_factual_benchmark,
    score_factual,
)
from simulation.llm_compare.kr_epi_bench import load_kr_epi_law


def _resp(text):
    return LLMResponse(backend_id="t:b", model="m", text=text, latency_ms=1.0)


def test_extract_mcqa_letter_formats():
    assert extract_mcqa_letter("정답: C") == "C"
    assert extract_mcqa_letter("답은 (D)이다") == "D"
    assert extract_mcqa_letter("B. 2급감염병") == "B"
    assert extract_mcqa_letter("제 생각엔 A 입니다") == "A"
    assert extract_mcqa_letter("") == ""
    assert extract_mcqa_letter("모르겠습니다") == ""   # no letter → wrong, no crash


def test_score_factual_kr_epi_grading():
    item = next(i for i in load_kr_epi_law() if i.must_contain and i.must_avoid)
    full = score_factual("kr_epi", item, _resp(" ".join(item.must_contain)))
    assert full.total == 1.0 and full.missing_must_contain == []
    partial = score_factual("kr_epi", item, _resp(item.must_contain[0]))
    assert 0.0 < partial.total < 1.0 and partial.missing_must_contain
    # a must_avoid hit halves an otherwise-perfect answer (factual-error penalty)
    viol = score_factual("kr_epi", item, _resp(" ".join(item.must_contain) + " " + item.must_avoid[0]))
    assert viol.total == 0.5 and item.must_avoid[0] in viol.hit_must_avoid


def test_score_factual_mcqa_exact_match():
    item = {"question": "인플루엔자는 몇 급?", "options": {"A": "1급", "D": "4급"},
            "answer_letter": "D", "answer_text": "4급"}
    assert score_factual("kormedmcqa", item, _resp("정답: D")).total == 1.0
    assert score_factual("kormedmcqa", item, _resp("A 입니다")).total == 0.0


def test_score_factual_error_is_zero():
    item = next(iter(load_kr_epi_law()))
    err = LLMResponse(backend_id="t:b", model="m", text="", latency_ms=0.0,
                      error="timeout")
    assert score_factual("kr_epi", item, err).total == 0.0


def test_factual_prompt_has_no_tools_preamble():
    item = next(iter(load_kr_epi_law()))
    p = factual_prompt("kr_epi", item)
    assert NO_TOOLS_PREAMBLE in p and item.question in p
    assert "웹 검색" in p  # confound control present


def test_iter_factual_items_epi_only_no_network():
    pairs = list(iter_factual_items(0))  # 0 KorMedMCQA → no network dependency
    assert len(pairs) >= 40 and all(k == "kr_epi" for k, _ in pairs)


def test_run_factual_benchmark_mock_pipeline():
    # two mock profiles → 2-backend path exercises ranking + compare_backends + manifest
    backends = [MockLLMBackend("balanced"), MockLLMBackend("aggressive")]
    rep = run_factual_benchmark(backends, n_kormedmcqa=0, repetitions=1, verbose=False)
    assert rep["n_items"] >= 40 and rep["n_kr_epi"] >= 40 and rep["n_kormedmcqa"] == 0
    assert len(rep["ranking"]) == 2
    assert rep["ranking"][0]["accuracy"] >= rep["ranking"][1]["accuracy"]  # sorted
    # SCI stats present for ≥2 backends
    assert "ranking" in rep["statistical_comparison"]
    # reproducibility manifest with stable config hash + confound note
    m = rep["repro_manifest"]
    assert len(m["config_sha256"]) == 16
    assert "CLI=agent not raw model" in m["confound_control"]


def test_openai_compat_backend_for_vllm_mlx():
    # ONE adapter for vLLM/MLX/SGLang/LiteLLM — used in place of Ollama.
    b = OpenAICompatBackend("Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1")
    assert b.tier == "openai_compat"
    assert b.backend_id == "oai:Qwen/Qwen2.5-7B-Instruct@http://localhost:8000/v1"
    assert b._endpoint == "http://localhost:8000/v1/chat/completions"
    # no server running → graceful unavailable + errored response (never raises)
    assert b.is_available() is False
    r = b.generate("ping")
    assert r.error and r.text == ""
    # custom label override
    assert OpenAICompatBackend("m", "http://h/v1", label="vllm:qwen").backend_id == "vllm:qwen"


def test_discover_backends_accepts_openai_compat_spec():
    # unreachable server is probed + filtered out without crashing
    out = discover_backends(
        include_api=False, include_cli=False, include_ollama=False, include_mock=False,
        include_openai_compat=[{"model": "m", "base_url": "http://localhost:1/v1"}])
    assert isinstance(out, list)  # no crash; unreachable → excluded


class _AuthFailBackend(MockLLMBackend):
    """Simulates a CLI that auth-fails on every call (e.g. claude in a nested
    agent session) — every response carries an error."""
    tier = "cli"

    def __init__(self):
        super().__init__("balanced")
        self.backend_id = "cli:fake:authfail"; self.model = "fake"

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        return LLMResponse(self.backend_id, self.model, "", 1.0,
                           error="fake CLI auth failed (not logged in / 401)")


def test_auth_failed_backend_excluded_not_scored_zero():
    # an auth-failed backend must NOT appear in the ranking as 0.000 (false
    # "worst model"); it goes to `unavailable`. The working backend still ranks.
    rep = run_factual_benchmark([MockLLMBackend("balanced"), _AuthFailBackend()],
                                n_kormedmcqa=0, max_items=5, verbose=False)
    ranked = {r["backend_id"] for r in rep["ranking"]}
    unavail = {u["backend_id"] for u in rep["unavailable"]}
    assert "cli:fake:authfail" in unavail and "cli:fake:authfail" not in ranked
    assert "mock:balanced" in ranked
    # stats computed only over available backends (1 here → no pairwise)
    assert rep["statistical_comparison"] == {}


def test_max_items_caps_benchmark():
    rep = run_factual_benchmark([MockLLMBackend("balanced")], n_kormedmcqa=0,
                                max_items=3, verbose=False)
    assert rep["n_items"] == 3


def test_run_factual_repetition_variance():
    rep = run_factual_benchmark([MockLLMBackend("balanced")], n_kormedmcqa=0,
                                repetitions=3, verbose=False)
    # per-item variance across reps (NOT pooled across items). Mock is
    # deterministic → each item's rep-sd is 0 → mean_item_sd 0.
    rv = rep["repetition_variance"]["mock:balanced"]
    assert rv["reps_per_item"] == 3 and rv["n_items"] >= 40
    assert rv["mean_item_sd"] == 0.0 and rv["frac_unstable"] == 0.0
