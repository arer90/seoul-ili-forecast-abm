"""Standalone REAL-ragas runner — executed by .venv_ragas, NOT the main venv.

The reference ``ragas`` package is incompatible with the main environment's
langchain (1.3.x): ragas pins an old langchain stack (vertexai / old core), so it
lives in an isolated venv (``.venv_ragas``: ragas 0.1.21 + langchain 0.2.x). The
main-env adapter (``ragas_metrics.ragas_real_eval``) invokes THIS script as a
subprocess with a JSON record and reads the real RAGAS scores back from stdout.

Offline: the LLM judge is a local Ollama model (default qwen2.5:3b, override with
MPH_RAGAS_MODEL); no network, no OpenAI key. Only the three LLM-only metrics
(faithfulness, context_precision, context_recall) are computed here — they need
no embedding model; answer_relevancy (which needs an embedder not available
offline) is handled by the main-env LLM-judge equivalent.

Usage:  .venv_ragas/bin/python -m simulation.llm_compare.ragas_runner <input.json>
   or:  .venv_ragas/bin/python simulation/llm_compare/ragas_runner.py <input.json>
input.json:  {"question","answer","contexts":[...],"ground_truth"?}
stdout JSON: {"faithfulness","context_precision","context_recall","ragas_version","model","method"}  | {"error", ...}
"""
import json
import os
import sys


def _score(result, name):
    """Pull a metric score from a ragas Result across 0.1.x API shapes."""
    try:
        v = result[name]
        return float(v)
    except Exception:
        pass
    try:
        df = result.to_pandas()
        if name in df.columns:
            return float(df[name].mean())
    except Exception:
        pass
    return None


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: ragas_runner.py <input.json>"}))
        return
    try:
        rec = json.loads(open(sys.argv[1], encoding="utf-8").read())
    except Exception as e:
        print(json.dumps({"error": f"input read failed: {e}"}))
        return
    try:
        import ragas
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import faithfulness, context_precision, context_recall
        from langchain_community.chat_models import ChatOllama

        model = os.environ.get("MPH_RAGAS_MODEL", "mistral:7b")  # 7B judges faithfulness well; 3B is noisy
        llm = LangchainLLMWrapper(ChatOllama(model=model, temperature=0.0))
        contexts = list(rec.get("contexts", []) or [""])
        ds = Dataset.from_dict({
            "question": [rec.get("question", "")],
            "answer": [rec.get("answer", "")],
            "contexts": [contexts],
            "ground_truth": [rec.get("ground_truth") or rec.get("answer", "")],
        })
        metrics = [faithfulness, context_precision, context_recall]
        try:
            result = evaluate(ds, metrics=metrics, llm=llm, raise_exceptions=False)
        except TypeError:
            result = evaluate(ds, metrics=metrics, llm=llm)
        out = {m: _score(result, m) for m in
               ("faithfulness", "context_precision", "context_recall")}
        out.update(ragas_version=ragas.__version__, model=model, method="ragas_package")
        print(json.dumps(out))
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"{type(e).__name__}: {e}",
                          "tb": traceback.format_exc()[-600:]}))


if __name__ == "__main__":
    main()
