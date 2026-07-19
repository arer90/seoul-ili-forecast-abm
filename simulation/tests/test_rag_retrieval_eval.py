"""D5 (M7): retrieval-quality eval harness for the literature RAG.

Verifies the metric math (recall@k / hit@k / MRR) on toy retrievers and that the
always-available token-overlap retriever clears a quality bar on the gold set —
turning "we have RAG" into a reportable retrieval metric.
"""
from simulation.server.rag.eval_retrieval import GOLD, evaluate, static_retriever


def test_evaluate_metric_math_on_toy_retrievers():
    gold = [("q1", frozenset({"A"})), ("q2", frozenset({"B"}))]

    perfect = lambda q, k: {"q1": ["A", "X"], "q2": ["B", "Y"]}[q]  # noqa: E731
    r = evaluate(perfect, gold, k=5)
    assert r["recall_at_k"] == 1.0 and r["hit_at_k"] == 1.0 and r["mrr"] == 1.0

    rank2 = lambda q, k: {"q1": ["X", "A"], "q2": ["Y", "B"]}[q]  # noqa: E731
    r2 = evaluate(rank2, gold, k=5)
    assert r2["hit_at_k"] == 1.0 and abs(r2["mrr"] - 0.5) < 1e-9  # 1/2

    miss = lambda q, k: ["X", "Y"]  # noqa: E731
    r3 = evaluate(miss, gold, k=5)
    assert r3["hit_at_k"] == 0.0 and r3["mrr"] == 0.0


def test_evaluate_marks_unavailable_retriever():
    r = evaluate(lambda q, k: None, [("q1", frozenset({"A"}))], k=5)
    assert r["available"] is False
    assert r["recall_at_k"] == 0.0


def test_static_retriever_meets_quality_bar_on_gold():
    r = evaluate(static_retriever, GOLD, k=5)
    assert r["available"]
    assert r["n_queries"] == len(GOLD)
    assert r["hit_at_k"] >= 0.8, f"served fallback retriever too weak: hit@5={r['hit_at_k']}"
    assert r["mrr"] >= 0.6, f"MRR too low: {r['mrr']}"
