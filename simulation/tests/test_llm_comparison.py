"""P5: statistical multi-LLM comparison — Wilcoxon+Holm, Fleiss κ, Cohen κ.

Pure statistics over a synthetic score matrix (dicts stand in for ScoredResponse;
_pivot accepts both), so no live LLM is needed.
"""
import numpy as np

from simulation.llm_compare.comparison import (
    bootstrap_ci,
    cohen_kappa,
    compare_backends,
    fleiss_kappa_binary,
    holm_correction,
    pairwise_wilcoxon_holm,
)


def _rows(backend, totals, prefix="q"):
    return [{"item_id": f"{prefix}{i}", "backend_id": backend, "total": t}
            for i, t in enumerate(totals)]


def test_holm_is_monotone_and_bounded():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert all(0.0 <= a <= 1.0 for a in adj)
    # smallest raw p gets ×m; order preserved as monotone step-down
    assert adj[0] == min(adj)
    assert holm_correction([0.5])[0] == 0.5


def test_wilcoxon_detects_consistent_winner():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.4, 0.6, 24)
    a = _rows("A", base + 0.2)          # A strictly better on every item
    b = _rows("B", base)
    res = pairwise_wilcoxon_holm(a + b)
    comp = res["comparisons"][0]
    assert comp["p_holm"] < 0.05 and comp["significant"] is True
    assert comp["median_diff"] > 0.1


def test_wilcoxon_no_difference_not_significant():
    base = np.linspace(0.4, 0.9, 20)
    res = pairwise_wilcoxon_holm(_rows("A", base) + _rows("B", base.copy()))
    assert res["comparisons"][0]["significant"] is False


def test_fleiss_kappa_perfect_agreement():
    # both backends pass exactly the same items → κ = 1
    totals = [0.9, 0.9, 0.3, 0.3, 0.95, 0.2]
    r = fleiss_kappa_binary(_rows("A", totals) + _rows("B", totals), threshold=0.7)
    assert r["kappa"] == 1.0


def test_fleiss_kappa_disagreement_is_low():
    # A passes the items B fails and vice-versa → κ ≤ 0
    a = _rows("A", [0.9, 0.9, 0.2, 0.2])
    b = _rows("B", [0.2, 0.2, 0.9, 0.9])
    r = fleiss_kappa_binary(a + b, threshold=0.7)
    assert r["kappa"] <= 0.0


def test_cohen_kappa_identical_and_independent():
    a = [1, 0, 1, 1, 0, 1, 0, 0]
    assert cohen_kappa(a, a)["kappa"] == 1.0
    opp = [0, 1, 0, 0, 1, 0, 1, 1]
    assert cohen_kappa(a, opp)["kappa"] < 0.0


def test_bootstrap_ci_brackets_mean():
    ci = bootstrap_ci([0.5, 0.6, 0.55, 0.7, 0.65], n_boot=2000)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_compare_backends_full_report():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.4, 0.6, 30)
    scored = _rows("A", base + 0.15) + _rows("B", base) + _rows("C", base - 0.1)
    rep = compare_backends(scored)
    assert rep["n_backends"] == 3
    assert rep["ranking"][0]["backend"] == "A"  # best mean on top
    assert "comparisons" in rep["pairwise"]
    assert "kappa" in rep["agreement"]


def test_degenerate_inputs_return_error_not_raise():
    assert "error" in pairwise_wilcoxon_holm(_rows("A", [0.5, 0.6]))  # 1 backend
    assert "error" in cohen_kappa([1, 0], [1])  # mismatched
