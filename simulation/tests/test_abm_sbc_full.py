"""Guard: full ABM-level SBC (Talts 2018) was executed and persisted — not toy-only.

D-3 TDD (Red -> Green): fails while simulation/results/abm_sbc/result.json still has
"abm": null and verdict "toy only (ABM skipped)"; passes once

    .venv/bin/python scripts/abm_sbc_check.py --abm-train 2000 --abm-sbc 1000 --abm-post 1000

(no --skip-abm) has run and populated the ABM behavioral-parameter (alpha/kappa/tau/theta)
SBC block.

This guards that the full ABM SBC was *executed* — the ABM block is populated, all four
behavioural parameters are present, and the overall verdict is no longer the skip string.
It deliberately does NOT assert the posterior is calibrated: a per-parameter KS p < 0.05
(miscalibration) is reported honestly in `abm.per_param_verdict`/`verdict`, never gated
away (no p-hacking by budget inflation).

Run per-file (macOS test-suite policy):
    .venv/bin/python -m pytest simulation/tests/test_abm_sbc_full.py -q
"""
import json
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1] / "results" / "abm_sbc" / "result.json"
NSF_RESULT = RESULT.with_name("result_nsf.json")        # NSF-flow SBC improvement
ABC_RESULT = RESULT.with_name("abc_coverage.json")      # ABC-SMC coverage calibration win
ABM_PARAMS = {"alpha", "kappa", "tau", "theta"}


def _n_calibrated(abm: dict) -> int:
    """Count behavioral params whose SBC rank is uniform (KS p >= 0.05)."""
    return sum(1 for p in abm["rank_uniform_pvalue"].values() if p >= 0.05)


def _load() -> dict:
    assert RESULT.exists(), f"SBC result.json missing: {RESULT}"
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_toy_gate_passes():
    """toy SBC must pass — it gates trust in the ABM SBC."""
    r = _load()
    assert r.get("toy_passed") is True, f"toy SBC gate must pass; toy block = {r.get('toy')}"


def test_abm_block_populated_with_four_params():
    """ABM SBC block is non-null with all four behavioural parameters."""
    r = _load()
    abm = r.get("abm")
    assert abm is not None, "ABM SBC block is null -> full ABM SBC was never executed"
    assert set(abm.get("params", [])) == ABM_PARAMS, f"ABM params = {abm.get('params')}"
    assert set(abm.get("rank_uniform_pvalue", {})) == ABM_PARAMS, "missing per-param KS p-values"
    assert set(abm.get("per_param_verdict", {})) == ABM_PARAMS, "missing per-param verdicts"


def test_verdict_no_longer_toy_only():
    """Overall verdict reflects an executed ABM SBC, not the skip placeholder."""
    r = _load()
    assert r.get("verdict") != "toy only (ABM skipped)", (
        "verdict is still 'toy only (ABM skipped)' -> ABM SBC path was not run/persisted"
    )


# ── calibration IMPROVEMENTS (genuine method fixes, not budget p-hacking) ──────
# Baseline single-round NPE-maf is SBC-miscalibrated (0/4). Two method fixes
# improve it WITHOUT inflating the simulation budget: NSF flow (better in-box fit)
# and ABC-SMC (proposes only inside prior support -> immune to flow leakage).
# These guards lock the improvement in; honesty caveat (weak identifiability /
# wide intervals) lives in the thesis, not gated away.

def test_nsf_strictly_improves_over_maf_baseline():
    """NSF density estimator calibrates strictly more params than the MAF baseline."""
    base = _load()                                   # result.json = baseline maf
    assert NSF_RESULT.exists(), f"NSF SBC result missing: {NSF_RESULT}"
    nsf = json.loads(NSF_RESULT.read_text(encoding="utf-8"))
    assert nsf.get("abm") is not None, "NSF ABM block is null"
    assert nsf["abm"].get("density_estimator") == "nsf", "NSF run not tagged density_estimator=nsf"
    assert set(nsf["abm"]["rank_uniform_pvalue"]) == ABM_PARAMS
    n_base, n_nsf = _n_calibrated(base["abm"]), _n_calibrated(nsf["abm"])
    assert n_nsf > n_base, f"NSF must improve calibration: maf {n_base}/4 -> nsf {n_nsf}/4"


def test_abc_smc_coverage_near_nominal():
    """ABC-SMC reaches near-nominal 90% credible-interval coverage (calibration win)."""
    assert ABC_RESULT.exists(), f"ABC coverage result missing: {ABC_RESULT}"
    abc = json.loads(ABC_RESULT.read_text(encoding="utf-8"))
    assert abc.get("leak_free") is True, "ABC coverage must be leak-free"
    cov = abc.get("coverage_90", {})
    assert set(cov) == ABM_PARAMS, f"coverage_90 params = {set(cov)}"
    near = sum(1 for p in ABM_PARAMS if abs(cov[p] - 0.90) <= 0.10)
    assert near >= 3, f"ABC-SMC 90% CI coverage off-nominal on >1 param: {cov}"
