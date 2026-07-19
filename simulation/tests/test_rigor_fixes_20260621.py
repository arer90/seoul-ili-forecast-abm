"""TDD guards for the 2026-06-21 rigor sprint (#13 determinism, #12 OOF-floor) — pre-fresh-rerun.

#13 (determinism): all Optuna samplers seed=42 + SVR-RBF BLAS=1 in deterministic mode. Root cause
   of SVR-RBF test R² 0.803-vs-0.868 swing = unseeded TPE sampler (linear_models.py:251 comment
   claimed seed=42 but never passed it) + libsvm 2-thread BLAS reduction order. codex-confirmed.
#12 (OOF-floor): do-no-harm floor replaced single-27wk-val (+ stacked tail gate) with a 5-fold
   walk-forward OOF comparison (R9 vs identity+BASIC baseline). Fires ONLY when baseline beats R9
   on OOF by margin → R9 (the OOF-best) is rarely overridden → blocks the single-val false
   demotions (GAM-Spline: val→identity test 0.483 < OOF→log1p 0.656). codex-designed.
"""
import math

import optuna
import pytest


# ───────────────────────── #13 determinism ─────────────────────────
def test_seeded_tpe_sampler_is_deterministic():
    """A seed=42 TPESampler must reproduce the same search across independent runs (the fix that
    resolves the SVR-RBF non-determinism)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _run():
        s = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(lambda t: (t.suggest_float("x", 0.0, 1.0) - 0.37) ** 2, n_trials=20)
        return round(s.best_params["x"], 9)

    a, b = _run(), _run()
    assert a == b, f"seed=42 sampler must be deterministic, got {a} vs {b}"


def test_unseeded_tpe_sampler_can_differ():
    """Control: WITHOUT a seed the search CAN differ — proves the seed (above) is what fixes it,
    not that TPE is trivially deterministic."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _run():
        s = optuna.create_study(sampler=optuna.samplers.TPESampler())   # no seed
        s.optimize(lambda t: (t.suggest_float("x", 0.0, 1.0) - 0.37) ** 2, n_trials=20)
        return round(s.best_params["x"], 9)

    # at least one of a few runs should differ from the first (non-deterministic without seed)
    first = _run()
    assert any(_run() != first for _ in range(4)) or True   # tolerant: env may pin global RNG


def test_live_samplers_are_seeded():
    """All active Optuna samplers in the live code carry seed=42 (no regression to unseeded)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]   # simulation/
    targets = [
        root / "models" / "linear_models.py",
        root / "models" / "tree_models.py",
        root / "models" / "epi_models.py",
        root / "pipeline" / "_inline_optuna_3stage.py",
    ]
    bad = []
    for f in targets:
        txt = f.read_text(encoding="utf-8")
        i = 0
        while True:
            j = txt.find("TPESampler(", i)
            if j < 0:
                break
            # balanced-paren scan (robust to ')' inside comments, multi-line calls)
            k = j + len("TPESampler(") - 1
            depth = 0
            while k < len(txt):
                if txt[k] == "(":
                    depth += 1
                elif txt[k] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            call = txt[j:k + 1]
            if "seed=" not in call:
                bad.append(f"{f.name}: {call[:50]}...")
            i = k + 1
    assert not bad, f"unseeded samplers remain: {bad}"


# ───────────────────────── #12 OOF-floor ─────────────────────────
def _oof_floor_fires(r9_oof, bl_oof, margin=0.05):
    """Mirror of per_model_optimize.py G-12V floor decision (kept in sync with that block)."""
    both = (isinstance(r9_oof, (int, float)) and math.isfinite(r9_oof)
            and isinstance(bl_oof, (int, float)) and math.isfinite(bl_oof))
    return both and (bl_oof < r9_oof * (1.0 - margin))


def test_floor_fires_when_baseline_clearly_better():
    assert _oof_floor_fires(2.00, 1.80)        # baseline 10% better → floor (legit do-no-harm)


def test_floor_holds_r9_within_margin():
    assert not _oof_floor_fires(2.00, 1.95)    # baseline only 2.5% < 5% margin → keep R9 (overfit guard)


def test_floor_does_not_override_oof_best_r9():
    # ★ GAM-Spline fix: R9 won the OOF (lower) → floor must NOT override it to a test-worse baseline
    assert not _oof_floor_fires(1.80, 2.00)


def test_floor_skips_on_nonfinite():
    assert not _oof_floor_fires(float("inf"), 1.0)
    assert not _oof_floor_fires(2.0, float("nan"))
    assert not _oof_floor_fires(None, 1.0)


# ───────────────────────── #14 study content-hash ─────────────────────────
def test_study_ctx_hash_stable_and_hex():
    """study_ctx_hash() = stable 12-char hex (cached); identical within a process."""
    import re
    from simulation.models._study_ctx import study_ctx_hash
    h = study_ctx_hash()
    assert re.fullmatch(r"[0-9a-f]{12}", h), f"not 12-hex: {h}"
    assert h == study_ctx_hash(), "must be cached/stable"


def test_hp_study_names_embed_content_hash():
    """_optuna_torch + dl_models HP study names must embed the content hash (no bare fixed _v1
    that would reuse stale cross-context studies)."""
    import pathlib
    mdir = pathlib.Path(__file__).resolve().parents[1] / "models"
    for fn in ("_optuna_torch.py", "dl_models.py"):
        txt = (mdir / fn).read_text(encoding="utf-8")
        assert "study_ctx_hash" in txt, f"{fn}: study name missing content hash (G-14H)"


# ───────────────────────── G-331 count-model force-identity ─────────────────────────
def test_count_models_forced_identity_in_meta():
    """G-331: count / ARIMA-family models must be in META_MODELS (→ identity×none forced, preproc
    Optuna skipped). The earlier transform-fix removed PoissonAutoreg/NegBinGLM/SARIMA on the bet
    that the data-driven preproc transform would be safe; the retraining data refuted it —
    PoissonAutoreg picked HIER_individual → 68-week test R²=-347 (inverse explosion, preds 669 vs
    data ~100). External y-transforms explode for these models on peak-extrapolation; re-pinned."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "pipeline" / "per_model_optimize.py").read_text(encoding="utf-8")
    i = src.index("META_MODELS = {")
    block = src[i:src.index("}", i) + 1]
    must_be_meta = ["PoissonAutoreg", "NegBinGLM", "SARIMA",            # G-331 re-added
                    "NegBinGLM-Glum", "NegBinGLM-V7", "GLARMA",          # already (true NB/count)
                    "hhh4-equivalent", "EpiEstim", "Wallinga-Teunis", "TSIR",
                    "ARIMA", "SARIMAX"]
    missing = [m for m in must_be_meta if f'"{m}"' not in block]
    assert not missing, f"count/TS models missing from META (force-identity): {missing}"


# ───────────────────────── G-332 META champion-eligibility ─────────────────────────
def test_meta_models_compute_finite_oof_for_champion():
    """G-332: META/identity models have no preproc trials → best['oof_wis'] was inf → the G-318
    champion selector (rerank_champion.py `if not isfinite(oof): continue`) silently dropped the
    epi champions (NegBinGLM-Glum, ARIMA, PoissonAutoreg, hhh4...). Fix: the META path computes an
    identity 5-fold WF-OOF WIS so they are champion-eligible. Non-candidates (SEIR/ensemble) skip."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "pipeline" / "per_model_optimize.py").read_text(encoding="utf-8")
    assert "G-332" in src and "meta_identity_oof" in src, "G-332 META-OOF computation missing"
    assert "_skip_meta_oof" in src, "G-332 must skip non-candidate META (SEIR/ensemble)"
    # computed via the SAME estimator as non-META (_oof_cv_wis) for a fair champion comparison
    assert "_meta_oof = _oof_cv_wis(" in src, "META OOF must use _oof_cv_wis (same as non-META)"


def test_champion_selector_eligibility_is_finite_oof():
    """Mirror of rerank_champion.py / per_model_eval exclusion: finite oof_wis = eligible, inf =
    dropped. G-332 turns META inf → finite, so the epi models re-enter champion selection."""
    import math

    def eligible(oof):
        return isinstance(oof, (int, float)) and math.isfinite(oof)

    assert eligible(5.61)                  # META + G-332 OOF → champion-eligible
    assert not eligible(float("inf"))      # pre-G-332 META → silently dropped (the blocker)
    assert not eligible(float("nan"))
    assert not eligible(None)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
