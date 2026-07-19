"""G-236 후속 regression — feature-optuna key mapping (inline 3-stage).

Bug (2026-05-29, Codex+Gemini): `_inline_optuna_3stage._model_to_optuna_key`
used `_COMMON_KEY_MAP`(19) which omits DL/sequence models → TCN/N-BEATS/PatchTST/
iTransformer/Mamba/TimesNet silently fell back to "elasticnet" proxy (wrong
feature subset), while `builder._OPTUNA_MODEL_MAP_INDIVIDUAL` mapped them to "dnn".
Fix: inline now uses `_CATEGORY_KEY_MAP`(DL→dnn, study exists in MODELS_QUICK/
REPRESENTATIVE) + WARN on unmapped fallback.
"""
from __future__ import annotations


def test_dl_sequence_models_map_to_dnn_not_elasticnet():
    """Active DL/sequence models must map to the 'dnn' feature-optuna proxy,
    NOT silently to 'elasticnet'. Regression guard for the inline drift."""
    from simulation.pipeline._inline_optuna_3stage import _model_to_optuna_key
    dl_seq = ["TCN", "TCN-Optuna", "N-BEATS", "N-HiTS", "PatchTST",
              "iTransformer", "Mamba", "TimesNet", "TiDE", "DNN"]
    bad = {m: _model_to_optuna_key(m) for m in dl_seq
           if _model_to_optuna_key(m) == "elasticnet"}
    assert not bad, (
        f"DL/sequence models silently using elasticnet proxy (G-236 drift): {bad}"
    )
    for m in dl_seq:
        assert _model_to_optuna_key(m) == "dnn", (
            f"{m} should map to 'dnn' proxy, got '{_model_to_optuna_key(m)}'"
        )


def test_tree_linear_keep_own_keys():
    """tree/linear models keep their dedicated optuna keys (not collapsed)."""
    from simulation.pipeline._inline_optuna_3stage import _model_to_optuna_key
    assert _model_to_optuna_key("XGBoost") == "xgboost"
    assert _model_to_optuna_key("LightGBM") == "lightgbm"
    assert _model_to_optuna_key("KRR") == "krr"


def test_inline_key_agrees_with_builder_for_active_models():
    """inline `_model_to_optuna_key` must agree with builder's
    `_OPTUNA_MODEL_MAP_INDIVIDUAL` for every active model that builder maps —
    prevents the two paths drifting again (the original G-236-class footgun).
    DNN-family keys {dnn, tabular_dnn} are treated as interchangeable: inline
    uses the guaranteed-present 'dnn' proxy for all DL/sequence (incl. TabularDNN),
    builder uses the more-specific 'tabular_dnn' — both are DNN-family, benign."""
    from simulation.models.registry import CATEGORY_MODELS, verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.pipeline._inline_optuna_3stage import _model_to_optuna_key
    from simulation.models.feature_engine.builder import _OPTUNA_MODEL_MAP_INDIVIDUAL as BLD
    dnn_family = {"dnn", "tabular_dnn"}
    active = {m for ms in CATEGORY_MODELS.values() for m in ms}
    disagree = {}
    for m in sorted(active):
        b = BLD.get(m)
        if b is None:
            continue  # builder doesn't map it → inline free to fall back
        i = _model_to_optuna_key(m)
        if i != b and not ({i, b} <= dnn_family):
            disagree[m] = (i, b)
    assert not disagree, (
        f"inline vs builder optuna-key disagreement beyond DNN-family (drift): {disagree}"
    )


if __name__ == "__main__":
    test_dl_sequence_models_map_to_dnn_not_elasticnet()
    print("PASS  DL→dnn")
    test_tree_linear_keep_own_keys()
    print("PASS  tree/linear own keys")
    test_inline_key_agrees_with_builder_for_active_models()
    print("PASS  inline≈builder (DNN-family interchangeable)")
    print("=== ALL PASS ===")
