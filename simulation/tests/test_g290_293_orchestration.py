"""G-290~293 (2026-06-17, 53×3 AI 감사): preproc/오케스트레이션 결함 일괄.

G-290 SARIMAX 수렴 fit 우선 · G-291 OOF fold 에 feature_names 전달(OverseasTransfer encoder) ·
G-292 active ensemble 7종 META(phase-13 preproc 175 trial 낭비 제거) · G-293 ensemble deploy skip.
"""
import re


def _src(p):
    return open(p, encoding="utf-8").read()


def test_g290_sarimax_convergence_guard():
    s = _src("simulation/models/ts_models.py")
    assert "best_conv_fit" in s and "mle_retvals" in s, "SARIMAX 수렴 guard 미적용"


def test_g291_oof_passes_feature_names():
    s = _src("simulation/pipeline/per_model_optimize.py")
    assert "model.fit(X_train_s, y_train_t, feature_names=feat_names_use)" in s, "OOF fit 에 feature_names 미전달"


def test_g292_ensembles_in_meta():
    s = _src("simulation/pipeline/per_model_optimize.py")
    m = re.search(r"META_MODELS\s*=\s*\{(.+?)\}", s, re.DOTALL)
    body = m.group(1)
    for e in ["Ensemble-NNLS", "Ensemble-BMA", "Ensemble-Adaptive", "Ensemble-ResidualAR"]:
        assert f'"{e}"' in body, f"{e} META 미등록 (preproc 낭비)"


def test_g293_ensemble_deploy_skip():
    s = _src("simulation/pipeline/per_model_optimize.py")
    assert 'str(model_name).startswith("Ensemble-")' in s and "deploy refit skip" in s, "ensemble deploy skip 미적용"


