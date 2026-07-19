"""audit (2026-06-01) R11(shap)/R12(comprehensive_eval) fix 의 functional smoke + 회귀 가드.

발견+수정:
  - R12 deep-dive 가 미존재 키 fi[model]["top_features"] 읽어 SHAP 섹션 영구 공백 →
    shap 실제 출력 model_importance/shap_analysis 읽도록 정정.
  - R11 xai 가 죽은 키(옛 per_model_opt) 읽어 항상 skip → per_model_optimize 우선.
(P1 real_forecaster / overseas 풀 e2e 는 실 champion artifact / 해외 fetch 필요 → 별도 deferred.)

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q -s
"""
import inspect

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_phase16_deep_dive_shows_model_importance(tmp_path):
    """R12 per-model deep-dive 가 feature_importance.model_importance 를 실제 표시 (공백 아님)."""
    from simulation.pipeline.comprehensive_eval import _per_model_deep_dive
    all_results = {
        "feature_importance": {
            "model_importance": {
                "XGBoost": [
                    {"feature": "ili_rate_lag1", "score": 0.42},
                    {"feature": "temp_avg", "score": 0.18},
                ]
            }
        }
    }
    p = _per_model_deep_dive("XGBoost", all_results, tmp_path)
    txt = p.read_text(encoding="utf-8")
    assert "ili_rate_lag1" in txt, "deep-dive 가 model_importance feature 미표시 (SHAP키 fix 회귀)"
    assert "R11" in txt, "feature importance 섹션 헤딩이 R11(shap) 라벨과 불일치"


def test_phase16_deep_dive_falls_back_to_shap_analysis(tmp_path):
    """model_importance 없고 shap_analysis 만 있어도 표시 (fallback)."""
    from simulation.pipeline.comprehensive_eval import _per_model_deep_dive
    all_results = {
        "feature_importance": {
            "shap_analysis": {"KRR": [{"feature": "rt_pm_avg", "score": 0.31}]}
        }
    }
    txt = _per_model_deep_dive("KRR", all_results, tmp_path).read_text(encoding="utf-8")
    assert "rt_pm_avg" in txt, "shap_analysis fallback 미작동"


def test_phase16_deep_dive_no_crash_on_empty(tmp_path):
    """feature_importance 비어도 crash 없이 MD 생성 (degrade-and-continue)."""
    from simulation.pipeline.comprehensive_eval import _per_model_deep_dive
    p = _per_model_deep_dive("AnyModel", {}, tmp_path)
    assert p.exists()


def test_phase15_xai_reads_per_model_optimize_key():
    """R11 xai 가 runner 의 실제 키 per_model_optimize 를 읽음 (죽은 키 fix 회귀)."""
    import simulation.pipeline.xai as m
    src = inspect.getsource(m.run_xai)
    assert 'all_results.get("per_model_optimize")' in src, "R11 xai 죽은 키 회귀"


def test_phase15_xai_degrades_on_empty():
    """빈 all_results → skipped dict 반환, crash 없음 (degrade-and-continue)."""
    from simulation.pipeline.xai import run_xai
    from simulation.pipeline.config import PipelineConfig
    r = run_xai({}, {}, PipelineConfig())
    assert isinstance(r, dict) and r.get("skipped") is True
