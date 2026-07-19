"""G-354 (2026-06-25, P1 감사 #4): R10 PI 반폭 residual = leak-free 출처만.

근본: per_model_eval 옛 else-branch 가 leak-free OOF residual 없는 모델(현 라인업 ~46/51)에
`res = y_test - pred` 로 PI 반폭을 **채점 대상 test 점에 self-calibrate** → empirical WIS(보고값)
낙관 편향. fix = R9 in-sample(train-pool fit error 또는 model native conformal cal-split 잔차)
→ WF-CV OOF → 둘 다 없으면 WIS=NaN(test-residual 절대 금지).

leak-free 가드(핵심): **어떤 모델도 PI residual 이 (y_test - pred) 와 동일하면 안 된다.**

macOS: per-file (.venv/bin/python -m pytest tests/test_g354_pi_leakfree.py -p no:cacheprovider).
"""
import inspect

import numpy as np
import pytest

import simulation.pipeline.per_model_eval as PME


# ─────────────────────────────────────────────────────────────────────────
# 1. SOURCE-LEVEL 가드: 옛 test-residual else-branch 가 코드에서 제거됐는가
# ─────────────────────────────────────────────────────────────────────────
def test_no_test_residual_in_pi_source_block():
    """PI residual 출처 루프에 `res = y_test - pred` (test-leak) 가 없어야 한다."""
    src = inspect.getsource(PME)
    # 옛 leak: 'res = y_test - pred' (else-branch). 주석/문서 언급(은 '금지'를 설명)은 OK.
    code_lines = [ln for ln in src.splitlines()
                  if "y_test - pred" in ln and "residuals=" not in ln
                  and not ln.lstrip().startswith("#")]
    assert not code_lines, f"test-residual leak 잔존(코드): {code_lines}"


def test_pi_source_priority_and_tag_present():
    """leak-free 우선순위(r9_leakfree → wfcv_oof → unavailable) + pi_source 태그 배선 확인."""
    src = inspect.getsource(PME)
    assert "pi_source_per_model" in src, "pi_source_per_model 미배선"
    assert '"r9_leakfree"' in src, "R9 in-sample 출처 미배선"
    assert '"wfcv_oof"' in src, "WF-CV OOF 출처 미배선"
    assert '"pi_source"' in src, "row table pi_source 태그 미배선"
    # unavailable → WIS NaN 가드(test-leak 금지)
    assert "PI source unavailable" in src, "unavailable NaN 가드 미배선"


# ─────────────────────────────────────────────────────────────────────────
# 2. BEHAVIOURAL 가드: 합성 데이터로 leak-free 출처 우선순위 동작
#    (per_model_eval 의 residual 선택 로직을 in-line 으로 재현 — 같은 분기 계약)
# ─────────────────────────────────────────────────────────────────────────
def _select_residual(name, pred, y_test, pm_configs, wfcv_oof, y_in, test_start):
    """per_model_eval B1 분기 계약의 self-contained 복제 (leak-free 출처 우선순위)."""
    _base = name[:-4] if name.endswith("[fs]") else name
    res, src = None, "unavailable"
    _cfg = pm_configs.get(_base, {}) if isinstance(pm_configs, dict) else {}
    _ires = (_cfg.get("val_metrics", {}) or {}).get("insample_residuals") if isinstance(_cfg, dict) else None
    if _ires is not None:
        _a = np.asarray(_ires, dtype=np.float64)
        _a = _a[np.isfinite(_a)]
        if len(_a) >= 2:
            res, src = _a, "r9_leakfree"
    if res is None and name in wfcv_oof and wfcv_oof.get(name) is not None:
        oof_pred = np.asarray(wfcv_oof[name], dtype=np.float64)[:test_start]
        oof_y = y_in[:test_start]
        mask = np.isfinite(oof_pred) & np.isfinite(oof_y)
        _a = (oof_y - oof_pred)[mask]
        if len(_a) >= 2:
            res, src = _a, "wfcv_oof"
    return res, src


def test_r9_insample_preferred_and_not_equal_test_residual():
    """R9 in-sample residual 보유 모델은 그 출처 사용 + (y_test-pred) 와 절대 불일치."""
    y_test = np.array([10.0, 12.0, 8.0, 15.0], dtype=float)
    pred = np.array([9.0, 13.0, 7.0, 14.0], dtype=float)
    insample = [0.3, -0.2, 0.5, -0.1, 0.4, 0.05]   # leak-free (train-pool fit error)
    pm_configs = {"FusedEpi": {"val_metrics": {"insample_residuals": insample}}}
    res, src = _select_residual("FusedEpi", pred, y_test, pm_configs, {}, np.array([]), 0)
    assert src == "r9_leakfree"
    assert res is not None
    # ★ leak-free: 선택된 residual 이 test-residual(y_test-pred) 과 같으면 안 됨
    test_resid = (y_test - pred)
    assert not (len(res) == len(test_resid) and np.allclose(np.sort(res), np.sort(test_resid))), \
        "PI residual 이 test-residual 과 동일 = leak!"


def test_wfcv_oof_when_no_insample():
    """in-sample 결손 모델은 WF-CV OOF(test_start 이전 = leak-free) 사용."""
    test_start = 5
    y_in = np.array([5, 6, 7, 8, 9, 100, 101, 102], dtype=float)  # [test_start:] = test era
    oof = [4.5, 6.5, 6.0, 8.5, 8.0, 999, 999, 999]               # test era 값은 잘려야 함
    res, src = _select_residual("ARIMA", None, None, {}, {"ARIMA": oof}, y_in, test_start)
    assert src == "wfcv_oof"
    assert len(res) == test_start, "WF-CV residual 이 test era 를 포함(leak)"
    assert 999 not in [oof[i] for i in range(test_start)]   # sanity


def test_unavailable_yields_no_residual_not_test_leak():
    """leak-free 출처 둘 다 없으면 residual=None(→ WIS NaN). 절대 test-residual fallback 금지."""
    y_test = np.array([10.0, 12.0, 8.0], dtype=float)
    pred = np.array([9.0, 13.0, 7.0], dtype=float)
    res, src = _select_residual("SomeBaselineOnly", pred, y_test, {}, {}, np.array([]), 0)
    assert res is None and src == "unavailable", \
        "leak-free 출처 부재 시 test-residual 로 채우면 안 됨(낙관 편향)"


# ─────────────────────────────────────────────────────────────────────────
# 3. FusedEpi native conformal cal-split 잔차 노출(leak-free held-out split)
# ─────────────────────────────────────────────────────────────────────────
def test_fused_epi_exposes_calib_residuals_attribute():
    """FusedEpi 가 _calib_residuals 속성을 선언(미학습=None, fit 후 list)."""
    from simulation.models.fused_epi import FusedEpiForecaster
    m = FusedEpiForecaster()
    assert hasattr(m, "_calib_residuals")
    assert m._calib_residuals is None   # 미학습 default


def test_fused_epi_calib_residual_is_heldout_calsplit_not_test():
    """fit() 의 _calib_residuals 는 held-out cal split(yf[-K:] - fused_cal) = leak-free.

    소스 계약 가드: 대입식이 yf[-K:] - fused_cal 형태이고 test/X_test 를 참조하지 않는다.
    """
    from simulation.models import fused_epi
    src = inspect.getsource(fused_epi.FusedEpiForecaster.fit)
    assign = [ln.strip() for ln in src.splitlines()
              if "_calib_residuals" in ln and "=" in ln and "self._calib_residuals" in ln]
    assert assign, "fit() 가 _calib_residuals 미설정"
    line = assign[0]
    assert "yf[-K:]" in line and "fused_cal" in line, \
        f"_calib_residuals 가 held-out cal split 이 아님: {line}"
    # leak 방지: 대입식이 X_test/y_test/X_te 를 참조하면 안 됨
    assert "test" not in line.lower(), f"_calib_residuals 가 test 참조 = leak: {line}"
