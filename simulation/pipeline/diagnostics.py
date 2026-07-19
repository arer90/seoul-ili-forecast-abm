"""
R5: 잔차 진단
===================
Shapiro-Wilk, Ljung-Box, Durbin-Watson, Jarque-Bera 검정.
V16: OOF 예측값의 잔차를 사용 (WF-CV 기반).
"""
import logging
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
import time
import numpy as np
from scipy import stats
from typing import Dict

log = logging.getLogger(__name__)


def _ljung_box(residuals, lags=10):
    """Ljung-Box 검정 (자기상관 검사)."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        stat = float(result["lb_stat"].values[0])
        pval = float(result["lb_pvalue"].values[0])
        return {"statistic": stat, "p_value": pval, "significant": pval < 0.05}
    except Exception:
        return {"statistic": None, "p_value": None, "significant": None}


def _durbin_watson(residuals):
    """Durbin-Watson 검정."""
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw = float(durbin_watson(residuals))
        return {"statistic": dw, "autocorrelation": "positive" if dw < 1.5 else "negative" if dw > 2.5 else "none"}
    except Exception:
        return {"statistic": None, "autocorrelation": None}


from simulation.utils.resource_tracker import track_resources


@track_resources("phase7_diagnostics")
def run_diagnostics(y_all, oof_predictions: Dict[str, np.ndarray], config) -> dict:
    """R5: 잔차 진단 실행.

    Args:
        oof_predictions: {model_name: oof_pred_array} from R4
    """
    from .utils.logging_util import phase_banner, fmt_time
    phase_banner("R5", "잔차 진단 (Residual Diagnostics)")

    t0 = time.time()
    diagnostics = {}

    for model_name, oof_pred in oof_predictions.items():
        valid = ~np.isnan(oof_pred)
        if valid.sum() < 10:
            continue

        residuals = y_all[valid] - oof_pred[valid]

        # Shapiro-Wilk (정규성)
        n_test = min(len(residuals), 5000)
        sw_stat, sw_p = stats.shapiro(residuals[:n_test])

        # Jarque-Bera (정규성)
        jb_stat, jb_p = stats.jarque_bera(residuals)

        # Ljung-Box (자기상관)
        lb = _ljung_box(residuals)

        # Durbin-Watson
        dw = _durbin_watson(residuals)

        diagnostics[model_name] = {
            "n_residuals": int(valid.sum()),
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "skew": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals)),
            "shapiro_wilk": {"statistic": float(sw_stat), "p_value": float(sw_p),
                             "normal": sw_p > 0.05},
            "jarque_bera": {"statistic": float(jb_stat), "p_value": float(jb_p),
                            "normal": jb_p > 0.05},
            "ljung_box": lb,
            "durbin_watson": dw,
        }

        # R8.3 (2026-05-26): full 134-key SSOT eval on OOF predictions (R5 perspective).
        # Trajectory: R4 OOF → R5 residual-diagnostic 시점 134-key snapshot.
        # Toggle via env MPH_FULL_EVAL_TRAJECTORY (default '1').
        from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
        if _GCFG.filter.full_eval_trajectory:
            try:
                from simulation.pipeline.phase_evaluator import evaluate_predictions_full
                full_r8 = evaluate_predictions_full(
                    y_test=y_all[valid],
                    y_pred=oof_pred[valid],
                    residuals=residuals,
                    y_train_pool=None,
                    threshold=GLOBAL.filter.alert_threshold,
                    phase_id=f"R5_diag_{model_name}",
                    enable_bootstrap_ci=False,
                )
                diagnostics[model_name]["phase_eval_r8"] = full_r8
            except Exception as _e:
                diagnostics[model_name]["phase_eval_r8_err"] = str(_e)

        log.info(f"  [{model_name:20s}] SW p={sw_p:.4f} JB p={jb_p:.4f} DW={dw.get('statistic', 0):.3f} "
                 f"LB p={lb.get('p_value', 0):.4f}")

    elapsed = time.time() - t0
    log.info(f"  ✓ R5 완료 [{fmt_time(elapsed)}]")
    return {"diagnostics": diagnostics, "elapsed": elapsed}


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase7 = run_diagnostics
