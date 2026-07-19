# simulation/models/__init__.py
"""
다중모델 예측 프레임워크
6범주: 시계열 / 선형·커널 / 트리 / 딥러닝 / 물리기반 / 앙상블

구조:
 base.py -- BaseForecaster ABC + ModelRegistry
 ts_models.py -- SARIMA, SARIMAX (시계열)
 linear_models.py -- SVR-Linear, SVR-RBF, ElasticNet, KRR (선형/커널)
 tree_models.py -- XGBoost, LightGBM, RandomForest (트리)
 dl_models.py -- DNN, TCN (딥러닝, PyTorch)
 tft_wrapper.py -- TFT (Temporal Fusion Transformer)
 ensemble.py -- Inv-RMSE, Stacking, Blending (메타)

 변경 (2026-03-25):
 - MLP → DNN 명칭 변경
 - LSTM/GRU/BiLSTM 제거 (소표본+distribution shift에서 WF R²<0.5)
"""

from simulation.models.base import BaseForecaster, ModelMeta, ModelRegistry, TimeSeriesForecaster

__all__ = ["BaseForecaster", "ModelMeta", "ModelRegistry", "TimeSeriesForecaster"]