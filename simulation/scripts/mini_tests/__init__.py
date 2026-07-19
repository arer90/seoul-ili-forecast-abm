"""Mini test framework — per-model 빠른 검증 (1-2분/model).

목적 (G-181 정정 sprint): 70 모델 각각 fit/predict 작동 + diagnostic 항목 검증.
- NaN/inf
- Negative pred (ILI ≥ 0)
- Pred 범위 (vs y_true)
- Transform/Scaler 작동
- α (anchor blend) 값

사용:
    from simulation.scripts.mini_tests.runner import run_all_models
    results = run_all_models()  # 70 model × 6 diagnostic
"""
