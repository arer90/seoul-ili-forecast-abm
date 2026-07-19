"""seir_count.py — SeirCountForecaster: 단일 fused 모델 (기능 융합·조율).

설계 SSOT: docs/NEW_MODEL_IDEAS_20260622.md (커버리지 20행 + 강점 10행 + 가속 메뉴).

기능 융합 (이름 아닌 기능):
- 입력 X = engineered feature (ILI lag/Fourier + optional SEIR/decomp/spline 채널은
  feature 파이프라인이 공급; per-fold train-only 누수가드는 _loaders/mechanistic.py — 후속).
- 소표본 엔진: #1 ``engine="tabpfn"`` (in-context, 학습0 = 경량·고속, default) /
  #2 ``engine="tirex_lora"`` (무거운 변형, TiRexLoraForecaster 별도 빌드 — 현재 NotImplemented).
- count 출력: NegBin(μ̂, ψ) — Cameron-Trivedi α. predict()=count 평균, predict_quantiles()=헤드라인 WIS용 보정구간.
- conformal·direct multi-horizon = 하니스/후속(task #16 phase C5 / #17).

20-coverage 안전장치 (NEW_MODEL_IDEAS §커버리지):
- #1 역변환 폭발 불가: **count-native (transform/log1p 자체 없음)** + G-334 cap(2·y_max).
- #16 과분산: NegBin(Poisson 아님). #7 fail-loud(미fit→RuntimeError). #20 sanitize(NaN/inf→0/cap).
- #4 결정성: random_state=42. #17 portability: device='cpu' default(MPS 옵션).

가속 #3 (모델-내부): TabPFN = **학습0(O(1) fit)·CPU·11M** = 본질적 경량·고속. + inference_precision='float16'(A/B 후).

⚠ audit_and_retrain 진행 중 빌드(2026-06-23): **자동 registry 등록 안 함** — ``register_seircount()``
   명시 호출(post-run)로만 등록. heavy import(tabpfn/statsmodels/scipy)는 전부 lazy(메서드 내부).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

_ALPHA_MAX = 10.0


class SeirCountForecaster(BaseForecaster):
    """기능 융합 단일 모델: 소표본 엔진 평균 + NegBin count 출력 + 안전장치.

    Args:
        engine: "tabpfn"(#1, default 경량·고속) | "tirex_lora"(#2, 무거운 변형, 미빌드).
        n_estimators: TabPFN in-context ensemble 수 (default 4).
        inference_precision: "auto"|"float32"|"float16" (가속 #3 — float16은 A/B 후 채택).
        device: "cpu"(default, 결정성·portability) | "mps"(가속, #13 determinism 해결 후).
        alpha_max: NegBin 분산 상한 (Cameron-Trivedi clip).

    Returns (BaseForecaster 계약):
        fit(X, y) -> self. predict(X) -> μ̂ (n,) count 평균 (nonneg·cap·sanitize).
        predict_quantiles(X, levels) -> {q: (n,)} NegBin 보정 count 분위 (헤드라인 WIS용).

    Performance: TabPFN fit O(1)(학습0, context 저장) · CPU. predict in-context(~수초/CPU).
    Side effects: 최초 fit 시 TabPFN 공개 가중치 lazy 로드.
    Caller responsibility: y ≥ 0 (count). X = engineered feature (SEIR/lag/Fourier 융합은 feature 파이프라인 책임).
    """

    USES_FEATURES = True
    meta = ModelMeta(
        name="SeirCount-TabPFN",
        category="dl",
        level=16,
        min_data=50,
        description="기능 융합 단일 모델(NEW_MODEL_IDEAS): TabPFN in-context 평균 + NegBin count 출력 "
                    "+ 20-coverage 안전장치. 경량·고속(학습0, O(1) fit). #2 tirex_lora 변형 별도 빌드.",
        requires_gpu=False,
        dependencies=["tabpfn", "statsmodels", "scipy"],
    )

    def __init__(self, engine: str = "tabpfn", n_estimators: int = 4,
                 inference_precision: str = "auto", device: str = "cpu",
                 alpha_max: float = _ALPHA_MAX, mc: str = "auto",
                 mc_thresholds=(0.9, 0.8)):
        super().__init__()
        self._engine_name = str(engine)
        self._n_est = int(n_estimators)
        self._precision = str(inference_precision)
        self._device = str(device)
        self._alpha_max = float(alpha_max)
        self._engine = None
        self._alpha = 1.0          # NegBin dispersion (Cameron-Trivedi)
        self._y_train_max = 0.0    # G-334 cap base (fold-불변은 caller가 set_y_ref 주입 가능, 후속)
        # ── mc(다중공선성) 처리 = 내장 기능 (R9 per-model none/corr/pca OOF 정신) ──
        self._mc = str(mc)                 # "auto"(OOF 선택) | "none" | "corr" | "pca"
        self._mc_thresholds = tuple(mc_thresholds)
        self._mc_method = "none"           # 선택 결과
        self._mc_keep = None               # corr: 유지 feature index
        self._mc_pca = None                # pca: (scaler, pca)

    # ── NegBin dispersion: Cameron & Trivedi (1990) aux regression ──
    def _estimate_alpha(self, X: np.ndarray, y: np.ndarray) -> float:
        """((y-μ)²-μ)/μ ~ α·μ (origin OLS). Poisson GLM 으로 μ 추정 후 α."""
        try:
            import statsmodels.api as sm
            pois = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=200, disp=0)
            mu = np.clip(pois.fittedvalues, 1e-3, None)
            aux_y = ((y - mu) ** 2 - mu) / mu
            alpha = float(sm.OLS(aux_y, mu).fit().params[0])
            return float(np.clip(alpha, 1e-4, self._alpha_max))
        except Exception as e:  # 미수렴 등 → α=1.0 (NegBin 여전히 유효, 보수적 과분산)
            log.warning(f"  [SeirCount] α 추정 실패: {e} → α=1.0 fallback")
            return 1.0

    def _build_engine(self):
        """소표본 엔진 lazy 생성. #1 TabPFN(경량·고속) / #2 tirex_lora(미빌드)."""
        if self._engine_name == "tabpfn":
            from tabpfn import TabPFNRegressor
            from simulation.models.tabpfn_wrapper import _ensure_weights, _load_tabpfn_token
            _load_tabpfn_token()
            ckpt = _ensure_weights()
            kw = {"device": self._device, "ignore_pretraining_limits": True,
                  "n_estimators": self._n_est, "random_state": 42}
            if self._precision in ("float16", "float32"):
                kw["inference_precision"] = self._precision  # 가속 #3 (A/B 후)
            if ckpt is not None:
                kw["model_path"] = str(ckpt)
            return TabPFNRegressor(**kw)
        if self._engine_name == "tirex_lora":
            # #2: TiRexLoraForecaster (hand-rolled LoRA, peft 0.19.1) — 별도 빌드 (task #16 phase C2).
            raise NotImplementedError(
                "engine='tirex_lora' 미빌드 — TiRexLoraForecaster(hand-rolled LoRA) 별도 빌드 필요")
        raise ValueError(f"unknown engine: {self._engine_name}")

    # ── mc(다중공선성) 처리: 내장 기능 ──
    def _corr_keep(self, Xtr: np.ndarray, thr: float) -> list:
        """공선성 prune: 이미 keep된 feature와 |corr|≥thr 이면 제외 (train-only, 누수 0)."""
        C = np.nan_to_num(np.corrcoef(Xtr.T))
        keep = []
        for i in range(Xtr.shape[1]):
            if all(abs(C[i, j]) < thr for j in keep):
                keep.append(i)
        return keep

    def _select_mc(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """OOF holdout 으로 mc 방법 선택({none, corr@thr…, pca}) + transform 저장 (R9 per-model 정신).

        Returns: 선택된 mc 적용 후 train X. 누수 0 (transform은 train-portion 으로만 fit).
        """
        if self._mc == "none" or X.shape[1] < 4:
            self._mc_method = "none"
            return X
        # 후보 빌드
        cands = ["none"]
        if self._mc in ("auto", "corr"):
            cands += [("corr", t) for t in self._mc_thresholds]
        if self._mc in ("auto", "pca"):
            cands += [("pca", None)]
        if self._mc != "auto" and len(cands) == 2:          # 명시 단일 방법 → OOF 없이 적용
            cands = [cands[1]]
        # OOF: 마지막 20% holdout (시계열 순서 보존)
        n = len(y); k = max(10, n // 5)
        Xtr_o, ytr_o, Xva, yva = X[:n - k], y[:n - k], X[n - k:], y[n - k:]
        from simulation.pipeline.phase_evaluator import evaluate_predictions_full

        def _fit_apply(spec, Xa, Xb):
            """spec → (Xa_t, Xb_t, store_fn). store_fn()=최종 저장."""
            if spec == "none":
                return Xa, Xb, lambda: ("none", None, None)
            kind, par = spec
            if kind == "corr":
                keep = self._corr_keep(Xa, par)
                return Xa[:, keep], Xb[:, keep], lambda: ("corr", keep, None)
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xa); pca = PCA(n_components=0.95).fit(sc.transform(Xa))
            return (pca.transform(sc.transform(Xa)), pca.transform(sc.transform(Xb)),
                    lambda: ("pca", None, (sc, pca)))

        best, best_wis, best_store = "none", float("inf"), None
        for spec in cands:
            try:
                Xa_t, Xb_t, store = _fit_apply(spec, Xtr_o, Xva)
                eng = self._build_engine()
                import warnings as _w
                with np.errstate(all="ignore"), _w.catch_warnings():
                    _w.simplefilter("ignore")
                    eng.fit(Xa_t, ytr_o); pred = np.asarray(eng.predict(Xb_t), dtype=float)
                wis = float(evaluate_predictions_full(yva, pred, sigma=1.0,
                            y_train_pool=ytr_o, phase_id="mc_sel").get("wis", np.inf))
                if np.isfinite(wis) and wis < best_wis:
                    best, best_wis, best_store = spec, wis, store
            except Exception as e:
                log.debug(f"  [SeirCount] mc 후보 skip {spec}: {e}")
        # 선택 결과 저장 + 전체 train 에 재적용 (transform 은 full-train 으로 refit)
        if best == "none" or best_store is None:
            self._mc_method = "none"; return X
        kind, par = best
        if kind == "corr":
            self._mc_keep = self._corr_keep(X, par); self._mc_method = f"corr{par}"
            return X[:, self._mc_keep]
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X); pca = PCA(n_components=0.95).fit(sc.transform(X))
        self._mc_pca = (sc, pca); self._mc_method = "pca"
        return pca.transform(sc.transform(X))

    def _apply_mc(self, X: np.ndarray) -> np.ndarray:
        if self._mc_method.startswith("corr") and self._mc_keep is not None:
            return X[:, self._mc_keep]
        if self._mc_method == "pca" and self._mc_pca is not None:
            sc, pca = self._mc_pca
            return pca.transform(sc.transform(X))
        return X

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "SeirCountForecaster":
        import warnings as _w
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        self._y_train_max = float(np.max(y)) if y.size else 0.0
        X = self._select_mc(X, y)                           # ★mc 내장 기능 (OOF 선택)
        self._alpha = self._estimate_alpha(X, y)            # NegBin 과분산
        eng = self._build_engine()
        with np.errstate(all="ignore"), _w.catch_warnings():
            _w.simplefilter("ignore")
            eng.fit(X, y)                                   # TabPFN: O(1) context 저장 (학습0)
        self._engine = eng
        self._fitted = True
        log.info(f"  [SeirCount] mc={self._mc_method} (p {X_train.shape[1]}→{X.shape[1]}) α={self._alpha:.3f}")
        return self

    def _cap(self, arr: np.ndarray) -> np.ndarray:
        """count-native nonneg + cap(2·train_max) + sanitize (G-334/G-159). 역변환 폭발 불가(transform 없음)."""
        cap = 2.0 * self._y_train_max if self._y_train_max > 0 else np.inf
        a = np.asarray(arr, dtype=float)
        a = np.nan_to_num(a, nan=0.0, posinf=(cap if np.isfinite(cap) else 0.0), neginf=0.0)
        return np.clip(a, 0.0, cap)

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if not self._fitted or self._engine is None:
            raise RuntimeError("SeirCount: fit() 먼저 호출")     # G-237 fail-loud
        import warnings as _w
        Xt = self._apply_mc(np.asarray(X_test, dtype=float))    # ★mc transform (fit서 선택된 것)
        with np.errstate(all="ignore"), _w.catch_warnings():
            _w.simplefilter("ignore")
            mu = np.asarray(self._engine.predict(Xt), dtype=float)
        return self._cap(mu)                                    # count 평균

    def predict_quantiles(self, X_test: np.ndarray,
                          levels=(0.025, 0.25, 0.5, 0.75, 0.975), **kwargs) -> dict:
        """NegBin(μ̂, ψ) count 보정 분위 — 헤드라인 WIS용. r=1/α, p=r/(r+μ)."""
        from scipy.stats import nbinom
        mu = np.clip(self.predict(X_test), 1e-6, None)
        r = 1.0 / max(self._alpha, 1e-4)
        p = r / (r + mu)
        cap = 2.0 * self._y_train_max if self._y_train_max > 0 else np.inf
        return {q: np.clip(np.asarray(nbinom.ppf(q, r, p), dtype=float), 0.0, cap) for q in levels}


def register_seircount() -> None:
    """명시/자동 등록 (import 시 자동호출 — 검증·완성 2026-06-23 후 활성)."""
    try:
        REGISTRY.register(SeirCountForecaster)
        log.info("[seir_count] SeirCountForecaster 등록됨")
    except Exception as e:
        log.debug(f"[seir_count] 등록 skip: {e}")


# ── import 시 자동 등록 (FusedEpi 등과 동일 패턴) ──
register_seircount()
