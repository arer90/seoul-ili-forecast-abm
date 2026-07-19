"""fused_epi.py — FusedEpiForecaster: TiRex+TabPFN을 **1개 모델**로 융합 (보건역학 특화).

NEW_MODEL_IDEAS 최종 융합 모델. "모델 여러 개"가 아니라 **단일 모델 1개**가 내부에 두 foundation을
컴포넌트로 흡수하고, 그 위에 모든 기법을 *기능*으로 탑재:

융합 컴포넌트 (1 모델 내부):
- **TiRex**(35M xLSTM, 시계열 foundation) = 강한 base 예측 (rolling 1-step).
- **TabPFN**(tabular foundation, 소표본 in-context) = 잔차(y−TiRex) 보정.
  → 최종 = TiRex + α·TabPFN-보정. 한 fit/predict/artifact = 단일 모델.

탑재 기법 (1 모델 내부 기능):
- ★**NegBin count 출력**: predict_quantiles = count 보정구간 (보건역학 over-dispersion).
- ★**mc 처리**: 보정 feature 공선성 corr-prune (R9 정신, train-only).
- ★**mechanistic**: Rt·S/N·FoI 인과 채널 (1-lag, 누수 0).
- ★**동적 데이터적응**: blend α(n) — 소표본→base 신뢰(α↓), 대표본→보정(α↑).
- ★**do-no-harm 안정화**: 보정이 cal서 해로우면 α→0.
- ★**conformal(CQR)**: cal 잔차로 NegBin 분위 보정 → coverage 보장.
- ★**캐시화**: TiRex 예측·mechanistic 메모리 캐시 → 재계산 0.

Performance: fit = TiRex rolling(캐시) + TabPFN O(1) + conformal. CPU. Side effects: TiRex/TabPFN lazy 로드.
Caller responsibility: y ≥ 0. USES_FEATURES=True. rolling 1-step eval(predict에 y_observed).
"""
from __future__ import annotations

import logging

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

# 모듈-레벨 캐시 (단일 모델 내부 기능 — 재평가/멀티-fit 공유).
_TIREX_CACHE: dict = {}
_MECH_CACHE: dict = {}


def _ctx_key(arr: np.ndarray) -> int:
    return hash(np.asarray(arr, dtype=np.float32).tobytes())


class FusedEpiForecaster(BaseForecaster):
    """TiRex+TabPFN 융합 단일 모델 + NegBin·mc·mechanistic·동적α·conformal·캐시.

    Args:
        n_ref: 동적 α 데이터규모 기준 (α=clip(n/n_ref, α_min, 1)).
        alpha_min: 최소 보정 가중(소표본 안정 바닥).
        use_mc / mc_thr: 보정 feature 공선성 corr-prune (|r|≥thr 제외).
        cal_frac: conformal·do-no-harm 용 calibration tail 비율.
        max_context: TiRex 컨텍스트. use_cache: TiRex/mech 캐시. nb_alpha_max: NegBin 분산 상한.
    """

    USES_FEATURES = True
    meta = ModelMeta(
        name="FusedEpi",
        category="foundation",
        level=16,
        min_data=70,
        description="TiRex+TabPFN 1모델 융합: TiRex base + TabPFN 잔차보정 + NegBin count + mc(corr-prune) "
                    "+ mechanistic + 동적 데이터적응 α(n) + do-no-harm + conformal(CQR) + 캐시. 단일 fit/predict.",
        requires_gpu=False,
        dependencies=["tirex", "tabpfn", "statsmodels", "scipy"],
    )

    def __init__(self, n_ref: int = 500, alpha_min: float = 0.2, use_mc="auto",
                 mc_margin: float = 0.02, mc_thr: float = 0.8, cal_frac: float = 0.2, max_context: int = 256,
                 min_ctx: int = 52, use_cache: bool = True, nb_alpha_max: float = 10.0,
                 adaptive_conf: bool = True, conf_window: int = 30,
                 pid_ki: float = 0.2, asym_conf="auto", min_asym_n: int = 80,
                 skew_thr: float = 0.5, pi_method: str = "negbin", tweedie_p: float = 1.5,
                 repo_id: str = "NX-AI/TiRex"):
        super().__init__()
        self.n_ref = int(n_ref); self.alpha_min = float(alpha_min)
        self.use_mc = use_mc; self.mc_margin = float(mc_margin)   # "auto"|True|False — do-no-harm mc
        self.mc_thr = float(mc_thr); self._mc_reason = ""
        self.cal_frac = float(cal_frac); self.max_context = int(max_context)
        self.min_ctx = int(min_ctx); self.use_cache = bool(use_cache)
        self.nb_alpha_max = float(nb_alpha_max); self.repo_id = repo_id
        self.adaptive_conf = bool(adaptive_conf)    # ★동적 conformal (팬데믹/충격 대응)
        self.conf_window = int(conf_window)         # rolling 점수 윈도우
        self.pid_ki = float(pid_ki)                 # ★Conformal PID I-term 게인 (coverage 오차 적분, 0=끔)
        self.asym_conf = asym_conf                  # "auto"|True|False — 비대칭 nonconformity (정점 우편향)
        self.min_asym_n = int(min_asym_n)           # auto: 비대칭 viable 최소 cal 크기
        self.skew_thr = float(skew_thr)             # auto: 잔차 우편향 임계
        self._use_asym = False; self._asym_reason = ""   # auto 결정 결과
        self._tx = None; self._corr = None; self._train_y = None
        self._alpha = 0.0; self._nb_disp = 1.0; self._y_max = 0.0
        self._mc_keep = None; self._conf = {}
        self._conf_scores = {}; self._conf_beta = {}   # adaptive: cal 점수 + 분위레벨
        self._conf_lo = {}; self._conf_hi = {}          # ★비대칭: 하단/상단 per-side cal 점수
        self._resid_scale = 1.0                          # 정상 잔차 스케일 (충격 magnitude 게이트)
        self._calib_residuals = None                     # G-354: leak-free held-out cal-split 잔차 (R10 PI 출처)
        self.pi_method = str(pi_method)                  # "negbin"(default; PID adaptive conformal — 정본 test서 우세) | "tweedie"(선택지; 분산-안정화 residual-scale 헤드, supplementary/robustness용)
        self.tweedie_p = float(tweedie_p)                # Tweedie 분산함수 power p (구간 폭 ∝ μ^(p/2)); 기본 1.5
        self._fused_cal = None                           # Tweedie 표준화용 cal-split 융합 point (leak-free)

    # ── 캐시된 mechanistic feature (1-lag, 누수 0) ──
    def _mech_features(self, y_full: np.ndarray) -> np.ndarray:
        key = _ctx_key(y_full)
        if self.use_cache and key in _MECH_CACHE:
            return _MECH_CACHE[key]
        from simulation.models.feature_engine._loaders.mechanistic import mechanistic_features
        mech = mechanistic_features(np.asarray(y_full, dtype=float))
        mech_lag = np.vstack([mech[:1], mech[:-1]])
        if self.use_cache:
            _MECH_CACHE[key] = mech_lag
        return mech_lag

    # ── 캐시된 TiRex 1-step ──
    def _tirex_1step(self, ctx: np.ndarray) -> float:
        import torch
        key = _ctx_key(ctx[-self.max_context:])
        if self.use_cache and key in _TIREX_CACHE:
            return _TIREX_CACHE[key]
        t = torch.tensor(ctx[-self.max_context:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _q, mean = self._tx.forecast(context=t, prediction_length=1)
        v = float(np.asarray(mean).ravel()[0])
        if self.use_cache:
            _TIREX_CACHE[key] = v
        return v

    def _tirex_roll(self, y_full: np.ndarray, idxs) -> np.ndarray:
        return np.array([self._tirex_1step(y_full[:t]) for t in idxs], dtype=float)

    # ── mc: 공선성 corr-prune (train-only) ──
    def _corr_keep(self, Xtr: np.ndarray, thr: float) -> list:
        C = np.nan_to_num(np.corrcoef(Xtr.T)); keep = []
        for i in range(Xtr.shape[1]):
            if all(abs(C[i, j]) < thr for j in keep):
                keep.append(i)
        return keep

    def _select_mc_keep(self, Xf_all, resid, K):
        """★mc do-no-harm 선택 — auto=OOF none vs corr-prune(margin 못 넘으면 none) / True=corr / False=none.

        모델 동적-적응 철학 일관: 공선성 제거가 cal서 실제 도움될 때만 적용, 아니면 none(보수적).
        Returns: keep index 또는 None(=전체 사용).
        """
        if self.use_mc is False or self.use_mc == "none":
            self._mc_reason = "none(off)"
            return None
        if self.use_mc is True or self.use_mc == "corr":
            self._mc_reason = "corr(forced)"
            return self._corr_keep(Xf_all, self.mc_thr)
        # auto: OOF none vs corr-prune (do-no-harm margin)
        keep_corr = self._corr_keep(Xf_all[:-K], self.mc_thr)
        err = {}
        for name, keep in (("none", None), ("corr", keep_corr)):
            Xc = Xf_all[:, keep] if keep is not None else Xf_all
            t = self._tab(); t.fit(Xc[:-K], resid[:-K])
            err[name] = float(np.mean((resid[-K:] - np.asarray(t.predict(Xc[-K:]), dtype=float)) ** 2))
        use_corr = err["corr"] < err["none"] * (1.0 - self.mc_margin)   # corr가 margin 넘게 우수할 때만
        self._mc_reason = (f"auto: none={err['none']:.1f} corr={err['corr']:.1f} "
                           f"→ {'corr' if use_corr else 'none'}")
        return self._corr_keep(Xf_all, self.mc_thr) if use_corr else None

    def _nb_dispersion(self, mu, y):
        mu = np.clip(np.asarray(mu, float), 1e-3, None); y = np.asarray(y, float)
        v = float(np.mean(((y - mu) ** 2 - mu) / np.maximum(mu, 1e-3)))
        return float(np.clip(v, 1e-4, self.nb_alpha_max))

    # ── conformal(CQR): cal 잔차로 분위 보정량 + 동적용 점수 저장 ──
    def _fit_conformal(self, fused_cal, y_cal) -> dict:
        from scipy.stats import nbinom
        mu = np.clip(np.asarray(fused_cal, float), 1e-6, None)
        r = 1.0 / max(self._nb_disp, 1e-4); p = r / (r + mu)
        yc = np.asarray(y_cal, float); n = len(yc); conf = {}
        self._conf_scores = {}; self._conf_beta = {}
        for lo, hi in [(0.025, 0.975), (0.25, 0.75)]:
            qlo = np.asarray(nbinom.ppf(lo, r, p), float); qhi = np.asarray(nbinom.ppf(hi, r, p), float)
            E = np.maximum(qlo - yc, yc - qhi)               # CQR conformity score
            beta = min(1.0, (hi - lo) * (1 + 1.0 / max(n, 1)))   # (1-α)(1+1/n), α=1-(hi-lo)
            conf[(lo, hi)] = max(0.0, float(np.quantile(E, beta)))   # static Q (widen-only)
            self._conf_scores[(lo, hi)] = E                  # ★동적용: cal 점수 버퍼 시드 (대칭)
            self._conf_lo[(lo, hi)] = qlo - yc               # ★비대칭 하단 점수(양수=하단 위반)
            self._conf_hi[(lo, hi)] = yc - qhi               # ★비대칭 상단 점수(양수=상단 위반)
            self._conf_beta[(lo, hi)] = beta
        return conf

    def _decide_asym(self, resid_cal, k_cal) -> None:
        """★비대칭 conformal auto 결정 — 데이터 size + content(왜도) 기반 (모델 동적-적응 철학 일관).

        규칙: cal 충분(K≥min_asym_n) ∧ 잔차 우편향(|skew|>skew_thr) 일 때만 비대칭. 그 외 대칭.
        소표본(per-side 노이즈)·대칭분포(이득無)는 자동 symmetric, 대표본·skewed만 asymmetric.
        """
        if self.asym_conf == "auto":
            try:
                from scipy.stats import skew as _skew
                sk = float(_skew(np.asarray(resid_cal, dtype=float))) if k_cal > 3 else 0.0
            except Exception:
                sk = 0.0
            self._use_asym = bool(k_cal >= self.min_asym_n and abs(sk) > self.skew_thr)
            self._asym_reason = (f"auto: K={k_cal}{'≥' if k_cal >= self.min_asym_n else '<'}{self.min_asym_n}, "
                                 f"skew={sk:+.2f}{'>' if abs(sk) > self.skew_thr else '≤'}{self.skew_thr} "
                                 f"→ {'비대칭' if self._use_asym else '대칭'}")
        else:
            self._use_asym = bool(self.asym_conf)
            self._asym_reason = f"manual={self.asym_conf}"

    def _tab(self):
        from tabpfn import TabPFNRegressor
        from simulation.models.tabpfn_wrapper import _ensure_weights, _load_tabpfn_token
        _load_tabpfn_token(); ck = _ensure_weights()
        kw = dict(device="cpu", ignore_pretraining_limits=True, n_estimators=4, random_state=42)
        if ck:
            kw["model_path"] = str(ck)
        return TabPFNRegressor(**kw)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "FusedEpiForecaster":
        from tirex import load_model
        y = np.asarray(y_train, dtype=float).ravel(); X = np.asarray(X_train, dtype=float)
        self._train_y = y; self._y_max = float(np.max(y)) if y.size else 0.0
        n = len(y)
        if self._tx is None:
            self._tx = load_model(self.repo_id, device="cpu")

        # 1) TiRex base (train rolling, 캐시) → 잔차
        tr_idx = list(range(self.min_ctx, n))
        tx_tr = self._tirex_roll(y, tr_idx)
        resid = y[self.min_ctx:] - tx_tr
        yf = y[self.min_ctx:]

        # 2) feature = lag/seasonal + mechanistic(1-lag)
        mech = self._mech_features(y)
        Xf_all = np.hstack([X, mech])[self.min_ctx:]

        # 3) ★mc do-no-harm: auto=OOF none vs corr-prune(margin 못 넘으면 none) / True=corr / False=none
        K = max(10, int(len(yf) * self.cal_frac))
        self._mc_keep = self._select_mc_keep(Xf_all, resid, K)
        Xf = Xf_all[:, self._mc_keep] if self._mc_keep is not None else Xf_all

        # 4) conformal/do-no-harm split: 마지막 K = calibration
        corr_pt = self._tab(); corr_pt.fit(Xf[:-K], resid[:-K])      # proper-train 보정
        corr_cal = np.asarray(corr_pt.predict(Xf[-K:]), float)

        # 5) ★동적 α(n) + do-no-harm (cal 기준, soft 비율 — hard-zero는 작은 cal서 과함)
        alpha_size = float(np.clip(n / self.n_ref, self.alpha_min, 1.0))
        base_err = float(np.mean(resid[-K:] ** 2))
        corr_err = float(np.mean((resid[-K:] - corr_cal) ** 2))
        harm = float(np.clip(base_err / (corr_err + 1e-9), 0.0, 1.0))   # 보정 우수→1, 열등→비율 shrink
        self._alpha = alpha_size * harm

        # 6) NegBin 분산 + ★conformal(CQR) — cal 기준
        fused_cal = np.clip(tx_tr[-K:] + self._alpha * corr_cal, 0, None)
        self._nb_disp = self._nb_dispersion(fused_cal, yf[-K:])
        self._conf = self._fit_conformal(fused_cal, yf[-K:])
        self._resid_scale = float(np.std(yf[-K:] - fused_cal)) + 1e-6   # 정상 잔차 스케일(충격 magnitude 게이트용)
        self._decide_asym(yf[-K:] - fused_cal, len(yf[-K:]))            # ★비대칭 auto 결정 (size+content)

        # 7) 최종 보정 = ALL train refit (배포 정확도)
        self._corr = self._tab(); self._corr.fit(Xf, resid)
        # G-354 (2026-06-25, P1 감사 #4): leak-free held-out cal-split 잔차 노출.
        #   R10 PI 반폭 보정의 누수-free 출처 — test-residual self-calibration(y_test-pred) 대체.
        #   yf[-K:]·fused_cal·K 모두 (5)~(6) 단계서 산출된 held-out conformal cal split → test 미접근.
        #   rolling 1-step eval 의 오차분포와 동일 레짐(마지막 K주) → np.std·K11 반폭 정합.
        self._calib_residuals = (yf[-K:] - fused_cal).tolist()
        self._fused_cal = np.asarray(fused_cal, dtype=float)     # Tweedie 표준화용 (leak-free cal split)
        self._fitted = True
        log.info(f"  [FusedEpi] α={self._alpha:.3f} mc={'corr%.2f' % self.mc_thr if self._mc_keep else 'none'}"
                 f"(p {Xf_all.shape[1]}→{Xf.shape[1]}) nb_disp={self._nb_disp:.3f} "
                 f"conf={ {k: round(v,2) for k,v in self._conf.items()} } cache(tx={len(_TIREX_CACHE)})")
        return self

    def _corr_features(self, X, y_full, n_test, obs):
        mech = self._mech_features(y_full)
        mech_te = mech[-n_test:] if obs is not None else np.tile(mech[-1], (n_test, 1))
        Xf = np.hstack([X, mech_te])
        return Xf[:, self._mc_keep] if self._mc_keep is not None else Xf

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FusedEpi: fit() 먼저")            # fail-loud
        X = np.asarray(X_test, dtype=float); n_test = len(X)
        obs = np.asarray(y_observed, dtype=float).ravel() if y_observed is not None else None
        base = np.empty(n_test, dtype=float)
        for i in range(n_test):
            hist = np.concatenate([self._train_y, obs[:i]]) if (obs is not None and i > 0) else self._train_y
            base[i] = self._tirex_1step(hist)
        y_full = np.concatenate([self._train_y, obs]) if obs is not None else self._train_y
        Xf = self._corr_features(X, y_full, n_test, obs)
        corr = np.asarray(self._corr.predict(Xf), dtype=float)
        fused = base + self._alpha * corr
        cap = 2.0 * self._y_max if self._y_max > 0 else np.inf
        return np.clip(np.nan_to_num(fused, nan=0.0), 0.0, cap)

    @staticmethod
    def _pid_conformal_adjust(qlo, qhi, obs, init_scores, beta, target,
                              window=30, ki=0.2, cap=np.inf):
        """★Conformal PID (P+I) 컨트롤러 — 분포·n 무관 long-run coverage 보장 (Angelopoulos 2024).

        P (분위추적): 최근 window nonconformity 점수의 beta-분위 (rolling, 점수 LEVEL 추적).
        I (적분): coverage 오차 누적 Σ(miscov−target) → 충격 후 coverage drift 교정 (현 rolling엔 없던 조각).
        D (scorecaster): **미채택** — 잔차추세 학습이 소표본 outbreak peak에 과적합 위험.

        소표본/대표본 공통: 학습 파라미터 0개, 스칼라 게인 ki만. I-term이 empirical miscoverage→target 구동.
        Args: qlo/qhi=NegBin 분위, obs=관측(rolling), init_scores=cal 점수 시드, beta=P 분위레벨,
              target=목표 miscoverage(=1−신뢰수준), ki=I 게인, cap=상한.
        Returns: (nlo, nhi) 보정 구간.
        """
        qlo = np.asarray(qlo, dtype=float); qhi = np.asarray(qhi, dtype=float)
        obs = np.asarray(obs, dtype=float).ravel()
        n = len(qlo); nlo = qlo.copy(); nhi = qhi.copy()
        buf = list(init_scores); integral = 0.0
        for i in range(n):
            q_p = max(0.0, float(np.quantile(buf[-window:], beta))) if buf else 0.0   # P: 분위추적
            scale = max(q_p, 1.0)
            Q = max(0.0, q_p + ki * scale * integral)                                 # P + I
            nlo[i] = max(0.0, qlo[i] - Q); nhi[i] = min(cap, qhi[i] + Q)
            miscov = 1.0 if (obs[i] < qlo[i] - Q or obs[i] > qhi[i] + Q) else 0.0
            integral = float(np.clip(integral + (miscov - target), -5.0, 5.0))        # I: 적분(windup clip)
            buf.append(float(max(qlo[i] - obs[i], obs[i] - qhi[i])))                  # 점수 버퍼 갱신
        return nlo, nhi

    @staticmethod
    def _wquantile(vals, q, recency=True):
        """recency-weighted 분위 — 최근 점수 가중↑ (drift 추종). vals=window 점수, q=분위레벨."""
        v = np.asarray(vals, dtype=float)
        if v.size == 0:
            return 0.0
        if not recency or v.size < 3:
            return float(np.quantile(v, q))
        w = np.linspace(0.5, 1.5, v.size)            # 최근일수록 가중↑
        order = np.argsort(v); vs = v[order]; ws = w[order]
        cw = np.cumsum(ws); cw = (cw - 0.5 * ws) / cw[-1]
        return float(np.interp(q, cw, vs))

    @staticmethod
    def _pid_conformal_adjust_asym(qlo, qhi, obs, init_lo, init_hi, target_side,
                                   window=30, ki=0.2, cap=np.inf, recency=True):
        """★비대칭 + recency Conformal PID — 하단/상단 *독립* Q (정점 우편향 sharpness).

        ILI count 잔차는 정점서 우편향(상단 heavy tail) → 대칭 Q는 상단 위반 시 하단까지 과확장(폭 낭비).
        하단 점수 E_lo=qlo−y, 상단 E_hi=y−qhi 각각 (1−target_side) 분위 + 양측 독립 I-term. 각 Q는 부호 자유
        (과대피복 측은 음수=narrowing). target_side=α/2(양측 분할).
        """
        qlo = np.asarray(qlo, dtype=float); qhi = np.asarray(qhi, dtype=float)
        obs = np.asarray(obs, dtype=float).ravel()
        n = len(qlo); nlo = qlo.copy(); nhi = qhi.copy()
        blo = list(init_lo); bhi = list(init_hi); ilo = 0.0; ihi = 0.0
        beta_side = 1.0 - target_side
        for i in range(n):
            qp_lo = FusedEpiForecaster._wquantile(blo[-window:], beta_side, recency)
            qp_hi = FusedEpiForecaster._wquantile(bhi[-window:], beta_side, recency)
            Q_lo = qp_lo + ki * max(abs(qp_lo), 1.0) * ilo   # 하단 radius (음수=과대피복 시 조임)
            Q_hi = qp_hi + ki * max(abs(qp_hi), 1.0) * ihi   # 상단 radius
            lo_b = max(0.0, qlo[i] - Q_lo); hi_b = min(cap, qhi[i] + Q_hi)
            if lo_b > hi_b:
                lo_b = hi_b
            nlo[i] = lo_b; nhi[i] = hi_b
            m_lo = 1.0 if obs[i] < lo_b else 0.0             # 하단 위반
            m_hi = 1.0 if obs[i] > hi_b else 0.0             # 상단 위반
            ilo = float(np.clip(ilo + (m_lo - target_side), -5.0, 5.0))
            ihi = float(np.clip(ihi + (m_hi - target_side), -5.0, 5.0))
            blo.append(float(qlo[i] - obs[i])); bhi.append(float(obs[i] - qhi[i]))
        return nlo, nhi

    def predict_quantiles(self, X_test, y_observed=None, levels=(0.025, 0.25, 0.5, 0.75, 0.975), **kwargs) -> dict:
        """NegBin count 보정구간 + ★adaptive conformal — 팬데믹/충격 시 동적 확장.

        rolling(y_observed) 이면 Q를 *최근 conf_window 잔차 점수*에서 계산 → 분포이동(COVID 등) 시
        최근 점수 급증 → 구간 자동 확장 → coverage 유지. static(단일원점)이면 고정 Q.
        """
        from scipy.stats import nbinom
        mu = np.clip(self.predict(X_test, y_observed=y_observed), 1e-6, None)
        cap = 2.0 * self._y_max if self._y_max > 0 else np.inf
        if self.pi_method == "tweedie" and self._fused_cal is not None:
            # ★Tweedie 분산-안정화 residual-scale 헤드 + EXPANDING split-CQR (flag-gated; NegBin+PID와 배타):
            #   q(τ)=μ+Qz(τ)·μ^(pw/2), Qz=EXPANDING 표준화잔차 분위(held-out cal seed→rolling 관측 확장, leak-free).
            #   그 위에 per-alpha expanding CQR 보정. 폭 ∝ μ^0.75 → peak서 확장. (검증된 campaign 방법과 일치.)
            pw = self.tweedie_p; mu_cal = np.clip(self._fused_cal, 1e-3, None)
            z_hist = list(np.asarray(self._calib_residuals, dtype=float) / (mu_cal ** (pw / 2.0)))
            obs2 = np.asarray(y_observed, dtype=float).ravel() if y_observed is not None else None
            lv = sorted(set(levels)); med = min(lv, key=lambda q: abs(q - 0.5))
            pairs = [(q, round(1.0 - q, 4)) for q in lv if q < 0.5 and round(1.0 - q, 4) in lv]
            out = {q: np.zeros(len(mu)) for q in lv}; E = {pr: [] for pr in pairs}
            for i in range(len(mu)):
                mu_s = float(np.clip(mu[i], 1e-3, None) ** (pw / 2.0))
                qraw = {q: mu[i] + float(np.quantile(z_hist, q)) * mu_s for q in lv}
                out[med][i] = float(np.clip(qraw[med], 0.0, cap))
                for (lo, hi) in pairs:
                    pe = E[(lo, hi)]
                    Q = float(np.quantile(pe, min(1.0, (hi - lo) * (1 + 1.0 / max(len(pe), 1))))) if len(pe) >= 5 else 0.0
                    out[lo][i] = float(np.clip(qraw[lo] - Q, 0.0, cap)); out[hi][i] = float(np.clip(qraw[hi] + Q, 0.0, cap))
                if obs2 is not None and i < len(obs2):                # rolling 관측으로 expanding (leak-free: i 예측 후 갱신)
                    z_hist.append((obs2[i] - mu[i]) / mu_s)
                    for (lo, hi) in pairs:
                        E[(lo, hi)].append(max(qraw[lo] - obs2[i], obs2[i] - qraw[hi]))
            return out
        r = 1.0 / max(self._nb_disp, 1e-4); p = r / (r + mu)
        out = {q: np.clip(np.asarray(nbinom.ppf(q, r, p), float), 0.0, cap) for q in levels}
        if not self._conf_scores:
            return out
        obs = np.asarray(y_observed, float).ravel() if y_observed is not None else None
        for (lo, hi), E0 in self._conf_scores.items():
            if lo not in out or hi not in out:
                continue
            qlo = np.asarray(out[lo], float).copy(); qhi = np.asarray(out[hi], float).copy()
            beta = self._conf_beta[(lo, hi)]
            if obs is None or not self.adaptive_conf:
                Q = self._conf[(lo, hi)]                      # static 고정 Q
                out[lo] = np.clip(qlo - Q, 0.0, cap); out[hi] = np.clip(qhi + Q, 0.0, cap)
            elif self._use_asym:                             # ★비대칭 PID(auto 결정): 하단/상단 독립 Q (정점 우편향)
                target_side = (1.0 - (hi - lo)) / 2.0
                nlo, nhi = self._pid_conformal_adjust_asym(
                    qlo, qhi, obs, self._conf_lo[(lo, hi)], self._conf_hi[(lo, hi)],
                    target_side, window=self.conf_window, ki=self.pid_ki, cap=cap)
                out[lo] = nlo; out[hi] = nhi
            else:                                            # ★adaptive Conformal PID (P+I 대칭)
                target = 1.0 - (hi - lo)                      # 목표 miscoverage (e.g. 0.05)
                nlo, nhi = self._pid_conformal_adjust(
                    qlo, qhi, obs, E0, beta, target,
                    window=self.conf_window, ki=self.pid_ki, cap=cap)
                out[lo] = nlo; out[hi] = nhi
        return out


    def predict_adaptive(self, X_test, y_observed, bias_decay: float = 0.6,
                         window: int = 4, consistency_thr: float = 0.6,
                         shock_mult: float = 2.0, **kwargs) -> np.ndarray:
        """★동적 online 시간/충격 적응 (LoRA/메타 형식, Seoul 내) — *일관된* regime 이동에만 bias 적용.

        교훈(앞 시도): 항상-켜진 bias는 노이즈 추종(평상시 악화) + rolling base와 이중계산(충격 overshoot).
        → **동적 게이팅**: 최근 잔차가 *부호-일관*(지속적 under/over = 진짜 regime 이동/충격)일 때만 bias
        적용, 노이즈(부호 혼재)면 끔. consistency∈[0,1]로 연속 게이트. TabPFN(meta-learned)+이 동적 bias
        = "메타학습 형식". rolling base가 이미 적응하므로 *잔여 지속 편향*만 보정 → 이중계산 회피.

        Args: bias_decay=EWMA 감쇠. window=부호-일관성 판정 최근 잔차 수.
        Returns: (n,) 적응 예측.
        """
        preds = np.asarray(self.predict(X_test, y_observed=y_observed), dtype=float)
        obs = np.asarray(y_observed, dtype=float).ravel()
        cap = 2.0 * self._y_max if self._y_max > 0 else np.inf
        out = preds.copy(); bias = 0.0; recent = []
        for i in range(len(preds)):
            if len(recent) >= 2:
                signs = np.sign(recent[-window:])
                consistency = float(abs(np.mean(signs)))        # 부호 일관성 (지속 이동 = regime/충격)
            else:
                consistency = 0.0
            cw = consistency if consistency >= consistency_thr else 0.0   # ★동적 게이트(부호-일관 시만)
            out[i] = float(np.clip(preds[i] + cw * bias, 0.0, cap))
            resid = obs[i] - preds[i]
            recent.append(resid)
            bias = bias_decay * bias + (1.0 - bias_decay) * resid
        # 주의: magnitude/상대-급증 게이트는 실측서 더 나빴음(scale 비대표·sustained surge 놓침) → 부호-일관만.
        return np.clip(out, 0.0, cap)

    def predict_multi(self, H: int, context_y=None, seasonal_period: int = 52,
                      blend_start: int = 4) -> np.ndarray:
        """★Track A: h=1..H 다중horizon 예측 — TiRex multi-step(단기 강) + seasonal 장기 blend.

        실측 근거: TiRex multi-step은 h=1~4 최강이나 h≥5는 seasonal-naive(작년 같은 주)가 추월.
        → blend_start 이후 seasonal로 점진 전환 = 전 horizon 강건. context_y 끝점서 forecast.

        Args: H=horizon 수. context_y=히스토리(None이면 train). seasonal_period=계절 주기(주).
              blend_start=TiRex 순수 유지 horizon(이후 seasonal 가중↑).
        Returns: (H,) 예측 (h=1..H), nonneg·cap.
        """
        import torch
        if not self._fitted:
            raise RuntimeError("FusedEpi: fit() 먼저")        # fail-loud
        y = np.asarray(context_y, dtype=float).ravel() if context_y is not None else self._train_y
        ctx = torch.tensor(y[-self.max_context:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _q, mean = self._tx.forecast(context=ctx, prediction_length=H)
        base = np.asarray(mean, dtype=float).ravel()[:H]
        cap = 2.0 * self._y_max if self._y_max > 0 else np.inf
        out = np.empty(H, dtype=float)
        for h in range(1, H + 1):
            idx = len(y) - seasonal_period + (h - 1)         # 작년 같은 주 (seasonal-naive)
            seas = y[idx] if 0 <= idx < len(y) else base[h - 1]
            w = 1.0 if h <= blend_start else max(0.0, 1.0 - (h - blend_start) / float(blend_start))
            out[h - 1] = w * base[h - 1] + (1.0 - w) * seas  # 단기=TiRex, 장기→seasonal
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, cap)


def register_fused_epi() -> None:
    """명시/자동 등록. 다른 모델과 동일하게 import 시 자동 호출(아래) — 빌드 완료 후 lineup 편입."""
    try:
        REGISTRY.register(FusedEpiForecaster)
        log.info("[fused_epi] FusedEpiForecaster 등록됨")
    except Exception as e:
        log.debug(f"[fused_epi] 등록 skip: {e}")


# ── import 시 자동 등록 (TabPFN/NegBin 등과 동일 패턴) — 검증·스모크 완료(2026-06-23) 후 활성 ──
register_fused_epi()
