"""Neural surrogate (emulator) for the Seoul SEIR-V-D agent kernel.

느린 agent-based ``run_agent_world`` 시뮬레이션을 빠른 신경망으로 대체(emulate)한다.
역학 파라미터 벡터 ``(beta, sigma, gamma, delta, nu)`` 를 입력으로 받아 25개 구 서울
인구를 굴린 **집계 궤적**(compartment counts × T일)을 한 번의 forward pass 로 근사한다.

핵심 아이디어
-------------
- ``run_agent_world`` 는 매 호출 O(T·N) 의 daily binomial tau-leap 을 돈다(rich-pop 경로).
  파라미터 sweep / 베이지안 보정에서 수천 번 호출하면 병목이 된다.
- surrogate 는 (param → trajectory) 사상을 작은 MLP 로 학습한다. 학습은 한 번,
  이후 예측은 행렬곱 한 번이라 kernel 보다 수십~수백 배 빠르다(실측 배율은
  :func:`surrogate_vs_kernel_speedup` 가 보고).

설계 규율
---------
- **base 코드 미수정**: ``agent_kernel.run_agent_world`` 와 ``synthetic_population.
  generate_population`` 은 import 후 *재사용*만 한다. 편집하지 않는다.
- **deep module**: 공개 인터페이스는 작다(3 함수 + 1 클래스), 구현은 표준화·
  torch/sklearn 이중 백엔드·궤적 평탄화/복원·결정적 샘플링을 캡슐화한다.
- **결정성**: 모든 난수는 ``np.random.default_rng(seed)`` / ``torch.manual_seed`` 로 고정.
- **torch 없으면 sklearn fallback**: 같은 인터페이스로 ``MLPRegressor`` 사용.
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

from simulation.abm.agent_kernel import run_agent_world
from simulation.abm.synthetic_population import generate_population

try:  # primary backend
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path exercised when torch absent
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

__all__ = [
    "PARAM_NAMES",
    "PARAM_BOUNDS",
    "COMPARTMENTS",
    "generate_training_data",
    "ABMSurrogate",
    "surrogate_vs_kernel_speedup",
    "TORCH_AVAILABLE",
]

log = logging.getLogger(__name__)

TORCH_AVAILABLE = _TORCH_AVAILABLE

# 학습되는 역학 파라미터(일 단위 hazard) — run_agent_world 시그니처의 핵심 5종.
PARAM_NAMES: tuple[str, ...] = ("beta", "sigma", "gamma", "delta", "nu")

# 각 파라미터의 uniform 샘플링 범위. 인플루엔자류 ILI 동역학에서 현실적인
# 구간(생성-시간 자릿수, base.py 기본값 부근)으로, 폭주/즉시소멸을 피하면서
# 충분한 동적 다양성을 확보한다.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "beta": (0.15, 0.60),    # 전파 hazard 배수
    "sigma": (0.10, 0.40),   # E->I (잠복 2.5~10일)
    "gamma": (0.07, 0.25),   # I->R (감염 4~14일)
    "delta": (1e-4, 5e-3),   # I->D 압력(severity 로 추가 배율)
    "nu": (0.0, 5e-3),       # S->V 백신 hazard
}

# 집계 궤적으로 emit 하는 compartment(순서 고정 = 출력 채널 k 순서).
COMPARTMENTS: tuple[str, ...] = ("S", "E", "I", "R", "V", "D")

_FIXED_DELTA_FALLBACK = 1e-3


def _sample_params(
    n_samples: int,
    rng: np.random.Generator,
    param_names: Sequence[str],
) -> np.ndarray:
    """각 파라미터를 PARAM_BOUNDS 안에서 독립 uniform 샘플 → (n_samples, d)."""
    cols = []
    for name in param_names:
        lo, hi = PARAM_BOUNDS[name]
        cols.append(rng.uniform(lo, hi, size=n_samples))
    return np.stack(cols, axis=1).astype(np.float64)


def generate_training_data(
    n_samples: int,
    *,
    seed: int = 0,
    N_agents: int = 2000,
    T_days: int = 60,
    param_names: Sequence[str] = PARAM_NAMES,
    compartments: Sequence[str] = COMPARTMENTS,
    pop_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """역학 파라미터를 샘플해 agent kernel 을 굴린 (param, trajectory) 쌍을 만든다.

    각 샘플마다 ``PARAM_BOUNDS`` 안에서 파라미터를 uniform 추출하고, DB-grounded
    서울 25구 합성 인구(``generate_population``) 위에서 ``run_agent_world`` 의 rich-pop
    경로를 돌려 일별 compartment count 궤적을 수집한다. 인구는 한 번만 생성해 모든
    샘플이 공유하므로(파라미터만 변동) param→trajectory 사상이 깨끗하게 정의된다.

    Args:
        n_samples: 생성할 (param, trajectory) 쌍 개수. >= 1.
        seed: 마스터 시드. 파라미터 샘플링·인구 생성·각 kernel 호출의 ``global_seed``
            가 모두 이 시드에서 결정적으로 파생된다. 같은 인자 = 같은 데이터.
        N_agents: 시뮬레이션 에이전트 수(서울 인구 다운스케일). >= 25 권장.
        T_days: 궤적 길이(일). Day 0 = 초기화 상태 포함, 정확히 T_days 행.
        param_names: 학습/샘플할 파라미터 이름(기본 5종). ``PARAM_BOUNDS`` 의 부분집합.
        compartments: 출력 채널로 emit 할 compartment 이름(기본 6종, 순서 고정).
        pop_year: 인구 기준연도(``generate_population`` 로 전달). None=최신.

    Returns:
        ``(params, trajectories)`` 튜플.
          - params: ``(n_samples, len(param_names))`` float64, 샘플한 파라미터 행렬.
          - trajectories: ``(n_samples, T_days, len(compartments))`` float64,
            각 샘플의 일별 compartment count.

    Raises:
        ValueError: ``n_samples < 1``, ``T_days < 1``, ``N_agents < 1`` 이거나
            ``param_names``/``compartments`` 가 알 수 없는 이름을 담을 때.

    Performance: O(n_samples · T_days · N_agents) time, O(n_samples · T_days · k) memory.
    Side effects: ``epi_real_seoul.db`` 를 read-only 로 1회 연다(인구 생성). DB write 없음.
    Caller responsibility: N_agents 는 25 의 배수가 아니어도 되나 구별 분해능을 위해
        충분히 커야 한다(PoC 권장 ~2000).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")
    if T_days < 1:
        raise ValueError(f"T_days must be >= 1; got {T_days}")
    if N_agents < 1:
        raise ValueError(f"N_agents must be >= 1; got {N_agents}")
    unknown_p = [p for p in param_names if p not in PARAM_BOUNDS]
    if unknown_p:
        raise ValueError(f"unknown param_names {unknown_p}; allowed {sorted(PARAM_BOUNDS)}")
    unknown_c = [c for c in compartments if c not in COMPARTMENTS]
    if unknown_c:
        raise ValueError(f"unknown compartments {unknown_c}; allowed {list(COMPARTMENTS)}")

    rng = np.random.default_rng(seed)
    params = _sample_params(n_samples, rng, param_names)

    # 인구는 한 번만 — param→trajectory 사상의 confounder 제거(고정 인구).
    population = generate_population(N_agents, seed=seed, year=pop_year)

    # delta/nu 가 학습 파라미터가 아닐 때 kernel 에 줄 고정값.
    default_kwargs = {
        "beta": float(np.mean([PARAM_BOUNDS["beta"][0], PARAM_BOUNDS["beta"][1]])),
        "sigma": 0.2,
        "gamma": 0.15,
        "delta": _FIXED_DELTA_FALLBACK,
        "nu": 0.0,
    }
    name_to_col = {name: j for j, name in enumerate(param_names)}

    k = len(compartments)
    trajectories = np.empty((n_samples, T_days, k), dtype=np.float64)
    # kernel 호출별 global_seed 를 결정적으로 파생(샘플마다 독립 stochastic 실현).
    call_seeds = rng.integers(1, 2**31 - 1, size=n_samples)

    for i in range(n_samples):
        kwargs = dict(default_kwargs)
        for name in param_names:
            kwargs[name] = float(params[i, name_to_col[name]])
        out = run_agent_world(
            N_agents,
            T_days,
            population=population,
            global_seed=int(call_seeds[i]),
            **kwargs,
        )
        for c_idx, comp in enumerate(compartments):
            trajectories[i, :, c_idx] = out[comp].astype(np.float64)

    return params, trajectories


def _standardize_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """열별 평균/표준편차 반환(std==0 → 1.0 로 치환해 0-나눗셈 회피)."""
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return mean.astype(np.float64), std.astype(np.float64)


class _TorchMLP(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """작은 2-은닉층 MLP. param(d) → flattened trajectory(T*k)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, int]):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):  # noqa: D401 - torch forward
        return self.net(x)


class ABMSurrogate:
    """Agent-kernel 궤적 emulator (param 벡터 → 집계 궤적).

    파라미터·궤적 모두 내부에서 z-score 표준화한 뒤 작은 MLP 를 학습한다.
    torch 가 있으면 :class:`_TorchMLP`(2 은닉층, Adam, early-stop), 없으면 sklearn
    ``MLPRegressor`` 로 *동일 인터페이스*(fit/predict)를 제공한다. 예측은 표준화
    공간에서 수행한 뒤 역표준화해 원래 count 스케일로 되돌린다.

    인터페이스(작음)
    ----------------
    - ``fit(params, trajectories)`` — 학습. self 반환.
    - ``predict(params)`` — ``(n, T, k)`` 궤적 근사 반환.

    Attributes:
        backend: ``"torch"`` 또는 ``"sklearn"``. 어떤 백엔드가 쓰였는지.
        T_days, n_channels: fit 시 추론된 궤적 형상.
    """

    def __init__(
        self,
        *,
        hidden: tuple[int, int] = (128, 64),
        max_epochs: int = 400,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 30,
        n_jobs: int = 2,
        prefer_torch: bool = True,
        seed: int = 0,
    ):
        """surrogate 하이퍼파라미터를 설정한다(학습은 ``fit`` 에서).

        Args:
            hidden: 두 은닉층 폭. 소표본(n~40)에 과적합하지 않게 작게.
            max_epochs: torch 최대 epoch(early-stop 으로 보통 더 일찍 멈춤).
            lr: Adam 학습률(torch 전용).
            batch_size: torch 미니배치 크기.
            patience: held-in val loss 가 개선 안 될 때 멈추기까지 epoch(torch).
            n_jobs: sklearn 백엔드의 병렬도. ENGINEERING_PRINCIPLES.md §2 규율로 <= 2 강제(clip).
            prefer_torch: True 면 torch 사용 가능 시 torch, 아니면 sklearn.
            seed: 가중치 초기화/셔플 결정성 시드.

        Side effects: none (초기화만).
        """
        self.hidden = hidden
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        # ENGINEERING_PRINCIPLES.md §2: n_jobs <= 2 (절대 -1 금지 — CPU deadlock).
        self.n_jobs = max(1, min(2, int(n_jobs)))
        self.seed = int(seed)
        self.backend = "torch" if (prefer_torch and _TORCH_AVAILABLE) else "sklearn"

        self._model = None
        self._x_mean = self._x_std = None
        self._y_mean = self._y_std = None
        self.T_days: int | None = None
        self.n_channels: int | None = None
        self._fitted = False

    # --- 표준화 helpers ------------------------------------------------------
    def _transform_x(self, params: np.ndarray) -> np.ndarray:
        return (params - self._x_mean) / self._x_std

    def _flatten_y(self, trajectories: np.ndarray) -> np.ndarray:
        n = trajectories.shape[0]
        return trajectories.reshape(n, -1)

    def _unflatten_y(self, flat: np.ndarray) -> np.ndarray:
        n = flat.shape[0]
        return flat.reshape(n, self.T_days, self.n_channels)

    # --- 공개 인터페이스 -----------------------------------------------------
    def fit(self, params: np.ndarray, trajectories: np.ndarray) -> "ABMSurrogate":
        """param→trajectory 사상을 학습한다.

        Args:
            params: ``(n, d)`` 파라미터 행렬(generate_training_data 출력).
            trajectories: ``(n, T, k)`` 집계 궤적.

        Returns:
            self (학습 완료, predict 가능).

        Raises:
            ValueError: 형상이 안 맞거나(n 불일치, 차원 오류) n < 2 일 때.

        Performance: torch ~O(max_epochs · n · params), sklearn 은 solver 의존.
            소규모(n~40)면 수 초 내. n_jobs <= 2.
        Side effects: 내부 모델/표준화 통계를 채운다. 전역 상태 변경 없음(시드는
            로컬 generator/Generator 로만 사용).
        """
        params = np.asarray(params, dtype=np.float64)
        trajectories = np.asarray(trajectories, dtype=np.float64)
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (n, d); got {params.shape}")
        if trajectories.ndim != 3:
            raise ValueError(f"trajectories must be 3D (n, T, k); got {trajectories.shape}")
        if params.shape[0] != trajectories.shape[0]:
            raise ValueError(
                f"params/trajectories row mismatch: {params.shape[0]} vs {trajectories.shape[0]}"
            )
        n = params.shape[0]
        if n < 2:
            raise ValueError(f"need at least 2 samples to fit; got {n}")

        self.T_days = int(trajectories.shape[1])
        self.n_channels = int(trajectories.shape[2])

        self._x_mean, self._x_std = _standardize_fit(params)
        y_flat = self._flatten_y(trajectories)
        self._y_mean, self._y_std = _standardize_fit(y_flat)

        x_std = self._transform_x(params)
        y_std = (y_flat - self._y_mean) / self._y_std

        if self.backend == "torch":
            self._fit_torch(x_std, y_std)
        else:
            self._fit_sklearn(x_std, y_std)

        self._fitted = True
        return self

    def predict(self, params: np.ndarray) -> np.ndarray:
        """학습된 surrogate 로 궤적을 근사한다.

        Args:
            params: ``(n, d)`` 또는 ``(d,)`` 파라미터. 단일 벡터도 허용.

        Returns:
            ``(n, T, k)`` 궤적(입력이 1D 면 n=1). 원래 count 스케일.

        Raises:
            RuntimeError: ``fit`` 호출 전.
            ValueError: 파라미터 차원이 학습 차원과 다를 때.

        Performance: O(n) forward pass — kernel 보다 훨씬 빠름(실측은
            surrogate_vs_kernel_speedup).
        Side effects: none. 음수 예측은 count 도메인 제약상 0 으로 clip.
        Caller responsibility: 입력 파라미터는 학습 PARAM_BOUNDS 안일 때 신뢰.
        """
        if not self._fitted:
            raise RuntimeError("ABMSurrogate.predict called before fit")
        params = np.asarray(params, dtype=np.float64)
        single = params.ndim == 1
        if single:
            params = params[None, :]
        if params.ndim != 2 or params.shape[1] != self._x_mean.shape[0]:
            raise ValueError(
                f"params must have {self._x_mean.shape[0]} columns; got {params.shape}"
            )

        x_std = self._transform_x(params)
        if self.backend == "torch":
            self._model.eval()
            with torch.no_grad():
                y_std = self._model(torch.from_numpy(x_std.astype(np.float32))).numpy()
            y_std = y_std.astype(np.float64)
        else:
            y_std = self._model.predict(x_std)
            if y_std.ndim == 1:  # single-output sklearn edge
                y_std = y_std[:, None]

        y_flat = y_std * self._y_std + self._y_mean
        traj = self._unflatten_y(y_flat)
        np.clip(traj, 0.0, None, out=traj)  # counts >= 0
        return traj[0] if single else traj

    def score_r2(self, params: np.ndarray, trajectories: np.ndarray) -> float:
        """held-out 궤적에 대한 전역 R²(모든 시점·채널 평탄화 후).

        Args:
            params: ``(n, d)`` held-out 파라미터.
            trajectories: ``(n, T, k)`` 참 궤적(kernel 산출).

        Returns:
            R² (float). 1.0=완벽, 0.0=평균예측 수준, 음수=평균보다 나쁨.

        Raises:
            RuntimeError: fit 전.
        """
        pred = self.predict(np.asarray(params, dtype=np.float64))
        if pred.ndim == 2:  # single sample
            pred = pred[None, ...]
        truth = np.asarray(trajectories, dtype=np.float64)
        return _r2_score(truth.reshape(-1), pred.reshape(-1))

    # --- 백엔드 구현 ---------------------------------------------------------
    def _fit_torch(self, x_std: np.ndarray, y_std: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        x = torch.from_numpy(x_std.astype(np.float32))
        y = torch.from_numpy(y_std.astype(np.float32))
        n = x.shape[0]

        # 작은 held-in 검증으로 early-stop(과적합 가드). n 이 아주 작으면 전부 train.
        n_val = max(1, int(round(0.2 * n))) if n >= 5 else 0
        perm = torch.randperm(n, generator=gen)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = (x[val_idx], y[val_idx]) if n_val else (x_tr, y_tr)

        model = _TorchMLP(x.shape[1], y.shape[1], self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        bs = min(self.batch_size, x_tr.shape[0])
        best_val = float("inf")
        best_state = None
        bad = 0
        for _epoch in range(self.max_epochs):
            model.train()
            order = torch.randperm(x_tr.shape[0], generator=gen)
            for start in range(0, x_tr.shape[0], bs):
                sel = order[start:start + bs]
                opt.zero_grad()
                pred = model(x_tr[sel])
                loss = loss_fn(pred, y_tr[sel])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(model(x_val), y_val))
            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model

    def _fit_sklearn(self, x_std: np.ndarray, y_std: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor

        # MLPRegressor 는 자체적으로 multi-output 회귀를 지원. early_stopping 으로
        # 과적합 가드(n>=10 일 때만 — 너무 작으면 validation split 불가).
        early = x_std.shape[0] >= 10
        model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            solver="adam",
            learning_rate_init=self.lr,
            max_iter=max(self.max_epochs, 500),
            batch_size=min(self.batch_size, x_std.shape[0]),
            early_stopping=early,
            n_iter_no_change=self.patience,
            random_state=self.seed,
            tol=1e-5,
        )
        model.fit(x_std, y_std)
        self._model = model


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - SS_res/SS_tot. SS_tot==0(상수 타깃) → pred 일치 시 1.0, 아니면 0.0."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def surrogate_vs_kernel_speedup(
    *,
    n_train: int = 40,
    n_test: int = 12,
    seed: int = 0,
    N_agents: int = 2000,
    T_days: int = 60,
    n_timing_repeats: int = 3,
) -> dict[str, float]:
    """surrogate 의 속도 배율과 held-out 정확도(R²)를 실측한다.

    학습 데이터를 ``generate_training_data`` 로 만들고 :class:`ABMSurrogate` 를 학습한
    뒤, **동일한 held-out 파라미터 집합**에 대해 (1) kernel 직접 호출과 (2) surrogate
    예측의 wall-clock 을 비교하고, surrogate 예측의 R² 를 kernel 산출 궤적 대비 잰다.
    학습/평가 파라미터는 시드로 분리해 leak-free(test 파라미터는 학습에 미사용).

    Args:
        n_train: 학습 샘플 수.
        n_test: held-out 평가 샘플 수.
        seed: 마스터 시드. 학습은 ``seed``, 평가는 ``seed + 10_000`` 로 분리.
        N_agents: 에이전트 수.
        T_days: 궤적 길이(일).
        n_timing_repeats: 타이밍 안정화를 위한 반복 횟수(최소 시간 채택).

    Returns:
        dict:
          - ``speedup``: kernel_time / surrogate_time (>1 이면 surrogate 가 빠름).
          - ``r2``: held-out 전역 R² (>0 이면 학습됨).
          - ``kernel_time_s``, ``surrogate_time_s``: 각 wall-clock 최소시간(초).
          - ``backend``: "torch"/"sklearn".
          - ``n_train``, ``n_test``: 사용한 샘플 수.

    Performance: O((n_train + n_test) · T_days · N_agents) — 학습 데이터 생성이 지배.
    Side effects: DB read-only 1회(인구). 표준출력/파일 쓰기 없음.
    Caller responsibility: PoC 규모(n_train~40, N~2000, T~60) 권장.
    """
    # 학습 데이터(시드 seed) — leak-free 위해 평가와 분리된 시드 사용.
    X_tr, Y_tr = generate_training_data(
        n_train, seed=seed, N_agents=N_agents, T_days=T_days
    )
    # 평가 데이터(다른 시드) — 파라미터·실현 모두 학습과 겹치지 않음.
    X_te, Y_te = generate_training_data(
        n_test, seed=seed + 10_000, N_agents=N_agents, T_days=T_days
    )

    surrogate = ABMSurrogate(seed=seed).fit(X_tr, Y_tr)
    r2 = surrogate.score_r2(X_te, Y_te)

    population = generate_population(N_agents, seed=seed)
    name_to_col = {name: j for j, name in enumerate(PARAM_NAMES)}

    def _run_kernel_batch() -> None:
        for i in range(n_test):
            kwargs = {name: float(X_te[i, name_to_col[name]]) for name in PARAM_NAMES}
            run_agent_world(
                N_agents, T_days, population=population, global_seed=1 + i, **kwargs
            )

    # warm-up(JIT/캐시 정상화) 후 최소시간 채택.
    surrogate.predict(X_te)
    _run_kernel_batch()

    kernel_time = min(
        _timeit(_run_kernel_batch) for _ in range(max(1, n_timing_repeats))
    )
    surrogate_time = min(
        _timeit(lambda: surrogate.predict(X_te)) for _ in range(max(1, n_timing_repeats))
    )
    surrogate_time = max(surrogate_time, 1e-9)

    return {
        "speedup": kernel_time / surrogate_time,
        "r2": r2,
        "kernel_time_s": kernel_time,
        "surrogate_time_s": surrogate_time,
        "backend": surrogate.backend,
        "n_train": float(n_train),
        "n_test": float(n_test),
    }


def _timeit(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
