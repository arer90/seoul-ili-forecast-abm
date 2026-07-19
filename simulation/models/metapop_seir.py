"""
simulation/models/metapop_seir.py
=================================
서울시 25개 구(자치구)를 노드로 하는 메타개체군 SEIR 모델.

[설계 원칙]
  1. CommutingMatrix: DB의 시간별 인구 데이터로부터 통근 흐름 행렬 구성
  2. MetapopSEIRSimulator: 25개 구의 연결된 SEIR 미분 방정식 풀이
  3. MetapopSEIRForecaster: BaseForecaster 상속, 훈련 데이터로 β 보정 후 예측

[수학적 모델]
  각 구 k에 대해:
    dS_k/dt = -λ_k·S_k + ω·R_k
    dE_k/dt = λ_k·S_k - σ·E_k
    dI_k/dt = σ·E_k - γ·I_k
    dR_k/dt = γ·(1-cfr)·I_k - ω·R_k
    dD_k/dt = γ·cfr·I_k

  여기서 λ_k = β·Σ_j (c_{j→k}·I_j / N_j)
  c_{j→k}: j구에서 k구로의 일일 통근 비율

[통근 행렬 구성]
  - 야간(0~6시): 거주지 인구 ≈ 야간 인구
  - 낮(9~18시): 근무지 인구 ≈ 낮 인구
  - 흐름: 낮과 야간의 인구 차이로 추정
  - 중력 모형 폴백: 거리 데이터 없을 경우
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY
from simulation.disease_params import DiseaseParams, get_disease_params

log = logging.getLogger(__name__)

# ── 서울시 25개 구 인구 (2024년 기준, 대략값) ──
SEOUL_GU_POPULATION = {
    "종로구": 144_000,
    "중구": 120_000,
    "용산구": 220_000,
    "성동구": 290_000,
    "광진구": 340_000,
    "동대문구": 340_000,
    "중랑구": 380_000,
    "성북구": 420_000,
    "강북구": 290_000,
    "도봉구": 310_000,
    "노원구": 490_000,
    "은평구": 460_000,
    "서대문구": 300_000,
    "마포구": 360_000,
    "양천구": 430_000,
    "강서구": 560_000,
    "구로구": 390_000,
    "금천구": 230_000,
    "영등포구": 370_000,
    "동작구": 380_000,
    "관악구": 480_000,
    "서초구": 410_000,
    "강남구": 520_000,
    "송파구": 650_000,
    "강동구": 430_000,
}

SEOUL_GU_ORDERED = list(SEOUL_GU_POPULATION.keys())


# ═══════════════════════════════════════════════════════════════════════════
# 통근 행렬 생성 (CommutingMatrix)
# ═══════════════════════════════════════════════════════════════════════════

class CommutingMatrix:
    """
    DB의 시간별 인구 데이터로부터 통근 흐름 행렬을 구성.

    로직:
      - 야간(0~6시) 인구 ≈ 거주지 인구
      - 낮(9AM~6PM) 인구 ≈ 일터 인구
      - 흐름: 낮 인구와 야간 인구의 차이로 추정
      - Fallback: 중력 모형 (거리 데이터 불가할 때)
    """

    def __init__(self, db_path: str = "data/db/epi_real_seoul.db"):
        """
        Parameters
        ----------
        db_path : str
            SQLite DB 경로
        """
        self.db_path = Path(db_path)
        self.districts = SEOUL_GU_ORDERED
        self.n_districts = len(self.districts)
        self.populations = np.array([
            SEOUL_GU_POPULATION[gu] for gu in self.districts
        ], dtype=float)

        self._matrix = None
        self._build_matrix()

    def _build_matrix(self) -> None:
        """통근 행렬 구성 (DB 또는 중력 모형)."""
        if self.db_path.exists():
            try:
                self._build_from_db()
                log.info(f"[CommutingMatrix] DB에서 통근 행렬 구성 완료")
                return
            except Exception as e:
                log.warning(f"[CommutingMatrix] DB 읽기 실패: {e}. 중력 모형 사용.")

        # Fallback: 중력 모형
        self._build_gravity_model()
        log.info(f"[CommutingMatrix] 중력 모형으로 통근 행렬 구성 완료")

    def _build_from_db(self) -> None:
        """
        DB의 daily_population_gu_hourly로부터 통근 행렬 추정.

        논리:
          1. 각 구별 야간(00:00~06:00) 평균 인구 → 거주지 인구
          2. 각 구별 낮(09:00~18:00) 평균 인구 → 일터 인구
          3. 각 구의 낮-야간 인구 차이 → 구 밖 통근자 수
          4. 전체 통근자 = Σ(낮-야간) / 2 (왕복)
          5. 구 j→k 통근자 = j의 초과 인구 × (k의 초과 수요 / 전체 초과)
        """
        from simulation.database import safe_connect  # : quick_check + WAL
        conn = safe_connect(str(self.db_path))
        try:
            # 시간별 인구 데이터 조회
            query = """
                SELECT gu_nm, hour, AVG(tot_pop) as avg_pop
                FROM daily_population_gu_hourly
                WHERE gu_nm IN ({})
                GROUP BY gu_nm, hour
            """.format(
                ",".join(["?" for _ in self.districts])
            )

            df = pd.read_sql_query(query, conn, params=self.districts)

            if df.empty:
                log.warning("[CommutingMatrix] 인구 데이터 없음, 중력 모형 사용")
                self._build_gravity_model()
                return

            # 야간(0~6) vs 낮(9~18) 인구
            night_pop = df[df["hour"].isin([0, 1, 2, 3, 4, 5, 6])] \
                .groupby("gu_nm")["avg_pop"].mean()
            day_pop = df[df["hour"].isin([9, 10, 11, 12, 13, 14, 15, 16, 17, 18])] \
                .groupby("gu_nm")["avg_pop"].mean()

            # 구별 인구 차이 (양수 = 유입, 음수 = 유출)
            delta = {}
            for gu in self.districts:
                n_night = night_pop.get(gu, 0)
                n_day = day_pop.get(gu, 0)
                delta[gu] = max(0, n_day - n_night)  # 순 유입

            # 통근 행렬 초기화
            self._matrix = np.zeros((self.n_districts, self.n_districts))

            # 단순화: j→k 흐름 = j의 유출 × (k의 유입 수요 / 전체 유입 수요)
            total_inflow = sum(delta.values())
            if total_inflow > 0:
                for j, gu_j in enumerate(self.districts):
                    outflow = -min(0, delta[gu_j]) if delta[gu_j] < 0 else 0

                    # j에서 외부로 나가는 인구
                    if outflow > 0:
                        for k, gu_k in enumerate(self.districts):
                            if j != k:
                                # k의 유입 수요 비율로 배분
                                inflow_k = delta[gu_k]
                                if inflow_k > 0:
                                    self._matrix[j, k] = outflow * (inflow_k / total_inflow)

            # 대각 원소 0 (자기 자신과의 통근 불가)
            np.fill_diagonal(self._matrix, 0)

            # 행 정규화 (각 구의 총 통근자가 1 이하)
            row_sums = self._matrix.sum(axis=1)
            for i in range(self.n_districts):
                if row_sums[i] > 0:
                    self._matrix[i, :] /= self.populations[i]

        finally:
            conn.close()

    def _build_gravity_model(self) -> None:
        """
        중력 모형: Flow_ij ∝ P_i * P_j / d_ij^2

        거리 데이터 없으므로, 인접도 기반 완화된 모형 사용.
        """
        # 지리적 인접도 근사 (인구 기반)
        self._matrix = np.zeros((self.n_districts, self.n_districts))

        for i in range(self.n_districts):
            for j in range(self.n_districts):
                if i != j:
                    # 단순 중력: P_i * P_j
                    self._matrix[i, j] = self.populations[i] * self.populations[j]

        # 행 정규화: 각 구의 통근 비율이 1% ~ 5% 수준
        row_sums = self._matrix.sum(axis=1)
        self._matrix = self._matrix / (row_sums[:, np.newaxis] + 1e-10)
        self._matrix *= 0.03  # 3% 통근율 (현실적)
        np.fill_diagonal(self._matrix, 0)

    def get_matrix(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray : (25, 25) 통근 흐름 행렬
            [i, j] = i구에서 j구로의 일일 통근 비율
        """
        return self._matrix

    def get_district_names(self) -> list[str]:
        """Returns 순서대로 정렬된 25개 구 이름."""
        return self.districts


# ═══════════════════════════════════════════════════════════════════════════
# 메타개체군 SEIR 시뮬레이터
# ═══════════════════════════════════════════════════════════════════════════

class MetapopSEIRSimulator:
    """
    25개 자치구를 노드로 하는 메타개체군 SEIR 시뮬레이션.

    각 구 k: S_k, E_k, I_k, R_k, D_k
    교차 전파: λ_k = β × Σ_j (c_{j→k} × I_j / N_j_eff)
    """

    def __init__(
        self,
        disease: DiseaseParams,
        commuting_matrix: np.ndarray,
        district_names: list[str],
        district_populations: dict[str, int],
        beta: float = None,
    ):
        """
        Parameters
        ----------
        disease : DiseaseParams
            질환 파라미터
        commuting_matrix : np.ndarray
            (25, 25) 통근 흐름 행렬
        district_names : list[str]
            순서대로 정렬된 구 이름
        district_populations : dict
            구별 인구
        beta : float, optional
            전파율. None이면 disease.beta 사용
        """
        self.disease = disease
        self.C = commuting_matrix  # (25, 25)
        self.districts = district_names
        self.n_districts = len(district_names)
        self.N = np.array([district_populations[gu] for gu in district_names], dtype=float)
        self.beta = beta if beta is not None else disease.beta

        # 질환 파라미터
        self.sigma = disease.sigma
        self.gamma = disease.gamma
        self.cfr = disease.cfr

    def _deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        메타개체군 SEIR 미분 방정식.

        y = [S_1, E_1, I_1, R_1, D_1, S_2, E_2, ..., D_25]
        형태: (5*25,) = (125,)
        """
        # 구획 추출
        compartments = y.reshape((self.n_districts, 5))
        S = compartments[:, 0]
        E = compartments[:, 1]
        I = compartments[:, 2]
        R = compartments[:, 3]
        D = compartments[:, 4]

        N_eff = S + E + I + R
        N_eff = np.maximum(N_eff, 1e-6)  # 0 방지

        # 교차 감염: λ_k = β × Σ_j (c_{j→k} × I_j / N_j_eff)
        # c_{j→k} = C[j, k] (j에서 k로의 통근 비율)
        I_normalized = I / N_eff
        lambda_vec = self.beta * (self.C @ I_normalized)

        # 각 구별 미분
        dS = -lambda_vec * S
        dE = lambda_vec * S - self.sigma * E
        dI = self.sigma * E - self.gamma * I
        dR = self.gamma * (1 - self.cfr) * I
        dD = self.gamma * self.cfr * I

        # 평탄화
        dydt = np.zeros(5 * self.n_districts)
        dydt[0::5] = dS
        dydt[1::5] = dE
        dydt[2::5] = dI
        dydt[3::5] = dR
        dydt[4::5] = dD

        return dydt

    def run(
        self,
        initial_conditions: dict[str, dict],
        days: int = 365,
    ) -> pd.DataFrame:
        """
        메타개체군 SEIR 시뮬레이션 실행.

        Parameters
        ----------
        initial_conditions : dict
            {
                "district_name": {"S": ..., "E": ..., "I": ..., "R": ..., "D": ...},
                ...
            }
        days : int
            시뮬레이션 기간 (일)

        Returns
        -------
        pd.DataFrame
            day, district, S, E, I, R, D, daily_cases, cumulative_cases, I_total
        """
        # 초기 조건 벡터화
        y0 = np.zeros(5 * self.n_districts)
        for k, gu in enumerate(self.districts):
            ic = initial_conditions.get(gu, {
                "S": self.N[k], "E": 0, "I": 0, "R": 0, "D": 0
            })
            y0[5*k:5*k+5] = [
                ic.get("S", self.N[k]),
                ic.get("E", 0),
                ic.get("I", 0),
                ic.get("R", 0),
                ic.get("D", 0),
            ]

        # ODE 풀이
        log.info(f"[MetapopSEIRSimulator] {self.n_districts}개 구, {days}일 시뮬레이션 시작")

        t_span = (0, days)
        t_eval = np.arange(0, days, 1.0)

        sol = solve_ivp(
            self._deriv,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            max_step=1.0,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            log.warning(f"[MetapopSEIRSimulator] 적분 경고: {sol.message}")

        # 결과 DataFrame 구성
        results = []
        for t_idx, t in enumerate(sol.t):
            y_t = sol.y[:, t_idx]
            compartments = y_t.reshape((self.n_districts, 5))

            for k, gu in enumerate(self.districts):
                S, E, I, R, D = compartments[k, :]
                results.append({
                    "day": int(t),
                    "district": gu,
                    "S": max(0, S),
                    "E": max(0, E),
                    "I": max(0, I),
                    "R": max(0, R),
                    "D": max(0, D),
                    "daily_cases": max(0, self.sigma * E),
                    "cumulative_cases": 0,  # 사후 계산
                })

        df = pd.DataFrame(results)

        # 누적 감염 계산
        df["cumulative_cases"] = df.groupby("district")["daily_cases"].cumsum()

        # 전체 감염자
        df["I_total"] = df.groupby("day")["I"].transform("sum")

        log.info(f"[MetapopSEIRSimulator] 완료: "
                 f"총 누적 감염 {df['cumulative_cases'].max():,.0f}")

        return df


# ═══════════════════════════════════════════════════════════════════════════
# 예측기 (BaseForecaster 상속)
# ═══════════════════════════════════════════════════════════════════════════

class MetapopSEIRForecaster(BaseForecaster):
    """
    메타개체군 SEIR을 기반으로 한 ILI 예측기.

    [훈련]
      1. 훈련 데이터에서 시계열 ILI rate 추출
      2. DB에서 구별 감염자 데이터 (가능시) 로드
      3. β 값 보정 (RMSE 최소화)

    [예측]
      1. 메타개체군 SEIR 실행 (보정된 β)
      2. 전체 감염자 I(t) → city-level ILI rate 변환
      3. 예측값 반환
    """

    meta = ModelMeta(
        name="Metapop-SEIR",
        category="physics",
        level=18,
        min_data=50,
        description="25개 자치구 메타개체군 SEIR (통근행렬 기반)",
    )

    def __init__(self):
        super().__init__()
        self.disease = None
        self.commuting_matrix = None
        self.districts = None
        self.populations = None
        self.beta_calibrated = None
        self._db_path = Path("data/db/epi_real_seoul.db")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> MetapopSEIRForecaster:
        """
        훈련 데이터로 메타개체군 SEIR 보정.

        Parameters
        ----------
        X_train : np.ndarray
            (n_samples, n_features) 특성 행렬
        y_train : np.ndarray
            (n_samples,) 목표값 (ILI rate ‰)
        """
        log.info(f"[MetapopSEIRForecaster] 훈련 시작: {len(y_train)} 주간 데이터")

        # 질환 파라미터: 기본 "인플루엔자". MPH_DISEASE 환경변수로 catalog 의
        # 임의 질환(67 catalog / 42 active)을 코드 수정 없이 선택 가능
        # (parameterizable base). 미설정/빈값 = 인플루엔자 → 기본 경로 불변.
        import os
        disease_nm = os.environ.get("MPH_DISEASE", "").strip() or "인플루엔자"
        try:
            self.disease = get_disease_params(disease_nm)
        except Exception:
            log.warning(
                f"[MetapopSEIRForecaster] {disease_nm} 파라미터 로드 실패, "
                f"인플루엔자 기본값 사용"
            )
            from simulation.disease_params import DiseaseParams
            self.disease = DiseaseParams(
                name="인플루엔자",
                R0_mean=1.3,
                latent_period=2.0,
                infectious_period=5.0,
                cfr=0.001,
                seasonal_amplitude=0.3,
                peak_week=5,  # 1월 중순
            )

        # 통근 행렬 생성
        commuting_obj = CommutingMatrix(str(self._db_path))
        self.commuting_matrix = commuting_obj.get_matrix()
        self.districts = commuting_obj.get_district_names()
        self.populations = {gu: SEOUL_GU_POPULATION[gu] for gu in self.districts}

        # β 보정: 훈련 데이터와 시뮬레이션 출력을 비교
        # 간단한 보정: y_train의 평균 → 초기 I 추정 → β 스케일 조정
        y_mean = np.mean(y_train[y_train > 0]) if np.any(y_train > 0) else 0.5

        # 초기 조건: 전체 감염자 = y_mean * 전체인구 / 1000
        # (ILI rate는 ‰ 단위)
        total_pop = sum(self.populations.values())
        initial_I = max(10, y_mean * total_pop / 1000.0)

        # 구별 초기 감염자 (인구 비례)
        initial_conditions = {}
        for gu in self.districts:
            gu_pop = self.populations[gu]
            gu_I = initial_I * (gu_pop / total_pop)
            initial_conditions[gu] = {
                "S": gu_pop - gu_I,
                "E": gu_I * 0.5,
                "I": gu_I,
                "R": 0,
                "D": 0,
            }

        # β 스케일: 훈련 데이터의 크기에 따라 조정
        # G-184 (2026-05-06 epi-advisor 권고): syndromic β caveat
        #   - ILI = "발열≥38°C + 호흡기증상" syndromic indicator (NOT lab-confirmed influenza)
        #   - β here = "syndromic respiratory-pathogen aggregate transmission rate"
        #     NOT influenza-specific R0 (RSV/SARS-CoV-2/hMPV 등 cocirculation 영향 받음)
        #   - paper 보고 시 단일 R0 X — Rt time-series (Cori 2013) 보고 권장
        #   - β scale heuristic (1.5/1.0/0.7) = data-driven adjustment, 학술 표준 X
        #     → 다음 sprint: ILI 분해 (FluNet subtype share × ILI = influenza-only proxy)
        #     또는 명시적 "syndromic SEIR" rename (epi-advisor 권고)
        beta_scale = 1.0
        if y_mean > 2.0:
            beta_scale = 1.5  # 높은 ILI → β 증가
        elif y_mean < 0.5:
            beta_scale = 0.7  # 낮은 ILI → β 감소

        self.beta_calibrated = self.disease.beta * beta_scale

        # BUG-B fix: predict 가 fit 상태를 쓸 수 있도록 persist.
        #   이전엔 predict 가 initial_I=20 (인구 9.4M 대비 2e-6 감염률)
        #   로 하드코딩돼 "총 누적 감염 1" 같은 degenerate 궤적이 나왔다.
        self._initial_I = float(initial_I)
        self._initial_conditions = {
            gu: dict(state) for gu, state in initial_conditions.items()
        }
        self._y_train_tail = float(y_train[-1]) if len(y_train) else 0.0

        log.info(f"[MetapopSEIRForecaster] 보정 완료: β={self.beta_calibrated:.6f} "
                 f"(scale={beta_scale:.2f}), 초기 I={initial_I:,.0f}")

        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        메타개체군 SEIR 시뮬레이션으로 예측.

        Parameters
        ----------
        X_test : np.ndarray
            (n_forecast, n_features)

        Returns
        -------
        np.ndarray
            (n_forecast,) ILI rate 예측값 (‰)
        """
        if not self.is_fitted:
            log.warning("[MetapopSEIRForecaster] 미훈련 상태, 예측 불가")
            return np.zeros(len(X_test))

        n_forecast = len(X_test)
        log.info(f"[MetapopSEIRForecaster] {n_forecast}주 예측 시작")

        # 초기 조건 (훈련 후 설정된 상태)
        total_pop = sum(self.populations.values())

        # BUG-B fix: fit 에서 calibrated 된 initial_conditions 를 재사용.
        #   fallback 도 과거 하드코드 20 이 아니라 y_train 기반 estimate.
        if getattr(self, "_initial_conditions", None):
            initial_conditions = {
                gu: dict(state) for gu, state in self._initial_conditions.items()
            }
            initial_I = getattr(self, "_initial_I", 0.0)
        else:
            # 미훈련 경로는 위에서 차단되지만 안전 fallback
            y_tail = float(getattr(self, "_y_train_tail", 0.5) or 0.5)
            initial_I = max(10.0, y_tail * total_pop / 1000.0)
            initial_conditions = {}
            for gu in self.districts:
                gu_pop = self.populations[gu]
                gu_I = initial_I * (gu_pop / total_pop)
                initial_conditions[gu] = {
                    "S": gu_pop - gu_I,
                    "E": gu_I * 0.5,
                    "I": gu_I,
                    "R": 0,
                    "D": 0,
                }
        log.info(f"[MetapopSEIRForecaster] 초기 조건: I_total={initial_I:,.0f}")

        # 시뮬레이터 생성 및 실행
        days = n_forecast * 7  # 주를 일로 변환
        sim = MetapopSEIRSimulator(
            disease=self.disease,
            commuting_matrix=self.commuting_matrix,
            district_names=self.districts,
            district_populations=self.populations,
            beta=self.beta_calibrated,
        )

        result_df = sim.run(initial_conditions, days=days)

        # 주별로 집계 (일별 → 주별)
        result_df["week"] = result_df["day"] // 7
        weekly = result_df.groupby("week").agg({
            "I_total": "mean",
        }).reset_index()

        # ILI rate로 변환: (I / 전체인구) × 1000
        predictions = (weekly["I_total"].values / total_pop * 1000).clip(min=0)

        # 예측값 수 조정
        if len(predictions) < n_forecast:
            # 부족분은 마지막 값으로 패드
            predictions = np.pad(
                predictions,
                (0, n_forecast - len(predictions)),
                mode="edge"
            )
        else:
            predictions = predictions[:n_forecast]

        log.info(f"[MetapopSEIRForecaster] 예측 완료: mean={predictions.mean():.3f}‰, "
                 f"max={predictions.max():.3f}‰")

        return np.maximum(predictions, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 모델 등록
# ═══════════════════════════════════════════════════════════════════════════

# (2026-04-19): forecasting REGISTRY 에서 제외.
#   근거: smoke_seir_salvage (n_tr=234, n_te=12)
#     default           → R² = -8.42 (mean_pred=0.76 vs true=9.73)
#     β_scale Optuna 10 → R² = -8.49 (best_val=-1.59, β=0.40 하한 고착)
#   구조적 한계: (a) X_train 피처 완전 무시 (y_mean 만 사용),
#   (b) 3-bin β scale 보정 (continuous Optuna 도 회복 안 됨),
#   (c) SEIR ODE 는 계절성 ILI observations 를 모사 못 함 (R0≈1.3 → 빠르게 equilibrium).
#   시뮬레이션 클래스 (MetapopSEIRSimulator) 는 그대로 유지 — import 는 가능.
# REGISTRY.register(MetapopSEIRForecaster)
log.info("[MetapopSEIRForecaster] 는 forecasting REGISTRY 에서 격리됨 (시뮬용)")
