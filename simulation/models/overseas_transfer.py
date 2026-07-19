"""
simulation/models/overseas_transfer.py
======================================
해외 ILI 데이터 기반 전이학습: LSTM 인코더 사전학습.

목표:
  1. WHO FluNet, CDC ILINet 등 여러 국가 ILI 데이터 수집
  2. 멀티국가 시계열을 함께 학습하는 LSTM 인코더 구축
  3. 서울 데이터로 파인튜닝 → 분포편이 대응

구조:
  - OverseasDataBuilder: DB에서 해외 ILI 데이터 로드
    (overseas_ili 테이블 기반, polars 사용)

  - OverseasLSTMPretrainer: 멀티국가 LSTM 사전학습
    - 인코더: 각 국가 시계열의 공유 특징 추출
    - 예측헤드: 국가별 다음 1주 예측
    - 다국가 손실 합산으로 강건한 인코더 학습

  - OverseasTransferForecaster: BaseForecaster 래퍼
    - 사전학습 LSTM 인코더 로드
    - 서울 데이터로 예측헤드 파인튜닝
    - 최종 예측 반환

특징:
  - Polars 기반 데이터 로드 (프로젝트 표준)
  - GPU 선택사항 (CPU 호환)
  - 멀티국가 앙상블 효과: 분포편이 완화
  - 신뢰도 점수: 국가별 일치도 기반
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Dict, List, Tuple

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

# G-279 병목 fix: 사전학습된 (frozen) encoder 를 (countries, 하이퍼) 키로 캐시 → 매 fit 마다 100ep
#   재학습 회피(수백×). encoder 는 fine-tuning 서 frozen 이라 fold 무관 = 누수 0. env
#   MPH_OVERSEAS_ENCODER_CACHE=1 일 때만 사용(기본 OFF = 라이브 run 안전, 기존 동작 동일).
_ENCODER_CACHE: dict = {}

# 의존성 확인
_HAS_TORCH = False
_HAS_POLARS = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    log.debug("torch not available")

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    log.debug("polars not available, fallback to pandas")


def _check_torch():
    """PyTorch 가용성 확인."""
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch이 설치되지 않았습니다.\n"
            "  uv pip install torch\n"
            "또는\n"
            "  pip install torch"
        )


def _check_polars():
    """Polars 가용성 확인 (pandas 폴백)."""
    if not _HAS_POLARS:
        log.warning("polars not available, using pandas")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# 1. 해외 ILI 데이터 로더
# ═══════════════════════════════════════════════════════════════════

class OverseasDataBuilder:
    """
    DB의 overseas_ili 테이블에서 멀티국가 ILI 데이터 로드.

    Schema (overseas_ili table):
        - source: 데이터 소스 (cdc_ilinet, japan_jihs, who_flunet)
        - country: 국가 코드 (US, JP, KR)
        - year: 연도
        - week_no: 주 번호 (1-53)
        - ili_rate: ILI 환자율 (CDC/JIHS) 또는 NULL (FluNet)
        - specimen_positive/total: 검체 양성수/총수 (FluNet)
        - influenza_a/b: A/B형 양성수 (FluNet)

    주의:
      - 데이터 누락 가능 → 보간(forward fill)
      - 국가별 시계열 길이 상이 → 패딩 또는 자르기
      - Log1p 변환 미리 적용 (모델 입력)
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Parameters:
            db_path: SQLite DB 경로 (None → SSOT simulation.database.DB_PATH).
        """
        # G-279: 옛 default "data/db/.." 는 SSOT(simulation/data/db/..) 와 불일치 →
        #   해외 데이터 로드 실패 → local LSTM(미학습 encoder) = phantom 의 또다른 원인. SSOT 사용.
        if db_path is None:
            from simulation.database import DB_PATH
            db_path = str(DB_PATH)
        self.db_path = db_path
        self._conn = None
        self._data = {}  # {country: (n_weeks, ili_rate)}

    def load_from_db(self, countries: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        DB에서 해외 ILI 데이터 로드.

        Parameters:
            countries: 로드할 국가 목록 (None이면 전체)

        Returns:
            {country: (n_weeks,)} dict, ILI rate (log1p 변환됨)
        """
        import sqlite3  # row_factory 참조용

        try:
            # : safe_connect 로 일원화
            from simulation.database import safe_connect
            conn = safe_connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # 사용 가능한 국가 조회 (ili_rate가 있는 소스 우선)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT country FROM overseas_ili
                WHERE ili_rate IS NOT NULL
                ORDER BY country
            """)
            available_countries = [row[0] for row in cursor.fetchall()]

            # ili_rate 없는 국가도 FluNet 양성률로 보완
            cursor.execute("""
                SELECT DISTINCT country FROM overseas_ili
                WHERE ili_rate IS NULL AND specimen_positive IS NOT NULL
                  AND specimen_total IS NOT NULL AND specimen_total > 0
                ORDER BY country
            """)
            flunet_only = [row[0] for row in cursor.fetchall()
                           if row[0] not in available_countries]
            available_countries.extend(flunet_only)

            if not available_countries:
                log.warning("overseas_ili 테이블 비어있음")
                conn.close()
                return {}

            # 필터링
            countries_to_load = (
                countries if countries
                else available_countries
            )
            countries_to_load = [c for c in countries_to_load if c in available_countries]

            log.info(f"  [OverseasDataBuilder] 로드 중: {countries_to_load}")

            # 국가별 데이터 로드
            for country in countries_to_load:
                # 주별 최적 값 추출 (중복 source 제거: ILI rate 우선, 없으면 양성률)
                cursor.execute(
                    """
                    SELECT year, week_no, ili_rate, specimen_positive, specimen_total
                    FROM overseas_ili
                    WHERE country = ?
                    ORDER BY year ASC, week_no ASC
                    """,
                    (country,),
                )
                rows_raw = cursor.fetchall()

                if not rows_raw:
                    log.warning(f"  {country}: 데이터 없음")
                    continue

                # 주별 중복 제거: (year, week_no) → 최적 rate
                week_rates = {}  # (year, week) → best rate
                for yr, wk, ili, pos, total in rows_raw:
                    key = (yr, wk)
                    rate = None
                    if ili is not None and ili > 0:
                        rate = float(ili)
                    elif pos is not None and total is not None and total > 0:
                        rate = float(pos) / float(total) * 100.0

                    # 기존 값보다 더 좋은 rate가 있으면 교체 (ili_rate > 양성률)
                    if rate is not None:
                        if key not in week_rates or (ili is not None and ili > 0):
                            week_rates[key] = rate

                if not week_rates:
                    log.warning(f"  {country}: 유효 데이터 없음")
                    continue

                # 정렬된 시계열 생성
                sorted_keys = sorted(week_rates.keys())
                rates = [week_rates[k] for k in sorted_keys]
                ili_rate = np.asarray(rates, dtype=np.float32)

                # bug fix: KR WHO FluNet 은 spec_processed==inf_total 인
                # 관측치가 대부분이라 positivity = 100% flat 이 됨 (upstream 수집 한계).
                # 분산이 사실상 0 인 시계열은 fine-tuning / transfer 에 해로우므로 제외.
                if float(np.std(ili_rate)) < 1e-3:
                    log.warning(
                        f"  {country}: 유효 변동성 없음 "
                        f"(std={float(np.std(ili_rate)):.6f}, mean={float(np.mean(ili_rate)):.3f}) → 제외. "
                        "WHO FluNet 의 positive/total 동일 보고 문제일 가능성 높음."
                    )
                    continue

                # Log1p 변환
                ili_rate_log = np.log1p(np.maximum(ili_rate, 0))

                self._data[country] = ili_rate_log
                log.info(f"  {country}: {len(ili_rate_log)} weeks loaded")

            conn.close()
            return dict(self._data)

        except Exception as e:
            log.error(f"  [OverseasDataBuilder] DB 로드 실패: {e}")
            return {}

    def align_series(
        self,
        target_length: Optional[int] = None,
        pad_mode: str = "forward_fill",
    ) -> Dict[str, np.ndarray]:
        """
        국가별 시계열 길이 정렬 (패딩 또는 자르기).

        Parameters:
            target_length: 목표 길이 (None이면 최대값)
            pad_mode: 'forward_fill' | 'pad_zero' | 'pad_mean'

        Returns:
            정렬된 시계열 dict
        """
        if not self._data:
            log.warning("  데이터 없음, load_from_db() 먼저 호출")
            return {}

        # 목표 길이 결정
        if target_length is None:
            target_length = max(len(s) for s in self._data.values())

        log.info(f"  [OverseasDataBuilder] 정렬 중: target_length={target_length}")

        aligned = {}
        for country, series in self._data.items():
            if len(series) == target_length:
                aligned[country] = series
            elif len(series) > target_length:
                # 자르기 (최근 데이터 유지)
                aligned[country] = series[-target_length:]
            else:
                # 패딩
                pad_length = target_length - len(series)
                if pad_mode == "forward_fill":
                    pad = np.full(pad_length, series[-1])
                elif pad_mode == "pad_zero":
                    pad = np.zeros(pad_length)
                else:  # pad_mean
                    pad = np.full(pad_length, np.mean(series))
                aligned[country] = np.concatenate([pad, series])

        return aligned

    def create_training_dataset(
        self,
        series_dict: Dict[str, np.ndarray],
        context_window: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        슬라이딩 윈도우로 학습 데이터셋 생성.

        Parameters:
            series_dict: {country: series} dict
            context_window: 컨텍스트 윈도우 크기

        Returns:
            (X, y): (n_samples, context_window, n_countries), (n_samples,)
                각 샘플: n_countries 국가의 과거 context_window 주 → 다음 1주 예측
        """
        countries = sorted(series_dict.keys())
        n_countries = len(countries)

        X_list = []
        y_list = []

        # 각 국가 시계열의 최소 길이
        min_length = min(len(series_dict[c]) for c in countries)

        if min_length <= context_window:
            raise ValueError(
                f"시계열 길이 {min_length} < context_window {context_window}"
            )

        # G-279 (2026-06-16, 3자 감사): 진짜 transfer — 국가별 **단일 ILI 시리즈**(input_dim=1)로
        #   샘플 생성(옛 버전은 (ctx, n_countries) 멀티채널 → encoder 가 Seoul 1-series 와 차원
        #   비호환 → phantom 우회의 근본원인). 각 샘플 = 한 국가의 과거 ctx주 → 다음 1주.
        #   encoder 가 country-agnostic 독감 동역학을 학습 → Seoul ILI 시퀀스에 그대로 전이.
        for country in countries:
            series = series_dict[country]
            for t in range(context_window, len(series)):
                w = series[t - context_window:t].astype(np.float32)
                # per-window z-score: scale-invariant 동역학(국가별 스케일 차 제거) → Seoul 전이 가능
                w = (w - w.mean()) / (w.std() + 1e-6)
                X_list.append(w.reshape(context_window, 1))
                y_list.append(float(series[t]))

        X = np.array(X_list, dtype=np.float32)  # (n_total, context_window, 1)
        y = np.array(y_list, dtype=np.float32)  # (n_total,)

        log.info(f"  [OverseasDataBuilder] 단일-시리즈 데이터셋: X={X.shape}, y={y.shape} "
                 f"({n_countries}개국 windows)")
        return X, y


# ═══════════════════════════════════════════════════════════════════
# 2. 멀티국가 LSTM 사전학습기
# ═══════════════════════════════════════════════════════════════════

class OverseasLSTMPretrainer:
    """
    멀티국가 ILI 시계열로 LSTM 인코더 사전학습.

    구조:
      - Encoder: LSTM (n_countries를 input으로)
      - Decoder: 선형층 (국가 구분 없는 일반적 예측)
      - Loss: 모든 국가의 MSE 합산

    사전학습 후 인코더를 FoundationModelTransferLearner에 이전.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Parameters:
            hidden_dim: LSTM 숨김 차원
            num_layers: LSTM 레이어 수
            dropout: Dropout 비율
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self._model = None
        self._device = None

    def build_model(self, n_countries: int) -> None:
        """LSTM 인코더 + 디코더 모델 구축."""
        _check_torch()
        import torch.nn as nn

        class MultiCountryLSTM(nn.Module):
            def __init__(
                self,
                n_countries: int,
                hidden_dim: int,
                num_layers: int,
                dropout: float,
            ):
                super().__init__()
                self.encoder = nn.LSTM(
                    input_size=n_countries,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.decoder = nn.Linear(hidden_dim, 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                x: (batch, context_window, n_countries)
                """
                lstm_out, (h_n, c_n) = self.encoder(x)
                last_hidden = h_n[-1]  # (batch, hidden_dim)
                last_hidden = self.dropout(last_hidden)
                out = self.decoder(last_hidden)  # (batch, 1)
                return out.squeeze(-1)  # (batch,)

        self._model = MultiCountryLSTM(
            n_countries=n_countries,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> float:
        """
        멀티국가 데이터로 모델 학습.

        Parameters:
            X: (n_samples, context_window, n_countries)
            y: (n_samples,)
            epochs: 에폭
            batch_size: 배치 크기
            lr: 학습률
            patience: Early stopping patience

        Returns:
            best_val_loss (float)
        """
        _check_torch()
        import torch
        import torch.nn as nn

        from simulation.models.base import pick_device
        self._device = pick_device()
        log.info(f"  [OverseasLSTMPretrainer] device={self._device}")

        if self._model is None:
            self.build_model(X.shape[2])

        self._model = self._model.to(self._device)

        # 데이터 준비
        X_t = torch.FloatTensor(X).to(self._device)
        y_t = torch.FloatTensor(y).to(self._device)

        # Validation split
        val_n = max(8, int(len(X) * 0.2))
        X_val, y_val = X_t[-val_n:], y_t[-val_n:]
        X_tr, y_tr = X_t[:-val_n], y_t[:-val_n]

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        no_improve = 0

        log.info(f"  [OverseasLSTMPretrainer] 학습 시작: {len(X_tr)} train, {len(X_val)} val")

        for epoch in range(epochs):
            # Training
            self._model.train()
            train_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_tr), batch_size):
                batch_X = X_tr[i : i + batch_size]
                batch_y = y_tr[i : i + batch_size]

                optimizer.zero_grad()
                pred = self._model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % max(1, epochs // 5) == 0:
                log.debug(
                    f"    epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if no_improve >= patience:
                log.info(f"    Early stopping at epoch {epoch + 1}")
                break

        log.info(f"  [OverseasLSTMPretrainer] 학습 완료: best_val_loss={best_val_loss:.4f}")
        return best_val_loss

    def get_encoder(self):
        """학습된 LSTM 인코더 반환 (서울 데이터 파인튜닝용)."""
        if self._model is None:
            raise RuntimeError("train() 먼저 호출 필수")
        return self._model.encoder


# ═══════════════════════════════════════════════════════════════════
# 3. BaseForecaster 래퍼: 사전학습 → 파인튜닝
# ═══════════════════════════════════════════════════════════════════

class OverseasTransferForecaster(BaseForecaster):
    """
    해외 데이터 사전학습 + 서울 데이터 파인튜닝.

    워크플로우:
      1. OverseasDataBuilder로 해외 ILI 로드 + 정렬
      2. OverseasLSTMPretrainer로 LSTM 인코더 사전학습
      3. 인코더를 BaseForecaster로 래핑 (predict 인터페이스 구현)
      4. 서울 데이터로 파인튜닝 (예측헤드만)

    BaseForecaster 호환성:
      - fit(X_train, y_train)
      - predict(X_test) → np.ndarray
      - save/load 지원
    """

    meta = ModelMeta(
        name="OverseasTransfer",
        category="dl",
        level=18,
        min_data=100,
        requires_gpu=False,
        dependencies=["torch"],
        description="멀티국가 LSTM 사전학습 → 서울 파인튜닝. "
                    "해외 ILI 데이터로 분포편이 대응.",
    )

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs_pretrain: int = 100,
        epochs_finetune: int = 150,
        batch_size: int = 16,
        lr: float = 1e-3,
        db_path: Optional[str] = None,   # G-279: None → SSOT DB_PATH (builder 가 해결)
    ):
        """
        Parameters:
            hidden_dim: LSTM 숨김 차원
            num_layers: LSTM 레이어 수
            dropout: Dropout
            epochs_pretrain: 해외 사전학습 에폭
            epochs_finetune: 서울 파인튜닝 에폭
            batch_size: 배치 크기
            lr: 학습률
            db_path: overseas_ili 테이블이 있는 DB 경로
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs_pretrain = epochs_pretrain
        self.epochs_finetune = epochs_finetune
        self.batch_size = batch_size
        self.lr = lr
        self.db_path = db_path

        self._encoder = None
        self._decoder_head = None
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._device = None
        self._lag_indices: list = []   # G-279: ILI 시퀀스 lag 컬럼 인덱스
        self._y_max = None             # G-279: 출력 cap 기준

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        countries: Optional[List[str]] = None,
        **kwargs,
    ) -> "OverseasTransferForecaster":
        """
        해외 사전학습 → 서울 파인튜닝.

        Parameters:
            X_train: (n_train, n_features) Seoul features
            y_train: (n_train,) Seoul ILI rate (log1p 변환됨)
            countries: 사전학습할 국가 목록 (None이면 전체)
            **kwargs: 추가 인자

        Returns:
            self
        """
        _check_torch()
        import torch
        import torch.nn as nn

        from simulation.models.base import pick_device
        self._device = pick_device()

        log.info(f"  [OverseasTransfer] device={self._device}")

        # 1단계: 해외 데이터 로드 + 사전학습 (G-279 병목: 매 fit 마다 encoder 100ep 재학습 = 수백×).
        #   encoder 는 fine-tuning 서 FROZEN(아래 line ~673 requires_grad=False, 디코더만 학습) →
        #   어떤 fold 에도 학습 안 됨 → 동일 하이퍼/국가면 캐시 재사용 안전(누수 0). 캐시 hit 시
        #   해외DB 로드+align+100ep 사전학습 전부 skip.
        #   ⚠ 라이브 run 안전: env 게이트(기본 OFF). run_pipeline.sh 가 MPH_OVERSEAS_ENCODER_CACHE=1
        #   export → 다음 run 부터 활성. 게이트 OFF 면 기존 동작 100% 동일.
        _use_enc_cache = os.environ.get("MPH_OVERSEAS_ENCODER_CACHE", "0") == "1"
        _enc_key = (tuple(countries) if countries else "__all__",
                    self.hidden_dim, self.num_layers, self.dropout,
                    self.epochs_pretrain, self.batch_size, self.lr)
        if _use_enc_cache and _enc_key in _ENCODER_CACHE:
            self._encoder = _ENCODER_CACHE[_enc_key]
            log.info("  [OverseasTransfer] encoder 캐시 재사용 — 100ep 사전학습 skip "
                     "(frozen=fold 무관, 누수0)")
        else:
            log.info(f"  [OverseasTransfer] 해외 데이터 로드 시작")
            builder = OverseasDataBuilder(db_path=self.db_path)
            overseas_dict = builder.load_from_db(countries=countries)

            if overseas_dict:
                aligned_dict = builder.align_series(target_length=len(y_train))
                X_overseas, y_overseas = builder.create_training_dataset(
                    aligned_dict,
                    context_window=4,   # G-279: Seoul ILI lag1-4 시퀀스와 ctx 정렬
                )

                log.info(f"  [OverseasTransfer] 해외 LSTM 사전학습 시작")
                pretrainer = OverseasLSTMPretrainer(
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                )
                pretrainer.train(
                    X_overseas,
                    y_overseas,
                    epochs=self.epochs_pretrain,
                    batch_size=self.batch_size,
                    lr=self.lr,
                )
                self._encoder = pretrainer.get_encoder()
            else:
                log.warning("  [OverseasTransfer] 해외 데이터 없음, 로컬 LSTM 사용")
                pretrainer = OverseasLSTMPretrainer(
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                )
                pretrainer.build_model(n_countries=1)
                self._encoder = pretrainer.get_encoder()
            if _use_enc_cache:
                _ENCODER_CACHE[_enc_key] = self._encoder

        # 2단계: 서울 데이터 정규화 — Sprint 1.5 R5 (2026-05-26) setup_xy_scalers
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_train_scaled, y_train_scaled = (
            setup_xy_scalers(X_train, y_train)
        )

        # 3단계: 디코더헤드 파인튜닝 — G-279: lag features 로 ILI 시퀀스 구성 → encoder 전이
        self._y_max = float(np.max(y_train))   # 출력 cap 기준 (누수 0)
        _fnames = kwargs.get("feature_names")
        lag_indices: list = []
        if _fnames is not None:
            _n2i = {n: i for i, n in enumerate(_fnames)}
            for _L in (4, 3, 2, 1):   # oldest→newest
                _idx = _n2i.get(f"ili_rate_lag{_L}")
                if _idx is not None and _idx < X_train.shape[1]:
                    lag_indices.append(int(_idx))
        self._lag_indices = lag_indices
        if not lag_indices:
            log.warning("  [OverseasTransfer] ili_rate_lag1-4 미발견 → transfer 생략(feature-only)")
        log.info(f"  [OverseasTransfer] 서울 파인튜닝 시작 (transfer lags={lag_indices})")
        self._build_finetuning_model(X_train.shape[1], lag_indices)

        self._model = self._model.to(self._device)

        # Freeze encoder, 디코더만 학습
        for param in self._encoder.parameters():
            param.requires_grad = False

        X_t = torch.FloatTensor(X_train_scaled).to(self._device)
        y_t = torch.FloatTensor(y_train_scaled).to(self._device)

        val_n = max(8, int(len(X_train) * 0.2))
        X_val, y_val = X_t[-val_n:], y_t[-val_n:]
        X_tr, y_tr = X_t[:-val_n], y_t[:-val_n]

        optimizer = torch.optim.Adam(self._decoder_head.parameters(), lr=self.lr)
        criterion = nn.MSELoss()  # G-218: huber 영구 제거 (huber-loss-banned-20260520)

        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(self.epochs_finetune):
            self._model.train()
            optimizer.zero_grad()
            pred = self._model(X_tr)
            loss = criterion(pred, y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % max(1, self.epochs_finetune // 5) == 0:
                log.debug(f"    epoch {epoch + 1}/{self.epochs_finetune}: val_loss={val_loss:.4f}")

            if no_improve >= 15:
                log.debug(f"    Early stopping at epoch {epoch + 1}")
                break

        self._fitted = True
        log.info(f"  [OverseasTransfer] fit 완료")
        return self

    def _build_finetuning_model(self, n_features: int, lag_indices: list) -> None:
        """G-279 (2026-06-16, 3자 감사): 진짜 cross-country transfer.

        frozen pretrained encoder(LSTM, input_size=1)가 Seoul ILI lag 시퀀스를 받아
        해외-학습 독감 동역학 embedding 을 산출 → Seoul features 와 concat → head.
        옛 버전은 encoder 를 forward 에 안 써 phantom(encoder 무기여)이었음 — 이제 실배선.
        lag_indices = [lag_ctx,...,lag1] 컬럼 인덱스(oldest→newest); 비면 transfer 생략(feature-only).
        """
        _check_torch()
        import torch
        import torch.nn as nn

        _encoder = self._encoder      # frozen LSTM(input_size=1) — get_encoder() 반환
        _hidden = self.hidden_dim
        _drop = self.dropout

        class TransferModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _encoder
                self.register_buffer("lag_idx", torch.tensor(list(lag_indices), dtype=torch.long))
                head_in = n_features + (_hidden if len(lag_indices) > 0 else 0)
                self.head = nn.Sequential(
                    nn.Linear(head_in, _hidden), nn.ReLU(), nn.Dropout(_drop),
                    nn.Linear(_hidden, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, n_features)
                if self.lag_idx.numel() >= 1:
                    seq = x.index_select(1, self.lag_idx)                # (batch, ctx) ILI lag window
                    # per-window z-score (해외 pretrain 과 동일 정규화 → scale-invariant 전이)
                    seq = (seq - seq.mean(dim=1, keepdim=True)) / (seq.std(dim=1, keepdim=True) + 1e-6)
                    _, (h_n, _c) = self.encoder(seq.unsqueeze(-1))       # (batch, ctx, 1)
                    emb = h_n[-1]                                        # (batch, hidden) 전이 표현
                    z = torch.cat([x, emb], dim=1)
                else:
                    z = x
                return self.head(z).squeeze(-1)

        self._decoder_head = TransferModel()
        self._model = self._decoder_head

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        예측 (정규화 역변환 포함).

        Parameters:
            X_test: (n_test, n_features)
            **kwargs: 추가 인자

        Returns:
            (n_test,) 점 예측
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("OverseasTransfer: fit() 먼저 호출 필수")

        _check_torch()
        import torch

        X_scaled = self._scaler_X.transform(X_test)
        X_t = torch.FloatTensor(X_scaled).to(self._device)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_t).cpu().numpy()  # (n_test,)

        # 역정규화
        pred = self._scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
        # G-279: 출력 cap (옛 버전은 cap 無 → test pred 669, rolling r2 −107 폭발).
        #   transform 공간이라도 2×y_max(=train 파생) clip 으로 발산 차단.
        # G-298 (2026-06-17): UPPER cap만 — 하한 0.0 floor 제거. _y_max=max(transformed y_train)
        #   이라 pred 도 transformed 공간 → median-centered transform(mcmc_robust/laplace)서 하한
        #   0.0 이 sub-median 예측을 floor(트리와 동일 버그). ILI≥0 는 phase-13 inverse 직후 원공간
        #   도메인 floor(_refit_and_predict_*)가 담당. nonneg 보존(NaN/inf만 sanitize).
        from simulation.models.base import sanitize_predictions
        _ymax = getattr(self, "_y_max", None)
        if _ymax is not None and _ymax > 0:
            pred = np.minimum(pred, 2.0 * float(_ymax))
        pred = sanitize_predictions(pred)
        return np.asarray(pred, dtype=np.float32)


# 2026-05-26 prune revision (user explicit re-include): OverseasTransfer kept.
# User wants this model active for transfer-learning experiments.
# 2026-06-15 (per-model 감사): Huber loss 는 :664 에서 MSELoss 로 이미 정정됨(영구금지 정합).
# 2026-06-16 (G-279, 3자 감사): phantom transfer 해소 — encoder 를 forward 에 실배선.
#   ① create_training_dataset 을 국가별 단일 ILI 시리즈(input_size=1, per-window z-score)로 →
#      encoder 가 country-agnostic 독감 동역학 학습. ② TransferModel.forward 가 Seoul ILI lag1-4
#      시퀀스를 frozen encoder 에 통과 → embedding 을 features 와 concat → head. ③ 출력 2×y_max cap.
#   이제 encoder 가 예측에 실제 기여(전이 실재) + 폭발 차단.
if _HAS_TORCH:
    REGISTRY.register(OverseasTransferForecaster)
    log.info("[overseas_transfer] OverseasTransferForecaster 등록됨 (re-include 2026-05-26)")
else:
    log.warning("[overseas_transfer] torch 없음 → OverseasTransferForecaster 스킵")
