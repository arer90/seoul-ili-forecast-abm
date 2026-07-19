"""
Optuna sampler 선택 + per-model DNN HP suggest helpers.
========================================================

사용자 요청 (2026-04-26):
1. TPE 보다 더 좋은 sampler 옵션 (multivariate / GP / CmaEs)
2. DNN 계열 25+ 모델별 개별 HP space — 모두 다른 search space 필요

=== 1. Sampler 선택 ===

`build_sampler(name, seed, n_startup_trials)` 가 다음 옵션 반환:

  • "tpe"          (default, current) — TPESampler(multivariate=False)
  • "tpe-mv"       (recommended)      — TPESampler(multivariate=True, group=True)
                                          → feature interaction 학습, +5-15% 성능
  • "gp"            (Optuna 4.0+)      — GPSampler (Gaussian Process Bayesian Opt)
                                          → continuous HP 에 최강, sample-efficient
  • "cma"           (CMAES)            — CmaEsSampler
                                          → high-dim continuous, evolutionary
  • "nsga2"         (multi-objective)  — NSGAIISampler
                                          → WIS + MAE 동시 최적화
  • "qmc"           (Quasi-Monte Carlo) — QMCSampler
                                          → 초기 exploration 균등

=== 2. Per-model DNN HP ===

각 DL 모델의 specific HP space 를 함수로 정의:

  suggest_tabular_dnn_hp(trial)        — TabularDNN, TabularDNN-Lite
  suggest_tft_hp(trial)                — Temporal Fusion Transformer
  suggest_patchtst_hp(trial)           — Patch-TST
  suggest_itransformer_hp(trial)       — iTransformer (inverted)
  suggest_timesnet_hp(trial)           — TimesNet (inception)
  suggest_mamba_hp(trial)              — Mamba state-space
  suggest_tcn_hp(trial)                — Temporal Convolutional Net
  suggest_nbeats_hp(trial)             — N-BEATS basis decomposition
  suggest_nhits_hp(trial)              — N-HiTS hierarchical interpolation
  suggest_tide_hp(trial)               — TiDE encoder/decoder
  suggest_deepar_hp(trial)             — DeepAR autoregressive
  suggest_rnn_pf_hp(trial)             — RNN probabilistic
  suggest_pinn_hp(trial)               — Physics-informed NN
  suggest_ge_dnn_hp(trial)             — Graph-Embedded DNN
  suggest_ge_dnn_gat_hp(trial)         — GE-DNN with GATv2 attention
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 1. Sampler factory
# ════════════════════════════════════════════════════════════════════
def build_sampler(name: str = "tpe-mv", *,
                    seed: int = 42,
                    n_startup_trials: int = 10) -> Any:
    """Build an Optuna sampler by name.

    Args:
      name:                "tpe", "tpe-mv", "gp", "cma", "nsga2", "qmc", "random"
      seed:                random seed
      n_startup_trials:    pre-TPE random samples
    """
    import optuna
    n = name.lower()

    if n == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup_trials,
            multivariate=False,
        )
    if n in ("tpe-mv", "tpe_mv", "tpemv", "tpe-multivariate"):
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup_trials,
            multivariate=True,           # ← interactions
            group=True,                  # ← group-wise sampling
            constant_liar=True,          # ← parallel-safe
        )
    if n == "gp":
        try:
            return optuna.samplers.GPSampler(seed=seed,
                                                n_startup_trials=n_startup_trials)
        except AttributeError:
            log.warning("  [sampler] GPSampler requires Optuna 4.0+ — "
                          "falling back to tpe-mv")
            return build_sampler("tpe-mv", seed=seed,
                                   n_startup_trials=n_startup_trials)
    if n in ("cma", "cmaes", "cma-es"):
        try:
            return optuna.samplers.CmaEsSampler(seed=seed,
                                                  n_startup_trials=n_startup_trials)
        except Exception as e:
            log.warning(f"  [sampler] CmaEs unavailable: {e} — fall back to tpe-mv")
            return build_sampler("tpe-mv", seed=seed,
                                   n_startup_trials=n_startup_trials)
    if n in ("nsga2", "nsga-ii"):
        return optuna.samplers.NSGAIISampler(seed=seed)
    if n in ("qmc", "qmc-sobol"):
        try:
            return optuna.samplers.QMCSampler(seed=seed)
        except Exception:
            return optuna.samplers.RandomSampler(seed=seed)
    if n in ("random", "rand"):
        return optuna.samplers.RandomSampler(seed=seed)
    log.warning(f"  [sampler] unknown '{name}' — using tpe-mv")
    return build_sampler("tpe-mv", seed=seed, n_startup_trials=n_startup_trials)


# ════════════════════════════════════════════════════════════════════
# 2. Per-model DNN HP suggesters
# ════════════════════════════════════════════════════════════════════
# 공통 helper — 작은 dataset (n=345) 에 적절히 cap
_DEFAULT_LR_LOW = 1e-5
_DEFAULT_LR_HIGH = 1e-2
_BS_CHOICES = [16, 32, 64, 128]


def suggest_anchor_hp(trial: Any, *, prefix: str = "") -> float:
    """α-blend HP 완전 제거 (2026-05-22 사용자 영구 명시: "완전히 흔적을 없애버려").

    이전: Bühlmann 2018 reference regression alpha_blend HP search.
    현재: 무조건 1.0 반환 (raw, blend effect 0).
    trial 에 어떤 HP 도 등록 안 함 (코드 잔존 제거).
    backward-compat 위해 함수 signature 유지, 내용만 no-op.
    """
    return 1.0


def suggest_tabular_dnn_hp(trial: Any, *,
                              n_train: int = None,
                              n_features: int = None,
                              device: str = None,
                              max_layers: int = 12,
                              max_hidden: int = 9999,
                              min_hidden: int = 16,
                              use_log_hidden: bool = True,
                              param_budget: int = None) -> Dict[str, Any]:
    """TabularDNN / TabularDNN-Lite — comprehensive HP search (사용자 요구 2026-05-12).

    Design philosophy (사용자 명시):
      - max_layers=12 / max_hidden=9999 **default 보존** (가능성 죽이지 X)
      - min_hidden=16 (이전 2 = degenerate; 16 부터 의미있음)
      - **param_budget** soft prune (cap 아님) — OOM/timeout 방지
      - **use_log_hidden** (log default) / False = linear step=8 옵션
      - **All categoricals 보존** (act 11 / opt 8 / norm 7 / init 6 / layer_type 4)
      - Hardware/sample-size aware (env override 가능)

    Search space:
      - **n_layers**:   1 ~ 12 (env: MPH_DNN_MAX_LAYERS)
      - **hidden_units**: 16 ~ 9999 (log/linear 선택, env: MPH_DNN_MAX_HIDDEN)
      - **param_budget**: device default 또는 env: MPH_DNN_PARAM_BUDGET
        · cuda  : 100,000,000 params (24-32GB GPU)
        · mps   :  25,000,000 params (16GB MPS, Adam state ×3 OK)
        · cpu   :   1,000,000 params (CPU 학습 속도 한계)
      - **lr**:         1e-5 ~ 1e-2 (log)
      - **l2_reg (weight_decay)**: 1e-6 ~ 1e-3 (log)
      - **dropout**:    0.1 ~ 0.5
      - **activation**: 11종
      - **optimizer**:  8종 (Adam, AdamW, SGD, RMSprop, Nadam, RAdam, Lamb, Ranger)
      - **norm**:       7종 (none, batch, layer, group, instance, weight, spectral)
      - **init**:       6종 (kaiming_u/n, xavier_u/n, orthogonal, trunc_normal)
      - **layer_type**: 4종 (linear, residual, dense_block, highway)
      - **bias, skip_connection, lr_schedule, gradient_clip**

    OOM 방지 (사용자 최우선):
      param_budget exceed 시 trial.prune() — Optuna 가 다음 trial 빨리 진행.
      Cap (max_layers/max_hidden) 자체는 보존 — env override 로 explicit unblock.
    """
    import os as _os_d
    try:
        from simulation.models.safety import device_str as _device_str
    except Exception:
        _device_str = lambda: "cpu"

    n_train = n_train or GLOBAL.data_split.n_train
    n_features = n_features or GLOBAL.data_split.n_features
    device = device or _device_str()

    # Env override — user explicit cap (가능성 보존, env 만 explicit cap)
    max_layers = int(_os_d.environ.get("MPH_DNN_MAX_LAYERS", str(max_layers)))
    max_hidden = int(_os_d.environ.get("MPH_DNN_MAX_HIDDEN", str(max_hidden)))

    # Param budget — OOM/timeout 방지 (cap 아님), device-aware default
    if param_budget is None:
        param_budget = {
            "cuda": 100_000_000,
            "mps":   25_000_000,
            "cpu":    1_000_000,
        }.get(device, 5_000_000)
    param_budget = int(_os_d.environ.get("MPH_DNN_PARAM_BUDGET", str(param_budget)))
    n_layers = trial.suggest_int("td_n_layers", 1, max_layers)

    # Hidden — log OR linear (사용자 NEW option)
    hidden_dims = []
    for i in range(n_layers):
        if use_log_hidden:
            hidden_dims.append(
                trial.suggest_int(f"td_h{i}", min_hidden, max_hidden, log=True)
            )
        else:
            # Linear scale with step=8 (sensible quantization)
            hidden_dims.append(
                trial.suggest_int(f"td_h{i}", min_hidden, max_hidden, step=8)
            )

    # Param budget soft prune — OOM/timeout 방지 (가능성 보존, env override 가능)
    est_params = (n_features * hidden_dims[0] + hidden_dims[0]
                  + sum(hidden_dims[i] * hidden_dims[i + 1] + hidden_dims[i + 1]
                        for i in range(len(hidden_dims) - 1))
                  + hidden_dims[-1])
    if est_params > param_budget:
        import optuna as _opt_pr
        raise _opt_pr.TrialPruned(
            f"params {est_params:,} > budget {param_budget:,} "
            f"(device={device}, n_train={n_train}). "
            f"Override: MPH_DNN_PARAM_BUDGET={est_params}"
        )

    # Cat 3 (Codex 절충안, 2026-05-12): MPH_HP_SPACE meta-flag
    # - "full" (default) = all categoricals (사용자 명시 보존)
    # - "stable"          = ANO/Codex 권장 subset (production 안정)
    _hp_space = GLOBAL.optuna.hp_space.lower()
    if _hp_space == "stable":
        _act_choices = ["gelu", "silu", "mish", "relu", "selu", "leaky_relu", "elu"]
        _opt_choices = ["adamw", "adam", "radam", "nadam"]
        _norm_choices = ["none", "batch", "layer", "group"]
        _init_choices = ["kaiming_uniform", "kaiming_normal",
                         "xavier_uniform", "xavier_normal"]
        _layer_choices = ["linear", "residual", "dense_block", "highway"]
    else:  # "full"
        _act_choices = ["relu", "gelu", "selu", "leaky_relu", "mish",
                        "swish", "elu", "tanh", "softplus", "prelu", "celu"]
        _opt_choices = ["adam", "adamw", "sgd", "rmsprop",
                        "nadam", "radam", "lamb", "ranger"]
        _norm_choices = ["none", "batch", "layer", "group", "instance",
                         "weight", "spectral"]
        _init_choices = ["kaiming_uniform", "kaiming_normal",
                         "xavier_uniform", "xavier_normal",
                         "orthogonal", "trunc_normal"]
        _layer_choices = ["linear", "residual", "dense_block", "highway"]

    hp: Dict[str, Any] = {
        "n_layers":      n_layers,
        # Per-layer hidden units (log/linear via use_log_hidden 사용자 옵션)
        "hidden_dims":   hidden_dims,
        # Dropout 0.1 ~ 0.5 (사용자 명시)
        "dropouts":      [trial.suggest_float(f"td_d{i}", 0.1, 0.5)
                            for i in range(n_layers)],
        # Learning rate 1e-5 ~ 1e-2 (사용자 명시: 0.00001 ~ 0.01)
        "lr":            trial.suggest_float("td_lr", 1e-5, 1e-2, log=True),
        # L2 regularization (weight_decay) 1e-6 ~ 1e-3 (사용자 명시)
        "weight_decay":  trial.suggest_float("td_l2", 1e-6, 1e-3, log=True),
        # Batch size — 더 다양하게
        "batch_size":    trial.suggest_categorical("td_bs",
                            [8, 16, 32, 64, 128, 256]),
        # Activation — 11 (full) / 7 (stable) 종, MPH_HP_SPACE 로 결정
        "activation":    trial.suggest_categorical("td_act", _act_choices),
        # Optimizer (compiler) — 8 (full) / 4 (stable) 종
        "optimizer":     trial.suggest_categorical("td_opt", _opt_choices),
        # Normalization — 7 (full) / 4 (stable) 종
        "norm":          trial.suggest_categorical("td_norm", _norm_choices),
        # Weight init — 6 (full) / 4 (stable) 종
        "init":          trial.suggest_categorical("td_init", _init_choices),
        # Layer architecture — 4 종 (full/stable 동일)
        "layer_type":    trial.suggest_categorical("td_layer_type", _layer_choices),
        # Optional features
        "use_attention": trial.suggest_categorical("td_attn", [False, True]),
        "use_fm":        trial.suggest_categorical("td_fm",   [False, True]),
        "use_bias":      trial.suggest_categorical("td_bias", [True, False]),
        "skip_connection": trial.suggest_categorical("td_skip",
                            [False, True]),
        # LR schedule + gradient clipping
        "lr_schedule":   trial.suggest_categorical("td_lr_sched",
                            ["none", "cosine", "step", "exp", "cyclic",
                             "reduce_on_plateau", "warmup_cosine"]),
        "gradient_clip": trial.suggest_float("td_grad_clip", 0.1, 5.0),
        # Loss function
        "loss":          trial.suggest_categorical("td_loss",
                            # G-218: huber + smooth_l1 (HuberLoss equivalent) 영구 제거 (huber-loss-banned-20260520)
                            ["mse", "mae", "logcosh", "quantile"]),
        # SGD/Lamb specific
        "momentum":      trial.suggest_float("td_momentum", 0.5, 0.999),
        "nesterov":      trial.suggest_categorical("td_nest", [False, True]),
        # AdamW specific betas
        "beta1":         trial.suggest_float("td_beta1", 0.7, 0.999),
        "beta2":         trial.suggest_float("td_beta2", 0.9, 0.9999),
        "eps":           trial.suggest_float("td_eps", 1e-9, 1e-6, log=True),
        # Warmup epochs
        "warmup_epochs": trial.suggest_int("td_warmup", 0, 20),
        # Package K (G-141, Bühlmann 2018): alpha_blend as model HP
    }
    return hp


def suggest_tft_hp(trial: Any) -> Dict[str, Any]:
    """Temporal Fusion Transformer (Lim 2019). + Package K alpha_blend."""
    n_heads = trial.suggest_categorical("tft_n_heads", [2, 4, 8])
    raw = trial.suggest_int("tft_d_model_raw", 16, 512, log=True)
    d_model = ((raw + n_heads - 1) // n_heads) * n_heads
    return {
        "d_model":           d_model,
        "n_heads":           n_heads,
        "n_lstm_layers":     trial.suggest_int("tft_n_lstm", 1, 3),
        "attention_head_size": trial.suggest_int("tft_attn_head", 4, 64, log=True),
        "dropout":           trial.suggest_float("tft_dropout", 0.0, 0.4),
        "lr":                trial.suggest_float("tft_lr", 1e-4, 5e-3, log=True),
        "weight_decay":      trial.suggest_float("tft_wd", 1e-6, 1e-3, log=True),
        "batch_size":        trial.suggest_categorical("tft_bs", _BS_CHOICES),
        "context_len":       trial.suggest_int("tft_ctx", 8, 52, step=4),
        "horizon_len":       trial.suggest_int("tft_hzn", 1, 13),
    }


def _ss_cap(hi: int, capped: int) -> int:
    """G-217 (2026-05-16): n=242 소표본 modern-ts over-capacity 처방.

    MPH_SMALL_SAMPLE_CAP=1 시 min(hi, capped) — d_model/n_layers/d_ff 상한↓.
    default (env unset) = hi 보존 (기존 동작 영향 0, RESUME_GUIDE 규칙 3 만족).

    근거 (mini-test 불필요 — 이미 입증된 over-capacity 의 직접 처방):
      · 4축 진단: ACF(1)=0.953 (n=242 << d_model² params = over-parameterized)
      · production-matched: TCN [32,16,8] R²=-0.945 폭주 vs [8,4] R²=0.46 (capacity↓로 발산 차단)
      · C-2: capacity 축소해도 lag1 0.854 미달 (paper §4 fragility 데이터, champion 아님)
    """
    if GLOBAL.optuna.small_sample_cap:
        return min(hi, capped)
    return hi


def suggest_patchtst_hp(trial: Any) -> Dict[str, Any]:
    """PatchTST (Nie 2022) — patch-based transformer. + Package K alpha_blend."""
    n_heads = trial.suggest_categorical("pt_n_heads", [2, 4, 8])
    raw = trial.suggest_int("pt_d_model_raw", 16, _ss_cap(256, 64), log=True)
    d_model = ((raw + n_heads - 1) // n_heads) * n_heads
    return {
        "d_model":      d_model,
        "n_heads":      n_heads,
        "n_layers":     trial.suggest_int("pt_n_layers", 1, _ss_cap(4, 2)),
        "patch_len":    trial.suggest_categorical("pt_patch", [4, 8, 13, 16, 26]),
        "stride":       trial.suggest_categorical("pt_stride", [2, 4, 8]),
        "dropout":      trial.suggest_float("pt_dropout", 0.0, 0.3),
        "fc_dropout":   trial.suggest_float("pt_fc_dropout", 0.0, 0.3),
        "head_dropout": trial.suggest_float("pt_head_dropout", 0.0, 0.3),
        "lr":           trial.suggest_float("pt_lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("pt_wd", 1e-6, 1e-3, log=True),
        "batch_size":   trial.suggest_categorical("pt_bs", _BS_CHOICES),
    }


def suggest_itransformer_hp(trial: Any) -> Dict[str, Any]:
    """iTransformer (Liu 2024) — inverted (variate-attention) transformer. + Package K."""
    n_heads = trial.suggest_categorical("it_n_heads", [2, 4, 8])
    raw = trial.suggest_int("it_d_model_raw", 16, 256, log=True)
    d_model = ((raw + n_heads - 1) // n_heads) * n_heads
    return {
        "d_model":     d_model,
        "n_heads":     n_heads,
        "e_layers":    trial.suggest_int("it_e_layers", 1, 4),
        "factor":      trial.suggest_int("it_factor", 1, 5),
        "d_ff":        trial.suggest_int("it_d_ff", 64, 1024, log=True),
        "dropout":     trial.suggest_float("it_dropout", 0.0, 0.3),
        "lr":          trial.suggest_float("it_lr", 1e-4, 5e-3, log=True),
        "batch_size":  trial.suggest_categorical("it_bs", _BS_CHOICES),
    }


def suggest_timesnet_hp(trial: Any) -> Dict[str, Any]:
    """TimesNet (Wu 2023) — 1D inception conv on top-k periods. + Package K."""
    return {
        "top_k":       trial.suggest_int("tn_top_k", 2, 5),
        "d_model":     trial.suggest_int("tn_d_model", 16, _ss_cap(256, 64), log=True),
        "e_layers":    trial.suggest_int("tn_e_layers", 1, _ss_cap(4, 2)),
        "d_ff":        trial.suggest_int("tn_d_ff", 32, _ss_cap(512, 128), log=True),
        "n_kernels":   trial.suggest_int("tn_n_kernels", 3, 8),
        "dropout":     trial.suggest_float("tn_dropout", 0.0, 0.3),
        "lr":          trial.suggest_float("tn_lr", 1e-4, 1e-2, log=True),
        "batch_size":  trial.suggest_categorical("tn_bs", _BS_CHOICES),
    }


def suggest_mamba_hp(trial: Any) -> Dict[str, Any]:
    """Mamba (Gu & Dao 2023) — state-space model. + Package K alpha_blend."""
    return {
        "d_model":    trial.suggest_int("mamba_d_model", 32, _ss_cap(256, 64), log=True),
        "n_layers":   trial.suggest_int("mamba_n_layers", 1, _ss_cap(6, 2)),
        "d_state":    trial.suggest_categorical("mamba_d_state", [8, 16, 32, 64]),
        "d_conv":     trial.suggest_categorical("mamba_d_conv", [2, 4, 8]),
        "expand":     trial.suggest_categorical("mamba_expand", [1, 2, 4]),
        "dropout":    trial.suggest_float("mamba_dropout", 0.0, 0.3),
        "lr":         trial.suggest_float("mamba_lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("mamba_bs", _BS_CHOICES),
    }


def suggest_tcn_hp(trial: Any) -> Dict[str, Any]:
    """Temporal Convolutional Network (Bai 2018). + Package K alpha_blend."""
    n_blocks = trial.suggest_int("tcn_n_blocks", 2, 8)
    return {
        "n_blocks":     n_blocks,
        "channels":     [trial.suggest_int(f"tcn_ch{i}", 16, 256, log=True)
                          for i in range(n_blocks)],
        "kernel_size":  trial.suggest_categorical("tcn_kernel", [2, 3, 5, 7]),
        "dropout":      trial.suggest_float("tcn_dropout", 0.0, 0.4),
        "lr":           trial.suggest_float("tcn_lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("tcn_wd", 1e-6, 1e-3, log=True),
        "batch_size":   trial.suggest_categorical("tcn_bs", _BS_CHOICES),
    }


def suggest_nbeats_hp(trial: Any) -> Dict[str, Any]:
    """N-BEATS (Oreshkin 2020) — basis function decomposition. + Package K."""
    return {
        "n_blocks":        trial.suggest_int("nb_n_blocks", 2, 6),
        "n_layers":        trial.suggest_int("nb_n_layers", 2, 6),
        "hidden_dim":      trial.suggest_int("nb_hidden", 32, 512, log=True),
        "basis_type":      trial.suggest_categorical("nb_basis",
                              ["generic", "trend", "seasonality", "interpretable"]),
        "share_weights":   trial.suggest_categorical("nb_share", [False, True]),
        "lr":              trial.suggest_float("nb_lr", 1e-4, 5e-3, log=True),
        "batch_size":      trial.suggest_categorical("nb_bs", _BS_CHOICES),
    }


def suggest_nhits_hp(trial: Any) -> Dict[str, Any]:
    """N-HiTS (Challu 2023) — hierarchical interpolation. + Package K."""
    n_blocks = trial.suggest_int("nh_n_blocks", 2, 5)
    return {
        "n_blocks":               n_blocks,
        "n_pool_kernel_size":     [trial.suggest_categorical(
            f"nh_pool_k{i}", [1, 2, 4, 8]) for i in range(n_blocks)],
        "n_freq_downsample":      [trial.suggest_categorical(
            f"nh_freq{i}", [1, 2, 4]) for i in range(n_blocks)],
        "hidden_dim":             trial.suggest_int("nh_hidden", 64, 512, log=True),
        "lr":                     trial.suggest_float("nh_lr", 1e-4, 5e-3, log=True),
        "batch_size":             trial.suggest_categorical("nh_bs", _BS_CHOICES),
    }


def suggest_tide_hp(trial: Any) -> Dict[str, Any]:
    """TiDE (Das 2023) — Time-series Dense Encoder. + Package K."""
    return {
        "encoder_layers":    trial.suggest_int("tide_enc", 1, 4),
        "decoder_layers":    trial.suggest_int("tide_dec", 1, 4),
        "hidden_size":       trial.suggest_int("tide_hidden", 64, 512, log=True),
        "decoder_output_dim": trial.suggest_int("tide_dec_out", 4, 32, log=True),
        "temporal_decoder_hidden": trial.suggest_int("tide_temp_hidden",
                                                          16, 128, log=True),
        "dropout":           trial.suggest_float("tide_dropout", 0.0, 0.3),
        "lr":                trial.suggest_float("tide_lr", 1e-4, 5e-3, log=True),
        "batch_size":        trial.suggest_categorical("tide_bs", _BS_CHOICES),
    }


def suggest_deepar_hp(trial: Any) -> Dict[str, Any]:
    """DeepAR (Salinas 2020) — autoregressive RNN with likelihood head. + Package K alpha_blend."""
    return {
        "rnn_type":     trial.suggest_categorical("dar_rnn", ["lstm", "gru"]),
        "n_layers":     trial.suggest_int("dar_n_layers", 1, 4),
        "hidden_size":  trial.suggest_int("dar_hidden", 32, 256, log=True),
        "dropout":      trial.suggest_float("dar_dropout", 0.0, 0.4),
        "lr":           trial.suggest_float("dar_lr", 1e-4, 5e-3, log=True),
        "batch_size":   trial.suggest_categorical("dar_bs", _BS_CHOICES),
        "likelihood":   trial.suggest_categorical("dar_lik",
                            ["normal", "studentt", "negativebinomial"]),
    }


def suggest_rnn_pf_hp(trial: Any) -> Dict[str, Any]:
    """RNN-pf (probabilistic) — same as DeepAR but standalone. + Package K alpha_blend."""
    return {
        "rnn_type":     trial.suggest_categorical("rnn_type", ["lstm", "gru"]),
        "n_layers":     trial.suggest_int("rnn_n_layers", 1, 3),
        "hidden_size":  trial.suggest_int("rnn_hidden", 32, 256, log=True),
        "dropout":      trial.suggest_float("rnn_dropout", 0.0, 0.4),
        "lr":           trial.suggest_float("rnn_lr", 1e-4, 5e-3, log=True),
        "batch_size":   trial.suggest_categorical("rnn_bs", _BS_CHOICES),
    }


def suggest_pinn_hp(trial: Any) -> Dict[str, Any]:
    """PINN-Lite / MP-PINN — physics-informed NN with SEIR ODE constraint."""
    return {
        "n_layers":         trial.suggest_int("pinn_n_layers", 2, 6),
        "hidden_size":      trial.suggest_int("pinn_hidden", 32, 256, log=True),
        "physics_weight":   trial.suggest_float("pinn_phys_w", 1e-3, 1.0, log=True),
        "data_weight":      trial.suggest_float("pinn_data_w", 0.5, 5.0),
        "ode_steps":        trial.suggest_int("pinn_ode_steps", 4, 32),
        "activation":       trial.suggest_categorical("pinn_act",
                                ["tanh", "sin", "gelu", "swish"]),
        "lr":               trial.suggest_float("pinn_lr", 1e-4, 1e-2, log=True),
        "batch_size":       trial.suggest_categorical("pinn_bs", _BS_CHOICES),
    }


def suggest_ge_dnn_hp(trial: Any) -> Dict[str, Any]:
    """Graph-Embedded DNN (commuter graph + node features). + Package K alpha_blend.

    Reference regression applies (graph 모델은 environment shift 견고성 필요).
    suggest_ge_dnn_gat_hp 가 이 함수를 base 로 호출하므로 alpha_blend 자동 상속.
    """
    return {
        "node_hidden":   trial.suggest_int("ge_node_h", 16, 128, log=True),
        "graph_hidden":  trial.suggest_int("ge_graph_h", 16, 128, log=True),
        "n_gcn_layers":  trial.suggest_int("ge_gcn_layers", 1, 3),
        "n_mlp_layers":  trial.suggest_int("ge_mlp_layers", 1, 3),
        "dropout":       trial.suggest_float("ge_dropout", 0.0, 0.4),
        "lr":            trial.suggest_float("ge_lr", 1e-4, 1e-2, log=True),
        "batch_size":    trial.suggest_categorical("ge_bs", [8, 16, 32]),
    }


def suggest_ge_dnn_gat_hp(trial: Any) -> Dict[str, Any]:
    """GE-DNN with GATv2 attention (Brody 2022).

    2026-04-28: 50 분/trial 폭주 fix — HP 공간 축소 + augment cap.
    - gat_hidden: 16-128 → 16-64 (-50%, attention 비용 ∝ d²)
    - gat_heads:  [2,4,8] → [2,4]  (heads=8 가 가장 느림)
    - gat_layers: 1-3 → 1-2 (multi-layer GAT 불필요 — 25 노드만)
    - augment_factor: 1-6 → 1-3 (데이터 augment 50% cut)
    """
    base = suggest_ge_dnn_hp(trial)
    base.update({
        "gat_hidden":      trial.suggest_int("gat_h", 16, 64, log=True),
        "gat_heads":       trial.suggest_categorical("gat_heads", [2, 4]),
        "gat_layers":      trial.suggest_int("gat_layers", 1, 2),
        # G-231 + 2026-05-26 archive: PI augmentation permanently disabled.
        # augment_factor fixed at 0 (was env-var suggest_int with MPH_PI_AUGMENT_LO/HI).
        "augment_factor": 0,
    })
    return base


# ════════════════════════════════════════════════════════════════════
# 3. Dispatch by model name
# ════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════
# 4. Model-specific BEST sampler recommendation
# ════════════════════════════════════════════════════════════════════
#
# 결정 기준:
#   - HP 차원 (low <10, mid 10-30, high 30+)
#   - HP 종류 (categorical 위주 / continuous 위주 / 혼합)
#   - Trial budget (적음 ≤20: GP 강함, 많음 ≥50: TPE-MV 충분)
#   - Sensitivity (PINN 의 physics_weight 같은 sensitive continuous → GP)
#   - Multi-objective 여부 (WIS+MAE 동시 → NSGA-II)
#
# 권장 매핑 (각 모델 별 검증된 best sampler):
_BEST_SAMPLER_BY_MODEL: dict[str, str] = {
    # ── Paper-primary 11 ──────────────────────────────────────────
    "SARIMA":            "tpe-mv",   # low-dim categorical + AR/MA orders
    "ElasticNet":        "tpe-mv",   # 2 continuous (alpha, l1_ratio) — TPE 충분
    "XGBoost":           "tpe-mv",   # mixed cat+cont, interaction 중요
    "NegBinGLM":         "tpe-mv",   # 단순 continuous (alpha)
    "BayesianMCMC":      "gp",       # sensitive (n_warmup, n_samples) — sample-efficient
    "PINN-Lite":         "gp",       # ⭐ sensitive (physics_weight) — GP best
    "TabularDNN-Lite":   "cma",      # ⭐ high-dim continuous (40+ HP) — CMA-ES best
    "TFT":               "tpe-mv",   # mixed (categorical n_heads + continuous)
    "PatchTST":          "tpe-mv",   # mixed (patch_len cat + continuous)
    # G-261 (2026-06-13): Chronos-2 제거 — Chronos retire. foundation 은 .get() 기본 "tpe-mv" 사용.
    "Ensemble-Stacking": "nsga2",    # ⭐ multi-objective (component weights + meta)

    # ── Tree (mixed cat+cont) ─────────────────────────────────────
    "LightGBM":          "tpe-mv",
    "RandomForest":      "tpe-mv",

    # ── Linear / kernel (low-dim continuous) ──────────────────────
    "KRR":               "tpe-mv",
    "SVR-RBF":           "tpe-mv",
    "SVR-Linear":        "tpe-mv",
    "BayesianRidge":     "tpe-mv",
    "PoissonAutoreg":    "tpe-mv",

    # ── Epi (low-dim) ─────────────────────────────────────────────
    "GAM-Spline":        "tpe-mv",
    "NegBinGLM-V7":      "tpe-mv",
    "GP-RBF-Periodic":   "gp",       # sensitive kernel HP

    # ── DL/MLP (high-dim continuous) — CMA-ES best ────────────────
    "TabularDNN":        "cma",
    "DNN":               "cma",
    "DNN-Optuna":        "cma",
    # G-231 (2026-05-22): DNN-Conformal, α-blend DNN 제거 — α-blend 폐기
    "TinyMLP":           "tpe-mv",   # 단순 (321→32→16→1) — TPE 충분

    # ── Transformer / modern TS (mixed) — TPE-MV ──────────────────
    "iTransformer":      "tpe-mv",
    "TimesNet":          "tpe-mv",
    "Mamba":             "tpe-mv",
    "TCN":               "tpe-mv",
    "TCN-Optuna":        "tpe-mv",
    "N-BEATS":           "tpe-mv",
    "N-HiTS":            "tpe-mv",
    "TiDE":              "tpe-mv",
    "DeepAR":         "tpe-mv",
    "RNN":            "tpe-mv",
    "PatchTST":          "tpe-mv",   # (paper-11 와 중복, OK)

    # ── Graph (medium-dim mixed) ──────────────────────────────────
    "GE-DNN":            "tpe-mv",
    "GE-GAT":        "tpe-mv",

    # ── Physics (sensitive continuous) — GP ───────────────────────
    "MP-PINN":           "gp",
    "Bayesian-SEIR":     "gp",
    "Metapop-SEIR":      "gp",
    "SEIR-V2-Forced":    "gp",
    "Rt-Augmented":      "tpe-mv",

    # ── Foundation (HP 적음) ──────────────────────────────────────
    # G-261 (2026-06-13): Chronos 전 변형(Chronos-2-FT/-FT-Real/-MultiCountry) 제거 — Chronos retire.
    #   TimesFM-2.5 / TiRex 는 .get() 기본 "tpe-mv" 사용 (HP 적은 foundation 동일).
    "FoundationModelTransfer": "tpe-mv",
    "OverseasTransfer":  "tpe-mv",

    # ── TS classical (low-dim) ────────────────────────────────────
    "SARIMAX":           "tpe-mv",
    "ARIMA":             "tpe-mv",

    # ── Ensemble (meta — multi-objective optional) ────────────────
    "Ensemble-NNLS":     "tpe-mv",
    "Ensemble-NNLS-Filtered": "tpe-mv",
    "Ensemble-BMA":      "tpe-mv",
    "Ensemble-SelectiveBMA": "tpe-mv",
    "Ensemble-InvRMSE":  "tpe-mv",
    "Ensemble-Temporal": "tpe-mv",
    "Ensemble-Diversity": "tpe-mv",
    "Ensemble-Adaptive": "tpe-mv",
    "Ensemble-Blending": "tpe-mv",
    "Ensemble-ResidualAR": "tpe-mv",
}


def get_best_sampler_for(model_name: str, *,
                            seed: int = 42,
                            n_startup_trials: int = 10) -> Any:
    """Return the best sampler instance for a specific model.

    근거:
      - tpe-mv: mixed cat/cont, +5-15% over univariate TPE
      - cma:    high-dim continuous (DL with 30+ HP)
      - gp:     sensitive continuous (PINN, BayesianMCMC, GP-RBF) — sample-efficient
      - nsga2:  multi-objective (Ensemble-Stacking with WIS+MAE)
    """
    name = _BEST_SAMPLER_BY_MODEL.get(model_name, "tpe-mv")
    return build_sampler(name, seed=seed, n_startup_trials=n_startup_trials)


def get_best_sampler_name(model_name: str) -> str:
    """Return the recommended sampler name string for a model."""
    return _BEST_SAMPLER_BY_MODEL.get(model_name, "tpe-mv")


# Universal default — for cases where model is unknown
UNIVERSAL_BEST_SAMPLER = "tpe-mv"   # multivariate TPE


def get_hp_suggester(model_name: str):
    """Return the right HP suggester for a model name."""
    m = model_name.lower().replace("-", "").replace("_", "")
    if "tabulardnn" in m:
        return suggest_tabular_dnn_hp
    if "tftpf" in m or m == "tft":
        return suggest_tft_hp
    if "patchtst" in m:
        return suggest_patchtst_hp
    if "itransformer" in m:
        return suggest_itransformer_hp
    if "timesnet" in m:
        return suggest_timesnet_hp
    if "mamba" in m:
        return suggest_mamba_hp
    if "tcn" in m:
        return suggest_tcn_hp
    if "nbeats" in m:
        return suggest_nbeats_hp
    if "nhits" in m:
        return suggest_nhits_hp
    if "tide" in m:
        return suggest_tide_hp
    if "deepar" in m:
        return suggest_deepar_hp
    if "rnnpf" in m:
        return suggest_rnn_pf_hp
    if "pinn" in m:
        return suggest_pinn_hp
    if "gednngat" in m:
        return suggest_ge_dnn_gat_hp
    if "gednn" in m:
        return suggest_ge_dnn_hp
    # 2026-04-27: Tree 모델 표준화
    if "xgboost" in m or "xgb" in m:
        return suggest_xgboost_hp
    if "lightgbm" in m or "lgbm" in m or "lgb" in m:
        return suggest_lightgbm_hp
    if "randomforest" in m or "rf" == m:
        return suggest_randomforest_hp
    if "gradientboost" in m or "gbm" in m:
        return suggest_gradientboost_hp
    if "catboost" in m:
        return suggest_catboost_hp
    # default — generic MLP
    return suggest_tabular_dnn_hp


# ══════════════════════════════════════════════════════════
# 2026-04-27: Tree 모델 HP suggester 표준화 (DNN 38 HP 와 동일 정책)
# ══════════════════════════════════════════════════════════

def suggest_xgboost_hp(trial: Any) -> Dict[str, Any]:
    """XGBoost — 12 HP search space (TPE-MV 권장)."""
    return {
        "n_estimators":       trial.suggest_int("xgb_n_est", 50, 500),
        "max_depth":          trial.suggest_int("xgb_depth", 2, 10),
        "learning_rate":      trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
        "subsample":          trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("xgb_colsample", 0.5, 1.0),
        "colsample_bylevel":  trial.suggest_float("xgb_colsample_lvl", 0.5, 1.0),
        "reg_alpha":          trial.suggest_float("xgb_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("xgb_lambda", 1e-3, 10.0, log=True),
        "min_child_weight":   trial.suggest_int("xgb_min_child", 1, 20),
        "gamma":              trial.suggest_float("xgb_gamma", 0.0, 5.0),
        "max_delta_step":     trial.suggest_int("xgb_max_delta", 0, 10),
        "tree_method":        trial.suggest_categorical("xgb_tree_method",
                                  ["hist", "exact", "approx"]),
        "random_state": 42, "n_jobs": 2, "verbosity": 0,
    }


def suggest_lightgbm_hp(trial: Any) -> Dict[str, Any]:
    """LightGBM — 11 HP search space (TPE-MV 권장)."""
    return {
        "n_estimators":       trial.suggest_int("lgb_n_est", 50, 500),
        "max_depth":          trial.suggest_int("lgb_depth", 2, 12),
        "num_leaves":         trial.suggest_int("lgb_leaves", 8, 256, log=True),
        "learning_rate":      trial.suggest_float("lgb_lr", 1e-3, 0.3, log=True),
        "subsample":          trial.suggest_float("lgb_subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("lgb_colsample", 0.5, 1.0),
        "reg_alpha":          trial.suggest_float("lgb_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("lgb_lambda", 1e-3, 10.0, log=True),
        "min_child_samples":  trial.suggest_int("lgb_min_child", 5, 100),
        "min_split_gain":     trial.suggest_float("lgb_min_gain", 0.0, 1.0),
        "boosting_type":      trial.suggest_categorical("lgb_boost",
                                  ["gbdt", "dart", "rf"]),
        "random_state": 42, "n_jobs": 2, "verbose": -1,
    }


def suggest_randomforest_hp(trial: Any) -> Dict[str, Any]:
    """RandomForest — 7 HP search space (TPE-MV 권장)."""
    return {
        "n_estimators":     trial.suggest_int("rf_n_est", 50, 500),
        "max_depth":        trial.suggest_int("rf_depth", 3, 20),
        "min_samples_split": trial.suggest_int("rf_min_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("rf_min_leaf", 1, 20),
        "max_features":     trial.suggest_categorical("rf_max_feat",
                                ["sqrt", "log2", 0.5, 0.7, 1.0]),
        "bootstrap":        trial.suggest_categorical("rf_bootstrap",
                                [True, False]),
        "random_state": 42, "n_jobs": 2,
    }


def suggest_gradientboost_hp(trial: Any) -> Dict[str, Any]:
    """sklearn GradientBoosting — 8 HP."""
    return {
        "n_estimators":     trial.suggest_int("gb_n_est", 50, 500),
        "max_depth":        trial.suggest_int("gb_depth", 2, 10),
        "learning_rate":    trial.suggest_float("gb_lr", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("gb_subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("gb_min_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("gb_min_leaf", 1, 20),
        "max_features":     trial.suggest_categorical("gb_max_feat",
                                ["sqrt", "log2", None]),
        "loss":             trial.suggest_categorical("gb_loss",
                                # G-218: huber 영구 제거 (huber-loss-banned-20260520)
                                ["squared_error", "absolute_error"]),
        "random_state": 42,
    }


def suggest_catboost_hp(trial: Any) -> Dict[str, Any]:
    """CatBoost — 9 HP (다음 sprint 에서 모델 추가 시 사용)."""
    return {
        "iterations":       trial.suggest_int("cb_iter", 50, 500),
        "depth":            trial.suggest_int("cb_depth", 2, 10),
        "learning_rate":    trial.suggest_float("cb_lr", 1e-3, 0.3, log=True),
        "l2_leaf_reg":      trial.suggest_float("cb_l2", 1.0, 30.0, log=True),
        "border_count":     trial.suggest_int("cb_border", 32, 255),
        "bagging_temperature": trial.suggest_float("cb_bag_temp", 0.0, 1.0),
        "random_strength":  trial.suggest_float("cb_rand_str", 0.0, 10.0),
        "grow_policy":      trial.suggest_categorical("cb_grow",
                                ["SymmetricTree", "Depthwise", "Lossguide"]),
        "od_type":          trial.suggest_categorical("cb_od", ["IncToDec", "Iter"]),
        "verbose": False, "random_seed": 42,
    }


__all__ = [
    "build_sampler",
    # DNN family
    "suggest_tabular_dnn_hp", "suggest_tft_hp", "suggest_patchtst_hp",
    "suggest_itransformer_hp", "suggest_timesnet_hp", "suggest_mamba_hp",
    "suggest_tcn_hp", "suggest_nbeats_hp", "suggest_nhits_hp",
    "suggest_tide_hp", "suggest_deepar_hp", "suggest_rnn_pf_hp",
    "suggest_pinn_hp", "suggest_ge_dnn_hp", "suggest_ge_dnn_gat_hp",
    # Tree family (2026-04-27 신규)
    "suggest_xgboost_hp", "suggest_lightgbm_hp", "suggest_randomforest_hp",
    "suggest_gradientboost_hp", "suggest_catboost_hp",
    "get_hp_suggester",
    "get_best_sampler_for", "get_best_sampler_name",
    "UNIVERSAL_BEST_SAMPLER",
]
