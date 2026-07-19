"""
simulation/models/runner.py
===========================
48개 모델 통합 실행기 (MultiModelRunner) v7c.

 변경: TargetTransformer 통합 (log1p/sqrt/boxcox/robust/none)
 - 모든 모델에 일관된 타겟 변환 적용
 - 분포 이동 완화: train/test ratio 0.30x → 0.64x (log1p 기준)
 - 역변환 후 원래 스케일에서 메트릭 계산
 - COVID-era 전략: 가중치 강화, 분포 클리핑 등

계획서 기술:
 "21개 예측 모델을 동일한 train/validation/test 분할(60/20/20)에서
 학습·검증·평가한다. 훈련 세트(202주)로 모델을 학습하고,
 검증 세트(67주)로 하이퍼파라미터를 조정하며,
 테스트 세트(68주)로 최종 성능을 평가한다."

사용법:
 from simulation.models.runner import MultiModelRunner
 from simulation.models.target_transform import TargetTransformer

 tt = TargetTransformer(method="log1p")
 runner = MultiModelRunner(data_size=341, target_transformer=tt)

 # 3-split (명시적 validation 제공)
 results = runner.run(X_train, y_train, X_val, y_val, X_test, y_test)

 # 2-split (80:20, validation은 내부 자동 생성)
 results = runner.run(X_train, y_train, X_test=X_test, y_test=y_test)
"""

from __future__ import annotations

import gc
import json
import logging
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd  # polars→pandas bridge for sklearn

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# G-044/G-045: subprocess.Popen 격리 -- OOM 시 자식만 죽고 부모 생존
# Windows multiprocessing.Process는 spawn 모드에서 __main__ 재실행 문제 발생
# → subprocess.Popen + pickle 파일 교환 패턴 사용
# ══════════════════════════════════════════════════════════════

# G-236 (2026-05-29): per-model 학습 루프(_run_pipeline individual loop)에서
# in-process 로 도는 모델이 OMP/BLAS thread 를 누적 → ~30 모델째 `OMP: System error
# #22 (pthread_create EINVAL)` → SIGSEGV (실측: full run 73분, CQR-LightGBM 30/56 crash).
# 원인: 이 set 이 "modern_ts"(dead — 어떤 model.meta.category 도 아님) 만 갖고 있어
# tree(XGBoost/LightGBM/CQR-LightGBM)·linear·ts·epi·physics 가 전부 in-process 였음.
# 해결: 모든 *개별* 모델 category 를 격리 → 각 모델 fresh subprocess → OS 가 thread
# 100% 회수 → 누적 0 (ENGINEERING_PRINCIPLES.md 원칙 #2 "격리로 회수"). worker(_WORKER_SCRIPT)는
# generic(fit/predict · fit_series/forecast)이라 비-DL 모델도 동일 처리.
# "meta"(ensemble 7) 는 제외 — OOF 예측을 in-memory 로 결합하는 별도 stage(run_ensembles).
# macOS PyG/MPS fork-unsafe 4 모델은 _INPROCESS_OVERRIDE_DARWIN 가 여전히 in-process 강제.
# G-342 (2026-06-24): "foundation" 추가 — FusedEpi(meta.category='foundation', TiRex+TabPFN 융합)가
#   격리목록 밖이라 in-process 실행→OMP 누적(G-236) 위험이었음. TimesFM/TiRex(meta.category='dl')는
#   이미 격리되나 FusedEpi 만 'foundation' 이라 누락. 'dl' 로 못 바꾸는 이유 = per_feature_preprocessor.py:174
#   가 'foundation'→"none"(자체 representation) 전처리에 의존 → category 보존하고 격리집합에 추가.
_SUBPROCESS_CATEGORIES = {"dl", "tree", "linear", "ts", "epi", "physics", "foundation"}

# ──────────────────────────────────────────────────────────────────────
# OS-별 subprocess 격리 전략 (2026-04-25 사용자 지적 반영)
# ──────────────────────────────────────────────────────────────────────
# Linux:   fork() 는 안정적. subprocess.Popen 정상 작동. 모든 DL 모델 격리.
# macOS:   fork() 가 PyG / 특정 torch 패턴에서 SIGSEGV → 화이트리스트는 in-process
#          (Apple libomp, MPS, PyG 의 spawn-unsafe 라이브러리 조합 때문)
# Windows: spawn 모드만 지원 (fork 없음). subprocess.Popen 이 자동으로 spawn 사용.
#          libomp 안전 — but VRAM cleanup 이 NVIDIA 에서 dirty exit 시 문제,
#          worker 가 _os._exit(0) 전에 cuda.empty_cache 실행해야 함.
import os as _os_mod
import sys as _sys_mod


def _get_fast_tmpdir(prefix: str = "mph_") -> str:
    """OS-aware 빠른 임시 디렉토리 자동 선택 (2026-04-26 사용자 요청).

    - Linux : `/dev/shm` (RAM-backed tmpfs) 가용 + 500MB+ 여유 시 사용 → IO 50% ↓
    - macOS : $TMPDIR (APFS + COW + SSD) — 이미 최적, 변경 없음
    - Windows: %TEMP% (NTFS) — 대안 없음, 변경 없음

    환경변수:
    - `MPH_FAST_TMPDIR=0` : 강제 default 사용 (back-compat)
    - `MPH_FAST_TMPDIR=1` : 강제 fast 시도 (default — Linux 만 효과)
    - 미설정 시 default = "1" (자동, 안전 fallback)

    Returns: tempfile.mkdtemp 결과 (사용 후 shutil.rmtree 필요).
    """
    # SSOT 예외(2026-05-28): MPH_FAST_TMPDIR 는 paths.fast_tmp(경로)와 dual-purpose →
    # `!= "0"` bool 의미를 _env_bool 로 옮기면 path 값에서 역전. 원시 read 유지.
    fast_mode = _os_mod.environ.get("MPH_FAST_TMPDIR", "1") != "0"
    if fast_mode and _sys_mod.platform.startswith("linux"):
        shm = "/dev/shm"
        if _os_mod.path.exists(shm) and _os_mod.access(shm, _os_mod.W_OK):
            try:
                free_mb = shutil.disk_usage(shm).free / (1024 * 1024)
                if free_mb > 500:    # 안전 임계 — 메모리 부족 시 fallback
                    return tempfile.mkdtemp(prefix=prefix, dir=shm)
            except Exception:
                pass    # 어떤 이유든 실패 시 default

    return tempfile.mkdtemp(prefix=prefix)


# ════════════════════════════════════════════════════════════════
# 2026-04-27: atexit 임시파일 cleanup 보장
# ─────────────────────────────────────────────────────────────────
# subprocess crash / SIGINT / SIGTERM / KeyboardInterrupt 시에도
# /tmp 또는 /dev/shm 의 mph_model_*, optuna_iso_* 디렉토리 정리.
# 학습 process 에 영향 없음 (이미 로드된 코드 변경 X).
# ════════════════════════════════════════════════════════════════
import atexit as _atexit_module
import glob as _glob_module

# 이 process 가 만든 tmpdir 만 추적 (다른 process tmpdir 보호)
_OWNED_TMPDIRS: list[str] = []


def _track_tmpdir(td: str) -> str:
    """_get_fast_tmpdir + _run_model_in_subprocess 가 만든 tmpdir 추적."""
    if td and td not in _OWNED_TMPDIRS:
        _OWNED_TMPDIRS.append(td)
    return td


def _cleanup_owned_tmpdirs():
    """atexit / signal 시 자체 추적한 tmpdir 정리."""
    if not _OWNED_TMPDIRS:
        return
    n = 0
    for td in list(_OWNED_TMPDIRS):
        try:
            if _glob_module.os.path.exists(td):
                shutil.rmtree(td, ignore_errors=True)
                n += 1
        except Exception:
            pass
    if n > 0:
        try:
            log.info(f"  [atexit] {n} 임시 디렉토리 정리")
        except Exception:
            pass


_atexit_module.register(_cleanup_owned_tmpdirs)


# macOS-specific fork-unsafe 모델 (검증된 화이트리스트)
# G-236 후속 (2026-05-29, Codex 교차검증): PyG 모델이 GE-DNN/GE-GAT → GCN/GAT 로 rename
# 됐는데 override 가 옛 이름을 들고 있어 매치 0 → GCN/GAT(meta.category=dl)가 macOS 에서
# subprocess 격리 대상이 됨(PyG GCNConv/GATv2Conv MPS-unsafe — override 가 막으려던 그것).
# 현재 등록명으로 정정.
_INPROCESS_OVERRIDE_DARWIN = frozenset({
    "GCN",                    # PyG GCNConv + MPS-unsafe (ex GE-DNN)
    "GAT",                    # PyG GATv2Conv + MPS-unsafe (ex GE-GAT)
    "TimesNet",               # subprocess SIGSEGV (검증)
    # G-261 (2026-06-13): Chronos-MultiCountry 제거 — Chronos retire.
})

# Linux: 거의 모든 모델이 subprocess 안전.
_INPROCESS_OVERRIDE_LINUX = frozenset()

# Windows: NVIDIA 드라이버 + spawn 모드. 일부 PyG 모델은 spawn 시 import 충돌.
_INPROCESS_OVERRIDE_WIN32 = frozenset()  # G-261: Chronos-MultiCountry 제거 (Chronos retire)


def _get_inprocess_override() -> frozenset:
    """현재 OS 의 in-process 강제 모델 집합."""
    if _sys_mod.platform == "darwin":
        return _INPROCESS_OVERRIDE_DARWIN
    if _sys_mod.platform == "win32":
        return _INPROCESS_OVERRIDE_WIN32
    # Linux + 기타 POSIX
    return _INPROCESS_OVERRIDE_LINUX


def _should_use_subprocess(category: str, name: str) -> bool:
    """모델별 subprocess 격리 여부 결정 (OS-aware).

    Linux:   거의 모든 DL 모델 → subprocess (fork 안정)
    macOS:   PyG/MPS fork-unsafe 4 모델 → in-process, 나머지 → subprocess
    Windows: spawn-unsafe 1 모델 → in-process, 나머지 → subprocess(spawn)
    """
    if category not in _SUBPROCESS_CATEGORIES:
        return False
    override = _get_inprocess_override()
    if name in override:
        log.info(f"  [subprocess/{_sys_mod.platform}] {name} 화이트리스트 → in-process 강제")
        return False
    return True


def get_subprocess_strategy_summary() -> dict:
    """현재 OS 의 subprocess 전략 요약 (doctor / preflight 용)."""
    override = _get_inprocess_override()
    return {
        "platform":        _sys_mod.platform,
        "default_strategy": "subprocess_popen",
        "isolation_categories": sorted(_SUBPROCESS_CATEGORIES),
        "in_process_override": sorted(override),
        "notes": {
            "darwin":  "fork() unsafe with PyG+MPS — 4 models forced in-process",
            "linux":   "fork() safe — all DL models isolated in subprocess",
            "win32":   "spawn-only — 1 model with spawn-unsafe imports forced in-process",
        }.get(_sys_mod.platform, "non-standard platform"),
    }

# 자식 프로세스에서 실행할 스크립트 (runner.py와 같은 디렉토리에 생성)
_WORKER_SCRIPT = r'''
import gc, pickle, sys, os, traceback, time
import numpy as np

args_path = sys.argv[1]
result_path = sys.argv[2]

try:
    _project_root = os.getcwd()
    sys.path.insert(0, _project_root)
    print("[WORKER] Loading args...", flush=True)
    with open(args_path, "rb") as f:
        args = pickle.load(f)
    model = args["model"]
    X_tr, y_tr = args["X_train"], args["y_train"]
    X_val, X_test = args["X_val"], args["X_test"]
    is_ts = args["is_ts"]
    y_val_len, y_test_len = args["y_val_len"], args["y_test_len"]
    save_dir = args.get("save_dir", "")
    _name = model.meta.name if hasattr(model, "meta") else "unknown"

    # G-121: subprocess 내 NaN/Inf 방어 — DL forward pass NaN 전파 방지
    _ns = {"X_tr": X_tr, "y_tr": y_tr, "X_val": X_val, "X_test": X_test}
    for _arr_name, _arr in list(_ns.items()):
        if _arr is not None and isinstance(_arr, np.ndarray):
            _bad = int(np.isnan(_arr).sum()) + int(np.isinf(_arr).sum())
            if _bad > 0:
                print(f"[WORKER] {_name} {_arr_name}: {_bad} NaN/Inf -> 0",
                      flush=True)
                _ns[_arr_name] = np.nan_to_num(_arr, nan=0.0, posinf=0.0,
                                                 neginf=0.0)
    X_tr, y_tr = _ns["X_tr"], _ns["y_tr"]
    X_val, X_test = _ns["X_val"], _ns["X_test"]

    print(f"[WORKER] {_name} training start (X={X_tr.shape}, y={y_tr.shape})",
          flush=True)
    _t0 = time.time()
    if is_ts:
        model.fit_series(y_tr)
        model._train_series = np.asarray(y_tr, dtype=float)   # G-321: base rolling_1step fallback 용
        print(f"[WORKER] {_name} fit done ({time.time()-_t0:.1f}s), "
              f"forecasting...", flush=True)
        _yv = args.get("y_val_values"); _yt = args.get("y_test_values")
        if _yv is not None and _yt is not None:
            # G-321: META classic-ts(부모가 y값 전달) → rolling-origin 1-step over [val++test]
            #   (test = train+val 조건부) = feature 모델과 동일 task(공정). 그 외는 단일원점(legacy).
            _post = np.concatenate([np.asarray(_yv, dtype=float), np.asarray(_yt, dtype=float)])
            _roll = np.asarray(model.rolling_1step(_post), dtype=float)
            val_pred = _roll[:len(_yv)]
            test_pred = _roll[len(_yv):]
        else:
            val_pred = model.forecast(y_val_len)
            test_pred = model.forecast(y_test_len)
    else:
        # : feature_names 전달 (DL 이 lag1 인덱스 찾는 데 사용)
        _fit_kw = {}
        _fnames = args.get("feature_names")
        if _fnames is not None:
            _fit_kw["feature_names"] = _fnames
        try:
            model.fit(X_tr, y_tr, **_fit_kw)
        except TypeError:
            model.fit(X_tr, y_tr)
        print(f"[WORKER] {_name} fit done ({time.time()-_t0:.1f}s), "
              f"predicting...", flush=True)
        # G-321: SARIMAX(is_ts=False=feature-path, exog) 도 y값 전달 시 rolling-origin 1-step
        #   (predict 가 y_observed 처리). 그 외 feature 모델은 y값 None → 기존 predict(실 lag=1-step).
        _yv = args.get("y_val_values"); _yt = args.get("y_test_values")
        if _yv is not None and _yt is not None:
            val_pred = model.predict(X_val, y_observed=np.asarray(_yv, dtype=float))
            test_pred = model.predict(X_test, y_observed=np.asarray(_yt, dtype=float))
        else:
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

    # : subprocess 내에서 training history 저장
    try:
        _hist = getattr(model, "_history", None)
        if _hist:
            import json as _json
            _hist_path = result_path + ".history.json"
            with open(_hist_path, "w", encoding="utf-8") as _hf:
                _json.dump({"model": _name, "history": _hist}, _hf, indent=2)
    except Exception:
        pass
    print(f"[WORKER] {_name} predict done ({time.time()-_t0:.1f}s total)",
          flush=True)

    # G-047: subprocess 내에서 모델 저장 (local class는 pickle 불가하므로)
    if save_dir:
        try:
            _save_name = _name.replace(" ", "_")
            _path = os.path.join(save_dir, f"{_save_name}.pt")
            os.makedirs(save_dir, exist_ok=True)
            model.save(_path)
        except Exception as se:
            print(f"[WORKER] model.save 실패: {se}", file=sys.stderr)

    # G-129: predictions atomic 저장 → model 별도 .model 파일
    result = {"val_pred": val_pred, "test_pred": test_pred}
    _tmp_result = result_path + ".tmp"
    with open(_tmp_result, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(_tmp_result, result_path)   # atomic on Windows/POSIX
    print(f"[WORKER] {_name} predictions saved (safe)", flush=True)

    try:
        # G-154 fix: cloudpickle 로 closure 지원 (PatchTST/iTransformer/Mamba/TimesNet/TiDE/N-BEATS/N-HiTS)
        # 이전: pickle.dump → "Can't get local object '_*Net.build.<locals>.*Model'" fail.
        # 수정: cloudpickle 사용 — closure 자동 처리.
        _model_path = result_path + ".model"
        try:
            import cloudpickle as _cp
            with open(_model_path, "wb") as mf:
                _cp.dump({"model": model}, mf, protocol=_cp.DEFAULT_PROTOCOL)
        except ImportError:
            # cloudpickle 없으면 일반 pickle (sequence 모델은 fail 가능)
            with open(_model_path, "wb") as mf:
                pickle.dump({"model": model}, mf, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

    # GPU/MPS cleanup best-effort + os._exit(0)
    import os as _os, sys as _sys
    try:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.synchronize()
                _t.cuda.empty_cache()
                try:
                    _t.cuda.ipc_collect()
                except Exception:
                    pass
            elif (hasattr(_t.backends, "mps")
                  and _t.backends.mps.is_available()):
                if hasattr(_t, "mps") and hasattr(_t.mps, "empty_cache"):
                    _t.mps.empty_cache()
        except Exception:
            pass
    except Exception:
        pass
    _sys.stdout.flush()
    _sys.stderr.flush()
    print(f"[WORKER] {_name} fast-exit (GPU cleanup attempted)", flush=True)
    _os._exit(0)

except Exception as e:
    try:
        with open(result_path, "wb") as f:
            pickle.dump({"error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()}, f)
    except Exception:
        pass
    traceback.print_exc()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif (hasattr(torch.backends, "mps")
              and torch.backends.mps.is_available()):
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()
    raise SystemExit(1)
'''


def _run_model_in_subprocess(model, X_train, y_train, X_val, X_test,
                              y_val_len, y_test_len, is_ts, name,
                              timeout=600, save_dir="",
                              stall_timeout=300, poll_interval=10,
                              feature_names=None,
                              y_val_values=None, y_test_values=None):
    """모델을 별도 subprocess.Popen으로 실행.

    적응형 타임아웃:
      - 총 timeout: 최대 실행 시간 (기본 600s)
      - stall_timeout: 로그 파일에 출력이 없으면 정지로 판단 (기본 300s)
      - 로그 파일이 계속 갱신되면 timeout까지 연장 허용
    OOM 탐지:
      - Windows: exit code != 0 + psutil 메모리 체크
      - Linux: SIGKILL (exit -9) 감지
    """
    import os as _os

    # 2026-04-26: OS-aware fast tmpdir (Linux /dev/shm) — IO 50% ↓
    # 2026-04-27: atexit cleanup 추적 (crash/SIGINT 시 자동 회수)
    tmp_dir = _track_tmpdir(_get_fast_tmpdir(prefix="mph_model_"))
    args_path = str(Path(tmp_dir) / "args.pkl")
    result_path = str(Path(tmp_dir) / "result.pkl")
    worker_path = str(Path(tmp_dir) / "_worker.py")

    with open(worker_path, "w", encoding="utf-8") as f:
        f.write(_WORKER_SCRIPT)

    args = {
        "model": model, "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "X_test": X_test, "is_ts": is_ts,
        "y_val_len": y_val_len, "y_test_len": y_test_len,
        "save_dir": save_dir,
        "feature_names": feature_names,
        # G-321: META classic-ts 만 채워짐(부모 supports_rolling_eval 게이트) → worker rolling 1-step.
        "y_val_values": y_val_values, "y_test_values": y_test_values,
    }
    with open(args_path, "wb") as f:
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)

    _sub_log_path = str(Path(tmp_dir) / "subprocess.log")
    _sub_log_f = open(_sub_log_path, "w", encoding="utf-8")
    # P0-3: 부모 프로세스의 device 선택을 자식에 명시 전파.
    # subprocess 는 기본적으로 parent env 를 상속하지만, MPH_DEVICE 가 미설정인 경우
    # 현재 가용 device 를 명시 고정해 자식이 같은 device 를 선택하도록 강제한다.
    _sub_env = _os.environ.copy()
    if "MPH_DEVICE" not in _sub_env and _sub_env.get("MPH_FORCE_CPU") != "1":
        try:
            from simulation.models.base import device_str as _dstr
            _sub_env["MPH_DEVICE"] = _dstr()
        except Exception:
            pass
    proc = subprocess.Popen(
        [sys.executable, worker_path, args_path, result_path],
        cwd=str(Path(__file__).parent.parent.parent),
        stdout=_sub_log_f,
        stderr=_sub_log_f,
        env=_sub_env,
    )

    # ── 적응형 타임아웃: polling loop ──
    _start = time.time()
    _last_log_size = 0
    _last_activity = _start
    _timed_out = False
    _stalled = False
    # BUG-HB: subprocess heartbeat — 조용한 구간에도 살아있음 신호 출력
    _heartbeat_interval = 60.0   # 60초마다 한 줄
    _last_heartbeat = _start

    while True:
        ret = proc.poll()
        if ret is not None:
            break  # 프로세스 종료

        elapsed = time.time() - _start

        # 로그 파일 활동 감지
        try:
            _cur_size = _os.path.getsize(_sub_log_path)
            if _cur_size > _last_log_size:
                _last_activity = time.time()
                _last_log_size = _cur_size
        except Exception:
            pass

        # BUG-HB: heartbeat — worker가 조용해도 60s마다 부모 로그에 한 줄
        _idle_log = time.time() - _last_activity
        if time.time() - _last_heartbeat >= _heartbeat_interval:
            try:
                _mem_mb = None
                try:
                    import psutil as _ps
                    _mem_mb = _ps.virtual_memory().available // (1024 * 1024)
                except Exception:
                    pass
                _mem_str = f", free_mem={_mem_mb}MB" if _mem_mb is not None else ""
                log.info(
                    f"  [{name}] alive... elapsed={elapsed:.0f}s / cap {timeout}s | "
                    f"log_idle={_idle_log:.0f}s (stall {stall_timeout}s){_mem_str}"
                )
            except Exception:
                pass
            _last_heartbeat = time.time()

        # stall 감지: 로그가 stall_timeout 동안 변화 없음
        _idle = time.time() - _last_activity
        if _idle > stall_timeout:
            log.warning(f"  [{name}] 정지 감지: {_idle:.0f}s 로그 무변화 → 강제 종료")
            _stalled = True
            break

        # 총 timeout 초과
        if elapsed > timeout:
            _timed_out = True
            break

        # OOM 사전 감지 (psutil) — platform-aware threshold
        #   Windows/Linux (dedicated RAM): 256MB (G-127)
        #   macOS (unified memory 16-128GB, swap-friendly): 512MB
        # 통합 메모리는 GPU 와 공유되므로 좀 더 여유를 둔다.
        try:
            import psutil
            import sys as _sys_p
            _mem = psutil.virtual_memory()
            _oom_threshold_mb = 512 if _sys_p.platform == "darwin" else 256
            if _mem.available < _oom_threshold_mb * 1024 * 1024:
                log.warning(
                    f"  [{name}] 메모리 위험: {_mem.available // (1024**2)}MB 가용"
                    f" (임계값 {_oom_threshold_mb}MB) → 강제 종료"
                )
                _timed_out = True
                break
        except ImportError:
            pass

        time.sleep(poll_interval)

    if _timed_out or _stalled:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        _reason = "정지" if _stalled else f"timeout ({timeout}s)"
        log.error(f"  [{name}] subprocess {_reason} -- 강제 종료 (elapsed={time.time()-_start:.0f}s)")
        _sub_log_f.close()
        # R-3: errors="replace" to tolerate HF progress bars / ANSI bytes
        try:
            with open(_sub_log_path, "r", encoding="utf-8", errors="replace") as lf:
                _tail = lf.readlines()[-20:]
            log.error(f"  [{name}] subprocess log (last 20 lines):\n" + "".join(_tail))
        except Exception as _log_err:
            log.error(f"  [{name}] subprocess log 읽기 실패: {type(_log_err).__name__}: {_log_err}")
        _cleanup_tmp(tmp_dir)
        return {"__failed__": True, "reason": _reason}  # G-237: real reason (정지/timeout), not guessed

    _sub_log_f.close()

    # G-130: Windows torch cleanup crash 후에도 result.pkl 이 완전히 저장되어 있으면
    # predictions 를 살린다. FoundationModelTransfer 등 slow_init DL 모델이 _os._exit(0) 직전
    # torch DLL exit-handler 에서 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) 를 던져 returncode
    # 가 3221226505 로 찍히는 false-negative 가 있음. worker 는 atomic replace 로 result.pkl 을
    # 저장한 뒤 "fast-exit" 를 찍고 exit 하므로, 그 파일이 val_pred/test_pred 를 담고 있으면
    # 크래시는 post-save 잡음으로 간주해도 안전하다.
    _salvaged = False
    if proc.returncode != 0:
        _rc = proc.returncode
        _salvage_ok = False
        if _os.path.exists(result_path):
            try:
                with open(result_path, "rb") as f:
                    _peek = pickle.load(f)
                if (
                    isinstance(_peek, dict)
                    and "error" not in _peek
                    and "val_pred" in _peek
                    and "test_pred" in _peek
                    and _peek["val_pred"] is not None
                    and _peek["test_pred"] is not None
                ):
                    try:
                        _vp = np.asarray(_peek["val_pred"])
                        _tp = np.asarray(_peek["test_pred"])
                        if _vp.size > 0 and _tp.size > 0:
                            _salvage_ok = True
                    except Exception:
                        _salvage_ok = False
            except Exception:
                _salvage_ok = False
        if _salvage_ok:
            log.warning(
                f"  [{name}] subprocess post-exit 이상 종료 (exit={_rc}) — "
                f"result.pkl 은 정상 저장됨 → predictions 살림 (salvaged)"
            )
            _salvaged = True
        else:
            # OOM 신호 감지
            if _rc == -9 or _rc == 137:  # SIGKILL (Linux OOM killer)
                _cause = "OOM (SIGKILL)"
            elif _rc < 0:
                _cause = f"signal {-_rc}"
            else:
                _cause = f"exit={_rc} (OOM 가능)"
            log.error(f"  [{name}] subprocess 비정상 종료: {_cause}")
            # R-3: HuggingFace progress bars / ANSI escapes can inject
            # non-UTF8 bytes into the subprocess log. Reading with strict
            # UTF-8 then raised UnicodeDecodeError and was silently swallowed
            # by the blanket `except Exception: pass`, hiding the real cause
            # (e.g., Chronos-MultiCountry exit=1 in 30s with no diagnostics).
            # Use `errors="replace"` so we always surface the last 20 lines.
            try:
                with open(_sub_log_path, "r", encoding="utf-8", errors="replace") as lf:
                    _tail = lf.readlines()[-20:]
                _log_text = "".join(_tail)
                if any(kw in _log_text.lower() for kw in ["memoryerror", "out of memory", "cannot allocate"]):
                    log.error(f"  [{name}] 로그에서 OOM 확인됨")
                log.error(f"  [{name}] subprocess log (last 20 lines):\n" + _log_text)
            except Exception as _log_err:
                log.error(f"  [{name}] subprocess log 읽기 실패: {type(_log_err).__name__}: {_log_err}")
            _cleanup_tmp(tmp_dir)
            return {"__failed__": True, "reason": _cause}  # G-237: real exit cause (OOM/signal/exit=N)

    # 결과 읽기
    try:
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        if "error" in result:
            log.error(f"  [{name}] subprocess 내부 에러: {result['error']}")
            _cleanup_tmp(tmp_dir)
            # G-237: propagate the REAL worker exception (KeyError/ValueError/NameError/…),
            # not the guessed "OOM 또는 timeout" that hid the augment_factor KeyError for 10h.
            return {"__failed__": True, "reason": f"worker: {result['error']}"}
        if _salvaged:
            result["_subprocess_salvaged"] = True
            result["_subprocess_exit_code"] = int(proc.returncode)
        # BUG-A fix: model 은 별도 .model 파일에 저장 → merge
        # Mac-migration: macOS 에서는 subprocess .model 사이드카 로드가 부모
        # Python (torch + MPS already initialized) 에 silent SIGSEGV 를 유발한다.
        # 증상: TinyMLP subprocess 정상 종료 후 부모 python 조용히 죽음 (no Traceback).
        # downstream 에서는 val_pred / test_pred 만 쓰므로 model 객체 불필요 — skip.
        import sys as _sys_r
        _model_path = result_path + ".model"
        if _sys_r.platform == "darwin":
            # macOS: model 사이드카 로드 스킵. 필요하면 save_models 경로에서 별도 처리.
            pass
        elif "model" not in result and _os.path.exists(_model_path):
            try:
                with open(_model_path, "rb") as mf:
                    _mdict = pickle.load(mf)
                if isinstance(_mdict, dict) and "model" in _mdict:
                    result["model"] = _mdict["model"]
            except Exception as _me:
                log.debug(f"  [{name}] model sidecar 로드 실패 (무시): {_me}")
        # : training history sidecar (DL only)
        _hist_sidecar = result_path + ".history.json"
        if _os.path.exists(_hist_sidecar):
            try:
                import json as _json
                with open(_hist_sidecar, "r", encoding="utf-8") as _hf:
                    result["history"] = _json.load(_hf).get("history", [])
            except Exception:
                pass
        _cleanup_tmp(tmp_dir)
        return result
    except Exception as e:
        log.error(f"  [{name}] 결과 읽기 실패: {e}")
        # BUG-A diagnostic: subprocess 로그 tail 도 함께 출력
        # R-3: errors="replace" to tolerate HF progress bars / ANSI bytes
        try:
            with open(_sub_log_path, "r", encoding="utf-8", errors="replace") as lf:
                _tail = lf.readlines()[-20:]
            if _tail:
                log.error(f"  [{name}] subprocess log (last 20 lines):\n" + "".join(_tail))
        except Exception as _log_err:
            log.error(f"  [{name}] subprocess log 읽기 실패: {type(_log_err).__name__}: {_log_err}")
        _cleanup_tmp(tmp_dir)
        return {"__failed__": True, "reason": f"result read failed: {e}"}  # G-237


def _cleanup_tmp(tmp_dir):
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# 실시간 진행률 파일 (progress.json)
# ══════════════════════════════════════════════════════════════

def _write_progress(save_dir: str, data: dict):
    """진행률을 JSON 파일로 저장 -- 외부에서 폴링 가능."""
    try:
        p = Path(save_dir) / "progress.json"
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # 진행률 저장 실패는 무시


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray,
                       train_actual: np.ndarray = None) -> dict:
    """평가 지표 (R², RMSE, MAE, sMAPE, MAPE) + 확장 (MASE, bias, peak metrics).

 train_actual : optional. 제공시 MASE 계산 (seasonal naive scale 기준).
 """
    n = min(len(actual), len(predicted))
    a, p = actual[:n].astype(float), predicted[:n].astype(float)

    # 기존 지표
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    denom = np.abs(a) + np.abs(p) + 1e-8
    smape = float(np.mean(2 * np.abs(a - p) / denom) * 100)
    mask = a > 0
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else None
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # 신규 지표 (보건역학 forecasting 표준)
    # 1. Bias (mean error 부호) — 과소/과대 예측 경향
    bias = float(np.mean(p - a))
    bias_pct = float(bias / (np.mean(a) + 1e-8) * 100)

    # 2. Pearson r — 추세 일관성 (R² 보다 분산 변동에 민감)
    if np.std(a) > 1e-8 and np.std(p) > 1e-8:
        pearson_r = float(np.corrcoef(a, p)[0, 1])
    else:
        pearson_r = 0.0

    # 3. Peak metrics — ILI forecasting 핵심
    peak_actual_idx = int(np.argmax(a))
    peak_pred_idx = int(np.argmax(p))
    peak_timing_err_w = int(peak_pred_idx - peak_actual_idx)  # +음수 = 늦게 예측
    peak_intensity_err_pct = float((p[peak_pred_idx] - a[peak_actual_idx])
                                     / (a[peak_actual_idx] + 1e-8) * 100)

    # 4. MASE (Mean Absolute Scaled Error, Hyndman&Koehler 2006)
    #    season=1 (random walk 기준). 1.0 미만 = naive 보다 좋음.
    mase = None
    if train_actual is not None and len(train_actual) > 1:
        ta = np.asarray(train_actual, dtype=float)
        scale = float(np.mean(np.abs(np.diff(ta))))  # naive 1-step error
        if scale > 0:
            mase = float(mae / scale)

    # 5. Direction accuracy — 상승/하강 방향 일치
    if n >= 2:
        d_actual = np.sign(np.diff(a))
        d_pred = np.sign(np.diff(p))
        direction_acc_pct = float(np.mean(d_actual == d_pred) * 100)
    else:
        direction_acc_pct = None

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "smape": round(smape, 2),
        "mape": round(mape, 2) if mape is not None else None,
        "n": n,
        # 확장
        "bias": round(bias, 4),
        "bias_pct": round(bias_pct, 2),
        "pearson_r": round(pearson_r, 4),
        "peak_timing_err_w": peak_timing_err_w,
        "peak_intensity_err_pct": round(peak_intensity_err_pct, 2),
        "mase": round(mase, 4) if mase is not None else None,
        "direction_acc_pct": round(direction_acc_pct, 2) if direction_acc_pct is not None else None,
    }


def _ar_correct_predictions(
    val_actual: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    test_actual: np.ndarray,
    max_order: int = 2,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> tuple:
    """반복 Cochrane-Orcutt AR 보정 — val 잔차로 계수 추정, test에 순차 적용.

    Parameters:
        val_actual, val_pred: 검증 세트 (AR 계수 추정용)
        test_pred, test_actual: 테스트 세트 (보정 대상)
        max_order: AR(1)~AR(max_order) 중 최적 선택
        max_iter: CO-GLS 반복 횟수
        tol: 수렴 판정 기준

    Returns:
        (corrected_test_pred, ar_info_dict)
    """
    import warnings
    from statsmodels.stats.stattools import durbin_watson

    # ── 사전 검증: NaN/Inf/상수 예측 → 보정 불가 ──
    _skip = {"corrected": False, "dw_before": np.nan, "reason": "invalid_predictions"}
    if (not np.all(np.isfinite(test_pred)) or not np.all(np.isfinite(val_pred))
            or not np.all(np.isfinite(test_actual)) or not np.all(np.isfinite(val_actual))):
        return test_pred.copy(), _skip
    if np.std(test_pred) < 1e-10 or np.std(val_pred) < 1e-10:
        return test_pred.copy(), {**_skip, "reason": "constant_predictions"}

    val_resid = val_actual[:len(val_pred)] - val_pred
    test_resid = test_actual[:len(test_pred)] - test_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig_dw = durbin_watson(test_resid)

    if not np.isfinite(orig_dw):
        return test_pred.copy(), {**_skip, "reason": "nan_dw"}
    if orig_dw >= 1.5:
        return test_pred.copy(), {"corrected": False, "dw_before": round(float(orig_dw), 4)}

    best_pred = test_pred.copy()
    best_dw = orig_dw
    best_order = 0

    for ar_order in range(1, max_order + 1):
        if len(val_resid) <= ar_order + 5:
            continue

        _vr = val_resid.copy()
        prev_rho = np.zeros(ar_order)
        corrected = test_pred.copy()  # fallback

        for _ in range(max_iter):
            X_ar = np.column_stack([
                _vr[ar_order - i - 1: len(_vr) - i - 1]
                for i in range(ar_order)
            ])
            y_ar = _vr[ar_order:]

            # degenerate 행렬 체크
            if X_ar.shape[0] < X_ar.shape[1] + 1 or np.std(X_ar) < 1e-10:
                break

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ar_coefs = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]
                if not np.all(np.isfinite(ar_coefs)):
                    break
                ar_coefs = np.clip(ar_coefs, -0.95, 0.95)
            except Exception:
                break

            if np.max(np.abs(ar_coefs - prev_rho)) < tol:
                break
            prev_rho = ar_coefs.copy()

            # 보정 적용
            corrected = test_pred.copy()
            prev_resids = list(_vr[-ar_order:])
            for t in range(len(test_pred)):
                expected_error = sum(ar_coefs[i] * prev_resids[-(i + 1)] for i in range(ar_order))
                if t == 0 and ar_order == 1:
                    pw_w = np.sqrt(max(1 - ar_coefs[0] ** 2, 0.01))
                    expected_error *= pw_w
                corrected[t] = test_pred[t] + expected_error
                actual_r = test_actual[t] - test_pred[t] if t < len(test_actual) else expected_error
                prev_resids.append(actual_r)

            corrected = np.maximum(corrected, 0)
            new_resid = test_actual[:len(corrected)] - corrected
            _vr = new_resid

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _new_resid = test_actual[:len(corrected)] - corrected
                final_dw = durbin_watson(_new_resid)
            if not np.isfinite(final_dw):
                continue
        except Exception:
            continue
        if final_dw > best_dw:
            best_pred = corrected
            best_dw = final_dw
            best_order = ar_order

    # 보정 후 R²
    ss_res_before = np.sum(test_resid ** 2)
    ss_res_after = np.sum((test_actual[:len(best_pred)] - best_pred) ** 2)
    ss_tot = np.sum((test_actual[:len(best_pred)] - test_actual[:len(best_pred)].mean()) ** 2)
    r2_before = float(1 - ss_res_before / ss_tot) if ss_tot > 0 else 0.0
    r2_after = float(1 - ss_res_after / ss_tot) if ss_tot > 0 else 0.0

    # NaN 방지
    if not np.isfinite(r2_before):
        r2_before = 0.0
    if not np.isfinite(r2_after):
        r2_after = 0.0

    return best_pred, {
        "corrected": True,
        "ar_order": best_order,
        "dw_before": round(float(orig_dw), 4),
        "dw_after": round(float(best_dw), 4) if np.isfinite(best_dw) else None,
        "r2_before": round(r2_before, 4),
        "r2_after": round(r2_after, 4),
        "r2_gain": round(r2_after - r2_before, 4),
    }


class MultiModelRunner:
    """
 48개 모델 통합 실행기 ( — AR 보정 내장).

 Parameters:
 data_size: 전체 데이터 주수 (모델 필터링용)
 exclude_categories: 제외할 범주 (예: ["physics"])
 max_models: 최대 사용 모델 수 (기본: 전체)
 target_transformer: TargetTransformer 인스턴스 (None이면 log1p 기본값)
 ar_correct: AR 잔차 보정(옛 phase 8 Cochrane-Orcutt) 적용 여부 (기본: False — 2026-06-05 은퇴,
            출력 미사용=no AR variant push; True 명시 시에만 동작)
 """

    def __init__(
        self,
        data_size: int = 341,
        exclude_categories: list[str] = None,
        max_models: int = 0,
        target_transformer=None,
        per_model_transform: dict[str, str] = None,
        per_model_features: dict[str, tuple] = None,
        skip_models: list[str] = None,
        include_only: list[str] = None,
        ar_correct: bool = False,   # 옛 phase 8 AR_correction 은퇴(2026-06-05) — 출력 미사용. True 시에만 동작
        feature_names: Optional[list[str]] = None,
    ):
        self.data_size = data_size
        self.exclude_categories = exclude_categories or []
        self.max_models = max_models
        self.skip_models = skip_models or []
        # : partial refit — if non-empty, only these model names run.
        # Used by --models CLI filter together with the R2 baseline per-model sidecar merge.
        self.include_only = list(include_only) if include_only else []
        self.ar_correct = ar_correct
        # G-231 cleanup: feature_names kept for lag1-index lookup uses
        self.feature_names = feature_names

        # 타겟 변환기 (기본: log1p)
        if target_transformer is None:
            from simulation.models.target_transform import TargetTransformer
            self.target_transformer = TargetTransformer(method="log1p")
        else:
            self.target_transformer = target_transformer

        # : 모델별 변환 전략
        self.per_model_transform = per_model_transform or {}

        # : 모델별 피처 서브셋 {name: (X_train, X_val, X_test)}
        self.per_model_features = per_model_features or {}

        # 모듈 임포트 → 레지스트리에 등록
        self._import_all_models()

    def _import_all_models(self):
        """모든 모델 모듈 임포트 (레지스트리 등록 트리거)."""
        try:
            import simulation.models.ts_models
            import simulation.models.linear_models
            import simulation.models.tree_models
            import simulation.models.dl_models
            import simulation.models.modern_ts
            import simulation.models.tft_wrapper
            import simulation.models.ensemble
            import simulation.models.pinn_model
            import simulation.models.rt_estimator
            import simulation.models.bayesian_seir
            import simulation.models.conformal
            import simulation.models.timesfm_wrapper   # TimesFM-2.5 (Chronos-2 대체, G-261)
            import simulation.models.tabpfn_wrapper     # TabPFN v2 tabular foundation (G-264)
            import simulation.models.dlinear            # DLinear (Zeng 2023, G-265)
            import simulation.models.tirex_wrapper      # TiRex xLSTM foundation (G-265)
            import simulation.models.metapop_seir
            import simulation.models.phase_ensemble
            import simulation.models.foundation_model
            import simulation.models.overseas_transfer
            import simulation.models.epi_models
            import simulation.models.graph_models
            # G-231 (2026-05-22): dl_anchored 제거 — α-blend DNN 모두 α≈0 collapse 확인 → 폐기
            # G-261 (2026-06-13): chronos_wrapper / chronos_finetune_real 제거 — Chronos retire.
            # 추가 등록자
            try:
                import simulation.models.negbin_glm  # NegBinGLM-V7
            except Exception as _ne:
                log.debug(f"  [opt] negbin_glm skip: {_ne}")
            try:
                import simulation.models.seir_forced  # SEIR-V2-Forced
            except Exception as _se:
                log.debug(f"  [opt] seir_forced skip: {_se}")
            # 2026-05-12 (사용자 5.b + epi 9): 신규 보건역학 모델 9개
            try:
                import simulation.models.cox_models       # CoxPH
            except Exception as _cx:
                log.debug(f"  [opt] cox_models skip: {_cx}")
            try:
                import simulation.models.epiestim_models  # EpiEstim
            except Exception as _ep:
                log.debug(f"  [opt] epiestim_models skip: {_ep}")
            try:
                import simulation.models.hhh4_models      # hhh4-equivalent
            except Exception as _hh:
                log.debug(f"  [opt] hhh4_models skip: {_hh}")
            try:
                import simulation.models.wallinga_teunis  # Wallinga-Teunis
            except Exception as _wt:
                log.debug(f"  [opt] wallinga_teunis skip: {_wt}")
            # renewal_models archived 2026-05-26 (Sprint D1, MERGE-drop Renewal → EpiEstim)
            try:
                import simulation.models.glarma_models    # GLARMA
            except Exception as _gl:
                log.debug(f"  [opt] glarma_models skip: {_gl}")
            try:
                import simulation.models.tsir_models      # TSIR
            except Exception as _ts:
                log.debug(f"  [opt] tsir_models skip: {_ts}")
            try:
                import simulation.models.prophet_models   # PROPHET
            except Exception as _pp:
                log.debug(f"  [opt] prophet_models skip: {_pp}")
            try:
                import simulation.models.cqr_models       # CQR-LightGBM/GBR/QuantReg (codex+gemini)
            except Exception as _ce:
                log.debug(f"  [opt] cqr_models skip: {_ce}")
            try:
                import simulation.models.ears_models      # EARS C1/C2/C3
            except Exception as _ea:
                log.debug(f"  [opt] ears_models skip: {_ea}")
        except ImportError as e:
            log.warning(f"일부 모델 모듈 임포트 실패: {e}")

    def get_available_models(self) -> list:
        """사용 가능한 모델 인스턴스 목록."""
        from simulation.models.base import REGISTRY

        models = REGISTRY.get_available(
            data_size=self.data_size,
            has_gpu=False,
            exclude_categories=self.exclude_categories + ["meta"],  # 메타는 별도 처리
        )

        # skip_models: 이름으로 특정 모델 제외
        if self.skip_models:
            models = [m for m in models if m.meta.name not in self.skip_models]

        # partial refit: --models CLI 필터가 주어졌을 때 해당 모델만 남긴다.
        # include_only = [] (기본) 이면 전체 통과.
        if self.include_only:
            keep = set(self.include_only)
            kept = [m for m in models if m.meta.name in keep]
            dropped = [m.meta.name for m in models if m.meta.name not in keep]
            log.info(f"  [include_only] 유지 {len(kept)}개: {[m.meta.name for m in kept]}")
            log.info(f"  [include_only] 제외 {len(dropped)}개: {dropped}")
            models = kept

        if self.max_models > 0:
            models = models[:self.max_models]

        return models

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        run_ensembles: bool = True,
        save_models: bool = True,
        save_dir: str = "results/trained_models",
    ) -> dict:
        """
        전체 파이프라인 실행.

        If X_val/y_val are None, internally carves the last 25% of y_train
        as validation holdout for ensemble weight learning.

        Returns:
            dict with:
                individual_results: 모델별 {name, category, val_metrics, test_metrics, val_pred, test_pred, elapsed_s}
                ensemble_results: 앙상블별 결과
                best_individual: 최고 개별 모델
                best_overall: 최고 모델 (앙상블 포함)
                summary: 요약 DataFrame
        """
        models = self.get_available_models()
        total_models = len(models)
        t_start = time.time()

        # ── 내부 검증 세트 자동 생성 (80:20 split용) ──
        _internal_val = False
        if X_val is None or y_val is None:
            _internal_val = True
            n_internal = max(int(len(y_train) * 0.25), 10)
            X_val = X_train[-n_internal:]
            y_val = y_train[-n_internal:]
            # X_train/y_train은 축소하지 않음 -- 모델은 전체 데이터로 학습

        def _fmt_time(s):
            h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
            return f"{h}시간 {m}분 {sec}초" if h else (f"{m}분 {sec}초" if m else f"{sec}초")

        # ── 타겟 변환 (분포 이동 완화) ──
        tt = self.target_transformer
        if tt.method != "none":
            # 원본 보관 (메트릭 계산 + 앙상블에서 사용)
            y_train_orig = y_train.copy()
            y_val_orig = y_val.copy()
            y_test_orig = y_test.copy()

            # 변환 적용 (학습 데이터 기준 fit)
            tt.fit(y_train)
            shift_info = tt.describe_shift(y_train, y_test)
            y_train = tt.transform(y_train)
            y_val = tt.transform(y_val)
            y_test_t = tt.transform(y_test)  # 참고용 (메트릭은 원본으로 계산)

            log.info(f"  [*] Target Transform: {tt.method}")
            log.info(f"    원본 분포: train={shift_info['original']['train_mean']:.2f}, "
                     f"test={shift_info['original']['test_mean']:.2f}, "
                     f"ratio={shift_info['original']['ratio']:.4f}")
            log.info(f"    변환 후:   train={shift_info['transformed']['train_mean']:.2f}, "
                     f"test={shift_info['transformed']['test_mean']:.2f}, "
                     f"ratio={shift_info['transformed']['ratio']:.4f}")
        else:
            y_train_orig = y_train
            y_val_orig = y_val
            y_test_orig = y_test

        log.info(f"\n{'='*60}")
        log.info(f"  MultiModelRunner -- {total_models}개 모델 실행")
        val_label = "Val(internal)" if _internal_val else "Val"
        log.info(f"  Train: {len(y_train)}주, {val_label}: {len(y_val)}주, Test: {len(y_test)}주")
        log.info(f"  Target Transform: {tt.method}")
        log.info(f"{'='*60}")

        # ── 개별 모델 실행 ──
        individual_results = {}
        val_predictions = {}
        test_predictions = {}
        elapsed_times = []  # ETA 계산용

        # G-237: the per-model "Test R2" printed below is on the HARD distribution-shifted
        # holdout (test mean ≫ train) — a diagnostic stress metric. Negative individual
        # Test R2 is EXPECTED and is NOT the champion signal. The champion DECISION metric
        # is OOF-CV WIS (MPH_BEST_BY=oof_cv); champion = 순수 best-WIS (4-criteria/g175
        # 완전 제거 2026-06-05). (progress.py:76 parses "Test R2=".)
        log.info("  [note] per-model 'Test R2' = hard distribution-shifted holdout "
                 "(diagnostic); champion = best-WIS (OOF-CV, R9 per_model_optimize).")

        for model_idx, model in enumerate(models):
            name = model.meta.name
            category = model.meta.category
            t0 = time.time()

            # G-209 (2026-05-14): R2 baseline per-model sidecar load
            # 12차 crash (32/60) 이후 13차에서 31 model 재학습 1h 낭비 → 사용자 명시
            # "통과된것은 괜찮은것 같은데 실패한 것에서만 다시 시작하자".
            # 해결: 각 모델 학습 결과 dict 를 .pkl sidecar 로 별도 저장 → 다음 run 에서
            # sidecar 존재 시 load 하고 학습 skip. .pt 와 별도 — model weights vs results.
            # 강제 재학습: MPH_PHASE4_FORCE_RETRAIN=1 (default 0 = skip if exists).
            # 14차 첫 run 은 12차 sidecar 없음 → 영향 X. 15차+ 부터 효과.
            _g209_skipped = False
            if save_dir:
                _force_retrain209 = GLOBAL.ops.phase4_force_retrain
                if not _force_retrain209:
                    try:
                        from pathlib import Path as _Path209
                        _sc_path209 = _Path209(save_dir) / f"{name.replace(' ', '_')}_phase2_result.pkl"
                        if _sc_path209.exists() and _sc_path209.stat().st_size > 0:
                            with open(_sc_path209, "rb") as _f209:
                                _saved209 = pickle.load(_f209)
                            if (isinstance(_saved209, dict)
                                and "val_pred" in _saved209 and "test_pred" in _saved209
                                and "val_metrics" in _saved209 and "test_metrics" in _saved209):
                                individual_results[name] = _saved209
                                _vp209 = np.asarray(_saved209["val_pred"], dtype=np.float64)
                                _tp209 = np.asarray(_saved209["test_pred"], dtype=np.float64)
                                if np.all(np.isfinite(_vp209)) and np.all(np.isfinite(_tp209)):
                                    val_predictions[name] = _vp209
                                    test_predictions[name] = _tp209
                                elapsed_times.append(float(_saved209.get("elapsed_s", 0.0)))
                                log.info(
                                    f"  [{name}] G-209: sidecar load → 학습 skip "
                                    f"(R²={_saved209.get('test_metrics', {}).get('r2', float('nan')):.4f})"
                                )
                                _g209_skipped = True
                    except Exception as _le209:
                        log.warning(f"  [{name}] G-209 sidecar load 실패 → 정상 학습: {_le209}")
            if _g209_skipped:
                continue  # 다음 모델로

            # 진행 바 + ETA 표시
            done = model_idx
            pct = int(done / total_models * 100)
            bar_len = 25
            filled = int(bar_len * done / total_models)
            bar = "#" * filled + "-" * (bar_len - filled)
            total_elapsed = time.time() - t_start

            if elapsed_times:
                avg_time = np.mean(elapsed_times)
                remaining = avg_time * (total_models - done)
                eta_str = f"ETA {_fmt_time(remaining)}"
            else:
                eta_str = "ETA ..."

            try:
                print(f"\r  [{bar}] {pct}% ({done}/{total_models}) {name:18s} training... "
                      f"[{_fmt_time(total_elapsed)} | {eta_str}]     ", end="", flush=True)
            except OSError:
                pass  # Windows console encoding issue -- skip progress bar

            # 학습 시작 시 진행률 파일 업데이트
            _write_progress(save_dir, {
                "phase": "individual",
                "current": model_idx,
                "total": total_models,
                "model": name,
                "status": "training",
                "total_elapsed_s": round(total_elapsed, 2),
                "completed": {k: round(v["test_metrics"]["r2"], 4)
                              for k, v in individual_results.items()
                              if "test_metrics" in v},
            })

            try:
                # : 모델별 변환 전략 확인
                model_tf = self.per_model_transform.get(name, None)
                use_orig = (model_tf == "none") and (tt.method != "none")

                if use_orig:
                    y_tr_m = y_train_orig
                else:
                    y_tr_m = y_train

                # : 모델별 피처 서브셋 (없으면 기본 X 사용)
                if name in self.per_model_features:
                    X_tr_m, X_val_m, X_test_m = self.per_model_features[name]
                else:
                    X_tr_m, X_val_m, X_test_m = X_train, X_val, X_test

                # G-044: 메모리 사전 체크 -- 1.2GB 미만이면 강제 GC 후 재확인
                try:
                    import psutil
                    _mem = psutil.virtual_memory()
                    _free_mb = _mem.available / (1024**2)
                    if _free_mb < 1200:
                        log.warning(f"  [메모리 경고] {name}: {_free_mb:.0f}MB 가용 -- 강제 GC 실행")
                        gc.collect()
                        gc.collect()  # 2회: 순환 참조 해제
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                                    torch.mps.empty_cache()
                        except Exception:
                            pass  # torch 없거나 device 호출 실패 — 무시
                        _mem2 = psutil.virtual_memory()
                        _free_mb2 = _mem2.available / (1024**2)
                        if _free_mb2 < 800:
                            raise MemoryError(f"가용 메모리 부족: {_free_mb2:.0f}MB < 800MB")
                        log.info(f"  [메모리 회복] GC 후 {_free_mb2:.0f}MB 가용")
                except ImportError:
                    pass

                # G-044: DL/modern_ts 모델은 subprocess 격리 (OOM 방지)
                # Mac-migration: macOS PyG/특정 DL 은 fork 시 SIGSEGV → in-process 강제.
                is_ts = hasattr(model, 'fit_series')
                _use_subp = _should_use_subprocess(category, name)
                if _use_subp:
                    log.info(f"  [{name}] subprocess 격리 모드로 실행")
                    # R-5 (2026-04-21): time-tier system based on observed
                    # elapsed times from simulation/logs/train_20260421_080345.log
                    # (the full 54-model run).
                    #
                    # Observed P2.1/P2.2 max elapsed per model (seconds):
                    #   iTransformer            1951 s  ← exceeded old 1800 s cap
                    #   Mamba                   1665 s
                    #   TFT                     1110 s
                    #   TFT-pf                   676 s
                    #   TabularDNN               105 s
                    #   Chronos-2-FT              60 s
                    #   TabularDNN-Lite           45 s
                    #   DeepAR-pf                 30 s
                    #   Chronos-2                 30 s
                    #   Chronos-2-FT-Real         15 s
                    #   FoundationModelTransfer   15 s
                    #   GE-DNN / GE-GAT      (crashed, bug R-1/R-1b; silent
                    #                             final fit observed ≥300 s in
                    #                             verify_v22_7_fixes.log)
                    #   Chronos-MultiCountry     (crashed, bug R-2; T5 HF pipeline
                    #                             load + multi-country fetch)
                    #
                    # Policy: total-timeout ≥ 2× observed max (safety margin so
                    # a single unlucky Optuna seed can't exceed budget); stall-
                    # timeout ≥ 2× typical silent-epoch gap.
                    #
                    # TIER 3 (ultra, 3 h total / 30 min stall):
                    #   (G-261 2026-06-13: 비어 있음 — Chronos 전 변형 retire. ultra-tier 모델 없음.)
                    # TIER 2 (heavy, 2 h total / 20 min stall):
                    #   Heavy attention/recurrent DL with long silent epochs.
                    #     iTransformer, Mamba, TFT, TFT-pf, FoundationModel,
                    #     GE-DNN / GE-GAT (R-4).
                    # TIER 1 (optuna-moderate, 1 h total / 10 min stall):
                    #   "*-Optuna" models with many trials but light per-trial.
                    # TIER 0 (light, 30 min total / 5 min stall):
                    #   Simple MLPs, PINN, DeepAR-pf, Chronos-2.
                    # G-261 (2026-06-13): Chronos 전 변형 retire — ultra-tier(외부 HF download +
                    #   multi-context fetch / T5 fine-tune) 모델 부재 → _ultra 항상 False.
                    _ultra = False
                    _heavy = _ultra or any(kw in name for kw in (
                        "iTransformer", "Mamba",
                        "FoundationModel",
                        "GE-DNN",      # R-4: covers GE-DNN and GE-GAT
                    ))
                    _moderate = ("Optuna" in name)

                    if _ultra:
                        _sub_timeout = 10800   # 3 h
                        _stall_timeout = 1800   # 30 min
                    elif _heavy:
                        _sub_timeout = 7200    # 2 h  (iTransformer observed 32 m)
                        _stall_timeout = 1200   # 20 min
                    elif _moderate:
                        _sub_timeout = 3600    # 1 h
                        _stall_timeout = 600    # 10 min
                    else:
                        _sub_timeout = 1800    # 30 min
                        _stall_timeout = 300    # 5 min
                    # R-5 diagnostic: surface the tier pick so the training
                    # log is auditable when a stall/timeout fires.
                    _tier_name = (
                        "T3-ultra" if _ultra
                        else "T2-heavy" if _heavy
                        else "T1-optuna" if _moderate
                        else "T0-light"
                    )
                    log.info(
                        f"  [{name}] time-tier={_tier_name} "
                        f"total={_sub_timeout}s stall={_stall_timeout}s"
                    )
                    # G-321: META classic-ts(ARIMA/SARIMA/SARIMAX/Theta/FluSight) 만 y값 전달 →
                    #   worker 가 rolling-origin 1-step(공정 평가). 그 외는 None → 단일원점(legacy).
                    from simulation.models.base import (
                        supports_rolling_eval as _sre,
                        supports_baseline_rolling as _sbr,
                    )
                    _rollv = _sre(model) or _sbr(model)   # G-327c: foundation/pf baseline-only rolling
                    sub_result = _run_model_in_subprocess(
                        model, X_tr_m, y_tr_m, X_val_m, X_test_m,
                        y_val_len=len(y_val), y_test_len=len(y_test),
                        is_ts=is_ts, name=name, timeout=_sub_timeout,
                        save_dir=save_dir if save_models else "",
                        stall_timeout=_stall_timeout, poll_interval=15,
                        feature_names=self.feature_names,
                        # G-321: RAW(y_*_orig) — META classic-ts 는 항상 raw 학습(worker 도 y_tr_m=raw).
                        y_val_values=(y_val_orig if _rollv else None),
                        y_test_values=(y_test_orig if _rollv else None),
                    )
                    if sub_result is None or sub_result.get("__failed__"):
                        # G-237: surface the REAL failure reason (worker exc / timeout /
                        # OOM / read-fail) instead of the guessed "OOM 또는 timeout".
                        _why = sub_result.get("reason", "unknown") if sub_result else "unknown (sub_result=None)"
                        raise RuntimeError(f"subprocess 실패: {_why}")
                    val_pred = sub_result["val_pred"]
                    test_pred = sub_result["test_pred"]
                    # G-047: subprocess에서 model 반환 가능하면 사용, 아니면 원본 유지
                    # (local class 모델은 pickle 불가 → subprocess에서 직접 저장)
                    _sub_model_saved = "model" not in sub_result
                    model = sub_result.get("model", model)
                else:
                    _sub_model_saved = False
                    # 경량 모델: 직접 실행 (feature_names 전달)
                    _fit_kwargs = {}
                    if self.feature_names is not None:
                        _fit_kwargs["feature_names"] = self.feature_names
                    if is_ts:
                        model.fit_series(y_tr_m)
                        model._train_series = np.asarray(y_tr_m, dtype=float)   # G-321: base rolling fallback
                        from simulation.models.base import (
                            supports_rolling_eval as _sre,
                            supports_baseline_rolling as _sbr,
                        )
                        if _sre(model) or _sbr(model):   # G-327c: +foundation(TimesFM/TiRex/DLinear)
                            # G-321: META classic-ts → rolling-origin 1-step over [val++test] (공정).
                            #   y_*_orig = RAW(pre-transform). META 는 항상 raw 학습(use_orig 또는
                            #   tt.method=none) → y_observed 도 raw. y_val(=transform됨)이 아니라 orig.
                            _post = np.concatenate([np.asarray(y_val_orig, dtype=float),
                                                    np.asarray(y_test_orig, dtype=float)])
                            _roll = np.asarray(model.rolling_1step(_post), dtype=float)
                            val_pred = _roll[:len(y_val_orig)]
                            test_pred = _roll[len(y_val_orig):]
                        else:
                            val_pred = model.forecast(len(y_val))
                            test_pred = model.forecast(len(y_test))
                    else:
                        try:
                            model.fit(X_tr_m, y_tr_m, **_fit_kwargs)
                        except TypeError:
                            model.fit(X_tr_m, y_tr_m)
                        from simulation.models.base import (
                            supports_rolling_eval as _sre,
                            supports_baseline_rolling as _sbr,
                        )
                        if _sre(model) or _sbr(model):   # G-327c: +pf(N-BEATS/N-HiTS/TiDE) predict(y_observed)
                            # G-321: SARIMAX(feature-path, exog) rolling-origin 1-step (predict y_observed).
                            #   y_*_orig = RAW(META 는 항상 raw 학습) → y_observed 도 raw.
                            val_pred = model.predict(X_val_m, y_observed=np.asarray(y_val_orig, dtype=float))
                            test_pred = model.predict(X_test_m, y_observed=np.asarray(y_test_orig, dtype=float))
                        else:
                            val_pred = model.predict(X_val_m)
                            test_pred = model.predict(X_test_m)

                # 역변환: log1p 모델만 inverse_transform
                if (not use_orig) and tt.method != "none":
                    val_pred = tt.inverse_transform(val_pred)
                    test_pred = tt.inverse_transform(test_pred)

                val_pred = np.maximum(val_pred, 0)
                test_pred = np.maximum(test_pred, 0)

                # FIX: raw-output sanity clip. Raw DL output can
                # blow up to 10³–10⁶ when OOD features hit inverse transforms
                # (sqrt→square, log1p→expm1). Clipping to 3× train_max bounds
                # the explosion before downstream clipping sees it.
                _pre_cap = float(np.max(y_train_orig)) * 3.0
                _v_over = int(np.sum(val_pred > _pre_cap))
                _t_over = int(np.sum(test_pred > _pre_cap))
                if _v_over + _t_over > 0:
                    log.info(
                        f"  [{name}] raw clip → [0, {_pre_cap:.1f}] "
                        f"({_v_over} val + {_t_over} test samples)"
                    )
                    val_pred = np.clip(val_pred, 0, _pre_cap)
                    test_pred = np.clip(test_pred, 0, _pre_cap)

                # G-231 (2026-05-22) + 2026-05-26 archive: the universal
                # post-proc residual α-blend (OLS α → blend) for DL/TSF/Graph
                # is permanently removed. Its module was moved to
                # `simulation/_archive/anchor_deprecated_20260526/`. The
                # α≈0 collapse made it a no-op; only the raw/pipeline clips
                # below remain.

                # D-1: pipeline-level clip = train y_max * 2.5 (was 1.5)
                #   1.5x → 100.4 는 test 피크 (~100) 와 경계가 겹쳐 정상 예측까지
                #   깎음. 2.5x ≈ 167 은 2.5배 외삽 여유를 주되 명백한 폭주는 가둠.
                _y_clip_max = float(np.max(y_train_orig)) * 2.5
                if np.any(test_pred > _y_clip_max) or np.any(val_pred > _y_clip_max):
                    log.warning(f"  [{name}] 예측 범위 초과 → clip(0, {_y_clip_max:.1f})")
                    val_pred = np.clip(val_pred, 0, _y_clip_max)
                    test_pred = np.clip(test_pred, 0, _y_clip_max)

                # NaN/Inf 보정: 평균으로 대체
                for _arr, _label in [(val_pred, "val"), (test_pred, "test")]:
                    _bad = ~np.isfinite(_arr)
                    if np.any(_bad):
                        _fill = float(np.nanmean(_arr[~_bad])) if np.any(~_bad) else float(np.mean(y_train_orig))
                        _arr[_bad] = _fill
                        log.warning(f"  [{name}] {_label}: {_bad.sum()}개 NaN/Inf → {_fill:.2f}로 대체")

                # 메트릭은 항상 원본 스케일에서 계산
                val_metrics = _compute_metrics(y_val_orig, val_pred)
                test_metrics = _compute_metrics(y_test_orig, test_pred)
                elapsed = time.time() - t0
                elapsed_times.append(elapsed)

                # : training history (subprocess sidecar OR model._history)
                _history = None
                try:
                    if _use_subp:
                        _sr = locals().get("sub_result") or {}
                        _history = _sr.get("history")
                    if _history is None and hasattr(model, "_history"):
                        _history = getattr(model, "_history")
                except Exception:
                    _history = None

                individual_results[name] = {
                    "name": name,
                    "category": category,
                    "level": model.meta.level,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "val_pred": val_pred.tolist(),
                    "test_pred": test_pred.tolist(),
                    "elapsed_s": round(elapsed, 2),
                    "history": _history,  # list[{epoch,train_loss,val_loss,lr}] or None
                }
                # NaN/Inf 예측은 앙상블에 포함하지 않음
                if np.all(np.isfinite(val_pred)) and np.all(np.isfinite(test_pred)):
                    val_predictions[name] = val_pred
                    test_predictions[name] = test_pred
                else:
                    log.warning(f"  [{name}] NaN/Inf 예측 → 앙상블 제외")

                # G-209 (2026-05-14): R2 baseline per-model sidecar SAVE
                # individual_results[name] dict 전체 (metrics + predictions + history) 를
                # .pkl sidecar 로 별도 저장 → 다음 run 의 G-209 load path 가 사용.
                # .pt 는 model weights 만; sidecar = 학습 결과 dict.
                # 14차 가 sidecar 안 남기면 15차 가 14차 결과 재사용 — 점진적 robust.
                if save_dir:
                    try:
                        from pathlib import Path as _Path209s
                        _sc_path209s = _Path209s(save_dir) / f"{name.replace(' ', '_')}_phase2_result.pkl"
                        with open(_sc_path209s, "wb") as _f209s:
                            pickle.dump(individual_results[name], _f209s,
                                        protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as _se209:
                        log.warning(f"  [{name}] G-209 sidecar save 실패 (계속 진행): {_se209}")

                # 모델 저장 (G-047: subprocess에서 이미 저장했으면 skip)
                # G-179 (2026-05-05): 학습 종료 직후 .pt 무결성 audit
                #   size>0 + magic bytes (zip/pickle) + load 1회 시도
                #   → fail 시 log.warning + 재시도 1회
                if save_models and not (_use_subp and _sub_model_saved):
                    try:
                        from pathlib import Path
                        model_path = Path(save_dir) / f"{name.replace(' ', '_')}.pt"
                        model.save(str(model_path))

                        # G-179 audit
                        _audit_ok = False
                        if model_path.exists() and model_path.stat().st_size > 0:
                            try:
                                with open(model_path, 'rb') as _af:
                                    _magic = _af.read(8)
                                _is_valid = (_magic[:4] == b'PK\x03\x04' or _magic[:2] in (b'\x80\x05', b'\x80\x04', b'\x80\x03'))
                                if _is_valid:
                                    _audit_ok = True
                                else:
                                    log.warning(f"  [{name}] G-179: .pt magic invalid {_magic.hex()[:16]} — 재저장 시도")
                            except Exception as _ae:
                                log.warning(f"  [{name}] G-179 audit fail: {_ae}")
                        else:
                            log.warning(f"  [{name}] G-179: .pt EMPTY/missing — 재저장 시도")

                        if not _audit_ok:
                            # 재시도 1회
                            try:
                                model.save(str(model_path))
                                if model_path.exists() and model_path.stat().st_size > 0:
                                    log.info(f"  [{name}] G-179 재저장 성공: {model_path.stat().st_size} bytes")
                                else:
                                    log.error(f"  [{name}] G-179 재저장 후에도 EMPTY — 모델 손상")
                            except Exception as _re:
                                log.error(f"  [{name}] G-179 재저장 실패: {_re}")
                    except Exception as se:
                        log.debug(f"  [{name}] 모델 저장 실패: {se}")

                # 완료 표시 (줄바꿈으로 진행 바 아래에)
                star = " *" if test_metrics['r2'] >= 0.90 else ""
                try:
                    print(f"\r  OK [{model_idx+1}/{total_models}] {name:18s} "
                          f"Test R2={test_metrics['r2']:7.4f} RMSE={test_metrics['rmse']:8.4f} "
                          f"({_fmt_time(elapsed)}){star}                    ")
                except OSError:
                    log.info(f"  [{name}] R2={test_metrics['r2']:.4f} RMSE={test_metrics['rmse']:.4f}")

                # 실시간 진행률 파일 업데이트
                _write_progress(save_dir, {
                    "phase": "individual",
                    "current": model_idx + 1,
                    "total": total_models,
                    "model": name,
                    "r2": test_metrics['r2'],
                    "rmse": test_metrics['rmse'],
                    "elapsed_s": round(elapsed, 2),
                    "total_elapsed_s": round(time.time() - t_start, 2),
                    "status": "completed",
                    "completed": {k: round(v["test_metrics"]["r2"], 4)
                                  for k, v in individual_results.items()
                                  if "test_metrics" in v},
                })
            except Exception as e:
                elapsed = time.time() - t0
                elapsed_times.append(elapsed)
                try:
                    print(f"\r  FAIL [{model_idx+1}/{total_models}] {name:18s} "
                          f"err: {type(e).__name__}: {str(e)[:60]} ({_fmt_time(elapsed)})                    ")
                except OSError:
                    log.error(f"  [{name}] FAIL: {type(e).__name__}: {str(e)[:80]}")
                individual_results[name] = {
                    "name": name,
                    "category": category,
                    "level": model.meta.level,
                    "error": str(e),
                    "error_type": type(e).__name__,  # G-237: structured classification (not guessed)
                    "elapsed_s": round(elapsed, 2),
                }
            finally:
                # : 공격적 메모리 정리 -- OOM 연쇄 장애 방지
                try:
                    del model
                except NameError:
                    pass
                gc.collect()
                gc.collect()  # 2회 호출: 순환 참조 해제
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        # macOS Apple Silicon: MPS 캐시 정리
                        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                    # CPU 텐서 캐시도 정리
                    if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
                        torch._C._cuda_clearCublasWorkspaces()
                except (ImportError, AttributeError):
                    pass
                import ctypes, sys
                if sys.platform.startswith("linux"):
                    try:
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except (OSError, AttributeError):
                        pass
                elif sys.platform == "win32":
                    try:
                        kernel32 = ctypes.windll.kernel32
                        heap = kernel32.GetProcessHeap()
                        kernel32.HeapCompact(heap, 0)
                    except (OSError, AttributeError):
                        pass
                # darwin: malloc_trim / HeapCompact 없음 — gc.collect() 만으로 충분

        # 개별 모델 완료 표시
        # G-174 (2026-05-04 fix): total_models == 0 (ensemble 카테고리 = base 모델 X)
        # 시 for loop 안 돌아 `bar_len` 정의 안 됨 → UnboundLocalError. default 25 강제.
        total_elapsed = time.time() - t_start
        bar_len = locals().get('bar_len', 25)
        try:
            print(f"\r  [{'#'*bar_len}] 100% ({total_models}/{total_models}) "
                  f"Individual done! [{_fmt_time(total_elapsed)}]                    ")
        except OSError:
            log.info(f"  Individual models done. [{_fmt_time(total_elapsed)}]")
        log.info("")

        # ── 앙상블 실행 ──
        ensemble_results = {}
        if run_ensembles and len(val_predictions) >= 2:
            log.info(f"\n  --- 앙상블 학습 ({len(val_predictions)}개 모델) ---")

            from simulation.models.ensemble import (
                InverseRMSEEnsemble, StackingEnsemble, BlendingEnsemble,
                BMAEnsemble, NNLSEnsemble, NNLSFilteredEnsemble, TemporalWeightEnsemble,
                DiversityEnsemble, SelectiveBMAEnsemble,
                ResidualCorrectedEnsemble, AdaptiveWeightEnsemble,
            )

            # G-283 (2026-06-16, 3자 감사): NNLSFilteredEnsemble(=Ensemble-NNLS-Filtered, G-169 등록)이
            #   하드코딩 list 에서 누락 → NEVER BUILT/EVALUATED(active 53 인데 R2 baseline 미생성). 추가.
            ensemble_classes = [
                InverseRMSEEnsemble,
                StackingEnsemble,
                BlendingEnsemble,
                BMAEnsemble,
                NNLSEnsemble,
                NNLSFilteredEnsemble,   # G-283: 누락 복구
                TemporalWeightEnsemble,
                DiversityEnsemble,
                SelectiveBMAEnsemble,
                ResidualCorrectedEnsemble,
                AdaptiveWeightEnsemble,
            ]

            total_ens = len(ensemble_classes)
            ens_elapsed_times = []
            t_ens_start = time.time()

            for ens_idx, EnsembleCls in enumerate(ensemble_classes):
                ens_name = EnsembleCls.meta.name
                t0 = time.time()

                # 앙상블 진행 바 + ETA
                done_ens = ens_idx
                pct_ens = int(done_ens / total_ens * 100)
                filled_ens = int(bar_len * done_ens / total_ens)
                bar_ens = "#" * filled_ens + "-" * (bar_len - filled_ens)
                ens_total_elapsed = time.time() - t_ens_start

                if ens_elapsed_times:
                    avg_ens_time = np.mean(ens_elapsed_times)
                    remaining_ens = avg_ens_time * (total_ens - done_ens)
                    eta_ens_str = f"ETA {_fmt_time(remaining_ens)}"
                else:
                    eta_ens_str = "ETA ..."

                try:
                    print(f"\r  [{bar_ens}] {pct_ens}% ({done_ens}/{total_ens}) {ens_name:22s} "
                          f"training... [{_fmt_time(ens_total_elapsed)} | {eta_ens_str}]     ",
                          end="", flush=True)
                except OSError:
                    pass

                try:
                    ens = EnsembleCls()
                    # 앙상블은 원본 스케일 y + 원본 스케일 predictions 사용
                    ens.fit(
                        X_train, y_train_orig,
                        val_predictions=val_predictions,
                        val_actual=y_val_orig,
                    )

                    # Validation 예측
                    val_ens_pred = ens.predict(
                        X_val, model_predictions=val_predictions,
                    )
                    # Test 예측
                    test_ens_pred = ens.predict(
                        X_test, model_predictions=test_predictions,
                    )

                    val_m = _compute_metrics(y_val_orig, val_ens_pred)
                    test_m = _compute_metrics(y_test_orig, test_ens_pred)
                    elapsed = time.time() - t0
                    ens_elapsed_times.append(elapsed)

                    ensemble_results[ens_name] = {
                        "name": ens_name,
                        "category": "meta",
                        "val_metrics": val_m,
                        "test_metrics": test_m,
                        "val_pred": val_ens_pred.tolist(),
                        "test_pred": test_ens_pred.tolist(),
                        "elapsed_s": round(elapsed, 2),
                    }
                    if hasattr(ens, 'weights'):
                        ensemble_results[ens_name]["weights"] = ens.weights

                    star_ens = " *" if test_m['r2'] >= 0.90 else ""
                    try:
                        print(f"\r  OK [{ens_idx+1}/{total_ens}] {ens_name:22s} "
                              f"Test R2={test_m['r2']:7.4f} RMSE={test_m['rmse']:8.4f} "
                              f"({_fmt_time(elapsed)}){star_ens}                    ")
                    except OSError:
                        log.info(f"  [{ens_name}] R2={test_m['r2']:.4f} RMSE={test_m['rmse']:.4f}")
                except Exception as e:
                    elapsed = time.time() - t0
                    ens_elapsed_times.append(elapsed)
                    try:
                        print(f"\r  FAIL [{ens_idx+1}/{total_ens}] {ens_name:22s} "
                              f"err: {type(e).__name__}: {str(e)[:60]} ({_fmt_time(elapsed)})                    ")
                    except OSError:
                        log.error(f"  [{ens_name}] FAIL: {type(e).__name__}: {str(e)[:80]}")
                    # P0-2: 실패 앙상블도 집계에 포함되도록 error 기록
                    ensemble_results[ens_name] = {
                        "name": ens_name,
                        "category": "meta",
                        "error": str(e),
                        "error_type": type(e).__name__,  # G-237
                        "elapsed_s": round(elapsed, 2),
                    }

            # 앙상블 완료 표시
            # G-174 (2026-05-04): bar_len safety (total_ens == 0 시 동일 issue)
            ens_total_elapsed = time.time() - t_ens_start
            bar_len = locals().get('bar_len', 25)
            try:
                print(f"\r  [{'#'*bar_len}] 100% ({total_ens}/{total_ens}) "
                      f"Ensemble done! [{_fmt_time(ens_total_elapsed)}]                    ")
            except OSError:
                log.info(f"  Ensemble done. [{_fmt_time(ens_total_elapsed)}]")

        # ── AR 잔차 보정 (전체 모델) ──
        ar_correction_report = {}
        if self.ar_correct:
            log.info(f"\n  --- AR 잔차 보정 (반복 Cochrane-Orcutt GLS) ---")
            _all_preds = {**individual_results, **ensemble_results}
            for name, r in _all_preds.items():
                if "test_pred" not in r or "val_pred" not in r:
                    continue
                try:
                    _vp = np.array(r["val_pred"])
                    _tp = np.array(r["test_pred"])
                    _corr_pred, _ar_info = _ar_correct_predictions(
                        val_actual=y_val_orig, val_pred=_vp,
                        test_pred=_tp, test_actual=y_test_orig,
                    )
                    ar_correction_report[name] = _ar_info
                    if _ar_info.get("corrected", False):
                        # 보정된 예측 + 메트릭 업데이트
                        _corr_metrics = _compute_metrics(y_test_orig, _corr_pred)
                        if name in individual_results:
                            individual_results[name]["test_pred_ar"] = _corr_pred.tolist()
                            individual_results[name]["test_metrics_ar"] = _corr_metrics
                            individual_results[name]["ar_info"] = _ar_info
                        elif name in ensemble_results:
                            ensemble_results[name]["test_pred_ar"] = _corr_pred.tolist()
                            ensemble_results[name]["test_metrics_ar"] = _corr_metrics
                            ensemble_results[name]["ar_info"] = _ar_info
                except Exception as e:
                    log.debug(f"  [{name}] AR 보정 실패: {e}")

            _n_corrected = sum(1 for v in ar_correction_report.values() if v.get("corrected"))
            log.info(f"  AR 보정 적용: {_n_corrected}/{len(ar_correction_report)}개 모델 (DW<1.5)")
            for _n, _ai in sorted(ar_correction_report.items(),
                                   key=lambda x: x[1].get("dw_after", 0), reverse=True):
                if _ai.get("corrected"):
                    log.info(f"    {_n:25s} DW {_ai['dw_before']:.3f}→{_ai['dw_after']:.3f} "
                             f"R² {_ai['r2_before']:.4f}→{_ai['r2_after']:.4f} "
                             f"(AR{_ai.get('ar_order', '?')})")

        # ── 최고 모델 선정 ──
        all_results = {}
        for name, r in individual_results.items():
            if "test_metrics" in r:
                all_results[name] = r["test_metrics"]["rmse"]
        for name, r in ensemble_results.items():
            if "test_metrics" in r:
                all_results[name] = r["test_metrics"]["rmse"]

        best_individual = min(
            ((n, r) for n, r in individual_results.items() if "test_metrics" in r),
            key=lambda x: x[1]["test_metrics"]["rmse"],
            default=(None, None),
        )[0]

        best_overall = min(all_results, key=all_results.get) if all_results else None

        # G-237: these rank by Test_RMSE on the HARD holdout (diagnostic only), NOT the
        # OOF-CV champion (MPH_BEST_BY=oof_cv) — champion = best-WIS (R9 per_model_optimize, no gate).
        log.info(f"\n  🏆 Best Individual (by hard-holdout Test_RMSE): {best_individual}")
        log.info(f"  🏆 Best Overall (by hard-holdout Test_RMSE): {best_overall}")

        # ── 요약 테이블 ──
        summary_rows = []
        for name, r in {**individual_results, **ensemble_results}.items():
            if "test_metrics" not in r:
                continue
            _ar_m = r.get("test_metrics_ar", {})
            summary_rows.append({
                "Model": name,
                "Category": r.get("category", ""),
                # G-237 provenance: Test_* below are on the HARD distribution-shifted
                # holdout (test mean ≫ train) — negative R² for weak models is EXPECTED,
                # not a regression. The champion DECISION metric is OOF-CV
                # (MPH_BEST_BY=oof_cv) gated in R9 per_model_optimize + 4-criteria, NOT these columns.
                "decision_metric": "oof_cv",
                "Test_R2_split": "hard_holdout_shifted",
                "Val_R2": r["val_metrics"]["r2"],
                "Val_RMSE": r["val_metrics"]["rmse"],
                "Test_R2": r["test_metrics"]["r2"],
                "Test_RMSE": r["test_metrics"]["rmse"],
                "Test_MAE": r["test_metrics"]["mae"],
                "Test_MAPE": r["test_metrics"].get("mape"),
                "Test_sMAPE": r["test_metrics"].get("smape"),
                "AR_R2": _ar_m.get("r2"),
                "AR_RMSE": _ar_m.get("rmse"),
                "Time_s": r.get("elapsed_s", 0),
            })
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty and "Test_RMSE" in summary_df.columns:
            summary_df = summary_df.sort_values("Test_RMSE")

        # ── P0-2: 실패 모델 집계 리포트 ──
        # subprocess "Ran out of input", OOM, timeout 등으로 silently drop 된
        # 모델들을 사후 추적 가능하도록 individual/ensemble 양쪽에서 error 키가
        # 있는 항목을 모아 카운트/로그. ensemble 의 가중치가 정상 모델만으로
        # 재정규화되어 전체 R² 가 내려가도 원인이 보이게 한다.
        failed_individual = [
            {
                "name": k,
                "category": v.get("category", ""),
                "error": str(v.get("error", ""))[:200],
                "elapsed_s": v.get("elapsed_s", 0.0),
            }
            for k, v in individual_results.items()
            if "error" in v and "test_pred" not in v
        ]
        failed_ensemble = [
            {
                "name": k,
                "error": str(v.get("error", ""))[:200],
                "elapsed_s": v.get("elapsed_s", 0.0),
            }
            for k, v in ensemble_results.items()
            if "error" in v and "test_pred" not in v
        ]
        _n_fail = len(failed_individual) + len(failed_ensemble)
        _n_attempted = len(individual_results) + len(ensemble_results)
        _fail_rate = _n_fail / _n_attempted if _n_attempted else 0.0

        log.info("")
        log.info(f" === 실패 모델 집계 (P0-2) ===")
        log.info(f"  전체 시도: {_n_attempted}개 / 실패: {_n_fail}개 ({_fail_rate*100:.1f}%)")
        if failed_individual:
            log.warning(f"  개별 모델 실패 ({len(failed_individual)}개):")
            for _f in failed_individual:
                log.warning(f"    ✗ {_f['name']:22s} [{_f['category']}] "
                            f"err={_f['error'][:120]}")
        if failed_ensemble:
            log.warning(f"  앙상블 실패 ({len(failed_ensemble)}개):")
            for _f in failed_ensemble:
                log.warning(f"    ✗ {_f['name']:22s} err={_f['error'][:120]}")
        # 실패율 20% 초과 시 CRITICAL 로깅 (앙상블 왜곡 경고)
        if _fail_rate > 0.20 and _n_attempted >= 5:
            log.error(
                f"  [CRITICAL] 실패율 {_fail_rate*100:.1f}% > 20% -- "
                f"앙상블 가중치가 남은 모델만으로 재정규화됨. "
                f"리더보드 해석 시 주의."
            )

        # ── 메모리 해제: 원본 y 배열 + 중간 예측 딕셔너리 ──
        del y_train_orig, y_val_orig, y_test_orig
        del val_predictions, test_predictions
        gc.collect()

        return {
            "individual_results": individual_results,
            "ensemble_results": ensemble_results,
            "best_individual": best_individual,
            "best_overall": best_overall,
            "summary": summary_df,
            "n_models_run": len([k for k, v in individual_results.items() if "test_pred" in v]),
            "n_ensembles_run": len(ensemble_results),
            "ar_correction": ar_correction_report,
            "failed_models": {
                "individual": failed_individual,
                "ensemble": failed_ensemble,
                "total": _n_fail,
                "attempted": _n_attempted,
                "fail_rate": round(_fail_rate, 4),
            },
        }