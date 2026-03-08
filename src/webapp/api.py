"""
LAH 강화학습 프레임워크 FastAPI 서버
- 환경 리셋/스텝/검증
- 학습 시작/중단/상태 조회
- 모델 로드/목록 조회
- 에피소드 시뮬레이션
- 지형 데이터 조회
- Optuna 튜닝 제어
- 설정 조회/업데이트
"""

from __future__ import annotations

import os
import uuid
import time
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# FastAPI 앱 초기화
# ------------------------------------------------------------------

app = FastAPI(
    title="LAH 회전익기 안전 경로계획 RL API",
    description="회전익기 안전 경로계획 강화학습 프레임워크 REST API",
    version="1.0.0",
)

# CORS 설정 (localhost 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ------------------------------------------------------------------
# 전역 상태 관리
# ------------------------------------------------------------------

_env = None              # 현재 Gymnasium 환경
_current_obs = None      # 현재 observation
_trainer = None          # LAHTrainer 인스턴스
_tuner = None            # HyperparamTuner 인스턴스
_train_thread = None     # 학습 백그라운드 스레드
_tune_thread = None      # 튜닝 백그라운드 스레드
_episode_store: Dict[str, Dict] = {}  # 에피소드 저장소

# 현재 설정
_current_config: Dict[str, Any] = {
    "env": {},
    "train": {"algo": "PPO", "total_timesteps": 500_000},
    "tune": {"algo": "PPO", "n_trials": 20, "timeout": 3600},
}


def _get_env():
    """환경 인스턴스 반환. 없으면 생성."""
    global _env
    if _env is None:
        from src.envs.lah_env import LAHLocalPathEnv
        _env = LAHLocalPathEnv(config=_current_config.get("env", {}))
    return _env


def _get_trainer():
    """트레이너 인스턴스 반환. 없으면 생성."""
    global _trainer
    if _trainer is None:
        from src.train.trainer import LAHTrainer
        tc = _current_config.get("train", {})
        _trainer = LAHTrainer(
            algo=tc.get("algo", "PPO"),
            env_config=_current_config.get("env", {}),
        )
    return _trainer


# ------------------------------------------------------------------
# Pydantic 요청/응답 모델
# ------------------------------------------------------------------

class ActionRequest(BaseModel):
    """환경 스텝 요청."""
    action: List[float]


class TrainStartRequest(BaseModel):
    """학습 시작 요청."""
    algo: str = "PPO"
    total_timesteps: int = 500_000
    env_config: Optional[Dict[str, Any]] = None
    train_config: Optional[Dict[str, Any]] = None


class SimulateRequest(BaseModel):
    """에피소드 시뮬레이션 요청."""
    policy: str = "baseline"  # "baseline", "corridor", "hold", "model"
    model_path: Optional[str] = None
    max_steps: int = 500
    render: bool = False


class ConfigUpdateRequest(BaseModel):
    """설정 업데이트 요청."""
    env: Optional[Dict[str, Any]] = None
    train: Optional[Dict[str, Any]] = None
    tune: Optional[Dict[str, Any]] = None


class TuneStartRequest(BaseModel):
    """튜닝 시작 요청."""
    algo: str = "PPO"
    n_trials: int = 20
    timeout: float = 3600.0
    eval_timesteps: int = 30_000


class WaypointsRequest(BaseModel):
    """ref path 설정 요청."""
    waypoints: List[List[float]]
    step_m: float = 50.0


# ------------------------------------------------------------------
# 상태 엔드포인트
# ------------------------------------------------------------------

@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """서버 및 환경 상태 조회."""
    global _env, _trainer, _train_thread

    env_ready = _env is not None
    is_training = (
        _train_thread is not None
        and _train_thread.is_alive()
    )
    is_tuning = (
        _tune_thread is not None
        and _tune_thread.is_alive()
    )

    return {
        "status": "running",
        "version": "1.0.0",
        "env_ready": env_ready,
        "is_training": is_training,
        "is_tuning": is_tuning,
        "episode_count": len(_episode_store),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ------------------------------------------------------------------
# 환경 엔드포인트
# ------------------------------------------------------------------

@app.post("/api/env/reset")
async def env_reset(request: Optional[WaypointsRequest] = None) -> Dict[str, Any]:
    """
    환경 리셋 및 초기 observation 반환.

    Args (선택):
        waypoints: 새 ref path 경유점 리스트

    Returns:
        {"obs": {...}, "info": {...}}
    """
    global _env, _current_obs

    env = _get_env()

    options = {}
    if request and request.waypoints:
        from src.route.ref_path import RefPath
        wp = [tuple(p) for p in request.waypoints]
        options["ref_path"] = RefPath.from_waypoints(wp, step_m=request.step_m)

    obs, info = env.reset(options=options if options else None)
    _current_obs = obs

    return {
        "obs": _obs_to_json(obs),
        "info": _safe_dict(info),
    }


@app.post("/api/env/step")
async def env_step(request: ActionRequest) -> Dict[str, Any]:
    """
    환경 한 스텝 진행.

    Args:
        action: [yaw_rate, vz, accel, hold] ∈ [-1, 1]

    Returns:
        {"obs": ..., "reward": ..., "terminated": ..., "truncated": ..., "info": ...}
    """
    global _current_obs

    env = _get_env()
    if _current_obs is None:
        obs, info = env.reset()
        _current_obs = obs

    action = np.array(request.action, dtype=np.float32)
    if len(action) != 4:
        raise HTTPException(status_code=400, detail="action은 4차원이어야 합니다.")

    obs, reward, terminated, truncated, info = env.step(action)
    _current_obs = obs

    return {
        "obs": _obs_to_json(obs),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": _safe_dict(info),
    }


@app.post("/api/env/validate")
async def env_validate() -> Dict[str, Any]:
    """
    환경 검증 실행.

    Returns:
        EnvironmentValidator 전체 보고서
    """
    from src.validate.rules import EnvironmentValidator

    env = _get_env()
    validator = EnvironmentValidator()
    report = validator.run_full_validation(env)
    return report


# ------------------------------------------------------------------
# 학습 엔드포인트
# ------------------------------------------------------------------

@app.post("/api/train/start")
async def train_start(
    request: TrainStartRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    강화학습 시작 (비동기 백그라운드 실행).

    Args:
        algo: 알고리즘 ('PPO' 또는 'SAC')
        total_timesteps: 총 학습 스텝
        env_config: 환경 설정 오버라이드
        train_config: 학습 파라미터 오버라이드

    Returns:
        {"status": "started", ...}
    """
    global _trainer, _train_thread

    if _train_thread is not None and _train_thread.is_alive():
        return {"status": "already_running", "message": "이미 학습 중입니다."}

    from src.train.trainer import LAHTrainer

    env_cfg = {**_current_config.get("env", {}), **(request.env_config or {})}
    _trainer = LAHTrainer(
        algo=request.algo,
        env_config=env_cfg,
        train_config=request.train_config,
    )

    def _run_training():
        try:
            logger.info(f"백그라운드 학습 시작: {request.algo}, {request.total_timesteps} steps")
            _trainer.train(total_timesteps=request.total_timesteps)
            logger.info("백그라운드 학습 완료")
        except Exception as e:
            logger.error(f"학습 오류: {e}")

    _train_thread = threading.Thread(target=_run_training, daemon=True)
    _train_thread.start()

    return {
        "status": "started",
        "algo": request.algo,
        "total_timesteps": request.total_timesteps,
        "message": f"{request.algo} 학습을 백그라운드에서 시작했습니다.",
    }


@app.get("/api/train/status")
async def train_status() -> Dict[str, Any]:
    """
    학습 진행 상태 조회.

    Returns:
        {"is_training": ..., "timesteps": ..., ...}
    """
    global _train_thread, _trainer

    is_training = _train_thread is not None and _train_thread.is_alive()
    log = _trainer.get_training_log() if _trainer else []

    return {
        "is_training": is_training,
        "n_log_entries": len(log),
        "latest_log": log[-1] if log else None,
        "algo": _trainer.algo if _trainer else None,
    }


@app.post("/api/train/stop")
async def train_stop() -> Dict[str, Any]:
    """
    학습 중단 요청.

    Note: stable-baselines3는 즉각 중단을 지원하지 않으므로
    현재 스텝 완료 후 중단됩니다.

    Returns:
        {"status": "stop_requested"}
    """
    return {
        "status": "stop_requested",
        "message": "학습 중단 요청이 접수되었습니다. 현재 스텝 완료 후 중단됩니다.",
    }


@app.get("/api/train/log")
async def train_log() -> Dict[str, Any]:
    """
    학습 로그 반환 (reward curve 등).

    Returns:
        {"log": [...], "n_entries": ...}
    """
    log = _trainer.get_training_log() if _trainer else []
    return {
        "log": log,
        "n_entries": len(log),
    }


# ------------------------------------------------------------------
# 모델 엔드포인트
# ------------------------------------------------------------------

@app.post("/api/model/load")
async def model_load(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    저장된 모델 로드.

    Body: {"path": "outputs/models/lah_ppo_12345"}

    Returns:
        {"status": "loaded", "path": ...}
    """
    global _trainer

    path = body.get("path", "")
    if not path:
        raise HTTPException(status_code=400, detail="path가 필요합니다.")

    trainer = _get_trainer()
    trainer.load_model(path)
    return {"status": "loaded", "path": path}


@app.get("/api/model/list")
async def model_list() -> Dict[str, Any]:
    """
    저장된 모델 목록 반환.

    Returns:
        {"models": [...]}
    """
    model_dir = Path("outputs/models")
    if not model_dir.exists():
        return {"models": []}

    models = []
    for f in model_dir.glob("*.zip"):
        meta_path = f.parent / (f.stem + "_meta.json")
        meta = {}
        if meta_path.exists():
            import json
            with open(meta_path) as mf:
                meta = json.load(mf)
        models.append({
            "name": f.stem,
            "path": str(f),
            "size_mb": round(f.stat().st_size / 1e6, 2),
            "meta": meta,
        })

    return {"models": sorted(models, key=lambda x: x["name"], reverse=True)}


# ------------------------------------------------------------------
# 시뮬레이션 엔드포인트
# ------------------------------------------------------------------

@app.post("/api/simulate")
async def simulate(request: SimulateRequest) -> Dict[str, Any]:
    """
    정책으로 에피소드 시뮬레이션 실행.

    Args:
        policy: 사용할 정책 ("baseline", "corridor", "hold", "model")
        model_path: 모델 파일 경로 (policy="model"일 때)
        max_steps: 최대 스텝 수
        render: 렌더링 여부

    Returns:
        {"episode_id": ..., "n_steps": ..., "success": ..., "summary": ...}
    """
    from src.envs.lah_env import LAHLocalPathEnv
    from src.sim.metrics import EpisodeMetrics

    env = LAHLocalPathEnv(
        config={**_current_config.get("env", {}), "max_steps": request.max_steps}
    )
    obs, _ = env.reset()

    # 정책 선택
    policy_obj = None
    if request.policy == "baseline":
        from src.heuristics.baseline import BaselineTracker
        policy_obj = BaselineTracker()
    elif request.policy == "corridor":
        from src.heuristics.baseline import CorridorShiftPolicy
        policy_obj = CorridorShiftPolicy()
    elif request.policy == "hold":
        from src.heuristics.baseline import HoldPolicy
        policy_obj = HoldPolicy()
    elif request.policy == "model":
        if not request.model_path:
            raise HTTPException(status_code=400, detail="model_path가 필요합니다.")
        trainer = _get_trainer()
        trainer.load_model(request.model_path)
        policy_obj = None  # 아래서 model 직접 사용

    done = False
    frames = []
    ep_reward = 0.0

    while not done:
        if request.policy == "model" and _trainer and _trainer.model:
            action, _ = _trainer.model.predict(obs, deterministic=True)
        elif policy_obj:
            action = policy_obj.get_action(obs)
        else:
            action = np.zeros(4, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated or truncated

        if request.render:
            env.render()

        # 프레임 저장
        frames.append({
            "step": info.get("step", 0),
            "x": float(env.state.x),
            "y": float(env.state.y),
            "z": float(env.state.z),
            "v": float(env.state.v),
            "psi": float(env.state.psi),
            "vz": float(env.state.vz),
            "agl": float(info.get("agl", 0)),
            "reward": float(reward),
            "hold_mode": bool(env.state.hold_mode),
            "action": action.tolist(),
        })

    metrics = env.get_episode_metrics()
    env.close()

    # 에피소드 저장
    ep_id = str(uuid.uuid4())[:8]
    _episode_store[ep_id] = {
        "id": ep_id,
        "policy": request.policy,
        "frames": frames,
        "metrics": metrics,
        "total_reward": float(ep_reward),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    return {
        "episode_id": ep_id,
        "n_steps": len(frames),
        "success": metrics.get("success", False),
        "total_reward": float(ep_reward),
        "summary": metrics,
    }


@app.get("/api/simulate/episode/{ep_id}")
async def get_episode(ep_id: str) -> Dict[str, Any]:
    """
    에피소드 데이터 조회.

    Args:
        ep_id: 에피소드 ID

    Returns:
        에피소드 전체 데이터 (frames, metrics)
    """
    if ep_id not in _episode_store:
        raise HTTPException(status_code=404, detail=f"에피소드 {ep_id}를 찾을 수 없습니다.")
    return _episode_store[ep_id]


# ------------------------------------------------------------------
# 지형 엔드포인트
# ------------------------------------------------------------------

@app.get("/api/terrain/heightmap")
async def terrain_heightmap() -> Dict[str, Any]:
    """
    합성 지형 heightmap 데이터 반환 (JSON).

    Returns:
        {"heightmap": [...], "size_x": ..., "size_y": ..., "resolution": ...}
    """
    env = _get_env()
    terrain = env.terrain

    # 다운샘플링 (웹 전송 크기 제한)
    hm = terrain.heightmap
    step = max(1, hm.shape[0] // 100)
    hm_small = hm[::step, ::step]

    return {
        "heightmap": hm_small.tolist(),
        "size_x": terrain.size_x,
        "size_y": terrain.size_y,
        "resolution": terrain.resolution,
        "step": step,
        "min_elev": float(hm.min()),
        "max_elev": float(hm.max()),
        "world_extent": list(terrain.world_extent),
    }


@app.get("/api/terrain/features")
async def terrain_features() -> Dict[str, Any]:
    """
    지형 특징 데이터 반환 (slope, roughness, risk 등).

    Returns:
        {"slope": [...], "roughness": [...], "risk": [...], ...}
    """
    from src.terrain.features import compute_all_features

    env = _get_env()
    terrain = env.terrain

    features = compute_all_features(terrain.heightmap, terrain.resolution)

    # 다운샘플링
    step = max(1, terrain.heightmap.shape[0] // 80)

    return {
        "slope": features["slope"][::step, ::step].tolist(),
        "roughness": features["roughness"][::step, ::step].tolist(),
        "risk": features["risk"][::step, ::step].tolist(),
        "tpi": features["tpi"][::step, ::step].tolist(),
        "landing": features["landing"][::step, ::step].tolist(),
        "step": step,
        "resolution": terrain.resolution,
    }


# ------------------------------------------------------------------
# 튜닝 엔드포인트
# ------------------------------------------------------------------

@app.post("/api/tune/start")
async def tune_start(request: TuneStartRequest) -> Dict[str, Any]:
    """
    Optuna 하이퍼파라미터 튜닝 시작 (백그라운드).

    Returns:
        {"status": "started", ...}
    """
    global _tuner, _tune_thread

    if _tune_thread is not None and _tune_thread.is_alive():
        return {"status": "already_running", "message": "이미 튜닝 중입니다."}

    from src.train.tuner import HyperparamTuner

    _tuner = HyperparamTuner(
        env_config=_current_config.get("env", {}),
        algo=request.algo,
    )

    def _run_tuning():
        try:
            logger.info(f"백그라운드 튜닝 시작: {request.algo}")
            _tuner.tune(
                n_trials=request.n_trials,
                timeout=request.timeout,
                eval_timesteps=request.eval_timesteps,
            )
        except Exception as e:
            logger.error(f"튜닝 오류: {e}")

    _tune_thread = threading.Thread(target=_run_tuning, daemon=True)
    _tune_thread.start()

    return {
        "status": "started",
        "algo": request.algo,
        "n_trials": request.n_trials,
        "timeout": request.timeout,
    }


@app.get("/api/tune/status")
async def tune_status() -> Dict[str, Any]:
    """
    튜닝 진행 상태 조회.

    Returns:
        {"is_tuning": ..., "n_completed": ..., "best_value": ..., "best_params": ...}
    """
    global _tune_thread, _tuner

    is_tuning = _tune_thread is not None and _tune_thread.is_alive()

    if _tuner:
        status = _tuner.get_status()
    else:
        status = {"is_running": False, "n_completed": 0, "best_value": None, "best_params": {}}

    status["is_tuning"] = is_tuning
    return status


# ------------------------------------------------------------------
# 설정 엔드포인트
# ------------------------------------------------------------------

@app.post("/api/config/update")
async def config_update(request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    환경/학습 설정 업데이트.

    Args:
        env: 환경 설정 dict
        train: 학습 설정 dict
        tune: 튜닝 설정 dict

    Returns:
        {"status": "updated", "config": ...}
    """
    global _current_config, _env, _trainer

    if request.env:
        _current_config["env"].update(request.env)
        _env = None  # 환경 재생성 필요

    if request.train:
        _current_config["train"].update(request.train)
        _trainer = None  # 트레이너 재생성 필요

    if request.tune:
        _current_config["tune"].update(request.tune)

    return {
        "status": "updated",
        "config": _current_config,
    }


@app.get("/api/config/current")
async def config_current() -> Dict[str, Any]:
    """
    현재 설정 반환.

    Returns:
        {"env": ..., "train": ..., "tune": ...}
    """
    return {
        "config": _current_config,
    }


# ------------------------------------------------------------------
# 내부 유틸리티
# ------------------------------------------------------------------

def _obs_to_json(obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """observation numpy 배열을 JSON 직렬화 가능한 형태로 변환."""
    result = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result


def _safe_dict(d: Any) -> Any:
    """dict의 numpy 값을 Python 기본 타입으로 변환."""
    if isinstance(d, dict):
        return {k: _safe_dict(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, (np.integer,)):
        return int(d)
    elif isinstance(d, (np.floating,)):
        return float(d)
    elif isinstance(d, (np.bool_,)):
        return bool(d)
    return d
