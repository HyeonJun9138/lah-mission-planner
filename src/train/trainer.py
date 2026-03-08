"""
LAH 강화학습 학습기(Trainer)
- stable-baselines3 기반 PPO, SAC 지원
- 학습, 평가, 모델 저장/로드
- 이어학습(resume) 지원
- 학습 로그 관리
"""

from __future__ import annotations

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Type

import gymnasium as gym

logger = logging.getLogger(__name__)


class LAHTrainer:
    """
    LAH 환경용 강화학습 트레이너.

    PPO 또는 SAC 알고리즘으로 LAHLocalPathEnv를 학습한다.
    학습 로그를 내부적으로 관리하며 평가 지표를 반환한다.

    Attributes:
        algo (str): 사용 알고리즘 ('PPO' 또는 'SAC')
        env_config (Dict): 환경 파라미터
        train_config (Dict): 학습 파라미터
        model: stable-baselines3 모델 인스턴스
        training_log (List[Dict]): 학습 로그 (timestep, reward 등)
    """

    # 기본 학습 파라미터
    DEFAULT_PPO_CONFIG: Dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "ent_coef": 0.005,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "policy": "MultiInputPolicy",
    }

    DEFAULT_SAC_CONFIG: Dict[str, Any] = {
        "learning_rate": 3e-4,
        "buffer_size": 500_000,
        "learning_starts": 5_000,
        "batch_size": 512,
        "gamma": 0.995,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "verbose": 1,
        "policy": "MultiInputPolicy",
    }

    def __init__(
        self,
        algo: str = "PPO",
        env_config: Optional[Dict[str, Any]] = None,
        train_config: Optional[Dict[str, Any]] = None,
        log_dir: str = "outputs/logs",
        model_dir: str = "outputs/models",
    ) -> None:
        """
        Args:
            algo: 학습 알고리즘 ('PPO' 또는 'SAC')
            env_config: LAHLocalPathEnv 환경 설정
            train_config: 알고리즘 하이퍼파라미터
            log_dir: 학습 로그 저장 경로
            model_dir: 모델 저장 경로
        """
        self.algo = algo.upper()
        self.env_config = env_config or {}
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 학습 파라미터 설정
        if self.algo == "PPO":
            self.train_config = {**self.DEFAULT_PPO_CONFIG, **(train_config or {})}
        elif self.algo == "SAC":
            self.train_config = {**self.DEFAULT_SAC_CONFIG, **(train_config or {})}
        else:
            raise ValueError(f"지원하지 않는 알고리즘: {algo}. 'PPO' 또는 'SAC' 사용.")

        self.model = None
        self.training_log: List[Dict[str, Any]] = []
        self._env: Optional[gym.Env] = None
        self._is_training: bool = False
        self._total_timesteps_trained: int = 0

    # ------------------------------------------------------------------
    # 환경 생성
    # ------------------------------------------------------------------

    def _make_env(self) -> gym.Env:
        """
        학습용 환경 생성.

        Returns:
            LAHLocalPathEnv 인스턴스
        """
        from src.envs.lah_env import LAHLocalPathEnv
        env = LAHLocalPathEnv(config=self.env_config)
        return env

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 500_000,
        callback: Optional[Any] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
    ) -> Any:
        """
        모델 학습.

        Args:
            total_timesteps: 총 학습 스텝 수
            callback: stable-baselines3 콜백 (None이면 기본 로깅 콜백 사용)
            eval_freq: 평가 주기 [스텝]
            n_eval_episodes: 평가 에피소드 수

        Returns:
            학습된 stable-baselines3 모델
        """
        if self._env is None:
            self._env = self._make_env()

        # 모델 초기화 (처음 학습 시)
        if self.model is None:
            self.model = self._create_model(self._env)

        self._is_training = True
        logger.info(f"[LAHTrainer] {self.algo} 학습 시작: {total_timesteps:,} 스텝")

        # 학습 로그 콜백
        log_callback = self._make_log_callback(eval_freq, n_eval_episodes)
        callbacks = [log_callback]
        if callback is not None:
            callbacks.append(callback)

        try:
            from stable_baselines3.common.callbacks import CallbackList
            cb = CallbackList(callbacks)
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=cb,
                reset_num_timesteps=False,
            )
            self._total_timesteps_trained += total_timesteps
        except Exception as e:
            logger.error(f"학습 오류: {e}")
            raise
        finally:
            self._is_training = False

        logger.info(f"[LAHTrainer] 학습 완료. 총 학습 스텝: {self._total_timesteps_trained:,}")
        return self.model

    def resume_training(
        self,
        model_path: str,
        additional_timesteps: int = 100_000,
    ) -> Any:
        """
        저장된 모델에서 이어학습.

        Args:
            model_path: 저장된 모델 경로
            additional_timesteps: 추가 학습 스텝 수

        Returns:
            계속 학습된 모델
        """
        logger.info(f"[LAHTrainer] 이어학습 시작: {model_path}")
        self.load_model(model_path)
        return self.train(additional_timesteps)

    def _create_model(self, env: gym.Env) -> Any:
        """
        알고리즘에 맞는 모델 인스턴스 생성.

        Args:
            env: Gymnasium 환경

        Returns:
            stable-baselines3 모델 인스턴스
        """
        policy = self.train_config.pop("policy", "MultiInputPolicy")
        verbose = self.train_config.get("verbose", 1)

        if self.algo == "PPO":
            from stable_baselines3 import PPO
            ppo_kwargs = {k: v for k, v in self.train_config.items()
                         if k not in ("verbose",)}
            return PPO(
                policy=policy,
                env=env,
                verbose=verbose,
                tensorboard_log=str(self.log_dir / "tb"),
                **ppo_kwargs,
            )
        elif self.algo == "SAC":
            from stable_baselines3 import SAC
            sac_kwargs = {k: v for k, v in self.train_config.items()
                         if k not in ("verbose",)}
            return SAC(
                policy=policy,
                env=env,
                verbose=verbose,
                tensorboard_log=str(self.log_dir / "tb"),
                **sac_kwargs,
            )
        raise ValueError(f"지원하지 않는 알고리즘: {self.algo}")

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Optional[Any] = None,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        모델 평가.

        Args:
            model: 평가할 모델 (None이면 self.model 사용)
            n_episodes: 평가 에피소드 수
            deterministic: 결정적(greedy) 행동 사용 여부

        Returns:
            {
                "success_rate": ...,
                "avg_reward": ...,
                "mean_agl": ...,
                "avg_risk": ...,
                ...
            }
        """
        from src.sim.metrics import MultiEpisodeMetrics

        m = model or self.model
        if m is None:
            raise RuntimeError("평가할 모델이 없습니다. 먼저 train()을 실행하세요.")

        eval_env = self._make_env()
        multi_metrics = MultiEpisodeMetrics()

        for ep in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = m.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated

            ep_metrics = eval_env.get_episode_metrics()
            ep_metrics["episode_reward"] = ep_reward
            multi_metrics.add(ep_metrics)

        summary = multi_metrics.summarize()
        logger.info(
            f"[평가 결과] success={summary.get('success_rate', 0):.2%}, "
            f"avg_return={summary.get('avg_return', 0):.2f}"
        )
        eval_env.close()
        return summary

    # ------------------------------------------------------------------
    # 모델 저장/로드
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[str] = None) -> str:
        """
        학습된 모델 저장.

        Args:
            path: 저장 경로 (None이면 자동 생성)

        Returns:
            실제 저장된 경로
        """
        if self.model is None:
            raise RuntimeError("저장할 모델이 없습니다.")

        if path is None:
            ts = int(time.time())
            path = str(self.model_dir / f"lah_{self.algo.lower()}_{ts}")

        self.model.save(path)

        # 메타데이터 저장
        meta = {
            "algo": self.algo,
            "total_timesteps": self._total_timesteps_trained,
            "env_config": self.env_config,
            "train_config": self.train_config,
            "save_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(path + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"[LAHTrainer] 모델 저장 완료: {path}")
        return path

    def load_model(self, path: str) -> Any:
        """
        저장된 모델 로드.

        Args:
            path: 모델 경로

        Returns:
            로드된 모델
        """
        env = self._make_env()

        if self.algo == "PPO":
            from stable_baselines3 import PPO
            self.model = PPO.load(path, env=env)
        elif self.algo == "SAC":
            from stable_baselines3 import SAC
            self.model = SAC.load(path, env=env)
        else:
            raise ValueError(f"지원하지 않는 알고리즘: {self.algo}")

        self._env = env
        logger.info(f"[LAHTrainer] 모델 로드 완료: {path}")
        return self.model

    # ------------------------------------------------------------------
    # 학습 로그
    # ------------------------------------------------------------------

    def get_training_log(self) -> List[Dict[str, Any]]:
        """
        학습 로그 반환.

        Returns:
            [{"timestep": ..., "reward": ..., "success_rate": ...}, ...]
        """
        return self.training_log.copy()

    def _make_log_callback(self, eval_freq: int, n_eval_episodes: int) -> Any:
        """
        학습 중 주기적 평가 및 로깅 콜백 생성.

        Args:
            eval_freq: 평가 주기
            n_eval_episodes: 에피소드 수

        Returns:
            stable-baselines3 콜백
        """
        from stable_baselines3.common.callbacks import BaseCallback

        trainer_ref = self

        class TrainingLogCallback(BaseCallback):
            """학습 로그 기록 콜백."""

            def __init__(self, eval_freq_: int, n_ep: int) -> None:
                super().__init__(verbose=0)
                self.eval_freq = eval_freq_
                self.n_ep = n_ep
                self._last_eval_step = 0

            def _on_step(self) -> bool:
                if self.num_timesteps - self._last_eval_step >= self.eval_freq:
                    self._last_eval_step = self.num_timesteps
                    # 간이 평가 (에러 방지를 위해 try/except)
                    try:
                        rewards_list = self.model.ep_info_buffer
                        if rewards_list:
                            recent = list(rewards_list)[-min(10, len(rewards_list)):]
                            avg_r = float(np.mean([ep.get("r", 0) for ep in recent]))
                        else:
                            avg_r = 0.0

                        log_entry = {
                            "timestep": int(self.num_timesteps),
                            "avg_reward": avg_r,
                            "time": time.strftime("%H:%M:%S"),
                        }
                        trainer_ref.training_log.append(log_entry)
                        logger.info(
                            f"[{self.num_timesteps:,} steps] avg_reward={avg_r:.3f}"
                        )
                    except Exception:
                        pass
                return True  # 학습 계속

        return TrainingLogCallback(eval_freq, n_eval_episodes)

    def __repr__(self) -> str:
        return (
            f"LAHTrainer(algo={self.algo}, "
            f"trained_steps={self._total_timesteps_trained:,})"
        )
