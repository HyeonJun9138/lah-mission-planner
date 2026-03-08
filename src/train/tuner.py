"""
Optuna 기반 하이퍼파라미터 자동 튜닝 모듈
- PPO/SAC 핵심 하이퍼파라미터 탐색
- 튜닝 결과 저장 및 최적 파라미터 반환
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HyperparamTuner:
    """
    Optuna 기반 하이퍼파라미터 자동 튜닝 클래스.

    LAHLocalPathEnv에서 PPO 또는 SAC의 하이퍼파라미터를 자동으로 탐색한다.

    Attributes:
        env_config (Dict): 환경 설정
        algo (str): 알고리즘 이름 ('PPO' 또는 'SAC')
        study: Optuna study 객체
        best_params (Dict): 최적 하이퍼파라미터
    """

    def __init__(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        algo: str = "PPO",
        study_name: str = "lah_tuning",
        output_dir: str = "outputs/tuning",
    ) -> None:
        """
        Args:
            env_config: 환경 파라미터 dict
            algo: 알고리즘 ('PPO' 또는 'SAC')
            study_name: Optuna study 이름
            output_dir: 튜닝 결과 저장 디렉토리
        """
        self.env_config = env_config or {}
        self.algo = algo.upper()
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.algo not in ("PPO", "SAC"):
            raise ValueError(f"지원하지 않는 알고리즘: {algo}")

        self.study = None
        self.best_params: Dict[str, Any] = {}
        self._is_running: bool = False

    # ------------------------------------------------------------------
    # 탐색 공간 정의
    # ------------------------------------------------------------------

    def _suggest_ppo_params(self, trial: Any) -> Dict[str, Any]:
        """
        PPO 하이퍼파라미터 탐색 공간 정의.

        Args:
            trial: Optuna trial 객체

        Returns:
            제안된 하이퍼파라미터 dict
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
            "n_epochs": trial.suggest_int("n_epochs", 5, 20),
            "gamma": trial.suggest_float("gamma", 0.99, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
            "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.9),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        }

    def _suggest_sac_params(self, trial: Any) -> Dict[str, Any]:
        """
        SAC 하이퍼파라미터 탐색 공간 정의.

        Args:
            trial: Optuna trial 객체

        Returns:
            제안된 하이퍼파라미터 dict
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 300_000, 500_000]),
            "learning_starts": trial.suggest_categorical("learning_starts", [1_000, 5_000, 10_000]),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "gamma": trial.suggest_float("gamma", 0.98, 0.999),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        }

    # ------------------------------------------------------------------
    # Objective 함수
    # ------------------------------------------------------------------

    def _objective(
        self,
        trial: Any,
        eval_timesteps: int = 30_000,
        n_eval_episodes: int = 5,
    ) -> float:
        """
        Optuna objective 함수.

        Args:
            trial: Optuna trial
            eval_timesteps: 각 trial 학습 스텝 수
            n_eval_episodes: 평가 에피소드 수

        Returns:
            목표값 (평균 보상, 최대화)
        """
        from src.envs.lah_env import LAHLocalPathEnv
        from src.sim.metrics import MultiEpisodeMetrics

        # 하이퍼파라미터 제안
        if self.algo == "PPO":
            params = self._suggest_ppo_params(trial)
        else:
            params = self._suggest_sac_params(trial)

        trial_num = trial.number
        logger.info(f"[Trial {trial_num}] 파라미터: {params}")

        try:
            # 환경 생성
            env = LAHLocalPathEnv(config=self.env_config)

            # 모델 생성 및 학습
            if self.algo == "PPO":
                from stable_baselines3 import PPO
                model = PPO(
                    "MultiInputPolicy",
                    env,
                    verbose=0,
                    **params,
                )
            else:
                from stable_baselines3 import SAC
                model = SAC(
                    "MultiInputPolicy",
                    env,
                    verbose=0,
                    **params,
                )

            model.learn(total_timesteps=eval_timesteps, reset_num_timesteps=True)

            # 평가
            eval_env = LAHLocalPathEnv(config=self.env_config)
            multi_metrics = MultiEpisodeMetrics()
            total_reward = 0.0

            for _ in range(n_eval_episodes):
                obs, _ = eval_env.reset()
                done = False
                ep_r = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = eval_env.step(action)
                    ep_r += r
                    done = terminated or truncated
                total_reward += ep_r
                multi_metrics.add(eval_env.get_episode_metrics())

            eval_env.close()
            env.close()

            summary = multi_metrics.summarize()
            avg_reward = total_reward / max(n_eval_episodes, 1)

            # 목표: 평균 보상 + 성공률 보너스
            objective_val = avg_reward + 100.0 * summary.get("success_rate", 0.0)

            logger.info(
                f"[Trial {trial_num}] 완료: obj={objective_val:.2f}, "
                f"success={summary.get('success_rate', 0):.2%}"
            )
            return objective_val

        except Exception as e:
            logger.warning(f"[Trial {trial_num}] 실패: {e}")
            return float("-inf")

    # ------------------------------------------------------------------
    # 튜닝 실행
    # ------------------------------------------------------------------

    def tune(
        self,
        n_trials: int = 20,
        timeout: float = 3600.0,
        eval_timesteps: int = 30_000,
        n_eval_episodes: int = 5,
        direction: str = "maximize",
    ) -> Dict[str, Any]:
        """
        하이퍼파라미터 튜닝 실행.

        Args:
            n_trials: 탐색 횟수
            timeout: 최대 실행 시간 [s]
            eval_timesteps: 각 trial 학습 스텝 수
            n_eval_episodes: 평가 에피소드 수
            direction: 최적화 방향 ('maximize' 또는 'minimize')

        Returns:
            최적 하이퍼파라미터 dict
        """
        import optuna

        logger.info(
            f"[HyperparamTuner] {self.algo} 튜닝 시작: "
            f"{n_trials} trials, timeout={timeout}s"
        )
        self._is_running = True

        # Optuna study 생성
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        try:
            self.study.optimize(
                lambda trial: self._objective(
                    trial,
                    eval_timesteps=eval_timesteps,
                    n_eval_episodes=n_eval_episodes,
                ),
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
            )
        finally:
            self._is_running = False

        # 최적 파라미터 저장
        if self.study.best_trial:
            self.best_params = self.study.best_trial.params
            self._save_results()
            logger.info(f"[HyperparamTuner] 최적 파라미터: {self.best_params}")
        else:
            logger.warning("[HyperparamTuner] 유효한 trial이 없습니다.")

        return self.best_params

    def _save_results(self) -> None:
        """튜닝 결과 저장."""
        result = {
            "algo": self.algo,
            "study_name": self.study_name,
            "best_params": self.best_params,
            "best_value": self.study.best_value if self.study.best_trial else None,
            "n_trials": len(self.study.trials),
            "save_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = self.output_dir / f"tuning_{self.algo.lower()}_{int(time.time())}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"[HyperparamTuner] 결과 저장: {path}")

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """
        현재 튜닝 상태 반환.

        Returns:
            {
                "is_running": bool,
                "n_completed": int,
                "best_value": float,
                "best_params": dict,
            }
        """
        if self.study is None:
            return {
                "is_running": self._is_running,
                "n_completed": 0,
                "best_value": None,
                "best_params": {},
            }

        completed = [t for t in self.study.trials
                     if t.state.name == "COMPLETE"]
        return {
            "is_running": self._is_running,
            "n_completed": len(completed),
            "best_value": self.study.best_value if completed else None,
            "best_params": self.best_params,
        }

    def __repr__(self) -> str:
        n = len(self.study.trials) if self.study else 0
        return f"HyperparamTuner(algo={self.algo}, trials={n})"
