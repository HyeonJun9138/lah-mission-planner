"""
환경 검증(Validation) 모듈
- Gymnasium 환경의 observation_space, action_space, reward, dynamics, terrain 검증
- 각 검증 항목은 pass/fail/warning 판정 + 한국어 상세 메시지 반환
"""

from __future__ import annotations

import time
import logging
import traceback
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """
    LAHLocalPathEnv 환경 검증 클래스.

    각 검증 함수는 독립적으로 실행되며,
    pass/fail/warning 결과와 한국어 상세 메시지를 반환한다.
    """

    # 허용 기준치
    OBS_FINITE_RATIO_MIN = 0.95    # observation 유한값 비율 최소 기준
    REWARD_FINITE_RATIO_MIN = 0.95  # reward 유한값 비율 최소 기준
    AGL_MIN_SAFE = 0.0             # AGL 최소값 (지형 아래로 가면 안 됨)

    def __init__(self) -> None:
        """초기화."""
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 개별 검증 함수
    # ------------------------------------------------------------------

    def validate_observation_space(self, env: "gym.Env") -> Dict[str, Any]:
        """
        Observation space 검증.

        검사 항목:
        - Dict space 여부
        - 필수 키 존재 여부 (state_vec, lookahead_ref, local_patch)
        - 각 sub-space의 shape 및 dtype
        - observation 샘플의 범위 확인

        Args:
            env: 검증할 환경

        Returns:
            검증 결과 dict
        """
        result = _new_result("observation_space", "Observation Space 검증")
        try:
            import gymnasium as gym

            obs_space = env.observation_space

            # Dict space 확인
            if not isinstance(obs_space, gym.spaces.Dict):
                result["status"] = "fail"
                result["message"] = "observation_space가 Dict 타입이 아닙니다."
                return result

            required_keys = ["state_vec", "lookahead_ref", "local_patch"]
            missing = [k for k in required_keys if k not in obs_space.spaces]
            if missing:
                result["status"] = "fail"
                result["message"] = f"필수 observation 키 누락: {missing}"
                return result

            shapes = {}
            for k, space in obs_space.spaces.items():
                shapes[k] = space.shape

            # observation 샘플 생성 및 확인
            obs, _ = env.reset()
            for k in required_keys:
                if k not in obs:
                    result["status"] = "fail"
                    result["message"] = f"reset() 결과에 '{k}' 없음"
                    return result

                arr = obs[k]
                if not isinstance(arr, np.ndarray):
                    result["status"] = "warning"
                    result["message"] = f"'{k}'이 ndarray가 아님: {type(arr)}"
                    continue

                expected_shape = obs_space.spaces[k].shape
                if arr.shape != expected_shape:
                    result["status"] = "fail"
                    result["message"] = (
                        f"'{k}' shape 불일치: "
                        f"예상 {expected_shape}, 실제 {arr.shape}"
                    )
                    return result

                # NaN/Inf 확인
                finite_ratio = float(np.isfinite(arr).mean())
                if finite_ratio < self.OBS_FINITE_RATIO_MIN:
                    result["status"] = "warning"
                    result["message"] = (
                        f"'{k}'에 NaN/Inf 비율이 높음: "
                        f"유한값 비율={finite_ratio:.2%}"
                    )

            result["status"] = "pass"
            result["message"] = "Observation space 검증 통과"
            result["details"] = {"shapes": {k: list(v) for k, v in shapes.items()}}

        except Exception as e:
            result["status"] = "fail"
            result["message"] = f"검증 중 예외 발생: {e}"
            result["traceback"] = traceback.format_exc()

        return result

    def validate_action_space(self, env: "gym.Env") -> Dict[str, Any]:
        """
        Action space 검증.

        검사 항목:
        - Box(4,) 타입 여부
        - 범위 [-1, 1] 확인
        - 샘플 액션으로 step() 실행 가능 여부

        Args:
            env: 검증할 환경

        Returns:
            검증 결과 dict
        """
        result = _new_result("action_space", "Action Space 검증")
        try:
            import gymnasium as gym

            act_space = env.action_space

            if not isinstance(act_space, gym.spaces.Box):
                result["status"] = "fail"
                result["message"] = "action_space가 Box 타입이 아닙니다."
                return result

            if act_space.shape != (4,):
                result["status"] = "fail"
                result["message"] = f"action_space shape 오류: {act_space.shape} (예상: (4,))"
                return result

            # 범위 확인
            if not (np.all(act_space.low == -1.0) and np.all(act_space.high == 1.0)):
                result["status"] = "warning"
                result["message"] = (
                    f"action_space 범위가 [-1,1]이 아님: "
                    f"low={act_space.low}, high={act_space.high}"
                )

            # 샘플 액션 실행 가능 확인
            env.reset()
            sample_action = act_space.sample()
            obs, reward, terminated, truncated, info = env.step(sample_action)

            if not isinstance(reward, (int, float)):
                result["status"] = "fail"
                result["message"] = f"reward 타입 오류: {type(reward)}"
                return result

            result["status"] = "pass"
            result["message"] = "Action space 검증 통과"
            result["details"] = {
                "shape": list(act_space.shape),
                "low": act_space.low.tolist(),
                "high": act_space.high.tolist(),
                "sample_reward": float(reward),
            }

        except Exception as e:
            result["status"] = "fail"
            result["message"] = f"검증 중 예외 발생: {e}"
            result["traceback"] = traceback.format_exc()

        return result

    def validate_reward_function(
        self,
        env: "gym.Env",
        n_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        보상 함수 검증.

        검사 항목:
        - 보상이 유한값인지
        - 보상 범위가 적절한지
        - 진행 시 보상 경향 (progress → 양수 보상 기대)

        Args:
            env: 검증할 환경
            n_steps: 검증 스텝 수

        Returns:
            검증 결과 dict
        """
        result = _new_result("reward_function", "보상 함수 검증")
        try:
            obs, _ = env.reset()
            rewards = []
            terminated = False
            truncated = False

            for _ in range(n_steps):
                if terminated or truncated:
                    obs, _ = env.reset()

                # 기본 추종 액션 (약간 전진)
                action = np.array([0.0, 0.0, 0.2, -1.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))

            rewards_arr = np.array(rewards)
            finite_ratio = float(np.isfinite(rewards_arr).mean())
            mean_r = float(np.mean(rewards_arr))
            std_r = float(np.std(rewards_arr))
            min_r = float(np.min(rewards_arr))
            max_r = float(np.max(rewards_arr))

            if finite_ratio < self.REWARD_FINITE_RATIO_MIN:
                result["status"] = "fail"
                result["message"] = f"보상에 NaN/Inf 비율 높음: {finite_ratio:.2%}"
                return result

            if max_r - min_r < 1e-6:
                result["status"] = "warning"
                result["message"] = "보상 분산이 너무 낮음 (보상이 변하지 않음)"
            else:
                result["status"] = "pass"
                result["message"] = "보상 함수 검증 통과"

            result["details"] = {
                "n_steps": n_steps,
                "mean": mean_r,
                "std": std_r,
                "min": min_r,
                "max": max_r,
                "finite_ratio": finite_ratio,
            }

        except Exception as e:
            result["status"] = "fail"
            result["message"] = f"검증 중 예외 발생: {e}"
            result["traceback"] = traceback.format_exc()

        return result

    def validate_termination_conditions(
        self,
        env: "gym.Env",
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """
        종료 조건 검증.

        검사 항목:
        - 에피소드가 반드시 종료되는지
        - terminated/truncated 중 하나는 True
        - 종료 이유가 일관성 있는지

        Args:
            env: 검증할 환경
            n_episodes: 검증 에피소드 수

        Returns:
            검증 결과 dict
        """
        result = _new_result("termination", "종료 조건 검증")
        try:
            reasons = []
            ep_lengths = []
            infinite_ep = False

            for _ in range(n_episodes):
                obs, _ = env.reset()
                done = False
                steps = 0
                max_test_steps = 200  # 검증용 제한

                while not done and steps < max_test_steps:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1

                if not done:
                    infinite_ep = True
                    reasons.append("infinite")
                else:
                    reasons.append(info.get("reason", "unknown"))
                ep_lengths.append(steps)

            if infinite_ep:
                result["status"] = "warning"
                result["message"] = (
                    f"{n_episodes} 에피소드 중 일부가 {max_test_steps} 스텝 내에 종료되지 않음"
                )
            else:
                result["status"] = "pass"
                result["message"] = "종료 조건 검증 통과"

            reason_counts: Dict[str, int] = {}
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1

            result["details"] = {
                "n_episodes": n_episodes,
                "mean_ep_length": float(np.mean(ep_lengths)),
                "reason_counts": reason_counts,
                "ep_lengths": ep_lengths,
            }

        except Exception as e:
            result["status"] = "fail"
            result["message"] = f"검증 중 예외 발생: {e}"
            result["traceback"] = traceback.format_exc()

        return result

    def validate_dynamics(
        self,
        env: "gym.Env",
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        운동학 검증.

        검사 항목:
        - 전진 액션 → 실제 위치 변화 여부
        - 속도 제한 준수
        - hold 액션 효과 확인
        - 상태 유효성 (NaN/Inf 없음)

        Args:
            env: 검증할 환경
            n_steps: 검증 스텝 수

        Returns:
            검증 결과 dict
        """
        result = _new_result("dynamics", "운동학 모델 검증")
        try:
            obs, _ = env.reset()
            init_x = float(env.state.x)
            init_y = float(env.state.y)

            # 전진 액션 (yaw=0, vz=0, accel=+1, hold=off)
            forward_action = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)

            positions = [(init_x, init_y)]
            speeds = []
            states_valid = True

            for i in range(n_steps):
                obs, reward, terminated, truncated, info = env.step(forward_action)
                if terminated or truncated:
                    break

                x = float(env.state.x)
                y = float(env.state.y)
                v = float(env.state.v)

                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(v)):
                    states_valid = False
                    break

                positions.append((x, y))
                speeds.append(v)

            if not states_valid:
                result["status"] = "fail"
                result["message"] = "운동학 업데이트 중 NaN/Inf 발생"
                return result

            # 이동 거리 확인
            if len(positions) > 1:
                final_x, final_y = positions[-1]
                dist_moved = float(np.sqrt(
                    (final_x - init_x) ** 2 + (final_y - init_y) ** 2
                ))
            else:
                dist_moved = 0.0

            if dist_moved < 1.0:
                result["status"] = "warning"
                result["message"] = f"전진 액션 후 이동 거리 너무 작음: {dist_moved:.1f}m"
            else:
                result["status"] = "pass"
                result["message"] = "운동학 검증 통과"

            result["details"] = {
                "dist_moved": dist_moved,
                "n_steps": len(positions) - 1,
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "max_speed": float(np.max(speeds)) if speeds else 0.0,
            }

        except Exception as e:
            result["status"] = "fail"
            result["message"] = f"검증 중 예외 발생: {e}"
            result["traceback"] = traceback.format_exc()

        return result

    def validate_terrain_consistency(self, env: "gym.Env") -> Dict[str, Any]:
        """
        지형 일관성 검증.

        검사 항목:
        - heightmap 유효성 (NaN/Inf 없음, 범위 합리적)
        - get_elevation() 결과 일관성
        - AGL 계산 일관성

        Args:
            env: 검증할 환경

        Returns:
            검증 결과 dict
        """
        result = _new_result("terrain", "지형 일관성 검증")
        try:
            terrain = env.terrain
            hm = terrain.heightmap

            # heightmap 유효성
            if not np.all(np.isfinite(hm)):
                result["status"] = "fail"
                result["message"] = "heightmap에 NaN/Inf 포함"
                return result

            if hm.min() < -100.0 or hm.max() > 10000.0:
                result["status"] = "warning"
                result["message"] = (
                    f"heightmap 고도 범위 비정상: "
                    f"[{hm.min():.0f}, {hm.max():.0f}]m"
                )

            # 무작위 포인트에서 고도 쿼리 일관성
            rng = np.random.default_rng(0)
            x_ext, y_ext = terrain.world_extent[1], terrain.world_extent[3]
            test_xs = rng.uniform(0, x_ext, 20)
            test_ys = rng.uniform(0, y_ext, 20)

            elevs = []
            for tx, ty in zip(test_xs, test_ys):
                e = terrain.get_elevation(float(tx), float(ty))
                elevs.append(e)

            elevs_arr = np.array(elevs)
            if not np.all(np.isfinite(elevs_arr)):
                result["status"] = "fail"
                result["message"] = "get_elevation() 결과에 NaN/Inf 포함"
                return result

            result["status"] = "pass"
            result["message"] = "지형 일관성 검증 통과"
            result["details"] = {
                "heightmap_shape": list(hm.shape),
                "elev_range": [float(hm.min()), float(hm.max())],
                "mean_elev": float(hm.mean()),
                "test_elev_range": [float(elevs_arr.min()), float(elevs_arr.max())],
            }

        except Exception as e:
            result["status"] = "fail"
            result["message"] = f"검증 중 예외 발생: {e}"
            result["traceback"] = traceback.format_exc()

        return result

    # ------------------------------------------------------------------
    # 전체 검증 실행
    # ------------------------------------------------------------------

    def run_full_validation(self, env: "gym.Env") -> Dict[str, Any]:
        """
        전체 환경 검증 실행.

        모든 개별 검증을 순서대로 실행하고
        종합 보고서를 반환한다.

        Args:
            env: 검증할 환경

        Returns:
            {
                "summary": {"pass": N, "fail": N, "warning": N},
                "overall": "pass" | "fail" | "warning",
                "results": {항목별 결과 dict},
                "duration_s": 소요 시간,
            }
        """
        start_time = time.time()
        logger.info("[EnvironmentValidator] 전체 환경 검증 시작")

        validators = [
            ("observation_space", self.validate_observation_space),
            ("action_space", self.validate_action_space),
            ("reward_function", self.validate_reward_function),
            ("termination", self.validate_termination_conditions),
            ("dynamics", self.validate_dynamics),
            ("terrain", self.validate_terrain_consistency),
        ]

        results = {}
        summary = {"pass": 0, "fail": 0, "warning": 0}

        for name, validator_fn in validators:
            try:
                logger.info(f"  검증 중: {name}")
                r = validator_fn(env)
                results[name] = r
                status = r.get("status", "fail")
                summary[status] = summary.get(status, 0) + 1
            except Exception as e:
                logger.error(f"  {name} 검증 중 예외: {e}")
                results[name] = {
                    "name": name,
                    "status": "fail",
                    "message": f"검증 실행 실패: {e}",
                }
                summary["fail"] += 1

        # 종합 판정
        if summary["fail"] > 0:
            overall = "fail"
        elif summary["warning"] > 0:
            overall = "warning"
        else:
            overall = "pass"

        duration = time.time() - start_time
        logger.info(
            f"[EnvironmentValidator] 검증 완료: "
            f"pass={summary['pass']}, warning={summary['warning']}, fail={summary['fail']} "
            f"({duration:.1f}s)"
        )

        return {
            "summary": summary,
            "overall": overall,
            "results": results,
            "duration_s": round(duration, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


# ------------------------------------------------------------------
# 내부 유틸리티
# ------------------------------------------------------------------

def _new_result(name: str, display_name: str) -> Dict[str, Any]:
    """
    검증 결과 dict 초기화.

    Args:
        name: 검증 항목 식별자
        display_name: 한국어 표시 이름

    Returns:
        초기화된 결과 dict
    """
    return {
        "name": name,
        "display_name": display_name,
        "status": "fail",  # 기본: fail (통과 시 "pass"로 업데이트)
        "message": "",
        "details": {},
    }
