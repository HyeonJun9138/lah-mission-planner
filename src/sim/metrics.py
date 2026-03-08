"""
에피소드 지표(Episode Metrics) 계산 모듈
- 설계서 21절 평가 지표 구현
- 학습 진행 평가 및 에피소드 분석에 사용
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional


class EpisodeMetrics:
    """
    단일 에피소드 지표 계산 클래스.

    에피소드 동안 수집된 데이터를 분석하여
    설계서 21절의 모든 평가 지표를 계산한다.

    Attributes:
        frames (List[Dict]): 각 스텝의 데이터 (x,y,z,v,agl,risk,s,d,hold_mode,action,reward 등)
        success (bool): 에피소드 성공 여부
        terminate_reason (str): 종료 이유
    """

    def __init__(self) -> None:
        """초기화."""
        self.frames: List[Dict[str, Any]] = []
        self.success: bool = False
        self.truncated: bool = False
        self.terminate_reason: str = "running"
        self.total_steps: int = 0
        self._ref_path_length: Optional[float] = None

    # ------------------------------------------------------------------
    # 데이터 수집
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        state_dict: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        reward_breakdown: Optional[Dict[str, float]] = None,
        agl: float = 0.0,
        s: float = 0.0,
        d: float = 0.0,
        local_risk: float = 0.0,
        terrain_h: float = 0.0,
    ) -> None:
        """
        스텝 데이터 기록.

        Args:
            step: 현재 스텝 번호
            state_dict: AircraftState.to_dict() 결과
            action: 4차원 액션 벡터
            reward: 보상값
            reward_breakdown: 보상 항목별 세부값
            agl: 지형 위 고도 [m]
            s: 경로 진행 거리 [m]
            d: cross-track 이탈 거리 [m]
            local_risk: 현재 위치 위험도
            terrain_h: 지형 고도 [m]
        """
        frame = {
            "t": step,
            **state_dict,
            "action": action.tolist() if hasattr(action, "tolist") else list(action),
            "reward": float(reward),
            "agl": float(agl),
            "s": float(s),
            "d": float(d),
            "local_risk": float(local_risk),
            "terrain_h": float(terrain_h),
        }
        if reward_breakdown:
            frame["reward_breakdown"] = reward_breakdown

        self.frames.append(frame)
        self.total_steps = step + 1

    def set_result(
        self,
        success: bool,
        truncated: bool,
        reason: str,
        ref_path_length: Optional[float] = None,
    ) -> None:
        """
        에피소드 최종 결과 설정.

        Args:
            success: 성공 여부
            truncated: 시간 초과 여부
            reason: 종료 이유 문자열
            ref_path_length: 참조 경로 전체 길이 [m]
        """
        self.success = success
        self.truncated = truncated
        self.terminate_reason = reason
        self._ref_path_length = ref_path_length

    # ------------------------------------------------------------------
    # 지표 계산
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, Any]:
        """
        모든 에피소드 지표 계산.

        Returns:
            설계서 21절 필수 지표 포함 dict
        """
        if not self.frames:
            return self._empty_metrics()

        # 기본 데이터 추출
        agls = np.array([f["agl"] for f in self.frames], dtype=np.float32)
        risks = np.array([f["local_risk"] for f in self.frames], dtype=np.float32)
        ds = np.array([f["d"] for f in self.frames], dtype=np.float32)
        rewards = np.array([f["reward"] for f in self.frames], dtype=np.float32)
        holds = np.array([f.get("hold_mode", False) for f in self.frames], dtype=bool)
        speeds = np.array([f.get("v", 40.0) for f in self.frames], dtype=np.float32)
        vzs = np.array([f.get("vz", 0.0) for f in self.frames], dtype=np.float32)

        # 경로 길이 계산 (실제 비행 거리)
        xs = np.array([f.get("x", 0.0) for f in self.frames])
        ys = np.array([f.get("y", 0.0) for f in self.frames])
        dxy = np.diff(np.stack([xs, ys], axis=1), axis=0)
        path_length = float(np.sum(np.linalg.norm(dxy, axis=1)))

        # 제어 부드러움 지수
        actions = np.array([f.get("action", [0, 0, 0, 0]) for f in self.frames])
        if len(actions) > 1:
            action_diffs = np.diff(actions, axis=0)
            control_smoothness = float(1.0 / (1.0 + np.mean(np.sum(action_diffs ** 2, axis=1))))
        else:
            control_smoothness = 1.0

        # 시간 투 goal
        if self.success and len(self.frames) > 0:
            time_to_goal = float(self.total_steps)
        else:
            time_to_goal = float("nan")

        # 경로 길이 비율
        ref_len = self._ref_path_length or 1.0
        path_length_ratio = path_length / max(ref_len, 1.0)

        # hold 사용률
        hold_usage_ratio = float(np.mean(holds)) if len(holds) > 0 else 0.0

        return {
            # 핵심 성능 지표
            "success": self.success,
            "truncated": self.truncated,
            "terminate_reason": self.terminate_reason,
            "total_steps": self.total_steps,

            # 보상
            "avg_return": float(np.mean(rewards)),
            "total_return": float(np.sum(rewards)),

            # AGL 지표
            "mean_agl": float(np.mean(agls)),
            "min_agl": float(np.min(agls)),
            "p5_agl": float(np.percentile(agls, 5)),

            # 위험도 지표
            "avg_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
            "p95_risk": float(np.percentile(risks, 95)),

            # 경로 이탈 지표
            "mean_cross_track": float(np.mean(np.abs(ds))),
            "p95_cross_track": float(np.percentile(np.abs(ds), 95)),
            "max_corridor_deviation": float(np.max(np.abs(ds))),

            # 시간/경로 지표
            "time_to_goal": time_to_goal,
            "path_length": path_length,
            "path_length_ratio": path_length_ratio,

            # 제어 지표
            "hold_usage_ratio": hold_usage_ratio,
            "control_smoothness": control_smoothness,

            # 속도 지표
            "mean_speed": float(np.mean(speeds)),
            "mean_vz": float(np.mean(np.abs(vzs))),
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """데이터 없을 때 기본 지표 반환."""
        return {
            "success": False,
            "truncated": False,
            "terminate_reason": "no_data",
            "total_steps": 0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "mean_agl": 0.0,
            "min_agl": 0.0,
            "p5_agl": 0.0,
            "avg_risk": 0.0,
            "max_risk": 0.0,
            "p95_risk": 0.0,
            "mean_cross_track": 0.0,
            "p95_cross_track": 0.0,
            "max_corridor_deviation": 0.0,
            "time_to_goal": float("nan"),
            "path_length": 0.0,
            "path_length_ratio": 0.0,
            "hold_usage_ratio": 0.0,
            "control_smoothness": 1.0,
            "mean_speed": 0.0,
            "mean_vz": 0.0,
        }

    def to_frames_list(self) -> List[Dict[str, Any]]:
        """기록된 프레임 데이터 반환."""
        return self.frames.copy()

    def reset(self) -> None:
        """에피소드 지표 초기화."""
        self.frames.clear()
        self.success = False
        self.truncated = False
        self.terminate_reason = "running"
        self.total_steps = 0
        self._ref_path_length = None

    def __repr__(self) -> str:
        m = self.compute()
        return (
            f"EpisodeMetrics("
            f"steps={self.total_steps}, "
            f"success={self.success}, "
            f"avg_return={m.get('avg_return', 0):.2f}, "
            f"min_agl={m.get('min_agl', 0):.0f}m)"
        )


class MultiEpisodeMetrics:
    """
    다중 에피소드 집계 지표.

    여러 에피소드의 EpisodeMetrics를 집계하여
    학습 성능을 평가한다.
    """

    def __init__(self) -> None:
        self.episode_results: List[Dict[str, Any]] = []

    def add(self, metrics: Dict[str, Any]) -> None:
        """에피소드 지표 추가."""
        self.episode_results.append(metrics)

    def summarize(self) -> Dict[str, Any]:
        """
        집계 지표 계산.

        Returns:
            {
                "n_episodes": ...,
                "success_rate": ...,
                "collision_rate": ...,
                "avg_return": ...,
                ...
            }
        """
        if not self.episode_results:
            return {"n_episodes": 0}

        n = len(self.episode_results)

        successes = [r.get("success", False) for r in self.episode_results]
        returns = [r.get("avg_return", 0.0) for r in self.episode_results]
        min_agls = [r.get("min_agl", 0.0) for r in self.episode_results]
        mean_agls = [r.get("mean_agl", 0.0) for r in self.episode_results]
        avg_risks = [r.get("avg_risk", 0.0) for r in self.episode_results]
        p95_cts = [r.get("p95_cross_track", 0.0) for r in self.episode_results]
        max_devs = [r.get("max_corridor_deviation", 0.0) for r in self.episode_results]
        ttgs = [r.get("time_to_goal", float("nan")) for r in self.episode_results]
        plrs = [r.get("path_length_ratio", 1.0) for r in self.episode_results]
        hurs = [r.get("hold_usage_ratio", 0.0) for r in self.episode_results]
        css = [r.get("control_smoothness", 1.0) for r in self.episode_results]

        collision_reasons = ["terrain_collision", "out_of_bounds"]
        collision_rate = float(
            sum(1 for r in self.episode_results
                if r.get("terminate_reason", "") in collision_reasons) / n
        )

        valid_ttgs = [t for t in ttgs if not np.isnan(t)]

        return {
            "n_episodes": n,
            "success_rate": float(np.mean(successes)),
            "collision_rate": collision_rate,
            "avg_return": float(np.mean(returns)),
            "mean_agl": float(np.mean(mean_agls)),
            "min_agl": float(np.min(min_agls)),
            "avg_risk": float(np.mean(avg_risks)),
            "p95_cross_track": float(np.mean(p95_cts)),
            "max_corridor_deviation": float(np.mean(max_devs)),
            "time_to_goal": float(np.mean(valid_ttgs)) if valid_ttgs else float("nan"),
            "path_length_ratio": float(np.mean(plrs)),
            "hold_usage_ratio": float(np.mean(hurs)),
            "control_smoothness": float(np.mean(css)),
        }
