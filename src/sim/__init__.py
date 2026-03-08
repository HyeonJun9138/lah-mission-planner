"""
sim 패키지
- 회전익기 시뮬레이션 핵심 모듈
- 상태(state), 운동학(dynamics), 보상(reward), 종료조건(termination), 지표(metrics)
"""

from src.sim.state import AircraftState
from src.sim.dynamics import kinematic_step
from src.sim.reward import compute_reward, compute_reward_breakdown
from src.sim.termination import check_termination
from src.sim.metrics import EpisodeMetrics

__all__ = [
    "AircraftState",
    "kinematic_step",
    "compute_reward",
    "compute_reward_breakdown",
    "check_termination",
    "EpisodeMetrics",
]
