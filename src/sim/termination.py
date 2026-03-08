"""
종료 조건(Termination) 판정 모듈
- 설계서 13절: 성공/실패/truncation 판정
- 성공: 목표 지점 도달
- 실패: 지형 충돌, 맵 이탈, hard corridor 이탈
- Truncation: max_steps 초과
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, Optional, TYPE_CHECKING

from src.sim.state import AircraftState

if TYPE_CHECKING:
    from src.terrain.synthetic import SyntheticTerrain
    from src.route.ref_path import RefPath
    from src.route.corridor import Corridor


# ------------------------------------------------------------------
# 기본 판정 파라미터
# ------------------------------------------------------------------

GOAL_RADIUS: float = 200.0       # 목표 도달 반경 [m]
GOAL_Z_MARGIN: float = 150.0     # 목표 고도 허용 오차 [m]
TERRAIN_COLLISION_MARGIN: float = 10.0  # 지형 충돌 판정 AGL [m]
HARD_CORRIDOR_OVERTIME: int = 10  # hard corridor 이탈 허용 스텝 수
OUT_OF_BOUNDS_MARGIN: float = 500.0  # 맵 경계 외부 여유 [m]


def check_termination(
    state: AircraftState,
    terrain: "SyntheticTerrain",
    ref_path: "RefPath",
    corridor: "Corridor",
    step: int,
    max_steps: int = 1000,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, bool, Dict[str, Any]]:
    """
    종료 조건 검사.

    Args:
        state: 현재 항공기 상태
        terrain: 지형 객체
        ref_path: 참조 경로
        corridor: corridor 정의
        step: 현재 스텝 수
        max_steps: 최대 에피소드 스텝 수
        config: 파라미터 오버라이드 dict

    Returns:
        (terminated, truncated, info)
        - terminated: 성공 또는 실패로 종료 (Gymnasium 기준)
        - truncated: 시간 초과로 종료
        - info: 종료 이유 및 상세 정보
    """
    cfg = _resolve_config(config)

    # --- 수치 유효성 검사 ---
    if not state.is_valid():
        return True, False, {
            "reason": "invalid_state",
            "reason_kr": "상태 수치 오류(NaN/Inf)",
            "success": False,
            "step": step,
        }

    # --- 지형 고도 계산 ---
    terrain_h = terrain.get_elevation(state.x, state.y)
    agl = state.z - terrain_h

    # --- 1) 지형 충돌 판정 ---
    if agl < cfg["collision_margin"]:
        return True, False, {
            "reason": "terrain_collision",
            "reason_kr": f"지형 충돌 (AGL={agl:.1f}m)",
            "success": False,
            "step": step,
            "agl": float(agl),
            "terrain_h": float(terrain_h),
        }

    # --- 2) 맵 경계 이탈 판정 ---
    x_min, x_max, y_min, y_max = terrain.world_extent
    margin = cfg["out_of_bounds_margin"]
    if (
        state.x < x_min - margin
        or state.x > x_max + margin
        or state.y < y_min - margin
        or state.y > y_max + margin
    ):
        return True, False, {
            "reason": "out_of_bounds",
            "reason_kr": f"맵 경계 이탈 (x={state.x:.0f}, y={state.y:.0f})",
            "success": False,
            "step": step,
        }

    # --- 3) Hard corridor 이탈 판정 ---
    from src.route.frenet import project_to_path
    s_curr, d_curr, _ = project_to_path(state.x, state.y, ref_path)

    if abs(d_curr) > corridor.hard_width:
        return True, False, {
            "reason": "hard_corridor_exit",
            "reason_kr": f"hard corridor 이탈 (d={d_curr:.0f}m, 한계={corridor.hard_width:.0f}m)",
            "success": False,
            "step": step,
            "cross_track": float(d_curr),
        }

    # --- 4) 성공 판정 (목표 도달) ---
    goal_xy = ref_path.end_point[:2]
    dist_to_goal = float(np.linalg.norm(
        np.array([state.x, state.y]) - goal_xy
    ))
    goal_z_ref = ref_path.get_reference_altitude(ref_path.total_length)
    z_error = abs(state.z - goal_z_ref)

    if dist_to_goal < cfg["goal_radius"] and z_error < cfg["goal_z_margin"]:
        return True, False, {
            "reason": "goal_reached",
            "reason_kr": f"목표 도달 성공 (거리={dist_to_goal:.0f}m)",
            "success": True,
            "step": step,
            "dist_to_goal": float(dist_to_goal),
            "z_error": float(z_error),
        }

    # --- 5) Truncation (최대 스텝 초과) ---
    if step >= max_steps:
        return False, True, {
            "reason": "max_steps",
            "reason_kr": f"최대 스텝({max_steps}) 초과",
            "success": False,
            "step": step,
            "s_progress": float(s_curr),
            "s_ratio": float(s_curr / max(ref_path.total_length, 1.0)),
        }

    # --- 계속 진행 ---
    return False, False, {
        "reason": "running",
        "reason_kr": "진행 중",
        "success": False,
        "step": step,
        "agl": float(agl),
        "cross_track": float(d_curr),
        "s_progress": float(s_curr),
        "dist_to_goal": float(dist_to_goal),
    }


def get_terminal_reward(info: Dict[str, Any]) -> float:
    """
    종료 사유에 따른 terminal reward 반환.

    Args:
        info: check_termination()이 반환한 info dict

    Returns:
        terminal reward 값
    """
    reason = info.get("reason", "running")

    reward_map = {
        "goal_reached": 150.0,
        "terrain_collision": -150.0,
        "hard_corridor_exit": -100.0,
        "out_of_bounds": -100.0,
        "invalid_state": -100.0,
        "max_steps": -30.0,
        "running": 0.0,
    }
    return float(reward_map.get(reason, 0.0))


def _resolve_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """종료 조건 파라미터 처리."""
    defaults = {
        "goal_radius": GOAL_RADIUS,
        "goal_z_margin": GOAL_Z_MARGIN,
        "collision_margin": TERRAIN_COLLISION_MARGIN,
        "out_of_bounds_margin": OUT_OF_BOUNDS_MARGIN,
    }
    if config is None:
        return defaults
    return {**defaults, **config}
